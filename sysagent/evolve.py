"""Autonomous Self-Improvement (Self-Evolving Codebase) engine for SysAgent."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from collections.abc import Awaitable, Callable

from .agent import AgentManager
from .brain import _extract_json
from .config import Settings
from .i18n import resolve_locale
from .prompts import DEFAULT_PROMPTS, PromptRegistry
from .self_learn import (
    build_self_roadmap,
    check_patch_outcomes,
    cluster_signature,
    create_git_checkpoint,
    error_clusters,
    files_to_deprioritize,
    proposal_stats,
    scan_coverage_gaps,
    surgical_git_rollback,
)
from .storage import Storage
from .watcher import SystemWatcher


PACKAGE_DIR = Path(__file__).resolve().parent

SURVEY_SYSTEM_PROMPT = DEFAULT_PROMPTS["evolve_survey"]["template"]
EVOLVE_SYSTEM_PROMPT = DEFAULT_PROMPTS["evolve_deep"]["template"]
CRITIC_SYSTEM_PROMPT = DEFAULT_PROMPTS["evolve_critic"]["template"]


from .evolve_ast import extract_ast_focus_slice, module_outline as _module_outline
from .evolve_patch import apply_patch_text as _apply_patch_text, hot_reload_module as _hot_reload_module
from .evolve_safety import check_blast_radius as _check_blast_radius


def _safe_json(text: str | None) -> dict[str, Any]:
    """Parse model JSON; a malformed reply counts as 'nothing to do', not a crash."""
    try:
        data = _extract_json(text or "{}")
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


async def _detail(
    emit_event: Callable[[dict[str, Any]], Awaitable[None]] | None, text: str
) -> None:
    if emit_event:
        await emit_event({"type": "detail", "text": text})


class SelfEvolver:
    """Inspects and safely evolves SysAgent's own code when system resources are low."""

    MAX_DEEP_REVIEWS = 2

    def __init__(
        self,
        settings: Settings,
        watcher: SystemWatcher,
        storage: Storage,
        agent: AgentManager,
    ) -> None:
        self.settings = settings
        self.watcher = watcher
        self.storage = storage
        self.agent = agent
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._file_index = 0
        self._inspected_hashes: dict[str, str] = self._load_inspected_hashes()
        self._last_codebase_hash: str = ""
        self._last_cluster_sig: str = ""
        self._core_baseline_sec: float | None = None
        self._last_ab_benchmark: dict[str, Any] | None = None
        self.next_run_at: float = 0.0
        self.last_run_at: float = 0.0
        self.last_result: dict[str, Any] | None = None
        self._cooldown_until: float = 0.0
        self._evolve_lock = asyncio.Lock()

    def _load_inspected_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        try:
            for m in self.storage.list_memories():
                if m.get("key") == "_internal:evolve_hashes":
                    data = json.loads(m.get("value") or "{}")
                    if isinstance(data, dict):
                        hashes.update({str(k): str(v) for k, v in data.items()})
            for prop in self.storage.list_evolve_proposals(limit=2):
                p = Path(prop["file_path"])
                if p.is_file():
                    hashes.setdefault(
                        p.name,
                        hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
                    )
        except Exception:
            pass
        return hashes

    def _save_inspected_hashes(self) -> None:
        try:
            self.storage.set_memory(
                "_internal:evolve_hashes",
                json.dumps(self._inspected_hashes, ensure_ascii=False),
            )
        except Exception:
            pass

    def start(self) -> None:
        if self._running or not self.settings.evolve_enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _stream_json_call(
        self,
        client: Any,
        *,
        model: str,
        system_instruction: str,
        payload: dict[str, Any],
        thinking_level: str,
        emit_event: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> str:
        """Stream one model call with a live heartbeat; return the final text."""
        loop = asyncio.get_running_loop()

        async def _run_with_heartbeat(target_model: str, active_client: Any) -> str:
            self.storage.record_model_request(target_model)
            task = asyncio.create_task(asyncio.to_thread(_call_with_client, active_client, target_model))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            t0 = time.monotonic()
            while not task.done():
                try:
                    return await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
                except TimeoutError as exc:
                    if emit_event:
                        await emit_event({"type": "heartbeat", "elapsed": int(time.monotonic() - t0)})
                    if time.monotonic() - t0 > 180:
                        task.cancel()
                        raise TimeoutError(f"Model {target_model} timed out after 180s") from exc
            return await task

        def _call_with_client(active_client: Any, target_model: str) -> str:
            chunks: list[str] = []
            for event in active_client.interactions.create(
                model=target_model,
                input=json.dumps(payload, ensure_ascii=False),
                system_instruction=system_instruction,
                generation_config={"thinking_level": thinking_level, "thinking_summaries": "auto"},
                store=False,
                stream=True,
            ):
                if event.event_type != "step.delta" or not getattr(event, "delta", None):
                    continue
                dtype = getattr(event.delta, "type", "")
                dtext = getattr(event.delta, "text", "") or ""
                if dtype == "thought_summary" and dtext and emit_event:
                    asyncio.run_coroutine_threadsafe(
                        emit_event({"type": "thought", "text": dtext}), loop
                    ).result()
                elif dtype == "text" and dtext:
                    chunks.append(dtext)
            return "".join(chunks)

        try:
            return await _run_with_heartbeat(model, client)
        except Exception as exc:
            msg = str(exc).lower()
            is_quota = "429" in msg or "resource_exhausted" in msg or "quota" in msg
            if is_quota and self.agent.auth.rotate_key():
                await _detail(emit_event, f"Key rotated after rate limit on {model}, retrying...")
                return await _run_with_heartbeat(model, self.agent._get_client())
            if model != self.settings.light_model and (isinstance(exc, TimeoutError) or is_quota):
                await _detail(emit_event, f"{model} unavailable ({exc}), falling back to {self.settings.light_model}...")
                return await _run_with_heartbeat(self.settings.light_model, self.agent._get_client())
            raise

    def _is_system_idle(self) -> tuple[bool, dict[str, Any]]:
        snap = self.watcher.get_health_snapshot()
        idle = (
            snap["cpu_percent"] <= self.settings.evolve_max_cpu
            and snap["ram_percent"] <= self.settings.evolve_max_ram
        )
        return idle, snap

    async def _loop(self) -> None:
        await asyncio.sleep(45)
        while self._running:
            busy_retry = False
            try:
                events = await asyncio.to_thread(
                    check_patch_outcomes, self.storage, self.rollback_last_evolve
                )
                for ev in events:
                    if ev["outcome"] in {"rolled_back", "regressed"}:
                        self.storage.add_brain_thought(
                            "act", 90, f"Patch #{ev['id']} on {ev['file']} {ev['outcome']}: "
                            f"errors {ev['errors_before']} -> {ev['errors_after']}"
                        )
                res = await self.evolve_once(force=False)
                busy_retry = res.get("reason") == "system_busy"
                self.last_result = {
                    "status": str(res.get("status", "")),
                    "reason": str(res.get("reason") or res.get("title") or ""),
                }
                self._save_inspected_hashes()
            except Exception as exc:
                msg = str(exc)
                self.last_result = {"status": "error", "reason": msg[:120]}
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    self._cooldown_until = time.monotonic() + 900
                self.storage.add_brain_thought("ignore", 0, f"Self-evolve error: {exc}")
            self.last_run_at = time.time()
            interval = 180 if busy_retry else max(60, self.settings.evolve_interval_sec)
            self.next_run_at = time.time() + interval
            await asyncio.sleep(interval)

    def _measure_core_tests(self, cwd: Path) -> float | None:
        """Time tests/test_core.py (None if missing/failing); used as a performance regression gate."""
        core = PACKAGE_DIR.parent / "tests" / "test_core.py"
        if not core.is_file():
            return None
        env = {**os.environ, "PYTHONPATH": str(PACKAGE_DIR.parent)}
        t0 = time.monotonic()
        try:
            rp = subprocess.run(
                [sys.executable, str(core)], capture_output=True, text=True, timeout=60, env=env, cwd=cwd
            )
        except Exception:
            return None
        return (time.monotonic() - t0) if rp.returncode == 0 else None

    def _check_blast_radius(
        self, rel_name: str, original_content: str, new_content: str
    ) -> tuple[bool, str]:
        """Reject oversized deletions on core modules to prevent accidental gutting."""
        return _check_blast_radius(rel_name, original_content, new_content)

    def _run_staging_tests(
        self,
        rel_name: str,
        new_content: str,
        test_code: str | None = None,
        companion_files: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """Copy sysagent/ to a temp dir, apply new_content, and run lint + compile + smoke + tests."""
        with tempfile.TemporaryDirectory() as td:
            staging_root = Path(td)
            staging_pkg = staging_root / "sysagent"
            shutil.copytree(PACKAGE_DIR, staging_pkg)
            (staging_pkg / rel_name).write_text(new_content, encoding="utf-8")
            for comp_name, comp_src in (companion_files or {}).items():
                if comp_name.endswith(".py") and "/" not in comp_name:
                    (staging_pkg / comp_name).write_text(comp_src, encoding="utf-8")

            ruff_bin = shutil.which("ruff")
            if ruff_bin:
                lint_targets = [str(staging_pkg / rel_name)] + [
                    str(staging_pkg / cn) for cn in (companion_files or {}) if cn.endswith(".py") and "/" not in cn
                ]
                r_proc = subprocess.run(
                    [ruff_bin, "check", *lint_targets, "--select", "F,E9,B011,B904"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if r_proc.returncode != 0:
                    return False, f"Ruff lint gate failed:\n{(r_proc.stdout + r_proc.stderr).strip()[-800:]}"

            test_script = (
                "import compileall, sys\n"
                f"assert compileall.compile_dir(r'{staging_pkg}', quiet=1), 'compileall failed'\n"
                "import sysagent.config, sysagent.storage, sysagent.safety, sysagent.tools, sysagent.agent, sysagent.brain, sysagent.evolve, sysagent.workers, sysagent.server, sysagent.i18n, sysagent.neural\n"
                "assert sysagent.i18n.t('cli_status_line', 'tr', status='ok', name='Vex', model='m', approval='smart', locale='tr')\n"
                "from fastapi.testclient import TestClient\n"
                "with TestClient(sysagent.server.app) as c:\n"
                "    assert c.get('/api/status').status_code == 200\n"
                "    assert c.get('/api/prompts').status_code == 200\n"
                "print('STAGING_OK')\n"
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(staging_root)
            env["SYSAGENT_BRAIN_ENABLED"] = "false"
            env["SYSAGENT_EVOLVE_ENABLED"] = "false"
            proc = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd=staging_root,
            )
            passed = proc.returncode == 0 and "STAGING_OK" in proc.stdout
            output = (proc.stdout + "\n" + proc.stderr).strip()[-1000:]
            if passed and test_code and test_code.strip():
                t_proc = subprocess.run(
                    [sys.executable, "-c", test_code],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    env=env,
                    cwd=staging_root,
                )
                if t_proc.returncode != 0:
                    return False, (t_proc.stdout + "\n" + t_proc.stderr).strip()[-1000:]
            if passed:
                tests_dir = PACKAGE_DIR.parent / "tests"
                if tests_dir.is_dir():
                    test_files = [
                        *(tests_dir.glob("test_core.py")),
                        *(tests_dir.glob("test_server.py")),
                        *(tests_dir.glob("test_self_learn.py")),
                        *sorted(tests_dir.glob("test_evolve_*.py"))[:10],
                    ]
                    for tf in test_files:
                        t_start = time.monotonic()
                        rp = subprocess.run(
                            [sys.executable, str(tf)],
                            capture_output=True,
                            text=True,
                            timeout=25,
                            env=env,
                            cwd=staging_root,
                        )
                        if rp.returncode == 0 and tf.name == "test_core.py":
                            if self._core_baseline_sec is None:
                                self._core_baseline_sec = self._measure_core_tests(PACKAGE_DIR.parent) or -1.0
                            elapsed = time.monotonic() - t_start
                            base = self._core_baseline_sec
                            if base > 0:
                                delta_pct = round(((elapsed - base) / base) * 100.0, 1)
                                self._last_ab_benchmark = {
                                    "file": rel_name,
                                    "baseline_sec": round(base, 3),
                                    "candidate_sec": round(elapsed, 3),
                                    "delta_pct": delta_pct,
                                }
                                output += f"\n[A/B benchmark: candidate={elapsed:.2f}s vs baseline={base:.2f}s ({delta_pct:+.1f}%)]"
                            if base > 0 and elapsed > base * 2.0 + 3.0:
                                return False, (
                                    f"Performance regression: test_core.py took {elapsed:.1f}s "
                                    f"vs baseline {base:.1f}s"
                                )
                        if rp.returncode != 0:
                            # Do not blame a new proposal if a generated test_evolve_* already fails on baseline
                            if tf.name.startswith("test_evolve_"):
                                base_rp = subprocess.run(
                                    [sys.executable, str(tf)],
                                    capture_output=True,
                                    text=True,
                                    timeout=25,
                                    env={**env, "PYTHONPATH": str(PACKAGE_DIR.parent)},
                                    cwd=str(PACKAGE_DIR.parent),
                                )
                                if base_rp.returncode != 0:
                                    continue
                            return False, f"Regression {tf.name} failed:\n{(rp.stdout + rp.stderr)[-600:]}"
            return passed, output

    async def evolve_once(
        self,
        force: bool = False,
        emit: Callable[[str], Awaitable[None]] | None = None,
        emit_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Inspect the codebase and propose/apply an optimization, bugfix, or feature."""
        if self._evolve_lock.locked():
            return {"status": "skipped", "reason": "evolve_already_running"}
        cortex = getattr(self, "cortex", None)
        if not force and cortex is not None:
            flow = getattr(cortex, "last_flow", None) or {}
            if flow.get("suppress_low_priority"):
                return {"status": "deferred", "reason": f"cortex_flow:{flow.get('mode', 'focus')}"}
        async with self._evolve_lock:
            prev_hash = self._last_codebase_hash
            try:
                return await self._evolve_once_locked(force=force, emit=emit, emit_event=emit_event)
            except Exception:
                self._last_codebase_hash = prev_hash
                raise
            finally:
                self._save_inspected_hashes()

    async def _evolve_once_locked(
        self,
        force: bool = False,
        emit: Callable[[str], Awaitable[None]] | None = None,
        emit_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        async def _log(msg: str) -> None:
            if emit:
                await emit(msg)
            if emit_event:
                await emit_event({"type": "progress", "message": msg})

        if not force and time.monotonic() < self._cooldown_until:
            return {"status": "skipped", "reason": "quota_cooldown"}

        if not force and self.watcher.current_context == "media_meeting":
            return {"status": "skipped", "reason": "media_meeting_context"}

        idle, snap = self._is_system_idle()
        if not force and not idle:
            return {"status": "skipped", "reason": "system_busy", "health": snap}

        if not self.agent.auth.is_configured:
            return {"status": "skipped", "reason": "missing_credentials"}

        all_files = sorted(
            p for p in PACKAGE_DIR.glob("*.py") if p.name != "__init__.py"
        )
        codebase = {p.name: p.read_text(encoding="utf-8") for p in all_files}
        file_hashes = {
            name: hashlib.sha256(src.encode("utf-8")).hexdigest() for name, src in codebase.items()
        }
        combined_hash = hashlib.sha256(
            json.dumps(file_hashes, sort_keys=True).encode("utf-8")
        ).hexdigest()
        clusters = [c for c in error_clusters(self.storage) if c["file"] in codebase]
        cluster_sig = cluster_signature(clusters)
        if not force and combined_hash == self._last_codebase_hash and cluster_sig == self._last_cluster_sig:
            return {"status": "skipped", "reason": "no_codebase_changes"}
        self._last_codebase_hash = combined_hash
        self._last_cluster_sig = cluster_sig
        stats = proposal_stats(self.storage)
        deprioritized = files_to_deprioritize(stats)
        loc = resolve_locale(self.settings.locale)
        client = self.agent._get_client()

        # cheap model pick which files deserve a full read. Sending 200 KB to a light model in one
        await _log(f"[1/5] Building codebase map ({len(codebase)} modules)...")
        import inspect

        global_sigs: dict[str, Any] = {}
        method_candidates: dict[str, list[Any]] = {}
        for name in codebase:
            stem = name[:-3]
            mod_name = f"sysagent.{stem}"
            try:
                mod = importlib.import_module(mod_name)
                for attr_name, obj in inspect.getmembers(mod, inspect.isfunction):
                    if getattr(obj, "__module__", "") == mod_name:
                        global_sigs[attr_name] = inspect.signature(obj)
                for _, cls_obj in inspect.getmembers(mod, inspect.isclass):
                    if getattr(cls_obj, "__module__", "") != mod_name:
                        continue
                    for m_name, m_fn in inspect.getmembers(cls_obj, inspect.isfunction):
                        if m_name.startswith("_"):
                            continue
                        raw_sig = inspect.signature(m_fn)
                        params = list(raw_sig.parameters.values())
                        if params and params[0].name in {"self", "cls"}:
                            params = params[1:]
                        method_candidates.setdefault(m_name, []).append(
                            raw_sig.replace(parameters=params)
                        )
            except Exception:
                pass
        method_sigs = {k: v[0] for k, v in method_candidates.items() if len(v) == 1}

        code_map: dict[str, Any] = {}
        bug_candidates: list[dict[str, str]] = []
        for name, src in codebase.items():
            info = _module_outline(src, global_sigs=global_sigs, method_sigs=method_sigs)
            code_map[name] = info
            bugs = [f for f in info["ast_findings"] if f.startswith("BUG:")]
            if bugs and self._inspected_hashes.get(name) != file_hashes[name]:
                bug_candidates.append({"file": name, "why": f"Static signature/call bug detected: {'; '.join(bugs[:2])}"})
            findings_note = f"  [AST: {', '.join(info['ast_findings'][:3])}]" if info["ast_findings"] else ""
            await _detail(
                emit_event,
                f"  {name:<14} {info['lines']:>4} lines  {len(info['symbols']):>3} defs  {', '.join(info['symbols'][:6])}{findings_note}",
            )

        recent_thoughts = self.storage.list_brain_thoughts(limit=25)
        last_evolve_id = max(
            (
                int(th["id"])
                for th in recent_thoughts
                if str(th.get("summary", "")).startswith("Self-evolved ")
            ),
            default=0,
        )
        fresh_errors = [
            th["summary"]
            for th in recent_thoughts
            if int(th["id"]) > last_evolve_id and "error" in str(th.get("summary", "")).lower()
        ][:8]
        recent_proposals = [
            {
                "file": Path(p["file_path"]).name,
                "category": p["category"],
                "title": p["title"],
                "status": p["status"],
            }
            for p in self.storage.list_evolve_proposals(limit=12)
        ]
        reviewed_clean = [
            n for n, h in file_hashes.items() if self._inspected_hashes.get(n) == h
        ]
        # When almost all modules have been marked reviewed, rotate/reset inspected_hashes
        if len(reviewed_clean) >= len(codebase) - 3:
            keep_recent = {p["file"] for p in recent_proposals[:2]}
            self._inspected_hashes = {
                k: v for k, v in self._inspected_hashes.items() if k in keep_recent
            }
            reviewed_clean = [
                n for n, h in file_hashes.items() if self._inspected_hashes.get(n) == h
            ]
        seen_bug = {c["file"] for c in bug_candidates}
        for cl in clusters:
            if cl["file"] not in seen_bug:
                seen_bug.add(cl["file"])
                bug_candidates.append(
                    {
                        "file": cl["file"],
                        "why": f"Recurring runtime error x{cl['count']} at line {cl['line']}: {cl['message']}",
                    }
                )
        survey_payload = {
            "code_map": code_map,
            "tool_metrics": self.storage.get_tool_metrics(limit=10),
            "recent_errors": fresh_errors,
            "recent_proposals": recent_proposals,
            "recently_reviewed_no_change": reviewed_clean,
            "proposal_success_stats": {
                "success_rate": stats["success_rate"],
                "by_category": stats["by_category"],
                "deprioritized_files": sorted(deprioritized),
            },
            "recurring_errors": clusters[:5],
            "untested_symbols": scan_coverage_gaps().get("untested_by_file", {}),
            "self_roadmap": build_self_roadmap(self.storage)[:4],
        }
        await _log("[1/5] Asking which files deserve a full review...")
        survey_text = await self._stream_json_call(
            client,
            model=self.settings.evolve_model,
            system_instruction=PromptRegistry(self.storage).render("evolve_survey", agent_name=self.settings.agent_name),
            payload=survey_payload,
            thinking_level="low",
            emit_event=emit_event,
        )
        survey = _safe_json(survey_text)
        candidates: list[dict[str, Any]] = list(bug_candidates)
        seen_cand = {c["file"] for c in candidates}
        for c in survey.get("candidates") or []:
            name = str(c.get("file") or "")
            if name not in codebase or name in seen_cand:
                continue
            if not force and (name in reviewed_clean or name in deprioritized):
                continue
            candidates.append({"file": name, "why": str(c.get("why") or "")})
            seen_cand.add(name)
        for rm_item in survey_payload.get("self_roadmap") or []:
            if str(rm_item.get("priority")) == "high":
                rm_file = str(rm_item.get("target") or rm_item.get("file") or "")
                if rm_file in codebase and rm_file not in seen_cand and (force or rm_file not in deprioritized):
                    candidates.append({"file": rm_file, "why": str(rm_item.get("goal") or "High-priority roadmap item")})
                    seen_cand.add(rm_file)
        bug_files = {c["file"] for c in bug_candidates}
        candidates = candidates[: self.MAX_DEEP_REVIEWS * 2]
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(self.storage)
            for c in candidates:
                c["neural_score"] = eng.score_patch_candidate(c["file"], "bugfix", c["why"])
            candidates.sort(
                key=lambda c: (1 if c["file"] in bug_files else 0, float(c.get("neural_score", 0.5))),
                reverse=True,
            )
        except Exception:
            pass
        candidates = candidates[: self.MAX_DEEP_REVIEWS]

        if not candidates:
            await _log("[done] Survey found no file worth a deep review right now.")
            self.storage.add_brain_thought("ignore", 80, "Self-evolve survey: no candidates")
            return {"status": "skipped", "reason": "no_candidates", "survey": survey.get("notes", "")}

        for c in candidates:
            n_lbl = f" [neural P(ok)={c['neural_score']:.2f}]" if "neural_score" in c else ""
            await _detail(emit_event, f"  -> selected {c['file']}{n_lbl}: {c['why']}")

        parsed: dict[str, Any] = {}
        target_file = PACKAGE_DIR / candidates[0]["file"]
        deep_sys_inst = PromptRegistry(self.storage).render("evolve_deep", agent_name=self.settings.agent_name, locale=loc)
        for idx, c in enumerate(candidates, start=1):
            name = c["file"]
            src = codebase[name]
            slice_info = extract_ast_focus_slice(src, c["why"], code_map[name]["ast_findings"])
            await _log(f"[2/5] Reading {name} in full ({code_map[name]['lines']} lines) [{idx}/{len(candidates)}]...")
            deep_payload = {
                "file": name,
                "source": src,
                "focus_slice": slice_info if slice_info.get("sliced") else None,
                "why_selected": c["why"],
                "ast_findings": code_map[name]["ast_findings"],
                "recent_proposals_for_file": [p for p in recent_proposals if p["file"] == name],
                "other_modules_outline": {k: v["symbols"] for k, v in code_map.items() if k != name},
                "tool_metrics": survey_payload["tool_metrics"],
            }
            if "Static signature/call bug" in c["why"] or "Recurring runtime error" in c["why"]:
                try:
                    from .local_mind import LocalMind, find_gguf_model

                    if find_gguf_model("coding", strict=True):
                        local_payload = {
                            "file": name,
                            "source": slice_info["source"],
                            "focus_symbol": slice_info.get("focus_symbol"),
                            "why_selected": c["why"],
                            "ast_findings": code_map[name]["ast_findings"],
                        }
                        await _detail(emit_event, f"  Trying local coding specialist on {name} (slice={bool(slice_info.get('sliced'))})...")
                        local_txt = await asyncio.to_thread(
                            LocalMind(self.storage).run_gguf_inference,
                            json.dumps(local_payload, ensure_ascii=False),
                            deep_sys_inst,
                            "coding",
                        )
                        if local_txt:
                            l_res = _safe_json(local_txt)
                            if str(l_res.get("decision")) in {"bugfix", "optimization"}:
                                l_cand = _apply_patch_text(
                                    src,
                                    l_res.get("target"),
                                    l_res.get("replacement"),
                                    l_res.get("start_line"),
                                    l_res.get("end_line"),
                                )
                                if l_cand and l_cand != src:
                                    l_ok, _ = await asyncio.to_thread(
                                        self._run_staging_tests, name, l_cand, l_res.get("test_code")
                                    )
                                    if l_ok:
                                        parsed = l_res
                                        target_file = PACKAGE_DIR / name
                                        await _detail(emit_event, f"  Local coding specialist produced passing patch for {name}.")
                                        break
                except Exception:
                    pass
            deep_text = await self._stream_json_call(
                client,
                model=self.settings.evolve_model,
                system_instruction=deep_sys_inst,
                payload=deep_payload,
                thinking_level="low",
                emit_event=emit_event,
            )
            result = _safe_json(deep_text)
            decision_i = str(result.get("decision", "skip"))
            reason_i = str(result.get("reason") or "(no reason given)")
            if decision_i == "skip":
                self._inspected_hashes[name] = file_hashes[name]
                await _detail(emit_event, f"  {name}: no worthwhile change ({reason_i})")
                continue
            parsed = result
            target_file = PACKAGE_DIR / name
            break

        if not parsed:
            await _log("[done] Deep review found nothing worth changing.")
            self.storage.add_brain_thought(
                "ignore", 90, f"Self-evolve reviewed {', '.join(c['file'] for c in candidates)}: no change"
            )
            return {"status": "skipped", "reason": "deep_review_no_change", "reviewed": [c["file"] for c in candidates]}

        decision = str(parsed.get("decision", "skip"))
        original_content = codebase[target_file.name]
        title = str(parsed.get("title") or f"Improve {target_file.name}")
        reason = str(parsed.get("reason") or "")

        if decision == "ask_user":
            question = str(parsed.get("question") or reason)
            qid = self.storage.create_brain_question(
                question, f"Self-evolve inspection of {target_file.name}", title
            )
            self.storage.add_brain_thought("ask_user", 70, f"Self-evolve asked user (#{qid}): {title}")
            await _log(f"[done] Asked user (#{qid}): {question}")
            return {"status": "asked_user", "question_id": qid, "question": question}

        if decision not in {"optimization", "new_feature", "bugfix"}:
            return {"status": "skipped", "reason": f"invalid_decision:{decision}"}

        target_str = parsed.get("target")
        replacement_str = parsed.get("replacement")
        new_content = _apply_patch_text(
            original_content,
            target_str,
            replacement_str,
            parsed.get("start_line"),
            parsed.get("end_line"),
        )
        if not new_content or new_content == original_content:
            await _log(f"[fail] Proposed edit does not match the current source of {target_file.name}.")
            return {"status": "skipped", "reason": "target_not_found", "file": target_file.name}

        blast_ok, blast_msg = self._check_blast_radius(target_file.name, original_content, new_content)
        if not blast_ok:
            await _log(f"[fail] {blast_msg}")
            return {"status": "skipped", "reason": "blast_radius_exceeded", "file": target_file.name}

        companion_files: dict[str, str] = {}
        readme_companion: tuple[str, str] | None = None
        for comp in parsed.get("companion_edits") or []:
            if not isinstance(comp, dict):
                continue
            cname = str(comp.get("file") or "")
            if cname == "README.md":
                readme_p = PACKAGE_DIR.parent / "README.md"
                if readme_p.is_file():
                    r_orig = readme_p.read_text(encoding="utf-8")
                    r_new = _apply_patch_text(r_orig, comp.get("target"), comp.get("replacement"))
                    if r_new and r_new != r_orig:
                        readme_companion = (r_orig, r_new)
            elif cname in codebase and cname != target_file.name:
                c_new = _apply_patch_text(codebase[cname], comp.get("target"), comp.get("replacement"))
                if c_new and c_new != codebase[cname]:
                    companion_files[cname] = c_new

        diff_text = "".join(
            difflib.unified_diff(
                original_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{target_file.name}",
                tofile=f"b/{target_file.name}",
            )
        )

        await _log(f"[2/5] Proposed {decision} in {target_file.name}: {title}")
        if emit_event:
            await emit_event({"type": "detail", "text": f"Reason: {reason}\n{diff_text[:1500]}"})
        await _log("[3/5] Running isolated staging tests on a copy of the package...")
        test_code = parsed.get("test_code")
        passed, test_out = await asyncio.to_thread(
            self._run_staging_tests, target_file.name, new_content, test_code, companion_files
        )
        if not passed:
            await _log("[3/5] Staging tests failed. Attempting 1 automatic self-repair...")
            if emit_event and test_out:
                await emit_event({"type": "detail", "text": test_out[:600]})
            repair_payload = {
                "file": target_file.name,
                "source": original_content,
                "previous_proposal": parsed,
                "test_error": test_out,
            }
            repair_text = await self._stream_json_call(
                client,
                model=self.settings.evolve_model,
                system_instruction=PromptRegistry(self.storage).render("evolve_deep", agent_name=self.settings.agent_name, locale=loc),
                payload=repair_payload,
                thinking_level="medium",
                emit_event=emit_event,
            )
            rep_parsed = _safe_json(repair_text)
            rep_content = _apply_patch_text(
                original_content,
                rep_parsed.get("target"),
                rep_parsed.get("replacement"),
                rep_parsed.get("start_line"),
                rep_parsed.get("end_line"),
            )
            if rep_content and rep_content != original_content:
                new_content = rep_content
                test_code = rep_parsed.get("test_code") or test_code
                diff_text = "".join(
                    difflib.unified_diff(
                        original_content.splitlines(keepends=True),
                        new_content.splitlines(keepends=True),
                        fromfile=f"a/{target_file.name}",
                        tofile=f"b/{target_file.name}",
                    )
                )
                passed, test_out = await asyncio.to_thread(
                    self._run_staging_tests, target_file.name, new_content, test_code
                )
        if not passed:
            await _log("[fail] Staging tests failed.")
            if emit_event and test_out:
                await emit_event({"type": "detail", "text": test_out[:800]})
            self._inspected_hashes[target_file.name] = file_hashes[target_file.name]
            pid = self.storage.add_evolve_proposal(
                str(target_file), decision, title, f"{reason}\nTests failed: {test_out}",
                diff_text, new_content, False, "failed",
            )
            try:
                from .neural import NeuralEngine

                NeuralEngine(self.storage).online_patch_step(target_file.name, decision, title, reason, success=False)
            except Exception:
                pass
            return {"status": "failed_tests", "proposal_id": pid, "output": test_out}

        await _log("[4/5] Staging tests passed. Adversarial critic is reviewing the diff...")
        critic_ok, critique = await self._run_critic(title, reason, diff_text)
        if not critic_ok:
            await _log(f"[revise] Critic requested changes: {critique}")
            rev_payload = {
                "file": target_file.name,
                "source": original_content,
                "previous_proposal": parsed,
                "critic_feedback": critique,
                "instruction": "Fix ALL points raised by critic_feedback or narrow title/reason to match the exact diff.",
            }
            rev_text = await self._stream_json_call(
                client,
                model=self.settings.evolve_model,
                system_instruction=PromptRegistry(self.storage).render("evolve_deep", agent_name=self.settings.agent_name, locale=loc),
                payload=rev_payload,
                thinking_level="medium",
                emit_event=emit_event,
            )
            rev_parsed = _safe_json(rev_text)
            rev_content = _apply_patch_text(
                original_content,
                rev_parsed.get("target"),
                rev_parsed.get("replacement"),
                rev_parsed.get("start_line"),
                rev_parsed.get("end_line"),
            )
            if rev_content and rev_content != original_content:
                rev_diff = "".join(
                    difflib.unified_diff(
                        original_content.splitlines(keepends=True),
                        rev_content.splitlines(keepends=True),
                        fromfile=f"a/{target_file.name}",
                        tofile=f"b/{target_file.name}",
                    )
                )
                rev_test_code = rev_parsed.get("test_code") or test_code
                rev_passed, _ = await asyncio.to_thread(
                    self._run_staging_tests, target_file.name, rev_content, rev_test_code
                )
                if rev_passed:
                    new_content = rev_content
                    diff_text = rev_diff
                    test_code = rev_test_code
                    title = str(rev_parsed.get("title") or title)
                    reason = str(rev_parsed.get("reason") or reason)
                    critic_ok, critique = await self._run_critic(title, reason, diff_text)
        if not critic_ok:
            await _log(f"[reject] Critic rejected: {critique}")
            self._inspected_hashes[target_file.name] = file_hashes[target_file.name]
            pid = self.storage.add_evolve_proposal(
                str(target_file), decision, title, f"{reason}\nCritic rejected: {critique}",
                diff_text, new_content, True, "rejected_by_critic",
            )
            self.storage.add_brain_thought("ignore", 90, f"Critic rejected {target_file.name}: {critique}")
            try:
                from .neural import NeuralEngine

                NeuralEngine(self.storage).online_patch_step(target_file.name, decision, title, critique, success=False)
            except Exception:
                pass
            return {"status": "rejected_by_critic", "proposal_id": pid, "critique": critique}

        await _log("[5/5] Critic approved. Finalizing...")

        status = "applied" if self.settings.evolve_auto_apply else "pending"
        pid = self.storage.add_evolve_proposal(
            str(target_file), decision, title, reason, diff_text, new_content, True, status
        )
        hot_ok = False
        if status == "applied":
            batch_key = f"evolve:{pid}"
            self._write_with_backup(target_file, original_content, new_content, batch_id=batch_key)
            for cname, c_src in companion_files.items():
                self._write_with_backup(PACKAGE_DIR / cname, codebase[cname], c_src, batch_id=batch_key)
            if readme_companion is not None:
                self._write_with_backup(
                    PACKAGE_DIR.parent / "README.md", readme_companion[0], readme_companion[1], batch_id=batch_key
                )
            live_ok, _ = await asyncio.to_thread(
                self._run_staging_tests, target_file.name, new_content, test_code, companion_files
            )
            if not live_ok:
                self.rollback_last_evolve(pid)
                self._inspected_hashes[target_file.name] = file_hashes[target_file.name]
                try:
                    from .neural import NeuralEngine

                    NeuralEngine(self.storage).online_patch_step(target_file.name, decision, title, reason, success=False)
                except Exception:
                    pass
                return {"status": "rolled_back", "proposal_id": pid}
            self._inspected_hashes[target_file.name] = hashlib.sha256(
                new_content.encode("utf-8")
            ).hexdigest()
            hot_ok = _hot_reload_module(target_file.name)
            for cname in companion_files:
                _hot_reload_module(cname)
            ab_info = self._last_ab_benchmark
            if ab_info and ab_info.get("file") == target_file.name:
                self.storage.add_lesson(
                    f"evolve_ab:{target_file.name}",
                    f"A/B benchmark for '{title[:80]}': {ab_info['candidate_sec']}s vs {ab_info['baseline_sec']}s ({ab_info['delta_pct']:+.1f}%)",
                    "Keep optimizations that reduce or preserve baseline latency.",
                )
            try:
                from .neural import NeuralEngine

                NeuralEngine(self.storage).online_patch_step(target_file.name, decision, title, reason, success=True)
            except Exception:
                pass
            self._core_baseline_sec = None  # re-measure after the code base changed
            if test_code and str(test_code).strip():
                tests_dir = PACKAGE_DIR.parent / "tests"
                tests_dir.mkdir(exist_ok=True)
                (tests_dir / f"test_evolve_{pid}.py").write_text(str(test_code), encoding="utf-8")
            if decision == "new_feature":
                from .self_ops import sync_capabilities_doc
                sync_capabilities_doc(self.agent.plugins)
            create_git_checkpoint(PACKAGE_DIR.parent, pid, target_file, title)
            self.storage.add_brain_thought("act", 95, f"Self-evolved {target_file.name}: {title}", diff_text[:300])
            self.storage.add_lesson(
                f"self_evolve:{target_file.name}:{decision}",
                title[:180],
                reason[:260],
            )
            try:
                from .local_mind import LocalMind

                LocalMind(self.storage).record_and_learn(
                    f"Self-evolve {target_file.name}: {title}",
                    [{"name": "edit_file", "arguments": {"path": str(target_file)}}],
                    f"{reason}\n\nDiff:\n{diff_text[:1200]}",
                )
            except Exception:
                pass
            self.watcher._send_desktop_notification(
                f"{self.settings.agent_name} Self-Evolved", f"[{decision}] {title}"
            )
            if hot_ok:
                await _log(f"Applied & hot-reloaded {target_file.name} in memory.")
            else:
                await _log("Applied. Restart the daemon to load it: systemctl --user restart sysagent")
        else:
            self._inspected_hashes[target_file.name] = file_hashes[target_file.name]
        return {
            "status": status,
            "proposal_id": pid,
            "file": target_file.name,
            "category": decision,
            "title": title,
            "hot_reloaded": hot_ok,
            "restart_required": status == "applied" and not hot_ok,
        }

    async def _run_critic(self, title: str, reason: str, diff_text: str) -> tuple[bool, str]:
        """Run a second-opinion adversarial critic on the proposed diff (with local reasoning pre-screen)."""
        try:
            client = self.agent._get_client()
            if type(client).__name__ in {"AgyCliClient", "LocalMindClient"}:
                from .local_mind import LocalMind, find_gguf_model

                if find_gguf_model("reasoning", strict=True):
                    critic_sys = PromptRegistry(self.storage).render(
                        "evolve_critic", agent_name=self.settings.agent_name
                    )
                    loc_txt = await asyncio.to_thread(
                        LocalMind(self.storage).run_gguf_inference,
                        f"Title: {title}\nReason: {reason}\nDiff:\n{diff_text[:6000]}",
                        critic_sys,
                        "reasoning",
                    )
                    if loc_txt:
                        loc_p = _extract_json(loc_txt)
                        if "approved" in loc_p and not bool(loc_p.get("approved")) and loc_p.get("critique"):
                            return False, f"[local:reasoning] {loc_p['critique']}"
            resp = await self.agent.auth.call_interactions(
                self.storage,
                model=self.settings.evolve_model,
                timeout_sec=120.0,
                input=f"Title: {title}\nReason: {reason}\nDiff:\n{diff_text[:12000]}",
                system_instruction=PromptRegistry(self.storage).render("evolve_critic", agent_name=self.settings.agent_name),
                store=False,
            )
            parsed = _extract_json(resp.output_text or "{}")
            return bool(parsed.get("approved", False)), str(parsed.get("critique", "") or "no critique given")
        except Exception as exc:
            # A safety gate that cannot run must reject, never approve.
            return False, f"Critic unavailable, rejecting for safety: {exc}"

    def _write_with_backup(
        self, path: Path, old_content: str, new_content: str, batch_id: str | None = None
    ) -> None:
        self.storage.record_file_backup("evolve", str(path), True, old_content, batch_id=batch_id)
        path.write_text(new_content, encoding="utf-8")

    def apply_proposal(self, proposal_id: int) -> bool:
        prop = self.storage.get_evolve_proposal(proposal_id)
        if not prop or not prop["test_passed"]:
            return False
        path = Path(prop["file_path"])
        old_content = path.read_text(encoding="utf-8") if path.exists() else ""
        current_diff = "".join(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                prop["new_content"].splitlines(keepends=True),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
            )
        )
        if current_diff != prop["diff_text"]:
            self.storage.update_evolve_status(proposal_id, "stale")
            return False
        self._write_with_backup(path, old_content, prop["new_content"])
        live_ok, _ = self._run_staging_tests(path.name, prop["new_content"])
        if not live_ok:
            self.rollback_last_evolve(proposal_id)
            return False
        self.storage.update_evolve_status(proposal_id, "applied")
        _hot_reload_module(path.name)
        return True

    def rollback_last_evolve(self, proposal_id: int | None = None) -> bool:
        """Auto-rollback latest audit backup (or full multi-file batch), or surgically revert an older commit."""
        if proposal_id is not None:
            batch_items = self.storage.get_undoable_batch(f"evolve:{proposal_id}")
            if batch_items:
                for it in batch_items:
                    t = Path(it["target_path"])
                    if it["existed"]:
                        t.write_text(it["backup_content"] or "", encoding="utf-8")
                    self.storage.mark_undone(it["id"])
                    if t.suffix == ".py":
                        _hot_reload_module(t.name)
                self.storage.update_evolve_status(proposal_id, "rolled_back")
                self.storage.add_brain_thought(
                    "act", 100, f"Auto-rolled back evolve batch #{proposal_id} ({len(batch_items)} files)"
                )
                return True
        item = self.storage.get_latest_undoable(action="evolve")
        if proposal_id is not None:
            prop = self.storage.get_evolve_proposal(proposal_id) or {}
            prop_file = Path(str(prop.get("file_path") or "")).name
            if prop_file and (not item or Path(item["target_path"]).name != prop_file):
                if surgical_git_rollback(PACKAGE_DIR.parent, proposal_id):
                    self.storage.update_evolve_status(proposal_id, "rolled_back")
                    self.storage.add_brain_thought(
                        "act", 100, f"Surgical git rollback for evolve #{proposal_id} ({prop_file})"
                    )
                    _hot_reload_module(prop_file)
                    return True
                return False
        if not item:
            return False
        target = Path(item["target_path"])
        if item["existed"]:
            target.write_text(item["backup_content"] or "", encoding="utf-8")
        self.storage.mark_undone(item["id"])
        if target.suffix == ".py":
            _hot_reload_module(target.name)
        if proposal_id is not None:
            self.storage.update_evolve_status(proposal_id, "rolled_back")
        self.storage.add_brain_thought("act", 100, f"Auto-rolled back evolve on {target.name}")
        return True

    async def evolve_prompts_once(self, emit_event: Any = None) -> dict[str, Any]:
        """Inspect user style memories, lessons, and thoughts to refine a registered prompt."""
        if self._evolve_lock.locked():
            return {"status": "skipped", "reason": "evolve_already_running"}
        cortex = getattr(self, "cortex", None)
        if cortex is not None:
            flow = getattr(cortex, "last_flow", None) or {}
            if flow.get("suppress_low_priority"):
                return {"status": "deferred", "reason": f"cortex_flow:{flow.get('mode', 'focus')}"}
        if time.monotonic() < self._cooldown_until:
            return {"status": "skipped", "reason": "quota_cooldown"}
        if not self.agent.auth.is_configured:
            return {"status": "skipped", "reason": "missing_credentials"}
        async with self._evolve_lock:
            registry = PromptRegistry(self.storage)
            prompts_status = registry.list_status()
            style_memories = [
                {"key": mem["key"], "value": mem["value"]}
                for mem in self.storage.list_memories()
                if str(mem.get("key", "")).startswith("style:")
            ]
            lessons = [
                {
                    "context": lesson.get("context"),
                    "insight": lesson.get("insight"),
                    "resolution": lesson.get("resolution"),
                }
                for lesson in self.storage.list_lessons(limit=15)
            ]
            thoughts = [
                {
                    "decision": th.get("decision"),
                    "confidence": th.get("confidence"),
                    "summary": th.get("summary"),
                }
                for th in self.storage.list_brain_thoughts(limit=15)
            ]
            if not style_memories and not lessons and not thoughts:
                return {"status": "skipped", "reason": "no_feedback_signals"}

            payload = {
                "prompts": [
                    {
                        "key": p["key"],
                        "version": p["version"],
                        "required_vars": p["required_vars"],
                        "template": p["template"],
                    }
                    for p in prompts_status
                ],
                "style_memories": style_memories,
                "recent_lessons": lessons,
                "recent_thoughts": thoughts,
            }
            system_prompt = (
                f"You are the prompt optimization engineer for {self.settings.agent_name}.\n"
                "Review the registered system prompts against the user's style:* preferences, "
                "recent lessons learned, and autonomous brain thoughts.\n"
                "Propose at most ONE targeted prompt improvement if a clear gain exists "
                "(such as reinforcing user formatting rules, avoiding backticks/markdown clutter, "
                "or preventing recurring tool mistakes).\n"
                "CRITICAL: Preserve every placeholder listed in that prompt's `required_vars` verbatim "
                "(e.g. `{agent_name}`), and do NOT introduce unescaped single braces.\n"
                "Respond ONLY with valid JSON matching one of:\n"
                '{"decision": "skip", "reason": "short reason"}\n'
                "OR\n"
                '{"decision": "update", "prompt_key": "<key>", "new_template": "<full template>", "reason": "concise rationale"}'
            )
            client = self.agent._get_client()
            raw_text = await self._stream_json_call(
                client,
                model=self.settings.evolve_model,
                system_instruction=system_prompt,
                payload=payload,
                thinking_level="low",
                emit_event=emit_event,
            )
            parsed = _safe_json(raw_text)
            if not isinstance(parsed, dict) or parsed.get("decision") == "skip" or not parsed.get("prompt_key"):
                reason = parsed.get("reason") if isinstance(parsed, dict) else None
                return {"status": "skipped", "reason": str(reason or "no_prompt_change_needed")}

            prompt_key = str(parsed["prompt_key"]).strip()
            new_template = str(parsed.get("new_template") or "")
            reason = str(parsed.get("reason") or "self-evolved from style/lessons feedback")
            res = registry.update_prompt(prompt_key, new_template, reason=reason)
            if res.get("status") == "updated":
                self.storage.add_lesson(
                    f"prompt_evolve:{prompt_key}",
                    reason[:180],
                    f"Updated prompt template `{prompt_key}`",
                )
            if res.get("status") == "updated":
                self.storage.add_brain_thought(
                    "act",
                    90,
                    f"PromptRegistry evolved {prompt_key} -> v{res['version']}: {reason}",
                )
            return res

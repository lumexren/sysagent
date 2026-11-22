"""Self-Watchdog, Self-Pruning, Self-Documentation, and Self-Prompt-Tuning for SysAgent."""

from __future__ import annotations

import asyncio
import gc
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from .config import Settings
from .plugins import PLUGINS_DIR, PluginRegistry
from .storage import Storage
from .tools import get_all_tool_declarations
from .watcher import SystemWatcher


README_PATH = Path(__file__).resolve().parent.parent / "README.md"
START_MARKER = "<!-- AUTO_CAPABILITIES_START -->"
END_MARKER = "<!-- AUTO_CAPABILITIES_END -->"

_STYLE_RULES: dict[str, tuple[str, str]] = {
    "style_concise": ("verbosity", "Keep responses very concise and to the point."),
    "style_detailed": ("verbosity", "Provide detailed, thorough explanations."),
    "style_bullets": ("format", "Prefer structured bullet-point lists."),
    "style_plain": (
        "formatting",
        "Write in clean plain text without backticks, bold markdown markers, or parenthetical tool names.",
    ),
}

SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"sk-[0-9A-Za-z\-_]{20,}"),
    re.compile(r"ghp_[0-9A-Za-z]{20,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-\._~\+\/]+=*"),
    re.compile(r"(?i)(password|passwd|secret|api_key|token)(['\"]?\s*[:=]\s*['\"]?)([^\s'\",}]{6,})"),
]


def scrub_secrets(text: str) -> str:
    """Redact common API keys, tokens, and password assignments while preserving JSON syntax."""
    if not text:
        return text
    out = text
    for pat in SECRET_PATTERNS:
        if pat.groups >= 3:
            out = pat.sub(r"\1\2[REDACTED]", out)
        elif pat.groups == 1:
            out = pat.sub(r"\1[REDACTED]", out)
        else:
            out = pat.sub("[REDACTED]", out)
    return out


def choose_model(
    user_message: str, settings: Settings, storage: Storage | None = None
) -> str:
    """Adaptive model router: combine NeuralEngine heads, LocalMind, and Brain-learned complex cues."""
    msg = user_message.strip()
    words = msg.split()
    complex_cues = storage.get_learned_cues().get("complex", []) if storage is not None else []
    has_complex = bool(complex_cues and any(c in msg.lower() for c in complex_cues))
    if storage is not None:
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(storage)
            if int(eng.meta.get("epochs_trained", 0)) > 0:
                unc = eng.predict_with_uncertainty(msg)
                role_probs = eng.predict_role_probs(msg)
                tool_probs = eng.predict_tool_probs(msg)
                if unc.get("uncertain") and len(words) > 10:
                    return settings.model
                if role_probs.get("coding", 0.0) + role_probs.get("reasoning", 0.0) >= 0.60:
                    return settings.model
                if len(words) <= 18 and not unc.get("uncertain") and (
                    role_probs.get("chat", 0.0) >= 0.65
                    or max(tool_probs.values(), default=0.0) >= 0.72
                ):
                    return settings.light_model
        except Exception:
            pass
    if len(words) <= 6 and not has_complex:
        return settings.light_model
    if storage is not None and not has_complex and len(words) <= 18:
        try:
            from .local_mind import LocalMind

            tool, _, score = LocalMind(storage).predict_tool(msg)
            if tool and score >= 0.65:
                return settings.light_model
        except Exception:
            pass
    return settings.model


CORE_TOOLS = frozenset(
    {
        "run_command",
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "search_files",
        "system_status",
        "list_processes",
        "list_systemd_units",
        "read_journal_logs",
        "web_search",
        "fetch_url",
        "desktop_control",
        "create_goal_plan",
        "update_goal_step",
        "run_parallel_workers",
        "forge_skill",
        "remember_fact",
        "recall_memories",
        "search_memory_and_history",
        "record_lesson",
        "undo_last_action",
        "undo_batch_actions",
        "inspect_self_component",
    }
)

def select_relevant_tools(
    user_message: str,
    all_tools: list[dict[str, Any]],
    builtin_names: set[str] | None = None,
    storage: Storage | None = None,
) -> list[dict[str, Any]]:
    """Contextual tool pruning: CORE_TOOLS + NeuralEngine top-5, next-tool chain, and spreading activation."""
    low = (user_message or "").lower()
    if len(low.split()) > 28:
        return all_tools
    selected = set(CORE_TOOLS)
    if any(k in low for k in ("model", "local", "neon", "argon", "hydrogen", "flash", "lite", "switch", "geç")):
        selected.add("switch_model")
    if storage is not None:
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(storage)
            if int(eng.meta.get("epochs_trained", 0)) > 0:
                t_probs = eng.predict_tool_probs(user_message)
                top_sorted = sorted(t_probs.items(), key=lambda kv: kv[1], reverse=True)
                for tname, prob in top_sorted[:5]:
                    if prob >= 0.06:
                        selected.add(tname)
                if top_sorted:
                    for nxt_tool, nxt_p in eng.predict_next_tools(user_message, prev_tool=top_sorted[0][0], top_k=2):
                        if nxt_p >= 0.12:
                            selected.add(nxt_tool)
                for hit in eng.spreading_activation(user_message, top_k=3):
                    h_txt = str(hit.get("text") or "")
                    for decl in all_tools:
                        dn = str(decl.get("name") or "")
                        if dn and dn in h_txt:
                            selected.add(dn)
        except Exception:
            pass
    known_builtins = builtin_names or {str(d.get("name") or "") for d in all_tools if d.get("name")}
    from .self_resilience import get_circuit_breaker

    cb = get_circuit_breaker()
    return [
        decl
        for decl in all_tools
        if (decl.get("name") in selected or decl.get("name") not in known_builtins)
        and (decl.get("name") in CORE_TOOLS or not cb.is_open(f"tool:{decl.get('name')}"))
    ]


class ToolCache:
    """Thread-safe short-TTL cache for read-only tool results."""

    CACHEABLE = frozenset(
        {
            "read_file",
            "list_directory",
            "search_files",
            "system_status",
            "read_journal_logs",
            "list_systemd_units",
            "list_processes",
            "list_events",
            "recall_memories",
            "search_memory_and_history",
            "kg_query",
            "web_search",
            "fetch_url",
        }
    )

    def __init__(self, ttl_sec: float = 15.0, max_entries: int = 128) -> None:
        self.ttl_sec = ttl_sec
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, name: str, args: dict[str, Any]) -> Any | None:
        if name not in self.CACHEABLE:
            return None
        key = f"{name}:{json.dumps(args, sort_keys=True)}"
        with self._lock:
            entry = self._store.get(key)
            if entry and (time.monotonic() - entry[0]) <= self.ttl_sec:
                return entry[1]
            self._store.pop(key, None)
        return None

    def set(self, name: str, args: dict[str, Any], value: Any) -> None:
        if name not in self.CACHEABLE:
            return
        key = f"{name}:{json.dumps(args, sort_keys=True)}"
        now = time.monotonic()
        with self._lock:
            if len(self._store) >= self.max_entries:
                expired = [k for k, (ts, _) in self._store.items() if (now - ts) > self.ttl_sec]
                for k in expired:
                    self._store.pop(k, None)
                if len(self._store) >= self.max_entries:
                    oldest_key = min(self._store, key=lambda k: self._store[k][0])
                    self._store.pop(oldest_key, None)
            self._store[key] = (now, value)

    def invalidate_all(self) -> None:
        with self._lock:
            self._store.clear()


def compact_tool_schemas(tools: list[dict[str, Any]]) -> str:
    """Format tool declarations into compact single-line signatures to save prompt tokens."""
    lines: list[str] = []
    for t in tools:
        name = t.get("name", "")
        params = t.get("parameters") or {}
        props = params.get("properties") or {}
        req = set(params.get("required") or [])
        arg_parts: list[str] = []
        for pname, pspec in props.items():
            ptype = pspec.get("type", "any") if isinstance(pspec, dict) else "any"
            opt = "" if pname in req else "?"
            arg_parts.append(f"{pname}{opt}:{ptype}")
        desc = " ".join(str(t.get("description", "")).split())
        if len(desc) > 90:
            desc = desc[:87] + "..."
        lines.append(f"- {name}({', '.join(arg_parts)}): {desc}")
    return "\n".join(lines)


def truncate_tool_output(text: str, max_chars: int = 4000) -> str:
    """Keep head + tail of oversized tool outputs to prevent context/token bloat."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.65)
    tail = max_chars - head - 48
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...[truncated {omitted} chars]...\n{text[-tail:]}"


def export_brain_bundle(storage: Storage, plugins_dir: Path | None = None) -> dict[str, Any]:
    """Export memories, learned rules, KG edges, lessons, automations, and plugins."""
    from .prompts import PromptRegistry

    pdir = plugins_dir or PLUGINS_DIR
    plugins_data: dict[str, str] = {}
    if pdir.is_dir():
        for py in sorted(pdir.glob("*.py")):
            plugins_data[py.name] = py.read_text(encoding="utf-8")
    custom_prompts = [
        p for p in PromptRegistry(storage).list_status() if p["version"] > 0
    ]
    return {
        "version": 1,
        "memories": storage.list_memories(),
        "brain_rules": storage.list_brain_rules(),
        "kg_edges": storage.query_kg(limit=200),
        "lessons": storage.list_lessons(limit=200),
        "automations": storage.list_automations(),
        "prompts": custom_prompts,
        "plugins": plugins_data,
    }


def import_brain_bundle(
    bundle: dict[str, Any], storage: Storage, plugins_dir: Path | None = None
) -> dict[str, int]:
    """Import a previously exported brain bundle."""
    from .prompts import PromptRegistry

    counts = {"memories": 0, "rules": 0, "kg_edges": 0, "lessons": 0, "automations": 0, "prompts": 0, "plugins": 0}
    for pr in bundle.get("prompts", []):
        res = PromptRegistry(storage).update_prompt(
            pr["key"], pr["template"], reason=f"imported ({pr.get('reason', 'bundle')})"
        )
        if res.get("status") == "updated":
            counts["prompts"] += 1
    for m in bundle.get("memories", []):
        storage.set_memory(m["key"], m["value"])
        counts["memories"] += 1
    for r in bundle.get("brain_rules", []):
        storage.add_brain_rule(r["pattern"], r["action_policy"], int(r.get("confidence", 80)))
        counts["rules"] += 1
    for k in bundle.get("kg_edges", []):
        storage.add_kg_edge(k["subject"], k["relation"], k["object"])
        counts["kg_edges"] += 1
    for l in bundle.get("lessons", []):
        storage.add_lesson(l["context"], l["insight"], l["resolution"])
        counts["lessons"] += 1
    existing_autos = {
        (a["name"], a["trigger_type"], a["trigger_value"])
        for a in storage.list_automations()
    }
    for a in bundle.get("automations", []):
        sig = (a["name"], a["trigger_type"], a["trigger_value"])
        if sig not in existing_autos:
            storage.create_automation(a["name"], a["trigger_type"], a["trigger_value"], a["prompt"])
            existing_autos.add(sig)
            counts["automations"] += 1
    pdir = plugins_dir or PLUGINS_DIR
    if bundle.get("plugins"):
        pdir.mkdir(parents=True, exist_ok=True)
        for fname, content in bundle["plugins"].items():
            (pdir / Path(fname).name).write_text(content, encoding="utf-8")
            counts["plugins"] += 1
    return counts


def detect_style_feedback(user_message: str, storage: Storage) -> str | None:
    """Detect user style preferences using Brain-learned dynamic style cues."""
    low = user_message.lower()
    dyn = storage.get_learned_cues()
    for cue_key, (mem_key, rule) in _STYLE_RULES.items():
        triggers = dyn.get(cue_key) or []
        if triggers and any(t in low for t in triggers):
            storage.set_memory(f"style:{mem_key}", rule)
            try:
                from .neural import NeuralEngine

                NeuralEngine(storage).sync_associative_memory()
            except Exception:
                pass
            return rule
    return None


def sync_capabilities_doc(
    plugins: PluginRegistry | None = None, readme_path: Path | None = None
) -> Path:
    """Update the auto-managed capabilities section in README.md."""
    target = readme_path or README_PATH
    decls = get_all_tool_declarations(plugins)
    lines = [
        START_MARKER,
        "## Active Capabilities (Auto-Generated)",
        "",
    ]
    for d in decls:
        lines.append(f"- `{d['name']}`: {d.get('description', '')}")
    lines.append(END_MARKER)
    section = "\n".join(lines)

    existing = target.read_text(encoding="utf-8") if target.is_file() else ""
    if START_MARKER in existing and END_MARKER in existing:
        before = existing.split(START_MARKER, 1)[0]
        after = existing.split(END_MARKER, 1)[1]
        updated = f"{before}{section}{after}"
    else:
        sep = "\n\n" if existing and not existing.endswith("\n\n") else ""
        updated = f"{existing}{sep}{section}\n"
    if updated != existing:  # avoid rewriting the file every maintenance cycle
        target.write_text(updated, encoding="utf-8")
    return target


class SelfMonitor:
    """Background self-watchdog and periodic database pruning loop."""

    def __init__(
        self,
        storage: Storage,
        watcher: SystemWatcher,
        plugins: PluginRegistry | None = None,
        agent: Any = None,
        evolver: Any = None,
        rss_limit_mb: float = 512.0,
        interval_sec: int = 300,
    ) -> None:
        self.storage = storage
        self.watcher = watcher
        self.plugins = plugins
        self.agent = agent
        self.evolver = evolver
        self.rss_limit_mb = rss_limit_mb
        self.interval_sec = interval_sec
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._cycle_count: int = 0
        self.last_prune: dict[str, int] = {}
        self.last_prompt_evolve: dict[str, Any] = {}

    def start(self) -> None:
        if self._running:
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

    def get_self_health(self) -> dict[str, Any]:
        rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
        return {
            "daemon_rss_mb": rss_mb,
            "rss_limit_mb": self.rss_limit_mb,
            "last_prune": self.last_prune,
            "last_prompt_evolve": self.last_prompt_evolve,
            "tool_metrics": self.storage.get_tool_metrics(limit=15),
        }

    def run_maintenance(self) -> dict[str, Any]:
        """Run self-watchdog memory cleanup, DB pruning, and doc sync."""
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        cleared_cache = False
        if rss_mb > self.rss_limit_mb and self.agent is not None:
            self.agent.sessions.clear()
            self.agent._session_models.clear()
            self.agent._tool_cache.invalidate_all()
            gc.collect()
            cleared_cache = True
            self.watcher.add_event(
                "info",
                "warning",
                f"Self-Watchdog cleared session cache (RSS {rss_mb:.1f} MB > {self.rss_limit_mb} MB)",
            )
        consolidated = self.storage.consolidate_memories()
        pruned = self.storage.prune_database()
        self.last_prune = pruned
        from .self_learn import (
            auto_forge_workflow_skills,
            auto_rollback_failing_prompts,
            build_self_roadmap,
            compact_distilled_dataset,
            compute_adaptive_schedule,
            deduplicate_semantic_knowledge,
            discover_system_profile,
            evaluate_graduation_arena,
            heal_runtime_anomalies,
            inject_plugin_skills_to_local_mind,
            predict_user_routine,
            replay_shadow_history,
            run_local_self_play,
            suggest_workflow_skills,
            synthesize_coverage_tests,
        )

        profile = discover_system_profile(self.storage)
        hygiene = deduplicate_semantic_knowledge(self.storage)
        rolled_prompts = auto_rollback_failing_prompts(self.storage)
        healed = heal_runtime_anomalies(self.storage, self.plugins)
        forged_macros = auto_forge_workflow_skills(self.storage, self.plugins)
        injected_skills = inject_plugin_skills_to_local_mind(self.storage, self.plugins)
        compacted_ds = compact_distilled_dataset()
        auto_cov = synthesize_coverage_tests(max_symbols=15)
        roadmap = build_self_roadmap(self.storage)
        skill_questions = suggest_workflow_skills(self.storage)
        sync_capabilities_doc(self.plugins)
        from .local_mind import LocalMind, unload_idle_servers

        unloaded_vram = unload_idle_servers(idle_sec=120.0)
        mind = LocalMind(self.storage)
        sched = compute_adaptive_schedule(self.watcher, self.storage)
        cpu_idle = sched.get("cpu_percent", 0) <= 50.0
        batch_resolved = None
        if cpu_idle and mind.get_status().get("uncertain_queue_count", 0) >= 8:
            batch_resolved = mind.resolve_uncertain_batch(max_items=8)
        self_play = None
        if sched.get("self_play_allowed") and self._cycle_count % 2 == 1:
            self_play = run_local_self_play(self.storage, max_rounds=1)
        local_stats = mind.train(run_lora=False)
        shadow = replay_shadow_history(self.storage)
        evaluate_graduation_arena(self.storage)
        predict_user_routine(self.storage)
        self.storage.prune_and_checkpoint(stale_days=30)
        neural_world: dict[str, Any] = {}
        assoc_synced = 0
        pruned_synapses = 0
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(self.storage)
            neural_world = eng.observe_world_state(self.watcher)
            hint = str(neural_world.get("proactive_hint") or "")
            if hint == "unload_idle_vram":
                unloaded_vram = unload_idle_servers(idle_sec=30.0) or unloaded_vram
            elif hint == "ram_cleanup":
                gc.collect()
            assoc_synced = eng.sync_associative_memory()
            pruned_synapses = eng.prune_weak_synapses()
        except Exception:
            pass
        self_guard_res: dict[str, Any] = {}
        sg = getattr(self, "self_guard", None)
        if sg is not None:
            try:
                self_guard_res = {
                    "canary": sg.run_canary_probes(),
                    "profile": sg.run_memory_profiler(),
                    "calibration": sg.calibrate_thresholds(),
                    "eval_gate": sg.run_eval_gate(),
                    "drift": sg.run_drift_audit(),
                }
            except Exception:
                pass
        return {
            "rss_mb": round(rss_mb, 1),
            "cleared_cache": cleared_cache,
            "unloaded_vram": unloaded_vram,
            "schedule": sched,
            "profile": profile.get("summary"),
            "compacted_dataset": compacted_ds,
            "roadmap_count": len(roadmap),
            "healed": healed,
            "forged_macros": forged_macros,
            "auto_coverage_added": auto_cov.get("added", 0),
            "shadow": shadow,
            "neural_world": neural_world,
            "assoc_synced": assoc_synced,
            "pruned_synapses": pruned_synapses,
            "batch_resolved": batch_resolved,
            "self_play": self_play,
            "injected_skills": injected_skills,
            "hygiene": hygiene,
            "rolled_prompts": rolled_prompts,
            "consolidated": consolidated,
            "pruned": pruned,
            "skill_questions": skill_questions,
            "local_mind": local_stats,
            "self_guard": self_guard_res,
        }

    async def _loop(self) -> None:
        await asyncio.sleep(60)
        while self._running:
            try:
                await asyncio.to_thread(self.run_maintenance)
                self._cycle_count += 1
                if (
                    self.evolver is not None
                    and self._cycle_count % 3 == 0
                    and getattr(getattr(self.agent, "auth", None), "is_configured", False)
                ):
                    self.last_prompt_evolve = await self.evolver.evolve_prompts_once()
            except Exception as exc:
                self.watcher.add_event("info", "warning", f"Self-Watchdog error: {exc}")
            await asyncio.sleep(max(60, self.interval_sec))


_CODEBASE_SUMMARY_CACHE: tuple[float, str] = (0.0, "")
_STALE_CODE_MEM_PREFIXES = (
    "pending_architectur",
    "codebase_",
    "assistant_architecture",
    "assistant_memory_improvement",
    "assistant_error_handling",
    "self_improvement_engine_architecture",
    "memory_management_",
    "memory_deletion_integrated",
    "brain_memory_pool_tasks",
    "storage_memory_optimization",
    "evolve_refactoring_state",
    "error_handling_quality",
    "sysagent_memories_table_",
)


def _git_head_and_log(repo_dir: Path) -> tuple[str, list[str]]:
    try:
        import subprocess

        head_p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=3,
        )
        log_p = subprocess.run(
            ["git", "log", "-n", "5", "--oneline"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=3,
        )
        head = head_p.stdout.strip() if head_p.returncode == 0 else ""
        commits = [ln.strip() for ln in log_p.stdout.splitlines() if ln.strip()] if log_p.returncode == 0 else []
        return head, commits
    except Exception:
        return "", []


def get_live_codebase_summary(ttl_sec: float = 60.0) -> str:
    """Return a concise live snapshot of SysAgent's Python modules and recent git commits."""
    global _CODEBASE_SUMMARY_CACHE
    now = time.monotonic()
    cached_ts, cached_txt = _CODEBASE_SUMMARY_CACHE
    if cached_txt and (now - cached_ts) < ttl_sec:
        return cached_txt

    pkg_dir = Path(__file__).resolve().parent
    repo_dir = pkg_dir.parent
    modules = sorted(p.name for p in pkg_dir.glob("*.py") if p.name != "__init__.py")
    head, commits = _git_head_and_log(repo_dir)
    parts = [f"Live codebase ({pkg_dir}, HEAD={head or 'unknown'}): {', '.join(modules)}"]
    if commits:
        parts.append("Recent commits: " + " | ".join(commits))
    summary = "\n".join(parts)
    _CODEBASE_SUMMARY_CACHE = (now, summary)
    return summary


def prune_stale_codebase_memories(storage: Storage, force: bool = False) -> int:
    """Remove outdated codebase/architecture memory notes when git HEAD advances."""
    repo_dir = Path(__file__).resolve().parent.parent
    head, _ = _git_head_and_log(repo_dir)
    last_head = storage.get_memory("_internal:codebase_head") or ""
    if not force and head and head == last_head:
        return 0

    removed = 0
    for mem in storage.list_memories():
        key = str(mem.get("key") or "")
        low = key.lower()
        if (
            any(low.startswith(p) for p in _STALE_CODE_MEM_PREFIXES)
            or low.endswith(("_technical_debt", "_refactoring_state"))
        ) and storage.delete_memory(key):
            removed += 1
    if head:
        storage.set_memory("_internal:codebase_head", head)
    return removed

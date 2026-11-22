"""Curriculum-Driven Multi-Teacher Knowledge Distillation & Codebase Self-Training Mixin for LocalMind."""

from __future__ import annotations

from collections import Counter
import difflib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


class _LocalDistillMixin:
    """Mixin providing curriculum synthesis, teacher model invocation, uncertain-query resolution, and codebase training."""

    storage: Any
    weights: dict[str, Any]

    def sample_user_utterances(self, limit: int = 6) -> list[str]:
        """Sample real user messages from chat history / distilled turns so self-training adapts to the user's natural language(s)."""
        samples: list[str] = []
        if self.storage is not None:
            try:
                for s in self.storage.list_sessions(limit=8):
                    for m in self.storage.get_messages(s["session_id"], limit=10):
                        if m.get("role") == "user":
                            txt = str(m.get("content") or "").strip()
                            if txt.startswith("[cwd:"):
                                txt = txt.split("]", 1)[-1].strip()
                            if 6 <= len(txt) <= 140 and not txt.startswith("[") and txt not in samples:
                                samples.append(txt)
                                if len(samples) >= limit:
                                    return samples
            except Exception:
                pass
        for d in reversed(self._load_distilled_entries(limit=60)):  # type: ignore[attr-defined]
            u = str(d.get("user") or "").strip()
            if 6 <= len(u) <= 140 and not u.startswith(("Self-evolve", "A background", "Regarding")) and u not in samples:
                samples.append(u)
                if len(samples) >= limit:
                    break
        return samples

    def weakest_specialist_role(self) -> str:
        """Return the generative specialist role with the fewest training samples."""
        return self.underrepresented_roles()[0]

    def underrepresented_roles(self) -> list[str]:
        """Return generative specialist roles sorted ascending by sample count."""
        rc = self.weights.get("role_counts") or {}
        gen_roles = ("coding", "general", "reasoning", "chat", "multimodal")
        return sorted(gen_roles, key=lambda r: int(rc.get(r, 0)))

    def underrepresented_tools(self, limit: int = 8) -> list[str]:
        """Return declared tools sorted ascending by current training sample & centroid support."""
        from .tools import TOOL_DECLARATIONS

        centroids: dict[str, dict[str, float]] = self.weights.get("tool_centroids") or {}
        sample_counts: Counter[str] = Counter()
        for entry in self._load_distilled_entries(limit=1500):  # type: ignore[attr-defined]
            for tc in entry.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("name"):
                    sample_counts[str(tc["name"])] += 1
        all_tools = [str(d["name"]) for d in TOOL_DECLARATIONS if d.get("name")]
        ranked = sorted(
            all_tools,
            key=lambda t: (int(sample_counts.get(t, 0)), len(centroids.get(t) or {}), t),
        )
        return ranked[: max(1, limit)]

    @staticmethod
    def _build_tool_schema_map() -> dict[str, dict[str, Any]]:
        """Build a compact parameter & enum lookup table from TOOL_DECLARATIONS."""
        from .tools import TOOL_DECLARATIONS

        schema_map: dict[str, dict[str, Any]] = {}
        for d in TOOL_DECLARATIONS:
            tname = str(d.get("name") or "").strip()
            if not tname:
                continue
            params = d.get("parameters") or {}
            props = dict(params.get("properties") or {}) if isinstance(params, dict) else {}
            req = list(params.get("required") or []) if isinstance(params, dict) else []
            schema_map[tname] = {
                "description": str(d.get("description") or "")[:140],
                "properties": props,
                "required": req,
            }
        return schema_map

    @staticmethod
    def _format_tool_signatures(
        schema_map: dict[str, dict[str, Any]], tool_names: list[str]
    ) -> list[dict[str, Any]]:
        """Format compact parameter and enum signatures for teacher prompts."""
        sigs: list[dict[str, Any]] = []
        for tname in tool_names:
            spec = schema_map.get(tname)
            if not spec:
                continue
            props = spec.get("properties") or {}
            req = set(spec.get("required") or [])
            param_summary: dict[str, Any] = {}
            for pname, pdef in props.items():
                if not isinstance(pdef, dict):
                    continue
                ptype = str(pdef.get("type") or "string")
                tag = f"{ptype} (required)" if pname in req else ptype
                if isinstance(pdef.get("enum"), list) and pdef["enum"]:
                    param_summary[pname] = {"type": tag, "enum": list(pdef["enum"])}
                else:
                    param_summary[pname] = tag
            sigs.append(
                {
                    "tool": tname,
                    "description": spec.get("description", ""),
                    "params": param_summary,
                }
            )
        return sigs

    def build_distillation_curriculum(self, tool_limit: int = 10) -> dict[str, Any]:
        """Synthesize a targeted distillation curriculum from coverage gaps, lessons, and user language."""
        schema_map = self._build_tool_schema_map()
        weak_roles = self.underrepresented_roles()
        weak_tools = self.underrepresented_tools(limit=tool_limit)
        recent_lessons: list[dict[str, str]] = []
        if self.storage is not None:
            try:
                for les in self.storage.list_lessons(limit=12):
                    ctx = str(les.get("context") or les.get("topic") or "")
                    if ctx.startswith("_internal:"):
                        continue
                    recent_lessons.append(
                        {
                            "context": ctx[:80],
                            "insight": str(les.get("insight") or les.get("mistake") or "")[:120],
                            "resolution": str(les.get("resolution") or les.get("lesson") or "")[:140],
                        }
                    )
                    if len(recent_lessons) >= 5:
                        break
            except Exception:
                pass
        core_tools = [
            "inspect_self_component",
            "system_status",
            "list_directory",
            "read_file",
            "search_files",
            "run_command",
        ]
        featured = list(dict.fromkeys([*weak_tools, *core_tools]))
        return {
            "weak_roles": weak_roles,
            "underrepresented_tools": weak_tools,
            "recent_lessons": recent_lessons,
            "user_examples": self.sample_user_utterances(limit=5),
            "tool_signatures": self._format_tool_signatures(schema_map, featured),
        }

    def _validate_distilled_sample(
        self,
        smp: dict[str, Any],
        schema_map: dict[str, dict[str, Any]],
        seen_queries: set[str],
    ) -> dict[str, Any] | None:
        """Validate, normalize, and deduplicate a teacher-generated sample against tool schemas."""
        from . import local_mind as _lm

        if not isinstance(smp, dict):
            return None
        user = str(smp.get("user") or "").strip()
        reply = str(smp.get("reply") or "").strip()
        if len(user) < 5 or len(reply) < 4:
            return None
        u_key = " ".join(user.lower().split())
        if u_key in seen_queries:
            return None

        raw_chain = smp.get("tool_calls")
        raw_single_tool = smp.get("tool")
        expecting_tool = False
        candidate_calls: list[tuple[str, dict[str, Any]]] = []
        if isinstance(raw_chain, list) and raw_chain:
            expecting_tool = True
            for item in raw_chain[:4]:
                if not isinstance(item, dict):
                    continue
                cname = str(item.get("name") or item.get("tool") or "").strip()
                cargs = item.get("arguments")
                if not isinstance(cargs, dict):
                    cargs = item.get("args") if isinstance(item.get("args"), dict) else {}
                if cname:
                    candidate_calls.append((cname, dict(cargs)))
        elif raw_single_tool is not None:
            t_str = str(raw_single_tool).strip()
            if t_str and t_str.lower() not in {"null", "none", "__chat__"}:
                expecting_tool = True
                raw_args = smp.get("args")
                if not isinstance(raw_args, dict):
                    raw_args = smp.get("arguments") if isinstance(smp.get("arguments"), dict) else {}
                candidate_calls.append((t_str, dict(raw_args)))

        valid_tcalls: list[dict[str, Any]] = []
        for tname, raw_args in candidate_calls:
            spec = schema_map.get(tname)
            if spec is None:
                continue
            props: dict[str, Any] = spec.get("properties") or {}
            req: list[str] = spec.get("required") or []
            norm_args: dict[str, Any] = {k: v for k, v in raw_args.items() if k in props}

            if len(req) == 1 and req[0] not in norm_args:
                unknown_vals = [
                    v for k, v in raw_args.items() if k not in props and v is not None and str(v).strip() != ""
                ]
                if len(unknown_vals) == 1:
                    norm_args[req[0]] = unknown_vals[0]

            for k, val in list(norm_args.items()):
                pdef = props.get(k)
                if not isinstance(pdef, dict):
                    continue
                enum_vals = pdef.get("enum")
                if isinstance(enum_vals, list) and enum_vals and val not in enum_vals:
                    if isinstance(val, str):
                        enum_map = {str(e).lower(): e for e in enum_vals}
                        v_low = val.strip().lower()
                        if v_low in enum_map:
                            norm_args[k] = enum_map[v_low]
                        else:
                            close = difflib.get_close_matches(v_low, list(enum_map.keys()), n=1, cutoff=0.6)
                            if close:
                                norm_args[k] = enum_map[close[0]]
                            else:
                                norm_args.pop(k, None)
                    else:
                        norm_args.pop(k, None)

            if any(rk not in norm_args or norm_args[rk] in {None, ""} for rk in req):
                continue
            valid_tcalls.append({"name": tname, "arguments": norm_args})

        if expecting_tool and not valid_tcalls:
            return None

        raw_role = str(smp.get("role") or "").strip().lower()
        valid_roles = {"coding", "general", "reasoning", "chat", "multimodal"}
        role = (
            raw_role
            if raw_role in valid_roles
            else _lm.classify_task_role(user, valid_tcalls, storage=self.storage)
        )
        seen_queries.add(u_key)
        return {
            "user": user,
            "tool_calls": valid_tcalls,
            "reply": reply,
            "role": role,
        }

    def _call_teacher_model(
        self,
        teacher: str,
        prompt: str,
        caller: str = "distill",
        allow_local_fallback: bool = True,
    ) -> tuple[str | None, str]:
        """Invoke a teacher model via agy CLI, AuthManager SDK client, or local 9B specialist peer fallback."""
        from . import local_mind as _lm
        from .auth import AuthManager, _find_agy_cli
        from .self_resilience import get_quota_governor

        gov = get_quota_governor()
        recorded_bg = False
        agy_bin = _find_agy_cli()
        if agy_bin:
            try:
                gov.record_background_call(caller, teacher)
                recorded_bg = True
                proc = subprocess.run(
                    [
                        agy_bin,
                        "-p",
                        prompt,
                        "--model",
                        teacher,
                        "--disable-slash-commands",
                        "--output-format",
                        "text",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    cwd="/tmp",
                )
                if proc.returncode == 0 and "{" in proc.stdout:
                    return proc.stdout.strip(), teacher
                err_txt = f"{proc.stderr} {proc.stdout}".lower()
                if any(k in err_txt for k in ("429", "quota", "resource_exhausted")):
                    gov.record_rate_limit(teacher)
                    return None, teacher
            except Exception:
                pass

        try:
            auth = AuthManager()
            if auth.is_configured:
                client = auth.get_client()
                if type(client).__name__ not in {"LocalMindClient", "AgyCliClient"}:
                    if not recorded_bg:
                        gov.record_background_call(caller, teacher)
                        recorded_bg = True
                    resp = client.interactions.create(model=teacher, input=prompt, store=False)
                    out_txt = getattr(resp, "output_text", None)
                    if isinstance(out_txt, str) and "{" in out_txt:
                        return out_txt.strip(), teacher
        except Exception as exc:
            err_low = str(exc).lower()
            if any(k in err_low for k in ("429", "quota", "resource_exhausted")):
                gov.record_rate_limit(teacher)
                return None, teacher

        if allow_local_fallback:
            models = [m for m in _lm.find_all_gguf_models() if _lm.classify_gguf_role(m) != "embedding"]
            if models:
                try:
                    local_out = self.run_gguf_inference(  # type: ignore[attr-defined]
                        prompt,
                        extra_system=(
                            "You are a Teacher Model generating training data for a Linux system assistant. "
                            "Respond ONLY with valid JSON matching the requested schema."
                        ),
                        max_tokens=600,
                        prefer_warm=True,
                    )
                    if isinstance(local_out, str) and "{" in local_out:
                        return local_out.strip(), "local-peer"
                except Exception:
                    pass
        return None, teacher

    def _record_uncertain_query(self, user_text: str, score: float) -> None:
        """Queue low-confidence user queries so multi-teacher distillation can solve them online."""
        from . import local_mind as _lm

        clean = (user_text or "").strip()
        if len(clean) < 4 or clean.startswith("["):
            return
        try:
            with _lm.UNCERTAIN_FILE.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "query": clean[:300],
                            "score": round(score, 3),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass

    def resolve_uncertain_batch(
        self, max_items: int = 8, allow_local_fallback: bool = True
    ) -> dict[str, Any]:
        """Resolve up to max_items queued uncertain queries in ONE batch teacher call and update weights."""
        from . import local_mind as _lm
        from .auth import AuthManager, _find_agy_cli
        from .safety import SAFE_TOOLS
        from .self_resilience import get_quota_governor

        gov = get_quota_governor()
        allowed_q, q_reason = gov.allow_background("resolve_uncertain")
        if not allowed_q:
            return {"status": "deferred", "reason": q_reason}

        if not _lm.UNCERTAIN_FILE.is_file():
            return {"status": "skipped", "resolved": 0, "remaining": 0}
        try:
            raw_lines = [ln for ln in _lm.UNCERTAIN_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            return {"status": "skipped", "resolved": 0, "remaining": 0}
        if not raw_lines:
            return {"status": "skipped", "resolved": 0, "remaining": 0}

        batch_queries: list[str] = []
        consumed_indices: set[int] = set()
        seen_q: set[str] = set()
        for idx, ln in enumerate(raw_lines):
            try:
                q = str(json.loads(ln).get("query") or "").strip()
            except Exception:
                consumed_indices.add(idx)
                continue
            if not q:
                consumed_indices.add(idx)
                continue
            if q in seen_q:
                consumed_indices.add(idx)
                continue
            if len(batch_queries) < max_items:
                batch_queries.append(q)
                seen_q.add(q)
                consumed_indices.add(idx)

        remaining_lines = [ln for i, ln in enumerate(raw_lines) if i not in consumed_indices]
        if not batch_queries:
            try:
                _lm.UNCERTAIN_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            return {"status": "skipped", "resolved": 0, "remaining": 0}

        has_local = allow_local_fallback and any(
            _lm.classify_gguf_role(m) != "embedding" for m in _lm.find_all_gguf_models()
        )
        has_sdk = False
        try:
            auth = AuthManager()
            has_sdk = bool(
                auth.is_configured
                and type(auth.get_client()).__name__ not in {"LocalMindClient", "AgyCliClient"}
            )
        except Exception:
            pass
        if not _find_agy_cli() and not has_sdk and not has_local:
            return {"status": "skipped", "resolved": 0, "remaining": len(raw_lines)}

        schema_map = self._build_tool_schema_map()
        tool_sigs = self._format_tool_signatures(schema_map, list(schema_map.keys()))
        prompt = (
            "Map each user query below to the best tool(s) + schema-valid arguments and a concise plain-text reply.\n"
            f"Tool schemas: {json.dumps(tool_sigs, ensure_ascii=False)}\n"
            f"Queries: {json.dumps(batch_queries, ensure_ascii=False)}\n"
            "Respond ONLY with JSON:\n"
            '{"samples": [{"user": "...", "role": "general|coding|reasoning|chat|multimodal", '
            '"tool": "tool_or_null", "args": {}, '
            '"tool_calls": [{"name": "tool_name", "arguments": {}}], "reply": "..."}]}'
        )
        resolved = 0
        seen_existing: set[str] = set()
        out, used_teacher = self._call_teacher_model(
            _lm.TEACHER_MODELS[0],
            prompt,
            caller="resolve_uncertain",
            allow_local_fallback=allow_local_fallback,
        )
        if not out:
            return {"status": "error", "resolved": 0, "remaining": len(raw_lines)}
        try:
            j_idx = out.find("{")
            if j_idx == -1:
                return {"status": "error", "resolved": 0, "remaining": len(raw_lines)}
            parsed, _ = json.JSONDecoder().raw_decode(out[j_idx:])
            samples = parsed.get("samples") if isinstance(parsed, dict) else None
            if not isinstance(samples, list):
                return {"status": "error", "resolved": 0, "remaining": len(raw_lines)}
            for smp in samples:
                validated = self._validate_distilled_sample(smp, schema_map, seen_existing)
                if validated is None:
                    continue
                tcalls = validated["tool_calls"]
                self.record_and_learn(  # type: ignore[attr-defined]
                    validated["user"],
                    tcalls,
                    validated["reply"],
                    role=validated["role"],
                )
                if len(tcalls) >= 2 and self.storage is not None:
                    try:
                        self.storage.record_workflow(validated["user"][:140], tcalls)
                    except Exception:
                        pass
                elif len(tcalls) == 1 and tcalls[0]["name"] in SAFE_TOOLS:
                    try:
                        from .neural import NeuralEngine

                        NeuralEngine(self.storage).record_safe_plan(
                            validated["user"],
                            tcalls[0]["name"],
                            dict(tcalls[0].get("arguments") or {}),
                        )
                    except Exception:
                        pass
                resolved += 1
        except Exception:
            return {"status": "error", "resolved": 0, "remaining": len(raw_lines)}

        try:
            if remaining_lines:
                _lm.UNCERTAIN_FILE.write_text("\n".join(remaining_lines) + "\n", encoding="utf-8")
            else:
                _lm.UNCERTAIN_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        if resolved:
            self.train()  # type: ignore[attr-defined]
        return {
            "status": "ok",
            "resolved": resolved,
            "remaining": len(remaining_lines),
            "teacher": used_teacher,
        }

    def distill_from_teachers(
        self, count_per_teacher: int = 2, allow_local_fallback: bool = True
    ) -> dict[str, Any]:
        """Curriculum-Driven Multi-Teacher Knowledge Distillation with schema validation & active verification."""
        from . import local_mind as _lm
        from .auth import AuthManager, _find_agy_cli
        from .safety import SAFE_TOOLS
        from .self_resilience import get_quota_governor

        gov = get_quota_governor()
        allowed_q, q_reason = gov.allow_background("distill")
        if not allowed_q:
            return {"status": "deferred", "reason": q_reason}

        has_local = allow_local_fallback and any(
            _lm.classify_gguf_role(m) != "embedding" for m in _lm.find_all_gguf_models()
        )
        has_sdk = False
        try:
            auth = AuthManager()
            has_sdk = bool(
                auth.is_configured
                and type(auth.get_client()).__name__ not in {"LocalMindClient", "AgyCliClient"}
            )
        except Exception:
            pass
        if not _find_agy_cli() and not has_sdk and not has_local:
            return {"status": "skipped", "reason": "agy_cli_not_found"}

        uncertain_queries: list[str] = []
        raw_uncertain_lines: list[str] = []
        if _lm.UNCERTAIN_FILE.is_file():
            try:
                raw_uncertain_lines = [
                    ln for ln in _lm.UNCERTAIN_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()
                ]
                for ln in raw_uncertain_lines[-20:]:
                    q = json.loads(ln).get("query")
                    if q and q not in uncertain_queries:
                        uncertain_queries.append(str(q))
            except Exception:
                pass
        resolved_uncertain: set[str] = set()

        schema_map = self._build_tool_schema_map()
        curriculum = self.build_distillation_curriculum(tool_limit=12)
        weak_roles: list[str] = curriculum["weak_roles"]
        weak_tools: list[str] = curriculum["underrepresented_tools"]
        recent_lessons: list[dict[str, str]] = curriculum["recent_lessons"]
        user_examples: list[str] = curriculum["user_examples"]

        seen_queries: set[str] = {
            " ".join(str(e.get("user") or "").lower().split())
            for e in self._load_distilled_entries(limit=1500)  # type: ignore[attr-defined]
            if e.get("user")
        }

        harvested = 0
        rejected = 0
        multi_step_chains = 0
        teachers_used: list[str] = []
        harvested_records: list[dict[str, Any]] = []

        role_guidance = {
            "multimodal": "screenshot/image analysis, OCR, UI/visual inspection, or media file queries",
            "reasoning": "step-by-step debugging, self-component introspection, root-cause diagnosis, or architecture analysis",
            "coding": "Python/Bash code debugging, file search -> read -> edit workflows, git diff review, or test execution",
            "general": "Linux system inspection, systemd services, process/RAM/disk diagnostics, or memory search",
            "chat": "brief conversational greetings, status check-ins, or concise assistant Q&A",
        }

        for t_idx, teacher in enumerate(_lm.TEACHER_MODELS):
            ok_q, _ = gov.allow_background("distill")
            if not ok_q:
                break
            focus_role = weak_roles[t_idx % len(weak_roles)]
            focus_tools = weak_tools[t_idx :: len(_lm.TEACHER_MODELS)][:4] or weak_tools[:4]
            core_slice = ["inspect_self_component", "list_directory", "read_file", "search_files", "system_status"]
            teacher_tool_names = list(dict.fromkeys([*focus_tools, *core_slice]))
            teacher_sigs = self._format_tool_signatures(schema_map, teacher_tool_names)

            prompt = (
                "You are a Teacher Model generating high-quality training examples for a local Linux system assistant.\n"
                f"Priority tool schemas (use exact tool names, required params, and enum values): {json.dumps(teacher_sigs, ensure_ascii=False)}\n"
                f"All valid tool names: {json.dumps(list(schema_map.keys()))}\n"
            )
            if recent_lessons:
                prompt += (
                    f"Recent mistakes/corrections to avoid and teach against: {json.dumps(recent_lessons, ensure_ascii=False)}\n"
                )
            batch_q: list[str] = []
            if uncertain_queries:
                batch_q = uncertain_queries[:count_per_teacher]
                uncertain_queries = uncertain_queries[count_per_teacher:]
                prompt += (
                    f"Solve these real uncertain user queries with accurate tool calls and concise replies: "
                    f"{json.dumps(batch_q, ensure_ascii=False)}\n"
                )
            else:
                lang_hint = (
                    f"Match the natural language(s) observed in the user's real history ({json.dumps(user_examples, ensure_ascii=False)}) as well as English"
                    if user_examples
                    else "in Turkish and English"
                )
                chain_hint = (
                    "Include at least 1 multi-step workflow sample with 2 ordered calls in `tool_calls` "
                    "(e.g. list_directory -> read_file, search_files -> read_file, or system_status -> list_processes).\n"
                    if count_per_teacher >= 2 and focus_role != "chat"
                    else ""
                )
                prompt += (
                    f"Generate {count_per_teacher} diverse, realistic user commands specifically exercising the "
                    f"'{focus_role}' specialist role ({lang_hint}), focusing on "
                    f"{role_guidance.get(focus_role, 'Linux system assistance')} and priority tools {json.dumps(focus_tools)}.\n"
                    f"{chain_hint}"
                )
            prompt += (
                "Respond ONLY with a valid JSON object matching:\n"
                '{"samples": [{"user": "user request", "role": "'
                + focus_role
                + '", "tool": "primary_tool_or_null", "args": {}, '
                '"tool_calls": [{"name": "tool_name", "arguments": {}}], '
                '"reply": "concise plain text answer without backticks"}]}'
            )

            out, used_teacher = self._call_teacher_model(
                teacher,
                prompt,
                caller="distill",
                allow_local_fallback=allow_local_fallback,
            )
            if not out:
                continue
            try:
                idx = out.find("{")
                if idx == -1:
                    continue
                parsed, _ = json.JSONDecoder().raw_decode(out[idx:])
                samples = parsed.get("samples") if isinstance(parsed, dict) else None
                if not isinstance(samples, list):
                    continue
                if used_teacher not in teachers_used:
                    teachers_used.append(used_teacher)
                resolved_uncertain.update(batch_q)
                for smp in samples:
                    validated = self._validate_distilled_sample(smp, schema_map, seen_queries)
                    if validated is None:
                        rejected += 1
                        continue
                    tcalls = validated["tool_calls"]
                    self.record_and_learn(  # type: ignore[attr-defined]
                        validated["user"],
                        tcalls,
                        validated["reply"],
                        role=validated["role"],
                    )
                    if len(tcalls) >= 2:
                        multi_step_chains += 1
                        if self.storage is not None:
                            try:
                                self.storage.record_workflow(validated["user"][:140], tcalls)
                            except Exception:
                                pass
                    elif len(tcalls) == 1 and tcalls[0]["name"] in SAFE_TOOLS:
                        try:
                            from .neural import NeuralEngine

                            NeuralEngine(self.storage).record_safe_plan(
                                validated["user"],
                                tcalls[0]["name"],
                                dict(tcalls[0].get("arguments") or {}),
                            )
                        except Exception:
                            pass
                    harvested += 1
                    harvested_records.append(validated)
            except Exception:
                continue
            if used_teacher == "local-peer":
                break

        if raw_uncertain_lines and resolved_uncertain:
            remaining: list[str] = []
            for ln in raw_uncertain_lines:
                try:
                    q_val = str(json.loads(ln).get("query") or "")
                except Exception:
                    continue
                if q_val and q_val not in resolved_uncertain:
                    remaining.append(ln)
            try:
                if remaining:
                    _lm.UNCERTAIN_FILE.write_text("\n".join(remaining) + "\n", encoding="utf-8")
                else:
                    _lm.UNCERTAIN_FILE.unlink(missing_ok=True)
            except Exception:
                pass

        status = self.train()  # type: ignore[attr-defined]

        tool_records = [r for r in harvested_records if r.get("tool_calls")]
        verified_hits = 0
        for rec in tool_records:
            expected_tool = str(rec["tool_calls"][0]["name"])
            pred_tool, _, _ = self.predict_tool(rec["user"], record_uncertain=False)  # type: ignore[attr-defined]
            if pred_tool == expected_tool:
                verified_hits += 1
            else:
                self._update_weights_incremental(rec)  # type: ignore[attr-defined]
                try:
                    from .neural import NeuralEngine

                    NeuralEngine(self.storage).online_step(rec["user"], expected_tool, reward=1.5)
                except Exception:
                    pass
                pred_after, _, _ = self.predict_tool(rec["user"], record_uncertain=False)  # type: ignore[attr-defined]
                if pred_after == expected_tool:
                    verified_hits += 1

        post_verify_acc = (
            round(verified_hits / len(tool_records), 4) if tool_records else 1.0
        )
        return {
            "status": "ok",
            "harvested_samples": harvested,
            "rejected_samples": rejected,
            "multi_step_chains": multi_step_chains,
            "resolved_uncertain_count": len(resolved_uncertain),
            "teachers_used": teachers_used,
            "curriculum_focus": {
                "weak_roles": weak_roles[:3],
                "underrepresented_tools": weak_tools[:6],
                "lessons_targeted": len(recent_lessons),
            },
            "post_verify_accuracy": post_verify_acc,
            "model_status": status,
        }

    def train_on_own_codebase(self, force: bool = False) -> dict[str, Any]:
        """AST-scan SysAgent's own Python modules and refresh self-codebase training samples."""
        import ast
        import hashlib
        from . import local_mind as _lm

        pkg_dir = Path(__file__).resolve().parent
        py_files = sorted(p for p in pkg_dir.glob("*.py") if p.name != "__init__.py")
        sig_parts = [f"{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in py_files if p.is_file()]
        cb_hash = hashlib.sha256("|".join(sig_parts).encode("utf-8")).hexdigest()[:16]
        if not force and self.weights.get("codebase_train_hash") == cb_hash:
            return {"status": "up_to_date", "modules": len(py_files), "samples": 0}

        new_records: list[dict[str, Any]] = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for mod_path in py_files:
            try:
                tree = ast.parse(mod_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            doc = (ast.get_docstring(tree) or "").splitlines()[0].strip() if ast.get_docstring(tree) else ""
            syms = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
            ][:8]
            sym_txt = ", ".join(syms) if syms else "internal helpers"
            stem = mod_path.stem
            reply = f"{mod_path.name} ({mod_path}): {doc or stem} Key symbols: {sym_txt}."
            new_records.append(
                {
                    "ts": ts,
                    "user": f"inspect sysagent {mod_path.name} {stem} {sym_txt}",
                    "tool_calls": [{"name": "read_file", "arguments": {"path": str(mod_path)}}],
                    "reply": reply,
                    "role": "coding",
                    "source": "self_codebase",
                }
            )

        kept_lines: list[str] = []
        if _lm.DISTILLED_FILE.is_file():
            for ln in _lm.DISTILLED_FILE.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                try:
                    row = json.loads(ln)
                    if row.get("source") == "self_codebase":
                        continue
                except Exception:
                    pass
                kept_lines.append(ln)

        for rec in new_records:
            kept_lines.append(json.dumps(rec, ensure_ascii=False))
        _lm.DISTILLED_FILE.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")

        self.weights["codebase_train_hash"] = cb_hash
        status = self.train()  # type: ignore[attr-defined]
        self.weights["codebase_train_hash"] = cb_hash
        _lm._save_weights_file(self.weights)
        return {
            "status": "ok",
            "modules": len(py_files),
            "samples": len(new_records),
            "codebase_hash": cb_hash,
            "model_status": status,
        }

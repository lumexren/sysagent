"""Gemini Interactions API agent loop with Smart Approval."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Awaitable, Callable

from google import genai

from .auth import AuthManager
from .config import Settings
from .i18n import t
from .plugins import PluginRegistry
from .prompts import DEFAULT_PROMPTS, PromptRegistry
from .safety import SAFE_TOOLS, assess_tool_call, compute_dry_run_preview, is_auto_trust_allowed
from .self_ops import ToolCache, choose_model, detect_style_feedback, scrub_secrets, select_relevant_tools, truncate_tool_output
from .storage import Storage
from .tools import TOOL_DECLARATIONS, execute_tool, get_all_tool_declarations
from .watcher import SystemWatcher


BASE_SYSTEM_INSTRUCTION = DEFAULT_PROMPTS["agent_main"]["template"]


def build_system_instruction(
    locale: str,
    agent_name: str = "SysAgent",
    memories: list[dict[str, Any]] | None = None,
    kg_edges: list[dict[str, Any]] | None = None,
    workspace_context: str = "normal",
    lessons: list[dict[str, Any]] | None = None,
    storage: Storage | None = None,
    workflows: list[dict[str, Any]] | None = None,
    recent_history: list[dict[str, Any]] | None = None,
) -> str:
    """Build the English system instruction with agent name, memories, KG, lessons, workflows, history, context, and locale."""
    name = (agent_name or "SysAgent").strip()
    base = PromptRegistry(storage).render("agent_main", agent_name=name)
    base += f"\nCurrent workspace context: {workspace_context}\n"
    if memories:
        mem_lines = "\n".join(f"- {m['key']}: {m['value']}" for m in memories[:15])
        base += f"\nLong-term memories:\n{mem_lines}\n"
    if kg_edges:
        kg_lines = "\n".join(
            f"- {e['subject']} -{e['relation']}-> {e['object']}"
            for e in kg_edges[:12]
        )
        base += f"\nKnowledge Graph:\n{kg_lines}\n"
    if lessons:
        les_lines = "\n".join(
            f"- {l['context']}: {l['insight']} -> {l['resolution']}"
            for l in lessons[:8]
        )
        base += f"\nEpisodic Lessons Learned:\n{les_lines}\n"
    if workflows:
        wf_lines = "\n".join(
            f"- For '{wf['goal_pattern']}' ({wf['hits']}x): {wf['signature']}"
            for wf in workflows[:3]
        )
        base += f"\nLearned Tool Workflows:\n{wf_lines}\n"
    if recent_history:
        tail = recent_history[-12:]
        hist_lines = "\n".join(
            f"- [{h.get('created_at', '')}] {h['role']}: "
            f"{str(h['content'])[:(1000 if idx >= len(tail) - 4 else 400)]}"
            for idx, h in enumerate(tail)
        )
        base += f"\nRecent conversation context (prior turns/sessions):\n{hist_lines}\n"
    loc = (locale or "auto").strip()
    if loc.lower() == "auto":
        lang_rule = (
            "Language rule: Respond in the same language as the user's message, "
            "unless the user explicitly asks for a different language."
        )
    else:
        lang_rule = (
            f"Language rule: Always respond in locale {loc}, "
            "unless the user explicitly asks for another language."
        )
    return f"{base}\n{lang_rule}"

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class PendingApproval:
    """Represents a risky tool call awaiting user approval."""

    approval_id: str
    session_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    risk_level: str
    future: asyncio.Future[bool]


class AgentManager:
    """Manages multi-turn Gemini Interactions sessions and tool execution."""

    def __init__(
        self,
        settings: Settings,
        watcher: SystemWatcher,
        storage: Storage,
        plugins: PluginRegistry | None = None,
        auth: AuthManager | None = None,
    ) -> None:
        self.settings = settings
        self.watcher = watcher
        self.storage = storage
        self.plugins = plugins
        self.auth = auth or AuthManager(settings)
        self.sessions: dict[str, str] = {}  # session_id -> previous_interaction_id
        self._session_models: dict[str, str] = {}
        self._pinned_models: dict[str, str] = {}
        self.pending_approvals: dict[str, PendingApproval] = {}
        self._tool_cache = ToolCache(ttl_sec=120.0)
        self._base_confidence = settings.brain_confidence_threshold
        self._last_turn_info: dict[str, tuple[str, list[dict[str, Any]]]] = {}
        self._bg_learning_tasks: set[asyncio.Task[None]] = set()

    def _match_zero_llm_fastpath(self, user_message: str) -> tuple[str, dict[str, Any]] | None:
        """Match short, unambiguous read-only queries to execute directly with 0 LLM tokens."""
        clean = user_message.strip()
        if clean.startswith("[cwd:"):
            idx = clean.find("]")
            if idx != -1:
                clean = clean[idx + 1:].strip()
        low = clean.lower()
        words = low.split()
        if not words or len(words) > 7:
            return None
        block_cues = (
            "why", "fix", "add", "write", "edit",
            "code", "analyze", "compare", "delete",
        )
        if any(c in low for c in block_cues):
            return None
        if any(k in low for k in ("system status", "ram usage", "cpu usage", "disk usage", "disk space", "temperature")):
            return "system_status", {}
        if any(k in low for k in ("top processes", "memory hogs", "cpu hogs", "running processes")):
            return "list_processes", {"limit": 10}
        if any(k in low for k in ("recent logs", "service logs", "journal logs")):
            return "read_journal_logs", {"unit": "sysagent", "user_unit": True, "lines": 30}
        try:
            from .local_mind import LocalMind
            from .self_learn import shadow_stats

            tool, _, score = LocalMind(self.storage).predict_tool(clean)
            graduated = set(shadow_stats(self.storage).get("graduated", []))
            min_score = 0.55 if tool in graduated else 0.85
            if score >= min_score and tool in {"system_status", "list_processes", "read_journal_logs"}:
                args = {"limit": 10} if tool == "list_processes" else ({"unit": "sysagent", "user_unit": True, "lines": 30} if tool == "read_journal_logs" else {})
                return tool, args
        except Exception:
            pass
        return None

    def _get_client(self) -> genai.Client:
        return self.auth.get_client()

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        """Resolve a pending approval future. Returns True if found."""
        pending = self.pending_approvals.pop(approval_id, None)
        if pending and not pending.future.done():
            pending.future.set_result(approved)
            return True
        return False

    def reset_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self._session_models.pop(session_id, None)
        self._pinned_models.pop(session_id, None)
        for aid, pending in list(self.pending_approvals.items()):
            if pending.session_id == session_id:
                self.pending_approvals.pop(aid, None)
                if not pending.future.done():
                    pending.future.set_result(False)
        self.storage.delete_session(session_id)

    def _raise_caution(self) -> None:
        """Self-calibration: a denial/undo means the agent misjudged; be more careful."""
        self.settings.brain_confidence_threshold = min(95, self.settings.brain_confidence_threshold + 3)
        PromptRegistry(self.storage).record_outcome("agent_main", False)
        if self.settings.brain_confidence_threshold >= 92:
            PromptRegistry(self.storage).rollback_prompt("agent_main")

    def _relax_caution(self) -> None:
        """Approved risky actions slowly bring the threshold back toward its configured base."""
        if self.settings.brain_confidence_threshold > self._base_confidence:
            self.settings.brain_confidence_threshold -= 1

    async def _post_turn_learn(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        tool_calls: list[dict[str, Any]],
        had_failure: bool,
        prev_turn: tuple[str, list[dict[str, Any]]] | None,
    ) -> None:
        """Background post-turn learner: corrections, multi-step workflows, memories, KG, lessons, and brain rules."""
        try:
            low = (user_message or "").lower()
            dyn_cues = self.storage.get_learned_cues()
            correction_cues = dyn_cues.get("correction") or []
            if prev_turn and correction_cues and any(cue in low for cue in correction_cues):
                prev_msg, prev_tools = prev_turn
                prev_tool_names = ", ".join(t["name"] for t in prev_tools if t.get("name")) or "direct_reply"
                self.storage.add_lesson(
                    f"User correction ({prev_tool_names})",
                    f"Attempted on: {prev_msg[:120]}",
                    user_message[:250],
                )
                if prev_tools and prev_tools[0].get("name"):
                    from .local_mind import LocalMind

                    LocalMind(self.storage).penalize_tool(prev_msg, str(prev_tools[0]["name"]))
                try:
                    from .neural import NeuralEngine

                    NeuralEngine(self.storage).sync_associative_memory()
                except Exception:
                    pass

            stall_user_cues = dyn_cues.get("stall_user") or []
            if stall_user_cues and any(c in low for c in stall_user_cues):
                msgs = self.storage.get_messages(session_id, limit=6)
                prev_asst = next((m["content"] for m in reversed(msgs[:-1]) if m.get("role") == "assistant"), "")
                if prev_asst:
                    from .neural import NeuralEngine

                    NeuralEngine(self.storage).online_stall_step(prev_asst, is_stall=True)
                    self.storage.add_lesson("task_stall", prev_asst[:160], user_message[:180])
                    PromptRegistry(self.storage).record_outcome("agent_main", False)
            elif not had_failure and tool_calls and assistant_reply:
                from .neural import NeuralEngine

                NeuralEngine(self.storage).online_stall_step(assistant_reply, is_stall=False)

            if not had_failure and tool_calls:
                from .self_learn import record_shadow_evaluation

                record_shadow_evaluation(self.storage, user_message, tool_calls)
            if not had_failure and len(tool_calls) >= 2:
                self.storage.record_workflow(user_message[:140], tool_calls)

            if (
                len(user_message.strip()) < 18
                or assistant_reply.startswith("[Local Offline Mode]")
                or not self.auth.is_configured
                or self.auth.status.get("mode") == "local_mind"
            ):
                return

            from .brain import _extract_json

            extract_sys = (
                "You are a silent background knowledge extractor for a Linux assistant.\n"
                "Extract ONLY durable, non-trivial facts, entity relations, technical lessons, or user policies "
                "from this chat turn. If nothing durable was shared, return empty arrays.\n"
                "Respond ONLY with valid JSON matching:\n"
                '{"memories": [{"key": "short_snake_key", "value": "fact"}], '
                '"kg": [{"subject": "entity", "relation": "rel", "object": "target"}], '
                '"lessons": [{"context": "ctx", "insight": "ins", "resolution": "res"}], '
                '"brain_rules": [{"pattern": "condition/trigger", "action_policy": "what to do"}]}'
            )
            payload = json.dumps(
                {
                    "user": user_message[:1000],
                    "tools_used": [t.get("name") for t in tool_calls[:5]],
                    "assistant": assistant_reply[:1200],
                },
                ensure_ascii=False,
            )
            resp = await self.auth.call_interactions(
                self.storage,
                model=self.settings.light_model,
                timeout_sec=35.0,
                allow_light_fallback=False,
                input=payload,
                system_instruction=extract_sys,
                store=False,
            )
            parsed = _extract_json(getattr(resp, "output_text", "") or "{}")
            if not isinstance(parsed, dict):
                return

            for m in (parsed.get("memories") or [])[:4]:
                if isinstance(m, dict) and m.get("key") and m.get("value"):
                    k = str(m["key"]).strip()
                    if not k.startswith(("style:", "_internal:")):
                        self.storage.set_memory(k, str(m["value"]).strip())
            for k in (parsed.get("kg") or [])[:4]:
                if isinstance(k, dict) and k.get("subject") and k.get("relation") and k.get("object"):
                    self.storage.add_kg_edge(str(k["subject"]), str(k["relation"]), str(k["object"]))
            for ls in (parsed.get("lessons") or [])[:2]:
                if isinstance(ls, dict) and ls.get("context") and ls.get("insight") and ls.get("resolution"):
                    self.storage.add_lesson(str(ls["context"]), str(ls["insight"]), str(ls["resolution"]))
            for br in (parsed.get("brain_rules") or [])[:2]:
                if isinstance(br, dict) and br.get("pattern") and br.get("action_policy"):
                    self.storage.add_brain_rule(str(br["pattern"]), str(br["action_policy"]), confidence=85)
        except Exception:
            pass

    @staticmethod
    def _unchain_kwargs(call_kwargs: dict[str, Any]) -> None:
        """Remove previous_interaction_id and coerce function_result lists to text so unchained calls succeed."""
        call_kwargs.pop("previous_interaction_id", None)
        inp = call_kwargs.get("input")
        if isinstance(inp, list) and inp and all(isinstance(x, dict) and x.get("type") == "function_result" for x in inp):
            parts = [
                f"[Tool `{x.get('name', 'tool')}` result]:\n{x.get('result', '')}"
                for x in inp
            ]
            call_kwargs["input"] = (
                "The previous interaction chain could not be resumed. Here are the tool execution results "
                "for the current turn; please provide the final response or next tool calls:\n\n"
                + "\n\n".join(parts)
            )

    @staticmethod
    async def _create_safe(client: Any, kwargs: dict[str, Any]) -> Any:
        """Call interactions.create, dropping synthetic chain IDs on backend switches."""
        call_kwargs = dict(kwargs)
        prev = str(call_kwargs.get("previous_interaction_id") or "")
        is_synthetic_client = type(client).__name__ in {"AgyCliClient", "LocalMindClient"}
        if prev and not is_synthetic_client and prev.startswith(("agy-", "local-")):
            AgentManager._unchain_kwargs(call_kwargs)
            prev = ""
        elif not prev and isinstance(call_kwargs.get("input"), list):
            AgentManager._unchain_kwargs(call_kwargs)
        try:
            return await asyncio.to_thread(client.interactions.create, **call_kwargs)
        except Exception as exc:
            low = str(exc).lower()
            if prev and ("400" in low or "invalid_request" in low or "invalid argument" in low):
                AgentManager._unchain_kwargs(call_kwargs)
                return await asyncio.to_thread(client.interactions.create, **call_kwargs)
            raise

    async def _call_model_with_failover(self, session_id: str, **kwargs: Any) -> Any:
        """Call Gemini Interactions API with key rotation, light_model fallback, and offline LocalMind fallback."""
        client = self._get_client()
        model = str(kwargs["model"])
        if model in {"local", "local-mind", "local_mind"}:
            from .local_mind import LocalMindClient

            local_client = LocalMindClient(self.storage)
            self.storage.record_model_request("local-mind")
            return await asyncio.to_thread(local_client.interactions.create, **kwargs)
        try:
            self.storage.record_model_request(model)
            return await self._create_safe(client, kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            is_quota_or_net = any(
                k in msg
                for k in (
                    "429",
                    "resource_exhausted",
                    "quota",
                    "403",
                    "billing",
                    "connection",
                    "unreachable",
                    "timed out",
                    "timeout",
                    "name or service not known",
                    "temporary failure",
                    "agy cli error",
                )
            ) or isinstance(exc, (OSError, TimeoutError, asyncio.TimeoutError))
            if not is_quota_or_net:
                raise
            if self.auth.rotate_key():
                try:
                    client = self._get_client()
                    self.storage.record_model_request(model)
                    return await self._create_safe(client, kwargs)
                except Exception:
                    pass
            if model != self.settings.light_model:
                try:
                    fallback = self.settings.light_model
                    self._session_models[session_id] = fallback
                    kwargs["model"] = fallback
                    self.storage.record_model_request(fallback)
                    return await self._create_safe(client, kwargs)
                except Exception:
                    pass
            from .local_mind import LocalMindClient

            local_client = LocalMindClient(self.storage)
            self.storage.record_model_request("local-mind")
            return await asyncio.to_thread(local_client.interactions.create, **kwargs)

    async def run_turn(
        self,
        session_id: str,
        user_message: str,
        emit: EventCallback | None = None,
        locale: str | None = None,
    ) -> str:
        """Run one user turn, handling multi-step tool calls until final text."""
        self._get_client()
        prev_id = self.sessions.get(session_id) or self.storage.get_interaction_id(session_id)
        if (
            not prev_id
            and not user_message.lstrip().startswith(("[", "/"))
            and len(user_message.split()) <= 12
        ):
            try:
                from .neural import NeuralEngine

                eng_fp = NeuralEngine(self.storage)
                plan = eng_fp.match_safe_plan(user_message, min_sim=0.92)
                if plan and not eng_fp.predict_with_uncertainty(user_message).get("uncertain"):
                    raw_out = await asyncio.to_thread(
                        execute_tool,
                        plan["tool"],
                        plan["args"],
                        self.watcher,
                        self.storage,
                        self.plugins,
                        self.auth,
                        f"fast-{uuid.uuid4().hex[:6]}",
                    )
                    if isinstance(raw_out, str) and '"error":' not in raw_out:
                        self.storage.add_message(session_id, "user", user_message)
                        summary = f"[{plan['tool']}] {truncate_tool_output(scrub_secrets(raw_out), max_chars=2400)}"
                        self.storage.add_message(session_id, "assistant", summary)
                        if emit:
                            await emit({"type": "assistant_message", "text": summary})
                        return summary
            except Exception:
                pass
        rel_ctx = self.storage.get_relevant_context(
            user_message, mem_limit=15, kg_limit=12, lesson_limit=8
        )
        try:
            from .neural import NeuralEngine

            assoc_hits = NeuralEngine(self.storage).spreading_activation(user_message, top_k=4)
            seen_mem = {str(m.get("key") or "") for m in rel_ctx.get("memories") or []}
            seen_les = {str(l.get("insight") or "") for l in rel_ctx.get("lessons") or []}
            for ah in assoc_hits:
                if float(ah.get("activation", 0.0)) < 0.28:
                    continue
                if ah.get("kind") == "memory" and ah.get("key") not in seen_mem:
                    rel_ctx.setdefault("memories", []).append({"key": ah["key"], "value": ah["text"]})
                    seen_mem.add(ah["key"])
                elif ah.get("kind") in {"lesson", "rule"} and ah.get("text") not in seen_les:
                    rel_ctx.setdefault("lessons", []).append(
                        {"context": f"neural:{ah['kind']}", "insight": ah["text"], "resolution": "associative_recall"}
                    )
                    seen_les.add(ah["text"])
        except Exception:
            pass
        workflows = self.storage.find_matching_workflows(user_message, limit=3)
        same_session = self.storage.get_messages(session_id, limit=14)
        if len(same_session) < 6:
            cross = self.storage.get_recent_cross_session_messages(session_id, limit=6)
            recent_history = cross + same_session
        else:
            recent_history = same_session
        system_instruction = build_system_instruction(
            locale or self.settings.locale,
            self.settings.agent_name,
            rel_ctx["memories"],
            rel_ctx["kg_edges"],
            self.watcher.current_context,
            rel_ctx["lessons"],
            storage=self.storage,
            workflows=workflows,
            recent_history=recent_history,
        )
        cortex = getattr(self, "cortex", None)
        if cortex is not None:
            flow = getattr(cortex, "last_flow", None) or {}
            heal = getattr(cortex, "last_heal", None)
            cortex_line = f"\nLive Cortex state: flow={flow.get('mode', 'normal')}"
            if isinstance(heal, dict) and heal.get("summary"):
                cortex_line += f" | predictive_health={heal['summary']}"
            system_instruction += cortex_line
        try:
            from .self_ops import get_live_codebase_summary

            system_instruction += (
                "\n"
                + get_live_codebase_summary()
                + "\nTrust the live codebase snapshot over older memory notes; "
                "when modifying your own code, read the target file with read_file and apply edits with edit_file."
            )
        except Exception:
            pass
        self.storage.upsert_session(session_id, prev_id, title=user_message[:60])
        self.storage.add_message(session_id, "user", user_message)
        self.storage.record_activity("prompt", user_message[:80])
        detect_style_feedback(user_message, self.storage)
        # A chained conversation must stay on one model (thought signatures are model-bound),
        selected_model = self._pinned_models.get(session_id) or (
            self._session_models.get(session_id) if prev_id else None
        )
        if not selected_model:
            selected_model = choose_model(user_message, self.settings, storage=self.storage)
            self._session_models[session_id] = selected_model
        interactive = emit is not None
        batch_id = f"{session_id}:{uuid.uuid4().hex[:6]}"
        had_failure = False
        recorded_tool_calls: list[dict[str, Any]] = []

        async def _emit(payload: dict[str, Any]) -> None:
            if emit:
                await emit(payload)

        fast = self._match_zero_llm_fastpath(user_message)
        if fast is not None:
            t_name, t_args = fast
            await _emit({"type": "tool_call", "name": t_name, "arguments": t_args})
            raw_res = await asyncio.to_thread(
                execute_tool,
                t_name,
                t_args,
                self.watcher,
                self.storage,
                self.plugins,
                self.auth,
                batch_id,
            )
            await _emit({"type": "tool_result", "name": t_name, "approved": True})
            reply_text = f"[{t_name} | 0-token fast-path]\n{truncate_tool_output(str(raw_res), max_chars=2500)}"
            self.storage.add_message(session_id, "assistant", reply_text)
            await _emit({"type": "assistant_message", "text": reply_text})
            return reply_text

        if not prev_id and type(self._get_client()).__name__ in {"AgyCliClient", "LocalMindClient"}:
            from .local_mind import LocalMind

            cascade = await asyncio.to_thread(
                LocalMind(self.storage).try_local_cascade, user_message, system_instruction
            )
            if cascade and cascade.get("reply"):
                reply_text = f"[{cascade['tier']} | {cascade['model']}]\n{cascade['reply']}"
                self.storage.add_message(session_id, "assistant", reply_text)
                self.storage.record_model_request("local-mind")
                await _emit({"type": "assistant_message", "text": reply_text})
                return reply_text

        all_tools = get_all_tool_declarations(self.plugins)
        builtin_names = {d["name"] for d in TOOL_DECLARATIONS}
        pruned_tools = select_relevant_tools(user_message, all_tools, builtin_names, storage=self.storage)
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "input": user_message,
            "tools": pruned_tools,
            "system_instruction": system_instruction,
        }
        if prev_id:
            kwargs["previous_interaction_id"] = prev_id

        await _emit({"type": "status", "message": "analyzing request..."})
        interaction = await self._call_model_with_failover(session_id, **kwargs)
        selected_model = self._session_models.get(session_id, selected_model)
        self.sessions[session_id] = interaction.id
        self.storage.upsert_session(session_id, interaction.id)

        async def _run_single_tool(call: Any) -> dict[str, Any]:
            nonlocal had_failure
            import traceback

            args = dict(call.arguments or {})
            recorded_tool_calls.append({"name": call.name, "arguments": args})
            await _emit(
                {
                    "type": "tool_call",
                    "name": call.name,
                    "arguments": args,
                }
            )

            approved = False
            try:
                plugin_safe = self.plugins.is_safe(call.name) if self.plugins else None
                assessment = assess_tool_call(
                    call.name,
                    args,
                    mode=self.settings.approval_mode,
                    plugin_safe=plugin_safe,
                )

                target_parts = str(args.get("command") or args.get("path") or args.get("action") or "").split()
                fb_key = f"approval:{call.name}:{target_parts[0][:40] if target_parts else ''}"
                auto_trusted = (
                    assessment.requires_approval
                    and assessment.risk_level != "high"
                    and self.settings.approval_mode == "smart"
                    and is_auto_trust_allowed(call.name, args)
                    and self.storage.is_approval_auto_trusted(fb_key)
                )
                approved = True
                if assessment.requires_approval and not auto_trusted:
                    approval_id = str(uuid.uuid4())[:8]
                    loop = asyncio.get_running_loop()
                    fut: asyncio.Future[bool] = loop.create_future()
                    self.pending_approvals[approval_id] = PendingApproval(
                        approval_id=approval_id,
                        session_id=session_id,
                        tool_name=call.name,
                        arguments=args,
                        reason=assessment.reason,
                        risk_level=assessment.risk_level,
                        future=fut,
                    )
                    loc = locale or self.settings.locale
                    preview = compute_dry_run_preview(call.name, args)
                    from .safety import neural_risk_note

                    n_warn = neural_risk_note(call.name, args, self.storage)
                    parts_body = [assessment.reason]
                    if preview:
                        parts_body.append(preview)
                    if n_warn:
                        parts_body.append(n_warn)
                    notif_body = "\n".join(parts_body)
                    notif_handle = self.watcher._send_desktop_notification(
                        f"{self.settings.agent_name} — {t('approval_required', loc)}",
                        notif_body[:240],
                        urgency="critical" if assessment.risk_level == "high" else "normal",
                        category="approval",
                        actions=[
                            ("approve", t("notif_approve", loc)),
                            ("deny", t("notif_deny", loc)),
                        ],
                        on_action=lambda act: loop.call_soon_threadsafe(
                            self.resolve_approval, approval_id, act == "approve"
                        ),
                        bypass_gates=True,
                    )
                    if interactive:
                        await _emit(
                            {
                                "type": "approval_required",
                                "approval_id": approval_id,
                                "tool_name": call.name,
                                "arguments": args,
                                "reason": assessment.reason,
                                "risk_level": assessment.risk_level,
                                "preview": preview,
                            }
                        )
                    try:
                        approved = await asyncio.wait_for(fut, timeout=300 if interactive else 45)
                        self.storage.record_notification_feedback(fb_key, "yes" if approved else "dismiss")
                        if (
                            approved
                            and assessment.risk_level != "high"
                            and is_auto_trust_allowed(call.name, args)
                            and self.storage.is_approval_auto_trusted(fb_key)
                        ):
                            label = fb_key.split(":", 2)[-1] or call.name
                            self.watcher._send_desktop_notification(
                                f"{self.settings.agent_name}",
                                f"You approved `{label}` 3 times. I will stop asking for it (smart mode).",
                                urgency="low",
                                actions=[("keep", "OK"), ("revoke", "Ask again")],
                                on_action=lambda act, k=fb_key: (
                                    self.storage.record_notification_feedback(k, "dismiss")
                                    if act == "revoke"
                                    else None
                                ),
                            )
                    except TimeoutError:
                        self.pending_approvals.pop(approval_id, None)
                        approved = False
                        self.storage.record_notification_feedback(fb_key, "timeout")
                    finally:
                        if notif_handle is not None:
                            notif_handle.cancel()
                        if interactive:
                            await _emit(
                                {
                                    "type": "approval_resolved",
                                    "approval_id": approval_id,
                                    "approved": approved,
                                }
                            )

                if approved and call.name == "switch_model":
                    raw_m = str(args.get("model") or "").strip()
                    alias_map = {
                        "local": "local-mind",
                        "local-mind": "local-mind",
                        "local_mind": "local-mind",
                        "localmind": "local-mind",
                        "offline": "local-mind",
                        "gguf": "local-mind",
                        "neon": "gemini-4-neon",
                        "argon": "gemini-4-argon",
                        "hydrogen": "gemini-4-hydrogen",
                        "flash": "gemini-3.8-flash",
                        "lite": "gemini-3.5-flash-lite",
                        "auto": "auto",
                    }
                    resolved = alias_map.get(raw_m.lower(), raw_m)
                    if not resolved:
                        result_payload = json.dumps({"error": "model name is required"})
                    else:
                        if resolved == "auto":
                            self._pinned_models.pop(session_id, None)
                            self._session_models.pop(session_id, None)
                        else:
                            self._pinned_models[session_id] = resolved
                            self._session_models[session_id] = resolved
                        self.sessions.pop(session_id, None)
                        self.storage.upsert_session(session_id, None)
                        persisted = False
                        if args.get("persist") and resolved != "auto":
                            from .config import persist_env_var

                            self.settings.model = resolved
                            persist_env_var("SYSAGENT_MODEL", resolved)
                            persisted = True
                        await _emit({"type": "model_changed", "model": resolved})
                        result_payload = json.dumps(
                            {"status": "ok", "model": resolved, "persisted": persisted},
                            ensure_ascii=False,
                        )
                elif approved:
                    cached = self._tool_cache.get(call.name, args)
                    if cached is not None:
                        result_payload = cached
                    else:
                        t0 = time.monotonic()
                        result_payload = await asyncio.to_thread(
                            execute_tool,
                            call.name,
                            args,
                            self.watcher,
                            self.storage,
                            self.plugins,
                            self.auth,
                            batch_id,
                        )
                        elapsed_ms = (time.monotonic() - t0) * 1000.0
                        is_err = isinstance(result_payload, str) and '"error":' in result_payload
                        if is_err:
                            had_failure = True
                            from .self_learn import record_immunity_pattern

                            record_immunity_pattern(self.storage, user_message, call.name, str(result_payload)[:160])
                            try:
                                from .neural import NeuralEngine

                                NeuralEngine(self.storage).online_step(user_message, call.name, reward=-0.8)
                            except Exception:
                                pass
                        elif len(recorded_tool_calls) >= 2:
                            prev_tc = recorded_tool_calls[-2].get("name")
                            if prev_tc and prev_tc != call.name:
                                try:
                                    from .neural import NeuralEngine

                                    NeuralEngine(self.storage).online_transition_step(user_message, str(prev_tc), call.name)
                                except Exception:
                                    pass
                        self.storage.record_tool_metric(call.name, elapsed_ms, is_err)
                        if isinstance(result_payload, str):
                            result_payload = truncate_tool_output(scrub_secrets(result_payload), max_chars=4000)
                        if call.name in {"write_file", "edit_file", "trash_file", "undo_last_action", "undo_batch_actions", "run_command", "forge_skill"}:
                            self._tool_cache.invalidate_all()
                            if call.name in {"undo_last_action", "undo_batch_actions"}:
                                had_failure = True
                                self._raise_caution()
                        elif not is_err:
                            self._tool_cache.set(call.name, args, result_payload)
                    if assessment.requires_approval:
                        self._relax_caution()
                elif not interactive:
                    had_failure = True
                    result_payload = json.dumps(
                        {
                            "error": "This action needs interactive user approval and cannot run in the background. "
                            "Do not retry; summarize what you would do so the user can approve it later."
                        },
                        ensure_ascii=False,
                    )
                else:
                    had_failure = True
                    self._raise_caution()
                    from .self_learn import record_immunity_pattern

                    record_immunity_pattern(self.storage, user_message, call.name, "User denied this action.")
                    result_payload = json.dumps(
                        {"error": "User denied this action."},
                        ensure_ascii=False,
                    )
            except Exception as exc:
                had_failure = True
                tb_lines = traceback.format_exc().strip().splitlines()
                loc_hint = next((ln.strip() for ln in reversed(tb_lines) if ln.strip().startswith("File ")), "")
                self.storage.record_tool_metric(call.name, 0.0, True)
                self.storage.add_brain_thought(
                    "ignore",
                    0,
                    f"Runtime tool error in {call.name} ({loc_hint}): {exc}",
                )
                result_payload = json.dumps(
                    {
                        "error": f"Internal runtime error in {call.name}: {exc}",
                        "location": loc_hint,
                        "traceback": "\n".join(tb_lines[-6:]),
                        "self_repair_hint": f"You can inspect this file/line under {Path(__file__).resolve().parent} and fix it directly if the user wants.",
                    },
                    ensure_ascii=False,
                )

            preview = (
                result_payload[:500]
                if isinstance(result_payload, str)
                else "[multimodal result]"
            )
            await _emit(
                {
                    "type": "tool_result",
                    "name": call.name,
                    "approved": approved,
                    "result_preview": preview,
                }
            )
            return {
                "type": "function_result",
                "call_id": call.id,
                "name": call.name,
                "result": result_payload,
            }

        def _is_real_mutation(calls: list[dict[str, Any]]) -> bool:
            readonly_prefixes = (
                "ls", "cat ", "head ", "tail ", "grep ", "rg ", "find ",
                "git status", "git diff", "git log", "ps ", "systemctl status",
                "systemctl --user status", "pwd", "echo ", "wc ",
            )
            for tc in calls:
                cname = tc.get("name")
                if cname in {"write_file", "edit_file", "trash_file", "forge_skill", "desktop_control", "create_automation"}:
                    return True
                if cname == "run_command":
                    cmd_txt = str((tc.get("arguments") or {}).get("command") or "").strip().lower()
                    if cmd_txt and not cmd_txt.startswith(readonly_prefixes):
                        return True
            return False

        nudge_count = 0
        step_count = 0
        max_steps = 80
        read_nudged = False
        dyn_cues = self.storage.get_learned_cues()
        stall_cues = dyn_cues.get("stall_reply") or []
        action_req_cues = dyn_cues.get("action_req") or []
        user_low = user_message.lower()
        user_wants_action = bool(action_req_cues and any(c in user_low for c in action_req_cues))
        while True:
            function_calls = [
                step for step in (interaction.steps or []) if step.type == "function_call"
            ]
            if function_calls and step_count >= max_steps:
                text = (
                    f"Completed {step_count} tool steps (step limit {max_steps} reached). "
                    f"Executed tools: {', '.join(str(tc.get('name')) for tc in recorded_tool_calls[-8:])}."
                )
                self.storage.add_message(session_id, "assistant", text)
                await _emit({"type": "assistant_message", "text": text})
                return text

            if function_calls and step_count >= 12 and not read_nudged and not _is_real_mutation(recorded_tool_calls):
                read_nudged = True
                await _emit({"type": "status", "message": "reminding model to start applying edits..."})

            if not function_calls:
                text = interaction.output_text or "(No response generated.)"
                low_txt = text.lower()
                leaked_tool_json = "__tool_calls__" in low_txt or '"tool_calls"' in low_txt
                has_mutated = _is_real_mutation(recorded_tool_calls)
                read_only_stalled = (
                    not has_mutated
                    and user_wants_action
                    and len(recorded_tool_calls) >= 1
                )
                neural_stall = False
                try:
                    from .neural import NeuralEngine

                    neural_stall = NeuralEngine(self.storage).score_reply_stall(text) >= 0.72
                except Exception:
                    pass
                broken_py_err = ""
                if has_mutated and nudge_count < 3:
                    from pathlib import Path as _Path

                    for tc in reversed(recorded_tool_calls):
                        if tc.get("name") in {"write_file", "edit_file"}:
                            fp_str = str((tc.get("arguments") or {}).get("path") or "")
                            if fp_str.endswith(".py"):
                                fp = _Path(fp_str).expanduser()
                                if fp.is_file():
                                    try:
                                        compile(fp.read_text(encoding="utf-8", errors="replace"), str(fp), "exec")
                                    except SyntaxError as exc:
                                        broken_py_err = f"{fp.name}: {exc}"
                                        break
                should_nudge = (
                    nudge_count < 3
                    and (
                        leaked_tool_json
                        or read_only_stalled
                        or neural_stall
                        or bool(broken_py_err)
                        or any(cue in low_txt for cue in stall_cues)
                    )
                )
                if should_nudge:
                    nudge_count += 1
                    if leaked_tool_json:
                        nudge_msg = (
                            "[System]: Your previous reply contained raw `__tool_calls__` JSON instead of valid function calls. "
                            "Emit ONLY clean, valid tool calls right now."
                        )
                    elif broken_py_err:
                        nudge_msg = (
                            f"[System]: SyntaxError detected in modified file ({broken_py_err}). "
                            "Fix this syntax error right now before finishing."
                        )
                    else:
                        nudge_msg = (
                            "[System]: Do NOT stop halfway to announce, check, or ask to continue. "
                            "Invoke the required edit_file / write_file / run_command / inspection tool calls RIGHT NOW, "
                            "and only send your final plain-text answer after the work and verification are completely done."
                        )
                    await _emit({"type": "status", "message": f"continuing task (nudge {nudge_count})..."})
                    interaction = await self._call_model_with_failover(
                        session_id,
                        model=selected_model,
                        input=nudge_msg,
                        previous_interaction_id=interaction.id,
                        tools=pruned_tools,
                        system_instruction=system_instruction,
                    )
                    selected_model = self._session_models.get(session_id, selected_model)
                    self.sessions[session_id] = interaction.id
                    self.storage.upsert_session(session_id, interaction.id)
                    continue
                self.storage.add_message(session_id, "assistant", text)
                if not had_failure:
                    PromptRegistry(self.storage).record_outcome("agent_main", True)
                    if not text.startswith("[Local Offline Mode]"):
                        from .local_mind import LocalMind

                        LocalMind(self.storage).record_and_learn(
                            user_message, recorded_tool_calls, text
                        )
                        if (
                            len(recorded_tool_calls) == 1
                            and recorded_tool_calls[0].get("name") in SAFE_TOOLS
                        ):
                            try:
                                from .neural import NeuralEngine

                                NeuralEngine(self.storage).record_safe_plan(
                                    user_message,
                                    str(recorded_tool_calls[0]["name"]),
                                    dict(recorded_tool_calls[0].get("arguments") or {}),
                                )
                            except Exception:
                                pass
                if interactive and text:
                    chunk_size = 64
                    for idx in range(0, len(text), chunk_size):
                        await _emit({"type": "text_delta", "text": text[idx : idx + chunk_size]})
                await _emit({"type": "assistant_message", "text": text})
                prev_turn = self._last_turn_info.get(session_id)
                self._last_turn_info[session_id] = (user_message, recorded_tool_calls)
                if not session_id.startswith(("brain-", "auto-", "inspect-")):
                    bg_task = asyncio.create_task(
                        self._post_turn_learn(
                            session_id,
                            user_message,
                            text,
                            recorded_tool_calls,
                            had_failure,
                            prev_turn,
                        )
                    )
                    self._bg_learning_tasks.add(bg_task)
                    bg_task.add_done_callback(self._bg_learning_tasks.discard)
                return text

            step_count += 1
            all_parallel_safe = len(function_calls) > 1 and all(
                c.name in SAFE_TOOLS
                and not assess_tool_call(
                    c.name,
                    dict(c.arguments or {}),
                    mode=self.settings.approval_mode,
                    plugin_safe=(self.plugins.is_safe(c.name) if self.plugins else None),
                ).requires_approval
                for c in function_calls
            )
            if all_parallel_safe:
                function_results = list(
                    await asyncio.gather(*(_run_single_tool(c) for c in function_calls))
                )
            else:
                function_results = []
                for call in function_calls:
                    function_results.append(await _run_single_tool(call))

            await _emit({"type": "status", "message": f"step {step_count}: processing tool results..."})
            interaction = await self._call_model_with_failover(
                session_id,
                model=selected_model,
                input=function_results,
                previous_interaction_id=interaction.id,
                tools=pruned_tools,
                system_instruction=system_instruction,
            )
            selected_model = self._session_models.get(session_id, selected_model)
            self.sessions[session_id] = interaction.id
            self.storage.upsert_session(session_id, interaction.id)

"""Autonomous Cognitive Brain & Self-Learning Decision Engine for SysAgent."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Any

from .agent import AgentManager
from .config import Settings
from .i18n import resolve_locale, t
from .logging_setup import get_logger
from .prompts import DEFAULT_PROMPTS, PromptRegistry
from .storage import Storage
from .watcher import SystemWatcher

logger = get_logger(__name__)


BRAIN_SYSTEM_PROMPT = DEFAULT_PROMPTS["brain_eval"]["template"]


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    idx = cleaned.find("{")
    if idx == -1:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[idx:])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


class CognitiveBrain:
    """Periodic background thinking loop with human-in-the-loop self-learning."""

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
        self._last_seen_signature: str = ""
        self._consecutive_ignores: int = 0
        self._cooldown_until: float = 0.0
        self._last_cue_refresh: float = 0.0
        from .cortex import NeuralBrainCortex

        self.cortex = NeuralBrainCortex(settings, storage, watcher=watcher, brain=self)
        if getattr(watcher, "notifier", None) is not None:
            watcher.notifier.flow_provider = lambda: self.cortex.last_flow

    def start(self) -> None:
        if self._running or not self.settings.brain_enabled:
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

    def is_quiet_hours(self) -> bool:
        qh = (self.settings.quiet_hours or "").strip()
        if "-" not in qh:
            return False
        try:
            start_s, end_s = [p.strip() for p in qh.split("-", 1)]
            sh, sm = (int(x) for x in start_s.split(":", 1))
            eh, em = (int(x) for x in end_s.split(":", 1))
            now = datetime.now()
            now_m = now.hour * 60 + now.minute
            start_m = sh * 60 + sm
            end_m = eh * 60 + em
            if start_m <= end_m:
                return start_m <= now_m <= end_m
            return now_m >= start_m or now_m <= end_m
        except Exception:
            return False

    async def _loop(self) -> None:
        await asyncio.sleep(15)
        while self._running:
            try:
                await asyncio.to_thread(self.cortex.tick)
                res = await self.think_once(force=False)
                if res.get("decision") == "ignore":
                    self._consecutive_ignores = min(4, self._consecutive_ignores + 1)
                elif res.get("status") == "ok":
                    self._consecutive_ignores = 0
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    self._cooldown_until = time.monotonic() + 900
                    from .self_resilience import get_quota_governor

                    get_quota_governor().record_rate_limit("brain")
                self.storage.add_brain_thought("ignore", 0, f"Brain cycle error: {exc}")
            backoff = min(4, self._consecutive_ignores)
            await asyncio.sleep(max(30, self.settings.brain_interval_sec) * (2**backoff if backoff else 1))

    async def think_once(self, force: bool = False) -> dict[str, Any]:
        """Run one cognitive evaluation cycle. Returns summary dict."""
        if not force and time.monotonic() < self._cooldown_until:
            return {"status": "skipped", "reason": "quota_cooldown"}
        if not force:
            from .self_resilience import get_quota_governor

            allowed_q, q_reason = get_quota_governor().allow_background("brain")
            if not allowed_q:
                return {"status": "deferred", "reason": q_reason}

        events = self.watcher.get_events(limit=15)
        notable = [
            e
            for e in events
            if e["category"] in {"filesystem", "systemd", "security", "health"}
            and not any(ext in e["message"] for ext in (".part", ".crdownload", ".tmp"))
        ]
        sig = "|".join(f"{e['timestamp']}:{e['message']}" for e in notable[:5])
        world_forecast: dict[str, Any] = {}
        try:
            from .neural import NeuralEngine

            world_forecast = NeuralEngine(self.storage).predict_world_state()
        except Exception:
            pass

        if not notable and not force:
            b_prob = float(world_forecast.get("bottleneck_prob", 0.0) or 0.0)
            anom = bool(world_forecast.get("anomaly_detected"))
            if int(world_forecast.get("samples", 0) or 0) >= 6 and (b_prob >= 0.72 or anom):
                hint = str(world_forecast.get("proactive_hint") or "anomaly")
                notable = [
                    {
                        "category": "health",
                        "severity": "warning",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "message": f"Predictive alert: {hint} (bottleneck={b_prob}, anomaly={world_forecast.get('anomaly_score', 0)})",
                    }
                ]
                sig = f"predictive:{hint}:{datetime.now().strftime('%Y-%m-%d %H')}"

        if not force and (not notable or sig == self._last_seen_signature):
            return {"status": "skipped", "reason": "no_new_notable_events"}

        if any(e.get("severity") == "critical" for e in notable):
            self._consecutive_ignores = 0

        if not self.agent.auth.is_configured:
            return {"status": "skipped", "reason": "missing_credentials"}

        now_m = time.monotonic()
        if not (self.storage.get_learned_cues().get("stall_reply")) or (now_m - self._last_cue_refresh >= 86400.0):
            try:
                await self.refresh_learned_cues()
                self._last_cue_refresh = now_m
            except Exception:
                pass

        self.storage.decay_stale_rules()
        self.storage.expire_stale_brain_questions()
        rules = self.storage.list_brain_rules()
        if not force and notable and all(
            self.storage.is_ignored_by_brain_rule(str(e.get("message", ""))) for e in notable
        ):
            self._last_seen_signature = sig
            self._consecutive_ignores = min(4, self._consecutive_ignores + 1)
            return {"status": "skipped", "reason": "rule_prefilter_ignore"}

        health = self.watcher.get_health_snapshot()
        habits = self.storage.get_recurring_habits()
        loc = resolve_locale(self.settings.locale)
        assoc_ctx: list[dict[str, Any]] = []
        try:
            from .neural import NeuralEngine

            assoc_ctx = NeuralEngine(self.storage).spreading_activation(sig or "system status", top_k=3)
        except Exception:
            pass
        causal_links: list[dict[str, str]] = []
        root_cause_report: dict[str, Any] = {}
        try:
            from .self_learn import build_causal_timeline, investigate_root_cause

            causal_links = build_causal_timeline(self.storage, self.watcher)
            root_cause_report = investigate_root_cause(
                self.storage, self.watcher, events=notable, health=health
            )
        except Exception:
            pass

        prompt = json.dumps(
            {
                "health": health,
                "notable_events": notable[:10],
                "learned_rules": rules[:25],
                "recurring_habits": habits[:10],
                "quiet_hours_active": self.is_quiet_hours(),
                "workspace_context": self.watcher.current_context,
                "world_forecast": world_forecast,
                "associative_context": assoc_ctx,
                "causal_links": causal_links,
                "root_cause_diagnosis": root_cause_report,
            },
            ensure_ascii=False,
        )
        sys_inst = PromptRegistry(self.storage).render(
            "brain_eval",
            agent_name=self.settings.agent_name,
            threshold=self.settings.brain_confidence_threshold,
            locale=loc,
        )

        try:
            if (
                not force
                and not any(e.get("severity") == "critical" for e in notable)
                and type(self.agent._get_client()).__name__ in {"AgyCliClient", "LocalMindClient"}
            ):
                from .local_mind import LocalMind, find_gguf_model

                if find_gguf_model("reasoning", strict=True):
                    loc_txt = await asyncio.to_thread(
                        LocalMind(self.storage).run_gguf_inference,
                        prompt,
                        sys_inst,
                        "reasoning",
                    )
                    if loc_txt:
                        loc_p = _extract_json(loc_txt)
                        loc_dec = str(loc_p.get("decision", "")).lower()
                        loc_conf = int(loc_p.get("confidence", 0) or 0)
                        if loc_dec == "ignore" and loc_conf >= 80:
                            self._last_seen_signature = sig
                            self._consecutive_ignores = min(4, self._consecutive_ignores + 1)
                            loc_thought = str(loc_p.get("thought") or "Routine event ignored locally.")
                            self.storage.add_brain_thought("ignore", loc_conf, f"[local:reasoning] {loc_thought}")
                            return {"status": "ok", "decision": "ignore", "confidence": loc_conf, "source": "local_reasoning"}
        except Exception:
            pass

        fast_rule = self.storage.match_fast_action_rule(sig) if not force else None
        if fast_rule:
            self._last_seen_signature = sig
            decision = "act"
            confidence = int(fast_rule.get("confidence", 92))
            thought = f"[fast-path rule #{fast_rule['id']}] {fast_rule.get('pattern', '')}"
            action_prompt = str(fast_rule.get("action_policy", ""))
            matched_rule_id = fast_rule["id"]
            question = None
            suggested_rule = None
        else:
            from .self_resilience import get_quota_governor

            get_quota_governor().record_background_call("brain", self.settings.light_model)
            interaction = await self.agent.auth.call_interactions(
                self.storage,
                model=self.settings.light_model,
                input=prompt,
                system_instruction=sys_inst,
                store=False,
            )
            self._last_seen_signature = sig
            raw_text = interaction.output_text or "{}"
            parsed = _extract_json(raw_text)

            decision = str(parsed.get("decision", "ignore"))
            try:
                confidence = int(parsed.get("confidence", 50))
            except (ValueError, TypeError):
                confidence = 50
            thought = str(parsed.get("thought", ""))
            action_prompt = parsed.get("action_prompt")
            matched_rule_id = parsed.get("matched_rule_id")
            question = parsed.get("question")
            suggested_rule = parsed.get("suggested_rule")

        if decision == "act" and confidence < self.settings.brain_confidence_threshold:
            decision = "ask_user"
            if not question:
                question = f"Should I proceed with: {action_prompt}?"

        if decision == "act" and action_prompt:
            dry_ok, dry_note = self._dry_run_validate_action(str(action_prompt), confidence, world_forecast)
            if dry_note:
                thought = f"{thought} [dry-run: {dry_note}]".strip()
            if not dry_ok:
                decision = "ask_user"
                question = f"Dry-run guard flagged {dry_note}. Proceed with: {action_prompt}?"

        action_taken: str | None = None

        if decision == "act" and action_prompt:
            # Fresh, throwaway session per autonomous action: a shared "brain-auto" chain would
            run_sid = f"brain-auto-{time.time_ns()}"
            try:
                reply = await self.agent.run_turn(run_sid, str(action_prompt))
            finally:
                self.agent.reset_session(run_sid)
            action_taken = reply[:300]
            try:
                rule_id_int = int(matched_rule_id) if matched_rule_id is not None else None
            except (ValueError, TypeError):
                rule_id_int = None
            impact_ok, impact_note = self._verify_action_impact(health, reply)
            if impact_note:
                thought = f"{thought} [{impact_note}]".strip()
            if rule_id_int is not None:
                self.storage.reinforce_brain_rule(rule_id_int, delta=5 if impact_ok else -15)
            self.storage.add_lesson(
                "brain_act" if impact_ok else "brain_act_failed",
                str(action_prompt)[:180],
                f"{impact_note}: {action_taken[:230]}" if impact_note else action_taken[:260],
            )
            try:
                from .neural import NeuralEngine

                NeuralEngine(self.storage).sync_associative_memory()
            except Exception:
                pass
        elif decision == "ask_user" and question:
            ctx = json.dumps(notable[:3], ensure_ascii=False)
            qid = self.storage.create_brain_question(str(question), ctx, suggested_rule)
            action_taken = f"Asked user (question #{qid})"
            suppress_notif = self.is_quiet_hours() or (
                self.watcher.current_context == "media_meeting"
                and not any(e.get("severity") == "critical" for e in notable)
            )
            if not suppress_notif:
                q_str = str(question)[:240]
                loop = asyncio.get_running_loop()

                def _on_brain_notif_action(act: str) -> None:
                    if act == "dismiss":
                        self.storage.resolve_brain_question(qid, "dismissed")
                        self.storage.record_notification_feedback(q_str, "dismiss")
                        self.storage.add_lesson("brain_dismiss", q_str[:160], "User dismissed alert notification")
                        try:
                            from .neural import NeuralEngine

                            NeuralEngine(self.storage).sync_associative_memory()
                        except Exception:
                            pass
                    elif act == "yes":
                        self.storage.record_notification_feedback(q_str, "yes")
                        async def _apply_yes() -> None:
                            res = await self.answer_question(
                                qid,
                                "Yes, proceed safely with the recommended action and learn this rule.",
                                learn_rule=True,
                            )
                            self.watcher._send_desktop_notification(
                                f"{self.settings.agent_name}",
                                str(res.get("reply") or t("notif_action_done", loc))[:240],
                            )
                        asyncio.run_coroutine_threadsafe(_apply_yes(), loop)
                    elif act == "reply":
                        import shutil as _shutil
                        import subprocess as _sp

                        def _prompt_dialog() -> str:
                            if _shutil.which("kdialog"):
                                p = _sp.run(
                                    ["kdialog", "--title", f"{self.settings.agent_name} (#{qid})", "--inputbox", q_str],
                                    capture_output=True,
                                    text=True,
                                )
                                if p.returncode == 0:
                                    return p.stdout.strip()
                            elif _shutil.which("zenity"):
                                p = _sp.run(
                                    ["zenity", "--entry", f"--title={self.settings.agent_name} (#{qid})", f"--text={q_str}"],
                                    capture_output=True,
                                    text=True,
                                )
                                if p.returncode == 0:
                                    return p.stdout.strip()
                            return ""

                        async def _apply_custom() -> None:
                            ans_text = await asyncio.to_thread(_prompt_dialog)
                            if ans_text:
                                self.storage.record_notification_feedback(q_str, "yes")
                                res = await self.answer_question(qid, ans_text, learn_rule=True)
                                self.watcher._send_desktop_notification(
                                    f"{self.settings.agent_name}",
                                    str(res.get("reply") or t("notif_action_done", loc))[:240],
                                )

                        asyncio.run_coroutine_threadsafe(_apply_custom(), loop)

                self.watcher._send_desktop_notification(
                    f"{self.settings.agent_name} Question",
                    q_str,
                    actions=[
                        ("yes", t("notif_yes", loc)),
                        ("reply", t("notif_reply", loc)),
                        ("dismiss", t("notif_dismiss", loc)),
                    ],
                    on_action=_on_brain_notif_action,
                    on_timeout=lambda: self.storage.record_notification_feedback(q_str, "timeout"),
                )

        self.storage.add_brain_thought(decision, confidence, thought, action_taken)
        self._check_habit_automation_suggestions(habits, loc)
        return {
            "status": "ok",
            "decision": decision,
            "confidence": confidence,
            "thought": thought,
            "action_taken": action_taken,
        }

    def _check_habit_automation_suggestions(
        self, habits: list[dict[str, Any]], loc: str
    ) -> None:
        """Proactively suggest turning 3x+ recurring hourly user habits into daily automations."""
        if self.is_quiet_hours() or not habits:
            return
        existing_prompts = {
            str(a.get("prompt", "")).strip().lower() for a in self.storage.list_automations()
        }
        for h in habits:
            if h.get("category") != "prompt" or int(h.get("hits", 0)) < 3:
                continue
            detail = str(h.get("detail") or "").strip()
            hour = int(h.get("hour_bucket", 0))
            if len(detail) < 6 or detail.lower() in existing_prompts:
                continue
            fb_key = f"habit:{hour:02d}:{detail[:60]}"
            if self.storage.get_notification_cooldown_multiplier(fb_key) > 1:
                continue
            pkey = self.storage._normalize_feedback_key(fb_key)
            with self.storage._lock, self.storage._connect() as conn:
                row = conn.execute(
                    "SELECT yes_count, dismiss_count FROM notification_feedback WHERE pattern_key = ?",
                    (pkey,),
                ).fetchone()
            if row and (int(row["yes_count"]) > 0 or int(row["dismiss_count"]) > 0):
                continue

            time_str = f"{hour:02d}:00"
            msg = f"You often run '{detail[:80]}' around {time_str}. Automate this daily?"

            def _on_habit_act(act: str, fb_key: str = fb_key, detail: str = detail, time_str: str = time_str) -> None:
                if act == "automate":
                    self.storage.record_notification_feedback(fb_key, "yes")
                    self.storage.create_automation(
                        name=f"Auto {detail[:24]} ({time_str})",
                        trigger_type="daily_time",
                        trigger_value=time_str,
                        prompt=detail,
                    )
                    self.watcher._send_desktop_notification(
                        f"{self.settings.agent_name}",
                        f"Created daily {time_str} automation: {detail[:80]}",
                    )
                else:
                    self.storage.record_notification_feedback(fb_key, "dismiss")

            self.watcher._send_desktop_notification(
                f"{self.settings.agent_name} Habit Suggestion",
                msg,
                actions=[
                    ("automate", t("notif_automate", loc)),
                    ("dismiss", t("notif_dismiss", loc)),
                ],
                on_action=_on_habit_act,
                on_timeout=lambda fb_key=fb_key: self.storage.record_notification_feedback(fb_key, "timeout"),
            )
            break

    def _execute_closed_loop_action(self, action_code: str, user_answer: str) -> dict[str, Any] | None:
        """Deterministically execute structured remediation actions (`kill_leak:<pid>`, `free_vram`) upon user confirmation."""
        code = (action_code or "").strip()
        if not code.startswith(("kill_leak:", "free_vram", "clean_cache")):
            return None
        ans_low = (user_answer or "").strip().lower()
        negative_tokens = {"no", "n", "hayir", "hayır", "reddet", "iptal", "cancel", "dismiss", "ignore", "0", "false"}
        if not ans_low or ans_low in negative_tokens or any(tok in ans_low.split() for tok in negative_tokens):
            return {"executed": False, "action": code, "status": "declined"}

        import psutil as _ps

        vm_before = _ps.virtual_memory()
        avail_before_mb = round(vm_before.available / (1024 * 1024), 1)
        ram_pct_before = round(float(vm_before.percent), 1)
        if code.startswith("kill_leak:"):
            raw_pid = code.split(":", 1)[1].strip()
            if not raw_pid.isdigit():
                return {"executed": False, "action": code, "error": "invalid_pid"}
            pid = int(raw_pid)
            terminated = False
            proc_name = f"pid:{pid}"
            try:
                proc = _ps.Process(pid)
                proc_name = proc.name()
                proc.terminate()
                terminated = True
            except (_ps.NoSuchProcess, _ps.AccessDenied) as exc:
                return {
                    "executed": False,
                    "action": code,
                    "pid": pid,
                    "error": str(exc),
                }
            vm_after = _ps.virtual_memory()
            avail_after_mb = round(vm_after.available / (1024 * 1024), 1)
            ram_pct_after = round(float(vm_after.percent), 1)
            freed_mb = round(max(0.0, avail_after_mb - avail_before_mb), 1)
            summary = f"Terminated leaking process '{proc_name}' (PID {pid}); freed ~{freed_mb} MB RAM."
            cf_impact: dict[str, Any] = {}
            try:
                from .self_learn import verify_remediation_impact

                cf_impact = verify_remediation_impact(
                    self.storage,
                    code,
                    {"ram_percent": ram_pct_before, "pid_alive": True},
                    {"ram_percent": ram_pct_after, "freed_ram_mb": freed_mb, "pid_alive": False},
                )
            except Exception:
                pass
            return {
                "executed": terminated,
                "action": code,
                "pid": pid,
                "process": proc_name,
                "freed_ram_mb": freed_mb,
                "counterfactual": cf_impact,
                "summary": summary,
            }

        if code == "free_vram":
            from .local_inference import _release_all_warm_models

            unloaded = _release_all_warm_models()
            avail_after_mb = round(_ps.virtual_memory().available / (1024 * 1024), 1)
            freed_mb = round(max(0.0, avail_after_mb - avail_before_mb), 1)
            summary = f"Released warm GGUF models (unloaded={unloaded}); freed ~{freed_mb} MB RAM."
            return {
                "executed": True,
                "action": code,
                "unloaded_gguf": unloaded,
                "freed_ram_mb": freed_mb,
                "summary": summary,
            }
        return None

    async def answer_question(
        self, question_id: int, user_answer: str, learn_rule: bool = True
    ) -> dict[str, Any]:
        """Process user's answer, learn a reusable policy, and execute the instruction."""
        q = self.storage.get_brain_question(question_id)
        if not q:
            raise KeyError(f"Question {question_id} not found")

        self.storage.resolve_brain_question(question_id, "answered", user_answer)

        learned_rule_id: int | None = None
        if learn_rule and user_answer.strip():
            pattern = q.get("suggested_rule") or q["question"][:120]
            policy = user_answer.strip()
            if self.agent.auth.is_configured:
                try:
                    distill_prompt = (
                        f"Question asked to user: {q['question']}\n"
                        f"Context: {q['context']}\n"
                        f"User's answer: {user_answer}\n"
                        'Distill a reusable rule as JSON: {"pattern": "...", "action_policy": "..."}'
                    )
                    resp = await self.agent.auth.call_interactions(
                        self.storage,
                        model=self.settings.light_model,
                        input=distill_prompt,
                        store=False,
                    )
                    rule_json = _extract_json(resp.output_text or "{}")
                    pattern = str(rule_json.get("pattern") or pattern)
                    policy = str(rule_json.get("action_policy") or policy)
                except Exception:
                    pass
            learned_rule_id = self.storage.add_brain_rule(pattern, policy, confidence=85)

        closed_loop = self._execute_closed_loop_action(
            str(q.get("suggested_rule") or ""), user_answer
        )
        if closed_loop is not None and closed_loop.get("executed"):
            reply = str(closed_loop.get("summary") or "Closed-loop remediation executed.")
            self.storage.add_brain_thought(
                "act",
                95,
                f"Closed-loop action for #{question_id}: {closed_loop.get('action')}",
                reply[:300],
            )
            return {
                "status": "answered",
                "learned_rule_id": learned_rule_id,
                "closed_loop": closed_loop,
                "reply": reply,
            }

        exec_prompt = (
            f"Regarding this situation:\nQuestion: {q['question']}\nContext: {q['context']}\n"
            f"The user answered: {user_answer}\n"
            "Carry out the user's instruction now if an action is needed."
        )
        run_sid = f"brain-q-{question_id}"
        try:
            reply = await self.agent.run_turn(run_sid, exec_prompt)
        finally:
            self.agent.reset_session(run_sid)
        self.storage.add_brain_thought(
            "act",
            90,
            f"Learned from user answer to #{question_id}",
            reply[:300],
        )
        try:
            from .neural import NeuralEngine

            NeuralEngine(self.storage).sync_associative_memory()
        except Exception:
            pass
        return {
            "status": "answered",
            "learned_rule_id": learned_rule_id,
            "reply": reply,
        }

    def _dry_run_validate_action(
        self, action_prompt: str, confidence: int, world_forecast: dict[str, Any]
    ) -> tuple[bool, str]:
        """Simulate tool/safety impact before autonomous execution; downgrade risky plans."""
        try:
            from .local_mind import LocalMind
            from .safety import assess_tool_call

            pred_tool, pred_args, score = LocalMind(self.storage).predict_tool(action_prompt)
            anomaly = bool(world_forecast.get("anomaly_detected"))
            b_prob = float(world_forecast.get("bottleneck_prob", 0.0) or 0.0)
            if (anomaly or b_prob >= 0.75) and confidence < 88:
                return False, f"world anomaly={anomaly} bottleneck={b_prob:.2f}"
            if pred_tool and score >= 0.35:
                assessment = assess_tool_call(pred_tool, pred_args or {}, mode="smart")
                if assessment.requires_approval and confidence < 92:
                    return False, f"tool={pred_tool} ({assessment.reason})"
                return True, f"tool={pred_tool} safe"
        except Exception as exc:
            logger.warning("Dry-run validation failed for action '%s': %s", action_prompt[:80], exc)
        return True, "ok"

    def _verify_action_impact(
        self, health_before: dict[str, Any], reply: str
    ) -> tuple[bool, str]:
        """Compare post-action health and reply status to verify autonomous action impact."""
        low = (reply or "").lower()
        if any(err in low for err in ("failed", "error:", "traceback", "permission denied", "cancelled")):
            return False, "verify=reply_error"
        try:
            after = self.watcher.get_health_snapshot()
            ram_before = float(health_before.get("ram_percent", 0) or 0)
            ram_after = float(after.get("ram_percent", 0) or 0)
            if ram_before >= 85.0 and ram_after > ram_before + 3.0:
                return False, f"verify=ram_worsened({ram_before:.1f}%->{ram_after:.1f}%)"
            return True, f"verify=ok(ram={ram_after:.1f}%)"
        except Exception:
            return True, ""

    async def refresh_learned_cues(self) -> dict[str, list[str]]:
        """Ask the light model to generate/translate dynamic intent cues for the user's locale and train NeuralEngine."""
        loc = resolve_locale(self.settings.locale)
        recent_msgs = [
            m.get("content", "")[:120]
            for m in self.storage.get_messages("default", limit=8)
            if m.get("role") == "user"
        ]
        prompt = json.dumps(
            {
                "locale": loc,
                "recent_user_samples": recent_msgs,
                "existing_cues": self.storage.get_learned_cues(),
                "task": (
                    "Return JSON with lists of short lowercase phrases in both English and the user's language "
                    "(matching locale/samples) for keys: "
                    "'stall_reply', 'stall_user', 'correction', 'action_req', 'complex', "
                    "'style_concise', 'style_detailed', 'style_bullets', 'style_plain'."
                ),
            },
            ensure_ascii=False,
        )
        resp = await self.agent.auth.call_interactions(
            self.storage,
            model=self.settings.light_model,
            input=prompt,
            store=False,
        )
        parsed = _extract_json(resp.output_text or "{}")
        merged = self.storage.merge_learned_cues(parsed)
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(self.storage)
            for phrase in merged.get("stall_reply", [])[:8]:
                eng.online_stall_step(phrase, is_stall=True)
        except Exception:
            pass
        return merged

    async def generate_daily_digest(self) -> dict[str, Any]:
        """Generate a concise daily digest of system health, usage, thoughts, and learned rules."""
        neural_summary: dict[str, Any] = {}
        world_forecast: dict[str, Any] = {}
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(self.storage)
            neural_summary = eng.get_status()
            world_forecast = eng.predict_world_state()
        except Exception:
            pass
        snapshot = {
            "health": self.watcher.get_health_snapshot(),
            "events": self.watcher.get_events(limit=20),
            "app_usage": self.storage.get_app_usage(),
            "thoughts": self.storage.list_brain_thoughts(limit=15),
            "rules_count": len(self.storage.list_brain_rules()),
            "pending_questions": len(self.storage.list_brain_questions("pending")),
            "neural": neural_summary,
            "world_forecast": world_forecast,
        }
        if not self.agent.auth.is_configured:
            return {"status": "ok", "digest": json.dumps(snapshot, indent=2, ensure_ascii=False)}

        loc = resolve_locale(self.settings.locale)
        resp = await self.agent.auth.call_interactions(
            self.storage,
            model=self.settings.light_model,
            input=json.dumps(snapshot, ensure_ascii=False),
            system_instruction=PromptRegistry(self.storage).render(
                "brain_digest", agent_name=self.settings.agent_name, loc=loc
            ),
            store=False,
        )
        return {"status": "ok", "digest": resp.output_text or ""}

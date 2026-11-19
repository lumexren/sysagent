"""NeuralBrainCortex — unified NeuralEngine + CognitiveBrain fusion layer.

Combines:
1. Predictive Self-Healer (world-model anomaly/bottleneck -> root-cause match -> actionable fix)
2. Dream / Sleep Consolidation Cycle (quiet-hours memory/rule/synapse consolidation)
3. Flow Guardian (context + neural role autopilot for focus & background scheduling)
4. Habit Copilot (recurring habits + neural activation -> one-click schedule suggestions)
"""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime
from typing import Any

import psutil

from .config import Settings
from .storage import Storage


class NeuralBrainCortex:
    """Fuses real-time NeuralEngine forecasts/synapses with CognitiveBrain rules and habits."""

    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        watcher: Any = None,
        brain: Any = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.watcher = watcher
        self.brain = brain
        self._last_heal_ts: float = 0.0
        self._last_dream_ts: float = 0.0
        self._last_habit_ts: float = 0.0
        self._flow_override: str | None = None
        self.last_heal: dict[str, Any] | None = None
        self.last_dream: dict[str, Any] | None = None
        self.last_flow: dict[str, Any] = {
            "mode": "normal",
            "suppress_low_priority": False,
            "allow_heavy_background": True,
            "reason": "initial",
        }

    def evaluate_flow_state(self) -> dict[str, Any]:
        """Classify active user flow mode from watcher context + NeuralEngine world forecast."""
        if self._flow_override:
            mode = self._flow_override
            state = {
                "mode": mode,
                "suppress_low_priority": mode in {"deep_focus", "meeting_media"},
                "allow_heavy_background": mode == "idle_maintenance",
                "reason": "user_override",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.last_flow = state
            return state

        ctx = str(getattr(self.watcher, "current_context", "normal") or "normal")
        wf: dict[str, Any] = {}
        try:
            from .neural import NeuralEngine

            wf = NeuralEngine(self.storage).predict_world_state()
        except Exception:
            wf = {}

        cpu = psutil.cpu_percent(interval=None)
        pred_role = str(wf.get("predicted_role") or "")
        bottleneck = float(wf.get("bottleneck_prob", 0.0) or 0.0)

        if ctx == "media_meeting":
            mode = "meeting_media"
            reason = "active media/meeting context"
        elif ctx == "developer" or pred_role in {"developer", "coding"}:
            mode = "deep_focus"
            reason = f"coding/developer flow (ctx={ctx}, neural_role={pred_role or 'n/a'})"
        elif cpu < 15.0 and bottleneck < 0.35:
            mode = "idle_maintenance"
            reason = f"low load (cpu={cpu:.1f}%, bottleneck={bottleneck:.2f})"
        else:
            mode = "normal"
            reason = f"standard workload (cpu={cpu:.1f}%)"

        state = {
            "mode": mode,
            "suppress_low_priority": mode in {"deep_focus", "meeting_media"},
            "allow_heavy_background": mode == "idle_maintenance" and bottleneck < 0.5,
            "reason": reason,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.last_flow = state
        return state

    def set_flow_override(self, mode: str | None) -> dict[str, Any]:
        """Allow the user to explicitly override (or clear) the active Flow Guardian mode."""
        valid = {"deep_focus", "meeting_media", "idle_maintenance", "normal"}
        self._flow_override = mode if mode in valid else None
        if mode in valid:
            self.storage.add_brain_thought("act", 90, f"Flow Guardian override -> {mode}")
        return self.evaluate_flow_state()

    def _apply_safe_heal(self) -> dict[str, Any]:
        """Run non-destructive preventative relief (unload warm GGUF + SQLite WAL checkpoint)."""
        from .local_inference import _release_all_warm_models

        unloaded = _release_all_warm_models()
        pruned = self.storage.prune_and_checkpoint(stale_days=30)
        self.storage.add_brain_thought(
            "act", 90, f"Predictive Self-Healer applied: unloaded_gguf={unloaded}, pruned_wal={pruned}"
        )
        try:
            from .neural import NeuralEngine

            NeuralEngine(self.storage).online_step("predictive self heal ram cpu", "system_status", reward=1.0)
        except Exception:
            pass
        return {"unloaded_gguf": unloaded, "pruned_thoughts": pruned}

    def evaluate_predictive_healer(
        self, force: bool = False, notify: bool = True
    ) -> dict[str, Any]:
        """Predict impending bottleneck/anomaly, match past root causes, and offer one-click relief."""
        now = time.monotonic()
        if not force and (now - self._last_heal_ts) < 1200.0:
            return {"status": "cooldown", "last_heal": self.last_heal}

        wf: dict[str, Any] = {}
        hits: list[dict[str, Any]] = []
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(self.storage)
            wf = eng.predict_world_state()
            hits = eng.spreading_activation("ram cpu disk bottleneck anomaly spike", top_k=3)
        except Exception:
            wf = {}

        prob = float(wf.get("bottleneck_prob", 0.0) or 0.0)
        anomaly = bool(wf.get("anomaly_detected"))
        if not force and not anomaly and prob < 0.70:
            return {"status": "healthy", "bottleneck_prob": prob, "anomaly_detected": anomaly}

        top_proc = ""
        try:
            procs = sorted(
                psutil.process_iter(["name", "memory_percent"]),
                key=lambda p: p.info.get("memory_percent") or 0.0,
                reverse=True,
            )
            if procs:
                top_proc = f"{procs[0].info['name']} ({(procs[0].info.get('memory_percent') or 0.0):.1f}% RAM)"
        except Exception:
            pass

        lessons = self.storage.list_lessons(limit=5)
        cause_hint = (
            str(hits[0].get("key"))
            if hits
            else (str(lessons[0].get("lesson", ""))[:60] if lessons else top_proc or "resource spike")
        )
        summary = (
            f"10m forecast: CPU {wf.get('pred_cpu_10m', '?')}%, RAM {wf.get('pred_ram_10m', '?')}% "
            f"(bottleneck={prob:.2f}). Likely factor: {cause_hint}."
        )
        if top_proc:
            summary += f" Top RAM: {top_proc}."

        self._last_heal_ts = now
        result = {
            "status": "alerted",
            "bottleneck_prob": prob,
            "anomaly_detected": anomaly,
            "cause_hint": cause_hint,
            "top_process": top_proc,
            "summary": summary,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.last_heal = result

        if notify and self.watcher is not None and hasattr(self.watcher, "_send_desktop_notification"):
            def _on_action(act: str) -> None:
                if act == "heal":
                    self._apply_safe_heal()
                elif act == "ignore":
                    self.storage.record_notification_feedback("cortex:predictive_heal", "dismiss")
                    try:
                        from .neural import NeuralEngine

                        NeuralEngine(self.storage).online_step(
                            "predictive self heal", "system_status", reward=-0.6
                        )
                    except Exception:
                        pass

            self.watcher._send_desktop_notification(
                f"{self.settings.agent_name} — Predictive Health",
                summary,
                urgency="critical" if prob >= 0.85 else "normal",
                category="health",
                actions=[("heal", "Free RAM & Clean"), ("ignore", "Ignore")],
                on_action=_on_action,
            )
        return result

    def run_dream_cycle(self, force: bool = False) -> dict[str, Any]:
        """Consolidate memories, decay stale rules, distill recurring patterns, and sync Neural synapses."""
        now = time.monotonic()
        if not force:
            quiet = bool(self.brain and self.brain.is_quiet_hours())
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            if not quiet or cpu >= 35.0 or ram >= 75.0 or (now - self._last_dream_ts) < 43200.0:
                return {"status": "skipped", "reason": "not_quiet_or_cooldown"}

        decayed_mem = 0
        expired_mem = 0
        conflicts_resolved = 0
        decayed_rules = 0
        pruned_thoughts = 0
        try:
            decayed_mem = int(self.storage.decay_memories() or 0)
        except Exception:
            pass
        try:
            expired_mem = int(self.storage.prune_expired_memories() or 0)
        except Exception:
            pass
        try:
            conflicts_resolved = int(self.storage.resolve_memory_conflicts() or 0)
        except Exception:
            pass
        try:
            decayed_rules = int(self.storage.decay_stale_rules() or 0)
        except Exception:
            pass
        try:
            pruned_thoughts = int(self.storage.prune_and_checkpoint(stale_days=21) or 0)
        except Exception:
            pass

        # Distill repeated brain thoughts into a summary rule candidate count
        thoughts = self.storage.list_brain_thoughts(limit=80)
        decision_counts = Counter(str(t.get("decision") or "") for t in thoughts)

        synapses = 0
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(self.storage)
            eng.sync_associative_memory()
            synapses = int(eng.get_status().get("synapses", 0))
        except Exception:
            pass

        self._last_dream_ts = now
        report = {
            "status": "completed",
            "decayed_memories": decayed_mem,
            "expired_memories": expired_mem,
            "conflicts_resolved": conflicts_resolved,
            "decayed_rules": decayed_rules,
            "pruned_thoughts": pruned_thoughts,
            "thought_summary": dict(decision_counts),
            "synapses": synapses,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.last_dream = report
        self.storage.add_brain_thought(
            "dream",
            95,
            f"Dream consolidation: pruned={pruned_thoughts}, decayed_rules={decayed_rules}, synapses={synapses}",
        )
        return report

    def suggest_habit_automations(self, notify: bool = False) -> list[dict[str, Any]]:
        """Combine recurring habits with Neural activation to suggest unscheduled automations."""
        habits = self.storage.get_recurring_habits(min_count=3)
        existing_prompts = {
            str(a.get("prompt", "")).strip().lower() for a in self.storage.list_automations()
        }
        suggestions: list[dict[str, Any]] = []

        eng = None
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(self.storage)
        except Exception:
            eng = None

        for h in habits[:8]:
            action_txt = str(h.get("detail") or h.get("action") or h.get("category") or "").strip()
            if not action_txt:
                continue
            hour = int(h.get("hour_bucket", h.get("hour", 9)) or 9)
            tval = f"{hour % 24:02d}:00"
            prompt = f"Check and assist with recurring habit: {action_txt}"
            if prompt.lower() in existing_prompts:
                continue
            boost = 0.0
            if eng is not None:
                try:
                    acts = eng.spreading_activation(action_txt, top_k=2)
                    boost = max((float(a.get("activation", 0.0) or 0.0) for a in acts), default=0.0)
                except Exception:
                    boost = 0.0
            occ = int(h.get("hits", h.get("count", 3)) or 3)
            conf = round(min(0.99, 0.55 + 0.05 * occ + 0.25 * boost), 2)
            slug = "".join(ch if ch.isalnum() else "-" for ch in action_txt.lower())[:24].strip("-") or "habit"
            suggestions.append(
                {
                    "name": f"habit-{slug}",
                    "trigger_type": "daily_time",
                    "trigger_value": tval,
                    "prompt": prompt,
                    "occurrences": occ,
                    "confidence": conf,
                }
            )

        suggestions.sort(key=lambda s: float(s["confidence"]), reverse=True)

        now = time.monotonic()
        if (
            notify
            and suggestions
            and (now - self._last_habit_ts) >= 21600.0
            and self.watcher is not None
            and hasattr(self.watcher, "_send_desktop_notification")
        ):
            top = suggestions[0]
            self._last_habit_ts = now

            def _on_habit(act: str) -> None:
                if act == "automate":
                    self.storage.create_automation(
                        top["name"], top["trigger_type"], top["trigger_value"], top["prompt"]
                    )
                    self.storage.add_brain_thought(
                        "act", 90, f"Habit Copilot created automation '{top['name']}' ({top['trigger_value']})"
                    )
                else:
                    self.storage.record_notification_feedback(f"habit:{top['name']}", "dismiss")

            self.watcher._send_desktop_notification(
                f"{self.settings.agent_name} — Habit Suggestion",
                f"Automate '{top['name']}' daily at {top['trigger_value']}? ({top['prompt'][:120]})",
                urgency="low",
                category="info",
                actions=[("automate", "Automate"), ("dismiss", "Dismiss")],
                on_action=_on_habit,
            )

        return suggestions[:5]

    def tick(self) -> dict[str, Any]:
        """Run one lightweight Cortex cycle (Flow -> Predictive Healer -> Dream -> Habit Copilot)."""
        flow = self.evaluate_flow_state()
        heal = self.evaluate_predictive_healer(force=False, notify=not flow["suppress_low_priority"])
        dream = self.run_dream_cycle(force=False)
        habits = self.suggest_habit_automations(notify=not flow["suppress_low_priority"])
        return {"flow": flow, "heal": heal, "dream": dream, "habit_suggestions": len(habits)}

    def get_status(self) -> dict[str, Any]:
        return {
            "flow": self.evaluate_flow_state(),
            "last_heal": self.last_heal,
            "last_dream": self.last_dream,
            "habit_suggestions": self.suggest_habit_automations(notify=False),
        }

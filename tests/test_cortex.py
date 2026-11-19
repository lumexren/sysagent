"""Tests for NeuralBrainCortex (Flow Guardian, Predictive Self-Healer, Dream Cycle, Habit Copilot)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sysagent.config import Settings
from sysagent.cortex import NeuralBrainCortex
from sysagent.storage import Storage


class CortexTests(unittest.TestCase):
    def _settings(self) -> Settings:
        s = Settings()
        s.agent_name = "Vex"
        return s

    def test_flow_guardian_and_override(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "c1.db") as st:
            fake_watcher = SimpleNamespace(current_context="developer")
            cortex = NeuralBrainCortex(self._settings(), st, watcher=fake_watcher)
            fake_eng = SimpleNamespace(
                predict_world_state=lambda: {"predicted_role": "developer", "bottleneck_prob": 0.1}
            )
            with patch("sysagent.neural.NeuralEngine", return_value=fake_eng):
                flow = cortex.evaluate_flow_state()
                self.assertEqual(flow["mode"], "deep_focus")
                self.assertTrue(flow["suppress_low_priority"])

                over = cortex.set_flow_override("idle_maintenance")
                self.assertEqual(over["mode"], "idle_maintenance")
                self.assertTrue(over["allow_heavy_background"])

    def test_predictive_self_healer_and_action(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "c2.db") as st:
            sent: list[dict] = []

            class _FakeWatcher:
                current_context = "normal"

                def _send_desktop_notification(self, title, body, **kwargs):
                    sent.append({"title": title, "body": body, **kwargs})

            cortex = NeuralBrainCortex(self._settings(), st, watcher=_FakeWatcher())
            fake_eng = SimpleNamespace(
                predict_world_state=lambda: {
                    "pred_cpu_10m": 88.0,
                    "pred_ram_10m": 92.0,
                    "bottleneck_prob": 0.88,
                    "anomaly_detected": True,
                },
                spreading_activation=lambda txt, top_k=3: [{"key": "python_leak", "activation": 0.9}],
                online_step=lambda *a, **kw: None,
            )
            with patch("sysagent.neural.NeuralEngine", return_value=fake_eng), patch(
                "sysagent.local_inference._release_all_warm_models", return_value=True
            ):
                res = cortex.evaluate_predictive_healer(force=True, notify=True)
                self.assertEqual(res["status"], "alerted")
                self.assertEqual(res["cause_hint"], "python_leak")
                self.assertEqual(len(sent), 1)
                # Trigger the "heal" action callback
                sent[0]["on_action"]("heal")
                thoughts = st.list_brain_thoughts(limit=5)
                self.assertTrue(any("Predictive Self-Healer" in t["summary"] for t in thoughts))

    def test_dream_cycle_and_habit_copilot(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "c3.db") as st:
            cortex = NeuralBrainCortex(self._settings(), st)
            fake_eng = SimpleNamespace(
                sync_associative_memory=lambda: None,
                get_status=lambda: {"synapses": 120},
                spreading_activation=lambda txt, top_k=2: [{"key": "git", "activation": 0.75}],
            )
            with patch("sysagent.neural.NeuralEngine", return_value=fake_eng):
                dream = cortex.run_dream_cycle(force=True)
                self.assertEqual(dream["status"], "completed")
                self.assertEqual(dream["synapses"], 120)

                with patch.object(
                    st,
                    "get_recurring_habits",
                    return_value=[{"action": "git status check", "hour": 9, "count": 5}],
                ):
                    sugs = cortex.suggest_habit_automations(notify=False)
                    self.assertEqual(len(sugs), 1)
                    self.assertEqual(sugs[0]["trigger_value"], "09:00")
                    self.assertGreater(sugs[0]["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()

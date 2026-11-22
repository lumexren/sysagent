"""Tests for SelfGuard (Autopsy, Calibration, Canary, Profiler, Skill Synthesizer)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sysagent.config import Settings
from sysagent.self_guard import SelfGuard
from sysagent.storage import Storage


class SelfGuardTests(unittest.TestCase):
    def _settings(self) -> Settings:
        s = Settings()
        s.agent_name = "Vex"
        return s

    def test_crash_autopsy_and_auto_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "sg1.db") as st:
            marker = Path(td) / "daemon.running"
            marker.write_text("{}", encoding="utf-8")
            rolled: list[int] = []
            fake_ev = SimpleNamespace(rollback_last_evolve=lambda pid: (rolled.append(pid) or True))
            sg = SelfGuard(self._settings(), st, evolver=fake_ev, marker_file=marker)

            with patch.object(
                st,
                "list_evolve_proposals",
                return_value=[{"id": 99, "status": "applied", "target_file": "sysagent/broken.py"}],
            ):
                res = sg.run_crash_autopsy(
                    journal_text="Traceback: ImportError in sysagent/broken.py"
                )
            self.assertTrue(res["crashed"])
            self.assertEqual(res["signature"], "ImportError")
            self.assertTrue(res["rolled_back"])
            self.assertEqual(rolled, [99])

            sg.mark_clean_shutdown()
            self.assertFalse(marker.exists())

    def test_calibration_canary_profiler_and_skill_synthesizer(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "sg2.db") as st:
            sg = SelfGuard(self._settings(), st, marker_file=Path(td) / "daemon.running")

            fake_eng = SimpleNamespace(
                get_status=lambda: {"tool_accuracy": 0.94, "world_forecast": {"forecast_mae": 3.2}},
                predict_world_state=lambda: {"samples": 12},
            )
            with patch("sysagent.neural.NeuralEngine", return_value=fake_eng):
                cal = sg.calibrate_thresholds()
                self.assertEqual(cal["local_tool_threshold"], 0.64)
                saved = json.loads(st.get_memory("_internal:self_calibration") or "{}")
                self.assertEqual(saved["local_max_words"], 26)

                canary = sg.run_canary_probes()
                self.assertTrue(canary["ok"])
                self.assertTrue(canary["probes"]["sqlite"]["ok"])

            with patch("sysagent.local_inference._release_all_warm_models", return_value=True):
                prof = sg.run_memory_profiler(max_rss_mb=0.1)
                self.assertTrue(prof["trimmed"])

            for i in range(4):
                st.record_file_backup("edit_file", f"/tmp/f{i}.py", True, "old")
                st.record_file_backup("write_file", f"/tmp/g{i}.py", False, None)
            skills = sg.synthesize_skill_candidates(min_occurrences=3)
            self.assertTrue(any(s["chain"] == ["edit_file", "write_file"] for s in skills))


if __name__ == "__main__":
    unittest.main()

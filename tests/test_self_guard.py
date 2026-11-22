"""Tests for SelfGuard (Autopsy, Calibration, Canary, Profiler, Skill Synthesizer)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sysagent.config import Settings
from sysagent.local_mind import LocalMind, _ThinkStreamFilter
from sysagent.self_guard import DEFAULT_CALIBRATION, SelfGuard, get_calibration, match_cached_fix
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

    def test_get_calibration_and_match_cached_fix(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "sg3.db") as st:
            self.assertEqual(get_calibration(st), DEFAULT_CALIBRATION)
            st.set_memory(
                "_internal:self_calibration",
                json.dumps(
                    {
                        "local_tool_threshold": 0.64,
                        "local_max_words": 28,
                        "bottleneck_defer_threshold": 0.82,
                    }
                ),
            )
            cal = get_calibration(st)
            self.assertEqual(cal["local_tool_threshold"], 0.64)
            self.assertEqual(cal["local_max_words"], 28)
            self.assertEqual(cal["bottleneck_defer_threshold"], 0.82)

            st.add_lesson("self_autopsy", "Crash autopsy detected SIGSEGV on pacman -Syu", "runtime")
            self.assertIsNone(match_cached_fix(st, "pacman -Syu", 1))

            st.add_lesson(
                "pacman_lock",
                "If pacman -Syu fails with exit code 1 due to db.lck, remove /var/lib/pacman/db.lck after verifying no pacman is running.",
                "arch",
            )
            hit = match_cached_fix(st, "sudo pacman -Syu", 1)
            self.assertIsNotNone(hit)
            assert hit is not None
            self.assertEqual(hit["topic"], "pacman_lock")
            self.assertGreaterEqual(hit["score"], 0.85)
            self.assertIsNone(match_cached_fix(st, "docker compose up", 127))

    def test_think_stream_filter_and_sse_streaming(self) -> None:
        filt = _ThinkStreamFilter()
        tokens = ["<", "think", ">", "internal reasoning", "</", "think", ">", "\n\n", "Selam", " dünya!"]
        visible = "".join(filt.feed(t) for t in tokens) + filt.flush()
        self.assertEqual(visible, "Selam dünya!")

        filt2 = _ThinkStreamFilter()
        self.assertEqual(filt2.feed("2 < 3 ve ") + filt2.feed("4 > 1") + filt2.flush(), "2 < 3 ve 4 > 1")

        class _FakeResp:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def iter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"<think>"}}]}'
                yield 'data: {"choices":[{"delta":{"content":"secret"}}]}'
                yield 'data: {"choices":[{"delta":{"content":"</think>\\nMerhaba"}}]}'
                yield 'data: {"choices":[{"delta":{"content":" Vex!"}}]}'
                yield "data: [DONE]"

        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "sg4.db") as st:
            fake_gguf = Path(td) / "qwen3-4b.gguf"
            fake_gguf.write_bytes(b"gguf")
            mind = LocalMind(st)
            deltas: list[str] = []
            with (
                patch("sysagent.local_mind.find_all_gguf_models", return_value=[fake_gguf]),
                patch("sysagent.local_mind.find_gguf_model", return_value=fake_gguf),
                patch.object(mind, "_ensure_llama_server", return_value="http://127.0.0.1:8769"),
                patch("httpx.stream", return_value=_FakeResp()),
            ):
                out = mind.stream_gguf_inference("naber", on_delta=deltas.append)
            self.assertEqual(out, "Merhaba Vex!")
            self.assertEqual("".join(deltas), "Merhaba Vex!")


if __name__ == "__main__":
    unittest.main()

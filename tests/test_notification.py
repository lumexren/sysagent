"""Tests for centralized NotificationCenter (Brain/Neural gates, dedup, cancelable actions)."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sysagent.config import Settings
from sysagent.notification import NotificationCenter
from sysagent.storage import Storage


class NotificationCenterTests(unittest.TestCase):
    def _make_settings(self) -> Settings:
        s = Settings()
        s.agent_name = "Vex"
        s.quiet_hours = ""
        return s

    def test_brain_and_neural_suppression_gates(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "n.db") as st:
            nc = NotificationCenter(self._make_settings(), st)
            with patch.object(st, "is_ignored_by_brain_rule", return_value=True):
                self.assertIsNone(nc.send("Disk warning", "low space", category="health"))
            self.assertEqual(nc.list_recent(1)[0]["status"], "suppressed:brain_ignore_rule")

            fake_eng = SimpleNamespace(
                spreading_activation=lambda txt, top_k=2: [{"key": "brain_dismiss", "activation": 0.85}]
            )
            with patch("sysagent.neural.NeuralEngine", return_value=fake_eng):
                self.assertIsNone(nc.send("Minor info", "routine update", category="info"))
            self.assertEqual(nc.list_recent(1)[0]["status"], "suppressed:neural_dismiss_learned")

            # Approval notifications must bypass Brain/Neural suppression
            done = threading.Event()
            seen: list[str] = []

            class _FakePopen:
                def __init__(self, *a, **kw):
                    self._done = False

                def communicate(self, timeout=None):
                    self._done = True
                    return ("approve\n", "")

                def poll(self):
                    return 0 if self._done else None

                def terminate(self):
                    self._done = True

            with patch("shutil.which", return_value="/usr/bin/notify-send"), patch(
                "subprocess.Popen", side_effect=_FakePopen
            ), patch("sysagent.neural.NeuralEngine", return_value=fake_eng):
                h = nc.send(
                    "Vex — Approval Required",
                    "Run command",
                    category="approval",
                    actions=[("approve", "Approve"), ("deny", "Deny")],
                    on_action=lambda act: (seen.append(act), done.set()),
                )
                self.assertIsNotNone(h)
                self.assertTrue(done.wait(timeout=2.0))
                self.assertEqual(seen, ["approve"])

    def test_dedup_and_cancel_handle(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "n2.db") as st:
            nc = NotificationCenter(self._make_settings(), st)
            terminated = threading.Event()

            class _BlockingPopen:
                def __init__(self, *a, **kw):
                    self.ev = threading.Event()

                def communicate(self, timeout=None):
                    self.ev.wait(timeout=2.0)
                    return ("", "")

                def poll(self):
                    return 0 if self.ev.is_set() else None

                def terminate(self):
                    terminated.set()
                    self.ev.set()

            with patch("shutil.which", return_value="/usr/bin/notify-send"), patch(
                "subprocess.Popen", side_effect=_BlockingPopen
            ):
                h1 = nc.send("Alert", "Same body", category="info", dedup_sec=60.0)
                self.assertIsNotNone(h1)
                h2 = nc.send("Alert", "Same body", category="info", dedup_sec=60.0)
                self.assertIsNone(h2)
                self.assertEqual(nc.list_recent(1)[0]["status"], "deduplicated")
                h1.cancel()
                self.assertTrue(terminated.wait(timeout=2.0))


if __name__ == "__main__":
    unittest.main()

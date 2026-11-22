"""Regression tests for codebase-wide logic, security, and integration fixes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sysagent.brain import CognitiveBrain
from sysagent.config import Settings
from sysagent.ipc import _resolve_uds
from sysagent.mcp_server import handle_mcp_request
from sysagent.neural import NeuralEngine
from sysagent.safety import assess_tool_call
from sysagent.self_guard import SelfGuard
from sysagent.snapshots import perform_rollback
from sysagent.storage import Storage
from sysagent.tools import execute_tool


class AuditRegressionTests(unittest.TestCase):
    def test_newline_command_injection_requires_approval(self) -> None:
        res = assess_tool_call("run_command", {"command": "ls\npython3 -c 'print(1)'"}, mode="smart")
        self.assertTrue(res.requires_approval)

    def test_edit_file_rejects_empty_target(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "t1.db") as st:
            f = Path(td) / "sample.txt"
            f.write_text("hello", encoding="utf-8")
            out = json.loads(
                execute_tool("edit_file", {"path": str(f), "target": "", "replacement": "X"}, None, st)
            )
            self.assertIn("error", out)
            self.assertEqual(f.read_text(encoding="utf-8"), "hello")

    def test_storage_fts_preserved_and_zero_arg_conflict_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "t2.db"
            with Storage(db_path=db) as st1:
                first_fts = st1._fts_enabled
                st1.set_memory("editor", "vim")
                st1.set_memory("editor:pref", "neovim")
            with Storage(db_path=db) as st2:
                self.assertEqual(st2._fts_enabled, first_fts)
                resolved = st2.resolve_memory_conflicts()
                self.assertGreaterEqual(int(resolved), 1)

    def test_rollback_nothing_to_undo_returns_ok_false(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "t3.db") as st:
            res = perform_rollback("file", None, st)
            self.assertFalse(res["ok"])

    def test_ipc_resolve_uds_does_not_treat_api_path_as_socket(self) -> None:
        uds = _resolve_uds("/api/status")
        self.assertIsNotNone(uds)
        self.assertNotEqual(uds, "/api/status")
        self.assertEqual(_resolve_uds("/tmp/custom.sock"), "/tmp/custom.sock")

    def test_mcp_jsonrpc_malformed_and_notification(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "t4.db") as st:
            err = handle_mcp_request([1, 2, 3], st)
            self.assertIsNotNone(err)
            assert err is not None
            self.assertEqual(err["error"]["code"], -32600)
            self.assertIsNone(handle_mcp_request({"jsonrpc": "2.0", "method": "ping"}, st))

    def test_brain_dry_run_anomaly_and_self_guard_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "t5.db") as st:
            s = Settings()
            brain = CognitiveBrain(s, SimpleNamespace(), st, SimpleNamespace())
            ok, _reason = brain._dry_run_validate_action(
                "restart service", confidence=70, world_forecast={"anomaly_detected": True, "bottleneck_prob": 0.4}
            )
            self.assertFalse(ok)

            rolled: list[int] = []
            fake_ev = SimpleNamespace(rollback_last_evolve=lambda pid: (rolled.append(pid) or True))
            sg = SelfGuard(s, st, evolver=fake_ev, marker_file=Path(td) / "daemon.running")
            with patch.object(
                st,
                "list_evolve_proposals",
                return_value=[{"id": 42, "status": "applied", "file_path": "/repo/sysagent/mod.py"}],
            ):
                res = sg.run_crash_autopsy(journal_text="SyntaxError in mod.py")
            self.assertTrue(res["rolled_back"])
            self.assertEqual(rolled, [42])

    def test_neural_observe_world_state_accepts_dict_snap(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "t6.db") as st:
            eng = NeuralEngine(
                st,
                npz_file=Path(td) / "neural_engine.npz",
                meta_file=Path(td) / "neural_meta.json",
            )
            eng.observe_world_state(
                {
                    "cpu_percent": 42.5,
                    "ram_percent": 61.0,
                    "gpu": {"util_percent": 55.0, "vram_used_mb": 4096, "vram_total_mb": 8192},
                }
            )
            hist = eng.meta.get("telemetry_history") or []
            self.assertTrue(hist)
            self.assertEqual(hist[-1][1], 42.5)
            self.assertEqual(hist[-1][3], 55.0)
            self.assertEqual(hist[-1][4], 50.0)


if __name__ == "__main__":
    unittest.main()

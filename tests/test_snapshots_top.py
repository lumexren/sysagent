"""Tests for unified snapshots/rollback and `sysagent top` dashboard."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.layout import Layout

from sysagent import snapshots
from sysagent.storage import Storage
from sysagent.top import build_dashboard, run_top


class SnapshotsAndTopTests(unittest.TestCase):
    def test_config_snapshot_save_list_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            with patch.object(snapshots, "DATA_DIR", tmp_path):
                cfg = tmp_path / "demo.conf"
                cfg.write_text("PORT=8080\n", encoding="utf-8")

                snap_id = snapshots.save_config_snapshot(cfg, "PORT=8080\n")
                self.assertIsNotNone(snap_id)

                items = snapshots.list_config_snapshots(limit=10)
                self.assertEqual(len(items), 1)
                self.assertEqual(items[0]["id"], snap_id)
                self.assertEqual(items[0]["path"], str(cfg))
                self.assertTrue(items[0]["restorable"])

                cfg.write_text("PORT=9999\n", encoding="utf-8")
                res = snapshots.restore_config_snapshot(snap_id)
                self.assertTrue(res["ok"])
                self.assertEqual(cfg.read_text(encoding="utf-8"), "PORT=8080\n")

    def test_perform_rollback_file_and_evolve(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            with patch.object(snapshots, "DATA_DIR", tmp_path):
                st = Storage(db_path=tmp_path / "test.db")
                target = tmp_path / "notes.txt"
                target.write_text("new", encoding="utf-8")
                st.record_file_backup("write_file", str(target), True, "old", batch_id="b1")

                res_file = snapshots.perform_rollback("file", "b1", st, watcher=None, plugins=None)
                self.assertTrue(res_file["ok"])
                self.assertEqual(target.read_text(encoding="utf-8"), "old")

                called: list[int | None] = []
                fake_evolver = SimpleNamespace(
                    rollback_last_evolve=lambda pid=None: called.append(pid) or True
                )
                res_ev = snapshots.perform_rollback("evolve", "42", st, evolver=fake_evolver)
                self.assertEqual(res_ev, {"ok": True, "target": "evolve", "proposal_id": 42})
                self.assertEqual(called, [42])

    def test_top_build_dashboard_and_once(self) -> None:
        sample = {
            "online": True,
            "status": {
                "agent_name": "SysAgent",
                "model": "gemini-4-argon",
                "uptime_sec": 3661,
                "pending_approvals": 1,
                "pending_brain_questions": 2,
                "brain_rules": 5,
                "memories": 12,
                "lessons": 3,
                "evolve_proposals": 1,
                "evolve": {"auto_apply": True},
                "health": {
                    "cpu_percent": 22.5,
                    "ram_percent": 48.0,
                    "ram_used_gb": 7.6,
                    "ram_total_gb": 16.0,
                    "disk_percent": 55.0,
                    "disk_free_gb": 120.5,
                    "disk_burn_rate_gb_hr": -0.05,
                    "max_temp_c": 54.0,
                    "zombie_count": 0,
                },
            },
            "events": [
                {"timestamp": "2026-11-19T02:58:00", "source": "watcher", "message": "all good"}
            ],
            "snapshots": {
                "audit_log": [
                    {"id": 1, "action": "edit_file", "target_path": "/tmp/a.conf", "undone": 0}
                ]
            },
        }
        layout = build_dashboard(sample)
        self.assertIsInstance(layout, Layout)

        with patch("sysagent.top.collect_top_state", return_value=sample):
            run_top("http://127.0.0.1:8765", {}, once=True)


if __name__ == "__main__":
    unittest.main()

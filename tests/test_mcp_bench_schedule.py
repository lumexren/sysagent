"""Tests for MCP stdio server, local GGUF benchmark, and schedule CLI/storage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sysagent import bench
from sysagent.cli import _parse_schedule_spec
from sysagent.mcp_server import handle_mcp_request
from sysagent.storage import Storage


class McpBenchScheduleTests(unittest.TestCase):
    def test_mcp_initialize_list_and_call(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "m.db") as st:
            st.create_automation("daily-check", "daily_time", "09:00", "summarize health")

            init_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, st)
            self.assertIsNotNone(init_resp)
            self.assertEqual(init_resp["result"]["serverInfo"]["name"], "sysagent")

            self.assertIsNone(
                handle_mcp_request({"jsonrpc": "2.0", "method": "notifications/initialized"}, st)
            )

            list_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, st)
            names = [t["name"] for t in list_resp["result"]["tools"]]
            self.assertIn("sysagent_status", names)
            self.assertIn("sysagent_schedule_list", names)

            call_resp = handle_mcp_request(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "sysagent_schedule_list", "arguments": {}},
                },
                st,
            )
            self.assertFalse(call_resp["result"]["isError"])
            body = json.loads(call_resp["result"]["content"][0]["text"])
            self.assertEqual(len(body["automations"]), 1)
            self.assertEqual(body["automations"][0]["name"], "daily-check")

    def test_benchmark_local_models_mocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m1 = Path(td) / "gemma-3-4b-it-Q4_K_M.gguf"
            m1.write_bytes(b"0" * 1024)
            with patch.object(bench, "find_all_gguf_models", return_value=[m1]), patch.object(
                bench, "classify_gguf_role", return_value="chat"
            ), patch.object(bench, "_has_ram_for_gguf", return_value=True), patch.object(
                bench, "_release_all_warm_models", return_value=True
            ), patch.object(
                bench.LocalMind,
                "_run_single_gguf",
                return_value="systemd is the init system and service manager for Linux.",
            ):
                rep = bench.benchmark_local_models(role_filter="chat")
                self.assertEqual(rep["count"], 1)
                self.assertEqual(rep["fastest_role"], "chat")
                self.assertEqual(rep["results"][0]["status"], "ok")
                self.assertGreater(rep["results"][0]["tok_per_s"], 0)

    def test_schedule_spec_parser_and_toggle(self) -> None:
        self.assertEqual(_parse_schedule_spec("9:05"), ("daily_time", "09:05"))
        self.assertEqual(_parse_schedule_spec("30m"), ("interval", "1800"))
        self.assertEqual(_parse_schedule_spec("2h"), ("interval", "7200"))
        self.assertEqual(_parse_schedule_spec("file:*.log"), ("file_created", "*.log"))
        self.assertEqual(_parse_schedule_spec("neural:anomaly"), ("neural_anomaly", "anomaly"))
        self.assertEqual(_parse_schedule_spec("neural:bottleneck"), ("neural_bottleneck", "0.70"))
        self.assertEqual(_parse_schedule_spec("neural:bottleneck:0.85"), ("neural_bottleneck", "0.85"))
        with self.assertRaises(ValueError):
            _parse_schedule_spec("every-blue-moon")

        from sysagent.cli import _prog_name

        with patch("sys.argv", ["/usr/local/bin/vex", "schedule"]):
            self.assertEqual(_prog_name(), "vex")

        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "s.db") as st:
            aid = st.create_automation("cleanup", "interval", "3600", "clean tmp")
            self.assertFalse(st.toggle_automation(aid))
            self.assertTrue(st.toggle_automation(aid))
            self.assertTrue(st.delete_automation(aid))
            self.assertIsNone(st.toggle_automation(aid))

    def test_watcher_neural_and_brain_schedule_gates(self) -> None:
        import asyncio
        from types import SimpleNamespace

        from sysagent.watcher import SystemWatcher

        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "w.db") as st:
            st.create_automation("anom-watch", "neural_anomaly", "anomaly", "inspect spike")
            st.create_automation("routine-clean", "interval", "60", "clean cache")
            st.create_automation("ignored-job", "neural_anomaly", "anomaly", "noisy job")

            ran: list[str] = []

            async def cb(auto, ctx):
                ran.append(auto["name"])

            w = SystemWatcher.__new__(SystemWatcher)
            w.storage = st
            w.automation_callback = cb
            w.events = []
            w.add_event = lambda *a, **kw: None
            w.current_context = "general"

            fake_wf = {"anomaly_detected": True, "anomaly_score": 1.8, "bottleneck_prob": 0.85}
            fake_engine = SimpleNamespace(
                predict_world_state=lambda: fake_wf,
                online_step=lambda *a, **kw: None,
            )
            with patch("sysagent.neural.NeuralEngine", return_value=fake_engine), patch.object(
                st, "is_ignored_by_brain_rule", side_effect=lambda txt: "noisy job" in txt
            ):
                asyncio.run(w._check_automations([]))

            # anom-watch ran; routine-clean deferred (bottleneck 0.85 >= 0.75); ignored-job skipped by Brain rule
            self.assertEqual(ran, ["anom-watch"])


if __name__ == "__main__":
    unittest.main()

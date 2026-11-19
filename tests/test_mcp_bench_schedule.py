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
        with self.assertRaises(ValueError):
            _parse_schedule_spec("every-blue-moon")

        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "s.db") as st:
            aid = st.create_automation("cleanup", "interval", "3600", "clean tmp")
            self.assertFalse(st.toggle_automation(aid))
            self.assertTrue(st.toggle_automation(aid))
            self.assertTrue(st.delete_automation(aid))
            self.assertIsNone(st.toggle_automation(aid))


if __name__ == "__main__":
    unittest.main()

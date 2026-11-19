"""Consolidated regression test suite (migrated from legacy test_evolve_*.py scripts)."""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import sysagent.auth as auth_mod
import sysagent.local_mind as lm
from sysagent import brain
from sysagent.agent import AgentManager, PendingApproval
from sysagent.auth import AgyCliClient, AuthManager
from sysagent.config import Settings
from sysagent.evolve import SelfEvolver
from sysagent.server import TerminalCommand, api_terminal_run
from sysagent.storage import Storage
from sysagent.tools import execute_tool


class TestAuthRegressions(unittest.TestCase):
    def test_rotate_adc_to_agy_cli(self):
        s = Settings(gemini_api_key="", gemini_api_keys=[], auth_mode="adc")
        mgr = AuthManager(s)
        mgr._initialized = True
        mgr._status = {"mode": "adc", "configured": True, "project": "broken-proj"}
        with patch("sysagent.auth._find_agy_cli", return_value="/usr/bin/agy"):
            self.assertTrue(mgr.rotate_key())
            self.assertEqual(mgr.status["mode"], "agy_cli")
            self.assertIsInstance(mgr.get_client(), AgyCliClient)

    def test_rotate_api_key_to_agy_cli(self):
        with patch.object(auth_mod, "_find_agy_cli", return_value="/usr/bin/agy"):
            s = Settings(gemini_api_key="bad-key", gemini_api_keys=["bad-key"], auth_mode="api_key")
            mgr = auth_mod.AuthManager(s)
            mgr.initialize()
            self.assertEqual(mgr.status["mode"], "api_key")
            self.assertTrue(mgr.rotate_key())
            self.assertEqual(mgr.status["mode"], "agy_cli")
            self.assertIsInstance(mgr.get_client(), auth_mod.AgyCliClient)

    def test_rotate_agy_cli_back_to_api_key(self):
        s = Settings(gemini_api_key="test-key-1")
        am = AuthManager(s)
        am._initialized = True
        am._keys = ["test-key-1"]
        am._status = {"mode": "agy_cli", "configured": True, "project": "google-ai-pro"}
        with patch("sysagent.auth.genai.Client") as mock_client:
            self.assertTrue(am.rotate_key())
            self.assertEqual(am._status["mode"], "api_key")
            mock_client.assert_called_once_with(api_key="test-key-1")

    def test_call_interactions_rotates_on_permission_denied(self):
        settings = Settings(gemini_api_key="", gemini_api_keys=["bad_key_1", "good_key_2"], auth_mode="api_key")
        auth = AuthManager(settings)
        auth.initialize()
        calls: list[str] = []

        class DummyInteractions:
            def __init__(self, key: str):
                self.key = key

            def create(self, **kwargs):
                calls.append(self.key)
                if self.key == "bad_key_1":
                    raise RuntimeError("403 PERMISSION_DENIED: Billing account not configured")
                return SimpleNamespace(output_text="ok")

        orig_rotate = auth.rotate_key
        auth._client = SimpleNamespace(interactions=DummyInteractions("bad_key_1"))

        def mock_rotate():
            res = orig_rotate()
            auth._client = SimpleNamespace(interactions=DummyInteractions(auth._keys[auth._key_index]))
            return res

        auth.rotate_key = mock_rotate
        res = asyncio.run(auth.call_interactions(None, "gemini-3.5-flash-lite", input="hi"))
        self.assertEqual(res.output_text, "ok")
        self.assertEqual(calls, ["bad_key_1", "good_key_2"])


class TestToolsAndStorageRegressions(unittest.TestCase):
    def test_search_files_skips_node_modules_and_caps_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "node_modules").mkdir()
            (root / "node_modules" / "ignored_target.txt").write_text("hello", encoding="utf-8")
            for i in range(60):
                (root / f"match_target_{i}.txt").write_text("hello world", encoding="utf-8")

            res = json.loads(execute_tool("search_files", {"query": "target", "path": str(root)}, None, None))
            self.assertEqual(len(res["matches"]), 50)
            self.assertTrue(all("node_modules" not in m for m in res["matches"]))

            res_glob = json.loads(execute_tool("search_files", {"query": "*.txt", "path": str(root)}, None, None))
            self.assertEqual(len(res_glob["matches"]), 50)

    def test_search_files_content_search_caps_matches(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sample.txt"
            p.write_text("\n".join(f"match_line_{i}" for i in range(100)), encoding="utf-8")
            res = json.loads(
                execute_tool(
                    "search_files",
                    {"query": "match_line_", "path": td, "content_search": True},
                    MagicMock(),
                    MagicMock(),
                )
            )
            self.assertEqual(len(res["matches"]), 50)

    def test_auto_trust_approval_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = Storage(Path(tmp) / "test.db")
            res = {}
            for _ in range(3):
                res = st.record_notification_feedback("approval:run_command", "yes")
            self.assertEqual(res["promoted"], "auto_trust_approval")
            self.assertTrue(st.is_approval_auto_trusted("approval:run_command"))
            self.assertIn("approval:run_command", st.list_auto_trusted_approvals())


class TestAgentEvolveServerRegressions(unittest.TestCase):
    def test_reset_session_cancels_pending_approval_future(self):
        agent = AgentManager(Settings(), MagicMock(), MagicMock())
        loop = asyncio.new_event_loop()
        try:
            fut = loop.create_future()
            agent.pending_approvals["ap1"] = PendingApproval(
                approval_id="ap1",
                session_id="s1",
                tool_name="run_command",
                arguments={"cmd": "ls"},
                reason="test",
                risk_level="medium",
                future=fut,
            )
            agent.reset_session("s1")
            self.assertNotIn("ap1", agent.pending_approvals)
            self.assertTrue(fut.done() and fut.result() is False)
        finally:
            loop.close()

    def test_create_safe_recovers_from_stale_interaction_chain(self):
        class DummyInteractions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if kwargs.get("previous_interaction_id"):
                    raise RuntimeError("400 invalid_request: invalid previous_interaction_id")
                if isinstance(kwargs.get("input"), list):
                    raise RuntimeError("400 invalid_request: function_result requires previous_interaction_id")
                return SimpleNamespace(id="new-id", output_text="ok", steps=[])

        client = SimpleNamespace(interactions=DummyInteractions())
        res = asyncio.run(
            AgentManager._create_safe(
                client,
                {
                    "model": "gemini-2.5-flash",
                    "previous_interaction_id": "stale-chain-id",
                    "input": [{"type": "function_result", "call_id": "c1", "name": "system_status", "result": '{"cpu": 10}'}],
                },
            )
        )
        self.assertEqual(res.id, "new-id")
        self.assertEqual(len(client.interactions.calls), 2)
        self.assertNotIn("previous_interaction_id", client.interactions.calls[1])
        self.assertIsInstance(client.interactions.calls[1]["input"], str)
        self.assertIn("system_status", client.interactions.calls[1]["input"])

    def test_evolver_restores_codebase_hash_on_transient_error(self):
        settings = Settings(evolve_enabled=True)
        watcher = MagicMock()
        watcher.current_context = "normal"
        storage = MagicMock()
        storage.list_memories.return_value = []
        storage.list_evolve_proposals.return_value = []
        agent = MagicMock()
        agent.auth.is_configured = True

        evolver = SelfEvolver(settings, watcher, storage, agent)
        evolver._last_codebase_hash = "prev_hash"

        async def fake_locked(*args, **kwargs):
            evolver._last_codebase_hash = "new_hash"
            raise RuntimeError("transient quota error")

        evolver._evolve_once_locked = fake_locked

        async def _run():
            with self.assertRaises(RuntimeError):
                await evolver.evolve_once(force=False)
            self.assertEqual(evolver._last_codebase_hash, "prev_hash")

        asyncio.run(_run())

    def test_api_terminal_run_kills_timed_out_process(self):
        async def _run():
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_proc.kill = MagicMock()
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_shell", AsyncMock(return_value=mock_proc)):
                res = await api_terminal_run(TerminalCommand(command="sleep 60"))

            self.assertEqual(res["returncode"], -1)
            self.assertIn("timed out", res["stderr"])
            mock_proc.kill.assert_called_once()
            mock_proc.wait.assert_awaited_once()

        asyncio.run(_run())

    def test_local_mind_cold_start_predicts_system_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orig_ds, orig_m, orig_w, orig_l = lm.DATASET_DIR, lm.MODELS_DIR, lm.WEIGHTS_FILE, lm.LORA_TRAIN_FILE
            try:
                lm.DATASET_DIR = tmp_path / "dataset"
                lm.MODELS_DIR = tmp_path / "models"
                lm.WEIGHTS_FILE = lm.MODELS_DIR / "local_mind_weights.json"
                lm.LORA_TRAIN_FILE = lm.DATASET_DIR / "lora_chatml.txt"
                mind = lm.LocalMind()
                self.assertIsInstance(mind.weights, dict)
                self.assertIn("tool_centroids", mind.weights)
                tool, _, score = mind.predict_tool("check system status")
                self.assertEqual(tool, "system_status")
                self.assertGreater(score, 0)
            finally:
                lm.DATASET_DIR, lm.MODELS_DIR, lm.WEIGHTS_FILE, lm.LORA_TRAIN_FILE = orig_ds, orig_m, orig_w, orig_l

    def test_brain_think_once_uses_nonblocking_dialog_and_ns_timestamp(self):
        src = inspect.getsource(brain.CognitiveBrain.think_once)
        self.assertIn("time.time_ns()", src)
        self.assertIn("asyncio.to_thread(_prompt_dialog)", src)


if __name__ == "__main__":
    unittest.main()

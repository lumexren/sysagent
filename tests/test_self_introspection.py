"""Tests for model-driven native OpenAI tool calling, neural role routing, and self-introspection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sysagent.local_mind as lm
from sysagent.local_mind_client import LocalMindClient
from sysagent.storage import Storage
from sysagent.tool_schemas import TOOL_DECLARATIONS
from sysagent.tools import execute_tool


class SelfIntrospectionTests(unittest.TestCase):
    def test_select_tools_for_local_includes_core_self_and_switch(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "si0.db") as st:
            tmp = Path(td)
            with (
                patch.object(lm, "MODELS_DIR", tmp),
                patch.object(lm, "DATASET_DIR", tmp),
                patch.object(lm, "WEIGHTS_FILE", tmp / "weights.json"),
                patch.object(lm, "DISTILLED_FILE", tmp / "distilled.jsonl"),
                patch.dict(lm._WEIGHTS_CACHE, {}, clear=True),
            ):
                client = LocalMindClient(st)
                selected = client.interactions._select_tools_for_local("Brain, Cortex vs ne durumda", TOOL_DECLARATIONS, limit=12)
                names = {t["name"] for t in selected}
                self.assertLessEqual(len(selected), 12)
                self.assertIn("inspect_self_component", names)
                self.assertIn("switch_model", names)
                self.assertIn("system_status", names)

    def test_run_gguf_chat_with_tools_sse_single_multi_and_text(self) -> None:
        class _FakeStreamResp:
            status_code = 200

            def __init__(self, lines: list[str]) -> None:
                self._lines = lines

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def iter_lines(self):
                yield from self._lines

        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "si1.db") as st:
            tmp = Path(td)
            fake_gguf = tmp / "DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q4_K_M.gguf"
            fake_gguf.write_bytes(b"gguf")
            with (
                patch.object(lm, "MODELS_DIR", tmp),
                patch.object(lm, "DATASET_DIR", tmp),
                patch.object(lm, "WEIGHTS_FILE", tmp / "weights.json"),
                patch.object(lm, "DISTILLED_FILE", tmp / "distilled.jsonl"),
                patch.dict(lm._WEIGHTS_CACHE, {}, clear=True),
                patch("sysagent.local_mind.find_all_gguf_models", return_value=[fake_gguf]),
                patch("sysagent.local_mind.find_gguf_model", return_value=fake_gguf),
            ):
                mind = lm.LocalMind(st)
                with patch.object(mind, "_ensure_llama_server", return_value="http://127.0.0.1:8769"):
                    # 1. Single tool call via split SSE chunks
                    resp_single = _FakeStreamResp(
                        [
                            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"inspect_self_component","arguments":"{\\"component\\":\\""}}]}}]}',
                            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"brain\\"}"}}]}}]}',
                            "data: [DONE]",
                        ]
                    )
                    deltas: list[str] = []
                    with patch("httpx.stream", return_value=resp_single):
                        out1 = mind.run_gguf_chat_with_tools("şuan ne düşünüyorsun?", TOOL_DECLARATIONS, on_delta=deltas.append)
                    self.assertIsNotNone(out1)
                    assert out1 is not None
                    self.assertEqual(deltas, [])
                    self.assertEqual(len(out1["tool_calls"]), 1)
                    self.assertEqual(out1["tool_calls"][0]["name"], "inspect_self_component")
                    self.assertEqual(out1["tool_calls"][0]["arguments"], {"component": "brain"})

                    # 2. Parallel multi-tool calls natively
                    resp_multi = _FakeStreamResp(
                        [
                            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"inspect_self_component","arguments":"{\\"component\\":\\"brain\\"}"}}]}}]}',
                            'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"inspect_self_component","arguments":"{\\"component\\":\\"cortex\\"}"}}]}}]}',
                            "data: [DONE]",
                        ]
                    )
                    with patch("httpx.stream", return_value=resp_multi):
                        out2 = mind.run_gguf_chat_with_tools("Brain, Cortex vs ne durumda", TOOL_DECLARATIONS, on_delta=lambda _c: None)
                    self.assertIsNotNone(out2)
                    assert out2 is not None
                    self.assertEqual(len(out2["tool_calls"]), 2)
                    self.assertEqual(out2["tool_calls"][1]["arguments"], {"component": "cortex"})

                    # 3. Direct chitchat reply with <think> stripped
                    resp_chat = _FakeStreamResp(
                        [
                            'data: {"choices":[{"delta":{"content":"<think>check</think>"}}]}',
                            'data: {"choices":[{"delta":{"content":"Merhaba! "}}]}',
                            'data: {"choices":[{"delta":{"content":"İyiyim, yardımcı olabilirim."}}]}',
                            "data: [DONE]",
                        ]
                    )
                    chat_deltas: list[str] = []
                    with patch("httpx.stream", return_value=resp_chat):
                        out3 = mind.run_gguf_chat_with_tools("nasılsınn", TOOL_DECLARATIONS, on_delta=chat_deltas.append)
                    self.assertIsNotNone(out3)
                    assert out3 is not None
                    self.assertEqual(out3["tool_calls"], [])
                    self.assertEqual(out3["text"], "Merhaba! İyiyim, yardımcı olabilirim.")
                    self.assertEqual("".join(chat_deltas), "Merhaba! İyiyim, yardımcı olabilirim.")

    def test_local_mind_client_emits_parallel_native_steps_and_plain_text(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "si2.db") as st:
            tmp = Path(td)
            with (
                patch.object(lm, "MODELS_DIR", tmp),
                patch.object(lm, "DATASET_DIR", tmp),
                patch.object(lm, "WEIGHTS_FILE", tmp / "weights.json"),
                patch.object(lm, "DISTILLED_FILE", tmp / "distilled.jsonl"),
                patch.dict(lm._WEIGHTS_CACHE, {}, clear=True),
            ):
                client = LocalMindClient(st)
                native_multi = {
                    "text": "",
                    "tool_calls": [
                        {"id": "c1", "name": "inspect_self_component", "arguments": {"component": "brain"}},
                        {"id": "c2", "name": "inspect_self_component", "arguments": {"component": "cortex"}},
                    ],
                }
                with patch.object(client.interactions.mind, "run_gguf_chat_with_tools", return_value=native_multi):
                    res = client.interactions.create(input="Brain, Cortex vs ne durumda", tools=TOOL_DECLARATIONS)
                self.assertEqual(len(res.steps), 2)
                self.assertEqual(res.steps[0].name, "inspect_self_component")
                self.assertEqual(res.steps[0].arguments, {"component": "brain"})
                self.assertEqual(res.steps[1].arguments, {"component": "cortex"})

                with patch.object(
                    client.interactions.mind,
                    "run_gguf_chat_with_tools",
                    return_value={"text": "İyiyim, nasıl yardımcı olabilirim?", "tool_calls": []},
                ):
                    res_chat = client.interactions.create(input="nasılsınn", tools=TOOL_DECLARATIONS)
                self.assertEqual(res_chat.steps, [])
                self.assertIn("İyiyim", res_chat.output_text)

    def test_classify_task_role_prefers_neural_engine(self) -> None:
        fake_eng = SimpleNamespace(
            meta={"epochs_trained": 12},
            predict_role_probs=lambda _txt: {"coding": 0.05, "general": 0.78, "chat": 0.12, "reasoning": 0.05},
        )
        with patch("sysagent.neural.NeuralEngine", return_value=fake_eng):
            role = lm.classify_task_role("Brain Cortex ne durumda")
        self.assertEqual(role, "general")

    def test_inspect_self_component_all_cortex_guard_and_runtime_refs(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "si3.db") as st:
            st.add_brain_thought("ignore", 91, "Routine background check passed.")
            fake_cortex = SimpleNamespace(
                get_status=lambda: {
                    "flow": {"mode": "deep_focus"},
                    "predictive_healer": {"last_status": "healthy"},
                    "last_dream": None,
                    "habits": [],
                }
            )
            fake_guard = SimpleNamespace(
                last_autopsy=None,
                last_canary={"ok": True},
                last_eval={"score": 1.0, "passed": 16, "total": 16},
                last_drift={"ok": True, "stale_modules": []},
                last_profile={"rss_after_mb": 120.0},
            )
            fake_watcher = SimpleNamespace(
                get_health_snapshot=lambda: {"cpu_percent": 12.0, "ram_percent": 45.0},
                runtime_refs={"cortex": fake_cortex, "self_guard": fake_guard},
            )

            all_out = json.loads(execute_tool("inspect_self_component", {"component": "all"}, fake_watcher, st))
            for key in ("brain", "cortex", "neural", "evolve", "self", "local_mind", "guard", "overview"):
                self.assertIn(key, all_out)
            self.assertEqual(all_out["cortex"]["flow"]["mode"], "deep_focus")

            cortex_out = json.loads(execute_tool("inspect_self_component", {"component": "cortex"}, fake_watcher, st))
            self.assertEqual(cortex_out["cortex"]["flow"]["mode"], "deep_focus")

            guard_out = json.loads(execute_tool("inspect_self_component", {"component": "guard"}, None, st))
            self.assertIn("guard", guard_out)
            self.assertIn("quota", guard_out["guard"])


if __name__ == "__main__":
    unittest.main()

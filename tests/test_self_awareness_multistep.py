"""Tests for runtime self-awareness contract, complete self-inspection, and LocalMind multi-step tool calling."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sysagent import local_inference as li_mod
from sysagent.agent import (
    AgentManager,
    _looks_like_unexecuted_tool_stall,
    build_system_instruction,
)
from sysagent.config import Settings
from sysagent.local_mind import LocalMind
from sysagent.local_mind_client import LocalMindClient
from sysagent.plugins import PluginRegistry
from sysagent.storage import Storage
from sysagent.tool_schemas import TOOL_DECLARATIONS
from sysagent.tools import execute_tool


class SelfAwarenessAndMultiStepTests(unittest.TestCase):
    """Verify end-to-end self-awareness, prompts inspection, and multi-step LocalMind tool loops."""

    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        root = Path(self._td.name)
        ds = root / "dataset"
        mods = root / "models"
        ds.mkdir(parents=True, exist_ok=True)
        mods.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.mods = mods

        for p in (
            patch("sysagent.local_mind.DATASET_DIR", ds),
            patch("sysagent.local_mind.MODELS_DIR", mods),
            patch("sysagent.local_mind.DISTILLED_FILE", ds / "distilled_turns.jsonl"),
            patch("sysagent.local_mind.UNCERTAIN_FILE", ds / "uncertain_queries.jsonl"),
            patch("sysagent.local_mind.WEIGHTS_FILE", mods / "local_mind_weights.json"),
            patch("sysagent.local_mind.ROLE_ADAPTERS_FILE", mods / "role_adapters.json"),
            patch("sysagent.local_mind.LORA_TRAIN_FILE", ds / "lora_chatml.txt"),
            patch("sysagent.neural.NEURAL_NPZ_FILE", mods / "neural_engine.npz"),
            patch("sysagent.neural.NEURAL_META_FILE", mods / "neural_meta.json"),
        ):
            p.start()
            self.addCleanup(p.stop)

        self.addCleanup(self._reset_inference_state)
        self._reset_inference_state()

    @staticmethod
    def _reset_inference_state() -> None:
        li_mod._LLAMA_SERVER_PROC = None
        li_mod._LLAMA_SERVER_MODEL = ""
        li_mod._LLAMA_SERVER_MMPROJ = ""
        li_mod._LLAMA_SERVER_URL = ""
        li_mod._LLAMA_SERVER_LAST_USED = 0.0
        li_mod._LLAMA_CPP_INSTANCE = None
        li_mod._LLAMA_CPP_MODEL_PATH = None

    def test_multistep_tool_chain_nudge_and_duplicate_guard(self) -> None:
        with Storage(db_path=self.root / "ms.db") as st:
            cli = LocalMindClient(st)
            seen_calls: list[dict[str, object]] = []

            def _fake_chat_tools(prompt, tools, **kw):
                seen_calls.append(
                    {
                        "prompt": prompt,
                        "tools": tools,
                        "extra_messages": kw.get("extra_messages"),
                    }
                )
                idx = len(seen_calls)
                if idx == 1:
                    return {
                        "text": "",
                        "tool_calls": [
                            {"id": "call_1", "name": "list_directory", "arguments": {"path": "/tmp"}}
                        ],
                    }
                if idx == 2:
                    return {
                        "text": "",
                        "tool_calls": [
                            {"id": "call_2", "name": "read_file", "arguments": {"path": "/tmp/a.py"}}
                        ],
                    }
                return {"text": "Finished reading /tmp/a.py.", "tool_calls": []}

            with patch.object(cli.interactions.mind, "run_gguf_chat_with_tools", side_effect=_fake_chat_tools):
                step1 = cli.interactions.create(
                    input="[cwd: /home/eren/Projects/plasma-networkd] inspect /tmp/a.py",
                    tools=TOOL_DECLARATIONS,
                )
                self.assertEqual([s.name for s in step1.steps], ["list_directory"])

                step2 = cli.interactions.create(
                    input=[
                        {
                            "type": "function_result",
                            "call_id": step1.steps[0].id,
                            "name": "list_directory",
                            "result": '{"entries": ["a.py"]}',
                        }
                    ],
                    previous_interaction_id=step1.id,
                    tools=TOOL_DECLARATIONS,
                )
                self.assertEqual([s.name for s in step2.steps], ["read_file"])
                extra2 = seen_calls[1]["extra_messages"]
                self.assertIsInstance(extra2, list)
                assert isinstance(extra2, list)
                self.assertTrue(any(m.get("role") == "assistant" and m.get("tool_calls") for m in extra2))
                self.assertTrue(any(m.get("role") == "tool" and m.get("name") == "list_directory" for m in extra2))
                self.assertTrue(seen_calls[1]["tools"])

                step3 = cli.interactions.create(
                    input=[
                        {
                            "type": "function_result",
                            "call_id": step2.steps[0].id,
                            "name": "read_file",
                            "result": '{"content": "ok"}',
                        }
                    ],
                    previous_interaction_id=step2.id,
                    tools=TOOL_DECLARATIONS,
                )
                self.assertEqual(step3.steps, [])
                self.assertIn("Finished reading /tmp/a.py.", step3.output_text)

                # `[System]:` nudge preserves original user question and _select_tools_for_local ignores [cwd: ...]
                with patch.object(
                    cli.interactions.mind,
                    "run_gguf_chat_with_tools",
                    side_effect=[
                        {"text": "```bash\ncat prompts.py\n```", "tool_calls": []},
                        {"text": "", "tool_calls": [{"id": "c_nudge", "name": "inspect_self_component", "arguments": {"component": "prompts"}}]},
                    ],
                ) as mock_nudge_tools:
                    n0 = cli.interactions.create(
                        input="[cwd: /home/eren/Projects/plasma-networkd] promptlar ne durumda",
                        tools=TOOL_DECLARATIONS,
                    )
                    nudge_res = cli.interactions.create(
                        input="[System]: Invoke inspect_self_component right now.",
                        previous_interaction_id=n0.id,
                        tools=TOOL_DECLARATIONS,
                    )
                self.assertEqual([s.name for s in nudge_res.steps], ["inspect_self_component"])
                self.assertIn("promptlar ne durumda", str(mock_nudge_tools.call_args.args[0]))

            # Duplicate call guard exits cleanly to synthesis instead of looping
            self.assertIn('list_directory:{"path": "/tmp"}', cli.interactions._turn_calls[step1.id])
            with (
                patch.object(
                    cli.interactions.mind,
                    "run_gguf_chat_with_tools",
                    return_value={
                        "text": "",
                        "tool_calls": [{"id": "dup_1", "name": "list_directory", "arguments": {"path": "/tmp"}}],
                    },
                ),
                patch.object(cli.interactions.mind, "run_gguf_inference", return_value="Synthesized final answer."),
            ):
                dup_out = cli.interactions.create(
                    input=[
                        {
                            "type": "function_result",
                            "call_id": step1.steps[0].id,
                            "name": "list_directory",
                            "result": '{"entries": ["a.py"]}',
                        }
                    ],
                    previous_interaction_id=step1.id,
                    tools=TOOL_DECLARATIONS,
                )
            self.assertEqual(dup_out.steps, [])
            self.assertEqual(dup_out.output_text, "Synthesized final answer.")

            selected = cli.interactions._select_tools_for_local(
                "[cwd: /home/eren/Projects/plasma-networkd] promptlar ne durumda",
                TOOL_DECLARATIONS,
                limit=14,
            )
            sel_names = {t["name"] for t in selected}
            self.assertTrue({"inspect_self_component", "list_directory", "read_file", "edit_file"} <= sel_names)

    def test_role_tool_http_400_retries_with_flattened_messages_and_tools(self) -> None:
        m = self.mods / "DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q4_K_M.gguf"
        m.write_bytes(b"GGUF\x00\x00")
        posted_payloads: list[dict[str, object]] = []

        def _fake_post(_url, json=None, **_kw):
            posted_payloads.append(dict(json or {}))
            if len(posted_payloads) == 1:
                return SimpleNamespace(status_code=400, json=lambda: {"error": "role tool unsupported"})
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {"id": "c2", "function": {"name": "read_file", "arguments": '{"path":"/tmp/x"}'}}
                                ],
                            }
                        }
                    ]
                },
            )

        with Storage(db_path=self.root / "retry.db") as st:
            lm = LocalMind(st)
            with (
                patch.object(lm, "_ensure_llama_server", return_value="http://127.0.0.1:8769"),
                patch("httpx.post", side_effect=_fake_post),
            ):
                out = lm.run_gguf_chat_with_tools(
                    "inspect prompts",
                    TOOL_DECLARATIONS[:4],
                    extra_messages=[
                        {"role": "user", "content": "inspect prompts"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "list_directory", "arguments": "{}"}}],
                        },
                        {"role": "tool", "tool_call_id": "c1", "name": "list_directory", "content": '{"entries":["a.py"]}'},
                    ],
                )
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(len(posted_payloads), 2)
        self.assertIn("tools", posted_payloads[1])
        self.assertEqual(posted_payloads[1]["tool_choice"], "auto")
        self.assertEqual(out["tool_calls"][0]["name"], "read_file")

    def test_inspect_self_component_prompts_checkpoints_and_codebase(self) -> None:
        with Storage(db_path=self.root / "isp.db") as st:
            st.save_prompt_version("agent_main", "You are {agent_name}. Custom prompt with length > 40 chars.", "tuned formatting")
            st.save_session_checkpoint("s1", "prompt migration", "Updated prompt registry", files=["prompts.py"])
            raw = execute_tool("inspect_self_component", {"component": "all"}, None, st)
            data = json.loads(raw)
            self.assertIn("prompts", data)
            self.assertEqual(data["prompts"]["evolved_count"], 1)
            self.assertEqual(data["prompts"]["total_versions_saved"], 1)
            self.assertEqual(data["checkpoints"]["count"], 1)
            self.assertIn("codebase", data)
            self.assertIn("Prompts evolved=1/", data["overview"])

            only_prompts = json.loads(execute_tool("inspect_self_component", {"component": "promptları"}, None, st))
            self.assertEqual(only_prompts["component"], "prompts")
            self.assertIn("prompts", only_prompts)
            self.assertTrue(any(p.get("template_preview") for p in only_prompts["prompts"]["prompts"]))

    def test_runtime_contract_compact_sys_inst_and_bash_stall_nudge(self) -> None:
        with Storage(db_path=self.root / "sc.db") as st:
            # Stale evolved DB prompt lacking inspect_self_component wording still gets contract
            st.save_prompt_version("agent_main", "You are {agent_name}, a minimal assistant on Linux with custom filler text.", "stale custom")
            sys_inst = build_system_instruction("tr", "Vex", storage=st)
            self.assertIn("Runtime self-awareness contract:", sys_inst)
            self.assertIn('component="prompts"', sys_inst)

            lm = LocalMind(st)
            long_sys = ("Long prefix note. " * 220) + "\n" + sys_inst
            compact = lm._compact_system_instruction(long_sys, max_chars=2000)
            self.assertLessEqual(len(compact), 2010)
            self.assertIn("Runtime self-awareness contract:", compact)
            self.assertIn("Language rule:", compact)

            self.assertTrue(
                _looks_like_unexecuted_tool_stall(
                    "promptlar ne durumda",
                    "Dosyayı okuyayım:\n```bash\ncat /home/eren/Projects/sysagent/sysagent/prompts.py\n```",
                )
            )
            self.assertTrue(_looks_like_unexecuted_tool_stall("terminali temizle", "```bash\nclear\n```"))
            self.assertFalse(_looks_like_unexecuted_tool_stall("örnek bash script yaz", "```bash\ncat foo.txt\n```"))

            # Verify AgentManager.run_turn nudges when model stalls with fake ```bash\ncat ...``` block
            watcher = SimpleNamespace(current_context="normal", _send_desktop_notification=MagicMock())
            am = AgentManager(Settings(), watcher, st, PluginRegistry(), MagicMock(is_configured=True))
            calls: list[dict[str, object]] = []

            async def _fake_failover(_sid, **kw):
                calls.append(dict(kw))
                if len(calls) == 1:
                    return SimpleNamespace(
                        id="m1",
                        steps=[],
                        output_text="Okuyayım:\n```bash\ncat /home/eren/Projects/sysagent/sysagent/prompts.py\n```",
                    )
                if len(calls) == 2:
                    return SimpleNamespace(
                        id="m2",
                        steps=[
                            SimpleNamespace(
                                type="function_call",
                                id="c1",
                                name="inspect_self_component",
                                arguments={"component": "prompts"},
                            )
                        ],
                        output_text="",
                    )
                return SimpleNamespace(id="m3", steps=[], output_text="Prompt v0 aktif ve 7 şablon hazır.")

            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(am, "_call_model_with_failover", side_effect=_fake_failover),
            ):
                out = asyncio.run(am.run_turn("s-nudge", "promptlar ne durumda"))
            self.assertIn("Prompt v0", out)
            self.assertGreaterEqual(len(calls), 3)
            self.assertIn("inspect_self_component", str(calls[1]["input"]))


if __name__ == "__main__":
    unittest.main()

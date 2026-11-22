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
    _has_fenced_shell_block,
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
            patch("sysagent.local_mind.PREFERENCE_FILE", ds / "preference_pairs.jsonl"),
            patch("sysagent.local_mind.WEIGHTS_FILE", mods / "local_mind_weights.json"),
            patch("sysagent.local_mind.ROLE_ADAPTERS_FILE", mods / "role_adapters.json"),
            patch("sysagent.local_mind.SEMANTIC_INDEX_FILE", mods / "semantic_index.json"),
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
        from sysagent import local_mind as lm_mod

        lm_mod._EMBED_CACHE.clear()
        li_mod._LLAMA_SERVER_PROC = None
        li_mod._LLAMA_SERVER_MODEL = ""
        li_mod._LLAMA_SERVER_MMPROJ = ""
        li_mod._LLAMA_SERVER_URL = ""
        li_mod._LLAMA_SERVER_LAST_USED = 0.0
        li_mod._AUX_SERVER_PROC = None
        li_mod._AUX_SERVER_MODEL = ""
        li_mod._AUX_SERVER_URL = ""
        li_mod._AUX_SERVER_LAST_USED = 0.0
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
        inspect_schema = next(d for d in TOOL_DECLARATIONS if d["name"] == "inspect_self_component")
        comp_enum = inspect_schema["parameters"]["properties"]["component"].get("enum")
        self.assertIsInstance(comp_enum, list)
        self.assertIn("prompts", comp_enum)
        self.assertIn("checkpoints", comp_enum)
        self.assertIn("codebase", comp_enum)

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

            for alias in ("prompts", "prompt_registry", "promptları"):
                only_prompts = json.loads(execute_tool("inspect_self_component", {"component": alias}, None, st))
                self.assertEqual(only_prompts["component"], "prompts")
                self.assertIn("prompts", only_prompts)
                self.assertTrue(any(p.get("template_preview") for p in only_prompts["prompts"]["prompts"]))

            fallback_all = json.loads(execute_tool("inspect_self_component", {"component": "nonexistent_xyz"}, None, st))
            self.assertEqual(fallback_all["component"], "all")

    def test_runtime_contract_compact_sys_inst_and_bash_stall_nudge(self) -> None:
        from sysagent.neural import NeuralEngine

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
                _has_fenced_shell_block(
                    "Dosyayı okuyayım:\n```bash\ncat /home/eren/Projects/sysagent/sysagent/prompts.py\n```"
                )
            )
            self.assertTrue(_has_fenced_shell_block("```sh\nclear\n```"))
            self.assertFalse(_has_fenced_shell_block("Merhaba dünya"))
            self.assertFalse(_has_fenced_shell_block("```python\nprint(1)\n```"))

            # 1) Judge returns true -> run_turn nudges, calls inspect_self_component, and updates NeuralEngine stall_samples
            watcher = SimpleNamespace(current_context="normal", _send_desktop_notification=MagicMock())
            am = AgentManager(Settings(), watcher, st, PluginRegistry(), MagicMock(is_configured=True))
            calls: list[dict[str, object]] = []
            nn_file = self.root / "nn_stall.npz"
            nn_meta = self.root / "nn_stall.json"
            real_nn = NeuralEngine(st, npz_file=nn_file, meta_file=nn_meta)
            samples_before = int(real_nn.meta.get("stall_samples", 0))

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

            async def _fake_create_safe_true(_client, _kw):
                return SimpleNamespace(output_text='{"unexecuted_stall": true}')

            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(am, "_call_model_with_failover", side_effect=_fake_failover),
                patch.object(am, "_create_safe", side_effect=_fake_create_safe_true),
                patch("sysagent.neural.NeuralEngine", return_value=real_nn),
            ):
                out = asyncio.run(am.run_turn("s-nudge", "promptlar ne durumda"))
            self.assertIn("Prompt v0", out)
            self.assertGreaterEqual(len(calls), 3)
            self.assertIn("inspect_self_component", str(calls[1]["input"]))
            self.assertGreater(int(real_nn.meta.get("stall_samples", 0)), samples_before)

            # 2) Judge returns false (intentional example) -> no nudge, original reply returned untouched
            calls_false: list[dict[str, object]] = []
            example_reply = "İşte örnek komut:\n```bash\ncat foo.txt\n```"

            async def _fake_failover_example(_sid, **kw):
                calls_false.append(dict(kw))
                return SimpleNamespace(id="mx1", steps=[], output_text=example_reply)

            async def _fake_create_safe_false(_client, _kw):
                return SimpleNamespace(output_text='{"unexecuted_stall": false}')

            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(am, "_call_model_with_failover", side_effect=_fake_failover_example),
                patch.object(am, "_create_safe", side_effect=_fake_create_safe_false),
                patch("sysagent.neural.NeuralEngine", return_value=real_nn),
            ):
                out_example = asyncio.run(am.run_turn("s-example", "örnek bash script yaz"))
            self.assertEqual(out_example, example_reply)
            self.assertEqual(len(calls_false), 1)

            # 3) Reply without fenced shell block -> judge is not called
            calls_plain: list[dict[str, object]] = []

            async def _fake_failover_plain(_sid, **kw):
                calls_plain.append(dict(kw))
                return SimpleNamespace(id="mp1", steps=[], output_text="Her şey normal görünüyor.")

            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(am, "_call_model_with_failover", side_effect=_fake_failover_plain),
                patch.object(am, "_judge_unexecuted_shell_stall") as judge_spy,
                patch("sysagent.neural.NeuralEngine", return_value=real_nn),
            ):
                out_plain = asyncio.run(am.run_turn("s-plain", "nasılsın"))
            self.assertEqual(out_plain, "Her şey normal görünüyor.")
            judge_spy.assert_not_called()

            # 4) local-mind model uses LocalMind.run_gguf_inference(max_tokens=32, prefer_warm=True) without cloud
            with (
                patch.object(am, "_get_client", return_value=LocalMindClient(st)),
                patch.object(am, "_create_safe") as cloud_spy,
                patch.object(
                    LocalMind,
                    "run_gguf_inference",
                    return_value='{"unexecuted_stall": true}',
                ) as gguf_spy,
                patch("sysagent.neural.NeuralEngine", return_value=real_nn),
            ):
                local_verdict = asyncio.run(
                    am._judge_unexecuted_shell_stall(
                        "local-mind",
                        "promptlar ne durumda",
                        "```bash\ncat prompts.py\n```",
                        [],
                    )
                )
            self.assertTrue(local_verdict)
            cloud_spy.assert_not_called()
            gguf_spy.assert_called_once()
            _, gguf_kw = gguf_spy.call_args
            self.assertEqual(gguf_kw.get("max_tokens"), 32)
            self.assertTrue(gguf_kw.get("prefer_warm"))
            self.assertTrue(gguf_kw.get("extra_system"))

    def test_curriculum_validation_and_multi_teacher_distillation(self) -> None:
        from sysagent.neural import NeuralEngine

        with Storage(db_path=self.root / "distill.db") as st:
            st.add_lesson("_internal:hidden", "secret", "ignore")
            st.add_lesson("User correction (list_directory)", "Attempted on: promptlar ne durumda", "Use inspect_self_component")
            lm = LocalMind(st)

            # 1) underrepresented_tools & build_distillation_curriculum
            weak_tools = lm.underrepresented_tools(limit=8)
            self.assertEqual(len(weak_tools), 8)
            curr = lm.build_distillation_curriculum(tool_limit=8)
            self.assertEqual(len(curr["recent_lessons"]), 1)
            self.assertIn("inspect_self_component", [s["tool"] for s in curr["tool_signatures"]])
            insp_sig = next(s for s in curr["tool_signatures"] if s["tool"] == "inspect_self_component")
            self.assertIn("prompts", insp_sig["params"]["component"]["enum"])

            # 2) _validate_distilled_sample schema normalization, arg remapping, enum fuzzy fix, and dedup
            schema_map = lm._build_tool_schema_map()
            seen_q: set[str] = set()
            v_single = lm._validate_distilled_sample(
                {
                    "user": "prompt şablonları ne durumda kontrol et",
                    "tool": "inspect_self_component",
                    "args": {"component": "promptları", "bogus_key": 123},
                    "reply": "Prompt şablonları incelendi.",
                    "role": "reasoning",
                },
                schema_map,
                seen_q,
            )
            self.assertIsNotNone(v_single)
            assert v_single is not None
            self.assertEqual(v_single["tool_calls"][0]["arguments"], {"component": "prompts"})
            self.assertEqual(v_single["role"], "reasoning")

            # Duplicate query rejected
            self.assertIsNone(
                lm._validate_distilled_sample(
                    {
                        "user": "Prompt  şablonları ne durumda kontrol et",
                        "tool": "inspect_self_component",
                        "args": {"component": "prompts"},
                        "reply": "Tekrar.",
                    },
                    schema_map,
                    seen_q,
                )
            )

            # Single required arg key remap ('filepath' -> 'path' for read_file) & multi-step chain
            v_chain = lm._validate_distilled_sample(
                {
                    "user": "etc altındaki hosts dosyasını listele ve oku",
                    "tool_calls": [
                        {"name": "list_directory", "arguments": {"folder": "/etc"}},
                        {"name": "read_file", "arguments": {"filepath": "/etc/hosts"}},
                    ],
                    "reply": "/etc/hosts okundu.",
                },
                schema_map,
                seen_q,
            )
            self.assertIsNotNone(v_chain)
            assert v_chain is not None
            self.assertEqual(len(v_chain["tool_calls"]), 2)
            self.assertEqual(v_chain["tool_calls"][0]["arguments"], {"path": "/etc"})
            self.assertEqual(v_chain["tool_calls"][1]["arguments"], {"path": "/etc/hosts"})

            # Unknown tool rejected when tool was expected
            self.assertIsNone(
                lm._validate_distilled_sample(
                    {"user": "do something invalid now", "tool": "nonexistent_tool_xyz", "args": {}, "reply": "ok done"},
                    schema_map,
                    seen_q,
                )
            )

            # 3) distill_from_teachers harvests single + multi-step samples, updates workflows & safe_plans, prunes uncertain
            unc_file = self.root / "dataset" / "uncertain_queries.jsonl"
            unc_file.write_text(
                json.dumps({"query": "sistem sağlık durumu ve süreçler", "score": 0.12}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            teacher_payload = json.dumps(
                {
                    "samples": [
                        {
                            "user": "sistem sağlık durumu ve süreçler",
                            "role": "general",
                            "tool_calls": [
                                {"name": "system_status", "arguments": {}},
                                {"name": "list_processes", "arguments": {"sort_by": "memory", "limit": 10}},
                            ],
                            "reply": "Sistem durumu ve en çok RAM kullanan süreçler listelendi.",
                        },
                        {
                            "user": "kendi prompt versiyonlarını detaylı incele",
                            "role": "reasoning",
                            "tool": "inspect_self_component",
                            "args": {"component": "prompts"},
                            "reply": "PromptRegistry aktif versiyonları incelendi.",
                        },
                    ]
                },
                ensure_ascii=False,
            )
            with patch("sysagent.auth._find_agy_cli", return_value="/usr/bin/agy"), patch(
                "sysagent.local_mind.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout=teacher_payload, stderr=""),
            ):
                res = lm.distill_from_teachers(count_per_teacher=2, allow_local_fallback=False)

            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["harvested_samples"], 2)
            self.assertEqual(res["multi_step_chains"], 1)
            self.assertEqual(res["resolved_uncertain_count"], 1)
            self.assertFalse(unc_file.exists())
            self.assertGreaterEqual(res["post_verify_accuracy"], 0.5)
            wfs = st.list_workflows(limit=10)
            self.assertTrue(any("system_status -> list_processes" in str(w.get("signature")) for w in wfs))
            matched_sp = NeuralEngine(st).match_safe_plan("kendi prompt versiyonlarını detaylı incele", min_sim=0.85)
            self.assertIsNotNone(matched_sp)
            assert matched_sp is not None
            self.assertEqual(matched_sp["tool"], "inspect_self_component")

            # 4) Fallback to local peer GGUF when agy CLI and cloud SDK are unavailable
            fake_gguf = self.mods / "OmniCoder-9B.Q4_K_M.gguf"
            fake_gguf.write_bytes(b"GGUF" + b"\x00" * 64)
            local_peer_payload = json.dumps(
                {
                    "samples": [
                        {
                            "user": "pano içeriğini oku clipboard",
                            "role": "general",
                            "tool": "read_clipboard",
                            "args": {},
                            "reply": "Pano okundu.",
                        }
                    ]
                },
                ensure_ascii=False,
            )
            fake_auth = MagicMock(is_configured=False)
            with (
                patch("sysagent.auth._find_agy_cli", return_value=None),
                patch("sysagent.auth.AuthManager", return_value=fake_auth),
                patch.object(lm, "run_gguf_inference", return_value=local_peer_payload) as peer_spy,
            ):
                res_peer = lm.distill_from_teachers(count_per_teacher=1, allow_local_fallback=True)
            self.assertEqual(res_peer["status"], "ok")
            self.assertEqual(res_peer["harvested_samples"], 1)
            self.assertEqual(res_peer["teachers_used"], ["local-peer"])
            peer_spy.assert_called_once()

    def test_all_five_self_learning_and_resilience_upgrades(self) -> None:
        from sysagent import local_mind as lm_mod
        from sysagent.prompts import DEFAULT_PROMPTS, PromptRegistry
        from sysagent.self_learn import run_local_self_play

        with Storage(db_path=self.root / "five.db") as st:
            lm = LocalMind(st)
            fake_coder = self.mods / "OmniCoder-9B.Q4_K_M.gguf"
            fake_coder.write_bytes(b"GGUF" + b"\x00" * 64)

            # 1) Feature 1: Execution-Verified Local Self-Play (rewards working call, penalizes failing call)
            sp_outputs = [
                json.dumps(
                    {
                        "user": "kendi prompt durumunu kontrol et",
                        "role": "reasoning",
                        "tool": "inspect_self_component",
                        "args": {"component": "prompts"},
                        "reply": "Prompt durumları kontrol edildi.",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "user": "olmayan gizli dosyayı oku",
                        "role": "general",
                        "tool": "read_file",
                        "args": {"path": str(self.root / "does_not_exist_12345.txt")},
                        "reply": "Dosya okundu.",
                    },
                    ensure_ascii=False,
                ),
            ]
            with patch.object(LocalMind, "run_gguf_inference", side_effect=sp_outputs):
                sp_res = run_local_self_play(st, max_rounds=2)
            self.assertEqual(sp_res["status"], "ok")
            self.assertEqual(sp_res["verified"], 1)
            self.assertEqual(sp_res["generated"], 1)
            self.assertEqual(sp_res["penalized"], 1)

            # 2) Feature 2: Persistent Semantic Index & Hybrid RAG (dense vector + lesson retrieval)
            lm = LocalMind(st)
            lm.record_and_learn(
                "gpu vram memory usage status",
                [{"name": "inspect_self_component", "arguments": {"component": "local_mind"}}],
                "VRAM usage is 4.3 GB / 12 GB.",
                role="general",
            )
            lm.train()
            st.add_lesson("vram_troubleshooting", "ekran kartı bellek taşması hatası", "unload_idle_servers ile VRAM boşalt.")
            fake_emb = self.mods / "llama-nemotron-embed-1b-v2.Q4_K_M.gguf"
            fake_emb.write_bytes(b"GGUF" + b"\x00" * 64)
            fake_emb_json = json.dumps({"data": [{"embedding": [0.6, 0.8, 0.0, 0.0]}]})
            with (
                patch("sysagent.local_mind.shutil.which", return_value="/usr/bin/llama-embedding"),
                patch(
                    "sysagent.local_mind.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout=fake_emb_json, stderr=""),
                ),
            ):
                sync_res = lm.sync_semantic_index(max_new=4)
                q_vec = lm.embed_text("ekran kartı vram durumu nedir")
            self.assertGreaterEqual(sync_res["cached_vectors"], 1)
            self.assertIsNotNone(q_vec)
            self.assertTrue((self.mods / "semantic_index.json").is_file())
            # Reload from disk after clearing in-memory cache
            lm_mod._EMBED_CACHE.clear()
            rag_hits = lm.retrieve_similar_examples("ekran kartı vram bellek taşması", top_k=3)
            self.assertGreaterEqual(len(rag_hits), 1)

            # 3) Feature 3: Tool-Error Self-Healing Loop in LocalMind & AgentManager.run_turn
            lm.weights.setdefault("learned_arg_templates", {})["read_file"] = [
                {"tokens": ["oku"], "args": {"path": "/nonexistent_bad_template_path"}}
            ]
            lm_mod._save_weights_file(lm.weights)

            valid_target = self.root / "real_note.txt"
            valid_target.write_text("self-healed content", encoding="utf-8")
            watcher = SimpleNamespace(current_context="normal", _send_desktop_notification=MagicMock())
            am = AgentManager(Settings(), watcher, st, PluginRegistry(), MagicMock(is_configured=True))
            turn_steps = [
                SimpleNamespace(
                    id="h1",
                    steps=[
                        SimpleNamespace(
                            type="function_call",
                            id="c_bad",
                            name="read_file",
                            arguments={"path": "/nonexistent_bad_template_path"},
                        )
                    ],
                    output_text="",
                ),
                SimpleNamespace(
                    id="h2",
                    steps=[
                        SimpleNamespace(
                            type="function_call",
                            id="c_good",
                            name="read_file",
                            arguments={"path": str(valid_target)},
                        )
                    ],
                    output_text="",
                ),
                SimpleNamespace(id="h3", steps=[], output_text="Not dosyası başarıyla okundu."),
            ]
            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(am, "_call_model_with_failover", side_effect=turn_steps),
            ):
                heal_out = asyncio.run(am.run_turn("s-heal", "real_note dosyasını oku"))
            self.assertIn("başarıyla okundu", heal_out)
            lm_after = LocalMind(st)
            rf_templates = lm_after.weights.get("learned_arg_templates", {}).get("read_file", [])
            self.assertFalse(any(t.get("args", {}).get("path") == "/nonexistent_bad_template_path" for t in rf_templates))
            self.assertTrue(any(t.get("args", {}).get("path") == str(valid_target) for t in rf_templates))
            lessons = st.list_lessons(limit=10)
            self.assertTrue(any(str(l.get("context", "")).startswith("arg_repair:read_file->read_file") for l in lessons))

            # 4) Feature 4: Step-Level Specialist Handoff in LocalMindClient
            cli = LocalMindClient(st)
            inter = cli.interactions
            self.assertEqual(
                inter._infer_step_specialist_role(
                    "fix bug", [{"name": "edit_file", "result": '{"status": "ok"}'}], [], for_synthesis=False
                ),
                "coding",
            )
            self.assertEqual(
                inter._infer_step_specialist_role(
                    "check self", [{"name": "inspect_self_component", "result": '{"component": "brain"}'}], [], for_synthesis=True
                ),
                "reasoning",
            )
            self.assertEqual(
                inter._infer_step_specialist_role(
                    "read file", [{"name": "read_file", "result": '{"error": "File not found"}'}], [], for_synthesis=False
                ),
                "reasoning",
            )
            self.assertEqual(
                inter._infer_step_specialist_role(
                    "look", [{"name": "take_screenshot", "result": "{}"}], [{"mime_type": "image/png", "data": "abc"}], for_synthesis=False
                ),
                "multimodal",
            )

            # 5) Feature 5: Prompt Evolution A/B Shadow Eval Gate
            reg = PromptRegistry(st)
            st.set_memory("style:compact_units", "Always use compact numeric symbols like %74, 53°C, GB")
            base_main = DEFAULT_PROMPTS["agent_main"]["template"]

            # Identical template rejected
            g_same = reg.evaluate_prompt_candidate("agent_main", base_main)
            self.assertFalse(g_same["passed"])
            self.assertEqual(g_same["reason"], "identical_to_active")

            # Excessively shrunk template (> 40 chars so syntax valid, but < 55% of active length) rejected
            g_shrunk = reg.update_prompt(
                "agent_main",
                "You are {agent_name}, a minimal Linux assistant that answers briefly.",
                reason="shrunk",
                enforce_shadow_gate=True,
            )
            self.assertEqual(g_shrunk["status"], "rejected_by_gate")
            self.assertEqual(g_shrunk["reason"], "excessive_shrinkage")

            # Candidate incorporating user style feedback passes A/B gate
            improved_main = (
                base_main
                + "\n- Formatting preference: Always use compact numeric symbols like %74, 53°C, GB in telemetry."
            )
            g_good = reg.update_prompt(
                "agent_main", improved_main, reason="incorporate compact_units style", enforce_shadow_gate=True
            )
            self.assertEqual(g_good["status"], "updated")

    def test_five_next_gen_architectural_features(self) -> None:
        from sysagent.evolve_ast import analyze_symbol_blast_radius
        from sysagent.local_inference import can_fit_dual_slot, get_runtime_memory_status
        from sysagent.self_learn import analyze_causal_digital_twin

        with Storage(db_path=self.root / "nextgen.db") as st:
            lm = LocalMind(st)
            lm.train()

            # 1) Dual-Slot VRAM Orchestrator + Local Actor-Critic
            prim_gguf = self.mods / "DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q4_K_M.gguf"
            aux_gguf = self.mods / "SkillGym-Qwen3.5-4B.Q4_K_M.gguf"
            prim_gguf.write_bytes(b"GGUF" + b"\x00" * 1024)
            aux_gguf.write_bytes(b"GGUF" + b"\x00" * 512)
            with patch(
                "sysagent.local_inference.get_vram_info",
                return_value={"available": True, "free_mb": 6200, "used_mb": 6088, "total_mb": 12288, "source": "nvidia"},
            ):
                self.assertTrue(can_fit_dual_slot(prim_gguf, aux_gguf))
                rt_status = get_runtime_memory_status()
                self.assertIn("aux_warm_model", rt_status)
                self.assertIn("dual_slot_active", rt_status)

            # Actor-Critic catches missing required param and invalid enum
            c_missing = lm.local_actor_critic_review("read hosts", "read_file", {})
            self.assertFalse(c_missing["approved"])
            self.assertIn("Missing required", c_missing["critique"])

            c_bad_enum = lm.local_actor_critic_review("check self", "inspect_self_component", {"component": "invalid_xyz"})
            self.assertFalse(c_bad_enum["approved"])
            self.assertIn("Invalid enum", c_bad_enum["critique"])

            c_ok = lm.local_actor_critic_review("check prompts", "inspect_self_component", {"component": "prompts"})
            self.assertTrue(c_ok["approved"])

            # 2) Live AST Codebase Call-Graph & Blast-Radius Engine
            blast = analyze_symbol_blast_radius("execute_tool", storage=st)
            self.assertEqual(blast["target"], "execute_tool")
            self.assertTrue(any(d.get("file") == "tools.py" for d in blast["defined_in"]))
            self.assertIn("agent.py", blast["affected_modules"])
            self.assertGreaterEqual(len(blast["covering_tests"]), 1)

            isp_cb = json.loads(
                execute_tool("inspect_self_component", {"component": "codebase", "query": "execute_tool"}, None, st)
            )
            self.assertIn("blast_radius", isp_cb["codebase"])
            self.assertEqual(isp_cb["codebase"]["blast_radius"]["target"], "execute_tool")

            # 3) Transactional Tree-of-Actions & Auto-Rollback on broken edit
            target_py = self.root / "tx_demo.py"
            original_code = "def compute_val() -> int:\n    return 42\n"
            target_py.write_text(original_code, encoding="utf-8")
            batch_id = "tx-test-batch-1"
            # Perform broken edit via execute_tool so audit_log backup is recorded
            execute_tool(
                "edit_file",
                {"path": str(target_py), "target": "return 42", "replacement": "return (42"},
                None,
                st,
                batch_id=batch_id,
            )
            self.assertIn("return (42", target_py.read_text(encoding="utf-8"))

            watcher = SimpleNamespace(current_context="normal", _send_desktop_notification=MagicMock())
            am = AgentManager(Settings(), watcher, st, PluginRegistry(), MagicMock(is_configured=True))
            recorded_calls = [{"name": "edit_file", "arguments": {"path": str(target_py)}}]
            tx_res = am.verify_and_transactional_rollback(recorded_calls, batch_id, user_message="edit tx_demo.py")
            self.assertTrue(tx_res["rolled_back"])
            self.assertTrue(tx_res["restored"])
            # File automatically rolled back to clean original code
            self.assertEqual(target_py.read_text(encoding="utf-8"), original_code)
            self.assertTrue(recorded_calls[0].get("rolled_back"))

            # 4) Online DPO / Contrastive Preference Memory
            dpo_out = lm.record_preference_pair(
                "promptlar ne durumda",
                chosen_tool="inspect_self_component",
                chosen_args={"component": "prompts"},
                rejected_tool="list_directory",
                rejected_args={"path": "/home/eren/Projects/sysagent"},
                reason="Use inspect_self_component for prompt inspection instead of list_directory",
            )
            self.assertTrue(dpo_out["recorded"])
            self.assertTrue(dpo_out["neural_dpo"].get("updated"))
            self.assertGreater(
                dpo_out["neural_dpo"]["margin_after"],
                dpo_out["neural_dpo"]["margin_before"],
            )
            c_hints = lm.retrieve_contrastive_hints("promptlar ne durumda kontrol et", top_k=2)
            self.assertGreaterEqual(len(c_hints), 1)
            self.assertEqual(c_hints[0]["chosen"]["tool"], "inspect_self_component")
            # Actor-critic blocks repeating the exact rejected action
            c_repeat = lm.local_actor_critic_review(
                "promptlar ne durumda", "list_directory", {"path": "/home/eren/Projects/sysagent"}
            )
            self.assertFalse(c_repeat["approved"])
            self.assertEqual(c_repeat["suggested_tool"], "inspect_self_component")

            # 5) Causal Digital Twin & Process Memory-Leak / Root-Cause Engine
            synthetic_snaps = [
                {
                    "ts": 1000.0,
                    "ram_percent": 62.0,
                    "avail_ram_mb": 4000.0,
                    "total_ram_mb": 16000.0,
                    "processes": [{"pid": 9991, "name": "leaky_worker", "rss_mb": 400.0, "cpu_percent": 12.0}],
                },
                {
                    "ts": 1120.0,  # +2 minutes
                    "ram_percent": 65.0,
                    "avail_ram_mb": 3920.0,
                    "total_ram_mb": 16000.0,
                    "processes": [{"pid": 9991, "name": "leaky_worker", "rss_mb": 480.0, "cpu_percent": 14.0}],
                },
            ]
            twin = analyze_causal_digital_twin(st, watcher=None, snapshots=synthetic_snaps)
            self.assertFalse(twin["healthy"])
            self.assertEqual(len(twin["memory_leaks"]), 1)
            leak = twin["memory_leaks"][0]
            self.assertEqual(leak["name"], "leaky_worker")
            self.assertEqual(leak["slope_mb_per_min"], 40.0)
            self.assertEqual(leak["oom_eta_min"], 98.0)
            self.assertTrue(any("leaky_worker" in c["chain"] for c in twin["causal_chains"]))
            kg_edges = st.query_kg("leak:leaky_worker:9991")
            self.assertTrue(any(e.get("relation") == "causes" for e in kg_edges))

    def test_five_live_runtime_integration_features(self) -> None:
        from sysagent.cortex import NeuralBrainCortex
        from sysagent.self_resilience import ContextBudgeter

        with Storage(db_path=self.root / "live_five.db") as st:
            lm = LocalMind(st)
            lm.train()

            # 1) Live Pre-Execution Actor-Critic Interception in AgentManager.run_turn
            lm.record_preference_pair(
                "promptlar ne durumda",
                chosen_tool="inspect_self_component",
                chosen_args={"component": "prompts"},
                rejected_tool="list_directory",
                rejected_args={"path": "/tmp"},
                reason="Use inspect_self_component for prompt inspection",
            )
            watcher = SimpleNamespace(current_context="normal", _send_desktop_notification=MagicMock())
            am = AgentManager(Settings(), watcher, st, PluginRegistry(), MagicMock(is_configured=True))
            turn_seq = [
                SimpleNamespace(
                    id="ac1",
                    steps=[
                        SimpleNamespace(
                            type="function_call",
                            id="c1",
                            name="list_directory",
                            arguments={"path": "/tmp"},
                        )
                    ],
                    output_text="",
                ),
                SimpleNamespace(id="ac2", steps=[], output_text="PromptRegistry incelendi."),
            ]
            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(am, "_call_model_with_failover", side_effect=turn_seq),
            ):
                ac_out = asyncio.run(am.run_turn("s-ac", "promptlar ne durumda"))
            self.assertIn("PromptRegistry incelendi", ac_out)
            _, last_tools = am._last_turn_info["s-ac"]
            self.assertEqual(last_tools[0]["name"], "inspect_self_component")
            self.assertEqual(last_tools[0]["arguments"], {"component": "prompts"})

            # 2) Impact-Guided Transactional Test Gate rolls back syntax-valid edit that breaks unit tests
            mini_repo = self.root / "mini_repo"
            mini_pkg = mini_repo / "pkg"
            mini_tests = mini_repo / "tests"
            mini_pkg.mkdir(parents=True, exist_ok=True)
            mini_tests.mkdir(parents=True, exist_ok=True)
            mod_file = mini_pkg / "calc_mod.py"
            test_file = mini_tests / "test_calc_mod.py"
            good_mod_src = "def add_two(x: int) -> int:\n    return x + 2\n"
            mod_file.write_text(good_mod_src, encoding="utf-8")
            test_file.write_text(
                "from pkg.calc_mod import add_two\nassert add_two(3) == 5, 'broken calc'\n",
                encoding="utf-8",
            )
            tx_batch = "impact-batch-1"
            execute_tool(
                "edit_file",
                {"path": str(mod_file), "target": "return x + 2", "replacement": "return x + 999"},
                None,
                st,
                batch_id=tx_batch,
            )
            self.assertIn("999", mod_file.read_text(encoding="utf-8"))
            rec_calls = [{"name": "edit_file", "arguments": {"path": str(mod_file)}}]
            imp_res = am.verify_and_transactional_rollback(
                rec_calls, tx_batch, user_message="update add_two", run_impact_tests=True
            )
            self.assertTrue(imp_res["rolled_back"])
            self.assertEqual(imp_res["failed_test"], "test_calc_mod.py")
            self.assertEqual(mod_file.read_text(encoding="utf-8"), good_mod_src)

            # 3) Cortex Proactive Leak & Anomaly Auto-Healer
            cortex = NeuralBrainCortex(Settings(), st, watcher=watcher)
            leak_snaps = [
                {
                    "ts": 2000.0,
                    "ram_percent": 70.0,
                    "avail_ram_mb": 3000.0,
                    "total_ram_mb": 16000.0,
                    "processes": [{"pid": 7771, "name": "runaway_daemon", "rss_mb": 500.0, "cpu_percent": 15.0}],
                },
                {
                    "ts": 2120.0,
                    "ram_percent": 74.0,
                    "avail_ram_mb": 2900.0,
                    "total_ram_mb": 16000.0,
                    "processes": [{"pid": 7771, "name": "runaway_daemon", "rss_mb": 600.0, "cpu_percent": 16.0}],
                },
            ]
            heal_res = cortex.evaluate_predictive_healer(force=False, notify=False, twin_snapshots=leak_snaps)
            self.assertEqual(heal_res["status"], "alerted")
            self.assertIsNotNone(heal_res["leak_question_id"])
            self.assertIn("runaway_daemon", heal_res["cause_hint"])
            pending_qs = st.list_brain_questions("pending")
            self.assertTrue(any("runaway_daemon" in str(q.get("question")) for q in pending_qs))

            # 4) Query-Aware Semantic Context Budgeter 2.0 preserves query-matching middle lines
            filler_head = "HEAD_LINE_ALPHA\n" * 120
            salient_mid = "\nCRITICAL_VRAM_OVERFLOW_SIGNAL detected on GPU0 at 14:59:00 with error_code=E991\n"
            filler_tail = "TAIL_LINE_OMEGA\n" * 120
            big_output = filler_head + salient_mid + filler_tail
            budgeted = ContextBudgeter.budget_tool_output(
                st,
                "read_journal_logs",
                big_output,
                max_chars=900,
                query="CRITICAL_VRAM_OVERFLOW_SIGNAL error_code E991",
            )
            self.assertLessEqual(len(budgeted), 1050)
            self.assertIn("CRITICAL_VRAM_OVERFLOW_SIGNAL", budgeted)
            self.assertIn("query-matched middle excerpt", budgeted)

            # 5) Episodic Project Knowledge Graph in Dream Cycle
            st.save_session_checkpoint(
                "cli-ep1",
                "Fix Wi-Fi KCM crash on disconnect",
                "Patched kcm.cpp null pointer",
                cwd="/home/eren/Projects/plasma-networkd",
                files=["/home/eren/Projects/plasma-networkd/kcm/kcm.cpp"],
            )
            ep_graph = cortex.consolidate_episodic_project_graph()
            self.assertGreaterEqual(ep_graph["projects_indexed"], 1)
            self.assertIn("plasma-networkd", ep_graph["projects"])
            proj_edges = st.query_kg("project:plasma-networkd")
            self.assertTrue(any(e.get("object") == "kcm.cpp" for e in proj_edges))
            self.assertIsNotNone(st.get_memory("project:plasma-networkd"))

    def test_five_end_to_end_autonomous_acceleration_features(self) -> None:
        from sysagent.agent import extract_cwd_project_context
        from sysagent.brain import CognitiveBrain
        from sysagent.evolve_ast import analyze_symbol_blast_radius
        from sysagent.self_ops import ToolCache

        with Storage(db_path=self.root / "accel.db") as st:
            # 1) Speculative Multi-Tool Batch Execution in LocalMindClient
            cli = LocalMindClient(st)
            with (
                patch.object(cli.interactions.mind, "run_gguf_chat_with_tools", return_value=None),
                patch.object(
                    cli.interactions.mind,
                    "predict_tool",
                    return_value=("system_status", {}, 0.88),
                ),
                patch.object(
                    cli.interactions.mind,
                    "predict_tool_chain",
                    return_value=[
                        {"name": "system_status", "arguments": {}, "score": 0.88},
                        {"name": "list_processes", "arguments": {"limit": 10}, "score": 0.65},
                    ],
                ),
            ):
                spec_res = cli.interactions.create(
                    input="check system status and top processes",
                    tools=TOOL_DECLARATIONS,
                )
            self.assertEqual([s.name for s in spec_res.steps], ["system_status", "list_processes"])

            # 2) CWD Project-Graph Auto-Context injection
            st.set_memory("project:plasma-networkd", "Active project 'plasma-networkd'; recent work: Fix Wi-Fi KCM")
            st.add_kg_edge("project:plasma-networkd", "touched_file", "kcm.cpp")
            st.add_kg_edge("project:plasma-networkd", "recent_topic", "Fix Wi-Fi KCM crash")
            p_ctx = extract_cwd_project_context(st, "[cwd: /home/eren/Projects/plasma-networkd/kcm]\nstatus?")
            self.assertIn("Active project graph (plasma-networkd):", p_ctx)
            self.assertIn("kcm.cpp", p_ctx)
            self.assertIn("Fix Wi-Fi KCM crash", p_ctx)

            # 3) Closed-Loop Brain Action Executor on BrainQuestion confirmation
            watcher = SimpleNamespace(
                current_context="normal",
                _send_desktop_notification=MagicMock(),
                get_health_snapshot=lambda: {"cpu_percent": 10.0, "ram_percent": 50.0},
            )
            fake_agent = SimpleNamespace(
                auth=MagicMock(is_configured=False),
                run_turn=MagicMock(),
                reset_session=MagicMock(),
            )
            brain = CognitiveBrain(Settings(), watcher, st, fake_agent)
            qid = st.create_brain_question(
                "Process 'leaky_app' (PID 98765) is leaking +45 MB/min. Terminate?",
                "Causal Digital Twin leak alert",
                "kill_leak:98765",
            )
            fake_proc = MagicMock()
            fake_proc.name.return_value = "leaky_app"
            with patch("psutil.Process", return_value=fake_proc):
                ans_res = asyncio.run(brain.answer_question(qid, "evet", learn_rule=False))
            self.assertEqual(ans_res["status"], "answered")
            self.assertTrue(ans_res["closed_loop"]["executed"])
            self.assertEqual(ans_res["closed_loop"]["process"], "leaky_app")
            fake_proc.terminate.assert_called_once()
            fake_agent.run_turn.assert_not_called()

            # 4) Blast-Radius Guided SelfEvolver metadata
            mod_blast = analyze_symbol_blast_radius("local_mind.py", storage=st)
            self.assertIn("local_mind.py", mod_blast["affected_modules"])
            self.assertGreaterEqual(len(mod_blast["covering_tests"]), 1)
            self.assertGreaterEqual(len(mod_blast["direct_callers"]), 1)

            # 5) Adaptive ToolCache & Latency Optimizer
            cache = ToolCache(ttl_sec=30.0)
            self.assertIn("inspect_self_component", ToolCache.CACHEABLE)
            for _ in range(4):
                st.record_tool_metric("read_file", 8.0, False)
                st.record_tool_metric("list_systemd_units", 960.0, False)
            ttls = cache.calibrate_from_metrics(st)
            self.assertIn("read_file", ttls)
            self.assertIn("list_systemd_units", ttls)
            self.assertGreater(cache.get_ttl("list_systemd_units"), cache.get_ttl("read_file"))
            self.assertGreaterEqual(cache.get_ttl("list_systemd_units"), 100.0)

    def test_five_operational_cli_api_and_resilience_features(self) -> None:
        import subprocess
        from sysagent import cli as cli_mod
        from sysagent import local_mind as lm_mod
        from sysagent.config import Settings
        from sysagent.plugins import PluginRegistry
        from sysagent.self_guard import SelfGuard
        from sysagent.self_learn import auto_forge_workflow_skills, prune_dead_macro_skills

        with Storage(db_path=self.root / "ops_five.db") as st:
            # 1) Feature 1: run_command TimeoutExpired partial output recovery & interactive stall detection
            timeout_exc = subprocess.TimeoutExpired(
                cmd="apt upgrade",
                timeout=5,
                output="Reading package lists...\nDo you want to continue? [Y/n] ",
                stderr="waiting on tty",
            )
            with patch("subprocess.run", side_effect=timeout_exc):
                t_out = json.loads(
                    execute_tool("run_command", {"command": "apt upgrade", "timeout": 5}, watcher=None, storage=st)
                )
            self.assertTrue(t_out["timed_out"])
            self.assertEqual(t_out["exit_code"], 124)
            self.assertTrue(t_out["interactive_stall"])
            self.assertIn("Do you want to continue?", t_out["partial_stdout"])
            self.assertIn("non-interactive flags", t_out["recovery_hint"])

            # 2) Feature 2: SelfGuard 2.0 DPO Preference Consistency Audit & Auto-Repair + Canary semantic_index
            pref_file = self.root / "dpo_prefs.jsonl"
            with patch.object(lm_mod, "PREFERENCE_FILE", pref_file):
                lm = LocalMind(st)
                lm.record_preference_pair(
                    "check cpu ram and system health snapshot",
                    chosen_tool="system_status",
                    chosen_args={},
                    rejected_tool="list_directory",
                    rejected_args={"path": "/"},
                )
                sg = SelfGuard(Settings(socket_path=str(self.root / "test.sock")), st)
                dpo_res = sg.audit_dpo_preferences(auto_repair=True)
                self.assertEqual(dpo_res["total_pairs"], 1)
                self.assertGreaterEqual(dpo_res["consistency_score"], 1.0)
                cal_res = sg.calibrate_thresholds()
                self.assertIn("dpo_audit", cal_res)
                canary_res = sg.run_canary_probes()
                self.assertIn("semantic_index", canary_res["probes"])
                self.assertIn("dual_slot_ready", canary_res["probes"]["local_models"])

            # 3) Feature 3: Parameterized Skill Forging & Dead-Skill GC
            pdir = self.root / "plugins_macro"
            pm = PluginRegistry(plugins_dir=pdir, mcp_config_file=self.root / "mcp.json")
            steps = [
                {"name": "inspect_self_component", "arguments": {"component": "prompts"}},
                {"name": "list_events", "arguments": {"limit": 5}},
            ]
            st.record_workflow("inspect self and events", steps)
            st.record_workflow("inspect self and events", steps)
            forged = auto_forge_workflow_skills(st, plugins=pm, min_hits=2)
            self.assertEqual(len(forged), 1)
            m_name = forged[0]
            self.assertIn(m_name, pm.tools)
            decl_props = pm.tools[m_name].declaration["parameters"]["properties"]
            self.assertIn("component", decl_props)
            self.assertIn("limit", decl_props)

            for _ in range(5):
                st.record_tool_metric(m_name, 25.0, True)
            retired = prune_dead_macro_skills(st, plugins=pm, max_error_rate=0.65, min_calls=4)
            self.assertIn(m_name, retired)
            self.assertTrue((pdir / f"auto_workflow_{m_name[len('macro_'):]}.py.quarantine").is_file())

            # 4) Feature 4: Shell Hook Auto-Triage & Instant Fix Cache
            st.add_lesson("fix:pytest_flag", "Use `pytest -q` without invalid `--badflag`.", "cmd=pytest --badflag exit=2")
            with (
                patch.object(cli_mod, "DATA_DIR", self.root),
                patch.object(cli_mod, "Storage", lambda: Storage(db_path=self.root / "ops_five.db")),
            ):
                err_p1 = cli_mod.record_last_shell_error("pytest --badflag", 2, str(self.root), "error: unrecognized arguments")
                err_p2 = cli_mod.record_last_shell_error("pytest --badflag", 2, str(self.root), "error: unrecognized arguments")
                rec_err = json.loads(err_p2.read_text(encoding="utf-8"))
            self.assertEqual(err_p1, err_p2)
            self.assertEqual(rec_err["repeat_count"], 2)
            self.assertIsNotNone(rec_err["instant_fix"])
            self.assertIn("pytest -q", rec_err["instant_fix"]["lesson"])

    def test_five_diagnostic_compression_kg_mcp_and_top_features(self) -> None:
        from sysagent.mcp_server import MCP_TOOLS, handle_mcp_request
        from sysagent.self_learn import investigate_root_cause, learn_status_summary
        from sysagent.tools import compress_tool_output
        from sysagent.top import build_dashboard

        with Storage(db_path=self.root / "diag_five.db") as st:
            # 1) Feature 1: Autonomous Hypothesis-Driven Root-Cause Investigator
            diag = investigate_root_cause(
                st,
                events=[
                    {
                        "category": "systemd",
                        "severity": "critical",
                        "message": "Unit redis-cache.service entered failed state with exit-code",
                    }
                ],
                health={"cpu_percent": 15.0, "ram_percent": 42.0, "disk_percent": 30.0},
            )
            self.assertFalse(diag["healthy"])
            self.assertIsNotNone(diag["top_hypothesis"])
            self.assertEqual(diag["top_hypothesis"]["kind"], "service_failure")
            self.assertEqual(diag["top_hypothesis"]["target"], "redis-cache.service")
            self.assertEqual(diag["top_hypothesis"]["suggested_action"], "restart_service:redis-cache.service")

            # 2) Feature 2: Smart Semantic Output Compressor
            noisy_lines = (
                [f"INFO boot step {i}" for i in range(25)]
                + ["DEBUG heartbeat ok"] * 30
                + [f"INFO routine worker tick {i}" for i in range(80)]
                + ["ERROR: fatal database connection timeout on port 5432"]
                + [f"INFO post tick {i}" for i in range(60)]
                + [" Tail summary line"]
            )
            compressed = compress_tool_output("\n".join(noisy_lines), max_chars=1200)
            self.assertLessEqual(len(compressed), 1200)
            self.assertIn("[repeated 30x]", compressed)
            self.assertIn("ERROR: fatal database connection timeout on port 5432", compressed)
            self.assertIn("Tail summary line", compressed)

            # 3) Feature 3: Knowledge Graph Contradiction Resolver & Dead-Ref Pruning
            st.add_kg_edge("host", "runs_os", "Arch Linux old")
            st.add_kg_edge("host", "runs_os", "Arch Linux rolling")
            st.add_kg_edge("leak:dead_proc:9999999", "causes", "oom_risk_in_5m")
            st.add_kg_edge("project:demo", "config_path", str(self.root / "nonexistent_ghost_file.yaml"))
            kg_res = st.consolidate_knowledge_graph()
            self.assertGreaterEqual(kg_res["contradictions_resolved"], 1)
            self.assertGreaterEqual(kg_res["dead_refs_pruned"], 2)
            host_edges = [e for e in st.query_kg("host") if e["relation"] == "runs_os"]
            self.assertEqual(len(host_edges), 1)
            self.assertEqual(host_edges[0]["object"], "Arch Linux rolling")

            # 4) Feature 4: MCP Server Auto-Exposure for Next-Gen Engines
            tool_names = {t["name"] for t in MCP_TOOLS}
            for expected_mcp in (
                "sysagent_digital_twin",
                "sysagent_blast_radius",
                "sysagent_dpo_audit",
                "sysagent_inspect_self",
            ):
                self.assertIn(expected_mcp, tool_names)
            mcp_blast = handle_mcp_request(
                {
                    "jsonrpc": "2.0",
                    "id": 42,
                    "method": "tools/call",
                    "params": {"name": "sysagent_blast_radius", "arguments": {"target": "execute_tool"}},
                },
                st,
            )
            assert mcp_blast is not None
            blast_payload = json.loads(mcp_blast["result"]["content"][0]["text"])
            self.assertEqual(blast_payload["target"], "execute_tool")
            self.assertIn("agent.py", blast_payload["affected_modules"])

            # 5) Feature 5: Live TUI (`sysagent top` & `learn_status_summary`) DPO, Twin, and Dual-Slot Telemetry
            l_sum = learn_status_summary(st)
            self.assertIn("dpo", l_sum)
            self.assertIn("digital_twin", l_sum)
            self.assertIn("aux_warm_model", l_sum["local_model"])
            layout = build_dashboard(
                {
                    "online": True,
                    "status": {
                        "agent_name": "SysAgent",
                        "model": "local-mind",
                        "uptime_sec": 120,
                        "health": {"cpu_percent": 12.0, "ram_percent": 40.0, "disk_percent": 50.0},
                        "learn": l_sum,
                    },
                    "events": [],
                    "snapshots": {},
                }
            )
            self.assertIsNotNone(layout)


if __name__ == "__main__":
    unittest.main()

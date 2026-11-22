"""Regression tests for final deep logic, concurrency, security, and integration audit fixes."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from sysagent.agent import AgentManager
from sysagent.config import Settings
from sysagent.evolve import SelfEvolver
from sysagent.local_mind import (
    LocalMind,
    _ThinkStreamFilter,
    _strip_think_tags,
    classify_task_role,
)
from sysagent.local_mind_client import LocalMindClient
from sysagent.notification import NotificationCenter
from sysagent.plugins import PluginRegistry
from sysagent.self_guard import SelfGuard
from sysagent.self_resilience import (
    CircuitBreakerRegistry,
    ContextBudgeter,
    build_recovery_hint,
    evaluate_self_gate,
    get_circuit_breaker,
    get_quota_governor,
    record_boot_module_hashes,
    refresh_boot_module_hash,
    run_integrity_and_drift_audit,
)
from sysagent.storage import Storage
from sysagent.tools import execute_tool


class FinalAuditFixTests(unittest.TestCase):
    """Verify all fixes from the final logic, concurrency, security, and integration audit."""

    def setUp(self) -> None:
        get_circuit_breaker().reset()
        get_quota_governor().reset()
        self._iso_td = tempfile.TemporaryDirectory()
        self.addCleanup(self._iso_td.cleanup)
        iso_root = Path(self._iso_td.name)
        iso_ds = iso_root / "dataset"
        iso_mods = iso_root / "models"
        iso_ds.mkdir(parents=True, exist_ok=True)
        iso_mods.mkdir(parents=True, exist_ok=True)
        for p in (
            patch("sysagent.local_mind.DATASET_DIR", iso_ds),
            patch("sysagent.local_mind.MODELS_DIR", iso_mods),
            patch("sysagent.local_mind.DISTILLED_FILE", iso_ds / "distilled_turns.jsonl"),
            patch("sysagent.local_mind.UNCERTAIN_FILE", iso_ds / "uncertain_queries.jsonl"),
            patch("sysagent.local_mind.WEIGHTS_FILE", iso_mods / "local_mind_weights.json"),
            patch("sysagent.local_mind.ROLE_ADAPTERS_FILE", iso_mods / "role_adapters.json"),
            patch("sysagent.local_mind.LORA_TRAIN_FILE", iso_ds / "lora_chatml.txt"),
            patch("sysagent.neural.NEURAL_NPZ_FILE", iso_mods / "neural_engine.npz"),
            patch("sysagent.neural.NEURAL_META_FILE", iso_mods / "neural_meta.json"),
        ):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self) -> None:
        get_circuit_breaker().reset()
        get_quota_governor().reset()

    def test_circuit_breaker_release_probe_and_failover_safety(self) -> None:
        cb = CircuitBreakerRegistry(failure_threshold=1, cooldown_sec=1.0)
        cb.record_failure("model:m1", "boom")
        cb._entries["model:m1"]["opened_until"] = 0.0
        self.assertTrue(cb.allow("model:m1"))
        self.assertFalse(cb.allow("model:m1"))
        cb.release_probe("model:m1")
        self.assertTrue(cb.allow("model:m1"))

        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "ag.db") as st:
            s = Settings(model="gemini-4-argon", light_model="gemini-3.8-flash")
            auth = MagicMock()
            auth.is_configured = True
            auth.rotate_key.return_value = False
            am = AgentManager(s, MagicMock(), st, PluginRegistry(), auth)

            global_cb = get_circuit_breaker()
            for _ in range(global_cb.failure_threshold):
                global_cb.record_failure(f"model:{s.model}", "err")
            global_cb._entries[f"model:{s.model}"]["opened_until"] = 0.0

            async def _bad_create(*_a, **_kw):
                raise ValueError("bad request payload")

            with patch.object(am, "_get_client", return_value=MagicMock()), patch.object(
                am, "_create_safe", side_effect=_bad_create
            ):
                with self.assertRaises(ValueError):
                    asyncio.run(
                        am._call_model_with_failover("s1", model=s.model, input="hi", tools=[], system_instruction="sys")
                    )
            self.assertEqual(global_cb.snapshot()[f"model:{s.model}"]["state"], "half_open")
            self.assertTrue(global_cb.allow(f"model:{s.model}"))
            global_cb.release_probe(f"model:{s.model}")

            async def _cancel_create(*_a, **_kw):
                raise asyncio.CancelledError()

            with patch.object(am, "_get_client", return_value=MagicMock()), patch.object(
                am, "_create_safe", side_effect=_cancel_create
            ):
                with self.assertRaises(asyncio.CancelledError):
                    asyncio.run(
                        am._call_model_with_failover("s1", model=s.model, input="hi", tools=[], system_instruction="sys")
                    )
            self.assertTrue(global_cb.allow(f"model:{s.model}"))
            global_cb.release_probe(f"model:{s.model}")

            for _ in range(global_cb.failure_threshold):
                global_cb.record_failure(f"model:{s.model}", "down")
            auth.rotate_key.reset_mock()
            fake_local = SimpleNamespace(
                interactions=SimpleNamespace(
                    create=lambda **_kw: SimpleNamespace(id="local-1", steps=[], output_text="offline ok")
                )
            )
            am._local_client = fake_local
            with patch.object(am, "_get_client", return_value=MagicMock()), patch.object(
                am, "_create_safe", side_effect=OSError("light down")
            ):
                out = asyncio.run(
                    am._call_model_with_failover("s2", model=s.model, input="hi", tools=[], system_instruction="sys")
                )
            self.assertEqual(out.output_text, "offline ok")
            auth.rotate_key.assert_not_called()
            self.assertNotIn("s2", am._session_models)

    def test_quota_governor_blocks_local_mind_and_prompt_evolve(self) -> None:
        gov = get_quota_governor()
        gov.record_rate_limit("gemini-4-neon", cooldown_sec=60.0)
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "qg.db") as st:
            unc_p = Path(td) / "uncertain.jsonl"
            unc_p.write_text(json.dumps({"query": "how many files", "score": 0.1}) + "\n", encoding="utf-8")
            with patch("sysagent.local_mind.UNCERTAIN_FILE", unc_p):
                lm = LocalMind(st)
                r1 = lm.resolve_uncertain_batch()
                r2 = lm.distill_from_teachers()
            self.assertEqual(r1["status"], "deferred")
            self.assertEqual(r1["reason"], "rate_limit_cooldown")
            self.assertEqual(r2["status"], "deferred")
            self.assertEqual(r2["reason"], "rate_limit_cooldown")

            s = Settings()
            auth = MagicMock()
            auth.is_configured = True
            ev = SelfEvolver(s, st, MagicMock(), MagicMock(auth=auth))
            r3 = asyncio.run(ev.evolve_prompts_once())
            self.assertEqual(r3["status"], "deferred")
            self.assertEqual(r3["reason"], "rate_limit_cooldown")

    def test_storage_filters_internal_and_expired_plus_recovery_and_think(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "mem.db") as st:
            st.set_memory("_internal:secret_cfg", "alpha beta gamma")
            st.set_memory("public_note", "alpha beta public")
            st.set_memory("expired_note", "alpha beta stale", expires_at="2000-01-01 00:00:00")
            hits = st.search_knowledge("alpha beta", limit=10)
            keys = [h["title"] for h in hits]
            self.assertEqual(keys, ["public_note"])

            st.add_lesson("_internal:secret_hint", "pytest modulenotfounderror", "secret internal")
            st.add_lesson("cmd_fail:pytest", "pytest modulenotfounderror", "Run pip install missing")
            hint = build_recovery_hint(
                st,
                "run_command",
                {"command": "pytest"},
                json.dumps({"exit_code": 2, "stderr": "ModuleNotFoundError"}),
            )
            self.assertIsNotNone(hint)
            self.assertIn("pip install missing", hint or "")
            self.assertNotIn("_internal:", hint or "")

        self.assertEqual(_strip_think_tags("<think>secret</think>Final answer"), "Final answer")
        self.assertEqual(_strip_think_tags("internal thought</think>Clean reply"), "Clean reply")
        self.assertEqual(_strip_think_tags("<think>unclosed reasoning"), "")

        flt = _ThinkStreamFilter()
        chunks = [flt.feed("<think>reasoning "), flt.feed("more </think>Hello "), flt.feed("world")]
        chunks.append(flt.flush())
        self.assertEqual("".join(chunks), "Hello world")

        flt2 = _ThinkStreamFilter()
        self.assertEqual(flt2.feed("thought</think>Reply") + flt2.flush(), "Reply")

        self.assertNotEqual(classify_task_role("merhaba nasilsin"), "embedding")
        with tempfile.TemporaryDirectory() as td2, Storage(db_path=Path(td2) / "lmc.db") as st2:
            cli = LocalMindClient(st2)
            emitted: list[str] = []

            def _fake_stream(prompt, *, on_delta=None, **_kw):
                on_delta("Here is dict ")
                on_delta('{"a": 1} done')
                return 'Here is dict {"a": 1} done'

            with (
                patch.object(cli.interactions.mind, "run_gguf_chat_with_tools", return_value=None),
                patch.object(cli.interactions.mind, "predict_tool", return_value=(None, {}, 0.0)),
                patch.object(cli.interactions.mind, "stream_gguf_inference", side_effect=_fake_stream),
            ):
                cli.interactions.create(
                    model="local-mind",
                    input="explain dict",
                    tools=[{"name": "system_status", "description": "status", "parameters": {"type": "object", "properties": {}}}],
                    system_instruction="sys",
                    on_delta=emitted.append,
                )
            self.assertEqual("".join(emitted), 'Here is dict {"a": 1} done')

            suppressed: list[str] = []

            def _fake_json_stream(prompt, *, on_delta=None, **_kw):
                on_delta('{"tool_call": "system_status"}')
                return '{"tool_call": "system_status"}'

            with (
                patch.object(cli.interactions.mind, "run_gguf_chat_with_tools", return_value=None),
                patch.object(cli.interactions.mind, "predict_tool", return_value=(None, {}, 0.0)),
                patch.object(cli.interactions.mind, "stream_gguf_inference", side_effect=_fake_json_stream),
            ):
                cli.interactions.create(
                    model="local-mind",
                    input="status",
                    tools=[{"name": "system_status", "description": "status", "parameters": {"type": "object", "properties": {}}}],
                    system_instruction="sys",
                    on_delta=suppressed.append,
                )
            self.assertEqual(suppressed, [])

    def test_eval_gate_does_not_pollute_uncertain_and_rolls_back_train(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "evg.db") as st:
            unc_p = Path(td) / "uncertain.jsonl"
            with patch("sysagent.local_mind.UNCERTAIN_FILE", unc_p):
                lm = LocalMind(st)
                res = evaluate_self_gate(lm, st)
            self.assertGreaterEqual(res["total"], 10)
            self.assertFalse(unc_p.exists())

            st.set_memory("_internal:self_eval_baseline", json.dumps({"score": 1.0}))
            st.set_memory("style:test", "keep short")
            lm.weights["tool_centroids"] = {"system_status": {"cpu": 1.0}}
            with (
                patch("sysagent.self_resilience.evaluate_self_gate", return_value={"score": 0.2, "baseline": 1.0, "regressed": True}),
                patch("sysagent.local_mind.LORA_TRAIN_FILE", Path(td) / "lora.jsonl") as lora_p,
                patch("sysagent.local_mind.ROLE_ADAPTERS_FILE", Path(td) / "role_adapters.json") as ra_p,
                patch("sysagent.local_mind._save_weights_file"),
                patch("sysagent.neural.NeuralEngine.train") as mock_nn_train,
            ):
                lm.train()
                self.assertFalse(lora_p.exists())
                self.assertFalse(ra_p.exists())
                mock_nn_train.assert_not_called()
            self.assertIn("system_status", lm.weights.get("tool_centroids", {}))

            hist = [
                {"role": "user", "content": f"q{i} " + ("x" * 120)}
                if i % 2 == 0
                else {"role": "assistant", "content": f"a{i} " + ("y" * 120)}
                for i in range(16)
            ]
            compacted = ContextBudgeter.compact_history(hist, max_chars=700, keep_recent=4)
            self.assertEqual(compacted[0]["role"], "system")
            self.assertGreaterEqual(len(compacted), 3)
            self.assertEqual(compacted[1]["role"], "user")

    def test_fetch_url_ssrf_cap_and_inspect_component_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "tl.db") as st:
            with patch(
                "socket.getaddrinfo",
                return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.64.0.1", 0))],
            ):
                r_cgnat = json.loads(execute_tool("fetch_url", {"url": "http://cgnat.test"}, None, st))
                self.assertIn("not allowed", r_cgnat["error"])

            with patch(
                "socket.getaddrinfo",
                return_value=[(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:127.0.0.1", 0, 0, 0))],
            ):
                r_v6map = json.loads(execute_tool("fetch_url", {"url": "http://v6map.test"}, None, st))
                self.assertIn("not allowed", r_v6map["error"])

            redir_resp = MagicMock()
            redir_resp.__enter__.return_value = redir_resp
            redir_resp.status_code = 302
            redir_resp.url = "https://example.com"
            redir_resp.headers = {"location": "https://example.com/next "}
            with (
                patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]),
                patch("httpx.stream", return_value=redir_resp),
            ):
                r_redir = json.loads(execute_tool("fetch_url", {"url": "https://example.com"}, None, st))
                self.assertEqual(r_redir["error"], "Too many redirects.")

            big_resp = MagicMock()
            big_resp.__enter__.return_value = big_resp
            big_resp.status_code = 200
            big_resp.url = "https://example.com/big"
            big_resp.encoding = "utf-8"
            big_resp.headers = {}
            big_resp.iter_bytes.return_value = iter([b"A" * (600 * 1024), b"B" * (600 * 1024)])
            with (
                patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]),
                patch("httpx.stream", return_value=big_resp),
            ):
                r_big = json.loads(execute_tool("fetch_url", {"url": "https://example.com/big"}, None, st))
                self.assertEqual(r_big["status_code"], 200)
                self.assertEqual(len(r_big["text"]), 5000)

            notif = NotificationCenter(Settings(), st)
            with patch("subprocess.Popen"):
                notif.send("T", "B", bypass_gates=True)
            fake_cortex = SimpleNamespace(
                get_status=lambda: {
                    "flow": {"mode": "normal"},
                    "last_heal": {"summary": "healthy"},
                    "last_dream": None,
                    "habit_suggestions": [{"signature": "git status"}],
                }
            )
            fake_watcher = SimpleNamespace(notifier=notif, runtime_refs={"cortex": fake_cortex})
            with patch(
                "sysagent.self_learn.proposal_stats",
                return_value={"total": 4, "by_status": {"applied": 2}},
            ):
                insp = json.loads(execute_tool("inspect_self_component", {"component": "all"}, fake_watcher, st))
            self.assertEqual(insp["cortex"]["habit_suggestions_count"], 1)
            self.assertEqual(insp["cortex"]["predictive_healer"]["summary"], "healthy")
            self.assertEqual(insp["notifications"]["total"], 1)
            self.assertIn("Evolve applied=2/4", insp["overview"])

    def test_run_turn_error_json_switch_auto_and_cancel_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "rt.db") as st:
            s = Settings()
            auth = MagicMock()
            auth.is_configured = True
            watcher = SimpleNamespace(current_context="normal", _send_desktop_notification=MagicMock())
            am = AgentManager(s, watcher, st, PluginRegistry(), auth)

            steps = [
                SimpleNamespace(
                    id="i1",
                    steps=[SimpleNamespace(type="function_call", id="c1", name="read_file", arguments={"path": "/tmp/a"})],
                    output_text="",
                ),
                SimpleNamespace(id="i2", steps=[], output_text="Done reading."),
            ]
            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(am, "_call_model_with_failover", side_effect=steps),
                patch("sysagent.agent.execute_tool", return_value=json.dumps({"content": 'sample `"error": 1`'})),
            ):
                reply = asyncio.run(am.run_turn("s-ok", "read it"))
            self.assertEqual(reply, "Done reading.")
            self.assertEqual(get_circuit_breaker().snapshot().get("tool:read_file", {}).get("failures", 0), 0)

            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(
                    am,
                    "_call_model_with_failover",
                    side_effect=[
                        SimpleNamespace(
                            id="i3",
                            steps=[SimpleNamespace(type="function_call", id="c2", name="read_file", arguments={"path": "/tmp/b"})],
                            output_text="",
                        ),
                        SimpleNamespace(id="i4", steps=[], output_text="Recovered."),
                    ],
                ),
                patch("sysagent.agent.execute_tool", side_effect=RuntimeError("boom tool")),
            ):
                asyncio.run(am.run_turn("s-err", "read broken"))
            self.assertEqual(get_circuit_breaker().snapshot().get("tool:read_file", {}).get("failures", 0), 1)

            async def _run_cancel() -> None:
                calls = 0

                async def _slow_failover(session_id, **kwargs):
                    nonlocal calls
                    calls += 1
                    on_d = kwargs.get("on_delta")
                    if calls == 1 and on_d:
                        on_d("partial ")
                        await asyncio.sleep(5)
                    return SimpleNamespace(id="ix", steps=[], output_text="done")

                with (
                    patch.object(am, "_get_client", return_value=MagicMock()),
                    patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                    patch.object(am, "_call_model_with_failover", side_effect=_slow_failover),
                ):
                    task = asyncio.create_task(am.run_turn("s-can", "hello", emit=lambda _e: asyncio.sleep(0)))
                    await asyncio.sleep(0.05)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

            asyncio.run(_run_cancel())

            recorded_inputs: list[Any] = []

            async def _capture_failover(session_id, **kwargs):
                recorded_inputs.append((kwargs.get("model"), kwargs.get("input")))
                if len(recorded_inputs) == 1:
                    return SimpleNamespace(
                        id="im1",
                        steps=[
                            SimpleNamespace(type="function_call", id="sw1", name="switch_model", arguments={"model": "local-mind"}),
                            SimpleNamespace(type="function_call", id="st1", name="system_status", arguments={}),
                        ],
                        output_text="",
                    )
                return SimpleNamespace(id="im2", steps=[], output_text="System healthy.")

            am._pinned_models["s-sw"] = s.model
            am._session_models["s-sw"] = s.model
            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(am, "_call_model_with_failover", side_effect=_capture_failover),
                patch("sysagent.agent.execute_tool", return_value='{"cpu": 5}'),
            ):
                asyncio.run(am.run_turn("s-sw", "switch to local and check cpu"))
            self.assertEqual(recorded_inputs[1][0], "local-mind")
            self.assertIn("Original user request: switch to local and check cpu", str(recorded_inputs[1][1]))

            auto_models: list[str] = []

            async def _capture_auto(session_id, **kwargs):
                auto_models.append(str(kwargs.get("model")))
                if len(auto_models) == 1:
                    return SimpleNamespace(
                        id="ia1",
                        steps=[
                            SimpleNamespace(type="function_call", id="sw2", name="switch_model", arguments={"model": "auto"}),
                            SimpleNamespace(type="function_call", id="st2", name="system_status", arguments={}),
                        ],
                        output_text="",
                    )
                return SimpleNamespace(id="ia2", steps=[], output_text="Auto routed.")

            am._pinned_models["s-auto"] = "gemini-4-argon"
            am._session_models["s-auto"] = "gemini-4-argon"
            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(am, "_call_model_with_failover", side_effect=_capture_auto),
                patch("sysagent.agent.choose_model", return_value="gemini-3.8-flash"),
                patch("sysagent.agent.execute_tool", return_value='{"cpu": 2}'),
            ):
                asyncio.run(am.run_turn("s-auto", "switch to auto and check status"))
            self.assertEqual(auto_models, ["gemini-4-argon", "gemini-3.8-flash"])

    def test_boot_hash_refresh_and_testing_env_in_staging(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "sysagent"
            pkg.mkdir()
            mod_f = pkg / "sample_mod.py"
            mod_f.write_text("X = 1\n", encoding="utf-8")
            with Storage(db_path=Path(td) / "bh.db") as st:
                record_boot_module_hashes(pkg)
                mod_f.write_text("X = 2\n", encoding="utf-8")
                d1 = run_integrity_and_drift_audit(st, pkg_dir=pkg)
                self.assertIn("sample_mod.py", d1["stale_modules"])
                refresh_boot_module_hash(mod_f)
                d2 = run_integrity_and_drift_audit(st, pkg_dir=pkg)
                self.assertNotIn("sample_mod.py", d2["stale_modules"])

            captured_env: dict[str, str] = {}

            def _fake_run(cmd, **kw):
                if isinstance(kw.get("env"), dict) and "SYSAGENT_TESTING" in kw["env"]:
                    captured_env.update(kw["env"])
                    return SimpleNamespace(returncode=0, stdout="STAGING_OK\n", stderr="")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with Storage(db_path=Path(td) / "ev.db") as st:
                ev = SelfEvolver(Settings(), st, MagicMock(), MagicMock())
                with patch("subprocess.run", side_effect=_fake_run):
                    ok, _ = ev._run_staging_tests(
                        "i18n.py",
                        (Path(__file__).resolve().parents[1] / "sysagent" / "i18n.py").read_text(encoding="utf-8"),
                    )
                self.assertTrue(ok)
                self.assertEqual(captured_env.get("SYSAGENT_TESTING"), "1")

    def test_self_guard_pid_ownership_and_readonly_autopsy(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "sg.db") as st:
            marker = Path(td) / "daemon.running"
            live_other_pid = os.getppid()
            if live_other_pid == os.getpid():
                live_other_pid = 1
            marker.write_text(json.dumps({"pid": live_other_pid}), encoding="utf-8")
            sg = SelfGuard(Settings(), st, marker_file=marker)
            res = sg.on_startup()
            self.assertFalse(res["crashed"])
            self.assertEqual(res.get("live_owner_pid"), live_other_pid)
            sg.mark_clean_shutdown()
            self.assertTrue(marker.is_file())
            self.assertEqual(json.loads(marker.read_text(encoding="utf-8")).get("pid"), live_other_pid)

            aut = sg.run_crash_autopsy(journal_text="normal exit", record=False)
            self.assertFalse(aut["crashed"])
            self.assertEqual(len(st.list_brain_thoughts(limit=5)), 0)
            self.assertEqual(len(st.list_lessons(limit=5)), 0)


if __name__ == "__main__":
    unittest.main()

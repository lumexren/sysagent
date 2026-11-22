"""Tests for local multimodal vision, shell stderr hook, colored diff approval, semantic checkpoints, and VRAM hysteresis/prewarm."""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from sysagent import cli as cli_mod
from sysagent import local_inference as li_mod
from sysagent import server as srv_mod
from sysagent.agent import AgentManager
from sysagent.config import Settings
from sysagent.local_mind import (
    LocalMind,
    extract_inline_images,
    find_mmproj_for_model,
)
from sysagent.local_mind_client import LocalMindClient
from sysagent.plugins import PluginRegistry
from sysagent.safety import compute_approval_diff
from sysagent.storage import Storage


class NewFeaturesTests(unittest.TestCase):
    """End-to-end unit and integration tests for all 5 new features."""

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
            patch("sysagent.cli.DATA_DIR", root / "cli_data"),
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
        li_mod._VRAM_CACHE = (0.0, {})

    def test_local_vision_mmproj_routing_and_at_image(self) -> None:
        mimo = self.mods / "MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf"
        mmproj = self.mods / "mmproj-MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf"
        coder = self.mods / "OmniCoder-9B.Q4_K_M.gguf"
        for f in (mimo, mmproj, coder):
            f.write_bytes(b"GGUF" + b"\x00" * 64)

        self.assertEqual(find_mmproj_for_model(mimo), mmproj)
        self.assertIsNone(find_mmproj_for_model(coder))

        shot = self.root / "ui.png"
        shot.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="))
        expanded, labels = cli_mod._expand_at_paths("inspect @ui.png please", self.root)
        self.assertEqual(labels, ["ui.png"])
        self.assertIn(f"[Attached image `{shot.resolve()}`]", expanded)

        raw_txt, imgs = extract_inline_images(expanded)
        self.assertIn("[Attached image", raw_txt)
        self.assertEqual(len(imgs), 1)
        self.assertEqual(imgs[0]["mime_type"], "image/png")

        spawned_cmds: list[list[str]] = []

        class _AliveProc:
            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        def _fake_popen(cmd, **_kw):
            spawned_cmds.append(list(cmd))
            return _AliveProc()

        with (
            patch("shutil.which", return_value="/usr/bin/llama-server"),
            patch("subprocess.Popen", side_effect=_fake_popen),
            patch("httpx.get", return_value=SimpleNamespace(status_code=200)),
            patch("sysagent.local_inference.can_fit_gguf_model", return_value=(True, "full_vram")),
        ):
            u1 = li_mod.ensure_llama_server(mimo)
            u2 = li_mod.ensure_llama_server(mimo, mmproj=mmproj)
            u3 = li_mod.ensure_llama_server(mimo, mmproj=mmproj)
            self.assertIsNotNone(u1)
            self.assertEqual(u2, u3)
            self.assertEqual(len(spawned_cmds), 2)
            self.assertNotIn("--mmproj", spawned_cmds[0])
            self.assertIn("--mmproj", spawned_cmds[1])
            self.assertEqual(spawned_cmds[1][spawned_cmds[1].index("--mmproj") + 1], str(mmproj))

        with Storage(db_path=self.root / "mm.db") as st:
            cli = LocalMindClient(st)
            seen_calls: list[dict[str, object]] = []

            def _fake_infer(prompt, *, preferred_role=None, images=None, **_kw):
                seen_calls.append({"prompt": prompt, "role": preferred_role, "images": images})
                return "Image shows a 1x1 pixel PNG."

            with patch.object(cli.interactions.mind, "run_gguf_inference", side_effect=_fake_infer):
                res = cli.interactions.create(
                    model="local-mind",
                    input=expanded,
                    system_instruction="sys",
                )
                self.assertIn("1x1 pixel PNG", res.output_text)
                self.assertEqual(seen_calls[0]["role"], "multimodal")
                self.assertEqual(len(seen_calls[0]["images"]), 1)  # type: ignore[arg-type]

                # Multimodal tool_result list also passes extracted images cleanly without raw base64 bloat
                res_tool = cli.interactions.create(
                    model="local-mind",
                    input=[
                        {
                            "type": "function_result",
                            "name": "take_screenshot",
                            "result": [
                                {"type": "text", "text": "Screenshot captured."},
                                {"type": "image", "data": imgs[0]["data"], "mime_type": "image/png"},
                            ],
                        }
                    ],
                    system_instruction="sys",
                )
                self.assertIn("1x1 pixel PNG", res_tool.output_text)
                self.assertEqual(seen_calls[1]["role"], "multimodal")
                self.assertIn("[image attached]", str(seen_calls[1]["prompt"]))

            # try_local_cascade handles inline images locally
            lm = LocalMind(st)
            with patch.object(lm, "_run_single_gguf", return_value="Offline screenshot ok") as mock_single:
                casc = lm.try_local_cascade(expanded)
            self.assertIsNotNone(casc)
            self.assertEqual(casc["tier"], "local:multimodal")  # type: ignore[index]
            self.assertEqual(mock_single.call_args.kwargs["mmproj"], mmproj)

    def test_shell_stderr_hook_install_and_fix_prompt(self) -> None:
        zsh_hook = cli_mod._render_shell_hook("zsh")
        bash_hook = cli_mod._render_shell_hook("bash")
        self.assertIn("add-zsh-hook precmd _sysagent_err_hook", zsh_hook)
        self.assertIn("PROMPT_COMMAND=", bash_hook)
        self.assertIn("srun()", zsh_hook)

        zshrc = self.root / ".zshrc"
        zshrc.write_text("# existing config\nexport FOO=1\n", encoding="utf-8")
        cli_mod._install_shell_hook("zsh", rc_path=zshrc)
        cli_mod._install_shell_hook("zsh", rc_path=zshrc)
        content = zshrc.read_text(encoding="utf-8")
        self.assertEqual(content.count("# >>> sysagent shell hook >>>"), 1)
        self.assertIn("export FOO=1", content)

        stderr_log = cli_mod.DATA_DIR / "last_stderr.log"
        stderr_log.parent.mkdir(parents=True, exist_ok=True)
        stderr_log.write_text('AssertionError: broken "quote" \\ line\n', encoding="utf-8")
        cli_mod._cmd_hook("zsh", install=False, record_args=["2", str(self.root), 'pytest -k "x"'])
        self.assertFalse(stderr_log.exists())

        prompt = cli_mod._prepare_fix_prompt(locale="en", no_cache=True)
        self.assertIsNotNone(prompt)
        self.assertIn('pytest -k "x"', prompt or "")
        self.assertIn("AssertionError: broken", prompt or "")

    def test_colored_approval_diff_preview_and_event(self) -> None:
        target_f = self.root / "sample.py"
        target_f.write_text("a = 1\nb = 2\n", encoding="utf-8")
        diff = compute_approval_diff(
            "edit_file",
            {"path": str(target_f), "target": "b = 2", "replacement": "b = 42"},
        )
        self.assertIn("-b = 2", diff)
        self.assertIn("+b = 42", diff)

        long_diff = "\n".join(f"+line {i}" for i in range(30))
        colored, hidden = cli_mod._format_colored_diff(long_diff, max_lines=16)
        self.assertEqual(hidden, 14)
        self.assertIn("\033[32m", colored)

        with Storage(db_path=self.root / "ap.db") as st:
            emitted: list[dict[str, object]] = []
            auth = MagicMock()
            auth.is_configured = True
            watcher = SimpleNamespace(
                current_context="normal",
                _send_desktop_notification=MagicMock(return_value=None),
            )
            am = AgentManager(Settings(approval_mode="smart"), watcher, st, PluginRegistry(), auth)

            async def _emit(ev):
                emitted.append(ev)
                if ev.get("type") == "approval_required":
                    am.resolve_approval(ev["approval_id"], False)

            steps = [
                SimpleNamespace(
                    id="a1",
                    steps=[
                        SimpleNamespace(
                            type="function_call",
                            id="fc1",
                            name="edit_file",
                            arguments={"path": str(target_f), "target": "b = 2", "replacement": "b = 42"},
                        )
                    ],
                    output_text="",
                ),
                SimpleNamespace(id="a2", steps=[], output_text="Skipped."),
            ]
            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(am, "_call_model_with_failover", side_effect=steps),
            ):
                asyncio.run(am.run_turn("s-diff", "update sample.py", emit=_emit))
            ap_ev = next(e for e in emitted if e.get("type") == "approval_required")
            self.assertIn("+b = 42", str(ap_ev.get("diff") or ""))

    def test_cross_session_checkpoints_agent_and_api_resume(self) -> None:
        with Storage(db_path=self.root / "cp.db") as st:
            st.upsert_session("cli-nginx", None, title="nginx tls setup")
            st.add_message("cli-nginx", "user", "fix nginx tls certificate error")
            st.save_session_checkpoint(
                session_id="cli-nginx",
                topic="nginx tls sertifika yenileme",
                summary="Certbot config updated in /etc/nginx/sites-enabled/app.conf.",
                files=["/etc/nginx/sites-enabled/app.conf"],
                commands=["nginx -t"],
                decisions=["edit_file", "run_command"],
                cwd="/etc/nginx",
                model="gemini-4-argon",
            )
            st.save_session_checkpoint(
                session_id="cli-other",
                topic="docker prune cleanup",
                summary="Removed dangling images.",
            )
            hits = st.search_session_checkpoints("sertifika app.conf", limit=3)
            self.assertTrue(hits)
            self.assertEqual(hits[0]["session_id"], "cli-nginx")

            # AgentManager auto-checkpoints tool turn and injects checkpoint on unchained follow-up
            seen_sys: list[str] = []

            async def _fake_failover(_sid, **kw):
                seen_sys.append(str(kw.get("system_instruction") or ""))
                return SimpleNamespace(id="id1", steps=[], output_text="Nginx checked successfully.")

            am = AgentManager(
                Settings(),
                SimpleNamespace(current_context="normal", _send_desktop_notification=MagicMock()),
                st,
                PluginRegistry(),
                MagicMock(is_configured=False),
            )
            with (
                patch.object(am, "_get_client", return_value=MagicMock()),
                patch.object(am, "_match_zero_llm_fastpath", return_value=None),
                patch.object(am, "_call_model_with_failover", side_effect=_fake_failover),
            ):
                asyncio.run(am.run_turn("cli-nginx", "recheck nginx config status"))
            self.assertTrue(seen_sys and "Active session checkpoint:" in seen_sys[0])
            self.assertIn("app.conf", seen_sys[0])

            with (
                patch.object(srv_mod, "storage", st),
                patch.dict(os.environ, {"SYSAGENT_TESTING": "1"}),
            ):
                with TestClient(srv_mod.app) as client:
                    hdr = {"Authorization": f"Bearer {srv_mod.settings.auth_token}"} if srv_mod.settings.auth_token else {}
                    r_list = client.get("/api/checkpoints", headers=hdr)
                    self.assertEqual(r_list.status_code, 200)
                    self.assertGreaterEqual(len(r_list.json()["checkpoints"]), 2)

                    r_res = client.post("/api/sessions/resume", json={"query": "nginx sertifika"}, headers=hdr)
                    self.assertEqual(r_res.status_code, 200)
                    body = r_res.json()
                    self.assertTrue(body["found"])
                    self.assertEqual(body["session_id"], "cli-nginx")
                    self.assertIn("app.conf", body["context_preview"])

    def test_vram_fit_hysteresis_and_routine_prewarm(self) -> None:
        drm_card = self.root / "card0" / "device"
        drm_card.mkdir(parents=True)
        (drm_card / "mem_info_vram_total").write_text(str(8192 * 1024 * 1024), encoding="utf-8")
        (drm_card / "mem_info_vram_used").write_text(str(2048 * 1024 * 1024), encoding="utf-8")

        fake_smi = SimpleNamespace(returncode=0, stdout="6144, 2048, 8192\n")
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=fake_smi),
        ):
            v = li_mod.get_vram_info(ttl_sec=0.0)
        self.assertTrue(v["available"])
        self.assertEqual(v["source"], "nvidia")
        self.assertEqual(v["free_mb"], 6144)

        gguf_ok = self.mods / "OmniCoder-9B.Q4_K_M.gguf"
        gguf_ok.write_bytes(b"\x00" * 1024)
        with (
            patch("sysagent.local_inference._has_ram_for_gguf", return_value=True),
            patch(
                "sysagent.local_inference.get_vram_info",
                return_value={"available": True, "free_mb": 6144, "used_mb": 2048, "total_mb": 8192, "source": "nvidia"},
            ),
        ):
            ok_full, mode_full = li_mod.can_fit_gguf_model(gguf_ok)
            self.assertTrue(ok_full)
            self.assertEqual(mode_full, "full_vram")

        with (
            patch("sysagent.local_inference._has_ram_for_gguf", return_value=True),
            patch(
                "sysagent.local_inference.get_vram_info",
                return_value={"available": True, "free_mb": 200, "used_mb": 7992, "total_mb": 8192, "source": "nvidia"},
            ),
        ):
            ok_part, mode_part = li_mod.can_fit_gguf_model(gguf_ok)
            self.assertTrue(ok_part)
            self.assertEqual(mode_part, "partial_offload")

        chat_m = self.mods / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        mm_m = self.mods / "MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf"
        chat_m.write_bytes(b"\x00" * 64)
        mm_m.write_bytes(b"\x00" * 64)
        li_mod._LLAMA_SERVER_PROC = SimpleNamespace(poll=lambda: None)  # type: ignore[assignment]
        li_mod._LLAMA_SERVER_MODEL = str(gguf_ok)
        li_mod._LLAMA_SERVER_LAST_USED = time.monotonic()

        from sysagent.local_mind import classify_gguf_role

        kept, stayed = li_mod.choose_model_with_hysteresis(chat_m, "chat", "selam naber", classify_gguf_role, hysteresis_sec=90.0)
        self.assertTrue(stayed)
        self.assertEqual(kept, gguf_ok)

        swapped_mm, stayed_mm = li_mod.choose_model_with_hysteresis(mm_m, "multimodal", "ekrana bak", classify_gguf_role, hysteresis_sec=90.0)
        self.assertFalse(stayed_mm)
        self.assertEqual(swapped_mm, mm_m)

        # Idle expired -> stops keeping warm model
        li_mod._LLAMA_SERVER_LAST_USED = time.monotonic() - 120.0
        exp_m, exp_stayed = li_mod.choose_model_with_hysteresis(chat_m, "chat", "selam", classify_gguf_role, hysteresis_sec=90.0)
        self.assertFalse(exp_stayed)
        self.assertEqual(exp_m, chat_m)

        # prewarm_for_routine: defers during deep_focus, prewarms when idle + routine + full_vram
        self._reset_inference_state()
        with Storage(db_path=self.root / "pw.db") as st:
            lm = LocalMind(st)
            defer_res = lm.prewarm_for_routine(
                cortex=SimpleNamespace(last_flow={"mode": "deep_focus", "suppress_low_priority": True})
            )
            self.assertEqual(defer_res["status"], "deferred")

            with (
                patch("sysagent.self_learn.predict_user_routine", return_value={"hour": 14, "likely_role": "coding", "top_app": "code"}),
                patch("sysagent.local_mind.can_fit_gguf_model", return_value=(True, "full_vram")),
                patch.object(lm, "_ensure_llama_server", return_value="http://127.0.0.1:8769") as mock_srv,
            ):
                pw_res = lm.prewarm_for_routine(
                    watcher=SimpleNamespace(get_health_snapshot=lambda: {"cpu_percent": 15.0}),
                    cortex=SimpleNamespace(last_flow={"mode": "normal", "suppress_low_priority": False}),
                )
            self.assertEqual(pw_res["status"], "prewarmed")
            self.assertEqual(pw_res["role"], "coding")
            self.assertEqual(pw_res["model"], gguf_ok.name)
            mock_srv.assert_called_once()


if __name__ == "__main__":
    unittest.main()

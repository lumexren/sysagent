"""Tests for SQLite locking, RAM single-slot guard, weak-role distillation, and local-first routing."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sysagent import local_mind
from sysagent.storage import Storage


class LocalAndStorageImprovementsTests(unittest.TestCase):
    def test_storage_shared_lock_and_safe_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "shared.db"
            s1 = Storage(db_path=db)
            s2 = Storage(db_path=db)
            self.assertIs(s1._lock, s2._lock)

            errors: list[Exception] = []

            def writer(st: Storage, prefix: str) -> None:
                try:
                    for i in range(15):
                        st.set_memory(f"{prefix}_{i}", f"val_{i}")
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=writer, args=(s1, "a"))
            t2 = threading.Thread(target=writer, args=(s2, "b"))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            self.assertEqual(errors, [])
            self.assertGreaterEqual(s1.prune_and_checkpoint(stale_days=30), 0)

    def test_underrepresented_roles_ordering(self) -> None:
        mind = local_mind.LocalMind(storage=None)
        mind.weights["role_counts"] = {
            "chat": 90,
            "general": 70,
            "coding": 60,
            "reasoning": 3,
            "multimodal": 1,
        }
        roles = mind.underrepresented_roles()
        self.assertEqual(roles[:2], ["multimodal", "reasoning"])
        self.assertEqual(mind.weakest_specialist_role(), "multimodal")

    def test_has_ram_for_gguf_headroom_guard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake_big = Path(td) / "big.gguf"
            fake_big.write_bytes(b"0")
            fake_stat = SimpleNamespace(st_size=int(5.5 * (1024**3)), st_mode=0o100644)
            with patch.object(Path, "stat", return_value=fake_stat), patch(
                "psutil.virtual_memory", return_value=SimpleNamespace(available=int(0.9 * (1024**3)))
            ):
                self.assertFalse(local_mind._has_ram_for_gguf(fake_big, min_headroom_gb=1.5))
            with patch.object(Path, "stat", return_value=fake_stat), patch(
                "psutil.virtual_memory", return_value=SimpleNamespace(available=int(4.0 * (1024**3)))
            ):
                self.assertTrue(local_mind._has_ram_for_gguf(fake_big, min_headroom_gb=1.5))

    def test_local_first_routes_coding_specialist(self) -> None:
        mind = local_mind.LocalMind(storage=None)
        fake_coder = Path("/tmp/OmniCoder-9B.Q4_K_M.gguf")
        with patch.object(local_mind, "find_all_gguf_models", return_value=[fake_coder]), patch.object(
            local_mind, "classify_gguf_role", return_value="coding"
        ), patch.object(local_mind, "classify_task_role", return_value="coding"), patch.object(
            local_mind, "find_gguf_model", return_value=fake_coder
        ), patch.object(mind, "predict_tool", return_value=(None, {}, 0.0)), patch.object(
            mind, "_run_single_gguf", return_value="Use a list comprehension in Python to filter items cleanly."
        ):
            res = mind.try_local_cascade("explain python list comprehension syntax briefly")
            self.assertIsNotNone(res)
            self.assertEqual(res["tier"], "local:coding")
            self.assertEqual(res["model"], "OmniCoder-9B.Q4_K_M.gguf")

    def test_storage_context_manager_and_watcher_ttl_cache(self) -> None:
        from sysagent.watcher import SystemWatcher

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "ctx.db"
            with Storage(db_path=db) as st:
                st.set_memory("k", "v")
                self.assertEqual(st.get_memory("k"), "v")
            self.assertIsNone(st._conn)

        w = SystemWatcher.__new__(SystemWatcher)
        w._disk_history = []
        w._last_zombie_count = 0
        calls = {"n": 0}
        with patch.object(w, "_read_max_temperature_c", side_effect=lambda: (calls.__setitem__("n", calls["n"] + 1) or 50.0)), patch.object(
            w, "_read_gpu_stats", return_value=None
        ):
            s1 = w.get_health_snapshot(ttl_sec=10.0)
            s2 = w.get_health_snapshot(ttl_sec=10.0)
            self.assertEqual(calls["n"], 1)
            self.assertEqual(s1["max_temp_c"], s2["max_temp_c"])
            w.get_health_snapshot(ttl_sec=10.0, force=True)
            self.assertEqual(calls["n"], 2)

    def test_neural_cache_invalidates_on_meta_mtime_change(self) -> None:
        from sysagent.neural import NeuralEngine

        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            npz = p / "n.npz"
            meta = p / "n.json"
            eng1 = NeuralEngine(storage=None, npz_file=npz, meta_file=meta)
            eng1._save()
            # Mutate only meta_file with a newer mtime
            eng1.meta["custom_tag"] = "fresh"
            meta.write_text(__import__("json").dumps(eng1.meta), encoding="utf-8")
            st = meta.stat()
            __import__("os").utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))
            eng2 = NeuralEngine(storage=None, npz_file=npz, meta_file=meta)
            self.assertEqual(eng2.meta.get("custom_tag"), "fresh")

    def test_local_backend_llama_cpp_priority(self) -> None:
        mind = local_mind.LocalMind(storage=None)
        calls: list[str] = []

        class _FakeLlama:
            def create_chat_completion(self, **kwargs):
                return {"choices": [{"message": {"content": "hello from llama_cpp"}}]}

        with patch.dict("os.environ", {"SYSAGENT_LOCAL_BACKEND": "llama_cpp"}), patch.object(
            mind, "retrieve_similar_examples", return_value=[]
        ), patch.object(
            mind, "_ensure_llama_server", side_effect=lambda g: calls.append("server") or None
        ), patch.object(
            mind, "_get_llama_cpp_model", side_effect=lambda g: calls.append("llama_cpp") or _FakeLlama()
        ):
            out = mind._run_single_gguf(Path("/tmp/fake.gguf"), "hi")
            self.assertEqual(out, "hello from llama_cpp")
            self.assertEqual(calls, ["llama_cpp"])

    def test_train_on_own_codebase_dedup_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            ds = p / "distilled.jsonl"
            wf = p / "weights.json"
            with patch.object(local_mind, "DISTILLED_FILE", ds), patch.object(
                local_mind, "WEIGHTS_FILE", wf
            ):
                mind = local_mind.LocalMind(storage=None)
                res1 = mind.train_on_own_codebase(force=False)
                self.assertEqual(res1["status"], "ok")
                self.assertGreater(res1["samples"], 10)
                lines_after_first = len(ds.read_text(encoding="utf-8").splitlines())

                res2 = mind.train_on_own_codebase(force=False)
                self.assertEqual(res2["status"], "up_to_date")

                res3 = mind.train_on_own_codebase(force=True)
                self.assertEqual(res3["status"], "ok")
                lines_after_force = len(ds.read_text(encoding="utf-8").splitlines())
                self.assertEqual(lines_after_first, lines_after_force)

    def test_local_mind_client_gguf_tool_calling_and_history(self) -> None:
        from sysagent.local_mind_client import LocalMindClient

        client = LocalMindClient(storage=None)
        tools = [
            {
                "name": "list_directory",
                "description": "List files in a directory.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ]
        with patch.object(client.interactions.mind, "predict_tool", return_value=(None, {}, 0.0)), patch.object(
            client.interactions.mind,
            "run_gguf_inference",
            return_value='{"tool": "list_directory", "args": {"path": "~/Downloads"}}',
        ):
            r1 = client.interactions.create(input="Downloads klasöründe ne var?", tools=tools)
            self.assertEqual(len(r1.steps), 1)
            self.assertEqual(r1.steps[0].name, "list_directory")
            self.assertEqual(r1.steps[0].arguments, {"path": "~/Downloads"})

        with patch.object(
            client.interactions.mind, "run_gguf_inference", return_value="Downloads klasöründe 2 dosya var."
        ):
            r2 = client.interactions.create(
                input=[{"name": "list_directory", "result": '{"entries": ["a.pdf", "b.png"]}'}],
                previous_interaction_id=r1.id,
            )
            self.assertIn("2 dosya", r2.output_text)
            self.assertGreaterEqual(len(client.interactions._history[r2.id]), 2)


if __name__ == "__main__":
    unittest.main()

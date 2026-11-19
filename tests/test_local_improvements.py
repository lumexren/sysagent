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
            fake_stat = SimpleNamespace(st_size=int(5.5 * (1024**3)))
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


if __name__ == "__main__":
    unittest.main()

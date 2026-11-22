"""Tests for the 6 self-resilience capabilities in sysagent.self_resilience."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sysagent.local_mind as lm
from sysagent.brain import CognitiveBrain
from sysagent.config import Settings
from sysagent.evolve import SelfEvolver
from sysagent.self_ops import select_relevant_tools
from sysagent.self_resilience import (
    CircuitBreakerRegistry,
    ContextBudgeter,
    QuotaGovernor,
    build_recovery_hint,
    evaluate_self_gate,
    get_circuit_breaker,
    get_quota_governor,
    record_boot_module_hashes,
    run_integrity_and_drift_audit,
    verify_written_file,
)
from sysagent.storage import Storage
from sysagent.tool_schemas import TOOL_DECLARATIONS
from sysagent.tools import execute_tool


class SelfResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        get_circuit_breaker().reset()
        get_quota_governor().reset()

    def tearDown(self) -> None:
        get_circuit_breaker().reset()
        get_quota_governor().reset()

    def test_circuit_breaker_lifecycle_and_tool_pruning(self) -> None:
        cb = CircuitBreakerRegistry(failure_threshold=3, cooldown_sec=60.0)
        self.assertTrue(cb.allow("tool:mcp_demo"))
        cb.record_failure("tool:mcp_demo", "timeout 1")
        cb.record_failure("tool:mcp_demo", "timeout 2")
        self.assertTrue(cb.allow("tool:mcp_demo"))
        state = cb.record_failure("tool:mcp_demo", "timeout 3")
        self.assertEqual(state, "open")
        self.assertFalse(cb.allow("tool:mcp_demo"))

        # Simulate cooldown expiry -> half_open probe -> success closes circuit.
        cb._entries["tool:mcp_demo"]["opened_until"] = 0.0
        self.assertTrue(cb.allow("tool:mcp_demo"))
        self.assertEqual(cb._entries["tool:mcp_demo"]["state"], "half_open")
        cb.record_success("tool:mcp_demo")
        self.assertEqual(cb._entries["tool:mcp_demo"]["state"], "closed")

        # Global breaker hides non-core tools from select_relevant_tools when open.
        g_cb = get_circuit_breaker()
        for _ in range(3):
            g_cb.record_failure("tool:mcp_broken", "err")
        all_tools = list(TOOL_DECLARATIONS) + [
            {"name": "mcp_broken", "description": "broken", "parameters": {"type": "object", "properties": {}}}
        ]
        pruned = select_relevant_tools("hello", all_tools, {d["name"] for d in TOOL_DECLARATIONS})
        self.assertNotIn("mcp_broken", {d["name"] for d in pruned})

    def test_self_eval_gate_and_regression_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "sr1.db") as st:
            tmp = Path(td)
            weights_file = tmp / "weights.json"
            distilled = tmp / "distilled.jsonl"
            with (
                patch.object(lm, "MODELS_DIR", tmp),
                patch.object(lm, "DATASET_DIR", tmp),
                patch.object(lm, "WEIGHTS_FILE", weights_file),
                patch.object(lm, "DISTILLED_FILE", distilled),
                patch.dict(lm._WEIGHTS_CACHE, {}, clear=True),
            ):
                mind = lm.LocalMind(st)
                good_res = evaluate_self_gate(mind, st)
                self.assertEqual(good_res["passed"], good_res["total"])
                self.assertFalse(good_res["regressed"])

                orig_weights = dict(mind.weights)
                # Force predict_tool regression during retrain and ensure previous weights are kept.
                with patch.object(lm.LocalMind, "predict_tool", return_value=("wrong_tool", {}, 0.99)):
                    mind.train()
                self.assertEqual(mind.weights["classes"], orig_weights["classes"])

    def test_post_write_verification_and_recovery_hint(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "sr2.db") as st:
            tmp = Path(td)
            py_ok = tmp / "ok.py"
            out_ok = json.loads(
                execute_tool("write_file", {"path": str(py_ok), "content": "x = 1 + 2\n"}, None, st)
            )
            self.assertEqual(out_ok["status"], "ok")
            self.assertTrue(out_ok["verification"]["ok"])

            py_bad = tmp / "bad.py"
            out_bad = json.loads(
                execute_tool("write_file", {"path": str(py_bad), "content": "def broken(:\n"}, None, st)
            )
            self.assertEqual(out_bad["status"], "ok")
            self.assertFalse(out_bad["verification"]["ok"])

            js_bad = tmp / "cfg.json"
            js_bad.write_text("{bad json", encoding="utf-8")
            self.assertFalse(verify_written_file(js_bad)["ok"])

            st.add_lesson(
                "pacman_lock",
                "pacman -Syu failed exit code 1 lock /var/lib/pacman/db.lck",
                "Remove stale db.lck if no pacman process is running.",
            )
            hint = build_recovery_hint(st, "run_command", {"command": "sudo pacman -Syu"}, '{"returncode": 1}')
            self.assertIsNotNone(hint)
            assert hint is not None
            self.assertIn("Self-Recovery Hint", hint)

    def test_context_budgeter_history_and_tool_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "sr3.db") as st:
            hist = [
                {"role": "user", "content": "inspect sysagent/server.py " + ("a" * 350)},
                {"role": "assistant", "content": "checked sysagent/cli.py " + ("b" * 350)},
                {"role": "user", "content": "fix sysagent/agent.py " + ("c" * 350)},
                {"role": "assistant", "content": "recent reply 1"},
                {"role": "user", "content": "recent query 2"},
            ]
            compacted = ContextBudgeter.compact_history(hist, max_chars=400, keep_recent=2)
            self.assertTrue(compacted[0]["content"].startswith("[Working Summary]"))
            self.assertIn("sysagent/server.py", compacted[0]["content"])
            self.assertEqual(compacted[-1]["content"], "recent query 2")

            big_text = "HEAD_" + ("X" * 500) + "_TAIL"
            budgeted = ContextBudgeter.budget_tool_output(st, "read_file", big_text, max_chars=220)
            self.assertIn("_internal:tool_out:read_file:", budgeted)
            ref_key = "_internal:tool_out:read_file:" + budgeted.split("_internal:tool_out:read_file:", 1)[1].split("]", 1)[0]
            self.assertEqual(st.get_memory(ref_key), big_text)

    def test_quota_governor_and_background_deferral(self) -> None:
        gov = QuotaGovernor(default_hourly_bg_cap=2)
        self.assertTrue(gov.allow_background("brain")[0])
        gov.record_background_call("brain")
        gov.record_background_call("evolve")
        allowed, reason = gov.allow_background("distill")
        self.assertFalse(allowed)
        self.assertEqual(reason, "hourly_bg_quota_exceeded")

        gov.reset()
        gov.record_rate_limit("gemini-4-neon", cooldown_sec=60.0)
        allowed2, reason2 = gov.allow_background("brain")
        self.assertFalse(allowed2)
        self.assertEqual(reason2, "rate_limit_cooldown")

        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "sr4.db") as st:
            s = Settings()
            get_quota_governor().record_rate_limit("gemini-4-neon", cooldown_sec=60.0)
            br = CognitiveBrain(s, SimpleNamespace(), st, SimpleNamespace())
            ev = SelfEvolver(s, SimpleNamespace(), st, SimpleNamespace())
            res_b = asyncio.run(br.think_once(force=False))
            res_e = asyncio.run(ev.evolve_once(force=False))
            self.assertEqual(res_b["status"], "deferred")
            self.assertEqual(res_e["status"], "deferred")

    def test_integrity_and_drift_auditor(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "sr5.db") as st:
            pkg = Path(td) / "pkg"
            pkg.mkdir()
            mod_a = pkg / "a.py"
            mod_a.write_text("x = 1\n", encoding="utf-8")
            record_boot_module_hashes(pkg)

            clean = run_integrity_and_drift_audit(st, pkg_dir=pkg, repo_dir=Path(td))
            self.assertTrue(clean["ok"])
            self.assertTrue(clean["db_ok"])
            self.assertEqual(clean["stale_modules"], [])

            mod_a.write_text("x = 2\n", encoding="utf-8")
            stale = run_integrity_and_drift_audit(st, pkg_dir=pkg, repo_dir=Path(td))
            self.assertEqual(stale["stale_modules"], ["a.py"])
            self.assertTrue(stale["ok"])

            mod_a.write_text("def broken(:\n", encoding="utf-8")
            broken = run_integrity_and_drift_audit(st, pkg_dir=pkg, repo_dir=Path(td))
            self.assertFalse(broken["ok"])
            self.assertEqual(broken["syntax_errors"][0]["file"], "a.py")


if __name__ == "__main__":
    unittest.main()

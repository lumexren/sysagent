"""Tests for self-introspection routing, component inspection, and LocalMind self-awareness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sysagent.local_mind as lm
from sysagent.local_mind_client import LocalMindClient, _LocalMindInteractions, detect_self_introspection
from sysagent.storage import Storage
from sysagent.tool_schemas import TOOL_DECLARATIONS
from sysagent.tools import execute_tool


class SelfIntrospectionTests(unittest.TestCase):
    def test_detect_self_introspection_mapping_and_non_false_triggers(self) -> None:
        self.assertEqual(detect_self_introspection("şuan ne düşünüyorsun?"), {"component": "brain"})
        self.assertEqual(detect_self_introspection("Brain, Cortex vs ne durumda"), {"component": "all"})
        self.assertEqual(detect_self_introspection("cortex flow durumu nasıl"), {"component": "cortex"})
        self.assertEqual(detect_self_introspection("self guard drift ve circuit breaker durumu"), {"component": "guard"})
        self.assertIsNone(detect_self_introspection("local mind modele geç"))
        self.assertIsNone(detect_self_introspection("nasılsınn"))
        self.assertIsNone(detect_self_introspection("merhaba"))
        self.assertIsNone(detect_self_introspection("şuan hangi modelsin adını söyle"))

    def test_tool_catalog_ranking_includes_inspect_self_component(self) -> None:
        catalog = _LocalMindInteractions._format_tool_catalog(TOOL_DECLARATIONS, "Brain, Cortex ne durumda")
        self.assertIn("inspect_self_component", catalog)

    def test_local_mind_client_emits_inspect_self_component_steps(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "si1.db") as st:
            tmp = Path(td)
            with (
                patch.object(lm, "MODELS_DIR", tmp),
                patch.object(lm, "DATASET_DIR", tmp),
                patch.object(lm, "WEIGHTS_FILE", tmp / "weights.json"),
                patch.object(lm, "DISTILLED_FILE", tmp / "distilled.jsonl"),
                patch.dict(lm._WEIGHTS_CACHE, {}, clear=True),
            ):
                client = LocalMindClient(st)
                res_all = client.interactions.create(input="Brain, Cortex vs ne durumda", tools=TOOL_DECLARATIONS)
                self.assertTrue(res_all.steps)
                self.assertEqual(res_all.steps[0].name, "inspect_self_component")
                self.assertEqual(res_all.steps[0].arguments, {"component": "all"})

                res_brain = client.interactions.create(input="şuan ne düşünüyorsun?", tools=TOOL_DECLARATIONS)
                self.assertTrue(res_brain.steps)
                self.assertEqual(res_brain.steps[0].name, "inspect_self_component")
                self.assertEqual(res_brain.steps[0].arguments, {"component": "brain"})

    def test_inspect_self_component_all_cortex_guard_and_runtime_refs(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "si2.db") as st:
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

    def test_try_local_cascade_does_not_bypass_self_introspection(self) -> None:
        with tempfile.TemporaryDirectory() as td, Storage(db_path=Path(td) / "si3.db") as st:
            tmp = Path(td)
            fake_gguf = tmp / "chat.gguf"
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
                self.assertIsNone(mind.try_local_cascade("Brain, Cortex vs ne durumda"))


if __name__ == "__main__":
    unittest.main()

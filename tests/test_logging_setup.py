"""Unit tests for centralized SysAgent logging setup."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from sysagent.logging_setup import configure_logging, get_logger


class TestLoggingSetup(unittest.TestCase):
    def test_get_logger_namespaces(self):
        self.assertEqual(get_logger("safety").name, "sysagent.safety")
        self.assertEqual(get_logger("sysagent.brain").name, "sysagent.brain")
        self.assertEqual(get_logger("sysagent").name, "sysagent")

    def test_configure_logging_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "logs" / "test.log"
            root = configure_logging(level="DEBUG", log_file=log_path, force=True)
            self.assertEqual(root.level, logging.DEBUG)
            child = get_logger("test_child")
            child.info("hello sysagent logging")
            for h in root.handlers:
                h.flush()
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("hello sysagent logging", text)
            self.assertIn("sysagent.test_child", text)


if __name__ == "__main__":
    unittest.main()

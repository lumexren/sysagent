"""Unit tests for safety policies and self-evolution blast-radius guards."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from sysagent.evolve_safety import PROTECTED_CORE_MODULES, check_blast_radius
from sysagent.safety import (
    assess_tool_call,
    is_auto_trust_allowed,
    neural_risk_note,
)


class TestSafetyShellPolicies(unittest.TestCase):
    def test_env_read_only_vs_exec(self):
        self.assertFalse(assess_tool_call("run_command", {"command": "env"}).requires_approval)
        self.assertTrue(
            assess_tool_call("run_command", {"command": "env python3 -c 'print(1)'"}).requires_approval
        )
        self.assertTrue(
            assess_tool_call("run_command", {"command": "env FOO=1 bash"}).requires_approval
        )

    def test_find_read_only_vs_dangerous_flags(self):
        self.assertFalse(
            assess_tool_call("run_command", {"command": "find . -name '*.py' -maxdepth 2"}).requires_approval
        )
        self.assertTrue(
            assess_tool_call("run_command", {"command": "find /tmp -name '*.tmp' -delete"}).requires_approval
        )
        self.assertTrue(
            assess_tool_call("run_command", {"command": "find . -exec python3 {} +"}).requires_approval
        )
        self.assertTrue(
            assess_tool_call("run_command", {"command": "find . -execdir sh -c id +"}).requires_approval
        )

    def test_git_allowlist_and_mutators(self):
        for safe_cmd in [
            "git status",
            "git -C /tmp/repo status -s",
            "git log -n 5 --oneline",
            "git diff HEAD~1",
            "git show HEAD",
            "git branch --list",
            "git tag -l",
            "git remote -v",
            "git rev-parse HEAD",
        ]:
            with self.subTest(safe_cmd=safe_cmd):
                self.assertFalse(assess_tool_call("run_command", {"command": safe_cmd}).requires_approval)

        for risky_cmd in [
            "git config --global user.name attacker",
            "git -c core.sshCommand=evil fetch",
            "git branch -D main",
            "git branch new-branch",
            "git tag v1.0",
            "git tag -d v1.0",
            "git remote add origin https://example.com/repo.git",
            "git stash drop",
            "git push origin main",
            "git reflog expire --all",
        ]:
            with self.subTest(risky_cmd=risky_cmd):
                self.assertTrue(assess_tool_call("run_command", {"command": risky_cmd}).requires_approval)

    def test_systemctl_allowlist_and_mutators(self):
        for safe_cmd in [
            "systemctl --user status sysagent",
            "systemctl is-active sshd",
            "systemctl --user --no-pager list-units --type=service",
            "systemctl list-timers --all",
            "systemctl cat sysagent.service",
        ]:
            with self.subTest(safe_cmd=safe_cmd):
                self.assertFalse(assess_tool_call("run_command", {"command": safe_cmd}).requires_approval)

        for risky_cmd in [
            "systemctl --user restart sysagent",
            "systemctl stop nginx",
            "systemctl daemon-reload",
            "systemctl set-environment FOO=bar",
            "systemctl edit sysagent",
        ]:
            with self.subTest(risky_cmd=risky_cmd):
                self.assertTrue(assess_tool_call("run_command", {"command": risky_cmd}).requires_approval)


class TestAutoTrustAndEvolveSafety(unittest.TestCase):
    def test_never_auto_trust_mutating_tools(self):
        for tool in ("forge_skill", "write_file", "edit_file", "trash_file"):
            self.assertFalse(is_auto_trust_allowed(tool, {"path": "/tmp/a"}))
        self.assertTrue(is_auto_trust_allowed("run_command", {"command": "pytest tests"}))
        self.assertFalse(is_auto_trust_allowed("run_command", {"command": "python3 x.py"}))

    def test_protected_core_modules_coverage(self):
        for mod in ("safety.py", "evolve_safety.py", "evolve_patch.py", "auth.py", "config.py", "tools.py"):
            self.assertIn(mod, PROTECTED_CORE_MODULES)

    def test_blast_radius_rejects_large_core_deletion(self):
        orig = "\n".join(f"line_{i}" for i in range(200))
        shrunk = "\n".join(f"line_{i}" for i in range(100))
        ok, reason = check_blast_radius("evolve_safety.py", orig, shrunk)
        self.assertFalse(ok)
        self.assertIn("Blast-radius guard", reason)

    def test_neural_risk_note_uses_injected_engine(self):
        engine = MagicMock()
        engine.spreading_activation.return_value = [
            {"key": "immunity_rm", "activation": 0.82, "text": "Previous rollback on rm"}
        ]
        note = neural_risk_note("run_command", {"command": "rm x"}, storage=None, neural_engine=engine)
        self.assertIn("Neural warning (0.82)", note)
        engine.spreading_activation.assert_called_once()


if __name__ == "__main__":
    unittest.main()

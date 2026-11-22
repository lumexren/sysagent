"""Smart approval classifier for system tool calls."""

from __future__ import annotations

import re
import shlex
import weakref
from dataclasses import dataclass
from typing import Any

from .logging_setup import get_logger

logger = get_logger(__name__)


SAFE_TOOLS = frozenset(
    {
        "read_file",
        "list_directory",
        "search_files",
        "system_status",
        "list_processes",
        "list_systemd_units",
        "list_events",
        "send_notification",
        "remember_fact",
        "recall_memories",
        "forget_memory",
        "create_automation",
        "list_automations",
        "delete_automation",
        "read_clipboard",
        "take_screenshot",
        "read_journal_logs",
        "list_audit_log",
        "undo_last_action",
        "undo_batch_actions",
        "search_memory_and_history",
        "desktop_control",
        "start_focus_timer",
        "get_app_usage",
        "reload_plugins",
        "kg_add_relation",
        "kg_query",
        "run_parallel_workers",
        "switch_model",
        "record_lesson",
        "create_goal_plan",
        "update_goal_step",
        "web_search",
        "fetch_url",
        "inspect_self_component",
    }
)

SAFE_SHELL_COMMANDS = frozenset(
    {
        "ls",
        "ll",
        "la",
        "pwd",
        "whoami",
        "id",
        "uname",
        "hostname",
        "uptime",
        "date",
        "df",
        "du",
        "free",
        "top",
        "htop",
        "ps",
        "pgrep",
        "ss",
        "netstat",
        "ip",
        "ifconfig",
        "ping",
        "dig",
        "nslookup",
        "cat",
        "head",
        "tail",
        "less",
        "wc",
        "grep",
        "rg",
        "fd",
        "which",
        "whereis",
        "file",
        "stat",
        "printenv",
        "echo",
        "journalctl",
        "dmesg",
        "lscpu",
        "lsblk",
        "lsusb",
        "lspci",
        "sensors",
    }
)

_FIND_DANGEROUS_FLAGS = frozenset(
    {
        "-exec",
        "-execdir",
        "-ok",
        "-okdir",
        "-delete",
        "-fprint",
        "-fprint0",
        "-fprintf",
        "-fls",
    }
)

_SAFE_GIT_SUBCOMMANDS = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "rev-parse",
        "describe",
        "ls-files",
        "ls-tree",
        "blame",
        "shortlog",
        "reflog",
        "cat-file",
        "branch",
        "tag",
        "remote",
    }
)

_GIT_DANGEROUS_GLOBAL_FLAGS = frozenset(
    {"-c", "--config-env", "--exec-path", "--upload-pack", "--receive-pack"}
)

_GIT_MUTATING_BRANCH_TAG_FLAGS = frozenset(
    {
        "-d",
        "-D",
        "-m",
        "-M",
        "-c",
        "-C",
        "-f",
        "--force",
        "--delete",
        "--move",
        "--copy",
        "--edit-description",
        "--set-upstream-to",
        "--unset-upstream",
        "-a",
        "-s",
        "-u",
        "--annotate",
        "--sign",
    }
)

_SAFE_SYSTEMCTL_SUBCOMMANDS = frozenset(
    {
        "status",
        "is-active",
        "is-enabled",
        "is-failed",
        "is-system-running",
        "list-units",
        "list-unit-files",
        "list-timers",
        "list-sockets",
        "list-dependencies",
        "list-jobs",
        "show",
        "cat",
    }
)

_SYSTEMCTL_FLAGS_WITH_VALUE = frozenset({"-t", "--type", "--state", "-n", "--lines", "-p", "--property", "-H", "--host", "-M", "--machine"})


def _is_safe_conditional_command(base_cmd: str, tokens: list[str]) -> bool:
    """Validate read-only usage of shell commands that also have mutating subcommands/flags."""
    args = tokens[1:]
    if base_cmd == "env":
        # Safe only when printing environment variables (no command execution).
        return len(args) == 0

    if base_cmd == "find":
        return not any(tok in _FIND_DANGEROUS_FLAGS for tok in args)

    if base_cmd == "git":
        idx = 0
        while idx < len(args) and args[idx].startswith("-"):
            flag = args[idx]
            flag_name = flag.split("=", 1)[0]
            if flag_name in _GIT_DANGEROUS_GLOBAL_FLAGS:
                return False
            if flag in {"-C", "--git-dir", "--work-tree"}:
                idx += 2
            else:
                idx += 1
        if idx >= len(args):
            return False
        sub = args[idx]
        if sub not in _SAFE_GIT_SUBCOMMANDS:
            return False
        sub_args = args[idx + 1 :]
        if sub in {"branch", "tag"}:
            # Allow only listing/read-only flags; reject mutating flags or creating a new branch/tag.
            for a in sub_args:
                if a.split("=", 1)[0] in _GIT_MUTATING_BRANCH_TAG_FLAGS:
                    return False
                if not a.startswith("-"):
                    return False
        elif sub == "remote":
            # Only `git remote` or `git remote -v` / `--verbose`
            if any(a not in {"-v", "--verbose"} for a in sub_args):
                return False
        elif sub == "reflog" and any(a in {"expire", "delete"} for a in sub_args):
            return False
        return True

    if base_cmd == "systemctl":
        idx = 0
        while idx < len(args) and args[idx].startswith("-"):
            flag = args[idx]
            if flag in _SYSTEMCTL_FLAGS_WITH_VALUE:
                idx += 2
            else:
                idx += 1
        if idx >= len(args):
            return True  # bare `systemctl` / `systemctl --user` lists units
        return args[idx] in _SAFE_SYSTEMCTL_SUBCOMMANDS

    return False

RISKY_PATTERNS = [
    r"[;&|`$><\r\n]",  # Shell chaining, newlines, pipes to mutators, redirection, command substitution
    r"\bsudo\b",
    r"\brm\b",
    r"\bmv\b",
    r"\bcp\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"\bkill\b",
    r"\bpkill\b",
    r"\bkillall\b",
    r"\breboot\b",
    r"\bshutdown\b",
    r"\bpoweroff\b",
    r"\bdd\b",
    r"\bmkfs\b",
    r"\bapt\b",
    r"\bpacman\b",
    r"\bdnf\b",
    r"\byum\b",
    r"\bpip\b",
    r"\bnpm\b",
    r"\bcurl\b",
    r"\bwget\b",
    r"\bsystemctl\s+(start|stop|restart|enable|disable|mask|unmask|daemon-reload)",
    r"\bgit\s+(push|commit|reset|checkout|rebase|merge|clean|rm|add)",
]


@dataclass
class SafetyAssessment:
    """Result of evaluating whether a tool call needs user approval."""

    requires_approval: bool
    reason: str
    risk_level: str  # "safe" | "moderate" | "high"


def assess_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    mode: str = "smart",
    plugin_safe: bool | None = None,
) -> SafetyAssessment:
    """Determine whether a tool call requires user confirmation."""
    if mode == "autonomous":
        return SafetyAssessment(False, "Autonomous mode enabled", "safe")

    if mode == "always_ask":
        return SafetyAssessment(True, "Always-ask mode enabled", "moderate")

    if plugin_safe is True:
        return SafetyAssessment(False, f"Trusted plugin tool: {tool_name}", "safe")
    if plugin_safe is False:
        return SafetyAssessment(True, f"Plugin tool requires approval: {tool_name}", "moderate")

    if tool_name == "desktop_control" and str(arguments.get("action")) == "launch_app":
        return SafetyAssessment(True, f"Launch application: {arguments.get('value', '')}", "moderate")

    if tool_name in SAFE_TOOLS:
        return SafetyAssessment(False, "Read-only / non-destructive tool", "safe")

    if tool_name in {"write_file", "edit_file"}:
        path = str(arguments.get("path", ""))
        return SafetyAssessment(
            True, f"File write/modify operation: {path}", "moderate"
        )

    if tool_name == "trash_file":
        return SafetyAssessment(True, f"Move to trash: {arguments.get('path', '')}", "moderate")

    if tool_name == "forge_skill":
        tname = str(arguments.get("tool_name", ""))
        return SafetyAssessment(
            True, f"Create and load custom Python plugin tool: {tname}", "moderate"
        )

    if tool_name == "run_command":
        cmd = str(arguments.get("command", "")).strip()
        if not cmd:
            return SafetyAssessment(True, "Empty command", "moderate")

        for pattern in RISKY_PATTERNS:
            if re.search(pattern, cmd):
                return SafetyAssessment(
                    True,
                    f"Potentially destructive or risky command: `{cmd}`",
                    "high",
                )

        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return SafetyAssessment(True, f"Could not parse command: `{cmd}`", "high")

        if not tokens:
            return SafetyAssessment(True, "Empty command", "moderate")

        base_cmd = tokens[0].split("/")[-1]
        if base_cmd in SAFE_SHELL_COMMANDS or _is_safe_conditional_command(base_cmd, tokens):
            return SafetyAssessment(
                False, f"Read-only system command: `{base_cmd}`", "safe"
            )

        return SafetyAssessment(
            True, f"Command not on allowlist: `{cmd}`", "moderate"
        )

    return SafetyAssessment(True, f"Unknown tool: {tool_name}", "high")


def compute_dry_run_preview(tool_name: str, arguments: dict[str, Any]) -> str:
    """Compute a lightweight impact preview (dry-run) for an approval-gated tool call."""
    from pathlib import Path
    import glob

    if tool_name == "edit_file":
        p = Path(str(arguments.get("path", ""))).expanduser()
        old_lines = len(str(arguments.get("target", "")).splitlines())
        new_lines = len(str(arguments.get("replacement", "")).splitlines())
        return f"Edit {p} ({old_lines} -> {new_lines} lines in target block)"

    if tool_name == "write_file":
        p = Path(str(arguments.get("path", ""))).expanduser()
        new_lines = len(str(arguments.get("content", "")).splitlines())
        if p.exists() and p.is_file():
            old_lines = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
            return f"Overwrite existing file {p} ({old_lines} -> {new_lines} lines)"
        return f"Create new file {p} ({new_lines} lines)"

    if tool_name == "forge_skill":
        tname = str(arguments.get("tool_name", ""))
        code = str(arguments.get("python_code", ""))
        try:
            compile(code, f"{tname}.py", "exec")
            syntax = "syntax OK"
        except SyntaxError as exc:
            syntax = f"syntax error: {exc}"
        return f"Write plugin {tname}.py ({syntax}, {len(code.splitlines())} lines)"

    if tool_name == "run_command":
        cmd = str(arguments.get("command", "")).strip()
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()
        matched_paths: list[str] = []
        for tok in tokens[1:]:
            if tok.startswith("-"):
                continue
            expanded = str(Path(tok).expanduser())
            hits = glob.glob(expanded)
            if hits:
                matched_paths.extend(hits[:5])
            elif Path(expanded).exists():
                matched_paths.append(expanded)
        if matched_paths:
            return f"Targets found ({len(matched_paths)}+): {', '.join(matched_paths[:5])}"
        return f"Shell command impact: `{cmd[:120]}`"

    return ""


def compute_approval_diff(
    tool_name: str, arguments: dict[str, Any], max_lines: int = 200
) -> str:
    """Compute a unified diff string for file/plugin-mutating approval prompts."""
    import difflib
    from pathlib import Path
    from .config import DATA_DIR

    orig = ""
    new_txt = ""
    label = "file"

    if tool_name == "edit_file":
        p = Path(str(arguments.get("path", ""))).expanduser()
        label = p.name or "file"
        target = str(arguments.get("target", ""))
        replacement = str(arguments.get("replacement", ""))
        if p.is_file():
            orig_file = p.read_text(encoding="utf-8", errors="replace")
            if target and target in orig_file:
                orig = orig_file
                new_txt = orig_file.replace(target, replacement)
            else:
                orig = target
                new_txt = replacement
        else:
            orig = target
            new_txt = replacement
    elif tool_name == "write_file":
        p = Path(str(arguments.get("path", ""))).expanduser()
        label = p.name or "file"
        orig = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
        new_txt = str(arguments.get("content", ""))
    elif tool_name == "forge_skill":
        tname = str(arguments.get("tool_name", "")).strip()
        label = f"{tname or 'skill'}.py"
        p = DATA_DIR / "plugins" / label
        orig = p.read_text(encoding="utf-8", errors="replace") if (tname and p.is_file()) else ""
        new_txt = str(arguments.get("python_code", ""))
    else:
        return ""

    if orig == new_txt:
        return ""

    diff_lines = list(
        difflib.unified_diff(
            orig.splitlines(),
            new_txt.splitlines(),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            n=3,
            lineterm="",
        )
    )
    if not diff_lines:
        return ""
    if len(diff_lines) > max_lines:
        extra = len(diff_lines) - max_lines
        diff_lines = [*diff_lines[:max_lines], f"... (+{extra} more diff lines)"]
    return "\n".join(diff_lines)


# Commands that can run arbitrary code or destroy data: approving them 3 times must never silence the prompt.
NEVER_AUTO_TRUST = frozenset(
    {
        "rm", "rmdir", "dd", "mkfs", "shred", "sudo", "su", "doas", "chmod", "chown", "kill", "pkill",
        "killall", "bash", "sh", "zsh", "fish", "python", "python3", "perl", "ruby", "node", "eval",
        "exec", "curl", "wget", "ssh", "scp", "rsync", "systemctl", "mv", "cp", "xargs", "env", "nohup",
        "pip", "pip3", "npm", "apt", "dnf", "pacman", "yay", "flatpak", "snap",
    }
)


NEVER_AUTO_TRUST_TOOLS = frozenset(
    {"forge_skill", "write_file", "edit_file", "trash_file"}
)

_NEURAL_ENGINE_CACHE: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def _get_cached_neural_engine(storage: Any) -> Any:
    """Return a cached NeuralEngine per storage instance to avoid repeated cold loads."""
    try:
        cached = _NEURAL_ENGINE_CACHE.get(storage)
        if cached is not None:
            return cached
        from .neural import NeuralEngine

        engine = NeuralEngine(storage)
        _NEURAL_ENGINE_CACHE[storage] = engine
        return engine
    except TypeError:
        from .neural import NeuralEngine

        return NeuralEngine(storage)


def neural_risk_note(
    tool_name: str,
    arguments: dict[str, Any],
    storage: Any,
    neural_engine: Any = None,
) -> str:
    """Query NeuralEngine associative memory for past failure/immunity warnings related to this tool call."""
    if storage is None and neural_engine is None:
        return ""
    try:
        import json

        engine = neural_engine if neural_engine is not None else _get_cached_neural_engine(storage)
        q = f"{tool_name} {json.dumps(arguments, ensure_ascii=False)[:180]}"
        hits = engine.spreading_activation(q, top_k=2)
        for h in hits:
            key = str(h.get("key") or "").lower()
            act = float(h.get("activation", 0.0) or 0.0)
            if act >= 0.55 and any(k in key for k in ("immunity", "failed", "rollback", "regressed")):
                return f"Neural warning ({act:.2f}): {str(h.get('text') or '')[:140]}"
    except Exception as exc:
        logger.debug("neural_risk_note lookup failed for %s: %s", tool_name, exc)
    return ""


def is_auto_trust_allowed(tool_name: str, arguments: dict[str, Any]) -> bool:
    """Whether repeated approvals may turn this call into an auto-trusted (no-prompt) pattern."""
    if tool_name in NEVER_AUTO_TRUST_TOOLS:
        return False
    if tool_name != "run_command":
        return True
    first = str(arguments.get("command", "")).strip().split(None, 1)
    return bool(first) and first[0].split("/")[-1] not in NEVER_AUTO_TRUST

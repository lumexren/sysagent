"""Runtime self-resilience: Circuit Breaker, Eval Gate, Recovery Hints, Context Budgeter, Quota Governor, and Drift Auditor."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import tomllib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .storage import Storage

_CB_LOCK = threading.RLock()
_QUOTA_LOCK = threading.RLock()
_BOOT_MODULE_HASHES: dict[str, str] = {}
_PACKAGE_DIR = Path(__file__).resolve().parent


class CircuitBreakerRegistry:
    """Thread-safe circuit breaker (closed -> open -> half_open) for tools, plugins, and models."""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_sec: float = 300.0,
        probe_timeout_sec: float = 120.0,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_sec = max(1.0, float(cooldown_sec))
        self.probe_timeout_sec = max(1.0, float(probe_timeout_sec))
        self._lock = threading.RLock()
        self._entries: dict[str, dict[str, Any]] = {}

    def allow(self, key: str) -> bool:
        """Return True if calls to `key` are allowed (closed or ready for half-open probe)."""
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if not entry or entry["state"] == "closed":
                return True
            if entry["state"] == "open":
                if now >= float(entry["opened_until"]):
                    entry["state"] = "half_open"
                    entry["probe_in_flight"] = True
                    entry["probe_started_at"] = now
                    return True
                return False
            if entry["state"] == "half_open" and (
                not entry.get("probe_in_flight")
                or (now - float(entry.get("probe_started_at") or 0.0) >= self.probe_timeout_sec)
            ):
                entry["probe_in_flight"] = True
                entry["probe_started_at"] = now
                return True
            return False

    def release_probe(self, key: str) -> None:
        """Release an in-flight half-open probe lock when a call aborts without a breaker verdict."""
        with self._lock:
            entry = self._entries.get(key)
            if entry and entry.get("state") == "half_open":
                entry["probe_in_flight"] = False

    def is_open(self, key: str) -> bool:
        """Return True when `key` is actively cooling down in the open state."""
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if not entry or entry["state"] != "open":
                return False
            return now < float(entry["opened_until"])

    def record_success(self, key: str) -> None:
        """Reset failure count and close circuit for `key`."""
        with self._lock:
            if key in self._entries:
                self._entries[key] = {
                    "state": "closed",
                    "failures": 0,
                    "opened_until": 0.0,
                    "probe_in_flight": False,
                    "last_error": "",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }

    def record_failure(self, key: str, reason: str = "") -> str:
        """Increment failure count for `key` and open the circuit if threshold is reached."""
        now = time.monotonic()
        with self._lock:
            entry = self._entries.setdefault(
                key,
                {
                    "state": "closed",
                    "failures": 0,
                    "opened_until": 0.0,
                    "probe_in_flight": False,
                    "last_error": "",
                    "updated_at": "",
                },
            )
            entry["failures"] = int(entry.get("failures", 0)) + 1
            entry["last_error"] = str(reason or "")[:160]
            entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
            entry["probe_in_flight"] = False
            if entry["state"] == "half_open" or entry["failures"] >= self.failure_threshold:
                entry["state"] = "open"
                entry["opened_until"] = now + self.cooldown_sec
            return str(entry["state"])

    def reset(self, key: str | None = None) -> None:
        """Clear circuit breaker state for one key or all keys."""
        with self._lock:
            if key is None:
                self._entries.clear()
            else:
                self._entries.pop(key, None)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-serializable view of active circuit states."""
        now = time.monotonic()
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            for k, v in self._entries.items():
                rem = max(0.0, float(v.get("opened_until", 0.0)) - now) if v.get("state") == "open" else 0.0
                out[k] = {
                    "state": v["state"],
                    "failures": int(v.get("failures", 0)),
                    "cooldown_remaining_sec": round(rem, 1),
                    "last_error": v.get("last_error", ""),
                    "updated_at": v.get("updated_at", ""),
                }
        return out


_DEFAULT_BREAKERS = CircuitBreakerRegistry()


def get_circuit_breaker() -> CircuitBreakerRegistry:
    """Return the process-wide CircuitBreakerRegistry singleton."""
    return _DEFAULT_BREAKERS


GOLDEN_EVAL_CASES: list[dict[str, Any]] = [
    {"kind": "route", "prompt": "check system status cpu ram", "expected_tool": "system_status"},
    {"kind": "route", "prompt": "list running processes top cpu ram memory ps", "expected_tool": "list_processes"},
    {"kind": "route", "prompt": "list systemd units services failed active status", "expected_tool": "list_systemd_units"},
    {"kind": "route", "prompt": "read file content show text cat", "expected_tool": "read_file"},
    {"kind": "route", "prompt": "search files find by name or content", "expected_tool": "search_files"},
    {"kind": "route", "prompt": "check system journal logs service errors journalctl", "expected_tool": "read_journal_logs"},
    {"kind": "route", "prompt": "recall saved memories facts what do you remember", "expected_tool": "recall_memories"},
    {"kind": "route", "prompt": "what did brain do recently what rules exist", "expected_tool": "inspect_self_component"},
    {"kind": "route", "prompt": "brain cortex ne durumda kendi bileşenlerini incele", "expected_tool": "inspect_self_component"},
    {"kind": "route", "prompt": "promptlar ne durumda prompt versions status", "expected_tool": "inspect_self_component"},
    {"kind": "chat", "prompt": "naber"},
    {"kind": "chat", "prompt": "merhaba nasilsin"},
    {"kind": "chat", "prompt": "hello thank you"},
    {"kind": "safety", "tool": "run_command", "args": {"command": "rm -rf /tmp/x"}, "expected_approval": True},
    {"kind": "safety", "tool": "run_command", "args": {"command": "ls\npython3 -c 'print(1)'"}, "expected_approval": True},
    {"kind": "safety", "tool": "run_command", "args": {"command": "ls -la"}, "expected_approval": False},
    {"kind": "safety", "tool": "system_status", "args": {}, "expected_approval": False},
]


def evaluate_self_gate(mind: Any, storage: Storage | None = None) -> dict[str, Any]:
    """Run deterministic routing, chat-vs-tool, and safety regression probes on `mind`."""
    from .safety import assess_tool_call
    from .self_guard import get_calibration

    tool_thresh = float(get_calibration(storage).get("local_tool_threshold", 0.72))
    passed = 0
    failures: list[dict[str, Any]] = []

    for case in GOLDEN_EVAL_CASES:
        kind = case["kind"]
        if kind == "route":
            try:
                pred, _, score = mind.predict_tool(case["prompt"], record_uncertain=False)
            except TypeError:
                pred, _, score = mind.predict_tool(case["prompt"])
            ok = pred == case["expected_tool"] and float(score) > 0.15
            if ok:
                passed += 1
            else:
                failures.append({"kind": kind, "prompt": case["prompt"], "expected": case["expected_tool"], "got": pred})
        elif kind == "chat":
            try:
                pred, _, score = mind.predict_tool(case["prompt"], record_uncertain=False)
            except TypeError:
                pred, _, score = mind.predict_tool(case["prompt"])
            ok = pred is None or float(score) < tool_thresh
            if ok:
                passed += 1
            else:
                failures.append({"kind": kind, "prompt": case["prompt"], "got": pred, "score": score})
        elif kind == "safety":
            res = assess_tool_call(case["tool"], case["args"], mode="smart")
            ok = bool(res.requires_approval) == bool(case["expected_approval"])
            if ok:
                passed += 1
            else:
                failures.append({"kind": kind, "tool": case["tool"], "expected": case["expected_approval"], "got": res.requires_approval})

    total = len(GOLDEN_EVAL_CASES)
    score = round(passed / max(1, total), 4)
    prev_score: float | None = None
    regressed = False
    if storage is not None:
        try:
            raw = storage.get_memory("_internal:self_eval_baseline")
            if raw:
                prev_score = float(json.loads(raw).get("score"))
        except Exception:
            prev_score = None
        if prev_score is not None and score < (prev_score - 0.10):
            regressed = True
        if not regressed and score >= (prev_score or 0.0):
            storage.set_memory(
                "_internal:self_eval_baseline",
                json.dumps({"score": score, "passed": passed, "total": total, "updated_at": datetime.now().isoformat(timespec="seconds")}),
            )

    return {
        "score": score,
        "passed": passed,
        "total": total,
        "baseline": prev_score,
        "regressed": regressed,
        "failures": failures[:5],
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }


def verify_written_file(path: Path) -> dict[str, Any]:
    """Run fast syntax/schema verification on newly written `.py`, `.json`, `.toml`, `.yaml`, `.ini`/`.conf`/`.service`, or `.sh` files."""
    if not path.is_file():
        return {"ok": False, "kind": "missing", "error": f"File not found: {path}"}
    suffix = path.suffix.lower()
    src = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".py":
        try:
            compile(src, str(path), "exec")
            return {"ok": True, "kind": "python"}
        except SyntaxError as exc:
            return {"ok": False, "kind": "python", "error": f"SyntaxError line {exc.lineno}: {exc.msg}"}
    if suffix == ".json":
        try:
            json.loads(src)
            return {"ok": True, "kind": "json"}
        except json.JSONDecodeError as exc:
            return {"ok": False, "kind": "json", "error": f"JSONDecodeError line {exc.lineno}: {exc.msg}"}
    if suffix == ".toml":
        try:
            tomllib.loads(src)
            return {"ok": True, "kind": "toml"}
        except tomllib.TOMLDecodeError as exc:
            return {"ok": False, "kind": "toml", "error": str(exc)}
    if suffix in {".yaml", ".yml"}:
        for idx, raw_ln in enumerate(src.splitlines(), start=1):
            stripped = raw_ln.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "\t" in raw_ln[: len(raw_ln) - len(raw_ln.lstrip())]:
                return {"ok": False, "kind": "yaml", "error": f"YAML syntax error line {idx}: tab indentation not allowed"}
            if stripped.count('"') % 2 == 1 and not stripped.endswith("\\"):
                return {"ok": False, "kind": "yaml", "error": f"YAML syntax error line {idx}: unclosed double quote"}
        return {"ok": True, "kind": "yaml"}
    if suffix in {".sh", ".bash"} and shutil.which("bash"):
        try:
            proc = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if proc.returncode == 0:
                return {"ok": True, "kind": "shell"}
            err = (proc.stderr or proc.stdout or "").strip().splitlines()[0:1]
            return {"ok": False, "kind": "shell", "error": err[0] if err else f"rc={proc.returncode}"}
        except Exception:
            pass
    if suffix in {".ini", ".cfg", ".service", ".timer"}:
        import configparser

        cp = configparser.ConfigParser(interpolation=None, strict=False)
        try:
            cp.read_string(src)
        except configparser.Error as exc:
            return {"ok": False, "kind": suffix.lstrip("."), "error": str(exc).splitlines()[0]}
        if suffix in {".service", ".timer"}:
            secs = set(cp.sections())
            req_sec = "Service" if suffix == ".service" else "Timer"
            if req_sec not in secs and "Unit" not in secs:
                return {"ok": False, "kind": "systemd", "error": f"Missing [{req_sec}] or [Unit] section in {path.name}"}
            return {"ok": True, "kind": "systemd"}
        return {"ok": True, "kind": suffix.lstrip(".")}
    return {"ok": True, "kind": "skipped"}


_CRITICAL_PROTECTED_ROOTS = frozenset(
    {
        "/",
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/home",
        "/lib",
        "/lib64",
        "/proc",
        "/root",
        "/sbin",
        "/sys",
        "/usr",
        "/var",
    }
)


def analyze_command_preflight(command: str, cwd: str | Path | None = None) -> dict[str, Any]:
    """Perform a 0-token pre-flight blast-radius simulation on destructive or state-mutating shell commands."""
    import shlex

    cmd_str = (command or "").strip()
    base_cwd = Path(cwd).expanduser().resolve() if cwd else Path.home().resolve()
    home_dir = Path.home().resolve()
    if not cmd_str:
        return {"destructive": False, "blocked": False, "summary": "empty command"}

    try:
        tokens = shlex.split(cmd_str)
    except ValueError:
        tokens = cmd_str.split()

    low_cmd = cmd_str.lower()
    affected_files = 0
    dirty_git_files = 0
    targets: list[str] = []
    blocked = False
    block_reason = ""
    destructive = False
    kind = "normal"

    if any(
        pat in low_cmd
        for pat in ("rm ", "rm\t", "find ", "git reset --hard", "git clean", "chmod -r", "chown -r", "systemctl stop", "systemctl disable")
    ):
        destructive = True

    if tokens and tokens[0] in {"rm", "sudo"}:
        rm_idx = 0 if tokens[0] == "rm" else (1 if len(tokens) > 1 and tokens[1] == "rm" else -1)
        if rm_idx >= 0:
            destructive = True
            kind = "file_deletion"
            flags = [t for t in tokens[rm_idx + 1 :] if t.startswith("-")]
            recursive = any("r" in f.lower() or "--recursive" in f.lower() for f in flags)
            path_args = [t for t in tokens[rm_idx + 1 :] if not t.startswith("-")]
            for raw_p in path_args:
                if raw_p in {"/", "/*"}:
                    blocked = True
                    block_reason = f"Refusing catastrophic deletion target `{raw_p}`"
                    break
                p_obj = Path(raw_p).expanduser()
                resolved = p_obj.resolve() if p_obj.is_absolute() or raw_p.startswith("~") else (base_cwd / p_obj).resolve()
                targets.append(str(resolved))
                if str(resolved) in _CRITICAL_PROTECTED_ROOTS or (recursive and resolved == home_dir):
                    blocked = True
                    block_reason = f"Refusing recursive/root deletion on protected path `{resolved}`"
                    break
                if resolved.is_file():
                    affected_files += 1
                elif resolved.is_dir():
                    try:
                        affected_files += sum(
                            1 for _ in zip(range(500), resolved.rglob("*"), strict=False)
                        )
                    except Exception:
                        affected_files += 1

    if "git reset --hard" in low_cmd or "git clean" in low_cmd:
        destructive = True
        kind = "git_worktree_discard"
        if shutil.which("git") and base_cwd.is_dir():
            try:
                proc = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(base_cwd),
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                if proc.returncode == 0:
                    dirty_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
                    dirty_git_files = len(dirty_lines)
                    affected_files = max(affected_files, dirty_git_files)
            except Exception:
                pass

    summary_parts: list[str] = [f"kind={kind}"]
    if targets:
        summary_parts.append(f"targets={targets[:3]}")
    if affected_files:
        summary_parts.append(f"affected_files~={affected_files}")
    if dirty_git_files:
        summary_parts.append(f"dirty_git_files={dirty_git_files}")
    if blocked:
        summary_parts.append(f"BLOCKED({block_reason})")

    return {
        "destructive": destructive,
        "kind": kind,
        "blocked": blocked,
        "block_reason": block_reason,
        "targets": targets,
        "affected_files": affected_files,
        "dirty_git_files": dirty_git_files,
        "summary": " | ".join(summary_parts),
    }


def build_recovery_hint(
    storage: Storage | None, tool_name: str, arguments: dict[str, Any], raw_result: str
) -> str | None:
    """Look up cached lessons or knowledge entries to suggest a concise recovery hint after tool errors."""
    if storage is None:
        return None
    if tool_name == "run_command":
        from .self_guard import match_cached_fix

        cmd = str(arguments.get("command") or "")
        rc = 1
        try:
            parsed = json.loads(raw_result)
            if isinstance(parsed, dict):
                raw_rc = parsed.get("exit_code") if parsed.get("exit_code") is not None else parsed.get("returncode")
                if raw_rc is not None:
                    rc = int(raw_rc)
        except Exception:
            pass
        hit = match_cached_fix(storage, cmd, rc)
        if hit is not None:
            return f"[Self-Recovery Hint] ({hit['topic']}) {str(hit['lesson'])[:220]}"

    q_parts = [tool_name]
    for k in ("path", "unit", "command", "query"):
        if arguments.get(k):
            q_parts.append(str(arguments[k])[:60])
    try:
        hits = storage.search_knowledge(" ".join(q_parts), limit=5)
    except Exception:
        return None
    for h in hits:
        title = str(h.get("title") or "")
        if title.startswith("_internal:"):
            continue
        detail = str(h.get("detail") or "").strip()
        if detail and h.get("type") in {"lesson", "memory"}:
            return f"[Self-Recovery Hint] {title or tool_name}: {detail[:200]}"
    return None


class ContextBudgeter:
    """Token/character budgeter for session histories and oversized tool outputs."""

    _PATH_RE = re.compile(r"(?:[\w.~-]+/)*[\w.-]+\.(?:py|json|toml|md|sh|service|conf)")

    @classmethod
    def compact_history(
        cls, history: list[dict[str, str]], max_chars: int = 3600, keep_recent: int = 4
    ) -> list[dict[str, str]]:
        """Summarize older turns into a working-memory header while preserving recent turns."""
        if not history or len(history) <= keep_recent:
            return list(history)
        total = sum(len(str(t.get("content") or "")) for t in history)
        if total <= max_chars:
            return list(history)

        older = history[:-keep_recent]
        recent = list(history[-keep_recent:])
        paths: list[str] = []
        goals: list[str] = []
        for turn in older:
            content = " ".join(str(turn.get("content") or "").split())
            if not content:
                continue
            for p in cls._PATH_RE.findall(content):
                if p not in paths:
                    paths.append(p)
            if turn.get("role") == "user" and len(goals) < 3:
                goals.append(content[:90])

        summary_parts: list[str] = ["[Working Summary]"]
        if goals:
            summary_parts.append("Prior requests: " + " | ".join(goals))
        if paths:
            summary_parts.append("Files referenced: " + ", ".join(paths[:8]))
        summary_msg = {"role": "system", "content": " ; ".join(summary_parts)[:600]}

        compacted = [summary_msg, *recent]
        while len(compacted) > 3 and sum(len(str(t.get("content") or "")) for t in compacted) > max_chars:
            if len(compacted) >= 4 and compacted[1].get("role") == "user" and compacted[2].get("role") == "assistant":
                del compacted[1:3]
            else:
                compacted.pop(1)
        return compacted

    @staticmethod
    def budget_tool_output(
        storage: Storage | None,
        tool_name: str,
        text: str,
        max_chars: int = 4000,
        query: str = "",
    ) -> str:
        """Cache oversized tool outputs in SQLite and return a query-aware head+salient_middle+tail slice."""
        if len(text) <= max_chars:
            return text
        ref_id = ""
        if storage is not None and max_chars >= 200:
            ref_key = f"_internal:tool_out:{tool_name}:{uuid.uuid4().hex[:8]}"
            exp = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            try:
                storage.set_memory(ref_key, text[:60000], expires_at=exp)
                ref_id = ref_key
            except Exception:
                ref_id = ""

        q_toks = set(re.findall(r"\w{3,}", (query or "").lower()))
        if q_toks and max_chars >= 400:
            head_len = max(60, int(max_chars * 0.45))
            tail_len = max(40, int(max_chars * 0.20))
            mid_budget = max(80, int(max_chars * 0.28))
            mid_region = text[head_len : max(head_len, len(text) - tail_len)]
            if len(mid_region) > mid_budget:
                win_size = min(len(mid_region), mid_budget)
                step = max(60, win_size // 2)
                best_win = ""
                best_score = 0
                for start in range(0, max(1, len(mid_region) - win_size + 1), step):
                    chunk = mid_region[start : start + win_size]
                    c_toks = set(re.findall(r"\w{3,}", chunk.lower()))
                    overlap = len(q_toks & c_toks)
                    if overlap > best_score:
                        best_score = overlap
                        best_win = chunk
                if best_score > 0 and best_win:
                    cache_note = f"; cached as {ref_id}" if ref_id else ""
                    return (
                        text[:head_len]
                        + f"\n...[query-matched middle excerpt ({best_score} terms{cache_note})]...\n"
                        + best_win
                        + "\n...[compacted tail]...\n"
                        + text[-tail_len:]
                    )

        head = max(40, int(max_chars * 0.65))
        tail = max(20, int(max_chars * 0.25))
        marker = (
            f"\n...[compacted {len(text)} chars; full output cached as {ref_id}]...\n"
            if ref_id
            else f"\n...[truncated {len(text) - head - tail} chars]...\n"
        )
        return text[:head] + marker + text[-tail:]


class QuotaGovernor:
    """Rate-limit and hourly background quota governor prioritizing interactive user chat."""

    def __init__(self, default_hourly_bg_cap: int = 40) -> None:
        self.default_hourly_bg_cap = default_hourly_bg_cap
        self._lock = _QUOTA_LOCK
        self._cooldown_until: float = 0.0
        self._last_limited_model: str = ""
        self._bg_events: list[tuple[float, str]] = []

    def _hourly_cap(self) -> int:
        try:
            return max(1, int(os.environ.get("SYSAGENT_BG_QUOTA_PER_HOUR", str(self.default_hourly_bg_cap))))
        except ValueError:
            return self.default_hourly_bg_cap

    def record_rate_limit(self, model: str = "", cooldown_sec: float = 300.0) -> None:
        """Activate background cooldown after a 429/quota exhaustion signal."""
        now = time.monotonic()
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, now + max(10.0, float(cooldown_sec)))
            if model:
                self._last_limited_model = model

    def record_background_call(self, caller: str, model: str = "") -> None:
        """Record one background cloud call against the rolling 1-hour budget."""
        now = time.monotonic()
        cutoff = now - 3600.0
        with self._lock:
            self._bg_events = [(ts, c) for ts, c in self._bg_events if ts >= cutoff]
            self._bg_events.append((now, caller))

    def allow_background(self, caller: str = "") -> tuple[bool, str]:
        """Return `(allowed, reason)` indicating whether a background job may consume cloud quota."""
        now = time.monotonic()
        cutoff = now - 3600.0
        cap = self._hourly_cap()
        with self._lock:
            if now < self._cooldown_until:
                return False, "rate_limit_cooldown"
            self._bg_events = [(ts, c) for ts, c in self._bg_events if ts >= cutoff]
            if len(self._bg_events) >= cap:
                return False, "hourly_bg_quota_exceeded"
        return True, "ok"

    def reset(self) -> None:
        """Reset cooldown and background counters (mainly for unit tests)."""
        with self._lock:
            self._cooldown_until = 0.0
            self._last_limited_model = ""
            self._bg_events.clear()

    def get_status(self) -> dict[str, Any]:
        """Return current quota cooldown and rolling 1-hour background usage."""
        now = time.monotonic()
        cutoff = now - 3600.0
        with self._lock:
            self._bg_events = [(ts, c) for ts, c in self._bg_events if ts >= cutoff]
            by_caller: dict[str, int] = {}
            for _, c in self._bg_events:
                by_caller[c] = by_caller.get(c, 0) + 1
            rem = max(0.0, self._cooldown_until - now)
            return {
                "cooldown_remaining_sec": round(rem, 1),
                "last_limited_model": self._last_limited_model,
                "bg_calls_last_hour": len(self._bg_events),
                "hourly_bg_cap": self._hourly_cap(),
                "by_caller": by_caller,
            }


_DEFAULT_GOVERNOR = QuotaGovernor()


def get_quota_governor() -> QuotaGovernor:
    """Return the process-wide QuotaGovernor singleton."""
    return _DEFAULT_GOVERNOR


def record_boot_module_hashes(pkg_dir: Path | None = None) -> dict[str, str]:
    """Snapshot SHA-256 hashes of loaded Python modules at daemon startup for drift detection."""
    target = (pkg_dir or _PACKAGE_DIR).resolve()
    hashes: dict[str, str] = {}
    for pyf in sorted(target.glob("*.py")):
        try:
            hashes[str(pyf.resolve())] = hashlib.sha256(pyf.read_bytes()).hexdigest()[:16]
        except Exception:
            pass
    _BOOT_MODULE_HASHES.update(hashes)
    return dict(hashes)


def refresh_boot_module_hash(path: Path) -> None:
    """Update the in-memory module hash baseline after an intentional hot-reload."""
    if not _BOOT_MODULE_HASHES:
        return
    try:
        resolved = path.resolve()
        if resolved.is_file():
            _BOOT_MODULE_HASHES[str(resolved)] = hashlib.sha256(resolved.read_bytes()).hexdigest()[:16]
    except Exception:
        pass


def run_integrity_and_drift_audit(
    storage: Storage | None = None,
    pkg_dir: Path | None = None,
    repo_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify package syntax, stale daemon modules vs disk, SQLite quick_check + WAL size, and git drift."""
    target = (pkg_dir or _PACKAGE_DIR).resolve()
    root_dir = repo_dir or target.parent
    stale_modules: list[str] = []
    syntax_errors: list[dict[str, str]] = []

    for pyf in sorted(target.glob("*.py")):
        try:
            raw_bytes = pyf.read_bytes()
            cur_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
            boot_h = _BOOT_MODULE_HASHES.get(str(pyf.resolve()))
            if boot_h and cur_hash != boot_h:
                stale_modules.append(pyf.name)
            compile(raw_bytes.decode("utf-8", errors="replace"), str(pyf), "exec")
        except SyntaxError as exc:
            syntax_errors.append({"file": pyf.name, "error": f"line {exc.lineno}: {exc.msg}"})
        except Exception as exc:
            syntax_errors.append({"file": pyf.name, "error": str(exc)})

    db_ok = True
    wal_mb = 0.0
    if storage is not None:
        try:
            with storage._lock, storage._connect() as conn:
                row = conn.execute("PRAGMA quick_check").fetchone()
                db_ok = bool(row and str(row[0]).lower() == "ok")
            wal_file = Path(f"{storage.db_path}-wal")
            if wal_file.is_file():
                wal_mb = round(wal_file.stat().st_size / (1024 * 1024), 3)
        except Exception:
            db_ok = False

    dirty_files: list[str] = []
    if (root_dir / ".git").exists() and shutil.which("git"):
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain", "sysagent/"],
                cwd=root_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if proc.returncode == 0:
                dirty_files = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()][:20]
        except Exception:
            pass

    return {
        "ok": not syntax_errors and db_ok,
        "stale_modules": stale_modules,
        "syntax_errors": syntax_errors,
        "db_ok": db_ok,
        "wal_mb": wal_mb,
        "dirty_files": dirty_files,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }

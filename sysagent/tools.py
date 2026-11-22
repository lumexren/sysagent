"""System tools exposed to the Gemini Interactions API."""

from __future__ import annotations

import base64
import difflib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .plugins import PluginRegistry
from .storage import Storage
from .watcher import SystemWatcher


from .tool_schemas import TOOL_DECLARATIONS

def get_all_tool_declarations(plugins: PluginRegistry | None = None) -> list[dict[str, Any]]:
    if not plugins:
        return TOOL_DECLARATIONS
    return [*TOOL_DECLARATIONS, *plugins.declarations()]


def _backup_critical_config(path: Path, old_content: str) -> None:
    """Save a timestamped snapshot before editing system or user config files."""
    p_str = str(path)
    cfg_dir = str(Path.home() / ".config")
    is_cfg = (
        p_str.startswith(("/etc/", cfg_dir))
        or path.suffix.lower() in {".service", ".conf", ".env", ".toml", ".yaml", ".yml"}
    )
    if not is_cfg:
        return
    from .snapshots import save_config_snapshot

    save_config_snapshot(path, old_content)


def _resolve(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


_IMPORTANT_LOG_TOKENS = (
    "error",
    "exception",
    "traceback",
    "failed",
    "fatal",
    "critical",
    "panic",
    "denied",
    "warning",
    "sigsegv",
    "oom",
)


def compress_tool_output(text: str, max_chars: int = 4000) -> str:
    """Fold consecutive duplicate lines and preserve head, high-signal error lines, and tail within budget."""
    raw = text or ""
    if not raw:
        return ""
    lines = raw.splitlines()
    folded: list[str] = []
    idx = 0
    n = len(lines)
    while idx < n:
        cur = lines[idx]
        run_len = 1
        while idx + run_len < n and lines[idx + run_len].strip() == cur.strip() and cur.strip():
            run_len += 1
        if run_len >= 3:
            folded.append(f"{cur}  [repeated {run_len}x]")
        else:
            for k in range(run_len):
                folded.append(lines[idx + k])
        idx += run_len

    joined = "\n".join(folded)
    if len(joined) <= max_chars:
        return joined

    if len(folded) <= 36:
        half = max(200, (max_chars - 60) // 2)
        omitted_c = len(joined) - (half * 2)
        return f"{joined[:half]}\n... [{omitted_c} chars compressed] ...\n{joined[-half:]}"

    head = folded[:12]
    tail = folded[-18:]
    mid = folded[12:-18]
    important: list[str] = []
    seen_imp: set[str] = set()
    for ln in mid:
        low = ln.lower()
        if "[repeated " in low or any(tok in low for tok in _IMPORTANT_LOG_TOKENS):
            norm = ln.strip()[:160]
            if norm not in seen_imp:
                seen_imp.add(norm)
                important.append(ln[:260])
                if len(important) >= 18:
                    break

    omitted_lines = max(0, len(mid) - len(important))
    marker = f"... [{omitted_lines} routine lines compressed; {len(important)} high-signal lines retained] ..."
    parts = [*head, marker, *important, *tail]
    out = "\n".join(parts)
    if len(out) > max_chars:
        head_budget = min(1100, max_chars // 3)
        tail_budget = max(500, max_chars - head_budget - 80)
        out = f"{out[:head_budget]}\n... [compressed to fit {max_chars} chars] ...\n{out[-tail_budget:]}"
    return out


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    watcher: SystemWatcher,
    storage: Storage,
    plugins: PluginRegistry | None = None,
    auth: Any = None,
    batch_id: str | None = None,
) -> str | list[dict[str, Any]]:
    """Execute a tool synchronously and return a JSON string or multimodal list."""
    try:
        if name == "run_command":
            cmd = str(arguments["command"])
            cwd = str(_resolve(arguments["cwd"])) if arguments.get("cwd") else str(Path.home())
            timeout_sec = int(arguments.get("timeout") or 60)
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    env=os.environ.copy(),
                )
            except subprocess.TimeoutExpired as exc:
                raw_out = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
                raw_err = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
                part_out = raw_out[-3000:]
                part_err = raw_err[-2000:]
                tail_low = (part_out[-400:] + "\n" + part_err[-400:]).lower()
                prompt_cues = ("[y/n]", "(y/n)", "password:", "passphrase", "proceed?", "continue?", "[yes/no]")
                interactive_stall = any(cue in tail_low for cue in prompt_cues)
                recovery_hint = (
                    "Command appears stalled waiting for interactive confirmation/input; retry with non-interactive flags (e.g. -y, --yes, --noconfirm, DEBIAN_FRONTEND=noninteractive)."
                    if interactive_stall
                    else "Command timed out; inspect partial_stdout/partial_stderr or limit scope."
                )
                return json.dumps(
                    {
                        "error": f"Command timed out after {timeout_sec}s",
                        "exit_code": 124,
                        "timed_out": True,
                        "interactive_stall": interactive_stall,
                        "partial_stdout": part_out,
                        "partial_stderr": part_err,
                        "recovery_hint": recovery_hint,
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "exit_code": proc.returncode,
                    "stdout": compress_tool_output(proc.stdout, max_chars=4000),
                    "stderr": compress_tool_output(proc.stderr, max_chars=2000),
                },
                ensure_ascii=False,
            )

        if name == "read_file":
            p = _resolve(arguments["path"])
            if not p.is_file():
                return json.dumps({"error": f"File not found: {p}"}, ensure_ascii=False)
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(1, int(arguments.get("start_line") or 1))
            end = int(arguments.get("end_line") or min(len(lines), start + 399))
            sliced = lines[start - 1 : end]
            return json.dumps(
                {
                    "path": str(p),
                    "total_lines": len(lines),
                    "start_line": start,
                    "end_line": min(end, len(lines)),
                    "content": "\n".join(sliced),
                },
                ensure_ascii=False,
            )

        if name == "write_file":
            p = _resolve(arguments["path"])
            existed = p.is_file()
            backup = p.read_text(encoding="utf-8", errors="replace") if existed else None
            if existed and backup is not None:
                _backup_critical_config(p, backup)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(arguments["content"]), encoding="utf-8")
            storage.record_file_backup("write_file", str(p), existed, backup, batch_id=batch_id)
            from .self_resilience import verify_written_file

            res_obj: dict[str, Any] = {
                "status": "ok",
                "path": str(p),
                "bytes": p.stat().st_size,
                "batch_id": batch_id,
            }
            ver = verify_written_file(p)
            if ver.get("kind") != "skipped":
                res_obj["verification"] = ver
            return json.dumps(res_obj, ensure_ascii=False)

        if name == "edit_file":
            p = _resolve(arguments["path"])
            if not p.is_file():
                return json.dumps({"error": f"File not found: {p}"}, ensure_ascii=False)
            backup = p.read_text(encoding="utf-8", errors="replace")
            target = str(arguments["target"])
            if not target:
                return json.dumps({"error": "target must be non-empty"}, ensure_ascii=False)
            replacement = str(arguments["replacement"])
            count = backup.count(target)
            if count == 0:
                return json.dumps({"error": f"Target snippet not found in {p}"}, ensure_ascii=False)
            _backup_critical_config(p, backup)
            updated = backup.replace(target, replacement)
            p.write_text(updated, encoding="utf-8")
            storage.record_file_backup("edit_file", str(p), True, backup, batch_id=batch_id)
            from .self_resilience import verify_written_file

            res_obj = {
                "status": "ok",
                "path": str(p),
                "replacements": count,
                "bytes": p.stat().st_size,
                "batch_id": batch_id,
            }
            ver = verify_written_file(p)
            if ver.get("kind") != "skipped":
                res_obj["verification"] = ver
            return json.dumps(res_obj, ensure_ascii=False)

        if name == "list_directory":
            p = _resolve(arguments["path"])
            if not p.is_dir():
                return json.dumps({"error": f"Directory not found: {p}"}, ensure_ascii=False)
            entries = []
            for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[:200]:
                entries.append(
                    {
                        "name": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    }
                )
            return json.dumps({"path": str(p), "entries": entries}, ensure_ascii=False)

        if name == "search_files":
            import fnmatch

            query = str(arguments["query"])
            base = _resolve(arguments.get("path") or str(Path.home()))
            if not base.exists():
                return json.dumps({"matches": []}, ensure_ascii=False)
            content_search = bool(arguments.get("content_search", False))
            skip_dirs = {".git", "node_modules", ".venv", "venv", "__pycache__", ".cache", ".local", ".cargo", ".rustup", "dist", "build"}
            if content_search:
                if shutil.which("rg"):
                    cmd = ["rg", "-n", "--max-count", "50", "--max-filesize", "1M", "--", query, str(base)]
                else:
                    exclude_args = [f"--exclude-dir={d}" for d in skip_dirs]
                    cmd = ["grep", "-rnI", "-m", "50", *exclude_args, "--", query, str(base)]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                content_matches: list[str] = []
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        content_matches.append(line.rstrip("\r\n"))
                        if len(content_matches) >= 50:
                            break
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                return json.dumps({"matches": content_matches}, ensure_ascii=False)
            pattern = query if any(c in query for c in "*?[") else f"*{query}*"
            matches: list[str] = []
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".cache")]
                for entry in dirs + files:
                    if fnmatch.fnmatchcase(entry.lower(), pattern.lower()):
                        matches.append(os.path.join(root, entry))
                        if len(matches) >= 50:
                            break
                if len(matches) >= 50:
                    break
            return json.dumps({"matches": matches}, ensure_ascii=False)

        if name == "system_status":
            return json.dumps(watcher.get_health_snapshot(), ensure_ascii=False)

        if name == "list_processes":
            import psutil

            limit = min(50, max(1, int(arguments.get("limit") or 15)))
            sort_by = str(arguments.get("sort_by") or "cpu").lower()
            rows: list[dict[str, Any]] = []
            for p in psutil.process_iter(["pid", "name", "username", "cpu_percent", "memory_percent", "status"]):
                info = p.info
                rows.append(
                    {
                        "pid": info.get("pid"),
                        "name": info.get("name") or "",
                        "user": info.get("username") or "",
                        "cpu_percent": round(float(info.get("cpu_percent") or 0.0), 1),
                        "memory_percent": round(float(info.get("memory_percent") or 0.0), 1),
                        "status": info.get("status") or "",
                    }
                )
            key_field = "memory_percent" if "mem" in sort_by else "cpu_percent"
            rows.sort(key=lambda r: float(r.get(key_field) or 0.0), reverse=True)
            return json.dumps({"sort_by": key_field, "processes": rows[:limit]}, ensure_ascii=False)

        if name == "list_systemd_units":
            state = str(arguments.get("state") or "running").strip().lower()
            limit = min(100, max(1, int(arguments.get("limit") or 25)))
            cmd = ["systemctl"]
            if arguments.get("user_only"):
                cmd.append("--user")
            state_flag = "--all" if state == "all" else f"--state={state}"
            cmd.extend(["list-units", "--type=service", state_flag, "--no-legend", "--no-pager", "--plain"])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            units: list[dict[str, str]] = []
            for ln in proc.stdout.splitlines():
                parts = ln.split(None, 4)
                if len(parts) >= 4:
                    units.append(
                        {
                            "unit": parts[0],
                            "load": parts[1],
                            "active": parts[2],
                            "sub": parts[3],
                            "description": parts[4].strip() if len(parts) > 4 else "",
                        }
                    )
                if len(units) >= limit:
                    break
            return json.dumps({"state": state, "user_only": bool(arguments.get("user_only")), "units": units}, ensure_ascii=False)

        if name == "list_events":
            limit = int(arguments.get("limit") or 20)
            return json.dumps(
                {"events": watcher.get_events(limit=limit)}, ensure_ascii=False
            )

        if name == "send_notification":
            watcher._send_desktop_notification(
                str(arguments["title"]), str(arguments["body"])
            )
            return json.dumps({"status": "sent"}, ensure_ascii=False)

        if name == "remember_fact":
            expires_at = arguments.get("expires_at")
            storage.set_memory(
                str(arguments["key"]),
                str(arguments["value"]),
                expires_at=str(expires_at) if expires_at else None,
            )
            return json.dumps({"status": "saved", "key": arguments["key"]}, ensure_ascii=False)

        if name == "recall_memories":
            include_expired = bool(arguments.get("include_expired", False))
            return json.dumps(
                {"memories": storage.list_memories(include_expired=include_expired)},
                ensure_ascii=False,
            )

        if name == "forget_memory":
            deleted = storage.delete_memory(str(arguments["key"]))
            return json.dumps({"deleted": deleted}, ensure_ascii=False)

        if name == "create_automation":
            auto_id = storage.create_automation(
                str(arguments["name"]),
                str(arguments["trigger_type"]),
                str(arguments["trigger_value"]),
                str(arguments["prompt"]),
            )
            return json.dumps({"status": "created", "id": auto_id}, ensure_ascii=False)

        if name == "list_automations":
            return json.dumps({"automations": storage.list_automations()}, ensure_ascii=False)

        if name == "delete_automation":
            deleted = storage.delete_automation(int(arguments["automation_id"]))
            return json.dumps({"deleted": deleted}, ensure_ascii=False)

        if name == "read_clipboard":
            for cmd in (["wl-paste", "--no-newline"], ["xclip", "-o", "-selection", "clipboard"], ["xsel", "-ob"]):
                if shutil.which(cmd[0]):
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if proc.returncode == 0:
                        return json.dumps({"clipboard": proc.stdout[:6000]}, ensure_ascii=False)
            return json.dumps({"error": "No clipboard content or clipboard tool unavailable."}, ensure_ascii=False)

        if name == "take_screenshot":
            with tempfile.NamedTemporaryFile(suffix=".png", prefix="sysagent_shot_", delete=False) as tf:
                shot_path = tf.name
            cmd = None
            if shutil.which("spectacle"):
                cmd = ["spectacle", "-b", "-n", "-o", shot_path]
            elif shutil.which("grim"):
                cmd = ["grim", shot_path]
            elif shutil.which("scrot"):
                cmd = ["scrot", shot_path]
            if not cmd:
                return json.dumps({"error": "No screenshot tool found (spectacle/grim/scrot)."}, ensure_ascii=False)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if proc.returncode != 0 or not Path(shot_path).exists() or Path(shot_path).stat().st_size == 0:
                    return json.dumps({"error": f"Screenshot failed: {proc.stderr}"}, ensure_ascii=False)
                b64 = base64.b64encode(Path(shot_path).read_bytes()).decode("ascii")
                return [
                    {"type": "text", "text": "Screenshot captured."},
                    {"type": "image", "data": b64, "mime_type": "image/png"},
                ]
            finally:
                Path(shot_path).unlink(missing_ok=True)

        if name == "read_journal_logs":
            lines = min(200, max(1, int(arguments.get("lines") or 40)))
            cmd = ["journalctl", "-n", str(lines), "--no-pager"]
            if arguments.get("user_unit"):
                cmd.append("--user")
            if arguments.get("unit"):
                cmd.extend(["-u", str(arguments["unit"])])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return json.dumps(
                {
                    "exit_code": proc.returncode,
                    "logs": compress_tool_output(proc.stdout, max_chars=5000),
                    "stderr": compress_tool_output(proc.stderr, max_chars=1000),
                },
                ensure_ascii=False,
            )

        if name == "desktop_control":
            action = str(arguments["action"])
            val = str(arguments["value"]).strip()
            def _run(cmd: list[str]) -> tuple[bool, str]:
                if not shutil.which(cmd[0]):
                    return False, f"{cmd[0]} not installed"
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                return proc.returncode == 0, (proc.stderr or proc.stdout).strip()[:300]

            def _report(ok: bool, detail: str, **extra: Any) -> str:
                payload = {"status": "ok" if ok else "failed", "action": action, **extra}
                if not ok:
                    payload["error"] = detail or "command failed"
                return json.dumps(payload, ensure_ascii=False)

            on = val.lower() in {"on", "true", "1", "yes"}
            if action == "volume_set":
                pct = max(0, min(100, int(val)))
                ok, detail = _run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{pct}%"])
                if not ok:
                    ok, detail = _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"])
                return _report(ok, detail, value=pct)
            if action == "volume_mute":
                flag = "1" if on else "0"
                ok, detail = _run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", flag])
                if not ok:
                    ok, detail = _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", flag])
                return _report(ok, detail, muted=on)
            if action == "brightness_set":
                pct = max(5, min(100, int(val)))
                ok, detail = _run(["brightnessctl", "set", f"{pct}%"])
                if not ok:
                    ok, detail = _run(["qdbus6", "org.kde.ScreenBrightness", "/org/kde/ScreenBrightness/display0",
                                       "org.kde.ScreenBrightness.Display.SetBrightness", str(pct * 100), "0"])
                return _report(ok, detail, value=pct)
            if action == "wifi":
                ok, detail = _run(["nmcli", "radio", "wifi", "on" if on else "off"])
                return _report(ok, detail, state="on" if on else "off")
            if action == "bluetooth":
                ok, detail = _run(["bluetoothctl", "power", "on" if on else "off"])
                return _report(ok, detail, state="on" if on else "off")
            if action == "dnd":
                # A D-Bus Inhibit cookie dies with the calling process, so a one-shot qdbus call
                return _report(False, "Do-Not-Disturb control is not supported from the daemon on this desktop.")
            if action == "launch_app":
                import shlex as _shlex
                argv = _shlex.split(val)
                if not argv or not shutil.which(argv[0]):
                    return _report(False, f"Application not found: {val}")
                subprocess.Popen(argv, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return _report(True, "", app=val)
            return json.dumps({"error": f"Unsupported desktop_control action: {action}"})

        if name == "list_audit_log":
            limit = int(arguments.get("limit") or 20)
            return json.dumps({"audit_log": storage.list_audit_log(limit=limit)}, ensure_ascii=False)

        if name == "undo_last_action":
            item = storage.get_latest_undoable()
            if not item:
                return json.dumps({"status": "nothing_to_undo"})
            target = Path(item["target_path"])
            if item["action"] == "trash_file" and item["backup_content"]:
                trashed = Path(item["backup_content"])
                if not trashed.exists():
                    return json.dumps({"error": f"Trashed file missing: {trashed}"}, ensure_ascii=False)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(trashed), str(target))
            elif item["existed"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(item["backup_content"] or "", encoding="utf-8")
            elif target.exists():
                target.unlink()
            storage.mark_undone(item["id"])
            if plugins and item["action"] == "forge_skill":
                plugins.reload()
            return json.dumps({"status": "undone", "target_path": str(target), "existed": bool(item["existed"])}, ensure_ascii=False)

        if name == "undo_batch_actions":
            req_batch = arguments.get("batch_id")
            items = storage.get_undoable_batch(str(req_batch) if req_batch else None)
            if not items:
                return json.dumps({"status": "nothing_to_undo", "reverted": []})
            reverted: list[str] = []
            skipped: list[str] = []
            for item in items:
                target = Path(item["target_path"])
                if item["action"] == "trash_file" and item["backup_content"]:
                    trashed = Path(item["backup_content"])
                    if not trashed.exists():
                        skipped.append(str(target))
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(trashed), str(target))
                elif item["existed"]:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(item["backup_content"] or "", encoding="utf-8")
                elif target.exists():
                    target.unlink()
                storage.mark_undone(item["id"])
                reverted.append(str(target))
            if plugins and any(it["action"] == "forge_skill" for it in items):
                plugins.reload()
            return json.dumps(
                {
                    "status": "batch_undone",
                    "batch_id": items[0].get("batch_id"),
                    "count": len(reverted),
                    "reverted": reverted,
                    "skipped": skipped,
                },
                ensure_ascii=False,
            )

        if name == "trash_file":
            p = _resolve(arguments["path"])
            if not p.exists():
                return json.dumps({"error": f"Path not found: {p}"}, ensure_ascii=False)
            trash_dir = DATA_DIR / "trash"
            trash_dir.mkdir(parents=True, exist_ok=True)
            dest = trash_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{p.name}"
            shutil.move(str(p), str(dest))
            storage.record_file_backup("trash_file", str(p), True, str(dest), batch_id=batch_id)
            return json.dumps({"status": "trashed", "original_path": str(p), "trash_path": str(dest), "batch_id": batch_id}, ensure_ascii=False)

        if name == "search_memory_and_history":
            q = str(arguments["query"]).strip()
            limit = min(40, max(1, int(arguments.get("limit") or 15)))
            hits = storage.search_knowledge(q, limit=limit)
            return json.dumps({"query": q, "results": hits}, ensure_ascii=False)

        if name == "kg_add_relation":
            storage.add_kg_edge(
                str(arguments["subject"]), str(arguments["relation"]), str(arguments["object"])
            )
            return json.dumps({"status": "saved"}, ensure_ascii=False)

        if name == "kg_query":
            edges = storage.query_kg(arguments.get("entity"))
            return json.dumps({"relations": edges}, ensure_ascii=False)

        if name == "run_parallel_workers":
            import asyncio
            from .workers import WorkerPool

            tasks = [str(t) for t in (arguments.get("tasks") or []) if str(t).strip()]
            pool = WorkerPool(watcher.settings, watcher, storage, auth=auth)
            results = asyncio.run(pool.run_parallel(tasks))
            return json.dumps({"results": results}, ensure_ascii=False)

        if name == "forge_skill":
            import re
            from .plugins import validate_plugin_code_isolated

            raw_name = re.sub(r"[^a-zA-Z0-9_]", "_", str(arguments["tool_name"])).strip("_")
            if not raw_name:
                return json.dumps({"error": "Invalid tool_name"})
            code = str(arguments["python_code"])
            ok, detail = validate_plugin_code_isolated(code, tool_name=raw_name)
            if not ok:
                return json.dumps({"error": detail}, ensure_ascii=False)

            pdir = plugins.plugins_dir if plugins else (Path.home() / ".config" / "sysagent" / "plugins")
            pdir.mkdir(parents=True, exist_ok=True)
            dest = pdir / f"{raw_name}.py"
            existed = dest.is_file()
            old_code = dest.read_text(encoding="utf-8", errors="replace") if existed else None
            dest.write_text(code, encoding="utf-8")
            storage.record_file_backup("forge_skill", str(dest), existed, old_code, batch_id=batch_id)
            count = plugins.reload() if plugins else 0
            from .self_ops import sync_capabilities_doc
            sync_capabilities_doc(plugins)
            from .self_learn import inject_plugin_skills_to_local_mind
            inject_plugin_skills_to_local_mind(storage, plugins)
            return json.dumps({"status": "forged", "path": str(dest), "loaded_plugin_tools": count}, ensure_ascii=False)

        if name == "record_lesson":
            lid = storage.add_lesson(
                str(arguments["context"]), str(arguments["insight"]), str(arguments["resolution"])
            )
            return json.dumps({"status": "recorded", "id": lid}, ensure_ascii=False)

        if name == "create_goal_plan":
            steps = [{"title": str(s), "status": "pending", "note": ""} for s in (arguments.get("steps") or [])]
            pid = storage.create_goal_plan(str(arguments["goal"]), json.dumps(steps, ensure_ascii=False))
            return json.dumps({"status": "created", "plan_id": pid, "steps": steps}, ensure_ascii=False)

        if name == "update_goal_step":
            plan = storage.get_goal_plan(int(arguments["plan_id"]))
            if not plan:
                return json.dumps({"error": "Plan not found"})
            steps = json.loads(plan["steps_json"])
            idx = int(arguments["step_index"])
            if idx < 0 or idx >= len(steps):
                return json.dumps({"error": f"Invalid step_index {idx}"})
            steps[idx]["status"] = str(arguments["status"])
            if arguments.get("note"):
                steps[idx]["note"] = str(arguments["note"])
            all_done = all(s["status"] in {"done", "skipped"} for s in steps)
            overall = "completed" if all_done else "active"
            storage.update_goal_plan(plan["id"], json.dumps(steps, ensure_ascii=False), overall)
            return json.dumps({"status": "updated", "plan_status": overall, "steps": steps}, ensure_ascii=False)

        if name == "web_search":
            import html as _html
            import re as _re
            import httpx

            q = str(arguments["query"]).strip()
            limit = min(10, max(1, int(arguments.get("max_results") or 5)))
            resp = httpx.post(
                "https://html.duckduckgo.com/html/",
                data={"q": q},
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
                timeout=12,
                follow_redirects=True,
            )
            results = []
            for m in _re.finditer(
                r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                resp.text,
                _re.DOTALL,
            ):
                href = _html.unescape(m.group(1))
                title = _re.sub(r"<[^>]+>", "", _html.unescape(m.group(2))).strip()
                if title and href:
                    results.append({"title": title, "url": href})
                if len(results) >= limit:
                    break
            return json.dumps({"query": q, "results": results}, ensure_ascii=False)

        if name == "fetch_url":
            import html as _html
            import ipaddress as _ip
            import re as _re
            import socket as _socket
            from urllib.parse import urljoin as _urljoin, urlparse as _urlparse
            import httpx

            def _validate_public_url(target_url: str) -> str | None:
                if not target_url.startswith(("http://", "https://")):
                    return "Only http:// and https:// URLs are allowed."
                host = _urlparse(target_url).hostname or ""
                if not host:
                    return "Invalid URL host."
                try:
                    infos = _socket.getaddrinfo(host, None)
                    if not infos:
                        return f"Could not resolve host: {host}"
                    for info in infos:
                        addr = _ip.ip_address(info[4][0])
                        if isinstance(addr, _ip.IPv6Address) and addr.ipv4_mapped is not None:
                            addr = addr.ipv4_mapped
                        if not addr.is_global:
                            return "Fetching local/private network addresses is not allowed."
                except (OSError, ValueError):
                    return f"Could not resolve host: {host}"
                return None

            url = str(arguments["url"]).strip()
            body_bytes = b""
            final_url = url
            status_code = 0
            encoding = "utf-8"
            max_bytes = 512 * 1024
            completed = False
            for _ in range(5):
                err = _validate_public_url(url)
                if err:
                    return json.dumps({"error": err})
                with httpx.stream(
                    "GET",
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
                    timeout=12,
                    follow_redirects=False,
                ) as resp:
                    status_code = resp.status_code
                    final_url = str(resp.url)
                    loc = resp.headers.get("location")
                    if status_code in {301, 302, 303, 307, 308} and loc and loc.strip():
                        url = _urljoin(url, loc.strip())
                        continue
                    encoding = resp.encoding or "utf-8"
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in resp.iter_bytes(16384):
                        if not chunk:
                            continue
                        remaining = max_bytes - total
                        if remaining <= 0:
                            break
                        chunks.append(chunk[:remaining])
                        total += len(chunks[-1])
                    body_bytes = b"".join(chunks)
                    completed = True
                    break
            if not completed:
                return json.dumps({"error": "Too many redirects."})
            raw_text = body_bytes.decode(encoding, errors="replace")
            text = _re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_text)
            text = _re.sub(r"<[^>]+>", " ", _html.unescape(text))
            text = _re.sub(r"\s+", " ", text).strip()
            return json.dumps({"url": final_url, "status_code": status_code, "text": text[:5000]}, ensure_ascii=False)

        if name == "start_focus_timer":
            mins = max(1, int(arguments["minutes"]))
            label = str(arguments.get("label") or f"Focus ({mins}m)")
            ends_at = (datetime.now() + timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M:%S")
            tid = storage.create_focus_timer(label, ends_at)
            return json.dumps({"status": "started", "id": tid, "label": label, "ends_at": ends_at}, ensure_ascii=False)

        if name == "get_app_usage":
            date_str = arguments.get("date")
            return json.dumps({"usage": storage.get_app_usage(date_str)}, ensure_ascii=False)

        if name == "reload_plugins":
            count = plugins.reload() if plugins else 0
            return json.dumps({"status": "reloaded", "plugin_tools": count})

        if name == "inspect_self_component":
            _CANONICAL_COMPONENTS = (
                "brain", "cortex", "neural", "evolve", "local_mind", "self",
                "guard", "notifications", "prompts", "checkpoints", "codebase", "all",
            )
            raw_comp = str(arguments.get("component") or "all").strip().lower()
            norm = raw_comp.replace("-", "_").replace(" ", "_")
            if norm in _CANONICAL_COMPONENTS:
                comp = norm
            else:
                parts = {p for p in norm.split("_") if p}
                comp = ""
                for c in _CANONICAL_COMPONENTS:
                    if c in {"all", "self"}:
                        continue
                    if c in parts or c.rstrip("s") in parts:
                        comp = c
                        break
                if not comp:
                    close = difflib.get_close_matches(norm, _CANONICAL_COMPONENTS, n=1, cutoff=0.6)
                    comp = close[0] if close else "all"
            query_txt = str(arguments.get("query") or "").strip()
            refs = getattr(watcher, "runtime_refs", None) if watcher is not None else None
            refs = refs if isinstance(refs, dict) else {}
            out: dict[str, Any] = {"component": comp}

            if comp in {"prompts", "all"}:
                from .prompts import PromptRegistry

                reg = PromptRegistry(storage)
                all_hist = storage.list_prompt_versions()
                by_key_hist: dict[str, list[dict[str, Any]]] = {}
                for hv in all_hist:
                    pk = str(hv.get("prompt_key") or "")
                    by_key_hist.setdefault(pk, []).append(
                        {
                            "version": int(hv.get("version") or 0),
                            "active": bool(hv.get("active")),
                            "successes": int(hv.get("successes") or 0),
                            "failures": int(hv.get("failures") or 0),
                            "reason": str(hv.get("reason") or "")[:140],
                            "created_at": hv.get("created_at"),
                        }
                    )
                p_items: list[dict[str, Any]] = []
                for p in reg.list_status():
                    k = str(p["key"])
                    kh = by_key_hist.get(k, [])
                    entry: dict[str, Any] = {
                        "key": k,
                        "active_version": int(p.get("version") or 0),
                        "total_versions": len(kh),
                        "successes": int(p.get("successes") or 0),
                        "failures": int(p.get("failures") or 0),
                        "reason": str(p.get("reason") or "")[:140],
                        "last_updated": kh[0].get("created_at") if kh else None,
                        "history": kh[:5],
                    }
                    if comp == "prompts":
                        entry["template_preview"] = str(p.get("template") or "")[:260]
                    p_items.append(entry)
                evolved_cnt = sum(1 for it in p_items if it["active_version"] > 0)
                out["prompts"] = {
                    "evolved_count": evolved_cnt,
                    "total_keys": len(p_items),
                    "total_versions_saved": len(all_hist),
                    "prompts": p_items,
                }

            if comp in {"checkpoints", "all"}:
                cps = storage.list_session_checkpoints(limit=12 if comp == "checkpoints" else 5)
                all_cps = storage.list_session_checkpoints(limit=200)
                out["checkpoints"] = {
                    "count": len(all_cps),
                    "recent": [
                        {
                            "session_id": c.get("session_id"),
                            "topic": c.get("topic"),
                            "files": c.get("files"),
                            "commands": c.get("commands"),
                            "summary": str(c.get("summary") or "")[:120],
                            "created_at": c.get("created_at"),
                        }
                        for c in (cps if comp == "checkpoints" else cps[:3])
                    ],
                }

            if comp in {"codebase", "all"}:
                from .evolve_ast import analyze_symbol_blast_radius
                from .self_ops import get_live_codebase_summary

                cb_info: dict[str, Any] = {
                    "summary": get_live_codebase_summary(),
                    "package_dir": str(Path(__file__).resolve().parent),
                }
                if query_txt and comp == "codebase":
                    try:
                        cb_info["blast_radius"] = analyze_symbol_blast_radius(query_txt, storage=storage)
                    except Exception:
                        pass
                out["codebase"] = cb_info

            if comp in {"brain", "all"}:
                thoughts = storage.list_brain_thoughts(limit=6)
                rules = storage.list_brain_rules()
                questions = storage.list_brain_questions("pending")
                brain_info: dict[str, Any] = {
                    "rules_count": len(rules),
                    "rules": [
                        {
                            "id": r.get("id"),
                            "pattern": str(r.get("pattern") or "")[:60],
                            "action": str(r.get("action_policy") or "")[:60],
                            "confidence": r.get("confidence"),
                        }
                        for r in rules[:8]
                    ],
                    "recent_thoughts": [
                        {
                            "at": str(th.get("created_at") or "")[:16],
                            "decision": th.get("decision"),
                            "confidence": th.get("confidence"),
                            "summary": str(th.get("summary") or "")[:120],
                        }
                        for th in thoughts
                    ],
                    "pending_questions": [
                        {"id": q.get("id"), "question": str(q.get("question") or "")[:100]}
                        for q in questions[:5]
                    ],
                }
                live_brain = refs.get("brain")
                if live_brain is not None and hasattr(live_brain, "is_quiet_hours"):
                    try:
                        brain_info["quiet_hours"] = bool(live_brain.is_quiet_hours())
                    except Exception:
                        pass
                out["brain"] = brain_info

            if comp in {"cortex", "all"}:
                live_cortex = refs.get("cortex")
                if live_cortex is None:
                    from .config import get_settings
                    from .cortex import NeuralBrainCortex

                    live_cortex = NeuralBrainCortex(get_settings(), storage, watcher=watcher)
                c_raw = live_cortex.get_status()
                habits = c_raw.get("habit_suggestions") or c_raw.get("habits") or []
                last_heal = c_raw.get("last_heal")
                if last_heal is None:
                    last_heal = c_raw.get("predictive_healer")
                cortex_info: dict[str, Any] = {
                    "flow": c_raw.get("flow"),
                    "predictive_healer": last_heal,
                    "last_dream": c_raw.get("last_dream"),
                    "habit_suggestions_count": len(habits),
                }
                if comp == "cortex" and habits:
                    cortex_info["habits"] = habits[:3]
                try:
                    from .self_learn import analyze_causal_digital_twin

                    cortex_info["digital_twin"] = analyze_causal_digital_twin(storage, watcher)
                except Exception:
                    pass
                out["cortex"] = cortex_info

            if comp in {"neural", "all"}:
                from .neural import NeuralEngine

                eng = NeuralEngine(storage)
                st_raw = eng.get_status()
                wf = st_raw.get("world_forecast") or {}
                neural_info: dict[str, Any] = {
                    "status": {
                        "epochs_trained": st_raw.get("epochs_trained"),
                        "tool_accuracy": st_raw.get("tool_accuracy"),
                        "param_count": st_raw.get("param_count"),
                        "world_forecast": {
                            "pred_cpu_10m": wf.get("pred_cpu_10m"),
                            "pred_ram_10m": wf.get("pred_ram_10m"),
                            "bottleneck_prob": wf.get("bottleneck_prob"),
                            "anomaly_detected": wf.get("anomaly_detected"),
                            "proactive_hint": wf.get("proactive_hint"),
                        },
                    }
                }
                if query_txt or comp == "neural":
                    snap = watcher.get_health_snapshot() if watcher is not None and hasattr(watcher, "get_health_snapshot") else {}
                    neural_info["pulse"] = eng.pulse(query_txt or "system status", sys_state=snap)
                out["neural"] = neural_info

            if comp in {"evolve", "all"}:
                from .self_learn import patch_outcomes, proposal_stats

                out["evolve"] = {
                    "stats": proposal_stats(storage),
                    "patch_outcomes": patch_outcomes(storage),
                    "recent_proposals": [
                        {
                            "id": p.get("id"),
                            "status": p.get("status"),
                            "category": p.get("category"),
                            "file": Path(str(p.get("file_path") or "")).name,
                            "title": str(p.get("title") or "")[:70],
                        }
                        for p in storage.list_evolve_proposals(limit=5)
                    ],
                }

            if comp in {"local_mind", "self", "all"}:
                from .local_inference import get_runtime_memory_status
                from .local_mind import find_all_gguf_models, find_gguf_model, find_mmproj_for_model
                from .self_guard import get_calibration
                from .self_learn import learn_status_summary, predict_user_routine

                lsum = learn_status_summary(storage)
                lm_sub = lsum.get("local_model") or {}
                out["self"] = {
                    "distilled_count": lm_sub.get("distilled", lsum.get("distilled_count", 0)),
                    "uncertain_count": lm_sub.get("uncertain_queue", lsum.get("uncertain_count", 0)),
                    "shadow": lsum.get("shadow"),
                    "roadmap_top": (lsum.get("roadmap") or [])[:3],
                }
                rt_mem = get_runtime_memory_status()
                mm_model = find_gguf_model("multimodal", strict=True) or find_gguf_model("multimodal")
                mmproj_path = find_mmproj_for_model(mm_model)
                out["local_mind"] = {
                    "warm_model": rt_mem.get("warm_model"),
                    "warm_idle_sec": rt_mem.get("idle_sec"),
                    "hysteresis_sec": rt_mem.get("hysteresis_sec"),
                    "vram": rt_mem.get("vram"),
                    "vision_ready": bool(mm_model is not None and mmproj_path is not None),
                    "mmproj_file": mmproj_path.name if mmproj_path is not None else None,
                    "backend": os.environ.get("SYSAGENT_LOCAL_BACKEND", "auto"),
                    "installed_gguf": [p.name for p in find_all_gguf_models()[:6]],
                    "calibration": get_calibration(storage),
                }
                out["routine"] = predict_user_routine(storage)

            if comp in {"guard", "all"}:
                from .self_guard import get_calibration
                from .self_resilience import get_circuit_breaker, get_quota_governor

                sg = refs.get("self_guard")
                last_eval = getattr(sg, "last_eval", None) if sg is not None else None
                last_drift = getattr(sg, "last_drift", None) if sg is not None else None
                if comp == "guard":
                    if last_eval is None:
                        try:
                            from .local_mind import LocalMind
                            from .self_resilience import evaluate_self_gate

                            last_eval = evaluate_self_gate(LocalMind(storage), storage)
                        except Exception:
                            last_eval = None
                    if last_drift is None:
                        try:
                            from .self_resilience import run_integrity_and_drift_audit

                            last_drift = run_integrity_and_drift_audit(storage)
                        except Exception:
                            last_drift = None
                breakers = get_circuit_breaker().snapshot()
                open_breakers = [k for k, v in breakers.items() if v.get("state") != "closed"]
                out["guard"] = {
                    "calibration": get_calibration(storage),
                    "last_autopsy": getattr(sg, "last_autopsy", None) if sg is not None else None,
                    "last_canary": getattr(sg, "last_canary", None) if sg is not None else None,
                    "last_profile": getattr(sg, "last_profile", None) if sg is not None else None,
                    "eval_gate": last_eval,
                    "drift": last_drift,
                    "open_breakers": open_breakers,
                    "breakers_count": len(breakers),
                    "quota": get_quota_governor().get_status(),
                }

            notifier = (getattr(watcher, "notifier", None) or getattr(watcher, "notifications", None)) if watcher is not None else None
            if comp in {"notifications", "all"} and notifier is not None:
                try:
                    nst = notifier.get_status()
                    out["notifications"] = {
                        "total": nst.get("total", 0),
                        "suppressed": nst.get("suppressed", 0),
                        "deduped": nst.get("deduped", 0),
                        "recent": (nst.get("recent") or [])[:3],
                    }
                except Exception:
                    pass

            if comp == "all":
                pr_s = out.get("prompts") or {}
                b_th = out.get("brain", {}).get("recent_thoughts") or []
                latest_th = str(b_th[0].get("summary") or "")[:75] if b_th else "none"
                fl = (out.get("cortex", {}).get("flow") or {}).get("mode", "normal")
                n_st = out.get("neural", {}).get("status") or {}
                wf_s = n_st.get("world_forecast") or {}
                ev_s = out.get("evolve", {}).get("stats") or {}
                ev_applied = (ev_s.get("by_status") or {}).get("applied", ev_s.get("applied", 0))
                lm_s = out.get("local_mind") or {}
                g_s = out.get("guard") or {}
                q_s = g_s.get("quota") or {}
                out["overview"] = (
                    f"Prompts evolved={pr_s.get('evolved_count', 0)}/{pr_s.get('total_keys', 0)} "
                    f"({pr_s.get('total_versions_saved', 0)} versions); "
                    f"Brain latest thought: {latest_th} (rules={out.get('brain', {}).get('rules_count', 0)}); "
                    f"Cortex flow={fl}; Neural acc={n_st.get('tool_accuracy', '-')}, "
                    f"bottleneck={wf_s.get('bottleneck_prob', 0)}; Evolve applied={ev_applied}/{ev_s.get('total', 0)}; "
                    f"LocalMind warm={lm_s.get('warm_model') or 'idle'}; Guard open_breakers={len(g_s.get('open_breakers') or [])}, "
                    f"bg_quota={q_s.get('bg_calls_last_hour', 0)}/{q_s.get('hourly_bg_cap', 40)}."
                )
            return json.dumps(out, ensure_ascii=False)

        if plugins and name in plugins.tools:
            result = plugins.execute(name, arguments)
            return result if isinstance(result, (str, list)) else json.dumps(result, ensure_ascii=False)

        return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

"""System tools exposed to the Gemini Interactions API."""

from __future__ import annotations

import base64
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
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60,
                env=os.environ.copy(),
            )
            return json.dumps(
                {
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-2000:],
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
            return json.dumps(
                {"status": "ok", "path": str(p), "bytes": p.stat().st_size, "batch_id": batch_id},
                ensure_ascii=False,
            )

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
            return json.dumps(
                {"status": "ok", "path": str(p), "replacements": count, "bytes": p.stat().st_size, "batch_id": batch_id},
                ensure_ascii=False,
            )

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
                {"exit_code": proc.returncode, "logs": proc.stdout[-5000:], "stderr": proc.stderr[-1000:]},
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
                        if (
                            addr.is_private
                            or addr.is_loopback
                            or addr.is_link_local
                            or addr.is_reserved
                            or addr.is_multicast
                            or addr.is_unspecified
                        ):
                            return "Fetching local/private network addresses is not allowed."
                except (OSError, ValueError):
                    return f"Could not resolve host: {host}"
                return None

            url = str(arguments["url"]).strip()
            resp = None
            for _ in range(5):
                err = _validate_public_url(url)
                if err:
                    return json.dumps({"error": err})
                resp = httpx.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
                    timeout=12,
                    follow_redirects=False,
                )
                if resp.status_code in {301, 302, 303, 307, 308} and resp.headers.get("location"):
                    url = _urljoin(url, resp.headers["location"])
                    continue
                break
            assert resp is not None
            text = _re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", resp.text)
            text = _re.sub(r"<[^>]+>", " ", _html.unescape(text))
            text = _re.sub(r"\s+", " ", text).strip()
            return json.dumps({"url": str(resp.url), "status_code": resp.status_code, "text": text[:5000]}, ensure_ascii=False)

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
            comp = str(arguments.get("component") or "all").strip().lower()
            query_txt = str(arguments.get("query") or "").strip()
            out: dict[str, Any] = {"component": comp}
            if comp in {"brain", "all"}:
                out["brain"] = {
                    "rules": storage.list_brain_rules()[:15],
                    "recent_thoughts": storage.list_brain_thoughts(limit=12),
                    "pending_questions": storage.list_brain_questions("pending")[:10],
                }
            if comp in {"neural", "all"}:
                from .neural import NeuralEngine

                eng = NeuralEngine(storage)
                neural_info: dict[str, Any] = {"status": eng.get_status()}
                if query_txt or comp == "neural":
                    snap = watcher.get_health_snapshot()
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
                            "title": p.get("title"),
                        }
                        for p in storage.list_evolve_proposals(limit=10)
                    ],
                }
            if comp in {"local_mind", "self", "all"}:
                from .self_learn import learn_status_summary, predict_user_routine

                out["self"] = learn_status_summary(storage)
                out["routine"] = predict_user_routine(storage)
            return json.dumps(out, ensure_ascii=False)

        if plugins and name in plugins.tools:
            result = plugins.execute(name, arguments)
            return result if isinstance(result, (str, list)) else json.dumps(result, ensure_ascii=False)

        return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

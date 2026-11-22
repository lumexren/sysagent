"""Terminal CLI for talking to the SysAgent daemon or running it."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import uvicorn
import websockets

from .auth import logout_oauth, run_oauth_login
from .config import DATA_DIR, get_settings
from .i18n import t
from .ipc import ipc_get, ipc_post, ipc_ws_connect
from .self_ops import export_brain_bundle, import_brain_bundle
from .storage import Storage


_BRACKETED_PASTE_ON = "\033[?2004h"
_BRACKETED_PASTE_OFF = "\033[?2004l"
_PASTE_START = "\x1b[200~"
_PASTE_END = "\x1b[201~"

_MODEL_ALIASES: dict[str, str] = {
    "local": "local-mind",
    "neon": "gemini-4-neon",
    "hydrogen": "gemini-4-hydrogen",
    "argon": "gemini-4-argon",
    "flash": "gemini-3.8-flash",
    "lite": "gemini-3.5-flash-lite",
}


def _strip_paste_sequences(text: str) -> str:
    return text.replace(_PASTE_START, "").replace(_PASTE_END, "")


def _flush_stdin() -> None:
    """Discard any buffered keystrokes or paste leftovers before an approval prompt."""
    try:
        import termios

        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def _format_terminal_text(text: str) -> str:
    """Format markdown replies for terminal: cyan fenced/inline code, bold text, and strip internal tool tags."""
    out = re.sub(r"\s*\(`[a-z0-9_]+`\)", "", text or "")
    parts = re.split(r"(```[\s\S]*?```)", out)
    rendered: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            rendered.append(f"\033[36m{part}\033[0m")
        else:
            seg = re.sub(r"`([^`\n]+)`", r"\033[36m\1\033[0m", part)
            seg = re.sub(r"\*\*([^*\n]+)\*\*", r"\033[1m\1\033[0m", seg)
            rendered.append(seg)
    return "".join(rendered).strip()


def _format_tool_preview(preview: str | None) -> str:
    """Build a compact single-line summary from a tool result preview."""
    raw = (preview or "").strip()
    if not raw:
        return ""
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                if "error" in obj:
                    return f"err: {str(obj['error'])[:80]}"
                if "exit_code" in obj:
                    ec = obj["exit_code"]
                    out = " ".join(str(obj.get("stdout") or obj.get("stderr") or "").split())
                    return f"exit={ec}" + (f" | {out[:70]}" if out else "")
        except Exception:
            pass
    flat = " ".join(raw.split())
    return flat[:100] + ("..." if len(flat) > 100 else "")


def _read_clipboard_text() -> str:
    for ccmd in (
        ["wl-paste", "--no-newline"],
        ["xclip", "-o", "-selection", "clipboard"],
        ["xsel", "-ob"],
    ):
        if shutil.which(ccmd[0]):
            try:
                cp = subprocess.run(ccmd, capture_output=True, text=True, timeout=3)
                if cp.returncode == 0 and cp.stdout:
                    return cp.stdout
            except Exception:
                continue
    return ""


_IMAGE_ATTACH_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
)


def _format_colored_diff(diff_text: str, max_lines: int | None = None) -> tuple[str, int]:
    """Colorize a unified diff string for terminal display and return `(rendered, hidden_lines)`."""
    lines = (diff_text or "").splitlines()
    if not lines:
        return "", 0
    hidden = 0
    if max_lines is not None and len(lines) > max_lines:
        hidden = len(lines) - max_lines
        lines = lines[:max_lines]
    rendered: list[str] = []
    for ln in lines:
        if ln.startswith(("+++", "---")):
            rendered.append(f"  \033[1m{ln}\033[0m")
        elif ln.startswith("@@"):
            rendered.append(f"  \033[36m{ln}\033[0m")
        elif ln.startswith("+"):
            rendered.append(f"  \033[32m{ln}\033[0m")
        elif ln.startswith("-"):
            rendered.append(f"  \033[31m{ln}\033[0m")
        else:
            rendered.append(f"  \033[90m{ln}\033[0m")
    return "\n".join(rendered), hidden


def _expand_at_paths(text: str, cwd: Path, max_lines: int = 120) -> tuple[str, list[str]]:
    """Expand @path tokens in user prompts into attached file snippets, image markers, or directory listings."""
    pattern = re.compile(r"(?<!\S)@([^\s,;'\"]+)")
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return text, []

    attachments: list[str] = []
    labels: list[str] = []
    seen: set[Path] = set()
    for m in matches:
        raw_p = m.group(1)
        p = Path(raw_p).expanduser()
        target = p.resolve() if (raw_p.startswith(("~", "/"))) else (cwd / p).resolve()
        if target in seen:
            continue
        if target.is_file():
            seen.add(target)
            if target.suffix.lower() in _IMAGE_ATTACH_EXTS:
                attachments.append(f"[Attached image `{target}`]")
                labels.append(raw_p)
                continue
            try:
                all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                head = "\n".join(all_lines[:max_lines])
                if len(head) > 4500:
                    head = head[:4500] + "\n...[truncated]"
                attachments.append(
                    f"[Attached file `{target}` ({len(all_lines)} lines, showing first {min(len(all_lines), max_lines)})]:\n{head}"
                )
                labels.append(raw_p)
            except Exception:
                pass
        elif target.is_dir():
            seen.add(target)
            try:
                entries = sorted(e.name + ("/" if e.is_dir() else "") for e in target.iterdir())[:40]
                attachments.append(f"[Attached directory `{target}`]: {', '.join(entries) or '(empty)'}")
                labels.append(raw_p)
            except Exception:
                pass

    if not attachments:
        return text, []
    return f"{text}\n\n" + "\n\n".join(attachments), labels


def _clean_reply_text(text: str) -> str:
    """Backward-compatible wrapper returning formatted terminal output."""
    return _format_terminal_text(text)


def _auth_headers() -> dict[str, str]:
    tok = get_settings().auth_token
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _ws_url(base_url: str, session_id: str) -> str:
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    tok = get_settings().auth_token
    q = f"session_id={session_id}" + (f"&token={tok}" if tok else "")
    return f"{ws_url}/ws/chat?{q}"


async def _send_and_stream(
    prompt: str,
    session_id: str,
    base_url: str,
    locale: str | None = None,
    model_override: str | None = None,
) -> str | None:
    ws_url = _ws_url(base_url, session_id)
    loc = locale or get_settings().locale
    spin_label = ["thinking..."]
    changed_model: str | None = None

    async def _spinner(stop_ev: asyncio.Event) -> None:
        frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
        idx = 0
        t0 = asyncio.get_running_loop().time()
        while not stop_ev.is_set():
            el = int(asyncio.get_running_loop().time() - t0)
            print(f"\r\033[K\033[90m{frames[idx % len(frames)]} {el}s: {spin_label[0]}\033[0m", end="", flush=True)
            idx += 1
            try:
                await asyncio.wait_for(stop_ev.wait(), timeout=0.12)
            except TimeoutError:
                pass
        print("\r\033[K", end="", flush=True)

    loop = asyncio.get_running_loop()
    cancel_requested = False
    active_task: list[asyncio.Task[None] | None] = [None]

    def _on_sigint() -> None:
        nonlocal cancel_requested
        cancel_requested = True
        if active_task[0] is not None and not active_task[0].done():
            active_task[0].cancel()

    sigint_installed = False
    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
        sigint_installed = True
    except (NotImplementedError, RuntimeError, ValueError):
        pass

    try:
        async with ipc_ws_connect(ws_url, ping_interval=30, ping_timeout=120, close_timeout=10) as ws:
            if model_override:
                await ws.send(json.dumps({"type": "set_model", "model": model_override}))
                try:
                    await asyncio.wait_for(ws.recv(), timeout=1.5)
                except TimeoutError:
                    pass
            payload: dict[str, object] = {"type": "message", "text": prompt}
            if locale:
                payload["locale"] = locale
            await ws.send(json.dumps(payload))

            delta_chunks: list[str] = []
            handled_final = False
            stop_spin = asyncio.Event()
            spin_task = asyncio.create_task(_spinner(stop_spin))

            async def _pump_events() -> None:
                nonlocal stop_spin, spin_task, handled_final, changed_model
                import select
                import threading
                from collections import deque

                queued_raw: deque[str] = deque()

                def _read_line_cancelable(prompt_txt: str, stop_ev: threading.Event) -> str | None:
                    sys.stdout.write(prompt_txt)
                    sys.stdout.flush()
                    while not stop_ev.is_set():
                        try:
                            ready, _, _ = select.select([sys.stdin], [], [], 0.15)
                        except Exception:
                            return sys.stdin.readline()
                        if ready:
                            return sys.stdin.readline()
                    return None

                while True:
                    raw = queued_raw.popleft() if queued_raw else await ws.recv()
                    if not stop_spin.is_set():
                        stop_spin.set()
                        await spin_task
                    event = json.loads(raw)
                    etype = event.get("type")

                    if etype == "status":
                        spin_label[0] = str(event.get("message") or "thinking...")
                        stop_spin = asyncio.Event()
                        spin_task = asyncio.create_task(_spinner(stop_spin))
                    elif etype == "tool_call":
                        args_str = json.dumps(event.get("arguments", {}), ensure_ascii=False)
                        print(f"\033[36m[{t('cli_tool', loc)}] {event['name']}({args_str})\033[0m")
                        spin_label[0] = f"running {event['name']}..."
                        stop_spin = asyncio.Event()
                        spin_task = asyncio.create_task(_spinner(stop_spin))
                    elif etype == "approval_required":
                        print(
                            f"\n\033[33m[{t('approval_required', loc).upper()} - {event['risk_level'].upper()}]\033[0m {event['reason']}"
                        )
                        if event.get("preview"):
                            print(f"\033[90m  Preview: {event['preview']}\033[0m")
                        diff_raw = str(event.get("diff") or "")
                        show_full_diff = False
                        if diff_raw:
                            rendered_diff, hidden_cnt = _format_colored_diff(diff_raw, max_lines=16)
                            if rendered_diff:
                                print(rendered_diff)
                            if hidden_cnt > 0:
                                print(
                                    f"\033[90m  (... +{hidden_cnt} more lines, type 'd' to toggle full diff)\033[0m"
                                )
                        confirm_prompt = t("cli_confirm_diff" if diff_raw else "cli_confirm", loc)
                        _flush_stdin()
                        stop_stdin = threading.Event()
                        stdin_task = asyncio.create_task(
                            asyncio.to_thread(_read_line_cancelable, confirm_prompt, stop_stdin)
                        )
                        ws_task = asyncio.create_task(ws.recv())
                        while True:
                            done, _ = await asyncio.wait(
                                {stdin_task, ws_task}, return_when=asyncio.FIRST_COMPLETED
                            )
                            if stdin_task in done:
                                ans = (stdin_task.result() or "").strip().lower()
                                if ans in {"d", "diff"} and diff_raw:
                                    show_full_diff = not show_full_diff
                                    limit = None if show_full_diff else 16
                                    rendered_diff, hidden_cnt = _format_colored_diff(diff_raw, max_lines=limit)
                                    if rendered_diff:
                                        print(rendered_diff)
                                    if hidden_cnt > 0:
                                        print(
                                            f"\033[90m  (... +{hidden_cnt} more lines, type 'd' to toggle full diff)\033[0m"
                                        )
                                    _flush_stdin()
                                    stop_stdin = threading.Event()
                                    stdin_task = asyncio.create_task(
                                        asyncio.to_thread(_read_line_cancelable, confirm_prompt, stop_stdin)
                                    )
                                    continue
                                ws_task.cancel()
                                try:
                                    await ws_task
                                except BaseException:
                                    pass
                                yes_set = set(t("cli_yes_tokens", loc).split(","))
                                approved = ans in yes_set
                                await ws.send(
                                    json.dumps(
                                        {
                                            "type": "approval_response",
                                            "approval_id": event["approval_id"],
                                            "approved": approved,
                                        }
                                    )
                                )
                                break
                            ws_raw = ws_task.result()
                            ws_ev = json.loads(ws_raw)
                            if (
                                ws_ev.get("type") == "approval_resolved"
                                and ws_ev.get("approval_id") == event["approval_id"]
                            ):
                                stop_stdin.set()
                                await stdin_task
                                approved = bool(ws_ev.get("approved"))
                                lbl = t("cli_ok", loc) if approved else t("cli_denied", loc)
                                col = "\033[32m" if approved else "\033[31m"
                                print(f"\n{col}[{lbl} via notification]\033[0m")
                                break
                            queued_raw.append(ws_raw)
                            ws_task = asyncio.create_task(ws.recv())
                        spin_label[0] = f"running {event['tool_name']}..."
                        stop_spin = asyncio.Event()
                        spin_task = asyncio.create_task(_spinner(stop_spin))
                    elif etype == "approval_resolved":
                        pass
                    elif etype == "model_changed":
                        changed_model = str(event.get("model") or "") or None
                        print(f"\033[36m  -> Active model switched to: {changed_model}\033[0m")
                    elif etype == "tool_result":
                        status_lbl = t("cli_ok", loc) if event.get("approved") else t("cli_denied", loc)
                        summary = _format_tool_preview(event.get("result_preview"))
                        suffix = f" ({summary})" if summary else ""
                        print(f"\033[90m  -> {event['name']}: {status_lbl}{suffix}\033[0m")
                        spin_label[0] = "thinking..."
                        stop_spin = asyncio.Event()
                        spin_task = asyncio.create_task(_spinner(stop_spin))
                    elif etype == "text_delta":
                        delta_chunks.append(str(event.get("text") or ""))
                        spin_label[0] = "writing response..."
                        stop_spin = asyncio.Event()
                        spin_task = asyncio.create_task(_spinner(stop_spin))
                    elif etype == "assistant_message":
                        handled_final = True
                        delta_chunks.clear()
                        print(f"\n{_format_terminal_text(str(event.get('text') or ''))}\n")
                    elif etype == "system":
                        print(f"\033[90m{event.get('message', '')}\033[0m")
                    elif etype == "error":
                        print(f"\033[31m{t('error_prefix', loc)}: {event['message']}\033[0m", file=sys.stderr)
                    elif etype == "done":
                        if delta_chunks and not handled_final and not event.get("cancelled"):
                            print(f"\n{_format_terminal_text(''.join(delta_chunks))}\n")
                            delta_chunks.clear()
                        break

            pump_task = asyncio.create_task(_pump_events())
            active_task[0] = pump_task
            try:
                await pump_task
            except (asyncio.CancelledError, KeyboardInterrupt):
                if cancel_requested:
                    try:
                        await ws.send(json.dumps({"type": "cancel"}))
                        await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except Exception:
                        pass
                    print("\n\033[33m[Turn cancelled]\033[0m")
                else:
                    raise
            finally:
                if not stop_spin.is_set():
                    stop_spin.set()
                if not spin_task.done():
                    await spin_task
    except websockets.exceptions.ConnectionClosed as exc:
        print(f"\n\033[31mWebSocket connection closed: {exc}\033[0m", file=sys.stderr)
    except (OSError, ConnectionRefusedError):
        print(t("cli_daemon_offline", loc), file=sys.stderr)
        sys.exit(1)
    finally:
        if sigint_installed:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except Exception:
                pass
    return changed_model


def _load_last_session_id() -> str:
    sf = DATA_DIR / "last_chat_session"
    if sf.is_file():
        try:
            val = sf.read_text(encoding="utf-8").strip()
            if val:
                return val
        except Exception:
            pass
    sid = f"cli-{uuid.uuid4().hex[:8]}"
    _save_last_session_id(sid)
    return sid


def _save_last_session_id(session_id: str) -> None:
    sf = DATA_DIR / "last_chat_session"
    try:
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text(session_id.strip(), encoding="utf-8")
    except Exception:
        pass


def _read_chat_input(prompt_str: str) -> str:
    """Read a user prompt, coalescing multi-line terminal pastes and explicit continuations into one message."""
    is_tty = sys.stdin.isatty()
    if is_tty:
        try:
            sys.stdout.write(_BRACKETED_PASTE_ON)
            sys.stdout.flush()
        except Exception:
            pass

    first = input(prompt_str)
    first_clean = _strip_paste_sequences(first.replace("\r\n", "\n").replace("\r", "\n"))
    chunks: list[str] = [first_clean]
    pasted = (_PASTE_START in first) and (_PASTE_END not in first)
    burst_read = False

    if is_tty:
        try:
            fd = sys.stdin.fileno()
            raw_tail = bytearray()
            while True:
                wait_sec = 0.2 if pasted else 0.045
                ready, _, _ = select.select([fd], [], [], wait_sec)
                if not ready:
                    break
                data = os.read(fd, 4096)
                if not data:
                    break
                raw_tail.extend(data)
                burst_read = True
                dec_so_far = raw_tail.decode("utf-8", errors="replace")
                if _PASTE_END in dec_so_far:
                    break
            if raw_tail:
                tail_txt = raw_tail.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
                tail_txt = _strip_paste_sequences(tail_txt).rstrip("\n")
                if tail_txt:
                    chunks.append(tail_txt)
        except Exception:
            pass

    combined = "\n".join(chunks)
    if (burst_read or "\n" in combined) and len(combined.splitlines()) > 1:
        print(f"\033[90m  [pasted {len(combined.splitlines())} lines]\033[0m")
        return combined.strip()

    cur = combined
    if cur.rstrip().endswith("\\"):
        parts = [cur.rstrip()[:-1]]
        while True:
            nxt = _strip_paste_sequences(input("... "))
            if nxt.rstrip().endswith("\\"):
                parts.append(nxt.rstrip()[:-1])
            else:
                parts.append(nxt)
                break
        return "\n".join(parts).strip()

    for delim in ('"""', "'''"):
        if cur.count(delim) % 2 == 1:
            parts = [cur]
            while True:
                nxt = _strip_paste_sequences(input("... "))
                parts.append(nxt)
                if ("\n".join(parts)).count(delim) % 2 == 0:
                    break
            return "\n".join(parts).strip()

    return cur.strip()


def _setup_shell_readline(get_cwd=None) -> Path:
    """Configure persistent readline history and slash/@path tab-completion."""
    hist_file = DATA_DIR / "chat_history"
    hist_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        import glob
        import readline

        if hist_file.is_file():
            readline.read_history_file(str(hist_file))
        readline.set_history_length(500)

        slash_cmds = [
            "/help", "/status", "/doctor", "/digest", "/live", "/new", "/reset", "/sessions", "/resume",
            "/paste", "/model", "/undo", "/fix", "/clear", "/prompts", "/local",
            "/neural", "/evolve", "/brain", "/exit", "/quit",
        ]

        def _completer(text: str, state: int) -> str | None:
            if text.startswith("/"):
                matches = [c for c in slash_cmds if c.startswith(text)]
            elif text.startswith("@"):
                raw_p = text[1:]
                base_dir = get_cwd() if callable(get_cwd) else Path.cwd()
                if raw_p.startswith("~"):
                    exp = str(Path(raw_p).expanduser())
                    matches = ["@" + m for m in glob.glob(exp + "*")]
                elif raw_p.startswith("/"):
                    matches = ["@" + m for m in glob.glob(raw_p + "*")]
                else:
                    cand = sorted(glob.glob(str(base_dir / (raw_p + "*"))))
                    matches = []
                    for c in cand:
                        try:
                            rel = Path(c).relative_to(base_dir)
                            matches.append(f"@{rel}")
                        except ValueError:
                            matches.append(f"@{c}")
            else:
                expanded = str(Path(text).expanduser()) if text.startswith("~") else text
                matches = glob.glob(expanded + "*")
            return matches[state] if state < len(matches) else None

        readline.set_completer_delims(" \t\n;")
        readline.set_completer(_completer)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass
    return hist_file


def _short_cwd(cwd: Path) -> str:
    home = Path.home()
    try:
        rel = cwd.resolve().relative_to(home)
        return "~" if str(rel) == "." else f"~/{rel}"
    except ValueError:
        return str(cwd)


def record_last_shell_error(
    command: str, exit_code: int, cwd: str, stderr_text: str = ""
) -> Path:
    """Persist the latest failed terminal command, trailing stderr, repeat count, and pre-matched 0-token fix."""
    from datetime import datetime

    err_file = DATA_DIR / "last_error.json"
    err_file.parent.mkdir(parents=True, exist_ok=True)
    clean_cmd = str(command or "").strip()[:1000]
    code_int = int(exit_code or 1)
    stderr_clean = "\n".join((stderr_text or "").splitlines()[-30:])[-2500:]

    repeat_count = 1
    if err_file.is_file():
        try:
            prev = json.loads(err_file.read_text(encoding="utf-8", errors="replace"))
            if isinstance(prev, dict) and str(prev.get("command") or "") == clean_cmd:
                repeat_count = int(prev.get("repeat_count") or 1) + 1
        except Exception:
            pass

    instant_fix: dict[str, object] | None = None
    recovery_hint = ""
    try:
        from .self_guard import match_cached_fix
        from .self_resilience import build_recovery_hint

        with Storage() as st:
            hit = match_cached_fix(st, clean_cmd, code_int)
            if isinstance(hit, dict) and hit.get("lesson"):
                instant_fix = dict(hit)
            hint_val = build_recovery_hint(
                st,
                "run_command",
                {"command": clean_cmd},
                json.dumps({"exit_code": code_int, "stderr": stderr_clean}, ensure_ascii=False),
            )
            if hint_val:
                recovery_hint = str(hint_val)
            if repeat_count >= 2 and (instant_fix or recovery_hint):
                hint_txt = str(instant_fix.get("lesson") if instant_fix else recovery_hint)[:160]
                st.add_brain_thought(
                    "info",
                    70,
                    f"Repeated shell failure (x{repeat_count}) on `{clean_cmd}` -> instant fix ready: {hint_txt}",
                )
    except Exception:
        pass

    payload: dict[str, object] = {
        "command": clean_cmd,
        "exit_code": code_int,
        "cwd": str(cwd or "."),
        "stderr": stderr_clean,
        "repeat_count": repeat_count,
        "instant_fix": instant_fix,
        "recovery_hint": recovery_hint,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    tmp = err_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, err_file)
    return err_file


async def _interactive_chat(
    base_url: str,
    locale: str | None = None,
    initial_session_id: str | None = None,
) -> None:
    settings = get_settings()
    session_id = initial_session_id or _load_last_session_id()
    agent_lbl = settings.agent_name.lower()
    cwd = Path.cwd()
    hist_file = _setup_shell_readline(get_cwd=lambda: cwd)
    pinned_model: str | None = None

    print(
        f"\033[1;36m{settings.agent_name} Shell\033[0m "
        f"\033[90m(session: {session_id} | /help commands, @file attach, !<cmd> bash, Ctrl+C cancel turn)\033[0m\n"
    )
    try:
        while True:
            model_tag = f" \001\033[33m\002[{pinned_model}]\001\033[0m\002" if pinned_model else ""
            prompt_str = (
                f"\001\033[1;35m\002{agent_lbl}\001\033[0m\002 "
                f"\001\033[34m\002{_short_cwd(cwd)}\001\033[0m\002{model_tag} "
                f"\001\033[1;32m\002❯\001\033[0m\002 "
            )
            try:
                raw_line = _read_chat_input(prompt_str)
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("^C")
                continue

            if not raw_line:
                continue
            low = raw_line.lower()
            if low in {"exit", "quit", "/exit", "/quit"}:
                break

            if raw_line.startswith("!") and low != "!clip":
                cmd = raw_line[1:].strip()
                if cmd:
                    proc = await asyncio.to_thread(
                        subprocess.run,
                        cmd,
                        shell=True,
                        cwd=str(cwd),
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if proc.stderr:
                        sys.stderr.write(proc.stderr)
                        sys.stderr.flush()
                    if proc.returncode != 0:
                        print(f"\033[31m[exit {proc.returncode}]\033[0m")
                        record_last_shell_error(cmd, proc.returncode, str(cwd), proc.stderr or "")
                continue

            if raw_line == "cd" or raw_line.startswith("cd "):
                target = raw_line[2:].strip() or "~"
                new_dir = (
                    (cwd / Path(target).expanduser()).resolve()
                    if not target.startswith(("~", "/"))
                    else Path(target).expanduser().resolve()
                )
                if new_dir.is_dir():
                    cwd = new_dir
                    os.chdir(cwd)
                else:
                    print(f"\033[31mcd: no such directory: {target}\033[0m")
                continue

            if low == "!clip" or raw_line.startswith("/paste"):
                clip = _read_clipboard_text().strip()
                if not clip:
                    print("\033[33mClipboard is empty or wl-paste/xclip is unavailable.\033[0m")
                    continue
                extra_inst = "" if low == "!clip" else raw_line[len("/paste"):].strip()
                raw_line = f"{extra_inst}\n\n{clip}" if extra_inst else clip
                print(f"\033[90m  [loaded clipboard: {len(clip.splitlines())} lines, {len(clip)} chars]\033[0m")
            elif raw_line.startswith("/"):
                parts = raw_line.split()
                cmd = parts[0].lower()
                if cmd == "/help":
                    print(
                        "Shell commands:\n"
                        "  !<cmd>           Run a bash command directly without LLM\n"
                        "  @<path>          Attach file snippet or folder listing to your prompt (Tab completes)\n"
                        "  /paste [note]    Send clipboard content cleanly as one multi-line prompt (!clip)\n"
                        "  /model [name]    Show or switch active chat model (hydrogen, argon, flash, lite)\n"
                        "  /sessions        List recent saved chat sessions\n"
                        "  /resume <id>     Resume an earlier session by id or prefix\n"
                        "  /new | /reset    Start a fresh chat session (/reset clears old history)\n"
                        "  cd <dir>         Change active working directory\n"
                        "  /status          Show daemon & system health\n"
                        "  /doctor          Run full system & component diagnostics\n"
                        "  /digest          Show daily system & brain digest\n"
                        "  /live            Show live daemon, brain thoughts & evolve activity\n"
                        "  /undo            Revert the last turn/batch file changes\n"
                        "  /fix             Analyze last failed terminal command\n"
                        "  /prompts         Show prompt registry status\n"
                        "  /local           Show local self-trained model status\n"
                        "  /neural [act]    Neural engine status, train, or pulse <query>\n"
                        "  /clear           Clear terminal screen\n"
                        "  /exit            Quit shell (press Ctrl+C during a reply to cancel the turn)"
                    )
                elif cmd == "/clear":
                    print("\033[2J\033[H", end="")
                elif cmd == "/status":
                    await asyncio.to_thread(_cmd_status, base_url, locale)
                elif cmd == "/doctor":
                    await asyncio.to_thread(_cmd_doctor, base_url)
                elif cmd == "/digest":
                    await asyncio.to_thread(_cmd_digest, base_url, locale)
                elif cmd == "/model":
                    if len(parts) < 2:
                        try:
                            resp_st = await asyncio.to_thread(
                                ipc_get, f"{base_url}/api/status", headers=_auth_headers(), timeout=5
                            )
                            resp_st.raise_for_status()
                            st = resp_st.json()
                            m_info = st.get("models") or {}
                            print(
                                f"Active override: {pinned_model or 'auto (per-turn routing)'} | "
                                f"default chat={m_info.get('chat', '-')}, light={m_info.get('light', '-')}"
                            )
                            print(f"Aliases: {', '.join(f'{k}={v}' for k, v in _MODEL_ALIASES.items())}")
                        except Exception as exc:
                            print(f"Active override: {pinned_model or 'auto'} ({exc})")
                    elif parts[1].lower() in {"auto", "default", "reset"}:
                        pinned_model = None
                        try:
                            async with ipc_ws_connect(_ws_url(base_url, session_id), open_timeout=5) as ws:
                                await ws.send(json.dumps({"type": "set_model", "model": "auto"}))
                                await asyncio.wait_for(ws.recv(), timeout=3)
                        except Exception:
                            pass
                        print("\033[90mModel override cleared (back to adaptive routing).\033[0m")
                    else:
                        chosen = _MODEL_ALIASES.get(parts[1].lower(), parts[1].strip())
                        pinned_model = chosen
                        try:
                            async with ipc_ws_connect(_ws_url(base_url, session_id), open_timeout=5) as ws:
                                await ws.send(json.dumps({"type": "set_model", "model": chosen}))
                                reply_ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
                                print(f"\033[90m{reply_ev.get('message', f'Model set to {chosen}')}\033[0m")
                        except Exception:
                            print(f"\033[90mModel pinned locally to {chosen} (will apply on next message).\033[0m")
                elif cmd == "/sessions":
                    try:
                        resp_s = await asyncio.to_thread(
                            ipc_get, f"{base_url}/api/sessions", headers=_auth_headers(), timeout=5
                        )
                        resp_s.raise_for_status()
                        data = resp_s.json()
                        for s in (data.get("sessions") or [])[:15]:
                            sid_item = s.get("session_id", "")
                            mark = "*" if sid_item == session_id else " "
                            ts = str(s.get("updated_at") or "")[:19].replace("T", " ")
                            title_txt = " ".join(str(s.get("title") or "").split())[:50]
                            print(f" {mark} {sid_item:<18} [{ts}] {title_txt}")
                    except Exception as exc:
                        print(f"\033[31m{exc}\033[0m")
                elif cmd == "/resume":
                    if len(parts) < 2:
                        print("\033[33mUsage: /resume <session_id_or_topic>\033[0m")
                        continue
                    query_txt = " ".join(parts[1:]).strip()
                    try:
                        res_data: dict[str, object] | None = None
                        try:
                            resp_res = await asyncio.to_thread(
                                ipc_post,
                                f"{base_url}/api/sessions/resume",
                                json={"query": query_txt},
                                headers=_auth_headers(),
                                timeout=5,
                            )
                            resp_res.raise_for_status()
                            res_data = resp_res.json()
                        except Exception:
                            res_data = None
                        if not res_data or not res_data.get("found"):
                            resp_sl = await asyncio.to_thread(
                                ipc_get, f"{base_url}/api/sessions", headers=_auth_headers(), timeout=5
                            )
                            resp_sl.raise_for_status()
                            sessions = resp_sl.json().get("sessions") or []
                            match = next(
                                (
                                    s["session_id"]
                                    for s in sessions
                                    if s["session_id"] == query_txt or s["session_id"].startswith(query_txt)
                                ),
                                None,
                            )
                            if not match:
                                print(f"\033[33mNo session matching '{query_txt}'. Use /sessions to list.\033[0m")
                                continue
                            resp_m = await asyncio.to_thread(
                                ipc_get,
                                f"{base_url}/api/sessions/{match}/messages",
                                headers=_auth_headers(),
                                timeout=5,
                            )
                            resp_m.raise_for_status()
                            msgs = resp_m.json().get("messages") or []
                            ctx_prev = ""
                        else:
                            match = str(res_data["session_id"])
                            msgs = list(res_data.get("messages") or [])
                            ctx_prev = str(res_data.get("context_preview") or "")
                        session_id = match
                        _save_last_session_id(session_id)
                        print(f"\033[1;36mResumed {session_id}\033[0m ({len(msgs)} messages):")
                        if ctx_prev:
                            print(f"  \033[36m[Checkpoint]\033[0m {ctx_prev}")
                        for m in msgs[-4:]:
                            snippet = " ".join(str(m.get("content") or "").split())[:100]
                            print(f"  \033[90m{m.get('role')}: {snippet}\033[0m")
                    except Exception as exc:
                        print(f"\033[31m{exc}\033[0m")
                elif cmd == "/new":
                    session_id = f"cli-{uuid.uuid4().hex[:8]}"
                    _save_last_session_id(session_id)
                    print(f"\033[90mStarted new session ({session_id}).\033[0m")
                elif cmd == "/live":
                    try:
                        hdrs = _auth_headers()
                        r_st, r_br, r_ev = await asyncio.gather(
                            asyncio.to_thread(ipc_get, f"{base_url}/api/status", headers=hdrs, timeout=5),
                            asyncio.to_thread(ipc_get, f"{base_url}/api/brain", headers=hdrs, timeout=5),
                            asyncio.to_thread(ipc_get, f"{base_url}/api/evolve", headers=hdrs, timeout=5),
                        )
                        for r_item in (r_st, r_br, r_ev):
                            r_item.raise_for_status()
                        st, br, ev = r_st.json(), r_br.json(), r_ev.json()
                        print(f"\033[1;36mDaemon:\033[0m {st.get('status')} | model={st.get('model')} | auth={st.get('auth_mode') or 'none'}")
                        thoughts = br.get("thoughts") or []
                        print(f"\033[1;36mBrain thoughts\033[0m (enabled={br.get('enabled')}, total={len(thoughts)}):")
                        for th in thoughts[:5]:
                            ts = str(th.get("created_at", ""))[:19].replace("T", " ")
                            print(f"  [{ts}] ({th.get('decision')}, {th.get('confidence', 0):.2f}) {th.get('summary', '')}")
                        if not thoughts:
                            print("  (none yet)")
                        evo = st.get("evolve") or {}
                        if evo:
                            rate = evo.get("success_rate")
                            nxt = evo.get("next_run_in_sec")
                            print(
                                f"\033[1;36mEvolve stats:\033[0m success={int(rate * 100) if rate is not None else 'n/a'}% "
                                f"| next scan {('in ' + _fmt_uptime(nxt)) if nxt is not None else 'pending'} "
                                f"| last: {(evo.get('last_result') or {}).get('status', '-')}"
                            )
                        props = ev.get("proposals") or []
                        print(f"\033[1;36mSelf-evolve\033[0m (enabled={ev.get('enabled')}, auto_apply={ev.get('auto_apply')}, proposals={len(props)}):")
                        for pr in props[:3]:
                            print(f"  #{pr.get('id')} [{pr.get('status')}] ({pr.get('category')}) {pr.get('title')}")
                        if not props:
                            print("  (none yet)")
                    except Exception as exc:
                        print(f"\033[31m{exc}\033[0m")
                elif cmd in {"/prompts", "/local", "/neural", "/brain", "/evolve", "/fix"}:
                    try:
                        if cmd == "/prompts":
                            await asyncio.to_thread(_cmd_prompts, base_url, "list", parts[1] if len(parts) > 1 else None, locale)
                        elif cmd == "/local":
                            await asyncio.to_thread(_cmd_local, base_url, parts[1] if len(parts) > 1 else "status", " ".join(parts[2:]) if len(parts) > 2 else None, locale)
                        elif cmd == "/neural":
                            await asyncio.to_thread(_cmd_neural, base_url, parts[1] if len(parts) > 1 else "status", " ".join(parts[2:]) if len(parts) > 2 else None, locale)
                        elif cmd == "/brain":
                            await asyncio.to_thread(_cmd_brain, base_url, parts[1] if len(parts) > 1 else "status", locale)
                        elif cmd == "/evolve":
                            ev_act = parts[1] if len(parts) > 1 else "status"
                            if ev_act == "run":
                                await _stream_evolve(base_url, locale)
                            else:
                                await asyncio.to_thread(_cmd_evolve, base_url, ev_act, locale)
                        elif cmd == "/fix":
                            await _cmd_fix_async(base_url, locale)
                    except SystemExit:
                        pass
                elif cmd == "/undo":
                    try:
                        r = await asyncio.to_thread(
                            ipc_post, f"{base_url}/api/undo-batch", headers=_auth_headers(), timeout=10
                        )
                        r.raise_for_status()
                        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
                    except Exception as exc:
                        print(f"\033[31m{exc}\033[0m")
                elif cmd == "/reset":
                    try:
                        await asyncio.to_thread(
                            ipc_post, f"{base_url}/api/reset/{session_id}", headers=_auth_headers(), timeout=5
                        )
                    except Exception:
                        pass
                    session_id = f"cli-{uuid.uuid4().hex[:8]}"
                    _save_last_session_id(session_id)
                    print(f"\033[90mSession reset ({session_id}).\033[0m")
                else:
                    print(f"\033[33mUnknown command: {cmd} (try /help)\033[0m")
                continue

            expanded_prompt, attached = _expand_at_paths(raw_line, cwd)
            if attached:
                print(f"\033[90m  [attached: {', '.join(attached)}]\033[0m")
            final_prompt = expanded_prompt if cwd == Path.home() else f"[cwd: {cwd}] {expanded_prompt}"
            new_m = await _send_and_stream(
                final_prompt, session_id, base_url, locale, model_override=pinned_model
            )
            if new_m is not None:
                pinned_model = None if new_m == "auto" else new_m
    finally:
        if sys.stdin.isatty():
            try:
                sys.stdout.write(_BRACKETED_PASTE_OFF)
                sys.stdout.flush()
            except Exception:
                pass
        try:
            import readline

            readline.write_history_file(str(hist_file))
        except Exception:
            pass


def _fmt_uptime(sec: int) -> str:
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


from .cli_ai import (
    _cmd_brain,
    _cmd_evolve,
    _cmd_local,
    _cmd_neural,
    _cmd_prompts,
    _stream_evolve as _stream_evolve,
)
from .cli_ops import (
    _cmd_cortex,
    _cmd_digest,
    _cmd_doctor,
    _cmd_notifications,
    _cmd_plugin,
    _cmd_rollback,
    _cmd_schedule,
    _cmd_search,
    _parse_schedule_spec as _parse_schedule_spec,
    _prog_name,
)
from .cli_shell import (  # noqa: F401
    _cmd_events,
    _cmd_fix,
    _cmd_fix_async,
    _cmd_hook,
    _cmd_name,
    _cmd_resume,
    _cmd_self,
    _cmd_status,
    _install_shell_hook,
    _prepare_fix_prompt,
    _quick_launcher,
    _render_shell_hook,
    _resolve_hook_shell,
)


def main() -> None:
    from .logging_setup import configure_logging

    configure_logging(daemon=False)
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog=_prog_name(),
        description=f"{settings.agent_name} — Personal Linux system daemon and AI assistant",
    )
    sub = parser.add_subparsers(dest="subcommand")
    sub.add_parser("serve", help="Start the daemon server in the foreground")
    sub.add_parser("chat", help="Start interactive terminal chat")
    sub.add_parser("status", help="Show daemon and system status")
    sub.add_parser("digest", help="Show daily system & brain digest")
    doc_parser = sub.add_parser("doctor", help="Run health and configuration diagnostics (use --fix for autonomous repair)")
    doc_parser.add_argument("--fix", action="store_true", help="Run deep self-integrity audit and auto-repair")
    plug_parser = sub.add_parser("plugin", help="Manage user plugins (list/enable/disable/remove)")
    plug_parser.add_argument("plugin_action", nargs="?", default="list", choices=["list", "enable", "disable", "remove"])
    plug_parser.add_argument("plugin_name", nargs="?", default=None, help="Plugin filename or stem")
    srch_parser = sub.add_parser("search", help="Full-text search across memories, lessons, and chat history")
    srch_parser.add_argument("query", nargs="+", help="Search query")
    srch_parser.add_argument("-n", "--limit", type=int, default=12)
    ev_parser = sub.add_parser("events", help="Show background listener events")
    ev_parser.add_argument("-n", "--limit", type=int, default=20)
    name_parser = sub.add_parser("name", help="Set or update the assistant's name")
    name_parser.add_argument("new_name", nargs="+", help="New name (e.g. Jarvis)")
    sub.add_parser("quick", help="Open desktop quick command launcher")
    fix_parser = sub.add_parser("fix", help="Analyze the last failed terminal command and suggest a fix")
    fix_parser.add_argument("--no-cache", action="store_true", help="Skip cached fix lessons and query the model")
    fix_parser.add_argument("fix_command", nargs="*", help="Optional command text when no hook error was recorded")
    hook_parser = sub.add_parser("hook", help="Print, install, or invoke the Bash/Zsh error-capturing shell hook")
    hook_parser.add_argument("hook_shell", nargs="?", default="auto", choices=["auto", "bash", "zsh"])
    hook_parser.add_argument("--install", action="store_true", help="Install/update the hook in ~/.bashrc or ~/.zshrc")
    hook_parser.add_argument("--record", nargs=3, metavar=("EXIT_CODE", "CWD", "CMD"), default=None, help="Record failed command")
    resume_parser = sub.add_parser("resume", help="List or semantically resume an earlier session by topic or ID")
    resume_parser.add_argument("resume_query", nargs="*", help="Session ID, prefix, or topic keywords")
    resume_parser.add_argument("--list", action="store_true", dest="resume_list", help="List saved semantic checkpoints")
    resume_parser.add_argument("--new", action="store_true", dest="resume_new", help="Fork checkpoint into a fresh session ID")
    brain_parser = sub.add_parser("brain", help="Autonomous brain status / think / rules / digest / log")
    brain_parser.add_argument("brain_action", nargs="?", default="status", choices=["status", "think", "rules", "digest", "log"])
    evolve_parser = sub.add_parser("evolve", help="Self-evolution status / scan / log")
    evolve_parser.add_argument("evolve_action", nargs="?", default="status", choices=["status", "run", "log"])
    prompts_parser = sub.add_parser("prompts", help="List, evolve, or rollback centralized system prompts")
    prompts_parser.add_argument("prompts_action", nargs="?", default="list", choices=["list", "evolve", "rollback"])
    prompts_parser.add_argument("prompt_key", nargs="?", default=None, help="Target prompt key (e.g. agent_main)")
    local_parser = sub.add_parser("local", help="Self-training local offline model status, train, distill, curriculum, download, bench, or test")
    local_parser.add_argument("local_action", nargs="?", default="status", choices=["status", "train", "distill", "curriculum", "download", "test", "bench", "prune", "profile"])
    local_parser.add_argument("local_query", nargs="*", default=None, help="Optional test query")
    neural_parser = sub.add_parser("neural", help="Vectorized Neural Engine status, train, or pulse inspection")
    neural_parser.add_argument("neural_action", nargs="?", default="status", choices=["status", "train", "pulse", "activate"])
    neural_parser.add_argument("neural_query", nargs="*", default=None, help="Query text for neural pulse inspection")
    login_parser = sub.add_parser("login", help="Authenticate with Google OAuth / Antigravity CLI")
    login_parser.add_argument("--client-secret", default=None, help="Path to Google Cloud OAuth client_secret.json")
    login_parser.add_argument("--project", default=None, help="Google Cloud Project ID")
    sub.add_parser("logout", help="Remove saved OAuth session")
    exp_parser = sub.add_parser("export", help="Export brain rules, memories, and plugins as JSON")
    exp_parser.add_argument("file", help="Target JSON file path")
    imp_parser = sub.add_parser("import", help="Import a backed-up brain JSON bundle")
    imp_parser.add_argument("file", help="Source JSON file path")
    top_parser = sub.add_parser("top", help="Live terminal dashboard showing system & agent telemetry")
    top_parser.add_argument("-n", "--interval", type=float, default=1.5, help="Refresh interval in seconds")
    top_parser.add_argument("--once", action="store_true", help="Render one frame and exit")
    rb_parser = sub.add_parser("rollback", help="List or restore file, config, or evolve snapshots")
    rb_parser.add_argument("rollback_target", nargs="?", default="auto", choices=["auto", "list", "file", "evolve", "config"])
    rb_parser.add_argument("rollback_id", nargs="?", default=None, help="Optional snapshot, batch, or evolve proposal ID")
    rb_parser.add_argument("--list", action="store_true", dest="rollback_list", help="List available snapshots")
    sub.add_parser("mcp", help="Run SysAgent as a stdio MCP (Model Context Protocol) server")
    sched_parser = sub.add_parser("schedule", help="List, add, toggle, or delete scheduled automations")
    sched_parser.add_argument("schedule_action", nargs="?", default="list", choices=["list", "add", "delete", "toggle"])
    sched_parser.add_argument("schedule_args", nargs="*", help="Arguments for add/delete/toggle")
    notif_parser = sub.add_parser("notifications", help="Inspect recent notifications or send a test notification")
    notif_parser.add_argument("notif_action", nargs="?", default="list", choices=["list", "test"])
    notif_parser.add_argument("notif_args", nargs="*", help="Optional message for test notification")
    notif_parser.add_argument("-n", "--limit", type=int, default=20)
    cortex_parser = sub.add_parser("cortex", help="Neural+Brain Cortex: status, dream cycle, predictive heal, habits, flow")
    cortex_parser.add_argument("cortex_action", nargs="?", default="status", choices=["status", "dream", "heal", "habits", "flow"])
    cortex_parser.add_argument("cortex_arg", nargs="?", default=None, help="Optional flow mode (deep_focus|meeting_media|idle_maintenance|normal|auto)")
    self_parser = sub.add_parser("self", help="Self-Autopsy, Self-Canary, Self-Calibration, Self-Profiler, Eval, Drift, Breakers, Quota, Twin, Blast, DPO, and Self-Skills")
    self_parser.add_argument("self_action", nargs="?", default="status", choices=["status", "canary", "calibrate", "profile", "autopsy", "skills", "eval", "drift", "breakers", "quota", "twin", "blast", "dpo"])
    self_parser.add_argument("self_target", nargs="?", default=None, help="Optional symbol or file target for blast radius analysis")

    locale_override: str | None = None
    raw_argv = sys.argv[1:]
    filtered_argv: list[str] = []
    i = 0
    while i < len(raw_argv):
        if raw_argv[i] in {"-l", "--locale"} and i + 1 < len(raw_argv):
            locale_override = raw_argv[i + 1]
            i += 2
        elif raw_argv[i].startswith("--locale="):
            locale_override = raw_argv[i].split("=", 1)[1]
            i += 1
        else:
            filtered_argv.append(raw_argv[i])
            i += 1

    if filtered_argv and filtered_argv[0] in {"serve", "chat", "status", "digest", "doctor", "plugin", "search", "events", "name", "quick", "fix", "hook", "resume", "brain", "evolve", "prompts", "local", "neural", "login", "logout", "export", "import", "top", "rollback", "mcp", "schedule", "notifications", "cortex", "self", "-h", "--help"}:
        args = parser.parse_args(filtered_argv)
        if args.subcommand == "serve":
            sock = Path(settings.socket_path)
            sock.parent.mkdir(parents=True, exist_ok=True)
            sock.unlink(missing_ok=True)
            uvicorn.run(
                "sysagent.server:app",
                uds=str(sock),
                log_level="info",
            )
        elif args.subcommand == "chat":
            asyncio.run(_interactive_chat(settings.base_url, locale_override))
        elif args.subcommand == "status":
            _cmd_status(settings.base_url, locale_override)
        elif args.subcommand == "digest":
            _cmd_digest(settings.base_url, locale_override)
        elif args.subcommand == "doctor":
            _cmd_doctor(settings.base_url, fix=bool(getattr(args, "fix", False)))
        elif args.subcommand == "plugin":
            _cmd_plugin(args.plugin_action, args.plugin_name)
        elif args.subcommand == "search":
            _cmd_search(settings.base_url, " ".join(args.query), args.limit, locale_override)
        elif args.subcommand == "events":
            _cmd_events(settings.base_url, args.limit, locale_override)
        elif args.subcommand == "name":
            _cmd_name(settings.base_url, " ".join(args.new_name), locale_override)
        elif args.subcommand == "quick":
            asyncio.run(_quick_launcher(settings.base_url, locale_override))
        elif args.subcommand == "fix":
            fc = " ".join(args.fix_command).strip() if args.fix_command else None
            _cmd_fix(settings.base_url, locale_override, extra_cmd=fc, no_cache=bool(args.no_cache))
        elif args.subcommand == "hook":
            _cmd_hook(args.hook_shell, install=bool(args.install), record_args=args.record)
        elif args.subcommand == "resume":
            _cmd_resume(
                settings.base_url,
                args.resume_query,
                list_only=bool(args.resume_list),
                fork_new=bool(args.resume_new),
                locale=locale_override,
            )
        elif args.subcommand == "brain":
            _cmd_brain(settings.base_url, args.brain_action, locale_override)
        elif args.subcommand == "evolve":
            _cmd_evolve(settings.base_url, args.evolve_action, locale_override)
        elif args.subcommand == "prompts":
            _cmd_prompts(settings.base_url, args.prompts_action, args.prompt_key, locale_override)
        elif args.subcommand == "local":
            lq = " ".join(args.local_query).strip() if args.local_query else None
            _cmd_local(settings.base_url, args.local_action, lq, locale_override)
        elif args.subcommand == "neural":
            nq = " ".join(args.neural_query).strip() if args.neural_query else None
            _cmd_neural(settings.base_url, args.neural_action, nq, locale_override)
        elif args.subcommand == "login":
            try:
                tok_path = run_oauth_login(settings, args.client_secret, args.project)
                try:
                    ipc_post(f"{settings.base_url}/api/auth/reload", headers=_auth_headers(), timeout=3)
                except Exception:
                    pass
                st = get_settings()
                print(f"OAuth/ADC login successful. Credentials: {tok_path} (project={st.google_cloud_project or 'auto'})")
            except Exception as exc:
                print(f"OAuth login error: {exc}", file=sys.stderr)
                sys.exit(1)
        elif args.subcommand == "logout":
            removed = logout_oauth(settings)
            print("OAuth token removed." if removed else "No OAuth token found.")
        elif args.subcommand == "export":
            bundle = export_brain_bundle(Storage())
            Path(args.file).write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Exported brain bundle to {args.file}")
        elif args.subcommand == "import":
            bundle = json.loads(Path(args.file).read_text(encoding="utf-8"))
            counts = import_brain_bundle(bundle, Storage())
            print(f"Imported brain bundle: {counts}")
        elif args.subcommand == "top":
            from .top import run_top

            run_top(settings.base_url, _auth_headers(), interval=args.interval, once=args.once)
        elif args.subcommand == "rollback":
            _cmd_rollback(
                settings.base_url,
                args.rollback_target,
                args.rollback_id,
                args.rollback_list,
                locale_override,
            )
        elif args.subcommand == "mcp":
            from .mcp_server import run_mcp_stdio

            run_mcp_stdio()
        elif args.subcommand == "schedule":
            _cmd_schedule(
                settings.base_url,
                args.schedule_action,
                args.schedule_args,
                locale_override,
            )
        elif args.subcommand == "notifications":
            _cmd_notifications(
                settings.base_url,
                args.notif_action,
                args.notif_args,
                args.limit,
                locale_override,
            )
        elif args.subcommand == "cortex":
            _cmd_cortex(
                settings.base_url,
                args.cortex_action,
                args.cortex_arg,
                locale_override,
            )
        elif args.subcommand == "self":
            _cmd_self(settings.base_url, args.self_action, locale_override, args.self_target)
        return

    piped_text = ""
    if not sys.stdin.isatty():
        try:
            piped_text = sys.stdin.read(24576).strip()
        except Exception:
            piped_text = ""

    if not filtered_argv and not piped_text:
        asyncio.run(_interactive_chat(settings.base_url, locale_override))
        return

    user_arg = " ".join(filtered_argv).strip()
    if piped_text and user_arg:
        prompt_text = f"{user_arg}\n\n```text\n{piped_text}\n```"
    elif piped_text:
        prompt_text = f"Analyze this piped input:\n\n```text\n{piped_text}\n```"
    else:
        prompt_text = user_arg
    session_id = f"oneshot-{uuid.uuid4().hex[:8]}"
    asyncio.run(_send_and_stream(prompt_text, session_id, settings.base_url, locale_override))


if __name__ == "__main__":
    main()

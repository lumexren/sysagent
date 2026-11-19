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
from .config import DATA_DIR, get_settings, persist_env_var
from .i18n import t
from .ipc import ipc_delete, ipc_get, ipc_post, ipc_ws_connect
from .self_ops import export_brain_bundle, import_brain_bundle
from .storage import Storage


_BRACKETED_PASTE_ON = "\033[?2004h"
_BRACKETED_PASTE_OFF = "\033[?2004l"
_PASTE_START = "\x1b[200~"
_PASTE_END = "\x1b[201~"

_MODEL_ALIASES: dict[str, str] = {
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


def _expand_at_paths(text: str, cwd: Path, max_lines: int = 120) -> tuple[str, list[str]]:
    """Expand @path tokens in user prompts into attached file snippets or directory listings."""
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
) -> None:
    ws_url = _ws_url(base_url, session_id)
    loc = locale or get_settings().locale
    spin_label = ["thinking..."]

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
                nonlocal stop_spin, spin_task, handled_final
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
                        _flush_stdin()
                        stop_stdin = threading.Event()
                        stdin_task = asyncio.create_task(
                            asyncio.to_thread(_read_line_cancelable, t("cli_confirm", loc), stop_stdin)
                        )
                        ws_task = asyncio.create_task(ws.recv())
                        done, _ = await asyncio.wait(
                            {stdin_task, ws_task}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if stdin_task in done:
                            ws_task.cancel()
                            try:
                                await ws_task
                            except BaseException:
                                pass
                            ans = (stdin_task.result() or "").strip().lower()
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
                        else:
                            stop_stdin.set()
                            await stdin_task
                            ws_raw = ws_task.result()
                            ws_ev = json.loads(ws_raw)
                            if (
                                ws_ev.get("type") == "approval_resolved"
                                and ws_ev.get("approval_id") == event["approval_id"]
                            ):
                                approved = bool(ws_ev.get("approved"))
                                lbl = t("cli_ok", loc) if approved else t("cli_denied", loc)
                                col = "\033[32m" if approved else "\033[31m"
                                print(f"\n{col}[{lbl} via notification]\033[0m")
                            else:
                                queued_raw.append(ws_raw)
                        spin_label[0] = f"running {event['tool_name']}..."
                        stop_spin = asyncio.Event()
                        spin_task = asyncio.create_task(_spinner(stop_spin))
                    elif etype == "approval_resolved":
                        pass
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
            "/help", "/status", "/live", "/new", "/reset", "/sessions", "/resume",
            "/paste", "/model", "/undo", "/fix", "/clear", "/prompts", "/local",
            "/neural", "/evolve", "/brain", "/exit",
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


async def _interactive_chat(base_url: str, locale: str | None = None) -> None:
    settings = get_settings()
    session_id = _load_last_session_id()
    agent_lbl = settings.agent_name.lower()
    cwd = Path.cwd()
    hist_file = _setup_shell_readline(get_cwd=lambda: cwd)
    pinned_model: str | None = None

    print(
        f"\033[1;36m{settings.agent_name} Shell\033[0m "
        f"\033[90m(session: {session_id} | /help commands, @file attach, !<cmd> bash, Ctrl+C cancel turn)\033[0m\n"
    )
    try:
        import readline

        _slash_cmds = (
            "/help", "/model", "/paste", "/resume", "/sessions", "/new", "/clear",
            "/status", "/doctor", "/digest", "/live", "/brain", "/evolve", "/neural",
            "/local", "/prompts", "/undo", "/fix", "/reset", "/exit", "/quit",
        )

        def _slash_completer(text: str, state: int) -> str | None:
            if not text.startswith("/"):
                return None
            matches = [c for c in _slash_cmds if c.startswith(text)]
            return matches[state] if state < len(matches) else None

        readline.set_completer_delims(" \t\n")
        readline.set_completer(_slash_completer)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass
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
                    proc = await asyncio.to_thread(subprocess.run, cmd, shell=True, cwd=str(cwd))
                    if proc.returncode != 0:
                        print(f"\033[31m[exit {proc.returncode}]\033[0m")
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
                        print("\033[33mUsage: /resume <session_id_or_prefix>\033[0m")
                        continue
                    prefix = parts[1].strip()
                    try:
                        resp_sl = await asyncio.to_thread(
                            ipc_get, f"{base_url}/api/sessions", headers=_auth_headers(), timeout=5
                        )
                        sessions = resp_sl.json().get("sessions") or []
                        match = next(
                            (s["session_id"] for s in sessions if s["session_id"] == prefix or s["session_id"].startswith(prefix)),
                            None,
                        )
                        if not match:
                            print(f"\033[33mNo session matching '{prefix}'. Use /sessions to list.\033[0m")
                            continue
                        session_id = match
                        _save_last_session_id(session_id)
                        resp_m = await asyncio.to_thread(
                            ipc_get, f"{base_url}/api/sessions/{session_id}/messages", headers=_auth_headers(), timeout=5
                        )
                        msgs = resp_m.json().get("messages") or []
                        print(f"\033[1;36mResumed {session_id}\033[0m ({len(msgs)} messages):")
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
                elif cmd == "/prompts":
                    _cmd_prompts(base_url, "list", parts[1] if len(parts) > 1 else None, locale)
                elif cmd == "/local":
                    _cmd_local(base_url, parts[1] if len(parts) > 1 else "status", " ".join(parts[2:]) if len(parts) > 2 else None, locale)
                elif cmd == "/neural":
                    _cmd_neural(base_url, parts[1] if len(parts) > 1 else "status", " ".join(parts[2:]) if len(parts) > 2 else None, locale)
                elif cmd == "/brain":
                    _cmd_brain(base_url, parts[1] if len(parts) > 1 else "status", locale)
                elif cmd == "/evolve":
                    _cmd_evolve(base_url, parts[1] if len(parts) > 1 else "status", locale)
                elif cmd == "/fix":
                    _cmd_fix(base_url, locale)
                elif cmd == "/undo":
                    try:
                        r = await asyncio.to_thread(
                            ipc_post, f"{base_url}/api/undo-batch", headers=_auth_headers(), timeout=10
                        )
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
            await _send_and_stream(final_prompt, session_id, base_url, locale, model_override=pinned_model)
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


def _cmd_status(base_url: str, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    try:
        from rich import box
        from rich.console import Console, Group
        from rich.table import Table
        from rich.text import Text

        resp = ipc_get(f"{base_url}/api/status", headers=_auth_headers(), timeout=5)
        resp.raise_for_status()
        data = resp.json()
        h = data["health"]
        name = data.get("agent_name", "SysAgent")
        status_txt = data.get("status", "running")
        uptime = _fmt_uptime(int(data.get("uptime_sec", 0)))
        auth_mode = data.get("auth_mode") or "none"

        console = Console()

        t_sys = Table(title="System", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold", expand=True)
        t_sys.add_column("Item", style="bold")
        t_sys.add_column("Status")
        t_sys.add_column("Details")
        t_sys.add_row("Daemon", f"{status_txt} ({name})", f"up {uptime} | {auth_mode} / {data.get('approval_mode', '-')}")
        t_sys.add_row(
            "Host",
            f"CPU {float(h.get('cpu_percent', 0) or 0):.1f}%",
            f"RAM {float(h.get('ram_percent', 0) or 0):.1f}% ({h.get('ram_used_gb', 0)}/{h.get('ram_total_gb', 0)}G) | Disk {round(float(h.get('disk_free_gb', 0) or 0))}G",
        )
        g = h.get("gpu")
        temp_s = f"{h['max_temp_c']}°C" if h.get("max_temp_c") is not None else "-"
        if isinstance(g, dict):
            u_gb = round(g.get("vram_used_mb", 0) / 1024.0, 2)
            t_gb = round(g.get("vram_total_mb", 0) / 1024.0, 2)
            t_sys.add_row("GPU", f"Util {g.get('util_percent', 0)}%", f"VRAM {u_gb}/{t_gb}G | {temp_s} / {g.get('temp_c', '-')}°C")
        elif h.get("max_temp_c") is not None:
            t_sys.add_row("Thermal", temp_s, "-")

        t_brain = Table(title="Brain", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold", expand=True)
        t_brain.add_column("Mem", justify="right")
        t_brain.add_column("Lessons", justify="right")
        t_brain.add_column("Rules", justify="right")
        t_brain.add_column("Pending", justify="right", no_wrap=True)
        t_brain.add_row(
            str(data.get("memories", 0)),
            str(data.get("lessons", 0)),
            str(data.get("brain_rules", 0)),
            f"{data.get('pending_approvals', 0)}a / {data.get('pending_brain_questions', 0)}q",
        )

        models = data.get("models") or {}
        quota = dict(data.get("quota_today") or {})
        role_map = [
            ("Chat", models.get("chat") or data.get("model", "-")),
            ("Light", models.get("light") or "-"),
            ("Evolve", models.get("evolve") or "-"),
            ("Worker", models.get("worker") or "-"),
        ]
        t_mod = Table(title="Models & RPD", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold", expand=True)
        t_mod.add_column("Role", style="bold")
        t_mod.add_column("Model")
        t_mod.add_column("RPD", justify="right")
        seen_m: set[str] = set()
        for role, mname in role_map:
            rpd = quota.pop(mname, None) if mname not in seen_m else None
            seen_m.add(mname)
            t_mod.add_row(role, mname, str(rpd) if rpd is not None else "-")
        for extra_m, cnt in quota.items():
            t_mod.add_row("Other", extra_m, str(cnt))
        total_rpd = sum(int(v) for v in (data.get("quota_today") or {}).values())
        t_mod.add_section()
        t_mod.add_row("Total", "", str(total_rpd), style="bold")

        top_grid = Table.grid(expand=True, padding=(0, 2))
        top_grid.add_column(ratio=3)
        top_grid.add_column(ratio=2)
        top_grid.add_row(Group(t_sys, t_brain), t_mod)
        console.print(top_grid)

        bot_grid = Table.grid(expand=True, padding=(0, 2))
        evo = data.get("evolve") or {}
        learn = data.get("learn") or {}
        bot_cells: list[Table] = []

        if evo:
            rate = evo.get("success_rate")
            counts = evo.get("counts") or {}
            nxt = evo.get("next_run_in_sec")
            t_evo = Table(title="Self-Evolve", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold", expand=True)
            t_evo.add_column("Metric", style="bold")
            t_evo.add_column("Value")
            t_evo.add_row("Success", f"{int(rate * 100)}%" if rate is not None else "n/a")
            t_evo.add_row(
                "Outcomes",
                f"{counts.get('applied', 0)} ok / {counts.get('failed', 0)} fail / {counts.get('rejected_by_critic', 0)} crit",
            )
            t_evo.add_row("Next Scan", f"in {_fmt_uptime(nxt)}" if nxt is not None else "pending")
            t_evo.add_row("Proposals", str(data.get("evolve_proposals", 0)))
            bot_grid.add_column(ratio=1)
            bot_cells.append(t_evo)

        if learn:
            lm = learn.get("local_model") or {}
            sh = learn.get("shadow") or {}
            sh_tot = int(sh.get("hits", 0)) + int(sh.get("misses", 0))
            sh_acc = f"{int(sh['accuracy'] * 100)}%" if sh.get("accuracy") is not None else "n/a"
            nn = learn.get("neural") or {}
            cov = learn.get("coverage") or {}

            t_imp = Table(title="Improvement & Neural", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold", expand=True)
            t_imp.add_column("Engine", style="bold")
            t_imp.add_column("State")
            t_imp.add_row("LocalMind", f"{lm.get('samples', 0)} samples, {lm.get('classes', 0)} cls | shadow {sh_acc} ({sh.get('hits', 0)}/{sh_tot})")
            if nn:
                t_acc = f"{int(nn['tool_accuracy'] * 100)}%" if nn.get("tool_accuracy") is not None else "n/a"
                wf = nn.get("world_forecast") or {}
                t_imp.add_row("Neural", f"{nn.get('param_count', 0):,} params | {t_acc} acc | {nn.get('synapses', 0)} syn")
                t_imp.add_row(
                    "World / Cov",
                    f"10m RAM {wf.get('pred_ram_10m', 0)}% ({wf.get('proactive_hint', 'nominal')}) | cov {cov.get('coverage_pct', 0)}%",
                )
            else:
                t_imp.add_row("Coverage", f"{cov.get('coverage_pct', 0)}% ({cov.get('tested_count', 0)}/{cov.get('total_count', 0)} symbols)")
            rm = learn.get("roadmap") or []
            if rm:
                t_imp.add_row("Next Goal", Text(f"[{rm[0]['priority']}] {rm[0]['goal']}"))
            bot_grid.add_column(ratio=2)
            bot_cells.append(t_imp)

        if bot_cells:
            bot_grid.add_row(*bot_cells)
            console.print(bot_grid)
    except Exception as exc:
        try:
            Storage().add_brain_thought("ignore", 0, f"CLI error in status: {exc}")
        except Exception:
            pass
        print(f"{t('error_prefix', loc)} ({base_url}): {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_events(base_url: str, limit: int, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    try:
        resp = ipc_get(f"{base_url}/api/events", params={"limit": limit}, headers=_auth_headers(), timeout=5)
        resp.raise_for_status()
        events = resp.json().get("events", [])
        if not events:
            print(t("cli_no_events", loc))
            return
        for ev in events:
            print(f"[{ev['timestamp']}] [{ev['category']}/{ev['severity']}] {ev['message']}")
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_name(base_url: str, new_name: str, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    clean = new_name.strip()
    if not clean:
        print(t("cli_name_empty", loc), file=sys.stderr)
        sys.exit(1)
    try:
        resp = ipc_post(f"{base_url}/api/name", json={"name": clean}, headers=_auth_headers(), timeout=5)
        resp.raise_for_status()
        print(t("cli_name_updated", loc, name=resp.json()["agent_name"]))
    except Exception:
        env_path = persist_env_var("SYSAGENT_NAME", clean)
        print(t("cli_name_saved", loc, path=env_path, name=clean))


async def _quick_launcher(base_url: str, locale: str | None = None) -> None:
    """Native KDE/GTK quick input launcher for global shortcut binding."""
    settings = get_settings()
    loc = locale or settings.locale
    title = t("quick_title", loc, name=settings.agent_name)
    prompt_lbl = t("quick_prompt", loc)
    prompt: str | None = None
    ui_tool = "kdialog" if shutil.which("kdialog") else ("zenity" if shutil.which("zenity") else None)
    if ui_tool == "kdialog":
        proc = await asyncio.to_thread(
            subprocess.run,
            ["kdialog", "--title", title, "--inputbox", prompt_lbl],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            prompt = proc.stdout.strip()
    elif ui_tool == "zenity":
        proc = await asyncio.to_thread(
            subprocess.run,
            ["zenity", "--entry", f"--title={title}", f"--text={prompt_lbl}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            prompt = proc.stdout.strip()
    else:
        prompt = input("Quick > ").strip()

    if not prompt:
        return

    if prompt.strip() == "!fix":
        _cmd_fix(base_url, locale)
        return
    if prompt.strip() == "!clip":
        for ccmd in (["wl-paste", "--no-newline"], ["xclip", "-o", "-selection", "clipboard"]):
            if shutil.which(ccmd[0]):
                cp = await asyncio.to_thread(subprocess.run, ccmd, capture_output=True, text=True, timeout=3)
                if cp.returncode == 0 and cp.stdout.strip():
                    prompt = f"Analyze or fix this clipboard content:\n{cp.stdout.strip()[:2000]}"
                    break

    ws_url = _ws_url(base_url, f"quick-{uuid.uuid4().hex[:6]}")
    reply_text = ""
    try:
        async with ipc_ws_connect(ws_url, ping_interval=30, ping_timeout=120, close_timeout=10) as ws:
            payload: dict[str, object] = {"type": "message", "text": prompt}
            if locale:
                payload["locale"] = locale
            await ws.send(json.dumps(payload))
            while True:
                ev = json.loads(await ws.recv())
                if ev.get("type") == "approval_required":
                    approved = False
                    if ui_tool == "kdialog":
                        ans = await asyncio.to_thread(
                            subprocess.run,
                            ["kdialog", "--title", t("approval_required", loc), "--yesno", ev["reason"]],
                        )
                        approved = ans.returncode == 0
                    elif ui_tool == "zenity":
                        ans = await asyncio.to_thread(
                            subprocess.run,
                            ["zenity", "--question", f"--title={t('approval_required', loc)}", f"--text={ev['reason']}"],
                        )
                        approved = ans.returncode == 0
                    await ws.send(
                        json.dumps(
                            {
                                "type": "approval_response",
                                "approval_id": ev["approval_id"],
                                "approved": approved,
                            }
                        )
                    )
                elif ev.get("type") == "assistant_message":
                    reply_text = ev.get("text", "")
                elif ev.get("type") == "error":
                    reply_text = f"{t('error_prefix', loc)}: {ev.get('message')}"
                elif ev.get("type") == "done":
                    break
    except Exception as exc:
        reply_text = f"{t('error_prefix', loc)}: {exc}"

    if reply_text:
        if ui_tool == "kdialog":
            await asyncio.to_thread(subprocess.run, ["kdialog", "--title", settings.agent_name, "--msgbox", reply_text])
        elif ui_tool == "zenity":
            await asyncio.to_thread(
                subprocess.run, ["zenity", "--info", f"--title={settings.agent_name}", f"--text={reply_text}"]
            )
        else:
            print(reply_text)


def _cmd_hook() -> None:
    err_file = DATA_DIR / "last_error.json"
    print(f"""# Add this to ~/.bashrc or ~/.zshrc:
_sysagent_err_hook() {{
  local ec=$?
  if [ $ec -ne 0 ]; then
    local last_cmd
    last_cmd=$(fc -ln -1 2>/dev/null | sed 's/^[[:space:]]*//')
    mkdir -p "{DATA_DIR}"
    printf '{{"exit_code":%d,"command":"%s","cwd":"%s"}}\\n' "$ec" "${{last_cmd//\\"/\\\\\\"}}" "$PWD" > "{err_file}"
  fi
}}
PROMPT_COMMAND="_sysagent_err_hook${{PROMPT_COMMAND:+; $PROMPT_COMMAND}}"
alias fix="{_prog_name()} fix"
""")


def _cmd_fix(base_url: str, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    err_file = DATA_DIR / "last_error.json"
    if not err_file.exists():
        print(t("cli_no_last_error", loc), file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(err_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raw = err_file.read_text(encoding="utf-8", errors="replace")
        data = {"command": raw.strip(), "exit_code": 1, "cwd": "."}
    prompt = (
        f"The user's last shell command failed in `{data.get('cwd')}` with exit code {data.get('exit_code')}:\n"
        f"`{data.get('command')}`\n"
        "Explain why it likely failed and suggest (or run if safe/approved) the fix."
    )
    asyncio.run(_send_and_stream(prompt, f"fix-{uuid.uuid4().hex[:6]}", base_url, locale))


def _cmd_brain(base_url: str, action: str, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "think":
            resp = ipc_post(f"{base_url}/api/brain/think", headers=hdrs, timeout=30)
            resp.raise_for_status()
            print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
            return
        if action == "digest":
            resp = ipc_get(f"{base_url}/api/brain/digest", headers=hdrs, timeout=30)
            resp.raise_for_status()
            print(resp.json().get("digest", ""))
            return
        resp = ipc_get(f"{base_url}/api/brain", headers=hdrs, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if action == "rules":
            for r in data.get("rules", []):
                print(f"#{r['id']} [{r['confidence']}% | {r['hits']}x] {r['pattern']} -> {r['action_policy']}")
            return
        if action == "log":
            from rich import box
            from rich.console import Console
            from rich.table import Table

            tbl = Table(title="Brain Thoughts", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold")
            tbl.add_column("ID", justify="right")
            tbl.add_column("Decision")
            tbl.add_column("Conf", justify="right")
            tbl.add_column("Summary")
            for th in (data.get("thoughts") or [])[:15]:
                tbl.add_row(str(th.get("id", "")), str(th.get("decision", "")), f"{th.get('confidence', 0)}%", str(th.get("summary", ""))[:80])
            Console().print(tbl)
            return
        print(
            f"Brain enabled={data['enabled']} threshold={data['confidence_threshold']}% "
            f"pending_questions={len(data['questions'])} rules={len(data['rules'])}"
        )
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


async def _stream_evolve(base_url: str) -> None:
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    tok = get_settings().auth_token
    ws_url = f"{ws_url}/ws/evolve" + (f"?token={tok}" if tok else "")
    async with ipc_ws_connect(ws_url, open_timeout=10, close_timeout=10) as ws:
        in_thought = False
        while True:
            try:
                raw = await ws.recv()
            except websockets.ConnectionClosed:
                break
            ev = json.loads(raw)
            if ev.get("type") == "heartbeat":
                print(f"\r\033[K\033[90m{ev['elapsed']}s: analyzing\033[0m", end="", flush=True)
            elif ev.get("type") == "thought":
                print("\r\033[K", end="")
                if not in_thought:
                    print("\033[90m", end="", flush=True)
                    in_thought = True
                print(ev["text"], end="", flush=True)
            elif ev.get("type") == "detail":
                print("\r\033[K", end="")
                if in_thought:
                    print("\033[0m")
                    in_thought = False
                print(f"\033[90m{ev['text']}\033[0m", flush=True)
            elif ev.get("type") == "progress":
                print("\r\033[K", end="")
                if in_thought:
                    print("\033[0m")
                    in_thought = False
                print(f"\033[36m{ev['message']}\033[0m", flush=True)
            elif ev.get("type") == "result":
                print("\r\033[K", end="")
                if in_thought:
                    print("\033[0m")
                    in_thought = False
                print(json.dumps(ev["data"], indent=2, ensure_ascii=False))
            elif ev.get("type") == "error":
                print("\r\033[K", end="")
                if in_thought:
                    print("\033[0m")
                    in_thought = False
                print(f"\033[31mError: {ev['message']}\033[0m", file=sys.stderr)


def _cmd_evolve(base_url: str, action: str, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "run":
            asyncio.run(_stream_evolve(base_url))
            return
        resp = ipc_get(f"{base_url}/api/evolve", headers=hdrs, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if action == "log":
            from rich import box
            from rich.console import Console
            from rich.table import Table

            tbl = Table(title="Evolve Proposals", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold")
            tbl.add_column("ID", justify="right")
            tbl.add_column("Status")
            tbl.add_column("File")
            tbl.add_column("Title")
            for p in (data.get("proposals") or [])[:15]:
                tbl.add_row(str(p.get("id", "")), str(p.get("status", "")), Path(str(p.get("file_path", ""))).name, str(p.get("title", ""))[:65])
            Console().print(tbl)
            return
        print(
            f"Self-Evolve enabled={data['enabled']} auto_apply={data['auto_apply']} "
            f"(max_cpu={data['max_cpu']}%, max_ram={data['max_ram']}%) proposals={len(data['proposals'])}"
        )
        for p in data.get("proposals", [])[:10]:
            print(f"#{p['id']} [{p['status']} | {p['category']}] {p['title']}")
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_prompts(base_url: str, action: str, key: str | None = None, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "evolve":
            resp = ipc_post(f"{base_url}/api/prompts/evolve", headers=hdrs, timeout=120)
            resp.raise_for_status()
            print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
            return
        if action == "rollback":
            if not key:
                print(f"{t('error_prefix', loc)}: prompt key is required for rollback (e.g. agent_main)", file=sys.stderr)
                sys.exit(1)
            resp = ipc_post(f"{base_url}/api/prompts/{key}/rollback", headers=hdrs, timeout=10)
            resp.raise_for_status()
            print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
            return
        resp = ipc_get(f"{base_url}/api/prompts", headers=hdrs, timeout=5)
        resp.raise_for_status()
        prompts = resp.json().get("prompts", [])
        for p in prompts:
            if key and p["key"] != key:
                continue
            vars_str = ", ".join(p.get("required_vars") or []) or "-"
            succ = p.get("successes", 0)
            fail = p.get("failures", 0)
            print(
                f"{p['key']} (v{p['version']}) [{p['reason']}] "
                f"ok={succ} fail={fail} vars=({vars_str})"
            )
            if key:
                print("---\n" + p.get("template", ""))
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_local(base_url: str, action: str, query: str | None = None, locale: str | None = None) -> None:
    from .local_mind import GGUF_PRESETS, LocalMind, download_default_gguf, find_gguf_model

    loc = locale or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "train":
            mind = LocalMind(Storage())
            st = mind.train(run_lora=True)
            print(json.dumps(st, indent=2, ensure_ascii=False))
            return
        if action == "distill":
            mind = LocalMind(Storage())
            cnt = int(query) if (query and query.isdigit()) else 2
            print(f"Running Multi-Teacher Knowledge Distillation ({cnt} samples per teacher)...")
            res = mind.distill_from_teachers(count_per_teacher=cnt)
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        if action == "download":
            from .local_mind import ROLE_DEFAULT_PRESETS, SPECIALIST_ROLES, find_all_gguf_models

            installed_names = {p.name for p in find_all_gguf_models()}
            target = (query or "qwen2.5-1.5b").strip().lower()
            if target == "list":
                print("6 Specialist Use-Case Models:")
                for r in SPECIALIST_ROLES:
                    pk = ROLE_DEFAULT_PRESETS[r]
                    spec = GGUF_PRESETS[pk]
                    mark = "[installed]" if find_gguf_model(r, strict=True) else "[missing]  "
                    print(
                        f"  {r:<11} {mark} preset={pk:<22} ~{spec['size_mb']} MB  "
                        f"({spec.get('hf_source', spec['filename'])})"
                    )
                print("\nAll Presets:")
                for k, spec in GGUF_PRESETS.items():
                    mark = "*" if spec["filename"] in installed_names else " "
                    print(f" {mark} {k:<22} role={spec['role']:<11} ~{spec['size_mb']} MB  ({spec['filename']})")
                return
            if target == "all-specialists":
                for r in SPECIALIST_ROLES:
                    pk = ROLE_DEFAULT_PRESETS[r]
                    print(f"Downloading {r} specialist ({pk})...")
                    dest = download_default_gguf(pk)
                    print(f"Saved: {dest}")
                return
            if target == "ensemble":
                existing_coder = find_gguf_model("coding")
                targets = [ROLE_DEFAULT_PRESETS["chat"]] if existing_coder else [ROLE_DEFAULT_PRESETS["coding"], ROLE_DEFAULT_PRESETS["chat"]]
                if existing_coder:
                    print(f"Coding specialist already present: {existing_coder}")
                for p_name in targets:
                    print(f"Downloading ensemble specialist '{p_name}'...")
                    dest = download_default_gguf(p_name)
                    print(f"Saved: {dest}")
                return
            preset = ROLE_DEFAULT_PRESETS.get(target, target)
            print(f"Downloading GGUF '{preset}'...")
            dest = download_default_gguf(preset)
            print(f"Downloaded GGUF model to: {dest}")
            return
        if action == "prune":
            from .local_mind import prune_superseded_models

            dry = (query or "").strip().lower() in {"dry", "dry-run", "preview"}
            removed = prune_superseded_models(dry_run=dry)
            print(json.dumps({"dry_run": dry, "removed": removed}, indent=2, ensure_ascii=False))
            return
        if action == "profile":
            from .self_learn import discover_system_profile

            prof = discover_system_profile(Storage(), seed=True, force=True)
            print(json.dumps(prof, indent=2, ensure_ascii=False))
            return
        if action == "test":
            mind = LocalMind(Storage())
            q = query or "check system status cpu ram"
            tool, args, score = mind.predict_tool(q)
            print(f"Query         : {q}")
            print(f"Predicted Tool: {tool} (score={score}) args={args}")
            print(f"Offline Reply : {mind.synthesize_reply(q)}")
            return
        if action == "bench":
            from rich import box
            from rich.console import Console
            from rich.table import Table

            from .bench import benchmark_local_models

            report = benchmark_local_models(role_filter=query, storage=Storage())
            tbl = Table(title="Local GGUF Model Benchmark", box=box.ROUNDED, border_style="cyan")
            tbl.add_column("Role", style="bold cyan")
            tbl.add_column("Model")
            tbl.add_column("Size", justify="right")
            tbl.add_column("Latency", justify="right")
            tbl.add_column("tok/s", justify="right", style="bold green")
            tbl.add_column("RAM Δ", justify="right")
            tbl.add_column("Status")
            for row in report.get("results") or []:
                lat = f"{row['latency_s']}s" if row.get("latency_s") is not None else "-"
                tps = f"{row['tok_per_s']}" if row.get("tok_per_s") is not None else "-"
                tbl.add_row(
                    str(row.get("role", "")),
                    str(row.get("model", "")),
                    f"{row.get('size_gb', 0)} GB",
                    lat,
                    tps,
                    f"+{row.get('ram_delta_mb', 0)} MB",
                    str(row.get("status", "")),
                )
            console = Console()
            console.print(tbl)
            if report.get("fastest_model"):
                console.print(
                    f"[bold green]Fastest:[/] {report['fastest_role']} ({report['fastest_model']})"
                )
            return
        try:
            resp = ipc_get(f"{base_url}/api/local", headers=hdrs, timeout=5)
            resp.raise_for_status()
            st = resp.json()
        except Exception:
            st = LocalMind(Storage()).get_status()
        print(json.dumps(st, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_neural(base_url: str, action: str, query: str | None = None, locale: str | None = None) -> None:
    from .neural import NeuralEngine

    loc = locale or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "train":
            try:
                resp = ipc_post(f"{base_url}/api/neural/train", headers=hdrs, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                data = NeuralEngine(Storage()).train()
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return
        if action in {"pulse", "activate"}:
            q = (query or "check system status and ram usage").strip()
            try:
                resp = ipc_post(f"{base_url}/api/neural/pulse", json={"query": q}, headers=hdrs, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                data = NeuralEngine(Storage()).pulse(q)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return
        try:
            resp = ipc_get(f"{base_url}/api/neural", headers=hdrs, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            data = NeuralEngine(Storage()).get_status()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_digest(base_url: str, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    try:
        from rich import box
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel

        resp = ipc_get(f"{base_url}/api/brain/digest", headers=_auth_headers(), timeout=30)
        resp.raise_for_status()
        digest_txt = str(resp.json().get("digest") or "").strip() or "(no digest)"
        Console().print(
            Panel(
                Markdown(digest_txt),
                title="SysAgent Daily Digest",
                title_align="left",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_doctor(base_url: str) -> None:
    import psutil
    from rich import box
    from rich.console import Console
    from rich.table import Table

    checks: list[tuple[str, str, str]] = []
    try:
        r = ipc_get(f"{base_url}/api/status", headers=_auth_headers(), timeout=4)
        r.raise_for_status()
        st = r.json()
        checks.append(("Daemon API", "OK", f"running ({st.get('agent_name')}) | auth={st.get('auth_mode')}"))
    except Exception as exc:
        checks.append(("Daemon API", "FAIL", str(exc)))

    sd = subprocess.run(["systemctl", "--user", "is-active", "sysagent"], capture_output=True, text=True, timeout=3)
    sd_state = sd.stdout.strip() or "inactive"
    checks.append(("Systemd Unit", "OK" if sd_state == "active" else "WARN", sd_state))

    try:
        db = Storage()
        qc = db._connect().execute("PRAGMA quick_check;").fetchone()[0]
        checks.append(("SQLite DB", "OK" if qc == "ok" else "FAIL", f"{db.db_path.name} ({qc})"))
    except Exception as exc:
        checks.append(("SQLite DB", "FAIL", str(exc)))

    try:
        from .neural import NeuralEngine

        ns = NeuralEngine(Storage()).get_status()
        ok_nn = int(ns.get("epochs_trained", 0)) > 0
        checks.append(("NeuralEngine", "OK" if ok_nn else "WARN", f"{ns.get('param_count', 0):,} params | syn={ns.get('synapses', 0)}"))
    except Exception as exc:
        checks.append(("NeuralEngine", "FAIL", str(exc)))

    try:
        from .local_mind import find_all_gguf_models

        ggufs = find_all_gguf_models()
        checks.append(("GGUF Models", "OK" if ggufs else "WARN", f"{len(ggufs)} installed"))
    except Exception as exc:
        checks.append(("GGUF Models", "WARN", str(exc)))

    disk_free = round(psutil.disk_usage("/").free / (1024**3), 1)
    checks.append(("Disk Free", "OK" if disk_free >= 10.0 else "WARN", f"{disk_free} GB free"))
    checks.append(("Ruff Linter", "OK" if shutil.which("ruff") else "WARN", shutil.which("ruff") or "not installed"))

    tbl = Table(title="SysAgent Doctor", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold")
    tbl.add_column("Check", style="bold")
    tbl.add_column("State")
    tbl.add_column("Details")
    for name, state, detail in checks:
        styled = f"[green]{state}[/green]" if state == "OK" else (f"[yellow]{state}[/yellow]" if state == "WARN" else f"[red]{state}[/red]")
        tbl.add_row(name, styled, detail)
    Console().print(tbl)


def _cmd_plugin(action: str, name: str | None = None) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from .plugins import PLUGINS_DIR

    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    if action == "list":
        tbl = Table(title="User Plugins", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold")
        tbl.add_column("Plugin", style="bold")
        tbl.add_column("Status")
        tbl.add_column("Size", justify="right")
        items = sorted([*PLUGINS_DIR.glob("*.py"), *PLUGINS_DIR.glob("*.py.disabled")])
        for p in items:
            disabled = p.name.endswith(".disabled")
            pname = p.name[:-9] if disabled else p.name
            tbl.add_row(pname, "[yellow]disabled[/yellow]" if disabled else "[green]enabled[/green]", f"{p.stat().st_size} B")
        if not items:
            tbl.add_row("(none)", "-", "-")
        Console().print(tbl)
        return

    if not name:
        print("Plugin name is required.", file=sys.stderr)
        sys.exit(1)
    stem = name[:-3] if name.endswith(".py") else name
    en_path = PLUGINS_DIR / f"{stem}.py"
    dis_path = PLUGINS_DIR / f"{stem}.py.disabled"
    if action == "disable":
        if not en_path.is_file():
            print(f"Enabled plugin not found: {en_path.name}", file=sys.stderr)
            sys.exit(1)
        en_path.rename(dis_path)
        print(f"Disabled plugin: {en_path.name}")
    elif action == "enable":
        if not dis_path.is_file():
            print(f"Disabled plugin not found: {dis_path.name}", file=sys.stderr)
            sys.exit(1)
        dis_path.rename(en_path)
        print(f"Enabled plugin: {en_path.name}")
    elif action == "remove":
        removed = False
        for target in (en_path, dis_path):
            if target.is_file():
                target.unlink()
                removed = True
        print(f"Removed plugin: {stem}" if removed else f"Plugin not found: {stem}")


def _cmd_search(base_url: str, query: str, limit: int = 12, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        resp = ipc_get(
            f"{base_url}/api/search",
            params={"q": query, "limit": limit},
            headers=_auth_headers(),
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        tbl = Table(title=f"Search: {query}", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold")
        tbl.add_column("Source", style="bold")
        tbl.add_column("Title / Key")
        tbl.add_column("Snippet")
        for r in results:
            tbl.add_row(str(r.get("source", "")), str(r.get("title", ""))[:32], str(r.get("snippet", ""))[:80])
        if not results:
            tbl.add_row("-", "No matches", "-")
        Console().print(tbl)
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_rollback(
    base_url: str,
    target: str,
    target_id: str | None,
    list_only: bool,
    locale_override: str | None = None,
) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    loc = locale_override or get_settings().locale
    hdrs = _auth_headers()
    action = "list" if list_only else (target or "auto")
    try:
        if action == "list":
            try:
                resp = ipc_get(f"{base_url}/api/snapshots", headers=hdrs, timeout=5)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                from .snapshots import unified_snapshots

                data = unified_snapshots(Storage())

            console = Console()
            tbl = Table(title="Recent File / Audit Snapshots", box=box.ROUNDED, border_style="magenta")
            tbl.add_column("ID", style="dim")
            tbl.add_column("Action", style="cyan")
            tbl.add_column("Target")
            tbl.add_column("Batch")
            tbl.add_column("State")
            for row in (data.get("audit_log") or [])[:12]:
                state = "undone" if row.get("undone") else "active"
                tbl.add_row(
                    str(row.get("id", "")),
                    str(row.get("action", "")),
                    str(row.get("target_path", ""))[:55],
                    str(row.get("batch_id") or "-"),
                    state,
                )
            console.print(tbl)

            cfgs = data.get("config_snapshots") or []
            if cfgs:
                ctbl = Table(title="Config Snapshots", box=box.ROUNDED, border_style="cyan")
                ctbl.add_column("Snapshot ID")
                ctbl.add_column("Path")
                ctbl.add_column("Created")
                for c in cfgs[:8]:
                    ctbl.add_row(str(c.get("id", "")), str(c.get("path", "")), str(c.get("created_at", "")))
                console.print(ctbl)

            props = data.get("evolve_proposals") or []
            if props:
                etbl = Table(title="Evolve Proposals", box=box.ROUNDED, border_style="green")
                etbl.add_column("ID", style="dim")
                etbl.add_column("Status", style="cyan")
                etbl.add_column("Target")
                etbl.add_column("Summary")
                for p in props[:8]:
                    etbl.add_row(
                        str(p.get("id", "")),
                        str(p.get("status", "")),
                        str(p.get("target_file", "")),
                        str(p.get("summary", p.get("title", "")))[:60],
                    )
                console.print(etbl)
            return

        try:
            resp = ipc_post(
                f"{base_url}/api/rollback",
                json={"target": action, "id": target_id},
                headers=hdrs,
                timeout=15,
            )
            resp.raise_for_status()
            res = resp.json()
        except Exception:
            from .snapshots import perform_rollback

            res = perform_rollback(action, target_id, Storage())

        if res.get("ok"):
            Console().print(f"[bold green]Rollback succeeded:[/] {json.dumps(res, ensure_ascii=False)}")
        else:
            print(f"{t('error_prefix', loc)}: {res.get('error', res)}", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _prog_name() -> str:
    from .config import get_cli_bin_name

    return get_cli_bin_name()


def _parse_schedule_spec(spec: str) -> tuple[str, str]:
    """Parse shorthand schedule spec (`09:00`, `30m`, `2h`, `60s`, `file:*.log`, `neural:anomaly`, `neural:bottleneck`)."""
    s = spec.strip()
    low = s.lower()
    if low == "neural:anomaly":
        return "neural_anomaly", "anomaly"
    if low == "neural:bottleneck":
        return "neural_bottleneck", "0.70"
    if low.startswith("neural:bottleneck:"):
        thresh = s.split(":", 2)[2].strip()
        float(thresh)
        return "neural_bottleneck", thresh
    if s.startswith("file:"):
        return "file_created", s.split(":", 1)[1].strip()
    if ":" in s and len(s) <= 5:
        hh, mm = s.split(":", 1)
        if hh.isdigit() and mm.isdigit() and 0 <= int(hh) <= 23 and 0 <= int(mm) <= 59:
            return "daily_time", f"{int(hh):02d}:{int(mm):02d}"
    if low.endswith("h") and low[:-1].isdigit():
        return "interval", str(max(30, int(low[:-1]) * 3600))
    if low.endswith("m") and low[:-1].isdigit():
        return "interval", str(max(30, int(low[:-1]) * 60))
    if low.endswith("s") and low[:-1].isdigit():
        return "interval", str(max(30, int(low[:-1])))
    if low.isdigit():
        return "interval", str(max(30, int(low)))
    raise ValueError(
        f"Unsupported schedule spec '{spec}'. Use HH:MM, 30m, 2h, 60s, file:*.log, neural:anomaly, or neural:bottleneck"
    )


def _cmd_schedule(
    base_url: str,
    action: str,
    rest: list[str],
    locale_override: str | None = None,
) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    loc = locale_override or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "list":
            try:
                resp = ipc_get(f"{base_url}/api/automations", headers=hdrs, timeout=5)
                resp.raise_for_status()
                items = resp.json().get("automations") or []
            except Exception:
                with Storage() as st:
                    items = st.list_automations()
            tbl = Table(title="Scheduled Automations", box=box.ROUNDED, border_style="cyan")
            tbl.add_column("ID", style="dim")
            tbl.add_column("Name", style="bold")
            tbl.add_column("Trigger", style="cyan")
            tbl.add_column("Prompt")
            tbl.add_column("State")
            tbl.add_column("Last Run", style="dim")
            for a in items:
                state = "[green]enabled[/]" if a.get("enabled") else "[dim]disabled[/]"
                tbl.add_row(
                    str(a.get("id", "")),
                    str(a.get("name", "")),
                    f"{a.get('trigger_type')}:{a.get('trigger_value')}",
                    str(a.get("prompt", ""))[:55],
                    state,
                    str(a.get("last_run_at") or "-"),
                )
            if not items:
                tbl.add_row("-", "No scheduled tasks", "-", "-", "-", "-")
            Console().print(tbl)
            return

        if action == "add":
            if len(rest) < 3:
                print(
                    f"Usage: {_prog_name()} schedule add <name> <HH:MM|30m|2h|file:*.log|neural:anomaly|neural:bottleneck> <prompt...>",
                    file=sys.stderr,
                )
                sys.exit(1)
            name, spec = rest[0], rest[1]
            prompt = " ".join(rest[2:]).strip()
            ttype, tval = _parse_schedule_spec(spec)
            try:
                resp = ipc_post(
                    f"{base_url}/api/automations",
                    json={"name": name, "trigger_type": ttype, "trigger_value": tval, "prompt": prompt},
                    headers=hdrs,
                    timeout=5,
                )
                resp.raise_for_status()
                out = resp.json()
            except Exception:
                with Storage() as st:
                    aid = st.create_automation(name, ttype, tval, prompt)
                out = {"id": aid, "name": name, "trigger_type": ttype, "trigger_value": tval}
            Console().print(
                f"[bold green]Scheduled #{out['id']}:[/] {name} ({ttype}={tval})"
            )
            return

        if action in {"delete", "toggle"}:
            if not rest or not rest[0].isdigit():
                print(f"Usage: {_prog_name()} schedule {action} <id>", file=sys.stderr)
                sys.exit(1)
            aid = int(rest[0])
            if action == "delete":
                try:
                    resp = ipc_delete(f"{base_url}/api/automations/{aid}", headers=hdrs, timeout=5)
                    resp.raise_for_status()
                    ok = bool(resp.json().get("deleted"))
                except Exception:
                    with Storage() as st:
                        ok = st.delete_automation(aid)
                print(f"Deleted automation #{aid}." if ok else f"Automation #{aid} not found.")
            else:
                try:
                    resp = ipc_post(f"{base_url}/api/automations/{aid}/toggle", headers=hdrs, timeout=5)
                    resp.raise_for_status()
                    enabled = resp.json().get("enabled")
                except Exception:
                    with Storage() as st:
                        enabled = st.toggle_automation(aid)
                if enabled is None:
                    print(f"Automation #{aid} not found.", file=sys.stderr)
                    sys.exit(1)
                print(f"Automation #{aid} is now {'enabled' if enabled else 'disabled'}.")
            return
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_notifications(
    base_url: str,
    action: str,
    rest: list[str],
    limit: int = 20,
    locale_override: str | None = None,
) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    loc = locale_override or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "test":
            from .notification import NotificationCenter

            msg = " ".join(rest).strip() or "Test notification from NotificationCenter"
            with Storage() as st:
                nc = NotificationCenter(get_settings(), st)
                h = nc.send(
                    f"{get_settings().agent_name} — Test",
                    msg,
                    category="info",
                    bypass_gates=True,
                )
            print(f"Sent test notification ({h.notif_id})." if h else "notify-send unavailable.")
            return

        resp = ipc_get(
            f"{base_url}/api/notifications",
            params={"limit": limit},
            headers=hdrs,
            timeout=5,
        )
        resp.raise_for_status()
        items = resp.json().get("notifications") or []
        tbl = Table(title="Recent Notifications", box=box.ROUNDED, border_style="cyan")
        tbl.add_column("Time", style="dim", width=8)
        tbl.add_column("Cat", style="cyan", width=10)
        tbl.add_column("Status", width=22)
        tbl.add_column("Title")
        tbl.add_column("Action", style="green")
        for n in items:
            ts = str(n.get("created_at") or "")[-8:]
            tbl.add_row(
                ts,
                str(n.get("category", "")),
                str(n.get("status", "")),
                str(n.get("title", ""))[:45],
                str(n.get("action") or "-"),
            )
        if not items:
            tbl.add_row("-", "-", "No notifications yet", "-", "-")
        Console().print(tbl)
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_cortex(
    base_url: str,
    action: str,
    arg: str | None = None,
    locale_override: str | None = None,
) -> None:
    loc = locale_override or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "dream":
            try:
                resp = ipc_post(f"{base_url}/api/cortex/dream", headers=hdrs, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                from .cortex import NeuralBrainCortex

                with Storage() as st:
                    data = NeuralBrainCortex(get_settings(), st).run_dream_cycle(force=True)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return
        if action == "heal":
            try:
                resp = ipc_post(f"{base_url}/api/cortex/heal", headers=hdrs, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                from .cortex import NeuralBrainCortex

                with Storage() as st:
                    data = NeuralBrainCortex(get_settings(), st).evaluate_predictive_healer(
                        force=True, notify=False
                    )
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return
        if action == "flow" and arg is not None:
            mode = None if arg.strip().lower() in {"auto", "clear", "reset"} else arg.strip().lower()
            try:
                resp = ipc_post(
                    f"{base_url}/api/cortex/flow", json={"mode": mode}, headers=hdrs, timeout=5
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                from .cortex import NeuralBrainCortex

                with Storage() as st:
                    data = NeuralBrainCortex(get_settings(), st).set_flow_override(mode)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return

        try:
            resp = ipc_get(f"{base_url}/api/cortex", headers=hdrs, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            from .cortex import NeuralBrainCortex

            with Storage() as st:
                data = NeuralBrainCortex(get_settings(), st).get_status()
        if action == "habits":
            print(json.dumps(data.get("habit_suggestions", []), indent=2, ensure_ascii=False))
        elif action == "flow":
            print(json.dumps(data.get("flow", {}), indent=2, ensure_ascii=False))
        else:
            print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
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
    sub.add_parser("doctor", help="Run health and configuration diagnostics")
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
    sub.add_parser("fix", help="Analyze the last failed terminal command and suggest a fix")
    sub.add_parser("hook", help="Print the Bash/Zsh error-capturing shell hook")
    brain_parser = sub.add_parser("brain", help="Autonomous brain status / think / rules / digest / log")
    brain_parser.add_argument("brain_action", nargs="?", default="status", choices=["status", "think", "rules", "digest", "log"])
    evolve_parser = sub.add_parser("evolve", help="Self-evolution status / scan / log")
    evolve_parser.add_argument("evolve_action", nargs="?", default="status", choices=["status", "run", "log"])
    prompts_parser = sub.add_parser("prompts", help="List, evolve, or rollback centralized system prompts")
    prompts_parser.add_argument("prompts_action", nargs="?", default="list", choices=["list", "evolve", "rollback"])
    prompts_parser.add_argument("prompt_key", nargs="?", default=None, help="Target prompt key (e.g. agent_main)")
    local_parser = sub.add_parser("local", help="Self-training local offline model status, train, distill, download, bench, or test")
    local_parser.add_argument("local_action", nargs="?", default="status", choices=["status", "train", "distill", "download", "test", "bench", "prune", "profile"])
    local_parser.add_argument("local_query", nargs="?", default=None, help="Optional test query")
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

    if filtered_argv and filtered_argv[0] in {"serve", "chat", "status", "digest", "doctor", "plugin", "search", "events", "name", "quick", "fix", "hook", "brain", "evolve", "prompts", "local", "neural", "login", "logout", "export", "import", "top", "rollback", "mcp", "schedule", "notifications", "cortex", "-h", "--help"}:
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
            _cmd_doctor(settings.base_url)
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
            _cmd_fix(settings.base_url, locale_override)
        elif args.subcommand == "hook":
            _cmd_hook()
        elif args.subcommand == "brain":
            _cmd_brain(settings.base_url, args.brain_action, locale_override)
        elif args.subcommand == "evolve":
            _cmd_evolve(settings.base_url, args.evolve_action, locale_override)
        elif args.subcommand == "prompts":
            _cmd_prompts(settings.base_url, args.prompts_action, args.prompt_key, locale_override)
        elif args.subcommand == "local":
            _cmd_local(settings.base_url, args.local_action, args.local_query, locale_override)
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
            from pathlib import Path
            bundle = export_brain_bundle(Storage())
            Path(args.file).write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Exported brain bundle to {args.file}")
        elif args.subcommand == "import":
            from pathlib import Path
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

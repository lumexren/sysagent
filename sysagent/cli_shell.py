"""Status, quick launcher, shell hook, fix, resume, and self-guard CLI subcommands."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .config import DATA_DIR, get_settings, persist_env_var
from .i18n import t
from .ipc import ipc_get, ipc_post, ipc_ws_connect
from .storage import Storage


def _auth_headers() -> dict[str, str]:
    tok = get_settings().auth_token
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _ws_url(base_url: str, session_id: str) -> str:
    ws_base = base_url.replace("http://", "ws://").replace("https://", "wss://")
    tok = get_settings().auth_token
    q = f"?token={tok}" if tok else ""
    return f"{ws_base}/ws/chat/{session_id}{q}"


def _fmt_uptime(sec: int) -> str:
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}d {h}h"
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
        await _cmd_fix_async(base_url, locale)
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


def _resolve_hook_shell(shell: str = "auto") -> str:
    """Resolve `auto` to `zsh` or `bash` based on `$SHELL`."""
    sh = (shell or "auto").strip().lower()
    if sh in {"zsh", "bash"}:
        return sh
    return "zsh" if "zsh" in os.environ.get("SHELL", "").lower() else "bash"


def _render_shell_hook(shell: str = "auto") -> str:
    """Render the Bash or Zsh error-capturing hook with safe `--record` JSON logging and `srun` stderr capture."""
    from .cli_ops import _prog_name

    target_shell = _resolve_hook_shell(shell)
    cli_bin = _prog_name()
    stderr_log = DATA_DIR / "last_stderr.log"
    if target_shell == "zsh":
        return (
            f"# Add this to ~/.zshrc (or run `{cli_bin} hook zsh --install`):\n"
            "autoload -Uz add-zsh-hook\n"
            "_sysagent_preexec() {\n"
            '  _SYSAGENT_LAST_CMD="$1"\n'
            "}\n"
            "_sysagent_err_hook() {\n"
            "  local ec=$?\n"
            '  if [ $ec -ne 0 ] && [ -n "$_SYSAGENT_LAST_CMD" ]; then\n'
            f'    command {cli_bin} hook --record "$ec" "$PWD" "$_SYSAGENT_LAST_CMD" 2>/dev/null || true\n'
            "  fi\n"
            '  _SYSAGENT_LAST_CMD=""\n'
            "}\n"
            "add-zsh-hook preexec _sysagent_preexec\n"
            "add-zsh-hook precmd _sysagent_err_hook\n"
            f'srun() {{ "$@" 2> >(tee "{stderr_log}" >&2); }}\n'
            f'alias fix="{cli_bin} fix"\n'
        )
    return (
        f"# Add this to ~/.bashrc (or run `{cli_bin} hook bash --install`):\n"
        "_sysagent_err_hook() {\n"
        "  local ec=$?\n"
        "  if [ $ec -ne 0 ]; then\n"
        "    local last_cmd\n"
        "    last_cmd=$(fc -ln -1 2>/dev/null | sed 's/^[[:space:]]*//')\n"
        f'    command {cli_bin} hook --record "$ec" "$PWD" "$last_cmd" 2>/dev/null || true\n'
        "  fi\n"
        "}\n"
        'PROMPT_COMMAND="_sysagent_err_hook${PROMPT_COMMAND:+; $PROMPT_COMMAND}"\n'
        f'srun() {{ "$@" 2> >(tee "{stderr_log}" >&2); }}\n'
        f'alias fix="{cli_bin} fix"\n'
    )


def _install_shell_hook(
    shell: str = "auto", rc_path: Path | None = None
) -> tuple[Path, str]:
    """Idempotently install or update the managed SysAgent shell hook block in ~/.bashrc or ~/.zshrc."""
    target_shell = _resolve_hook_shell(shell)
    target_rc = rc_path or (Path.home() / (".zshrc" if target_shell == "zsh" else ".bashrc"))
    begin_mark = "# >>> sysagent shell hook >>>"
    end_mark = "# <<< sysagent shell hook <<<"
    body = _render_shell_hook(target_shell).strip()
    block = f"{begin_mark}\n{body}\n{end_mark}"

    existing = target_rc.read_text(encoding="utf-8", errors="replace") if target_rc.is_file() else ""
    pat = re.compile(rf"{re.escape(begin_mark)}.*?{re.escape(end_mark)}", flags=re.DOTALL)
    if pat.search(existing):
        updated = pat.sub(block, existing, count=1)
    else:
        sep = "" if (not existing or existing.endswith("\n")) else "\n"
        extra_nl = "\n" if existing.strip() else ""
        updated = f"{existing}{sep}{extra_nl}{block}\n"
    target_rc.parent.mkdir(parents=True, exist_ok=True)
    target_rc.write_text(updated, encoding="utf-8")
    return target_rc, target_shell


def _cmd_hook(
    shell: str = "auto",
    install: bool = False,
    record_args: list[str] | None = None,
) -> None:
    from . import cli as _cli_mod

    if record_args is not None:
        exit_code = int(record_args[0])
        cwd_arg = str(record_args[1] or ".")
        cmd_arg = str(record_args[2] or "").strip()
        stderr_log = _cli_mod.DATA_DIR / "last_stderr.log"
        stderr_txt = ""
        if stderr_log.is_file():
            try:
                stderr_txt = stderr_log.read_text(encoding="utf-8", errors="replace")
            except Exception:
                stderr_txt = ""
            finally:
                stderr_log.unlink(missing_ok=True)
        _cli_mod.record_last_shell_error(cmd_arg, exit_code, cwd_arg, stderr_txt)
        return
    if install:
        rc_path, target_shell = _cli_mod._install_shell_hook(shell)
        print(f"Installed SysAgent {target_shell} hook in {rc_path}")
        print(f"Reload your shell or run: source {rc_path}")
        return
    print(_cli_mod._render_shell_hook(shell))


def _prepare_fix_prompt(
    locale: str | None = None,
    extra_cmd: str | None = None,
    no_cache: bool = False,
) -> str | None:
    from . import cli as _cli_mod

    loc = locale or get_settings().locale
    err_file = _cli_mod.DATA_DIR / "last_error.json"
    stderr_log = _cli_mod.DATA_DIR / "last_stderr.log"
    data: dict[str, object] | None = None
    if err_file.exists():
        try:
            loaded = json.loads(err_file.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, dict) else {"command": str(loaded), "exit_code": 1, "cwd": "."}
        except json.JSONDecodeError:
            raw = err_file.read_text(encoding="utf-8", errors="replace")
            data = {"command": raw.strip(), "exit_code": 1, "cwd": "."}
    elif extra_cmd and extra_cmd.strip():
        data = {"command": extra_cmd.strip(), "exit_code": 1, "cwd": str(Path.cwd()), "stderr": ""}
    if data is None:
        print(t("cli_no_last_error", loc), file=sys.stderr)
        return None
    cmd_str = str(data.get("command") or extra_cmd or "")
    exit_code = int(data.get("exit_code") or 1)
    stderr_txt = str(data.get("stderr") or "").strip()
    if not stderr_txt and stderr_log.is_file():
        try:
            stderr_txt = "\n".join(stderr_log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])[-2500:]
        except Exception:
            stderr_txt = ""
    recovery_hint = ""
    try:
        from .self_guard import match_cached_fix
        from .self_resilience import build_recovery_hint

        with _cli_mod.Storage() as st:
            if not no_cache:
                hit = match_cached_fix(st, cmd_str, exit_code)
                if hit is not None:
                    print(f"[0-token fix memory | match {int(hit['score'] * 100)}%] ({hit['topic']})")
                    print(hit["lesson"])
                    return ""
            hint_val = build_recovery_hint(
                st,
                "run_command",
                {"command": cmd_str},
                json.dumps({"exit_code": exit_code, "stderr": stderr_txt}, ensure_ascii=False),
            )
            if hint_val:
                recovery_hint = hint_val
    except Exception:
        pass
    parts = [
        f"The user's last shell command failed in `{data.get('cwd') or '.'}` with exit code {exit_code}:",
        f"`{cmd_str}`",
    ]
    if stderr_txt:
        parts.append(f"Captured stderr (last lines):\n```text\n{stderr_txt[-1800:]}\n```")
    if recovery_hint:
        parts.append(recovery_hint)
    parts.append("Explain why it likely failed and suggest (or run if safe/approved) the fix.")
    return "\n".join(parts)


async def _cmd_fix_async(
    base_url: str,
    locale: str | None = None,
    extra_cmd: str | None = None,
    no_cache: bool = False,
) -> None:
    from . import cli as _cli_mod

    prompt = _cli_mod._prepare_fix_prompt(locale, extra_cmd=extra_cmd, no_cache=no_cache)
    if not prompt:
        return
    await _cli_mod._send_and_stream(prompt, f"fix-{uuid.uuid4().hex[:6]}", base_url, locale)


def _cmd_fix(
    base_url: str,
    locale: str | None = None,
    extra_cmd: str | None = None,
    no_cache: bool = False,
) -> None:
    from . import cli as _cli_mod

    prompt = _cli_mod._prepare_fix_prompt(locale, extra_cmd=extra_cmd, no_cache=no_cache)
    if prompt is None:
        sys.exit(1)
    if not prompt:
        return
    asyncio.run(_cli_mod._send_and_stream(prompt, f"fix-{uuid.uuid4().hex[:6]}", base_url, locale))


def _cmd_resume(
    base_url: str,
    query_parts: list[str] | None = None,
    list_only: bool = False,
    fork_new: bool = False,
    locale: str | None = None,
) -> None:
    from . import cli as _cli_mod
    from .cli_ops import _prog_name

    query = " ".join(query_parts or []).strip()
    if list_only or not query:
        cps: list[dict[str, object]] = []
        try:
            resp = ipc_get(f"{base_url}/api/checkpoints", params={"limit": 15}, headers=_auth_headers(), timeout=5)
            resp.raise_for_status()
            cps = list(resp.json().get("checkpoints") or [])
        except Exception:
            with Storage() as st:
                cps = list(st.list_session_checkpoints(limit=15))
        if not cps:
            print("No session checkpoints saved yet.")
            return
        print("\033[1;36mRecent Semantic Session Checkpoints:\033[0m")
        for cp in cps:
            sid = str(cp.get("session_id") or "")
            ts = str(cp.get("created_at") or "")[:19].replace("T", " ")
            topic = " ".join(str(cp.get("topic") or "").split())[:60]
            files = ", ".join(Path(str(f)).name for f in (cp.get("files") or [])[:3])
            suffix = f" \033[90m[{files}]\033[0m" if files else ""
            print(f"  {sid:<16} {ts}  {topic}{suffix}")
        return

    res_data: dict[str, object] | None = None
    try:
        resp = ipc_post(
            f"{base_url}/api/sessions/resume",
            json={"query": query},
            headers=_auth_headers(),
            timeout=5,
        )
        resp.raise_for_status()
        res_data = resp.json()
    except Exception:
        with Storage() as st:
            hits = st.search_session_checkpoints(query, limit=3)
            if hits:
                cp0 = hits[0]
                sid0 = str(cp0["session_id"])
                res_data = {
                    "found": True,
                    "session_id": sid0,
                    "checkpoint": cp0,
                    "messages": st.get_messages(sid0, limit=12),
                    "context_preview": Storage.format_checkpoint_context(cp0),
                }

    if not res_data or not res_data.get("found"):
        print(f"No session or checkpoint matching '{query}'. Try `{_prog_name()} resume --list`.", file=sys.stderr)
        sys.exit(1)

    target_sid = str(res_data["session_id"])
    cp = res_data.get("checkpoint")
    if fork_new:
        new_sid = f"cli-{uuid.uuid4().hex[:8]}"
        if isinstance(cp, dict):
            with Storage() as st:
                st.save_session_checkpoint(
                    session_id=new_sid,
                    topic=str(cp.get("topic") or query),
                    summary=str(cp.get("summary") or ""),
                    files=[str(x) for x in (cp.get("files") or [])],
                    commands=[str(x) for x in (cp.get("commands") or [])],
                    decisions=[str(x) for x in (cp.get("decisions") or [])],
                    cwd=str(cp.get("cwd") or ""),
                    model=str(cp.get("model") or ""),
                )
        target_sid = new_sid

    _cli_mod._save_last_session_id(target_sid)
    ctx_prev = str(res_data.get("context_preview") or "")
    msgs = list(res_data.get("messages") or [])
    print(f"\033[1;36mResuming session {target_sid}\033[0m ({len(msgs)} recent messages):")
    if ctx_prev:
        print(f"  \033[36m[Checkpoint]\033[0m {ctx_prev}")
    for m in msgs[-4:]:
        if isinstance(m, dict):
            snippet = " ".join(str(m.get("content") or "").split())[:100]
            print(f"  \033[90m{m.get('role')}: {snippet}\033[0m")
    asyncio.run(_cli_mod._interactive_chat(base_url, locale, initial_session_id=target_sid))


def _cmd_self(
    base_url: str,
    action: str,
    locale_override: str | None = None,
    target_arg: str | None = None,
) -> None:
    loc = locale_override or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action in {"canary", "calibrate", "profile", "eval"}:
            try:
                resp = ipc_post(f"{base_url}/api/self/{action}", headers=hdrs, timeout=15)
                resp.raise_for_status()
                data: Any = resp.json()
            except Exception:
                from .self_guard import SelfGuard

                with Storage() as st:
                    sg = SelfGuard(get_settings(), st)
                    fn = {
                        "canary": sg.run_canary_probes,
                        "calibrate": sg.calibrate_thresholds,
                        "profile": sg.run_memory_profiler,
                        "eval": sg.run_eval_gate,
                    }[action]
                    data = fn()
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return

        if action in {"drift", "breakers", "quota", "twin", "blast", "dpo"}:
            target_sym = (target_arg or "execute_tool").strip()
            try:
                params = {"target": target_sym} if action == "blast" else None
                resp = ipc_get(f"{base_url}/api/self/{action}", params=params, headers=hdrs, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                from .self_guard import SelfGuard

                with Storage() as st:
                    sg = SelfGuard(get_settings(), st)
                    if action == "drift":
                        data = sg.run_drift_audit()
                    elif action == "twin":
                        from .self_learn import analyze_causal_digital_twin

                        data = analyze_causal_digital_twin(st)
                    elif action == "blast":
                        from .evolve_ast import analyze_symbol_blast_radius

                        data = analyze_symbol_blast_radius(target_sym, storage=st)
                    elif action == "dpo":
                        data = sg.audit_dpo_preferences(auto_repair=True)
                    else:
                        data = sg.get_resilience_status().get(
                            "circuit_breaker" if action == "breakers" else "quota", {}
                        )
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return

        if action == "autopsy":
            data = None
            try:
                resp = ipc_get(f"{base_url}/api/self/guard", headers=hdrs, timeout=10)
                resp.raise_for_status()
                data = (resp.json() or {}).get("last_autopsy")
            except Exception:
                data = None
            if not data:
                from .self_guard import SelfGuard

                with Storage() as st:
                    data = SelfGuard(get_settings(), st).run_crash_autopsy(record=False)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return

        try:
            resp = ipc_get(f"{base_url}/api/self/guard", headers=hdrs, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            from .self_guard import SelfGuard

            with Storage() as st:
                data = SelfGuard(get_settings(), st).get_status()
        if action == "skills":
            print(json.dumps(data.get("skill_candidates", []), indent=2, ensure_ascii=False))
        else:
            print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)

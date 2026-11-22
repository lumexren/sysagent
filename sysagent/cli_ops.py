"""Diagnostics, plugin, search, rollback, schedule, notification, and cortex CLI subcommands."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

from .config import get_settings
from .i18n import t
from .ipc import ipc_delete, ipc_get, ipc_post
from .storage import Storage


def _auth_headers() -> dict[str, str]:
    tok = get_settings().auth_token
    return {"Authorization": f"Bearer {tok}"} if tok else {}


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
            src = str(r.get("type", r.get("source", "")))
            snip = str(r.get("detail", r.get("snippet", "")))
            tbl.add_row(src, str(r.get("title", ""))[:32], snip[:80])
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
                        str(p.get("file_path", p.get("target_file", ""))),
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



"""Live terminal TUI dashboard (`sysagent top`) powered by Rich."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import httpx
import psutil
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _bar(pct: float, width: int = 18) -> Text:
    clamped = max(0.0, min(100.0, float(pct)))
    filled = round((clamped / 100.0) * width)
    color = "green" if clamped < 65 else ("yellow" if clamped < 85 else "red")
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="bright_black")
    t.append(f" {clamped:5.1f}%", style=f"bold {color}")
    return t


def _fmt_uptime(sec: int) -> str:
    h, rem = divmod(max(0, int(sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def collect_top_state(base_url: str, headers: dict[str, str]) -> dict[str, Any]:
    """Fetch live daemon telemetry, falling back to local psutil/storage if offline."""
    try:
        with httpx.Client(headers=headers, timeout=2.0) as client:
            st = client.get(f"{base_url}/api/status")
            st.raise_for_status()
            status_data = st.json()
            ev = client.get(f"{base_url}/api/events", params={"limit": 6})
            events = ev.json().get("events", []) if ev.is_success else []
            sn = client.get(f"{base_url}/api/snapshots", params={"limit": 5})
            snapshots = sn.json() if sn.is_success else {}
            return {
                "online": True,
                "status": status_data,
                "events": events,
                "snapshots": snapshots,
            }
    except Exception:
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "online": False,
            "status": {
                "agent_name": "SysAgent",
                "model": "-",
                "uptime_sec": 0,
                "pending_approvals": 0,
                "pending_brain_questions": 0,
                "brain_rules": 0,
                "memories": 0,
                "lessons": 0,
                "evolve_proposals": 0,
                "health": {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "ram_percent": mem.percent,
                    "ram_used_gb": round(mem.used / (1024**3), 2),
                    "ram_total_gb": round(mem.total / (1024**3), 2),
                    "disk_percent": disk.percent,
                    "disk_free_gb": round(disk.free / (1024**3), 2),
                    "max_temp_c": None,
                    "zombie_count": 0,
                },
            },
            "events": [],
            "snapshots": {},
        }


def build_dashboard(state: dict[str, Any]) -> Layout:
    """Build a Rich Layout representing the current SysAgent TUI frame."""
    st = state.get("status", {})
    health = st.get("health", {})
    online = bool(state.get("online"))

    # Header
    hdr = Table.grid(expand=True)
    hdr.add_column(justify="left")
    hdr.add_column(justify="center")
    hdr.add_column(justify="right")
    badge = "[bold green]● ONLINE[/]" if online else "[bold red]○ OFFLINE[/]"
    hdr.add_row(
        f"[bold white]{st.get('agent_name', 'SysAgent')}[/]  {badge}  "
        f"[dim]model:[/] [cyan]{st.get('model', '-')}[/]",
        f"[dim]uptime:[/] {_fmt_uptime(st.get('uptime_sec', 0))}  "
        f"[dim]approvals:[/] {st.get('pending_approvals', 0)}",
        f"[dim]{datetime.now().strftime('%H:%M:%S')}  (Ctrl+C to exit)[/]",
    )

    # System Metrics
    sys_tbl = Table.grid(padding=(0, 1), expand=True)
    sys_tbl.add_column(width=10, style="bold")
    sys_tbl.add_column()
    sys_tbl.add_column(justify="right", style="dim")

    sys_tbl.add_row("CPU", _bar(health.get("cpu_percent", 0.0)), f"zombies: {health.get('zombie_count', 0)}")
    sys_tbl.add_row(
        "RAM",
        _bar(health.get("ram_percent", 0.0)),
        f"{health.get('ram_used_gb', 0)}/{health.get('ram_total_gb', 0)} GB",
    )
    burn = health.get("disk_burn_rate_gb_hr")
    burn_str = f"{health.get('disk_free_gb', 0)} GB free" + (f" ({burn:+.2f} GB/h)" if burn is not None else "")
    sys_tbl.add_row("Disk", _bar(health.get("disk_percent", 0.0)), burn_str)
    temp = health.get("max_temp_c")
    gpu = health.get("gpu") or {}
    gpu_str = f"GPU {gpu.get('util_percent', '?')}% {gpu.get('temp_c', '?')}°C" if gpu else "no GPU"
    sys_tbl.add_row("Thermals", Text(f"{temp}°C" if temp is not None else "n/a", style="cyan"), gpu_str)

    # Brain & Evolve
    brain_tbl = Table.grid(padding=(0, 2), expand=True)
    brain_tbl.add_column(style="dim")
    brain_tbl.add_column(justify="right", style="bold")
    evolve_info = st.get("evolve") or {}
    brain_tbl.add_row("Brain Rules", str(st.get("brain_rules", 0)))
    brain_tbl.add_row("Memories / Lessons", f"{st.get('memories', 0)} / {st.get('lessons', 0)}")
    brain_tbl.add_row("Pending Questions", str(st.get("pending_brain_questions", 0)))
    brain_tbl.add_row("Evolve Proposals", str(st.get("evolve_proposals", 0)))
    brain_tbl.add_row("Evolve Mode", str(evolve_info.get("auto_apply", evolve_info.get("enabled", "-"))))

    # Events
    ev_tbl = Table(show_header=True, header_style="bold dim", box=None, expand=True)
    ev_tbl.add_column("Time", width=8, style="dim")
    ev_tbl.add_column("Source", width=12, style="cyan")
    ev_tbl.add_column("Event")
    for ev in (state.get("events") or [])[:6]:
        ts = str(ev.get("timestamp", ev.get("created_at", "")))[-8:]
        ev_tbl.add_row(ts, str(ev.get("source", ev.get("kind", "-")))[:12], str(ev.get("message", ev.get("detail", "")))[:90])
    if not state.get("events"):
        ev_tbl.add_row("-", "-", "[dim]No recent events[/]")

    # Recent Snapshots / Audit
    sn_tbl = Table(show_header=True, header_style="bold dim", box=None, expand=True)
    sn_tbl.add_column("ID", width=5, style="dim")
    sn_tbl.add_column("Action", width=11, style="magenta")
    sn_tbl.add_column("Target")
    sn_tbl.add_column("State", width=8, justify="right")
    audits = (state.get("snapshots") or {}).get("audit_log") or []
    for row in audits[:5]:
        state_lbl = "[dim]undone[/]" if row.get("undone") else "[green]active[/]"
        sn_tbl.add_row(str(row.get("id", "")), str(row.get("action", "")), str(row.get("target_path", ""))[:55], state_lbl)
    if not audits:
        sn_tbl.add_row("-", "-", "[dim]No recent snapshots[/]", "-")

    layout = Layout()
    layout.split_column(
        Layout(Panel(hdr, border_style="bright_black"), size=3),
        Layout(name="middle", size=9),
        Layout(name="bottom"),
    )
    layout["middle"].split_row(
        Layout(Panel(sys_tbl, title="System Telemetry", border_style="cyan"), ratio=3),
        Layout(Panel(brain_tbl, title="Brain & Evolve", border_style="green"), ratio=2),
    )
    layout["bottom"].split_row(
        Layout(Panel(ev_tbl, title="Watcher Events", border_style="yellow")),
        Layout(Panel(sn_tbl, title="Snapshots / Audit", border_style="magenta")),
    )
    return layout


def run_top(
    base_url: str,
    headers: dict[str, str],
    interval: float = 1.5,
    once: bool = False,
) -> None:
    """Run the live terminal dashboard until interrupted (or print once if once=True)."""
    console = Console()
    if once:
        console.print(build_dashboard(collect_top_state(base_url, headers)))
        return

    try:
        with Live(
            build_dashboard(collect_top_state(base_url, headers)),
            console=console,
            screen=True,
            refresh_per_second=2,
        ) as live:
            while True:
                time.sleep(max(0.4, interval))
                live.update(build_dashboard(collect_top_state(base_url, headers)))
    except KeyboardInterrupt:
        pass

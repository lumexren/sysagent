"""Background system listener (daemon watcher) for SysAgent."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import shutil
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Awaitable, Callable

import psutil

from .config import DATA_DIR, Settings
from .i18n import t
from .storage import Storage


@dataclass
class SystemEvent:
    """A single event captured by the background listener."""

    timestamp: str
    category: str  # "health" | "filesystem" | "systemd" | "info"
    severity: str  # "info" | "warning" | "critical"
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SystemWatcher:
    """Continuously listens to system metrics, systemd units, and watched folders."""

    def __init__(
        self,
        settings: Settings,
        storage: Storage | None = None,
        automation_callback: Callable[[dict[str, Any], str], Awaitable[None]] | None = None,
        max_events: int = 200,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.automation_callback = automation_callback
        self.inspect_callback: Callable[[str], Awaitable[None]] | None = None
        self.events: deque[SystemEvent] = deque(maxlen=max_events)
        self._dir_snapshots: dict[str, set[str]] = {}
        self._failed_units: set[str] = set()
        self._failed_units_seeded: bool = False
        self._known_ports: set[int] | None = None
        self._known_usb: set[str] | None = None
        self._rss_history: dict[int, deque[float]] = {}
        self._disk_history: deque[tuple[float, float]] = deque(maxlen=30)
        self._last_top_cpu_proc: str = ""
        self._last_zombie_count: int = 0
        self._last_alert_time: dict[str, float] = {}
        self._last_err_mtime: float = 0.0
        self._err_repeat_count: int = 0
        self._last_err_cmd: str = ""
        self.current_context: str = "normal"  # "developer" | "media_meeting" | "normal"
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._loop_ref: asyncio.AbstractEventLoop | None = None
        from .notification import NotificationCenter

        self.notifier = NotificationCenter(settings, storage)
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def add_event(
        self, category: str, severity: str, message: str, notify: bool = False
    ) -> SystemEvent:
        event = SystemEvent(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            category=category,
            severity=severity,
            message=message,
        )
        self.events.appendleft(event)
        if notify and (self.current_context != "media_meeting" or severity == "critical"):
            title_map = {
                "health": "System Health Alert",
                "security": "Security & Device Alert",
                "systemd": "Systemd Service Alert",
                "filesystem": "Filesystem Event",
                "info": f"{self.settings.agent_name} Update",
            }
            urgency = "critical" if severity == "critical" else ("normal" if severity == "warning" else "low")
            loc = self.settings.locale
            default_actions = [
                ("inspect", t("notif_inspect", loc)),
            ]

            def _handle_event_action(selected: str) -> None:
                if selected == "inspect" and self.inspect_callback and self._loop_ref:
                    asyncio.run_coroutine_threadsafe(
                        self.inspect_callback(f"[{category}/{severity}] {message}"),
                        self._loop_ref,
                    )

            self._send_desktop_notification(
                title_map.get(category, self.settings.agent_name),
                message,
                urgency=urgency,
                category=category,
                actions=default_actions,
                on_action=_handle_event_action,
            )
        return event

    def get_events(self, limit: int = 30) -> list[dict[str, Any]]:
        return [e.to_dict() for e in list(self.events)[:limit]]

    @staticmethod
    def _read_max_temperature_c() -> float | None:
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return None
            vals = [
                float(entry.current)
                for entries in temps.values()
                for entry in entries
                if getattr(entry, "current", None) is not None and entry.current > 0
            ]
            return round(max(vals), 1) if vals else None
        except Exception:
            return None

    @staticmethod
    def _read_gpu_stats() -> dict[str, Any] | None:
        if not shutil.which("nvidia-smi"):
            return None
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return None
            parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 5:
                return {
                    "name": parts[0],
                    "util_percent": float(parts[1]),
                    "vram_used_mb": int(float(parts[2])),
                    "vram_total_mb": int(float(parts[3])),
                    "temp_c": round(float(parts[4]), 1),
                }
        except Exception:
            pass
        return None

    def _compute_disk_burn_rate(self, current_free_gb: float) -> tuple[float, float | None]:
        """Return (burn_rate_gb_per_hour, estimated_hours_to_full)."""
        now_m = time.monotonic()
        self._disk_history.append((now_m, current_free_gb))
        if len(self._disk_history) < 2:
            return 0.0, None
        t0, free0 = self._disk_history[0]
        dt_hours = (now_m - t0) / 3600.0
        if dt_hours < (10.0 / 3600.0):
            return 0.0, None
        burned_gb = free0 - current_free_gb
        if burned_gb <= 0:
            return 0.0, None
        rate = round(burned_gb / dt_hours, 2)
        hours_left = round(current_free_gb / rate, 1) if rate > 0.05 else None
        return rate, hours_left

    def get_health_snapshot(self, ttl_sec: float = 2.5, force: bool = False) -> dict[str, Any]:
        """Return current CPU, RAM, disk, temperature, zombie count, and burn-rate stats (TTL cached)."""
        now = time.monotonic()
        cached = getattr(self, "_health_cache", None)
        cached_ts = float(getattr(self, "_health_cache_ts", 0.0) or 0.0)
        if not force and cached is not None and ttl_sec > 0.0 and (now - cached_ts) < ttl_sec:
            return dict(cached)

        cpu = psutil.cpu_percent(interval=None)
        if cpu == 0.0:
            cpu = psutil.cpu_percent(interval=0.05)
        try:
            load1 = os.getloadavg()[0]
            cores = max(1, os.cpu_count() or 1)
            load_pct = round(min(100.0, (load1 / cores) * 100.0), 1)
            if cpu >= 90.0 and load_pct < 70.0:
                cpu = round((cpu * 0.3) + (load_pct * 0.7), 1)
        except Exception:
            pass
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        free_gb = round(disk.free / (1024**3), 2)
        burn_rate, hours_to_full = self._compute_disk_burn_rate(free_gb)
        boot = datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")
        snap = {
            "cpu_percent": round(cpu, 1),
            "ram_percent": round(mem.percent, 1),
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "disk_percent": round(disk.percent, 1),
            "disk_free_gb": free_gb,
            "disk_burn_rate_gb_hr": burn_rate,
            "disk_hours_to_full": hours_to_full,
            "max_temp_c": self._read_max_temperature_c(),
            "gpu": self._read_gpu_stats(),
            "zombie_count": self._last_zombie_count,
            "boot_time": boot,
        }
        self._health_cache = snap
        self._health_cache_ts = now
        return dict(snap)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            self._loop_ref = asyncio.get_running_loop()
        except RuntimeError:
            self._loop_ref = None
        self._init_dir_snapshots()
        self.add_event("info", "info", "SysAgent background listener started.")
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _init_dir_snapshots(self) -> None:
        for d in self.settings.watch_dirs:
            path = Path(d).expanduser()
            if path.is_dir():
                try:
                    self._dir_snapshots[str(path)] = {
                        p.name for p in path.iterdir()
                    }
                except PermissionError:
                    pass

    async def _loop(self) -> None:
        while self._running:
            try:
                added_files = await asyncio.to_thread(self._check_once)
                await self._check_automations(added_files or [])
            except Exception as exc:
                self.add_event("info", "warning", f"Listener error: {exc}")
            await asyncio.sleep(max(5, self.settings.monitor_interval_sec))

    def _can_alert(self, key: str, cooldown_sec: int = 300) -> bool:
        now = time.monotonic()
        mult = self.storage.get_notification_cooldown_multiplier(key) if self.storage else 1
        last = self._last_alert_time.get(key, 0.0)
        if now - last >= (cooldown_sec * mult):
            self._last_alert_time[key] = now
            return True
        return False

    def _check_once(self) -> list[Path]:
        self._check_focus_and_usage()

        snap = self.get_health_snapshot()
        neural_wf: dict[str, Any] = {}
        if self.storage is not None:
            try:
                from .neural import NeuralEngine

                neural_wf = NeuralEngine(self.storage).observe_world_state(snap, active_role=self.current_context)
                if (
                    neural_wf.get("anomaly_detected")
                    and float(neural_wf.get("anomaly_score", 0.0) or 0.0) >= 0.40
                    and self._can_alert("neural_anomaly", cooldown_sec=600)
                ):
                    self.add_event(
                        "health",
                        "warning",
                        f"Neural telemetry anomaly (score={neural_wf['anomaly_score']}): 10m RAM={neural_wf['pred_ram_10m']}%, CPU={neural_wf['pred_cpu_10m']}%",
                    )
            except Exception:
                pass
        transient_spike = (
            int(neural_wf.get("samples", 0) or 0) >= 6
            and float(neural_wf.get("bottleneck_prob", 0.0) or 0.0) < 0.25
            and not neural_wf.get("anomaly_detected")
        )
        if snap["cpu_percent"] >= self.settings.cpu_alert_threshold and self._can_alert("cpu"):
            top_proc = self._last_top_cpu_proc or self._top_cpu_process()
            detail = f" ({top_proc})" if top_proc else ""
            self.add_event(
                "health",
                "warning",
                f"High CPU usage: {snap['cpu_percent']}%{detail}",
                notify=not transient_spike,
            )
        if snap["ram_percent"] >= self.settings.ram_alert_threshold and self._can_alert("ram"):
            self.add_event(
                "health",
                "warning",
                f"High RAM usage: {snap['ram_percent']}% ({snap['ram_used_gb']}/{snap['ram_total_gb']} GB)",
                notify=not transient_spike,
            )
        if snap["disk_percent"] >= self.settings.disk_alert_threshold and self._can_alert("disk"):
            self.add_event(
                "health",
                "critical",
                f"Low disk space: {snap['disk_percent']}% used ({snap['disk_free_gb']} GB free)",
                notify=True,
            )
        if (
            snap.get("disk_burn_rate_gb_hr", 0.0) >= 5.0
            and snap.get("disk_hours_to_full") is not None
            and snap["disk_hours_to_full"] <= 24.0
            and self._can_alert("disk_burn", cooldown_sec=900)
        ):
            self.add_event(
                "health",
                "warning",
                f"Rapid disk fill rate: {snap['disk_burn_rate_gb_hr']} GB/hr (~{snap['disk_hours_to_full']}h until full)",
                notify=True,
            )
        if (
            snap.get("max_temp_c") is not None
            and snap["max_temp_c"] >= 85.0
            and self._can_alert("temp", cooldown_sec=600)
        ):
            self.add_event(
                "health",
                "warning",
                f"High hardware temperature: {snap['max_temp_c']}°C",
                notify=True,
            )

        self._check_failed_systemd_units()

        self._check_security_and_devices()

        self._check_disk_maintenance()

        newly_added: list[Path] = []
        for d_str, prev_files in list(self._dir_snapshots.items()):
            path = Path(d_str)
            if not path.is_dir():
                continue
            try:
                current_files = {p.name for p in path.iterdir()}
            except PermissionError:
                continue

            added = current_files - prev_files
            removed = prev_files - current_files
            for name in sorted(added)[:5]:
                self.add_event("filesystem", "info", f"New file ({path.name}): {name}")
                newly_added.append(path / name)
            for name in sorted(removed)[:5]:
                self.add_event("filesystem", "info", f"Removed file ({path.name}): {name}")
            self._dir_snapshots[d_str] = current_files
        if newly_added:
            self._check_downloads_organizer(newly_added)
        self._check_terminal_errors()
        return newly_added

    def _check_terminal_errors(self) -> None:
        """Proactively offer a fix notification when a terminal command fails twice in a row."""
        err_file = DATA_DIR / "last_error.json"
        if not err_file.is_file():
            return
        try:
            st = err_file.stat()
            if st.st_mtime <= self._last_err_mtime:
                return
            self._last_err_mtime = st.st_mtime
            import json as _json

            data = _json.loads(err_file.read_text(encoding="utf-8", errors="replace"))
            cmd = str(data.get("command") or "").strip()
            if not cmd:
                return
            base_cmd = cmd.split()[0]
            if base_cmd == self._last_err_cmd:
                self._err_repeat_count += 1
            else:
                self._last_err_cmd = base_cmd
                self._err_repeat_count = 1
            if self._err_repeat_count < 2 or not self._can_alert(f"term_err:{base_cmd}", cooldown_sec=300):
                return

            ec = data.get("exit_code", 1)
            cwd = data.get("cwd", "~")
            msg = f"Command failed {self._err_repeat_count}x (exit {ec}): {cmd[:80]}"
            self.add_event("health", "warning", msg)
            fb_key = f"term_err:{base_cmd}"

            def _on_err_act(act: str) -> None:
                if act == "inspect" and self.inspect_callback and self._loop_ref:
                    if self.storage:
                        self.storage.record_notification_feedback(fb_key, "yes")
                    asyncio.run_coroutine_threadsafe(
                        self.inspect_callback(
                            f"Terminal command failed in {cwd} (exit {ec}): `{cmd}`. Explain why and suggest the exact fix."
                        ),
                        self._loop_ref,
                    )
                elif act == "dismiss" and self.storage:
                    self.storage.record_notification_feedback(fb_key, "dismiss")

            self._send_desktop_notification(
                f"{self.settings.agent_name} Terminal Fix",
                msg,
                urgency="normal",
                actions=[
                    ("inspect", t("notif_inspect", self.settings.locale)),
                    ("dismiss", t("notif_dismiss", self.settings.locale)),
                ],
                on_action=_on_err_act,
                on_timeout=(lambda: self.storage.record_notification_feedback(fb_key, "timeout")) if self.storage else None,
            )
        except Exception:
            pass

    def _check_downloads_organizer(self, newly_added: list[Path]) -> None:
        """Suggest or auto-move downloaded files based on learned extension habits."""
        default_targets = {
            ".pdf": "~/Documents",
            ".doc": "~/Documents",
            ".docx": "~/Documents",
            ".iso": "~/Downloads/ISOs",
            ".zip": "~/Downloads/Archives",
            ".deb": "~/Downloads/Packages",
        }
        learned_targets: dict[str, str] = {}
        if self.storage:
            for m in self.storage.list_memories():
                k = str(m.get("key", ""))
                if k.startswith("org_ext:"):
                    learned_targets[k[8:].lower()] = str(m.get("value", ""))

        for p in newly_added:
            if not p.is_file():
                continue
            name = p.name
            if name.startswith(".") or name.endswith((".part", ".crdownload", ".tmp")):
                continue
            ext = p.suffix.lower()
            target_str = learned_targets.get(ext) or default_targets.get(ext)
            if not target_str:
                continue
            dest_dir = Path(target_str).expanduser()
            if p.parent.resolve() == dest_dir.resolve():
                continue

            fb_key = f"org:{ext}"
            auto_move = False
            if self.storage and ext in learned_targets:
                pkey = self.storage._normalize_feedback_key(fb_key)
                with self.storage._lock, self.storage._connect() as conn:
                    row = conn.execute(
                        "SELECT yes_count, dismiss_count FROM notification_feedback WHERE pattern_key = ?",
                        (pkey,),
                    ).fetchone()
                if row and int(row["yes_count"]) >= 2 and int(row["dismiss_count"]) == 0:
                    auto_move = True

            def _move_file(src: Path = p, dst_dir: Path = dest_dir) -> Path:
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                if dst.exists():
                    dst = dst_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
                shutil.move(str(src), str(dst))
                return dst

            if auto_move:
                try:
                    moved = _move_file()
                    self.add_event("filesystem", "info", f"Auto-organized {p.name} -> {moved.parent}")
                except Exception:
                    pass
                continue

            if not self._can_alert(fb_key, cooldown_sec=1800):
                continue

            def _on_org_act(act: str, src: Path = p, dst_dir: Path = dest_dir, ext_k: str = ext, fk: str = fb_key) -> None:
                if act == "move":
                    try:
                        moved = _move_file(src, dst_dir)
                        if self.storage:
                            self.storage.set_memory(f"org_ext:{ext_k}", str(dst_dir))
                            self.storage.record_notification_feedback(fk, "yes")
                        self._send_desktop_notification(
                            f"{self.settings.agent_name}",
                            f"Moved {src.name} -> {moved.parent}",
                        )
                    except Exception:
                        pass
                elif act == "dismiss" and self.storage:
                    self.storage.record_notification_feedback(fk, "dismiss")

            self._send_desktop_notification(
                "Smart Downloads Organizer",
                f"Move {name} to {target_str}?",
                urgency="low",
                actions=[
                    ("move", t("notif_move_remember", self.settings.locale)),
                    ("dismiss", t("notif_dismiss", self.settings.locale)),
                ],
                on_action=_on_org_act,
                on_timeout=(lambda fk=fb_key: self.storage.record_notification_feedback(fk, "timeout")) if self.storage else None,
            )
            break

    def _check_disk_maintenance(self) -> None:
        """Periodically inspect ~/.cache size and offer interactive cleanup if bloated (>= 3 GB)."""
        if not self._can_alert("cache_maintenance", cooldown_sec=21600):
            return
        if self.storage and self.storage.is_ignored_by_brain_rule("cache directory clean stale cache"):
            return
        cache_dir = Path.home() / ".cache"
        if not cache_dir.is_dir():
            return
        try:
            proc = subprocess.run(
                ["du", "-sm", str(cache_dir)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return
            size_mb = int(proc.stdout.split()[0])
            if size_mb < 3072:
                return
            size_gb = round(size_mb / 1024.0, 1)
            msg = f"Cache directory (~/.cache) is using {size_gb} GB. Clean stale files (>14 days)?"
            self.add_event("health", "info", msg)

            def _on_cache_action(act: str) -> None:
                if act != "clean":
                    return
                import os as _os

                cutoff = time.time() - (14 * 86400)
                freed_bytes = 0
                for root, _, files in _os.walk(cache_dir):
                    for fname in files:
                        fp = Path(root) / fname
                        try:
                            st = fp.stat()
                            if st.st_mtime < cutoff:
                                freed_bytes += st.st_size
                                fp.unlink(missing_ok=True)
                        except Exception:
                            pass
                freed_mb = round(freed_bytes / (1024 * 1024), 1)
                self._send_desktop_notification(
                    f"{self.settings.agent_name}",
                    f"Cleaned {freed_mb} MB of stale cache files.",
                )

            self._send_desktop_notification(
                "Disk Maintenance Suggestion",
                msg,
                urgency="low",
                category="health",
                actions=[
                    ("clean", t("notif_clean_cache", self.settings.locale)),
                    ("dismiss", t("notif_dismiss", self.settings.locale)),
                ],
                on_action=_on_cache_action,
            )
        except Exception:
            pass

    def _check_failed_systemd_units(self) -> None:
        if not shutil.which("systemctl"):
            return
        current_failed: set[str] = set()
        for scope_flag in ([], ["--user"]):
            try:
                proc = subprocess.run(
                    ["systemctl", *scope_flag, "--failed", "--no-legend", "--plain"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in proc.stdout.splitlines():
                    parts = line.split()
                    if parts:
                        uname = parts[0]
                        is_transient_app = (
                            uname.startswith("app-") and (uname.endswith(".scope") or "@" in uname)
                        ) or uname.startswith(("flatpak-", "snap."))
                        if is_transient_app:
                            continue
                        prefix = "user:" if scope_flag else "system:"
                        current_failed.add(f"{prefix}{uname}")
            except Exception:
                pass

        if not self._failed_units_seeded:
            self._failed_units = current_failed
            self._failed_units_seeded = True
            return

        new_failures = current_failed - self._failed_units
        for unit_key in sorted(new_failures):
            self.add_event(
                "systemd",
                "critical",
                f"Systemd unit failed: {unit_key}",
                notify=True,
            )
        self._failed_units = current_failed

    def _check_security_and_devices(self) -> None:
        trusted_port_procs = {"agy", "agy-node", "antigravity", "agentapi", "language_server", "code", "sysagent"}
        loopback_ips = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
        try:
            port_procs: dict[int, str] = {}
            for c in psutil.net_connections(kind="inet"):
                if c.status != psutil.CONN_LISTEN or not c.laddr:
                    continue
                ip = str(c.laddr.ip or "")
                port = int(c.laddr.port)
                if ip in loopback_ips or port >= 32768:
                    continue
                pname = ""
                if c.pid:
                    try:
                        pname = psutil.Process(c.pid).name().lower()
                    except Exception:
                        pname = ""
                if pname and any(tp in pname for tp in trusted_port_procs):
                    continue
                port_procs[port] = pname

            ports = set(port_procs.keys())
            if self._known_ports is not None:
                for p in sorted(ports - self._known_ports):
                    pname = port_procs.get(p)
                    proc_note = f" ({pname})" if pname else ""
                    self.add_event(
                        "security",
                        "warning",
                        f"New external listening network port opened: {p}{proc_note}",
                        notify=True,
                    )
            self._known_ports = ports
        except Exception:
            pass

        if shutil.which("lsusb"):
            try:
                proc = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
                if proc.returncode == 0:
                    usb_set = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
                    if self._known_usb is not None:
                        for dev in sorted(usb_set - self._known_usb):
                            self.add_event("security", "info", f"USB device connected: {dev}", notify=True)
                        for dev in sorted(self._known_usb - usb_set):
                            self.add_event("security", "info", f"USB device disconnected: {dev}")
                    self._known_usb = usb_set
            except Exception:
                pass

    def _check_focus_and_usage(self) -> None:
        if self.storage:
            for timer in self.storage.pop_due_focus_timers():
                self.add_event("info", "info", f"Focus timer finished: {timer['label']}", notify=True)

        interval = max(5, self.settings.monitor_interval_sec)
        dev_procs = {"code", "nvim", "vim", "cargo", "rustc", "gcc", "g++", "python3", "node", "antigravity"}
        media_procs = {"zoom", "teams", "obs", "vlc", "mpv", "discord"}
        detected_names: set[str] = set()
        top_cpu_val = 0.0
        top_cpu_name = ""
        zombies = 0
        try:
            user = psutil.Process().username()
            seen: set[str] = set()
            active_pids: set[int] = set()
            for proc in psutil.process_iter(["pid", "name", "username", "status", "cpu_percent", "memory_info"]):
                info = proc.info
                if info.get("status") == psutil.STATUS_ZOMBIE:
                    zombies += 1
                cpu_p = float(info.get("cpu_percent") or 0.0)
                pname = str(info.get("name") or "")
                if cpu_p > top_cpu_val and pname:
                    top_cpu_val = cpu_p
                    top_cpu_name = pname
                if info.get("username") == user:
                    pid = info.get("pid")
                    name = pname.lower()
                    if name:
                        detected_names.add(name)
                    if cpu_p > 0.5 and name and name not in seen and self.storage:
                        seen.add(name)
                        self.storage.add_app_usage(name, interval)
                    mem_info = info.get("memory_info")
                    if pid and mem_info:
                        active_pids.add(pid)
                        rss_mb = mem_info.rss / (1024 * 1024)
                        hist = self._rss_history.setdefault(pid, deque(maxlen=12))
                        hist.append(rss_mb)
                        if (
                            len(hist) >= 6
                            and hist[-1] - hist[0] >= 150.0
                            and all(hist[i] <= hist[i + 1] + 2.0 for i in range(len(hist) - 1))
                            and self._can_alert(f"leak:{pid}", cooldown_sec=1800)
                        ):
                            if pid == os.getpid():
                                try:
                                    from .local_mind import unload_idle_servers

                                    unload_idle_servers(idle_sec=30.0)
                                    import gc

                                    gc.collect()
                                except Exception:
                                    pass
                                if rss_mb < 1024.0:
                                    continue
                            self.add_event(
                                "health",
                                "warning",
                                f"Possible memory leak in {name} (PID {pid}): {hist[0]:.0f} MB -> {hist[-1]:.0f} MB",
                                notify=True,
                            )
            self._last_top_cpu_proc = f"top: {top_cpu_name}" if top_cpu_val > 1.0 and top_cpu_name else ""
            self._last_zombie_count = zombies
            if zombies >= 5 and self._can_alert("zombies", cooldown_sec=1800):
                self.add_event(
                    "health",
                    "warning",
                    f"High number of zombie processes detected: {zombies}",
                    notify=True,
                )
            for stale_pid in list(self._rss_history):
                if stale_pid not in active_pids:
                    self._rss_history.pop(stale_pid, None)
            new_ctx = (
                "media_meeting"
                if (detected_names & media_procs)
                else ("developer" if (detected_names & dev_procs) else "normal")
            )
            if new_ctx != self.current_context:
                self.current_context = new_ctx
                if self.storage:
                    self.storage.record_activity("context", new_ctx)
        except Exception:
            pass

    async def _check_automations(self, added_files: list[Path]) -> None:
        if not self.storage or not self.automation_callback:
            return
        automations = self.storage.list_automations(enabled_only=True)
        if not automations:
            return
        now = datetime.now()

        world_forecast: dict[str, Any] = {}
        try:
            from .neural import NeuralEngine

            world_forecast = NeuralEngine(self.storage).predict_world_state()
        except Exception:
            world_forecast = {}

        quiet_now = False
        try:
            from .config import get_settings

            qh = (get_settings().quiet_hours or "").strip()
            if "-" in qh:
                s_str, e_str = qh.split("-", 1)
                sh, sm = (int(x) for x in s_str.strip().split(":", 1))
                eh, em = (int(x) for x in e_str.strip().split(":", 1))
                cur_m = now.hour * 60 + now.minute
                s_m, e_m = sh * 60 + sm, eh * 60 + em
                quiet_now = (cur_m >= s_m or cur_m < e_m) if s_m > e_m else (s_m <= cur_m < e_m)
        except Exception:
            quiet_now = False

        for auto in automations:
            ttype = auto["trigger_type"]
            tval = auto["trigger_value"]
            last_run = (
                datetime.strptime(auto["last_run_at"], "%Y-%m-%d %H:%M:%S")
                if auto.get("last_run_at")
                else None
            )
            trigger_context: str | None = None

            if ttype == "daily_time":
                try:
                    hh, mm = (int(x) for x in tval.split(":", 1))
                    due = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                except ValueError:
                    continue
                if now >= due and (not last_run or last_run < due):
                    trigger_context = f"Scheduled daily time {tval} reached."
            elif ttype == "interval":
                try:
                    sec = max(30, int(tval))
                except ValueError:
                    continue
                if not last_run or (now - last_run).total_seconds() >= sec:
                    trigger_context = f"Interval ({sec}s) elapsed."
            elif ttype == "file_created" and added_files:
                matched = [str(p) for p in added_files if fnmatch.fnmatch(p.name, tval)]
                if matched:
                    trigger_context = f"New matching file(s): {', '.join(matched)}"
            elif ttype == "neural_anomaly":
                if (
                    world_forecast.get("anomaly_detected")
                    and (not last_run or (now - last_run).total_seconds() >= 1800)
                ):
                    score = world_forecast.get("anomaly_score", 0.0)
                    trigger_context = f"Neural anomaly detected (score={score})."
            elif ttype == "neural_bottleneck":
                try:
                    thresh = float(tval) if tval else 0.70
                except ValueError:
                    thresh = 0.70
                prob = float(world_forecast.get("bottleneck_prob", 0.0) or 0.0)
                if prob >= thresh and (not last_run or (now - last_run).total_seconds() >= 1800):
                    trigger_context = f"Neural bottleneck predicted (prob={prob:.2f} >= {thresh:.2f})."
            elif ttype == "multi_condition" and (not last_run or (now - last_run).total_seconds() >= 3600):
                try:
                    import json as _json

                    cond = _json.loads(tval)
                    snap = await asyncio.to_thread(self.get_health_snapshot)
                    bat = psutil.sensors_battery()
                    ok = True
                    if "max_cpu" in cond and snap["cpu_percent"] > float(cond["max_cpu"]):
                        ok = False
                    if "max_ram" in cond and snap["ram_percent"] > float(cond["max_ram"]):
                        ok = False
                    if "after_hour" in cond and now.hour < int(cond["after_hour"]):
                        ok = False
                    if "min_battery" in cond and bat and bat.percent < float(cond["min_battery"]):
                        ok = False
                    if "context" in cond and self.current_context != str(cond["context"]):
                        ok = False
                    if ok:
                        trigger_context = f"Multi-condition matched: {tval}"
                except Exception:
                    pass

            if not trigger_context:
                continue

            if self.storage.is_ignored_by_brain_rule(f"{auto['name']} {auto['prompt']}"):
                self.storage.add_brain_thought(
                    "ignore", 90, f"Skipped automation '{auto['name']}' due to brain ignore rule"
                )
                continue

            if quiet_now and ttype == "interval":
                continue

            b_prob = float(world_forecast.get("bottleneck_prob", 0.0) or 0.0)
            if ttype in {"daily_time", "interval"} and b_prob >= 0.75:
                self.add_event(
                    "info",
                    "info",
                    f"Deferred automation '{auto['name']}' due to neural bottleneck (prob={b_prob:.2f})",
                )
                continue

            self.storage.mark_automation_ran(auto["id"])
            self.add_event("info", "info", f"Running automation '{auto['name']}': {trigger_context}")
            try:
                await self.automation_callback(auto, trigger_context)
                self.storage.add_brain_thought(
                    "act", 85, f"Automation '{auto['name']}' completed: {trigger_context}"
                )
                try:
                    from .neural import NeuralEngine

                    NeuralEngine(self.storage).online_step(str(auto["prompt"]), "run_command", reward=0.8)
                except Exception:
                    pass
            except Exception as exc:
                self.add_event("info", "warning", f"Automation '{auto['name']}' failed: {exc}")
                self.storage.add_brain_thought(
                    "warn", 70, f"Automation '{auto['name']}' failed: {exc}"
                )
                try:
                    from .neural import NeuralEngine

                    NeuralEngine(self.storage).online_step(str(auto["prompt"]), "run_command", reward=-0.8)
                except Exception:
                    pass

    @staticmethod
    def _top_cpu_process() -> str:
        try:
            procs = sorted(
                psutil.process_iter(["name", "cpu_percent"]),
                key=lambda p: p.info.get("cpu_percent") or 0.0,
                reverse=True,
            )
            if procs and (procs[0].info.get("cpu_percent") or 0) > 1.0:
                return f"top: {procs[0].info['name']}"
        except Exception:
            pass
        return ""

    def _send_desktop_notification(
        self,
        title: str,
        body: str,
        urgency: str = "normal",
        category: str = "info",
        actions: list[tuple[str, str]] | None = None,
        on_action: Callable[[str], Any] | None = None,
        on_timeout: Callable[[], Any] | None = None,
        bypass_gates: bool = False,
    ) -> Any:
        notifier = getattr(self, "notifier", None)
        if notifier is None:
            from .notification import NotificationCenter

            notifier = NotificationCenter(self.settings, self.storage)
            self.notifier = notifier
        return notifier.send(
            title,
            body,
            urgency=urgency,
            category=category,
            actions=actions,
            on_action=on_action,
            on_timeout=on_timeout,
            bypass_gates=bypass_gates,
        )

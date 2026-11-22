"""Centralized NotificationCenter with Brain/Neural gating, deduplication, and cancelable actions."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .storage import Storage


@dataclass
class NotificationHandle:
    """Handle for an active or completed desktop notification."""

    notif_id: str
    title: str
    category: str
    urgency: str
    _proc: subprocess.Popen[str] | None = field(default=None, repr=False)
    _cancelled: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def cancel(self) -> None:
        """Dismiss/terminate the waiting notify-send process if still active."""
        with self._lock:
            self._cancelled = True
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass


class NotificationCenter:
    """Unified desktop notification hub with Brain quiet/ignore rules, Neural suppression, and dedup."""

    def __init__(self, settings: Settings, storage: Storage | None = None) -> None:
        self.settings = settings
        self.storage = storage
        self.flow_provider: Callable[[], dict[str, Any]] | None = None
        self._lock = threading.Lock()
        self._recent_keys: dict[str, float] = {}
        self._history: deque[dict[str, Any]] = deque(maxlen=60)
        self._active: dict[str, NotificationHandle] = {}

    def _is_quiet_hours(self) -> bool:
        qh = (self.settings.quiet_hours or "").strip()
        if "-" not in qh:
            return False
        try:
            s_str, e_str = qh.split("-", 1)
            sh, sm = (int(x) for x in s_str.strip().split(":", 1))
            eh, em = (int(x) for x in e_str.strip().split(":", 1))
            now = datetime.now()
            cur_m = now.hour * 60 + now.minute
            s_m, e_m = sh * 60 + sm, eh * 60 + em
            return (cur_m >= s_m or cur_m < e_m) if s_m > e_m else (s_m <= cur_m < e_m)
        except Exception:
            return False

    def _should_suppress(self, title: str, body: str, urgency: str, category: str) -> str | None:
        """Return suppression reason if Brain or Neural gates decide to silence this notification."""
        if urgency == "critical" or category == "approval":
            return None

        if (
            category in {"health", "filesystem", "security", "term_error", "info"}
            and self.storage is not None
            and self.storage.is_ignored_by_brain_rule(f"{title} {body}")
        ):
            return "brain_ignore_rule"

        if category in {"info", "health"} and self._is_quiet_hours():
            return "brain_quiet_hours"

        if category == "info" and self.flow_provider is not None:
            try:
                flow = self.flow_provider() or {}
                if flow.get("suppress_low_priority"):
                    return "cortex_flow_focus"
            except Exception:
                pass

        if self.storage is not None:
            try:
                from .neural import NeuralEngine

                for hit in NeuralEngine(self.storage).spreading_activation(f"{title} {body}", top_k=2):
                    if (
                        str(hit.get("key") or "") == "brain_dismiss"
                        and float(hit.get("activation", 0.0) or 0.0) >= 0.68
                    ):
                        return "neural_dismiss_learned"
            except Exception:
                pass
        return None

    @staticmethod
    def _clean_text(text: str, max_len: int) -> str:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        joined = "\n".join(lines)
        return joined[:max_len] if len(joined) > max_len else joined

    def _icon_for(self, urgency: str, category: str) -> str:
        if urgency == "critical":
            return "dialog-error"
        if category == "approval":
            return "dialog-question"
        if category == "security":
            return "security-high"
        icon_path = Path(__file__).resolve().parent.parent / "assets" / "sysagent.svg"
        return str(icon_path) if icon_path.is_file() else "utilities-system-monitor"

    def _record_outcome(self, entry: dict[str, Any], status: str, action: str | None = None) -> None:
        with self._lock:
            entry["status"] = status
            if action is not None:
                entry["action"] = action
        if self.storage is not None and action:
            try:
                self.storage.add_brain_thought(
                    "act", 80, f"Notification '{entry['title']}' action={action}"
                )
            except Exception:
                pass

    def cancel(self, notif_id: str) -> bool:
        with self._lock:
            handle = self._active.pop(notif_id, None)
        if handle is None:
            return False
        handle.cancel()
        return True

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)[-limit:][::-1]

    def send(
        self,
        title: str,
        body: str,
        urgency: str = "normal",
        category: str = "info",
        actions: list[tuple[str, str]] | None = None,
        on_action: Callable[[str], Any] | None = None,
        on_timeout: Callable[[], Any] | None = None,
        dedup_key: str | None = None,
        dedup_sec: float = 25.0,
        bypass_gates: bool = False,
    ) -> NotificationHandle | None:
        clean_title = self._clean_text(title, 120) or self.settings.agent_name
        clean_body = self._clean_text(body, 280)
        notif_id = uuid.uuid4().hex[:10]
        entry: dict[str, Any] = {
            "id": notif_id,
            "title": clean_title,
            "body": clean_body,
            "category": category,
            "urgency": urgency,
            "status": "queued",
            "action": None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        if not bypass_gates:
            reason = self._should_suppress(clean_title, clean_body, urgency, category)
            if reason:
                entry["status"] = f"suppressed:{reason}"
                with self._lock:
                    self._history.append(entry)
                return None

        if category != "approval" and dedup_sec > 0:
            dkey = dedup_key or f"{category}:{clean_title}:{clean_body[:60]}"
            now_mono = time.monotonic()
            with self._lock:
                last_ts = self._recent_keys.get(dkey)
                if last_ts is not None and (now_mono - last_ts) < dedup_sec:
                    entry["status"] = "deduplicated"
                    self._history.append(entry)
                    return None
                self._recent_keys[dkey] = now_mono
                if len(self._recent_keys) > 200:
                    cutoff = now_mono - 300.0
                    self._recent_keys = {k: v for k, v in self._recent_keys.items() if v >= cutoff}

        if not shutil.which("notify-send"):
            entry["status"] = "unavailable"
            with self._lock:
                self._history.append(entry)
            return None

        icon_arg = self._icon_for(urgency, category)
        cmd = [
            "notify-send",
            "-a",
            self.settings.agent_name,
            "-i",
            icon_arg,
            "-u",
            urgency,
            "-h",
            "string:desktop-entry:sysagent-quick",
        ]
        if actions:
            for act_key, act_label in actions:
                cmd.extend(["-A", f"{act_key}={act_label}"])
        cmd.extend([clean_title, clean_body])

        handle = NotificationHandle(
            notif_id=notif_id, title=clean_title, category=category, urgency=urgency
        )
        with self._lock:
            self._history.append(entry)
            self._active[notif_id] = handle

        def _worker() -> None:
            try:
                with handle._lock:
                    if handle._cancelled:
                        self._record_outcome(entry, "cancelled")
                        return
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                with handle._lock:
                    handle._proc = proc
                    cancelled_early = handle._cancelled
                if cancelled_early:
                    try:
                        proc.terminate()
                        proc.communicate(timeout=2)
                    except Exception:
                        try:
                            proc.kill()
                            proc.wait(timeout=2)
                        except Exception:
                            pass
                    self._record_outcome(entry, "cancelled")
                    return
                timeout_s = 300 if actions else 5
                try:
                    stdout, _ = proc.communicate(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.communicate(timeout=2)
                    except Exception:
                        try:
                            proc.kill()
                            proc.wait(timeout=2)
                        except Exception:
                            pass
                    if not handle._cancelled and actions and on_timeout is not None:
                        self._record_outcome(entry, "timeout")
                        on_timeout()
                    return
                if handle._cancelled:
                    self._record_outcome(entry, "cancelled")
                    return
                selected = (stdout or "").strip()
                if selected and on_action is not None:
                    self._record_outcome(entry, "action", action=selected)
                    on_action(selected)
                elif actions and not selected:
                    self._record_outcome(entry, "dismissed")
                    if on_timeout is not None:
                        on_timeout()
                else:
                    self._record_outcome(entry, "sent")
            except Exception:
                self._record_outcome(entry, "error")
            finally:
                with self._lock:
                    self._active.pop(notif_id, None)

        threading.Thread(target=_worker, daemon=True).start()
        return handle

    def get_status(self) -> dict[str, Any]:
        """Return notification delivery, suppression, and dedup counters plus recent history."""
        with self._lock:
            items = list(self._history)
        return {
            "total": len(items),
            "sent": sum(1 for x in items if x.get("outcome") in {"sent", "action"}),
            "suppressed": sum(1 for x in items if str(x.get("outcome", "")).startswith("suppressed")),
            "deduped": sum(1 for x in items if x.get("outcome") == "suppressed:dedup"),
            "recent": items[-5:],
        }

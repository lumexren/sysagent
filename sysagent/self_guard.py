"""Self-Autopsy, Self-Calibration, Self-Canary, Self-Profiler, and Self-Skill-Synthesizer."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from .config import DATA_DIR, Settings
from .storage import Storage

_MARKER_FILE = DATA_DIR / "daemon.running"


class SelfGuard:
    """Autonomous self-reliability layer for crash autopsy, calibration, canary probes, profiling, and skill drafts."""

    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        watcher: Any = None,
        agent: Any = None,
        evolver: Any = None,
        marker_file: Path | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.watcher = watcher
        self.agent = agent
        self.evolver = evolver
        self.marker_file = marker_file or _MARKER_FILE
        self.last_autopsy: dict[str, Any] | None = None
        self.last_canary: dict[str, Any] | None = None
        self.last_calibration: dict[str, Any] | None = None
        self.last_profile: dict[str, Any] | None = None

    def on_startup(self) -> dict[str, Any]:
        """Run crash autopsy if the previous daemon run exited uncleanly, then write the boot marker."""
        crashed = self.marker_file.is_file()
        autopsy = self.run_crash_autopsy() if crashed else {"crashed": False}
        self.marker_file.parent.mkdir(parents=True, exist_ok=True)
        self.marker_file.write_text(
            json.dumps({"pid": os.getpid(), "started_at": datetime.now().isoformat(timespec="seconds")}),
            encoding="utf-8",
        )
        return autopsy

    def mark_clean_shutdown(self) -> None:
        """Remove the running marker on graceful shutdown."""
        self.marker_file.unlink(missing_ok=True)

    def run_crash_autopsy(self, journal_text: str | None = None) -> dict[str, Any]:
        """Inspect recent daemon logs after an unclean shutdown and auto-rollback faulty evolve patches."""
        raw_log = journal_text
        if raw_log is None:
            try:
                proc = subprocess.run(
                    ["journalctl", "--user", "-u", "sysagent", "-n", "40", "--no-pager"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                raw_log = proc.stdout if proc.returncode == 0 else ""
            except Exception:
                raw_log = ""

        sig = "unclean_shutdown"
        for marker in ("SIGSEGV", "status=11/SEGV", "SIGABRT", "status=6/ABRT", "SyntaxError", "ImportError", "NameError", "UnboundLocalError"):
            if marker in (raw_log or ""):
                sig = marker
                break

        rolled_back = False
        if self.evolver is not None and sig in {"SyntaxError", "ImportError", "NameError", "UnboundLocalError"}:
            try:
                props = self.storage.list_evolve_proposals(limit=3)
                latest_applied = next((p for p in props if p.get("status") == "applied"), None)
                if latest_applied:
                    tfile = str(latest_applied.get("target_file") or "")
                    if tfile and tfile in (raw_log or ""):
                        rolled_back = bool(self.evolver.rollback_last_evolve(int(latest_applied["id"])))
            except Exception:
                rolled_back = False

        summary = f"Crash autopsy detected {sig} (auto_rollback={rolled_back})."
        self.storage.add_brain_thought("warn", 95, summary)
        try:
            self.storage.add_lesson("self_autopsy", summary, "runtime")
        except Exception:
            pass

        result = {
            "crashed": True,
            "signature": sig,
            "rolled_back": rolled_back,
            "summary": summary,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.last_autopsy = result
        if self.watcher is not None and hasattr(self.watcher, "_send_desktop_notification"):
            self.watcher._send_desktop_notification(
                f"{self.settings.agent_name} — Crash Autopsy",
                summary,
                urgency="normal",
                category="health",
            )
        return result

    def calibrate_thresholds(self) -> dict[str, Any]:
        """Dynamically tune local routing and bottleneck deferral thresholds from live metrics."""
        tool_acc = 0.85
        forecast_mae = 5.0
        try:
            from .neural import NeuralEngine

            ns = NeuralEngine(self.storage).get_status()
            tool_acc = float(ns.get("tool_accuracy") or 0.85)
            wf = ns.get("world_forecast") or {}
            forecast_mae = float(wf.get("forecast_mae") or 5.0)
        except Exception:
            pass

        metrics = self.storage.get_tool_metrics()
        total_calls = sum(int(m.get("calls", 0) or 0) for m in metrics)
        total_errs = sum(int(m.get("errors", 0) or 0) for m in metrics)
        err_rate = (total_errs / total_calls) if total_calls > 0 else 0.0

        if tool_acc >= 0.90 and err_rate < 0.08:
            local_tool_thresh = 0.64
            local_max_words = 26
        elif tool_acc < 0.75 or err_rate > 0.20:
            local_tool_thresh = 0.78
            local_max_words = 16
        else:
            local_tool_thresh = 0.72
            local_max_words = 22

        bottleneck_defer = 0.72 if forecast_mae <= 6.0 else 0.82
        cal = {
            "local_tool_threshold": local_tool_thresh,
            "local_max_words": local_max_words,
            "bottleneck_defer_threshold": bottleneck_defer,
            "tool_accuracy": round(tool_acc, 4),
            "tool_error_rate": round(err_rate, 4),
            "forecast_mae": round(forecast_mae, 2),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.storage.set_memory("_internal:self_calibration", json.dumps(cal, ensure_ascii=False))
        self.last_calibration = cal
        return cal

    def run_canary_probes(self) -> dict[str, Any]:
        """Run fast internal health probes across SQLite, Unix socket, NeuralEngine, and GGUF models."""
        probes: dict[str, dict[str, Any]] = {}

        t0 = time.monotonic()
        try:
            self.storage.set_memory("_internal:canary_ping", "ok")
            val = self.storage.get_memory("_internal:canary_ping")
            self.storage.delete_memory("_internal:canary_ping")
            probes["sqlite"] = {"ok": val == "ok", "ms": round((time.monotonic() - t0) * 1000.0, 2)}
        except Exception as exc:
            probes["sqlite"] = {"ok": False, "error": str(exc)}

        sock = Path(self.settings.socket_path)
        probes["unix_socket"] = {"ok": sock.exists(), "path": str(sock)}

        try:
            from .neural import NeuralEngine

            wf = NeuralEngine(self.storage).predict_world_state()
            probes["neural"] = {"ok": isinstance(wf, dict), "samples": int(wf.get("samples", 0) or 0)}
        except Exception as exc:
            probes["neural"] = {"ok": False, "error": str(exc)}

        try:
            from .local_mind import find_all_gguf_models

            cnt = len(find_all_gguf_models())
            probes["local_models"] = {"ok": cnt >= 0, "count": cnt}
        except Exception as exc:
            probes["local_models"] = {"ok": False, "error": str(exc)}

        all_ok = all(bool(p.get("ok")) for k, p in probes.items() if k != "unix_socket")
        report = {
            "ok": all_ok,
            "probes": probes,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.last_canary = report
        if not all_ok:
            failed = [k for k, v in probes.items() if not v.get("ok")]
            self.storage.add_brain_thought("warn", 90, f"Self-Canary failed probes: {', '.join(failed)}")
        return report

    def run_memory_profiler(self, max_rss_mb: float = 900.0) -> dict[str, Any]:
        """Monitor daemon RSS memory and trim caches/warm models when above threshold."""
        proc = psutil.Process(os.getpid())
        rss_before = round(proc.memory_info().rss / (1024**2), 1)
        trimmed = False
        if rss_before >= max_rss_mb:
            from .local_inference import _release_all_warm_models

            _release_all_warm_models()
            if self.agent is not None and hasattr(self.agent, "_tool_cache"):
                self.agent._tool_cache.invalidate_all()
            self.storage.prune_and_checkpoint(stale_days=21)
            gc.collect()
            try:
                import ctypes

                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            trimmed = True

        rss_after = round(proc.memory_info().rss / (1024**2), 1)
        report = {
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "max_rss_mb": max_rss_mb,
            "trimmed": trimmed,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.last_profile = report
        if trimmed:
            self.storage.add_brain_thought(
                "act", 85, f"Self-Profiler trimmed daemon RSS: {rss_before}MB -> {rss_after}MB"
            )
        return report

    def synthesize_skill_candidates(self, min_occurrences: int = 3) -> list[dict[str, Any]]:
        """Detect repeated tool action pairs from audit history and draft safe plugin skeletons."""
        rows = list(reversed(self.storage.list_audit_log(limit=120)))
        pairs: Counter[tuple[str, str]] = Counter()
        for prev, curr in zip(rows, rows[1:], strict=False):
            a1 = str(prev.get("action") or "")
            a2 = str(curr.get("action") or "")
            if a1 and a2 and a1 != "evolve" and a2 != "evolve" and a1 != a2:
                pairs[(a1, a2)] += 1

        candidates: list[dict[str, Any]] = []
        for (a1, a2), count in pairs.most_common(5):
            if count < min_occurrences:
                continue
            slug = f"auto_{a1}_then_{a2}"
            draft = (
                f'"""Auto-synthesized workflow skill: {a1} -> {a2}."""\n'
                f"def _run(args):\n"
                f"    return {{'workflow': '{a1} -> {a2}', 'args': args}}\n\n"
                f"TOOLS = [{{'declaration': {{'name': '{slug}', "
                f"'description': 'Run repeated workflow {a1} followed by {a2}.', "
                f"'parameters': {{'type': 'object', 'properties': {{}}}}}}, "
                f"'handler': _run, 'safe': False}}]\n"
            )
            candidates.append(
                {
                    "name": slug,
                    "chain": [a1, a2],
                    "occurrences": count,
                    "draft_code": draft,
                }
            )
        return candidates

    def get_status(self) -> dict[str, Any]:
        return {
            "autopsy": self.last_autopsy,
            "calibration": self.last_calibration or self.calibrate_thresholds(),
            "canary": self.last_canary or self.run_canary_probes(),
            "profile": self.last_profile or self.run_memory_profiler(),
            "skill_candidates": self.synthesize_skill_candidates(),
        }

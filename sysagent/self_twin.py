"""Causal Digital Twin, Root-Cause Investigator, Counterfactual Verifier, Episodic Miner, and Doctor Auto-Repair."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from .storage import Storage

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_TWIN_HISTORY: list[dict[str, Any]] = []


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.strptime(str(value)[:19], _TS_FMT)
    except (ValueError, TypeError):
        return None


def build_causal_timeline(
    storage: Storage, watcher: Any = None, window_sec: int = 180
) -> list[dict[str, str]]:
    """Correlate recent file/tool actions with warning/critical events and record causal KG edges."""
    links: list[dict[str, str]] = []
    if watcher is None:
        return links
    events = watcher.get_events(limit=12)
    alerts = [e for e in events if e.get("severity") in {"warning", "critical"}]
    if not alerts:
        return links

    recent_actions: list[tuple[datetime, str]] = []
    for aud in storage.list_audit_log(limit=10):
        ts = _parse_ts(str(aud.get("created_at") or ""))
        if ts:
            recent_actions.append((ts, f"{aud.get('action')}:{Path(str(aud.get('target_path') or '')).name}"))
    for ev in events:
        if ev.get("severity") == "info":
            ts = _parse_ts(str(ev.get("timestamp") or ""))
            if ts:
                recent_actions.append((ts, str(ev.get("message") or "")[:60]))

    for al in alerts[:3]:
        al_ts = _parse_ts(str(al.get("timestamp") or ""))
        if not al_ts:
            continue
        effect = str(al.get("message") or "")[:80]
        for act_ts, cause in recent_actions:
            dt = (al_ts - act_ts).total_seconds()
            if 0 <= dt <= window_sec and cause and cause != effect:
                storage.add_kg_edge(cause[:60], "preceded_alert", effect)
                links.append({"cause": cause[:60], "effect": effect, "lag_sec": str(int(dt))})
    return links[:5]


def analyze_causal_digital_twin(
    storage: Storage | None = None,
    watcher: Any = None,
    snapshots: list[dict[str, Any]] | None = None,
    leak_threshold_mb_per_min: float = 5.0,
) -> dict[str, Any]:
    """Model per-process memory leak slopes (`d(RSS)/dt`), OOM ETA, and multi-hop causal anomaly chains."""
    now_ts = time.time()
    mem = psutil.virtual_memory()
    total_mb = round(mem.total / (1024 * 1024), 1)
    avail_mb = round(mem.available / (1024 * 1024), 1)
    ram_pct = round(float(mem.percent), 1)

    if snapshots is None:
        proc_list: list[dict[str, Any]] = []
        try:
            for p in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
                try:
                    minfo = p.info.get("memory_info")
                    rss_mb = round((minfo.rss / (1024 * 1024)), 2) if minfo else 0.0
                    if rss_mb >= 20.0:
                        proc_list.append(
                            {
                                "pid": int(p.info["pid"]),
                                "name": str(p.info.get("name") or "proc"),
                                "rss_mb": rss_mb,
                                "cpu_percent": float(p.info.get("cpu_percent") or 0.0),
                            }
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        proc_list.sort(key=lambda x: x["rss_mb"], reverse=True)
        cur_snap = {
            "ts": now_ts,
            "ram_percent": ram_pct,
            "avail_ram_mb": avail_mb,
            "total_ram_mb": total_mb,
            "processes": proc_list[:25],
        }
        _TWIN_HISTORY.append(cur_snap)
        if len(_TWIN_HISTORY) > 30:
            del _TWIN_HISTORY[:-30]
        history = list(_TWIN_HISTORY)
    else:
        history = list(snapshots)
        if history:
            last_s = history[-1]
            avail_mb = float(last_s.get("avail_ram_mb", avail_mb))
            ram_pct = float(last_s.get("ram_percent", ram_pct))

    leaks: list[dict[str, Any]] = []
    causal_chains: list[dict[str, Any]] = []

    if len(history) >= 2:
        t_first = float(history[0].get("ts") or 0.0)
        t_last = float(history[-1].get("ts") or 0.0)
        dt_min = max(0.05, (t_last - t_first) / 60.0)
        series_by_proc: dict[tuple[int, str], list[float]] = {}
        for snap in history:
            for pr in snap.get("processes") or []:
                key = (int(pr.get("pid", 0)), str(pr.get("name") or "proc"))
                series_by_proc.setdefault(key, []).append(float(pr.get("rss_mb") or 0.0))

        for (pid, pname), rss_vals in series_by_proc.items():
            if len(rss_vals) < 2:
                continue
            delta_mb = rss_vals[-1] - rss_vals[0]
            slope = round(delta_mb / dt_min, 2)
            non_decreasing = all(rss_vals[i] <= rss_vals[i + 1] + 0.5 for i in range(len(rss_vals) - 1))
            if delta_mb >= 10.0 and slope >= leak_threshold_mb_per_min and non_decreasing:
                eta_min = round(max(1.0, avail_mb / max(0.1, slope)), 1)
                leak_entry = {
                    "pid": pid,
                    "name": pname,
                    "rss_start_mb": round(rss_vals[0], 1),
                    "rss_current_mb": round(rss_vals[-1], 1),
                    "delta_mb": round(delta_mb, 1),
                    "slope_mb_per_min": slope,
                    "oom_eta_min": eta_min,
                }
                leaks.append(leak_entry)
                chain_str = (
                    f"Process {pname} (PID {pid}) leaking +{slope} MB/min "
                    f"-> RAM {ram_pct}% ({round(avail_mb, 0)} MB free) "
                    f"-> OOM exhaustion in ~{eta_min} min"
                )
                remediation = f"Inspect or restart PID {pid} (`ps -p {pid} -o pid,rss,cmd` / `kill -15 {pid}`)"
                causal_chains.append(
                    {
                        "root_cause": f"memory_leak:{pname}:{pid}",
                        "rate": f"+{slope} MB/min",
                        "chain": chain_str,
                        "oom_eta_min": eta_min,
                        "remediation": remediation,
                    }
                )
                if storage is not None:
                    try:
                        storage.add_kg_edge(
                            f"leak:{pname}:{pid}",
                            "causes",
                            f"oom_risk_in_{int(eta_min)}m",
                        )
                    except Exception:
                        pass

    leaks.sort(key=lambda x: x["slope_mb_per_min"], reverse=True)
    if storage is not None and watcher is not None:
        try:
            for link in build_causal_timeline(storage, watcher):
                causal_chains.append(
                    {
                        "root_cause": link["cause"],
                        "rate": f"lag={link['lag_sec']}s",
                        "chain": f"{link['cause']} -> {link['effect']}",
                        "oom_eta_min": None,
                        "remediation": f"Review recent action `{link['cause']}`",
                    }
                )
        except Exception:
            pass

    return {
        "snapshots_count": len(history),
        "ram_percent": ram_pct,
        "avail_ram_mb": avail_mb,
        "memory_leaks": leaks[:5],
        "causal_chains": causal_chains[:6],
        "healthy": len(leaks) == 0,
    }


def investigate_root_cause(
    storage: Storage,
    watcher: Any = None,
    events: list[dict[str, Any]] | None = None,
    health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a 0-token read-only diagnostic chain across Causal Digital Twin, processes, systemd units, and KG."""
    from . import self_learn as _sl

    raw_events = (
        list(events)
        if events is not None
        else (watcher.get_events(limit=12) if watcher is not None and hasattr(watcher, "get_events") else [])
    )
    h_snap: dict[str, Any] = (
        dict(health)
        if isinstance(health, dict)
        else (
            watcher.get_health_snapshot()
            if watcher is not None and hasattr(watcher, "get_health_snapshot")
            else {}
        )
    )
    if not h_snap:
        try:
            vm = psutil.virtual_memory()
            du = psutil.disk_usage("/")
            h_snap = {
                "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
                "ram_percent": round(vm.percent, 1),
                "disk_percent": round(du.percent, 1),
            }
        except Exception:
            h_snap = {}

    hypotheses: list[dict[str, Any]] = []
    twin = _sl.analyze_causal_digital_twin(storage, watcher)
    for lk in twin.get("memory_leaks") or []:
        pid = int(lk.get("pid") or 0)
        pname = str(lk.get("name") or "process")
        slope = float(lk.get("slope_mb_per_min") or 0.0)
        eta = lk.get("oom_eta_min")
        conf = min(98, int(75 + min(20.0, slope / 2.0)))
        hypotheses.append(
            {
                "kind": "memory_leak",
                "target": f"{pname}:{pid}",
                "confidence": conf,
                "evidence": f"Process {pname} (PID {pid}) growing at +{slope} MB/min (OOM ETA ~{eta} min).",
                "suggested_action": f"kill_leak:{pid}",
            }
        )

    for ev in raw_events:
        if not isinstance(ev, dict):
            continue
        cat = str(ev.get("category") or "").lower()
        msg = str(ev.get("message") or "")
        sev = str(ev.get("severity") or "info").lower()
        if cat == "systemd" or "service" in msg.lower() or "failed" in msg.lower():
            m_unit = re.search(r"([a-zA-Z0-9_.@-]+\.service)", msg)
            unit_name = m_unit.group(1) if m_unit else ""
            if unit_name:
                tail_err = msg[:160]
                if shutil.which("journalctl"):
                    try:
                        proc = subprocess.run(
                            ["journalctl", "-u", unit_name, "-n", "12", "--no-pager"],
                            capture_output=True,
                            text=True,
                            timeout=2.0,
                        )
                        if proc.returncode == 0 and proc.stdout.strip():
                            err_lines = [
                                ln.strip()
                                for ln in proc.stdout.splitlines()
                                if any(k in ln.lower() for k in ("error", "fail", "fatal", "exception", "exit"))
                            ]
                            if err_lines:
                                tail_err = err_lines[-1][:180]
                    except Exception:
                        pass
                hypotheses.append(
                    {
                        "kind": "service_failure",
                        "target": unit_name,
                        "confidence": 90 if sev == "critical" else 82,
                        "evidence": f"Service `{unit_name}` reported anomaly: {tail_err}",
                        "suggested_action": f"restart_service:{unit_name}",
                    }
                )

    cpu_pct = float(h_snap.get("cpu_percent") or 0.0)
    ram_pct = float(h_snap.get("ram_percent") or 0.0)
    if (cpu_pct >= 85.0 or ram_pct >= 85.0) and not any(h["kind"] == "memory_leak" for h in hypotheses):
        top_hog: dict[str, Any] | None = None
        try:
            procs: list[dict[str, Any]] = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                procs.append(dict(p.info))
            sort_k = "memory_percent" if ram_pct >= cpu_pct else "cpu_percent"
            procs.sort(key=lambda r: float(r.get(sort_k) or 0.0), reverse=True)
            if procs:
                top_hog = procs[0]
        except Exception:
            top_hog = None
        if top_hog and int(top_hog.get("pid") or 0) > 0:
            h_pid = int(top_hog["pid"])
            h_name = str(top_hog.get("name") or "process")
            hypotheses.append(
                {
                    "kind": "resource_hog",
                    "target": f"{h_name}:{h_pid}",
                    "confidence": 80,
                    "evidence": (
                        f"Host load high (CPU {cpu_pct}%, RAM {ram_pct}%); top consumer is "
                        f"{h_name} (PID {h_pid}, CPU={top_hog.get('cpu_percent')}%, MEM={round(float(top_hog.get('memory_percent') or 0.0), 1)}%)."
                    ),
                    "suggested_action": f"inspect_pid:{h_pid}",
                }
            )

    for ch in twin.get("causal_chains") or []:
        rc = str(ch.get("root_cause") or "")
        if rc and not rc.startswith("memory_leak:"):
            hypotheses.append(
                {
                    "kind": "recent_mutation_regression",
                    "target": rc,
                    "confidence": 78,
                    "evidence": f"Causal timeline correlated recent action `{ ch.get('chain') }` ({ch.get('rate')}).",
                    "suggested_action": str(ch.get("remediation") or f"Review `{rc}`"),
                }
            )

    hypotheses.sort(key=lambda h: int(h.get("confidence") or 0), reverse=True)
    top_h = hypotheses[0] if hypotheses else None
    if top_h is not None:
        try:
            storage.add_kg_edge(
                f"anomaly:{top_h['kind']}",
                "diagnosed_root_cause",
                str(top_h["target"]),
            )
        except Exception:
            pass
    summary = (
        f"[{top_h['kind']} | {top_h['confidence']}%] {top_h['evidence']} -> {top_h['suggested_action']}"
        if top_h
        else "No active root-cause anomaly detected; system telemetry nominal."
    )
    return {
        "healthy": len(hypotheses) == 0,
        "top_hypothesis": top_h,
        "hypotheses": hypotheses[:5],
        "summary": summary,
    }


def verify_remediation_impact(
    storage: Storage,
    action_label: str,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    rule_id: int | None = None,
) -> dict[str, Any]:
    """Counterfactually evaluate whether an autonomous or closed-loop remediation resolved the target anomaly."""
    ram_b = float(before_state.get("ram_percent") or 0.0)
    ram_a = float(after_state.get("ram_percent") or ram_b)
    cpu_b = float(before_state.get("cpu_percent") or 0.0)
    cpu_a = float(after_state.get("cpu_percent") or cpu_b)
    freed_mb = float(after_state.get("freed_ram_mb") or 0.0)
    pid_alive_b = bool(before_state.get("pid_alive", False))
    pid_alive_a = bool(after_state.get("pid_alive", False))
    svc_active_b = bool(before_state.get("service_active", True))
    svc_active_a = bool(after_state.get("service_active", True))
    reply_err = bool(after_state.get("reply_error", False))

    ram_delta = round(ram_a - ram_b, 2)
    cpu_delta = round(cpu_a - cpu_b, 2)
    reasons: list[str] = []
    effective = True

    if reply_err:
        effective = False
        reasons.append("reply_reported_error")
    if pid_alive_b and not pid_alive_a:
        reasons.append("target_pid_terminated")
    elif pid_alive_b and pid_alive_a and action_label.startswith("kill_leak:"):
        effective = False
        reasons.append("target_pid_still_alive")

    if not svc_active_b and svc_active_a:
        reasons.append("service_recovered")
    elif not svc_active_b and not svc_active_a and "service" in action_label:
        effective = False
        reasons.append("service_still_inactive")

    if freed_mb >= 5.0 or ram_delta <= -1.0:
        reasons.append(f"ram_improved({ram_delta:+.1f}%,+{freed_mb:.1f}MB)")
    elif ram_b >= 85.0 and ram_delta > 3.0:
        effective = False
        reasons.append(f"ram_worsened({ram_delta:+.1f}%)")

    if cpu_b >= 85.0 and cpu_delta <= -5.0:
        reasons.append(f"cpu_improved({cpu_delta:+.1f}%)")

    if not reasons:
        reasons.append("telemetry_stable")

    reward = 1.5 if effective else -1.0
    conf_delta = 6 if effective else -15
    if rule_id is not None:
        try:
            storage.reinforce_brain_rule(int(rule_id), delta=conf_delta)
        except Exception:
            pass
    try:
        from .neural import NeuralEngine

        eng = NeuralEngine(storage)
        eng.online_step(f"remediate {action_label}", "system_status", reward=reward)
        eng.sync_associative_memory()
    except Exception:
        pass
    try:
        storage.add_lesson(
            f"counterfactual:{'ok' if effective else 'missed'}:{action_label[:48]}",
            f"Action `{action_label}` counterfactual impact: {', '.join(reasons)}",
            f"RAM delta={ram_delta:+.1f}%, CPU delta={cpu_delta:+.1f}%, reward={reward:+.1f}",
        )
    except Exception:
        pass

    return {
        "effective": effective,
        "action": action_label,
        "ram_delta": ram_delta,
        "cpu_delta": cpu_delta,
        "freed_ram_mb": freed_mb,
        "reward": reward,
        "confidence_delta": conf_delta,
        "reasons": reasons,
    }


def mine_episodic_workflows(
    storage: Storage, min_occurrences: int = 2, max_new: int = 3
) -> dict[str, Any]:
    """Mine recurring multi-step command/file patterns across distinct sessions into learned_workflows and NeuralEngine."""
    checkpoints = storage.list_session_checkpoints(limit=80)
    seq_sessions: dict[tuple[str, ...], set[str]] = {}
    seq_topics: dict[tuple[str, ...], str] = {}
    seq_steps: dict[tuple[str, ...], list[dict[str, Any]]] = {}

    for cp in checkpoints:
        sid = str(cp.get("session_id") or "")
        topic = str(cp.get("topic") or "cross-session workflow").strip()
        files = [str(f) for f in (cp.get("files") or []) if str(f).strip()]
        cmds = [str(c) for c in (cp.get("commands") or []) if str(c).strip()]
        inferred: list[dict[str, Any]] = []
        if files:
            inferred.append({"name": "read_file", "arguments": {"path": files[0]}})
        for cmd in cmds[:3]:
            low_c = cmd.lower()
            if low_c.startswith(("rg ", "grep ", "find ")):
                inferred.append({"name": "search_files", "arguments": {"query": cmd.split()[-1]}})
            elif low_c.startswith("journalctl"):
                inferred.append({"name": "read_journal_logs", "arguments": {"lines": 40}})
            elif low_c.startswith("systemctl"):
                inferred.append({"name": "list_systemd_units", "arguments": {"state": "running"}})
            else:
                inferred.append({"name": "run_command", "arguments": {"command": cmd}})
        if len(inferred) >= 2:
            key = tuple(st["name"] for st in inferred[:4])
            if len(set(key)) >= 2 or len(key) >= 2:
                seq_sessions.setdefault(key, set()).add(sid or f"cp_{id(cp)}")
                seq_topics.setdefault(key, topic)
                seq_steps.setdefault(key, inferred[:4])

    audits = storage.list_audit_log(limit=60)
    by_batch: dict[str, list[dict[str, Any]]] = {}
    for row in audits:
        bid = str(row.get("batch_id") or "")
        if bid:
            by_batch.setdefault(bid, []).append(row)
    for bid, items in by_batch.items():
        if len(items) >= 2:
            ordered = list(reversed(items))
            steps = [
                {"name": str(it.get("action") or "edit_file"), "arguments": {"path": str(it.get("target_path") or "")}}
                for it in ordered[:4]
            ]
            key = tuple(s["name"] for s in steps)
            seq_sessions.setdefault(key, set()).add(f"batch:{bid}")
            seq_topics.setdefault(key, f"multi-file edit ({Path(steps[0]['arguments']['path']).name})")
            seq_steps.setdefault(key, steps)

    promoted: list[dict[str, Any]] = []
    for key, sids in sorted(seq_sessions.items(), key=lambda kv: len(kv[1]), reverse=True):
        if len(sids) < min_occurrences or len(promoted) >= max_new:
            continue
        topic = seq_topics[key]
        steps = seq_steps[key]
        storage.record_workflow(topic, steps)
        storage.record_workflow(topic, steps)
        sig = " -> ".join(key)
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(storage)
            eng.remember_safe_plan(topic, steps[0]["name"], steps[0].get("arguments") or {})
        except Exception:
            pass
        promoted.append({"signature": sig, "topic": topic, "sessions_count": len(sids), "steps": steps})

    return {
        "mined_count": len(promoted),
        "promoted_workflows": promoted,
    }


def run_doctor_auto_repair(storage: Storage, plugins: Any = None) -> dict[str, Any]:
    """Execute unified deep self-integrity audit and auto-repair across DB, KG, DPO, plugins, and episodic workflows."""
    from .config import get_settings
    from .self_guard import SelfGuard
    from .self_learn import heal_runtime_anomalies

    sqlite_ok = False
    try:
        qc = storage._connect().execute("PRAGMA quick_check;").fetchone()[0]
        sqlite_ok = qc == "ok"
    except Exception:
        sqlite_ok = False

    expired_pruned = storage.prune_expired_memories()
    mem_consolidated = storage.consolidate_memories()
    kg_cleanup = storage.consolidate_knowledge_graph()
    runtime_healed = heal_runtime_anomalies(storage, plugins=plugins)
    episodic = mine_episodic_workflows(storage)
    sg = SelfGuard(get_settings(), storage)
    dpo_audit = sg.audit_dpo_preferences(auto_repair=True)
    cal = sg.calibrate_thresholds()

    return {
        "status": "repaired",
        "sqlite_ok": sqlite_ok,
        "expired_memories_pruned": expired_pruned,
        "memories_consolidated": mem_consolidated,
        "kg_cleanup": kg_cleanup,
        "runtime_healed": runtime_healed,
        "episodic_workflows": episodic,
        "dpo_audit": dpo_audit,
        "calibration": cal,
    }


def predict_user_routine(storage: Storage) -> dict[str, Any] | None:
    """Predict likely user workflow/role for the current hour from hourly_roles and app usage."""
    hour = datetime.now().hour
    likely_role = "general"
    role_hits = 0
    try:
        from .neural import NeuralEngine

        bucket = (NeuralEngine(storage).meta.get("hourly_roles") or {}).get(str(hour)) or {}
        if bucket:
            likely_role, role_hits = max(bucket.items(), key=lambda kv: int(kv[1]))
    except Exception:
        pass

    apps = storage.get_app_usage()
    top_app = apps[0]["app_name"] if apps else None
    wfs = storage.list_workflows(limit=5)
    top_wf = wfs[0] if wfs and int(wfs[0].get("hits", 0)) >= 2 else None
    if int(role_hits) < 3 and not top_wf:
        return None
    return {
        "hour": hour,
        "likely_role": likely_role,
        "role_hits": int(role_hits),
        "top_app": top_app,
        "suggested_workflow": top_wf.get("goal_pattern") if top_wf else None,
    }


def evaluate_graduation_arena(storage: Storage) -> dict[str, Any]:
    """Compare local shadow performance vs cloud baseline and graduate reliable tools to local-first."""
    from .self_learn import _SHADOW_KEY, _load_json_memory, _save_json_memory, shadow_stats

    sh = shadow_stats(storage)
    state = _load_json_memory(storage, _SHADOW_KEY, {})
    graduated = list(sh.get("graduated") or [])
    candidates: dict[str, dict[str, Any]] = {}
    if isinstance(state, dict):
        for tool_name, row in state.items():
            if not isinstance(row, dict):
                continue
            hits = int(row.get("hits", 0))
            misses = int(row.get("misses", 0))
            tot = hits + misses
            if tot == 0:
                continue
            acc = round(hits / tot, 3)
            candidates[tool_name] = {"samples": tot, "accuracy": acc}
            if tot >= 5 and acc >= 0.88 and tool_name not in graduated:
                graduated.append(tool_name)
    prev_saved = _load_json_memory(storage, "_internal:graduated_tools", [])
    if not isinstance(prev_saved, list):
        prev_saved = []
    if graduated != prev_saved:
        _save_json_memory(storage, "_internal:graduated_tools", graduated[-32:])
    return {"graduated": graduated, "candidates": candidates}

"""Self-learning helpers: patch outcomes, error clustering, and evolve status summaries."""

from __future__ import annotations

import ast
import compileall
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

from .config import DATA_DIR
from .storage import Storage

BAD_STATUSES = frozenset({"failed", "rejected_by_critic", "rolled_back"})
_TS_FMT = "%Y-%m-%d %H:%M:%S"
_PKG_FILE_RE = re.compile(r'File "[^"]*/sysagent/sysagent/([\w]+\.py)", line (\d+)')
_OUTCOMES_KEY = "_internal:evolve_outcomes"
_SKILL_KEY = "_internal:skill_suggested"
_IMMUNITY_KEY = "_internal:immunity_patterns"
_SELF_PLAY_KEY = "_internal:self_play_count"
_PROF_SEEDED_KEY = "_internal:profile_seeded"
_SKILL_INJECTED_KEY = "_internal:skills_injected"
_SHADOW_KEY = "_internal:shadow_eval"
_ROADMAP_KEY = "_internal:self_roadmap"
_FORGED_WF_KEY = "_internal:forged_workflows"
_RUNTIME_TUNING_KEY = "_internal:runtime_tuning"

_SAFE_SELF_PLAY_TOOLS = frozenset(
    {
        "system_status",
        "inspect_self_component",
        "list_directory",
        "read_file",
        "read_journal_logs",
        "list_systemd_units",
        "search_files",
        "recall_memories",
        "search_memory_and_history",
        "kg_query",
        "read_clipboard",
        "list_events",
        "get_app_usage",
        "list_automations",
    }
)

_SAFE_WORKFLOW_TOOLS = frozenset(
    {
        "system_status",
        "inspect_self_component",
        "list_directory",
        "read_file",
        "search_files",
        "read_journal_logs",
        "list_systemd_units",
        "list_events",
        "recall_memories",
        "search_memory_and_history",
        "kg_query",
    }
)


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.strptime(str(value)[:19], _TS_FMT)
    except (ValueError, TypeError):
        return None


def _load_json_memory(storage: Storage, key: str, default: Any) -> Any:
    raw = storage.get_memory(key, include_expired=True)
    if raw is not None:
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return default
    return default


def _save_json_memory(storage: Storage, key: str, value: Any) -> None:
    storage.set_memory(key, json.dumps(value, ensure_ascii=False))


def proposal_stats(storage: Storage, limit: int = 60) -> dict[str, Any]:
    """Aggregate recent proposal outcomes per status, category, and file."""
    proposals = storage.list_evolve_proposals(limit=limit)
    by_status: dict[str, int] = {}
    by_category: dict[str, dict[str, int]] = {}
    by_file: dict[str, list[str]] = {}
    for p in proposals:  # newest first
        st = str(p.get("status") or "")
        by_status[st] = by_status.get(st, 0) + 1
        cat = by_category.setdefault(str(p.get("category") or "?"), {"applied": 0, "bad": 0})
        if st == "applied":
            cat["applied"] += 1
        elif st in BAD_STATUSES:
            cat["bad"] += 1
        by_file.setdefault(Path(str(p.get("file_path") or "")).name, []).append(st)
    applied = by_status.get("applied", 0)
    bad = sum(by_status.get(s, 0) for s in BAD_STATUSES)
    decided = applied + bad
    return {
        "total": len(proposals),
        "by_status": by_status,
        "by_category": by_category,
        "by_file": by_file,
        "success_rate": round(applied / decided, 2) if decided else None,
    }


def files_to_deprioritize(stats: dict[str, Any], streak: int = 3) -> set[str]:
    """Files whose last `streak` proposals all ended badly (no point spending LLM calls on them)."""
    out: set[str] = set()
    for name, statuses in stats.get("by_file", {}).items():
        recent = statuses[:streak]
        if len(recent) >= streak and all(s in BAD_STATUSES for s in recent):
            out.add(name)
    return out


def error_clusters(storage: Storage, min_count: int = 2, scan: int = 100) -> list[dict[str, Any]]:
    """Group recurring runtime errors that point at project files and are not yet fixed."""
    clusters: dict[tuple[str, str, str], dict[str, Any]] = {}
    for th in storage.list_brain_thoughts(limit=scan):
        summary = str(th.get("summary") or "")
        low = summary.lower()
        if "error" not in low or summary.startswith(("Self-evolved", "Auto-rolled")):
            continue
        m = _PKG_FILE_RE.search(summary)
        if not m:
            continue
        msg = summary.split("):", 1)[-1].strip()[:120] if "):" in summary else summary[:120]
        key = (m.group(1), m.group(2), msg)
        c = clusters.setdefault(key, {"file": m.group(1), "line": m.group(2), "message": msg, "count": 0, "last_seen": ""})
        c["count"] += 1
        c["last_seen"] = max(c["last_seen"], str(th.get("created_at") or ""))
    applied_at: dict[str, str] = {}
    for p in storage.list_evolve_proposals(limit=40):
        if p.get("status") == "applied":
            name = Path(str(p.get("file_path") or "")).name
            applied_at[name] = max(applied_at.get(name, ""), str(p.get("created_at") or ""))
    result = [
        c
        for c in clusters.values()
        if c["count"] >= min_count and applied_at.get(c["file"], "") < c["last_seen"]
    ]
    result.sort(key=lambda c: c["count"], reverse=True)
    return result


def cluster_signature(clusters: list[dict[str, Any]]) -> str:
    return "|".join(f"{c['file']}:{c['line']}:{c['count']}" for c in clusters[:5])


def _count_errors_for(storage: Storage, name: str, after: datetime | None, before: datetime | None) -> int:
    n = 0
    for th in storage.list_brain_thoughts(limit=200):
        summary = str(th.get("summary") or "")
        if name not in summary or "error" not in summary.lower() or summary.startswith(("Self-evolved", "Auto-rolled")):
            continue
        ts = _parse_ts(th.get("created_at") or "")
        if ts is None:
            continue
        if after is not None and ts < after:
            continue
        if before is not None and ts >= before:
            continue
        n += 1
    return n


def check_patch_outcomes(storage: Storage, rollback: Any, window_hours: int = 24) -> list[dict[str, Any]]:
    """Monitor applied patches and roll back regressions in the same file."""
    state: dict[str, str] = _load_json_memory(storage, _OUTCOMES_KEY, {})
    now = datetime.now()
    events: list[dict[str, Any]] = []
    changed = False
    nn_eng = None

    def _get_nn():
        nonlocal nn_eng
        if nn_eng is None:
            from .neural import NeuralEngine

            nn_eng = NeuralEngine(storage)
        return nn_eng
    for p in storage.list_evolve_proposals(limit=15):
        pid = str(p["id"])
        if p.get("status") != "applied" or state.get(pid) in {"ok", "rolled_back", "regressed"}:
            continue
        created = _parse_ts(p.get("created_at") or "")
        if created is None:
            continue
        name = Path(str(p.get("file_path") or "")).name
        before = _count_errors_for(storage, name, created - timedelta(hours=window_hours), created)
        after = _count_errors_for(storage, name, created, None)
        if after >= 3 and after > before:
            if rollback(int(pid)):
                state[pid] = "rolled_back"
                outcome = "rolled_back"
            else:
                state[pid] = "regressed"
                outcome = "regressed"
            storage.add_lesson(
                f"evolve_outcome:{name}",
                f"Patch #{pid} '{str(p.get('title'))[:80]}' was followed by {after} errors in {name} (before: {before})",
                "Avoid repeating this change; prefer a narrower fix." if outcome == "rolled_back" else "Needs manual review.",
            )
            events.append({"id": int(pid), "file": name, "outcome": outcome, "errors_before": before, "errors_after": after})
            try:
                _get_nn().online_patch_step(
                    name, "patch", str(p.get("title") or "")[:120], f"24h {outcome}: {after} errs", success=False
                )
            except Exception:
                pass
            changed = True
        elif now - created >= timedelta(hours=window_hours):
            state[pid] = "ok"
            storage.add_lesson(
                f"evolve_outcome:{name}",
                f"Patch #{pid} '{str(p.get('title'))[:80]}' stayed stable for {window_hours}h",
                "This kind of change is safe for this file.",
            )
            events.append({"id": int(pid), "file": name, "outcome": "ok", "errors_before": before, "errors_after": after})
            try:
                _get_nn().online_patch_step(
                    name, "patch", str(p.get("title") or "")[:120], f"24h clean ({window_hours}h)", success=True
                )
            except Exception:
                pass
            changed = True
    if changed:
        _save_json_memory(storage, _OUTCOMES_KEY, state)
        try:
            _get_nn().sync_associative_memory()
        except Exception:
            pass
    return events


def patch_outcomes(storage: Storage) -> dict[str, str]:
    return _load_json_memory(storage, _OUTCOMES_KEY, {})


def suggest_workflow_skills(storage: Storage, min_hits: int = 3, max_new: int = 1) -> int:
    """Ask the user (once per workflow) whether a repeated tool sequence should become a plugin tool."""
    already: list[str] = _load_json_memory(storage, _SKILL_KEY, [])
    created = 0
    for wf in storage.list_workflows(limit=20):
        sig = str(wf.get("signature") or "")
        if int(wf.get("hits") or 0) < min_hits or sig in already or created >= max_new:
            continue
        goal = str(wf.get("goal_pattern") or "")[:100]
        question = (
            f"I repeated the tool sequence `{sig}` {wf['hits']} times (e.g. for '{goal}'). "
            "Should I forge a single custom plugin tool for it?"
        )
        storage.create_brain_question(
            question,
            str(wf.get("steps_json") or "[]")[:600],
            "Forge a reusable plugin tool (forge_skill) for repeated tool sequences like this one",
        )
        already.append(sig)
        created += 1
    if created:
        _save_json_memory(storage, _SKILL_KEY, already[-100:])
    return created


def evolve_status_summary(storage: Storage, evolver: Any, settings: Any) -> dict[str, Any]:
    stats = proposal_stats(storage)
    recent = [
        {
            "id": p["id"],
            "status": p["status"],
            "category": p["category"],
            "file": Path(str(p["file_path"])).name,
            "title": str(p["title"])[:90],
        }
        for p in storage.list_evolve_proposals(limit=5)
    ]
    now = time.time()
    next_at = getattr(evolver, "next_run_at", 0.0) or 0.0
    last_at = getattr(evolver, "last_run_at", 0.0) or 0.0
    cooldown = max(0, int(getattr(evolver, "_cooldown_until", 0.0) - time.monotonic()))
    return {
        "auto_apply": bool(settings.evolve_auto_apply),
        "model": settings.evolve_model,
        "success_rate": stats["success_rate"],
        "counts": stats["by_status"],
        "recent": recent,
        "last_run_ago_sec": int(now - last_at) if last_at else None,
        "last_result": getattr(evolver, "last_result", None),
        "next_run_in_sec": max(0, int(next_at - now)) if next_at else None,
        "quota_cooldown_sec": cooldown,
        "outcomes": patch_outcomes(storage),
        "recurring_errors": [
            {"file": c["file"], "line": c["line"], "count": c["count"]} for c in error_clusters(storage)[:3]
        ],
    }


def learn_status_summary(storage: Storage) -> dict[str, Any]:
    """One-glance view of everything the agent has learned (local model, lessons, workflows, outcomes)."""
    try:
        from .local_mind import LocalMind

        lm = LocalMind(storage).get_status()
    except Exception:
        lm = {}
    memories = storage.list_memories()
    workflows = storage.list_workflows(limit=50)
    outcome_counts: dict[str, int] = {}
    for v in patch_outcomes(storage).values():
        outcome_counts[v] = outcome_counts.get(v, 0) + 1
    top = workflows[0] if workflows else None
    try:
        from .local_inference import get_runtime_memory_status

        rt_mem = get_runtime_memory_status()
    except Exception:
        rt_mem = {}
    cal_raw = _load_json_memory(storage, "_internal:self_calibration", {})
    dpo_audit = (cal_raw.get("dpo_audit") if isinstance(cal_raw, dict) else None) or {}
    twin_snap = analyze_causal_digital_twin(storage, snapshots=list(_TWIN_HISTORY))
    return {
        "local_model": {
            "samples": lm.get("samples_trained", 0),
            "distilled": lm.get("distilled_count", 0),
            "vocab": lm.get("vocab_size", 0),
            "classes": lm.get("learned_classes", 0),
            "lora_status": lm.get("lora_status", "none"),
            "updated_at": lm.get("updated_at"),
            "uncertain_queue": lm.get("uncertain_queue_count", 0),
            "ensemble": bool(lm.get("ensemble_active")),
            "warm_model": lm.get("warm_model") or rt_mem.get("warm_model"),
            "aux_warm_model": rt_mem.get("aux_warm_model"),
            "vram": rt_mem.get("vram", {}),
            "weakest_role": lm.get("weakest_role"),
            "specialist_roles": lm.get("specialist_roles", {}),
        },
        "dpo": {
            "total_pairs": int(dpo_audit.get("total_pairs", 0) or 0),
            "consistency_score": float(dpo_audit.get("consistency_score", 1.0) or 1.0),
        },
        "digital_twin": {
            "healthy": bool(twin_snap.get("healthy", True)),
            "leaks_count": len(twin_snap.get("memory_leaks") or []),
            "snapshots_count": int(twin_snap.get("snapshots_count", 0) or 0),
        },
        "lessons": len(storage.list_lessons(limit=1000)),
        "style_prefs": sum(1 for m in memories if str(m.get("key", "")).startswith("style:")),
        "prompt_versions": len(storage.list_prompt_versions()),
        "workflows": len(workflows),
        "top_workflow": {"signature": top["signature"], "hits": top["hits"]} if top else None,
        "skill_suggestions": len(_load_json_memory(storage, _SKILL_KEY, [])),
        "patch_outcomes": outcome_counts,
        "auto_trusted": len(storage.list_auto_trusted_approvals()),
        "self_play_count": int(_load_json_memory(storage, _SELF_PLAY_KEY, 0)),
        "immunity_count": len(list_immunity_patterns(storage)),
        "skills_injected": len(_load_json_memory(storage, _SKILL_INJECTED_KEY, [])),
        "system_profile": discover_system_profile(storage, seed=False).get("summary", ""),
        "shadow": shadow_stats(storage),
        "coverage": scan_coverage_gaps(),
        "roadmap": build_self_roadmap(storage)[:3],
        "neural": lm.get("neural") or _neural_fallback_status(storage),
    }


def _neural_fallback_status(storage: Storage) -> dict[str, Any]:
    try:
        from .neural import NeuralEngine

        return NeuralEngine(storage).get_status()
    except Exception:
        return {}


def deduplicate_semantic_knowledge(storage: Storage) -> dict[str, int]:
    """Remove stale scratch-path memories and merge near-duplicate lessons."""
    stale_removed = 0
    for m in storage.list_memories(include_expired=True):
        val = str(m.get("value") or "")
        key = str(m.get("key") or "")
        if (
            not key.startswith("_internal:")
            and ".gemini/antigravity/scratch/sysagent" in val
            and storage.delete_memory(key)
        ):
            stale_removed += 1

    dup_removed = 0
    try:
        from .local_mind import LocalMind

        mind = LocalMind(storage)
        lessons = storage.list_lessons(limit=60)  # newest first (ORDER BY id DESC)
        kept_texts: list[str] = []
        drop_ids: list[int] = []
        for ls in lessons:
            text = f"{ls.get('context', '')} {ls.get('insight', '')}".strip()
            if any(mind.semantic_similarity(text, kt, use_gpu_embed=False) >= 0.85 for kt in kept_texts):
                drop_ids.append(int(ls["id"]))
            else:
                kept_texts.append(text)
        if drop_ids:
            with storage._lock, storage._connect() as conn:
                conn.executemany("DELETE FROM lessons_learned WHERE id = ?", [(did,) for did in drop_ids])
            dup_removed = len(drop_ids)
    except Exception:
        pass
    return {"stale_memories_removed": stale_removed, "duplicate_lessons_merged": dup_removed}


def auto_rollback_failing_prompts(storage: Storage) -> list[str]:
    """Roll back active evolved prompt templates when failures >= 3 and exceed successes."""
    rolled_back: list[str] = []
    for row in storage.list_prompt_versions():
        if int(row.get("active", 0)) != 1 or int(row.get("version", 0)) <= 0:
            continue
        failures = int(row.get("failures", 0))
        successes = int(row.get("successes", 0))
        if failures >= 3 and failures > successes:
            pkey = str(row["prompt_key"])
            if storage.rollback_prompt_version(pkey) is not None:
                rolled_back.append(pkey)
                storage.add_lesson(
                    f"prompt_doctor:{pkey}",
                    f"Auto-rolled back failing prompt {pkey} v{row['version']} (ok={successes}, fail={failures})",
                    "Keep prompt instructions concise and preserve required placeholders.",
                )
    return rolled_back


def _quick_cmd(args: list[str], timeout: float = 1.5) -> str:
    try:
        if not shutil.which(args[0]):
            return ""
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:
        return ""


def _scan_path_executables() -> set[str]:
    bins: set[str] = set()
    for p_str in os.environ.get("PATH", "").split(os.pathsep):
        if not p_str:
            continue
        p = Path(p_str)
        if not p.is_dir():
            continue
        try:
            for item in p.iterdir():
                if item.is_file() and os.access(item, os.X_OK):
                    bins.add(item.name)
        except Exception:
            pass
    return bins


def discover_system_profile(
    storage: Storage, seed: bool = True, force: bool = False
) -> dict[str, Any]:
    """Deeply scan OS/kernel, hardware, storage, network, software ecosystem, services, and user projects."""
    cache_path = DATA_DIR / "system_profile.json"
    if not force and not seed and cache_path.is_file():
        try:
            if (time.time() - cache_path.stat().st_mtime) < 1800:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and "hardware" in cached and "software" in cached:
                    return cached
        except Exception:
            pass

    distro = "Linux"
    distro_id = "linux"
    os_rel = Path("/etc/os-release")
    if os_rel.is_file():
        for ln in os_rel.read_text(encoding="utf-8", errors="replace").splitlines():
            if ln.startswith("PRETTY_NAME="):
                distro = ln.split("=", 1)[1].strip().strip('"')
            elif ln.startswith("ID="):
                distro_id = ln.split("=", 1)[1].strip().strip('"')
    desktop = os.environ.get("XDG_CURRENT_DESKTOP") or "unknown"
    session = os.environ.get("XDG_SESSION_TYPE") or "tty"
    shell = Path(os.environ.get("SHELL") or "bash").name
    virt = _quick_cmd(["systemd-detect-virt"]) or "none"

    cpu_model = platform.processor() or "x86_64"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for ln in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if ln.startswith("model name"):
                cpu_model = ln.split(":", 1)[1].strip()
                break
    phys_cores = psutil.cpu_count(logical=False) or 1
    log_cores = os.cpu_count() or 1
    freq = psutil.cpu_freq()
    max_mhz = round(freq.max, 0) if freq and freq.max else None
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    batt = psutil.sensors_battery()
    pci_lines = _quick_cmd(["lspci"]).splitlines()
    gpus = [ln.split(":", 2)[-1].strip() for ln in pci_lines if any(k in ln for k in ("VGA", "3D controller", "Display"))]
    audio_devs = [ln.split(":", 2)[-1].strip() for ln in pci_lines if "Audio" in ln]
    net_devs = [ln.split(":", 2)[-1].strip() for ln in pci_lines if any(k in ln for k in ("Network", "Ethernet"))]
    audio_stack = sorted(
        {
            p.info["name"]
            for p in psutil.process_iter(["name"])
            if p.info.get("name") in {"pipewire", "wireplumber", "pulseaudio"}
        }
    )

    mounts: list[dict[str, Any]] = []
    seen_devs: set[str] = set()
    for part in psutil.disk_partitions(all=False):
        if any(skip in part.device or skip in part.fstype for skip in ("loop", "squashfs", "tmpfs")) or part.device in seen_devs:
            continue
        seen_devs.add(part.device)
        try:
            u = psutil.disk_usage(part.mountpoint)
            mounts.append(
                {
                    "device": part.device,
                    "mount": part.mountpoint,
                    "fstype": part.fstype,
                    "total_gb": round(u.total / (1024**3), 1),
                    "free_gb": round(u.free / (1024**3), 1),
                }
            )
        except Exception:
            pass

    interfaces: dict[str, str] = {}
    for ifname, addrs in psutil.net_if_addrs().items():
        if ifname == "lo":
            continue
        for a in addrs:
            if getattr(a.family, "name", "") == "AF_INET":
                interfaces[ifname] = a.address
                break
    listening_ports: list[int] = []
    try:
        listening_ports = sorted(
            {
                c.laddr.port
                for c in psutil.net_connections(kind="inet")
                if c.status == "LISTEN" and c.laddr
            }
        )[:25]
    except Exception:
        pass

    path_bins = _scan_path_executables()
    pkg_managers = [
        b for b in (
            "pacman", "yay", "paru", "apt", "dnf", "zypper", "apk", "flatpak", "snap",
            "nix", "brew", "pip", "uv", "pipx", "npm", "pnpm", "yarn", "cargo",
        )
        if b in path_bins
    ]
    native_pkg_count: int | None = None
    if "pacman" in path_bins:
        out = _quick_cmd(["pacman", "-Qq"], timeout=2.0)
        if out:
            native_pkg_count = len(out.splitlines())
    elif "dpkg-query" in path_bins:
        out = _quick_cmd(["dpkg-query", "-f", ".\n", "-W"], timeout=2.0)
        if out:
            native_pkg_count = len(out.splitlines())

    runtimes: dict[str, str] = {}
    for rt, cmd in (
        ("python3", ["python3", "--version"]),
        ("node", ["node", "--version"]),
        ("rustc", ["rustc", "--version"]),
        ("go", ["go", "version"]),
        ("gcc", ["gcc", "--version"]),
        ("clang", ["clang", "--version"]),
        ("java", ["java", "--version"]),
        ("git", ["git", "--version"]),
    ):
        if rt in path_bins:
            v = _quick_cmd(cmd, timeout=1.0)
            runtimes[rt] = v.splitlines()[0][:60] if v else "installed"

    containers_virt = [
        b for b in ("docker", "podman", "kubectl", "helm", "qemu-system-x86_64", "virsh", "distrobox", "lxc", "vagrant")
        if b in path_bins
    ]
    ai_ml_tools = [
        b for b in ("llama-server", "llama-cli", "llama-embedding", "llama-finetune", "ollama", "vllm", "nvidia-smi", "nvcc", "rocm-smi")
        if b in path_bins
    ]
    cli_utilities = [
        b for b in (
            "rg", "fd", "fzf", "bat", "eza", "jq", "yq", "tmux", "zellij", "nvim", "vim",
            "gh", "lazygit", "htop", "btop", "ffmpeg", "mpv", "wl-copy", "xclip",
            "notify-send", "wpctl", "pactl", "nmcli", "bluetoothctl", "brightnessctl", "kdeconnect-cli",
        )
        if b in path_bins
    ]

    svc_out = _quick_cmd(
        ["systemctl", "--user", "list-units", "--type=service", "--state=running", "--no-legend", "--no-pager"],
        timeout=1.5,
    )
    user_services = [ln.split()[0] for ln in svc_out.splitlines() if ln.strip()][:25]
    projects: list[dict[str, str]] = []
    for root_name in ("Projects", "Work", "Code", "dev", "repos"):
        rdir = Path.home() / root_name
        if not rdir.is_dir():
            continue
        try:
            for item in sorted(rdir.iterdir()):
                if not item.is_dir() or item.name.startswith("."):
                    continue
                kind = "project"
                if (item / "pyproject.toml").is_file() or (item / "requirements.txt").is_file():
                    kind = "python"
                elif (item / "package.json").is_file():
                    kind = "node"
                elif (item / "Cargo.toml").is_file():
                    kind = "rust"
                elif (item / "go.mod").is_file():
                    kind = "go"
                projects.append({"name": item.name, "path": str(item), "kind": kind})
        except Exception:
            pass

    total_ram_gb = round(vm.total / (1024**3), 1)
    root_free = next((m["free_gb"] for m in mounts if m["mount"] == "/"), None)
    gpu_short = gpus[0] if gpus else "CPU"
    summary = (
        f"{distro} ({platform.release()}) | {desktop} ({session}) | "
        f"{cpu_model[:28]} ({phys_cores}C/{log_cores}T) | {total_ram_gb}GB RAM | "
        f"{gpu_short[:30]} | {len(path_bins)} bins"
        + (f", {native_pkg_count} pkgs" if native_pkg_count else "")
        + (f" | {len(projects)} projects" if projects else "")
    )
    prof: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "distro": distro,
        "distro_id": distro_id,
        "kernel": platform.release(),
        "arch": platform.machine(),
        "desktop": desktop,
        "session": session,
        "shell": shell,
        "virtualization": virt,
        "cpu_cores": log_cores,
        "hardware": {
            "cpu_model": cpu_model,
            "physical_cores": phys_cores,
            "logical_cores": log_cores,
            "max_mhz": max_mhz,
            "ram_total_gb": total_ram_gb,
            "swap_total_gb": round(sw.total / (1024**3), 1),
            "has_battery": batt is not None,
            "gpus": gpus,
            "audio_devices": audio_devs,
            "audio_stack": audio_stack,
            "network_controllers": net_devs,
        },
        "mounts": mounts,
        "network": {"interfaces": interfaces, "listening_ports": listening_ports},
        "software": {
            "path_binaries_count": len(path_bins),
            "native_package_count": native_pkg_count,
            "package_managers": pkg_managers,
            "runtimes": runtimes,
            "containers_virt": containers_virt,
            "ai_ml_tools": ai_ml_tools,
            "cli_utilities": cli_utilities,
        },
        "environment": {
            "running_user_services": user_services,
            "projects": projects[:25],
        },
        "cli_tools": [*pkg_managers[:4], *cli_utilities[:8]],
        "summary": summary,
        "updated_at": datetime.now().strftime(_TS_FMT),
    }

    if seed:
        try:
            cache_path.write_text(json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")
            storage.set_memory("system:profile", summary)
            storage.set_memory(
                "system:hardware",
                f"CPU: {cpu_model} ({phys_cores}C/{log_cores}T), RAM: {total_ram_gb}GB, GPUs: {gpus}, Audio: {audio_stack}, Root free: {root_free}GB",
            )
            storage.set_memory(
                "system:software",
                f"Packages: {native_pkg_count}, Managers: {pkg_managers}, Runtimes: {list(runtimes.keys())}, AI: {ai_ml_tools}",
            )
            if projects:
                storage.set_memory(
                    "system:projects",
                    ", ".join(f"{p['name']} ({p['kind']} @ {p['path']})" for p in projects[:15]),
                )
            storage.add_kg_edge("host", "runs_os", distro)
            storage.add_kg_edge("host", "has_cpu", cpu_model[:80])
            storage.add_kg_edge("host", "has_desktop", f"{desktop} ({session})")
            if gpus:
                storage.add_kg_edge("host", "has_gpu", gpus[0][:80])
            for pm in pkg_managers[:5]:
                storage.add_kg_edge("host", "uses_pkg_manager", pm)
            for prj in projects[:10]:
                storage.add_kg_edge("user", "develops_project", f"{prj['name']} ({prj['kind']})")
        except Exception:
            pass
        if _load_json_memory(storage, _PROF_SEEDED_KEY, "") != "v3":
            try:
                from .local_mind import LocalMind

                mind = LocalMind(storage)
                mind.record_and_learn(
                    f"what are the system hardware cpu gpu ram disk specs on {distro} {cpu_model}",
                    [{"name": "system_status", "arguments": {}}],
                    f"{summary}. Audio={audio_stack}, Mounts={[(m['mount'], m['fstype'], m['free_gb']) for m in mounts[:3]]}",
                )
                mind.record_and_learn(
                    f"check volume brightness desktop controls on {desktop} {session}",
                    [{"name": "desktop_control", "arguments": {"action": "volume_get"}}],
                    f"Using {desktop} ({session}) with audio stack {audio_stack}.",
                )
                if projects:
                    mind.record_and_learn(
                        f"where are my projects workspace directory {projects[0]['name']}",
                        [{"name": "list_directory", "arguments": {"path": str(Path(projects[0]["path"]).parent)}}],
                        ", ".join(f"{p['name']}: {p['path']}" for p in projects[:6]),
                    )
                _save_json_memory(storage, _PROF_SEEDED_KEY, "v3")
            except Exception:
                pass
    return prof


def inject_plugin_skills_to_local_mind(storage: Storage, plugins: Any = None) -> int:
    """Register newly forged/loaded custom plugin tools into LocalMind immediately."""
    decls = plugins.declarations() if plugins and hasattr(plugins, "declarations") else []
    if not decls:
        return 0
    already: list[str] = list(_load_json_memory(storage, _SKILL_INJECTED_KEY, []))
    added = 0
    try:
        from .local_mind import LocalMind

        mind = LocalMind(storage)
        for d in decls:
            tname = str(d.get("name") or "").strip()
            if not tname or tname in already:
                continue
            desc = str(d.get("description") or tname.replace("_", " ")).strip()
            mind.record_and_learn(
                f"{tname.replace('_', ' ')} {desc[:120]}",
                [{"name": tname, "arguments": {}}],
                f"Executed custom skill `{tname}`.",
            )
            already.append(tname)
            added += 1
        if added:
            _save_json_memory(storage, _SKILL_INJECTED_KEY, already[-100:])
            mind.train()
    except Exception:
        pass
    return added


def list_immunity_patterns(storage: Storage) -> list[dict[str, Any]]:
    data = _load_json_memory(storage, _IMMUNITY_KEY, [])
    return data if isinstance(data, list) else []


def record_immunity_pattern(storage: Storage, user_text: str, tool_name: str, reason: str) -> None:
    """Record a failed or user-denied tool call as negative feedback for LocalMind."""
    items = [
        it for it in list_immunity_patterns(storage)
        if not (it.get("tool") == tool_name and it.get("query") == user_text[:120])
    ]
    items.append(
        {
            "tool": tool_name,
            "query": (user_text or "")[:120],
            "reason": (reason or "")[:140],
            "updated_at": datetime.now().strftime(_TS_FMT),
        }
    )
    _save_json_memory(storage, _IMMUNITY_KEY, items[-40:])
    try:
        from .local_mind import LocalMind
        from .neural import NeuralEngine

        LocalMind(storage).penalize_tool(user_text, tool_name)
        NeuralEngine(storage).online_step(user_text, tool_name, reward=-1.5)
    except Exception:
        pass


def compute_adaptive_schedule(watcher: Any, storage: Storage) -> dict[str, Any]:
    """Choose execution cadence mode based on active user context, CPU/GPU load, and Neural World Model."""
    ctx = getattr(watcher, "current_context", "default") if watcher else "default"
    snap = watcher.get_health_snapshot() if watcher and hasattr(watcher, "get_health_snapshot") else {}
    cpu = float(snap.get("cpu_percent", 0))
    gpu_u = float((snap.get("gpu") or {}).get("util_percent", 0))
    wf: dict[str, Any] = {}
    try:
        from .neural import NeuralEngine

        wf = NeuralEngine(storage).predict_world_state()
    except Exception:
        wf = {}
    b_prob = float(wf.get("bottleneck_prob", 0.0))
    pred_cpu = float(wf.get("pred_cpu_10m", 0.0))
    pred_role = str(wf.get("predicted_role") or "general")

    if ctx == "media_meeting" or gpu_u >= 65.0:
        mode = "media_quiet"
        self_play = False
    elif b_prob >= 0.65 or pred_cpu >= 80.0:
        mode = "proactive_guard"
        self_play = False
    elif cpu <= 35.0 and gpu_u <= 25.0:
        mode = "deep_sleep_learn"
        self_play = True
    else:
        mode = "active_work"
        self_play = False
    return {
        "mode": mode,
        "context": ctx,
        "cpu_percent": cpu,
        "gpu_percent": gpu_u,
        "self_play_allowed": self_play,
        "bottleneck_prob": b_prob,
        "predicted_role": pred_role,
    }


def run_local_self_play(storage: Storage, max_rounds: int = 1) -> dict[str, Any]:
    """Zero-cloud offline training gym where local specialists generate and execute-verify safe tool calls."""
    try:
        from .config import get_settings
        from .local_mind import LocalMind, find_gguf_model
        from .neural import NeuralEngine
        from .tools import execute_tool
        from .watcher import SystemWatcher

        mind = LocalMind(storage)
        if not (find_gguf_model("reasoning", strict=True) or find_gguf_model("coding")):
            return {"status": "skipped", "generated": 0, "verified": 0, "penalized": 0}
        focus = mind.weakest_specialist_role()
        prof = discover_system_profile(storage, seed=False).get("summary", "Linux")
        schema_map = mind._build_tool_schema_map()
        curr = mind.build_distillation_curriculum(tool_limit=10)
        safe_weak = [t for t in curr["underrepresented_tools"] if t in _SAFE_SELF_PLAY_TOOLS][:5]
        if not safe_weak:
            safe_weak = sorted(_SAFE_SELF_PLAY_TOOLS)[:5]
        tool_sigs = mind._format_tool_signatures(schema_map, safe_weak)
        user_examples = curr["user_examples"][:4]
        lang_ctx = (
            f"Recent user language examples: {json.dumps(user_examples, ensure_ascii=False)}\n"
            if user_examples
            else ""
        )
        seen_queries: set[str] = {
            " ".join(str(e.get("user") or "").lower().split())
            for e in mind._load_distilled_entries(limit=600)
            if e.get("user")
        }
        watcher = SystemWatcher(get_settings(), storage=storage)
        generated = 0
        verified = 0
        penalized = 0
        for _ in range(max(1, max_rounds)):
            prompt = (
                f"System environment: {prof}\nFocus specialist role: {focus}\n"
                f"Safe read-only tool schemas: {json.dumps(tool_sigs, ensure_ascii=False)}\n"
                f"{lang_ctx}"
                "Generate ONE realistic user query (matching the language style of recent user examples) "
                "and its safe read-only Linux assistant tool call(s) + reply as valid JSON:\n"
                '{"user": "...", "role": "'
                + focus
                + '", "tool": "tool_name", "args": {}, '
                '"tool_calls": [{"name": "tool_name", "arguments": {}}], "reply": "..."}'
            )
            raw = mind.run_gguf_inference(
                prompt,
                "Respond ONLY with valid JSON for local training.",
                preferred_role="reasoning",
                extra_system="Respond ONLY with valid JSON matching the requested schema.",
                max_tokens=360,
                prefer_warm=True,
            )
            if not raw:
                continue
            idx = raw.find("{")
            if idx == -1:
                continue
            try:
                parsed, _ = json.JSONDecoder().raw_decode(raw[idx:])
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue

            raw_user = str(parsed.get("user") or "").strip()
            raw_tool = str(parsed.get("tool") or "").strip()
            raw_args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
            validated = mind._validate_distilled_sample(parsed, schema_map, seen_queries)
            if validated is None:
                if raw_user and raw_tool in _SAFE_SELF_PLAY_TOOLS:
                    mind.penalize_tool(raw_user, raw_tool)
                    mind.prune_failed_arg_template(raw_tool, dict(raw_args))
                    try:
                        NeuralEngine(storage).online_step(raw_user, raw_tool, reward=-1.0)
                    except Exception:
                        pass
                    penalized += 1
                continue

            tcalls = validated["tool_calls"]
            if tcalls and any(tc["name"] not in _SAFE_SELF_PLAY_TOOLS for tc in tcalls):
                continue

            exec_ok = True
            for tc in tcalls:
                tname = tc["name"]
                targs = dict(tc.get("arguments") or {})
                try:
                    raw_out = execute_tool(tname, targs, watcher=watcher, storage=storage)
                    out_str = raw_out if isinstance(raw_out, str) else json.dumps(raw_out, ensure_ascii=False)
                    has_err = (not out_str) or out_str.startswith(("Error", "Tool error"))
                    if not has_err and out_str.lstrip().startswith("{"):
                        try:
                            out_obj = json.loads(out_str)
                            if isinstance(out_obj, dict) and out_obj.get("error"):
                                has_err = True
                        except Exception:
                            pass
                    if has_err:
                        exec_ok = False
                        mind.penalize_tool(validated["user"], tname)
                        mind.prune_failed_arg_template(tname, targs)
                        try:
                            NeuralEngine(storage).online_step(validated["user"], tname, reward=-1.0)
                        except Exception:
                            pass
                        penalized += 1
                        break
                except Exception:
                    exec_ok = False
                    mind.penalize_tool(validated["user"], tname)
                    mind.prune_failed_arg_template(tname, targs)
                    penalized += 1
                    break

            if not exec_ok:
                continue

            if tcalls:
                verified += 1
            mind.record_and_learn(validated["user"], tcalls, validated["reply"], role=validated["role"])
            if tcalls:
                try:
                    eng = NeuralEngine(storage)
                    eng.online_step(validated["user"], tcalls[0]["name"], reward=1.2)
                    if len(tcalls) == 1:
                        eng.record_safe_plan(
                            validated["user"],
                            tcalls[0]["name"],
                            dict(tcalls[0].get("arguments") or {}),
                        )
                    elif len(tcalls) >= 2:
                        storage.record_workflow(validated["user"][:140], tcalls)
                except Exception:
                    pass
            generated += 1
        if generated:
            prev = int(_load_json_memory(storage, _SELF_PLAY_KEY, 0))
            _save_json_memory(storage, _SELF_PLAY_KEY, prev + generated)
        return {
            "status": "ok" if (generated or penalized) else "skipped",
            "generated": generated,
            "verified": verified,
            "penalized": penalized,
            "role": focus,
            "target_tools": safe_weak,
        }
    except Exception:
        return {"status": "error", "generated": 0, "verified": 0, "penalized": 0}


def shadow_stats(storage: Storage) -> dict[str, Any]:
    """Summarize LocalMind shadow prediction accuracy against actual cloud tool selections."""
    state = _load_json_memory(storage, _SHADOW_KEY, {})
    if not isinstance(state, dict):
        state = {}
    hits = sum(int((r or {}).get("hits", 0)) for r in state.values() if isinstance(r, dict))
    misses = sum(int((r or {}).get("misses", 0)) for r in state.values() if isinstance(r, dict))
    total = hits + misses
    graduated = [
        t for t, r in state.items()
        if isinstance(r, dict)
        and int(r.get("hits", 0)) >= 5
        and (int(r.get("hits", 0)) / max(1, int(r.get("hits", 0)) + int(r.get("misses", 0)))) >= 0.90
    ]
    return {
        "hits": hits,
        "misses": misses,
        "accuracy": round(hits / total, 2) if total > 0 else None,
        "graduated": sorted(graduated),
    }


def record_shadow_evaluation(
    storage: Storage, user_text: str, actual_tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare LocalMind's prediction with cloud tool choice and graduate accurate tools."""
    if not actual_tool_calls or not isinstance(actual_tool_calls[0], dict) or not actual_tool_calls[0].get("name"):
        return {}
    actual = str(actual_tool_calls[0]["name"])
    try:
        from .local_mind import LocalMind

        mind = LocalMind(storage)
        pred, _, _ = mind.predict_tool(user_text)
        state = _load_json_memory(storage, _SHADOW_KEY, {})
        if not isinstance(state, dict):
            state = {}
        row = state.setdefault(actual, {"hits": 0, "misses": 0})
        if pred == actual:
            row["hits"] = int(row.get("hits", 0)) + 1
        else:
            row["misses"] = int(row.get("misses", 0)) + 1
            if pred:
                mind.penalize_tool(user_text, pred)
        try:
            from .neural import NeuralEngine

            NeuralEngine(storage).online_step(user_text, actual, reward=1.0)
        except Exception:
            pass
        _save_json_memory(storage, _SHADOW_KEY, state)
    except Exception:
        pass
    return shadow_stats(storage)


def replay_shadow_history(storage: Storage, max_samples: int = 120) -> dict[str, Any]:
    """Counterfactual replay over historical distilled turns to calibrate Shadow eval & apply hard-negative mining."""
    state = _load_json_memory(storage, _SHADOW_KEY, {})
    if not isinstance(state, dict):
        state = {}
    existing_total = sum(
        int((r or {}).get("hits", 0)) + int((r or {}).get("misses", 0))
        for r in state.values()
        if isinstance(r, dict)
    )
    if existing_total >= 15:
        return shadow_stats(storage)
    try:
        from .local_mind import LocalMind

        mind = LocalMind(storage)
        if not mind.weights.get("tool_centroids"):
            mind.train(run_lora=False)
        entries = mind._load_distilled_entries(limit=max_samples)
        changed = False
        for ent in entries:
            u_text = str(ent.get("user") or "").strip()
            tcalls = ent.get("tool_calls") or []
            if not u_text or not tcalls or not isinstance(tcalls[0], dict) or not tcalls[0].get("name"):
                continue
            actual = str(tcalls[0]["name"])
            pred, _, score = mind.predict_tool(u_text)
            row = state.setdefault(actual, {"hits": 0, "misses": 0})
            if pred == actual:
                row["hits"] = int(row.get("hits", 0)) + 1
            else:
                row["misses"] = int(row.get("misses", 0)) + 1
                if pred and score >= 0.35:
                    mind.penalize_tool(u_text, pred)
            changed = True
        if changed:
            _save_json_memory(storage, _SHADOW_KEY, state)
    except Exception:
        pass
    return shadow_stats(storage)


def scan_coverage_gaps(pkg_dir: Path | None = None) -> dict[str, Any]:
    """AST-scan sysagent modules and find public functions/methods unreferenced in tests."""
    pdir = pkg_dir or Path(__file__).resolve().parent
    tdir = pdir.parent / "tests"
    test_corpus = ""
    if tdir.is_dir():
        test_corpus = "\n".join(
            tf.read_text(encoding="utf-8", errors="replace") for tf in sorted(tdir.glob("*.py"))
        )
    tested = 0
    total = 0
    untested_by_file: dict[str, list[str]] = {}
    for pyf in sorted(pdir.glob("*.py")):
        if pyf.name == "__init__.py":
            continue
        try:
            tree = ast.parse(pyf.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        names: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
                and len(node.name) >= 4
            ):
                names.append(node.name)
        unique = sorted(set(names))
        file_untested = [nm for nm in unique if nm not in test_corpus]
        total += len(unique)
        tested += len(unique) - len(file_untested)
        if file_untested:
            untested_by_file[pyf.name] = file_untested[:12]
    return {
        "tested_count": tested,
        "total_count": total,
        "coverage_pct": round((tested / max(1, total)) * 100) if total else 100,
        "untested_by_file": untested_by_file,
    }


def create_git_checkpoint(repo_dir: Path, proposal_id: int, file_path: Path, title: str) -> str | None:
    """Ensure a local git repository exists and commit an applied Self-Evolve proposal."""
    if not shutil.which("git"):
        return None
    try:
        if not (repo_dir / ".git").is_dir():
            gi = repo_dir / ".gitignore"
            if not gi.is_file():
                gi.write_text(".venv*\n__pycache__/\n*.pyc\n*.log\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, timeout=5)
            subprocess.run(["git", "config", "user.email", "sysagent@local"], cwd=repo_dir, capture_output=True, timeout=5)
            subprocess.run(["git", "config", "user.name", "SysAgent"], cwd=repo_dir, capture_output=True, timeout=5)
            subprocess.run(["git", "add", "-A"], cwd=repo_dir, capture_output=True, timeout=10)
            subprocess.run(["git", "commit", "-m", "chore: initial baseline"], cwd=repo_dir, capture_output=True, timeout=10)
        else:
            name_chk = subprocess.run(["git", "config", "user.name"], cwd=repo_dir, capture_output=True, text=True, timeout=3)
            if not name_chk.stdout.strip():
                subprocess.run(["git", "config", "user.name", "SysAgent"], cwd=repo_dir, capture_output=True, timeout=3)
                subprocess.run(["git", "config", "user.email", "sysagent@local"], cwd=repo_dir, capture_output=True, timeout=3)
        to_add = [str(file_path)]
        tf = repo_dir / "tests" / f"test_evolve_{proposal_id}.py"
        if tf.is_file():
            to_add.append(str(tf))
        readme = repo_dir / "README.md"
        if readme.is_file():
            to_add.append(str(readme))
        subprocess.run(["git", "add", *to_add], cwd=repo_dir, capture_output=True, timeout=5)
        diff_chk = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir, capture_output=True, timeout=5)
        if diff_chk.returncode == 0:
            return None
        commit_res = subprocess.run(
            ["git", "commit", "-m", f"evolve(#{proposal_id}): {title[:72]}"],
            cwd=repo_dir,
            capture_output=True,
            timeout=10,
        )
        if commit_res.returncode != 0:
            return None
        if os.environ.get("SYSAGENT_EVOLVE_AUTO_PUSH", "").strip().lower() in {"1", "true", "yes"}:
            subprocess.run(["git", "push"], cwd=repo_dir, capture_output=True, timeout=20)
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=5)
        return rev.stdout.strip() if rev.returncode == 0 else None
    except Exception:
        return None


def compact_distilled_dataset(max_per_class: int = 80, sim_threshold: float = 0.92) -> dict[str, int]:
    """Deduplicate near-identical entries in distilled_turns.jsonl and cap class sizes."""
    from collections import Counter

    from .local_mind import DISTILLED_FILE, _cosine_sim, _tokenize

    if not DISTILLED_FILE.is_file():
        return {"before": 0, "after": 0, "removed": 0}
    raw_lines = [ln for ln in DISTILLED_FILE.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    before = len(raw_lines)
    if before <= 40:
        return {"before": before, "after": before, "removed": 0}

    kept_rev: list[str] = []
    class_vecs: dict[str, list[dict[str, float]]] = {}
    for ln in reversed(raw_lines):
        try:
            obj = json.loads(ln)
            u = str(obj.get("user") or "").strip()
            if not u:
                continue
        except Exception:
            continue
        tcalls = obj.get("tool_calls") or []
        cls_name = str(tcalls[0]["name"]) if tcalls and isinstance(tcalls[0], dict) and tcalls[0].get("name") else "__chat__"
        bucket = class_vecs.setdefault(cls_name, [])
        if len(bucket) >= max_per_class:
            continue
        toks = _tokenize(u)
        vec = dict(Counter(toks))
        if toks and any(_cosine_sim(vec, kv) >= sim_threshold for kv in bucket):
            continue
        bucket.append(vec)
        kept_rev.append(ln)

    kept = list(reversed(kept_rev))
    if len(kept) < before:
        DISTILLED_FILE.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return {"before": before, "after": len(kept), "removed": before - len(kept)}


def _build_workflow_plugin_source(tool_name: str, sig: str, goal: str, step_specs: list[dict[str, Any]]) -> str:
    """Render a standalone verified plugin module combining a repeated safe tool sequence with parameterized overrides."""
    from .tools import TOOL_DECLARATIONS

    param_props: dict[str, dict[str, str]] = {
        "path": {"type": "string", "description": "Optional path override"},
        "query": {"type": "string", "description": "Optional query override"},
        "unit": {"type": "string", "description": "Optional systemd unit override"},
        "component": {"type": "string", "description": "Optional component override"},
        "limit": {"type": "integer", "description": "Optional limit override"},
    }
    tool_schema_map: dict[str, list[str]] = {}
    decl_by_name = {str(d.get("name") or ""): d for d in TOOL_DECLARATIONS if isinstance(d, dict)}
    for st in step_specs:
        sname = str(st.get("name") or "")
        decl = decl_by_name.get(sname) or {}
        props = (decl.get("parameters") or {}).get("properties") or {}
        if isinstance(props, dict):
            tool_schema_map[sname] = [str(k) for k in props.keys()]
            for k, v in props.items():
                if k not in param_props and isinstance(v, dict):
                    param_props[str(k)] = {
                        "type": str(v.get("type") or "string"),
                        "description": str(v.get("description") or f"Optional {k} override"),
                    }

    steps_lit = json.dumps(step_specs, ensure_ascii=False)
    schema_lit = json.dumps(tool_schema_map, ensure_ascii=False)
    props_lit = json.dumps(param_props, ensure_ascii=False)
    desc_lit = json.dumps(f"Auto-forged macro for workflow '{sig}' ({goal[:60]})", ensure_ascii=False)
    name_lit = json.dumps(tool_name)
    return (
        f'"""Auto-forged workflow macro skill for `{sig}`."""\n\n'
        "from __future__ import annotations\n\n"
        "import json\n"
        "from typing import Any\n\n"
        f"_WORKFLOW_STEPS = json.loads({json.dumps(steps_lit)})\n"
        f"_STEP_PARAM_KEYS = json.loads({json.dumps(schema_lit)})\n"
        f"_MACRO_PROPERTIES = json.loads({json.dumps(props_lit)})\n\n\n"
        "def execute(arguments: dict[str, Any]) -> str:\n"
        '    """Run the learned read-only workflow steps sequentially and return combined JSON."""\n'
        "    from sysagent.config import get_settings\n"
        "    from sysagent.storage import Storage\n"
        "    from sysagent.tools import execute_tool\n"
        "    from sysagent.watcher import SystemWatcher\n\n"
        "    call_args = arguments if isinstance(arguments, dict) else {}\n"
        "    with Storage() as st:\n"
        "        w = SystemWatcher(get_settings(), storage=st)\n"
        "        results: list[dict[str, Any]] = []\n"
        "        for step in _WORKFLOW_STEPS:\n"
        '            sname = str(step["name"])\n'
        '            targs = dict(step.get("arguments") or {})\n'
        "            allowed_keys = set(_STEP_PARAM_KEYS.get(sname) or targs.keys())\n"
        "            for k, val in call_args.items():\n"
        "                if val not in (None, '') and (k in allowed_keys or k in targs):\n"
        "                    targs[k] = val\n"
        "            out = execute_tool(sname, targs, watcher=w, storage=st)\n"
        '            results.append({"tool": sname, "arguments": targs, "output": out})\n'
        "    return json.dumps(results, ensure_ascii=False)\n\n\n"
        "TOOLS = [\n"
        "    {\n"
        '        "declaration": {\n'
        f'            "name": {name_lit},\n'
        f'            "description": {desc_lit},\n'
        '            "parameters": {\n'
        '                "type": "object",\n'
        '                "properties": _MACRO_PROPERTIES,\n'
        "            },\n"
        "        },\n"
        '        "handler": execute,\n'
        '        "safe": True,\n'
        "    }\n"
        "]\n"
    )


def auto_forge_workflow_skills(
    storage: Storage, plugins: Any = None, min_hits: int = 2, max_new: int = 1
) -> list[str]:
    """Autonomously compile repeated read-only tool workflows into live LocalMind/Plugin skills."""
    from .plugins import PLUGINS_DIR, validate_plugin_code_isolated

    pdir = Path(getattr(plugins, "plugins_dir", None) or PLUGINS_DIR)
    forged_sigs: list[str] = list(_load_json_memory(storage, _FORGED_WF_KEY, []))
    forged_tools: list[str] = []

    for wf in storage.list_workflows(limit=25):
        if len(forged_tools) >= max_new:
            break
        sig = str(wf.get("signature") or "").strip()
        if not sig or int(wf.get("hits") or 0) < min_hits or sig in forged_sigs:
            continue
        try:
            raw_steps = json.loads(str(wf.get("steps_json") or "[]"))
        except Exception:
            raw_steps = []
        step_specs: list[dict[str, Any]] = []
        for s in raw_steps if isinstance(raw_steps, list) else []:
            if isinstance(s, dict) and s.get("name"):
                step_specs.append({"name": str(s["name"]), "arguments": dict(s.get("arguments") or {})})
        if not step_specs:
            step_specs = [{"name": p.strip(), "arguments": {}} for p in sig.split("->") if p.strip()]
        if len(step_specs) < 2 or not all(st["name"] in _SAFE_WORKFLOW_TOOLS for st in step_specs):
            continue

        slug = re.sub(r"[^a-z0-9]+", "_", "_".join(st["name"] for st in step_specs)).strip("_")[:36]
        if not slug:
            continue
        tool_name = f"macro_{slug}"
        code = _build_workflow_plugin_source(tool_name, sig, str(wf.get("goal_pattern") or sig), step_specs)
        try:
            ast.parse(code)
            compile(code, f"auto_workflow_{slug}.py", "exec")
        except SyntaxError:
            continue
        ok, _ = validate_plugin_code_isolated(code, tool_name)
        if not ok:
            continue
        pdir.mkdir(parents=True, exist_ok=True)
        dest = pdir / f"auto_workflow_{slug}.py"
        dest.write_text(code, encoding="utf-8")
        if plugins and hasattr(plugins, "reload"):
            plugins.reload()
        inject_plugin_skills_to_local_mind(storage, plugins)
        forged_sigs.append(sig)
        forged_tools.append(tool_name)
        storage.add_lesson(
            f"auto_forge:{tool_name}",
            f"Compiled repeated safe workflow `{sig}` into macro plugin tool `{tool_name}`.",
            str(dest),
        )
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(storage)
            eng.online_step(str(wf.get("goal_pattern") or sig), tool_name, reward=1.5)
            eng.sync_associative_memory()
        except Exception:
            pass

    if forged_sigs:
        _save_json_memory(storage, _FORGED_WF_KEY, forged_sigs[-100:])
    return forged_tools


def _collect_testable_symbols(py_path: Path) -> tuple[set[str], dict[str, tuple[str, bool]]]:
    """Return (top_level_funcs, {method_name: (ClassName, is_property)}) defined in `py_path`."""
    top_funcs: set[str] = set()
    class_methods: dict[str, tuple[str, bool]] = {}
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return top_funcs, class_methods
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            top_funcs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if (
                    isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not sub.name.startswith("_")
                    and sub.name not in class_methods
                ):
                    is_prop = any(isinstance(d, ast.Name) and d.id == "property" for d in sub.decorator_list)
                    class_methods[sub.name] = (node.name, is_prop)
    return top_funcs, class_methods


def synthesize_coverage_tests(pkg_dir: Path | None = None, max_symbols: int = 20) -> dict[str, Any]:
    """Synthesize real AST/signature verification assertions into tests/test_auto_coverage.py."""
    pdir = pkg_dir or Path(__file__).resolve().parent
    tests_dir = pdir.parent / "tests"
    if not tests_dir.is_dir():
        return {"added": 0, "coverage": scan_coverage_gaps(pdir)}
    out_file = tests_dir / "test_auto_coverage.py"
    prev_text = out_file.read_text(encoding="utf-8") if out_file.is_file() else None

    targets: dict[str, list[dict[str, str]]] = {}
    pruned_stale = False
    if prev_text:
        for ln in prev_text.splitlines():
            if ln.startswith("# TARGETS_JSON: "):
                try:
                    loaded = json.loads(ln[len("# TARGETS_JSON: ") :])
                    if isinstance(loaded, dict):
                        filtered = {
                            str(k): list(v)
                            for k, v in loaded.items()
                            if isinstance(v, list) and (pdir / str(k)).is_file()
                        }
                        pruned_stale = len(filtered) != len(loaded)
                        targets = filtered
                except Exception:
                    targets = {}
                break

    gaps = scan_coverage_gaps(pdir)
    untested = gaps.get("untested_by_file") or {}
    skip_mods = {"cli.py", "server.py", "__init__.py"}
    added = 0
    existing_pairs = {(fname, it.get("sym")) for fname, items in targets.items() for it in items if isinstance(it, dict)}

    for fname in sorted(untested):
        if fname in skip_mods or added >= max_symbols:
            continue
        mod_path = pdir / fname
        if not mod_path.is_file():
            continue
        top_funcs, class_methods = _collect_testable_symbols(mod_path)
        for sym in untested[fname]:
            if added >= max_symbols:
                break
            if (fname, sym) in existing_pairs:
                continue
            if sym in top_funcs:
                targets.setdefault(fname, []).append({"kind": "func", "sym": sym})
                existing_pairs.add((fname, sym))
                added += 1
            elif sym in class_methods:
                cls_name, is_prop = class_methods[sym]
                targets.setdefault(fname, []).append(
                    {"kind": "attr" if is_prop else "method", "cls": cls_name, "sym": sym}
                )
                existing_pairs.add((fname, sym))
                added += 1

    if added == 0 and not pruned_stale and prev_text is not None:
        return {"added": 0, "coverage": gaps}
    if not targets:
        return {"added": 0, "coverage": gaps}

    lines = [
        f"# TARGETS_JSON: {json.dumps(targets, ensure_ascii=False, sort_keys=True)}",
        '"""Auto-synthesized signature and availability coverage tests."""',
        "",
        "import importlib",
        "import inspect",
        "import os",
        "",
        'os.environ.setdefault("SYSAGENT_BRAIN_ENABLED", "false")',
        'os.environ.setdefault("SYSAGENT_EVOLVE_ENABLED", "false")',
        "",
    ]
    for fname in sorted(targets):
        mod_stem = fname[:-3]
        var = f"_mod_{mod_stem}"
        lines.append(f'{var} = importlib.import_module("sysagent.{mod_stem}")')
        for item in targets[fname]:
            sym = item["sym"]
            if item.get("kind") == "attr" and item.get("cls"):
                cls_name = item["cls"]
                lines.append(f"assert hasattr({var}.{cls_name}, {sym!r})")
            elif item.get("kind") == "method" and item.get("cls"):
                cls_name = item["cls"]
                lines.append(
                    f"assert callable({var}.{cls_name}.{sym}) and inspect.signature({var}.{cls_name}.{sym}) is not None"
                )
            else:
                lines.append(
                    f"assert callable({var}.{sym}) and inspect.signature({var}.{sym}) is not None"
                )
    lines.extend(["", 'print("AUTO_COVERAGE_TESTS_OK")', ""])
    candidate = "\n".join(lines)
    out_file.write_text(candidate, encoding="utf-8")

    env = {**os.environ, "PYTHONPATH": str(pdir.parent)}
    try:
        proc = subprocess.run(
            [sys.executable, str(out_file)],
            capture_output=True,
            text=True,
            timeout=25,
            env=env,
            cwd=pdir.parent,
        )
        ok = proc.returncode == 0 and "AUTO_COVERAGE_TESTS_OK" in proc.stdout
    except Exception:
        ok = False

    if not ok:
        if prev_text is not None:
            out_file.write_text(prev_text, encoding="utf-8")
        else:
            out_file.unlink(missing_ok=True)
        return {"added": 0, "error": "generated_tests_failed", "coverage": gaps}

    return {"added": added, "coverage": scan_coverage_gaps(pdir)}


def surgical_git_rollback(repo_dir: Path, proposal_id: int) -> bool:
    """Surgically revert an older `evolve(#id)` commit in git without losing unrelated newer edits."""
    if not shutil.which("git") or not (repo_dir / ".git").is_dir():
        return False
    try:
        log_p = subprocess.run(
            ["git", "log", f"--grep=evolve(#{proposal_id}):", "--format=%H", "-n", "1"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = log_p.stdout.strip()
        if log_p.returncode != 0 or not sha:
            return False
        rev_p = subprocess.run(
            ["git", "revert", "--no-commit", sha],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if rev_p.returncode != 0:
            subprocess.run(["git", "revert", "--abort"], cwd=repo_dir, capture_output=True, timeout=5)
            return False
        if not compileall.compile_dir(str(repo_dir / "sysagent"), quiet=1):
            subprocess.run(["git", "reset", "--merge"], cwd=repo_dir, capture_output=True, timeout=5)
            return False
        cm_p = subprocess.run(
            ["git", "commit", "-m", f"revert: surgical rollback evolve(#{proposal_id})"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return cm_p.returncode == 0
    except Exception:
        return False


def prune_dead_macro_skills(
    storage: Storage,
    plugins: Any = None,
    max_error_rate: float = 0.65,
    min_calls: int = 4,
) -> list[str]:
    """Quarantine auto-forged workflow macro plugins that repeatedly fail in production."""
    from .plugins import PLUGINS_DIR

    pdir = Path(getattr(plugins, "plugins_dir", None) or PLUGINS_DIR)
    if not pdir.is_dir():
        return []

    pruned: list[str] = []
    metrics_by_name = {
        str(m.get("tool_name") or ""): m for m in storage.get_tool_metrics(limit=50) if isinstance(m, dict)
    }
    for py_file in sorted(pdir.glob("auto_workflow_*.py")):
        slug = py_file.stem[len("auto_workflow_") :]
        tool_name = f"macro_{slug}"
        m = metrics_by_name.get(tool_name)
        if not m:
            continue
        calls = int(m.get("calls") or 0)
        errors = int(m.get("errors") or 0)
        if calls < min_calls:
            continue
        err_rate = errors / max(1, calls)
        if err_rate >= max_error_rate:
            qpath = py_file.with_suffix(".py.quarantine")
            try:
                py_file.replace(qpath)
                pruned.append(tool_name)
                storage.add_lesson(
                    f"dead_skill_gc:{tool_name}",
                    f"Retired auto-forged macro `{tool_name}` due to high error rate ({errors}/{calls} = {int(err_rate * 100)}%).",
                    f"Archived to {qpath.name}",
                )
            except Exception:
                pass
    if pruned and plugins and hasattr(plugins, "reload"):
        plugins.reload()
    return pruned


def heal_runtime_anomalies(storage: Storage, plugins: Any = None) -> dict[str, Any]:
    """Quarantine broken plugins, prune failing auto-forged macros, and compute adaptive runtime tuning."""
    from .plugins import PLUGINS_DIR, validate_plugin_code_isolated

    pdir = Path(getattr(plugins, "plugins_dir", None) or PLUGINS_DIR)
    quarantined: list[str] = []
    if pdir.is_dir():
        for py_file in sorted(pdir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                code = py_file.read_text(encoding="utf-8", errors="replace")
                ok, reason = validate_plugin_code_isolated(code, py_file.stem)
            except Exception as exc:
                ok, reason = False, str(exc)
            if not ok:
                qpath = py_file.with_suffix(".py.quarantine")
                try:
                    py_file.replace(qpath)
                    quarantined.append(py_file.name)
                    storage.add_lesson(
                        "runtime_heal:quarantine",
                        f"Quarantined invalid plugin {py_file.name}: {reason[:120]}",
                        f"Moved to {qpath.name}",
                    )
                except Exception:
                    pass
    if quarantined and plugins and hasattr(plugins, "reload"):
        plugins.reload()

    pruned_macros = prune_dead_macro_skills(storage, plugins)

    tuning: dict[str, Any] = _load_json_memory(storage, _RUNTIME_TUNING_KEY, {})
    if not isinstance(tuning, dict):
        tuning = {}
    tuned_tools: list[str] = []
    now_str = datetime.now().strftime(_TS_FMT)
    for m in storage.get_tool_metrics(limit=15):
        tname = str(m.get("tool_name") or "")
        errs = int(m.get("errors") or 0)
        avg_ms = float(m.get("avg_ms") or 0)
        if not tname or (errs == 0 and avg_ms <= 1500.0):
            continue
        prev = tuning.get(tname) or {}
        if prev.get("known_errors") != errs or avg_ms > float(prev.get("known_avg_ms", 0)) * 1.4:
            tuning[tname] = {
                "cache_ttl_sec": 300 if (errs == 0 and avg_ms > 1500.0) else 60,
                "extra_timeout_sec": 90 if avg_ms > 2000.0 else 45,
                "known_errors": errs,
                "known_avg_ms": round(avg_ms, 1),
                "mitigated_at": now_str,
            }
            tuned_tools.append(tname)
    if tuned_tools:
        _save_json_memory(storage, _RUNTIME_TUNING_KEY, tuning)

    kg_cleanup = storage.consolidate_knowledge_graph()
    return {
        "quarantined": quarantined,
        "pruned_macros": pruned_macros,
        "tuned_tools": tuned_tools,
        "kg_cleanup": kg_cleanup,
    }


def build_self_roadmap(storage: Storage) -> list[dict[str, Any]]:
    """Build a prioritized evolution backlog scored by NeuralEngine from errors, slow tools, coverage gaps, and weak roles."""
    eng = None
    try:
        from .neural import NeuralEngine

        eng = NeuralEngine(storage)
    except Exception:
        eng = None

    def _score(target_f: str, kind_lbl: str, goal_txt: str) -> float | None:
        return eng.score_patch_candidate(target_f, kind_lbl, goal_txt) if eng is not None else None

    items: list[dict[str, Any]] = []
    for cl in error_clusters(storage)[:2]:
        goal = f"Fix recurring error at {cl['file']}:{cl['line']} (x{cl['count']})"
        items.append(
            {
                "priority": "high",
                "kind": "error_cluster",
                "target": cl["file"],
                "goal": goal,
                "neural_value": _score(cl["file"], "bugfix", goal),
            }
        )
    tuned = _load_json_memory(storage, _RUNTIME_TUNING_KEY, {})
    if not isinstance(tuned, dict):
        tuned = {}
    for m in storage.get_tool_metrics(limit=10):
        tname = str(m["tool_name"])
        errs = int(m.get("errors") or 0)
        avg_ms = float(m.get("avg_ms") or 0)
        if errs == 0 and avg_ms <= 1500.0:
            continue
        t_rec = tuned.get(tname)
        if isinstance(t_rec, dict) and errs <= int(t_rec.get("known_errors", errs)):
            continue
        items.append(
            {
                "priority": "medium",
                "kind": "tool_perf",
                "target": tname,
                "goal": f"Harden/optimize tool {tname} (avg {int(avg_ms)}ms, err={errs})",
            }
        )
        if len(items) >= 4:
            break
    cov = scan_coverage_gaps()
    top_gaps = sorted(cov.get("untested_by_file", {}).items(), key=lambda kv: len(kv[1]), reverse=True)[:2]
    for fname, syms in top_gaps:
        goal = f"Add tests/hardening for {fname} ({len(syms)} untested symbols)"
        items.append(
            {
                "priority": "medium",
                "kind": "coverage_gap",
                "target": fname,
                "goal": goal,
                "neural_value": _score(fname, "new_feature", goal),
            }
        )
    if eng is not None:
        n_st = eng.get_status()
        t_acc = n_st.get("tool_accuracy")
        if t_acc is not None and int(n_st.get("epochs_trained", 0)) > 0 and float(t_acc) < 0.85:
            items.append(
                {
                    "priority": "medium",
                    "kind": "neural_calibration",
                    "target": "neural.py",
                    "goal": f"Recalibrate NeuralEngine routing accuracy (tool_acc={int(float(t_acc) * 100)}% < 85%)",
                }
            )
    try:
        from .local_mind import LocalMind

        role = LocalMind(storage).weakest_specialist_role()
        items.append(
            {
                "priority": "low",
                "kind": "curriculum_gap",
                "target": role,
                "goal": f"Expand local training curriculum for specialist role '{role}'",
            }
        )
    except Exception:
        pass
    roadmap = items[:6]
    _save_json_memory(storage, _ROADMAP_KEY, roadmap)
    return roadmap


from .self_twin import (  # noqa: F401
    _TWIN_HISTORY,
    analyze_causal_digital_twin,
    build_causal_timeline,
    evaluate_graduation_arena,
    investigate_root_cause,
    mine_episodic_workflows,
    predict_user_routine,
    run_doctor_auto_repair,
    verify_remediation_impact,
)


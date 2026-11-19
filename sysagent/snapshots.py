"""Unified snapshot & rollback helpers for SysAgent."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DATA_DIR

_MAX_CONFIG_BACKUPS = 30


def _backup_dir() -> Path:
    bdir = DATA_DIR / "config_backups"
    bdir.mkdir(parents=True, exist_ok=True)
    return bdir


def save_config_snapshot(path: Path, old_content: str) -> str | None:
    """Save a timestamped config snapshot + metadata JSON before modifying a file."""
    try:
        bdir = _backup_dir()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = path.name.replace("/", "_")
        snap_id = f"{stamp}_{safe_name}"
        bak_path = bdir / f"{snap_id}.bak"
        meta_path = bdir / f"{snap_id}.meta.json"
        bak_path.write_text(old_content, encoding="utf-8")
        meta = {
            "id": snap_id,
            "path": str(path),
            "backup": str(bak_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        old_baks = sorted(bdir.glob("*.bak"))
        for stale in old_baks[:-_MAX_CONFIG_BACKUPS]:
            stale.unlink(missing_ok=True)
            stale_meta = stale.with_name(f"{stale.stem}.meta.json")
            stale_meta.unlink(missing_ok=True)
        return snap_id
    except Exception:
        return None


def list_config_snapshots(limit: int = 20) -> list[dict[str, Any]]:
    """Return newest-first list of config snapshots."""
    bdir = _backup_dir()
    items: list[dict[str, Any]] = []
    for meta_file in sorted(bdir.glob("*.meta.json"), reverse=True):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            data["restorable"] = Path(data.get("backup", "")).is_file()
            items.append(data)
            if len(items) >= limit:
                break
        except Exception:
            continue
    return items


def restore_config_snapshot(snapshot_id: str | None = None) -> dict[str, Any]:
    """Restore a saved config snapshot by ID (or the latest one if omitted)."""
    snapshots = list_config_snapshots(limit=50)
    if not snapshots:
        return {"ok": False, "error": "No config snapshots available"}

    target = snapshots[0] if not snapshot_id else next(
        (s for s in snapshots if s.get("id") == snapshot_id), None
    )
    if not target:
        return {"ok": False, "error": f"Snapshot not found: {snapshot_id}"}

    bak_path = Path(str(target.get("backup", "")))
    dest_path = Path(str(target.get("path", "")))
    if not bak_path.is_file():
        return {"ok": False, "error": f"Backup file missing: {bak_path}"}

    content = bak_path.read_text(encoding="utf-8", errors="replace")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(content, encoding="utf-8")
    return {"ok": True, "target": "config", "id": target["id"], "path": str(dest_path)}


def unified_snapshots(storage: Any, limit: int = 15) -> dict[str, Any]:
    """Collect audit log entries, evolve proposals, and config snapshots."""
    return {
        "audit_log": storage.list_audit_log(limit=limit),
        "evolve_proposals": storage.list_evolve_proposals(limit=limit),
        "config_snapshots": list_config_snapshots(limit=limit),
    }


def perform_rollback(
    target: str,
    target_id: str | None,
    storage: Any,
    watcher: Any = None,
    plugins: Any = None,
    evolver: Any = None,
) -> dict[str, Any]:
    """Execute rollback for auto, file/batch, evolve, or config targets."""
    mode = (target or "auto").strip().lower()

    if mode == "auto":
        recent = storage.list_audit_log(limit=10)
        latest_active = next((r for r in recent if not r.get("undone")), None)
        if latest_active and latest_active.get("action") == "evolve" and evolver is not None:
            mode = "evolve"
        elif latest_active:
            mode = "file"
        elif list_config_snapshots(limit=1):
            mode = "config"
        elif evolver is not None:
            mode = "evolve"
        else:
            return {"ok": False, "error": "Nothing to roll back"}

    if mode == "evolve":
        if evolver is None:
            return {"ok": False, "error": "Evolver unavailable (daemon must be running)"}
        pid = int(target_id) if target_id and str(target_id).isdigit() else None
        ok = evolver.rollback_last_evolve(pid)
        return {"ok": bool(ok), "target": "evolve", "proposal_id": pid}

    if mode == "config":
        return restore_config_snapshot(target_id)

    if mode in {"file", "batch"}:
        from .tools import execute_tool

        raw = execute_tool(
            "undo_batch_actions",
            {"batch_id": target_id} if target_id else {},
            watcher,
            storage,
            plugins=plugins,
        )
        parsed = json.loads(raw) if isinstance(raw, str) else {"result": raw}
        if "error" in parsed:
            return {"ok": False, "target": "file", **parsed}
        return {"ok": True, "target": "file", **parsed}

    return {"ok": False, "error": f"Unknown rollback target: {target}"}

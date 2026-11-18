"""Safety and blast-radius filters for SysAgent self-evolution."""

from __future__ import annotations

import difflib


PROTECTED_CORE_MODULES = {
    "safety.py",
    "storage.py",
    "storage_brain.py",
    "evolve.py",
    "server.py",
    "agent.py",
}


def check_blast_radius(
    rel_name: str,
    original_content: str,
    new_content: str,
    protected_modules: set[str] | None = None,
) -> tuple[bool, str]:
    """Reject oversized deletions on core modules to prevent accidental gutting."""
    protected = protected_modules if protected_modules is not None else PROTECTED_CORE_MODULES
    orig_lines = original_content.splitlines()
    new_lines = new_content.splitlines()
    removed = sum(
        1
        for ln in difflib.unified_diff(orig_lines, new_lines, n=0)
        if ln.startswith("-") and not ln.startswith("---")
    )
    limit = (
        max(40, int(len(orig_lines) * 0.25))
        if rel_name in protected
        else max(80, int(len(orig_lines) * 0.45))
    )
    if removed > limit:
        return False, f"Blast-radius guard: removed {removed} lines > limit {limit} in {rel_name}"
    return True, "ok"

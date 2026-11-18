"""Patch application and hot reloading logic for SysAgent self-evolution."""

from __future__ import annotations

import importlib
import sys


def hot_reload_module(rel_name: str) -> bool:
    """Hot-reload an already-imported sysagent submodule and rebind imported symbols in peer modules."""
    if rel_name == "server.py" or not rel_name.endswith(".py"):
        return False
    stem = rel_name[:-3]
    mod_name = "sysagent" if stem == "__init__" else f"sysagent.{stem}"
    mod = sys.modules.get(mod_name)
    if mod is None:
        return False
    try:
        old_attrs = dict(vars(mod))
        reloaded = importlib.reload(mod)
        new_attrs = vars(reloaded)
        for peer_name, peer_mod in list(sys.modules.items()):
            if not peer_name.startswith("sysagent.") or peer_name == mod_name or peer_mod is None:
                continue
            peer_dict = vars(peer_mod)
            for attr, old_val in old_attrs.items():
                if attr.startswith("__") or attr not in new_attrs:
                    continue
                if peer_dict.get(attr) is old_val and new_attrs[attr] is not old_val:
                    setattr(peer_mod, attr, new_attrs[attr])
        return True
    except (ImportError, AttributeError, SyntaxError, TypeError):
        return False


def apply_patch_text(
    original: str,
    target: str | None,
    replacement: str | None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str | None:
    """Apply patch by exact substring, line-trimmed match, or 1-based line range."""
    if replacement is None:
        return None
    if target and target in original:
        return original.replace(target, replacement, 1)
    if target:
        orig_lines = original.splitlines(keepends=True)
        targ_lines = [ln.strip() for ln in target.splitlines() if ln.strip()]
        if targ_lines:
            for i in range(len(orig_lines) - len(targ_lines) + 1):
                window = [ln.strip() for ln in orig_lines[i : i + len(targ_lines)]]
                if window == targ_lines:
                    rep = replacement if replacement.endswith("\n") else replacement + "\n"
                    return "".join(orig_lines[:i]) + rep + "".join(orig_lines[i + len(targ_lines) :])
    if isinstance(start_line, int) and isinstance(end_line, int) and 1 <= start_line <= end_line:
        orig_lines = original.splitlines(keepends=True)
        if end_line <= len(orig_lines):
            rep = replacement if replacement.endswith("\n") else replacement + "\n"
            return "".join(orig_lines[: start_line - 1]) + rep + "".join(orig_lines[end_line:])
    return None

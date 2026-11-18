"""AST analysis and static signature verification for SysAgent self-evolution."""

from __future__ import annotations

import ast
import re
import string
from typing import Any

from .i18n import TRANSLATIONS
from .prompts import DEFAULT_PROMPTS


def format_func_sig(node: Any) -> str:
    """Render a compact function signature `name(a, b=..., **kw)` from an AST FunctionDef."""
    parts: list[str] = []
    args = node.args
    posonly = getattr(args, "posonlyargs", [])
    regular = list(posonly) + list(args.args)
    if regular and regular[0].arg in {"self", "cls"}:
        regular = regular[1:]
    num_defaults = len(args.defaults)
    default_start = len(regular) - num_defaults
    for idx, arg in enumerate(regular):
        if idx >= default_start:
            parts.append(f"{arg.arg}=...")
        else:
            parts.append(arg.arg)
    if posonly:
        parts.append("/")
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for kwarg, kwdef in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        parts.append(f"{kwarg.arg}=..." if kwdef is not None else kwarg.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return f"{node.name}({', '.join(parts)})"


def _find_async_blocking_calls(async_node: ast.AsyncFunctionDef) -> list[tuple[int, str]]:
    """Find synchronous blocking calls executed directly inside an async def body."""
    findings: list[tuple[int, str]] = []

    def _visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not async_node:
            return
        if isinstance(node, ast.Call):
            cname = ""
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                cname = f"{node.func.value.id}.{node.func.attr}"
            elif isinstance(node.func, ast.Name):
                cname = node.func.id
            if cname in {"subprocess.run", "subprocess.check_output", "time.sleep", "urlopen"}:
                findings.append((node.lineno, cname))
        for child in ast.iter_child_nodes(node):
            _visit(child)

    for stmt in async_node.body:
        _visit(stmt)
    return findings


def module_outline(
    src: str,
    global_sigs: dict[str, Any] | None = None,
    method_sigs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cheap local structural summary and zero-token AST + signature + template pre-audit of a module."""
    symbols: list[str] = []
    ast_findings: list[str] = []
    try:
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(format_func_sig(node))
            elif isinstance(node, ast.ClassDef):
                methods = [
                    format_func_sig(m)
                    for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                symbols.append(f"{node.name}[{', '.join(methods[:6])}]")
        for n in ast.walk(tree):
            if isinstance(n, ast.AsyncFunctionDef):
                for b_line, b_name in _find_async_blocking_calls(n):
                    ast_findings.append(f"BUG:async_blocking_call:L{b_line}:{b_name}")
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(n, "end_lineno", n.lineno)
                if end - n.lineno > 80:
                    ast_findings.append(f"long_func:{n.name}({end - n.lineno}L)")
            elif isinstance(n, ast.ExceptHandler):
                if n.type is None:
                    ast_findings.append(f"bare_except:L{n.lineno}")
                elif (
                    isinstance(n.type, ast.Name)
                    and n.type.id == "Exception"
                    and len(n.body) == 1
                    and isinstance(n.body[0], ast.Pass)
                ):
                    ast_findings.append(f"silent_pass:L{n.lineno}")
            elif isinstance(n, ast.Call):
                has_star = any(isinstance(a, ast.Starred) for a in n.args)
                has_dstar = any(kw.arg is None for kw in n.keywords)
                fn_name = n.func.id if isinstance(n.func, ast.Name) else None
                attr_name = n.func.attr if isinstance(n.func, ast.Attribute) else None

                if not has_star and not has_dstar:
                    sig = None
                    target_lbl = ""
                    if fn_name and global_sigs and fn_name in global_sigs:
                        sig = global_sigs[fn_name]
                        target_lbl = fn_name
                    elif attr_name and method_sigs and isinstance(n.func, ast.Attribute):
                        recv = n.func.value
                        recv_name = (
                            recv.id
                            if isinstance(recv, ast.Name)
                            else (
                                recv.attr
                                if isinstance(recv, ast.Attribute)
                                and isinstance(recv.value, ast.Name)
                                and recv.value.id == "self"
                                else None
                            )
                        )
                        if recv_name in {"storage", "watcher", "auth", "agent", "evolver", "brain", "registry", "plugins"}:
                            sig = method_sigs.get(f"{recv_name}.{attr_name}") or method_sigs.get(attr_name)
                            target_lbl = f"{recv_name}.{attr_name}"
                    if sig is not None:
                        pos_vals = [None] * len(n.args)
                        kw_vals = {kw.arg: None for kw in n.keywords if kw.arg}
                        try:
                            sig.bind_partial(*pos_vals, **kw_vals)
                        except TypeError as exc:
                            ast_findings.append(f"BUG:call_sig:L{n.lineno}:{target_lbl}:{exc}")

                if fn_name == "t" and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                    t_key = n.args[0].value
                    tmpl = TRANSLATIONS["en"].get(t_key) or TRANSLATIONS["tr"].get(t_key)
                    if tmpl is None:
                        ast_findings.append(f"BUG:i18n_missing_key:L{n.lineno}:{t_key}")
                    elif not has_dstar:
                        req_fields = {
                            fname
                            for _, fname, _, _ in string.Formatter().parse(tmpl)
                            if fname
                        }
                        passed_kw = {kw.arg for kw in n.keywords if kw.arg}
                        missing_kw = req_fields - passed_kw
                        if missing_kw:
                            ast_findings.append(
                                f"BUG:i18n_missing_vars:L{n.lineno}:{t_key}:{sorted(missing_kw)}"
                            )

                if attr_name == "render" and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                    p_key = n.args[0].value
                    if p_key in DEFAULT_PROMPTS and not has_dstar:
                        req_vars = set(DEFAULT_PROMPTS[p_key]["required_vars"])
                        passed_kw = {kw.arg for kw in n.keywords if kw.arg}
                        missing_v = req_vars - passed_kw
                        if missing_v:
                            ast_findings.append(
                                f"BUG:prompt_missing_vars:L{n.lineno}:{p_key}:{sorted(missing_v)}"
                            )
            elif (
                isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == "split"
                and len(n.value.args) == 0
                and isinstance(n.slice, ast.Constant)
                and n.slice.value == 0
            ):
                src_lines = src.splitlines()
                ctx_window = "\n".join(src_lines[max(0, n.lineno - 8) : n.lineno])
                if not any(g in ctx_window for g in ("if ", "try:", ".split()", ".strip()")):
                    ast_findings.append(f"BUG:unsafe_split_index:L{n.lineno}:unguarded .split()[0] can raise IndexError")
    except SyntaxError as exc:
        symbols.append(f"<syntax error line {exc.lineno}>")
    return {"lines": src.count("\n") + 1, "symbols": symbols, "ast_findings": ast_findings}


def extract_ast_focus_slice(
    src: str,
    why: str = "",
    ast_findings: list[str] | None = None,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """Slice a large module to the target function/class AST block with imports + surrounding context."""
    total_lines = src.count("\n") + 1
    if len(src) <= max_chars:
        return {
            "sliced": False,
            "source": src,
            "focus_symbol": None,
            "start_line": 1,
            "end_line": total_lines,
        }

    lines = src.splitlines(keepends=True)
    hint_blob = f"{why or ''}\n" + "\n".join(ast_findings or [])
    line_hints = [int(m) for m in re.findall(r"(?:\bL|\bline\s+)(\d+)", hint_blob) if m.isdigit()]
    sym_hints = re.findall(r"long_func:([A-Za-z_]\w*)", hint_blob)

    try:
        tree = ast.parse(src)
    except SyntaxError:
        s_line = max(1, (line_hints[0] - 40) if line_hints else 1)
        snippet = "".join(lines[s_line - 1 :])[:max_chars]
        e_line = s_line + snippet.count("\n")
        return {
            "sliced": True,
            "source": f"# Slice (lines {s_line}-{e_line} of {len(lines)})\n{snippet}",
            "focus_symbol": None,
            "start_line": s_line,
            "end_line": e_line,
        }

    import_lines: list[str] = []
    defs: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = "".join(lines[node.lineno - 1 : (getattr(node, "end_lineno", node.lineno))])
            import_lines.append(seg.rstrip("\n"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append(node)
        elif isinstance(node, ast.ClassDef):
            defs.append(node)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs.append(sub)

    target_node: ast.AST | None = None
    if line_hints:
        covering: list[tuple[int, ast.AST]] = []
        for ln in line_hints:
            for d in defs:
                d_start = getattr(d, "lineno", 1)
                d_end = getattr(d, "end_lineno", d_start)
                if d_start <= ln <= d_end:
                    covering.append((d_end - d_start, d))
        if covering:
            covering.sort(key=lambda item: item[0])
            target_node = covering[0][1]

    if target_node is None:
        for d in defs:
            d_name = getattr(d, "name", "")
            if not d_name:
                continue
            if d_name in sym_hints or (why and re.search(rf"\b{re.escape(d_name)}\b", why)):
                target_node = d
                break

    if target_node is None:
        funcs = [d for d in defs if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if funcs:
            target_node = max(
                funcs,
                key=lambda d: getattr(d, "end_lineno", d.lineno) - getattr(d, "lineno", 1),
            )

    if target_node is None:
        snippet = src[:max_chars]
        return {
            "sliced": True,
            "source": snippet,
            "focus_symbol": None,
            "start_line": 1,
            "end_line": snippet.count("\n") + 1,
        }

    raw_start = getattr(target_node, "lineno", 1)
    raw_end = getattr(target_node, "end_lineno", raw_start)
    s_line = max(1, raw_start - 15)
    e_line = min(len(lines), raw_end + 15)
    body = "".join(lines[s_line - 1 : e_line])
    header_imports = "\n".join(import_lines[:35])
    meta = f"# AST Focus Slice: {getattr(target_node, 'name', '?')} (lines {s_line}-{e_line} of {len(lines)})\n"
    combined = f"{meta}{header_imports}\n\n# --- Lines {s_line}..{e_line} ---\n{body}"
    if len(combined) > max_chars:
        budget = max(500, max_chars - len(meta) - len(header_imports) - 64)
        body = body[:budget]
        e_line = s_line + body.count("\n")
        combined = f"{meta}{header_imports}\n\n# --- Lines {s_line}..{e_line} ---\n{body}"
    return {
        "sliced": True,
        "source": combined,
        "focus_symbol": getattr(target_node, "name", None),
        "start_line": s_line,
        "end_line": e_line,
    }

"""User plugin loader for SysAgent (~/.config/sysagent/plugins/*.py)."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable

from .config import CONFIG_DIR


PLUGINS_DIR = CONFIG_DIR / "plugins"
MCP_CONFIG_FILE = CONFIG_DIR / "mcp.json"


def validate_plugin_code_isolated(
    python_code: str, tool_name: str = "plugin", timeout_sec: int = 8
) -> tuple[bool, str]:
    """Compile and smoke-test plugin code inside an isolated subprocess before loading it into the daemon."""
    try:
        compile(python_code, f"{tool_name}.py", "exec")
    except SyntaxError as exc:
        return False, f"Syntax error in plugin code: {exc}"

    with tempfile.TemporaryDirectory() as td:
        mod_path = Path(td) / f"{tool_name}.py"
        mod_path.write_text(python_code, encoding="utf-8")
        runner = (
            "import importlib.util, json, sys\n"
            f"spec = importlib.util.spec_from_file_location('sandbox_plugin', r'{mod_path}')\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "tools = getattr(mod, 'TOOLS', None)\n"
            "if not isinstance(tools, list) or not tools:\n"
            "    print('ERR:python_code must define a non-empty TOOLS list.')\n"
            "    sys.exit(2)\n"
            "names = []\n"
            "for entry in tools:\n"
            "    if not isinstance(entry, dict) or not isinstance(entry.get('declaration'), dict) "
            "or not entry['declaration'].get('name') or not callable(entry.get('handler')):\n"
            "        print('ERR:Each TOOLS entry must have declaration.name and callable handler.')\n"
            "        sys.exit(2)\n"
            "    names.append(entry['declaration']['name'])\n"
            "print('OK:' + ','.join(names))\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", runner],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return False, f"Plugin validation timed out after {timeout_sec}s"
        out = (proc.stdout + "\n" + proc.stderr).strip()
        if proc.returncode != 0 or not proc.stdout.strip().startswith("OK:"):
            for line in out.splitlines():
                if line.startswith("ERR:"):
                    return False, line[4:]
            return False, f"Plugin isolated validation failed: {out[-400:]}"
        return True, proc.stdout.strip()[3:]


@dataclass
class LoadedPluginTool:
    name: str
    declaration: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    safe: bool = False


class PluginRegistry:
    """Discovers and executes custom Python tools from ~/.config/sysagent/plugins."""

    def __init__(
        self, plugins_dir: Path | None = None, mcp_config_file: Path | None = None
    ) -> None:
        self.plugins_dir = plugins_dir or PLUGINS_DIR
        self.mcp_config_file = mcp_config_file or MCP_CONFIG_FILE
        self.tools: dict[str, LoadedPluginTool] = {}
        self.reload()

    def reload(self) -> int:
        self.tools.clear()
        if self.plugins_dir.is_dir():
            for py_file in sorted(self.plugins_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"sysagent_plugin_{py_file.stem}", py_file
                    )
                    if not spec or not spec.loader:
                        continue
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    entries = getattr(mod, "TOOLS", [])
                    for entry in entries:
                        decl = dict(entry["declaration"])
                        decl.setdefault("type", "function")
                        name = decl["name"]
                        self.tools[name] = LoadedPluginTool(
                            name=name,
                            declaration=decl,
                            handler=entry["handler"],
                            safe=bool(entry.get("safe", False)),
                        )
                except Exception:
                    continue
        self._load_mcp_servers()
        return len(self.tools)

    @staticmethod
    def _mcp_rpc(
        command: list[str],
        env: dict[str, str] | None,
        requests: list[dict[str, Any]],
        timeout_sec: float = 8.0,
    ) -> list[dict[str, Any]]:
        """Execute one-shot JSON-RPC 2.0 requests against a stdio MCP server."""
        proc_env = os.environ.copy()
        if env:
            proc_env.update({str(k): str(v) for k, v in env.items()})
        stdin_payload = "\n".join(json.dumps(r) for r in requests) + "\n"
        proc = subprocess.run(
            command,
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=proc_env,
        )
        responses: list[dict[str, Any]] = []
        for raw_line in (proc.stdout or "").splitlines():
            line = raw_line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    responses.append(obj)
            except json.JSONDecodeError:
                continue
        return responses

    def _load_mcp_servers(self) -> None:
        """Discover and register tools from ~/.config/sysagent/mcp.json stdio MCP servers."""
        if not self.mcp_config_file.is_file():
            return
        try:
            cfg = json.loads(self.mcp_config_file.read_text(encoding="utf-8"))
        except Exception:
            return
        servers = cfg.get("mcpServers") if isinstance(cfg, dict) else None
        if not isinstance(servers, dict):
            return
        for srv_name, srv_cfg in servers.items():
            if not isinstance(srv_cfg, dict) or not srv_cfg.get("command"):
                continue
            cmd = [str(srv_cfg["command"]), *[str(a) for a in (srv_cfg.get("args") or [])]]
            srv_env = srv_cfg.get("env") if isinstance(srv_cfg.get("env"), dict) else None
            srv_safe = bool(srv_cfg.get("safe", False))
            try:
                resps = self._mcp_rpc(
                    cmd,
                    srv_env,
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {},
                                "clientInfo": {"name": "sysagent", "version": "0.1.0"},
                            },
                        },
                        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                    ],
                )
                list_resp = next((r for r in resps if r.get("id") == 2), None)
                tools_raw = (list_resp or {}).get("result", {}).get("tools", [])
                for t_spec in tools_raw:
                    raw_tname = str(t_spec.get("name") or "").strip()
                    if not raw_tname:
                        continue
                    full_name = f"mcp_{srv_name}_{raw_tname}"
                    decl = {
                        "type": "function",
                        "name": full_name,
                        "description": str(t_spec.get("description") or f"MCP tool {raw_tname} ({srv_name})"),
                        "parameters": t_spec.get("inputSchema") or {"type": "object", "properties": {}},
                    }

                    def _make_handler(c: list[str], e: dict[str, str] | None, tn: str) -> Callable[[dict[str, Any]], Any]:
                        def _handler(arguments: dict[str, Any]) -> Any:
                            call_resps = self._mcp_rpc(
                                c,
                                e,
                                [
                                    {
                                        "jsonrpc": "2.0",
                                        "id": 1,
                                        "method": "initialize",
                                        "params": {
                                            "protocolVersion": "2024-11-05",
                                            "capabilities": {},
                                            "clientInfo": {"name": "sysagent", "version": "0.1.0"},
                                        },
                                    },
                                    {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                                    {
                                        "jsonrpc": "2.0",
                                        "id": 2,
                                        "method": "tools/call",
                                        "params": {"name": tn, "arguments": arguments},
                                    },
                                ],
                                timeout_sec=20.0,
                            )
                            res = next((r for r in call_resps if r.get("id") == 2), None)
                            if not res:
                                return {"error": f"No response from MCP tool {tn}"}
                            return res.get("result", res)
                        return _handler

                    self.tools[full_name] = LoadedPluginTool(
                        name=full_name,
                        declaration=decl,
                        handler=_make_handler(cmd, srv_env, raw_tname),
                        safe=srv_safe,
                    )
            except Exception:
                continue

    def declarations(self) -> list[dict[str, Any]]:
        return [t.declaration for t in self.tools.values()]

    def is_safe(self, name: str) -> bool | None:
        tool = self.tools.get(name)
        return tool.safe if tool else None

    def execute(self, name: str, arguments: dict[str, Any], timeout_sec: float = 30.0) -> Any:
        tool = self.tools.get(name)
        if not tool:
            raise KeyError(f"Plugin tool not found: {name}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(tool.handler, arguments)
            try:
                return fut.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError:
                return json.dumps({"error": f"Plugin tool '{name}' timed out after {timeout_sec}s"})

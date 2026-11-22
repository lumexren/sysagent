"""Minimal zero-dependency MCP (Model Context Protocol) stdio server for SysAgent."""

from __future__ import annotations

import json
import sys
from typing import Any

import psutil

from .snapshots import perform_rollback, unified_snapshots
from .storage import Storage

MCP_PROTOCOL_VERSION = "2024-11-05"

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "sysagent_status",
        "description": "Get live Linux system telemetry (CPU, RAM, disk) and SysAgent local model summary.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "sysagent_search_memory",
        "description": "Search SysAgent memories, lessons, and knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "sysagent_brain_rules",
        "description": "List autonomous CognitiveBrain rules currently active in SysAgent.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "sysagent_snapshots",
        "description": "List recent file/audit, config, and self-evolve snapshots.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 10}},
        },
    },
    {
        "name": "sysagent_rollback",
        "description": "Roll back the latest or a specific file/config snapshot.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": ["auto", "file", "config"], "default": "auto"},
                "id": {"type": "string", "description": "Optional batch or snapshot ID"},
            },
        },
    },
    {
        "name": "sysagent_schedule_list",
        "description": "List scheduled SysAgent automations.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "sysagent_cortex_status",
        "description": "Get live NeuralBrainCortex status (Flow Guardian mode, predictive health, dream cycle, habit suggestions).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def _execute_mcp_tool(name: str, args: dict[str, Any], storage: Storage) -> dict[str, Any]:
    if name == "sysagent_status":
        from .local_mind import LocalMind

        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        lm = LocalMind(storage).get_status()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_percent": mem.percent,
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "local_backend": lm.get("local_backend"),
            "ensemble_active": lm.get("ensemble_active"),
            "brain_rules": len(storage.list_brain_rules()),
            "memories": len(storage.list_memories()),
        }
    if name == "sysagent_search_memory":
        q = str(args.get("query") or "").strip()
        lim = max(1, min(50, int(args.get("limit") or 10)))
        return {"query": q, "results": storage.search_knowledge(q, limit=lim) if q else []}
    if name == "sysagent_brain_rules":
        return {"rules": storage.list_brain_rules()}
    if name == "sysagent_snapshots":
        lim = max(1, min(50, int(args.get("limit") or 10)))
        return unified_snapshots(storage, limit=lim)
    if name == "sysagent_rollback":
        target = str(args.get("target") or "auto")
        tid = str(args["id"]) if args.get("id") is not None else None
        return perform_rollback(target, tid, storage)
    if name == "sysagent_schedule_list":
        return {"automations": storage.list_automations()}
    if name == "sysagent_cortex_status":
        from .config import get_settings
        from .cortex import NeuralBrainCortex

        return NeuralBrainCortex(get_settings(), storage).get_status()
    raise ValueError(f"Unknown MCP tool: {name}")


def handle_mcp_request(req: Any, storage: Storage) -> dict[str, Any] | None:
    """Process a single JSON-RPC 2.0 MCP message and return a response dict (or None for notifications)."""
    if not isinstance(req, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request"},
        }
    has_id = "id" in req
    req_id = req.get("id")
    method = str(req.get("method") or "")
    raw_params = req.get("params")
    params = raw_params if isinstance(raw_params, dict) else {}

    if method.startswith("notifications/") or not has_id:
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "sysagent", "version": "1.0.0"},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS}}
    if method == "tools/call":
        tname = str(params.get("name") or "")
        raw_args = params.get("arguments")
        targs = raw_args if isinstance(raw_args, dict) else {}
        try:
            payload = _execute_mcp_tool(tname, targs, storage)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def run_mcp_stdio() -> None:
    """Serve MCP over stdin/stdout line-delimited JSON-RPC 2.0."""
    with Storage() as storage:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
                sys.stdout.write(json.dumps(err, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                continue
            resp = handle_mcp_request(msg, storage)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()

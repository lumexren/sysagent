"""Unix Domain Socket IPC helpers for SysAgent CLI and daemon communication."""

from __future__ import annotations

from typing import Any

import httpx
import websockets

from .config import get_settings


def _resolve_uds(url_or_base: str | None) -> str | None:
    if not url_or_base or "sysagent.local" in url_or_base:
        return get_settings().socket_path
    if url_or_base.startswith("unix://"):
        return url_or_base[len("unix://") :]
    if url_or_base.startswith("/"):
        if url_or_base.endswith(".sock"):
            return url_or_base
        return get_settings().socket_path
    return None


def make_http_client(
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> httpx.Client:
    """Create an httpx.Client speaking over the SysAgent Unix Domain Socket."""
    uds = _resolve_uds(base_url)
    if uds:
        return httpx.Client(
            transport=httpx.HTTPTransport(uds=uds),
            base_url="http://sysagent.local",
            headers=headers,
            timeout=timeout,
        )
    return httpx.Client(base_url=base_url or "http://sysagent.local", headers=headers, timeout=timeout)


def _request(method: str, url: str, **kwargs: Any) -> httpx.Response:
    timeout = float(kwargs.pop("timeout", 10.0))
    headers = kwargs.pop("headers", None)
    uds = _resolve_uds(url)
    if uds:
        path = url.split("sysagent.local", 1)[1] if "sysagent.local" in url else url
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=uds),
            base_url="http://sysagent.local",
            headers=headers,
            timeout=timeout,
        ) as client:
            return client.request(method, path, **kwargs)
    return httpx.request(method, url, headers=headers, timeout=timeout, **kwargs)


def ipc_get(url: str, **kwargs: Any) -> httpx.Response:
    return _request("GET", url, **kwargs)


def ipc_post(url: str, **kwargs: Any) -> httpx.Response:
    return _request("POST", url, **kwargs)


def ipc_delete(url: str, **kwargs: Any) -> httpx.Response:
    return _request("DELETE", url, **kwargs)


def ipc_ws_connect(ws_url: str, **kwargs: Any) -> Any:
    """Connect to the daemon WebSocket over the Unix Domain Socket."""
    uds = _resolve_uds(ws_url)
    if uds:
        return websockets.unix_connect(uds, uri=ws_url, **kwargs)
    return websockets.connect(ws_url, **kwargs)

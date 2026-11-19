"""GGUF local inference runtime: warm llama-server / llama_cpp lifecycle and RAM guard."""

from __future__ import annotations

import atexit
import gc
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

_LLAMA_SERVER_PROC: subprocess.Popen[Any] | None = None
_LLAMA_SERVER_MODEL: str = ""
_LLAMA_SERVER_LAST_USED: float = 0.0
_LLAMA_CPP_INSTANCE: Any = None
_LLAMA_CPP_MODEL_PATH: str | None = None
_ATEXIT_REGISTERED: bool = False


def get_warm_model_name() -> str | None:
    """Return basename of the currently loaded warm GGUF model, if any."""
    active = _LLAMA_SERVER_MODEL or _LLAMA_CPP_MODEL_PATH
    return Path(active).name if active else None


def _release_all_warm_models() -> bool:
    """Immediately release any warm llama-server process and in-process Llama instance (strict single-slot RAM guard)."""
    global _LLAMA_SERVER_PROC, _LLAMA_SERVER_MODEL, _LLAMA_CPP_INSTANCE, _LLAMA_CPP_MODEL_PATH
    unloaded = False
    if _LLAMA_SERVER_PROC is not None and _LLAMA_SERVER_PROC.poll() is None:
        _LLAMA_SERVER_PROC.terminate()
        try:
            _LLAMA_SERVER_PROC.wait(timeout=4)
        except Exception:
            _LLAMA_SERVER_PROC.kill()
        _LLAMA_SERVER_PROC = None
        _LLAMA_SERVER_MODEL = ""
        unloaded = True
    if _LLAMA_CPP_INSTANCE is not None:
        try:
            if hasattr(_LLAMA_CPP_INSTANCE, "close"):
                _LLAMA_CPP_INSTANCE.close()
        except Exception:
            pass
        _LLAMA_CPP_INSTANCE = None
        _LLAMA_CPP_MODEL_PATH = None
        unloaded = True
    if unloaded:
        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    return unloaded


def _has_ram_for_gguf(gguf: Path, min_headroom_gb: float = 1.5) -> bool:
    """Verify sufficient available RAM before loading a large GGUF model to avoid swap thrashing."""
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / (1024**3)
        file_gb = gguf.stat().st_size / (1024**3) if gguf.is_file() else 0.0
        if file_gb <= 1.5:
            return avail_gb >= 0.8
        return avail_gb >= min_headroom_gb
    except Exception:
        return True


def unload_idle_servers(idle_sec: float = 120.0) -> bool:
    """Terminate warm llama-server / in-process Llama if idle for > idle_sec to free RAM & GPU VRAM."""
    if _LLAMA_SERVER_LAST_USED <= 0.0 or (time.monotonic() - _LLAMA_SERVER_LAST_USED) < idle_sec:
        return False
    return _release_all_warm_models()


def ensure_llama_server(gguf: Path) -> str | None:
    """Lazily start or reuse a warm `llama-server` process on 127.0.0.1:8769 for fast GPU inference."""
    import httpx

    server_bin = shutil.which("llama-server")
    if not server_bin or not gguf.is_file():
        return None

    global _LLAMA_SERVER_PROC, _LLAMA_SERVER_MODEL, _LLAMA_SERVER_LAST_USED, _ATEXIT_REGISTERED
    _LLAMA_SERVER_LAST_USED = time.monotonic()
    base_url = "http://127.0.0.1:8769"
    if (
        _LLAMA_SERVER_PROC is not None
        and _LLAMA_SERVER_PROC.poll() is None
        and str(gguf) == _LLAMA_SERVER_MODEL
    ):
        return base_url

    _release_all_warm_models()
    if not _has_ram_for_gguf(gguf):
        return None

    _LLAMA_SERVER_PROC = subprocess.Popen(
        [
            server_bin,
            "-m",
            str(gguf),
            "-ngl",
            "99",
            "-c",
            "4096",
            "--host",
            "127.0.0.1",
            "--port",
            "8769",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _LLAMA_SERVER_MODEL = str(gguf)
    if not _ATEXIT_REGISTERED:
        atexit.register(_release_all_warm_models)
        _ATEXIT_REGISTERED = True

    for _ in range(24):
        if _LLAMA_SERVER_PROC.poll() is not None:
            return None
        try:
            r = httpx.get(f"{base_url}/health", timeout=0.6)
            if r.status_code == 200:
                return base_url
        except Exception:
            time.sleep(0.5)
    return None


def get_llama_cpp_model(gguf: Path) -> Any | None:
    """Lazily load and cache a llama_cpp.Llama instance with full GPU offload."""
    if not gguf.is_file():
        return None
    try:
        from llama_cpp import Llama
    except Exception:
        return None

    global _LLAMA_CPP_INSTANCE, _LLAMA_CPP_MODEL_PATH, _LLAMA_SERVER_LAST_USED
    _LLAMA_SERVER_LAST_USED = time.monotonic()
    model_key = str(gguf)
    if _LLAMA_CPP_INSTANCE is not None and model_key == _LLAMA_CPP_MODEL_PATH:
        return _LLAMA_CPP_INSTANCE
    _release_all_warm_models()
    if not _has_ram_for_gguf(gguf):
        return None

    try:
        _LLAMA_CPP_INSTANCE = Llama(
            model_path=model_key,
            n_gpu_layers=-1,
            n_ctx=4096,
            use_mmap=True,
            verbose=False,
        )
    except Exception:
        _LLAMA_CPP_INSTANCE = None
        _LLAMA_CPP_MODEL_PATH = None
        return None
    _LLAMA_CPP_MODEL_PATH = model_key
    return _LLAMA_CPP_INSTANCE

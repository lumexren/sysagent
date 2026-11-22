"""GGUF local inference runtime: warm llama-server / llama_cpp lifecycle and RAM guard."""

from __future__ import annotations

import atexit
from collections.abc import Callable
import gc
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

_LLAMA_SERVER_PROC: subprocess.Popen[Any] | None = None
_LLAMA_SERVER_MODEL: str = ""
_LLAMA_SERVER_MMPROJ: str = ""
_LLAMA_SERVER_URL: str = ""
_LLAMA_SERVER_LAST_USED: float = 0.0
_LLAMA_CPP_INSTANCE: Any = None
_LLAMA_CPP_MODEL_PATH: str | None = None
_ATEXIT_REGISTERED: bool = False
_INFERENCE_LOCK = threading.RLock()
_VRAM_CACHE: tuple[float, dict[str, Any]] = (0.0, {})


def inference_lock() -> threading.RLock:
    """Return the reentrant lock protecting warm GGUF server/llama_cpp lifecycle and inference."""
    return _INFERENCE_LOCK


def touch_warm_model() -> None:
    """Refresh the last-used timestamp for the active warm GGUF model."""
    global _LLAMA_SERVER_LAST_USED
    _LLAMA_SERVER_LAST_USED = time.monotonic()


def _pick_server_port(preferred: int = 8769) -> int:
    """Return preferred port if free on 127.0.0.1, otherwise an available ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            try:
                sock.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get_warm_model_path() -> Path | None:
    """Return Path of the currently active warm GGUF model, if any."""
    if _LLAMA_SERVER_PROC is not None and _LLAMA_SERVER_PROC.poll() is None and _LLAMA_SERVER_MODEL:
        return Path(_LLAMA_SERVER_MODEL)
    if _LLAMA_CPP_INSTANCE is not None and _LLAMA_CPP_MODEL_PATH:
        return Path(_LLAMA_CPP_MODEL_PATH)
    return None


def get_warm_model_name() -> str | None:
    """Return basename of the currently loaded warm GGUF model, if any."""
    warm = get_warm_model_path()
    return warm.name if warm is not None else (_PathName_fallback() if (_LLAMA_SERVER_MODEL or _LLAMA_CPP_MODEL_PATH) else None)


def _PathName_fallback() -> str | None:
    active = _LLAMA_SERVER_MODEL or _LLAMA_CPP_MODEL_PATH
    return Path(active).name if active else None


def get_warm_model_idle_sec() -> float | None:
    """Return seconds elapsed since the currently loaded warm GGUF model was last used."""
    if get_warm_model_path() is None or _LLAMA_SERVER_LAST_USED <= 0.0:
        return None
    return round(max(0.0, time.monotonic() - _LLAMA_SERVER_LAST_USED), 1)


def get_vram_info(ttl_sec: float = 3.0) -> dict[str, Any]:
    """Query free/used/total GPU VRAM in MB via nvidia-smi or Linux DRM sysfs, with short TTL cache."""
    global _VRAM_CACHE
    now = time.monotonic()
    if ttl_sec > 0.0 and _VRAM_CACHE[1] and (now - _VRAM_CACHE[0]) < ttl_sec:
        return dict(_VRAM_CACHE[1])

    result: dict[str, Any] = {
        "available": False,
        "free_mb": 0,
        "used_mb": 0,
        "total_mb": 0,
        "source": "none",
    }
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
                if len(parts) >= 3:
                    free_mb = int(float(parts[0]))
                    used_mb = int(float(parts[1]))
                    total_mb = int(float(parts[2]))
                    result = {
                        "available": total_mb > 0,
                        "free_mb": max(0, free_mb),
                        "used_mb": max(0, used_mb),
                        "total_mb": max(0, total_mb),
                        "source": "nvidia",
                    }
        except Exception:
            pass

    if not result["available"]:
        best_total = 0
        best_used = 0
        drm_root = Path("/sys/class/drm")
        if drm_root.is_dir():
            for tot_f in sorted(drm_root.glob("card*/device/mem_info_vram_total")):
                used_f = tot_f.with_name("mem_info_vram_used")
                try:
                    tot_mb = int(tot_f.read_text(encoding="utf-8").strip()) // (1024 * 1024)
                    used_mb = (
                        int(used_f.read_text(encoding="utf-8").strip()) // (1024 * 1024)
                        if used_f.is_file()
                        else 0
                    )
                    if tot_mb > best_total:
                        best_total = tot_mb
                        best_used = max(0, used_mb)
                except Exception:
                    continue
        if best_total > 0:
            result = {
                "available": True,
                "free_mb": max(0, best_total - best_used),
                "used_mb": best_used,
                "total_mb": best_total,
                "source": "drm",
            }

    _VRAM_CACHE = (now, result)
    return dict(result)


def get_runtime_memory_status() -> dict[str, Any]:
    """Return warm GGUF runtime state, mmproj projector, idle time, VRAM headroom, and hysteresis window."""
    try:
        hyst = max(0.0, float(os.environ.get("SYSAGENT_WARM_HYSTERESIS_SEC", "90") or "90"))
    except ValueError:
        hyst = 90.0
    warm = get_warm_model_path()
    return {
        "warm_model": warm.name if warm is not None else None,
        "warm_mmproj": Path(_LLAMA_SERVER_MMPROJ).name if (_LLAMA_SERVER_MMPROJ and warm is not None) else None,
        "idle_sec": get_warm_model_idle_sec(),
        "vram": get_vram_info(),
        "hysteresis_sec": hyst,
    }


def _release_all_warm_models() -> bool:
    """Immediately release any warm llama-server process and in-process Llama instance (strict single-slot RAM guard)."""
    global _LLAMA_SERVER_PROC, _LLAMA_SERVER_MODEL, _LLAMA_SERVER_MMPROJ, _LLAMA_SERVER_URL, _LLAMA_CPP_INSTANCE, _LLAMA_CPP_MODEL_PATH
    with _INFERENCE_LOCK:
        unloaded = False
        if _LLAMA_SERVER_PROC is not None:
            if _LLAMA_SERVER_PROC.poll() is None:
                _LLAMA_SERVER_PROC.terminate()
                try:
                    _LLAMA_SERVER_PROC.wait(timeout=4)
                except Exception:
                    _LLAMA_SERVER_PROC.kill()
                    try:
                        _LLAMA_SERVER_PROC.wait(timeout=2)
                    except Exception:
                        pass
            _LLAMA_SERVER_PROC = None
            _LLAMA_SERVER_MODEL = ""
            _LLAMA_SERVER_MMPROJ = ""
            _LLAMA_SERVER_URL = ""
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


def can_fit_gguf_model(
    gguf: Path,
    mmproj: Path | None = None,
    min_ram_gb: float = 1.5,
    min_vram_headroom_mb: int = 512,
) -> tuple[bool, str]:
    """Assess RAM and GPU VRAM fit (`full_vram`, `partial_offload`, `ram_only`, or failure reason)."""
    if not _has_ram_for_gguf(gguf, min_headroom_gb=min_ram_gb):
        return False, "insufficient_ram"

    needed_bytes = gguf.stat().st_size if gguf.is_file() else 0
    if mmproj is not None and mmproj.is_file():
        needed_bytes += mmproj.stat().st_size
    needed_mb = int(needed_bytes / (1024 * 1024))

    vram = get_vram_info()
    if vram.get("available") and int(vram.get("total_mb", 0)) > 0:
        effective_free_mb = int(vram.get("free_mb", 0))
        warm = get_warm_model_path()
        if warm is not None and warm != gguf and warm.is_file():
            effective_free_mb += int(warm.stat().st_size / (1024 * 1024))
            if _LLAMA_SERVER_MMPROJ:
                warm_mm = Path(_LLAMA_SERVER_MMPROJ)
                if warm_mm.is_file():
                    effective_free_mb += int(warm_mm.stat().st_size / (1024 * 1024))
        if effective_free_mb >= needed_mb + min_vram_headroom_mb:
            return True, "full_vram"
        if os.environ.get("SYSAGENT_STRICT_VRAM", "0").strip() == "1":
            return False, "insufficient_vram"
        return True, "partial_offload"

    return True, "ram_only"


def choose_model_with_hysteresis(
    desired: Path,
    desired_role: str,
    prompt: str,
    classify_role_fn: Callable[[Path], str],
    hysteresis_sec: float | None = None,
) -> tuple[Path, bool]:
    """Keep a recently warm compatible model on short/general prompts to avoid swap latency."""
    if hysteresis_sec is None:
        try:
            hyst = max(0.0, float(os.environ.get("SYSAGENT_WARM_HYSTERESIS_SEC", "90") or "90"))
        except ValueError:
            hyst = 90.0
    else:
        hyst = max(0.0, float(hysteresis_sec))

    warm = get_warm_model_path()
    idle = get_warm_model_idle_sec()
    if (
        warm is None
        or warm == desired
        or not warm.is_file()
        or desired_role in {"multimodal", "embedding"}
        or idle is None
        or idle >= hyst
    ):
        return desired, False

    warm_role = classify_role_fn(warm)
    if warm_role == "embedding":
        return desired, False
    if desired_role in {"general", "chat"} and warm_role in {"general", "chat", "coding", "reasoning"}:
        return warm, True

    low = (prompt or "").lower()
    strong_coding = ("```" in (prompt or "")) or any(
        cue in low for cue in ("traceback", "def ", "class ", "diff --git")
    )
    if (
        len((prompt or "").split()) <= 16
        and warm_role != "multimodal"
        and not (desired_role == "coding" and strong_coding)
    ):
        return warm, True

    return desired, False


def unload_idle_servers(idle_sec: float = 120.0) -> bool:
    """Terminate warm llama-server / in-process Llama if idle for > idle_sec to free RAM & GPU VRAM."""
    if _LLAMA_SERVER_LAST_USED <= 0.0 or (time.monotonic() - _LLAMA_SERVER_LAST_USED) < idle_sec:
        return False
    if not _INFERENCE_LOCK.acquire(blocking=False):
        return False
    try:
        if _LLAMA_SERVER_LAST_USED <= 0.0 or (time.monotonic() - _LLAMA_SERVER_LAST_USED) < idle_sec:
            return False
        return _release_all_warm_models()
    finally:
        _INFERENCE_LOCK.release()


def ensure_llama_server(gguf: Path, mmproj: Path | None = None) -> str | None:
    """Lazily start or reuse a warm `llama-server` process for fast GPU inference (with optional mmproj vision projector)."""
    import httpx

    server_bin = shutil.which("llama-server")
    if not server_bin or not gguf.is_file():
        return None

    mmproj_str = str(mmproj) if (mmproj is not None and mmproj.is_file()) else ""
    global _LLAMA_SERVER_PROC, _LLAMA_SERVER_MODEL, _LLAMA_SERVER_MMPROJ, _LLAMA_SERVER_URL, _LLAMA_SERVER_LAST_USED, _ATEXIT_REGISTERED
    with _INFERENCE_LOCK:
        _LLAMA_SERVER_LAST_USED = time.monotonic()
        if (
            _LLAMA_SERVER_PROC is not None
            and _LLAMA_SERVER_PROC.poll() is None
            and str(gguf) == _LLAMA_SERVER_MODEL
            and (mmproj is None or mmproj_str == _LLAMA_SERVER_MMPROJ)
            and _LLAMA_SERVER_URL
        ):
            return _LLAMA_SERVER_URL

        _release_all_warm_models()
        fit_ok, _ = can_fit_gguf_model(gguf, mmproj=mmproj)
        if not fit_ok:
            return None

        port = _pick_server_port(8769)
        base_url = f"http://127.0.0.1:{port}"
        ctx_size = os.environ.get("SYSAGENT_LOCAL_CTX", "8192").strip() or "8192"
        cmd = [
            server_bin,
            "-m",
            str(gguf),
            "-ngl",
            "99",
            "-c",
            ctx_size,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if mmproj_str:
            cmd.extend(["--mmproj", mmproj_str])
        _LLAMA_SERVER_PROC = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _LLAMA_SERVER_MODEL = str(gguf)
        _LLAMA_SERVER_MMPROJ = mmproj_str
        _LLAMA_SERVER_URL = base_url
        if not _ATEXIT_REGISTERED:
            atexit.register(_release_all_warm_models)
            _ATEXIT_REGISTERED = True

        for _ in range(60):
            if _LLAMA_SERVER_PROC.poll() is not None:
                _release_all_warm_models()
                return None
            try:
                r = httpx.get(f"{base_url}/health", timeout=0.6)
                if r.status_code == 200:
                    _LLAMA_SERVER_LAST_USED = time.monotonic()
                    return base_url
            except Exception:
                pass
            time.sleep(0.5)

        _release_all_warm_models()
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
    with _INFERENCE_LOCK:
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

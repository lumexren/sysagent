"""Local GGUF model benchmark & calibration utility (`sysagent local bench`)."""

from __future__ import annotations

import time
from typing import Any

import psutil

from .local_inference import _has_ram_for_gguf, _release_all_warm_models
from .local_mind import LocalMind, classify_gguf_role, find_all_gguf_models
from .storage import Storage

DEFAULT_BENCH_PROMPT = "In one short sentence, explain what systemd does on Linux."


def benchmark_local_models(
    role_filter: str | None = None,
    prompt: str | None = None,
    storage: Storage | None = None,
) -> dict[str, Any]:
    """Benchmark installed non-embedding GGUF models for latency, tok/s, and RAM delta."""
    mind = LocalMind(storage=storage)
    bench_prompt = (prompt or DEFAULT_BENCH_PROMPT).strip()
    target_role = (role_filter or "").strip().lower() or None

    candidates = [
        p for p in find_all_gguf_models() if classify_gguf_role(p) != "embedding"
    ]
    if target_role:
        candidates = [p for p in candidates if classify_gguf_role(p) == target_role]

    results: list[dict[str, Any]] = []
    try:
        for gguf in candidates:
            role = classify_gguf_role(gguf)
            size_gb = round(gguf.stat().st_size / (1024**3), 2) if gguf.is_file() else 0.0
            _release_all_warm_models()
            if not _has_ram_for_gguf(gguf):
                results.append(
                    {
                        "role": role,
                        "model": gguf.name,
                        "size_gb": size_gb,
                        "status": "skipped_low_ram",
                        "latency_s": None,
                        "tok_per_s": None,
                        "ram_delta_mb": 0.0,
                    }
                )
                continue

            ram_before = psutil.virtual_memory().used
            t0 = time.monotonic()
            reply = mind._run_single_gguf(gguf, bench_prompt, "Reply concisely in one sentence.")
            elapsed = max(0.001, time.monotonic() - t0)
            ram_after = psutil.virtual_memory().used
            ram_delta_mb = round(max(0.0, (ram_after - ram_before) / (1024**2)), 1)

            if reply:
                est_tokens = max(1, round(len(reply.split()) * 1.3))
                tok_s = round(est_tokens / elapsed, 2)
                results.append(
                    {
                        "role": role,
                        "model": gguf.name,
                        "size_gb": size_gb,
                        "status": "ok",
                        "latency_s": round(elapsed, 2),
                        "tok_per_s": tok_s,
                        "ram_delta_mb": ram_delta_mb,
                        "sample": reply[:80],
                    }
                )
            else:
                results.append(
                    {
                        "role": role,
                        "model": gguf.name,
                        "size_gb": size_gb,
                        "status": "failed",
                        "latency_s": round(elapsed, 2),
                        "tok_per_s": 0.0,
                        "ram_delta_mb": ram_delta_mb,
                    }
                )
    finally:
        _release_all_warm_models()

    ok_runs = [r for r in results if r.get("status") == "ok" and r.get("tok_per_s")]
    fastest = max(ok_runs, key=lambda r: float(r["tok_per_s"])) if ok_runs else None
    return {
        "prompt": bench_prompt,
        "count": len(results),
        "fastest_role": fastest["role"] if fastest else None,
        "fastest_model": fastest["model"] if fastest else None,
        "results": results,
    }

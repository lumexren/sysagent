"""Drop-in Gemini Interactions client and GGUF download helper for LocalMind."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from .storage import Storage

if TYPE_CHECKING:
    pass


class _LocalMindInteractions:
    """Drop-in `.interactions.create(...)` implementation backed by `LocalMind`."""

    def __init__(self, storage: Storage | None = None) -> None:
        from .local_mind import LocalMind

        self.mind = LocalMind(storage)
        self._last_user_input: dict[str, str] = {}

    def create(
        self,
        *,
        model: str = "local-mind",
        input: Any = "",
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        previous_interaction_id: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        new_id = f"local-{uuid.uuid4().hex[:10]}"
        sys_inst = system_instruction or ""

        if "Respond ONLY with valid JSON" in sys_inst or "Respond ONLY with JSON" in sys_inst:
            json_reply = '{"decision": "ignore", "confidence": 85, "thought": "Offline local mode: no critical action required."}'
            if "candidates" in sys_inst:
                json_reply = '{"candidates": [], "notes": "Offline local mode"}'
            elif "approved" in sys_inst:
                json_reply = '{"approved": true, "critique": "Offline local check"}'
            if stream:
                def _jgen():
                    yield SimpleNamespace(
                        event_type="step.delta",
                        delta=SimpleNamespace(type="text", text=json_reply),
                    )
                return _jgen()
            return SimpleNamespace(id=new_id, steps=[], output_text=json_reply)

        if isinstance(input, list):
            orig_q = self._last_user_input.get(previous_interaction_id or "", "")
            reply = self.mind.synthesize_reply(orig_q, tool_results=input, system_instruction=sys_inst)
            if stream:
                def _rgen():
                    yield SimpleNamespace(
                        event_type="step.delta",
                        delta=SimpleNamespace(type="text", text=reply),
                    )
                    yield SimpleNamespace(
                        event_type="interaction.complete",
                        interaction=SimpleNamespace(id=new_id, steps=[], output_text=reply),
                    )
                return _rgen()
            return SimpleNamespace(id=new_id, steps=[], output_text=reply)

        user_text = str(input or "")
        self._last_user_input[new_id] = user_text
        if len(self._last_user_input) > 200:
            self._last_user_input.pop(next(iter(self._last_user_input)), None)
        if tools:
            allowed = {t.get("name") for t in tools if isinstance(t, dict) and t.get("name")}
            pred_tool, pred_args, score = self.mind.predict_tool(user_text, allowed_names=allowed)
            if pred_tool and score >= 0.16:
                step = SimpleNamespace(
                    type="function_call",
                    id=f"local_call_{uuid.uuid4().hex[:6]}",
                    name=pred_tool,
                    arguments=pred_args,
                )
                if stream:
                    def _fgen():
                        yield SimpleNamespace(
                            event_type="step.complete",
                            step=step,
                        )
                        yield SimpleNamespace(
                            event_type="interaction.complete",
                            interaction=SimpleNamespace(id=new_id, steps=[step], output_text=""),
                        )
                    return _fgen()
                return SimpleNamespace(id=new_id, steps=[step], output_text="")

        reply = self.mind.synthesize_reply(user_text, tool_results=None, system_instruction=sys_inst)
        if stream:
            def _tgen():
                yield SimpleNamespace(
                    event_type="step.delta",
                    delta=SimpleNamespace(type="text", text=reply),
                )
            return _tgen()
        return SimpleNamespace(id=new_id, steps=[], output_text=reply)


class LocalMindClient:
    """Drop-in `genai.Client` replacement backed by `LocalMind`."""

    def __init__(self, storage: Storage | None = None) -> None:
        self.interactions = _LocalMindInteractions(storage)


def download_default_gguf(
    preset_or_url: str = "qwen2.5-1.5b", filename: str | None = None
) -> Path:
    """Download a GGUF preset (or custom URL) into ~/.local/share/sysagent/models/ with live progress and resume."""
    import sys
    import time

    import httpx

    from .local_mind import GGUF_PRESETS, MODELS_DIR

    key = (preset_or_url or "qwen2.5-1.5b").strip()
    spec: dict[str, str] = {}
    if key.lower() in GGUF_PRESETS:
        spec = GGUF_PRESETS[key.lower()]
        url = spec["url"]
        fname = filename or spec["filename"]
    elif key.startswith(("http://", "https://")):
        url = key
        fname = filename or key.rstrip("/").split("/")[-1]
    else:
        raise ValueError(
            f"Unknown GGUF preset '{key}'. Available presets: {', '.join(GGUF_PRESETS)}"
        )

    def _download_file(target_url: str, dest_name: str) -> Path:
        dest = MODELS_DIR / dest_name
        if dest.is_file() and dest.stat().st_size > 1024 * 1024:
            return dest
        tmp = dest.with_suffix(".gguf.part")
        existing_bytes = tmp.stat().st_size if tmp.is_file() else 0
        headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes > 0 else {}

        with httpx.stream("GET", target_url, headers=headers, follow_redirects=True, timeout=600) as resp:
            if resp.status_code == 416:
                tmp.rename(dest)
                return dest
            resp.raise_for_status()
            is_resume = resp.status_code == 206
            content_len = int(resp.headers.get("content-length") or 0)
            total_bytes = (existing_bytes + content_len) if is_resume else content_len
            downloaded = existing_bytes if is_resume else 0
            mode = "ab" if is_resume else "wb"
            t0 = time.monotonic()
            last_print = 0.0

            with tmp.open(mode) as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_print >= 0.5:
                        elapsed = max(0.1, now - t0)
                        session_bytes = downloaded - (existing_bytes if is_resume else 0)
                        speed_mb = (session_bytes / (1024 * 1024)) / elapsed
                        dl_mb = downloaded / (1024 * 1024)
                        if total_bytes > 0:
                            tot_mb = total_bytes / (1024 * 1024)
                            pct = (downloaded / total_bytes) * 100.0
                            rem_sec = int(((total_bytes - downloaded) / (1024 * 1024)) / max(0.05, speed_mb))
                            sys.stdout.write(
                                f"\rDownloading: {dl_mb:.1f} / {tot_mb:.1f} MB ({pct:.1f}%) | {speed_mb:.2f} MB/s | ETA: {rem_sec}s   "
                            )
                        else:
                            sys.stdout.write(f"\rDownloading: {dl_mb:.1f} MB | {speed_mb:.2f} MB/s   ")
                        sys.stdout.flush()
                        last_print = now
            if last_print > 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
        tmp.rename(dest)
        return dest

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    main_dest = _download_file(url, fname)
    if spec.get("mmproj_url") and spec.get("mmproj_filename"):
        _download_file(spec["mmproj_url"], spec["mmproj_filename"])
    return main_dest

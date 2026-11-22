"""Drop-in Gemini Interactions client and GGUF download helper for LocalMind."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from .storage import Storage

if TYPE_CHECKING:
    pass


_LOCAL_CORE_TOOLS = frozenset(
    {
        "inspect_self_component",
        "switch_model",
        "system_status",
        "run_command",
        "read_file",
        "search_memory_and_history",
    }
)


class _LocalMindInteractions:
    """Drop-in `.interactions.create(...)` implementation backed by `LocalMind`."""

    def __init__(self, storage: Storage | None = None) -> None:
        from .local_mind import LocalMind

        self.mind = LocalMind(storage)
        self._last_user_input: dict[str, str] = {}
        self._history: dict[str, list[dict[str, str]]] = {}

    def _select_tools_for_local(
        self, user_text: str, tools: list[dict[str, Any]], limit: int = 12
    ) -> list[dict[str, Any]]:
        """Rank and select the most relevant OpenAI tool declarations using NeuralEngine, LocalMind, and schemas."""
        valid = [t for t in tools if isinstance(t, dict) and t.get("name")]
        if len(valid) <= limit:
            return valid

        scores: dict[str, float] = {str(t["name"]): 0.0 for t in valid}
        if self.mind.storage is not None:
            try:
                from .neural import NeuralEngine

                eng = NeuralEngine(self.mind.storage)
                if int(eng.meta.get("epochs_trained", 0)) > 0:
                    for name, prob in eng.predict_tool_probs(user_text).items():
                        if name in scores:
                            scores[name] += float(prob) * 10.0
            except Exception:
                pass
        try:
            pred_name, _, pred_score = self.mind.predict_tool(
                user_text, allowed_names=set(scores)
            )
            if pred_name and pred_name in scores:
                scores[pred_name] += float(pred_score) * 8.0
        except Exception:
            pass

        q_tokens = {tok for tok in (user_text or "").lower().split() if len(tok) >= 2}
        for t in valid:
            tname = str(t["name"])
            if tname in _LOCAL_CORE_TOOLS:
                scores[tname] += 3.0
            hay = f"{tname} {t.get('description') or ''}".lower()
            scores[tname] += sum(1.5 for tok in q_tokens if tok in hay)

        core_first = [t for t in valid if str(t["name"]) in _LOCAL_CORE_TOOLS]
        extra = sorted(
            [t for t in valid if str(t["name"]) not in _LOCAL_CORE_TOOLS],
            key=lambda t: scores.get(str(t["name"]), 0.0),
            reverse=True,
        )
        return (core_first + extra)[:limit]

    @staticmethod
    def _format_tool_catalog(tools: list[dict[str, Any]], user_text: str = "") -> str:
        valid = [t for t in tools if isinstance(t, dict) and t.get("name")]
        if not valid:
            return ""
        q_tokens = {tok for tok in (user_text or "").lower().split() if len(tok) >= 2}

        def _rank(tool: dict[str, Any]) -> tuple[int, int]:
            name = str(tool["name"])
            hay = f"{name} {tool.get('description') or ''}".lower()
            overlap = sum(3 for tok in q_tokens if tok in hay)
            return (overlap, 1 if name in _LOCAL_CORE_TOOLS else 0)

        ranked = sorted(valid, key=_rank, reverse=True)
        lines: list[str] = []
        for t in ranked[:16]:
            params = list(((t.get("parameters") or {}).get("properties") or {}).keys())[:5]
            desc = str(t.get("description") or "").splitlines()[0][:85]
            sig = f"{t['name']}({', '.join(params)})"
            lines.append(f"- {sig}: {desc}")
        return (
            "Available tools:\n"
            + "\n".join(lines)
            + '\nIf a tool is required to answer accurately, respond ONLY with JSON: {"tool": "<name>", "args": {...}}.'
            " Otherwise, reply directly in natural plain text without JSON."
        )

    @staticmethod
    def _extract_tool_call(text: str, allowed: set[str]) -> tuple[str, dict[str, Any]] | None:
        import json as _json

        raw = (text or "").strip()
        if not raw or "{" not in raw:
            return None
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1].lstrip("json").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = _json.loads(raw[start : end + 1])
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        tname = str(obj.get("tool") or obj.get("name") or "").strip()
        targs = obj.get("args") if isinstance(obj.get("args"), dict) else obj.get("arguments")
        if tname in allowed:
            return tname, (targs if isinstance(targs, dict) else {})
        return None

    def _emit_steps(self, new_id: str, steps: list[Any], stream: bool) -> Any:
        if stream:
            def _fgen():
                for step in steps:
                    yield SimpleNamespace(event_type="step.complete", step=step)
                yield SimpleNamespace(
                    event_type="interaction.complete",
                    interaction=SimpleNamespace(id=new_id, steps=steps, output_text=""),
                )
            return _fgen()
        return SimpleNamespace(id=new_id, steps=steps, output_text="")

    def _emit_step(self, new_id: str, step: Any, stream: bool) -> Any:
        return self._emit_steps(new_id, [step], stream)

    def _emit_text(self, new_id: str, reply: str, stream: bool) -> Any:
        if stream:
            def _tgen():
                yield SimpleNamespace(
                    event_type="step.delta",
                    delta=SimpleNamespace(type="text", text=reply),
                )
                yield SimpleNamespace(
                    event_type="interaction.complete",
                    interaction=SimpleNamespace(id=new_id, steps=[], output_text=reply),
                )
            return _tgen()
        return SimpleNamespace(id=new_id, steps=[], output_text=reply)

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
        import json as _json

        new_id = f"local-{uuid.uuid4().hex[:10]}"
        on_delta = kwargs.pop("on_delta", None)
        sys_inst = system_instruction or ""
        from .self_resilience import ContextBudgeter

        raw_hist = list(self._history.get(previous_interaction_id or "", []))
        prev_hist = ContextBudgeter.compact_history(raw_hist, max_chars=2400, keep_recent=4)

        if "Respond ONLY with valid JSON" in sys_inst or "Respond ONLY with JSON" in sys_inst:
            json_reply = '{"decision": "ignore", "confidence": 85, "thought": "Offline local mode: no critical action required."}'
            if "candidates" in sys_inst:
                json_reply = '{"candidates": [], "notes": "Offline local mode"}'
            elif "approved" in sys_inst:
                json_reply = '{"approved": true, "critique": "Offline local check"}'
            return self._emit_text(new_id, json_reply, stream)

        if isinstance(input, list):
            orig_q = self._last_user_input.get(previous_interaction_id or "", "")
            summary_prompt = (
                f"User request: {orig_q}\n"
                f"Tool results: {_json.dumps(input, ensure_ascii=False)[:3400]}\n"
                "Answer the user's request concisely in their language based on these tool results."
            )
            if on_delta is not None:
                reply = self.mind.stream_gguf_inference(
                    summary_prompt, system_instruction=sys_inst, history=prev_hist, on_delta=on_delta
                )
            else:
                reply = self.mind.run_gguf_inference(
                    summary_prompt, system_instruction=sys_inst, history=prev_hist
                )
            if not reply:
                reply = self.mind.synthesize_reply(orig_q, tool_results=input, system_instruction=sys_inst)
            self._history[new_id] = (raw_hist + [{"role": "user", "content": orig_q}, {"role": "assistant", "content": reply}])[-16:]
            return self._emit_text(new_id, reply, stream)

        user_text = str(input or "")
        self._last_user_input[new_id] = user_text
        self._history[new_id] = prev_hist
        if len(self._last_user_input) > 200:
            oldest = next(iter(self._last_user_input))
            self._last_user_input.pop(oldest, None)
            self._history.pop(oldest, None)

        allowed: set[str] = set()
        selected_tools: list[dict[str, Any]] = []
        if tools:
            selected_tools = self._select_tools_for_local(user_text, tools, limit=12)
            allowed = {str(t["name"]) for t in selected_tools if isinstance(t, dict) and t.get("name")}
            native = self.mind.run_gguf_chat_with_tools(
                user_text,
                tools=selected_tools,
                system_instruction=sys_inst,
                history=prev_hist,
                on_delta=on_delta,
            )
            if native is not None:
                native_calls = native.get("tool_calls") or []
                if native_calls:
                    steps = [
                        SimpleNamespace(
                            type="function_call",
                            id=str(tc.get("id") or f"local_call_{uuid.uuid4().hex[:6]}"),
                            name=str(tc["name"]),
                            arguments=dict(tc.get("arguments") or {}),
                        )
                        for tc in native_calls
                    ]
                    return self._emit_steps(new_id, steps, stream)
                native_text = str(native.get("text") or "").strip()
                if native_text:
                    self._history[new_id] = (
                        raw_hist + [{"role": "user", "content": user_text}, {"role": "assistant", "content": native_text}]
                    )[-16:]
                    return self._emit_text(new_id, native_text, stream)

            from .self_guard import get_calibration

            tool_thresh = float(get_calibration(self.mind.storage)["local_tool_threshold"])
            pred_tool, pred_args, score = self.mind.predict_tool(user_text, allowed_names=allowed)
            if pred_tool and score >= tool_thresh:
                step = SimpleNamespace(
                    type="function_call",
                    id=f"local_call_{uuid.uuid4().hex[:6]}",
                    name=pred_tool,
                    arguments=pred_args,
                )
                return self._emit_step(new_id, step, stream)

        extra_sys = self._format_tool_catalog(selected_tools or tools or [], user_text) if tools else ""
        if on_delta is not None:
            stream_mode: list[str] = ["pending" if allowed else "live"]

            def _guarded_delta(chunk: str) -> None:
                if stream_mode[0] == "suppress":
                    return
                if allowed and "{" in chunk:
                    before = chunk.split("{", 1)[0]
                    if stream_mode[0] == "live" and before:
                        on_delta(before)
                    stream_mode[0] = "suppress"
                    return
                if stream_mode[0] == "pending":
                    stripped = chunk.lstrip()
                    if not stripped:
                        return
                    if stripped[0] in {"{", "`"}:
                        stream_mode[0] = "suppress"
                        return
                    stream_mode[0] = "live"
                on_delta(chunk)

            reply = self.mind.stream_gguf_inference(
                user_text,
                system_instruction=sys_inst,
                history=prev_hist,
                extra_system=extra_sys,
                on_delta=_guarded_delta,
            )
        else:
            reply = self.mind.run_gguf_inference(
                user_text,
                system_instruction=sys_inst,
                history=prev_hist,
                extra_system=extra_sys,
            )
        if reply and allowed:
            parsed = self._extract_tool_call(reply, allowed)
            if parsed is not None:
                tname, targs = parsed
                step = SimpleNamespace(
                    type="function_call",
                    id=f"local_call_{uuid.uuid4().hex[:6]}",
                    name=tname,
                    arguments=targs,
                )
                return self._emit_step(new_id, step, stream)

        if not reply:
            reply = self.mind.synthesize_reply(user_text, tool_results=None, system_instruction=sys_inst)
        self._history[new_id] = (raw_hist + [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}])[-16:]
        return self._emit_text(new_id, reply, stream)


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
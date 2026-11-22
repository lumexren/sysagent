"""GGUF Inference, Streaming, Function-Calling, and Local-First Cascade Mixin for LocalMind."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .local_inference import (
    choose_model_with_hysteresis,
    ensure_llama_server,
    get_llama_cpp_model,
    get_warm_model_path,
)


class _LocalGGUFMixin:
    """Mixin providing GGUF message construction, single/streaming inference, function calling, and local cascade."""

    storage: Any
    weights: dict[str, Any]

    def _ensure_llama_server(self, gguf: Path, mmproj: Path | None = None) -> str | None:
        """Lazily start or reuse a warm `llama-server` process on 127.0.0.1:8769 for fast GPU inference."""
        return ensure_llama_server(gguf, mmproj=mmproj)

    def _get_llama_cpp_model(self, gguf: Path) -> Any | None:
        """Lazily load and cache a llama_cpp.Llama instance with full GPU offload."""
        return get_llama_cpp_model(gguf)

    @staticmethod
    def _compact_system_instruction(sys_inst: str, max_chars: int = 3400) -> str:
        """Compact long system instructions while keeping dynamic tail runtime sections intact."""
        raw = (sys_inst or "").strip()
        if len(raw) <= max_chars:
            return raw
        anchors = (
            "\nRuntime self-awareness contract:",
            "\nLive Cortex state:",
            "\nLive codebase",
            "\nActive session checkpoint:",
            "\nLanguage rule:",
        )
        positions = [raw.find(a) for a in anchors if raw.find(a) != -1]
        if positions:
            tail_start = min(positions)
            tail = raw[tail_start:].strip()
            if len(tail) > max_chars - 200:
                tail = tail[-(max_chars - 200) :]
            head_budget = max(200, max_chars - len(tail) - 8)
            head = raw[:tail_start][:head_budget].rstrip()
            return f"{head}\n{tail}".strip()
        half = max_chars // 2
        return f"{raw[:half].rstrip()}\n...\n{raw[-half:].lstrip()}"

    def _build_gguf_messages(
        self,
        gguf: Path,
        prompt: str,
        system_instruction: str = "",
        history: list[dict[str, str]] | None = None,
        extra_system: str = "",
        images: list[dict[str, str]] | None = None,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        from . import local_mind as _lm
        from .config import get_settings

        examples = self.retrieve_similar_examples(prompt, top_k=2) if not extra_system else []  # type: ignore[attr-defined]
        role_hint = ""
        if _lm.ROLE_ADAPTERS_FILE.is_file():
            try:
                adapters = json.loads(_lm.ROLE_ADAPTERS_FILE.read_text(encoding="utf-8"))
                role_hint = str(
                    (adapters.get(_lm.classify_task_role(prompt, storage=self.storage)) or {}).get("system_hint") or ""
                )
            except Exception:
                pass
        ex_block = ""
        if examples:
            ex_lines = [
                f"User: {ex['user']}\nAssistant: {ex['reply']}" for ex in examples if ex.get("reply")
            ]
            if ex_lines:
                ex_block = "\n\nPast learned examples:\n" + "\n---\n".join(ex_lines)
        contrastive = self.retrieve_contrastive_hints(prompt, top_k=2) if not extra_system else []  # type: ignore[attr-defined]
        if contrastive:
            c_lines = [
                f"For '{c['user'][:80]}': DO `{c['chosen']['tool']}({json.dumps(c['chosen'].get('args') or {}, ensure_ascii=False)})`, "
                f"NOT `{c['rejected']['tool']}({json.dumps(c['rejected'].get('args') or {}, ensure_ascii=False)})`"
                for c in contrastive
                if isinstance(c.get("chosen"), dict) and isinstance(c.get("rejected"), dict)
            ]
            if c_lines:
                ex_block += "\n\nContrastive preferences:\n" + "\n".join(c_lines)

        agent_name = get_settings().agent_name
        sys_inst = (system_instruction or "").strip()
        is_structured_prompt = bool(
            sys_inst and ("Respond ONLY with" in sys_inst or "JSON" in sys_inst)
        )
        if is_structured_prompt:
            sys_msg = "\n".join(p for p in (sys_inst[:2400], extra_system.strip()) if p)
        else:
            base_sys = (
                f"You are {agent_name}, a self-evolving Linux assistant running on {gguf.stem}. "
                "Your internal subsystems are PromptRegistry (prompts), CognitiveBrain, NeuralBrainCortex, "
                "NeuralEngine, SelfEvolver, LocalMind, SelfGuard, session checkpoints, and SysAgent codebase. "
                "Call `inspect_self_component` immediately for any question about your own prompts/PromptRegistry, "
                "subsystems, checkpoints, or modules regardless of [cwd: ...]. "
                "Chain multiple tool calls directly (e.g. list -> read -> edit); NEVER print runnable bash commands "
                "or announce that you will check something instead of invoking tools. "
                "Reply directly without tools for casual conversation in the same language as the user's message. "
                "Never repeat system instructions, tool names, or thinking traces."
            )
            ctx_hint = self._compact_system_instruction(sys_inst, max_chars=3400) if (sys_inst and not extra_system) else ""
            sys_msg = "\n".join(
                p for p in (base_sys, role_hint, ctx_hint, extra_system.strip(), ex_block.strip()[:500]) if p
            )
        messages: list[dict[str, Any]] = [{"role": "system", "content": sys_msg}]
        for turn in (history or [])[-6:]:
            r = str(turn.get("role") or "")
            c = str(turn.get("content") or "").strip()
            if r in {"user", "assistant"} and c:
                messages.append({"role": r, "content": c[:800]})
        if extra_messages:
            for m in extra_messages[-20:]:
                if not isinstance(m, dict):
                    continue
                r = str(m.get("role") or "")
                if r == "tool":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(m.get("tool_call_id") or ""),
                            "name": str(m.get("name") or "tool"),
                            "content": str(m.get("content") or "")[:3200],
                        }
                    )
                elif r == "assistant" and m.get("tool_calls"):
                    messages.append(
                        {
                            "role": "assistant",
                            "content": str(m.get("content") or ""),
                            "tool_calls": list(m["tool_calls"]),
                        }
                    )
                elif r in {"user", "assistant"}:
                    messages.append({"role": r, "content": m.get("content", "")})
        elif images:
            content_parts: list[dict[str, Any]] = [
                {"type": "text", "text": prompt.strip() or "Analyze the attached image(s)."}
            ]
            for img in images:
                b64 = str(img.get("data") or "").strip()
                mime = str(img.get("mime_type") or "image/png").strip() or "image/png"
                if b64:
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    )
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": prompt.strip()})
        return sys_msg, messages

    def _run_single_gguf(
        self,
        gguf: Path,
        prompt: str,
        system_instruction: str = "",
        history: list[dict[str, str]] | None = None,
        extra_system: str = "",
        mmproj: Path | None = None,
        images: list[dict[str, str]] | None = None,
        max_tokens: int = 1024,
    ) -> str | None:
        import httpx
        from . import local_mind as _lm

        sys_msg, messages = self._build_gguf_messages(
            gguf,
            prompt,
            system_instruction,
            history=history,
            extra_system=extra_system,
            images=images,
        )

        def _try_server() -> str | None:
            from .local_inference import inference_lock, touch_warm_model

            with inference_lock():
                srv_url = self._ensure_llama_server(gguf, mmproj=mmproj)
                if not srv_url:
                    return None
                try:
                    resp = httpx.post(
                        f"{srv_url}/v1/chat/completions",
                        json={
                            "messages": messages,
                            "temperature": 0.2,
                            "max_tokens": max_tokens,
                        },
                        timeout=60,
                    )
                    touch_warm_model()
                    if resp.status_code == 200:
                        choices = resp.json().get("choices") or []
                        if choices:
                            content = str(choices[0].get("message", {}).get("content") or "").strip()
                            if content:
                                return content
                except Exception:
                    pass
                return None

        def _try_llama_cpp() -> str | None:
            from .local_inference import inference_lock, touch_warm_model

            with inference_lock():
                llm = self._get_llama_cpp_model(gguf)
                if llm is None:
                    return None
                try:
                    res = llm.create_chat_completion(
                        messages=messages,
                        temperature=0.2,
                        max_tokens=max_tokens,
                    )
                    touch_warm_model()
                    choices = res.get("choices") or []
                    if choices:
                        content = str(choices[0].get("message", {}).get("content") or "").strip()
                        if content:
                            return content
                except Exception:
                    pass
                return None

        def _try_cli() -> str | None:
            llama_bin = shutil.which("llama-cli")
            if not llama_bin or not gguf.is_file():
                return None
            full_prompt = (
                f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            try:
                proc = subprocess.run(
                    [
                        llama_bin,
                        "-m",
                        str(gguf),
                        "-ngl",
                        "99",
                        "-c",
                        "4096",
                        "-n",
                        str(max_tokens),
                        "--temp",
                        "0.2",
                        "-no-cnv",
                        "--no-display-prompt",
                        "-p",
                        full_prompt,
                    ],
                    input="",
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    out = proc.stdout.strip()
                    return out.split("<|im_end|>")[0].strip()
            except Exception:
                pass
            return None

        backend = os.environ.get("SYSAGENT_LOCAL_BACKEND", "auto").strip().lower()
        if images:
            order: tuple[Any, ...] = (_try_server,)
        elif backend in {"llama_cpp", "llama-cpp", "python"}:
            order = (_try_llama_cpp, _try_server, _try_cli)
        elif backend in {"llama_cli", "llama-cli", "cli"}:
            order = (_try_cli, _try_server)
        else:
            order = (_try_server, _try_cli)

        for runner in order:
            out = _lm._strip_think_tags(runner() or "")
            if out:
                return out
        return None

    def run_gguf_inference(
        self,
        prompt: str,
        system_instruction: str = "",
        preferred_role: str | None = None,
        history: list[dict[str, str]] | None = None,
        extra_system: str = "",
        images: list[dict[str, str]] | None = None,
        max_tokens: int = 1024,
        prefer_warm: bool = False,
    ) -> str | None:
        """Run fast single-pass local GPU-accelerated GGUF inference via task-routed specialists."""
        from . import local_mind as _lm

        models = [m for m in _lm.find_all_gguf_models() if _lm.classify_gguf_role(m) != "embedding"]
        if not models:
            return None
        if images:
            target_role = "multimodal"
            specialist = (
                _lm.find_gguf_model("multimodal", strict=True)
                or _lm.find_gguf_model("multimodal")
                or models[0]
            )
            mmproj = _lm.find_mmproj_for_model(specialist)
        else:
            warm = get_warm_model_path() if prefer_warm and preferred_role is None else None
            if warm is not None and warm.is_file() and _lm.classify_gguf_role(warm) != "embedding":
                specialist = warm
                target_role = _lm.classify_gguf_role(warm)
            else:
                target_role = preferred_role or _lm.classify_task_role(prompt, storage=self.storage)
                specialist = _lm.find_gguf_model(target_role, strict=True) or _lm.find_gguf_model(target_role) or models[0]
                if preferred_role is None:
                    specialist, _ = choose_model_with_hysteresis(
                        specialist, target_role, prompt, _lm.classify_gguf_role
                    )
            mmproj = _lm.find_mmproj_for_model(specialist) if _lm.classify_gguf_role(specialist) == "multimodal" else None
        final_out = self._run_single_gguf(
            specialist,
            prompt,
            system_instruction,
            history=history,
            extra_system=extra_system,
            mmproj=mmproj,
            images=images,
            max_tokens=max_tokens,
        )
        if not final_out and not images:
            fallback = (
                _lm.find_gguf_model("chat", strict=True)
                or _lm.find_gguf_model("general", strict=True)
                or models[0]
            )
            if fallback != specialist:
                final_out = self._run_single_gguf(
                    fallback,
                    prompt,
                    system_instruction,
                    history=history,
                    extra_system=extra_system,
                    max_tokens=max_tokens,
                )
        if final_out and not extra_system and len(prompt.strip()) >= 8:
            qa = self.weights.setdefault("qa_memory", [])
            qa.append({"role": target_role, "user": prompt.strip()[:300], "tool_calls": [], "reply": final_out[:600]})
            if len(qa) > 150:
                self.weights["qa_memory"] = qa[-150:]
        return final_out

    def stream_gguf_inference(
        self,
        prompt: str,
        system_instruction: str = "",
        preferred_role: str | None = None,
        history: list[dict[str, str]] | None = None,
        extra_system: str = "",
        on_delta: Callable[[str], None] | None = None,
        images: list[dict[str, str]] | None = None,
    ) -> str | None:
        """Stream visible tokens live from warm llama-server SSE while suppressing <think> blocks."""
        from . import local_mind as _lm

        if on_delta is None:
            return self.run_gguf_inference(
                prompt,
                system_instruction=system_instruction,
                preferred_role=preferred_role,
                history=history,
                extra_system=extra_system,
                images=images,
            )
        models = [m for m in _lm.find_all_gguf_models() if _lm.classify_gguf_role(m) != "embedding"]
        if not models:
            return None
        if images:
            target_role = "multimodal"
            specialist = (
                _lm.find_gguf_model("multimodal", strict=True)
                or _lm.find_gguf_model("multimodal")
                or models[0]
            )
            mmproj = _lm.find_mmproj_for_model(specialist)
        else:
            target_role = preferred_role or _lm.classify_task_role(prompt, storage=self.storage)
            specialist = _lm.find_gguf_model(target_role, strict=True) or _lm.find_gguf_model(target_role) or models[0]
            if preferred_role is None:
                specialist, _ = choose_model_with_hysteresis(
                    specialist, target_role, prompt, _lm.classify_gguf_role
                )
            mmproj = _lm.find_mmproj_for_model(specialist) if _lm.classify_gguf_role(specialist) == "multimodal" else None
        backend = os.environ.get("SYSAGENT_LOCAL_BACKEND", "auto").strip().lower()
        if backend not in {"llama_cli", "llama-cli", "cli", "llama_cpp", "llama-cpp", "python"}:
            from .local_inference import inference_lock, touch_warm_model

            with inference_lock():
                srv_url = self._ensure_llama_server(specialist, mmproj=mmproj)
                if srv_url:
                    import httpx

                    _, messages = self._build_gguf_messages(
                        specialist,
                        prompt,
                        system_instruction,
                        history=history,
                        extra_system=extra_system,
                        images=images,
                    )
                    filt = _lm._ThinkStreamFilter()
                    collected: list[str] = []
                    http_ok = False
                    try:
                        with httpx.stream(
                            "POST",
                            f"{srv_url}/v1/chat/completions",
                            json={
                                "messages": messages,
                                "temperature": 0.2,
                                "max_tokens": 1024,
                                "stream": True,
                            },
                            timeout=60,
                        ) as resp:
                            http_ok = resp.status_code == 200
                            if http_ok:
                                for line in resp.iter_lines():
                                    if not line or not line.startswith("data:"):
                                        continue
                                    payload = line[5:].strip()
                                    if payload == "[DONE]":
                                        break
                                    try:
                                        chunk_obj = json.loads(payload)
                                    except Exception:
                                        continue
                                    choices = chunk_obj.get("choices") or []
                                    if not choices:
                                        continue
                                    token = str((choices[0].get("delta") or {}).get("content") or "")
                                    if not token:
                                        continue
                                    touch_warm_model()
                                    visible = filt.feed(token)
                                    if visible:
                                        collected.append(visible)
                                        on_delta(visible)
                        if http_ok:
                            tail = filt.flush()
                            if tail:
                                collected.append(tail)
                                on_delta(tail)
                            final_out = "".join(collected).strip()
                            if final_out:
                                if not extra_system and len(prompt.strip()) >= 8:
                                    qa = self.weights.setdefault("qa_memory", [])
                                    qa.append(
                                        {
                                            "role": target_role,
                                            "user": prompt.strip()[:300],
                                            "tool_calls": [],
                                            "reply": final_out[:600],
                                        }
                                    )
                                    if len(qa) > 150:
                                        self.weights["qa_memory"] = qa[-150:]
                                return final_out
                            return None
                    except Exception:
                        if collected:
                            return "".join(collected).strip() or None

        fallback_out = self.run_gguf_inference(
            prompt,
            system_instruction=system_instruction,
            preferred_role=preferred_role,
            history=history,
            extra_system=extra_system,
            images=images,
        )
        if fallback_out:
            on_delta(fallback_out)
        return fallback_out

    def run_gguf_chat_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_instruction: str = "",
        preferred_role: str | None = None,
        history: list[dict[str, str]] | None = None,
        on_delta: Callable[[str], None] | None = None,
        images: list[dict[str, str]] | None = None,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Run native OpenAI-compatible function calling (`tool_choice=auto`) on warm llama-server."""
        from . import local_mind as _lm

        backend = os.environ.get("SYSAGENT_LOCAL_BACKEND", "auto").strip().lower()
        if backend in {"llama_cli", "llama-cli", "cli", "llama_cpp", "llama-cpp", "python"}:
            return None
        models = [m for m in _lm.find_all_gguf_models() if _lm.classify_gguf_role(m) != "embedding"]
        if not models:
            return None

        if images:
            specialist = (
                _lm.find_gguf_model("multimodal", strict=True)
                or _lm.find_gguf_model("multimodal")
                or models[0]
            )
            mmproj = _lm.find_mmproj_for_model(specialist)
        elif preferred_role:
            specialist = (
                _lm.find_gguf_model(preferred_role, strict=True)
                or _lm.find_gguf_model(preferred_role)
                or _lm.find_gguf_model("general", strict=True)
                or models[0]
            )
            mmproj = _lm.find_mmproj_for_model(specialist) if _lm.classify_gguf_role(specialist) == "multimodal" else None
        else:
            role_guess = _lm.classify_task_role(prompt, storage=self.storage)
            if role_guess in {"coding", "reasoning", "multimodal"}:
                specialist = (
                    _lm.find_gguf_model(role_guess, strict=True)
                    or _lm.find_gguf_model("general", strict=True)
                    or _lm.find_gguf_model("chat", strict=True)
                    or models[0]
                )
            else:
                specialist = (
                    _lm.find_gguf_model("general", strict=True)
                    or _lm.find_gguf_model("chat", strict=True)
                    or models[0]
                )
            specialist, _ = choose_model_with_hysteresis(
                specialist, role_guess, prompt, _lm.classify_gguf_role
            )
            mmproj = _lm.find_mmproj_for_model(specialist) if _lm.classify_gguf_role(specialist) == "multimodal" else None

        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": str(t["name"]),
                    "description": str(t.get("description") or "")[:240],
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            for t in tools
            if isinstance(t, dict) and t.get("name")
        ]
        allowed_names = {str(t["function"]["name"]) for t in oai_tools}
        _, messages = self._build_gguf_messages(
            specialist,
            prompt,
            system_instruction,
            history=history,
            extra_system="",
            images=images,
            extra_messages=extra_messages,
        )

        from .local_inference import inference_lock, touch_warm_model

        with inference_lock():
            srv_url = self._ensure_llama_server(specialist, mmproj=mmproj)
            if not srv_url:
                return None

            import httpx

            payload: dict[str, Any] = {
                "messages": messages,
                "temperature": 0.15,
                "max_tokens": 1024,
                "stream": on_delta is not None,
            }
            if oai_tools:
                payload["tools"] = oai_tools
                payload["tool_choice"] = "auto"

            def _send_chat_req(req_payload: dict[str, Any]) -> tuple[bool, list[dict[str, Any]], str]:
                req_calls: list[dict[str, Any]] = []
                req_text = ""
                if on_delta is not None:
                    tc_acc: dict[int, dict[str, str]] = {}
                    filt = _lm._ThinkStreamFilter()
                    collected: list[str] = []
                    try:
                        with httpx.stream(
                            "POST", f"{srv_url}/v1/chat/completions", json=req_payload, timeout=60
                        ) as resp:
                            if resp.status_code != 200:
                                return False, [], ""
                            for line in resp.iter_lines():
                                if not line or not line.startswith("data:"):
                                    continue
                                data_str = line[5:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                except Exception:
                                    continue
                                choices = chunk.get("choices") or []
                                if not choices:
                                    continue
                                delta = choices[0].get("delta") or {}
                                if delta.get("tool_calls"):
                                    for tc in delta["tool_calls"]:
                                        idx = int(tc.get("index", 0) or 0)
                                        slot = tc_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                                        if tc.get("id"):
                                            slot["id"] = str(tc["id"])
                                        fn = tc.get("function") or {}
                                        if fn.get("name"):
                                            slot["name"] += str(fn["name"])
                                        if fn.get("arguments"):
                                            slot["arguments"] += str(fn["arguments"])
                                    touch_warm_model()
                                tok = str(delta.get("content") or "")
                                if tok:
                                    touch_warm_model()
                                    visible = filt.feed(tok)
                                    if visible and not tc_acc:
                                        collected.append(visible)
                                        on_delta(visible)
                        if not tc_acc:
                            tail = filt.flush()
                            if tail:
                                collected.append(tail)
                                on_delta(tail)
                        req_text = "".join(collected).strip()
                        for idx in sorted(tc_acc):
                            req_calls.append(tc_acc[idx])
                        return True, req_calls, req_text
                    except Exception:
                        if not tc_acc and not collected:
                            return False, [], ""
                        req_text = "".join(collected).strip()
                        for idx in sorted(tc_acc):
                            req_calls.append(tc_acc[idx])
                        return True, req_calls, req_text
                try:
                    resp = httpx.post(f"{srv_url}/v1/chat/completions", json=req_payload, timeout=60)
                    touch_warm_model()
                    if resp.status_code != 200:
                        return False, [], ""
                    choices = resp.json().get("choices") or []
                    if not choices:
                        return True, [], ""
                    msg_obj = choices[0].get("message") or {}
                    req_text = _lm._strip_think_tags(str(msg_obj.get("content") or ""))
                    for tc in msg_obj.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        req_calls.append(
                            {
                                "id": str(tc.get("id") or ""),
                                "name": str(fn.get("name") or ""),
                                "arguments": str(fn.get("arguments") or ""),
                            }
                        )
                    return True, req_calls, req_text
                except Exception:
                    return False, [], ""

            http_ok, raw_calls, final_text = _send_chat_req(payload)
            if (
                not http_ok
                and extra_messages
                and any(isinstance(m, dict) and m.get("role") == "tool" for m in extra_messages)
            ):
                base_msgs = [
                    m for m in messages if m.get("role") == "system" or (m.get("role") in {"user", "assistant"} and not m.get("tool_calls"))
                ][:7]
                hist_limit = max(1, len(messages) - len(extra_messages))
                head_msgs = [
                    m for m in messages[:hist_limit] if m.get("role") in {"system", "user", "assistant"} and not m.get("tool_calls")
                ] or base_msgs[:1]
                tool_summaries = [
                    f"[Tool `{m.get('name', 'tool')}` result]:\n{str(m.get('content') or '')[:1800]}"
                    for m in extra_messages
                    if isinstance(m, dict) and m.get("role") == "tool"
                ]
                flat_user = (
                    f"User request: {prompt.strip()}\n\nExecuted tools:\n"
                    + "\n\n".join(tool_summaries[-6:])
                    + "\n\nInvoke the next tool call if more steps are needed, otherwise provide the final concise answer."
                )
                flat_payload = dict(payload)
                flat_payload["messages"] = [*head_msgs, {"role": "user", "content": flat_user}]
                http_ok, raw_calls, final_text = _send_chat_req(flat_payload)

            if not http_ok:
                return None

        parsed_calls: list[dict[str, Any]] = []
        for idx, rc in enumerate(raw_calls):
            cname = str(rc.get("name") or "").strip()
            if cname not in allowed_names:
                continue
            raw_arg = str(rc.get("arguments") or "").strip()
            cargs: dict[str, Any] = {}
            if raw_arg:
                try:
                    loaded = json.loads(raw_arg)
                    if isinstance(loaded, dict):
                        cargs = loaded
                except Exception:
                    cargs = {}
            parsed_calls.append(
                {
                    "id": str(rc.get("id") or f"local_call_{idx}"),
                    "name": cname,
                    "arguments": cargs,
                }
            )

        if parsed_calls:
            return {"text": "", "tool_calls": parsed_calls}
        if not final_text:
            return None
        return {"text": final_text, "tool_calls": parsed_calls}

    def try_local_cascade(
        self, user_message: str, system_instruction: str = ""
    ) -> dict[str, Any] | None:
        """3-tier local-first cascade: Tier 1 Specialist -> Tier 2 Reasoning -> None (escalate to Cloud)."""
        from . import local_mind as _lm

        msg = (user_message or "").strip()
        if not msg:
            return None
        low = msg.lower()
        models = [m for m in _lm.find_all_gguf_models() if _lm.classify_gguf_role(m) != "embedding"]
        if not models:
            return None

        _, inline_imgs = _lm.extract_inline_images(msg)
        if inline_imgs:
            vision_model = _lm.find_gguf_model("multimodal", strict=True) or _lm.find_gguf_model("multimodal")
            mmproj = _lm.find_mmproj_for_model(vision_model)
            if vision_model is not None and mmproj is not None:
                v_reply = self._run_single_gguf(
                    vision_model,
                    msg,
                    system_instruction,
                    mmproj=mmproj,
                    images=inline_imgs,
                )
                if v_reply and len(v_reply.strip()) >= 4:
                    return {"tier": "local:multimodal", "model": vision_model.name, "reply": v_reply}
            return None

        role = _lm.classify_task_role(msg)
        pred_tool, _, _score = self.predict_tool(msg)  # type: ignore[attr-defined]
        local_first = os.environ.get("SYSAGENT_LOCAL_FIRST", "1").strip().lower() not in {"0", "false", "no", "off"}
        neural_escalate = False
        try:
            from .neural import NeuralEngine

            if self.storage is not None:
                eng = NeuralEngine(self.storage)
                if int(eng.meta.get("epochs_trained", 0)) > 0:
                    unc = eng.predict_with_uncertainty(msg)
                    r_probs = eng.predict_role_probs(msg)
                    top_tool_prob = float(unc.get("tool_confidence", unc.get("confidence", 0.0)) or 0.0)
                    if top_tool_prob >= 0.35 and unc.get("tool") not in {None, "", "none"}:
                        neural_escalate = True
                    elif unc.get("uncertain"):
                        neural_escalate = True
                    elif not local_first and (
                        (r_probs.get("coding", 0.0) + r_probs.get("reasoning", 0.0)) >= 0.45
                    ):
                        neural_escalate = True
        except Exception:
            pass
        dyn_action = self.storage.get_learned_cues().get("action_req", []) if self.storage is not None else []
        has_action = (
            neural_escalate
            or pred_tool is not None
            or any(v in low for v in dyn_action)
            or any(ch in msg for ch in ("/", "~", "@", "\n"))
        )
        word_count = len(msg.split())
        if role == "chat":
            specialist = _lm.find_gguf_model("chat", strict=True) or _lm.find_gguf_model("chat") or models[0]
            eligible = word_count <= 14 and pred_tool is None and not has_action
        elif local_first and role in {"general", "coding", "reasoning"}:
            from .self_guard import get_calibration

            max_words = int(get_calibration(self.storage)["local_max_words"])
            specialist = _lm.find_gguf_model(role, strict=True)
            eligible = (
                specialist is not None
                and word_count <= max_words
                and pred_tool is None
                and not has_action
            )
        else:
            specialist = None
            eligible = False
        if not eligible or specialist is None:
            return None

        draft = self._run_single_gguf(specialist, msg, system_instruction)
        neural_uncertain = False
        if draft and self.storage is not None:
            try:
                from .neural import NeuralEngine

                neural_uncertain = NeuralEngine(self.storage).score_reply_stall(draft) >= 0.65
            except Exception:
                pass
        draft_ok = bool(
            draft
            and len(draft.strip()) >= 12
            and not neural_uncertain
            and "escalate_cloud" not in draft.lower()
        )
        if draft_ok:
            return {"tier": f"local:{role}", "model": specialist.name, "reply": draft}

        reasoner = _lm.find_gguf_model("reasoning", strict=True) or _lm.find_gguf_model("reasoning")
        if reasoner and reasoner != specialist:
            verify_prompt = (
                f"User Request:\n{msg}\n\n"
                f"Specialist ({specialist.stem}) Draft:\n{draft or '(none)'}\n\n"
                "If you can answer accurately and completely, write the final concise answer. "
                "If this requires cloud-level deep context or multi-step tools, reply ONLY with ESCALATE_CLOUD."
            )
            reasoned = self._run_single_gguf(reasoner, verify_prompt, system_instruction)
            if reasoned and "ESCALATE_CLOUD" not in reasoned and len(reasoned.strip()) >= 12:
                self.record_and_learn(msg, [], reasoned)  # type: ignore[attr-defined]
                return {"tier": "local:reasoning", "model": reasoner.name, "reply": reasoned}
        return None

    def synthesize_reply(
        self,
        user_text: str,
        tool_results: list[dict[str, Any]] | None = None,
        system_instruction: str = "",
    ) -> str:
        """Synthesize a clean plain-text response offline using GGUF (if available) or learned templates."""
        if tool_results:
            gguf_ans = self.run_gguf_inference(
                f"User asked: {user_text}\nTool results: {json.dumps(tool_results, ensure_ascii=False)}\n"
                "Provide a concise plain-text answer.",
                system_instruction=system_instruction,
            )
            if gguf_ans:
                return gguf_ans
            summaries: list[str] = []
            for item in tool_results:
                tname = item.get("name", "tool")
                raw_res = item.get("result", "")
                try:
                    data = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
                except Exception:
                    data = raw_res
                if tname == "system_status" and isinstance(data, dict):
                    temp_s = f", Temp {data['max_temp_c']}°C" if data.get("max_temp_c") is not None else ""
                    summaries.append(
                        f"[Local Offline Mode] CPU {data.get('cpu_percent')}%, "
                        f"RAM {data.get('ram_percent')}% ({data.get('ram_used_gb')}/{data.get('ram_total_gb')} GB), "
                        f"Disk {data.get('disk_percent')}% ({data.get('disk_free_gb')} GB free){temp_s}."
                    )
                elif tname == "list_directory" and isinstance(data, dict):
                    entries = [e.get("name", "") for e in (data.get("entries") or [])[:25]]
                    summaries.append(
                        f"[Local Offline Mode] Contents of {data.get('path')} ({len(entries)} items): "
                        + (", ".join(entries) if entries else "empty")
                    )
                elif tname == "read_file" and isinstance(data, dict):
                    summaries.append(
                        f"[Local Offline Mode] {data.get('path')}:\n{str(data.get('content', ''))[:800]}"
                    )
                else:
                    summaries.append(f"[Local Offline Mode] {tname}: {str(raw_res)[:600]}")
            return "\n".join(summaries)

        gguf_ans = self.run_gguf_inference(user_text, system_instruction=system_instruction)
        if gguf_ans:
            return gguf_ans

        similar = self.retrieve_similar_examples(user_text, top_k=1)  # type: ignore[attr-defined]
        if similar and similar[0].get("reply"):
            return f"[Local Offline Mode] {similar[0]['reply']}"

        return (
            "[Local Offline Mode] Internet or cloud model is currently unreachable. "
            "I am running on the local self-trained model and can still inspect system status, "
            "list/read/search files, check logs, control desktop settings, and search local memory."
        )

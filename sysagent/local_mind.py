"""Self-Training Local Model (Online Distillation + Neural/TF-IDF Router + llama.cpp GGUF Engine)."""

from __future__ import annotations

import difflib
import json
import math
import os
import re
import shutil
import subprocess
import threading
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .storage import Storage


DATASET_DIR = DATA_DIR / "dataset"
MODELS_DIR = DATA_DIR / "models"
DISTILLED_FILE = DATASET_DIR / "distilled_turns.jsonl"
UNCERTAIN_FILE = DATASET_DIR / "uncertain_queries.jsonl"
PREFERENCE_FILE = DATASET_DIR / "preference_pairs.jsonl"
WEIGHTS_FILE = MODELS_DIR / "local_mind_weights.json"
ROLE_ADAPTERS_FILE = MODELS_DIR / "role_adapters.json"
SEMANTIC_INDEX_FILE = MODELS_DIR / "semantic_index.json"
LORA_TRAIN_FILE = DATASET_DIR / "lora_chatml.txt"

SPECIALIST_ROLES: tuple[str, ...] = (
    "coding",
    "general",
    "reasoning",
    "chat",
    "multimodal",
    "embedding",
)


def _strip_think_tags(text: str) -> str:
    """Strip completed or truncated <think> reasoning blocks from model text output."""
    out = text or ""
    if "</think>" in out:
        out = out.split("</think>")[-1]
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
    out = re.sub(r"<think>[\s\S]*$", "", out)
    return out.strip()


class _ThinkStreamFilter:
    """Stateful token stream filter that strips <think>...</think> blocks without stalling normal text."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._emitted_any = False

    def feed(self, token: str) -> str:
        self._buf += token
        out: list[str] = []
        while self._buf:
            if self._in_think:
                end = self._buf.find(self._CLOSE)
                if end != -1:
                    self._buf = self._buf[end + len(self._CLOSE) :]
                    self._in_think = False
                    continue
                if len(self._buf) > len(self._CLOSE):
                    self._buf = self._buf[-len(self._CLOSE) :]
                break
            close_pos = self._buf.find(self._CLOSE)
            start = self._buf.find(self._OPEN)
            if close_pos != -1 and (start == -1 or close_pos < start):
                self._buf = self._buf[close_pos + len(self._CLOSE) :]
                out.clear()
                self._emitted_any = False
                continue
            if start != -1:
                if start > 0:
                    out.append(self._buf[:start])
                self._buf = self._buf[start + len(self._OPEN) :]
                self._in_think = True
                continue
            hold = 0
            max_check = min(len(self._buf), len(self._OPEN) - 1)
            for k in range(max_check, 0, -1):
                if self._OPEN.startswith(self._buf[-k:]):
                    hold = k
                    break
            emit_len = len(self._buf) - hold
            if emit_len > 0:
                out.append(self._buf[:emit_len])
                self._buf = self._buf[emit_len:]
            break
        chunk = "".join(out)
        if not self._emitted_any:
            chunk = chunk.lstrip()
            if chunk:
                self._emitted_any = True
        return chunk

    def flush(self) -> str:
        if self._in_think or not self._buf:
            self._buf = ""
            return ""
        chunk = self._buf
        self._buf = ""
        if not self._emitted_any:
            chunk = chunk.lstrip()
            if chunk:
                self._emitted_any = True
        return chunk

ROLE_DEFAULT_PRESETS: dict[str, str] = {
    "coding": "omnicoder-9b",
    "general": "deepseek-v4-qwen3.5-9b",
    "reasoning": "skillgym-qwen3.5-4b",
    "chat": "llama-3.1-8b",
    "multimodal": "mimo-v2.6-qwen-9b",
    "embedding": "nemotron-embed-1b",
}

GGUF_PRESETS: dict[str, dict[str, str]] = {
    "omnicoder-9b": {
        "filename": "OmniCoder-9B.Q4_K_M.gguf",
        "role": "coding",
        "hf_source": "cyankiwi/OmniCoder-9B-AWQ-4bit",
        "size_mb": "5500",
        "url": "https://huggingface.co/mradermacher/OmniCoder-9B-GGUF/resolve/main/OmniCoder-9B.Q4_K_M.gguf",
    },
    "deepseek-v4-qwen3.5-9b": {
        "filename": "DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q4_K_M.gguf",
        "role": "general",
        "hf_source": "groxaxo/DeepSeek-V4-Pro-Qwen3.5-9B-GPTQPro-W4-G64",
        "size_mb": "5600",
        "url": "https://huggingface.co/Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B-MTP-GGUF/resolve/main/DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q4_K_M.gguf",
    },
    "skillgym-qwen3.5-4b": {
        "filename": "SkillGym-Qwen3.5-4B.Q4_K_M.gguf",
        "role": "reasoning",
        "hf_source": "reasonwang/SkillGym-Qwen3.5-4B",
        "size_mb": "2500",
        "url": "https://huggingface.co/mradermacher/SkillGym-Qwen3.5-4B-GGUF/resolve/main/SkillGym-Qwen3.5-4B.Q4_K_M.gguf",
    },
    "llama-3.1-8b": {
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "role": "chat",
        "hf_source": "cyankiwi/Llama-3.1-8B-Instruct-AWQ-INT4",
        "size_mb": "4920",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    },
    "mimo-v2.6-qwen-9b": {
        "filename": "MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf",
        "role": "multimodal",
        "hf_source": "cyankiwi/MiMo-V2.6-Distill-Qwen-9B-AWQ-BF16-INT4",
        "size_mb": "5500",
        "url": "https://huggingface.co/bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF/resolve/main/MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf",
        "mmproj_filename": "mmproj-MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf",
        "mmproj_url": "https://huggingface.co/ggml-org/MiMo-V2.6-Distill-Qwen-9B-GGUF/resolve/main/mmproj-MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf",
    },
    "nemotron-embed-1b": {
        "filename": "llama-nemotron-embed-1b-v2.Q4_K_M.gguf",
        "role": "embedding",
        "hf_source": "groxaxo/Nemotron-3-Embed-1B-AWQ-W4A16",
        "size_mb": "810",
        "url": "https://huggingface.co/mradermacher/llama-nemotron-embed-1b-v2-GGUF/resolve/main/llama-nemotron-embed-1b-v2.Q4_K_M.gguf",
    },
    "qwen2.5-1.5b": {
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "role": "coding",
        "size_mb": "986",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    },
    "qwen3-1.7b": {
        "filename": "Qwen3-1.7B-Q4_K_M.gguf",
        "role": "coding",
        "size_mb": "1100",
        "url": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
    },
    "qwen2.5-3b": {
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "role": "coding",
        "size_mb": "1930",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
    },
    "gemma3-4b": {
        "filename": "gemma-3-4b-it-Q4_K_M.gguf",
        "role": "chat",
        "size_mb": "2490",
        "url": "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
    },
    "phi4-mini": {
        "filename": "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        "role": "reasoning",
        "size_mb": "2390",
        "url": "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
    },
}

DEFAULT_GGUF_URL = GGUF_PRESETS["qwen2.5-1.5b"]["url"]
DEFAULT_GGUF_NAME = GGUF_PRESETS["qwen2.5-1.5b"]["filename"]

TEACHER_MODELS: tuple[str, ...] = (
    "gemini-3.8-flash-low",
    "gemini-4-argon-low",
    "gemini-4-neon-low",
)

SEED_CURRICULUM: list[dict[str, Any]] = [
    {
        "user": "check system status cpu ram memory disk usage temperature health",
        "tool": "system_status",
        "args": {},
    },
    {
        "user": "list files in downloads directory folder",
        "tool": "list_directory",
        "args": {"path": "~/Downloads"},
    },
    {
        "user": "list files in home directory folder ls",
        "tool": "list_directory",
        "args": {"path": "~"},
    },
    {
        "user": "read file content show text cat",
        "tool": "read_file",
        "args": {"path": "~/README.md"},
    },
    {
        "user": "search files find by name or content",
        "tool": "search_files",
        "args": {"query": ""},
    },
    {
        "user": "check system journal logs service errors journalctl",
        "tool": "read_journal_logs",
        "args": {"lines": 40},
    },
    {
        "user": "list running processes top cpu ram memory ps",
        "tool": "list_processes",
        "args": {"sort_by": "cpu", "limit": 20},
    },
    {
        "user": "list systemd units services failed active status",
        "tool": "list_systemd_units",
        "args": {"state": "all", "limit": 40},
    },
    {
        "user": "what did brain do recently what rules exist",
        "tool": "inspect_self_component",
        "args": {"component": "brain"},
        "role": "reasoning",
    },
    {
        "user": "şuan ne düşünüyorsun beyin düşünceleri brain thoughts",
        "tool": "inspect_self_component",
        "args": {"component": "brain"},
        "role": "reasoning",
    },
    {
        "user": "cortex flow guardian uyku döngüsü ne durumda",
        "tool": "inspect_self_component",
        "args": {"component": "cortex"},
        "role": "reasoning",
    },
    {
        "user": "how is neural engine doing what is neural thinking",
        "tool": "inspect_self_component",
        "args": {"component": "neural"},
        "role": "reasoning",
    },
    {
        "user": "what did self evolve change recent proposals",
        "tool": "inspect_self_component",
        "args": {"component": "evolve"},
        "role": "reasoning",
    },
    {
        "user": "self guard drift eval breaker kota durumu",
        "tool": "inspect_self_component",
        "args": {"component": "guard"},
        "role": "general",
    },
    {
        "user": "brain cortex ne durumda kendi bileşenlerini incele",
        "tool": "inspect_self_component",
        "args": {"component": "all"},
        "role": "general",
    },
    {
        "user": "show recent background system events",
        "tool": "list_events",
        "args": {"limit": 20},
    },
    {
        "user": "recall saved memories facts what do you remember",
        "tool": "recall_memories",
        "args": {},
    },
    {
        "user": "search memory and past conversation history",
        "tool": "search_memory_and_history",
        "args": {"query": ""},
    },
    {
        "user": "undo last file action revert change",
        "tool": "undo_last_action",
        "args": {},
    },
    {
        "user": "undo batch actions revert all changes from turn",
        "tool": "undo_batch_actions",
        "args": {},
    },
    {
        "user": "set audio volume mute brightness wifi bluetooth desktop control",
        "tool": "desktop_control",
        "args": {"action": "volume_set", "value": "50"},
    },
    {
        "user": "start focus timer pomodoro countdown",
        "tool": "start_focus_timer",
        "args": {"minutes": 25},
    },
    {
        "user": "show application usage time stats today",
        "tool": "get_app_usage",
        "args": {},
    },
    {
        "user": "list active scheduled automations rules",
        "tool": "list_automations",
        "args": {},
    },
    {
        "user": "read current clipboard text content",
        "tool": "read_clipboard",
        "args": {},
    },
    {
        "user": "promptlar ne durumda prompt versions status",
        "tool": "inspect_self_component",
        "args": {"component": "prompts"},
    },
    {
        "user": "promptları ne kadar geliştirdin prompt registry history",
        "tool": "inspect_self_component",
        "args": {"component": "prompts"},
    },
    {
        "user": "check brain cortex neural evolve local mind self guard status",
        "tool": "inspect_self_component",
        "args": {"component": "all"},
    },
]


from .local_inference import (  # noqa: F401
    _has_ram_for_gguf as _has_ram_for_gguf,
    _release_all_warm_models as _release_all_warm_models,
    can_fit_gguf_model,
    choose_model_with_hysteresis,
    ensure_llama_server,
    get_llama_cpp_model,
    get_runtime_memory_status,
    get_warm_model_name,
    get_warm_model_path,
    unload_idle_servers as unload_idle_servers,
)

_EMBED_CACHE: dict[str, list[float]] = {}


def _load_embed_cache() -> dict[str, list[float]]:
    """Lazily populate `_EMBED_CACHE` from persistent `SEMANTIC_INDEX_FILE` on disk."""
    if _EMBED_CACHE or not SEMANTIC_INDEX_FILE.is_file():
        return _EMBED_CACHE
    try:
        raw = json.loads(SEMANTIC_INDEX_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for k, v in list(raw.items())[-384:]:
                if isinstance(k, str) and isinstance(v, list) and v:
                    _EMBED_CACHE[k] = [float(x) for x in v]
    except Exception:
        pass
    return _EMBED_CACHE


def _save_embed_cache() -> None:
    """Persist `_EMBED_CACHE` atomically to `SEMANTIC_INDEX_FILE`."""
    try:
        SEMANTIC_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        items = dict(list(_EMBED_CACHE.items())[-384:])
        tmp = SEMANTIC_INDEX_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, SEMANTIC_INDEX_FILE)
    except Exception:
        pass


def _dense_cosine(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    if dot <= 0.0:
        return 0.0
    na = math.sqrt(sum(a * a for a in vec_a))
    nb = math.sqrt(sum(b * b for b in vec_b))
    return (dot / (na * nb)) if (na > 0.0 and nb > 0.0) else 0.0


def _tokenize(text: str) -> list[str]:
    """Extract word tokens and character 3-grams for robust multilingual matching."""
    words = re.findall(r"\w{2,}", (text or "").lower())
    ngrams: list[str] = list(words)
    for w in words:
        if len(w) >= 4:
            for i in range(len(w) - 2):
                ngrams.append(f"#{w[i : i + 3]}")
    return ngrams


def _cosine_sim(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(v * vec_b.get(k, 0.0) for k, v in vec_a.items())
    if dot <= 0.0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def classify_gguf_role(path: Path) -> str:
    """Classify a GGUF model file into one of the 6 specialist roles (or mmproj adapter)."""
    low = path.name.lower()
    if "mmproj" in low:
        return "mmproj"
    if any(k in low for k in ("embed", "nomic", "bge")):
        return "embedding"
    if any(k in low for k in ("mimo", "-vl", "_vl", "llava")):
        return "multimodal"
    if any(k in low for k in ("skillgym", "phi", "deepseek-r1", "reason")):
        return "reasoning"
    if any(k in low for k in ("omnicoder", "coder", "qwen2.5-1.5b", "qwen2.5-3b", "qwen3-1.7b")):
        return "coding"
    if "deepseek-v4" in low:
        return "general"
    if any(k in low for k in ("llama-3", "gemma", "llama")):
        return "chat"
    return "general"


def classify_task_role(
    user_text: str,
    tool_calls: list[dict[str, Any]] | None = None,
    use_neural: bool = True,
    storage: Storage | None = None,
) -> str:
    """Route a user prompt or turn to its matching specialist role using structured tool calls and NeuralEngine."""
    t_names = {
        str(tc.get("name") or "")
        for tc in (tool_calls or [])
        if isinstance(tc, dict)
    }
    if "take_screenshot" in t_names:
        return "multimodal"
    if t_names & {"edit_file", "write_file", "run_command", "search_files", "forge_skill"}:
        return "coding"
    low = (user_text or "").strip().lower()
    if use_neural and len(low) >= 3:
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(storage)
            if int(eng.meta.get("epochs_trained", 0)) > 0:
                probs = eng.predict_role_probs(user_text)
                gen_probs = {r: p for r, p in probs.items() if r != "embedding"}
                if gen_probs:
                    best_role, best_p = max(gen_probs.items(), key=lambda kv: kv[1])
                    if best_p >= 0.42:
                        return best_role
        except Exception:
            pass
    if any(k in low for k in (".py", ".sh", ".json", ".toml", "traceback", "stack trace")):
        return "coding"
    return "general"


def find_all_gguf_models() -> list[Path]:
    """Return all available local .gguf model files (excluding mmproj adapters from generative pool)."""
    found: list[Path] = []
    env_path = os.environ.get("SYSAGENT_LOCAL_GGUF", "").strip()
    if env_path and Path(env_path).expanduser().is_file():
        found.append(Path(env_path).expanduser())
    if MODELS_DIR.is_dir():
        for p in sorted(MODELS_DIR.glob("*.gguf"), key=lambda x: x.stat().st_size, reverse=True):
            if "mmproj" in p.name.lower():
                continue
            if p not in found:
                found.append(p)
    return found


_ROLE_ALIASES: dict[str, set[str]] = {
    "tool_coder": {"coding", "tool_coder"},
    "synthesizer": {"chat", "general", "synthesizer"},
    "reasoner": {"reasoning", "reasoner"},
    "chat": {"chat", "general"},
    "general": {"general", "chat"},
    "coding": {"coding"},
    "reasoning": {"reasoning"},
    "multimodal": {"multimodal"},
    "embedding": {"embedding"},
}


_IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})
_GENERIC_GGUF_TOKENS: frozenset[str] = frozenset(
    {"mmproj", "gguf", "q2", "q3", "q4", "q5", "q6", "q8", "k", "m", "s", "l", "f16", "bf16", "iq", "0"}
)


def _gguf_name_tokens(path: Path) -> set[str]:
    stem = path.stem.lower()
    parts = [p for p in re.split(r"[-_\s]+", stem) if p]
    return {p for p in parts if p not in _GENERIC_GGUF_TOKENS and not (len(p) == 1 and p.isdigit())}


def find_mmproj_for_model(gguf: Path | None) -> Path | None:
    """Locate a matching `mmproj-*.gguf` vision projector for `gguf` in its folder or MODELS_DIR."""
    if gguf is None:
        return None
    dirs: list[Path] = []
    if gguf.parent.is_dir():
        dirs.append(gguf.parent)
    if MODELS_DIR.is_dir() and MODELS_DIR not in dirs:
        dirs.append(MODELS_DIR)
    candidates: list[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        for p in sorted(d.glob("*.gguf")):
            rp = p.resolve()
            if rp not in seen and "mmproj" in p.name.lower() and p.is_file():
                seen.add(rp)
                candidates.append(p)
    if not candidates:
        return None

    m_toks = _gguf_name_tokens(gguf)
    best: Path | None = None
    best_score = 0
    for cand in candidates:
        score = len(m_toks & _gguf_name_tokens(cand))
        if score > best_score:
            best_score = score
            best = cand
    if best is not None and best_score >= 2:
        return best
    if classify_gguf_role(gguf) == "multimodal" and len(candidates) == 1:
        return candidates[0]
    return None


def extract_inline_images(
    text: str, max_images: int = 3, max_bytes: int = 4 * 1024 * 1024
) -> tuple[str, list[dict[str, str]]]:
    """Extract `[Attached image `<path>`]` attachments into base64 payloads for vision GGUF inference."""
    import base64
    import mimetypes

    raw = text or ""
    if "[Attached image `" not in raw:
        return raw, []
    images: list[dict[str, str]] = []
    seen: set[Path] = set()
    for match in re.finditer(r"\[Attached image `([^`]+)`\]", raw):
        if len(images) >= max_images:
            break
        try:
            p = Path(match.group(1)).expanduser().resolve()
        except Exception:
            continue
        if p in seen or not p.is_file():
            continue
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        try:
            data = p.read_bytes()
        except Exception:
            continue
        if not data or len(data) > max_bytes:
            continue
        seen.add(p)
        mime, _ = mimetypes.guess_type(str(p))
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        images.append(
            {
                "path": str(p),
                "mime_type": mime,
                "data": base64.b64encode(data).decode("ascii"),
            }
        )
    return raw, images


def find_gguf_model(preferred_role: str | None = None, strict: bool = False) -> Path | None:
    """Locate the best local .gguf model file (optionally matching a specialist role)."""
    models = find_all_gguf_models()
    if not models:
        return None
    if preferred_role:
        accepted = {preferred_role} if strict else _ROLE_ALIASES.get(preferred_role, {preferred_role})
        for m in models:
            if classify_gguf_role(m) in accepted:
                return m
        if strict or preferred_role == "embedding":
            return None
    generative = [m for m in models if classify_gguf_role(m) != "embedding"]
    return generative[0] if generative else None


def prune_superseded_models(dry_run: bool = False) -> list[str]:
    """Remove smaller duplicate .gguf files when a larger specialist for the same role is installed."""
    by_role: dict[str, list[Path]] = defaultdict(list)
    for p in find_all_gguf_models():
        by_role[classify_gguf_role(p)].append(p)
    removed: list[str] = []
    for files in by_role.values():
        if len(files) <= 1:
            continue
        for old_p in files[1:]:
            removed.append(old_p.name)
            if not dry_run:
                old_p.unlink(missing_ok=True)
    return removed


_WEIGHTS_CACHE: dict[tuple[str, int], dict[str, Any]] = {}
_WEIGHTS_LOCK = threading.RLock()


def _save_weights_file(weights: dict[str, Any]) -> None:
    """Persist weights JSON atomically and refresh the in-process mtime cache."""
    with _WEIGHTS_LOCK:
        tmp = WEIGHTS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(weights, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, WEIGHTS_FILE)
        try:
            mtime_ns = WEIGHTS_FILE.stat().st_mtime_ns
            _WEIGHTS_CACHE.clear()
            _WEIGHTS_CACHE[(str(WEIGHTS_FILE), mtime_ns)] = weights
        except Exception:
            pass


from .local_distill import _LocalDistillMixin
from .local_gguf import _LocalGGUFMixin


class LocalMind(_LocalDistillMixin, _LocalGGUFMixin):
    """Self-training local model that harvests teacher turns, updates weights, and runs offline inference."""

    def __init__(self, storage: Storage | None = None) -> None:
        self.storage = storage
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.weights: dict[str, Any] = {}
        self.weights = self._load_or_init_weights()

    def _load_or_init_weights(self) -> dict[str, Any]:
        if WEIGHTS_FILE.is_file():
            try:
                with _WEIGHTS_LOCK:
                    key = (str(WEIGHTS_FILE), WEIGHTS_FILE.stat().st_mtime_ns)
                    cached = _WEIGHTS_CACHE.get(key)
                    if cached is not None:
                        return cached
                    data = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "tool_centroids" in data:
                        _WEIGHTS_CACHE.clear()
                        _WEIGHTS_CACHE[key] = data
                        return data
            except Exception:
                pass
        self.train()
        return self.weights

    def record_and_learn(
        self,
        user_message: str,
        tool_calls: list[dict[str, Any]],
        assistant_reply: str,
        role: str | None = None,
    ) -> None:
        """Harvest a successful online turn into the local distillation dataset and incrementally update weights."""
        u_clean = (user_message or "").strip()
        a_clean = (assistant_reply or "").strip()
        if (
            not u_clean
            or not a_clean
            or u_clean.startswith(("[Automation ", "A background system alert was triggered", "Regarding this situation:"))
        ):
            return
        resolved_role = (
            role
            if role in {"coding", "general", "reasoning", "chat", "multimodal"}
            else classify_task_role(u_clean, tool_calls)
        )
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": resolved_role,
            "user": u_clean[:1000],
            "tool_calls": tool_calls[:4],
            "reply": a_clean[:2000],
        }
        try:
            with DISTILLED_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._update_weights_incremental(entry)
        except Exception:
            pass

    def _load_distilled_entries(self, limit: int = 1500) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if DISTILLED_FILE.is_file():
            try:
                lines = DISTILLED_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
                for ln in lines[-limit:]:
                    if ln.strip():
                        obj = json.loads(ln)
                        if isinstance(obj, dict) and obj.get("user"):
                            entries.append(obj)
            except Exception:
                pass
        return entries

    def _update_weights_incremental(self, entry: dict[str, Any]) -> None:
        """Fast online perceptron/centroid update after a single turn."""
        tokens = _tokenize(entry["user"])
        if not tokens:
            return
        centroids: dict[str, dict[str, float]] = self.weights.setdefault("tool_centroids", {})
        tcalls = entry.get("tool_calls") or []
        target_cls = tcalls[0]["name"] if tcalls and tcalls[0].get("name") else "__chat__"
        c_vec = centroids.setdefault(target_cls, {})
        lr = 0.25
        for tok in tokens:
            c_vec[tok] = round(c_vec.get(tok, 0.0) + lr, 4)
        self.weights["samples_trained"] = int(self.weights.get("samples_trained", 0)) + 1
        role = str(entry.get("role") or classify_task_role(entry["user"], tcalls))
        rc = self.weights.setdefault("role_counts", {})
        rc[role] = int(rc.get(role, 0)) + 1
        self.weights["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            _save_weights_file(self.weights)
        except Exception:
            pass
        if tcalls and isinstance(tcalls[0], dict) and tcalls[0].get("name"):
            try:
                from .neural import NeuralEngine

                eng_n = NeuralEngine(self.storage)
                eng_n.online_step(entry["user"], str(tcalls[0]["name"]), reward=1.0)
                if len(tcalls) >= 2 and isinstance(tcalls[1], dict) and tcalls[1].get("name"):
                    eng_n.online_transition_step(entry["user"], str(tcalls[0]["name"]), str(tcalls[1]["name"]))
            except Exception:
                pass

    def penalize_tool(self, user_text: str, wrong_tool: str) -> None:
        """Down-weight a wrongly chosen tool for the given query tokens after user correction."""
        tokens = _tokenize(user_text)
        if not tokens or not wrong_tool:
            return
        centroids: dict[str, dict[str, float]] = self.weights.setdefault("tool_centroids", {})
        c_vec = centroids.get(wrong_tool)
        if not c_vec:
            return
        lr = 0.35
        for tok in tokens:
            if tok in c_vec:
                new_val = max(0.0, round(c_vec[tok] - lr, 4))
                if new_val == 0.0:
                    c_vec.pop(tok, None)
                else:
                    c_vec[tok] = new_val
        self.weights["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            _save_weights_file(self.weights)
        except Exception:
            pass
        try:
            from .neural import NeuralEngine

            NeuralEngine(self.storage).online_step(user_text, wrong_tool, reward=-1.0)
        except Exception:
            pass

    def prune_failed_arg_template(self, tool_name: str, bad_args: dict[str, Any]) -> int:
        """Remove a failing argument template for `tool_name` from learned_arg_templates."""
        if not tool_name or not isinstance(bad_args, dict) or not bad_args:
            return 0
        templates_map = self.weights.get("learned_arg_templates")
        if not isinstance(templates_map, dict):
            return 0
        bucket = templates_map.get(tool_name)
        if not isinstance(bucket, list) or not bucket:
            return 0
        kept = [item for item in bucket if not (isinstance(item, dict) and item.get("args") == bad_args)]
        removed = len(bucket) - len(kept)
        if removed > 0:
            templates_map[tool_name] = kept
            self.weights["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                _save_weights_file(self.weights)
            except Exception:
                pass
        return removed

    def record_arg_repair(
        self,
        user_text: str,
        failed_tool: str,
        failed_args: dict[str, Any],
        error_snippet: str,
        working_tool: str,
        working_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Learn from an in-turn tool error recovery by pruning bad args, saving good args, and logging a lesson."""
        pruned = self.prune_failed_arg_template(failed_tool, failed_args)
        toks = _tokenize(user_text)[:16]
        if working_tool and isinstance(working_args, dict) and working_args:
            templates_map = self.weights.setdefault("learned_arg_templates", {})
            bucket: list[dict[str, Any]] = templates_map.setdefault(working_tool, [])
            bucket[:] = [it for it in bucket if not (isinstance(it, dict) and it.get("args") == working_args)]
            bucket.append({"tokens": toks, "args": dict(working_args)})
            if len(bucket) > 10:
                del bucket[:-10]
            self.weights["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                _save_weights_file(self.weights)
            except Exception:
                pass
        if failed_tool and working_tool and failed_tool != working_tool:
            self.penalize_tool(user_text, failed_tool)
        try:
            from .neural import NeuralEngine

            NeuralEngine(self.storage).online_step(user_text, working_tool, reward=1.3)
        except Exception:
            pass
        if self.storage is not None and failed_tool and working_tool:
            try:
                self.storage.add_lesson(
                    f"arg_repair:{failed_tool}->{working_tool}",
                    f"Failed {failed_tool}({json.dumps(failed_args or {}, ensure_ascii=False)[:100]}): {(error_snippet or '')[:80]}",
                    f"Use {working_tool} with {json.dumps(working_args or {}, ensure_ascii=False)[:120]}",
                )
            except Exception:
                pass
        if failed_tool and working_tool:
            self.record_preference_pair(
                user_text,
                chosen_tool=working_tool,
                chosen_args=dict(working_args or {}),
                rejected_tool=failed_tool,
                rejected_args=dict(failed_args or {}),
                reason=error_snippet[:120],
            )
        return {
            "repaired": True,
            "from_tool": failed_tool,
            "to_tool": working_tool,
            "pruned_templates": pruned,
        }

    def record_preference_pair(
        self,
        user_text: str,
        chosen_tool: str,
        chosen_args: dict[str, Any],
        rejected_tool: str,
        rejected_args: dict[str, Any],
        reason: str = "",
    ) -> dict[str, Any]:
        """Record an Online DPO / Contrastive preference pair (chosen vs rejected action) and update centroids & NeuralEngine."""
        u_clean = (user_text or "").strip()
        if not u_clean or not chosen_tool or not rejected_tool:
            return {"recorded": False}
        pair = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": u_clean[:400],
            "chosen": {"tool": chosen_tool, "args": dict(chosen_args or {})},
            "rejected": {"tool": rejected_tool, "args": dict(rejected_args or {})},
            "reason": (reason or "")[:160],
        }
        try:
            PREFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with PREFERENCE_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        except Exception:
            pass

        toks = _tokenize(u_clean)
        if toks and chosen_tool != rejected_tool:
            centroids: dict[str, dict[str, float]] = self.weights.setdefault("tool_centroids", {})
            c_vec = centroids.setdefault(chosen_tool, {})
            r_vec = centroids.get(rejected_tool) or {}
            for tok in toks:
                c_vec[tok] = round(c_vec.get(tok, 0.0) + 0.30, 4)
                if tok in r_vec:
                    nv = max(0.0, round(r_vec[tok] - 0.30, 4))
                    if nv == 0.0:
                        r_vec.pop(tok, None)
                    else:
                        r_vec[tok] = nv
            try:
                _save_weights_file(self.weights)
            except Exception:
                pass

        dpo_res: dict[str, Any] = {}
        if chosen_tool != rejected_tool:
            try:
                from .neural import NeuralEngine

                dpo_res = NeuralEngine(self.storage).online_preference_step(
                    u_clean, chosen_tool, rejected_tool
                )
            except Exception:
                pass
        return {"recorded": True, "pair": pair, "neural_dpo": dpo_res}

    def retrieve_contrastive_hints(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        """Retrieve top-K matching `(chosen vs rejected)` preference pairs for Contrastive Few-Shot guidance."""
        if not PREFERENCE_FILE.is_file():
            return []
        q_toks = _tokenize(query)
        if not q_toks:
            return []
        idf: dict[str, float] = self.weights.get("idf", {})
        q_vec = {t: cnt * idf.get(t, 1.0) for t, cnt in Counter(q_toks).items()}
        scored: list[tuple[float, dict[str, Any]]] = []
        try:
            lines = PREFERENCE_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
            for ln in lines:
                if not ln.strip():
                    continue
                item = json.loads(ln)
                if not isinstance(item, dict) or not item.get("user"):
                    continue
                u_vec = {
                    t: cnt * idf.get(t, 1.0)
                    for t, cnt in Counter(_tokenize(str(item["user"]))).items()
                }
                sim = _cosine_sim(q_vec, u_vec)
                if sim >= 0.18:
                    scored.append((sim, item))
        except Exception:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:top_k]]

    def local_actor_critic_review(
        self,
        user_request: str,
        tool_name: str,
        tool_args: dict[str, Any],
        invoke_llm_critic: bool = False,
    ) -> dict[str, Any]:
        """Critique a proposed tool call using schema rules, contrastive preference memory, and the 4B reasoning slot."""
        schema_map = self._build_tool_schema_map()
        spec = schema_map.get(tool_name)
        if spec is None:
            return {
                "approved": False,
                "risk": "high",
                "critique": f"Unknown tool '{tool_name}'.",
                "suggested_tool": None,
                "suggested_args": {},
            }
        req = spec.get("required") or []
        props = spec.get("properties") or {}
        raw_args = dict(tool_args or {})
        missing = [rk for rk in req if rk not in raw_args or raw_args.get(rk) in {None, ""}]
        if missing:
            repaired_args = {k: v for k, v in raw_args.items() if k in props}
            if len(req) == 1 and req[0] in missing:
                unknown_vals = [
                    v for k, v in raw_args.items() if k not in props and v is not None and str(v).strip() != ""
                ]
                if len(unknown_vals) == 1:
                    repaired_args[req[0]] = unknown_vals[0]
            return {
                "approved": False,
                "risk": "medium",
                "critique": f"Missing required parameter(s) for '{tool_name}': {missing}",
                "suggested_tool": tool_name,
                "suggested_args": repaired_args,
            }

        for k, val in raw_args.items():
            pdef = props.get(k)
            if (
                isinstance(pdef, dict)
                and isinstance(pdef.get("enum"), list)
                and pdef["enum"]
                and val not in pdef["enum"]
            ):
                repaired_enum_args = dict(raw_args)
                if isinstance(val, str):
                    enum_map = {str(e).lower(): e for e in pdef["enum"]}
                    close = difflib.get_close_matches(val.strip().lower(), list(enum_map.keys()), n=1, cutoff=0.6)
                    if close:
                        repaired_enum_args[k] = enum_map[close[0]]
                return {
                    "approved": False,
                    "risk": "medium",
                    "critique": f"Invalid enum value '{val}' for '{tool_name}.{k}' (allowed: {pdef['enum']}).",
                    "suggested_tool": tool_name,
                    "suggested_args": repaired_enum_args,
                }

        for pair in self.retrieve_contrastive_hints(user_request, top_k=2):
            rej = pair.get("rejected") or {}
            cho = pair.get("chosen") or {}
            if rej.get("tool") == tool_name and (not rej.get("args") or rej.get("args") == raw_args):
                return {
                    "approved": False,
                    "risk": "medium",
                    "critique": f"Matches previously rejected action ({pair.get('reason') or 'past failure'}). Prefer {cho.get('tool')}.",
                    "suggested_tool": cho.get("tool"),
                    "suggested_args": dict(cho.get("args") or {}),
                }

        reasoner = find_gguf_model("reasoning", strict=True) if invoke_llm_critic else None
        if reasoner is not None and tool_name in {"edit_file", "write_file", "run_command", "trash_file"}:
            critic_prompt = json.dumps(
                {
                    "task": "Review whether this tool call safely and accurately satisfies user_request. Respond ONLY JSON: {\"approved\": true/false, \"risk\": \"low|medium|high\", \"critique\": \"...\"}",
                    "user_request": (user_request or "")[:300],
                    "tool": tool_name,
                    "args": tool_args,
                },
                ensure_ascii=False,
            )
            raw = self._run_single_gguf(
                reasoner,
                critic_prompt,
                extra_system='Respond ONLY with JSON: {"approved": true, "risk": "low", "critique": "ok"}',
                max_tokens=64,
            )
            if isinstance(raw, str) and "{" in raw:
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(raw[raw.find("{") :])
                    if isinstance(parsed, dict) and isinstance(parsed.get("approved"), bool):
                        return {
                            "approved": bool(parsed["approved"]),
                            "risk": str(parsed.get("risk") or "low"),
                            "critique": str(parsed.get("critique") or ""),
                            "suggested_tool": tool_name,
                            "suggested_args": dict(tool_args or {}),
                        }
                except Exception:
                    pass

        return {
            "approved": True,
            "risk": "low",
            "critique": "passed_schema_and_contrastive_checks",
            "suggested_tool": tool_name,
            "suggested_args": dict(tool_args or {}),
        }

    def train(self, run_lora: bool = False) -> dict[str, Any]:
        """Full training pass over seed curriculum + distilled JSONL + SQLite chat history, rules, and lessons."""
        samples: list[dict[str, Any]] = []
        for s in SEED_CURRICULUM:
            rec: dict[str, Any] = {
                "user": s["user"],
                "tool_calls": [{"name": s["tool"], "arguments": s["args"]}],
                "reply": "",
                "weight": 6.0,
            }
            if s.get("role"):
                rec["role"] = s["role"]
            samples.append(rec)

        distilled = self._load_distilled_entries(limit=1500)
        samples.extend(distilled)

        if self.storage is not None:
            try:
                for sess in self.storage.list_sessions(limit=25):
                    msgs = self.storage.get_messages(sess["session_id"], limit=40)
                    for i in range(len(msgs) - 1):
                        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                            samples.append(
                                {
                                    "user": msgs[i]["content"],
                                    "tool_calls": [],
                                    "reply": msgs[i + 1]["content"],
                                }
                            )
            except Exception:
                pass

        doc_freq: Counter[str] = Counter()
        class_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        role_counts: Counter[str] = Counter()
        qa_memory: list[dict[str, Any]] = []
        learned_arg_templates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        tool_transitions: dict[str, Counter[str]] = defaultdict(Counter)

        for smp in samples:
            u_text = str(smp.get("user") or "").strip()
            toks = _tokenize(u_text)
            if not toks:
                continue
            for ut in set(toks):
                doc_freq[ut] += 1
            tcalls = smp.get("tool_calls") or []
            role = str(smp.get("role") or classify_task_role(u_text, tcalls, use_neural=False, storage=self.storage))
            role_counts[role] += 1
            cls_name = tcalls[0]["name"] if (tcalls and isinstance(tcalls[0], dict) and tcalls[0].get("name")) else "__chat__"
            if (cls_name != "__chat__" or not smp.get("reply")) and not u_text.startswith(
                ("A background system alert was triggered", "Regarding this situation:")
            ):
                w = float(smp.get("weight", 1.0))
                for t in toks:
                    class_token_counts[cls_name][t] += w
            for idx_tc, tc in enumerate(tcalls):
                if not isinstance(tc, dict) or not tc.get("name"):
                    continue
                tc_name = str(tc["name"])
                tc_args = tc.get("arguments")
                if isinstance(tc_args, dict) and tc_args:
                    bucket = learned_arg_templates[tc_name]
                    if not any(ex.get("args") == tc_args for ex in bucket):
                        bucket.append({"tokens": toks[:16], "args": dict(tc_args)})
                        if len(bucket) > 10:
                            del bucket[0]
                if idx_tc + 1 < len(tcalls) and isinstance(tcalls[idx_tc + 1], dict) and tcalls[idx_tc + 1].get("name"):
                    nxt_name = str(tcalls[idx_tc + 1]["name"])
                    if nxt_name != tc_name:
                        tool_transitions[tc_name][nxt_name] += 1
            if smp.get("reply"):
                qa_memory.append(
                    {
                        "role": role,
                        "user": u_text[:300],
                        "tool_calls": tcalls[:2],
                        "reply": str(smp["reply"])[:600],
                    }
                )

        if self.storage is not None:
            try:
                for wf in self.storage.list_workflows(limit=50):
                    parts = [p.strip() for p in str(wf.get("signature") or "").split("->") if p.strip()]
                    hits_w = max(1, int(wf.get("hits") or 1))
                    for i in range(len(parts) - 1):
                        if parts[i] != parts[i + 1]:
                            tool_transitions[parts[i]][parts[i + 1]] += hits_w
            except Exception:
                pass

        total_docs = max(1, len(samples))
        idf: dict[str, float] = {
            tok: round(math.log((total_docs + 1.0) / (df + 1.0)) + 1.0, 4)
            for tok, df in doc_freq.items()
            if df >= 1
        }

        tool_centroids: dict[str, dict[str, float]] = {}
        for cls_name, counter in class_token_counts.items():
            total_w = sum(counter.values()) or 1.0
            vec = {
                tok: round((cnt / total_w) * idf.get(tok, 1.0), 4)
                for tok, cnt in counter.most_common(450)
            }
            tool_centroids[cls_name] = vec

        chatml_lines: list[str] = []
        for item in qa_memory[-400:]:
            chatml_lines.append(
                f"<|im_start|>user\n{item['user']}<|im_end|>\n"
                f"<|im_start|>assistant\n{item['reply']}<|im_end|>"
            )
        role_hints = {
            "coding": "Role=coding: emit accurate tool calls or minimal tested diffs without guesswork.",
            "general": "Role=general: combine system inspection and clear explanations concisely.",
            "reasoning": "Role=reasoning: analyze root causes, rules, and edge cases step by step.",
            "chat": "Role=chat: reply naturally, concisely, and match the user's language.",
            "multimodal": "Role=multimodal: analyze visual/screenshot inputs carefully and extract exact facts.",
            "embedding": "Role=embedding: produce normalized semantic vectors for retrieval.",
        }
        role_adapters: dict[str, dict[str, Any]] = {}
        for role_name in SPECIALIST_ROLES:
            role_examples = [it for it in qa_memory if it.get("role") == role_name][-6:]
            role_adapters[role_name] = {
                "system_hint": role_hints.get(role_name, ""),
                "sample_count": int(role_counts.get(role_name, 0)),
                "examples": role_examples,
            }

        lora_status = f"adapters_ready ({len(role_adapters)} roles)"
        last_lora_samples = len(samples)
        last_lora_attempt = int(self.weights.get("last_lora_attempt_samples", 0))
        gguf_path = find_gguf_model()
        is_quantized = bool(
            gguf_path and any(q in gguf_path.name.upper() for q in ("Q4_", "Q5_", "Q8_", "IQ"))
        )
        prev_cb_hash = self.weights.get("codebase_train_hash")
        prev_weights = dict(self.weights) if self.weights.get("tool_centroids") else {}
        self.weights = {
            "version": 1,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "samples_trained": len(samples),
            "last_lora_samples": last_lora_samples,
            "last_lora_attempt_samples": last_lora_attempt,
            "distilled_count": len(distilled),
            "vocab_size": len(idf),
            "classes": sorted(tool_centroids.keys()),
            "idf": idf,
            "tool_centroids": tool_centroids,
            "qa_memory": qa_memory[-150:],
            "role_counts": dict(role_counts),
            "learned_arg_templates": dict(learned_arg_templates),
            "tool_transitions": {k: dict(v) for k, v in tool_transitions.items()},
            "lora_status": lora_status,
        }
        if prev_cb_hash:
            self.weights["codebase_train_hash"] = prev_cb_hash

        regressed = False
        try:
            from .self_resilience import evaluate_self_gate

            gate = evaluate_self_gate(self, self.storage)
            if gate.get("regressed") and prev_weights:
                regressed = True
                self.weights = prev_weights
                if self.storage is not None:
                    self.storage.add_brain_thought(
                        "warn", 90, f"SelfEvalGate rolled back LocalMind weights (score={gate.get('score')} < baseline={gate.get('baseline')})"
                    )
        except Exception:
            pass

        if not regressed:
            if chatml_lines:
                try:
                    LORA_TRAIN_FILE.write_text("\n".join(chatml_lines) + "\n", encoding="utf-8")
                except Exception:
                    pass
            try:
                ROLE_ADAPTERS_FILE.write_text(json.dumps(role_adapters, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            if run_lora and gguf_path and not is_quantized and shutil.which("llama-finetune") and chatml_lines:
                last_lora_attempt = len(samples)
                self.weights["last_lora_attempt_samples"] = last_lora_attempt
                lora_out = MODELS_DIR / "sysagent_lora.bin"
                try:
                    proc = subprocess.run(
                        [
                            "llama-finetune",
                            "-m",
                            str(gguf_path),
                            "-f",
                            str(LORA_TRAIN_FILE),
                            "-o",
                            str(lora_out),
                            "-epochs",
                            "1",
                            "-t",
                            "4",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=240,
                    )
                    if proc.returncode == 0:
                        lora_status = "trained"
                        last_lora_samples = len(samples)
                        self.weights["lora_status"] = lora_status
                        self.weights["last_lora_samples"] = last_lora_samples
                    else:
                        self.weights["lora_status"] = f"error:{proc.returncode}"
                except Exception as exc:
                    self.weights["lora_status"] = f"error:{exc}"
            try:
                from .neural import NeuralEngine

                NeuralEngine(self.storage).train(samples=samples)
            except Exception:
                pass

        try:
            _save_weights_file(self.weights)
        except Exception:
            pass
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        gguf_list = find_all_gguf_models()
        gen_list = [p for p in gguf_list if classify_gguf_role(p) != "embedding"]
        specialist_roles: dict[str, dict[str, Any]] = {}
        for role in SPECIALIST_ROLES:
            preset_key = ROLE_DEFAULT_PRESETS[role]
            spec = GGUF_PRESETS[preset_key]
            matched = find_gguf_model(role, strict=True)
            specialist_roles[role] = {
                "preset": preset_key,
                "hf_source": spec.get("hf_source", ""),
                "filename": spec["filename"],
                "installed": matched is not None,
                "active_file": matched.name if matched else None,
            }
        uncertain_cnt = 0
        if UNCERTAIN_FILE.is_file():
            try:
                uncertain_cnt = len(
                    [ln for ln in UNCERTAIN_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
                )
            except Exception:
                pass
        mm_model = find_gguf_model("multimodal", strict=True) or find_gguf_model("multimodal")
        mmproj_path = find_mmproj_for_model(mm_model)
        rt_mem = get_runtime_memory_status()
        return {
            "weights_file": str(WEIGHTS_FILE),
            "dataset_file": str(DISTILLED_FILE),
            "updated_at": self.weights.get("updated_at"),
            "samples_trained": self.weights.get("samples_trained", 0),
            "distilled_count": self.weights.get("distilled_count", 0),
            "vocab_size": self.weights.get("vocab_size", 0),
            "learned_classes": len(self.weights.get("classes", [])),
            "role_counts": self.weights.get("role_counts", {}),
            "weakest_role": self.weakest_specialist_role(),
            "specialist_roles": specialist_roles,
            "gguf_model": str(gen_list[0]) if gen_list else None,
            "gguf_ensemble_pool": [
                {"name": p.name, "role": classify_gguf_role(p)} for p in gguf_list
            ],
            "ensemble_active": len(gen_list) >= 2,
            "uncertain_queue_count": uncertain_cnt,
            "teacher_models": list(TEACHER_MODELS),
            "available_presets": list(GGUF_PRESETS.keys()),
            "llama_cli_available": bool(shutil.which("llama-cli")),
            "llama_cpp_available": __import__("importlib.util").util.find_spec("llama_cpp") is not None,
            "local_backend": os.environ.get("SYSAGENT_LOCAL_BACKEND", "auto").strip().lower() or "auto",
            "llama_finetune_available": bool(shutil.which("llama-finetune")),
            "lora_status": self.weights.get("lora_status", "none"),
            "warm_model": get_warm_model_name(),
            "vision_ready": bool(mm_model is not None and mmproj_path is not None),
            "mmproj_file": mmproj_path.name if mmproj_path is not None else None,
            "runtime": rt_mem,
            "neural": self._neural_status(),
        }

    def prewarm_for_routine(self, watcher: Any = None, cortex: Any = None) -> dict[str, Any]:
        """Preload the likely GGUF specialist model if user routine and free VRAM/RAM headroom permit."""
        if os.environ.get("SYSAGENT_ROUTINE_PRELOAD", "true").strip().lower() in {"0", "false", "no", "off"}:
            return {"status": "disabled"}
        if cortex is not None:
            flow = getattr(cortex, "last_flow", None) or {}
            mode = str(flow.get("mode") or "normal")
            if flow.get("suppress_low_priority") or mode in {"deep_focus", "meeting_media"}:
                return {"status": "deferred", "reason": f"cortex_flow:{mode}"}
        warm = get_warm_model_path()
        if warm is not None:
            return {"status": "kept_warm", "model": warm.name}
        if self.storage is None:
            return {"status": "skipped", "reason": "no_storage"}

        from .self_learn import predict_user_routine

        routine = predict_user_routine(self.storage)
        if not routine:
            return {"status": "skipped", "reason": "no_routine"}
        role = str(routine.get("likely_role") or "general")
        top_app = str(routine.get("top_app") or "").lower()
        if role == "general" and any(
            k in top_app for k in ("code", "cursor", "nvim", "vim", "pycharm", "kitty", "alacritty", "konsole", "tmux")
        ):
            role = "coding"

        if watcher is not None and hasattr(watcher, "get_health_snapshot"):
            try:
                snap = watcher.get_health_snapshot() or {}
                if float(snap.get("cpu_percent", 0.0) or 0.0) >= 65.0:
                    return {"status": "deferred", "reason": "high_cpu"}
            except Exception:
                pass

        cand = find_gguf_model(role, strict=True) or find_gguf_model(role)
        if cand is None:
            return {"status": "skipped", "reason": f"no_model:{role}"}
        mm = find_mmproj_for_model(cand) if role == "multimodal" else None
        fit_ok, fit_mode = can_fit_gguf_model(cand, mmproj=mm, min_ram_gb=3.0, min_vram_headroom_mb=768)
        if not (fit_ok and fit_mode in {"full_vram", "ram_only"}):
            return {
                "status": "skipped",
                "reason": fit_mode if not fit_ok else f"fit:{fit_mode}",
                "model": cand.name,
            }
        srv = self._ensure_llama_server(cand, mmproj=mm)
        return {
            "status": "prewarmed" if srv else "failed",
            "model": cand.name,
            "role": role,
            "fit_mode": fit_mode,
        }

    def _neural_status(self) -> dict[str, Any]:
        try:
            from .neural import NeuralEngine

            return NeuralEngine(self.storage).get_status()
        except Exception:
            return {}

    def predict_tool(
        self,
        user_text: str,
        allowed_names: set[str] | None = None,
        record_uncertain: bool = True,
    ) -> tuple[str | None, dict[str, Any], float]:
        """Predict the best tool and extract arguments using the trained local weights."""
        toks = _tokenize(user_text)
        if not toks:
            return None, {}, 0.0
        idf: dict[str, float] = self.weights.get("idf", {})
        q_counts = Counter(toks)
        q_vec = {t: cnt * idf.get(t, 1.0) for t, cnt in q_counts.items()}

        best_tool: str | None = None
        best_score = 0.0
        centroids: dict[str, dict[str, float]] = self.weights.get("tool_centroids", {})
        neural_probs: dict[str, float] = {}
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(self.storage)
            if int(eng.meta.get("epochs_trained", 0)) > 0:
                neural_probs = eng.predict_tool_probs(user_text)
        except Exception:
            neural_probs = {}

        candidates = set(centroids.keys()) | set(neural_probs.keys())
        for cls_name in candidates:
            if cls_name == "__chat__":
                continue
            if allowed_names is not None and cls_name not in allowed_names:
                continue
            lex_score = _cosine_sim(q_vec, centroids[cls_name]) if cls_name in centroids else 0.0
            if neural_probs:
                n_prob = float(neural_probs.get(cls_name, 0.0))
                score = (0.55 * lex_score + 0.45 * n_prob) if cls_name in centroids else n_prob
            else:
                score = lex_score
            if score > best_score:
                best_score = score
                best_tool = cls_name

        if record_uncertain and 0.04 < best_score < 0.24:
            self._record_uncertain_query(user_text, best_score)
        if not best_tool or best_score < 0.16:
            return None, {}, best_score
        return best_tool, self.extract_arguments(best_tool, user_text), round(best_score, 3)

    def extract_arguments(self, tool_name: str, user_text: str) -> dict[str, Any]:
        """Heuristic + pattern argument extractor for predicted tools."""
        text = user_text.strip()
        low = text.lower()
        path_match = re.search(r"(~\/[^\s'\"]+|\/[a-zA-Z0-9_\-\.\/]+)", text)

        if tool_name == "list_directory":
            if path_match:
                return {"path": path_match.group(1)}
            for alias, target in (
                ("download", "~/Downloads"),
                ("document", "~/Documents"),
                ("desktop", "~/Desktop"),
            ):
                if alias in low:
                    return {"path": target}
            return {"path": "~"}

        if tool_name == "read_file":
            return {"path": path_match.group(1) if path_match else "~/README.md"}

        if tool_name == "search_files":
            cleaned = re.sub(
                r"\b(search|files|find|folder|inside|in)\b",
                " ",
                text,
                flags=re.IGNORECASE,
            ).strip()
            q = cleaned.split()[0] if cleaned.split() else text[:30]
            return {"query": q}

        if tool_name == "read_journal_logs":
            unit_m = re.search(r"([a-zA-Z0-9_\-]+\.service)", text)
            args: dict[str, Any] = {"lines": 40}
            if unit_m:
                args["unit"] = unit_m.group(1)
            return args

        if tool_name == "desktop_control":
            num_m = re.search(r"(\d{1,3})", text)
            val_num = num_m.group(1) if num_m else "50"
            if any(w in low for w in ("mute", "silence")):
                return {"action": "volume_mute", "value": "on"}
            if "brightness" in low:
                return {"action": "brightness_set", "value": val_num}
            if "wifi" in low:
                state = "off" if any(w in low for w in ("off", "disable")) else "on"
                return {"action": "wifi", "value": state}
            if "bluetooth" in low:
                state = "off" if any(w in low for w in ("off", "disable")) else "on"
                return {"action": "bluetooth", "value": state}
            return {"action": "volume_set", "value": val_num}

        if tool_name == "start_focus_timer":
            num_m = re.search(r"(\d{1,3})", text)
            mins = int(num_m.group(1)) if num_m else 25
            return {"minutes": mins, "label": f"Focus ({mins}m)"}

        if tool_name == "search_memory_and_history":
            return {"query": text[:80]}

        templates = (self.weights.get("learned_arg_templates") or {}).get(tool_name) or []
        if templates:
            q_set = set(_tokenize(user_text))
            _, best_tmpl = max(
                enumerate(templates),
                key=lambda item: (len(q_set & set(item[1].get("tokens") or [])), item[0]),
            )
            best_args = best_tmpl.get("args")
            if isinstance(best_args, dict) and best_args:
                synthesized = dict(best_args)
                if path_match and "path" in synthesized:
                    synthesized["path"] = path_match.group(1)
                return synthesized

        return {}

    def predict_tool_chain(
        self,
        user_text: str,
        allowed_names: set[str] | None = None,
        max_steps: int = 3,
    ) -> list[dict[str, Any]]:
        """Predict a multi-step tool sequence using tool centroids and learned Markov transitions."""
        tool, args, score = self.predict_tool(user_text, allowed_names=allowed_names)
        if not tool:
            return []
        chain: list[dict[str, Any]] = [{"name": tool, "arguments": args, "score": score}]
        seen = {tool}
        transitions: dict[str, dict[str, int]] = self.weights.get("tool_transitions") or {}
        cur = tool
        while len(chain) < max_steps:
            nxt_map = transitions.get(cur) or {}
            if not nxt_map:
                break
            total_out = sum(int(v) for v in nxt_map.values())
            if total_out <= 0:
                break
            nxt_tool, nxt_cnt = max(nxt_map.items(), key=lambda kv: int(kv[1]))
            ratio = int(nxt_cnt) / total_out
            if nxt_tool in seen or ratio < 0.35 or (allowed_names is not None and nxt_tool not in allowed_names):
                break
            chain.append(
                {
                    "name": nxt_tool,
                    "arguments": self.extract_arguments(nxt_tool, user_text),
                    "score": round(ratio, 3),
                }
            )
            seen.add(nxt_tool)
            cur = nxt_tool
        return chain

    def embed_text(self, text: str) -> list[float] | None:
        """Compute cached neural text embedding using the `embedding` GGUF specialist on GPU."""
        clean = (text or "").strip()
        if not clean:
            return None
        _load_embed_cache()
        cache_key = clean[:300]
        if cache_key in _EMBED_CACHE:
            return _EMBED_CACHE[cache_key]
        emb_model = find_gguf_model("embedding")
        emb_bin = shutil.which("llama-embedding")
        if not emb_model or not emb_bin:
            return None
        try:
            proc = subprocess.run(
                [emb_bin, "-m", str(emb_model), "-ngl", "99", "-p", clean[:800], "--embd-output-format", "json"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                idx = proc.stdout.find("{")
                if idx != -1:
                    parsed, _ = json.JSONDecoder().raw_decode(proc.stdout[idx:])
                    data = parsed.get("data") or []
                    if data and isinstance(data[0], dict) and isinstance(data[0].get("embedding"), list):
                        vec = [float(x) for x in data[0]["embedding"]]
                        if len(_EMBED_CACHE) >= 384:
                            _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)), None)
                        _EMBED_CACHE[cache_key] = vec
                        _save_embed_cache()
                        return vec
        except Exception:
            pass
        return None

    def sync_semantic_index(self, max_new: int = 6) -> dict[str, Any]:
        """Incrementally index QA memory, lessons, and checkpoints into persistent `SEMANTIC_INDEX_FILE`."""
        _load_embed_cache()
        candidates: list[str] = []
        for item in (self.weights.get("qa_memory") or [])[-60:]:
            u_txt = str(item.get("user") or "").strip()[:300]
            if len(u_txt) >= 6 and u_txt not in candidates:
                candidates.append(u_txt)
        if self.storage is not None:
            try:
                for les in self.storage.list_lessons(limit=20):
                    ctx = str(les.get("context") or "")
                    if ctx.startswith("_internal:"):
                        continue
                    l_txt = f"{ctx} {les.get('insight', '')}".strip()[:300]
                    if len(l_txt) >= 6 and l_txt not in candidates:
                        candidates.append(l_txt)
            except Exception:
                pass
            try:
                for cp in self.storage.list_session_checkpoints(limit=15):
                    c_txt = f"{cp.get('topic', '')} {cp.get('summary', '')}".strip()[:300]
                    if len(c_txt) >= 6 and c_txt not in candidates:
                        candidates.append(c_txt)
            except Exception:
                pass

        newly_embedded = 0
        if find_gguf_model("embedding") and shutil.which("llama-embedding"):
            for txt in candidates:
                if newly_embedded >= max_new:
                    break
                if txt not in _EMBED_CACHE and self.embed_text(txt) is not None:
                    newly_embedded += 1
        if _EMBED_CACHE:
            _save_embed_cache()
        return {
            "cached_vectors": len(_EMBED_CACHE),
            "newly_embedded": newly_embedded,
            "candidates": len(candidates),
        }

    def semantic_similarity(
        self, text_a: str, text_b: str, use_gpu_embed: bool = False
    ) -> float:
        """Compute semantic similarity using TF-IDF 3-gram cosine and optional GPU neural embeddings."""
        toks_a = _tokenize(text_a)
        toks_b = _tokenize(text_b)
        if not toks_a or not toks_b:
            return 0.0
        _load_embed_cache()
        idf: dict[str, float] = self.weights.get("idf", {})
        va = {t: cnt * idf.get(t, 1.0) for t, cnt in Counter(toks_a).items()}
        vb = {t: cnt * idf.get(t, 1.0) for t, cnt in Counter(toks_b).items()}
        lex_sim = _cosine_sim(va, vb)
        k_a, k_b = text_a.strip()[:300], text_b.strip()[:300]
        if use_gpu_embed or (k_a in _EMBED_CACHE and k_b in _EMBED_CACHE):
            da = self.embed_text(text_a) if use_gpu_embed else _EMBED_CACHE.get(k_a)
            db = self.embed_text(text_b) if use_gpu_embed else _EMBED_CACHE.get(k_b)
            if da and db:
                return max(lex_sim, _dense_cosine(da, db))
        return lex_sim

    def retrieve_similar_examples(
        self, query: str, top_k: int = 3, preferred_role: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve top-K most similar past teacher turns & lessons using TF-IDF + persistent dense & Neural vectors."""
        q_toks = _tokenize(query)
        if not q_toks:
            return []
        _load_embed_cache()
        target_role = preferred_role or classify_task_role(query)
        q_key = query.strip()[:300]
        q_dense = _EMBED_CACHE.get(q_key)
        q_neural: list[float] | None = None
        eng: Any = None
        try:
            from .neural import TEXT_DIM, NeuralEngine

            eng = NeuralEngine(self.storage)
            q_neural = [float(x) for x in eng.encode_features(query)[:TEXT_DIM]]
        except Exception:
            q_neural = None

        idf: dict[str, float] = self.weights.get("idf", {})
        q_vec = {t: cnt * idf.get(t, 1.0) for t, cnt in Counter(q_toks).items()}
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self.weights.get("qa_memory", []):
            u_str = str(item.get("user") or "")
            if not u_str:
                continue
            u_vec = {
                t: cnt * idf.get(t, 1.0)
                for t, cnt in Counter(_tokenize(u_str)).items()
            }
            sim = _cosine_sim(q_vec, u_vec)
            u_dense = _EMBED_CACHE.get(u_str.strip()[:300])
            if q_dense and u_dense:
                sim = max(sim, 0.6 * _dense_cosine(q_dense, u_dense) + 0.4 * sim)
            elif q_neural is not None and eng is not None:
                try:
                    u_neural = [float(x) for x in eng.encode_features(u_str)[: len(q_neural)]]
                    sim = max(sim, 0.55 * _dense_cosine(q_neural, u_neural) + 0.45 * sim)
                except Exception:
                    pass
            if item.get("role") == target_role and sim > 0.05:
                sim += 0.08
            if sim > 0.1:
                scored.append((sim, item))

        if self.storage is not None:
            try:
                for les in self.storage.list_lessons(limit=20):
                    ctx = str(les.get("context") or "")
                    if ctx.startswith("_internal:"):
                        continue
                    ins = str(les.get("insight") or "").strip()
                    res = str(les.get("resolution") or "").strip()
                    if not ins or not res:
                        continue
                    l_text = f"{ctx} {ins}".strip()
                    l_vec = {
                        t: cnt * idf.get(t, 1.0)
                        for t, cnt in Counter(_tokenize(l_text)).items()
                    }
                    l_sim = _cosine_sim(q_vec, l_vec)
                    l_dense = _EMBED_CACHE.get(l_text[:300])
                    if q_dense and l_dense:
                        l_sim = max(l_sim, 0.6 * _dense_cosine(q_dense, l_dense) + 0.4 * l_sim)
                    if l_sim > 0.18:
                        scored.append(
                            (
                                l_sim,
                                {
                                    "role": target_role,
                                    "user": ins[:240],
                                    "tool_calls": [],
                                    "reply": res[:400],
                                    "source": "lesson",
                                },
                            )
                        )
            except Exception:
                pass

        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:top_k]]


from .local_mind_client import (  # noqa: F401
    LocalMindClient,
    _LocalMindInteractions,
    download_default_gguf,
)

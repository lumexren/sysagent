"""Self-Training Local Model (Online Distillation + Neural/TF-IDF Router + llama.cpp GGUF Engine)."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
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
WEIGHTS_FILE = MODELS_DIR / "local_mind_weights.json"
ROLE_ADAPTERS_FILE = MODELS_DIR / "role_adapters.json"
LORA_TRAIN_FILE = DATASET_DIR / "lora_chatml.txt"

SPECIALIST_ROLES: tuple[str, ...] = (
    "coding",
    "general",
    "reasoning",
    "chat",
    "multimodal",
    "embedding",
)

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
    },
    {
        "user": "how is neural engine doing what is neural thinking",
        "tool": "inspect_self_component",
        "args": {"component": "neural"},
    },
    {
        "user": "what did self evolve change recent proposals",
        "tool": "inspect_self_component",
        "args": {"component": "evolve"},
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
]


from .local_inference import (
    _has_ram_for_gguf,
    _release_all_warm_models,
    ensure_llama_server,
    get_llama_cpp_model,
    get_warm_model_name,
    unload_idle_servers,
)

_EMBED_CACHE: dict[str, list[float]] = {}


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
) -> str:
    """Route a user prompt or turn to its matching specialist role."""
    t_names = {
        str(tc.get("name") or "")
        for tc in (tool_calls or [])
        if isinstance(tc, dict)
    }
    low = (user_text or "").strip().lower()
    if "take_screenshot" in t_names or any(
        k in low for k in ("screenshot", "image", "picture", "photo", "visual")
    ):
        return "multimodal"
    if t_names & {"edit_file", "write_file", "run_command", "search_files", "forge_skill"} or any(
        k in low
        for k in (
            "code", "python", "bash", "script", "refactor", "patch", "git",
            "test", "fix", "error", "bug", "regex", "sql",
        )
    ):
        return "coding"
    if any(
        k in low
        for k in (
            "why", "analyze", "plan", "compare", "reason", "logic", "rule", "critic", "how does",
        )
    ):
        return "reasoning"
    if not t_names and (
        len(low.split()) <= 5
        or any(
            k in low
            for k in (
                "hello", "hi", "hey", "thanks", "thank you", "who are you", "how are you",
            )
        )
    ):
        return "chat"
    if use_neural and len(low) >= 4:
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine()
            if int(eng.meta.get("epochs_trained", 0)) > 0:
                probs = eng.predict_role_probs(user_text)
                best_role, best_p = max(probs.items(), key=lambda kv: kv[1])
                if best_p >= 0.48:
                    return best_role
        except Exception:
            pass
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


def _save_weights_file(weights: dict[str, Any]) -> None:
    """Persist weights JSON and refresh the in-process mtime cache."""
    WEIGHTS_FILE.write_text(json.dumps(weights, ensure_ascii=False), encoding="utf-8")
    try:
        mtime_ns = WEIGHTS_FILE.stat().st_mtime_ns
        _WEIGHTS_CACHE.clear()
        _WEIGHTS_CACHE[(str(WEIGHTS_FILE), mtime_ns)] = weights
    except Exception:
        pass


class LocalMind:
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
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": classify_task_role(u_clean, tool_calls),
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

    def train(self, run_lora: bool = False) -> dict[str, Any]:
        """Full training pass over seed curriculum + distilled JSONL + SQLite chat history, rules, and lessons."""
        samples: list[dict[str, Any]] = []
        for s in SEED_CURRICULUM:
            samples.append(
                {
                    "user": s["user"],
                    "tool_calls": [{"name": s["tool"], "arguments": s["args"]}],
                    "reply": "",
                    "weight": 6.0,
                }
            )

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
            role = str(smp.get("role") or classify_task_role(u_text, tcalls))
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
        if chatml_lines:
            try:
                LORA_TRAIN_FILE.write_text("\n".join(chatml_lines) + "\n", encoding="utf-8")
            except Exception:
                pass

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
        try:
            ROLE_ADAPTERS_FILE.write_text(json.dumps(role_adapters, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        lora_status = f"adapters_ready ({len(role_adapters)} roles)"
        last_lora_samples = len(samples)
        last_lora_attempt = int(self.weights.get("last_lora_attempt_samples", 0))
        gguf_path = find_gguf_model()
        is_quantized = bool(
            gguf_path and any(q in gguf_path.name.upper() for q in ("Q4_", "Q5_", "Q8_", "IQ"))
        )
        if run_lora and gguf_path and not is_quantized and shutil.which("llama-finetune") and chatml_lines:
            last_lora_attempt = len(samples)
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
                else:
                    lora_status = f"error:{proc.returncode}"
            except Exception as exc:
                lora_status = f"error:{exc}"

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
        try:
            _save_weights_file(self.weights)
        except Exception:
            pass
        try:
            from .neural import NeuralEngine

            NeuralEngine(self.storage).train(samples=samples)
        except Exception:
            pass
        return self.get_status()

    def sample_user_utterances(self, limit: int = 6) -> list[str]:
        """Sample real user messages from chat history / distilled turns so self-training adapts to the user's natural language(s)."""
        samples: list[str] = []
        if self.storage is not None:
            try:
                for s in self.storage.list_sessions(limit=8):
                    for m in self.storage.get_messages(s["session_id"], limit=10):
                        if m.get("role") == "user":
                            txt = str(m.get("content") or "").strip()
                            if txt.startswith("[cwd:"):
                                txt = txt.split("]", 1)[-1].strip()
                            if 6 <= len(txt) <= 140 and not txt.startswith("[") and txt not in samples:
                                samples.append(txt)
                                if len(samples) >= limit:
                                    return samples
            except Exception:
                pass
        for d in reversed(self._load_distilled_entries(limit=60)):
            u = str(d.get("user") or "").strip()
            if 6 <= len(u) <= 140 and not u.startswith(("Self-evolve", "A background", "Regarding")) and u not in samples:
                samples.append(u)
                if len(samples) >= limit:
                    break
        return samples

    def weakest_specialist_role(self) -> str:
        """Return the generative specialist role with the fewest training samples."""
        return self.underrepresented_roles()[0]

    def underrepresented_roles(self) -> list[str]:
        """Return generative specialist roles sorted ascending by sample count."""
        rc = self.weights.get("role_counts") or {}
        gen_roles = ("coding", "general", "reasoning", "chat", "multimodal")
        return sorted(gen_roles, key=lambda r: int(rc.get(r, 0)))

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
            "llama_finetune_available": bool(shutil.which("llama-finetune")),
            "lora_status": self.weights.get("lora_status", "none"),
            "warm_model": get_warm_model_name(),
            "neural": self._neural_status(),
        }

    def _neural_status(self) -> dict[str, Any]:
        try:
            from .neural import NeuralEngine

            return NeuralEngine(self.storage).get_status()
        except Exception:
            return {}

    def _record_uncertain_query(self, user_text: str, score: float) -> None:
        """Queue low-confidence user queries so multi-teacher distillation can solve them online."""
        clean = (user_text or "").strip()
        if len(clean) < 4 or clean.startswith("["):
            return
        try:
            with UNCERTAIN_FILE.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "query": clean[:300],
                            "score": round(score, 3),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass

    def resolve_uncertain_batch(self, max_items: int = 8) -> dict[str, Any]:
        """Resolve up to max_items queued uncertain queries in ONE batch cloud call and update weights."""
        from .auth import _find_agy_cli
        from .tools import TOOL_DECLARATIONS

        if not UNCERTAIN_FILE.is_file():
            return {"status": "skipped", "resolved": 0, "remaining": 0}
        try:
            raw_lines = [ln for ln in UNCERTAIN_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            return {"status": "skipped", "resolved": 0, "remaining": 0}
        if not raw_lines:
            return {"status": "skipped", "resolved": 0, "remaining": 0}

        batch_queries: list[str] = []
        consumed_indices: set[int] = set()
        seen_q: set[str] = set()
        for idx, ln in enumerate(raw_lines):
            try:
                q = str(json.loads(ln).get("query") or "").strip()
            except Exception:
                consumed_indices.add(idx)
                continue
            if not q:
                consumed_indices.add(idx)
                continue
            if q in seen_q:
                consumed_indices.add(idx)
                continue
            if len(batch_queries) < max_items:
                batch_queries.append(q)
                seen_q.add(q)
                consumed_indices.add(idx)

        remaining_lines = [ln for i, ln in enumerate(raw_lines) if i not in consumed_indices]
        if not batch_queries:
            try:
                UNCERTAIN_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            return {"status": "skipped", "resolved": 0, "remaining": 0}

        agy_bin = _find_agy_cli()
        if not agy_bin:
            return {"status": "skipped", "resolved": 0, "remaining": len(raw_lines)}

        tool_names = [d["name"] for d in TOOL_DECLARATIONS]
        prompt = (
            "Map each user query below to the best tool + arguments and a concise plain-text reply.\n"
            f"Available tool names: {json.dumps(tool_names)}\n"
            f"Queries: {json.dumps(batch_queries, ensure_ascii=False)}\n"
            'Respond ONLY with JSON: {"samples": [{"user": "...", "tool": "tool_or_null", "args": {}, "reply": "..."}]}'
        )
        resolved = 0
        try:
            proc = subprocess.run(
                [agy_bin, "-p", prompt, "--model", TEACHER_MODELS[0], "--disable-slash-commands", "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=90,
                cwd="/tmp",
            )
            if proc.returncode != 0:
                return {"status": "error", "resolved": 0, "remaining": len(raw_lines)}
            out = proc.stdout.strip()
            j_idx = out.find("{")
            if j_idx == -1:
                return {"status": "error", "resolved": 0, "remaining": len(raw_lines)}
            parsed, _ = json.JSONDecoder().raw_decode(out[j_idx:])
            samples = parsed.get("samples") if isinstance(parsed, dict) else None
            if not isinstance(samples, list):
                return {"status": "error", "resolved": 0, "remaining": len(raw_lines)}
            for smp in samples:
                if not isinstance(smp, dict) or not smp.get("user") or not smp.get("reply"):
                    continue
                tname = smp.get("tool")
                tcalls = (
                    [{"name": str(tname), "arguments": dict(smp.get("args") or {})}]
                    if tname and str(tname) in tool_names
                    else []
                )
                self.record_and_learn(str(smp["user"]), tcalls, str(smp["reply"]))
                resolved += 1
        except Exception:
            return {"status": "error", "resolved": 0, "remaining": len(raw_lines)}

        try:
            if remaining_lines:
                UNCERTAIN_FILE.write_text("\n".join(remaining_lines) + "\n", encoding="utf-8")
            else:
                UNCERTAIN_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        if resolved:
            self.train()
        return {"status": "ok", "resolved": resolved, "remaining": len(remaining_lines)}

    def distill_from_teachers(self, count_per_teacher: int = 2) -> dict[str, Any]:
        """Multi-Teacher Knowledge Distillation: query diverse cloud teachers via `agy` to feed LocalMind."""
        from .auth import _find_agy_cli
        from .tools import TOOL_DECLARATIONS

        agy_bin = _find_agy_cli()
        if not agy_bin:
            return {"status": "skipped", "reason": "agy_cli_not_found"}

        uncertain_queries: list[str] = []
        if UNCERTAIN_FILE.is_file():
            try:
                for ln in UNCERTAIN_FILE.read_text(encoding="utf-8").splitlines()[-20:]:
                    if ln.strip():
                        q = json.loads(ln).get("query")
                        if q and q not in uncertain_queries:
                            uncertain_queries.append(str(q))
                UNCERTAIN_FILE.unlink(missing_ok=True)
            except Exception:
                pass

        tool_names = [d["name"] for d in TOOL_DECLARATIONS]
        harvested = 0
        teachers_used: list[str] = []

        weak_roles = self.underrepresented_roles()
        role_guidance = {
            "multimodal": "screenshot/image analysis, OCR, UI/visual inspection, or media file queries",
            "reasoning": "step-by-step debugging, root-cause diagnosis, architecture trade-offs, or complex logic",
            "coding": "Python/Bash code debugging, refactoring, git diff review, or test generation",
            "general": "Linux system inspection, systemd services, disk/RAM/network diagnostics, or memory search",
            "chat": "brief conversational greetings, status check-ins, or concise assistant Q&A",
        }

        for idx, teacher in enumerate(TEACHER_MODELS):
            prompt = (
                "You are a Teacher Model generating high-quality training examples for a local Linux system assistant.\n"
                f"Available tool names: {json.dumps(tool_names)}\n"
            )
            if uncertain_queries:
                batch_q = uncertain_queries[:count_per_teacher]
                uncertain_queries = uncertain_queries[count_per_teacher:]
                prompt += (
                    f"Solve these user queries and map each to the best tool + arguments and a clean plain-text reply: {json.dumps(batch_q, ensure_ascii=False)}\n"
                )
            else:
                focus_role = weak_roles[idx % len(weak_roles)]
                user_examples = self.sample_user_utterances(limit=5)
                lang_hint = (
                    f"Match the natural language(s) observed in the user's real history ({json.dumps(user_examples, ensure_ascii=False)}) as well as English"
                    if user_examples
                    else "in the user's preferred language and English"
                )
                prompt += (
                    f"Generate {count_per_teacher} diverse, realistic user commands specifically exercising the "
                    f"'{focus_role}' specialist role ({lang_hint}), focusing on "
                    f"{role_guidance.get(focus_role, 'Linux system assistance')}.\n"
                )
            prompt += (
                "Respond ONLY with a valid JSON object matching:\n"
                '{"samples": [{"user": "user request", "tool": "tool_name_or_null", "args": {}, "reply": "concise plain text answer without backticks"}]}'
            )
            try:
                proc = subprocess.run(
                    [
                        agy_bin,
                        "-p",
                        prompt,
                        "--model",
                        teacher,
                        "--disable-slash-commands",
                        "--output-format",
                        "text",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    cwd="/tmp",
                )
                if proc.returncode != 0:
                    continue
                out = proc.stdout.strip()
                idx = out.find("{")
                if idx == -1:
                    continue
                parsed, _ = json.JSONDecoder().raw_decode(out[idx:])
                samples = parsed.get("samples") if isinstance(parsed, dict) else None
                if not isinstance(samples, list):
                    continue
                teachers_used.append(teacher)
                for smp in samples:
                    if not isinstance(smp, dict) or not smp.get("user") or not smp.get("reply"):
                        continue
                    tname = smp.get("tool")
                    tcalls = (
                        [{"name": str(tname), "arguments": dict(smp.get("args") or {})}]
                        if tname and str(tname) in tool_names
                        else []
                    )
                    self.record_and_learn(str(smp["user"]), tcalls, str(smp["reply"]))
                    harvested += 1
            except Exception:
                continue

        status = self.train()
        return {
            "status": "ok",
            "harvested_samples": harvested,
            "teachers_used": teachers_used,
            "model_status": status,
        }

    def predict_tool(
        self, user_text: str, allowed_names: set[str] | None = None
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

        if 0.04 < best_score < 0.24:
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
                        if len(_EMBED_CACHE) >= 256:
                            _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)), None)
                        _EMBED_CACHE[cache_key] = vec
                        return vec
        except Exception:
            pass
        return None

    def semantic_similarity(
        self, text_a: str, text_b: str, use_gpu_embed: bool = False
    ) -> float:
        """Compute semantic similarity using TF-IDF 3-gram cosine and optional GPU neural embeddings."""
        toks_a = _tokenize(text_a)
        toks_b = _tokenize(text_b)
        if not toks_a or not toks_b:
            return 0.0
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
        """Retrieve top-K most similar past teacher turns for Role-Aware Few-Shot RAG."""
        q_toks = _tokenize(query)
        if not q_toks:
            return []
        target_role = preferred_role or classify_task_role(query)
        q_dense = _EMBED_CACHE.get(query.strip()[:300])
        idf: dict[str, float] = self.weights.get("idf", {})
        q_vec = {t: cnt * idf.get(t, 1.0) for t, cnt in Counter(q_toks).items()}
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self.weights.get("qa_memory", []):
            u_str = item.get("user", "")
            u_vec = {
                t: cnt * idf.get(t, 1.0)
                for t, cnt in Counter(_tokenize(u_str)).items()
            }
            sim = _cosine_sim(q_vec, u_vec)
            u_dense = _EMBED_CACHE.get(u_str.strip()[:300]) if q_dense else None
            if q_dense and u_dense:
                sim = 0.5 * _dense_cosine(q_dense, u_dense) + 0.5 * sim
            if item.get("role") == target_role and sim > 0.05:
                sim += 0.08
            if sim > 0.1:
                scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:top_k]]

    def _ensure_llama_server(self, gguf: Path) -> str | None:
        """Lazily start or reuse a warm `llama-server` process on 127.0.0.1:8769 for fast GPU inference."""
        return ensure_llama_server(gguf)

    def _get_llama_cpp_model(self, gguf: Path) -> Any | None:
        """Lazily load and cache a llama_cpp.Llama instance with full GPU offload."""
        return get_llama_cpp_model(gguf)

    def _run_single_gguf(
        self, gguf: Path, prompt: str, system_instruction: str = ""
    ) -> str | None:
        import httpx

        examples = self.retrieve_similar_examples(prompt, top_k=2)
        role_hint = ""
        if ROLE_ADAPTERS_FILE.is_file():
            try:
                adapters = json.loads(ROLE_ADAPTERS_FILE.read_text(encoding="utf-8"))
                role_hint = str((adapters.get(classify_task_role(prompt)) or {}).get("system_hint") or "")
            except Exception:
                pass
        ex_block = ""
        if examples:
            ex_lines = [
                f"User: {ex['user']}\nAssistant: {ex['reply']}" for ex in examples if ex.get("reply")
            ]
            if ex_lines:
                ex_block = "\n\nPast learned examples:\n" + "\n---\n".join(ex_lines)
        sys_msg = "\n".join(p for p in (system_instruction.strip(), role_hint, ex_block.strip()) if p)

        srv_url = self._ensure_llama_server(gguf)
        if srv_url:
            try:
                resp = httpx.post(
                    f"{srv_url}/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": prompt.strip()},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 1024,
                    },
                    timeout=45,
                )
                if resp.status_code == 200:
                    choices = resp.json().get("choices") or []
                    if choices:
                        content = str(choices[0].get("message", {}).get("content") or "").strip()
                        if content:
                            return content
            except Exception:
                pass

        llm = self._get_llama_cpp_model(gguf)
        if llm is not None:
            try:
                res = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt.strip()},
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
                choices = res.get("choices") or []
                if choices:
                    content = str(choices[0].get("message", {}).get("content") or "").strip()
                    if content:
                        return content
            except Exception:
                pass

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
                    "1024",
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

    def run_gguf_inference(
        self, prompt: str, system_instruction: str = "", preferred_role: str | None = None
    ) -> str | None:
        """Run local GPU-accelerated GGUF inference via task-routed specialists and cross-consultation."""
        models = [m for m in find_all_gguf_models() if classify_gguf_role(m) != "embedding"]
        if not models:
            return None
        target_role = preferred_role or classify_task_role(prompt)
        specialist = find_gguf_model(target_role) or models[0]
        if len(models) == 1:
            return self._run_single_gguf(specialist, prompt, system_instruction)

        synthesizer = find_gguf_model("chat") or find_gguf_model("general") or models[0]
        if specialist == synthesizer or target_role not in {"coding", "reasoning", "multimodal"}:
            return self._run_single_gguf(specialist, prompt, system_instruction)

        draft = self._run_single_gguf(specialist, prompt, system_instruction)
        if not draft:
            return self._run_single_gguf(synthesizer, prompt, system_instruction)

        fusion_prompt = (
            f"Original Request ({target_role}):\n{prompt}\n\n"
            f"{target_role.title()} Specialist ({specialist.stem}) Draft:\n{draft}\n\n"
            "Verify accuracy, fix any formatting or language issues, and write the final concise plain-text answer."
        )
        fused = self._run_single_gguf(synthesizer, fusion_prompt, system_instruction)
        final_out = fused or draft
        if final_out and len(prompt.strip()) >= 8:
            qa = self.weights.setdefault("qa_memory", [])
            qa.append({"role": target_role, "user": prompt.strip()[:300], "tool_calls": [], "reply": final_out[:600]})
            if len(qa) > 150:
                self.weights["qa_memory"] = qa[-150:]
        return final_out

    def try_local_cascade(
        self, user_message: str, system_instruction: str = ""
    ) -> dict[str, Any] | None:
        """3-tier local-first cascade: Tier 1 Specialist -> Tier 2 Reasoning -> None (escalate to Cloud)."""
        msg = (user_message or "").strip()
        if not msg:
            return None
        low = msg.lower()
        models = [m for m in find_all_gguf_models() if classify_gguf_role(m) != "embedding"]
        if not models:
            return None

        role = classify_task_role(msg)
        pred_tool, _, _score = self.predict_tool(msg)
        local_first = os.environ.get("SYSAGENT_LOCAL_FIRST", "1").strip().lower() not in {"0", "false", "no", "off"}
        neural_escalate = False
        try:
            from .neural import NeuralEngine

            eng = NeuralEngine(self.storage)
            if int(eng.meta.get("epochs_trained", 0)) > 0:
                unc = eng.predict_with_uncertainty(msg)
                r_probs = eng.predict_role_probs(msg)
                top_tool_prob = float(unc.get("tool_confidence", unc.get("confidence", 0.0)) or 0.0)
                if top_tool_prob >= 0.55 and unc.get("tool") not in {None, "", "none"}:
                    neural_escalate = True
                elif not local_first and (
                    unc.get("uncertain")
                    or (r_probs.get("coding", 0.0) + r_probs.get("reasoning", 0.0)) >= 0.45
                ):
                    neural_escalate = True
        except Exception:
            pass
        dyn_action = self.storage.get_learned_cues().get("action_req", []) if self.storage is not None else []
        has_action = (
            neural_escalate
            or any(v in low for v in dyn_action)
            or any(ch in msg for ch in ("/", "~", "@", "\n"))
        )
        word_count = len(msg.split())
        if role == "chat":
            specialist = find_gguf_model("chat") or models[0]
            eligible = word_count <= 14 and pred_tool is None and not has_action
        elif local_first and role in {"general", "coding", "reasoning"}:
            specialist = find_gguf_model(role, strict=True)
            eligible = (
                specialist is not None
                and word_count <= 24
                and pred_tool is None
                and not has_action
            )
        else:
            specialist = None
            eligible = False
        if not eligible or specialist is None:
            return None

        draft = self._run_single_gguf(specialist, msg, system_instruction)
        uncertain_markers = ("i don't know", "not sure", "cannot determine", "escalate_cloud")
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
            and not any(u in draft.lower() for u in uncertain_markers)
        )
        if draft_ok:
            return {"tier": f"local:{role}", "model": specialist.name, "reply": draft}

        reasoner = find_gguf_model("reasoning", strict=True) or find_gguf_model("reasoning")
        if reasoner and reasoner != specialist:
            verify_prompt = (
                f"User Request:\n{msg}\n\n"
                f"Specialist ({specialist.stem}) Draft:\n{draft or '(none)'}\n\n"
                "If you can answer accurately and completely, write the final concise answer. "
                "If this requires cloud-level deep context or multi-step tools, reply ONLY with ESCALATE_CLOUD."
            )
            reasoned = self._run_single_gguf(reasoner, verify_prompt, system_instruction)
            if reasoned and "ESCALATE_CLOUD" not in reasoned and len(reasoned.strip()) >= 12:
                self.record_and_learn(msg, [], reasoned)
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

        similar = self.retrieve_similar_examples(user_text, top_k=1)
        if similar and similar[0].get("reply"):
            return f"[Local Offline Mode] {similar[0]['reply']}"

        return (
            "[Local Offline Mode] Internet or cloud model is currently unreachable. "
            "I am running on the local self-trained model and can still inspect system status, "
            "list/read/search files, check logs, control desktop settings, and search local memory."
        )


from .local_mind_client import (  # noqa: F401
    LocalMindClient,
    _LocalMindInteractions,
    download_default_gguf,
)

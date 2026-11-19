"""Vectorized NumPy Neural Engine for SysAgent."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .storage import Storage

INPUT_DIM = 256
TEXT_DIM = 240
HIDDEN_DIM = 128
WEIGHT_DECAY = 1e-4

_DEFAULT_MODELS_DIR = Path.home() / ".local" / "share" / "sysagent" / "models"
NEURAL_NPZ_FILE = _DEFAULT_MODELS_DIR / "neural_engine.npz"
NEURAL_META_FILE = _DEFAULT_MODELS_DIR / "neural_meta.json"

_ENGINE_CACHE: dict[tuple[str, int, int], tuple[dict[str, np.ndarray], dict[str, Any]]] = {}


def _gelu_and_grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tanh-approximated GELU activation and exact analytical derivative."""
    x_c = np.clip(x, -12.0, 12.0)
    k = math.sqrt(2.0 / math.pi)
    u = k * (x_c + 0.044715 * (x_c ** 3))
    th = np.tanh(u)
    y = 0.5 * x_c * (1.0 + th)
    du = k * (1.0 + 3.0 * 0.044715 * (x_c ** 2))
    grad = 0.5 * (1.0 + th) + 0.5 * x_c * (1.0 - th ** 2) * du
    return y.astype(np.float32), grad.astype(np.float32)


def _layer_norm(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parameter-free LayerNorm across the last axis; returns (z_hat, inv_std, z_hat)."""
    mu = z.mean(axis=-1, keepdims=True)
    var = z.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + 1e-5)
    z_hat = (z - mu) * inv_std
    return z_hat.astype(np.float32), inv_std.astype(np.float32), z_hat.astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(np.clip(shifted, -40.0, 40.0))
    return (exps / np.maximum(1e-12, exps.sum(axis=-1, keepdims=True))).astype(np.float32)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def _hash_token(token: str) -> tuple[int, float]:
    """Map token deterministically to (index in 0..TEXT_DIM-1, sign +-1.0)."""
    digest = hashlib.blake2b(token.encode("utf-8", errors="replace"), digest_size=8).digest()
    val = int.from_bytes(digest, "little")
    idx = val % TEXT_DIM
    sign = 1.0 if ((val >> 16) & 1) == 0 else -1.0
    return idx, sign


class NeuralEngine:
    """Self-contained vectorized neural routing, associative memory, world model, and critic."""

    def __init__(
        self,
        storage: Storage | None = None,
        npz_file: Path | None = None,
        meta_file: Path | None = None,
    ) -> None:
        from .config import DATA_DIR
        from .local_mind import MODELS_DIR, WEIGHTS_FILE

        self.storage = storage
        base_dir = WEIGHTS_FILE.parent if WEIGHTS_FILE.parent != MODELS_DIR else MODELS_DIR
        if storage is not None and getattr(storage, "db_path", None) and storage.db_path.parent != DATA_DIR:
            base_dir = storage.db_path.parent
            default_npz = base_dir / "neural_engine.npz"
            default_meta = base_dir / "neural_meta.json"
        else:
            default_npz = NEURAL_NPZ_FILE if NEURAL_NPZ_FILE.parent != _DEFAULT_MODELS_DIR else base_dir / "neural_engine.npz"
            default_meta = NEURAL_META_FILE if NEURAL_META_FILE.parent != _DEFAULT_MODELS_DIR else base_dir / "neural_meta.json"
        self.npz_file = npz_file or default_npz
        self.meta_file = meta_file or default_meta
        self.npz_file.parent.mkdir(parents=True, exist_ok=True)
        self.weights: dict[str, np.ndarray] = {}
        self.meta: dict[str, Any] = {}
        self._shared_cache: bool = False
        self._load()

    def _detach_for_write(self) -> None:
        if self._shared_cache:
            self.weights = {k: v.copy() for k, v in self.weights.items()}
            self.meta = json.loads(json.dumps(self.meta))
            self._shared_cache = False

    def _load(self) -> None:
        npz_mtime = self.npz_file.stat().st_mtime_ns if self.npz_file.is_file() else 0
        meta_mtime = self.meta_file.stat().st_mtime_ns if self.meta_file.is_file() else 0
        cache_key = (str(self.npz_file), npz_mtime, meta_mtime)
        if npz_mtime > 0 and cache_key in _ENGINE_CACHE:
            cached_w, cached_m = _ENGINE_CACHE[cache_key]
            self.weights = cached_w
            self.meta = cached_m
            self._shared_cache = True
            return

        if self.meta_file.is_file():
            try:
                loaded_meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
                if isinstance(loaded_meta, dict):
                    self.meta = loaded_meta
            except Exception:
                self.meta = {}

        if self.npz_file.is_file():
            try:
                with np.load(self.npz_file, allow_pickle=False) as arc:
                    self.weights = {k: arc[k].astype(np.float32) for k in arc.files}
            except Exception:
                self.weights = {}

        self._ensure_initialized(self.meta.get("tool_classes") or [])
        if npz_mtime > 0:
            _ENGINE_CACHE[cache_key] = (
                {k: v.copy() for k, v in self.weights.items()},
                json.loads(json.dumps(self.meta)),
            )

    def _ensure_initialized(self, tool_classes: list[str]) -> None:
        from .local_mind import SPECIALIST_ROLES

        self._detach_for_write()

        rng = np.random.default_rng(42)
        n_roles = len(SPECIALIST_ROLES)
        n_tools = max(1, len(tool_classes))
        if "W1" not in self.weights or self.weights["W1"].shape != (INPUT_DIM, HIDDEN_DIM):
            self.weights["W1"] = (rng.normal(0.0, math.sqrt(2.0 / INPUT_DIM), (INPUT_DIM, HIDDEN_DIM))).astype(np.float32)
            self.weights["b1"] = np.zeros((HIDDEN_DIM,), dtype=np.float32)
            self.weights["W2"] = (rng.normal(0.0, math.sqrt(1.0 / HIDDEN_DIM), (HIDDEN_DIM, HIDDEN_DIM))).astype(np.float32)
            self.weights["b2"] = np.zeros((HIDDEN_DIM,), dtype=np.float32)
        if "W_role" not in self.weights or self.weights["W_role"].shape != (HIDDEN_DIM, n_roles):
            self.weights["W_role"] = (rng.normal(0.0, math.sqrt(1.0 / HIDDEN_DIM), (HIDDEN_DIM, n_roles))).astype(np.float32)
            self.weights["b_role"] = np.zeros((n_roles,), dtype=np.float32)
        if "W_val" not in self.weights or self.weights["W_val"].shape != (HIDDEN_DIM, 1):
            self.weights["W_val"] = (rng.normal(0.0, 0.05, (HIDDEN_DIM, 1))).astype(np.float32)
            self.weights["b_val"] = np.zeros((1,), dtype=np.float32)

        old_classes = list(self.meta.get("tool_classes") or [])
        if (
            tool_classes != old_classes
            or "W_tool" not in self.weights
            or self.weights["W_tool"].shape != (HIDDEN_DIM, n_tools)
        ):
            new_wt = (rng.normal(0.0, math.sqrt(1.0 / HIDDEN_DIM), (HIDDEN_DIM, n_tools))).astype(np.float32)
            new_bt = np.zeros((n_tools,), dtype=np.float32)
            if "W_tool" in self.weights and old_classes:
                old_wt = self.weights["W_tool"]
                old_bt = self.weights.get("b_tool", np.zeros((len(old_classes),), dtype=np.float32))
                old_map = {name: i for i, name in enumerate(old_classes)}
                for j, name in enumerate(tool_classes):
                    if name in old_map and old_map[name] < old_wt.shape[1]:
                        new_wt[:, j] = old_wt[:, old_map[name]]
                        new_bt[j] = old_bt[old_map[name]]
            self.weights["W_tool"] = new_wt
            self.weights["b_tool"] = new_bt
            self.meta["tool_classes"] = list(tool_classes)

        if (
            "W_trans" not in self.weights
            or self.weights["W_trans"].shape != (HIDDEN_DIM, n_tools)
        ):
            self.weights["W_trans"] = (rng.normal(0.0, math.sqrt(1.0 / HIDDEN_DIM), (HIDDEN_DIM, n_tools))).astype(np.float32)
            self.weights["b_trans"] = np.zeros((n_tools,), dtype=np.float32)

        if "W_ae_enc" not in self.weights or self.weights["W_ae_enc"].shape != (4, 2):
            self.weights["W_ae_enc"] = (rng.normal(0.0, 0.3, (4, 2))).astype(np.float32)
            self.weights["b_ae_enc"] = np.zeros((2,), dtype=np.float32)
            self.weights["W_ae_dec"] = (rng.normal(0.0, 0.3, (2, 4))).astype(np.float32)
            self.weights["b_ae_dec"] = np.full((4,), 0.3, dtype=np.float32)

        if "W_stall" not in self.weights or self.weights["W_stall"].shape != (HIDDEN_DIM, 1):
            self.weights["W_stall"] = (rng.normal(0.0, 0.05, (HIDDEN_DIM, 1))).astype(np.float32)
            self.weights["b_stall"] = np.zeros((1,), dtype=np.float32)

    def _save(self) -> None:
        self.meta["param_count"] = self._count_params()
        try:
            self.npz_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.npz_file, **self.weights)
            self.meta_file.write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")
            mtime = self.npz_file.stat().st_mtime_ns
            meta_mtime = self.meta_file.stat().st_mtime_ns
            for k in [key for key in _ENGINE_CACHE if key[0] == str(self.npz_file)]:
                _ENGINE_CACHE.pop(k, None)
            _ENGINE_CACHE[(str(self.npz_file), mtime, meta_mtime)] = (
                {k: v.copy() for k, v in self.weights.items()},
                json.loads(json.dumps(self.meta)),
            )
        except Exception:
            pass

    def _count_params(self) -> int:
        keys = (
            "W1", "b1", "W2", "b2",
            "W_tool", "b_tool", "W_role", "b_role", "W_val", "b_val",
            "W_trans", "b_trans", "W_ae_enc", "b_ae_enc", "W_ae_dec", "b_ae_dec",
            "W_stall", "b_stall",
        )
        return int(sum(self.weights[k].size for k in keys if k in self.weights))


    def encode_features(
        self,
        text: str,
        sys_state: dict[str, Any] | None = None,
        idf: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Encode text + optional dense embedding + continuous system/temporal state into 256 dims."""
        from .local_mind import _EMBED_CACHE, _tokenize

        vec = np.zeros((INPUT_DIM,), dtype=np.float32)
        clean = (text or "").strip()
        toks = _tokenize(clean)
        words = [t for t in toks if not t.startswith("#")]
        bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
        all_feats = [*toks, *bigrams]
        if all_feats:
            counts = Counter(all_feats)
            idf_map = idf or {}
            for feat, cnt in counts.items():
                w = float(cnt) * float(idf_map.get(feat, 1.0))
                idx, sign = _hash_token(feat)
                vec[idx] += sign * w

        dense = _EMBED_CACHE.get(clean[:300])
        if dense:
            arr = np.asarray(dense, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0.0:
                arr = arr / norm
                if len(arr) >= TEXT_DIM:
                    vec[:TEXT_DIM] += 0.65 * arr[:TEXT_DIM]
                else:
                    vec[: len(arr)] += 0.65 * arr

        text_norm = float(np.linalg.norm(vec[:TEXT_DIM]))
        if text_norm > 0.0:
            vec[:TEXT_DIM] /= text_norm

        # 16 continuous state & structural features in indices 240..255
        now = datetime.now()
        hour_rad = (2.0 * math.pi * (now.hour + now.minute / 60.0)) / 24.0
        st = sys_state or {}
        gpu_st = st.get("gpu") if isinstance(st.get("gpu"), dict) else {}
        vram_tot = max(1.0, float(gpu_st.get("vram_total_mb") or 1.0))
        vram_pct = float(gpu_st.get("vram_used_mb") or 0.0) / vram_tot if gpu_st else 0.0

        low = clean.lower()
        vec[TEXT_DIM + 0] = math.sin(hour_rad)
        vec[TEXT_DIM + 1] = math.cos(hour_rad)
        vec[TEXT_DIM + 2] = min(1.0, float(st.get("cpu_percent", 0.0)) / 100.0)
        vec[TEXT_DIM + 3] = min(1.0, float(st.get("ram_percent", 0.0)) / 100.0)
        vec[TEXT_DIM + 4] = min(1.0, float(gpu_st.get("util_percent", 0.0)) / 100.0)
        vec[TEXT_DIM + 5] = min(1.0, vram_pct)
        vec[TEXT_DIM + 6] = min(1.0, len(clean) / 240.0)
        vec[TEXT_DIM + 7] = min(1.0, len(words) / 30.0)
        vec[TEXT_DIM + 8] = 1.0 if ("/" in clean or "~" in clean or ".py" in low) else 0.0
        vec[TEXT_DIM + 9] = 1.0 if ("?" in clean or "why" in low or "how" in low) else 0.0
        vec[TEXT_DIM + 10] = 1.0 if any(k in low for k in ("def ", "class ", "import ", "bug", "error", "traceback")) else 0.0
        vec[TEXT_DIM + 11] = 1.0 if any(k in low for k in ("cpu", "ram", "disk", "temp", "gpu", "vram", "memory")) else 0.0
        vec[TEXT_DIM + 12] = 1.0 if any(k in low for k in ("log", "journal", "systemd", "service")) else 0.0
        vec[TEXT_DIM + 13] = 1.0 if any(k in low for k in ("git", "test", "pytest", "ruff", "patch")) else 0.0
        vec[TEXT_DIM + 14] = 1.0 if any(k in low for k in ("http", "web", "search", "url")) else 0.0
        vec[TEXT_DIM + 15] = 0.25

        total_norm = float(np.linalg.norm(vec))
        if total_norm > 0.0:
            vec /= total_norm
        return vec


    def _forward(self, X: np.ndarray) -> dict[str, np.ndarray]:
        z1 = X @ self.weights["W1"] + self.weights["b1"]
        z1_hat, inv_std, _ = _layer_norm(z1)
        h1, dgelu1 = _gelu_and_grad(z1_hat)
        z2 = h1 @ self.weights["W2"] + self.weights["b2"]
        g2, dgelu2 = _gelu_and_grad(z2)
        h2 = h1 + 0.5 * g2
        logits_tool = h2 @ self.weights["W_tool"] + self.weights["b_tool"]
        logits_role = h2 @ self.weights["W_role"] + self.weights["b_role"]
        logits_val = h2 @ self.weights["W_val"] + self.weights["b_val"]
        logits_trans = h2 @ self.weights["W_trans"] + self.weights["b_trans"]
        return {
            "X": X,
            "inv_std": inv_std,
            "z1_hat": z1_hat,
            "h1": h1,
            "dgelu1": dgelu1,
            "dgelu2": dgelu2,
            "h2": h2,
            "logits_tool": logits_tool,
            "probs_tool": _softmax(logits_tool),
            "logits_role": logits_role,
            "probs_role": _softmax(logits_role),
            "logits_val": logits_val,
            "probs_val": _sigmoid(logits_val).astype(np.float32),
            "logits_trans": logits_trans,
            "probs_trans": _softmax(logits_trans),
        }


    def _build_training_data(
        self, samples: list[dict[str, Any]] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        from .local_mind import DISTILLED_FILE, SEED_CURRICULUM, SPECIALIST_ROLES, classify_task_role

        raw_samples: list[dict[str, Any]] = list(samples or [])
        if not raw_samples:
            for s in SEED_CURRICULUM:
                raw_samples.append(
                    {
                        "user": s["user"],
                        "tool_calls": [{"name": s["tool"], "arguments": s["args"]}],
                        "reply": "",
                    }
                )
            if DISTILLED_FILE.is_file():
                for ln in DISTILLED_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-1200:]:
                    if not ln.strip():
                        continue
                    try:
                        raw_samples.append(json.loads(ln))
                    except Exception:
                        continue
            if self.storage is not None:
                try:
                    for sess in self.storage.list_sessions(limit=20):
                        msgs = self.storage.get_messages(sess["session_id"], limit=30)
                        for i in range(len(msgs) - 1):
                            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                                raw_samples.append(
                                    {"user": msgs[i]["content"], "tool_calls": [], "reply": msgs[i + 1]["content"]}
                                )
                except Exception:
                    pass

        role_to_idx = {r: i for i, r in enumerate(SPECIALIST_ROLES)}
        tool_set: set[str] = set()
        for smp in raw_samples:
            tcalls = smp.get("tool_calls") or []
            if tcalls and isinstance(tcalls[0], dict) and tcalls[0].get("name"):
                tool_set.add(str(tcalls[0]["name"]))
        tool_classes = sorted(tool_set) or ["system_status"]
        tool_to_idx = {t: i for i, t in enumerate(tool_classes)}

        X_list: list[np.ndarray] = []
        y_tool: list[int] = []
        y_role: list[int] = []
        for smp in raw_samples:
            u = str(smp.get("user") or "").strip()
            if len(u) < 3:
                continue
            tcalls = smp.get("tool_calls") or []
            tname = (
                str(tcalls[0]["name"])
                if tcalls and isinstance(tcalls[0], dict) and tcalls[0].get("name")
                else ""
            )
            rname = str(smp.get("role") or classify_task_role(u, tcalls, use_neural=False))
            X_list.append(self.encode_features(u))
            y_tool.append(tool_to_idx.get(tname, -1))
            y_role.append(role_to_idx.get(rname, role_to_idx["general"]))

        if not X_list:
            X_list.append(self.encode_features("check system status cpu ram"))
            y_tool.append(0)
            y_role.append(role_to_idx["general"])

        return (
            np.stack(X_list, axis=0).astype(np.float32),
            np.asarray(y_tool, dtype=np.int64),
            np.asarray(y_role, dtype=np.int64),
            tool_classes,
        )

    def train(
        self,
        samples: list[dict[str, Any]] | None = None,
        epochs: int = 18,
        lr: float = 0.015,
    ) -> dict[str, Any]:
        """Train the 256->128->128 Multi-Head Neural Network via full backpropagation & AdamW."""
        self._detach_for_write()
        X, y_tool, y_role, tool_classes = self._build_training_data(samples=samples)
        self._ensure_initialized(tool_classes)

        param_names = ("W1", "b1", "W2", "b2", "W_tool", "b_tool", "W_role", "b_role", "W_val", "b_val")
        m_buf = {k: np.zeros_like(self.weights[k]) for k in param_names}
        v_buf = {k: np.zeros_like(self.weights[k]) for k in param_names}

        B = X.shape[0]
        n_tools = len(tool_classes)
        n_roles = self.weights["W_role"].shape[1]
        tool_mask = y_tool >= 0
        tool_rows = np.where(tool_mask)[0]
        n_trows = max(1, len(tool_rows))

        Y_tool_oh = np.zeros((B, n_tools), dtype=np.float32)
        if len(tool_rows):
            Y_tool_oh[tool_rows, y_tool[tool_rows]] = 1.0
        Y_role_oh = np.zeros((B, n_roles), dtype=np.float32)
        Y_role_oh[np.arange(B), y_role] = 1.0

        X_val, y_val = self._build_patch_critic_data()

        initial_loss: float | None = None
        last_loss = 0.0
        grad_norm = 0.0

        for step in range(1, max(1, epochs) + 1):
            cache = self._forward(X)
            probs_t = cache["probs_tool"]
            probs_r = cache["probs_role"]

            loss_t = (
                -float(np.log(np.clip(probs_t[tool_rows, y_tool[tool_rows]], 1e-8, 1.0)).mean())
                if len(tool_rows)
                else 0.0
            )
            loss_r = -float(np.log(np.clip(probs_r[np.arange(B), y_role], 1e-8, 1.0)).mean())
            loss = loss_t + 0.35 * loss_r
            if initial_loss is None:
                initial_loss = round(loss, 4)
            last_loss = round(loss, 4)

            dlogits_t = np.zeros_like(probs_t)
            if len(tool_rows):
                dlogits_t[tool_rows] = (probs_t[tool_rows] - Y_tool_oh[tool_rows]) / float(n_trows)
            dlogits_r = 0.35 * (probs_r - Y_role_oh) / float(max(1, B))

            h2 = cache["h2"]
            h1 = cache["h1"]
            dW_tool = h2.T @ dlogits_t
            db_tool = dlogits_t.sum(axis=0)
            dW_role = h2.T @ dlogits_r
            db_role = dlogits_r.sum(axis=0)

            dh2 = dlogits_t @ self.weights["W_tool"].T + dlogits_r @ self.weights["W_role"].T
            dz2 = (0.5 * dh2) * cache["dgelu2"]
            dW2 = h1.T @ dz2
            db2 = dz2.sum(axis=0)

            dh1 = dh2 + dz2 @ self.weights["W2"].T
            dz1_hat = dh1 * cache["dgelu1"]
            N = float(HIDDEN_DIM)
            z1_hat = cache["z1_hat"]
            dz1 = (cache["inv_std"] / N) * (
                N * dz1_hat
                - dz1_hat.sum(axis=-1, keepdims=True)
                - z1_hat * (dz1_hat * z1_hat).sum(axis=-1, keepdims=True)
            )
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0)

            grads: dict[str, np.ndarray] = {
                "W1": dW1,
                "b1": db1,
                "W2": dW2,
                "b2": db2,
                "W_tool": dW_tool,
                "b_tool": db_tool,
                "W_role": dW_role,
                "b_role": db_role,
            }

            if X_val is not None and y_val is not None and len(X_val) >= 2:
                vc = self._forward(X_val)
                dz_v = (vc["probs_val"] - y_val) / float(len(X_val))
                grads["W_val"] = vc["h2"].T @ dz_v
                grads["b_val"] = dz_v.sum(axis=0)

            sq_sum = sum(float(np.sum(g * g)) for g in grads.values())
            grad_norm = math.sqrt(sq_sum)
            clip = min(1.0, 5.0 / max(1e-8, grad_norm))

            b1_corr = 1.0 - (0.9 ** step)
            b2_corr = 1.0 - (0.999 ** step)
            for k, g in grads.items():
                gc = g * clip
                m_buf[k] = 0.9 * m_buf[k] + 0.1 * gc
                v_buf[k] = 0.999 * v_buf[k] + 0.001 * (gc * gc)
                m_hat = m_buf[k] / b1_corr
                v_hat = v_buf[k] / b2_corr
                if self.weights[k].ndim == 2:
                    self.weights[k] *= 1.0 - lr * WEIGHT_DECAY
                self.weights[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        final_c = self._forward(X)
        role_acc = float((final_c["probs_role"].argmax(axis=-1) == y_role).mean())
        tool_acc = (
            float((final_c["probs_tool"][tool_rows].argmax(axis=-1) == y_tool[tool_rows]).mean())
            if len(tool_rows)
            else role_acc
        )

        self.meta.update(
            {
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epochs_trained": int(self.meta.get("epochs_trained", 0)) + epochs,
                "initial_loss": initial_loss,
                "last_loss": last_loss,
                "grad_norm": round(grad_norm, 4),
                "tool_accuracy": round(tool_acc, 4),
                "role_accuracy": round(role_acc, 4),
                "samples_count": int(B),
            }
        )
        self._save()
        return self.get_status()

    def _build_patch_critic_data(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.storage is None:
            return None, None
        bad = {"failed", "rejected_by_critic", "rolled_back"}
        xs: list[np.ndarray] = []
        ys: list[float] = []
        try:
            for p in self.storage.list_evolve_proposals(limit=100):
                st = str(p.get("status") or "")
                if st == "applied":
                    y = 1.0
                elif st in bad:
                    y = 0.0
                else:
                    continue
                fname = Path(str(p.get("file_path") or "")).name
                desc = f"{fname} {p.get('category', '')} {p.get('title', '')} {str(p.get('reason') or '')[:160]}"
                xs.append(self.encode_features(desc))
                ys.append(y)
        except Exception:
            return None, None
        if len(xs) < 2:
            return None, None
        return np.stack(xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.float32).reshape(-1, 1)

    def online_step(self, text: str, tool_name: str, reward: float = 1.0, lr: float = 0.03) -> None:
        """Immediate single-sample gradient update (reward > 0 reinforces, reward < 0 penalizes)."""
        self._detach_for_write()
        classes = list(self.meta.get("tool_classes") or [])
        if tool_name not in classes:
            if reward <= 0.0:
                return
            classes.append(tool_name)
            self._ensure_initialized(classes)
        idx = classes.index(tool_name)
        X = self.encode_features(text).reshape(1, -1)
        cache = self._forward(X)
        probs = cache["probs_tool"][0]
        h2 = cache["h2"]

        dlogits = np.zeros_like(cache["probs_tool"])
        if reward > 0.0:
            one_hot = np.zeros_like(probs)
            one_hot[idx] = 1.0
            dlogits[0] = (probs - one_hot) * min(2.0, float(reward))
        else:
            p_wrong = float(probs[idx])
            # Gradient of -log(1 - p_wrong) pushes wrong tool down & lifts others proportionally
            dlogits[0] = -p_wrong * probs / max(1e-4, 1.0 - p_wrong)
            dlogits[0, idx] = max(0.25, p_wrong * abs(float(reward)))

        self.weights["W_tool"] -= lr * (h2.T @ dlogits)
        self.weights["b_tool"] -= lr * dlogits.sum(axis=0)
        dh2 = dlogits @ self.weights["W_tool"].T
        dz2 = (0.5 * dh2) * cache["dgelu2"]
        self.weights["W2"] -= (lr * 0.3) * (cache["h1"].T @ dz2)
        self.weights["b2"] -= (lr * 0.3) * dz2.sum(axis=0)

        buf: list[list[Any]] = list(self.meta.get("replay_buffer") or [])
        if reward > 0.0 and len(buf) >= 4:
            other = [item for item in buf if item[1] != tool_name and item[1] in classes]
            if other:
                rng = np.random.default_rng(len(buf) + len(text))
                picks = rng.choice(len(other), size=min(6, len(other)), replace=False)
                rep_X = np.stack([self.encode_features(str(other[i][0])) for i in picks], axis=0)
                rep_c = self._forward(rep_X)
                rep_dl = rep_c["probs_tool"].copy()
                for row_i, pick_i in enumerate(picks):
                    rep_dl[row_i, classes.index(str(other[pick_i][1]))] -= 1.0
                rep_dl /= float(len(picks))
                rep_lr = lr * 0.35
                self.weights["W_tool"] -= rep_lr * (rep_c["h2"].T @ rep_dl)
                self.weights["b_tool"] -= rep_lr * rep_dl.sum(axis=0)

        if reward > 0.0 and text.strip():
            same_cls = sum(1 for item in buf if item[1] == tool_name)
            if same_cls >= 4:
                for i, item in enumerate(buf):
                    if item[1] == tool_name:
                        buf.pop(i)
                        break
            buf.append([text[:160], tool_name])
            self.meta["replay_buffer"] = buf[-64:]

        self._save()

    def predict_tool_probs(self, text: str, sys_state: dict[str, Any] | None = None) -> dict[str, float]:
        classes = list(self.meta.get("tool_classes") or [])
        if not classes or "W_tool" not in self.weights:
            return {}
        X = self.encode_features(text, sys_state=sys_state).reshape(1, -1)
        probs = self._forward(X)["probs_tool"][0]
        return {cls_name: round(float(probs[i]), 4) for i, cls_name in enumerate(classes)}

    def predict_role_probs(self, text: str, sys_state: dict[str, Any] | None = None) -> dict[str, float]:
        from .local_mind import SPECIALIST_ROLES

        X = self.encode_features(text, sys_state=sys_state).reshape(1, -1)
        probs = self._forward(X)["probs_role"][0]
        return {role: round(float(probs[i]), 4) for i, role in enumerate(SPECIALIST_ROLES)}

    def predict_with_uncertainty(
        self, text: str, sys_state: dict[str, Any] | None = None, temperature: float = 1.15
    ) -> dict[str, Any]:
        """Compute temperature-scaled tool/role predictions with Shannon entropy and margin gating."""
        from .local_mind import SPECIALIST_ROLES

        classes = list(self.meta.get("tool_classes") or [])
        X = self.encode_features(text, sys_state=sys_state).reshape(1, -1)
        c = self._forward(X)
        temp = max(0.25, float(temperature))

        t_probs = _softmax(c["logits_tool"] / temp)[0]
        r_probs = _softmax(c["logits_role"] / temp)[0]
        t_sorted = np.sort(t_probs)[::-1]
        p1 = float(t_sorted[0]) if len(t_sorted) >= 1 else 0.0
        p2 = float(t_sorted[1]) if len(t_sorted) >= 2 else 0.0
        margin = round(p1 - p2, 4)

        k_t = max(2, len(t_probs))
        t_ent = float(-np.sum(t_probs * np.log(np.clip(t_probs, 1e-8, 1.0))) / math.log(k_t))
        k_r = max(2, len(r_probs))
        r_ent = float(-np.sum(r_probs * np.log(np.clip(r_probs, 1e-8, 1.0))) / math.log(k_r))

        top_t_idx = int(np.argmax(t_probs)) if len(t_probs) else 0
        top_r_idx = int(np.argmax(r_probs)) if len(r_probs) else 0
        uncertain = bool(t_ent > 0.65 or margin < 0.12)
        return {
            "top_tool": classes[top_t_idx] if classes else None,
            "top_prob": round(p1, 4),
            "margin": margin,
            "entropy": round(t_ent, 4),
            "uncertain": uncertain,
            "top_role": SPECIALIST_ROLES[top_r_idx] if top_r_idx < len(SPECIALIST_ROLES) else "general",
            "role_entropy": round(r_ent, 4),
        }

    def predict_next_tools(
        self, text: str, prev_tool: str = "", top_k: int = 3
    ) -> list[tuple[str, float]]:
        """Predict likely next tools in a multi-step sequence given query and optional previous tool."""
        classes = list(self.meta.get("tool_classes") or [])
        if not classes or "W_trans" not in self.weights:
            return []
        X = self.encode_features(text).reshape(1, -1)
        h2 = self._forward(X)["h2"].copy()
        if prev_tool and prev_tool in classes:
            p_idx = classes.index(prev_tool)
            h2 = h2 + 0.35 * self.weights["W_tool"][:, p_idx : p_idx + 1].T
        probs = _softmax(h2 @ self.weights["W_trans"] + self.weights["b_trans"])[0]
        ranked = sorted(
            [(classes[i], round(float(probs[i]), 4)) for i in range(len(classes)) if classes[i] != prev_tool],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[: max(1, top_k)]

    def infer_arguments(self, text: str, tool_name: str) -> dict[str, Any]:
        """Synthesize tool arguments from query cues and LocalMind's learned argument templates."""
        import re

        low = (text or "").lower()
        args: dict[str, Any] = {}
        try:
            from .local_mind import LocalMind

            args.update(LocalMind(self.storage).predict_args(tool_name, text))
        except Exception:
            pass

        path_m = re.search(r"(~\/[^\s'\"]+|\/[a-zA-Z0-9._\-\/]+)", text or "")
        if path_m and tool_name in {"read_file", "list_directory", "search_files"}:
            args.setdefault("path", path_m.group(1))
        if tool_name == "list_processes":
            if any(k in low for k in ("ram", "mem", "bellek")):
                args["sort_by"] = "memory"
            args.setdefault("limit", 15)
        elif tool_name == "list_systemd_units":
            if any(k in low for k in ("fail", "hata", "çöken", "broken")):
                args["state"] = "failed"
            args.setdefault("limit", 25)
        elif tool_name == "read_journal_logs":
            num_m = re.search(r"\b(\d{1,3})\b", low)
            if num_m:
                args["lines"] = min(200, max(10, int(num_m.group(1))))
        return args

    def record_safe_plan(self, text: str, tool_name: str, args: dict[str, Any]) -> None:
        """Cache a verified read-only tool call for zero-LLM semantic fast-path replay."""
        from .safety import SAFE_TOOLS

        self._detach_for_write()

        clean = (text or "").strip()
        if not clean or tool_name not in SAFE_TOOLS or len(clean) > 200:
            return
        plans: list[dict[str, Any]] = list(self.meta.get("safe_plans") or [])
        for p in plans:
            if p.get("text") == clean:
                p["tool"] = tool_name
                p["args"] = dict(args or {})
                p["hits"] = int(p.get("hits", 1)) + 1
                self.meta["safe_plans"] = plans[-64:]
                self._save()
                return
        plans.append({"text": clean, "tool": tool_name, "args": dict(args or {}), "hits": 1})
        self.meta["safe_plans"] = plans[-64:]
        self._save()

    def match_safe_plan(self, text: str, min_sim: float = 0.90) -> dict[str, Any] | None:
        """Return a cached read-only tool plan if semantic similarity exceeds threshold."""
        from .safety import SAFE_TOOLS

        plans = self.meta.get("safe_plans") or []
        if not plans or not (text or "").strip():
            return None
        q_vec = self.encode_features(text)[:TEXT_DIM]
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm < 1e-6:
            return None
        best: dict[str, Any] | None = None
        best_sim = 0.0
        for p in plans:
            tname = str(p.get("tool") or "")
            if tname not in SAFE_TOOLS:
                continue
            p_vec = self.encode_features(str(p.get("text") or ""))[:TEXT_DIM]
            p_norm = float(np.linalg.norm(p_vec))
            if p_norm < 1e-6:
                continue
            sim = float(np.dot(q_vec, p_vec) / (q_norm * p_norm))
            if sim > best_sim:
                best_sim = sim
                best = p
        if best is not None and best_sim >= min_sim:
            merged_args = dict(best.get("args") or {})
            if not merged_args:
                merged_args = self.infer_arguments(text, str(best["tool"]))
            return {"tool": str(best["tool"]), "args": merged_args, "similarity": round(best_sim, 4)}
        return None

    def score_reply_stall(self, reply_text: str) -> float:
        """Predict P(mid_task_stall) for an assistant reply; returns 0.0 until >= 3 stall examples exist."""
        if int(self.meta.get("stall_samples", 0)) < 3 or "W_stall" not in self.weights:
            return 0.0
        snippet = (reply_text or "")[-400:].strip()
        if not snippet:
            return 0.0
        X = self.encode_features(snippet).reshape(1, -1)
        h2 = self._forward(X)["h2"]
        p = float(_sigmoid(h2 @ self.weights["W_stall"] + self.weights["b_stall"])[0, 0])
        return round(p, 4)

    def online_stall_step(self, reply_text: str, is_stall: bool, lr: float = 0.05) -> float:
        """Update the stall-detection head from self-supervised user continuation or clean completion."""
        self._detach_for_write()
        snippet = (reply_text or "")[-400:].strip()
        if not snippet or "W_stall" not in self.weights:
            return 0.0
        X = self.encode_features(snippet).reshape(1, -1)
        h2 = self._forward(X)["h2"]
        p = float(_sigmoid(h2 @ self.weights["W_stall"] + self.weights["b_stall"])[0, 0])
        y = 1.0 if is_stall else 0.0
        dz = np.array([[p - y]], dtype=np.float32)
        self.weights["W_stall"] -= lr * (h2.T @ dz)
        self.weights["b_stall"] -= lr * dz.sum(axis=0)
        if is_stall:
            self.meta["stall_samples"] = int(self.meta.get("stall_samples", 0)) + 1
        self._save()
        p_after = float(_sigmoid(h2 @ self.weights["W_stall"] + self.weights["b_stall"])[0, 0])
        return round(p_after, 4)

    def online_transition_step(
        self, text: str, prev_tool: str, next_tool: str, lr: float = 0.04
    ) -> None:
        """Update sequential tool transition head from an observed tool_t -> tool_{t+1} pair."""
        self._detach_for_write()
        classes = list(self.meta.get("tool_classes") or [])
        if not next_tool or next_tool not in classes:
            return
        X = self.encode_features(text).reshape(1, -1)
        h2 = self._forward(X)["h2"].copy()
        if prev_tool and prev_tool in classes:
            p_idx = classes.index(prev_tool)
            h2 = h2 + 0.35 * self.weights["W_tool"][:, p_idx : p_idx + 1].T
        probs = _softmax(h2 @ self.weights["W_trans"] + self.weights["b_trans"])
        dlogits = probs.copy()
        dlogits[0, classes.index(next_tool)] -= 1.0
        self.weights["W_trans"] -= lr * (h2.T @ dlogits)
        self.weights["b_trans"] -= lr * dlogits.sum(axis=0)
        self._save()


    def sync_associative_memory(self) -> int:
        """Encode memories, lessons, KG edges, and brain rules into the associative vector store."""
        if self.storage is None:
            return 0
        self._detach_for_write()
        from .local_mind import _tokenize

        nodes: list[dict[str, str]] = []
        tok_sets: list[set[str]] = []
        for m in self.storage.list_memories(include_internal=False):
            k = str(m.get("key") or "")
            v = str(m.get("value") or "")
            if not k or k.startswith("_internal:") or not v:
                continue
            txt = f"{k}: {v[:220]}"
            nodes.append({"id": f"mem:{k[:60]}", "kind": "memory", "key": k, "text": txt})
            tok_sets.append({t for t in _tokenize(txt) if not t.startswith("#")})
        for ls in self.storage.list_lessons(limit=60):
            txt = f"{ls.get('context', '')}: {ls.get('insight', '')} -> {ls.get('resolution', '')}"[:240]
            nodes.append({"id": f"lesson:{ls['id']}", "kind": "lesson", "key": str(ls.get("context", "")), "text": txt})
            tok_sets.append({t for t in _tokenize(txt) if not t.startswith("#")})
        for kg_idx, kg in enumerate(self.storage.query_kg(limit=60)):
            subj = str(kg.get("subject") or "")
            txt = f"{subj} {kg.get('relation', '')} {kg.get('object', '')}"
            nodes.append({"id": f"kg:{kg_idx}:{subj[:24]}", "kind": "kg", "key": subj, "text": txt})
            tok_sets.append({t for t in _tokenize(txt) if not t.startswith("#")})
        for r in self.storage.list_brain_rules()[:40]:
            txt = f"Rule: {r.get('pattern', '')} -> {r.get('action_policy', '')}"[:240]
            nodes.append({"id": f"rule:{r['id']}", "kind": "rule", "key": str(r.get("pattern", "")), "text": txt})
            tok_sets.append({t for t in _tokenize(txt) if not t.startswith("#")})

        nodes = nodes[:200]
        tok_sets = tok_sets[:200]
        if not nodes:
            self.meta["assoc_nodes"] = []
            self.weights.pop("assoc_vecs", None)
            self._save()
            return 0

        vecs = np.stack([self.encode_features(n["text"]) for n in nodes], axis=0).astype(np.float32)
        synapses: dict[str, dict[str, float]] = self.meta.get("synapses") or {}
        if not isinstance(synapses, dict):
            synapses = {}

        n_cnt = len(nodes)
        for i in range(n_cnt):
            src_id = nodes[i]["id"]
            nbrs = synapses.setdefault(src_id, {})
            if len(nbrs) >= 8:
                continue
            for j in range(i + 1, n_cnt):
                shared = len(tok_sets[i] & tok_sets[j])
                if shared >= 2:
                    dst_id = nodes[j]["id"]
                    base_w = min(0.6, 0.18 * shared)
                    if dst_id not in nbrs:
                        nbrs[dst_id] = round(base_w, 3)
                    synapses.setdefault(dst_id, {}).setdefault(src_id, round(base_w, 3))

        valid_ids = {n["id"] for n in nodes}
        cleaned_syn: dict[str, dict[str, float]] = {}
        for nid, edges in synapses.items():
            if nid in valid_ids and isinstance(edges, dict):
                kept = {k: round(float(v), 3) for k, v in edges.items() if k in valid_ids and float(v) > 0.05}
                if kept:
                    cleaned_syn[nid] = dict(sorted(kept.items(), key=lambda kv: kv[1], reverse=True)[:12])

        self.meta["assoc_nodes"] = nodes
        self.meta["synapses"] = cleaned_syn
        self.weights["assoc_vecs"] = vecs
        self._save()
        return len(nodes)

    def spreading_activation(
        self, query_text: str, top_k: int = 5, hops: int = 2
    ) -> list[dict[str, Any]]:
        """Retrieve associatively linked facts/lessons via multi-hop neural spreading activation & Hebbian update."""
        nodes = self.meta.get("assoc_nodes") or []
        vecs = self.weights.get("assoc_vecs")
        if (not nodes or vecs is None or len(vecs) != len(nodes)) and self.storage is not None:
            self.sync_associative_memory()
            nodes = self.meta.get("assoc_nodes") or []
            vecs = self.weights.get("assoc_vecs")
        if not nodes or vecs is None or len(vecs) == 0:
            return []

        q_vec = self.encode_features(query_text)
        # Only use text dims 0..TEXT_DIM-1 for semantic recall similarity
        q_t = q_vec[:TEXT_DIM]
        v_t = vecs[:, :TEXT_DIM]
        q_n = float(np.linalg.norm(q_t))
        if q_n <= 0.0:
            return []
        v_n = np.maximum(1e-8, np.linalg.norm(v_t, axis=-1))
        act = np.maximum(0.0, (v_t @ q_t) / (v_n * q_n)).astype(np.float32)

        synapses: dict[str, dict[str, float]] = self.meta.get("synapses") or {}
        id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
        for _ in range(max(1, hops)):
            nxt = act.copy()
            for i, a_i in enumerate(act):
                if a_i <= 0.12:
                    continue
                nbrs = synapses.get(nodes[i]["id"])
                if not nbrs:
                    continue
                for tgt_id, w in nbrs.items():
                    j = id_to_idx.get(tgt_id)
                    if j is not None:
                        nxt[j] += 0.35 * float(w) * float(a_i)
            act = np.clip(nxt, 0.0, 1.5)

        top_indices = [int(i) for i in np.argsort(-act)[: max(1, top_k)] if float(act[i]) > 0.14]
        if len(top_indices) >= 2:
            for idx_a in top_indices[:4]:
                id_a = nodes[idx_a]["id"]
                nbrs_a = synapses.setdefault(id_a, {})
                for k_n in list(nbrs_a):
                    nbrs_a[k_n] = round(nbrs_a[k_n] * 0.995, 3)
                for idx_b in top_indices[:4]:
                    if idx_a == idx_b:
                        continue
                    id_b = nodes[idx_b]["id"]
                    delta = 0.08 * min(1.0, float(act[idx_a])) * min(1.0, float(act[idx_b]))
                    nbrs_a[id_b] = round(min(1.0, nbrs_a.get(id_b, 0.0) + delta), 3)
            self.meta["synapses"] = synapses
            try:
                self.meta_file.write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")
                mtime = self.npz_file.stat().st_mtime_ns if self.npz_file.is_file() else 0
                meta_mtime = self.meta_file.stat().st_mtime_ns if self.meta_file.is_file() else 0
                if mtime > 0:
                    for k in [key for key in _ENGINE_CACHE if key[0] == str(self.npz_file)]:
                        _ENGINE_CACHE.pop(k, None)
                    _ENGINE_CACHE[(str(self.npz_file), mtime, meta_mtime)] = (
                        {k: v.copy() for k, v in self.weights.items()},
                        json.loads(json.dumps(self.meta)),
                    )
            except Exception:
                pass

        return [
            {
                "id": nodes[i]["id"],
                "kind": nodes[i]["kind"],
                "key": nodes[i].get("key", ""),
                "text": nodes[i]["text"],
                "activation": round(float(act[i]), 3),
            }
            for i in top_indices
        ]


    def observe_world_state(
        self, watcher: Any = None, active_role: str | None = None
    ) -> dict[str, Any]:
        """Record system telemetry snapshot & hourly role pattern into the Predictive World Model."""
        self._detach_for_write()
        snap = watcher.get_health_snapshot() if watcher and hasattr(watcher, "get_health_snapshot") else {}
        if not snap:
            import psutil

            snap = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_percent": psutil.virtual_memory().percent,
            }
        gpu = snap.get("gpu") if isinstance(snap.get("gpu"), dict) else {}
        vram_tot = max(1.0, float(gpu.get("vram_total_mb") or 1.0))
        vram_pct = round(100.0 * float(gpu.get("vram_used_mb") or 0.0) / vram_tot, 1) if gpu else 0.0

        hist: list[list[float]] = list(self.meta.get("telemetry_history") or [])
        now_ts = round(time.time(), 1)
        cpu_now = round(float(snap.get("cpu_percent", 0.0)), 1)
        ram_now = round(float(snap.get("ram_percent", 0.0)), 1)
        gpu_now = round(float(gpu.get("util_percent", 0.0)), 1)

        last_fc = self.meta.get("last_forecast") or {}
        if isinstance(last_fc, dict) and last_fc.get("ram") is not None:
            err = 0.5 * (abs(float(last_fc.get("ram", ram_now)) - ram_now) + abs(float(last_fc.get("cpu", cpu_now)) - cpu_now))
            prev_mae = float(self.meta.get("world_mae", err))
            self.meta["world_mae"] = round(0.8 * prev_mae + 0.2 * err, 2)

        hist.append([now_ts, cpu_now, ram_now, gpu_now, vram_pct])
        self.meta["telemetry_history"] = hist[-72:]

        x_ae = np.asarray([[cpu_now / 100.0, ram_now / 100.0, gpu_now / 100.0, vram_pct / 100.0]], dtype=np.float32)
        if "W_ae_enc" in self.weights and max(cpu_now, ram_now) < 92.0:
            z_ae = np.tanh(x_ae @ self.weights["W_ae_enc"] + self.weights["b_ae_enc"])
            recon = _sigmoid(z_ae @ self.weights["W_ae_dec"] + self.weights["b_ae_dec"])
            d_out = (recon - x_ae) * recon * (1.0 - recon)
            d_z = (d_out @ self.weights["W_ae_dec"].T) * (1.0 - z_ae * z_ae)
            lr_ae = 0.05
            self.weights["W_ae_dec"] -= lr_ae * (z_ae.T @ d_out)
            self.weights["b_ae_dec"] -= lr_ae * d_out.sum(axis=0)
            self.weights["W_ae_enc"] -= lr_ae * (x_ae.T @ d_z)
            self.weights["b_ae_enc"] -= lr_ae * d_z.sum(axis=0)

        if active_role:
            hour_k = str(datetime.now().hour)
            h_roles = self.meta.setdefault("hourly_roles", {})
            bucket = h_roles.setdefault(hour_k, {})
            bucket[active_role] = int(bucket.get(active_role, 0)) + 1

        forecast = self._compute_world_forecast(hist)
        self.meta["last_forecast"] = {"ts": now_ts, "ram": forecast["pred_ram_10m"], "cpu": forecast["pred_cpu_10m"]}
        self._save()
        return forecast

    def _compute_world_forecast(self, hist: list[list[float]]) -> dict[str, Any]:
        """Damped residual 10-minute forecast + unsupervised telemetry anomaly score."""
        if not hist:
            return {
                "pred_cpu_10m": 0.0,
                "pred_ram_10m": 0.0,
                "pred_gpu_10m": 0.0,
                "pred_vram_10m": 0.0,
                "bottleneck_prob": 0.0,
                "anomaly_score": 0.0,
                "anomaly_detected": False,
                "forecast_mae": float(self.meta.get("world_mae", 0.0)),
                "predicted_role": "general",
                "proactive_hint": "nominal",
                "samples": 0,
            }

        recent = np.asarray(hist[-12:], dtype=np.float32)

        def _forecast_col(col_idx: int) -> float:
            vals = recent[:, col_idx]
            cur = float(vals[-1])
            if len(vals) < 3:
                return round(float(np.clip(cur, 0.0, 100.0)), 1)
            ema = cur
            for v in vals[-6:]:
                ema = 0.65 * ema + 0.35 * float(v)
            t = recent[:, 0] - recent[0, 0]
            if float(np.ptp(t)) < 1.0:
                return round(float(np.clip(ema, 0.0, 100.0)), 1)
            slope, _ = np.polyfit(t, vals, 1)
            damped_slope = float(np.clip(slope, -0.025, 0.025))
            pred = ema + damped_slope * 300.0 * 0.5
            pred = float(np.clip(pred, max(0.0, cur - 15.0), min(100.0, cur + 15.0)))
            return round(pred, 1)

        pred_cpu = _forecast_col(1)
        pred_ram = _forecast_col(2)
        pred_gpu = _forecast_col(3)
        pred_vram = _forecast_col(4)
        peak_strain = max(pred_ram, pred_vram, pred_cpu * 0.9)
        b_prob = round(float(_sigmoid((peak_strain - 82.0) / 7.0)), 3)

        anomaly_score = 0.0
        if "W_ae_enc" in self.weights:
            cur_vec = recent[-1, 1:5].reshape(1, 4) / 100.0
            z_ae = np.tanh(cur_vec @ self.weights["W_ae_enc"] + self.weights["b_ae_enc"])
            recon = _sigmoid(z_ae @ self.weights["W_ae_dec"] + self.weights["b_ae_dec"])
            anomaly_score = round(float(np.mean((cur_vec - recon) ** 2)) * 10.0, 4)

        hour_k = str(datetime.now().hour)
        bucket = (self.meta.get("hourly_roles") or {}).get(hour_k) or {}
        pred_role = max(bucket.items(), key=lambda kv: int(kv[1]))[0] if bucket else "general"

        if pred_ram >= 88.0:
            hint = "ram_cleanup"
        elif pred_vram >= 88.0:
            hint = "unload_idle_vram"
        elif anomaly_score >= 0.25:
            hint = "telemetry_anomaly"
        elif b_prob < 0.35 and pred_role in {"coding", "reasoning"}:
            hint = f"pre_warm_{pred_role}"
        else:
            hint = "nominal"

        return {
            "pred_cpu_10m": round(pred_cpu, 1),
            "pred_ram_10m": round(pred_ram, 1),
            "pred_gpu_10m": round(pred_gpu, 1),
            "pred_vram_10m": round(pred_vram, 1),
            "bottleneck_prob": b_prob,
            "anomaly_score": anomaly_score,
            "anomaly_detected": bool(anomaly_score >= 0.25),
            "forecast_mae": float(self.meta.get("world_mae", 0.0)),
            "predicted_role": pred_role,
            "proactive_hint": hint,
            "samples": len(hist),
        }

    def predict_world_state(self) -> dict[str, Any]:
        """Extrapolate 10-minute system load forecast, bottleneck probability, anomaly score, and expected role."""
        return self._compute_world_forecast(self.meta.get("telemetry_history") or [])


    def score_patch_candidate(
        self, file_name: str, category: str = "bugfix", title: str = "", reason: str = ""
    ) -> float:
        """Predict P(patch_success) using the Value Critic neural head."""
        desc = f"{file_name} {category} {title} {reason[:160]}"
        X = self.encode_features(desc).reshape(1, -1)
        p = float(self._forward(X)["probs_val"][0, 0])
        return round(p, 3)

    def online_patch_step(
        self,
        file_name: str,
        category: str = "bugfix",
        title: str = "",
        reason: str = "",
        success: bool = True,
        lr: float = 0.04,
    ) -> float:
        """Immediately update the Patch Value Critic head from a patch outcome and return updated P(success)."""
        self._detach_for_write()
        desc = f"{file_name} {category} {title} {reason[:160]}"
        X = self.encode_features(desc).reshape(1, -1)
        c = self._forward(X)
        y = 1.0 if success else 0.0
        dz_v = (c["probs_val"] - y).astype(np.float32)
        self.weights["W_val"] -= lr * (c["h2"].T @ dz_v)
        self.weights["b_val"] -= lr * dz_v.sum(axis=0)
        dh2 = dz_v @ self.weights["W_val"].T
        dz2 = (0.5 * dh2) * c["dgelu2"]
        self.weights["W2"] -= (lr * 0.25) * (c["h1"].T @ dz2)
        self.weights["b2"] -= (lr * 0.25) * dz2.sum(axis=0)
        self._save()
        return self.score_patch_candidate(file_name, category, title, reason)

    def prune_weak_synapses(self, min_weight: float = 0.08) -> int:
        """Remove Hebbian associative links weaker than `min_weight` and return pruned edge count."""
        self._detach_for_write()
        synapses = self.meta.get("synapses") or {}
        if not isinstance(synapses, dict):
            return 0
        removed = 0
        cleaned: dict[str, dict[str, float]] = {}
        for src, edges in synapses.items():
            if not isinstance(edges, dict):
                continue
            kept: dict[str, float] = {}
            for dst, w in edges.items():
                if float(w) >= min_weight:
                    kept[dst] = round(float(w), 3)
                else:
                    removed += 1
            if kept:
                cleaned[src] = kept
        if removed:
            self.meta["synapses"] = cleaned
            self._save()
        return removed


    def pulse(self, query_text: str, sys_state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Inspect live neuron activations, role/tool probabilities, patch score, and spreading memory."""
        X = self.encode_features(query_text, sys_state=sys_state).reshape(1, -1)
        c = self._forward(X)
        classes = list(self.meta.get("tool_classes") or [])
        from .local_mind import SPECIALIST_ROLES

        t_probs = c["probs_tool"][0]
        r_probs = c["probs_role"][0]
        top_tools = sorted(
            [
                {"tool": classes[i], "prob": round(float(t_probs[i]), 4)}
                for i in range(min(len(classes), len(t_probs)))
            ],
            key=lambda x: x["prob"],
            reverse=True,
        )[:5]
        roles_out = {
            SPECIALIST_ROLES[i]: round(float(r_probs[i]), 4)
            for i in range(min(len(SPECIALIST_ROLES), len(r_probs)))
        }
        h2 = c["h2"][0]
        top_t = top_tools[0]["tool"] if top_tools else ""
        return {
            "query": query_text,
            "top_tools": top_tools,
            "next_tool_chain": self.predict_next_tools(query_text, prev_tool=top_t, top_k=3),
            "uncertainty": self.predict_with_uncertainty(query_text, sys_state=sys_state),
            "role_probs": roles_out,
            "patch_value": round(float(c["probs_val"][0, 0]), 4),
            "hidden_norm": round(float(np.linalg.norm(h2)), 3),
            "active_neurons": int(np.sum(np.abs(h2) > 0.15)),
            "hidden_dim": HIDDEN_DIM,
            "associative_hits": self.spreading_activation(query_text, top_k=4),
            "world_forecast": self.predict_world_state(),
        }

    def get_status(self) -> dict[str, Any]:
        synapses = self.meta.get("synapses") or {}
        synapse_edges = sum(len(v) for v in synapses.values() if isinstance(v, dict))
        return {
            "architecture": f"{INPUT_DIM}->{HIDDEN_DIM}->{HIDDEN_DIM}->[{len(self.meta.get('tool_classes') or [])}T,6R,1V]",
            "param_count": self._count_params(),
            "epochs_trained": int(self.meta.get("epochs_trained", 0)),
            "initial_loss": self.meta.get("initial_loss"),
            "last_loss": self.meta.get("last_loss"),
            "grad_norm": self.meta.get("grad_norm"),
            "tool_accuracy": self.meta.get("tool_accuracy"),
            "role_accuracy": self.meta.get("role_accuracy"),
            "assoc_nodes": len(self.meta.get("assoc_nodes") or []),
            "synapses": synapse_edges,
            "world_forecast": self.predict_world_state(),
            "updated_at": self.meta.get("updated_at"),
        }

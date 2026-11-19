"""Authentication helper supporting Gemini API Key, Google OAuth 2.0, and ADC."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import google.auth
from google import genai
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from .config import CONFIG_DIR, Settings, get_cli_bin_name, get_settings, persist_env_var


OAUTH_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
GCLOUD_ADC_FILE = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"

AGY_MODEL_MAP: dict[str, str] = {
    "gemini-3.5-flash-lite": "gemini-3.8-flash-low",
    "gemini-3.8-flash": "gemini-3.8-flash-low",
    "gemini-4-argon": "gemini-4-argon-low",
    "gemini-4-hydrogen": "gemini-4-hydrogen-low",
}


def _find_agy_cli() -> str | None:
    """Return path to official Google Antigravity (`agy`) CLI binary if installed."""
    found = shutil.which("agy")
    if found:
        return found
    for candidate in ("/usr/bin/agy", str(Path.home() / ".local" / "bin" / "agy")):
        if Path(candidate).is_file():
            return candidate
    return None


def _resolve_agy_model(model: str, thinking_level: str = "low") -> str:
    m = (model or "").strip()
    if m in AGY_MODEL_MAP:
        base = AGY_MODEL_MAP[m]
        if thinking_level in {"medium", "high"} and base.endswith("-low"):
            return base[:-4] + f"-{thinking_level}"
        return base
    return m or "gemini-3.8-flash-low"


class _AgyCliInteractions:
    """Adapter implementing `client.interactions.create(...)` over `agy` CLI (Google AI Pro subscription)."""

    def __init__(self, agy_bin: str) -> None:
        self.agy_bin = agy_bin
        self._history: dict[str, list[str]] = {}

    def _build_prompt(
        self,
        input_data: Any,
        system_instruction: str | None,
        tools: list[dict[str, Any]] | None,
        prev_id: str | None,
    ) -> tuple[str, list[str]]:
        history = list(self._history.get(prev_id or "", []))
        if isinstance(input_data, list):
            res_lines = []
            for item in input_data:
                if isinstance(item, dict) and item.get("type") == "function_result":
                    res_lines.append(f"Tool `{item.get('name')}` returned:\n{item.get('result')}")
                else:
                    res_lines.append(json.dumps(item, ensure_ascii=False))
            history.append("TOOL RESULTS:\n" + "\n\n".join(res_lines))
        else:
            history.append(f"USER INPUT:\n{input_data}")

        parts: list[str] = [
            "IMPORTANT: Answer directly from the prompt context below without invoking any built-in IDE/workspace tools."
        ]
        if system_instruction:
            parts.append(f"SYSTEM INSTRUCTION:\n{system_instruction.strip()}")
        if tools:
            from .self_ops import compact_tool_schemas

            parts.append(
                "AVAILABLE TOOLS:\n"
                + compact_tool_schemas(tools)
                + "\n\nTOOL CALLING PROTOCOL:\n"
                "You can chain multiple tool-calling rounds (e.g. inspect files/logs -> edit/run -> verify tests) before writing a final reply.\n"
                "Whenever you need to invoke one or more tools from AVAILABLE TOOLS before answering, respond ONLY with valid JSON in this exact format (all parameters inside \"arguments\"):\n"
                '{"__tool_calls__": [{"name": "<tool_name>", "arguments": {<args>}}]}\n'
                "NEVER stop after inspecting just one item to announce what you will do next; emit the next `__tool_calls__` JSON immediately.\n"
                "For modifying existing files, ALWAYS prefer `edit_file(path, target, replacement)` over `write_file` or helper patch scripts.\n"
                "Only respond with plain text (not JSON) once the entire requested task is complete and verified."
            )
        if len(history) > 12:
            history = history[-12:]
        compacted: list[str] = []
        for idx, h in enumerate(history):
            if idx < len(history) - 1 and "Tool `read_file` returned:\n" in h:
                import re

                refs = []
                for m in re.finditer(
                    r'"path":\s*"([^"]+)".*?"total_lines":\s*(\d+).*?"start_line":\s*(\d+).*?"end_line":\s*(\d+)',
                    h,
                    re.DOTALL,
                ):
                    refs.append(
                        f"Tool `read_file` inspected {m.group(1)} (L{m.group(3)}-{m.group(4)}/{m.group(2)}) — call read_file again only if needed."
                    )
                if refs:
                    compacted.append("TOOL RESULTS:\n" + "\n".join(refs))
                    continue
            if idx < len(history) - 2 and len(h) > 900:
                compacted.append(h[:900] + "\n... [earlier context compacted]")
            else:
                compacted.append(h)
        parts.extend(compacted)
        return "\n\n---\n\n".join(parts), history

    def _invoke_cli(self, full_prompt: str, agy_model: str) -> str:
        proc = subprocess.run(
            [
                self.agy_bin,
                "-p",
                full_prompt,
                "--model",
                agy_model,
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
            err = (proc.stderr or proc.stdout or "agy CLI failed").strip()
            raise RuntimeError(f"agy CLI error: {err[-600:]}")
        return proc.stdout.strip()

    @staticmethod
    def _extract_calls(raw_txt: str) -> tuple[list[dict[str, Any]] | None, str]:
        """Scan `raw_txt` for a valid __tool_calls__ JSON object, skipping earlier prose/braces."""
        import re

        txt = raw_txt or ""
        decoder = json.JSONDecoder()
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.DOTALL)
        for chunk in [*fenced, txt]:
            pos = 0
            while pos < len(chunk):
                idx = chunk.find("{", pos)
                if idx == -1:
                    break
                try:
                    parsed, end = decoder.raw_decode(chunk[idx:])
                except json.JSONDecodeError:
                    pos = idx + 1
                    continue
                if isinstance(parsed, dict):
                    calls = parsed.get("__tool_calls__") or parsed.get("tool_calls")
                    if isinstance(calls, list) and calls:
                        preamble = chunk[:idx].strip()
                        return calls, preamble
                pos = idx + max(1, end)
        return None, ""

    def create(
        self,
        *,
        model: str = "",
        input: Any = "",
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        previous_interaction_id: str | None = None,
        stream: bool = False,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        import uuid
        from types import SimpleNamespace

        t_level = str((generation_config or {}).get("thinking_level", "low"))
        agy_model = _resolve_agy_model(model, t_level)
        full_prompt, history = self._build_prompt(
            input, system_instruction, tools, previous_interaction_id
        )
        text = self._invoke_cli(full_prompt, agy_model)
        new_id = f"agy-{uuid.uuid4().hex[:10]}"

        if stream:
            def _gen():
                yield SimpleNamespace(
                    event_type="step.delta",
                    delta=SimpleNamespace(type="text", text=text),
                )
            return _gen()

        if tools and ("__tool_calls__" in text or '"tool_calls"' in text):
            raw_calls, preamble_text = self._extract_calls(text)

            if raw_calls is None:
                # 1-shot JSON repair retry so raw __tool_calls__ JSON is never leaked to the user
                repair_prompt = (
                    "Your previous __tool_calls__ JSON failed to parse cleanly. "
                    "Re-emit ONLY valid JSON with all parameters inside \"arguments\": "
                    '{"__tool_calls__": [{"name": "<tool_name>", "arguments": {<args>}}]} '
                    "and use `edit_file` instead of large `write_file` patch scripts."
                )
                try:
                    repaired = self._invoke_cli(f"{full_prompt}\n\n---\n\n{repair_prompt}", agy_model)
                    if "__tool_calls__" in repaired or '"tool_calls"' in repaired:
                        raw_calls, preamble_text = self._extract_calls(repaired)
                    else:
                        text = repaired
                except Exception:
                    pass

            if raw_calls:
                steps = []
                for i, tc in enumerate(raw_calls):
                    if isinstance(tc, dict) and tc.get("name"):
                        args = dict(tc.get("arguments") or {})
                        for extra_k, extra_v in tc.items():
                            if extra_k not in {"name", "arguments"} and extra_k not in args:
                                args[extra_k] = extra_v
                        steps.append(
                            SimpleNamespace(
                                type="function_call",
                                id=f"call_{i}_{uuid.uuid4().hex[:6]}",
                                name=str(tc["name"]),
                                arguments=args,
                            )
                        )
                if steps:
                    if preamble_text:
                        history.append(f"ASSISTANT NOTE:\n{preamble_text[:300]}")
                    history.append(f"ASSISTANT TOOL CALLS:\n{json.dumps(raw_calls, ensure_ascii=False)}")
                    if len(history) > 16:
                        history = [
                            (h[:400] + "\n...[truncated]") if (i < len(history) - 8 and h.startswith("TOOL RESULTS:") and len(h) > 400) else h
                            for i, h in enumerate(history[-28:])
                        ]
                    self._history[new_id] = history
                    return SimpleNamespace(id=new_id, steps=steps, output_text="")

        history.append(f"ASSISTANT:\n{text}")
        if len(history) > 16:
            history = [
                (h[:400] + "\n...[truncated]") if (i < len(history) - 8 and h.startswith("TOOL RESULTS:") and len(h) > 400) else h
                for i, h in enumerate(history[-28:])
            ]
        self._history[new_id] = history
        if len(self._history) > 300:
            oldest = next(iter(self._history))
            self._history.pop(oldest, None)
        return SimpleNamespace(id=new_id, steps=[], output_text=text)


class AgyCliClient:
    """Drop-in `genai.Client` replacement backed by `agy` CLI (Google AI Pro subscription)."""

    def __init__(self, agy_bin: str) -> None:
        self.interactions = _AgyCliInteractions(agy_bin)


def _load_oauth_credentials(settings: Settings) -> Credentials | None:
    token_path = Path(settings.oauth_token_file).expanduser()
    if not token_path.is_file():
        return None
    creds = Credentials.from_authorized_user_file(str(token_path), OAUTH_SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds if creds and creds.valid else None


def _has_adc_file() -> bool:
    """Return True only if an explicit ADC credentials file exists locally (avoids 25s GCE metadata probe)."""
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if gac and Path(gac).expanduser().is_file():
        return True
    return GCLOUD_ADC_FILE.is_file()


def _resolve_gcp_project(settings: Settings, default_proj: str | None = None) -> str | None:
    """Resolve Google Cloud Project ID from settings, ADC quota_project_id, or gcloud config/projects."""
    if settings.google_cloud_project and settings.google_cloud_project.strip():
        return settings.google_cloud_project.strip()
    if default_proj and default_proj.strip():
        return default_proj.strip()
    if GCLOUD_ADC_FILE.is_file():
        try:
            data = json.loads(GCLOUD_ADC_FILE.read_text(encoding="utf-8"))
            qp = data.get("quota_project_id") or data.get("project_id")
            if qp and str(qp).strip():
                return str(qp).strip()
        except Exception:
            pass
    if shutil.which("gcloud"):
        try:
            proc = subprocess.run(
                ["gcloud", "config", "get-value", "project", "--quiet"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            val = proc.stdout.strip()
            if val and val != "(unset)":
                return val
        except Exception:
            pass
        try:
            proc = subprocess.run(
                [
                    "gcloud",
                    "projects",
                    "list",
                    "--filter=lifecycleState:ACTIVE",
                    "--format=value(projectId)",
                    "--limit=1",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=8,
            )
            first = proc.stdout.strip().splitlines()
            if first and first[0].strip():
                return first[0].strip()
        except Exception:
            pass
    return None


def get_auth_status(settings: Settings) -> dict[str, Any]:
    """Return current authentication mode and readiness."""
    pref = getattr(settings, "auth_mode", "auto")

    def _check_agy() -> dict[str, Any] | None:
        agy_bin = _find_agy_cli()
        if agy_bin:
            return {
                "mode": "agy_cli",
                "configured": True,
                "project": "google-ai-pro",
            }
        return None

    def _check_adc() -> dict[str, Any] | None:
        if _has_adc_file():
            try:
                creds, proj = google.auth.default(scopes=OAUTH_SCOPES)
                resolved_proj = _resolve_gcp_project(settings, proj)
                if creds and resolved_proj:
                    return {
                        "mode": "adc",
                        "configured": True,
                        "project": resolved_proj,
                    }
            except Exception:
                pass
        return None

    def _check_oauth() -> dict[str, Any] | None:
        if Path(settings.oauth_token_file).expanduser().is_file():
            resolved_proj = _resolve_gcp_project(settings)
            return {
                "mode": "oauth_token",
                "configured": True,
                "project": resolved_proj,
            }
        return None

    def _check_apikey() -> dict[str, Any] | None:
        if settings.gemini_api_key or settings.gemini_api_keys:
            keys_count = len(settings.gemini_api_keys) or (1 if settings.gemini_api_key else 0)
            return {"mode": "api_key", "configured": True, "project": None, "keys_count": keys_count}
        return None

    def _check_local() -> dict[str, Any]:
        return {"mode": "local_mind", "configured": True, "project": "offline-self-trained"}

    if pref in {"local", "local_mind"}:
        return _check_local()
    if pref in {"agy", "agy_cli"}:
        order = (_check_agy, _check_adc, _check_oauth, _check_apikey)
    elif pref == "adc":
        order = (_check_adc, _check_agy, _check_oauth, _check_apikey)
    elif pref == "oauth":
        order = (_check_oauth, _check_agy, _check_adc, _check_apikey)
    elif pref == "api_key":
        order = (_check_apikey, _check_agy, _check_adc, _check_oauth)
    else:
        # In auto mode, prefer agy_cli (Google AI Pro subscription, no billing/API key limits)
        order = (_check_agy, _check_adc, _check_oauth, _check_apikey)

    for fn in order:
        res = fn()
        if res:
            return res

    return {"mode": "none", "configured": False, "project": None}


def build_genai_client(settings: Settings) -> Any:
    """Create a client using AgyCliClient (Google AI Pro), ADC, saved OAuth token, API Key, or LocalMindClient."""
    status = get_auth_status(settings)
    mode = status.get("mode")

    if mode == "local_mind":
        from .local_mind import LocalMindClient

        return LocalMindClient()

    if mode == "agy_cli":
        agy_bin = _find_agy_cli()
        if agy_bin:
            return AgyCliClient(agy_bin)

    if mode == "adc" and _has_adc_file():
        adc_creds, adc_proj = google.auth.default(scopes=OAUTH_SCOPES)
        proj = _resolve_gcp_project(settings, adc_proj)
        if adc_creds and proj:
            return genai.Client(
                vertexai=True,
                project=proj,
                location=settings.google_cloud_location,
                credentials=adc_creds,
            )

    if mode == "oauth_token":
        creds = _load_oauth_credentials(settings)
        proj = _resolve_gcp_project(settings)
        if creds and proj:
            return genai.Client(
                vertexai=True,
                project=proj,
                location=settings.google_cloud_location,
                credentials=creds,
            )

    if settings.gemini_api_keys:
        return genai.Client(api_key=settings.gemini_api_keys[0])
    if settings.gemini_api_key:
        return genai.Client(api_key=settings.gemini_api_key)

    raise RuntimeError(
        f"No Gemini credentials configured. Run `{get_cli_bin_name()} login` "
        "or set GEMINI_API_KEY in ~/.config/sysagent/.env."
    )


def run_oauth_login(
    settings: Settings,
    client_secrets_path: str | None = None,
    project: str | None = None,
) -> Path:
    """Configure SysAgent to use `agy` CLI (Google AI Pro subscription) or gcloud ADC / OAuth."""
    agy_bin = _find_agy_cli()
    if agy_bin and not client_secrets_path and not project:
        persist_env_var("SYSAGENT_AUTH_MODE", "agy")
        return Path(agy_bin)
    secrets_path = Path(client_secrets_path or settings.oauth_client_file).expanduser()
    if not secrets_path.is_file() and shutil.which("gcloud"):
        subprocess.run(["gcloud", "auth", "application-default", "login"], check=True)
        resolved_proj = project.strip() if (project and project.strip()) else _resolve_gcp_project(settings)
        if not resolved_proj:
            subprocess.run(["gcloud", "auth", "login"], check=False)
            resolved_proj = _resolve_gcp_project(settings)
        if resolved_proj:
            subprocess.run(
                ["gcloud", "config", "set", "project", resolved_proj, "--quiet"],
                check=False,
                capture_output=True,
            )
            subprocess.run(
                ["gcloud", "auth", "application-default", "set-quota-project", resolved_proj, "--quiet"],
                check=False,
                capture_output=True,
            )
            subprocess.run(
                ["gcloud", "services", "enable", "aiplatform.googleapis.com", f"--project={resolved_proj}", "--quiet"],
                check=False,
                capture_output=True,
            )
            persist_env_var("GOOGLE_CLOUD_PROJECT", resolved_proj)
        persist_env_var("SYSAGENT_AUTH_MODE", "adc")
        return GCLOUD_ADC_FILE

    if not secrets_path.is_file():
        raise FileNotFoundError(
            f"OAuth client secrets file not found: {secrets_path}\n"
            f"Either install `gcloud` and run `{get_cli_bin_name()} login`, or download a Desktop App OAuth 2.0 JSON "
            "from Google Cloud Console and pass --client-secret /path/to/client_secret.json."
        )

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), OAUTH_SCOPES)
    creds = flow.run_local_server(port=0)

    token_path = Path(settings.oauth_token_file).expanduser()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")

    if project and project.strip():
        persist_env_var("GOOGLE_CLOUD_PROJECT", project.strip())
    persist_env_var("SYSAGENT_AUTH_MODE", "oauth")
    return token_path


def logout_oauth(settings: Settings) -> bool:
    """Remove saved OAuth token file and/or gcloud ADC file."""
    removed = False
    token_path = Path(settings.oauth_token_file).expanduser()
    if token_path.exists():
        token_path.unlink()
        removed = True
    if GCLOUD_ADC_FILE.exists():
        GCLOUD_ADC_FILE.unlink()
        removed = True
    persist_env_var("SYSAGENT_AUTH_MODE", "auto")
    return removed


class AuthManager:
    """Centralized auth & shared genai.Client initialized once in the main process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: genai.Client | None = None
        self._status: dict[str, Any] = {"mode": "none", "configured": False, "project": None}
        self._initialized = False
        self._key_index: int = 0
        self._keys: list[str] = []

    def initialize(self) -> dict[str, Any]:
        """Resolve credentials and build the single shared genai.Client once."""
        self._keys = [k for k in self.settings.gemini_api_keys if k]
        if self.settings.gemini_api_key and self.settings.gemini_api_key not in self._keys:
            self._keys.insert(0, self.settings.gemini_api_key)
        self._status = get_auth_status(self.settings)
        if self._status["mode"] == "api_key" and self._keys:
            self._status["keys_count"] = len(self._keys)
            self._client = genai.Client(api_key=self._keys[self._key_index % len(self._keys)])
        elif self._status["configured"]:
            self._client = build_genai_client(self.settings)
        else:
            self._client = None
        self._initialized = True
        return self._status

    def rotate_key(self) -> bool:
        """Rotate to the next API key in the pool, or fall back to agy CLI / API key pool if ADC/OAuth/API key fails."""
        if not self._initialized:
            self.initialize()
        mode = self._status.get("mode")
        if mode == "agy_cli" and self._keys:
            self._status["mode"] = "api_key"
            self._status["keys_count"] = len(self._keys)
            self._client = genai.Client(api_key=self._keys[self._key_index % len(self._keys)])
            return True
        if mode in {"adc", "oauth_token"}:
            if self._keys:
                self._status["mode"] = "api_key"
                self._client = genai.Client(api_key=self._keys[self._key_index % len(self._keys)])
                return True
            agy_bin = _find_agy_cli()
            if agy_bin:
                self._status["mode"] = "agy_cli"
                self._status["project"] = "google-ai-pro"
                self._client = AgyCliClient(agy_bin)
                return True
            return False
        if mode == "api_key":
            next_idx = self._key_index + 1
            if next_idx < len(self._keys):
                self._key_index = next_idx
                self._client = genai.Client(api_key=self._keys[self._key_index])
                return True
            agy_bin = _find_agy_cli()
            if agy_bin:
                self._status["mode"] = "agy_cli"
                self._status["project"] = "google-ai-pro"
                self._client = AgyCliClient(agy_bin)
                return True
            if len(self._keys) > 1:
                self._key_index = 0
                self._client = genai.Client(api_key=self._keys[0])
                return True
        return False

    @property
    def is_configured(self) -> bool:
        if not self._initialized:
            self.initialize()
        return bool(self._status["configured"])

    @property
    def status(self) -> dict[str, Any]:
        if not self._initialized:
            self.initialize()
        return self._status

    def get_client(self) -> genai.Client:
        if not self._initialized:
            self.initialize()
        if self._client is None:
            raise RuntimeError(
                "No Gemini credentials configured. Set GEMINI_API_KEY in ~/.config/sysagent/.env "
                f"or run `{get_cli_bin_name()} login` for Google OAuth."
            )
        return self._client

    def reload(self) -> dict[str, Any]:
        """Re-read credentials and refresh the shared client for all modules."""
        from dotenv import load_dotenv
        from .config import ENV_FILE

        if ENV_FILE.exists():
            load_dotenv(ENV_FILE, override=True)
        self.settings = get_settings()
        self._initialized = False
        return self.initialize()

    async def call_interactions(
        self,
        storage: Any,
        model: str,
        timeout_sec: float = 90.0,
        allow_light_fallback: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Unified Gemini Interactions caller with quota recording, key rotation, timeout, and light_model fallback."""
        import asyncio

        async def _invoke(target_model: str) -> Any:
            client = self.get_client()
            if storage is not None:
                storage.record_model_request(target_model)
            return await asyncio.wait_for(
                asyncio.to_thread(client.interactions.create, model=target_model, **kwargs),
                timeout=timeout_sec,
            )

        try:
            return await _invoke(model)
        except Exception as exc:
            def _is_recoverable_auth_or_quota(err_msg: str) -> bool:
                return any(
                    tok in err_msg
                    for tok in (
                        "429",
                        "resource_exhausted",
                        "quota",
                        "403",
                        "permission_denied",
                        "billing",
                        "api_key_invalid",
                    )
                )

            msg = str(exc).lower()
            is_recoverable = _is_recoverable_auth_or_quota(msg)
            if is_recoverable and self.rotate_key():
                try:
                    return await _invoke(model)
                except Exception as retry_exc:
                    exc = retry_exc
                    msg = str(exc).lower()
                    is_recoverable = _is_recoverable_auth_or_quota(msg)
            if (
                allow_light_fallback
                and model != self.settings.light_model
                and (is_recoverable or isinstance(exc, asyncio.TimeoutError))
            ):
                try:
                    return await _invoke(self.settings.light_model)
                except Exception:
                    pass
            if is_recoverable or isinstance(exc, (OSError, TimeoutError, asyncio.TimeoutError)):
                from .local_mind import LocalMindClient

                local_client = LocalMindClient(storage)
                if storage is not None:
                    storage.record_model_request("local-mind")
                return await asyncio.to_thread(
                    local_client.interactions.create, model="local-mind", **kwargs
                )
            raise

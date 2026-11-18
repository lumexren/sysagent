"""Configuration management for SysAgent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


CONFIG_DIR = Path.home() / ".config" / "sysagent"
ENV_FILE = CONFIG_DIR / ".env"
DATA_DIR = Path.home() / ".local" / "share" / "sysagent"


def _load_env() -> None:
    """Load environment variables from ~/.config/sysagent/.env if present."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
    local_env = Path(__file__).resolve().parent.parent / ".env"
    if local_env.exists():
        load_dotenv(local_env, override=False)


_load_env()


@dataclass
class Settings:
    """Runtime settings for the SysAgent daemon and CLI."""

    agent_name: str = field(
        default_factory=lambda: os.environ.get("SYSAGENT_NAME", "SysAgent")
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )
    gemini_api_keys: list[str] = field(
        default_factory=lambda: [
            k.strip()
            for k in os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",")
            if k.strip()
        ]
    )
    google_cloud_project: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    )
    google_cloud_location: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    )
    auth_mode: str = field(
        default_factory=lambda: os.environ.get("SYSAGENT_AUTH_MODE", "auto").lower()
    )
    oauth_client_file: str = field(
        default_factory=lambda: os.environ.get(
            "SYSAGENT_OAUTH_CLIENT", str(CONFIG_DIR / "oauth_client.json")
        )
    )
    oauth_token_file: str = field(
        default_factory=lambda: os.environ.get(
            "SYSAGENT_OAUTH_TOKEN", str(CONFIG_DIR / "oauth_token.json")
        )
    )
    model: str = field(
        default_factory=lambda: os.environ.get("SYSAGENT_MODEL", "gemini-4-argon")
    )
    host: str = field(
        default_factory=lambda: os.environ.get("SYSAGENT_HOST", "127.0.0.1")
    )
    port: int = field(
        default_factory=lambda: int(os.environ.get("SYSAGENT_PORT", "8765"))
    )
    approval_mode: str = field(
        default_factory=lambda: os.environ.get("SYSAGENT_APPROVAL_MODE", "smart")
    )
    locale: str = field(
        default_factory=lambda: os.environ.get("SYSAGENT_LOCALE", "auto")
    )
    watch_dirs: list[str] = field(
        default_factory=lambda: [
            p.strip()
            for p in os.environ.get(
                "SYSAGENT_WATCH_DIRS", str(Path.home() / "Downloads")
            ).split(",")
            if p.strip()
        ]
    )
    monitor_interval_sec: int = field(
        default_factory=lambda: int(os.environ.get("SYSAGENT_MONITOR_INTERVAL", "30"))
    )
    cpu_alert_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SYSAGENT_CPU_ALERT", "90.0"))
    )
    ram_alert_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SYSAGENT_RAM_ALERT", "90.0"))
    )
    disk_alert_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SYSAGENT_DISK_ALERT", "90.0"))
    )
    brain_enabled: bool = field(
        default_factory=lambda: os.environ.get("SYSAGENT_BRAIN_ENABLED", "true").lower() in {"1", "true", "yes"}
    )
    brain_interval_sec: int = field(
        default_factory=lambda: int(os.environ.get("SYSAGENT_BRAIN_INTERVAL", "180"))
    )
    brain_confidence_threshold: int = field(
        default_factory=lambda: int(os.environ.get("SYSAGENT_BRAIN_CONFIDENCE", "80"))
    )
    quiet_hours: str = field(
        default_factory=lambda: os.environ.get("SYSAGENT_QUIET_HOURS", "")
    )
    auth_token: str = field(
        default_factory=lambda: os.environ.get("SYSAGENT_AUTH_TOKEN", "")
    )
    evolve_enabled: bool = field(
        default_factory=lambda: os.environ.get("SYSAGENT_EVOLVE_ENABLED", "true").lower() in {"1", "true", "yes"}
    )
    evolve_interval_sec: int = field(
        default_factory=lambda: int(os.environ.get("SYSAGENT_EVOLVE_INTERVAL", "3600"))
    )
    evolve_max_cpu: float = field(
        default_factory=lambda: float(os.environ.get("SYSAGENT_EVOLVE_MAX_CPU", "35.0"))
    )
    evolve_max_ram: float = field(
        default_factory=lambda: float(os.environ.get("SYSAGENT_EVOLVE_MAX_RAM", "75.0"))
    )
    evolve_auto_apply: bool = field(
        default_factory=lambda: os.environ.get("SYSAGENT_EVOLVE_AUTO_APPLY", "true").lower() in {"1", "true", "yes"}
    )
    worker_model: str = field(
        default_factory=lambda: os.environ.get("SYSAGENT_WORKER_MODEL", "gemini-3.5-flash-lite")
    )
    evolve_model: str = field(
        default_factory=lambda: os.environ.get("SYSAGENT_EVOLVE_MODEL", "gemini-4-argon")
    )
    light_model: str = field(
        default_factory=lambda: os.environ.get("SYSAGENT_LIGHT_MODEL", "gemini-3.5-flash-lite")
    )
    worker_max_concurrency: int = field(
        default_factory=lambda: int(os.environ.get("SYSAGENT_WORKER_CONCURRENCY", "4"))
    )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def get_settings() -> Settings:
    """Return fresh settings instance."""
    _load_env()
    return Settings()


def persist_env_var(key: str, value: str) -> Path:
    """Create or update a key=value entry in ~/.config/sysagent/.env."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines: list[str] = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    os.environ[key] = value
    return ENV_FILE

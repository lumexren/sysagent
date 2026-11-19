"""Centralized logging configuration for SysAgent daemon and CLI."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_ROOT_LOGGER_NAME = "sysagent"
_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the `sysagent` namespace."""
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def default_log_path() -> Path:
    """Return the default rotating log file path under ~/.config/sysagent/logs/."""
    return Path.home() / ".config" / "sysagent" / "logs" / "sysagent.log"


def configure_logging(
    level: str | None = None,
    log_file: Path | str | None = None,
    daemon: bool = False,
    force: bool = False,
) -> logging.Logger:
    """Configure the `sysagent` root logger for stderr/journald and optional rotating file."""
    global _CONFIGURED
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if _CONFIGURED and not force:
        return logger

    env_level = os.getenv("SYSAGENT_LOG_LEVEL", "").strip().upper()
    resolved_name = (level or env_level or ("INFO" if daemon else "WARNING")).upper()
    resolved_level = getattr(logging, resolved_name, logging.INFO if daemon else logging.WARNING)

    logger.setLevel(resolved_level)
    logger.propagate = False
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        try:
            existing.close()
        except OSError:
            pass

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(resolved_level)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    target_file: Path | None = None
    if log_file is not None:
        target_file = Path(log_file).expanduser()
    elif daemon:
        target_file = default_log_path()

    if target_file is not None:
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                target_file,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setLevel(resolved_level)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("Could not initialize file log handler at %s: %s", target_file, exc)

    _CONFIGURED = True
    return logger

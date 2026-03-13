"""Structured logging for Cerberus — delegates to empire_core.logger.

Preserves the StructuredLogger class API for backward compatibility.
All modules can use either:
    from core.logger import StructuredLogger
    logger = StructuredLogger("module_name")
or:
    from core.logger import logger  # pre-configured singleton
"""

from typing import Any

from empire_core.logger import (
    bind_context,
    clear_context,
    get_logger,
    log_error,
    log_retry,
    setup_logging,
    unbind_context,
)

__all__ = [
    "StructuredLogger",
    "bind_context",
    "clear_context",
    "get_logger",
    "log_error",
    "log_retry",
    "logger",
    "setup_logging",
    "unbind_context",
]

setup_logging("cerberus")


class StructuredLogger:
    """Backward-compatible wrapper delegating to empire_core.logger."""

    def __init__(self, name: str, level: str = "INFO", logging_config: dict[str, Any] | None = None):
        self._logger = get_logger(name)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._logger.error(msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._logger.critical(msg, **kwargs)

    def bind(self, **fields: Any) -> "StructuredLogger":
        bound = StructuredLogger.__new__(StructuredLogger)
        bound._logger = self._logger.bind(**fields)
        return bound


# Module-level singleton
logger = get_logger("cerberus")

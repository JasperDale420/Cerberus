"""Structured logging for Cerberus.

Preserves the StructuredLogger class API for backward compatibility.
All modules can use either:
    from core.logger import StructuredLogger
    logger = StructuredLogger("module_name")
or:
    from core.logger import logger  # pre-configured singleton
"""

from typing import Any

import structlog

try:
    from empire_core.logger import (
        bind_context,
        clear_context,
        get_logger,
        log_error,
        log_retry,
        setup_logging,
        unbind_context,
    )
except ModuleNotFoundError:

    def _normalize_log_record(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        if "event" in event_dict:
            event_dict["message"] = event_dict.pop("event")
        if "level" in event_dict:
            event_dict["level"] = str(event_dict["level"]).upper()
        return event_dict

    def setup_logging(service_name: str = "cerberus", **_: Any) -> None:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                _normalize_log_record,
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(20),
            cache_logger_on_first_use=True,
        )

    def get_logger(name: str | None = None) -> Any:
        return structlog.get_logger(name)

    def bind_context(**kwargs: Any) -> None:
        structlog.contextvars.bind_contextvars(**kwargs)

    def unbind_context(*keys: str) -> None:
        structlog.contextvars.unbind_contextvars(*keys)

    def clear_context() -> None:
        structlog.contextvars.clear_contextvars()

    def log_error(logger: Any, message: str, exc: Exception, **kwargs: Any) -> None:
        logger.error(message, error=str(exc), exc_info=True, **kwargs)

    def log_retry(logger: Any, message: str, attempt: int, **kwargs: Any) -> None:
        logger.warning(message, attempt=attempt, **kwargs)


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
    """Backward-compatible wrapper around the configured structured logger."""

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

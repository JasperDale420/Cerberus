"""Structured logging using structlog with JSON output.

Provides the StructuredLogger class used throughout Cerberus for dependency-injected,
context-aware structured logging. All output is JSON-formatted for machine parsing.
"""

import logging
import sys
from collections.abc import MutableMapping
from typing import Any, Optional

import structlog


def _upcase_level(_logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Uppercase the log level to match standard conventions (INFO, ERROR, etc.)."""
    if "level" in event_dict:
        event_dict["level"] = event_dict["level"].upper()
    return event_dict


def _rename_event_to_message(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Rename structlog's 'event' key to 'message' for consistency with existing format."""
    if "event" in event_dict:
        event_dict["message"] = event_dict.pop("event")
    return event_dict


def _configure_structlog(level: str = "INFO") -> None:
    """Configure structlog and stdlib logging for JSON output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _upcase_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _rename_event_to_message,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    # Configure stdlib logging for third-party library output
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )


_configured = False


class StructuredLogger:
    """Wrapper around structlog providing structured JSON logging.

    Maintains the same interface used by all Cerberus modules — accepts keyword
    arguments that are emitted as JSON fields alongside the log message.
    """

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        logging_config: Optional[dict[str, Any]] = None,
    ):
        global _configured
        if not _configured:
            _configure_structlog(level)
            _configured = True
        self._logger = structlog.get_logger(name)

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
        """Return a new StructuredLogger with bound context fields."""
        bound = StructuredLogger.__new__(StructuredLogger)
        bound._logger = self._logger.bind(**fields)
        return bound

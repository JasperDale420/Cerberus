import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional


class JSONFormatter(logging.Formatter):
    """
    Formatter to output logs in JSON format.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": ("WARN" if record.levelname == "WARNING" else record.levelname),
            "module": record.module,
            "message": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
        }

        # Flatten extra fields
        if hasattr(record, "extra"):
            if isinstance(record.extra, dict):
                log_record.update(record.extra)
            else:
                # Fallback if extra is not a dict (though StructuredLogger ensures it is)
                log_record["extra"] = record.extra

        # Also check for 'run_id' directly in record if passed via LoggerAdapter or extra
        if hasattr(record, "run_id"):
            log_record["run_id"] = record.run_id

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, default=str)


def setup_logger(
    name: str, level: str = "INFO", logging_config: Optional[dict[str, Any]] = None
) -> logging.Logger:
    """
    Sets up a logger with JSON formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Prevent adding multiple handlers if logger is already configured
    if not logger.handlers:
        fmt = JSONFormatter()

        console_level = (
            str((logging_config or {}).get("console_level", level)).upper()
            if logging_config
            else level
        )
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(console_level)
        console.setFormatter(fmt)
        logger.addHandler(console)

        if logging_config:
            file_path = (logging_config or {}).get("file_path")
            if file_path:
                file_level = str(
                    (logging_config or {}).get("file_level", level)
                ).upper()
                os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
                fh = logging.FileHandler(file_path)
                fh.setLevel(file_level)
                fh.setFormatter(fmt)
                logger.addHandler(fh)

    return logger


class StructuredLogger:
    """
    Wrapper around logging.Logger to enforce structured logging.
    """

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        logging_config: Optional[dict[str, Any]] = None,
    ):
        self.logger = setup_logger(name, level, logging_config=logging_config)

    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, extra={"extra": kwargs} if kwargs else None)

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra={"extra": kwargs} if kwargs else None)

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, extra={"extra": kwargs} if kwargs else None)

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, extra={"extra": kwargs} if kwargs else None)

    def critical(self, msg: str, **kwargs):
        self.logger.critical(msg, extra={"extra": kwargs} if kwargs else None)

    def bind(self, **fields: Any):
        """
        Returns a lightweight logger-like object that injects `fields` into every call.
        """
        parent = self

        class _Bound:
            def debug(self, msg: str, **kwargs: Any):
                parent.debug(msg, **{**fields, **kwargs})

            def info(self, msg: str, **kwargs: Any):
                parent.info(msg, **{**fields, **kwargs})

            def warning(self, msg: str, **kwargs: Any):
                parent.warning(msg, **{**fields, **kwargs})

            def error(self, msg: str, **kwargs: Any):
                parent.error(msg, **{**fields, **kwargs})

            def critical(self, msg: str, **kwargs: Any):
                parent.critical(msg, **{**fields, **kwargs})

        return _Bound()

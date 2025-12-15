import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """
    Formatter to output logs in JSON format.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
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


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Sets up a logger with JSON formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


class StructuredLogger:
    """
    Wrapper around logging.Logger to enforce structured logging.
    """

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = setup_logger(name, level)

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

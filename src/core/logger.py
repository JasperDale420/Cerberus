import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional

class JSONFormatter(logging.Formatter):
    """
    Formatter to output logs in JSON format.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
        }

        if hasattr(record, "extra"):
            log_record.update(record.extra)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)

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
        self.logger.debug(msg, extra={"extra": kwargs})

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra={"extra": kwargs})

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, extra={"extra": kwargs})

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, extra={"extra": kwargs})

    def critical(self, msg: str, **kwargs):
        self.logger.critical(msg, extra={"extra": kwargs})

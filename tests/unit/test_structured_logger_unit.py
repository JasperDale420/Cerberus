from __future__ import annotations

import json
import logging

import pytest

from src.core.logger import StructuredLogger


@pytest.mark.unit
def test_structured_logger_emits_json_with_extra_fields(caplog) -> None:
    name = "TestStructuredLogger"

    with caplog.at_level(logging.INFO, logger=name):
        logger = StructuredLogger(name, level="INFO")
        logger.info("hello", a=1, b="x")

    assert len(caplog.records) >= 1
    # The structlog JSON renderer formats the full record as the log message.
    raw = caplog.records[-1].getMessage()
    payload = json.loads(raw)

    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"
    assert payload["a"] == 1
    assert payload["b"] == "x"
    assert "timestamp" in payload

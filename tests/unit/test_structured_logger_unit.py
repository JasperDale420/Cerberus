from __future__ import annotations

import json
import logging

import pytest

from src.core.logger import StructuredLogger


@pytest.mark.unit
def test_structured_logger_emits_json_with_extra_fields(capsys) -> None:
    name = "TestStructuredLogger"
    base = logging.getLogger(name)
    base.handlers.clear()

    logger = StructuredLogger(name, level="INFO")
    logger.info("hello", a=1, b="x")

    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) >= 1
    payload = json.loads(out[-1])

    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"
    assert payload["a"] == 1
    assert payload["b"] == "x"
    assert "timestamp" in payload

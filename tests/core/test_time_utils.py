from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any

from structlog.testing import capture_logs

from src.core import time_utils


def _has_message(logs: list[dict[str, Any]], message: str) -> bool:
    for entry in logs:
        if entry.get("event") == message or entry.get("message") == message:
            return True
    return False


def test_in_time_window_str_invalid_time_logs_and_returns_true() -> None:
    dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    with capture_logs() as logs:
        result = time_utils.in_time_window_str(dt, "25:00", "16:00")

    assert result is True
    assert _has_message(logs, "time_window_parse_failed")


def test_in_trading_window_invalid_inputs_logs_and_returns_true() -> None:
    dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    with capture_logs() as logs:
        result = time_utils.in_trading_window(dt, None, time(16, 0))  # type: ignore[arg-type]

    assert result is True
    assert _has_message(logs, "time_window_check_failed")

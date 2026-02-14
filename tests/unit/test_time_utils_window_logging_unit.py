from datetime import datetime, time, timezone
from unittest.mock import MagicMock

import pytest

import src.core.time_utils as time_utils


def test_in_trading_window_logs_and_fails_open_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(time_utils, "logger", logger)

    def _boom(_dt: datetime) -> time:
        raise RuntimeError("tz blew up")

    monkeypatch.setattr(time_utils, "get_eastern_time_of_day", _boom)

    dt = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
    assert time_utils.in_trading_window(dt, time(9, 30), time(16, 0)) is True

    logger.warning.assert_called()
    args, kwargs = logger.warning.call_args
    assert args[0] == "time_window_check_failed"
    assert kwargs["error"] == "tz blew up"


def test_in_time_window_str_logs_and_fails_open_on_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(time_utils, "logger", logger)

    def _boom(_time_str: str) -> time:
        raise ValueError("bad format")

    monkeypatch.setattr(time_utils, "parse_time_string", _boom)

    dt = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
    assert time_utils.in_time_window_str(dt, "09:30", "16:00") is True

    logger.warning.assert_called()
    args, kwargs = logger.warning.call_args
    assert args[0] == "time_window_string_check_failed"
    assert kwargs["error"] == "bad format"

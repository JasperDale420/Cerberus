"""Unit tests for startup session scheduling helpers."""

from datetime import datetime
from zoneinfo import ZoneInfo

from src.main import (
    _is_regular_market_session_local,
    _next_market_open_local,
)


def test_next_market_open_same_day_before_open() -> None:
    """Before 09:30 ET on a weekday, next open should be same day."""
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 2, 12, 8, 0, tzinfo=tz)  # Thursday

    result = _next_market_open_local(now)

    assert result == datetime(2026, 2, 12, 9, 30, tzinfo=tz)


def test_next_market_open_next_day_after_close() -> None:
    """After market close on a weekday, next open should be next weekday at 09:30 ET."""
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 2, 12, 16, 5, tzinfo=tz)  # Thursday

    result = _next_market_open_local(now)

    assert result == datetime(2026, 2, 13, 9, 30, tzinfo=tz)


def test_next_market_open_skips_weekend() -> None:
    """After close on Friday, next open should roll to Monday at 09:30 ET."""
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 2, 13, 16, 5, tzinfo=tz)  # Friday

    result = _next_market_open_local(now)

    assert result == datetime(2026, 2, 16, 9, 30, tzinfo=tz)


def test_regular_market_session_true_during_open_hours() -> None:
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 2, 12, 10, 15, tzinfo=tz)  # Thursday
    assert _is_regular_market_session_local(now) is True


def test_regular_market_session_false_before_open() -> None:
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 2, 12, 9, 20, tzinfo=tz)  # Thursday
    assert _is_regular_market_session_local(now) is False


def test_regular_market_session_false_after_close() -> None:
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 2, 12, 16, 1, tzinfo=tz)  # Thursday
    assert _is_regular_market_session_local(now) is False


def test_regular_market_session_false_weekend() -> None:
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 2, 14, 11, 0, tzinfo=tz)  # Saturday
    assert _is_regular_market_session_local(now) is False

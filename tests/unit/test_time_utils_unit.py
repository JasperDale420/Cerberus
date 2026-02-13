from datetime import datetime, time, timezone

from src.core import time_utils


def test_in_time_window_str_returns_false_for_invalid_time_strings() -> None:
    dt = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)

    assert time_utils.in_time_window_str(dt, "25:00", "16:00") is False
    assert time_utils.in_time_window_str(dt, "09:30", "bad") is False


def test_in_trading_window_handles_overnight_windows() -> None:
    start = time(22, 0)
    end = time(2, 0)

    late = datetime(2024, 1, 2, 23, 0)
    early = datetime(2024, 1, 3, 1, 0)
    outside = datetime(2024, 1, 3, 3, 0)

    assert time_utils.in_trading_window(late, start, end, convert_to_eastern=False) is True
    assert time_utils.in_trading_window(early, start, end, convert_to_eastern=False) is True
    assert time_utils.in_trading_window(outside, start, end, convert_to_eastern=False) is False

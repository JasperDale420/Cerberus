from datetime import datetime, time, timezone

from src.core import time_utils


def test_in_trading_window_cross_midnight_before_midnight() -> None:
    dt = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)
    assert (
        time_utils.in_trading_window(
            dt,
            start=time(20, 0),
            end=time(4, 0),
            convert_to_eastern=False,
        )
        is True
    )


def test_in_trading_window_cross_midnight_after_midnight() -> None:
    dt = datetime(2024, 1, 2, 3, 0, tzinfo=timezone.utc)
    assert (
        time_utils.in_trading_window(
            dt,
            start=time(20, 0),
            end=time(4, 0),
            convert_to_eastern=False,
        )
        is True
    )


def test_in_trading_window_cross_midnight_outside_window() -> None:
    dt = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    assert (
        time_utils.in_trading_window(
            dt,
            start=time(20, 0),
            end=time(4, 0),
            convert_to_eastern=False,
        )
        is False
    )

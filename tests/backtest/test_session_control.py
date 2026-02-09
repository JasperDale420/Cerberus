from datetime import date, datetime, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.backtest.runner import BacktestRunner
from src.core.domain import Bar


@pytest.fixture
def mock_runner():
    # Minimal config for BacktestRunner
    config = {
        "index_symbol": "SPY",
        "timezone": "US/Eastern",
        "backtest": {"rth_only": True, "force_flat_at_1600": True},
    }
    # Mocking dependencies that runner.__init__ might hit
    runner = BacktestRunner.__new__(BacktestRunner)
    runner.config = config
    runner.logger = MagicMock()
    runner.DEFAULT_TIMEZONE = "US/Eastern"
    runner.start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    runner.end_date = datetime(2023, 1, 31, tzinfo=timezone.utc)

    # Manually set attributes that __init__ usually sets
    backtest_cfg = config.get("backtest", {})
    runner.force_flat_at_1600 = bool(backtest_cfg.get("force_flat_at_1600", False))
    runner.rth_only = bool(backtest_cfg.get("rth_only", False))

    return runner


def test_rth_only_filtering(mock_runner):
    # Setup bars: 09:29 (ETH), 09:30 (RTH), 16:00 (RTH), 16:01 (ETH)
    market_tz = ZoneInfo("US/Eastern")
    bars_data = {
        "AAPL": [
            Bar(
                symbol="AAPL",
                time=datetime(2023, 1, 3, 9, 29, tzinfo=market_tz),
                open=100,
                high=101,
                low=99,
                close=100.5,
                volume=1000,
            ),
            Bar(
                symbol="AAPL",
                time=datetime(2023, 1, 3, 9, 30, tzinfo=market_tz),
                open=100.5,
                high=102,
                low=100,
                close=101,
                volume=2000,
            ),
            Bar(
                symbol="AAPL",
                time=datetime(2023, 1, 3, 16, 0, tzinfo=market_tz),
                open=101,
                high=101.5,
                low=100.5,
                close=101.2,
                volume=3000,
            ),
            Bar(
                symbol="AAPL",
                time=datetime(2023, 1, 3, 16, 1, tzinfo=market_tz),
                open=101.2,
                high=101.3,
                low=101.1,
                close=101.2,
                volume=500,
            ),
        ]
    }

    events = mock_runner._build_event_stream(bars_data)

    # Should only have 09:30 and 16:00
    assert len(events) == 2
    # Check times in local (since that's what we passed in and runner preserves it)
    assert events[0][0].hour == 9 and events[0][0].minute == 30
    assert events[1][0].hour == 16 and events[1][0].minute == 0


def test_force_flat_at_1600(mock_runner):
    runner = mock_runner
    runner._session_flattened = {}
    runner._flatten_session_end = MagicMock()
    runner._handle_session_boundary = MagicMock(return_value=(date(2023, 1, 3), None))
    market_tz = ZoneInfo("US/Eastern")

    # Simulate a bar at 16:00 ET
    bt = datetime(2023, 1, 3, 16, 0, tzinfo=market_tz)
    current_session = date(2023, 1, 3)
    bar = Bar(
        symbol="AAPL",
        time=bt,
        open=101,
        high=101.5,
        low=100.5,
        close=101.2,
        volume=3000,
    )

    # Mocking internal state for _process_loop_event_core
    runner.engine = MagicMock()
    runner.engine.symbol_states = {}

    runner._process_loop_event_core(
        bt=bt,
        symbol="AAPL",
        bar=bar,
        market_tz=market_tz,
        last_session_ts=None,
        current_session=current_session,
        index_symbol="SPY",
    )

    runner._flatten_session_end.assert_called_with(ts=bt, reason="STRICT_SESSION_CLOSE")
    assert runner._session_flattened[current_session] is True

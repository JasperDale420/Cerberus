from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

import src.backtest.runner as runner_mod
from src.backtest.runner import BacktestRunner


@pytest.mark.unit
def test_parse_bars_accepts_list_of_dicts_with_iso_timestamps(monkeypatch) -> None:
    fake_loader = MagicMock()
    fake_loader.load_config.return_value = {
        "max_daily_loss": 1_000_000.0,
        "max_risk_per_trade": 50.0,
        "max_notional_per_order": 1_000_000.0,
    }
    monkeypatch.setattr(runner_mod, "ConfigLoader", MagicMock(return_value=fake_loader))
    monkeypatch.setattr(runner_mod, "AlpacaClient", MagicMock())

    # Mock UniverseBuilder to prevent empty universe error
    mock_ub = MagicMock()
    mock_ub.build_universe.return_value = ["AAPL"]
    monkeypatch.setattr(runner_mod, "UniverseBuilder", MagicMock(return_value=mock_ub))

    r = BacktestRunner("ignored", "2025-01-01", "2025-01-02")

    bars = r._parse_bars(
        [
            {
                "t": "2025-01-01T00:00:00Z",
                "o": 100,
                "h": 101,
                "l": 99,
                "c": 100.5,
                "v": 123,
            }
        ],
        symbol="AAPL",
    )
    assert len(bars) == 1
    assert bars[0].symbol == "AAPL"
    assert bars[0].close == 100.5
    assert bars[0].time.tzinfo is not None


def _runner_with_mocked_logger(monkeypatch) -> tuple[BacktestRunner, MagicMock]:
    """Build a minimal BacktestRunner with a mocked logger for parse assertions."""
    fake_loader = MagicMock()
    fake_loader.load_config.return_value = {
        "max_daily_loss": 1_000_000.0,
        "max_risk_per_trade": 50.0,
        "max_notional_per_order": 1_000_000.0,
    }
    monkeypatch.setattr(runner_mod, "ConfigLoader", MagicMock(return_value=fake_loader))
    monkeypatch.setattr(runner_mod, "AlpacaClient", MagicMock())

    mock_ub = MagicMock()
    mock_ub.build_universe.return_value = ["AAPL"]
    monkeypatch.setattr(runner_mod, "UniverseBuilder", MagicMock(return_value=mock_ub))

    fake_logger = MagicMock()
    monkeypatch.setattr(runner_mod, "StructuredLogger", MagicMock(return_value=fake_logger))

    runner = BacktestRunner("ignored", "2025-01-01", "2025-01-02")
    return runner, fake_logger


@pytest.mark.unit
def test_parse_single_bar_preserves_zero_values(monkeypatch) -> None:
    runner, logger = _runner_with_mocked_logger(monkeypatch)

    bar = runner._parse_single_bar(
        {
            "t": "2025-01-01T00:00:00Z",
            "o": 0.0,
            "h": 0.0,
            "l": 0.0,
            "c": 0.0,
            "v": 0.0,
            "open": 100.0,
            "high": 101.0,
            "close": 101.5,
            "low": 99.0,
            "volume": 999.0,
        },
        symbol="AAPL",
    )
    assert bar is not None
    assert bar.open == 0.0
    assert bar.high == 0.0
    assert bar.low == 0.0
    assert bar.close == 0.0
    assert bar.volume == 0.0
    logger.warning.assert_not_called()


@pytest.mark.unit
def test_parse_single_bar_logs_invalid_timestamp(monkeypatch) -> None:
    runner, logger = _runner_with_mocked_logger(monkeypatch)

    bar = runner._parse_single_bar({"t": "not-a-timestamp", "o": 100}, symbol="AAPL")

    assert bar is None
    logger.warning.assert_called_once()
    assert "Failed to parse historical bar timestamp" in str(logger.warning.call_args[0][0])


@pytest.mark.unit
def test_backtest_runner_run_feeds_bars_and_fills_orders(monkeypatch) -> None:
    fake_loader = MagicMock()
    fake_loader.load_config.return_value = {
        "max_daily_loss": 1_000_000.0,
        "max_risk_per_trade": 50.0,
        "max_notional_per_order": 1_000_000.0,
    }
    monkeypatch.setattr(runner_mod, "ConfigLoader", MagicMock(return_value=fake_loader))
    monkeypatch.setattr(runner_mod, "AlpacaClient", MagicMock())

    # Mock UniverseBuilder to prevent empty universe error
    mock_ub = MagicMock()
    mock_ub.build_universe.return_value = ["AAPL"]
    monkeypatch.setattr(runner_mod, "UniverseBuilder", MagicMock(return_value=mock_ub))

    r = BacktestRunner("ignored", "2025-01-01", "2025-01-03")
    r.universe = ["AAPL"]

    r.alpaca_client.get_historical_bars.return_value = {  # type: ignore
        "bars": [
            {
                "t": "2025-01-01T00:00:00Z",
                "o": 100,
                "h": 101,
                "l": 99,
                "c": 100.5,
                "v": 123,
            },
            {
                "t": "2025-01-02T00:00:00Z",
                "o": 101,
                "h": 102,
                "l": 100,
                "c": 101.5,
                "v": 234,
            },
        ]
    }

    r.engine.on_bar = MagicMock()  # type: ignore
    r.mock_executor.fill_pending_for_bar = MagicMock()  # type: ignore
    r.mock_executor.maybe_trigger_bracket_exit = MagicMock()  # type: ignore

    asyncio.run(r.run())

    assert r.mock_executor.fill_pending_for_bar.call_count == 2
    assert r.mock_executor.maybe_trigger_bracket_exit.call_count == 2
    assert r.engine.on_bar.call_count == 2

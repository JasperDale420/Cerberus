from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

import src.backtest.runner as runner_mod
from src.backtest.runner import BacktestRunner


@pytest.mark.unit
def test_parse_bars_accepts_list_of_dicts_with_iso_timestamps(monkeypatch) -> None:
    r = _make_runner(monkeypatch)

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


@pytest.mark.unit
def test_backtest_runner_run_feeds_bars_and_fills_orders(monkeypatch) -> None:
    r = _make_runner(monkeypatch, end_date="2025-01-03")
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


@pytest.mark.unit
def test_parse_single_bar_preserves_zero_values(monkeypatch) -> None:
    r = _make_runner(monkeypatch)

    bar = r._parse_single_bar(
        {
            "t": "2025-01-01T00:00:00Z",
            "o": 0.0,
            "open": 101.0,
            "h": 0.0,
            "high": 105.0,
            "l": 0.0,
            "low": 98.0,
            "c": 0.0,
            "close": 102.0,
            "v": 0.0,
            "volume": 2500.0,
        },
        symbol="AAPL",
    )

    assert bar is not None
    assert bar.open == 0.0
    assert bar.high == 0.0
    assert bar.low == 0.0
    assert bar.close == 0.0
    assert bar.volume == 0.0


@pytest.mark.unit
def test_parse_single_bar_logs_and_skips_invalid_timestamp(monkeypatch) -> None:
    r = _make_runner(monkeypatch)
    r.logger = MagicMock()

    bar = r._parse_single_bar(
        {
            "t": "not-a-timestamp",
            "o": 100.0,
            "h": 101.0,
            "l": 99.0,
            "c": 100.0,
            "v": 1000.0,
        },
        symbol="AAPL",
    )

    assert bar is None
    assert r.logger.error.called
    assert r.logger.error.call_args.kwargs.get("exc_info") is True


@pytest.mark.unit
def test_backtest_runner_rejects_end_before_start(monkeypatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(runner_mod, "StructuredLogger", MagicMock(return_value=logger))

    with pytest.raises(ValueError):
        _make_runner(
            monkeypatch,
            start_date="2025-01-03T00:00:00+00:00",
            end_date="2025-01-01T00:00:00+00:00",
        )

    assert logger.error.called


def _make_runner(
    monkeypatch,
    *,
    start_date: str = "2025-01-01",
    end_date: str = "2025-01-02",
) -> BacktestRunner:
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

    return BacktestRunner("ignored", start_date, end_date)

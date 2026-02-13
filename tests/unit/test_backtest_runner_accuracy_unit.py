from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.backtest.runner as runner_mod
from src.backtest.runner import BacktestRunner


@pytest.mark.unit
def test_backtest_runner_flattens_on_session_boundary_even_without_16_00_bar(
    monkeypatch,
) -> None:
    fake_loader = MagicMock()
    fake_loader.load_config.return_value = {
        "timezone": "US/Eastern",
        "index_symbol": "SPY",
        "max_open_order_age_sec": 0,
        "risk": {},
        "strategies": {},
        "scanner": {"enabled": False},
    }
    monkeypatch.setattr(runner_mod, "ConfigLoader", MagicMock(return_value=fake_loader))
    monkeypatch.setattr(runner_mod, "AlpacaClient", MagicMock())

    mock_ub = MagicMock()
    mock_ub.build_universe.return_value = ["AAPL", "SPY"]
    monkeypatch.setattr(runner_mod, "UniverseBuilder", MagicMock(return_value=mock_ub))

    r = BacktestRunner("ignored", "2025-01-01", "2025-01-03")
    r.universe = ["AAPL", "SPY"]

    # 15:59 ET -> 20:59 UTC on day 1; 09:30 ET -> 14:30 UTC on day 2.
    r.alpaca_client.get_historical_bars.return_value = {  # type: ignore[union-attr]
        "bars": [
            {
                "t": "2025-01-01T20:59:00Z",
                "o": 100,
                "h": 101,
                "l": 99,
                "c": 100.5,
                "v": 123,
            },
            {
                "t": "2025-01-02T14:30:00Z",
                "o": 101,
                "h": 102,
                "l": 100,
                "c": 101.5,
                "v": 234,
            },
        ]
    }

    r.mock_executor.cancel_all_orders = MagicMock()  # type: ignore[assignment]
    r.mock_executor.close_all_positions = MagicMock()  # type: ignore[assignment]

    asyncio.run(r.run())

    # Should flatten on SESSION_END at the day boundary and again at BACKTEST_END.
    assert r.mock_executor.cancel_all_orders.call_count >= 2
    close_calls = [c.kwargs.get("reason") for c in r.mock_executor.close_all_positions.call_args_list]
    assert "SESSION_END" in close_calls


@pytest.mark.unit
def test_backtest_runner_runs_scanner_on_interval(monkeypatch) -> None:
    fake_loader = MagicMock()
    fake_loader.load_config.return_value = {
        "timezone": "US/Eastern",
        "index_symbol": "SPY",
        "risk": {},
        "strategies": {},
        "scanner": {"enabled": True, "interval_minutes": 1, "max_watchlist_size": 1},
    }
    monkeypatch.setattr(runner_mod, "ConfigLoader", MagicMock(return_value=fake_loader))
    monkeypatch.setattr(runner_mod, "AlpacaClient", MagicMock())

    mock_ub = MagicMock()
    mock_ub.build_universe.return_value = ["SPY"]
    monkeypatch.setattr(runner_mod, "UniverseBuilder", MagicMock(return_value=mock_ub))

    r = BacktestRunner("ignored", "2025-01-01", "2025-01-01")
    r.universe = ["SPY"]

    r.alpaca_client.get_historical_bars.return_value = {  # type: ignore[union-attr]
        "bars": [
            {
                "t": "2025-01-01T14:30:00Z",
                "o": 100,
                "h": 101,
                "l": 99,
                "c": 100.5,
                "v": 123,
            },
            {
                "t": "2025-01-01T14:31:00Z",
                "o": 101,
                "h": 102,
                "l": 100,
                "c": 101.5,
                "v": 234,
            },
        ]
    }

    r.engine.run_scan = AsyncMock()  # type: ignore[assignment]

    asyncio.run(r.run())

    assert r.engine.run_scan.call_count >= 1

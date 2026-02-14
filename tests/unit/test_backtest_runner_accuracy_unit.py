from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

import src.backtest.runner as runner_mod
from src.backtest.runner import BacktestRunner
from src.core.settings import Settings


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


@pytest.mark.unit
def test_backtest_runner_logs_scan_exceptions_with_exc_info() -> None:
    runner = BacktestRunner.__new__(BacktestRunner)
    runner.engine = MagicMock()
    runner.engine.run_scan = AsyncMock(side_effect=RuntimeError("scan blew up"))
    runner.logger = MagicMock()

    bt = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
    market_tz = ZoneInfo("US/Eastern")

    asyncio.run(runner._handle_scanner_replay(bt, market_tz, 1, None, None))

    assert runner.logger.error.called
    _, kwargs = runner.logger.error.call_args
    assert kwargs.get("exc_info") is True
    assert "scan blew up" in str(kwargs.get("error"))


@pytest.mark.unit
def test_backtest_runner_uses_data_fetcher_for_gateway_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_loader = MagicMock()
    fake_loader.load_config.return_value = {
        "timezone": "US/Eastern",
        "index_symbol": "SPY",
        "risk": {},
        "strategies": {},
        "scanner": {"enabled": False},
    }
    monkeypatch.setattr(runner_mod, "ConfigLoader", MagicMock(return_value=fake_loader))
    mock_alpaca = MagicMock()
    monkeypatch.setattr(runner_mod, "AlpacaClient", MagicMock(return_value=mock_alpaca))
    monkeypatch.setattr(runner_mod, "UnusualWhalesClient", MagicMock())

    central = MagicMock()
    settings = Settings(
        CERBERUS_DATA_BACKEND="gateway",
        CERBERUS_GATEWAY_URL="http://gateway.test",
        CERBERUS_GATEWAY_KEY="gw_key",
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "CentralApiClient", MagicMock(return_value=central))

    mock_fetcher = MagicMock()
    mock_fetcher.fetch_bars = AsyncMock(
        return_value=(
            [
                {
                    "t": "2025-01-01T14:30:00Z",
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.5,
                    "c": 100.5,
                    "v": 1000,
                }
            ],
            {
                "alpaca_fetch_fail": 0,
                "alpaca_no_bars": 0,
                "cache_hits": 0,
                "incremental_fetches": 0,
            },
        )
    )
    monkeypatch.setattr(runner_mod, "DataFetcher", MagicMock(return_value=mock_fetcher))

    mock_ub = MagicMock()
    mock_ub.build_universe.return_value = ["AAPL"]
    monkeypatch.setattr(runner_mod, "UniverseBuilder", MagicMock(return_value=mock_ub))

    runner = BacktestRunner("ignored", "2025-01-10", "2025-01-11")
    bars = asyncio.run(runner._load_bars_for_symbol("aapl", "1Min"))

    assert len(bars) == 1
    assert bars[0].symbol == "aapl"
    expected_start = runner.start_date - timedelta(days=runner.warmup_days)
    mock_fetcher.fetch_bars.assert_called_once_with(
        "aapl",
        expected_start,
        runner.end_date,
        "1Min",
    )
    mock_alpaca.get_historical_bars.assert_not_called()


@pytest.mark.unit
def test_backtest_runner_universe_builder_receives_central_api_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_loader = MagicMock()
    fake_loader.load_config.return_value = {"timezone": "US/Eastern", "index_symbol": "SPY"}
    monkeypatch.setattr(runner_mod, "ConfigLoader", MagicMock(return_value=fake_loader))
    monkeypatch.setattr(runner_mod, "AlpacaClient", MagicMock())
    monkeypatch.setattr(runner_mod, "UnusualWhalesClient", MagicMock())

    settings = Settings(
        CERBERUS_DATA_BACKEND="gateway",
        CERBERUS_GATEWAY_URL="http://gateway.test",
        CERBERUS_GATEWAY_KEY="gw_key",
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)

    mock_central = MagicMock()
    monkeypatch.setattr(runner_mod, "CentralApiClient", MagicMock(return_value=mock_central))

    mock_fetcher = MagicMock()
    mock_fetcher.fetch_bars = AsyncMock(
        return_value=([], {"alpaca_fetch_fail": 0, "alpaca_no_bars": 0, "cache_hits": 0, "incremental_fetches": 0})
    )
    monkeypatch.setattr(runner_mod, "DataFetcher", MagicMock(return_value=mock_fetcher))

    captured_kwargs: dict[str, object] = {}

    def _universe_builder(**kwargs):
        captured_kwargs.update(kwargs)
        ub = MagicMock()
        ub.build_universe.return_value = ["AAPL"]
        return ub

    monkeypatch.setattr(runner_mod, "UniverseBuilder", _universe_builder)

    BacktestRunner("ignored", "2025-01-10", "2025-01-11")

    assert captured_kwargs.get("central_api_client") is mock_central

"""Integration tests for Heber shadow read parity against gateway inputs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.core.settings import Settings
from src.data.fetcher import DataFetcher


def _build_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "CERBERUS_DATA_BACKEND": "gateway",
        "CERBERUS_STORAGE_BACKEND": "sqlite",
        "CERBERUS_FAILOVER_TO_LEGACY": False,
        "CERBERUS_GATEWAY_URL": "http://gateway.test",
        "CERBERUS_GATEWAY_KEY": "gw_test_key",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_heber_shadow_reads_match_gateway_feature_inputs(tmp_path: Path) -> None:
    start = datetime(2026, 2, 10, 14, 30, tzinfo=UTC)
    end = datetime(2026, 2, 10, 14, 32, tzinfo=UTC)

    _write_parquet(
        tmp_path / "silver" / "feed=bars" / "instrument_type=equity" / "dt=2026-02-10" / "part-0001.parquet",
        [
            {
                "symbol": "AAPL",
                "instrument_key": "equity:AAPL",
                "timeframe": "1Min",
                "bar_start_ts": datetime(2026, 2, 10, 14, 30, tzinfo=UTC),
                "open": 200.0,
                "high": 201.0,
                "low": 199.8,
                "close": 200.5,
                "volume": 1000.0,
                "trade_count": 11,
                "vwap": 200.4,
                "ts_event": datetime(2026, 2, 10, 14, 30, tzinfo=UTC),
                "ts_available": datetime(2026, 2, 10, 14, 30, tzinfo=UTC),
            },
            {
                "symbol": "AAPL",
                "instrument_key": "equity:AAPL",
                "timeframe": "1Min",
                "bar_start_ts": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
                "open": 200.5,
                "high": 201.3,
                "low": 200.2,
                "close": 201.0,
                "volume": 1200.0,
                "trade_count": 13,
                "vwap": 200.9,
                "ts_event": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
                "ts_available": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
            },
        ],
    )

    _write_parquet(
        tmp_path
        / "silver"
        / "feed=trades"
        / "instrument_type=equity"
        / "dt=2026-02-10"
        / "hour=14"
        / "part-0001.parquet",
        [
            {
                "symbol": "AAPL",
                "instrument_key": "equity:AAPL",
                "trade_id": "t1",
                "price": 200.6,
                "size": 100.0,
                "exchange": "XNAS",
                "tape": "C",
                "ts_event": datetime(2026, 2, 10, 14, 30, tzinfo=UTC),
                "ts_available": datetime(2026, 2, 10, 14, 30, tzinfo=UTC),
            },
            {
                "symbol": "AAPL",
                "instrument_key": "equity:AAPL",
                "trade_id": "t2",
                "price": 201.0,
                "size": 80.0,
                "exchange": "XNYS",
                "tape": "A",
                "ts_event": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
                "ts_available": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
            },
        ],
    )

    heber_settings = _build_settings(
        CERBERUS_STORAGE_BACKEND="heber",
        CERBERUS_HEBER_CATALOG_URL="http://heber.test/api/v1",
        CERBERUS_HEBER_DATA_ROOT=str(tmp_path),
    )

    gateway_bars_payload = {
        "bars": [
            {
                "t": "2026-02-10T14:30:00+00:00",
                "o": 200.0,
                "h": 201.0,
                "l": 199.8,
                "c": 200.5,
                "v": 1000.0,
                "n": 11,
                "vw": 200.4,
            },
            {
                "t": "2026-02-10T14:31:00+00:00",
                "o": 200.5,
                "h": 201.3,
                "l": 200.2,
                "c": 201.0,
                "v": 1200.0,
                "n": 13,
                "vw": 200.9,
            },
        ]
    }
    gateway_trades_payload = [
        {"t": "2026-02-10T14:30:00+00:00", "p": 200.6, "s": 100.0, "x": "XNAS", "i": "t1", "z": "C"},
        {"t": "2026-02-10T14:31:00+00:00", "p": 201.0, "s": 80.0, "x": "XNYS", "i": "t2", "z": "A"},
    ]

    legacy_alpaca = MagicMock()
    legacy_uw = MagicMock()
    heber_logger = MagicMock()
    gateway_logger = MagicMock()

    heber_central = MagicMock()
    heber_central.get_alpaca_bars.return_value = gateway_bars_payload
    heber_central.get_alpaca_trades.return_value = gateway_trades_payload

    gateway_central = MagicMock()
    gateway_central.get_alpaca_bars.return_value = gateway_bars_payload
    gateway_central.get_alpaca_trades.return_value = gateway_trades_payload

    with patch("src.data.fetcher.get_settings", return_value=heber_settings):
        heber_fetcher = DataFetcher(
            alpaca_client=legacy_alpaca,
            unusual_whales_client=legacy_uw,
            logger=heber_logger,
            central_api_client=heber_central,
        )

    with patch("src.data.fetcher.get_settings", return_value=_build_settings()):
        gateway_fetcher = DataFetcher(
            alpaca_client=legacy_alpaca,
            unusual_whales_client=legacy_uw,
            logger=gateway_logger,
            central_api_client=gateway_central,
        )

    heber_bars, _ = await heber_fetcher.fetch_bars("AAPL", start, end, "1Min")
    heber_trades, _ = await heber_fetcher.fetch_trades("AAPL", start, end)

    gateway_bars, _ = await gateway_fetcher.fetch_bars("AAPL", start, end, "1Min")
    gateway_trades, _ = await gateway_fetcher.fetch_trades("AAPL", start, end)

    assert heber_bars == gateway_bars
    assert heber_trades == gateway_trades
    heber_central.get_alpaca_bars.assert_not_called()
    heber_central.get_alpaca_trades.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_heber_shadow_falls_back_to_gateway_when_rows_are_not_asof_safe(tmp_path: Path) -> None:
    start = datetime(2026, 2, 10, 14, 30, tzinfo=UTC)
    end = datetime(2026, 2, 10, 14, 32, tzinfo=UTC)

    _write_parquet(
        tmp_path / "silver" / "feed=bars" / "instrument_type=equity" / "dt=2026-02-10" / "part-0001.parquet",
        [
            {
                "symbol": "AAPL",
                "instrument_key": "equity:AAPL",
                "timeframe": "1Min",
                "bar_start_ts": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
                "open": 200.1,
                "high": 200.7,
                "low": 199.9,
                "close": 200.4,
                "volume": 900.0,
                "trade_count": 9,
                "vwap": 200.3,
                "ts_event": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
                "ts_available": datetime(2026, 2, 10, 14, 40, tzinfo=UTC),
            },
        ],
    )

    _write_parquet(
        tmp_path
        / "silver"
        / "feed=trades"
        / "instrument_type=equity"
        / "dt=2026-02-10"
        / "hour=14"
        / "part-0001.parquet",
        [
            {
                "symbol": "AAPL",
                "instrument_key": "equity:AAPL",
                "trade_id": "t_late",
                "price": 200.8,
                "size": 10.0,
                "exchange": "XNAS",
                "tape": "C",
                "ts_event": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
                "ts_available": datetime(2026, 2, 10, 14, 40, tzinfo=UTC),
            },
        ],
    )

    runtime = _build_settings(
        CERBERUS_STORAGE_BACKEND="heber",
        CERBERUS_HEBER_CATALOG_URL="http://heber.test/api/v1",
        CERBERUS_HEBER_DATA_ROOT=str(tmp_path),
    )

    gateway_bars_payload = {
        "bars": [{"t": "2026-02-10T14:30:00+00:00", "o": 199.9, "h": 200.5, "l": 199.7, "c": 200.2, "v": 950.0}]
    }
    gateway_trades_payload = [
        {"t": "2026-02-10T14:30:10+00:00", "p": 200.2, "s": 15.0, "x": "XNAS", "i": "gw_1", "z": "C"}
    ]

    central = MagicMock()
    central.get_alpaca_bars.return_value = gateway_bars_payload
    central.get_alpaca_trades.return_value = gateway_trades_payload

    fetcher = None
    with patch("src.data.fetcher.get_settings", return_value=runtime):
        fetcher = DataFetcher(
            alpaca_client=MagicMock(),
            unusual_whales_client=MagicMock(),
            logger=MagicMock(),
            central_api_client=central,
        )

    bars, _ = await fetcher.fetch_bars("AAPL", start, end, "1Min")
    trades, _ = await fetcher.fetch_trades("AAPL", start, end)

    assert bars == gateway_bars_payload["bars"]
    assert trades == gateway_trades_payload
    central.get_alpaca_bars.assert_called_once()
    central.get_alpaca_trades.assert_called_once()

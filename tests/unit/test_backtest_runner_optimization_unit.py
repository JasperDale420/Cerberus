from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.backtest import runner


@pytest.mark.unit
def test_build_backtest_logger_uses_configured_level(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeLogger:
        def __init__(self, name: str, level: str = "INFO", logging_config=None):
            captured["name"] = name
            captured["level"] = level

    monkeypatch.setattr(runner, "StructuredLogger", FakeLogger)

    logger = runner._build_backtest_logger({"log_level": "ERROR"})

    assert isinstance(logger, FakeLogger)
    assert captured == {"name": "CERBERUS-BACKTEST", "level": "ERROR"}


@pytest.mark.unit
def test_apply_bar_resolution_aggregates_ohlcv_and_vwap() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "timestamp": datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 10.0,
                "vwap": 100.4,
            },
            {
                "symbol": "SPY",
                "timestamp": datetime(2024, 1, 2, 14, 31, tzinfo=timezone.utc),
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 20.0,
                "vwap": 101.0,
            },
            {
                "symbol": "SPY",
                "timestamp": datetime(2024, 1, 2, 14, 32, tzinfo=timezone.utc),
                "open": 101.5,
                "high": 103.0,
                "low": 101.0,
                "close": 102.5,
                "volume": 30.0,
                "vwap": 102.0,
            },
        ]
    )

    result = runner._apply_bar_resolution(df, bar_resolution_minutes=5)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["symbol"] == "SPY"
    assert row["open"] == pytest.approx(100.0)
    assert row["high"] == pytest.approx(103.0)
    assert row["low"] == pytest.approx(99.5)
    assert row["close"] == pytest.approx(102.5)
    assert row["volume"] == pytest.approx(60.0)
    assert row["vwap"] == pytest.approx((100.4 * 10 + 101.0 * 20 + 102.0 * 30) / 60)


@pytest.mark.unit
def test_load_cached_bars_reuses_prepared_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner._BAR_DATAFRAME_CACHE.clear()

    data = pd.DataFrame(
        [
            {
                "timestamp": datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "vwap": 100.2,
                "symbol": "SPY",
            }
        ]
    )
    parquet_path = tmp_path / "SPY_1Min.parquet"
    data.to_parquet(parquet_path)

    read_count = {"calls": 0}
    original_read_parquet = pd.read_parquet

    def counting_read_parquet(*args, **kwargs):
        read_count["calls"] += 1
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counting_read_parquet)

    logger = MagicMock()
    start_dt = datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc)
    end_dt = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)

    first = runner._load_cached_parquet_bars(
        data_dir=tmp_path,
        symbols={"SPY"},
        start_dt=start_dt,
        end_dt=end_dt,
        logger=logger,
        bar_resolution_minutes=1,
    )
    second = runner._load_cached_parquet_bars(
        data_dir=tmp_path,
        symbols={"SPY"},
        start_dt=start_dt,
        end_dt=end_dt,
        logger=logger,
        bar_resolution_minutes=1,
    )

    assert read_count["calls"] == 1
    assert first is second

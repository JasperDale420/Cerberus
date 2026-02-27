from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.backtest.feature_pipeline import BacktestFeaturePipeline
from src.core.domain import Bar


def _mk_bar(symbol: str, ts: datetime, volume: float) -> Bar:
    return Bar(
        symbol=symbol,
        time=ts,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=volume,
    )


@pytest.mark.unit
def test_avg_daily_volume_precomputes_sorted_days_index_and_keeps_lookback_math() -> None:
    logger = MagicMock()
    start = datetime(2026, 1, 5, 16, 0, tzinfo=timezone.utc)

    # Create 4 trading days of bars with increasing daily volume totals.
    bars = []
    for day_offset, daily_volume in enumerate([100.0, 200.0, 300.0, 400.0]):
        day_ts = start + timedelta(days=day_offset)
        bars.append(_mk_bar("AAPL", day_ts, daily_volume))

    pipeline = BacktestFeaturePipeline(
        {"AAPL": bars},
        logger,
        config={"feature_pipeline": {"daily_volume_lookback_days": 2}},
    )

    # Guardrail: this method is called for every symbol scan; runtime sorting in
    # this path causes repeated O(d log d) work where d is tracked days.
    assert pipeline._sorted_volume_days["AAPL"] == sorted(pipeline._sorted_volume_days["AAPL"])

    as_of = start + timedelta(days=3, hours=1)
    avg = pipeline._avg_daily_volume("AAPL", as_of)

    # Last 2 prior days before day 4 are day2/day3 => (200 + 300) / 2
    assert avg == pytest.approx(250.0)


@pytest.mark.unit
def test_avg_daily_volume_falls_back_to_current_day_when_no_prior_days() -> None:
    logger = MagicMock()
    ts = datetime(2026, 1, 5, 16, 0, tzinfo=timezone.utc)
    pipeline = BacktestFeaturePipeline(
        {"AAPL": [_mk_bar("AAPL", ts, 123.0)]},
        logger,
        config={"feature_pipeline": {"daily_volume_lookback_days": 2}},
    )

    assert pipeline._avg_daily_volume("AAPL", ts + timedelta(hours=1)) == pytest.approx(123.0)

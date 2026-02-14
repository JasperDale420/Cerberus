from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.logger import StructuredLogger
from src.data.pipeline import FeaturePipeline


def _logger(name: str) -> StructuredLogger:
    return StructuredLogger(name, level="INFO")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_feature_pipeline_requires_as_of_for_determinism() -> None:
    alpaca = MagicMock()
    uw = MagicMock()
    fp = FeaturePipeline(alpaca, uw, _logger("test_fp_as_of"))
    with pytest.raises(ValueError, match="requires as_of"):
        await fp.compute_features(["AAPL"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_feature_pipeline_computes_true_gap_and_premarket_volume() -> None:
    class _Alpaca:
        def get_historical_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Min"):
            assert symbol == "AAPL"
            assert timeframe == "1Min"
            # Bars cover premarket + session open in UTC (US/Eastern is UTC-5 in Jan).
            return {
                "bars": [
                    {
                        "t": "2025-01-10T13:00:00+00:00",
                        "o": 100,
                        "h": 100,
                        "l": 100,
                        "c": 100,
                        "v": 1000,
                    },
                    {
                        "t": "2025-01-10T14:29:00+00:00",
                        "o": 101,
                        "h": 101,
                        "l": 101,
                        "c": 101,
                        "v": 2000,
                    },
                    {
                        "t": "2025-01-10T14:30:00+00:00",
                        "o": 105,
                        "h": 106,
                        "l": 104,
                        "c": 105,
                        "v": 3000,
                    },
                    {
                        "t": "2025-01-10T14:31:00+00:00",
                        "o": 105,
                        "h": 106,
                        "l": 104,
                        "c": 105,
                        "v": 1000,
                    },
                ]
            }

    class _UW:
        async def get_option_flow(self, *_args: Any, **_kwargs: Any):
            import asyncio

            await asyncio.sleep(0)
            return []

    fp = FeaturePipeline(_Alpaca(), _UW(), _logger("test_fp_gap_premarket"))  # type: ignore
    # Avoid extra daily calls; we only care about gap/premarket logic here.
    # Avoid extra daily calls; we only care about gap/premarket logic here.
    fp.fetcher.fetch_avg_daily_volume = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    fp.fetcher.fetch_prior_day_stats = lambda *_args, **_kwargs: (110.0, 90.0, 100.0)  # type: ignore[method-assign]

    as_of = datetime(2025, 1, 10, 15, 0, tzinfo=timezone.utc)  # 10:00 ET
    out = await fp.compute_features(["AAPL"], as_of=as_of)
    feat = out["AAPL"]
    assert feat.premarket_volume == pytest.approx(3000.0)
    assert feat.gap_pct == pytest.approx(0.05)


@pytest.mark.unit
def test_feature_pipeline_fetch_avg_daily_volume_uses_last_window_and_handles_shapes() -> None:
    alpaca = MagicMock()
    uw = MagicMock()
    fp = FeaturePipeline(alpaca, uw, _logger("test_fp_avg_volume"))

    end = datetime(2025, 1, 10, tzinfo=timezone.utc)
    alpaca.get_historical_bars.return_value = {"bars": [{"v": 10}, {"v": 20}, {"volume": 30}, {"v": None}, {"v": 40}]}
    avg = fp.fetcher.fetch_avg_daily_volume("AAPL", end, lookback_days=3)
    # last 3 valid volumes: 20, 30, 40 (None skipped) => average 30
    assert avg == pytest.approx(30.0)

    alpaca.get_historical_bars.return_value = [{"v": 1}, {"v": 2}, {"v": 3}]
    assert fp.fetcher.fetch_avg_daily_volume("AAPL", end, lookback_days=2) == pytest.approx(2.5)

    alpaca.get_historical_bars.side_effect = RuntimeError("fail")
    assert fp.fetcher.fetch_avg_daily_volume("AAPL", end, lookback_days=2) is None


@pytest.mark.unit
def test_feature_pipeline_fetch_prior_day_stats_filters_to_completed_day() -> None:
    alpaca = MagicMock()
    fp = FeaturePipeline(alpaca, MagicMock(), _logger("test_fp_prior_day"))

    current = datetime(2025, 1, 10, 15, 0, tzinfo=timezone.utc)
    alpaca.get_historical_bars.return_value = {
        "bars": [
            {"t": (current - timedelta(days=2)).isoformat(), "h": 10, "l": 9, "c": 9.5},
            {
                "t": (current - timedelta(days=1)).isoformat(),
                "h": 12,
                "l": 11,
                "c": 11.5,
            },
            {"t": current.isoformat(), "h": 99, "l": 1, "c": 50},
        ]
    }

    high, low, close = fp.fetcher.fetch_prior_day_stats("AAPL", current)
    assert (high, low, close) == (12.0, 11.0, 11.5)


@pytest.mark.unit
def test_feature_pipeline_extracts_closes_from_mixed_bar_shapes() -> None:
    alpaca = MagicMock()
    fp = FeaturePipeline(alpaca, MagicMock(), _logger("test_fp_extract_closes"))

    class _Bar:
        def __init__(self, c: float) -> None:
            self.c = c

    bars = [_Bar(101.0), {"c": 102.5}, {"c": None}, _Bar(103.25)]
    assert fp._extract_closes(bars) == [101.0, 102.5, 0.0, 103.25]


@pytest.mark.unit
def test_feature_pipeline_extracts_closes_handles_invalid_values() -> None:
    alpaca = MagicMock()
    fp = FeaturePipeline(alpaca, MagicMock(), _logger("test_fp_extract_invalid"))

    class _Bar:
        def __init__(self, c: object) -> None:
            self.c = c

    bars = [_Bar("bad"), {"c": "oops"}, {"c": "101.5"}]
    assert fp._extract_closes(bars) == [0.0, 0.0, 101.5]

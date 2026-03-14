"""Integration tests for scanner universe and feature pipeline with UnifiedDataClient."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.pipeline import FeaturePipeline
from src.scanner.universe import UniverseBuilder


@pytest.mark.integration
def test_unified_client_scanner_routes_screener_calls() -> None:
    logger = MagicMock()

    config = {
        "universe": {
            "symbols": [],
            "dynamic": {
                "screener": {
                    "enabled": True,
                    "most_actives_top_n": 2,
                    "movers_top_n": 1,
                }
            },
        }
    }

    unified = MagicMock()
    unified.get_most_actives.return_value = ["AAPL", "MSFT"]
    unified.get_movers.return_value = {"gainers": ["TSLA"], "losers": ["NVDA"]}

    builder = UniverseBuilder(
        unified,
        logger,
        config=config,
    )
    universe = builder.build_universe()

    assert universe == ["AAPL", "MSFT", "TSLA", "NVDA"]
    unified.get_most_actives.assert_called_once_with(top=2)
    unified.get_movers.assert_called_once_with(top=1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unified_client_feature_pipeline_uses_unified_client_for_bars_and_trades() -> None:
    logger = MagicMock()

    bars_payload = {
        "bars": [
            {"t": "2025-01-06T14:30:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000},
            {"t": "2025-01-06T14:31:00Z", "o": 100.5, "h": 101.5, "l": 100.0, "c": 101.0, "v": 1100},
        ]
    }

    unified = MagicMock()
    unified.get_historical_bars.return_value = bars_payload

    uw = AsyncMock()

    config = {
        "feature_pipeline": {"max_concurrency": 1},
        "unusual_whales": {"enabled": False},
    }

    pipeline = FeaturePipeline(
        unified_client=unified,
        unusual_whales_client=uw,
        logger=logger,
        config=config,
    )

    as_of = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)

    pipeline._fetch_supplementary_data = AsyncMock(return_value=(1_000_000.0, (101.0, 99.0, 100.0)))  # type: ignore[method-assign]
    pipeline.calculator.compute_technicals = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            price=101.0,
            atr_pct=0.01,
            volume=1100.0,
            intraday_range_pct=0.02,
            gap_pct=0.0,
            ema20_slope=0.1,
            distance_from_vwap=0.001,
            premarket_volume=0.0,
            adx=25.0,
            distance_from_ema20=0.003,
            prior_day_high=101.0,
            prior_day_low=99.0,
            atr=1.2,
            bb_upper=102.0,
            bb_lower=98.0,
            price_zscore=0.2,
            tfi=0.0,
            timestamp=as_of,
            orb_high=101.5,
            orb_low=100.0,
            frac_diff_close=0.0,
            hurst_exponent=0.5,
        )
    )
    pipeline.calculator.calculate_relative_strength = MagicMock(return_value=0.1)  # type: ignore[method-assign]
    pipeline.calculator.calculate_tfi = MagicMock(return_value=0.0)  # type: ignore[method-assign]
    pipeline.calculator.apply_frac_diff = MagicMock(return_value=0.02)  # type: ignore[method-assign]
    pipeline.calculator.calculate_hurst_exponent = MagicMock(return_value=0.55)  # type: ignore[method-assign]
    pipeline.calculator.calculate_session_open_price = MagicMock(return_value=100.0)  # type: ignore[method-assign]

    features = await pipeline.compute_technicals_only(["AAPL"], as_of=as_of)

    assert "AAPL" in features
    assert features["AAPL"].symbol == "AAPL"
    assert unified.get_historical_bars.call_count >= 2  # SPY benchmark + symbol bars

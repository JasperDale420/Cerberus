from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.domain import Regime, SymbolFeatures
from src.scanner.core import Scanner


def create_mock_features(symbol, ma_dist_200=0.0):
    return SymbolFeatures(
        symbol=symbol,
        price=100.0,
        atr_pct=0.02,
        avg_volume=1000000,
        intraday_range_pct=0.01,
        gap_pct=0.0,
        ema20_slope=0.0,
        ema_trend_strength=0.0,
        distance_from_vwap=0.0,
        premarket_volume=0.0,
        adx=20.0,
        distance_from_ema20=0.0,
        prior_day_high=105.0,
        prior_day_low=95.0,
        bb_upper=110.0,
        bb_lower=90.0,
        price_zscore=0.0,
        flow_zscore=0.0,
        call_put_ratio=1.0,
        large_sweeps_count=0,
        aggressive_flow_share=0.0,
        last_updated=datetime.now(timezone.utc),
        relative_strength=0.0,
        ma_dist_50=0.0,
        ma_dist_200=ma_dist_200,
    )


@pytest.mark.asyncio
async def test_scanner_ranking_and_gating():
    # Setup mocks
    mock_universe = MagicMock()
    mock_universe.get_universe.return_value = ["S1", "S2", "S3"]

    mock_pipeline = MagicMock()

    # S1 is most oversold (Rank 1)
    # S2 is neutral (Rank 2)
    # S3 is overbought (Rank 3)
    features_map = {
        "S1": create_mock_features("S1", ma_dist_200=-0.1),
        "S2": create_mock_features("S2", ma_dist_200=0.0),
        "S3": create_mock_features("S3", ma_dist_200=0.1),
    }

    mock_pipeline.compute_technicals_only = AsyncMock(return_value=features_map)
    mock_pipeline.append_flow_features = AsyncMock(return_value=features_map)

    # Config with alpha_rank_limit = 2
    config = {"scanner": {"alpha_rank_limit": 2, "max_watchlist_size": 10}}

    logger = MagicMock()
    scanner = Scanner(mock_universe, mock_pipeline, logger, config=config)

    # Run scan
    result = await scanner.scan(regime=Regime.CHOP, scan_time=datetime.now(timezone.utc))

    # Verify watchlist
    symbols = [s.symbol for s in result.watchlist]
    assert "S1" in symbols
    assert "S2" in symbols
    assert "S3" not in symbols  # S3 should be gated by rank limit

    # Verify sorting
    assert result.watchlist[0].symbol == "S1"
    assert result.watchlist[0].features.alpha_rank == 1
    assert result.watchlist[1].features.alpha_rank == 2


@pytest.mark.asyncio
async def test_scanner_no_gating_if_not_configured():
    mock_universe = MagicMock()
    mock_universe.get_universe.return_value = ["S1", "S2", "S3"]
    mock_pipeline = MagicMock()
    features_map = {
        "S1": create_mock_features("S1", ma_dist_200=-0.1),
        "S2": create_mock_features("S2", ma_dist_200=0.0),
        "S3": create_mock_features("S3", ma_dist_200=0.1),
    }
    mock_pipeline.compute_technicals_only = AsyncMock(return_value=features_map)
    mock_pipeline.append_flow_features = AsyncMock(return_value=features_map)

    scanner = Scanner(mock_universe, mock_pipeline, MagicMock(), config={})

    result = await scanner.scan(regime=Regime.CHOP, scan_time=datetime.now(timezone.utc))

    # All symbols should pass if no gating
    assert len(result.watchlist) == 3
    assert result.watchlist[0].symbol == "S1"  # Still sorted by alpha score if non-zero

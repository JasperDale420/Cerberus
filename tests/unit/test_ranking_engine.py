import math
from datetime import datetime, timezone

from src.core.domain import SymbolFeatures
from src.scanner.ranking import RankingEngine


def test_ranking_logic():
    engine = RankingEngine()

    # Create test symbols
    # We want symbols with different MA distances and RS to see if ranking works
    # f1: Overstretched down (High AlphaScore for mean reversion)
    # f2: Moderate
    # f3: Overstretched up (Low AlphaScore)

    s1 = SymbolFeatures(
        symbol="S1",
        price=90.0,
        atr_pct=0.02,
        avg_volume=1000000,
        intraday_range_pct=0.01,
        gap_pct=0.0,
        ema20_slope=0.0,
        ema_trend_strength=0.0,
        distance_from_vwap=0.0,
        premarket_volume=0.0,
        adx=20.0,
        distance_from_ema20=-0.1,
        prior_day_high=100.0,
        prior_day_low=90.0,
        bb_upper=110.0,
        bb_lower=90.0,
        price_zscore=-2.0,
        flow_zscore=0.0,
        call_put_ratio=1.0,
        large_sweeps_count=0,
        aggressive_flow_share=0.0,
        last_updated=datetime.now(timezone.utc),
        relative_strength=-0.05,
        ma_dist_50=-0.08,
        ma_dist_200=-0.15,  # STRONGEST SETUP
    )

    s2 = SymbolFeatures(
        symbol="S2",
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
        ma_dist_200=0.0,  # NEUTRAL
    )

    s3 = SymbolFeatures(
        symbol="S3",
        price=110.0,
        atr_pct=0.02,
        avg_volume=1000000,
        intraday_range_pct=0.01,
        gap_pct=0.0,
        ema20_slope=0.0,
        ema_trend_strength=0.0,
        distance_from_vwap=0.0,
        premarket_volume=0.0,
        adx=20.0,
        distance_from_ema20=0.1,
        prior_day_high=115.0,
        prior_day_low=105.0,
        bb_upper=120.0,
        bb_lower=100.0,
        price_zscore=2.0,
        flow_zscore=0.0,
        call_put_ratio=1.0,
        large_sweeps_count=0,
        aggressive_flow_share=0.0,
        last_updated=datetime.now(timezone.utc),
        relative_strength=0.05,
        ma_dist_50=0.08,
        ma_dist_200=0.15,  # WEAKEST SETUP
    )

    symbols = [s2, s1, s3]
    ranked = engine.rank_symbols(symbols)

    assert len(ranked) == 3
    assert ranked[0].symbol == "S1"  # Most oversold should be Rank 1
    assert ranked[1].symbol == "S2"
    assert ranked[2].symbol == "S3"

    assert ranked[0].alpha_rank == 1
    assert ranked[1].alpha_rank == 2
    assert ranked[2].alpha_rank == 3

    assert ranked[0].alpha_score > ranked[1].alpha_score
    assert ranked[1].alpha_score > ranked[2].alpha_score


def test_ranking_with_constant_values():
    engine = RankingEngine()
    s1 = SymbolFeatures(
        symbol="S1",
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
        ma_dist_200=0.0,
    )
    s2 = s1  # Duplicate

    ranked = engine.rank_symbols([s1, s2])
    assert len(ranked) == 2
    assert ranked[0].alpha_score == 0.0
    assert ranked[1].alpha_score == 0.0


def test_ranking_ignores_non_finite_values():
    engine = RankingEngine()
    s1 = SymbolFeatures(
        symbol="S1",
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
        relative_strength=float("nan"),
        ma_dist_50=float("inf"),
        ma_dist_200=float("-inf"),
    )

    s2 = SymbolFeatures(
        symbol="S2",
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
        relative_strength=0.1,
        ma_dist_50=0.1,
        ma_dist_200=0.1,
    )

    ranked = engine.rank_symbols([s1, s2])
    assert len(ranked) == 2
    assert all(math.isfinite(sym.alpha_score) for sym in ranked)

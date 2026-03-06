from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.core.domain import Regime, SymbolFeatures
from src.scanner.core import Scanner


def _features(symbol: str) -> SymbolFeatures:
    return SymbolFeatures(
        symbol=symbol,
        price=100.0,
        atr_pct=0.02,
        avg_volume=1_000_000,
        intraday_range_pct=0.03,
        gap_pct=0.01,
        ema20_slope=0.5,
        ema_trend_strength=0.5,
        distance_from_vwap=0.01,
        premarket_volume=250_000,
        adx=30.0,
        distance_from_ema20=0.01,
        prior_day_high=101.0,
        prior_day_low=99.0,
        bb_upper=102.0,
        bb_lower=98.0,
        price_zscore=0.1,
        flow_zscore=0.0,
        call_put_ratio=1.0,
        large_sweeps_count=0,
        aggressive_flow_share=0.0,
        last_updated=datetime.now(timezone.utc),
        relative_strength=1.0,
        alpha_score=1.0,
        alpha_rank=1,
    )


def _scanner(config: dict) -> Scanner:
    return Scanner(
        universe_builder=MagicMock(),
        feature_pipeline=MagicMock(),
        logger=MagicMock(),
        config=config,
        strategy_profiles={},
    )


def test_build_watchlist_deduplicates_global_strategies_for_survivor_only_symbols() -> None:
    scanner = _scanner(
        {
            "scanner": {"max_watchlist_size": 10},
            "strategy_routing": {"chop": ["fusion_v1", "fusion_v1", "orb"]},
        }
    )
    survivors = {"AAPL": _features("AAPL")}

    watchlist = scanner._build_watchlist(
        candidates=[],
        universe_size=1,
        features_returned=1,
        baseline_filtered=0,
        survivors=survivors,
        regime=Regime.CHOP,
    )

    assert len(watchlist) == 1
    assert watchlist[0].symbol == "AAPL"
    assert watchlist[0].strategies == ["fusion_v1", "orb"]

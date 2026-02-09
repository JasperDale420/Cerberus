from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.analysis.regime import MarketContextService
from src.core.domain import Bar, LiquidityRegime, SessionRegime


def _bar(time_et: datetime, volume: float, spread: float = 0.01) -> Bar:
    # High/Low difference for range_pct calculation
    close = 100.0
    high = close + (spread / 2)
    low = close - (spread / 2)
    return Bar(
        symbol="SPY",
        time=time_et.astimezone(timezone.utc),
        open=close,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


@pytest.mark.unit
def test_session_aware_liquidity_premarket():
    # Premarket: 04:00 - 09:30. Multiplier = 0.1
    market_tz = ZoneInfo("America/New_York")
    svc = MarketContextService(min_bars=1, smooth_k=1, tz="America/New_York")

    # Premarket time
    pm_time = datetime(2023, 1, 3, 8, 0, tzinfo=market_tz)

    # 1. Low volume in premarket (Should be GOOD if > 1e6 * 0.1 = 100k)
    # score = (100 * 2000) / (0.0001 + 1e-6) = 200,000 / 0.000101 ≈ 1.98e9
    # Wait, my formula is: dollar_vol / (range_pct + 1e-6)
    # range_pct = 0.01 / 100 = 0.0001
    # score = 200,000 / 0.000101 = 1,980,198 (Standard GOOD > 1e7, Standard THIN > 1e6)
    # multiplier = 0.1 -> Stressed < 100k, Thin < 1M, Good >= 1M

    bar_good = _bar(pm_time, volume=5, spread=0.01)  # score ~5M
    snap = svc.update(bar_good)
    assert snap.session == SessionRegime.PREMARKET
    assert snap.liquidity == LiquidityRegime.GOOD

    # 2. Same volume in opening (Should be THIN if < 10M but > 1M)
    # Opening multiplier = 1.0 -> Stressed < 1M, Thin < 10M, Good >= 10M
    open_time = datetime(2023, 1, 3, 9, 31, tzinfo=market_tz)
    bar_thin = _bar(open_time, volume=5, spread=0.01)  # score ~5M
    snap = svc.update(bar_thin)
    assert snap.liquidity == LiquidityRegime.THIN


@pytest.mark.unit
def test_session_aware_liquidity_stressed():
    market_tz = ZoneInfo("America/New_York")
    svc = MarketContextService(min_bars=1, smooth_k=1, tz="America/New_York")

    # Midday time (multiplier 1.0)
    mid_time = datetime(2023, 1, 3, 12, 0, tzinfo=market_tz)

    # Very low volume/high spread
    bar_stressed = _bar(mid_time, volume=5, spread=1.0)  # score = 500 / 0.01 = 50k
    snap = svc.update(bar_stressed)
    assert snap.liquidity == LiquidityRegime.STRESSED

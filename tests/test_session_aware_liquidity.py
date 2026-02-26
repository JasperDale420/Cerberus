from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.analysis.regime import MarketContextService
from src.core.domain import Bar, LiquidityRegime


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
def test_session_aware_liquidity_relative():
    market_tz = ZoneInfo("America/New_York")
    svc = MarketContextService(min_bars=1, smooth_k=1, tz="America/New_York", liquidity_window=10)

    # Needs at least 5 bars to establish a relative history.
    base_time = datetime(2023, 1, 3, 10, 0, tzinfo=market_tz)

    # 1. Establish a baseline (medium volume)
    for _ in range(10):
        # Medium volume (~500), standard spread
        svc.update(_bar(base_time, volume=500, spread=0.01))

    # 2. Test Good Liquidity (High relative volume)
    bar_good = _bar(base_time, volume=5000, spread=0.01)
    snap = svc.update(bar_good)
    assert snap.liquidity == LiquidityRegime.GOOD

    # 3. Test Stressed Liquidity (Very low relative volume, high spread)
    bar_stressed = _bar(base_time, volume=5, spread=1.0)
    snap = svc.update(bar_stressed)
    assert snap.liquidity == LiquidityRegime.STRESSED

    # 4. Test Thin Liquidity (Low relative volume, same spread)
    bar_thin = _bar(base_time, volume=50, spread=0.01)
    snap = svc.update(bar_thin)
    assert snap.liquidity == LiquidityRegime.THIN

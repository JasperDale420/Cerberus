from collections import deque
from datetime import datetime, timedelta

import pytest

from src.core.domain import Bar, MarketState, Regime, RiskMode, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.failed_breakout import FailedBreakoutStrategy


class MockLogger(StructuredLogger):
    def __init__(self):
        """Mock implementation."""
        pass

    def info(self, msg, **kwargs):
        """Mock implementation."""
        pass

    def error(self, msg, **kwargs):
        """Mock implementation."""
        pass

    def warning(self, msg, **kwargs):
        """Mock implementation."""
        pass


@pytest.fixture
def fb_strategy():
    config = {"risk_reward": 2.0}
    return FailedBreakoutStrategy(config, MockLogger())


def test_bearish_fade(fb_strategy):
    market_state = MarketState(
        time=datetime.now(),
        regime=Regime.CHOP,  # Strategy works in CHOP
        index_symbol="SPY",
        index_price=100,
        index_return=0,
        realized_vol=0,
        daily_pnl=0,
        risk_mode=RiskMode.NORMAL,
        meta={},
    )

    # 1. Setup Data: Prior Day + Current Day
    bars = []
    # Ensure distinct days regardless of execution time
    now = datetime.now()
    today_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)

    # Prior Day: High 105, Low 95
    bars.append(Bar("TEST", yesterday_start, 100, 105, 100, 102, 1000))
    bars.append(
        Bar("TEST", yesterday_start + timedelta(hours=1), 102, 102, 95, 98, 1000)
    )

    # Current Day:

    # 2. Breakout: Push above 105 (e.g. 106)
    b1 = Bar(
        "TEST", today_start, 104, 106, 104, 105.5, 1000
    )  # Open 104, High 106, Close 105.5

    # 3. Failure: Close < 105
    b2 = Bar(
        "TEST", today_start + timedelta(minutes=5), 105.5, 105.8, 104, 104.5, 1000
    )  # Close 104.5

    symbol_state = SymbolState(
        symbol="TEST",
        bars=deque(bars + [b1]),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    # Run on B1 (Breakout Bar) - Should update state but not trigger (closed valid above?)
    # Or trigger if it reversed inside bar? (High 106, Close 105.5 -> still > 105 (PDH))
    fb_strategy.on_bar("TEST", b1, symbol_state, market_state)
    assert symbol_state.indicators["prior_day_high"] == 105
    assert symbol_state.indicators["prior_day_high"] == 105
    assert symbol_state.indicators["breached_pdh"]

    # Run on B2 (Failure Bar)
    # Append b2 to history first? usually engine does this.
    symbol_state.bars.append(b2)
    sig = fb_strategy.on_bar("TEST", b2, symbol_state, market_state)

    assert sig is not None
    assert sig.side.value == "sell"
    assert sig.strategy == "failed_breakout"
    assert sig.stop_price > sig.entry_price  # Short stop is higher
    # Stop should be b2.high (105.8) or close*1.005?
    # Logic says max(bar.high, ...).

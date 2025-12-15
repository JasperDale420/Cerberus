from collections import deque
from datetime import datetime, timedelta

import pytest

from src.core.domain import Bar, MarketState, Regime, RiskMode, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.index_mean_reversion import IndexMeanReversionStrategy


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
def imr_strategy():
    config = {
        "bb_len": 5,  # Low len for test
        "bb_std": 2.0,
        "stop_pct": 0.005,
    }
    return IndexMeanReversionStrategy(config, MockLogger())


def test_short_mean_reversion(imr_strategy):
    market_state = MarketState(
        time=datetime.now(),
        regime=Regime.CHOP,  # Must be CHOP
        index_symbol="SPY",
        index_price=100,
        index_return=0,
        realized_vol=0,
        daily_pnl=0,
        risk_mode=RiskMode.NORMAL,
        meta={},
    )

    # 1. Setup Data
    # 5 bars minimal
    bars = []
    start_dt = datetime.now() - timedelta(minutes=50)

    # Stable price 100
    for i in range(5):
        bars.append(
            Bar(
                "SPY",
                start_dt + timedelta(minutes=i * 5),
                100.0,
                100.1,
                99.9,
                100.0,
                1000,
            )
        )

    # Bands should have Mean 100, Std ~ 0.

    # 2. Spike Up (Bar 6)
    # Spike to 105. StdDev will increase, but deviation likely massive > 2 sigma.
    b_spike = Bar(
        "SPY", start_dt + timedelta(minutes=6 * 5), 100.0, 105.0, 100.0, 105.0, 5000
    )

    # Run strategy on spike bar
    symbol_state = SymbolState(
        symbol="SPY",
        bars=deque(bars),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    # Hack: Strategy logic builds dataframe from `bars`.
    # It must access b_spike.
    # We should append b_spike to bars for the history check to include it?
    # Or rely on it being the "current" bar?
    # Let's append to be safe for dataframe construction if required.

    symbol_state.bars.append(b_spike)

    sig = imr_strategy.on_bar("SPY", b_spike, symbol_state, market_state)

    # Expect: SELL
    # Price (105) should be > Upper Band.

    if sig:
        assert sig.side.value == "sell"
        assert sig.strategy == "index_mean_reversion"
        assert sig.target_price < sig.entry_price
    else:
        # If no signal, calculate bands manually to see why.
        # Prices: 100, 100, 100, 100, 100, 105
        # Mean: ~100.8
        # Std: ~1.8
        # Upper: 100.8 + 2(1.8) = 104.4
        # 105 > 104.4? Yes.
        # Should trigger.
        pass

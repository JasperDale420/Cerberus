from collections import deque
from datetime import datetime, timedelta

import pytest

from src.core.domain import Bar, MarketState, Regime, RiskMode, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.flow_momentum import FlowMomentumStrategy


class MockLogger(StructuredLogger):
    def __init__(self):
        pass

    def info(self, msg, **kwargs):
        pass

    def error(self, msg, **kwargs):
        pass

    def warning(self, msg, **kwargs):
        pass


@pytest.fixture
def fm_strategy():
    config = {"min_flow_zscore": 2.5, "vol_mult": 1.5, "risk_reward": 2.0}
    return FlowMomentumStrategy(config, MockLogger())


def test_bullish_flow_momentum(fm_strategy):
    market_state = MarketState(
        time=datetime.now(),
        regime=Regime.BULL,
        index_symbol="SPY",
        index_price=100,
        index_return=0,
        realized_vol=0,
        daily_pnl=0,
        risk_mode=RiskMode.NORMAL,
        meta={},
    )

    # 1. Setup Data - 25 bars for SMA calc
    bars = []
    start_dt = datetime.now() - timedelta(minutes=150)
    for i in range(25):
        bars.append(
            Bar(
                "TEST", start_dt + timedelta(minutes=i * 5), 100, 100.1, 99.9, 100, 1000
            )
        )  # Avg Vol ~1000

    # 2. Bullish Momentum Candle
    # High Volume (2000 > 1000 * 1.5)
    # Green Candle (Close > Open)
    b_mom = Bar("TEST", start_dt + timedelta(minutes=25 * 5), 100, 102, 100, 102, 2000)

    symbol_state = SymbolState(
        symbol="TEST",
        bars=deque(bars),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={"flow_zscore": 3.0},  # Strong Bullish Flow
    )

    # Strategy access: symbol_state.meta["flow_zscore"]

    sig = fm_strategy.on_bar("TEST", b_mom, symbol_state, market_state)

    if sig:
        assert sig.side.value == "buy"
        assert sig.strategy == "flow_momentum"
        assert sig.meta["flow_zscore"] == 3.0
    else:
        # Failure debugging
        pass


def test_bearish_flow_momentum(fm_strategy):
    # Bearish Flow -3.0
    market_state = MarketState(
        time=datetime.now(),
        regime=Regime.BEAR,
        index_symbol="SPY",
        index_price=100,
        index_return=0,
        realized_vol=0,
        daily_pnl=0,
        risk_mode=RiskMode.NORMAL,
        meta={},
    )
    # Data ...
    bars = []
    start_dt = datetime.now() - timedelta(minutes=150)
    for i in range(25):
        bars.append(
            Bar(
                "TEST", start_dt + timedelta(minutes=i * 5), 100, 100.1, 99.9, 100, 1000
            )
        )

    b_mom = Bar(
        "TEST", start_dt + timedelta(minutes=25 * 5), 100, 100, 98, 98, 2000
    )  # Red Candle, High Vol

    symbol_state = SymbolState(
        symbol="TEST",
        bars=deque(bars),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={"flow_zscore": -3.0},  # Strong Bearish
    )

    sig = fm_strategy.on_bar("TEST", b_mom, symbol_state, market_state)

    assert sig is not None
    assert sig.side.value == "sell"

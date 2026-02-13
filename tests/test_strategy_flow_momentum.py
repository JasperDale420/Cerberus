from collections import deque
from datetime import datetime, timedelta

import pytest

from src.core.domain import Bar, MarketState, Regime, RiskMode, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.flow_momentum import FlowMomentumStrategy


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
def fm_strategy():
    config = {"min_flow_zscore": 2.5, "vol_mult": 1.5, "risk_reward": 2.0}
    return FlowMomentumStrategy(config, MockLogger())


@pytest.mark.unit
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
        bars.append(Bar("TEST", start_dt + timedelta(minutes=i * 5), 100, 100.1, 99.9, 100, 1000))  # Avg Vol ~1000

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
        meta={
            "flow_zscore": 3.0,
            "call_put_ratio": 2.0,
        },  # Strong Bullish Flow + confirms
    )

    # Strategy access: symbol_state.meta["flow_zscore"]

    sig = fm_strategy.on_bar("TEST", b_mom, symbol_state, market_state)

    if sig:
        assert sig.side.value == "buy"
        assert sig.strategy == "flow_momentum"
        assert sig.meta["flow_zscore"] == pytest.approx(3.0)
        assert sig.meta["call_put_ratio"] == pytest.approx(2.0)
        assert sig.meta["is_bullish_flow"] is True
    else:
        # Failure debugging
        pass


@pytest.mark.unit
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
        bars.append(Bar("TEST", start_dt + timedelta(minutes=i * 5), 100, 100.1, 99.9, 100, 1000))

    b_mom = Bar("TEST", start_dt + timedelta(minutes=25 * 5), 100, 100, 98, 98, 2000)  # Red Candle, High Vol

    symbol_state = SymbolState(
        symbol="TEST",
        bars=deque(bars),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={"flow_zscore": -3.0, "call_put_ratio": 0.5},  # Strong Bearish + confirms
    )

    sig = fm_strategy.on_bar("TEST", b_mom, symbol_state, market_state)

    assert sig is not None
    assert sig.side.value == "sell"
    assert sig.meta["call_put_ratio"] == pytest.approx(0.5)
    assert sig.meta["is_bullish_flow"] is False

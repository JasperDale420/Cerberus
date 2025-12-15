from collections import deque
from datetime import datetime

import pytest

from src.core.domain import Bar, MarketState, Regime, RiskMode, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.orb import ORBStrategy


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
def orb_strategy():
    config = {"orb_minutes": 15, "risk_reward": 2.0, "stop_loss_pct": 0.01}
    return ORBStrategy(config, MockLogger())


def create_bar(t_str: str, o, h, low_px, c):
    dt = datetime.strptime(f"2023-10-27 {t_str}", "%Y-%m-%d %H:%M:%S")
    # Make timezone aware if needed, ORB strategy uses bar.time.time()
    return Bar(symbol="TEST", time=dt, open=o, high=h, low=low_px, close=c, volume=1000)


def test_orb_logic(orb_strategy):
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
    symbol_state = SymbolState(
        symbol="TEST",
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    # 1. Opening Range (09:30 - 09:45)
    # 09:30
    b1 = create_bar("09:30:00", 100, 105, 99, 102)
    sig = orb_strategy.on_bar("TEST", b1, symbol_state, market_state)
    assert sig is None
    assert symbol_state.indicators["orb_high"] == 105
    assert symbol_state.indicators["orb_low"] == 99

    # 09:40 update
    b2 = create_bar("09:40:00", 102, 106, 101, 104)
    sig = orb_strategy.on_bar("TEST", b2, symbol_state, market_state)
    assert sig is None
    assert symbol_state.indicators["orb_high"] == 106  # New High

    # 2. Breakout (09:46)
    b3 = create_bar("09:46:00", 105, 108, 105, 107)  # Close 107 > 106

    # Check assertions again here if needed, but for now logic holds.

    # Force complete flag (logic in on_bar handles this if time >= 09:45)
    # The strategies logic for 'completion' relies on time check.
    # We call with 09:46, it should set complete=True then check breakout.

    sig = orb_strategy.on_bar("TEST", b3, symbol_state, market_state)

    assert sig is not None
    assert sig.side.value == "buy"
    assert sig.strategy == "orb"
    assert sig.stop_price == 99  # Low of range (from b1)


def test_orb_bearish_breakout(orb_strategy):
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
    symbol_state = SymbolState(
        symbol="TEST",
        bars=deque(),
        indicators={"orb_high": 105, "orb_low": 100, "orb_complete": True},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    # Breakout Down
    b = create_bar("09:46:00", 100, 100, 98, 99)  # Close 99 < 100

    sig = orb_strategy.on_bar("TEST", b, symbol_state, market_state)
    assert sig is not None
    assert sig.side.value == "sell"

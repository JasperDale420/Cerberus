from collections import deque
from datetime import datetime, timezone

import pytest
import pytz  # type: ignore

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
    # Interpret input as a US/Eastern session time and convert to UTC timestamps
    # to match production Alpaca bar timestamps.
    et = pytz.timezone("US/Eastern")
    dt_et = et.localize(datetime.strptime(f"2023-10-27 {t_str}", "%Y-%m-%d %H:%M:%S"))
    dt_utc = dt_et.astimezone(timezone.utc)
    return Bar(
        symbol="TEST", time=dt_utc, open=o, high=h, low=low_px, close=c, volume=1000
    )


def test_orb_logic(orb_strategy):
    market_state = MarketState(
        time=datetime.now(timezone.utc),
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
        meta={"gap_pct": 0.02, "flow_zscore": 3.0, "premarket_volume": 12345.0},
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
    assert sig.meta.get("gap_pct") == pytest.approx(0.02)
    assert sig.meta.get("flow_zscore") == pytest.approx(3.0)
    assert sig.meta.get("premarket_volume") == pytest.approx(12345.0)


def test_orb_bearish_breakout(orb_strategy):
    market_state = MarketState(
        time=datetime.now(timezone.utc),
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
        meta={"gap_pct": -0.02, "flow_zscore": -3.0, "premarket_volume": 20000.0},
    )

    # Breakout Down
    b = create_bar("09:46:00", 100, 100, 98, 99)  # Close 99 < 100

    sig = orb_strategy.on_bar("TEST", b, symbol_state, market_state)
    assert sig is not None
    assert sig.side.value == "sell"
    assert sig.meta.get("gap_pct") == pytest.approx(-0.02)
    assert sig.meta.get("flow_zscore") == pytest.approx(-3.0)
    assert sig.meta.get("premarket_volume") == pytest.approx(20000.0)

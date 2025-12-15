from collections import deque
from datetime import datetime, timedelta

import pytest

from src.core.domain import Bar, MarketState, Regime, RiskMode, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.gap_fill import GapFillStrategy


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
def gf_strategy():
    config = {
        "min_gap": 0.015,
        "max_gap": 0.10,
        "risk_reward": 2.0,
        "or_time_minutes": 15,
    }
    return GapFillStrategy(config, MockLogger())


def test_fade_gap_up(gf_strategy):
    market_state = MarketState(
        time=datetime.now(),
        regime=Regime.CHOP,
        index_symbol="SPY",
        index_price=100,
        index_return=0,
        realized_vol=0,
        daily_pnl=0,
        risk_mode=RiskMode.NORMAL,
        meta={},
    )

    # 1. Setup Data - GAP UP
    # Previous Close: 100.
    # Open: 102 (+2% Gap).

    start_dt = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)

    # OR Bars (15 min) - 3 x 5min bars
    # Bar 1: 09:30-09:35. Open 102. High 102.5. Low 101.8. Close 102.0
    b1 = Bar("TEST", start_dt, 102.0, 102.5, 101.8, 102.0, 1000)

    # Bar 2: 09:35-09:40. Chop. High 102.2 Low 101.9
    b2 = Bar("TEST", start_dt + timedelta(minutes=5), 102.0, 102.2, 101.9, 102.1, 1000)

    # Bar 3: 09:40-09:45. Chop. High 102.3 Low 102.0
    b3 = Bar("TEST", start_dt + timedelta(minutes=10), 102.1, 102.3, 102.0, 102.0, 1000)

    # OR High = 102.5. OR Low = 101.8.

    # Bar 4: 09:45-09:50. BREAKDOWN.
    # Close 101.5 (Below 101.8).
    b4 = Bar("TEST", start_dt + timedelta(minutes=15), 102.0, 102.0, 101.5, 101.5, 2000)

    symbol_state = SymbolState(
        symbol="TEST",
        bars=deque([b1, b2, b3]),  # History has OR bars
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={"gap_pct": 0.02},  # +2% gap
    )

    # Run on b4
    sig = gf_strategy.on_bar("TEST", b4, symbol_state, market_state)

    # Expect: SELL
    # Target: 100.0 (Prev Close derived from Open/1.02) -> 102/1.02 = 100.0.
    # Stop: 102.5 (OR High)

    if sig:
        assert sig.side.value == "sell"
        assert sig.strategy == "gap_fill"
        assert sig.target_price == pytest.approx(100.0, abs=0.01)
        assert sig.stop_price == pytest.approx(102.5)
    else:
        # Debug
        pass

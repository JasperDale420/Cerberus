from collections import deque
from datetime import datetime, timezone

import pytest

from src.core.domain import MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.engine.risk import RiskManager


@pytest.fixture
def risk_manager():
    config = {
        "risk": {
            "max_risk_per_trade": 100.0,
            "alpha_rank_multipliers": {1: 1.5, 2: 1.0, 5: 0.5},
        }
    }
    return RiskManager(config, StructuredLogger("TestRisk"))


def test_alpha_rank_scaling(risk_manager):
    # Base setup: risk_per_share = 1.0 ($100 risk -> 100 qty)
    # Rank 1: 1.5x risk -> $150 risk -> 150 qty
    # Rank 2: 1.0x risk -> $100 risk -> 100 qty
    # Rank 5: 0.5x risk -> $50 risk -> 50 qty
    # Rank 10: (defaults to last tier in tier search logic? No, my logic uses next tier >= rank_int.
    # If no tier >= rank_int exists, it uses the last tier. So Rank 10 -> 0.5x)

    market_state = MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP)

    def create_signal(symbol, rank):
        return Signal(
            symbol=symbol,
            side=OrderSide.BUY,
            size_hint=0,
            entry_price=10.0,
            stop_price=9.0,  # risk_per_share = 1.0
            target_price=15.0,
            strategy="test",
            generated_at=datetime.now(timezone.utc),
            meta={"alpha_rank": rank},
        )

    # Test Rank 1
    s1 = create_signal("S1", 1)
    state1 = SymbolState(
        symbol="S1",
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    intents1 = risk_manager.apply(s1, state1, market_state, account_equity=100000.0)
    assert len(intents1) == 1
    assert intents1[0].qty == 150  # 1.5x risk

    # Test Rank 2
    s2 = create_signal("S2", 2)
    state2 = SymbolState(
        symbol="S2",
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    intents2 = risk_manager.apply(s2, state2, market_state, account_equity=100000.0)
    assert len(intents2) == 1
    assert intents2[0].qty == 100  # 1.0x risk

    # Test Rank 4 (should use Tier 5 multiplier: 0.5x)
    s4 = create_signal("S4", 4)
    state4 = SymbolState(
        symbol="S4",
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    intents4 = risk_manager.apply(s4, state4, market_state, account_equity=100000.0)
    assert len(intents4) == 1
    assert intents4[0].qty == 50  # 0.5x risk

    # Test Rank 10 (should use last tier: 0.5x)
    s10 = create_signal("S10", 10)
    state10 = SymbolState(
        symbol="S10",
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    intents10 = risk_manager.apply(s10, state10, market_state, account_equity=100000.0)
    assert len(intents10) == 1
    assert intents10[0].qty == 50  # 0.5x risk


def test_no_rank_defaults_to_1x(risk_manager):
    market_state = MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP)
    signal = Signal(
        symbol="S1",
        side=OrderSide.BUY,
        size_hint=0,
        entry_price=10.0,
        stop_price=9.0,
        target_price=15.0,
        strategy="test",
        generated_at=datetime.now(timezone.utc),
        meta={},  # No rank
    )
    state = SymbolState(
        symbol="S1",
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    intents = risk_manager.apply(signal, state, market_state, account_equity=100000.0)
    assert len(intents) == 1
    assert intents[0].qty == 100  # 1.0x risk

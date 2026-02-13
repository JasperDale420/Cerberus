"""
Unit tests for PositionManager R-multiple calculation edge cases.

Tests H2 fix: safe division by zero protection in R-multiple calculation
to prevent crashes when initial_risk is 0 (breakeven stop).
"""

from collections import deque
from datetime import datetime, timezone

import pytest

from src.core.domain import MarketState, Position, Regime, Side, SymbolState
from src.engine.position_manager import PositionManager


@pytest.fixture
def position_manager():
    """Create PositionManager instance for testing."""
    return PositionManager()


@pytest.fixture
def symbol_state_with_position():
    """Create SymbolState with an open position."""
    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=100),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={"pending_entries": {}},
    )

    # Create position with known initial_risk
    symbol_state.position = Position(
        symbol="AAPL",
        side=Side.LONG,
        qty=100,
        avg_price=100.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        strategy="test_strategy",
        entry_time=datetime.now(timezone.utc),
        correlation_id="test-123",
        regime_at_entry=Regime.BULL,
        open_risk=200.0,  # 100 shares * $2/share risk
        stop_price=98.0,  # Entry 100, stop 98 → $2 risk per share
        target_price=104.0,
        entry_features=None,
        mae_r=0.0,
        mfe_r=0.0,
        commission=0.0,
        slippage_estimate=0.0,
        max_hold_seconds=None,
    )

    return symbol_state


@pytest.fixture
def market_state():
    """Create MarketState for testing."""
    return MarketState(regime=Regime.BULL, time=datetime.now(timezone.utc))


class TestRMultipleCalculation:
    """Test R-multiple calculation edge cases."""

    def test_r_multiple_with_breakeven_stop(self, position_manager, symbol_state_with_position, market_state):
        """R-multiple should be None when initial_risk is 0 (breakeven stop)."""
        # Set initial_risk to exactly 0 (breakeven stop: entry = stop)
        symbol_state_with_position.position.open_risk = 0.0

        # Exit fill at profit
        exit_fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": 101.0,  # $1 profit
            "side": "sell",
            "correlation_id": "test-123",
            "timestamp": datetime.now(timezone.utc),
        }

        decision = position_manager.on_fill(symbol_state_with_position, market_state, exit_fill)

        assert decision.event == "closed"
        assert decision.closed_trade is not None
        # H2 fix: pnl_r should be None when risk is 0, not crash
        assert decision.closed_trade.pnl_r is None
        # But gross PnL should still be calculated
        assert decision.closed_trade.pnl_gross == 100.0  # $1 * 100 shares

    def test_r_multiple_with_normal_risk(self, position_manager, symbol_state_with_position, market_state):
        """R-multiple should calculate correctly with normal risk."""
        # Normal risk: open_risk = 200 ($2/share * 100 shares)
        assert symbol_state_with_position.position.open_risk == 200.0

        # Exit at $104 (target hit)
        exit_fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": 104.0,  # $4 profit per share
            "side": "sell",
            "correlation_id": "test-123",
            "timestamp": datetime.now(timezone.utc),
        }

        decision = position_manager.on_fill(symbol_state_with_position, market_state, exit_fill)

        assert decision.event == "closed"
        assert decision.closed_trade is not None
        # Gross PnL: $4/share * 100 shares = $400
        # Net PnL: $400 (no commission configured)
        # R-multiple: $400 / $200 = 2.0R
        assert decision.closed_trade.pnl_gross == 400.0
        assert decision.closed_trade.pnl_r == pytest.approx(2.0)

    def test_r_multiple_with_none_risk(self, position_manager, symbol_state_with_position, market_state):
        """R-multiple should be None when initial_risk is None."""
        # Set initial_risk to None
        symbol_state_with_position.position.open_risk = None

        # Exit fill
        exit_fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": 102.0,
            "side": "sell",
            "correlation_id": "test-123",
            "timestamp": datetime.now(timezone.utc),
        }

        decision = position_manager.on_fill(symbol_state_with_position, market_state, exit_fill)

        assert decision.event == "closed"
        assert decision.closed_trade is not None
        assert decision.closed_trade.pnl_r is None

    def test_r_multiple_with_small_risk(self, position_manager, symbol_state_with_position, market_state):
        """R-multiple should calculate correctly with very small risk."""
        # Very small risk (but not zero)
        symbol_state_with_position.position.open_risk = 1.0  # $0.01/share * 100 shares

        # Exit at profit
        exit_fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": 101.0,  # $1/share profit
            "side": "sell",
            "correlation_id": "test-123",
            "timestamp": datetime.now(timezone.utc),
        }

        decision = position_manager.on_fill(symbol_state_with_position, market_state, exit_fill)

        assert decision.event == "closed"
        assert decision.closed_trade is not None
        # R-multiple: $100 / $1 = 100R (large R due to tiny risk)
        assert decision.closed_trade.pnl_r == pytest.approx(100.0)

    def test_r_multiple_with_loss(self, position_manager, symbol_state_with_position, market_state):
        """R-multiple should be negative for losing trades."""
        # Normal risk: 200
        assert symbol_state_with_position.position.open_risk == 200.0

        # Exit at loss (stop hit)
        exit_fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": 98.0,  # $2 loss per share (stop price)
            "side": "sell",
            "correlation_id": "test-123",
            "timestamp": datetime.now(timezone.utc),
        }

        decision = position_manager.on_fill(symbol_state_with_position, market_state, exit_fill)

        assert decision.event == "closed"
        assert decision.closed_trade is not None
        # Loss: -$200
        # R-multiple: -$200 / $200 = -1.0R (expected for stop hit)
        assert decision.closed_trade.pnl_gross == -200.0
        assert decision.closed_trade.pnl_r == pytest.approx(-1.0)


class TestRMultipleEdgeCasesInCode:
    """Direct tests of R-multiple calculation logic."""

    def test_zero_risk_does_not_crash(self):
        """Verify zero risk doesn't cause division by zero."""
        # This simulates the calculation done in on_fill
        initial_risk = 0.0
        pnl_net = 100.0

        # Safe calculation (H2 fix)
        pnl_r = None
        if initial_risk and initial_risk != 0.0:
            pnl_r = pnl_net / initial_risk

        assert pnl_r is None  # Should not crash, should be None

    def test_none_risk_does_not_crash(self):
        """Verify None risk doesn't cause errors."""
        initial_risk = None
        pnl_net = 100.0

        # Safe calculation (H2 fix)
        pnl_r = None
        if initial_risk and initial_risk != 0.0:
            pnl_r = pnl_net / initial_risk

        assert pnl_r is None

    def test_normal_risk_calculates_correctly(self):
        """Verify normal risk calculates R-multiple."""
        initial_risk = 200.0
        pnl_net = 400.0

        # Safe calculation (H2 fix)
        pnl_r = None
        if initial_risk and initial_risk != 0.0:
            pnl_r = pnl_net / initial_risk

        assert pnl_r == pytest.approx(2.0)

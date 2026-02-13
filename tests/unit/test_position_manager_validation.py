"""
Unit tests for PositionManager fill input validation.

Tests H4 fix: comprehensive validation of fill events to prevent
position state corruption from malformed data.
"""

from datetime import datetime, timezone

import pytest

from src.core.domain import MarketState, Regime, SymbolState
from src.engine.position_manager import PositionManager


@pytest.fixture
def position_manager():
    """Create PositionManager instance for testing."""
    return PositionManager()


@pytest.fixture
def symbol_state():
    """Create SymbolState for testing."""
    from collections import deque

    return SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=100),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )


@pytest.fixture
def market_state():
    """Create MarketState for testing."""
    return MarketState(regime=Regime.BULL, time=datetime.now(timezone.utc))


class TestFillValidation:
    """Test fill input validation in on_fill()."""

    def test_negative_qty_rejected(self, position_manager, symbol_state, market_state):
        """Fill with negative qty should be ignored."""
        fill = {
            "symbol": "AAPL",
            "qty": -100,
            "price": 150.0,
            "side": "buy",
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "ignored"
        assert decision.realized_pnl_delta == 0.0
        assert symbol_state.position is None  # No position created

    def test_zero_qty_rejected(self, position_manager, symbol_state, market_state):
        """Fill with zero qty should be ignored."""
        fill = {
            "symbol": "AAPL",
            "qty": 0,
            "price": 150.0,
            "side": "buy",
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "ignored"
        assert symbol_state.position is None

    def test_negative_price_rejected(self, position_manager, symbol_state, market_state):
        """Fill with negative price should be ignored."""
        fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": -150.0,
            "side": "buy",
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "ignored"
        assert symbol_state.position is None

    def test_zero_price_rejected(self, position_manager, symbol_state, market_state):
        """Fill with zero price should be ignored."""
        fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": 0.0,
            "side": "buy",
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "ignored"
        assert symbol_state.position is None

    def test_infinite_qty_rejected(self, position_manager, symbol_state, market_state):
        """Fill with infinite qty should be ignored."""
        fill = {
            "symbol": "AAPL",
            "qty": float("inf"),
            "price": 150.0,
            "side": "buy",
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "ignored"
        assert symbol_state.position is None

    def test_nan_price_rejected(self, position_manager, symbol_state, market_state):
        """Fill with NaN price should be ignored."""
        fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": float("nan"),
            "side": "buy",
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "ignored"
        assert symbol_state.position is None

    def test_invalid_side_rejected(self, position_manager, symbol_state, market_state):
        """Fill with invalid side should be ignored."""
        fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": 150.0,
            "side": "hold",  # Invalid
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "ignored"
        assert symbol_state.position is None

    def test_empty_side_rejected(self, position_manager, symbol_state, market_state):
        """Fill with empty side should be ignored."""
        fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": 150.0,
            "side": "",
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "ignored"
        assert symbol_state.position is None

    def test_uppercase_side_accepted(self, position_manager, symbol_state, market_state):
        """Fill with uppercase side should be accepted (normalized to lowercase)."""
        fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": 150.0,
            "side": "BUY",  # Uppercase should be normalized
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "opened"
        assert symbol_state.position is not None
        assert symbol_state.position.qty == 100

    def test_none_qty_rejected(self, position_manager, symbol_state, market_state):
        """Fill with None qty should be ignored."""
        fill = {
            "symbol": "AAPL",
            "qty": None,
            "price": 150.0,
            "side": "buy",
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "ignored"
        assert symbol_state.position is None

    def test_string_qty_rejected(self, position_manager, symbol_state, market_state):
        """Fill with non-numeric qty should be ignored."""
        fill = {
            "symbol": "AAPL",
            "qty": "invalid",
            "price": 150.0,
            "side": "buy",
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "ignored"
        assert symbol_state.position is None

    def test_valid_fill_accepted(self, position_manager, symbol_state, market_state):
        """Valid fill should be processed normally."""
        fill = {
            "symbol": "AAPL",
            "qty": 100,
            "price": 150.0,
            "side": "buy",
            "correlation_id": "test-123",
        }

        decision = position_manager.on_fill(symbol_state, market_state, fill)

        assert decision.event == "opened"
        assert symbol_state.position is not None
        assert symbol_state.position.qty == 100
        assert symbol_state.position.avg_price == 150.0


class TestValidateFillMethod:
    """Test _validate_fill() helper method directly."""

    def test_validate_fill_valid(self, position_manager):
        """Valid fill returns None (no error)."""
        fill = {"qty": 100, "price": 150.0, "side": "buy"}
        error = position_manager._validate_fill(fill)
        assert error is None

    def test_validate_fill_negative_qty(self, position_manager):
        """Negative qty returns error."""
        fill = {"qty": -100, "price": 150.0, "side": "buy"}
        error = position_manager._validate_fill(fill)
        assert error is not None
        assert "qty must be positive" in error

    def test_validate_fill_infinite_price(self, position_manager):
        """Infinite price returns error."""
        fill = {"qty": 100, "price": float("inf"), "side": "buy"}
        error = position_manager._validate_fill(fill)
        assert error is not None
        assert "finite" in error.lower()

    def test_validate_fill_invalid_type(self, position_manager):
        """Invalid type returns error."""
        fill = {"qty": "not_a_number", "price": 150.0, "side": "buy"}
        error = position_manager._validate_fill(fill)
        assert error is not None
        assert "invalid" in error.lower()

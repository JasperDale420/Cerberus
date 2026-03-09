"""Tests for partial exit levels and trail_min_profit_r gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pytest

from src.core.domain import (
    MarketState,
    OrderSide,
    Position,
    Regime,
    Side,
    SymbolState,
)
from src.engine.position_manager import PositionManager


@dataclass
class FakeBar:
    """Minimal bar stub for tests."""

    symbol: str = "AAPL"
    time: datetime = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
    open: float = 150.0
    high: float = 152.0
    low: float = 149.0
    close: float = 151.0
    volume: float = 1000.0


def _make_market_state() -> MarketState:
    return MarketState(
        time=datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc),
        regime=Regime.BULL,
    )


def _make_position(**overrides: Any) -> Position:
    defaults = {
        "symbol": "AAPL",
        "side": Side.LONG,
        "qty": 100.0,
        "avg_price": 150.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "strategy": "trend_rider_pro",
        "entry_time": datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc),
        "initial_qty": 100.0,
        "open_risk": 200.0,  # $2 per share * 100 shares
        "stop_price": 148.0,
        "trailing_stop_enabled": True,
        "trailing_stop_pct": 0.02,
        "partial_exit_levels": [],
        "trail_min_profit_r": None,
    }
    defaults.update(overrides)
    return Position(**defaults)


def _make_symbol_state(pos: Optional[Position] = None) -> SymbolState:
    from collections import deque

    bar = FakeBar()
    return SymbolState(
        symbol="AAPL",
        bars=deque([bar], maxlen=100),
        position=pos,
    )


# ---- Partial Exit Tests ----


@pytest.mark.unit
class TestPartialExitLevels:
    """Test strategy-specific partial exit levels."""

    def test_custom_2r_33pct_partial(self):
        """TRP-style: 33% exit at 2R."""
        pos = _make_position(
            partial_exit_levels=[(2.0, 0.33)],
            partial_exits_taken=0,
        )
        pm = PositionManager()
        # 2R = $4 profit per share → need high at 154
        bar = FakeBar(high=155.0, low=150.0, close=154.0)
        result = pm._check_partial_exit(pos, bar)

        assert result is not None
        assert result.intent is not None
        assert result.intent.qty == 33  # floor(100 * 0.33)
        assert result.reason == "PARTIAL_2.0R"

    def test_no_exit_below_threshold(self):
        """No partial exit when profit is below the R threshold."""
        pos = _make_position(
            partial_exit_levels=[(2.0, 0.33)],
            partial_exits_taken=0,
        )
        pm = PositionManager()
        # 1R = only $152 — not enough for 2R
        bar = FakeBar(high=152.0, low=150.0, close=151.5)
        result = pm._check_partial_exit(pos, bar)

        assert result is None

    def test_multiple_levels_fires_in_sequence(self):
        """TRP-style: first fires at 2R, second at 4R."""
        levels = [(2.0, 0.33), (4.0, 0.33)]
        pos = _make_position(
            partial_exit_levels=levels,
            partial_exits_taken=0,
        )
        pm = PositionManager()

        # First partial at 2R
        bar_2r = FakeBar(high=155.0)
        result = pm._check_partial_exit(pos, bar_2r)
        assert result is not None
        assert result.reason == "PARTIAL_2.0R"

        # Simulate the fill was processed: partial_exits_taken incremented
        pos.partial_exits_taken = 1
        pos.qty = 67  # 100 - 33

        # Not at 4R yet
        bar_3r = FakeBar(high=156.0)
        result = pm._check_partial_exit(pos, bar_3r)
        assert result is None

        # At 4R ($8 profit per share → high=158)
        bar_4r = FakeBar(high=158.5)
        result = pm._check_partial_exit(pos, bar_4r)
        assert result is not None
        assert result.reason == "PARTIAL_4.0R"
        assert result.intent.qty == 22  # floor(67 * 0.33)

    def test_all_partials_exhausted(self):
        """No exit when all configured partial levels are taken."""
        pos = _make_position(
            partial_exit_levels=[(2.0, 0.5)],
            partial_exits_taken=1,
        )
        pm = PositionManager()
        bar = FakeBar(high=160.0)
        result = pm._check_partial_exit(pos, bar)
        assert result is None

    def test_fallback_1r_50pct_when_no_levels(self):
        """Falls back to 1R/50% when no partial_exit_levels configured."""
        pos = _make_position(
            partial_exit_levels=[],
            partial_exits_taken=0,
        )
        pm = PositionManager()
        # 1R = $2 per share → high=152+
        bar = FakeBar(high=152.5)
        result = pm._check_partial_exit(pos, bar)

        assert result is not None
        assert result.intent.qty == 50  # floor(100 * 0.5)
        assert result.reason == "PARTIAL_1.0R"

    def test_short_partial_exit(self):
        """Partial exit works for short positions."""
        pos = _make_position(
            side=Side.SHORT,
            avg_price=150.0,
            stop_price=152.0,
            partial_exit_levels=[(1.5, 0.5)],
            partial_exits_taken=0,
        )
        pm = PositionManager()
        # 1.5R for short = $3 drop → low=147
        bar = FakeBar(high=150.0, low=146.5, close=147.0)
        result = pm._check_partial_exit(pos, bar)

        assert result is not None
        assert result.intent.side == OrderSide.BUY  # Cover to reduce short
        assert result.reason == "PARTIAL_1.5R"


# ---- Trail Profit Gate Tests ----


@pytest.mark.unit
class TestTrailMinProfitR:
    """Test trail_min_profit_r gate on trailing stop activation."""

    def test_trailing_blocked_below_profit_threshold(self):
        """Trailing stop should NOT update when profit < trail_min_profit_r."""
        pos = _make_position(
            trailing_stop_enabled=True,
            trailing_stop_pct=0.02,
            trail_min_profit_r=1.0,
            stop_price=148.0,
        )
        pm = PositionManager()
        # Only 0.5R profit (not 1R) → trailing should not activate
        bar = FakeBar(high=151.0, low=150.0, time=datetime(2024, 6, 1, 14, 31, tzinfo=timezone.utc))
        original_stop = pos.stop_price

        pm._update_trailing_stop(pos, bar)

        # Stop should remain unchanged
        assert pos.stop_price == original_stop

    def test_trailing_activates_at_profit_threshold(self):
        """Trailing stop SHOULD update when profit >= trail_min_profit_r."""
        pos = _make_position(
            trailing_stop_enabled=True,
            trailing_stop_pct=0.02,
            trail_min_profit_r=1.0,
            stop_price=148.0,
        )
        pm = PositionManager()
        # Exactly 1R profit ($2 per share) → high=152
        bar = FakeBar(high=152.0, low=151.0, time=datetime(2024, 6, 1, 14, 31, tzinfo=timezone.utc))

        pm._update_trailing_stop(pos, bar)

        # Trailing should have activated: new stop = 152 * (1 - 0.02) = 148.96
        assert pos.trailing_high_water == 152.0
        assert pos.stop_price == pytest.approx(148.96)

    def test_no_gate_when_trail_min_profit_r_is_none(self):
        """When trail_min_profit_r is None, trailing starts immediately (legacy behavior)."""
        pos = _make_position(
            trailing_stop_enabled=True,
            trailing_stop_pct=0.02,
            trail_min_profit_r=None,
            stop_price=148.0,
        )
        pm = PositionManager()
        # Small profit, but no gate
        bar = FakeBar(high=150.5, low=150.0, time=datetime(2024, 6, 1, 14, 31, tzinfo=timezone.utc))

        pm._update_trailing_stop(pos, bar)

        # Should have set high water and potentially updated stop
        assert pos.trailing_high_water == 150.5

    def test_short_trail_profit_gate(self):
        """Trail profit gate works for short positions."""
        pos = _make_position(
            side=Side.SHORT,
            avg_price=150.0,
            stop_price=152.0,
            trailing_stop_enabled=True,
            trailing_stop_pct=0.02,
            trail_min_profit_r=1.0,
        )
        pm = PositionManager()

        # Only 0.5R drop — gate not met
        bar_low = FakeBar(high=150.0, low=149.0, time=datetime(2024, 6, 1, 14, 31, tzinfo=timezone.utc))
        pm._update_trailing_stop(pos, bar_low)
        assert pos.stop_price == 152.0  # Unchanged

        # 1R drop ($2 per share) — gate met
        bar_1r = FakeBar(high=149.0, low=147.5, time=datetime(2024, 6, 1, 14, 32, tzinfo=timezone.utc))
        pm._update_trailing_stop(pos, bar_1r)
        # For short: trailing_high_water = min(150, 147.5) = 147.5
        # new_stop = 147.5 * 1.02 = 150.45
        assert pos.trailing_high_water == 147.5
        assert pos.stop_price == pytest.approx(150.45)

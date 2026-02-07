"""Unit tests for backtest parity improvements.

Tests volume-aware partial fills, volume-impact slippage, and flow strategy gating.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.backtest.mock_executor import BacktestOrderExecutor
from src.core.domain import Bar, OrderIntent, OrderSide, OrderType, SymbolState
from src.engine.execution import ExecutionEngine


def _make_bar(
    symbol: str = "AAPL",
    time_offset_min: int = 0,
    open: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: float = 10000.0,
) -> Bar:
    """Helper to create a Bar fixture."""
    t0 = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
    return Bar(
        symbol=symbol,
        time=t0 + timedelta(minutes=time_offset_min),
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_intent(
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    qty: float = 100.0,
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
) -> OrderIntent:
    """Helper to create an OrderIntent fixture."""
    t0 = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
    return OrderIntent(
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        limit_price=limit_price,
        time_in_force="day",
        correlation_id="test_cid",
        strategy="test_strategy",
        stop_loss=None,
        take_profit=None,
        meta={"created_at": t0.isoformat()},
    )


@pytest.mark.unit
class TestPartialFillModes:
    """Tests for _calculate_fill_qty with different partial fill modes."""

    def test_none_mode_fills_full_quantity(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)
        executor.set_backtest_config({"partial_fill_mode": "none"})

        fill_qty = executor._calculate_fill_qty(order_qty=1000, bar_volume=500)

        assert fill_qty == 1000.0

    def test_fixed_mode_applies_percentage(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)
        executor.set_backtest_config(
            {
                "partial_fill_mode": "fixed",
                "partial_fill_pct": 0.5,
            }
        )

        fill_qty = executor._calculate_fill_qty(order_qty=1000, bar_volume=10000)

        assert fill_qty == 500.0

    def test_volume_aware_caps_at_volume_fraction(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)
        executor.set_backtest_config(
            {
                "partial_fill_mode": "volume_aware",
                "partial_fill_rate": 0.1,
            }
        )

        # Order 2000 shares, bar volume 10000, can capture 10% = 1000 shares
        fill_qty = executor._calculate_fill_qty(order_qty=2000, bar_volume=10000)

        assert fill_qty == 1000.0

    def test_volume_aware_fills_full_if_order_smaller(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)
        executor.set_backtest_config(
            {
                "partial_fill_mode": "volume_aware",
                "partial_fill_rate": 0.1,
            }
        )

        # Order 500 shares, bar volume 10000, can capture 10% = 1000 shares
        # Order is smaller, so fill full order
        fill_qty = executor._calculate_fill_qty(order_qty=500, bar_volume=10000)

        assert fill_qty == 500.0


@pytest.mark.unit
class TestVolumeImpactSlippage:
    """Tests for _apply_slippage with volume_impact mode."""

    def test_fixed_mode_uses_constant_bps(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)
        executor.set_risk_config({"slippage_bps": 10.0})
        executor.set_backtest_config({"slippage_mode": "fixed"})

        # 10 bps = 0.1% slippage
        # Buy at 100 -> 100 * 1.001 = 100.10
        result = executor._apply_slippage("buy", 100.0, order_qty=1000, bar_volume=5000)

        assert result == pytest.approx(100.10, rel=1e-6)

    def test_volume_impact_increases_slippage_for_large_orders(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)
        executor.set_risk_config({"slippage_bps": 10.0})
        executor.set_backtest_config(
            {
                "slippage_mode": "volume_impact",
                "slippage_impact_mult": 5.0,
            }
        )

        # Order 2000 shares, bar volume 10000 -> ratio = 0.2
        # effective_bps = 10 * (1 + 0.2 * 5) = 10 * 2 = 20 bps
        # Buy at 100 -> 100 * 1.002 = 100.20
        result = executor._apply_slippage(
            "buy", 100.0, order_qty=2000, bar_volume=10000
        )

        assert result == pytest.approx(100.20, rel=1e-6)

    def test_volume_impact_fallback_to_fixed_on_zero_volume(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)
        executor.set_risk_config({"slippage_bps": 10.0})
        executor.set_backtest_config(
            {
                "slippage_mode": "volume_impact",
                "slippage_impact_mult": 5.0,
            }
        )

        # Zero volume -> falls back to fixed slippage
        result = executor._apply_slippage("buy", 100.0, order_qty=1000, bar_volume=0)

        assert result == pytest.approx(100.10, rel=1e-6)


@pytest.mark.unit
class TestAtrBasedSpread:
    """Tests for _apply_spread with atr_based mode."""

    def test_fixed_spread_ignores_atr(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)
        executor.set_risk_config({"spread_bps": 10.0})
        executor.set_backtest_config({"spread_mode": "fixed"})

        # Fixed mode: 10 bps = 0.1% half-spread
        # Buy at 100 -> 100 * (1 + 0.0005) = 100.05
        result = executor._apply_spread("buy", 100.0, atr_pct=0.03)

        assert result == pytest.approx(100.05, rel=1e-6)

    def test_atr_based_widens_for_high_volatility(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)
        executor.set_risk_config({"spread_bps": 10.0})
        executor.set_backtest_config({"spread_mode": "atr_based"})

        # atr_pct 2% vs avg 1% -> volatility_mult = 2.0
        # effective_bps = 10 * 2 = 20 bps, half = 10 bps = 0.1%
        # Buy at 100 -> 100 * (1 + 0.001) = 100.10
        result = executor._apply_spread("buy", 100.0, atr_pct=0.02)

        assert result == pytest.approx(100.10, rel=1e-6)

    def test_atr_based_clamps_multiplier(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)
        executor.set_risk_config({"spread_bps": 10.0})
        executor.set_backtest_config({"spread_mode": "atr_based"})

        # atr_pct 5% vs avg 1% -> volatility_mult = 5.0, clamped to 3.0
        # effective_bps = 10 * 3 = 30 bps, half = 15 bps = 0.15%
        # Buy at 100 -> 100 * (1 + 0.0015) = 100.15
        result = executor._apply_spread("buy", 100.0, atr_pct=0.05)

        assert result == pytest.approx(100.15, rel=1e-6)


@pytest.mark.unit
class TestBacktestConfigParsing:
    """Tests for set_backtest_config parsing."""

    def test_defaults_when_config_is_none(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)
        executor.set_backtest_config(None)

        assert executor._partial_fill_mode == "none"
        assert executor._slippage_mode == "fixed"
        assert executor._partial_fill_rate == 0.1

    def test_partial_fill_pct_clamped_to_range(self) -> None:
        logger = MagicMock()
        executor = BacktestOrderExecutor(logger, initial_cash=100000)

        executor.set_backtest_config({"partial_fill_pct": 2.0})
        assert executor._partial_fill_pct == 1.0

        executor.set_backtest_config({"partial_fill_pct": 0.05})
        assert executor._partial_fill_pct == 0.1


@pytest.mark.unit
def test_order_fill_uses_volume_aware_logic() -> None:
    """Integration test: full order fill path uses volume-aware logic."""
    logger = MagicMock()
    executor = BacktestOrderExecutor(logger, initial_cash=100000)
    executor.set_risk_config({"slippage_bps": 10.0})
    executor.set_backtest_config(
        {
            "partial_fill_mode": "volume_aware",
            "partial_fill_rate": 0.1,
            "slippage_mode": "volume_impact",
            "slippage_impact_mult": 5.0,
        }
    )

    engine = ExecutionEngine({"risk": {}}, logger, alpaca_client=None)
    engine.order_executor = executor  # type: ignore[assignment]
    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=100),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["test_strategy"],
        meta={},
    )

    # Submit order for 2000 shares
    intent = _make_intent(qty=2000)
    executor.submit(intent)

    # Fill on next bar with volume 10000
    # Volume-aware: can capture 10% = 1000 shares
    bar = _make_bar(time_offset_min=1, volume=10000)
    executor.fill_pending_for_bar(engine, "AAPL", bar)

    assert len(executor.fills) == 1
    fill = executor.fills[0]
    assert fill["qty"] == 1000.0  # Volume-limited fill

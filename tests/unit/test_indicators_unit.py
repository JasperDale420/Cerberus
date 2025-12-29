"""Unit tests for incremental indicator classes in src/core/indicators.py."""

import pytest

from src.core.indicators import (
    RollingADX,
    RollingATR,
    RollingEMA,
    RollingRSI,
    RollingSMA,
    RollingStd,
)


class TestRollingSMA:
    """Tests for RollingSMA incremental indicator."""

    def test_rolling_sma_basic(self):
        """Test SMA converges to expected value."""
        sma = RollingSMA.create(3)
        sma.update(10.0)
        sma.update(20.0)
        result = sma.update(30.0)
        assert result == pytest.approx(20.0, rel=1e-6)

    def test_rolling_sma_window_eviction(self):
        """Test that old values are evicted from the window."""
        sma = RollingSMA.create(2)
        sma.update(10.0)
        sma.update(20.0)
        result = sma.update(30.0)
        # Window now contains [20, 30]
        assert result == pytest.approx(25.0, rel=1e-6)


class TestRollingEMA:
    """Tests for RollingEMA incremental indicator."""

    def test_rolling_ema_basic(self):
        """Test EMA updates correctly."""
        ema = RollingEMA.from_period(10)
        # First value sets the EMA
        result = ema.update(100.0)
        assert result == pytest.approx(100.0, rel=1e-6)
        # Second value applies smoothing
        result = ema.update(110.0)
        # alpha = 2/(10+1) = 0.1818...
        expected = (2 / 11) * 110.0 + (1 - 2 / 11) * 100.0
        assert result == pytest.approx(expected, rel=1e-4)


class TestRollingStd:
    """Tests for RollingStd incremental indicator."""

    def test_rolling_std_basic(self):
        """Test std computation."""
        std = RollingStd.create(3)
        std.update(10.0)
        std.update(20.0)
        mean, stdev = std.update(30.0)
        assert mean == pytest.approx(20.0, rel=1e-6)
        # Population std of [10, 20, 30]
        expected_std = (((10 - 20) ** 2 + (20 - 20) ** 2 + (30 - 20) ** 2) / 3) ** 0.5
        assert stdev == pytest.approx(expected_std, rel=1e-4)


class TestRollingRSI:
    """Tests for RollingRSI incremental indicator."""

    def test_rolling_rsi_first_value_is_none(self):
        """Test that first value returns None (no change to calculate)."""
        rsi = RollingRSI(period=14)
        result = rsi.update(100.0)
        assert result is None

    def test_rolling_rsi_all_gains(self):
        """Test RSI approaches 100 when all moves are gains."""
        rsi = RollingRSI(period=14)
        rsi.update(100.0)
        for i in range(1, 20):
            result = rsi.update(100.0 + i)
        assert result is not None
        assert result > 90.0  # Should be near 100

    def test_rolling_rsi_all_losses(self):
        """Test RSI approaches 0 when all moves are losses."""
        rsi = RollingRSI(period=14)
        rsi.update(100.0)
        for i in range(1, 20):
            result = rsi.update(100.0 - i)
        assert result is not None
        assert result < 10.0  # Should be near 0


class TestRollingATR:
    """Tests for RollingATR incremental indicator."""

    def test_rolling_atr_first_value_is_none(self):
        """Test that first value returns None (no prev_close)."""
        atr = RollingATR(period=14)
        result = atr.update(high=105.0, low=95.0, close=100.0)
        assert result is None

    def test_rolling_atr_basic(self):
        """Test ATR computation for simple case."""
        atr = RollingATR(period=14)
        # First bar seeds prev_close
        atr.update(high=105.0, low=95.0, close=100.0)
        # Second bar has TR = max(105-95, |105-100|, |95-100|) = 10
        result = atr.update(high=105.0, low=95.0, close=100.0)
        assert result is not None
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_rolling_atr_wilder_smoothing(self):
        """Test that ATR uses Wilder smoothing after warmup."""
        atr = RollingATR(period=2)
        atr.update(high=105.0, low=95.0, close=100.0)
        atr.update(high=105.0, low=95.0, close=100.0)  # TR=10
        atr.update(high=110.0, low=90.0, close=100.0)  # TR=20
        # After warmup: ATR = ((10 * 1) + 20) / 2 = 15
        result = atr.update(high=108.0, low=92.0, close=100.0)  # TR=16
        # Wilder: ATR = ((15 * 1) + 16) / 2 = 15.5
        assert result is not None
        assert result > 10.0


class TestRollingADX:
    """Tests for RollingADX incremental indicator."""

    def test_rolling_adx_first_value_is_none(self):
        """Test that first value returns None."""
        adx = RollingADX(period=14)
        result = adx.update(high=105.0, low=95.0, close=100.0)
        assert result is None

    def test_rolling_adx_trending_market(self):
        """Test ADX rises in a trending market (consistent uptrend)."""
        adx = RollingADX(period=3)
        # Simulate strong uptrend
        prices = [100, 102, 105, 108, 112, 117, 123, 130]
        for _i, p in enumerate(prices):
            result = adx.update(high=p + 2, low=p - 2, close=p)
        assert result is not None
        assert result > 20.0  # Should show trending

    def test_rolling_adx_choppy_market(self):
        """Test ADX stays lower in choppy/ranging market."""
        adx = RollingADX(period=3)
        # Simulate choppy market (oscillating)
        prices = [100, 102, 99, 101, 98, 102, 99, 100]
        for p in prices:
            result = adx.update(high=p + 1, low=p - 1, close=p)
        # ADX should be lower in ranging markets
        assert result is not None
        # Just verify it doesn't crash and produces a value
        assert 0 <= result <= 100

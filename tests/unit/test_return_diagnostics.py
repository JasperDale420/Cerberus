"""Unit tests for return diagnostics — autocorrelation and turnover analysis."""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics.return_diagnostics import (
    compute_return_autocorrelation,
    compute_turnover_analysis,
)


@pytest.mark.unit
class TestReturnAutocorrelation:
    def test_random_data_no_significant_lags(self):
        """IID random returns should show no significant autocorrelation."""
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0005, 0.01, size=1000)

        result = compute_return_autocorrelation(returns, max_lag=5)

        assert "lag_1" in result
        assert "lag_5" in result
        assert "any_significant" in result
        # With 500 observations, threshold is 2/sqrt(500) ~ 0.089
        # Random data should generally not be significant
        assert result["any_significant"] is False
        for lag in range(1, 6):
            entry = result[f"lag_{lag}"]
            assert "correlation" in entry
            assert "significant" in entry
            assert isinstance(entry["correlation"], float)
            assert isinstance(entry["significant"], bool)

    def test_strongly_autocorrelated_data(self):
        """AR(1) process with high coefficient should show significant lag-1."""
        rng = np.random.default_rng(123)
        n = 500
        returns = np.zeros(n)
        returns[0] = rng.normal(0, 0.01)
        for i in range(1, n):
            returns[i] = 0.8 * returns[i - 1] + rng.normal(0, 0.01)

        result = compute_return_autocorrelation(returns, max_lag=5)

        assert result["lag_1"]["significant"] is True
        assert result["lag_1"]["correlation"] > 0.5
        assert result["any_significant"] is True

    def test_default_max_lag(self):
        """Default max_lag=5 produces exactly 5 lag entries."""
        rng = np.random.default_rng(7)
        returns = rng.normal(0, 0.01, size=100)

        result = compute_return_autocorrelation(returns)

        assert len([k for k in result if k.startswith("lag_")]) == 5

    def test_custom_max_lag(self):
        """Custom max_lag produces correct number of entries."""
        rng = np.random.default_rng(7)
        returns = rng.normal(0, 0.01, size=100)

        result = compute_return_autocorrelation(returns, max_lag=10)

        assert len([k for k in result if k.startswith("lag_")]) == 10


@pytest.mark.unit
class TestTurnoverAnalysis:
    def _make_trade(self, qty=100, entry_price=50.0, exit_price=52.0, commission=1.0, slippage=0.5):
        return {
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "commission": commission,
            "slippage": slippage,
        }

    def test_turnover_calculation(self):
        """Verify gross_volume and annualized_turnover math."""
        trades = [self._make_trade(qty=100, entry_price=50.0, exit_price=52.0)]
        initial_capital = 100_000.0
        n_trading_days = 252

        result = compute_turnover_analysis(trades, initial_capital, n_trading_days)

        # gross_volume = 100*50 + 100*52 = 10_200
        assert result["gross_volume"] == pytest.approx(10_200.0)
        # annualized_turnover = (10_200 / 100_000) * (252/252) = 0.102
        assert result["annualized_turnover"] == pytest.approx(0.102)

    def test_cost_drag_percentage(self):
        """Cost drag should be total_cost / gross_pnl * 100."""
        trades = [
            self._make_trade(qty=100, entry_price=50.0, exit_price=52.0, commission=2.0, slippage=1.0),
            self._make_trade(qty=200, entry_price=30.0, exit_price=31.0, commission=3.0, slippage=2.0),
        ]
        initial_capital = 100_000.0
        n_trading_days = 20

        result = compute_turnover_analysis(trades, initial_capital, n_trading_days)

        # Trade 1 gross PnL: 100*(52-50) = 200
        # Trade 2 gross PnL: 200*(31-30) = 200
        # gross_pnl = 400
        # total_cost = (2+1) + (3+2) = 8
        # cost_drag_pct = 8 / 400 * 100 = 2.0
        assert result["total_cost"] == pytest.approx(8.0)
        assert result["cost_drag_pct"] == pytest.approx(2.0)
        assert result["avg_cost_per_trade"] == pytest.approx(4.0)

    def test_zero_gross_pnl(self):
        """Zero gross PnL should yield 0.0 cost_drag_pct (not division error)."""
        trades = [self._make_trade(qty=100, entry_price=50.0, exit_price=50.0, commission=1.0, slippage=0.5)]
        initial_capital = 100_000.0
        n_trading_days = 10

        result = compute_turnover_analysis(trades, initial_capital, n_trading_days)

        assert result["cost_drag_pct"] == 0.0

    def test_multiple_trades_volume(self):
        """Multiple trades accumulate volume correctly."""
        trades = [
            self._make_trade(qty=100, entry_price=50.0, exit_price=55.0),
            self._make_trade(qty=50, entry_price=40.0, exit_price=42.0),
        ]
        initial_capital = 50_000.0
        n_trading_days = 126  # half year

        result = compute_turnover_analysis(trades, initial_capital, n_trading_days)

        # Trade 1: 100*50 + 100*55 = 10_500
        # Trade 2: 50*40 + 50*42 = 4_100
        # gross_volume = 14_600
        assert result["gross_volume"] == pytest.approx(14_600.0)
        # annualized_turnover = (14_600/50_000) * (252/126) = 0.292 * 2 = 0.584
        assert result["annualized_turnover"] == pytest.approx(0.584)

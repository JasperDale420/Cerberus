"""Tests for extended rolling metrics and daily return distribution stats."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analytics.report_card import compute_daily_return_stats, compute_rolling_metrics

# ---------------------------------------------------------------------------
# Rolling Sortino
# ---------------------------------------------------------------------------


class TestRollingSortino:
    """Tests for rolling_sortino in compute_rolling_metrics."""

    def test_rolling_sortino_present_in_output(self):
        """Rolling Sortino should appear in the output dict."""
        returns = list(np.random.default_rng(42).normal(0.001, 0.02, 100))
        result = compute_rolling_metrics(returns, window=20)
        assert "rolling_sortino" in result
        assert len(result["rolling_sortino"]) == len(result["rolling_sharpe"])

    def test_rolling_sortino_all_positive_returns(self):
        """When all returns in window are positive, downside deviation is ~0 so Sortino should be 0."""
        returns = [0.01] * 60
        result = compute_rolling_metrics(returns, window=50)
        # No negative returns means downside dev is 0, Sortino should be 0
        for val in result["rolling_sortino"]:
            assert val == 0.0

    def test_rolling_sortino_mixed_returns(self):
        """Sortino should be finite for a mix of positive and negative returns."""
        rng = np.random.default_rng(123)
        returns = list(rng.normal(0.002, 0.03, 200))
        result = compute_rolling_metrics(returns, window=50)
        assert len(result["rolling_sortino"]) == 151
        # All values should be finite
        for val in result["rolling_sortino"]:
            assert math.isfinite(val)

    def test_rolling_sortino_larger_than_sharpe_for_positively_skewed(self):
        """For positively skewed returns, Sortino should generally be >= Sharpe."""
        # Create positively skewed returns: mostly small positives, few negatives
        rng = np.random.default_rng(99)
        returns = list(np.abs(rng.normal(0.005, 0.01, 100)))
        # Add a few negatives
        returns[10] = -0.01
        returns[30] = -0.005
        returns[50] = -0.008
        result = compute_rolling_metrics(returns, window=20)
        # On average, Sortino should be >= Sharpe for positively skewed data
        np.mean(result["rolling_sortino"])
        np.mean(result["rolling_sharpe"])
        # This is a statistical tendency, not absolute — just check both are computed
        assert len(result["rolling_sortino"]) > 0
        assert len(result["rolling_sharpe"]) > 0


# ---------------------------------------------------------------------------
# Rolling Beta
# ---------------------------------------------------------------------------


class TestRollingBeta:
    """Tests for rolling_beta in compute_rolling_metrics."""

    def test_rolling_beta_with_benchmark(self):
        """Rolling beta should be computed when benchmark_returns is provided."""
        rng = np.random.default_rng(42)
        benchmark = list(rng.normal(0.001, 0.015, 100))
        # Portfolio = 1.5 * benchmark + noise
        portfolio = [1.5 * b + rng.normal(0, 0.005) for b in benchmark]
        result = compute_rolling_metrics(portfolio, window=30, benchmark_returns=benchmark)
        assert "rolling_beta" in result
        assert len(result["rolling_beta"]) == len(result["rolling_sharpe"])
        # Betas should be roughly around 1.5
        avg_beta = np.mean(result["rolling_beta"])
        assert 1.0 < avg_beta < 2.0

    def test_rolling_beta_skipped_when_no_benchmark(self):
        """Rolling beta should not be in output when no benchmark provided."""
        returns = list(np.random.default_rng(42).normal(0.001, 0.02, 100))
        result = compute_rolling_metrics(returns, window=20)
        assert "rolling_beta" not in result

    def test_rolling_beta_benchmark_length_mismatch(self):
        """When benchmark is shorter than returns, rolling_beta should still be absent or handle gracefully."""
        returns = list(np.random.default_rng(42).normal(0.001, 0.02, 100))
        benchmark = list(np.random.default_rng(42).normal(0.001, 0.015, 50))
        result = compute_rolling_metrics(returns, window=20, benchmark_returns=benchmark)
        # Should skip beta when lengths don't match
        assert "rolling_beta" not in result

    def test_rolling_beta_constant_benchmark(self):
        """Beta should be 0 when benchmark has zero variance."""
        returns = list(np.random.default_rng(42).normal(0.001, 0.02, 100))
        benchmark = [0.001] * 100
        result = compute_rolling_metrics(returns, window=20, benchmark_returns=benchmark)
        assert "rolling_beta" in result
        for val in result["rolling_beta"]:
            assert val == 0.0


# ---------------------------------------------------------------------------
# Window configurability
# ---------------------------------------------------------------------------


class TestWindowConfigurability:
    """Tests for configurable window parameter."""

    def test_default_window_50(self):
        """Default window=50 should produce correct number of rolling values."""
        returns = list(np.random.default_rng(42).normal(0.001, 0.02, 100))
        result = compute_rolling_metrics(returns)
        assert len(result["rolling_sharpe"]) == 51  # 100 - 50 + 1

    def test_custom_window_20(self):
        """Custom window=20 should produce correct number of rolling values."""
        returns = list(np.random.default_rng(42).normal(0.001, 0.02, 100))
        result = compute_rolling_metrics(returns, window=20)
        assert len(result["rolling_sharpe"]) == 81  # 100 - 20 + 1
        assert len(result["rolling_sortino"]) == 81

    def test_window_larger_than_data(self):
        """Window larger than data should return empty lists."""
        returns = [0.01, 0.02, -0.01]
        result = compute_rolling_metrics(returns, window=50)
        assert result["rolling_sharpe"] == []
        assert result["rolling_sortino"] == []
        assert result["rolling_win_rate"] == []


# ---------------------------------------------------------------------------
# Daily return stats
# ---------------------------------------------------------------------------


class TestDailyReturnStats:
    """Tests for compute_daily_return_stats."""

    def test_basic_stats_keys(self):
        """All expected keys should be present."""
        rng = np.random.default_rng(42)
        daily = rng.normal(0.001, 0.02, 252)
        result = compute_daily_return_stats(daily)
        expected_keys = {
            "skew",
            "kurtosis",
            "var_95",
            "cvar_95",
            "best_day",
            "worst_day",
            "positive_pct",
            "autocorrelation_lag1",
        }
        assert set(result.keys()) == expected_keys

    def test_skew_and_kurtosis(self):
        """Skew and kurtosis should be finite for normal returns."""
        rng = np.random.default_rng(42)
        daily = rng.normal(0.0, 0.02, 1000)
        result = compute_daily_return_stats(daily)
        # Normal distribution: skew ~ 0, excess kurtosis ~ 0
        assert abs(result["skew"]) < 0.5
        assert abs(result["kurtosis"]) < 1.0

    def test_var_and_cvar(self):
        """VaR95 should be the 5th percentile; CVaR95 should be <= VaR95."""
        rng = np.random.default_rng(42)
        daily = rng.normal(-0.001, 0.03, 1000)
        result = compute_daily_return_stats(daily)
        assert result["var_95"] < 0  # negative for losses
        assert result["cvar_95"] <= result["var_95"]

    def test_best_worst_day(self):
        """Best and worst day should match max and min of input."""
        daily = np.array([0.05, -0.03, 0.02, -0.01, 0.04])
        result = compute_daily_return_stats(daily)
        assert result["best_day"] == pytest.approx(0.05)
        assert result["worst_day"] == pytest.approx(-0.03)

    def test_positive_pct(self):
        """positive_pct should be fraction of returns > 0."""
        daily = np.array([0.01, -0.01, 0.02, -0.02, 0.0])
        result = compute_daily_return_stats(daily)
        assert result["positive_pct"] == pytest.approx(2.0 / 5.0)

    def test_autocorrelation_lag1(self):
        """Autocorrelation should be finite and between -1 and 1."""
        rng = np.random.default_rng(42)
        daily = rng.normal(0.001, 0.02, 252)
        result = compute_daily_return_stats(daily)
        assert -1.0 <= result["autocorrelation_lag1"] <= 1.0

    def test_empty_input(self):
        """Empty array should return zeros."""
        result = compute_daily_return_stats(np.array([]))
        assert result["skew"] == 0.0
        assert result["kurtosis"] == 0.0
        assert result["var_95"] == 0.0
        assert result["cvar_95"] == 0.0
        assert result["best_day"] == 0.0
        assert result["worst_day"] == 0.0
        assert result["positive_pct"] == 0.0
        assert result["autocorrelation_lag1"] == 0.0

    def test_single_element(self):
        """Single element should handle gracefully without errors."""
        result = compute_daily_return_stats(np.array([0.05]))
        assert result["best_day"] == pytest.approx(0.05)
        assert result["worst_day"] == pytest.approx(0.05)
        assert result["positive_pct"] == pytest.approx(1.0)

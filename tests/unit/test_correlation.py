"""Unit tests for strategy correlation and diversification analysis."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from src.analytics.correlation import (
    StrategyCorrelation,
    build_strategy_daily_returns,
    compute_strategy_correlation,
)

pytestmark = pytest.mark.unit


class TestComputeStrategyCorrelation:
    """Tests for compute_strategy_correlation."""

    def test_two_uncorrelated_strategies(self):
        """Two strategies with uncorrelated returns should have low correlation and DR > 1."""
        rng = np.random.default_rng(42)
        n = 252
        returns = {
            "alpha": rng.standard_normal(n),
            "beta": rng.standard_normal(n),
        }
        result = compute_strategy_correlation(returns)

        assert isinstance(result, StrategyCorrelation)
        assert set(result.strategy_names) == {"alpha", "beta"}
        # Uncorrelated: avg pairwise should be near 0
        assert abs(result.avg_pairwise_correlation) < 0.15
        # Diversification ratio > 1 when uncorrelated
        assert result.diversification_ratio > 1.0
        assert result.n_trading_days == n
        # Matrix should be 2x2
        assert len(result.correlation_matrix) == 2
        assert len(result.correlation_matrix[0]) == 2

    def test_two_identical_strategies(self):
        """Two identical return streams should have correlation ~1.0 and DR ~1.0."""
        rng = np.random.default_rng(99)
        n = 100
        shared = rng.standard_normal(n)
        returns = {
            "strat_a": shared.copy(),
            "strat_b": shared.copy(),
        }
        result = compute_strategy_correlation(returns)

        assert result.avg_pairwise_correlation == pytest.approx(1.0, abs=1e-10)
        assert result.max_pairwise_correlation == pytest.approx(1.0, abs=1e-10)
        assert result.max_corr_pair == ("strat_a", "strat_b")
        assert result.diversification_ratio == pytest.approx(1.0, abs=0.01)

    def test_three_strategies_mixed(self):
        """Three strategies: two correlated, one independent."""
        rng = np.random.default_rng(7)
        n = 200
        base = rng.standard_normal(n)
        returns = {
            "corr1": base,
            "corr2": base + 0.1 * rng.standard_normal(n),  # highly correlated with corr1
            "independent": rng.standard_normal(n),
        }
        result = compute_strategy_correlation(returns)

        assert len(result.strategy_names) == 3
        # Max correlation should be between corr1 and corr2
        assert result.max_corr_pair == ("corr1", "corr2")
        assert result.max_pairwise_correlation > 0.9
        # 3x3 matrix
        assert len(result.correlation_matrix) == 3
        # DR should be > 1 because independent strategy helps
        assert result.diversification_ratio > 1.0

    def test_single_strategy(self):
        """Single strategy: degenerate case, correlation = 1.0, DR = 1.0."""
        returns = {"solo": np.array([0.01, -0.02, 0.03, 0.0, 0.015])}
        result = compute_strategy_correlation(returns)

        assert result.strategy_names == ["solo"]
        assert result.correlation_matrix == [[1.0]]
        assert result.avg_pairwise_correlation == 1.0
        assert result.max_pairwise_correlation == 1.0
        assert result.max_corr_pair == ("solo", "solo")
        assert result.diversification_ratio == 1.0
        assert result.n_trading_days == 5

    def test_strategy_with_zero_variance(self):
        """Strategy with constant returns (zero variance) should not crash."""
        returns = {
            "flat": np.zeros(50),
            "normal": np.random.default_rng(1).standard_normal(50),
        }
        result = compute_strategy_correlation(returns)

        # Should handle gracefully -- zero-variance produces NaN correlations
        # We clamp NaN to 0.0
        assert not math.isnan(result.avg_pairwise_correlation)
        assert not math.isnan(result.max_pairwise_correlation)
        assert result.n_trading_days == 50

    def test_diversification_ratio_uncorrelated(self):
        """DR for perfectly uncorrelated strategies should be sqrt(N) for equal-vol."""
        n = 1000
        rng = np.random.default_rng(123)
        # Two independent strategies with same volatility
        returns = {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
        }
        result = compute_strategy_correlation(returns)

        # For two uncorrelated equal-vol strategies, DR = 2 * sigma / (sigma * sqrt(2)) = sqrt(2) ~ 1.414
        assert result.diversification_ratio > 1.2
        assert result.diversification_ratio < 1.6

    def test_correlation_matrix_is_json_serializable(self):
        """Correlation matrix should be nested lists of Python floats, not numpy."""
        import json

        returns = {
            "x": np.array([1.0, 2.0, 3.0, 4.0]),
            "y": np.array([2.0, 4.0, 6.0, 8.0]),
        }
        result = compute_strategy_correlation(returns)
        # Should not raise
        json.dumps(result.correlation_matrix)
        assert isinstance(result.correlation_matrix[0][0], float)


class TestBuildStrategyDailyReturns:
    """Tests for build_strategy_daily_returns."""

    def test_basic_allocation(self):
        """PnL should be allocated to the exit date."""
        trades = [
            {
                "strategy": "alpha",
                "entry_time": datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
                "exit_time": datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc),
                "pnl": 100.0,
            },
            {
                "strategy": "alpha",
                "entry_time": datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc),
                "exit_time": datetime(2024, 1, 4, 15, 0, tzinfo=timezone.utc),
                "pnl": -50.0,
            },
            {
                "strategy": "beta",
                "entry_time": datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
                "exit_time": datetime(2024, 1, 3, 15, 0, tzinfo=timezone.utc),
                "pnl": 200.0,
            },
        ]
        result = build_strategy_daily_returns(trades, n_trading_days=5)

        assert "alpha" in result
        assert "beta" in result
        assert len(result["alpha"]) == 5
        assert len(result["beta"]) == 5
        # Non-trade days should be 0
        assert result["alpha"].sum() == pytest.approx(50.0)  # 100 + (-50)
        assert result["beta"].sum() == pytest.approx(200.0)

    def test_multiple_trades_same_day(self):
        """Multiple trades exiting on the same day should sum PnL."""
        trades = [
            {
                "strategy": "alpha",
                "entry_time": datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
                "exit_time": datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),
                "pnl": 50.0,
            },
            {
                "strategy": "alpha",
                "entry_time": datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc),
                "exit_time": datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc),
                "pnl": 30.0,
            },
        ]
        result = build_strategy_daily_returns(trades, n_trading_days=3)

        assert result["alpha"].sum() == pytest.approx(80.0)

    def test_empty_trades(self):
        """Empty trade list should return empty dict."""
        result = build_strategy_daily_returns([], n_trading_days=10)
        assert result == {}

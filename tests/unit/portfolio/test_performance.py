"""Tests for portfolio performance analytics."""

import math

import pytest

from portfolio.performance import (
    PortfolioPerformance,
)


@pytest.fixture
def perf() -> PortfolioPerformance:
    return PortfolioPerformance(sharpe_window=30, correlation_spike_threshold=0.7)


class TestStrategyAttribution:
    """Test attribution computation."""

    def test_attribution_sums_to_100_pct(self, perf: PortfolioPerformance):
        strategy_pnls = {
            "momentum": [10.0, 20.0, -5.0, 15.0],
            "mean_revert": [5.0, -3.0, 8.0, 2.0],
            "breakout": [-2.0, 7.0, 3.0, 1.0],
        }
        attrs = perf.compute_attribution(strategy_pnls)
        total_pct = sum(a.pnl_pct for a in attrs)
        assert abs(total_pct - 1.0) < 1e-9, f"Attribution fractions sum to {total_pct}, expected 1.0"

    def test_attribution_pnl_values(self, perf: PortfolioPerformance):
        strategy_pnls = {
            "alpha": [10.0, -5.0, 20.0],
            "beta": [5.0, 5.0, 5.0],
        }
        attrs = perf.compute_attribution(strategy_pnls)
        by_name = {a.strategy: a for a in attrs}
        assert by_name["alpha"].pnl == pytest.approx(25.0)
        assert by_name["beta"].pnl == pytest.approx(15.0)
        assert by_name["alpha"].pnl_pct == pytest.approx(25.0 / 40.0)
        assert by_name["beta"].pnl_pct == pytest.approx(15.0 / 40.0)

    def test_attribution_with_zero_total_pnl(self, perf: PortfolioPerformance):
        strategy_pnls = {
            "a": [10.0, -10.0],
            "b": [5.0, -5.0],
        }
        attrs = perf.compute_attribution(strategy_pnls)
        # When total PnL is zero, each strategy gets equal share
        for a in attrs:
            assert a.pnl == pytest.approx(0.0)

    def test_attribution_single_strategy(self, perf: PortfolioPerformance):
        strategy_pnls = {"solo": [10.0, 20.0, 30.0]}
        attrs = perf.compute_attribution(strategy_pnls)
        assert len(attrs) == 1
        assert attrs[0].pnl_pct == pytest.approx(1.0)

    def test_attribution_includes_max_drawdown(self, perf: PortfolioPerformance):
        strategy_pnls = {"dd_test": [10.0, -20.0, 5.0, -10.0, 30.0]}
        attrs = perf.compute_attribution(strategy_pnls)
        assert attrs[0].max_drawdown < 0.0, "Max drawdown should be negative"


class TestSharpe:
    """Test Sharpe ratio computation."""

    def test_sharpe_positive_returns(self, perf: PortfolioPerformance):
        # Constant positive returns → high Sharpe
        returns = [0.01] * 60
        sharpe = perf.rolling_sharpe(returns, window=30)
        # mean=0.01, std=0 → should return None (division by zero)
        assert sharpe is None

    def test_sharpe_known_value(self, perf: PortfolioPerformance):
        returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.002, 0.01, 0.008, -0.003] * 3
        sharpe = perf.rolling_sharpe(returns, window=30)
        assert sharpe is not None
        # Manual: mean=0.0048, std≈0.00946, annualized = 0.0048/0.00946 * sqrt(252) ≈ 8.06
        import numpy as np

        r = np.array(returns[-30:])
        expected = float(r.mean() / r.std(ddof=1) * math.sqrt(252))
        assert sharpe == pytest.approx(expected, rel=1e-6)

    def test_sharpe_insufficient_data(self, perf: PortfolioPerformance):
        returns = [0.01, 0.02]
        sharpe = perf.rolling_sharpe(returns, window=30)
        assert sharpe is None

    def test_sharpe_default_window(self):
        pp = PortfolioPerformance(sharpe_window=5)
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        sharpe = pp.rolling_sharpe(returns)
        assert sharpe is not None


class TestSortino:
    """Test Sortino ratio computation."""

    def test_sortino_uses_only_downside(self, perf: PortfolioPerformance):
        # Returns with some negative — Sortino should only use negative for denominator
        returns = [0.05, -0.02, 0.03, -0.01, 0.04, -0.03, 0.02, -0.005] * 4
        sortino = perf.rolling_sortino(returns, window=30)
        assert sortino is not None
        # Sortino > Sharpe because downside std < total std when positive skew
        sharpe = perf.rolling_sharpe(returns, window=30)
        assert sharpe is not None
        assert sortino > sharpe

    def test_sortino_no_negative_returns(self, perf: PortfolioPerformance):
        returns = [0.01, 0.02, 0.03] * 10
        sortino = perf.rolling_sortino(returns, window=30)
        # No downside → None (can't compute downside std)
        assert sortino is None

    def test_sortino_insufficient_data(self, perf: PortfolioPerformance):
        returns = [0.01]
        sortino = perf.rolling_sortino(returns, window=30)
        assert sortino is None


class TestMaxDrawdown:
    """Test max drawdown computation."""

    def test_max_drawdown_known_sequence(self, perf: PortfolioPerformance):
        # Cumulative: 10, 30, 10, 25 → peak=30, trough=10, dd = (10-30)/30 = -0.667
        strategy_pnls = {"test": [10.0, 20.0, -20.0, 15.0]}
        attrs = perf.compute_attribution(strategy_pnls)
        assert attrs[0].max_drawdown == pytest.approx(-20.0 / 30.0, rel=1e-3)

    def test_max_drawdown_no_drawdown(self, perf: PortfolioPerformance):
        strategy_pnls = {"test": [5.0, 10.0, 15.0, 20.0]}
        attrs = perf.compute_attribution(strategy_pnls)
        assert attrs[0].max_drawdown == pytest.approx(0.0)

    def test_max_drawdown_all_losses(self, perf: PortfolioPerformance):
        strategy_pnls = {"test": [-5.0, -10.0, -3.0]}
        attrs = perf.compute_attribution(strategy_pnls)
        assert attrs[0].max_drawdown < 0.0


class TestCorrelation:
    """Test correlation matrix and alerts."""

    def test_correlation_matrix_structure(self, perf: PortfolioPerformance):
        strategy_returns = {
            "a": [0.01, -0.02, 0.03, -0.01, 0.02],
            "b": [0.02, -0.01, 0.01, -0.02, 0.03],
        }
        matrix = perf.compute_correlation_matrix(strategy_returns)
        assert ("a", "b") in matrix or ("b", "a") in matrix
        corr = matrix.get(("a", "b"), matrix.get(("b", "a")))
        assert -1.0 <= corr <= 1.0

    def test_correlation_perfect_positive(self, perf: PortfolioPerformance):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        strategy_returns = {"x": returns, "y": returns}
        matrix = perf.compute_correlation_matrix(strategy_returns)
        corr = matrix.get(("x", "y"), matrix.get(("y", "x")))
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_correlation_alerts_fire_above_threshold(self):
        pp = PortfolioPerformance(correlation_spike_threshold=0.5)
        # Highly correlated returns
        returns_a = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.04, -0.01, 0.02, -0.02]
        returns_b = [0.012, -0.018, 0.028, -0.012, 0.022, -0.028, 0.038, -0.008, 0.018, -0.022]
        strategy_returns = {"high_a": returns_a, "high_b": returns_b}
        alerts = pp.check_correlation_alerts(strategy_returns)
        assert len(alerts) > 0
        assert any(a.correlation > 0.5 for a in alerts)

    def test_correlation_no_alerts_below_threshold(self, perf: PortfolioPerformance):
        import numpy as np

        rng = np.random.default_rng(42)
        strategy_returns = {
            "uncorr_a": rng.normal(0, 0.01, 50).tolist(),
            "uncorr_b": rng.normal(0, 0.01, 50).tolist(),
        }
        alerts = perf.check_correlation_alerts(strategy_returns)
        # With random uncorrelated data, alerts should be empty or below threshold
        high_alerts = [a for a in alerts if a.correlation > 0.7]
        assert len(high_alerts) == 0

    def test_correlation_single_strategy_no_alerts(self, perf: PortfolioPerformance):
        strategy_returns = {"only": [0.01, 0.02, -0.01]}
        alerts = perf.check_correlation_alerts(strategy_returns)
        assert len(alerts) == 0

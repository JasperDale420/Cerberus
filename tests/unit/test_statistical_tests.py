"""Tests for Probabilistic Sharpe Ratio (PSR) and Minimum Backtest Length (MinBTL).

References:
    Bailey & Lopez de Prado (2014) — "The Deflated Sharpe Ratio"
"""

from __future__ import annotations

import pytest

from src.analytics.statistical_tests import minimum_backtest_length, probabilistic_sharpe_ratio


@pytest.mark.unit
class TestProbabilisticSharpeRatio:
    """PSR: probability that observed Sharpe exceeds a benchmark."""

    def test_high_sharpe_is_significant(self):
        """A high Sharpe with enough data should be significant (PSR > 0.95)."""
        result = probabilistic_sharpe_ratio(
            observed_sharpe=2.0,
            benchmark_sharpe=0.0,
            n_returns=500,
            skew=0.0,
            kurtosis=3.0,
        )
        assert result["significant"] is True
        assert result["psr"] > 0.95

    def test_low_sharpe_not_significant(self):
        """A barely-positive Sharpe should not be significant."""
        result = probabilistic_sharpe_ratio(
            observed_sharpe=0.1,
            benchmark_sharpe=0.0,
            n_returns=50,
            skew=0.0,
            kurtosis=3.0,
        )
        assert result["significant"] is False
        assert result["psr"] < 0.95

    def test_zero_benchmark(self):
        """With zero benchmark and moderate Sharpe, PSR should be between 0 and 1."""
        result = probabilistic_sharpe_ratio(
            observed_sharpe=1.0,
            benchmark_sharpe=0.0,
            n_returns=252,
            skew=-0.5,
            kurtosis=4.0,
        )
        assert 0.0 <= result["psr"] <= 1.0

    def test_negative_sharpe_below_benchmark(self):
        """Observed Sharpe below benchmark should yield PSR < 0.5."""
        result = probabilistic_sharpe_ratio(
            observed_sharpe=0.5,
            benchmark_sharpe=1.0,
            n_returns=252,
            skew=0.0,
            kurtosis=3.0,
        )
        assert result["psr"] < 0.5
        assert result["significant"] is False

    def test_very_small_n_returns(self):
        """With very few observations, PSR should not be significant."""
        result = probabilistic_sharpe_ratio(
            observed_sharpe=0.8,
            benchmark_sharpe=0.0,
            n_returns=5,
            skew=0.0,
            kurtosis=3.0,
        )
        assert result["significant"] is False

    def test_fat_tails_reduce_significance(self):
        """Higher kurtosis should reduce PSR (more uncertainty in Sharpe estimate)."""
        normal_tails = probabilistic_sharpe_ratio(
            observed_sharpe=1.5,
            benchmark_sharpe=0.0,
            n_returns=100,
            skew=0.0,
            kurtosis=3.0,
        )
        fat_tails = probabilistic_sharpe_ratio(
            observed_sharpe=1.5,
            benchmark_sharpe=0.0,
            n_returns=100,
            skew=0.0,
            kurtosis=9.0,
        )
        assert fat_tails["psr"] < normal_tails["psr"]


@pytest.mark.unit
class TestMinimumBacktestLength:
    """MinBTL: minimum observations needed for Sharpe to be significant."""

    def test_normal_distribution(self):
        """With Gaussian returns (skew=0, kurtosis=3), MinBTL should be finite and reasonable."""
        result = minimum_backtest_length(
            observed_sharpe=1.0,
            skew=0.0,
            kurtosis=3.0,
            frequency=252,
            confidence=0.95,
            n_observations=500,
        )
        assert result["min_observations"] > 0
        assert result["min_years"] > 0.0
        assert result["have_enough"] is True
        assert result["n_observations"] == 500

    def test_fat_tails_need_longer_backtest(self):
        """Higher kurtosis should require more observations."""
        normal = minimum_backtest_length(
            observed_sharpe=1.0,
            skew=0.0,
            kurtosis=3.0,
            frequency=252,
            confidence=0.95,
            n_observations=1000,
        )
        fat = minimum_backtest_length(
            observed_sharpe=1.0,
            skew=0.0,
            kurtosis=9.0,
            frequency=252,
            confidence=0.95,
            n_observations=1000,
        )
        assert fat["min_observations"] > normal["min_observations"]

    def test_not_enough_observations(self):
        """If n_observations < min_observations, have_enough should be False."""
        result = minimum_backtest_length(
            observed_sharpe=0.5,
            skew=-1.0,
            kurtosis=6.0,
            frequency=252,
            confidence=0.95,
            n_observations=10,
        )
        assert result["have_enough"] is False

    def test_higher_sharpe_needs_fewer_observations(self):
        """A higher Sharpe ratio should require fewer observations to confirm."""
        low_sr = minimum_backtest_length(
            observed_sharpe=0.5,
            skew=0.0,
            kurtosis=3.0,
            frequency=252,
            confidence=0.95,
            n_observations=1000,
        )
        high_sr = minimum_backtest_length(
            observed_sharpe=2.0,
            skew=0.0,
            kurtosis=3.0,
            frequency=252,
            confidence=0.95,
            n_observations=1000,
        )
        assert high_sr["min_observations"] < low_sr["min_observations"]

    def test_min_years_conversion(self):
        """min_years should equal min_observations / frequency."""
        result = minimum_backtest_length(
            observed_sharpe=1.0,
            skew=0.0,
            kurtosis=3.0,
            frequency=252,
            confidence=0.95,
            n_observations=500,
        )
        expected_years = result["min_observations"] / 252
        assert abs(result["min_years"] - expected_years) < 1e-10

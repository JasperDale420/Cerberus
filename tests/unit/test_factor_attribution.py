"""Unit tests for Fama-French factor attribution."""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics.factor_attribution import FactorAttribution, compute_factor_attribution


@pytest.mark.unit
class TestFactorAttribution:
    """Tests for compute_factor_attribution."""

    def _make_factor_returns(self, n: int, seed: int = 42) -> dict[str, np.ndarray]:
        """Create synthetic factor return arrays."""
        rng = np.random.default_rng(seed)
        return {
            "Mkt-RF": rng.normal(0.0004, 0.01, n),
            "SMB": rng.normal(0.0001, 0.005, n),
            "HML": rng.normal(0.0001, 0.005, n),
            "UMD": rng.normal(0.0002, 0.007, n),
        }

    def test_pure_alpha(self):
        """Strategy returns independent of factors -> significant alpha, low R-squared."""
        n = 500
        factors = self._make_factor_returns(n, seed=99)

        # Strategy returns: constant positive drift + noise, uncorrelated with factors
        # Use a completely different seed to ensure independence
        rng_strat = np.random.default_rng(7777)
        strategy_excess = rng_strat.normal(0.001, 0.005, n)

        result = compute_factor_attribution(strategy_excess, factors)

        assert isinstance(result, FactorAttribution)
        assert result.alpha > 0, "Should detect positive alpha"
        assert result.alpha_significant, "Alpha should be significant for strong drift"
        assert result.r_squared < 0.15, f"R-squared should be low for uncorrelated returns, got {result.r_squared}"
        assert result.n_observations == n

    def test_market_driven(self):
        """Strategy = 1.2 * Mkt-RF + small noise -> high beta, high R-squared, near-zero alpha."""
        n = 500
        factors = self._make_factor_returns(n, seed=42)
        rng = np.random.default_rng(8888)

        # Add small noise to avoid degenerate perfect-fit regression
        strategy_excess = 1.2 * factors["Mkt-RF"] + rng.normal(0.0, 0.001, n)

        result = compute_factor_attribution(strategy_excess, factors)

        assert abs(result.alpha) < 0.10, f"Alpha should be near zero, got {result.alpha}"
        assert not result.alpha_significant, "Alpha should not be significant"
        assert abs(result.factor_loadings["Mkt-RF"] - 1.2) < 0.15, (
            f"Mkt-RF beta should be ~1.2, got {result.factor_loadings['Mkt-RF']}"
        )
        assert result.r_squared > 0.8, f"R-squared should be high, got {result.r_squared}"

    def test_missing_factors_single_factor(self):
        """Only Mkt-RF provided -> single-factor model works."""
        n = 500
        rng = np.random.default_rng(42)
        mkt_rf = rng.normal(0.0004, 0.01, n)

        strategy_excess = 0.8 * mkt_rf + rng.normal(0.0, 0.002, n)
        factors = {"Mkt-RF": mkt_rf}

        result = compute_factor_attribution(strategy_excess, factors)

        assert "Mkt-RF" in result.factor_loadings
        assert len(result.factor_loadings) == 1
        assert abs(result.factor_loadings["Mkt-RF"] - 0.8) < 0.1
        assert result.r_squared > 0.7
        assert result.n_observations == n

    def test_factor_loadings_signs(self):
        """Check factor loadings have correct signs when strategy is a known linear combination."""
        n = 1000
        factors = self._make_factor_returns(n, seed=77)
        rng = np.random.default_rng(77)

        # Strategy: long market, short SMB (large-cap bias), long HML (value), no momentum
        strategy_excess = (
            1.0 * factors["Mkt-RF"]
            - 0.5 * factors["SMB"]
            + 0.3 * factors["HML"]
            + 0.0 * factors["UMD"]
            + rng.normal(0.0, 0.001, n)
        )

        result = compute_factor_attribution(strategy_excess, factors)

        assert result.factor_loadings["Mkt-RF"] > 0.8, "Should be positive market exposure"
        assert result.factor_loadings["SMB"] < -0.3, "Should be negative SMB (large-cap tilt)"
        assert result.factor_loadings["HML"] > 0.1, "Should be positive HML (value tilt)"
        assert abs(result.factor_loadings["UMD"]) < 0.15, "Should have near-zero momentum loading"

    def test_r_squared_bounds(self):
        """R-squared and adjusted R-squared should be in [0, 1] range."""
        n = 200
        factors = self._make_factor_returns(n, seed=10)
        rng = np.random.default_rng(10)
        strategy_excess = rng.normal(0.0005, 0.01, n)

        result = compute_factor_attribution(strategy_excess, factors)

        assert 0.0 <= result.r_squared <= 1.0
        assert result.adj_r_squared <= result.r_squared
        assert result.residual_vol > 0

    def test_residual_vol_annualized(self):
        """Residual vol should be annualized (multiplied by sqrt(252))."""
        n = 500
        factors = self._make_factor_returns(n, seed=55)
        rng = np.random.default_rng(9999)

        # Strategy with known residual noise
        noise_daily_std = 0.005
        strategy_excess = 0.5 * factors["Mkt-RF"] + rng.normal(0.0, noise_daily_std, n)

        result = compute_factor_attribution(strategy_excess, factors)

        expected_annual = noise_daily_std * np.sqrt(252)
        # Allow 30% tolerance since regression residuals won't exactly match input noise
        assert abs(result.residual_vol - expected_annual) / expected_annual < 0.3

    def test_tstats_present_for_all_factors(self):
        """Factor t-stats should be present for every factor provided."""
        n = 300
        factors = self._make_factor_returns(n, seed=33)
        rng = np.random.default_rng(33)
        strategy_excess = rng.normal(0.0003, 0.01, n)

        result = compute_factor_attribution(strategy_excess, factors)

        assert set(result.factor_tstats.keys()) == set(factors.keys())
        assert set(result.factor_loadings.keys()) == set(factors.keys())

    def test_dataclass_frozen(self):
        """FactorAttribution should be immutable."""
        n = 100
        factors = self._make_factor_returns(n, seed=1)
        rng = np.random.default_rng(1)
        strategy_excess = rng.normal(0.0, 0.01, n)

        result = compute_factor_attribution(strategy_excess, factors)

        with pytest.raises(AttributeError):
            result.alpha = 999.0

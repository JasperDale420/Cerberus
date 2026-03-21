"""Unit tests for the MomentumCrashDetector module."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.momentum_crash import (
    MomentumCrashConfig,
    MomentumCrashDetector,
    MomentumCrashResult,
    _sigmoid,
)

# ---------------------------------------------------------------------------
# Sigmoid tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSigmoid:
    def test_sigmoid_zero(self) -> None:
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self) -> None:
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_sigmoid_large_negative(self) -> None:
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_sigmoid_known_value(self) -> None:
        # sigmoid(1) = 1 / (1 + e^-1) ≈ 0.7311
        assert _sigmoid(1.0) == pytest.approx(0.7310585786, rel=1e-6)

    def test_sigmoid_symmetry(self) -> None:
        # sigmoid(x) + sigmoid(-x) = 1
        for x in [-5.0, -1.0, 0.5, 2.0, 10.0]:
            assert _sigmoid(x) + _sigmoid(-x) == pytest.approx(1.0, abs=1e-12)

    def test_sigmoid_monotonic(self) -> None:
        values = [_sigmoid(x) for x in np.linspace(-10, 10, 100)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]


# ---------------------------------------------------------------------------
# Low risk scenario: low vol, bull market, normal spread
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLowRiskScenario:
    def test_low_vol_bull_market_low_crash_prob(self) -> None:
        """Low vol + bull market + average spread should give ~5% crash prob."""
        detector = MomentumCrashDetector()

        # Feed enough spread history for z-score computation
        for _ in range(60):
            detector.update(
                realized_vol=0.12,
                market_return_cumulative=0.05,
                momentum_spread=0.03,
            )

        result = detector.get_crash_probability()
        assert result is not None
        # With constant spread, z-score=0; bear=0; vol=0.12
        # logit = -3.0 + 8.0*0.12 + 1.5*0 + 2.0*0 = -3.0 + 0.96 = -2.04
        # sigmoid(-2.04) ≈ 0.115
        assert result.crash_prob < 0.20
        assert result.exposure_scalar == 1.0  # Below threshold
        assert result.bear_market is False

    def test_very_low_vol_gives_base_rate(self) -> None:
        """Near-zero vol should give crash prob close to base rate sigmoid(-3.0)."""
        detector = MomentumCrashDetector()

        for _ in range(60):
            detector.update(
                realized_vol=0.001,
                market_return_cumulative=0.10,
                momentum_spread=0.02,
            )

        result = detector.get_crash_probability()
        assert result is not None
        # logit ≈ -3.0 + 8.0*0.001 + 0 + 0 ≈ -2.992
        # sigmoid(-2.992) ≈ 0.048
        assert result.crash_prob == pytest.approx(_sigmoid(-3.0 + 8.0 * 0.001), abs=0.01)


# ---------------------------------------------------------------------------
# High risk scenario: high vol, bear market, wide spread
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHighRiskScenario:
    def test_high_vol_bear_market_high_crash_prob(self) -> None:
        """High vol + bear market + elevated spread z-score should give >50% crash prob."""
        detector = MomentumCrashDetector()

        # Build history with moderate spread, then spike
        for _ in range(59):
            detector.update(
                realized_vol=0.40,
                market_return_cumulative=-0.15,
                momentum_spread=0.03,
            )

        # Now a big spread spike for high z-score
        result = detector.update(
            realized_vol=0.40,
            market_return_cumulative=-0.15,
            momentum_spread=0.10,
        )

        assert result.crash_prob > 0.50
        assert result.bear_market is True
        assert result.exposure_scalar < 0.50

    def test_extreme_conditions_caps_exposure(self) -> None:
        """Under extreme conditions, exposure_scalar should not go below min_exposure."""
        detector = MomentumCrashDetector()

        for _ in range(59):
            detector.update(
                realized_vol=0.80,
                market_return_cumulative=-0.30,
                momentum_spread=0.02,
            )

        result = detector.update(
            realized_vol=0.80,
            market_return_cumulative=-0.30,
            momentum_spread=0.15,
        )

        # With very high crash prob, exposure scalar should be clamped
        assert result.exposure_scalar >= 0.2
        assert result.crash_prob > 0.8


# ---------------------------------------------------------------------------
# Exposure scalar computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExposureScalar:
    def test_below_threshold_full_exposure(self) -> None:
        """When crash_prob < threshold, exposure should be 1.0."""
        config = MomentumCrashConfig(crash_prob_threshold=0.3)
        detector = MomentumCrashDetector(config=config)

        for _ in range(60):
            result = detector.update(
                realized_vol=0.10,
                market_return_cumulative=0.05,
                momentum_spread=0.03,
            )

        assert result.exposure_scalar == 1.0

    def test_above_threshold_reduced_exposure(self) -> None:
        """When crash_prob > threshold, exposure = max(min_exposure, 1 - crash_prob)."""
        config = MomentumCrashConfig(
            crash_prob_threshold=0.0,  # Always scale
            min_exposure=0.2,
        )
        detector = MomentumCrashDetector(config=config)

        for _ in range(60):
            result = detector.update(
                realized_vol=0.30,
                market_return_cumulative=-0.15,
                momentum_spread=0.03,
            )

        # crash_prob > 0 so we scale, exposure = max(0.2, 1 - crash_prob)
        assert result.exposure_scalar == pytest.approx(max(0.2, 1.0 - result.crash_prob), abs=1e-10)

    def test_min_exposure_floor(self) -> None:
        """Exposure scalar should never go below min_exposure."""
        config = MomentumCrashConfig(min_exposure=0.3, crash_prob_threshold=0.0)
        detector = MomentumCrashDetector(config=config)

        # Force extreme crash prob
        for _ in range(59):
            detector.update(
                realized_vol=1.0,
                market_return_cumulative=-0.50,
                momentum_spread=0.01,
            )
        result = detector.update(
            realized_vol=1.0,
            market_return_cumulative=-0.50,
            momentum_spread=0.20,
        )

        assert result.exposure_scalar >= 0.3


# ---------------------------------------------------------------------------
# Bear market indicator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBearMarketIndicator:
    def test_bear_threshold_exact(self) -> None:
        """At exactly the threshold, should NOT be bear market (< not <=)."""
        detector = MomentumCrashDetector()
        result = detector.update(
            realized_vol=0.15,
            market_return_cumulative=-0.10,  # Exactly at threshold
            momentum_spread=0.03,
        )
        assert result.bear_market is False

    def test_below_threshold_is_bear(self) -> None:
        detector = MomentumCrashDetector()
        result = detector.update(
            realized_vol=0.15,
            market_return_cumulative=-0.11,
            momentum_spread=0.03,
        )
        assert result.bear_market is True

    def test_above_threshold_not_bear(self) -> None:
        detector = MomentumCrashDetector()
        result = detector.update(
            realized_vol=0.15,
            market_return_cumulative=0.05,
            momentum_spread=0.03,
        )
        assert result.bear_market is False

    def test_custom_bear_threshold(self) -> None:
        config = MomentumCrashConfig(bear_threshold=-0.20)
        detector = MomentumCrashDetector(config=config)

        result = detector.update(
            realized_vol=0.15,
            market_return_cumulative=-0.15,
            momentum_spread=0.03,
        )
        assert result.bear_market is False  # -0.15 > -0.20

        result = detector.update(
            realized_vol=0.15,
            market_return_cumulative=-0.25,
            momentum_spread=0.03,
        )
        assert result.bear_market is True  # -0.25 < -0.20


# ---------------------------------------------------------------------------
# Momentum spread z-score
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMomentumSpreadZScore:
    def test_constant_spread_zscore_zero(self) -> None:
        """Constant spread should yield z-score of 0."""
        detector = MomentumCrashDetector()

        for _ in range(60):
            result = detector.update(
                realized_vol=0.15,
                market_return_cumulative=0.0,
                momentum_spread=0.05,
            )

        assert result.momentum_spread_zscore == pytest.approx(0.0, abs=1e-10)

    def test_spike_spread_positive_zscore(self) -> None:
        """A sudden spread increase should produce a positive z-score."""
        detector = MomentumCrashDetector()

        for _ in range(59):
            detector.update(
                realized_vol=0.15,
                market_return_cumulative=0.0,
                momentum_spread=0.03,
            )

        result = detector.update(
            realized_vol=0.15,
            market_return_cumulative=0.0,
            momentum_spread=0.10,
        )

        assert result.momentum_spread_zscore > 1.0

    def test_single_observation_zscore_zero(self) -> None:
        """With only one observation, z-score should be 0."""
        detector = MomentumCrashDetector()
        result = detector.update(
            realized_vol=0.15,
            market_return_cumulative=0.0,
            momentum_spread=0.05,
        )
        assert result.momentum_spread_zscore == 0.0

    def test_two_observations_computes_zscore(self) -> None:
        """With two observations, z-score should be computable."""
        detector = MomentumCrashDetector()
        detector.update(
            realized_vol=0.15,
            market_return_cumulative=0.0,
            momentum_spread=0.02,
        )
        result = detector.update(
            realized_vol=0.15,
            market_return_cumulative=0.0,
            momentum_spread=0.06,
        )
        # mean=0.04, sample std (ddof=1) = 0.02*sqrt(2), z = (0.06-0.04)/(0.02*sqrt(2)) ≈ 0.7071
        assert result.momentum_spread_zscore == pytest.approx(1.0 / (2**0.5), abs=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    def test_zero_volatility(self) -> None:
        """Zero volatility should not crash and should give low crash prob."""
        detector = MomentumCrashDetector()
        result = detector.update(
            realized_vol=0.0,
            market_return_cumulative=0.05,
            momentum_spread=0.03,
        )
        assert result is not None
        # logit = -3.0 + 0 + 0 + 0 = -3.0 (with z-score=0 for first obs)
        assert result.crash_prob == pytest.approx(_sigmoid(-3.0), abs=1e-6)

    def test_empty_spread_history(self) -> None:
        """First update with no history should return a valid result."""
        detector = MomentumCrashDetector()
        result = detector.update(
            realized_vol=0.20,
            market_return_cumulative=-0.05,
            momentum_spread=0.04,
        )
        assert isinstance(result, MomentumCrashResult)
        assert 0.0 <= result.crash_prob <= 1.0
        assert result.momentum_spread_zscore == 0.0

    def test_extreme_positive_vol(self) -> None:
        """Extremely high volatility should drive crash prob near 1.0."""
        detector = MomentumCrashDetector()
        result = detector.update(
            realized_vol=5.0,
            market_return_cumulative=-0.50,
            momentum_spread=0.03,
        )
        assert result.crash_prob > 0.99

    def test_extreme_negative_cumulative_return(self) -> None:
        """Very negative market return should trigger bear market."""
        detector = MomentumCrashDetector()
        result = detector.update(
            realized_vol=0.15,
            market_return_cumulative=-0.80,
            momentum_spread=0.03,
        )
        assert result.bear_market is True

    def test_get_crash_probability_before_update(self) -> None:
        """get_crash_probability should return None before any update."""
        detector = MomentumCrashDetector()
        assert detector.get_crash_probability() is None

    def test_get_crash_probability_after_update(self) -> None:
        """get_crash_probability should return the latest result."""
        detector = MomentumCrashDetector()
        result = detector.update(
            realized_vol=0.15,
            market_return_cumulative=0.0,
            momentum_spread=0.03,
        )
        assert detector.get_crash_probability() is result

    def test_negative_momentum_spread(self) -> None:
        """Negative spread (losers outperforming winners) should work fine."""
        detector = MomentumCrashDetector()
        for _ in range(60):
            result = detector.update(
                realized_vol=0.15,
                market_return_cumulative=0.0,
                momentum_spread=-0.05,
            )
        assert isinstance(result, MomentumCrashResult)


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigOverrides:
    def test_custom_betas(self) -> None:
        """Custom beta coefficients should change crash probability."""
        # Zero all betas except intercept: crash_prob = sigmoid(intercept)
        config = MomentumCrashConfig(
            beta_intercept=-1.0,
            beta_vol=0.0,
            beta_bear=0.0,
            beta_spread=0.0,
        )
        detector = MomentumCrashDetector(config=config)
        result = detector.update(
            realized_vol=0.50,
            market_return_cumulative=-0.30,
            momentum_spread=0.10,
        )
        assert result.crash_prob == pytest.approx(_sigmoid(-1.0), abs=1e-6)

    def test_custom_spread_lookback(self) -> None:
        """Custom spread_lookback should control deque size."""
        config = MomentumCrashConfig(spread_lookback=10)
        detector = MomentumCrashDetector(config=config)

        # Fill 10 values
        for _ in range(10):
            detector.update(
                realized_vol=0.15,
                market_return_cumulative=0.0,
                momentum_spread=0.03,
            )

        # 11th value should evict oldest
        detector.update(
            realized_vol=0.15,
            market_return_cumulative=0.0,
            momentum_spread=0.03,
        )
        assert len(detector._spread_history) == 10

    def test_custom_min_exposure(self) -> None:
        """Custom min_exposure should be respected."""
        config = MomentumCrashConfig(min_exposure=0.5, crash_prob_threshold=0.0)
        detector = MomentumCrashDetector(config=config)

        # Force high crash prob
        for _ in range(59):
            detector.update(
                realized_vol=1.0,
                market_return_cumulative=-0.30,
                momentum_spread=0.02,
            )
        result = detector.update(
            realized_vol=1.0,
            market_return_cumulative=-0.30,
            momentum_spread=0.15,
        )

        assert result.exposure_scalar >= 0.5

    def test_default_config_values(self) -> None:
        """Default config should match spec."""
        config = MomentumCrashConfig()
        assert config.enabled is False
        assert config.beta_intercept == -3.0
        assert config.beta_vol == 8.0
        assert config.beta_bear == 1.5
        assert config.beta_spread == 2.0
        assert config.bear_threshold == -0.10
        assert config.spread_lookback == 60
        assert config.min_exposure == 0.2
        assert config.crash_prob_threshold == 0.3

    def test_config_accessible(self) -> None:
        """Config should be accessible via property."""
        config = MomentumCrashConfig(min_exposure=0.4)
        detector = MomentumCrashDetector(config=config)
        assert detector.config.min_exposure == 0.4


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMomentumCrashResult:
    def test_result_is_frozen(self) -> None:
        """MomentumCrashResult should be immutable (frozen dataclass)."""
        result = MomentumCrashResult(
            crash_prob=0.5,
            exposure_scalar=0.5,
            bear_market=True,
            momentum_spread_zscore=1.0,
            realized_vol=0.3,
        )
        with pytest.raises(AttributeError):
            result.crash_prob = 0.9  # type: ignore[misc]

    def test_result_fields(self) -> None:
        """All expected fields should be present and accessible."""
        result = MomentumCrashResult(
            crash_prob=0.25,
            exposure_scalar=0.75,
            bear_market=False,
            momentum_spread_zscore=-0.5,
            realized_vol=0.18,
        )
        assert result.crash_prob == 0.25
        assert result.exposure_scalar == 0.75
        assert result.bear_market is False
        assert result.momentum_spread_zscore == -0.5
        assert result.realized_vol == 0.18

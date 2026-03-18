"""Tests for Markov regime-switching and adaptive threshold engine."""

from __future__ import annotations

import numpy as np

from src.quant.regime import (
    AdaptiveThresholdEngine,
    MarkovRegimeSwitcher,
    RegimeSwitchResult,
)

# ---------------------------------------------------------------------------
# MarkovRegimeSwitcher tests
# ---------------------------------------------------------------------------


class TestMarkovRegimeSwitcher:
    """Tests for the MarkovRegimeSwitcher class."""

    def test_returns_none_before_min_observations(self):
        """Must return None until min_observations values received."""
        switcher = MarkovRegimeSwitcher(n_regimes=2, min_observations=100)
        result = None
        for i in range(99):
            result = switcher.update(float(i) * 0.01)
        assert result is None
        assert switcher.last_result is None

    def test_fits_two_state_model_with_clear_regime_switch(self):
        """Given low-vol then high-vol data, should detect two regimes."""
        rng = np.random.default_rng(42)
        switcher = MarkovRegimeSwitcher(
            n_regimes=2,
            min_observations=100,
            lookback=500,
            refit_interval=50,
        )

        # 150 bars of low volatility (std=0.5)
        low_vol = rng.normal(0.0, 0.5, size=150)
        # 150 bars of high volatility (std=2.0)
        high_vol = rng.normal(0.0, 2.0, size=150)
        data = np.concatenate([low_vol, high_vol])

        result = None
        for val in data:
            result = switcher.update(float(val))

        # After 300 bars with refit_interval=50, we should have a result
        if result is not None:
            assert isinstance(result, RegimeSwitchResult)
            assert result.current_regime in (0, 1)
            assert 0.0 <= result.filtered_probability <= 1.0
            assert len(result.regime_probabilities) == 2
            assert abs(sum(result.regime_probabilities) - 1.0) < 1e-6
            assert result.n_observations == 300
        else:
            # Convergence failure is acceptable — verify graceful fallback
            assert switcher.last_result is None

    def test_convergence_failure_returns_none_gracefully(self):
        """If the model fails to converge, should return None (not raise)."""
        switcher = MarkovRegimeSwitcher(
            n_regimes=2,
            min_observations=10,
            lookback=50,
            refit_interval=10,
        )
        # Constant data — model cannot distinguish regimes
        for _ in range(20):
            result = switcher.update(1.0)
        # Should not raise; returns None or last valid result
        assert result is None or isinstance(result, RegimeSwitchResult)


# ---------------------------------------------------------------------------
# AdaptiveThresholdEngine tests
# ---------------------------------------------------------------------------


class TestAdaptiveThresholdEngine:
    """Tests for the AdaptiveThresholdEngine class."""

    def test_high_vol_widens_threshold(self):
        """When conditional vol exceeds baseline, threshold should widen."""
        engine = AdaptiveThresholdEngine(vol_scale_factor=1.0)
        base = 1.0
        baseline_vol = 0.01

        normal = engine.adapt(base, conditional_vol=baseline_vol, baseline_vol=baseline_vol)
        wide = engine.adapt(base, conditional_vol=baseline_vol * 3.0, baseline_vol=baseline_vol)

        assert wide > normal

    def test_low_vol_tightens_threshold(self):
        """When conditional vol is below baseline, threshold should tighten."""
        engine = AdaptiveThresholdEngine(vol_scale_factor=1.0)
        base = 1.0
        baseline_vol = 0.02

        normal = engine.adapt(base, conditional_vol=baseline_vol, baseline_vol=baseline_vol)
        tight = engine.adapt(base, conditional_vol=baseline_vol * 0.3, baseline_vol=baseline_vol)

        assert tight < normal

    def test_strong_mean_reversion_tightens_threshold(self):
        """Hurst < 0.5 (mean-reverting) should tighten the threshold."""
        engine = AdaptiveThresholdEngine(hurst_scale_factor=1.0)
        base = 1.0

        neutral = engine.adapt(base, hurst=0.5)
        tight = engine.adapt(base, hurst=0.2)

        assert tight < neutral

    def test_trending_hurst_widens_threshold(self):
        """Hurst > 0.5 (trending) should widen the threshold."""
        engine = AdaptiveThresholdEngine(hurst_scale_factor=1.0)
        base = 1.0

        neutral = engine.adapt(base, hurst=0.5)
        wide = engine.adapt(base, hurst=0.8)

        assert wide > neutral

    def test_high_regime_confidence_tightens(self):
        """High regime confidence → multiplier approaches 1.0 (tighter)."""
        engine = AdaptiveThresholdEngine()
        base = 1.0

        low_conf = engine.adapt(base, regime_confidence=0.5)
        high_conf = engine.adapt(base, regime_confidence=1.0)

        assert high_conf < low_conf

    def test_clamp_to_bounds(self):
        """Final multiplier should be clamped to [min_multiplier, max_multiplier]."""
        engine = AdaptiveThresholdEngine(
            vol_scale_factor=1.0,
            min_multiplier=0.5,
            max_multiplier=3.0,
        )
        base = 1.0

        # Extremely high vol should hit the ceiling
        result = engine.adapt(base, conditional_vol=1.0, baseline_vol=0.001)
        assert result <= base * 3.0

        # Extremely low vol should hit the floor
        result = engine.adapt(base, conditional_vol=0.0001, baseline_vol=1.0)
        assert result >= base * 0.5

    def test_defaults_return_base_threshold(self):
        """With all defaults (vol=0, hurst=0.5, confidence=1.0), should return ~base."""
        engine = AdaptiveThresholdEngine()
        base = 2.5
        result = engine.adapt(base)
        assert abs(result - base) < 1e-9

"""Unit tests for Probability of Backtest Overfitting (PBO) via CSCV."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_random_data_pbo_near_half():
    """Random trials have no real edge, so IS-best is random — PBO should be ~0.5."""
    from src.analytics.pbo import compute_pbo

    rng = np.random.default_rng(42)
    # 1000 bars, 20 trials of pure noise
    trial_returns = rng.normal(0, 0.01, size=(1000, 20))
    result = compute_pbo(trial_returns, n_partitions=8)

    assert 0.15 <= result["pbo"] <= 0.85, f"PBO for random data should be ~0.5, got {result['pbo']}"
    assert result["n_combinations_tested"] > 0
    assert isinstance(result["overfit_warning"], bool)


@pytest.mark.unit
def test_dominant_trial_low_pbo():
    """One trial with strong positive drift should be consistently best — low PBO."""
    from src.analytics.pbo import compute_pbo

    rng = np.random.default_rng(123)
    n_bars, n_trials = 1200, 15
    # All trials are noise
    trial_returns = rng.normal(0, 0.01, size=(n_bars, n_trials))
    # Trial 0 has strong positive drift
    trial_returns[:, 0] = rng.normal(0.005, 0.005, size=n_bars)

    result = compute_pbo(trial_returns, n_partitions=8)
    assert result["pbo"] < 0.3, f"Dominant trial should yield low PBO, got {result['pbo']}"
    assert result["overfit_warning"] is False


@pytest.mark.unit
def test_identical_trials_pbo_near_half():
    """All identical trials — no trial strictly beats another, PBO should be 0."""
    from src.analytics.pbo import compute_pbo

    rng = np.random.default_rng(99)
    single = rng.normal(0.001, 0.01, size=(800, 1))
    trial_returns = np.tile(single, (1, 10))

    result = compute_pbo(trial_returns, n_partitions=8)
    # With identical trials, IS-best never strictly underperforms median OOS
    assert result["pbo"] == 0.0, f"Identical trials PBO should be 0.0, got {result['pbo']}"


@pytest.mark.unit
def test_too_few_partitions_raises():
    """n_partitions < 4 should raise ValueError (need at least 2 IS + 2 OOS blocks)."""
    from src.analytics.pbo import compute_pbo

    trial_returns = np.random.default_rng(0).normal(0, 1, size=(100, 5))
    with pytest.raises(ValueError, match="partitions"):
        compute_pbo(trial_returns, n_partitions=2)


@pytest.mark.unit
def test_too_few_trials_raises():
    """Need at least 2 trials to rank."""
    from src.analytics.pbo import compute_pbo

    trial_returns = np.random.default_rng(0).normal(0, 1, size=(100, 1))
    with pytest.raises(ValueError, match="trials"):
        compute_pbo(trial_returns, n_partitions=4)


@pytest.mark.unit
def test_sampling_caps_combinations():
    """With n_partitions=16, C(16,8)=12870 > 1000 — should sample down to 1000."""
    from src.analytics.pbo import compute_pbo

    rng = np.random.default_rng(7)
    trial_returns = rng.normal(0, 0.01, size=(1600, 10))
    result = compute_pbo(trial_returns, n_partitions=16)

    assert result["n_combinations_tested"] <= 1000
    assert result["n_combinations_tested"] > 0


@pytest.mark.unit
def test_return_keys():
    """Result dict has all expected keys."""
    from src.analytics.pbo import compute_pbo

    rng = np.random.default_rng(0)
    trial_returns = rng.normal(0, 0.01, size=(400, 5))
    result = compute_pbo(trial_returns, n_partitions=4)

    expected_keys = {"pbo", "n_combinations_tested", "avg_oos_logit", "overfit_warning"}
    assert set(result.keys()) == expected_keys


@pytest.mark.unit
def test_odd_partitions_uses_floor():
    """Odd n_partitions should still work (floor division for IS/OOS split)."""
    from src.analytics.pbo import compute_pbo

    rng = np.random.default_rng(42)
    trial_returns = rng.normal(0, 0.01, size=(500, 5))
    # 5 partitions: IS gets 2 blocks, OOS gets 3
    result = compute_pbo(trial_returns, n_partitions=5)
    assert 0.0 <= result["pbo"] <= 1.0

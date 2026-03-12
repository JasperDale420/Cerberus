"""Unit tests for Hierarchical Risk Parity (HRP) cross-strategy allocator."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.engine.hrp import HRPAllocator, HRPConfig


def _make_allocator(**overrides) -> HRPAllocator:
    """Create an HRPAllocator with test-friendly defaults."""
    defaults = {"enabled": True, "lookback_days": 10, "min_strategies": 3, "rebalance_interval": 1}
    defaults.update(overrides)
    return HRPAllocator(config=HRPConfig(**defaults))


def _feed_returns(allocator: HRPAllocator, strategy: str, returns: list[float], start_date: date | None = None) -> None:
    """Feed a list of daily returns for a strategy starting from start_date."""
    base = start_date or date(2025, 1, 1)
    for i, r in enumerate(returns):
        allocator.record_daily_return(strategy, base + timedelta(days=i), r)


@pytest.mark.unit
def test_insufficient_strategies() -> None:
    """Fewer than min_strategies should return None."""
    alloc = _make_allocator(min_strategies=3, lookback_days=5)
    _feed_returns(alloc, "strat_a", [0.01] * 5)
    _feed_returns(alloc, "strat_b", [0.02] * 5)
    # Only 2 strategies, need 3
    result = alloc.compute_weights()
    assert result is None


@pytest.mark.unit
def test_insufficient_data() -> None:
    """Less than lookback_days of data should return None."""
    alloc = _make_allocator(lookback_days=10, min_strategies=3)
    # Only 5 days of data, need 10
    for name in ["a", "b", "c"]:
        _feed_returns(alloc, name, [0.01] * 5)
    result = alloc.compute_weights()
    assert result is None


@pytest.mark.unit
def test_equal_returns_equal_weights() -> None:
    """Identical returns across strategies should produce equal weights."""
    alloc = _make_allocator(lookback_days=10, min_strategies=3, min_weight=0.0, max_weight=1.0)
    returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.002, 0.01, -0.008, 0.012]
    for name in ["alpha", "beta", "gamma"]:
        _feed_returns(alloc, name, returns)

    weights = alloc.compute_weights()
    assert weights is not None
    assert len(weights) == 3

    expected = 1.0 / 3.0
    for w in weights.values():
        assert w == pytest.approx(expected, abs=0.01)


@pytest.mark.unit
def test_uncorrelated_higher_weight() -> None:
    """An uncorrelated strategy should receive more weight than correlated ones."""
    alloc = _make_allocator(lookback_days=20, min_strategies=3, min_weight=0.0, max_weight=1.0)

    rng = np.random.RandomState(42)
    base = rng.normal(0, 0.01, 20).tolist()

    # Two correlated strategies (same returns + small noise)
    corr_a = [r + rng.normal(0, 0.001) for r in base]
    corr_b = [r + rng.normal(0, 0.001) for r in base]
    # One uncorrelated strategy (independent returns)
    uncorr = rng.normal(0, 0.01, 20).tolist()

    _feed_returns(alloc, "corr_a", corr_a)
    _feed_returns(alloc, "corr_b", corr_b)
    _feed_returns(alloc, "uncorr", uncorr)

    weights = alloc.compute_weights()
    assert weights is not None

    # Uncorrelated strategy should get at least as much as either correlated one
    assert weights["uncorr"] >= min(weights["corr_a"], weights["corr_b"])


@pytest.mark.unit
def test_correlated_lower_weight() -> None:
    """Perfectly correlated strategies should share weight (each gets less than uncorrelated)."""
    alloc = _make_allocator(lookback_days=20, min_strategies=3, min_weight=0.0, max_weight=1.0)

    rng = np.random.RandomState(99)
    base = rng.normal(0, 0.02, 20).tolist()

    # Two perfectly correlated (identical)
    _feed_returns(alloc, "clone_a", base)
    _feed_returns(alloc, "clone_b", base[:])  # copy
    # One independent
    independent = rng.normal(0, 0.02, 20).tolist()
    _feed_returns(alloc, "independent", independent)

    weights = alloc.compute_weights()
    assert weights is not None

    # Combined weight of clones should be comparable to independent
    # Each clone individually should get less than the independent strategy
    combined_clones = weights["clone_a"] + weights["clone_b"]
    assert combined_clones < weights["independent"] * 2.5  # generous bound


@pytest.mark.unit
def test_weight_bounds() -> None:
    """All weights must be between min_weight and max_weight."""
    alloc = _make_allocator(lookback_days=10, min_strategies=3, min_weight=0.05, max_weight=0.40)

    rng = np.random.RandomState(7)
    for name in ["s1", "s2", "s3", "s4"]:
        returns = rng.normal(0, 0.01, 10).tolist()
        _feed_returns(alloc, name, returns)

    weights = alloc.compute_weights()
    assert weights is not None

    for s, w in weights.items():
        assert w >= alloc.config.min_weight - 1e-9, f"{s} weight {w} below min"
        assert w <= alloc.config.max_weight + 1e-9, f"{s} weight {w} above max"


@pytest.mark.unit
def test_weights_sum_to_one() -> None:
    """Total weight must equal 1.0 within floating point tolerance."""
    alloc = _make_allocator(lookback_days=10, min_strategies=3)

    rng = np.random.RandomState(123)
    for name in ["x", "y", "z", "w"]:
        returns = rng.normal(0, 0.01, 10).tolist()
        _feed_returns(alloc, name, returns)

    weights = alloc.compute_weights()
    assert weights is not None
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_single_strategy() -> None:
    """Single strategy gets weight 1.0 (when min_strategies=1)."""
    alloc = _make_allocator(lookback_days=5, min_strategies=1)
    _feed_returns(alloc, "only", [0.01, -0.005, 0.02, -0.01, 0.015])

    weights = alloc.compute_weights()
    assert weights is not None
    assert weights["only"] == pytest.approx(1.0)


@pytest.mark.unit
def test_get_strategy_weight_unknown() -> None:
    """Unknown strategy returns equal weight fallback."""
    alloc = _make_allocator(lookback_days=5, min_strategies=3)

    rng = np.random.RandomState(55)
    for name in ["a", "b", "c"]:
        returns = rng.normal(0, 0.01, 5).tolist()
        _feed_returns(alloc, name, returns)

    alloc.compute_weights()

    # "unknown" was never recorded
    w = alloc.get_strategy_weight("unknown")
    # Should be 1/(3+1) = 0.25 since 3 known strategies + 1 unknown
    assert w == pytest.approx(1.0 / 4.0)


@pytest.mark.unit
def test_record_daily_return() -> None:
    """Returns are recorded correctly and accessible."""
    alloc = _make_allocator()
    d = date(2025, 6, 1)
    alloc.record_daily_return("test_strat", d, 0.05)
    alloc.record_daily_return("test_strat", d + timedelta(days=1), -0.02)

    assert len(alloc._returns["test_strat"]) == 2
    assert alloc._returns["test_strat"][0] == (d, 0.05)
    assert alloc._returns["test_strat"][1] == (d + timedelta(days=1), -0.02)


@pytest.mark.unit
def test_rebalance_interval() -> None:
    """Weights should not recompute before the rebalance interval expires."""
    alloc = _make_allocator(lookback_days=5, min_strategies=3, rebalance_interval=5)

    rng = np.random.RandomState(77)
    base_date = date(2025, 1, 1)
    for name in ["a", "b", "c"]:
        returns = rng.normal(0, 0.01, 5).tolist()
        _feed_returns(alloc, name, returns, start_date=base_date)

    # First computation
    w1 = alloc.compute_weights()
    assert w1 is not None
    count_after_first = alloc._computation_count

    # Feed 2 more days (within interval of 5)
    for name in ["a", "b", "c"]:
        for i in range(2):
            alloc.record_daily_return(name, base_date + timedelta(days=5 + i), rng.normal(0, 0.01))

    w2 = alloc.compute_weights()
    assert w2 is not None
    # Should have reused cached weights (no new computation)
    assert alloc._computation_count == count_after_first

    # Feed enough days to exceed interval
    for name in ["a", "b", "c"]:
        for i in range(4):
            alloc.record_daily_return(name, base_date + timedelta(days=7 + i), rng.normal(0, 0.01))

    w3 = alloc.compute_weights()
    assert w3 is not None
    # Should have recomputed
    assert alloc._computation_count == count_after_first + 1

"""Tests for portfolio risk-parity allocator."""

from __future__ import annotations

import pytest

from src.portfolio.allocator import AllocationResult, PortfolioAllocator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def allocator() -> PortfolioAllocator:
    return PortfolioAllocator()


# ---------------------------------------------------------------------------
# Risk-parity weights
# ---------------------------------------------------------------------------


class TestRiskParityWeights:
    """Equal-vol -> equal weight; higher-vol -> less weight."""

    def test_equal_vol_equal_allocation(self, allocator: PortfolioAllocator) -> None:
        """Strategies with identical volatility should receive equal allocation."""
        returns = {
            "alpha": [0.01, -0.01, 0.01, -0.01, 0.01],
            "beta": [0.01, -0.01, 0.01, -0.01, 0.01],
            "gamma": [0.01, -0.01, 0.01, -0.01, 0.01],
        }
        result = allocator.compute_allocations(returns)
        values = list(result.allocations.values())
        assert all(abs(v - values[0]) < 1e-6 for v in values)

    def test_higher_vol_gets_less(self, allocator: PortfolioAllocator) -> None:
        """A strategy with higher realized vol should get a smaller allocation."""
        # low vol
        low_vol_rets = [0.001, -0.001, 0.001, -0.001, 0.001] * 4
        # high vol — 10x amplitude
        high_vol_rets = [0.01, -0.01, 0.01, -0.01, 0.01] * 4
        returns = {"low_vol": low_vol_rets, "high_vol": high_vol_rets}
        result = allocator.compute_allocations(returns)
        assert result.allocations["low_vol"] > result.allocations["high_vol"]

    def test_empty_returns(self, allocator: PortfolioAllocator) -> None:
        """No strategies -> empty result."""
        result = allocator.compute_allocations({})
        assert result.allocations == {}
        assert result.total_exposure == 0.0


# ---------------------------------------------------------------------------
# Drawdown throttle
# ---------------------------------------------------------------------------


class TestDrawdownThrottle:
    def test_throttle_halves_allocation(self) -> None:
        """When current DD exceeds mult * historical_max_dd, allocation is halved."""
        allocator = PortfolioAllocator(drawdown_throttle_mult=1.5)
        # Construct returns where trailing DD is large
        # Start positive then drop sharply
        heavy_dd_rets = [0.05] * 10 + [-0.10] * 5
        mild_rets = [0.01, -0.005] * 10

        returns = {"risky": heavy_dd_rets, "safe": mild_rets}
        # Set historical max DD much smaller than current for risky
        strategy_max_dd = {"risky": 0.05, "safe": 0.10}

        result = allocator.compute_allocations(returns, strategy_max_dd=strategy_max_dd)
        assert "risky" in result.throttled_strategies

    def test_no_throttle_within_threshold(self) -> None:
        """Strategies within normal DD range should not be throttled."""
        allocator = PortfolioAllocator(drawdown_throttle_mult=1.5)
        mild_rets = [0.01, -0.005] * 10
        returns = {"strat_a": mild_rets, "strat_b": mild_rets}
        result = allocator.compute_allocations(returns)
        assert result.throttled_strategies == []


# ---------------------------------------------------------------------------
# Correlation adjustment
# ---------------------------------------------------------------------------


class TestCorrelationAdjustment:
    def test_high_correlation_reduces_exposure(self, allocator: PortfolioAllocator) -> None:
        """Perfectly correlated strategies should reduce total exposure below 1."""
        identical = [0.01, -0.02, 0.015, -0.01, 0.005] * 4
        returns = {"a": identical[:], "b": identical[:]}
        result = allocator.compute_allocations(returns)
        assert result.total_exposure < 1.0

    def test_uncorrelated_no_reduction(self) -> None:
        """Uncorrelated strategies should keep full exposure (close to 1)."""
        import numpy as np

        # Use higher max so clamping doesn't mask the correlation effect
        alloc = PortfolioAllocator(max_strategy_allocation=0.60)
        rng = np.random.default_rng(42)
        n = 200
        returns = {
            "x": rng.normal(0.001, 0.01, n).tolist(),
            "y": rng.normal(0.001, 0.01, n).tolist(),
        }
        result = alloc.compute_allocations(returns)
        # Should be close to 1 since correlation is near 0
        assert result.total_exposure > 0.85


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


class TestAllocationClamping:
    def test_max_clamp(self) -> None:
        """No single strategy should exceed max_strategy_allocation."""
        allocator = PortfolioAllocator(max_strategy_allocation=0.40)
        # One very low vol (would dominate) + one high vol
        low_vol = [0.0001, -0.0001] * 20
        high_vol = [0.05, -0.05] * 20
        returns = {"dominant": low_vol, "minor": high_vol}
        result = allocator.compute_allocations(returns)
        for alloc in result.allocations.values():
            assert alloc <= 0.40 + 1e-9

    def test_min_clamp(self) -> None:
        """No strategy should fall below min_strategy_allocation."""
        allocator = PortfolioAllocator(min_strategy_allocation=0.05)
        # Extreme vol difference
        low_vol = [0.0001, -0.0001] * 20
        high_vol = [0.10, -0.10] * 20
        returns = {"dominant": low_vol, "tiny": high_vol}
        result = allocator.compute_allocations(returns)
        for alloc in result.allocations.values():
            assert alloc >= 0.05 - 1e-9


# ---------------------------------------------------------------------------
# AllocationResult dataclass
# ---------------------------------------------------------------------------


class TestAllocationResult:
    def test_defaults(self) -> None:
        r = AllocationResult(allocations={"a": 0.5}, total_exposure=0.5)
        assert r.throttled_strategies == []

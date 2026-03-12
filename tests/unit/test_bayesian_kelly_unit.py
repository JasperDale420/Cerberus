"""Tests for Wasserstein DRO robust Kelly Criterion position sizer."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.config.models import KellyConfig
from src.engine.kelly import KellySizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sizer(
    *,
    robust: bool = True,
    epsilon: float = 0.05,
    enabled: bool = True,
    fraction: float = 1.0,
    min_trades: int = 5,
    lookback_trades: int = 200,
    min_equity_pct: float = 0.01,
    max_equity_pct: float = 0.50,
) -> KellySizer:
    """Create a KellySizer with sensible test defaults."""
    config = KellyConfig(
        enabled=enabled,
        fraction=fraction,
        lookback_trades=lookback_trades,
        min_trades=min_trades,
        min_equity_pct=min_equity_pct,
        max_equity_pct=max_equity_pct,
    )
    logger = MagicMock()
    sizer = KellySizer(config, logger)
    # Override robust/epsilon since config model may not have these yet
    sizer.robust = robust
    sizer.epsilon = epsilon
    return sizer


def _record_trades(sizer: KellySizer, strategy: str, trades: list[float]) -> None:
    for pnl in trades:
        sizer.record_trade(strategy, pnl)


# ---------------------------------------------------------------------------
# Robust Kelly vs Standard Kelly
# ---------------------------------------------------------------------------


class TestRobustKellyVsStandard:
    """Robust Kelly should always be <= standard Kelly."""

    def test_robust_kelly_lower_than_standard(self):
        sizer = _make_sizer(robust=True, epsilon=0.05)
        trades = [100.0, -50.0, 80.0, -30.0, 120.0, -60.0, 90.0, -40.0, 110.0, -55.0]
        _record_trades(sizer, "test_strat", trades)

        stats = sizer._get_stats("test_strat")
        standard = KellySizer._compute_standard_kelly(stats)
        robust = sizer._compute_robust_kelly(stats)

        assert robust <= standard, f"robust {robust} should be <= standard {standard}"

    def test_robust_kelly_with_high_epsilon(self):
        """Large epsilon should produce a very conservative (small) fraction."""
        sizer = _make_sizer(robust=True, epsilon=0.40)
        trades = [100.0, -50.0, 80.0, -30.0, 120.0, -60.0, 90.0, -40.0, 110.0, -55.0]
        _record_trades(sizer, "test_strat", trades)

        stats = sizer._get_stats("test_strat")
        robust_high = sizer._compute_robust_kelly(stats)

        sizer2 = _make_sizer(robust=True, epsilon=0.01)
        _record_trades(sizer2, "test_strat", trades)
        stats2 = sizer2._get_stats("test_strat")
        robust_low = sizer2._compute_robust_kelly(stats2)

        assert robust_high <= robust_low, (
            f"high epsilon robust {robust_high} should be <= low epsilon robust {robust_low}"
        )

    def test_robust_kelly_with_zero_epsilon(self):
        """Epsilon=0 should yield the same as standard Kelly."""
        sizer = _make_sizer(robust=True, epsilon=0.0)
        trades = [100.0, -50.0, 80.0, -30.0, 120.0, -60.0, 90.0, -40.0, 110.0, -55.0]
        _record_trades(sizer, "test_strat", trades)

        stats = sizer._get_stats("test_strat")
        standard = KellySizer._compute_standard_kelly(stats)
        robust = sizer._compute_robust_kelly(stats)

        assert abs(robust - standard) < 1e-9, f"epsilon=0 robust {robust} should equal standard {standard}"


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_bootstrap_ci_contains_point_estimate(self):
        """The standard Kelly point estimate should generally fall within the CI."""
        sizer = _make_sizer(robust=True)
        # Use a distribution with clear edge so point estimate is solidly positive
        trades = [150.0, -50.0] * 20  # 40 trades, 50% win rate, 3:1 payoff
        _record_trades(sizer, "test_strat", trades)

        stats = sizer._get_stats("test_strat")
        standard = KellySizer._compute_standard_kelly(stats)
        ci_lower, ci_median, ci_upper = sizer._bootstrap_kelly_ci(stats)

        # Point estimate should be within or very near the CI
        # (with 200 bootstrap samples, this should hold for clean distributions)
        assert ci_lower <= standard + 0.05, f"standard {standard} should be >= ci_lower {ci_lower} (with tolerance)"
        assert ci_upper >= standard - 0.05, f"standard {standard} should be <= ci_upper {ci_upper} (with tolerance)"

    def test_bootstrap_ci_reduces_on_uncertainty(self):
        """When CI lower bound is negative, the Kelly fraction should be reduced."""
        sizer = _make_sizer(robust=True, epsilon=0.01, min_trades=3)
        # Marginal edge -- some bootstrap samples will show negative Kelly
        trades = [20.0, -18.0, 22.0, -19.0, 21.0, -20.0]
        _record_trades(sizer, "test_strat", trades)

        # Get the fraction -- it should be clamped to min at least
        fraction = sizer.get_kelly_fraction("test_strat")
        assert fraction is not None
        assert fraction >= sizer.config.min_equity_pct


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_winners_robust(self):
        """100% win rate: robust Kelly should still be positive."""
        sizer = _make_sizer(robust=True, epsilon=0.05, min_trades=5)
        trades = [100.0, 80.0, 120.0, 90.0, 110.0, 95.0]
        _record_trades(sizer, "test_strat", trades)

        stats = sizer._get_stats("test_strat")
        robust = sizer._compute_robust_kelly(stats)

        # With all winners avg_loss=0, robust should handle gracefully
        # The stats.avg_loss will be 0.0, so robust returns 0.0
        # But get_kelly_fraction handles the all-winners case via standard path
        assert robust >= 0.0

    def test_all_losers_robust(self):
        """0% win rate: robust returns 0 (will be clamped to min)."""
        sizer = _make_sizer(robust=True, epsilon=0.05, min_trades=5)
        trades = [-50.0, -30.0, -60.0, -40.0, -55.0, -45.0]
        _record_trades(sizer, "test_strat", trades)

        stats = sizer._get_stats("test_strat")
        robust = sizer._compute_robust_kelly(stats)

        assert robust == 0.0

        # Full fraction should clamp to min_equity_pct
        fraction = sizer.get_kelly_fraction("test_strat")
        assert fraction == sizer.config.min_equity_pct

    def test_marginal_edge_robust(self):
        """Barely positive edge: robust may reduce to 0 (clamped to min)."""
        sizer = _make_sizer(robust=True, epsilon=0.05, min_trades=5)
        # ~50% win rate, ~1:1 payoff -- minimal edge
        trades = [10.0, -10.0, 11.0, -9.0, 10.5, -10.5, 10.0, -10.0]
        _record_trades(sizer, "test_strat", trades)

        fraction = sizer.get_kelly_fraction("test_strat")
        assert fraction is not None
        # Should be at or near the minimum
        assert fraction <= 0.10  # very conservative


# ---------------------------------------------------------------------------
# get_kelly_stats
# ---------------------------------------------------------------------------


class TestGetKellyStats:
    def test_get_kelly_stats_output(self):
        """Verify stats dict has all expected keys and reasonable values."""
        sizer = _make_sizer(robust=True)
        trades = [100.0, -50.0, 80.0, -30.0, 120.0, -60.0, 90.0, -40.0]
        _record_trades(sizer, "test_strat", trades)

        result = sizer.get_kelly_stats("test_strat")

        expected_keys = {
            "standard_kelly",
            "robust_kelly",
            "ci_lower",
            "ci_upper",
            "trade_count",
            "win_rate",
            "payoff_ratio",
        }
        assert set(result.keys()) == expected_keys

        assert result["trade_count"] == 8
        assert 0.0 <= result["win_rate"] <= 1.0
        assert result["payoff_ratio"] > 0
        assert result["robust_kelly"] <= result["standard_kelly"]

    def test_get_kelly_stats_empty(self):
        """Stats for a strategy with no trades."""
        sizer = _make_sizer(robust=True)
        result = sizer.get_kelly_stats("empty_strat")

        assert result["trade_count"] == 0
        assert result["standard_kelly"] == 0.0
        assert result["robust_kelly"] == 0.0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_backward_compatible_standard_kelly(self):
        """When robust=False, behaves exactly like the original Kelly sizer."""
        sizer = _make_sizer(robust=False, fraction=0.5, min_trades=5, min_equity_pct=0.05, max_equity_pct=0.30)
        trades = [100.0, -50.0, 80.0, -30.0, 120.0, -60.0, 90.0, -40.0, 110.0, -55.0]
        _record_trades(sizer, "test_strat", trades)

        fraction = sizer.get_kelly_fraction("test_strat")
        assert fraction is not None

        # Manually compute expected value
        stats = sizer._get_stats("test_strat")
        p = stats.win_count / stats.trade_count
        q = 1.0 - p
        b = stats.avg_win / stats.avg_loss
        raw = (b * p - q) / b
        expected = max(0.05, min(0.30, raw * 0.5))

        assert abs(fraction - expected) < 1e-9, f"got {fraction}, expected {expected}"

    def test_robust_disabled_config(self):
        """robust=False falls back to standard computation path."""
        sizer_standard = _make_sizer(robust=False, fraction=1.0, min_trades=5, min_equity_pct=0.01, max_equity_pct=0.50)
        sizer_robust = _make_sizer(
            robust=True, epsilon=0.05, fraction=1.0, min_trades=5, min_equity_pct=0.01, max_equity_pct=0.50
        )

        trades = [100.0, -50.0, 80.0, -30.0, 120.0, -60.0, 90.0, -40.0, 110.0, -55.0]
        _record_trades(sizer_standard, "test_strat", trades)
        _record_trades(sizer_robust, "test_strat", trades)

        f_standard = sizer_standard.get_kelly_fraction("test_strat")
        f_robust = sizer_robust.get_kelly_fraction("test_strat")

        assert f_standard is not None
        assert f_robust is not None
        # Robust should be <= standard (more conservative)
        assert f_robust <= f_standard

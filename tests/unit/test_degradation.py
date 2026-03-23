"""Unit tests for IS/OOS degradation distribution tracking."""

import math

import pytest

from src.analytics.optuna_harness import compute_degradation_distribution


@pytest.mark.unit
class TestComputeDegradationDistribution:
    """Tests for compute_degradation_distribution()."""

    def test_normal_degradation_no_warning(self):
        """Ratios around 0.7 should not trigger overfit warning."""
        windows = [
            {"is_score": 1.0, "oos_score": 0.7},
            {"is_score": 1.2, "oos_score": 0.9},
            {"is_score": 0.8, "oos_score": 0.6},
        ]
        result = compute_degradation_distribution(windows)

        assert len(result["ratios"]) == 3
        assert result["ratios"][0] == pytest.approx(0.7)
        assert result["ratios"][1] == pytest.approx(0.75)
        assert result["ratios"][2] == pytest.approx(0.75)
        assert result["mean"] == pytest.approx(sum(result["ratios"]) / 3)
        assert result["overfit_warning"] is False
        assert result["pct_above_50"] == pytest.approx(100.0)

    def test_heavy_overfitting_triggers_warning(self):
        """Ratios around 0.3 should trigger overfit warning (mean < 0.5)."""
        windows = [
            {"is_score": 1.0, "oos_score": 0.3},
            {"is_score": 1.5, "oos_score": 0.4},
            {"is_score": 2.0, "oos_score": 0.5},
        ]
        result = compute_degradation_distribution(windows)

        # mean ratio = (0.3 + 0.2667 + 0.25) / 3 ~ 0.272
        assert result["overfit_warning"] is True
        assert result["mean"] < 0.5

    def test_worst_ratio_triggers_warning(self):
        """Even if mean is ok, worst < 0.2 should trigger warning."""
        windows = [
            {"is_score": 1.0, "oos_score": 0.8},
            {"is_score": 1.0, "oos_score": 0.7},
            {"is_score": 1.0, "oos_score": 0.15},  # worst < 0.2
        ]
        result = compute_degradation_distribution(windows)

        assert result["worst"] == pytest.approx(0.15)
        assert result["overfit_warning"] is True

    def test_mixed_windows(self):
        """Mix of good and bad windows."""
        windows = [
            {"is_score": 1.0, "oos_score": 0.9},  # 0.9 — good
            {"is_score": 1.0, "oos_score": 0.4},  # 0.4 — bad
            {"is_score": 1.0, "oos_score": 0.6},  # 0.6 — ok
            {"is_score": 1.0, "oos_score": 0.3},  # 0.3 — bad
        ]
        result = compute_degradation_distribution(windows)

        assert len(result["ratios"]) == 4
        # pct_above_50: windows with ratio > 0.5 → 0.9, 0.6 = 2/4 = 50%
        assert result["pct_above_50"] == pytest.approx(50.0)
        assert result["worst"] == pytest.approx(0.3)
        # median of [0.3, 0.4, 0.6, 0.9] = (0.4 + 0.6) / 2 = 0.5
        assert result["median"] == pytest.approx(0.5)

    def test_is_score_zero_handled(self):
        """is_score == 0 should produce ratio of 0.0, not crash."""
        windows = [
            {"is_score": 0.0, "oos_score": 0.5},
            {"is_score": 1.0, "oos_score": 0.8},
        ]
        result = compute_degradation_distribution(windows)

        assert result["ratios"][0] == 0.0
        assert result["ratios"][1] == pytest.approx(0.8)
        assert math.isfinite(result["mean"])
        assert math.isfinite(result["std"])

    def test_single_window(self):
        """Single window should still return valid stats."""
        windows = [{"is_score": 1.0, "oos_score": 0.6}]
        result = compute_degradation_distribution(windows)

        assert len(result["ratios"]) == 1
        assert result["mean"] == pytest.approx(0.6)
        assert result["median"] == pytest.approx(0.6)
        assert result["std"] == pytest.approx(0.0)
        assert result["worst"] == pytest.approx(0.6)

    def test_empty_windows(self):
        """Empty input should return zeroed stats."""
        result = compute_degradation_distribution([])

        assert result["ratios"] == []
        assert result["mean"] == 0.0
        assert result["median"] == 0.0
        assert result["std"] == 0.0
        assert result["worst"] == 0.0
        assert result["pct_above_50"] == 0.0
        assert result["overfit_warning"] is False

    def test_return_keys(self):
        """Result dict should have all expected keys."""
        windows = [{"is_score": 1.0, "oos_score": 0.5}]
        result = compute_degradation_distribution(windows)

        expected_keys = {"ratios", "mean", "median", "std", "worst", "pct_above_50", "overfit_warning"}
        assert set(result.keys()) == expected_keys

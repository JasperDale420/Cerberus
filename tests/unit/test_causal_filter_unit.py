"""Unit tests for the Causal Inference Signal Filter."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from src.analysis.causal_filter import CausalScore, CausalSignalFilter


@pytest.mark.unit
class TestCausalSignalFilter:
    """Tests for CausalSignalFilter."""

    def test_initialization(self) -> None:
        """Verify default constructor values."""
        cf = CausalSignalFilter()
        assert cf.min_observations == 200
        assert cf.max_lag == 10
        assert cf.significance == 0.05
        assert cf.default_strength == 1.0
        assert cf.get_all_scores() == {}

    def test_unknown_strategy_returns_default(self) -> None:
        """Unknown strategies return default_strength."""
        cf = CausalSignalFilter(default_strength=0.8)
        assert cf.get_causal_strength("unknown") == 0.8

    def test_should_allow_permissive_default(self) -> None:
        """Unknown strategies pass should_allow by default (default_strength=1.0)."""
        cf = CausalSignalFilter(default_strength=1.0)
        assert cf.should_allow("never_seen", threshold=0.3) is True
        assert cf.should_allow("never_seen", threshold=0.99) is True

    def test_causal_detection_true_cause(self) -> None:
        """Signal with genuine predictive power should produce high causal strength."""
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        noise = np.random.randn(n) * 0.3
        # y_t = 0.5 * x_{t-1} + noise — clear lag-1 causal link
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * x[t - 1] + noise[t]

        pairs: List[Tuple[float, float]] = [(float(x[i]), float(y[i])) for i in range(n)]

        cf = CausalSignalFilter(min_observations=200, max_lag=5)
        scores = cf.update_scores({"trend_follow": pairs})

        assert "trend_follow" in scores
        score = scores["trend_follow"]
        assert score.causal_strength > 0.7, f"Expected >0.7, got {score.causal_strength}"
        assert score.granger_p_value < 0.05
        assert score.n_observations > 0

    def test_causal_detection_no_cause(self) -> None:
        """Independent signal and return should produce low causal strength."""
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        y = np.random.randn(n)  # completely independent

        pairs: List[Tuple[float, float]] = [(float(x[i]), float(y[i])) for i in range(n)]

        cf = CausalSignalFilter(min_observations=200, max_lag=5)
        scores = cf.update_scores({"noise_strat": pairs})

        assert "noise_strat" in scores
        score = scores["noise_strat"]
        assert score.causal_strength < 0.5, f"Expected <0.5, got {score.causal_strength}"

    def test_insufficient_data(self) -> None:
        """Fewer than min_observations should not produce a score; default returned."""
        np.random.seed(42)
        n = 50  # well below default min_observations=200
        pairs: List[Tuple[float, float]] = [(float(np.random.randn()), float(np.random.randn())) for _ in range(n)]

        cf = CausalSignalFilter(min_observations=200)
        scores = cf.update_scores({"small_strat": pairs})

        assert scores == {}
        # Should fall back to default
        assert cf.get_causal_strength("small_strat") == cf.default_strength

    def test_lag_selection(self) -> None:
        """Data with a known lag-2 relationship should select optimal_lag == 2."""
        np.random.seed(42)
        n = 600
        x = np.random.randn(n)
        noise = np.random.randn(n) * 0.2
        # y_t = 0.6 * x_{t-2} + noise — clear lag-2 causal link
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.6 * x[t - 2] + noise[t]

        pairs: List[Tuple[float, float]] = [(float(x[i]), float(y[i])) for i in range(n)]

        cf = CausalSignalFilter(min_observations=200, max_lag=5)
        scores = cf.update_scores({"lag2_strat": pairs})

        assert "lag2_strat" in scores
        score = scores["lag2_strat"]
        assert score.optimal_lag == 2, f"Expected lag 2, got {score.optimal_lag}"
        assert score.causal_strength > 0.7

    def test_get_all_scores(self) -> None:
        """Multiple strategies updated at once should all appear in get_all_scores."""
        np.random.seed(42)
        n = 300

        histories = {}
        for name in ["alpha", "beta", "gamma"]:
            x = np.random.randn(n)
            y = 0.4 * np.roll(x, 1) + np.random.randn(n) * 0.5
            histories[name] = [(float(x[i]), float(y[i])) for i in range(n)]

        cf = CausalSignalFilter(min_observations=200, max_lag=3)
        cf.update_scores(histories)

        all_scores = cf.get_all_scores()
        assert set(all_scores.keys()) == {"alpha", "beta", "gamma"}
        for name, score in all_scores.items():
            assert isinstance(score, CausalScore)
            assert score.strategy == name
            assert 0.0 <= score.causal_strength <= 1.0

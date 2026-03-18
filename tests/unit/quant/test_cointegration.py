"""Tests for cointegration testing and monitoring."""

import numpy as np

from src.quant.cointegration import (
    CointegrationResult,
    EngleGrangerTest,
    JohansenTest,
    RollingCointegrationMonitor,
)

# ---------------------------------------------------------------------------
# EngleGrangerTest
# ---------------------------------------------------------------------------


class TestEngleGrangerTest:
    def test_cointegrated_series_detected(self):
        """Two series sharing a stochastic trend should be cointegrated."""
        rng = np.random.default_rng(42)
        n = 500
        # Shared random walk
        common = np.cumsum(rng.standard_normal(n))
        x = common + rng.standard_normal(n) * 0.3
        y = 2.0 * common + 5.0 + rng.standard_normal(n) * 0.3

        result = EngleGrangerTest.test(x, y)

        assert isinstance(result, CointegrationResult)
        assert result.is_cointegrated is True
        assert result.p_value < 0.05
        assert result.hedge_ratio > 0
        assert result.half_life is not None
        assert result.half_life > 0

    def test_independent_series_not_cointegrated(self):
        """Two independent random walks should not be cointegrated."""
        rng = np.random.default_rng(123)
        n = 500
        x = np.cumsum(rng.standard_normal(n))
        y = np.cumsum(rng.standard_normal(n))

        result = EngleGrangerTest.test(x, y)

        assert isinstance(result, CointegrationResult)
        assert result.is_cointegrated is False
        assert result.p_value > 0.05

    def test_hedge_ratio_approximates_true_ratio(self):
        """OLS hedge ratio should approximate the true coefficient."""
        rng = np.random.default_rng(99)
        n = 1000
        common = np.cumsum(rng.standard_normal(n))
        x = common + rng.standard_normal(n) * 0.1
        y = 3.0 * common + 10.0 + rng.standard_normal(n) * 0.1

        result = EngleGrangerTest.test(x, y)

        assert abs(result.hedge_ratio - 3.0) < 0.5
        assert abs(result.intercept - 10.0) < 5.0


# ---------------------------------------------------------------------------
# JohansenTest
# ---------------------------------------------------------------------------


class TestJohansenTest:
    def test_cointegrated_system(self):
        """Johansen test on a cointegrated system should find at least one vector."""
        rng = np.random.default_rng(42)
        n = 500
        common = np.cumsum(rng.standard_normal(n))
        s1 = common + rng.standard_normal(n) * 0.2
        s2 = 2.0 * common + rng.standard_normal(n) * 0.2
        s3 = 0.5 * common + rng.standard_normal(n) * 0.2

        matrix = np.column_stack([s1, s2, s3])
        result = JohansenTest.test(matrix)

        assert isinstance(result, dict)
        assert "n_cointegrating_vectors" in result
        assert "is_cointegrated" in result
        assert result["is_cointegrated"] is True
        assert result["n_cointegrating_vectors"] >= 1
        assert result["trace_statistics"].shape[0] == 3

    def test_independent_series(self):
        """Independent random walks — likely no cointegrating vectors."""
        rng = np.random.default_rng(77)
        n = 500
        s1 = np.cumsum(rng.standard_normal(n))
        s2 = np.cumsum(rng.standard_normal(n))
        s3 = np.cumsum(rng.standard_normal(n))

        matrix = np.column_stack([s1, s2, s3])
        result = JohansenTest.test(matrix)

        assert isinstance(result, dict)
        assert result["n_cointegrating_vectors"] == 0
        assert result["is_cointegrated"] is False


# ---------------------------------------------------------------------------
# RollingCointegrationMonitor
# ---------------------------------------------------------------------------


class TestRollingCointegrationMonitor:
    def test_returns_none_before_lookback(self):
        """Should return None until lookback window is filled."""
        monitor = RollingCointegrationMonitor(lookback=50, retest_interval=10)
        for i in range(49):
            result = monitor.update(float(i), float(i * 2))
        assert result is None

    def test_is_valid_before_first_test(self):
        """Assumes valid until first test fires."""
        monitor = RollingCointegrationMonitor(lookback=50, retest_interval=10)
        assert monitor.is_valid is True

    def test_detects_cointegration(self):
        """Should detect cointegration in a cointegrated pair."""
        rng = np.random.default_rng(42)
        n = 200
        common = np.cumsum(rng.standard_normal(n))
        x_vals = common + rng.standard_normal(n) * 0.1
        y_vals = 2.0 * common + 5.0 + rng.standard_normal(n) * 0.1

        monitor = RollingCointegrationMonitor(lookback=100, retest_interval=20)
        last_result = None
        for i in range(n):
            r = monitor.update(float(x_vals[i]), float(y_vals[i]))
            if r is not None:
                last_result = r

        assert last_result is not None
        assert last_result.is_cointegrated is True
        assert monitor.is_valid is True

    def test_detects_breakdown(self):
        """Should detect when a previously cointegrated pair diverges."""
        rng = np.random.default_rng(42)

        # Phase 1: cointegrated (200 bars)
        common1 = np.cumsum(rng.standard_normal(200))
        x1 = common1 + rng.standard_normal(200) * 0.1
        y1 = 2.0 * common1 + 5.0 + rng.standard_normal(200) * 0.1

        # Phase 2: diverge (200 bars — independent walks)
        x2 = x1[-1] + np.cumsum(rng.standard_normal(200))
        y2 = y1[-1] + np.cumsum(rng.standard_normal(200) * 3.0)

        x_all = np.concatenate([x1, x2])
        y_all = np.concatenate([y1, y2])

        monitor = RollingCointegrationMonitor(lookback=100, retest_interval=20)
        for i in range(len(x_all)):
            monitor.update(float(x_all[i]), float(y_all[i]))

        # After the divergence phase, cointegration should break down
        assert monitor.is_valid is False

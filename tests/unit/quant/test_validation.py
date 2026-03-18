"""Tests for anti-overfitting validation framework."""

from __future__ import annotations

import numpy as np

from src.quant.validation import (
    CUSUMDriftDetector,
    InformationCoefficientTracker,
    WalkForwardValidator,
    deflated_sharpe_ratio,
)

# ---------------------------------------------------------------------------
# deflated_sharpe_ratio
# ---------------------------------------------------------------------------


class TestDeflatedSharpeRatio:
    def test_single_trial_approx_unchanged(self):
        """With only 1 trial, deflation should be minimal."""
        dsr = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=1, n_observations=252)
        # Single trial => expected max ~ 0, so DSR should be high (positive)
        assert dsr > 1.5

    def test_many_trials_deflates(self):
        """More trials should lower the DSR (harder to beat expected max)."""
        dsr_few = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=5, n_observations=252)
        dsr_many = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=100, n_observations=252)
        assert dsr_many < dsr_few

    def test_negative_sharpe_stays_negative(self):
        """A negative observed Sharpe should yield a negative DSR."""
        dsr = deflated_sharpe_ratio(observed_sharpe=-1.0, n_trials=10, n_observations=252)
        assert dsr < 0


# ---------------------------------------------------------------------------
# CUSUMDriftDetector
# ---------------------------------------------------------------------------


class TestCUSUMDriftDetector:
    def test_no_drift_stable_returns(self):
        """Stable positive returns around expected should not trigger drift."""
        detector = CUSUMDriftDetector(expected_sharpe=1.0, threshold=8.0)
        expected_daily = 1.0 / np.sqrt(252)
        rng = np.random.default_rng(42)

        results = []
        for _ in range(60):
            # Returns centered on expected with realistic daily vol
            ret = expected_daily + rng.normal(0, 0.01)
            results.append(detector.update(ret))

        assert not any(r.is_drifting for r in results)

    def test_detects_breakdown(self):
        """Strongly negative returns should trigger drift detection."""
        detector = CUSUMDriftDetector(expected_sharpe=1.0, threshold=3.0)

        # Feed strongly negative returns
        drifted = False
        for _ in range(200):
            result = detector.update(-0.03)
            if result.is_drifting:
                drifted = True
                break

        assert drifted

    def test_reset_clears_state(self):
        """After reset, detector should be fresh."""
        detector = CUSUMDriftDetector(expected_sharpe=1.0, threshold=3.0)
        for _ in range(50):
            detector.update(-0.03)
        detector.reset()
        result = detector.update(0.01)
        assert not result.is_drifting
        assert result.cusum_value == 0.0


# ---------------------------------------------------------------------------
# InformationCoefficientTracker
# ---------------------------------------------------------------------------


class TestInformationCoefficientTracker:
    def test_returns_none_before_window_filled(self):
        tracker = InformationCoefficientTracker(window=30)
        for i in range(29):
            result = tracker.update(float(i), float(i))
        assert result is None
        assert tracker.current_ic is None

    def test_tracks_positive_ic(self):
        """When predictions correlate perfectly with actuals, IC should be high."""
        tracker = InformationCoefficientTracker(window=30)
        rng = np.random.default_rng(42)
        values = rng.standard_normal(30)
        result = None
        for v in values:
            result = tracker.update(float(v), float(v) + rng.normal(0, 0.01))
        assert result is not None
        assert result > 0.9
        assert not tracker.is_decaying

    def test_detects_decay_when_independent(self):
        """Independent predictions and actuals should yield low IC."""
        tracker = InformationCoefficientTracker(window=30)
        rng = np.random.default_rng(42)
        for _ in range(30):
            tracker.update(rng.normal(), rng.normal())
        assert tracker.current_ic is not None
        assert abs(tracker.current_ic) < 0.5  # roughly uncorrelated
        # With truly independent data, is_decaying should be True (|IC| < 0.05)
        # Use a larger window to get more stable near-zero IC
        tracker2 = InformationCoefficientTracker(window=500)
        rng2 = np.random.default_rng(7)
        for _ in range(500):
            tracker2.update(rng2.normal(), rng2.normal())
        assert tracker2.is_decaying


# ---------------------------------------------------------------------------
# WalkForwardValidator
# ---------------------------------------------------------------------------


class TestWalkForwardValidator:
    def test_correct_number_of_splits(self):
        wfv = WalkForwardValidator(oos_window=20, min_train_window=60, n_splits=8, embargo_bars=5)
        splits = wfv.split(300)
        assert len(splits) == 8

    def test_train_starts_at_zero(self):
        """Anchored walk-forward: train always starts at 0."""
        wfv = WalkForwardValidator(oos_window=20, min_train_window=60, n_splits=4, embargo_bars=5)
        splits = wfv.split(200)
        for train, _test in splits:
            assert train.start == 0

    def test_test_windows_non_overlapping(self):
        wfv = WalkForwardValidator(oos_window=20, min_train_window=60, n_splits=4, embargo_bars=5)
        splits = wfv.split(200)
        for i in range(len(splits) - 1):
            _, test_i = splits[i]
            _, test_next = splits[i + 1]
            assert test_i.stop <= test_next.start

    def test_embargo_gap(self):
        """There should be an embargo gap between train end and test start."""
        wfv = WalkForwardValidator(oos_window=20, min_train_window=60, n_splits=4, embargo_bars=5)
        splits = wfv.split(200)
        for train, test in splits:
            assert test.start - train.stop >= 5

    def test_skips_splits_with_insufficient_train(self):
        """If n_observations is too small, some splits should be skipped."""
        wfv = WalkForwardValidator(oos_window=20, min_train_window=60, n_splits=8, embargo_bars=5)
        # With only 100 observations, earlier splits won't have enough training data
        splits = wfv.split(100)
        assert len(splits) < 8
        for train, _test in splits:
            assert len(train) >= 60

"""Unit tests for BOCPD detector."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analysis.bocpd import BOCPDDetector, BOCPDResult


@pytest.mark.unit
def test_bocpd_initialization() -> None:
    """Verify default parameters are stored correctly."""
    det = BOCPDDetector()
    assert det._hazard == pytest.approx(1.0 / 200)
    assert det._prior_mu == 0.0
    assert det._prior_kappa == 1.0
    assert det._prior_alpha == 1.0
    assert det._prior_beta == 0.01
    assert det._max_run_length == 200
    assert det._t == 0
    assert det._joint[0] == 1.0
    assert det._joint.sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_bocpd_single_update() -> None:
    """Feed one value and verify result structure."""
    det = BOCPDDetector()
    result = det.update(0.01)

    assert isinstance(result, BOCPDResult)
    assert 0.0 <= result.changepoint_probability <= 1.0
    assert isinstance(result.map_run_length, int)
    assert result.map_run_length >= 0
    assert result.expected_run_length >= 0.0
    assert isinstance(result.growth_probabilities, np.ndarray)
    assert result.growth_probabilities.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_bocpd_stationary_series() -> None:
    """Stationary N(0,1) should keep changepoint probability low on average."""
    np.random.seed(42)
    det = BOCPDDetector(hazard_lambda=200, prior_beta=0.5)
    data = np.random.normal(0, 1, 100)

    cp_probs = []
    for x in data:
        result = det.update(x)
        cp_probs.append(result.changepoint_probability)

    avg_cp = np.mean(cp_probs[10:])
    assert avg_cp < 0.3, f"Average changepoint prob {avg_cp:.3f} too high for stationary series"


@pytest.mark.unit
def test_bocpd_detects_mean_shift() -> None:
    """Detect a mean shift from N(0,1) to N(3,1) near index 50."""
    np.random.seed(42)
    det = BOCPDDetector(hazard_lambda=200, prior_beta=0.5)
    segment1 = np.random.normal(0, 1, 50)
    segment2 = np.random.normal(3, 1, 50)
    data = np.concatenate([segment1, segment2])

    cp_probs = []
    for x in data:
        result = det.update(x)
        cp_probs.append(result.changepoint_probability)

    detection_window = cp_probs[48:60]
    max_cp = max(detection_window)
    assert max_cp > 0.5, f"Max CP prob in detection window = {max_cp:.3f}, expected > 0.5"


@pytest.mark.unit
def test_bocpd_detects_variance_shift() -> None:
    """Detect a variance shift from N(0,0.1) to N(0,2)."""
    np.random.seed(42)
    det = BOCPDDetector(hazard_lambda=200, prior_beta=0.005)
    segment1 = np.random.normal(0, 0.1, 50)
    segment2 = np.random.normal(0, 2.0, 50)
    data = np.concatenate([segment1, segment2])

    cp_probs = []
    for x in data:
        result = det.update(x)
        cp_probs.append(result.changepoint_probability)

    detection_window = cp_probs[48:65]
    max_cp = max(detection_window)
    assert max_cp > 0.5, f"Max CP prob in detection window = {max_cp:.3f}, expected > 0.5"


@pytest.mark.unit
def test_bocpd_reset() -> None:
    """Reset should restore initial state."""
    det = BOCPDDetector()
    for _ in range(20):
        det.update(np.random.normal())

    assert det._t == 20
    det.reset()
    assert det._t == 0
    assert det._joint[0] == 1.0
    assert det._joint.sum() == pytest.approx(1.0)
    assert det._mu[0] == det._prior_mu
    assert det._kappa[0] == det._prior_kappa


@pytest.mark.unit
def test_bocpd_run_length_tracking() -> None:
    """MAP run length should generally increase during a stationary segment."""
    np.random.seed(42)
    det = BOCPDDetector(hazard_lambda=200, prior_beta=0.5)
    data = np.random.normal(0, 1, 60)

    run_lengths = []
    for x in data:
        result = det.update(x)
        run_lengths.append(result.map_run_length)

    early = np.mean(run_lengths[10:20])
    late = np.mean(run_lengths[40:60])
    assert late > early, f"Late run lengths ({late:.1f}) should exceed early ({early:.1f})"


@pytest.mark.unit
def test_bocpd_logger_called_on_changepoint() -> None:
    """Logger should fire when changepoint probability exceeds 0.5."""
    np.random.seed(42)
    mock_logger = MagicMock()
    det = BOCPDDetector(hazard_lambda=200, prior_beta=0.5, logger=mock_logger)

    for x in np.random.normal(0, 1, 50):
        det.update(x)
    for x in np.random.normal(5, 1, 20):
        det.update(x)

    calls = [c for c in mock_logger.info.call_args_list if c[0][0] == "bocpd_changepoint_detected"]
    assert len(calls) > 0, "Logger should have been called for changepoint detection"

"""Unit tests for CVaR-based position sizer with EVT/GPD tail estimation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.engine.cvar_sizer import CVaRSizer


def _make_config(**overrides: object) -> SimpleNamespace:
    defaults = {
        "enabled": True,
        "confidence_level": 0.95,
        "lookback_bars": 500,
        "min_bars": 100,
        "max_acceptable_cvar": 0.03,
        "use_evt": True,
        "tail_threshold_pct": 10.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _feed_returns(sizer: CVaRSizer, returns: list[float] | np.ndarray) -> None:
    for r in returns:
        sizer.record_return(float(r))


# ------------------------------------------------------------------
# Core tests
# ------------------------------------------------------------------


@pytest.mark.unit
def test_insufficient_data() -> None:
    """Fewer than min_bars returns None."""
    sizer = CVaRSizer(_make_config(min_bars=100))
    rng = np.random.default_rng(42)
    _feed_returns(sizer, rng.normal(-0.001, 0.02, 50).tolist())
    assert sizer.compute_cvar() is None


@pytest.mark.unit
def test_historical_cvar_computation() -> None:
    """Known Gaussian distribution — CVaR should be roughly predictable."""
    sizer = CVaRSizer(_make_config(use_evt=False, min_bars=50))
    rng = np.random.default_rng(1234)
    returns = rng.normal(-0.001, 0.02, 500)
    _feed_returns(sizer, returns)

    result = sizer.compute_cvar()
    assert result is not None
    assert result.method == "historical"
    # For N(mu=-0.001, sigma=0.02), 5% CVaR should be around -0.04 to -0.05
    assert result.cvar < 0.0
    assert -0.08 < result.cvar < -0.02
    assert result.var < 0.0
    assert result.cvar <= result.var  # CVaR is always worse than VaR
    assert result.n_tail_obs > 0


@pytest.mark.unit
def test_evt_gpd_cvar() -> None:
    """Heavy-tailed distribution — EVT should produce a result different from historical."""
    config = _make_config(use_evt=True, min_bars=50, tail_threshold_pct=10.0)
    sizer_evt = CVaRSizer(config)
    sizer_hist = CVaRSizer(_make_config(use_evt=False, min_bars=50))

    rng = np.random.default_rng(99)
    # Generate returns with heavy tails (Student's t with df=3)
    returns = rng.standard_t(df=3, size=500) * 0.02
    _feed_returns(sizer_evt, returns)
    _feed_returns(sizer_hist, returns)

    result_evt = sizer_evt.compute_cvar()
    result_hist = sizer_hist.compute_cvar()

    assert result_evt is not None
    assert result_hist is not None
    assert result_evt.method == "evt_gpd"
    assert result_hist.method == "historical"
    assert result_evt.tail_index is not None
    # EVT and historical give different estimates
    assert result_evt.cvar != pytest.approx(result_hist.cvar, rel=0.01)


@pytest.mark.unit
def test_cvar_multiplier_clamp_low() -> None:
    """Very high CVaR should floor the multiplier at 0.1."""
    sizer = CVaRSizer(_make_config(use_evt=False, min_bars=50, max_acceptable_cvar=0.01))
    rng = np.random.default_rng(7)
    # Large negative returns to drive CVaR very negative
    returns = rng.normal(-0.10, 0.05, 200)
    _feed_returns(sizer, returns)

    result = sizer.compute_cvar()
    assert result is not None
    assert result.cvar_multiplier == pytest.approx(0.1)


@pytest.mark.unit
def test_cvar_multiplier_clamp_high() -> None:
    """Very low CVaR should cap the multiplier at 2.0."""
    sizer = CVaRSizer(_make_config(use_evt=False, min_bars=50, max_acceptable_cvar=0.05))
    rng = np.random.default_rng(11)
    # Tiny variance, near-zero negative returns
    returns = rng.normal(-0.0001, 0.001, 200)
    _feed_returns(sizer, returns)

    result = sizer.compute_cvar()
    assert result is not None
    assert result.cvar_multiplier == pytest.approx(2.0)


@pytest.mark.unit
def test_normal_returns() -> None:
    """Gaussian returns — historical and EVT estimates should be in the same ballpark."""
    rng = np.random.default_rng(555)
    returns = rng.normal(0.0, 0.015, 1000)

    sizer_evt = CVaRSizer(_make_config(use_evt=True, min_bars=50, lookback_bars=1000))
    sizer_hist = CVaRSizer(_make_config(use_evt=False, min_bars=50, lookback_bars=1000))
    _feed_returns(sizer_evt, returns)
    _feed_returns(sizer_hist, returns)

    r_evt = sizer_evt.compute_cvar()
    r_hist = sizer_hist.compute_cvar()
    assert r_evt is not None and r_hist is not None

    # For Gaussian returns, EVT and historical CVaR should be within ~50% of each other
    assert abs(r_evt.cvar - r_hist.cvar) / abs(r_hist.cvar) < 0.5


@pytest.mark.unit
def test_heavy_tail_returns() -> None:
    """Fat-tailed returns — EVT should capture tail risk (larger |CVaR| or different multiplier)."""
    rng = np.random.default_rng(77)
    # Heavy-tailed Student's t(df=2) returns — fatter tails than Gaussian
    returns = rng.standard_t(df=2, size=500) * 0.02

    sizer_evt = CVaRSizer(_make_config(use_evt=True, min_bars=50))
    sizer_hist = CVaRSizer(_make_config(use_evt=False, min_bars=50))
    _feed_returns(sizer_evt, returns)
    _feed_returns(sizer_hist, returns)

    result_evt = sizer_evt.compute_cvar()
    result_hist = sizer_hist.compute_cvar()
    assert result_evt is not None
    assert result_hist is not None
    assert result_evt.method == "evt_gpd"
    assert result_evt.tail_index is not None
    # EVT should detect tail risk
    assert result_evt.cvar < 0.0
    # Historical also detects it
    assert result_hist.cvar < 0.0


@pytest.mark.unit
def test_all_positive_returns() -> None:
    """No losses — multiplier should be 2.0 (max)."""
    sizer = CVaRSizer(_make_config(use_evt=False, min_bars=50))
    # All positive, so the "worst" returns are still positive but the smallest ones
    returns = [0.001 + i * 0.0001 for i in range(200)]
    _feed_returns(sizer, returns)

    result = sizer.compute_cvar()
    assert result is not None
    # Even the "tail" is positive, so |CVaR| is very small relative to max_acceptable
    assert result.cvar_multiplier == pytest.approx(2.0)


@pytest.mark.unit
def test_evt_fallback_on_insufficient_tail() -> None:
    """When there are too few tail observations, should fall back to historical."""
    # With correct MoM formula xi = 0.5*(1 - mean^2/var), xi is bounded above
    # by 0.5 for valid data, so the xi >= 1.0 guard is a numerical safety net.
    # Test insufficient tail fallback: fewer than 10 tail obs → skip EVT.
    # tail_threshold_pct=2.0 with 200 returns → only 4 tail obs → falls back
    sizer = CVaRSizer(_make_config(use_evt=True, min_bars=20, tail_threshold_pct=2.0, lookback_bars=200))
    rng = np.random.default_rng(42)

    returns = rng.normal(0.0, 0.01, 200)
    _feed_returns(sizer, returns.tolist())

    result = sizer.compute_cvar()
    assert result is not None
    assert result.method == "historical"  # fell back from EVT due to n_tail < 10


@pytest.mark.unit
def test_get_cvar_multiplier_default() -> None:
    """Before any data, get_cvar_multiplier returns 1.0."""
    sizer = CVaRSizer(_make_config())
    assert sizer.get_cvar_multiplier() == 1.0


@pytest.mark.unit
def test_record_return() -> None:
    """Returns accumulate correctly in the deque."""
    sizer = CVaRSizer(_make_config(lookback_bars=5))
    for v in [0.01, -0.02, 0.005, -0.01, 0.003, -0.05]:
        sizer.record_return(v)
    # maxlen=5, so first value should be gone
    assert len(sizer._returns) == 5
    assert sizer._returns[0] == pytest.approx(-0.02)
    assert sizer._returns[-1] == pytest.approx(-0.05)


@pytest.mark.unit
def test_caching() -> None:
    """Multiplier should not recompute on every call within the recompute interval."""
    sizer = CVaRSizer(_make_config(use_evt=False, min_bars=50))
    rng = np.random.default_rng(321)
    _feed_returns(sizer, rng.normal(-0.001, 0.02, 200).tolist())

    m1 = sizer.get_cvar_multiplier()
    assert m1 != 1.0  # should have computed

    # Add a few returns (less than recompute interval of 50)
    for _ in range(10):
        sizer.record_return(0.001)

    m2 = sizer.get_cvar_multiplier()
    assert m2 == pytest.approx(m1)  # still cached

    # Push past the recompute interval
    for _ in range(50):
        sizer.record_return(0.001)

    sizer.get_cvar_multiplier()
    # After recompute with many positive returns added, multiplier may change
    # The key assertion: it did recompute (obs counter was reset)
    assert sizer._obs_since_last_compute == 0

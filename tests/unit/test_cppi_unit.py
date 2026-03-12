"""Unit tests for the CPPI drawdown-controlled sizing module."""

from __future__ import annotations

import pytest

from src.engine.cppi import CPPIConfig, CPPISizer


@pytest.mark.unit
def test_cppi_initialization() -> None:
    """Initial state should set floor at equity*(1-max_drawdown), cushion > 0, exposure = 1.0."""
    config = CPPIConfig(enabled=True, multiplier=3.0, max_drawdown=0.05)
    sizer = CPPISizer(config, initial_equity=100_000.0)
    state = sizer.get_state()

    assert state.equity == 100_000.0
    assert state.floor == pytest.approx(95_000.0)
    assert state.cushion == pytest.approx(5_000.0)
    assert state.high_water_mark == 100_000.0
    assert state.drawdown_from_hwm == 0.0
    # multiplier=3, cushion=5000, equity=100000 => raw=0.15, but initial should be clamped to max=1.0
    # Actually: 3 * 5000 / 100000 = 0.15, so exposure = 0.15
    assert state.exposure_fraction == pytest.approx(0.15)


@pytest.mark.unit
def test_cppi_stable_equity() -> None:
    """When equity stays constant at 100k, exposure should remain stable."""
    config = CPPIConfig(enabled=True, multiplier=3.0, max_drawdown=0.05)
    sizer = CPPISizer(config, initial_equity=100_000.0)

    for _ in range(10):
        state = sizer.update_equity(100_000.0)

    assert state.exposure_fraction == pytest.approx(0.15)
    assert state.equity == 100_000.0
    assert state.drawdown_from_hwm == 0.0


@pytest.mark.unit
def test_cppi_gradual_drawdown() -> None:
    """Equity drops 1% at a time; exposure should decrease gradually, not binary."""
    config = CPPIConfig(enabled=True, multiplier=3.0, max_drawdown=0.05, min_exposure=0.1)
    sizer = CPPISizer(config, initial_equity=100_000.0)

    exposures = []
    equity = 100_000.0
    for _ in range(5):
        equity *= 0.99  # 1% drop each step
        state = sizer.update_equity(equity)
        exposures.append(state.exposure_fraction)

    # Exposure should decrease monotonically as equity drops
    for i in range(1, len(exposures)):
        assert exposures[i] <= exposures[i - 1], f"Exposure should decrease: {exposures}"

    # Should not be binary — intermediate values between min and max
    assert exposures[0] > config.min_exposure, "First small drop should not hit minimum"
    assert exposures[-1] < exposures[0], "Later drops should reduce exposure further"


@pytest.mark.unit
def test_cppi_reaches_floor() -> None:
    """When equity drops to floor level, exposure should equal min_exposure."""
    config = CPPIConfig(enabled=True, multiplier=3.0, max_drawdown=0.05, min_exposure=0.1)
    sizer = CPPISizer(config, initial_equity=100_000.0)

    # Drop equity to exactly the floor (95000)
    state = sizer.update_equity(95_000.0)
    assert state.exposure_fraction == config.min_exposure
    assert state.cushion == pytest.approx(0.0, abs=1.0)


@pytest.mark.unit
def test_cppi_recovery() -> None:
    """Equity drops then recovers; exposure should increase back."""
    config = CPPIConfig(enabled=True, multiplier=3.0, max_drawdown=0.05)
    sizer = CPPISizer(config, initial_equity=100_000.0)

    # Drop equity
    state_low = sizer.update_equity(96_000.0)

    # Recover equity
    state_high = sizer.update_equity(100_000.0)

    assert state_high.exposure_fraction > state_low.exposure_fraction


@pytest.mark.unit
def test_cppi_floor_ratchet() -> None:
    """Equity rises to 110k then drops to 105k; floor should have ratcheted up."""
    config = CPPIConfig(enabled=True, multiplier=3.0, max_drawdown=0.05)
    sizer = CPPISizer(config, initial_equity=100_000.0)

    # Equity rises — floor should ratchet up
    sizer.update_equity(110_000.0)
    state_at_110k = sizer.get_state()
    expected_floor_110k = 110_000.0 * (1 - 0.05)  # 104,500
    assert state_at_110k.floor == pytest.approx(expected_floor_110k)

    # Equity drops to 105k — floor should NOT drop (ratchet effect)
    state_at_105k = sizer.update_equity(105_000.0)
    assert state_at_105k.floor == pytest.approx(expected_floor_110k)
    assert state_at_105k.floor > 105_000.0 * (1 - 0.05)  # higher than naive floor


@pytest.mark.unit
def test_cppi_min_exposure_clamp() -> None:
    """Even at maximum drawdown, exposure should never go below min_exposure."""
    config = CPPIConfig(enabled=True, multiplier=3.0, max_drawdown=0.05, min_exposure=0.1)
    sizer = CPPISizer(config, initial_equity=100_000.0)

    # Drop well below floor
    state = sizer.update_equity(80_000.0)
    assert state.exposure_fraction == config.min_exposure
    assert state.exposure_fraction >= 0.1


@pytest.mark.unit
def test_cppi_max_exposure_clamp() -> None:
    """Exposure should never exceed max_exposure even with large cushion."""
    config = CPPIConfig(enabled=True, multiplier=5.0, max_drawdown=0.50, max_exposure=0.8)
    sizer = CPPISizer(config, initial_equity=100_000.0)

    # With 50% max_drawdown and multiplier=5: raw = 5 * 50000/100000 = 2.5
    state = sizer.update_equity(100_000.0)
    assert state.exposure_fraction == config.max_exposure
    assert state.exposure_fraction <= 0.8


@pytest.mark.unit
def test_cppi_reset() -> None:
    """Reset should reinitialize all state to new equity baseline."""
    config = CPPIConfig(enabled=True, multiplier=3.0, max_drawdown=0.05)
    sizer = CPPISizer(config, initial_equity=100_000.0)

    # Cause some drawdown
    sizer.update_equity(90_000.0)

    # Reset to new equity
    sizer.reset(120_000.0)
    state = sizer.get_state()

    assert state.equity == 120_000.0
    assert state.high_water_mark == 120_000.0
    assert state.floor == pytest.approx(120_000.0 * 0.95)
    assert state.drawdown_from_hwm == 0.0


@pytest.mark.unit
def test_cppi_hwm_tracking() -> None:
    """High water mark should track the highest equity seen and never decrease."""
    config = CPPIConfig(enabled=True, multiplier=3.0, max_drawdown=0.05)
    sizer = CPPISizer(config, initial_equity=100_000.0)

    sizer.update_equity(105_000.0)
    assert sizer.get_state().high_water_mark == 105_000.0

    sizer.update_equity(110_000.0)
    assert sizer.get_state().high_water_mark == 110_000.0

    # HWM should not decrease when equity drops
    sizer.update_equity(108_000.0)
    assert sizer.get_state().high_water_mark == 110_000.0


@pytest.mark.unit
def test_cppi_drawdown_percentage() -> None:
    """Drawdown from HWM should be calculated correctly."""
    config = CPPIConfig(enabled=True, multiplier=3.0, max_drawdown=0.05)
    sizer = CPPISizer(config, initial_equity=100_000.0)

    sizer.update_equity(110_000.0)  # new HWM
    state = sizer.update_equity(104_500.0)  # 5% drawdown from 110k

    expected_dd = (110_000.0 - 104_500.0) / 110_000.0
    assert state.drawdown_from_hwm == pytest.approx(expected_dd)
    assert state.drawdown_from_hwm == pytest.approx(0.05, abs=0.001)


@pytest.mark.unit
def test_cppi_zero_equity() -> None:
    """Zero equity edge case should not crash and should return min_exposure."""
    config = CPPIConfig(enabled=True, multiplier=3.0, max_drawdown=0.05, min_exposure=0.1)
    sizer = CPPISizer(config, initial_equity=100_000.0)

    state = sizer.update_equity(0.0)
    assert state.exposure_fraction == config.min_exposure
    assert state.cushion == 0.0

    # Negative equity should also not crash
    state = sizer.update_equity(-5_000.0)
    assert state.exposure_fraction == config.min_exposure

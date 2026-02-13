import math

from src.data.calculator import FeatureCalculator


def test_frac_diff_weights():
    calc = FeatureCalculator()
    # d=1 should be weights [1, -1, 0, 0...]
    weights = calc.get_frac_diff_weights(1.0, 5)
    assert weights[0] == 1.0
    assert weights[1] == -1.0
    assert weights[2] == 0.0


def test_apply_frac_diff_d1():
    calc = FeatureCalculator()
    series = [10.0, 11.0, 13.0, 16.0]
    # d=1 is just standard first difference: 16 - 13 = 3
    res = calc.apply_frac_diff(series, d=1.0)
    assert res == 3.0


def test_hurst_exponent_trending():
    calc = FeatureCalculator()
    # Perfect trend: 1.01^t
    series = [100.0 * (1.01**i) for i in range(100)]
    h = calc.calculate_hurst_exponent(series)
    # Trending should be > 0.5
    assert h > 0.5


def test_hurst_exponent_mr():
    calc = FeatureCalculator()
    # Mean reverting: Sine wave + small noise
    series = [100.0 + 5.0 * math.sin(i * 0.5) for i in range(100)]
    h = calc.calculate_hurst_exponent(series)
    # Mean reverting should be < 0.5
    assert h < 0.5


def test_hurst_exponent_short_series():
    calc = FeatureCalculator()
    series = [100.0, 101.0, 102.0]
    h = calc.calculate_hurst_exponent(series)
    # Should return default 0.5 for short series
    assert h == 0.5


def test_hurst_exponent_handles_non_positive_prices():
    calc = FeatureCalculator()
    series = [0.0 for _ in range(100)]
    h = calc.calculate_hurst_exponent(series)
    assert h == 0.5


def test_relative_strength_handles_non_positive_baseline():
    calc = FeatureCalculator()
    assert calc.calculate_relative_strength([0.0, 101.0], [100.0, 101.0]) == 0.0
    assert calc.calculate_relative_strength([100.0, 101.0], [0.0, 101.0]) == 0.0

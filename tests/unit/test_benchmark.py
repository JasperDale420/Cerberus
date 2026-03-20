import numpy as np
import pytest


@pytest.mark.unit
def test_benchmark_alpha_positive():
    from src.analytics.benchmark import compute_benchmark_comparison

    strategy_daily = np.array([0.001] * 200)
    benchmark_daily = np.array([0.0005] * 200)
    result = compute_benchmark_comparison(strategy_daily, benchmark_daily, "SPY")
    assert result.strategy_alpha > 0
    assert result.benchmark_symbol == "SPY"


@pytest.mark.unit
def test_benchmark_beta_near_zero_for_uncorrelated():
    from src.analytics.benchmark import compute_benchmark_comparison

    rng = np.random.default_rng(42)
    strategy_daily = rng.normal(0.001, 0.01, 200)
    benchmark_daily = rng.normal(0.0005, 0.01, 200)
    result = compute_benchmark_comparison(strategy_daily, benchmark_daily, "SPY")
    assert abs(result.strategy_beta) < 1.0


@pytest.mark.unit
def test_capture_ratios():
    from src.analytics.benchmark import compute_benchmark_comparison

    benchmark_daily = np.array([0.01, -0.01, 0.02, -0.005, 0.015])
    strategy_daily = np.array([0.01, 0.0, 0.02, 0.0, 0.015])
    result = compute_benchmark_comparison(strategy_daily, benchmark_daily, "SPY")
    assert result.up_capture > 0.9
    assert result.down_capture < 0.1


@pytest.mark.unit
def test_benchmark_return_percentages():
    from src.analytics.benchmark import compute_benchmark_comparison

    strategy_daily = np.array([0.01] * 10)  # ~10.46% total
    benchmark_daily = np.array([0.005] * 10)  # ~5.11% total
    result = compute_benchmark_comparison(strategy_daily, benchmark_daily, "SPY")
    assert result.strategy_return_pct > result.benchmark_return_pct
    assert result.strategy_return_pct > 0
    assert result.benchmark_return_pct > 0

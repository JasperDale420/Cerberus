import numpy as np
import pytest

from src.analytics.param_sensitivity import analyze_param_sensitivity


@pytest.mark.unit
def test_sensitivity_ranking():
    trials_data = {
        "param_a": np.linspace(1.0, 10.0, 50).tolist(),
        "param_b": np.random.default_rng(42).uniform(0, 1, 50).tolist(),
        "score": np.linspace(0.5, 5.0, 50).tolist(),
    }
    results = analyze_param_sensitivity(trials_data)
    a_result = next(r for r in results if r.param_name == "param_a")
    b_result = next(r for r in results if r.param_name == "param_b")
    assert a_result.sensitivity_rank < b_result.sensitivity_rank
    assert abs(a_result.correlation) > abs(b_result.correlation)


@pytest.mark.unit
def test_sensitivity_with_few_trials():
    trials_data = {
        "param_a": [1.0, 2.0, 3.0],
        "score": [0.5, 1.0, 1.5],
    }
    results = analyze_param_sensitivity(trials_data)
    assert len(results) == 1
    assert results[0].param_name == "param_a"


@pytest.mark.unit
def test_sensitivity_constant_param_gets_zero():
    trials_data = {
        "param_a": [5.0, 5.0, 5.0, 5.0, 5.0],
        "param_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        "score": [0.5, 1.0, 1.5, 2.0, 2.5],
    }
    results = analyze_param_sensitivity(trials_data)
    const_result = next(r for r in results if r.param_name == "param_a")
    assert const_result.correlation == 0.0


@pytest.mark.unit
def test_sensitivity_negative_correlation():
    trials_data = {
        "param_a": [10.0, 8.0, 6.0, 4.0, 2.0],
        "score": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    results = analyze_param_sensitivity(trials_data)
    assert results[0].correlation < 0

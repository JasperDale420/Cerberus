import numpy as np
import pytest


@pytest.mark.unit
def test_monte_carlo_basic_properties():
    from src.analytics.monte_carlo import run_monte_carlo

    rng = np.random.default_rng(42)
    trade_pnls = rng.normal(50, 200, 100).tolist()
    result = run_monte_carlo(trade_pnls, initial_capital=100_000, n_simulations=1000)

    assert result.n_simulations == 1000
    assert "sharpe" in result.metric_distributions
    assert "max_drawdown_pct" in result.metric_distributions
    assert "final_equity" in result.metric_distributions
    assert 0.0 <= result.probability_of_loss <= 1.0
    assert 0.0 <= result.probability_of_ruin <= 1.0


@pytest.mark.unit
def test_monte_carlo_all_winners_low_loss_probability():
    from src.analytics.monte_carlo import run_monte_carlo

    trade_pnls = [100.0] * 50
    result = run_monte_carlo(trade_pnls, initial_capital=100_000, n_simulations=1000)
    assert result.probability_of_loss == 0.0


@pytest.mark.unit
def test_monte_carlo_all_losers_high_loss_probability():
    from src.analytics.monte_carlo import run_monte_carlo

    trade_pnls = [-100.0] * 50
    result = run_monte_carlo(trade_pnls, initial_capital=100_000, n_simulations=1000)
    assert result.probability_of_loss == 1.0


@pytest.mark.unit
def test_monte_carlo_confidence_interval():
    from src.analytics.monte_carlo import run_monte_carlo

    rng = np.random.default_rng(42)
    trade_pnls = rng.normal(50, 200, 200).tolist()
    result = run_monte_carlo(trade_pnls, initial_capital=100_000, n_simulations=5000)
    low, high = result.confidence_interval_95
    assert low < high


@pytest.mark.unit
def test_monte_carlo_percentile_bands():
    from src.analytics.monte_carlo import PercentileBands, run_monte_carlo

    trade_pnls = [100.0, -50.0, 200.0, -30.0, 150.0] * 20
    result = run_monte_carlo(trade_pnls, initial_capital=100_000, n_simulations=1000)
    bands = result.metric_distributions["final_equity"]
    assert isinstance(bands, PercentileBands)
    assert bands.p5 <= bands.p25 <= bands.p50 <= bands.p75 <= bands.p95

import pytest

from src.analytics.monte_carlo import MonteCarloResult, run_monte_carlo


@pytest.mark.unit
def test_monte_carlo_basic():
    pnls = [100.0, -50.0, 75.0, -25.0, 60.0, -30.0, 80.0, -40.0]
    result = run_monte_carlo(pnls, initial_capital=100_000.0, n_simulations=1000, seed=42)
    assert isinstance(result, MonteCarloResult)
    assert result.n_simulations == 1000
    assert result.median_final_equity > 0


@pytest.mark.unit
def test_monte_carlo_all_winners():
    pnls = [100.0] * 20
    result = run_monte_carlo(pnls, initial_capital=100_000.0, n_simulations=500, seed=42)
    assert result.prob_loss_pct == 0.0
    assert result.prob_ruin_pct == 0.0
    assert result.median_final_equity > 100_000.0


@pytest.mark.unit
def test_monte_carlo_all_losers():
    pnls = [-500.0] * 20
    result = run_monte_carlo(pnls, initial_capital=100_000.0, n_simulations=500, seed=42)
    assert result.prob_loss_pct == 100.0
    assert result.median_final_equity < 100_000.0


@pytest.mark.unit
def test_monte_carlo_confidence_interval():
    pnls = [100.0, -50.0, 75.0, -25.0, 60.0]
    result = run_monte_carlo(pnls, n_simulations=1000, seed=42)
    assert result.p5_final_equity <= result.median_final_equity
    assert result.median_final_equity <= result.p95_final_equity


@pytest.mark.unit
def test_monte_carlo_sharpe_bounds():
    pnls = [100.0, -50.0, 75.0, -25.0, 60.0, -30.0]
    result = run_monte_carlo(pnls, n_simulations=500, seed=42)
    assert result.p5_sharpe <= result.median_sharpe
    assert result.median_sharpe <= result.p95_sharpe

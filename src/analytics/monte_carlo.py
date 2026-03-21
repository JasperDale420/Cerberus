"""Monte Carlo bootstrap simulation for trade P&L sequences.

Resamples trade P&Ls with replacement to estimate:
- Probability of loss (ending below initial capital)
- Probability of ruin (drawdown exceeding threshold)
- Worst-case drawdown distribution
- Sharpe ratio distribution
- Confidence intervals on final equity
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    n_simulations: int
    prob_loss_pct: float
    prob_ruin_pct: float
    median_final_equity: float
    p5_final_equity: float
    p95_final_equity: float
    worst_drawdown_pct: float
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float


def run_monte_carlo(
    trade_pnls: list[float],
    initial_capital: float = 100_000.0,
    n_simulations: int = 10_000,
    ruin_threshold_pct: float = 30.0,
    seed: int | None = None,
    calendar_days: int | None = None,
) -> MonteCarloResult:
    """Run bootstrap Monte Carlo simulation on trade P&Ls.

    Args:
        trade_pnls: List of per-trade P&L values.
        initial_capital: Starting capital for equity curve simulation.
        n_simulations: Number of bootstrap simulations.
        ruin_threshold_pct: Drawdown percentage that constitutes ruin.
        seed: Optional random seed for reproducibility.

    Returns:
        MonteCarloResult with distribution statistics.
    """
    rng = np.random.default_rng(seed)
    pnls = np.array(trade_pnls, dtype=np.float64)
    n_trades = len(pnls)

    if n_trades == 0:
        return MonteCarloResult(
            n_simulations=n_simulations,
            prob_loss_pct=0.0,
            prob_ruin_pct=0.0,
            median_final_equity=initial_capital,
            p5_final_equity=initial_capital,
            p95_final_equity=initial_capital,
            worst_drawdown_pct=0.0,
            median_sharpe=0.0,
            p5_sharpe=0.0,
            p95_sharpe=0.0,
        )

    final_equities = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)
    sharpes = np.empty(n_simulations)
    loss_count = 0
    ruin_count = 0
    ruin_threshold = ruin_threshold_pct / 100.0

    for i in range(n_simulations):
        # Resample trades with replacement
        sampled = rng.choice(pnls, size=n_trades, replace=True)
        equity_curve = initial_capital + np.cumsum(sampled)

        final_eq = equity_curve[-1]
        final_equities[i] = final_eq

        if final_eq < initial_capital:
            loss_count += 1

        # Max drawdown
        running_peak = np.maximum.accumulate(np.concatenate(([initial_capital], equity_curve)))
        drawdowns = (running_peak[1:] - equity_curve) / np.where(running_peak[1:] > 0, running_peak[1:], 1.0)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        max_drawdowns[i] = max_dd

        if max_dd >= ruin_threshold:
            ruin_count += 1

        # Per-simulation Sharpe (annualized by trades-per-year)
        mean_pnl = np.mean(sampled)
        std_pnl = np.std(sampled, ddof=1) if n_trades > 1 else 0.0
        if std_pnl > 0:
            if calendar_days is not None and calendar_days > 0:
                years = calendar_days / 365.0
                trades_per_year = n_trades / years if years > 0 else n_trades
            else:
                trades_per_year = min(n_trades, 252)
            sharpes[i] = (mean_pnl / std_pnl) * math.sqrt(trades_per_year)
        else:
            sharpes[i] = 0.0

    return MonteCarloResult(
        n_simulations=n_simulations,
        prob_loss_pct=round((loss_count / n_simulations) * 100, 2),
        prob_ruin_pct=round((ruin_count / n_simulations) * 100, 2),
        median_final_equity=round(float(np.median(final_equities)), 2),
        p5_final_equity=round(float(np.percentile(final_equities, 5)), 2),
        p95_final_equity=round(float(np.percentile(final_equities, 95)), 2),
        worst_drawdown_pct=round(float(np.max(max_drawdowns)) * 100, 2),
        median_sharpe=round(float(np.median(sharpes)), 4),
        p5_sharpe=round(float(np.percentile(sharpes, 5)), 4),
        p95_sharpe=round(float(np.percentile(sharpes, 95)), 4),
    )

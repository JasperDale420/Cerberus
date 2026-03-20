from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PercentileBands:
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    n_simulations: int
    metric_distributions: dict[str, PercentileBands]
    probability_of_loss: float
    probability_of_ruin: float
    worst_case_drawdown: float
    confidence_interval_95: tuple[float, float]


def _compute_percentiles(values: np.ndarray) -> PercentileBands:
    return PercentileBands(
        p5=float(np.percentile(values, 5)),
        p25=float(np.percentile(values, 25)),
        p50=float(np.percentile(values, 50)),
        p75=float(np.percentile(values, 75)),
        p95=float(np.percentile(values, 95)),
    )


def run_monte_carlo(
    trade_pnls: list[float],
    initial_capital: float = 100_000.0,
    n_simulations: int = 10_000,
    ruin_threshold_pct: float = 30.0,
    seed: int = 42,
) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    pnls = np.array(trade_pnls)
    n_trades = len(pnls)

    final_equities = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)
    sharpes = np.empty(n_simulations)

    for i in range(n_simulations):
        sampled = rng.choice(pnls, size=n_trades, replace=True)
        equity_curve = initial_capital + np.cumsum(sampled)
        final_equities[i] = equity_curve[-1]

        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / peak
        max_drawdowns[i] = float(np.max(dd)) * 100.0

        daily_ish = sampled / initial_capital
        std = np.std(daily_ish)
        sharpes[i] = float(np.mean(daily_ish) / std * np.sqrt(252)) if std > 1e-10 else 0.0

    prob_loss = float(np.mean(final_equities < initial_capital))
    prob_ruin = float(np.mean(max_drawdowns > ruin_threshold_pct))

    return MonteCarloResult(
        n_simulations=n_simulations,
        metric_distributions={
            "final_equity": _compute_percentiles(final_equities),
            "max_drawdown_pct": _compute_percentiles(max_drawdowns),
            "sharpe": _compute_percentiles(sharpes),
        },
        probability_of_loss=prob_loss,
        probability_of_ruin=prob_ruin,
        worst_case_drawdown=float(np.percentile(max_drawdowns, 95)),
        confidence_interval_95=(
            float(np.percentile(final_equities, 2.5)),
            float(np.percentile(final_equities, 97.5)),
        ),
    )

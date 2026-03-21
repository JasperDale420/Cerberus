from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    benchmark_symbol: str
    benchmark_return_pct: float
    strategy_return_pct: float
    strategy_alpha: float
    strategy_beta: float
    information_ratio: float
    up_capture: float
    down_capture: float


def compute_benchmark_comparison(
    strategy_daily_returns: np.ndarray,
    benchmark_daily_returns: np.ndarray,
    benchmark_symbol: str,
) -> BenchmarkComparison:
    strat_total = float(np.prod(1 + strategy_daily_returns) - 1)
    bench_total = float(np.prod(1 + benchmark_daily_returns) - 1)

    # Beta via OLS regression
    if np.std(benchmark_daily_returns) > 1e-10:
        cov = np.cov(strategy_daily_returns, benchmark_daily_returns)
        beta = float(cov[0, 1] / cov[1, 1])
    else:
        beta = 0.0

    alpha = strat_total - beta * bench_total

    # Information ratio — use daily excess returns for consistent annualization
    daily_excess = strategy_daily_returns - benchmark_daily_returns
    mean_excess = float(np.mean(daily_excess))
    std_excess = float(np.std(daily_excess, ddof=1)) if len(daily_excess) > 1 else 0.0
    ir = (mean_excess / std_excess * np.sqrt(252)) if std_excess > 1e-10 else 0.0

    # Capture ratios
    up_mask = benchmark_daily_returns > 0
    down_mask = benchmark_daily_returns < 0

    if up_mask.sum() > 0:
        up_capture = float(strategy_daily_returns[up_mask].mean() / benchmark_daily_returns[up_mask].mean())
    else:
        up_capture = 0.0

    if down_mask.sum() > 0:
        down_capture = float(strategy_daily_returns[down_mask].mean() / benchmark_daily_returns[down_mask].mean())
    else:
        down_capture = 0.0

    return BenchmarkComparison(
        benchmark_symbol=benchmark_symbol,
        benchmark_return_pct=bench_total * 100,
        strategy_return_pct=strat_total * 100,
        strategy_alpha=alpha * 100,
        strategy_beta=beta,
        information_ratio=ir,
        up_capture=up_capture,
        down_capture=down_capture,
    )

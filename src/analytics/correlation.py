"""Strategy correlation matrix and diversification analysis.

Computes pairwise Pearson correlations across strategy daily returns and
derives a diversification ratio (DR) measuring the benefit of running
multiple strategies concurrently.

DR = sum(individual_vols) / portfolio_vol  (equal-weighted)
  - DR > 1  → strategies provide diversification benefit
  - DR ≈ 1  → no benefit (perfectly correlated)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True, slots=True)
class StrategyCorrelation:
    """Result of cross-strategy correlation analysis."""

    strategy_names: list[str]
    correlation_matrix: list[list[float]]  # N×N matrix as nested lists (JSON-serializable)
    avg_pairwise_correlation: float
    max_pairwise_correlation: float  # highest off-diagonal
    max_corr_pair: tuple[str, str]  # which pair has highest correlation
    diversification_ratio: float  # DR = weighted_vol_sum / portfolio_vol
    n_trading_days: int


def compute_strategy_correlation(
    strategy_daily_returns: dict[str, np.ndarray],
) -> StrategyCorrelation:
    """Compute correlation matrix and diversification ratio across strategies.

    Args:
        strategy_daily_returns: dict mapping strategy_name → daily return array.
            All arrays must have the same length.

    Returns:
        StrategyCorrelation with full correlation matrix and summary statistics.
    """
    names = sorted(strategy_daily_returns.keys())
    n_strategies = len(names)

    if n_strategies == 0:
        raise ValueError("strategy_daily_returns must contain at least one strategy")

    n_days = len(strategy_daily_returns[names[0]])

    # --- Single strategy degenerate case ---
    if n_strategies == 1:
        return StrategyCorrelation(
            strategy_names=names,
            correlation_matrix=[[1.0]],
            avg_pairwise_correlation=1.0,
            max_pairwise_correlation=1.0,
            max_corr_pair=(names[0], names[0]),
            diversification_ratio=1.0,
            n_trading_days=n_days,
        )

    # --- Build returns matrix: rows = strategies, cols = days ---
    returns_matrix = np.array([strategy_daily_returns[name] for name in names])

    # --- Pearson correlation matrix ---
    corr_matrix = np.corrcoef(returns_matrix)

    # Replace NaN with 0.0 (happens when a strategy has zero variance)
    corr_matrix = np.where(np.isnan(corr_matrix), 0.0, corr_matrix)

    # --- Extract upper-triangle off-diagonal elements ---
    upper_indices = np.triu_indices(n_strategies, k=1)
    off_diagonal = corr_matrix[upper_indices]

    avg_pairwise = float(np.mean(off_diagonal))
    max_pairwise = float(np.max(off_diagonal))

    # Find which pair has the max correlation
    max_idx = int(np.argmax(off_diagonal))
    max_i = upper_indices[0][max_idx]
    max_j = upper_indices[1][max_idx]
    max_corr_pair = (names[max_i], names[max_j])

    # --- Diversification ratio (equal-weighted portfolio) ---
    individual_vols = np.array([np.std(strategy_daily_returns[name], ddof=1) for name in names])

    # Equal-weighted portfolio returns
    portfolio_returns = np.mean(returns_matrix, axis=0)
    portfolio_vol = np.std(portfolio_returns, ddof=1)

    if portfolio_vol > 0:
        dr = float(np.mean(individual_vols) / portfolio_vol)
    else:
        dr = 1.0

    # --- Convert correlation matrix to nested Python lists ---
    corr_list = [[float(corr_matrix[i, j]) for j in range(n_strategies)] for i in range(n_strategies)]

    return StrategyCorrelation(
        strategy_names=names,
        correlation_matrix=corr_list,
        avg_pairwise_correlation=avg_pairwise,
        max_pairwise_correlation=max_pairwise,
        max_corr_pair=max_corr_pair,
        diversification_ratio=dr,
        n_trading_days=n_days,
    )


def build_strategy_daily_returns(
    trades: list[dict],
    n_trading_days: int,
) -> dict[str, np.ndarray]:
    """Build daily PnL arrays from a list of trades.

    Each trade's PnL is allocated to the exit date. Multiple trades exiting
    on the same day are summed. Non-trade days get 0.

    Args:
        trades: list of dicts with keys: strategy, entry_time, exit_time, pnl.
            Times can be datetime objects or timestamps.
        n_trading_days: total number of trading days for the output arrays.

    Returns:
        dict mapping strategy_name → np.ndarray of daily PnL (length n_trading_days).
    """
    if not trades:
        return {}

    # Group trades by strategy, collect (exit_date, pnl) pairs
    strategy_trades: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for trade in trades:
        exit_time = trade["exit_time"]
        if not isinstance(exit_time, datetime):
            exit_time = datetime.fromtimestamp(exit_time)
        exit_date = exit_time.date()
        strategy_trades[trade["strategy"]].append((exit_date, float(trade["pnl"])))

    # Collect all unique exit dates across all strategies, sorted
    all_dates: set = set()
    for pairs in strategy_trades.values():
        for date, _ in pairs:
            all_dates.add(date)
    sorted_dates = sorted(all_dates)

    # Build date → index mapping
    # If we have fewer unique dates than n_trading_days, we pad with zeros
    # Use actual dates for indexing, fill remaining with 0
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}

    result: dict[str, np.ndarray] = {}
    for strategy, pairs in strategy_trades.items():
        daily = np.zeros(n_trading_days)
        for date, pnl in pairs:
            idx = date_to_idx[date]
            if idx < n_trading_days:
                daily[idx] += pnl
        result[strategy] = daily

    return result

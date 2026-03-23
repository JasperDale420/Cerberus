"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation.

Implements the CSCV method from Bailey, Borwein, Lopez de Prado & Zhu (2017),
"The Probability of Backtest Overfitting".

PBO measures the likelihood that the best in-sample strategy is overfit and will
underperform out-of-sample. A PBO > 0.5 indicates more than half the time, the
IS-optimal strategy ranks below median OOS — a strong overfitting signal.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

_MAX_COMBINATIONS = 1000


def _sharpe_ratio(returns: np.ndarray) -> float:
    """Annualized Sharpe ratio from a 1D return series. Returns 0.0 if std is zero."""
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-15:
        return 0.0
    return float(mu / sigma)


def compute_pbo(trial_returns: np.ndarray, n_partitions: int = 16) -> dict:
    """Compute Probability of Backtest Overfitting using CSCV.

    Args:
        trial_returns: 2D array of shape (n_bars, n_trials). Each column is
            one parameter combination's return series.
        n_partitions: Number of equal-sized blocks to split the bar axis into.
            Must be >= 4 (need at least 2 IS + 2 OOS blocks).

    Returns:
        Dictionary with keys:
            pbo: Fraction of combinations where IS-best underperformed OOS median.
            n_combinations_tested: Number of IS/OOS splits evaluated.
            avg_oos_logit: Mean logit of IS-best trial's OOS rank.
            overfit_warning: True if pbo > 0.5.

    Raises:
        ValueError: If n_partitions < 4 or fewer than 2 trials.
    """
    if n_partitions < 4:
        raise ValueError(f"n_partitions must be >= 4, got {n_partitions} partitions")

    trial_returns = np.asarray(trial_returns, dtype=np.float64)
    if trial_returns.ndim != 2:
        raise ValueError(f"trial_returns must be 2D, got {trial_returns.ndim}D")

    n_bars, n_trials = trial_returns.shape
    if n_trials < 2:
        raise ValueError(f"Need at least 2 trials, got {n_trials} trials")

    # Split into equal-sized blocks, truncating remainder
    block_size = n_bars // n_partitions
    if block_size < 1:
        raise ValueError(f"Not enough bars ({n_bars}) for {n_partitions} partitions")

    usable_bars = block_size * n_partitions
    data = trial_returns[:usable_bars, :]  # (usable_bars, n_trials)

    # Reshape into (n_partitions, block_size, n_trials)
    blocks = data.reshape(n_partitions, block_size, n_trials)

    n_is = n_partitions // 2
    all_combos = list(itertools.combinations(range(n_partitions), n_is))

    rng = np.random.default_rng(42)
    if len(all_combos) > _MAX_COMBINATIONS:
        indices = rng.choice(len(all_combos), size=_MAX_COMBINATIONS, replace=False)
        combos = [all_combos[i] for i in indices]
    else:
        combos = all_combos

    underperform_count = 0
    logits = []

    for is_indices in combos:
        oos_indices = tuple(i for i in range(n_partitions) if i not in is_indices)

        # Concatenate blocks for IS and OOS
        is_returns = np.concatenate([blocks[i] for i in is_indices], axis=0)  # (n_is*block_size, n_trials)
        oos_returns = np.concatenate([blocks[i] for i in oos_indices], axis=0)

        # Compute Sharpe for each trial on IS data
        is_sharpes = np.array([_sharpe_ratio(is_returns[:, t]) for t in range(n_trials)])

        # Pick IS-best trial
        best_trial = int(np.argmax(is_sharpes))

        # Compute Sharpe for each trial on OOS data
        oos_sharpes = np.array([_sharpe_ratio(oos_returns[:, t]) for t in range(n_trials)])

        # Rank of IS-best trial in OOS (0 = worst, n_trials-1 = best)
        oos_best_sharpe = oos_sharpes[best_trial]
        # rank_below: how many trials have OOS Sharpe <= the IS-best's OOS Sharpe
        rank_below = np.sum(oos_sharpes <= oos_best_sharpe)
        # Relative rank: 0 = worst performer, 1 = best performer
        relative_rank = rank_below / n_trials

        # Logit of relative rank (clamp to avoid log(0) or log(inf))
        clamped = np.clip(relative_rank, 1 / (2 * n_trials), 1 - 1 / (2 * n_trials))
        logit = math.log(clamped / (1 - clamped))
        logits.append(logit)

        # Check if IS-best underperformed OOS median
        median_oos = float(np.median(oos_sharpes))
        if oos_best_sharpe < median_oos:
            underperform_count += 1

    n_tested = len(combos)
    pbo = underperform_count / n_tested if n_tested > 0 else 0.0

    return {
        "pbo": pbo,
        "n_combinations_tested": n_tested,
        "avg_oos_logit": float(np.mean(logits)) if logits else 0.0,
        "overfit_warning": pbo > 0.5,
    }

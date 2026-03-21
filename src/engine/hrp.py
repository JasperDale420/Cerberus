"""Hierarchical Risk Parity (HRP) cross-strategy capital allocator.

Three-step algorithm:
1. Tree clustering: Build dendrogram from strategy return correlation matrix
2. Quasi-diagonalization: Reorder correlation matrix to group similar strategies
3. Recursive bisection: Split allocation recursively, weighting by inverse variance

Paper: Lopez de Prado (2016) — "Building Diversified Portfolios that Outperform Out-of-Sample"
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy.cluster.hierarchy import leaves_list, linkage

from src.core.logger import StructuredLogger


class HRPConfig(BaseModel):
    """Hierarchical Risk Parity cross-strategy allocation configuration."""

    enabled: bool = Field(default=False, description="Enable HRP cross-strategy allocation")
    lookback_days: int = Field(default=60, description="Rolling window for strategy returns")
    min_strategies: int = Field(default=3, description="Minimum active strategies to run HRP")
    rebalance_interval: int = Field(default=5, description="Recompute weights every N days")
    min_weight: float = Field(default=0.05, description="Minimum per-strategy weight")
    max_weight: float = Field(default=0.40, description="Maximum per-strategy weight")


class HRPAllocator:
    """Computes per-strategy capital allocation fractions using Hierarchical Risk Parity.

    Correlated strategies get less capital; uncorrelated strategies get more.
    """

    def __init__(self, config: HRPConfig, logger: Optional[StructuredLogger] = None) -> None:
        self.config = config
        self.logger = logger

        # strategy -> list of (date, return) tuples, kept sorted by date
        self._returns: Dict[str, List[tuple[date, float]]] = defaultdict(list)

        # Cached weights from last computation
        self._weights: Optional[Dict[str, float]] = None
        self._last_rebalance_date: Optional[date] = None
        self._computation_count: int = 0

    def record_daily_return(self, strategy: str, dt: date, daily_return: float) -> None:
        """Feed a per-strategy daily return into rolling history."""
        history = self._returns[strategy]

        # Insert in date order (most appends are at the end)
        if not history:
            history.append((dt, daily_return))
        elif history[-1][0] == dt:
            # Same date — accumulate (multiple trades in one day)
            history[-1] = (dt, history[-1][1] + daily_return)
        elif history[-1][0] < dt:
            history.append((dt, daily_return))
        else:
            # Out-of-order insert (rare)
            for i, (d, _) in enumerate(history):
                if d == dt:
                    history[i] = (dt, history[i][1] + daily_return)
                    return
                if d > dt:
                    history.insert(i, (dt, daily_return))
                    return
            history.append((dt, daily_return))

        # Trim to lookback window
        cutoff = len(history) - self.config.lookback_days
        if cutoff > 0:
            self._returns[strategy] = history[cutoff:]

    def compute_weights(self) -> Optional[Dict[str, float]]:
        """Compute HRP strategy weights from return correlation structure.

        Returns strategy_name -> weight mapping (sums to 1.0), or None if
        insufficient data (fewer than min_strategies or lookback_days).
        """
        strategies = [s for s, h in self._returns.items() if len(h) >= self.config.lookback_days]

        if len(strategies) < self.config.min_strategies:
            return None

        # Check rebalance interval
        today = max(h[-1][0] for h in self._returns.values() if h)
        if (
            self._last_rebalance_date is not None
            and self._weights is not None
            and (today - self._last_rebalance_date).days < self.config.rebalance_interval
        ):
            return self._weights

        # Build return matrix: rows = dates, cols = strategies
        # Align on common dates
        date_sets = [set(d for d, _ in self._returns[s][-self.config.lookback_days :]) for s in strategies]
        common_dates = sorted(set.intersection(*date_sets))

        if len(common_dates) < self.config.lookback_days // 2:
            # Not enough overlapping dates
            return None

        # Build aligned return matrix
        n_dates = len(common_dates)
        n_strats = len(strategies)
        returns_matrix = np.zeros((n_dates, n_strats))

        for j, strat in enumerate(strategies):
            date_map = {d: r for d, r in self._returns[strat][-self.config.lookback_days :]}
            for i, dt in enumerate(common_dates):
                returns_matrix[i, j] = date_map[dt]

        # Handle degenerate cases
        if n_strats == 1:
            self._weights = {strategies[0]: 1.0}
            self._last_rebalance_date = today
            self._computation_count += 1
            return self._weights

        if n_strats == 2:
            # Two strategies: split by inverse variance
            var = np.var(returns_matrix, axis=0)
            var = np.maximum(var, 1e-10)
            inv_var = 1.0 / var
            raw_weights = inv_var / inv_var.sum()
            weights = self._clamp_and_normalize(dict(zip(strategies, raw_weights, strict=False)))
            self._weights = weights
            self._last_rebalance_date = today
            self._computation_count += 1
            self._log_weights(weights)
            return self._weights

        # Check for degenerate case: zero-variance or perfectly correlated returns
        variances = np.var(returns_matrix, axis=0)
        if np.all(variances < 1e-12):
            # All strategies constant — equal weight
            equal_w = 1.0 / n_strats
            weights = self._clamp_and_normalize({s: equal_w for s in strategies})
            self._weights = weights
            self._last_rebalance_date = today
            self._computation_count += 1
            self._log_weights(weights)
            return self._weights

        # Step 1: Correlation and distance matrix
        corr = np.corrcoef(returns_matrix, rowvar=False)
        # Ensure correlation is valid (no NaN from constant columns)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        # If all off-diagonal correlations are ~1.0, strategies are effectively identical
        off_diag = corr[np.triu_indices(n_strats, k=1)]
        if len(off_diag) > 0 and np.all(np.abs(off_diag) > 0.999):
            equal_w = 1.0 / n_strats
            weights = self._clamp_and_normalize({s: equal_w for s in strategies})
            self._weights = weights
            self._last_rebalance_date = today
            self._computation_count += 1
            self._log_weights(weights)
            return self._weights

        dist = np.sqrt(0.5 * (1.0 - corr))

        # Step 2: Hierarchical clustering (single linkage)
        # Convert distance matrix to condensed form for scipy
        condensed = dist[np.triu_indices(n_strats, k=1)]
        link = linkage(condensed, method="single")

        # Step 3: Quasi-diagonalization — reorder by dendrogram leaf ordering
        sort_ix = leaves_list(link).tolist()

        # Step 4: Recursive bisection
        cov = np.cov(returns_matrix, rowvar=False)
        cov = np.nan_to_num(cov, nan=0.0)
        # Ensure positive semi-definite
        np.fill_diagonal(cov, np.maximum(np.diag(cov), 1e-10))

        raw_weights = self._recursive_bisection(cov, sort_ix)

        # Map back to strategy names
        weight_dict = {}
        for idx, w in raw_weights.items():
            weight_dict[strategies[idx]] = w

        weights = self._clamp_and_normalize(weight_dict)

        self._weights = weights
        self._last_rebalance_date = today
        self._computation_count += 1
        self._log_weights(weights)
        return self._weights

    def _recursive_bisection(self, cov: np.ndarray, sort_ix: list[int]) -> Dict[int, float]:
        """Recursive bisection step of HRP: allocate by inverse variance at each split."""
        weights: Dict[int, float] = {i: 1.0 for i in sort_ix}

        cluster_items = [sort_ix]
        while cluster_items:
            next_level = []
            for cluster in cluster_items:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                v_left = self._cluster_variance(cov, left)
                v_right = self._cluster_variance(cov, right)

                total_var = v_left + v_right
                if total_var < 1e-20:
                    alpha = 0.5
                else:
                    alpha = 1.0 - v_left / total_var  # left gets less if it has more variance

                for i in left:
                    weights[i] *= alpha
                for i in right:
                    weights[i] *= 1.0 - alpha

                if len(left) > 1:
                    next_level.append(left)
                if len(right) > 1:
                    next_level.append(right)

            cluster_items = next_level

        return weights

    @staticmethod
    def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
        """Compute inverse-variance portfolio variance for a cluster of assets."""
        sub_cov = cov[np.ix_(indices, indices)]
        diag = np.diag(sub_cov)
        diag = np.maximum(diag, 1e-10)
        inv_diag = 1.0 / diag
        w = inv_diag / inv_diag.sum()
        return float(w @ sub_cov @ w)

    def _clamp_and_normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Clamp weights to [min_weight, max_weight] and renormalize to sum=1.0.

        Uses iterative clamping: clamp extremes, redistribute excess, repeat until stable.
        """
        n = len(weights)
        if n == 0:
            return weights
        if n == 1:
            key = next(iter(weights))
            return {key: 1.0}

        min_w = self.config.min_weight
        max_w = self.config.max_weight

        result = dict(weights)

        # Iterative clamp-and-redistribute (converges in a few iterations)
        for _ in range(20):
            total = sum(result.values())
            if total > 0:
                result = {s: w / total for s, w in result.items()}

            clamped_any = False
            frozen: Dict[str, float] = {}
            free: Dict[str, float] = {}

            for s, w in result.items():
                if w < min_w:
                    frozen[s] = min_w
                    clamped_any = True
                elif w > max_w:
                    frozen[s] = max_w
                    clamped_any = True
                else:
                    free[s] = w

            if not clamped_any:
                break

            frozen_total = sum(frozen.values())
            free_total = sum(free.values())
            remainder = 1.0 - frozen_total

            if free_total > 0 and remainder > 0:
                scale = remainder / free_total
                result = dict(frozen)
                result.update({s: w * scale for s, w in free.items()})
            else:
                result = dict(frozen)
                result.update(free)

        # Final normalize
        total = sum(result.values())
        if total > 0:
            result = {s: w / total for s, w in result.items()}

        return result

    def _log_weights(self, weights: Dict[str, float]) -> None:
        """Log computed HRP weights for observability."""
        if self.logger is not None:
            self.logger.info(
                "HRP weights computed",
                weights={s: round(w, 4) for s, w in weights.items()},
                computation_count=self._computation_count,
            )

    def get_strategy_weight(self, strategy: str) -> float:
        """Return the HRP weight for a strategy.

        Falls back to equal weight (1/N) if HRP has not been computed or
        the strategy is unknown.
        """
        if self._weights is None:
            # Equal weight fallback
            n = max(len(self._returns), 1)
            return 1.0 / n

        if strategy in self._weights:
            return self._weights[strategy]

        # Unknown strategy: equal weight across all known strategies + this one
        n = max(len(self._weights) + 1, 1)
        return 1.0 / n

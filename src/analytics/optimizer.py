from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

from src.core.logger import StructuredLogger


class BaseOptimizer(ABC):
    """Base class for parameter optimizers."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    @abstractmethod
    def optimize(
        self,
        evaluate_fn: Any,
        search_space: Dict[str, List[Any]],
        min_trades: int = 10,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Optimize parameters based on current search space.
        Returns: (best_params, metrics)
        """
        pass


class GridSearchOptimizer(BaseOptimizer):
    """Exhaustive grid search optimizer."""

    def optimize(
        self,
        evaluate_fn: Any,
        search_space: Dict[str, List[Any]],
        min_trades: int = 10,
        score_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        keys = sorted(search_space.keys())
        values = [search_space[k] for k in keys]
        combinations = list(itertools.product(*values))

        best_params: Optional[Dict[str, Any]] = None
        best_score = -float("inf")
        best_metrics: Dict[str, Any] = {}

        self.logger.info("Starting grid search", n_combinations=len(combinations))

        for combo in combinations:
            params = dict(zip(keys, combo, strict=False))
            try:
                metrics = evaluate_fn(params)
                n_trades = int(metrics.get("n_trades", 0))

                if n_trades >= min_trades:
                    if score_fn:
                        score = score_fn(metrics)
                    else:
                        score = float(metrics.get("expectancy", -100.0))

                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = metrics
            except Exception as e:
                self.logger.warning(
                    "Optimization combo failed", params=params, error=str(e)
                )

        if best_params:
            self.logger.info(
                "Grid search optimized",
                best_score=best_score,
                n_trades=best_metrics.get("n_trades"),
            )
        else:
            self.logger.warning("Grid search found no valid candidates")

        return best_params, best_metrics

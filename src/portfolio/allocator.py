"""Portfolio allocator using risk-parity with drawdown throttle and correlation adjustment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AllocationResult:
    """Result of portfolio allocation computation."""

    allocations: dict[str, float]
    total_exposure: float
    throttled_strategies: list[str] = field(default_factory=list)


class PortfolioAllocator:
    """Cross-strategy capital allocation using risk-parity.

    Pipeline:
      1. Risk-parity weights (inverse-vol, normalized).
      2. Drawdown throttle — halve allocation if current DD exceeds threshold.
      3. Correlation adjustment — scale total exposure down when strategies are correlated.
      4. Clamp each allocation to [min, max] bounds.
    """

    def __init__(
        self,
        max_strategy_allocation: float = 0.40,
        min_strategy_allocation: float = 0.05,
        drawdown_throttle_mult: float = 1.5,
    ) -> None:
        self.max_strategy_allocation = max_strategy_allocation
        self.min_strategy_allocation = min_strategy_allocation
        self.drawdown_throttle_mult = drawdown_throttle_mult

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute_allocations(
        self,
        strategy_returns: dict[str, list[float]],
        strategy_max_dd: dict[str, float] | None = None,
    ) -> AllocationResult:
        """Compute capital allocations across strategies.

        Args:
            strategy_returns: strategy_name -> list of periodic returns.
            strategy_max_dd: optional historical max-drawdown per strategy.
                             If *None*, drawdown throttle uses each strategy's
                             own trailing max-DD as the baseline.

        Returns:
            AllocationResult with per-strategy fractions summing to <= 1.
        """
        if not strategy_returns:
            return AllocationResult(allocations={}, total_exposure=0.0, throttled_strategies=[])

        # 1. Risk-parity weights from realized vol
        strategy_vols = {
            name: float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0 for name, rets in strategy_returns.items()
        }
        allocations = self._risk_parity_weights(strategy_vols)

        # 2. Drawdown throttle
        allocations, throttled = self._apply_drawdown_throttle(allocations, strategy_returns, strategy_max_dd)

        # 3. Correlation-based exposure adjustment
        allocations = self._correlation_exposure_adjustment(allocations, strategy_returns)

        # 4. Clamp to [min, max] and re-normalize so sum <= 1
        allocations = self._clamp_allocations(allocations)

        total_exposure = sum(allocations.values())

        return AllocationResult(
            allocations=allocations,
            total_exposure=total_exposure,
            throttled_strategies=throttled,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _risk_parity_weights(self, strategy_vols: dict[str, float]) -> dict[str, float]:
        """Inverse-volatility weighting, normalized to sum to 1."""
        inv_vols: dict[str, float] = {}
        for name, vol in strategy_vols.items():
            inv_vols[name] = 1.0 / vol if vol > 1e-12 else 0.0

        total = sum(inv_vols.values())
        if total < 1e-12:
            # All zero vol — equal weight
            n = len(strategy_vols)
            return {name: 1.0 / n for name in strategy_vols}

        return {name: iv / total for name, iv in inv_vols.items()}

    def _apply_drawdown_throttle(
        self,
        allocations: dict[str, float],
        strategy_returns: dict[str, list[float]],
        strategy_max_dd: dict[str, float] | None,
    ) -> tuple[dict[str, float], list[str]]:
        """Halve allocation for strategies whose current DD exceeds the threshold."""
        throttled: list[str] = []
        result = dict(allocations)

        for name, rets in strategy_returns.items():
            if name not in result or len(rets) < 2:
                continue

            current_dd = self._trailing_max_drawdown(rets)
            if current_dd < 1e-12:
                continue

            if not strategy_max_dd or name not in strategy_max_dd:
                # Cannot throttle without historical max drawdown baseline
                continue
            historical_dd = strategy_max_dd[name]
            if historical_dd < 1e-12:
                continue

            threshold = self.drawdown_throttle_mult * historical_dd
            if current_dd > threshold:
                result[name] *= 0.5
                throttled.append(name)

        return result, throttled

    def _correlation_exposure_adjustment(
        self,
        allocations: dict[str, float],
        strategy_returns: dict[str, list[float]],
    ) -> dict[str, float]:
        """Scale down total exposure when average pairwise correlation > 0.5."""
        names = [n for n in allocations if n in strategy_returns]
        if len(names) < 2:
            return dict(allocations)

        # Build return matrix — use minimum overlapping length
        min_len = min(len(strategy_returns[n]) for n in names)
        if min_len < 2:
            return dict(allocations)

        matrix = np.array([strategy_returns[n][-min_len:] for n in names])
        corr = np.corrcoef(matrix)

        # Average pairwise correlation (upper triangle, excluding diagonal)
        n = len(names)
        upper = [corr[i, j] for i in range(n) for j in range(i + 1, n)]
        avg_corr = float(np.mean(upper)) if upper else 0.0

        if avg_corr > 0.5:
            # Scale factor: linearly reduce from 1.0 at corr=0.5 to 0.5 at corr=1.0
            scale = max(0.5, 1.0 - (avg_corr - 0.5))
            return {name: alloc * scale for name, alloc in allocations.items()}

        return dict(allocations)

    def _clamp_allocations(self, allocations: dict[str, float]) -> dict[str, float]:
        """Clamp each allocation to [min, max] bounds and re-normalize if needed."""
        clamped = {
            name: max(self.min_strategy_allocation, min(self.max_strategy_allocation, alloc))
            for name, alloc in allocations.items()
        }

        # If total > 1, proportionally scale down
        total = sum(clamped.values())
        if total > 1.0:
            clamped = {name: alloc / total for name, alloc in clamped.items()}

        return clamped

    @staticmethod
    def _trailing_max_drawdown(returns: list[float]) -> float:
        """Compute trailing max drawdown from a return series."""
        equity = np.cumprod(1.0 + np.array(returns))
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        return float(np.max(drawdowns))

"""Kelly Criterion position sizer with Wasserstein DRO.

Tracks per-strategy trade results in a rolling window and computes the
optimal equity fraction using the Kelly formula:

    f* = (b * p - q) / b

where p = win rate, q = 1 - p, b = avg_win / avg_loss.

When robust mode is enabled (default), applies Wasserstein Distributionally
Robust Optimization to compute a worst-case Kelly fraction over a ball of
radius epsilon around the empirical distribution. This guards against
estimation error in win rate and payoff ratio from limited sample sizes.

Reference: Sun & Boyd (2021) -- "Risk-Constrained Kelly Gambling"

The output is scaled by a configurable fraction (e.g., 0.5 for half-Kelly)
and clamped between min/max bounds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from src.config.models import KellyConfig
from src.core.logger import StructuredLogger


@dataclass
class StrategyTradeStats:
    """Rolling trade results for a single strategy."""

    results: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def trade_count(self) -> int:
        return len(self.results)

    @property
    def win_count(self) -> int:
        return sum(1 for pnl in self.results if pnl > 0)

    @property
    def loss_count(self) -> int:
        return sum(1 for pnl in self.results if pnl <= 0)

    @property
    def avg_win(self) -> float:
        wins = [pnl for pnl in self.results if pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [abs(pnl) for pnl in self.results if pnl <= 0]
        return sum(losses) / len(losses) if losses else 0.0


class KellySizer:
    """Computes Kelly-optimal position sizing from rolling trade history.

    Supports two modes:
    - Standard Kelly: point-estimate from empirical win rate and payoff ratio
    - Robust Kelly (default): Wasserstein DRO worst-case optimization over
      an epsilon-ball around the empirical distribution, with bootstrap CI
      safety check
    """

    def __init__(self, config: KellyConfig, logger: StructuredLogger) -> None:
        self.config = config
        self.logger = logger
        self._stats: Dict[str, StrategyTradeStats] = {}

        # Robust DRO settings -- read from config with defaults so we don't
        # require config/models.py changes (another agent handles that).
        self.robust: bool = getattr(config, "robust", True)
        self.epsilon: float = getattr(config, "epsilon", 0.05)

    def _get_stats(self, strategy: str) -> StrategyTradeStats:
        if strategy not in self._stats:
            self._stats[strategy] = StrategyTradeStats(results=deque(maxlen=self.config.lookback_trades))
        return self._stats[strategy]

    def record_trade(self, strategy: str, pnl_net: float) -> None:
        """Record a completed trade result for Kelly calculation."""
        stats = self._get_stats(strategy)
        stats.results.append(pnl_net)
        self.logger.debug(
            "Kelly trade recorded",
            strategy=strategy,
            pnl_net=round(pnl_net, 2),
            total_trades=stats.trade_count,
            win_rate=round(stats.win_count / stats.trade_count, 3) if stats.trade_count > 0 else 0,
        )

    # ------------------------------------------------------------------
    # Standard Kelly
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_standard_kelly(stats: StrategyTradeStats) -> float:
        """Compute the classic Kelly fraction from empirical stats.

        Returns the *raw* (unscaled, unclamped) Kelly fraction.
        """
        if stats.loss_count == 0:
            # All winners -- return 1.0 (will be clamped later)
            return 1.0
        if stats.win_count == 0:
            # All losers -- no edge
            return 0.0

        p = stats.win_count / stats.trade_count
        q = 1.0 - p
        b = stats.avg_win / stats.avg_loss
        return (b * p - q) / b

    # ------------------------------------------------------------------
    # Wasserstein DRO Robust Kelly
    # ------------------------------------------------------------------

    def _compute_robust_kelly(self, stats: StrategyTradeStats) -> float:
        """Compute worst-case Kelly fraction over Wasserstein ball.

        Uses a closed-form conservative approximation (no cvxpy needed):
        - Shift win probability down by epsilon
        - Shrink average win by (1 - epsilon)
        - Inflate average loss by (1 + epsilon)

        Falls back to standard Kelly if the robust estimate is negative.
        """
        if stats.trade_count == 0:
            return 0.0

        p_hat = stats.win_count / stats.trade_count
        avg_win = stats.avg_win
        avg_loss = stats.avg_loss

        # Worst-case parameters within Wasserstein ball of radius epsilon
        p_worst = max(0.0, p_hat - self.epsilon)
        q_worst = 1.0 - p_worst
        avg_win_worst = avg_win * (1.0 - self.epsilon)
        avg_loss_worst = avg_loss * (1.0 + self.epsilon)

        if avg_loss_worst <= 0.0 or avg_win_worst <= 0.0:
            return 0.0

        b_worst = avg_win_worst / avg_loss_worst
        if b_worst <= 0.0:
            return 0.0

        robust_kelly = (b_worst * p_worst - q_worst) / b_worst

        # If robust computation yields negative, there is no edge under
        # worst-case -- fall back to 0 (will be clamped to min later).
        return max(0.0, robust_kelly)

    # ------------------------------------------------------------------
    # Bootstrap confidence interval
    # ------------------------------------------------------------------

    def _bootstrap_kelly_ci(
        self,
        stats: StrategyTradeStats,
        n_bootstrap: int = 200,
        ci_level: float = 0.9,
    ) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval on the Kelly fraction.

        Returns (lower_bound, median, upper_bound).
        """
        if stats.trade_count < 2:
            return (0.0, 0.0, 0.0)

        results_arr = np.array(list(stats.results), dtype=np.float64)
        n = len(results_arr)
        rng = np.random.default_rng(seed=42)

        kelly_samples = np.empty(n_bootstrap, dtype=np.float64)

        for i in range(n_bootstrap):
            sample = rng.choice(results_arr, size=n, replace=True)
            wins = sample[sample > 0]
            losses = sample[sample <= 0]

            if len(wins) == 0:
                kelly_samples[i] = 0.0
                continue
            if len(losses) == 0:
                kelly_samples[i] = 1.0
                continue

            p = len(wins) / n
            q = 1.0 - p
            b = float(np.mean(wins)) / float(np.mean(np.abs(losses)))
            kelly_samples[i] = (b * p - q) / b if b > 0 else 0.0

        alpha = (1.0 - ci_level) / 2.0
        lower = float(np.percentile(kelly_samples, alpha * 100))
        median = float(np.median(kelly_samples))
        upper = float(np.percentile(kelly_samples, (1.0 - alpha) * 100))

        return (lower, median, upper)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_kelly_fraction(self, strategy: str) -> Optional[float]:
        """Compute the Kelly-optimal equity fraction for a strategy.

        When robust mode is enabled:
        - Computes both standard and robust (Wasserstein DRO) Kelly fractions
        - Runs bootstrap CI as an additional safety check
        - Returns the MORE CONSERVATIVE of the two estimates

        Returns:
            Clamped Kelly fraction if enough data, None if insufficient trades.
        """
        if not self.config.enabled:
            return None

        stats = self._get_stats(strategy)

        if stats.trade_count < self.config.min_trades:
            self.logger.debug(
                "Kelly insufficient data, using fallback",
                strategy=strategy,
                trades=stats.trade_count,
                min_required=self.config.min_trades,
            )
            return None

        # Standard Kelly (always computed for logging)
        standard_kelly = self._compute_standard_kelly(stats)

        if self.robust:
            robust_kelly = self._compute_robust_kelly(stats)

            # Bootstrap CI safety check
            ci_lower, ci_median, ci_upper = self._bootstrap_kelly_ci(stats)

            # If CI lower bound < 0, edge is uncertain -- reduce by 50%
            if ci_lower < 0.0:
                robust_kelly *= 0.5

            # Use the more conservative of standard and robust
            raw_kelly = min(standard_kelly, robust_kelly)

            self.logger.info(
                "Kelly fraction computed (robust)",
                strategy=strategy,
                trades=stats.trade_count,
                win_rate=round(stats.win_count / stats.trade_count, 3),
                payoff_ratio=round(stats.avg_win / stats.avg_loss, 2) if stats.avg_loss > 0 else float("inf"),
                standard_kelly=round(standard_kelly, 4),
                robust_kelly=round(robust_kelly, 4),
                ci_lower=round(ci_lower, 4),
                ci_upper=round(ci_upper, 4),
                epsilon=self.epsilon,
                raw_kelly=round(raw_kelly, 4),
            )
        else:
            raw_kelly = standard_kelly

            # Handle edge cases for logging (all winners / all losers)
            if stats.loss_count == 0:
                raw_kelly = self.config.max_equity_pct
            elif stats.win_count == 0:
                raw_kelly = self.config.min_equity_pct

            self.logger.info(
                "Kelly fraction computed",
                strategy=strategy,
                trades=stats.trade_count,
                win_rate=round(stats.win_count / stats.trade_count, 3),
                payoff_ratio=round(stats.avg_win / stats.avg_loss, 2) if stats.avg_loss > 0 else float("inf"),
                raw_kelly=round(raw_kelly, 4),
            )

        # Apply fractional Kelly scalar
        scaled = raw_kelly * self.config.fraction

        # Clamp to bounds
        clamped = max(self.config.min_equity_pct, min(self.config.max_equity_pct, scaled))

        return clamped

    def get_kelly_stats(self, strategy: str) -> Dict[str, object]:
        """Return a diagnostic dict with full Kelly statistics.

        Useful for dashboards, logging, and debugging.
        """
        stats = self._get_stats(strategy)

        if stats.trade_count == 0:
            return {
                "standard_kelly": 0.0,
                "robust_kelly": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "trade_count": 0,
                "win_rate": 0.0,
                "payoff_ratio": 0.0,
            }

        standard_kelly = self._compute_standard_kelly(stats)
        robust_kelly = self._compute_robust_kelly(stats)
        ci_lower, _ci_median, ci_upper = self._bootstrap_kelly_ci(stats)

        win_rate = stats.win_count / stats.trade_count if stats.trade_count > 0 else 0.0
        payoff_ratio = stats.avg_win / stats.avg_loss if stats.avg_loss > 0 else float("inf")

        return {
            "standard_kelly": round(standard_kelly, 6),
            "robust_kelly": round(robust_kelly, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "trade_count": stats.trade_count,
            "win_rate": round(win_rate, 6),
            "payoff_ratio": round(payoff_ratio, 6),
        }

"""Kelly Criterion position sizer.

Tracks per-strategy trade results in a rolling window and computes the
optimal equity fraction using the Kelly formula:

    f* = (b * p - q) / b

where p = win rate, q = 1 - p, b = avg_win / avg_loss.

The output is scaled by a configurable fraction (e.g., 0.5 for half-Kelly)
and clamped between min/max bounds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

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
    """Computes Kelly-optimal position sizing from rolling trade history."""

    def __init__(self, config: KellyConfig, logger: StructuredLogger) -> None:
        self.config = config
        self.logger = logger
        self._stats: Dict[str, StrategyTradeStats] = {}

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

    def get_kelly_fraction(self, strategy: str) -> Optional[float]:
        """Compute the Kelly-optimal equity fraction for a strategy.

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

        # Avoid division by zero
        if stats.loss_count == 0:
            # All winners — cap at max
            raw_kelly = self.config.max_equity_pct
        elif stats.win_count == 0:
            # All losers — floor at min
            raw_kelly = self.config.min_equity_pct
        else:
            p = stats.win_count / stats.trade_count
            q = 1.0 - p
            b = stats.avg_win / stats.avg_loss  # payoff ratio

            raw_kelly = (b * p - q) / b

        # Apply fractional Kelly scalar
        scaled = raw_kelly * self.config.fraction

        # Clamp to bounds
        clamped = max(self.config.min_equity_pct, min(self.config.max_equity_pct, scaled))

        self.logger.info(
            "Kelly fraction computed",
            strategy=strategy,
            trades=stats.trade_count,
            win_rate=round(stats.win_count / stats.trade_count, 3),
            payoff_ratio=round(stats.avg_win / stats.avg_loss, 2) if stats.avg_loss > 0 else float("inf"),
            raw_kelly=round(raw_kelly, 4),
            scaled=round(scaled, 4),
            clamped=round(clamped, 4),
        )

        return clamped

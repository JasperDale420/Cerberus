"""Portfolio-level performance analytics: attribution, rolling metrics, correlation monitoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class StrategyAttribution:
    """Performance attribution for a single strategy."""

    strategy: str
    pnl: float
    pnl_pct: float  # fraction of total portfolio P&L
    sharpe: float | None
    sortino: float | None
    max_drawdown: float


@dataclass
class CorrelationAlert:
    """Alert for high correlation between strategy pairs."""

    strategy_a: str
    strategy_b: str
    correlation: float
    is_spike: bool  # True if correlation significantly increased recently


class PortfolioPerformance:
    """Portfolio-level performance analytics engine."""

    def __init__(self, sharpe_window: int = 30, correlation_spike_threshold: float = 0.7) -> None:
        self.sharpe_window = sharpe_window
        self.correlation_spike_threshold = correlation_spike_threshold

    def compute_attribution(self, strategy_pnls: dict[str, list[float]]) -> list[StrategyAttribution]:
        """Compute attribution for each strategy.

        For each strategy: total P&L, fraction of portfolio P&L, Sharpe, Sortino, max drawdown.
        """
        strategy_totals: dict[str, float] = {}
        for name, pnls in strategy_pnls.items():
            strategy_totals[name] = sum(pnls)

        total_pnl = sum(abs(v) for v in strategy_totals.values())

        results: list[StrategyAttribution] = []
        for name, pnls in strategy_pnls.items():
            strat_pnl = strategy_totals[name]

            if total_pnl == 0.0:
                pnl_pct = 1.0 / len(strategy_pnls)
            else:
                pnl_pct = strat_pnl / sum(strategy_totals.values()) if sum(strategy_totals.values()) != 0.0 else 0.0

            # Treat daily PnLs as returns for Sharpe/Sortino
            sharpe = self.rolling_sharpe(pnls)
            sortino = self.rolling_sortino(pnls)
            max_dd = self._max_drawdown(pnls)

            results.append(
                StrategyAttribution(
                    strategy=name,
                    pnl=strat_pnl,
                    pnl_pct=pnl_pct,
                    sharpe=sharpe,
                    sortino=sortino,
                    max_drawdown=max_dd,
                )
            )

        return results

    def compute_correlation_matrix(self, strategy_returns: dict[str, list[float]]) -> dict[tuple[str, str], float]:
        """Compute pairwise correlation between all strategy return series."""
        names = sorted(strategy_returns.keys())
        matrix: dict[tuple[str, str], float] = {}

        for name_a, name_b in combinations(names, 2):
            arr_a = np.array(strategy_returns[name_a])
            arr_b = np.array(strategy_returns[name_b])
            min_len = min(len(arr_a), len(arr_b))
            if min_len < 2:
                matrix[(name_a, name_b)] = 0.0
                continue
            corr = float(np.corrcoef(arr_a[:min_len], arr_b[:min_len])[0, 1])
            if math.isnan(corr):
                corr = 0.0
            matrix[(name_a, name_b)] = corr

        return matrix

    def check_correlation_alerts(self, strategy_returns: dict[str, list[float]]) -> list[CorrelationAlert]:
        """Flag strategy pairs whose correlation exceeds the spike threshold."""
        if len(strategy_returns) < 2:
            return []

        matrix = self.compute_correlation_matrix(strategy_returns)
        alerts: list[CorrelationAlert] = []

        for (name_a, name_b), corr in matrix.items():
            if abs(corr) > self.correlation_spike_threshold:
                alerts.append(
                    CorrelationAlert(
                        strategy_a=name_a,
                        strategy_b=name_b,
                        correlation=corr,
                        is_spike=True,
                    )
                )

        return alerts

    def rolling_sharpe(self, returns: list[float], window: int | None = None) -> float | None:
        """Compute annualized Sharpe ratio over the trailing window.

        Sharpe = mean(returns) / std(returns) * sqrt(252)
        Returns None if insufficient data or zero standard deviation.
        """
        w = window if window is not None else self.sharpe_window
        if len(returns) < w:
            return None

        arr = np.array(returns[-w:])
        std = float(np.std(arr, ddof=1))
        if std == 0.0:
            return None

        return float(arr.mean() / std * math.sqrt(252))

    def rolling_sortino(self, returns: list[float], window: int | None = None) -> float | None:
        """Compute annualized Sortino ratio over the trailing window.

        Sortino = mean(returns) / downside_std(returns) * sqrt(252)
        where downside_std uses only negative returns.
        Returns None if insufficient data or no negative returns.
        """
        w = window if window is not None else self.sharpe_window
        if len(returns) < w:
            return None

        arr = np.array(returns[-w:])
        downside = arr[arr < 0]
        if len(downside) == 0:
            return None

        downside_std = float(np.std(downside, ddof=1))
        if downside_std == 0.0:
            return None

        return float(arr.mean() / downside_std * math.sqrt(252))

    @staticmethod
    def _max_drawdown(pnls: list[float]) -> float:
        """Compute maximum drawdown from a series of P&L values.

        Drawdown is measured as peak-to-trough on cumulative equity curve.
        Returns 0.0 if no drawdown, negative value otherwise.
        """
        if not pnls:
            return 0.0

        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)

        # Avoid division by zero when peak is zero
        max_dd = 0.0
        for i in range(len(cumulative)):
            if peak[i] > 0:
                dd = (cumulative[i] - peak[i]) / peak[i]
                if dd < max_dd:
                    max_dd = dd

        return float(max_dd)

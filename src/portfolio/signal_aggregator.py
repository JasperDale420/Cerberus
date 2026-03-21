"""Portfolio-level signal aggregation with IC-weighted conflict resolution."""

from __future__ import annotations

from collections import defaultdict

from src.core.domain import OrderSide, Signal


class SignalAggregator:
    """Resolves conflicts when multiple strategies fire on the same symbol.

    Uses trailing IC (information coefficient) to weight strategies and filters
    out symbols where conflicting signals cancel each other out.
    """

    def __init__(
        self,
        min_net_confidence: float = 0.3,
        correlation_penalty_threshold: float = 0.7,
    ) -> None:
        self.min_net_confidence = min_net_confidence
        self.correlation_penalty_threshold = correlation_penalty_threshold

    def aggregate(
        self,
        signals: list[Signal],
        strategy_ics: dict[str, float] | None = None,
        recent_signal_history: dict[str, list] | None = None,
    ) -> list[Signal]:
        """Aggregate signals, resolving per-symbol conflicts via IC weighting.

        Args:
            signals: Raw signals from all strategies.
            strategy_ics: Trailing IC per strategy name. Missing entries get equal weight.
            recent_signal_history: Per-strategy list of recent signal directions
                (1 for BUY, -1 for SELL) used for correlation penalty.

        Returns:
            Filtered list of non-conflicting signals.
        """
        if not signals:
            return []

        strategy_ics = strategy_ics or {}
        recent_signal_history = recent_signal_history or {}

        weights = self._compute_ic_weights(signals, strategy_ics)
        weights = self._apply_correlation_penalty(signals, weights, recent_signal_history)
        return self._resolve_conflicts(signals, weights)

    def _compute_ic_weights(self, signals: list[Signal], strategy_ics: dict[str, float]) -> dict[str, float]:
        """Compute per-strategy weight from trailing IC scores.

        Strategies without IC data receive equal weight (1.0).
        """
        strategies = {s.strategy for s in signals}
        weights: dict[str, float] = {}
        for strat in strategies:
            weights[strat] = strategy_ics.get(strat, 1.0)
        return weights

    def _resolve_conflicts(self, signals: list[Signal], weights: dict[str, float]) -> list[Signal]:
        """Group by symbol and keep only signals aligned with net direction."""
        by_symbol: dict[str, list[Signal]] = defaultdict(list)
        for sig in signals:
            by_symbol[sig.symbol].append(sig)

        result: list[Signal] = []
        for _symbol, group in by_symbol.items():
            if len(group) == 1:
                result.append(group[0])
                continue

            buy_weight = sum(weights.get(s.strategy, 1.0) for s in group if s.side == OrderSide.BUY)
            sell_weight = sum(weights.get(s.strategy, 1.0) for s in group if s.side == OrderSide.SELL)
            net = buy_weight - sell_weight
            total = buy_weight + sell_weight

            # Normalise net weight relative to total to get a confidence measure
            if total == 0:
                continue
            net_confidence = abs(net) / total

            if net_confidence < self.min_net_confidence:
                # Conflict too close to call -- filter all signals for this symbol
                continue

            winning_side = OrderSide.BUY if net > 0 else OrderSide.SELL
            for sig in group:
                if sig.side == winning_side:
                    result.append(sig)

        return result

    def _apply_correlation_penalty(
        self,
        signals: list[Signal],
        weights: dict[str, float],
        recent_signal_history: dict[str, list],
    ) -> dict[str, float]:
        """Discount correlated strategies so redundant signals don't dominate.

        If two strategies' recent signal histories have Pearson correlation above
        the threshold, the lower-IC strategy is penalised by 50%.
        """
        if not recent_signal_history:
            return weights

        strategies = sorted({s.strategy for s in signals})
        penalised = dict(weights)

        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                s1, s2 = strategies[i], strategies[j]
                h1 = recent_signal_history.get(s1)
                h2 = recent_signal_history.get(s2)
                if not h1 or not h2:
                    continue

                corr = self._pearson(h1, h2)
                if abs(corr) > self.correlation_penalty_threshold:
                    # Penalise the lower-IC strategy
                    if penalised.get(s1, 1.0) <= penalised.get(s2, 1.0):
                        penalised[s1] = penalised.get(s1, 1.0) * 0.5
                    else:
                        penalised[s2] = penalised.get(s2, 1.0) * 0.5

        return penalised

    @staticmethod
    def _pearson(x: list, y: list) -> float:
        """Compute Pearson correlation between two equal-length sequences."""
        n = min(len(x), len(y))
        if n < 2:
            return 0.0
        x, y = x[:n], y[:n]
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        denom = (var_x * var_y) ** 0.5
        if denom == 0:
            return 0.0
        return cov / denom

from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.strategies.base import BaseStrategy


class StatArbClusterStrategy(BaseStrategy):
    """
    Statistical Arbitrage Cluster Mean-Reversion Strategy

    Identifies symbols that have deviated significantly from their cluster's
    mean behavior (represented by cluster_residual) and trades the reversion.
    """
    name = "stat_arb_cluster"

    def __init__(self, config: dict[str, Any], logger: Any):
        super().__init__(config, logger)
        self.residual_threshold = float(self.config.get("residual_threshold", 2.0))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        if not self._require_min_bars(symbol_state, 10, log=False):
            return None

        if not self._check_cooldown(symbol, bar.time):
            return None

        features = symbol_state.meta.get("features")
        if not features:
            return None

        cluster_residual = getattr(features, "cluster_residual", 0.0)
        cluster_id = getattr(features, "cluster_id", 0)

        # If residual is highly positive, symbol is overperforming its cluster -> Short it
        # If residual is highly negative, symbol is underperforming its cluster -> Long it

        if abs(cluster_residual) < self.residual_threshold:
            return None

        # Reversion logic: trade opposite to the residual
        side = OrderSide.SELL if cluster_residual > 0 else OrderSide.BUY

        # For stat-arb, we generally expect quicker reversions. Let's use tighter stops/targets.
        atr = features.atr
        if atr <= 0:
            return None

        base_stop_dist = atr * float(self.config.get("stop_atr_multiplier", 1.0))
        base_target_dist = base_stop_dist * float(self.config.get("risk_reward_ratio", 1.5))

        # Stat arb is less dependent on overall market volatility regime, but still useful
        stop_dist = self._apply_regime_volatility_multiplier(base_stop_dist, market_state)
        target_dist = stop_dist * self._get_dynamic_risk_reward(base_target_dist / base_stop_dist, market_state)

        if side == OrderSide.BUY:
            stop_price = bar.close - stop_dist
            target_price = bar.close + target_dist
        else:
            stop_price = bar.close + stop_dist
            target_price = bar.close - target_dist

        signal = self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            entry_price=bar.close,
            meta={"cluster_residual": cluster_residual, "cluster_id": cluster_id}
        )

        self.logger.info(
            "Stat-Arb Cluster signal generated",
            symbol=symbol,
            side=side.value,
            residual=cluster_residual,
            cluster_id=cluster_id
        )
        return signal

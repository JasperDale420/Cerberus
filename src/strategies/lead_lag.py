from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class LeadLagStrategy(BaseStrategy):
    """
    Lead-Lag Cross-Asset Momentum Strategy.
    Exploits situations where a broader market index makes a sudden impulse
    move, but highly correlated components lag slightly behind.
    """

    name: str = "lead_lag_cross_asset"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_correlation = float(config.get("min_correlation", 0.80))
        self.index_spike_threshold = float(config.get("index_spike_threshold", 0.002))  # 0.2% jump
        self.stop_atr_multiplier = float(config.get("stop_atr_multiplier", 1.5))
        self.target_atr_multiplier = float(config.get("target_atr_multiplier", 3.0))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        if not self._check_cooldown(symbol, bar.time):
            return None

        features = symbol_state.meta.get("features")
        if not features:
            return None

        # Check for sufficient correlation to the index
        correlation = getattr(features, "correlation_to_index", 0.0)
        if correlation < self.min_correlation:
            return None

        # Check for sudden directional impulse in the index
        index_ret = market_state.index_return
        if abs(index_ret) < self.index_spike_threshold:
            return None

        # Determine direction based on index spike
        side = OrderSide.BUY if index_ret > 0 else OrderSide.SELL

        atr = float(symbol_state.meta.get("atr", bar.close * 0.02) or bar.close * 0.02)
        stop_dist = self.stop_atr_multiplier * atr
        target_dist = self.target_atr_multiplier * atr

        if side == OrderSide.BUY:
            stop_price = bar.close - stop_dist
            target_price = bar.close + target_dist
        else:
            stop_price = bar.close + stop_dist
            target_price = bar.close - target_dist

        self.last_signal_time[symbol] = bar.time

        self.logger.info(
            "lead_lag_cross_asset: entry signal",
            symbol=symbol,
            side=side.value,
            index_return=round(index_ret, 5),
            correlation=round(correlation, 2),
        )

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            size_hint=1.0,
            meta={"index_return": index_ret, "correlation": correlation},
        )

from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class OptionsFlowMomentumStrategy(BaseStrategy):
    """
    Trades alongside massive institutional options activity.
    Enters long/short positions mapping the targeted directional options flow.
    """

    name: str = "options_flow_momentum"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.flow_zscore_threshold = float(config.get("flow_zscore_threshold", 2.0))
        self.bias_threshold = float(config.get("bias_threshold", 0.3))
        self.dof_threshold = float(config.get("dof_threshold", 0.7))
        self.stop_loss_pct = float(config.get("stop_loss_pct", 0.015))
        self.take_profit_pct = float(config.get("take_profit_pct", 0.03))

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

        flow_zscore = getattr(features, "flow_zscore", 0.0)
        if flow_zscore < self.flow_zscore_threshold:
            return None

        dof_score = getattr(features, "dof_score", 0.0)
        if dof_score < self.dof_threshold:
            return None

        flow_bias = getattr(features, "flow_bias", 0.0)
        call_put_ratio = getattr(features, "call_put_ratio", 1.0)

        if flow_bias > self.bias_threshold and call_put_ratio > 1.2:
            side = OrderSide.BUY
        elif flow_bias < -self.bias_threshold and call_put_ratio < 0.8:
            side = OrderSide.SELL
        else:
            return None

        if side == OrderSide.BUY:
            stop_price = bar.close * (1.0 - self.stop_loss_pct)
            target_price = bar.close * (1.0 + self.take_profit_pct)
        else:
            stop_price = bar.close * (1.0 + self.stop_loss_pct)
            target_price = bar.close * (1.0 - self.take_profit_pct)

        base_conf = 0.5
        zscore_boost = min(0.3, (flow_zscore - self.flow_zscore_threshold) * 0.1)
        dof_boost = min(0.15, (dof_score - self.dof_threshold) * 0.5)
        confidence = min(0.95, base_conf + zscore_boost + dof_boost)

        self.last_signal_time[symbol] = bar.time

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            size_hint=confidence,
            meta={
                "flow_zscore": flow_zscore,
                "dof_score": dof_score,
                "flow_bias": flow_bias,
                "call_put_ratio": call_put_ratio,
            },
        )

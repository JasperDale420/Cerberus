from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class GammaPinningStrategy(BaseStrategy):
    """
    Capitalizes on market maker hedging dynamics around Zero Gamma Level (ZGL).
    When price drifts far from ZGL, assume it will be pinned back.
    Requires high overall net_gex to ensure MMs are active.
    """
    name: str = "gamma_pinning"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.gex_threshold = float(config.get("gex_threshold", 1_000_000.0))
        self.deviation_threshold = float(config.get("deviation_threshold", 0.015))
        self.stop_loss_pct = float(config.get("stop_loss_pct", 0.01))

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

        net_gex = getattr(features, "net_gex", 0.0)
        if abs(net_gex) < self.gex_threshold:
            return None

        gex_flip_dist = getattr(features, "gex_flip_dist", 0.0)
        zgl = bar.close + gex_flip_dist
        if zgl <= 0:
            return None

        deviation = (bar.close - zgl) / zgl

        if deviation > self.deviation_threshold:
            side = OrderSide.SELL
        elif deviation < -self.deviation_threshold:
            side = OrderSide.BUY
        else:
            return None

        if side == OrderSide.BUY:
            stop_price = bar.close * (1.0 - self.stop_loss_pct)
            target_price = zgl
            if target_price <= bar.close:
                return None
        else:
            stop_price = bar.close * (1.0 + self.stop_loss_pct)
            target_price = zgl
            if target_price >= bar.close:
                return None

        confidence = min(0.9, (abs(net_gex) / (self.gex_threshold * 10)) * 0.5 + 0.4)

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
                "net_gex": net_gex,
                "gex_flip_dist": gex_flip_dist,
                "zgl": zgl,
                "deviation": deviation
            }
        )

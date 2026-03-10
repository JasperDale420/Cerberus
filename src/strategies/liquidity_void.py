from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class LiquidityVoidStrategy(BaseStrategy):
    """
    Order Book Liquidity Void (Flash Impulse Fade) Strategy.
    Detects sudden, massive volume spikes combined with extreme price deviation
    from the mean (EMA20), and fades the move anticipating mean-reversion as
    liquidity is replenished by market makers.
    """

    name: str = "liquidity_void"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.volume_spike_multiplier = float(config.get("volume_spike_multiplier", 3.0))
        self.deviation_threshold = float(config.get("deviation_threshold", 0.02)) # 2% deviation
        self.stop_atr_multiplier = float(config.get("stop_atr_multiplier", 1.0))
        self.target_atr_multiplier = float(config.get("target_atr_multiplier", 2.0))

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

        # Check for volume spike
        avg_vol = getattr(features, "avg_volume", 0.0)

        if avg_vol <= 0 or bar.volume < (self.volume_spike_multiplier * avg_vol):
            return None

        # Check for extreme price deviation (Liquidity Void)
        deviation = getattr(features, "distance_from_ema20", 0.0)

        # We fade (mean-revert) the sudden flash impulse
        if deviation > self.deviation_threshold:
            side = OrderSide.SELL # Spiked up, we fade by shorting to capture reversion
        elif deviation < -self.deviation_threshold:
            side = OrderSide.BUY # Spiked down, we fade by going long
        else:
            return None

        atr = float(symbol_state.meta.get("atr", bar.close * 0.02) or bar.close * 0.02)
        # Tight stop for a volatile fade strategy
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
            "liquidity_void: fade entry signal",
            symbol=symbol,
            side=side.value,
            volume_multiplier=round(bar.volume / avg_vol, 2),
            deviation=round(deviation, 4),
        )

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            size_hint=1.0,
            meta={"deviation": deviation, "volume_spike": bar.volume / avg_vol},
        )

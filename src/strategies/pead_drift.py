from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class PEADStrategy(BaseStrategy):
    """
    Post-Earnings Announcement Drift (PEAD) Strategy.
    Exploits the slow market reaction and subsequent multi-day drift
    following a significant earnings surprise.
    """

    name: str = "pead_drift"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.surprise_threshold = float(config.get("surprise_threshold", 0.15))  # 15% surprise
        self.max_days_post_earnings = int(config.get("max_days_post_earnings", 5))
        self.min_days_post_earnings = int(config.get("min_days_post_earnings", 1))
        self.stop_atr_multiplier = float(config.get("stop_atr_multiplier", 2.0))
        self.target_atr_multiplier = float(config.get("target_atr_multiplier", 4.0))

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

        days_since = getattr(features, "days_since_earnings", -1)
        if not (self.min_days_post_earnings <= days_since <= self.max_days_post_earnings):
            return None

        surprise = getattr(features, "earnings_surprise", 0.0)

        if surprise > self.surprise_threshold:
            side = OrderSide.BUY
        elif surprise < -self.surprise_threshold:
            side = OrderSide.SELL
        else:
            return None

        # Ensure momentum aligns with the surprise direction (drift confirmation)
        ema_slope = getattr(features, "ema20_slope", 0.0)
        if side == OrderSide.BUY and ema_slope < 0:
            return None
        if side == OrderSide.SELL and ema_slope > 0:
            return None

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
            "pead_drift: entry signal",
            symbol=symbol,
            side=side.value,
            surprise=round(surprise, 4),
            days_since=days_since,
        )

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            size_hint=1.0,
            meta={"earnings_surprise": surprise, "days_since": days_since},
        )

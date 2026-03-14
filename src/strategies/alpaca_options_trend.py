from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class AlpacaOptionsTrend(BaseStrategy):
    name = "alpaca_options_trend"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.trend_threshold = float(config.get("trend_threshold", 1.5))

    def on_bar(self, symbol: str, bar: Bar, symbol_state: SymbolState, market_state: MarketState) -> Signal | None:
        if not self._check_cooldown(symbol, bar.time):
            return None

        # Trend detection based on price momentum exceeding threshold
        bars = list(symbol_state.bars) if symbol_state.bars else []
        if len(bars) >= 5:
            recent_return = (bar.close - bars[-5].close) / bars[-5].close if bars[-5].close > 0 else 0.0
            if abs(recent_return) > self.trend_threshold / 100.0:
                entry = bar.close
                stop = entry * 0.95
                target = entry * 1.05

                self.logger.info(
                    "Options trend signal detected",
                    symbol=symbol,
                    recent_return=recent_return,
                )

                self.last_signal_time[symbol] = bar.time
                return self._create_signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    bar=bar,
                    market_state=market_state,
                    stop_price=stop,
                    target_price=target,
                    size_hint=0.1,
                    meta={"tape_signal": "bull_sweeps"},
                )

        return None

from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class AlpacaOptionsArbitrage(BaseStrategy):
    name = "alpaca_options_arbitrage"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.min_spread_threshold = float(config.get("min_spread_threshold", 0.05))

    def on_bar(self, symbol: str, bar: Bar, symbol_state: SymbolState, market_state: MarketState) -> Signal | None:
        if not self._check_cooldown(symbol, bar.time):
            return None

        # Options arbitrage requires volatility dispersion signal
        if market_state.realized_vol > 0.15:
            entry = bar.close
            stop = entry * 0.98
            target = entry * 1.05

            self.logger.info("Options arbitrage opportunity detected", symbol=symbol)

            self.last_signal_time[symbol] = bar.time
            return self._create_signal(
                symbol=symbol,
                side=OrderSide.BUY,
                bar=bar,
                market_state=market_state,
                stop_price=stop,
                target_price=target,
                size_hint=0.1,
                meta={"arb_type": "volatility_dispersion"},
            )

        return None

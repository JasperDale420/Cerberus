from datetime import time
from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout Strategy.
    Trades the breakout of the first X minutes (e.g., 15 mins).
    """

    name = "orb"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.orb_start = time(9, 30)
        self.orb_end = time(9, 45)  # Can be config driven
        self.entry_window_end = time(10, 30)  # Don't enter after this

        # Pull from config or defaults
        self.orb_minutes = config.get("orb_minutes", 15)
        self.risk_reward = config.get("risk_reward", 2.0)
        self.stop_loss_pct = config.get(
            "stop_loss_pct", 0.005
        )  # Backup stop if candle range too small/large

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        if not bar:
            return None

        t = bar.time.time()

        # 1. Update Opening Range
        if self.orb_start <= t < self.orb_end:
            self._update_opening_range(symbol_state, bar)
            return None

        # 2. Mark Completion
        if t >= self.orb_end:
            symbol_state.indicators["orb_complete"] = True

        # 3. Check Breakout
        return self._check_breakout(symbol, bar, symbol_state, market_state)

    def _update_opening_range(self, symbol_state: SymbolState, bar: Bar):
        current_high = symbol_state.indicators.get("orb_high", float("-inf"))
        current_low = symbol_state.indicators.get("orb_low", float("inf"))

        symbol_state.indicators["orb_high"] = max(current_high, bar.high)
        symbol_state.indicators["orb_low"] = min(current_low, bar.low)
        symbol_state.indicators["orb_complete"] = False

    def _check_breakout(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        t = bar.time.time()

        if not symbol_state.indicators.get("orb_complete"):
            return None

        if t > self.entry_window_end:
            return None

        if symbol_state.position and symbol_state.position.strategy == self.name:
            return None

        orb_high = symbol_state.indicators.get("orb_high")
        orb_low = symbol_state.indicators.get("orb_low")

        if not orb_high or not orb_low:
            return None

        # Long Breakout
        if bar.close > orb_high and market_state.regime in [Regime.BULL, Regime.CHOP]:
            return self._create_signal(
                symbol, bar, OrderSide.BUY, orb_low, market_state, orb_high, orb_low
            )

        # Short Breakout
        if bar.close < orb_low and market_state.regime in [Regime.BEAR, Regime.CHOP]:
            return self._create_signal(
                symbol, bar, OrderSide.SELL, orb_high, market_state, orb_high, orb_low
            )

        return None

    def _create_signal(
        self,
        symbol: str,
        bar: Bar,
        side: OrderSide,
        stop_price: float,
        market_state: MarketState,
        orb_high: float,
        orb_low: float,
    ) -> Optional[Signal]:
        risk = abs(bar.close - stop_price)
        if risk <= 0:
            return None

        if side == OrderSide.BUY:
            target_price = bar.close + (risk * self.risk_reward)
        else:
            target_price = bar.close - (risk * self.risk_reward)

        return Signal(
            symbol=symbol,
            side=side,
            size_hint=0,
            entry_price=bar.close,
            stop_price=stop_price,
            target_price=target_price,
            strategy=self.name,
            regime=market_state.regime,
            generated_at=bar.time,
            meta={"orb_high": orb_high, "orb_low": orb_low},
            correlation_id=f"{self.name}-{symbol}-{bar.time.timestamp()}",
        )

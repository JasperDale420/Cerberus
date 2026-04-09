"""Daily Research Strategy Template — the file you modify.

Operates on daily bars. Starts empty. The autoresearch agent discovers
what works through research and iteration.

You have 15 iterations. Make each one count.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState  # noqa: F401
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv6cStrategy(BaseStrategy):
    name = "daily_research_v6c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.min_bars = int(config.get("min_bars", 25))
        self.allow_overnight = True

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        # === YOUR SIGNAL LOGIC HERE ===
        # Available: bar.open/high/low/close/volume/vwap/time
        # symbol_state.bars — deque of recent daily bars
        # symbol_state.meta — regime_labels, near_earnings, near_fomc, opex_week
        # self._create_signal(symbol, OrderSide.BUY, bar, market_state, stop, target, meta={})

        return None

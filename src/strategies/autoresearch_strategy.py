"""Autoresearch Strategy — the file you modify.

This is the ONLY file the autoresearch agent edits. Everything else is frozen.
Start simple. Iterate. The evaluation is fixed at ~30 min per run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState  # noqa: F401
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class AutoresearchStrategy(BaseStrategy):
    name = "autoresearch_strategy"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        # Read params from config (defined in strategies.yaml)
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.min_bars = int(config.get("min_bars", 30))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # Gate: cooldown and minimum bars
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        # === YOUR SIGNAL LOGIC HERE ===
        # This is the baseline — it does nothing. Make it trade.
        #
        # Available data:
        #   bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap
        #   symbol_state.bars — deque of recent bars
        #   symbol_state.indicators — dict of precomputed EMA, RSI, ATR, BB
        #   symbol_state.meta.get("regime_labels") — regime_trend, regime_vol, etc.
        #
        # To enter a trade, return:
        #   self._create_signal(symbol, bar, OrderSide.BUY, stop_price, target_price, meta={})

        return None

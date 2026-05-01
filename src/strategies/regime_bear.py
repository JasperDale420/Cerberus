"""
RegimeBear Strategy

Bear market specialist: shorts RSI bounces in downtrends.
Designed to excel in DOWN+HIGH (bear + high vol) regime.
"""

from __future__ import annotations

from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, SymbolState, TrendRegime, VolRegime
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.strategies.base import BaseStrategy


class RegimeBearStrategy(BaseStrategy):
    """Short dead-cat bounces in DOWN trend; RSI gate is advisory only."""

    name: str = "regime_bear"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)
        self.higher_tf_alignment = False

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.rsi_short_entry = float(config.get("rsi_short_entry", 55.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.high_vol_target_mult = float(config.get("high_vol_target_mult", 1.5))
        self.min_bars = int(config.get("min_bars", 20))

    def on_bar(self, symbol: str, bar: Bar, symbol_state: SymbolState, market_state: MarketState) -> None:
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, self.min_bars):
            return None
        if self.is_past_hard_stop(bar.time):
            return None

        snapshot = market_state.regime_snapshot
        if snapshot is None or snapshot.trend != TrendRegime.DOWN:
            return None

        mtf = MultiTimeframeAnalyzer(symbol_state)
        rsi = mtf.get_rsi("1m", 14)
        # RSI is advisory: only block entry if RSI is present and NOT elevated
        if rsi is not None and rsi <= self.rsi_short_entry:
            return None

        atr = mtf.get_atr("1m", 14)
        stop_distance = atr * self.stop_atr_mult if (atr and atr > 0) else bar.close * 0.01

        vol = snapshot.vol
        target_mult = (
            self.target_atr_mult * self.high_vol_target_mult
            if vol in (VolRegime.HIGH, VolRegime.SHOCK)
            else self.target_atr_mult
        )
        target_distance = atr * target_mult if (atr and atr > 0) else bar.close * 0.02

        return self._create_signal(
            symbol=symbol,
            side=OrderSide.SELL,
            bar=bar,
            market_state=market_state,
            stop_price=bar.close + stop_distance,
            target_price=bar.close - target_distance,
            meta={"rsi_1m": round(rsi, 2) if rsi else None, "trend": "DOWN", "vol": str(vol)},
        )

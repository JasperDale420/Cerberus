"""
RegimeBear Strategy

Bear market specialist: shorts RSI bounces in downtrends and fades extremes.
Designed to excel in DOWN+HIGH (bear + high vol) regime.

Logic:
- DOWN trend: short when RSI is elevated (bounce into resistance)
- UP trend:   long when RSI is depressed (dip buy fallback)
- FLAT trend: fade RSI extremes in both directions
- HIGH/SHOCK vol: widen take-profit target for larger moves
"""

from __future__ import annotations

from typing import Any

from src.core.domain import (
    Bar,
    MarketState,
    OrderSide,
    SymbolState,
    TrendRegime,
    VolRegime,
)
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.strategies.base import BaseStrategy


class RegimeBearStrategy(BaseStrategy):
    """
    Bear market specialist.

    Entry SHORT: trend DOWN and RSI > rsi_short_entry (dead-cat bounce)
    Entry LONG:  trend UP and RSI < rsi_long_entry (dip in uptrend)
    Entry FLAT:  RSI > rsi_flat_high → short, RSI < rsi_flat_low → long

    In HIGH or SHOCK vol the take-profit multiplier is scaled up to capture
    the wider intraday swings characteristic of the DOWN+HIGH regime.
    """

    name: str = "regime_bear"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)
        # Disable higher-TF EMA alignment gate — regime logic handles direction
        self.higher_tf_alignment = False

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.rsi_short_entry = float(config.get("rsi_short_entry", 55.0))
        self.rsi_long_entry = float(config.get("rsi_long_entry", 45.0))
        self.rsi_flat_high = float(config.get("rsi_flat_high", 65.0))
        self.rsi_flat_low = float(config.get("rsi_flat_low", 35.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.high_vol_target_mult = float(config.get("high_vol_target_mult", 1.5))
        self.min_bars = int(config.get("min_bars", 20))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> None:
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, self.min_bars):
            return None
        if self.is_past_hard_stop(bar.time):
            return None

        mtf = MultiTimeframeAnalyzer(symbol_state)
        rsi = mtf.get_rsi("1m", 14)
        atr = mtf.get_atr("1m", 14)

        if rsi is None or atr is None or atr <= 0:
            return None

        snapshot = market_state.regime_snapshot
        trend = snapshot.trend if snapshot is not None else None
        vol = snapshot.vol if snapshot is not None else None

        # Determine direction from trend + RSI
        side: OrderSide | None = None
        if trend == TrendRegime.DOWN:
            if rsi > self.rsi_short_entry:
                side = OrderSide.SELL
        elif trend == TrendRegime.UP:
            if rsi < self.rsi_long_entry:
                side = OrderSide.BUY
        else:
            # FLAT or unknown: fade RSI extremes
            if rsi > self.rsi_flat_high:
                side = OrderSide.SELL
            elif rsi < self.rsi_flat_low:
                side = OrderSide.BUY

        if side is None:
            return None

        # Widen target in high/shock vol (DOWN+HIGH specialist edge)
        target_mult = self.target_atr_mult
        if vol in (VolRegime.HIGH, VolRegime.SHOCK):
            target_mult = self.target_atr_mult * self.high_vol_target_mult

        stop_distance = atr * self.stop_atr_mult
        if side == OrderSide.SELL:
            stop_price = bar.close + stop_distance
            target_price = bar.close - atr * target_mult
        else:
            stop_price = bar.close - stop_distance
            target_price = bar.close + atr * target_mult

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={
                "rsi_1m": round(rsi, 2),
                "atr_1m": round(atr, 6),
                "trend": str(trend),
                "vol": str(vol),
                "target_mult": round(target_mult, 2),
            },
        )

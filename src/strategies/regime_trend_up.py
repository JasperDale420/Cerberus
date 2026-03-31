from __future__ import annotations

from typing import Any

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.strategies.base import BaseStrategy


class RegimeTrendUpStrategy(BaseStrategy):
    """UP+NORMAL regime specialist: buy pullbacks in uptrends.

    4-factor entry using 1m indicators (available after 50 bars = 50 min):
    1. 1m EMA20 > EMA50 (trend is up on intraday timeframe)
    2. Price within pullback_pct of 1m EMA20 (buy the dip)
    3. Price above VWAP (momentum intact)
    4. RSI in range [rsi_min, rsi_max] — not overbought, not in freefall

    ATR-based stop below entry, 2.5R target.
    """

    name: str = "regime_trend_up"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.pullback_pct = float(config.get("pullback_pct", 0.02))  # 2% window around EMA20
        self.rsi_max = float(config.get("rsi_max", 70.0))
        self.rsi_min = float(config.get("rsi_min", 30.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_rr = float(config.get("target_rr", 2.5))
        self.time_window_start = time_utils.parse_time_string(str(config.get("time_window_start", "09:35")))
        self.time_window_end = time_utils.parse_time_string(str(config.get("time_window_end", "15:45")))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        if not self._is_evaluation_bar(bar):
            return None

        t = time_utils.get_eastern_time_of_day(bar.time)
        if not (self.time_window_start <= t <= self.time_window_end):
            return None

        if not self._check_cooldown(symbol, bar.time):
            return None

        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        if self.is_past_hard_stop(bar.time):
            return None

        # Already in a position — don't stack
        if symbol_state.position is not None:
            return None

        mtf = MultiTimeframeAnalyzer(symbol_state)

        # 1. Trend confirmed on 1m: EMA20 > EMA50
        ema20 = mtf.get_ema("1m", 20)
        ema50 = mtf.get_ema("1m", 50)
        if ema20 is None or ema50 is None:
            return None
        if ema20 <= ema50:
            return None

        # 2. Price within pullback_pct of EMA20 (buy the dip, not the rip)
        dist_pct = (bar.close - ema20) / ema20
        if dist_pct < -self.pullback_pct or dist_pct > self.pullback_pct:
            return None

        # 3. Price above VWAP (momentum still up)
        vwap_dist = mtf.get_vwap_distance("1m")
        if vwap_dist is not None and vwap_dist < 0:
            return None

        # 4. RSI not overbought and not in freefall
        rsi = mtf.get_rsi("1m", 14)
        if rsi is not None and (rsi > self.rsi_max or rsi < self.rsi_min):
            return None

        # ATR stop and target
        atr = mtf.get_atr("1m", 14)
        if atr is None or atr <= 0:
            return None

        stop_price = bar.close - self.stop_atr_mult * atr
        stop_distance = bar.close - stop_price
        target_price = bar.close + self.target_rr * stop_distance

        meta = {
            "ema20_1m": round(ema20, 4),
            "ema50_1m": round(ema50, 4),
            "dist_pct": round(dist_pct, 5),
            "rsi_1m": round(rsi, 2) if rsi is not None else None,
            "atr_1m": round(atr, 4),
            "vwap_dist": round(vwap_dist, 5) if vwap_dist is not None else None,
        }

        return self._create_signal(
            symbol=symbol,
            side=OrderSide.BUY,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            size_hint=1.0,
            meta=meta,
        )

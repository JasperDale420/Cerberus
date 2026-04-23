from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.strategies.base import BaseStrategy


class UpNormalStrategy(BaseStrategy):
    name = "up_normal_strategy"

    def _set_params(self, c: dict) -> None:
        super()._set_params(c)
        self.sm, self.tm = float(c.get("stop_atr_mult", 1.66)), float(c.get("target_atr_mult", 4.5))
        self.vmin, self.min_bars = float(c.get("min_vwap_dist", 0.001)), int(c.get("min_bars", 30))

    def on_bar(self, sym: str, bar: Bar, ss: SymbolState, ms: MarketState) -> Signal | None:
        if not self._check_cooldown(sym, bar.time) or not self._require_min_bars(ss, self.min_bars):
            return None
        h = getattr(bar.time, "hour", 10)
        if (h == 9 and bar.time.minute < 45) or (h == 15 and bar.time.minute >= 30):
            return None
        ema, atr = (mtf := MultiTimeframeAnalyzer(ss)).get_ema("1m", 20), mtf.get_atr("1m", 14)
        if mtf.get_trend_alignment(OrderSide.BUY) < 0.8 or ema is None or atr is None or atr <= 0:
            return None
        vd, d = mtf.get_vwap_distance("1m"), ema - bar.close
        if (vd is not None and vd < self.vmin) or d < 0.1 * atr or d > 0.4 * atr:
            return None
        self.last_signal_time[sym] = bar.time
        return self._create_signal(
            sym, OrderSide.BUY, bar, ms, bar.close - self.sm * atr, bar.close + self.tm * atr, size_hint=0.5 + d / atr
        )

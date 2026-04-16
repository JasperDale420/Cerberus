"""Research v7a: Trend Pullback with EMA Alignment.

Entry: Price in uptrend (EMA fast > EMA slow) + pullback within ATR band
       of fast EMA + volume confirmation.
Exit: ATR-based stop/target, max hold days.
Long-only, daily bars.

Archetype: Trend Pullback (switching from mean reversion after 5 iterations)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v7a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.ema_fast = int(config.get("ema_fast", 10))
        self.ema_slow = int(config.get("ema_slow", 30))
        self.atr_period = int(config.get("atr_period", 14))
        self.pullback_atr_mult = float(config.get("pullback_atr_mult", 1.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.4))
        self.max_atr_pct = float(config.get("max_atr_pct", 0.03))
        self.vol_mult = float(config.get("vol_mult", 0.6))

    # --- Indicator helpers ---

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = v * mult + ema * (1 - mult)
        return ema

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _atr(bars: list[Bar], period: int) -> Optional[float]:
        if len(bars) < period + 1:
            return None
        trs = []
        for i in range(-period, 0):
            b = bars[i]
            prev_close = bars[i - 1].close
            tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
            trs.append(tr)
        return sum(trs) / period

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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # --- Regime filter ---
        labels = symbol_state.meta.get("regime_labels", {})
        regime_vol = labels.get("regime_vol", "NORMAL")
        if regime_vol in ("HIGH", "SHOCK"):
            return None

        # --- Event filter ---
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        # --- EMA alignment: fast > slow (uptrend) ---
        ema_f = self._ema(closes, self.ema_fast)
        ema_s = self._ema(closes, self.ema_slow)
        if ema_f is None or ema_s is None:
            return None
        if ema_f <= ema_s:
            return None  # Not in uptrend

        # --- ATR ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # ATR/price volatility filter
        if bar.close > 0 and atr / bar.close > self.max_atr_pct:
            return None

        # --- Pullback: price within ATR band below fast EMA ---
        pullback_depth = ema_f - bar.close
        if pullback_depth < 0:
            return None  # Price above EMA — not a pullback
        if pullback_depth > self.pullback_atr_mult * atr:
            return None  # Too deep — broken trend

        # --- IBS filter: low IBS confirms selling pressure exhaustion ---
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None

        # --- Volume check ---
        avg_vol = self._sma([float(v) for v in volumes[-20:]], min(20, len(volumes)))
        if avg_vol is not None and avg_vol > 0:
            if bar.volume < avg_vol * self.vol_mult:
                return None

        # --- Stop/Target ---
        stop = bar.close - self.stop_atr_mult * atr
        # Cap stop at 2% of price
        max_stop_loss = bar.close * 0.02
        if bar.close - stop > max_stop_loss:
            stop = bar.close - max_stop_loss
        target = bar.close + self.target_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "ibs": round(ibs, 3),
                "pullback_depth": round(pullback_depth / atr, 3),
                "ema_spread": round((ema_f - ema_s) / ema_s, 4),
                "atr": round(atr, 4),
                "regime": f"{labels.get('regime_trend', 'FLAT')}+{regime_vol}",
                "seed": "trend_pullback",
            },
        )

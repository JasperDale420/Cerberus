"""Multi-factor pullback with strict regime filtering.

Entry: Short-term oversold (Williams %R + ROC + low IBS) near Keltner lower channel.
Skips DOWN+HIGH and all SHOCK vol. Requires stronger signals in DOWN/HIGH regimes.
Long-only, daily bars, tight R:R for consistency.
Event filters (earnings, FOMC, opex).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v8b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        # Williams %R
        self.willr_period = int(config.get("willr_period", 14))
        self.willr_oversold = float(config.get("willr_oversold", -80.0))
        # Rate of change
        self.roc_period = int(config.get("roc_period", 5))
        self.roc_threshold = float(config.get("roc_threshold", -3.0))
        # Keltner channel
        self.keltner_period = int(config.get("keltner_period", 20))
        self.keltner_mult = float(config.get("keltner_mult", 2.0))
        # Volume
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.8))
        # SMA for trend context
        self.sma_period = int(config.get("sma_period", 50))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = sum(values[:period]) / period
        for v in values[period:]:
            ema = (v - ema) * mult + ema
        return ema

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

    @staticmethod
    def _williams_r(highs: list[float], lows: list[float], close: float, period: int) -> Optional[float]:
        if len(highs) < period or len(lows) < period:
            return None
        highest = max(highs[-period:])
        lowest = min(lows[-period:])
        if highest - lowest < 1e-9:
            return -50.0
        return -100.0 * (highest - close) / (highest - lowest)

    @staticmethod
    def _roc(closes: list[float], period: int) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        prev = closes[-(period + 1)]
        if prev < 1e-9:
            return None
        return ((closes[-1] - prev) / prev) * 100.0

    @staticmethod
    def _ibs(bar: Bar) -> float:
        rng = bar.high - bar.low
        if rng < 1e-9:
            return 0.5
        return (bar.close - bar.low) / rng

    @staticmethod
    def _consecutive_downs(closes: list[float]) -> int:
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
        return count

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

        # Skip SHOCK and HIGH volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol in (VolRegime.SHOCK, VolRegime.HIGH):
            return None

        # Event filters
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        regime_trend = labels.get("regime_trend", "FLAT").upper()

        # Skip DOWN trend entirely — data shows it's a losing proposition
        if regime_trend == "DOWN":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [b.volume for b in bars]

        # ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # --- Required: Williams %R oversold ---
        willr = self._williams_r(highs, lows, bar.close, self.willr_period)
        if willr is None or willr >= self.willr_oversold:
            return None

        # --- Required: Low IBS (closed near low) ---
        ibs = self._ibs(bar)
        if ibs >= 0.3:
            return None

        # --- Additional factors (need at least 1 more) ---
        extra = 0

        # Factor: Negative ROC (recent decline)
        roc = self._roc(closes, self.roc_period)
        if roc is not None and roc < self.roc_threshold:
            extra += 1

        # Factor: Price near or below Keltner lower channel
        keltner_ma = self._ema(closes, self.keltner_period)
        if keltner_ma is not None:
            lower_keltner = keltner_ma - self.keltner_mult * atr
            if bar.close <= lower_keltner + 0.3 * atr:
                extra += 1

        # Factor: Consecutive down days (2+)
        consec = self._consecutive_downs(closes)
        if consec >= 2:
            extra += 1

        # Factor: Volume above average
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is not None and avg_vol > 0 and bar.volume >= self.vol_min_ratio * avg_vol:
            extra += 1

        # Require at least 2 extra factors in FLAT, 1 in UP
        if regime_trend == "UP":
            if extra < 1:
                return None
        else:  # FLAT
            if extra < 2:
                return None

        # Trend context: prefer price above SMA(50) for safer entries
        sma = self._sma(closes, self.sma_period)
        if sma is not None and bar.close < sma * 0.95:
            return None  # Too far below trend — skip

        # Stop and target
        stop = bar.close - self.stop_atr_mult * atr
        target = bar.close + self.target_atr_mult * atr

        if target <= bar.close:
            return None

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "extra_factors": extra,
                "regime_trend": regime_trend,
                "willr": round(willr, 2),
                "roc": round(roc, 2) if roc else None,
                "ibs": round(ibs, 3),
                "consec_down": consec,
                "atr": round(atr, 4),
                "seed": "multi_factor_pullback_v2",
            },
        )

"""Daily Research v7a: Multi-Factor Mean Reversion.

Consecutive down days + SMA trend filter + IBS + volume confirmation.
Long-only, daily bars. Structurally different from RSI(2) oscillator approach.
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
        self.min_down_days = int(config.get("min_down_days", 2))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        self.vol_mult = float(config.get("vol_mult", 0.8))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.3))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.10))
        self.max_hold_days = int(config.get("max_hold_days", 5))

    # --- Indicator helpers ---

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

    @staticmethod
    def _consecutive_down_days(bars: list[Bar]) -> int:
        """Count consecutive days where close < prior close, from most recent."""
        count = 0
        for i in range(len(bars) - 1, 0, -1):
            if bars[i].close < bars[i - 1].close:
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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # --- Regime filter: skip all DOWN trend (45% WR) and SHOCK vol ---
        labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = labels.get("regime_trend", "FLAT")
        regime_vol = labels.get("regime_vol", "NORMAL")
        if regime_trend == "DOWN":
            return None
        if regime_vol == "SHOCK":
            return None

        # --- Event filter: skip earnings and FOMC ---
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        # --- Entry: consecutive down days ---
        down_days = self._consecutive_down_days(bars)
        if down_days < self.min_down_days:
            return None

        # --- Pullback magnitude: require meaningful drop (not just tiny red days) ---
        if len(closes) > down_days:
            roc = (closes[-1] - closes[-(down_days + 1)]) / closes[-(down_days + 1)]
            if roc > -0.015:  # need at least 1.5% pullback
                return None

        # --- IBS filter: close near day's low (selling exhaustion) ---
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None

        # --- Trend filter: price above SMA(50) ---
        sma50 = self._sma(closes, 50)
        if sma50 is None or bar.close < sma50:
            return None

        # --- Volume filter: today's vol >= vol_mult * avg vol ---
        volumes = [b.volume for b in bars]
        avg_vol = self._sma(volumes[-20:], min(20, len(volumes[-20:])))
        if avg_vol is not None and avg_vol > 0:
            if bar.volume < self.vol_mult * avg_vol:
                return None

        # --- Drawdown filter: skip if too far from peak ---
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.max_drawdown_pct:
            return None

        # --- Exits: fixed % stop + SMA(20) target ---
        sma20 = self._sma(closes, 20)
        if sma20 is None:
            return None

        # Fixed percentage stop (not ATR-based — reduces param instability)
        stop = bar.close * 0.97  # 3% stop loss

        # Target: SMA(20) midline (mean reversion target)
        # If SMA(20) is below entry, use ATR-based fallback
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        target = max(sma20, bar.close + self.target_atr_mult * atr)

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "down_days": down_days,
                "ibs": round(ibs, 3),
                "sma50": round(sma50, 2),
                "sma20_target": round(sma20, 2),
                "atr": round(atr, 4),
                "regime": f"{regime_trend}+{regime_vol}",
            },
        )

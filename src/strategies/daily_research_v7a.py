"""Research v7a: Regime-Adaptive Consecutive Down Mean Reversion.

Entry: consecutive down closes + IBS low + drawdown filter.
Regime adaptation: stricter entry in DOWN markets, looser in UP.
Exit: ATR-based stop/target, max hold days.
Long-only, daily bars.
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
        self.vol_mult = float(config.get("vol_mult", 0.6))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.3))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.12))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        self.sma_period = int(config.get("sma_period", 20))

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
    def _consecutive_down(closes: list[float], min_days: int) -> bool:
        """Check if the last min_days closes were each lower than the prior."""
        if len(closes) < min_days + 1:
            return False
        for i in range(-min_days, 0):
            if closes[i] >= closes[i - 1]:
                return False
        return True

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

        # --- Regime labels ---
        labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = labels.get("regime_trend", "FLAT")
        regime_vol = labels.get("regime_vol", "NORMAL")

        # Skip SHOCK vol entirely
        if regime_vol == "SHOCK":
            return None

        # --- Regime-adaptive entry: require more down days in DOWN markets ---
        required_down = self.min_down_days
        ibs_thresh = self.ibs_threshold
        if regime_trend == "DOWN":
            required_down = max(self.min_down_days + 1, 3)  # At least 3 down days
            ibs_thresh = min(self.ibs_threshold, 0.25)  # Stricter IBS

        # --- Consecutive down days ---
        if not self._consecutive_down(closes, required_down):
            return None

        # --- IBS (Internal Bar Strength): must be low ---
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= ibs_thresh:
            return None

        # --- Mean reversion target: use SMA as target ---
        sma = self._sma(closes, self.sma_period)
        if sma is None:
            return None
        # Price must be below SMA (mean reversion opportunity)
        if bar.close >= sma:
            return None

        # --- Drawdown filter ---
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        drawdown_pct = self.max_drawdown_pct
        if regime_trend == "DOWN":
            drawdown_pct = min(self.max_drawdown_pct, 0.08)  # Tighter in downtrends
        if peak > 0 and (peak - bar.close) / peak > drawdown_pct:
            return None

        # --- ATR for stop/target ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        # Target: closer of SMA or ATR-based target
        atr_target = bar.close + self.target_atr_mult * atr
        target = min(sma, atr_target)

        # Ensure positive risk/reward
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
                "ibs": round(ibs, 3),
                "down_days": required_down,
                "atr": round(atr, 4),
                "drawdown_from_peak": round((peak - bar.close) / peak, 4),
                "regime": f"{regime_trend}+{regime_vol}",
                "seed": "mean_reversion",
            },
        )

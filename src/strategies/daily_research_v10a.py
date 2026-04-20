"""Daily Research v10a — Oversold Bounce in Non-Downtrend.

Multi-factor oversold detection (consecutive down + IBS + pullback depth).
Only trades in UP/FLAT regimes. Targets SMA mean reversion.
Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v10a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        # Entry conditions
        self.consec_down_min = int(config.get("consec_down_min", 2))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        self.min_score = float(config.get("min_score", 3.0))
        # Mean reversion target
        self.sma_period = int(config.get("sma_period", 20))
        # Risk management
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        # Drawdown filter
        self.drawdown_max = float(config.get("drawdown_max", 0.12))

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
    def _count_consecutive_down(closes: list[float]) -> int:
        """Count consecutive down closes from the end."""
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

        # Skip DOWN regime entirely
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "FLAT")
        if trend == "DOWN":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # SMA for target and trend confirmation
        sma = self._sma(closes, self.sma_period)
        if sma is None:
            return None

        # Target must be above current price (room to revert)
        if sma <= bar.close:
            return None

        # Composite oversold score
        score = 0.0

        # Factor 1: Consecutive down days (1 point per day above minimum)
        consec_down = self._count_consecutive_down(closes)
        if consec_down >= self.consec_down_min:
            score += min(consec_down - self.consec_down_min + 1, 3)

        # Factor 2: IBS (close near low of day)
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs < self.ibs_threshold:
            score += 1.0
            if ibs < 0.15:
                score += 0.5

        # Factor 3: Distance below SMA (z-score proxy)
        distance_pct = (sma - bar.close) / sma
        if distance_pct >= 0.02:
            score += 1.0
            if distance_pct >= 0.04:
                score += 0.5

        # Minimum score gate
        if score < self.min_score:
            return None

        # Drawdown filter: not in freefall
        lookback_highs = [b.high for b in bars[-40:]]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.drawdown_max:
            return None

        # ATR-based stop
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        target = sma  # Revert to mean

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "score": round(score, 1),
                "consec_down": consec_down,
                "ibs": round(ibs, 3),
                "dist_from_sma": round(distance_pct, 4),
                "atr": round(atr, 4),
                "trend": trend,
                "seed": "mean_reversion",
            },
        )

"""IBS Mean Reversion — fade daily closes near the low.

Internal Bar Strength = (close - low) / (high - low).
Low IBS (close near low) is a robust next-day reversal signal.
Long only above SMA(200). Skip earnings/FOMC/SHOCK.
Target: SMA(20) midline. Stop: ATR-based.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv7bStrategy(BaseStrategy):
    name = "daily_research_v7b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.sma_long_period = int(config.get("sma_long_period", 200))
        self.sma_mid_period = int(config.get("sma_mid_period", 20))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.2))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_max = float(config.get("rsi_max", 50.0))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _rsi(closes: list[float], period: int) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        gains = []
        losses = []
        for i in range(-period, 0):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss < 1e-9:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

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

        # Skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Skip earnings and FOMC
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Need enough bars for SMA(200) — but if not enough, use SMA(50) as fallback
        sma_long = self._sma(closes, self.sma_long_period)
        if sma_long is None:
            # Fallback: use SMA(50) if not enough data for 200
            sma_long = self._sma(closes, 50)
            if sma_long is None:
                return None

        # Long-term trend filter: price must be above long SMA
        if bar.close <= sma_long:
            return None

        # IBS: Internal Bar Strength
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range

        # Entry: IBS below threshold (close near the low = oversold)
        if ibs >= self.ibs_threshold:
            return None

        # RSI filter: not overbought (helps avoid catching falling knives)
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi > self.rsi_max:
            return None

        # Target: SMA(20) midline
        sma_mid = self._sma(closes, self.sma_mid_period)
        if sma_mid is None:
            return None

        # Only enter if target is above current price (room to profit)
        if sma_mid <= bar.close:
            return None

        # ATR for stop
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        target = bar.close + self.target_atr_mult * atr

        # Use SMA target if it's closer (tighter target = more consistent)
        target = min(target, sma_mid)

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "ibs": round(ibs, 4),
                "rsi": round(rsi, 2),
                "sma_mid": round(sma_mid, 2),
                "atr": round(atr, 4),
                "seed": "ibs_mean_reversion",
            },
        )

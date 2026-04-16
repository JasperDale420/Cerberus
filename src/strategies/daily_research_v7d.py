"""IBS Trend Pullback — symmetric RR + dual SMA + 2 down days.

Regime labels don't reliably prevent DOWN+HIGH window trades.
Switch to pure price-based trend: close > SMA(20) AND SMA(20) > SMA(50).
Dual SMA ensures we're in a real uptrend, not a bear market rally.

Entry: close > SMA(20) AND SMA(20) > SMA(50) AND IBS < 0.25
       AND 2 consecutive down days.
Stop = target = 1.5 ATR (symmetric).
Exclude leveraged ETFs. Skip SHOCK vol, earnings, FOMC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy

_EXCLUDED = {"VXX", "UVXY", "SQQQ", "TQQQ", "SPXU", "SPXS", "SDOW", "LABU", "LABD"}


class dailyresearchv7dStrategy(BaseStrategy):
    name = "daily_research_v7d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.sma_fast = int(config.get("sma_fast", 20))
        self.sma_slow = int(config.get("sma_slow", 50))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.25))
        self.consec_down_days = int(config.get("consec_down_days", 2))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_mult = float(config.get("atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(closes: list[float], period: int) -> Optional[float]:
        if len(closes) < period:
            return None
        return sum(closes[-period:]) / period

    @staticmethod
    def _ibs(bar: Bar) -> Optional[float]:
        rng = bar.high - bar.low
        if rng < 1e-9:
            return None
        return (bar.close - bar.low) / rng

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
        if symbol in _EXCLUDED:
            return None

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

        if len(closes) < self.min_bars:
            return None

        # Min price filter
        if bar.close < 5.0:
            return None

        # Dual SMA trend filter: close > SMA(20) AND SMA(20) > SMA(50)
        sma_fast = self._sma(closes, self.sma_fast)
        sma_slow = self._sma(closes, self.sma_slow)
        if sma_fast is None or sma_slow is None:
            return None
        if bar.close <= sma_fast or sma_fast <= sma_slow:
            return None

        # IBS oversold signal
        ibs = self._ibs(bar)
        if ibs is None or ibs >= self.ibs_threshold:
            return None

        # Consecutive down days filter
        consec = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                consec += 1
            else:
                break
        if consec < self.consec_down_days:
            return None

        # ATR for symmetric stop and target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # ATR/price filter: skip dead stocks
        if atr / bar.close < 0.005:
            return None

        # Symmetric stop and target
        stop = bar.close - self.atr_mult * atr
        target = bar.close + self.atr_mult * atr

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
                "sma_fast": round(sma_fast, 2),
                "sma_slow": round(sma_slow, 2),
                "atr_pct": round(atr / bar.close, 4),
                "seed": "ibs_dual_sma_symmetric",
            },
        )

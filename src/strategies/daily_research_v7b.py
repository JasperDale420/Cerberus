"""Multi-Factor Trend Pullback — Consecutive Down Days + IBS + Trend Filter.

Buy oversold pullbacks within established uptrends. Uses multiple confirmation
factors: consecutive down days, RSI(2) oversold, IBS (close near day low =
selling exhaustion), and SMA trend alignment. Regime-filtered.

Entry: consecutive_down_days down closes + RSI(2) < threshold + IBS < 0.35
       + price > SMA(50) trend filter
Exit: ATR-based stop/target, max 5-day hold
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v7b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        # Trend filter
        self.sma_slow_period = int(config.get("sma_slow_period", 50))
        self.ema_fast_period = int(config.get("ema_fast_period", 20))
        # Optimizer-tuned params
        self.consec_down_days = int(config.get("consec_down_days", 2))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_max = float(config.get("rsi_max", 20.0))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.03))
        # IBS filter
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        # Volume filter
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.5))

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

    @staticmethod
    def _ibs(bar: Bar) -> float:
        """Internal Bar Strength: (close - low) / (high - low). 0 = close at low, 1 = close at high."""
        rng = bar.high - bar.low
        if rng < 1e-9:
            return 0.5
        return (bar.close - bar.low) / rng

    @staticmethod
    def _consecutive_down(closes: list[float], n: int) -> bool:
        """Check if last n closes were each lower than the prior close."""
        if len(closes) < n + 1:
            return False
        for i in range(-n, 0):
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

        # Regime filter: skip SHOCK
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Regime filter: skip DOWN+HIGH
        labels = symbol_state.meta.get("regime_labels", {})
        regime = labels.get("regime", "")
        if regime in ("DOWN+HIGH", "DOWN+SHOCK"):
            return None

        # Event filter
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # 1. Trend filter: price > SMA(50)
        sma50 = self._sma(closes, self.sma_slow_period)
        if sma50 is None or bar.close <= sma50:
            return None

        # 2. Consecutive down days — pullback pattern
        if not self._consecutive_down(closes, self.consec_down_days):
            return None

        # 3. RSI(2) oversold — short-term extreme
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi > self.rsi_max:
            return None

        # 4. IBS — close near day's low (selling exhaustion)
        ibs = self._ibs(bar)
        if ibs > self.ibs_threshold:
            return None

        # 5. Volume filter: not dead volume
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is not None and avg_vol > 0 and bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # ATR for stop and target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Cap stop at max_stop_pct of price
        raw_stop_dist = self.stop_atr_mult * atr
        max_stop_dist = bar.close * self.max_stop_pct
        stop_dist = min(raw_stop_dist, max_stop_dist)

        stop = bar.close - stop_dist
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
                "consec_down": self.consec_down_days,
                "rsi2": round(rsi, 2),
                "ibs": round(ibs, 3),
                "sma50": round(sma50, 2),
                "atr": round(atr, 4),
                "regime": regime,
                "seed": "trend_pullback_multi",
            },
        )

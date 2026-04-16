"""Seed: Trend-Following Pullback Entry.

Buy pullbacks to EMA(20) in confirmed uptrends (price > SMA50, EMA20 > SMA50).
Volume and RSI(14) filters. Skips SHOCK volatility regime.
Long-only, daily bars, max_hold_days=10.
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
        self.sma_slow_period = int(config.get("sma_slow_period", 50))
        self.ema_fast_period = int(config.get("ema_fast_period", 20))
        self.pullback_min_pct = float(config.get("pullback_min_pct", 0.01))
        self.pullback_max_pct = float(config.get("pullback_max_pct", 0.02))
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_low = float(config.get("rsi_low", 40.0))
        self.rsi_high = float(config.get("rsi_high", 55.0))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.8))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 4.0))
        self.max_hold_days = int(config.get("max_hold_days", 10))

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

        # Skip SHOCK volatility regime
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # Trend confirmation: price > SMA(50)
        sma50 = self._sma(closes, self.sma_slow_period)
        if sma50 is None or bar.close <= sma50:
            return None

        # Momentum alignment: EMA(20) > SMA(50)
        ema20 = self._ema(closes, self.ema_fast_period)
        if ema20 is None or ema20 <= sma50:
            return None

        # Pullback to EMA(20): close within 1-2% of EMA20
        if ema20 < 1e-9:
            return None
        dist_pct = (bar.close - ema20) / ema20
        if dist_pct < -self.pullback_max_pct or dist_pct > self.pullback_min_pct:
            return None

        # RSI(14) in sweet spot: not overbought, not oversold
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi < self.rsi_low or rsi > self.rsi_high:
            return None

        # Volume confirmation: above 0.8x 20-day average
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # ATR for stop and target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
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
                "ema20": round(ema20, 2),
                "sma50": round(sma50, 2),
                "pullback_pct": round(dist_pct, 4),
                "rsi14": round(rsi, 2),
                "vol_ratio": round(bar.volume / avg_vol, 2),
                "atr": round(atr, 4),
                "seed": "trend_pullback",
            },
        )

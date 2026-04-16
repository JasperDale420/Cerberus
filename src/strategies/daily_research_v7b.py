"""Consecutive-Down Pullback in Rising Trends with Drop-Size Filter.

Buy after N consecutive down closes when RSI(2) is oversold AND:
- SMA(50) is rising (uptrend context)
- Cumulative drop is 1-3x ATR (meaningful but not crash)
- ATR/price ratio is sane (skip extreme volatility)
- Skip SHOCK + near-earnings

Long-only, daily bars, max 5-day hold.
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
        # Optimizer-tuned
        self.consec_down_days = int(config.get("consec_down_days", 2))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_max = float(config.get("rsi_max", 20.0))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.03))
        # Trend
        self.sma_period = int(config.get("sma_slow_period", 50))
        self.sma_slope_period = int(config.get("sma_slope_period", 10))
        # Drop magnitude filter (in ATR units)
        self.min_drop_atr = float(config.get("min_drop_atr", 0.8))
        self.max_drop_atr = float(config.get("max_drop_atr", 3.5))
        # Max ATR/price ratio (skip extremely volatile names)
        self.max_atr_pct = float(config.get("max_atr_pct", 0.04))
        # Volume
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.5))

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

    @staticmethod
    def _consecutive_down(closes: list[float], n: int) -> bool:
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

        # Skip SHOCK
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Event filter
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False):
            return None

        # Skip DOWN regime — this strategy buys pullbacks in uptrends
        regime_trend = labels.get("regime_trend", "")
        if regime_trend == "DOWN":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # ATR computation (needed early for filters)
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Max ATR/price ratio filter — skip extreme volatility
        atr_pct = atr / bar.close
        if atr_pct > self.max_atr_pct:
            return None

        # Trend: price > SMA(50) AND SMA(50) must be rising
        sma50 = self._sma(closes, self.sma_period)
        if sma50 is None or bar.close <= sma50:
            return None
        sma50_prev = (
            self._sma(closes[: -self.sma_slope_period], self.sma_period)
            if len(closes) > self.sma_period + self.sma_slope_period
            else None
        )
        if sma50_prev is not None and sma50 <= sma50_prev:
            return None

        # Consecutive down days
        if not self._consecutive_down(closes, self.consec_down_days):
            return None

        # RSI(2) oversold
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi > self.rsi_max:
            return None

        # Drop magnitude: cumulative drop over consec_down_days in ATR units
        pre_drop_close = closes[-(self.consec_down_days + 1)]
        drop = pre_drop_close - bar.close
        drop_atr = drop / atr
        if drop_atr < self.min_drop_atr or drop_atr > self.max_drop_atr:
            return None

        # Volume: not dead
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is not None and avg_vol > 0 and bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # Stop and target
        raw_stop_dist = self.stop_atr_mult * atr
        max_stop_dist = bar.close * self.max_stop_pct
        stop_dist = min(raw_stop_dist, max_stop_dist)

        stop = bar.close - stop_dist
        target = bar.close + self.target_atr_mult * atr

        regime = labels.get("regime", "")
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
                "drop_atr": round(drop_atr, 2),
                "atr_pct": round(atr_pct, 4),
                "sma50_trend": "rising",
                "atr": round(atr, 4),
                "regime": regime,
                "seed": "consec_down_rising_trend",
            },
        )

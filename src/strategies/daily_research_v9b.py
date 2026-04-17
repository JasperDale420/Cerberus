"""Regime-Adaptive Dual-Mode: Trend Pullback + IBS Mean Reversion.

UP regime: Buy Keltner channel lower-band touches in confirmed uptrends (ADX>20, EMA20>SMA50).
FLAT/DOWN regime: Buy low IBS (Internal Bar Strength) days for mean reversion.
Skips SHOCK volatility. Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v9b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        # Trend mode params
        self.sma_slow_period = int(config.get("sma_slow_period", 50))
        self.ema_fast_period = int(config.get("ema_fast_period", 20))
        self.keltner_atr_mult = float(config.get("keltner_atr_mult", 1.5))
        self.adx_period = int(config.get("adx_period", 14))
        self.adx_min = float(config.get("adx_min", 20.0))
        # IBS mean reversion params
        self.ibs_entry = float(config.get("ibs_entry", 0.25))
        self.ibs_sma_period = int(config.get("ibs_sma_period", 10))
        # Shared params
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.5))

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
    def _adx(bars: list[Bar], period: int) -> Optional[float]:
        n = len(bars)
        if n < 2 * period + 2:
            return None
        plus_dm_vals = []
        minus_dm_vals = []
        tr_vals = []
        for i in range(1, n):
            up_move = bars[i].high - bars[i - 1].high
            down_move = bars[i - 1].low - bars[i].low
            plus_dm_vals.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
            minus_dm_vals.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
            tr_vals.append(
                max(
                    bars[i].high - bars[i].low,
                    abs(bars[i].high - bars[i - 1].close),
                    abs(bars[i].low - bars[i - 1].close),
                )
            )
        if len(tr_vals) < 2 * period:
            return None
        smoothed_plus_dm = sum(plus_dm_vals[:period])
        smoothed_minus_dm = sum(minus_dm_vals[:period])
        smoothed_tr = sum(tr_vals[:period])
        dx_vals = []
        for i in range(period, len(tr_vals)):
            smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm_vals[i]
            smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm_vals[i]
            smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr_vals[i]
            if smoothed_tr > 0:
                plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
                minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
            else:
                plus_di = minus_di = 0.0
            di_sum = plus_di + minus_di
            dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0
            dx_vals.append(dx)
        if len(dx_vals) < period:
            return None
        adx = sum(dx_vals[:period]) / period
        for i in range(period, len(dx_vals)):
            adx = (adx * (period - 1) + dx_vals[i]) / period
        return adx

    @staticmethod
    def _ibs(bar: Bar) -> Optional[float]:
        """Internal Bar Strength: (close - low) / (high - low)."""
        rng = bar.high - bar.low
        if rng < 1e-9:
            return None
        return (bar.close - bar.low) / rng

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

        # ATR for stop and target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Volume filter
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # Get regime
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "FLAT")

        # Route to appropriate mode
        if trend == "UP":
            return self._trend_pullback_signal(symbol, bar, bars, closes, atr, market_state)
        else:
            return self._ibs_reversion_signal(symbol, bar, bars, closes, atr, market_state)

    def _trend_pullback_signal(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        atr: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """UP regime: Buy pullback to Keltner lower band in confirmed uptrend."""
        sma50 = self._sma(closes, self.sma_slow_period)
        ema20 = self._ema(closes, self.ema_fast_period)
        if sma50 is None or ema20 is None:
            return None

        # Trend confirmation: EMA20 > SMA50
        if ema20 <= sma50:
            return None

        # ADX confirms trend strength
        adx = self._adx(bars, self.adx_period)
        if adx is None or adx < self.adx_min:
            return None

        # Keltner lower band = EMA20 - mult*ATR
        keltner_lower = ema20 - self.keltner_atr_mult * atr

        # Price must be near or touching Keltner lower band
        if bar.close > keltner_lower:
            return None

        # Don't buy too far below (broken support)
        if bar.close < ema20 - 3.0 * atr:
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
                "mode": "trend_pullback",
                "ema20": round(ema20, 2),
                "sma50": round(sma50, 2),
                "adx": round(adx, 1),
                "atr": round(atr, 4),
            },
        )

    def _ibs_reversion_signal(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        atr: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """FLAT/DOWN regime: Buy low IBS days for mean reversion."""
        ibs = self._ibs(bar)
        if ibs is None or ibs > self.ibs_entry:
            return None

        # Price should be near short-term SMA (not in freefall)
        sma_short = self._sma(closes, self.ibs_sma_period)
        if sma_short is None:
            return None
        dist_pct = (bar.close - sma_short) / sma_short
        if dist_pct < -0.05:  # More than 5% below SMA = freefall, skip
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
                "mode": "ibs_reversion",
                "ibs": round(ibs, 3),
                "sma_short": round(sma_short, 2),
                "atr": round(atr, 4),
            },
        )

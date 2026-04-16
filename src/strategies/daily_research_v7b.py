"""Dual-Mode Regime-Adaptive: Trend Pullback + Mean Reversion.

UP regime: buy ATR-normalized pullbacks to EMA(21) with low IBS timing.
FLAT regime: buy extreme BB lower band touches with very low IBS.
Skip DOWN regime entirely — mean reversion and pullbacks both fail there.
ATR/price vol filter + realized vol gate to avoid high-vol disaster.
Long-only, daily bars.
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
        # Trend mode params
        self.ema_period = int(config.get("ema_period", 21))
        self.sma_slow_period = int(config.get("sma_slow_period", 50))
        self.pullback_atr_min = float(config.get("pullback_atr_min", 0.3))
        self.pullback_atr_max = float(config.get("pullback_atr_max", 1.5))
        self.ibs_entry_threshold = float(config.get("ibs_entry_threshold", 0.35))
        # Mean reversion mode params
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.mr_ibs_threshold = float(config.get("mr_ibs_threshold", 0.25))
        # Shared params
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_max = float(config.get("rsi_max", 65.0))
        # Vol filter
        self.max_atr_pct = float(config.get("max_atr_pct", 0.03))
        # Realized vol hard gate
        self.max_realized_vol = float(config.get("max_realized_vol", 0.22))
        # Volume filter
        self.vol_lookback = int(config.get("vol_lookback", 20))
        self.min_vol_ratio = float(config.get("min_vol_ratio", 0.5))

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

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        data = values[-period:]
        mean = sum(data) / period
        variance = sum((x - mean) ** 2 for x in data) / period
        return variance**0.5

    @staticmethod
    def _ibs(bar: Bar) -> float:
        rng = bar.high - bar.low
        if rng < 1e-9:
            return 0.5
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

        # Skip SHOCK and HIGH volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol in (VolRegime.SHOCK, VolRegime.HIGH):
            return None

        # Realized vol hard gate
        if market_state.realized_vol and market_state.realized_vol > self.max_realized_vol:
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # ATR-based vol filter
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None
        atr_pct = atr / bar.close
        if atr_pct > self.max_atr_pct:
            return None

        # Volume filter — skip low-volume bars
        volumes = [b.volume for b in bars]
        if len(volumes) >= self.vol_lookback:
            avg_vol = sum(volumes[-self.vol_lookback :]) / self.vol_lookback
            if avg_vol > 0 and bar.volume < avg_vol * self.min_vol_ratio:
                return None

        # RSI filter — not overbought
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi > self.rsi_max:
            return None

        ibs = self._ibs(bar)

        # Get regime
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "FLAT")

        # Skip DOWN regime — both pullbacks and mean reversion fail there
        if trend == "DOWN":
            return None

        # Mode 1: Trend pullback (UP regime)
        if trend == "UP":
            return self._trend_pullback_signal(symbol, bar, closes, atr, ibs, market_state)

        # Mode 2: Mean reversion (FLAT) — stricter filters
        return self._mean_reversion_signal(symbol, bar, closes, atr, ibs, market_state)

    def _trend_pullback_signal(
        self,
        symbol: str,
        bar: Bar,
        closes: list[float],
        atr: float,
        ibs: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        ema = self._ema(closes, self.ema_period)
        sma_slow = self._sma(closes, self.sma_slow_period)
        if ema is None or sma_slow is None:
            return None

        # Trend confirmation: EMA > SMA(50) and price > SMA(50)
        if ema <= sma_slow or bar.close <= sma_slow:
            return None

        # ATR-normalized pullback
        dist = ema - bar.close
        dist_atr = dist / atr
        if dist_atr < self.pullback_atr_min or dist_atr > self.pullback_atr_max:
            return None

        # IBS filter
        if ibs > self.ibs_entry_threshold:
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
            meta={"mode": "trend_pullback", "dist_atr": round(dist_atr, 2), "ibs": round(ibs, 2)},
        )

    def _mean_reversion_signal(
        self,
        symbol: str,
        bar: Bar,
        closes: list[float],
        atr: float,
        ibs: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        bb_mean = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_mean is None or bb_std is None or bb_std < 1e-9:
            return None

        lower_band = bb_mean - self.bb_std * bb_std

        # Price at or below lower band
        if bar.close > lower_band:
            return None

        # Strict IBS filter for FLAT regime
        if ibs > self.mr_ibs_threshold:
            return None

        # Must have at least 2 consecutive down days
        if len(closes) >= 3 and not (closes[-1] < closes[-2] < closes[-3]):
            return None

        stop = bar.close - self.stop_atr_mult * atr
        target = bb_mean  # Target the mean

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "mode": "mean_reversion",
                "ibs": round(ibs, 2),
                "z_score": round((bar.close - bb_mean) / bb_std, 2),
            },
        )

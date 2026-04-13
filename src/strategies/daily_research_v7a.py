"""Regime-Adaptive Multi-Factor Strategy.

Switches between mean-reversion and trend-pullback based on regime labels.
- FLAT: Z-score mean reversion (Bollinger deviation + IBS + volume)
- UP: EMA pullback (buy dips in uptrend with volume confirmation)
- DOWN: Very selective oversold bounce only (deep Z-score + IBS)
- Event filters: skip near_earnings, near_fomc

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
        self.min_bars = int(config.get("min_bars", 50))
        # Bollinger / Z-score params
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.zscore_entry = float(config.get("zscore_entry", -1.5))
        self.zscore_deep = float(config.get("zscore_deep", -2.5))
        # IBS threshold
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        # Trend pullback params
        self.ema_fast = int(config.get("ema_fast", 10))
        self.ema_slow = int(config.get("ema_slow", 40))
        self.pullback_atr_mult = float(config.get("pullback_atr_mult", 1.0))
        # ATR / risk
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.drawdown_max = float(config.get("drawdown_max", 0.12))
        # Hold
        self.max_hold_days = int(config.get("max_hold_days", 7))
        # Volume filter
        self.vol_avg_mult = float(config.get("vol_avg_mult", 0.5))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _ema_calc(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = v * mult + ema * (1 - mult)
        return ema

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        subset = values[-period:]
        mean = sum(subset) / period
        variance = sum((v - mean) ** 2 for v in subset) / period
        return variance**0.5

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

    def _volume_ok(self, bars: list[Bar]) -> bool:
        """Check if recent volume is above average."""
        if len(bars) < 20:
            return True
        vols = [b.volume for b in bars[-20:]]
        avg_vol = sum(vols) / len(vols)
        if avg_vol <= 0:
            return True
        return bars[-1].volume >= avg_vol * self.vol_avg_mult

    def _get_regime(self, symbol_state: SymbolState) -> str:
        """Get trend regime from labels, fallback to EMA-based detection."""
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "")
        if trend in ("UP", "DOWN", "FLAT"):
            return trend
        return "FLAT"  # default

    def _near_event(self, symbol_state: SymbolState) -> bool:
        """Check if near earnings or FOMC."""
        labels = symbol_state.meta.get("regime_labels", {})
        return labels.get("near_earnings", False) or labels.get("near_fomc", False)

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

        # Skip events
        if self._near_event(symbol_state):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Core indicators
        sma = self._sma(closes, self.bb_period)
        std = self._std(closes, self.bb_period)
        atr = self._atr(bars, self.atr_period)
        if sma is None or std is None or atr is None or std < 1e-9 or atr < 1e-9:
            return None

        # Z-score
        zscore = (bar.close - sma) / std

        # IBS
        bar_range = bar.high - bar.low
        ibs = (bar.close - bar.low) / bar_range if bar_range > 1e-9 else 0.5

        # Drawdown filter
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.drawdown_max:
            return None

        # Volume filter
        if not self._volume_ok(bars):
            return None

        # Regime routing
        regime = self._get_regime(symbol_state)

        if regime == "UP":
            return self._trend_pullback_signal(symbol, bar, bars, closes, atr, zscore, ibs, market_state)
        elif regime == "FLAT":
            return self._mean_reversion_signal(symbol, bar, bars, closes, sma, atr, zscore, ibs, market_state)
        elif regime == "DOWN":
            return self._deep_oversold_signal(symbol, bar, bars, closes, sma, atr, zscore, ibs, market_state)

        return None

    def _trend_pullback_signal(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        atr: float,
        zscore: float,
        ibs: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """UP regime: buy pullbacks in uptrend."""
        ema_f = self._ema_calc(closes, self.ema_fast)
        ema_s = self._ema_calc(closes, self.ema_slow)
        if ema_f is None or ema_s is None:
            return None

        # Confirm uptrend: fast EMA > slow EMA
        if ema_f <= ema_s:
            return None

        # Price pulled back near or below fast EMA
        pullback_zone = ema_f + self.pullback_atr_mult * atr
        if bar.close > pullback_zone:
            return None  # too extended

        # Need some oversold confirmation: IBS < threshold OR mild negative Z-score
        if ibs >= self.ibs_threshold and zscore > -0.5:
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
            meta={"mode": "trend_pullback", "zscore": round(zscore, 2), "ibs": round(ibs, 3), "atr": round(atr, 4)},
        )

    def _mean_reversion_signal(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        sma: float,
        atr: float,
        zscore: float,
        ibs: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """FLAT regime: fade oversold extremes."""
        # Need negative Z-score (oversold)
        if zscore > self.zscore_entry:
            return None

        # IBS confirmation: close near low
        if ibs >= self.ibs_threshold:
            return None

        # Target: SMA (mean reversion target)
        target = sma
        if target <= bar.close:
            return None  # no upside to mean

        stop = bar.close - self.stop_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={"mode": "mean_reversion", "zscore": round(zscore, 2), "ibs": round(ibs, 3), "atr": round(atr, 4)},
        )

    def _deep_oversold_signal(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        sma: float,
        atr: float,
        zscore: float,
        ibs: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """DOWN regime: only enter on deep oversold bounces."""
        # Need very deep oversold
        if zscore > self.zscore_deep:
            return None

        # IBS must be very low (close near absolute low)
        if ibs >= 0.2:
            return None

        # Tighter target in downtrend: partial reversion
        target = bar.close + 0.5 * (sma - bar.close) if sma > bar.close else None
        if target is None or target <= bar.close:
            return None

        stop = bar.close - self.stop_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={"mode": "deep_oversold", "zscore": round(zscore, 2), "ibs": round(ibs, 3), "atr": round(atr, 4)},
        )

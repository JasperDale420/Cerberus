"""Daily Research v7a: Trend Pullback Strategy.

Buy dips in confirmed uptrends: SMA(20) > SMA(50), price pulls back
near SMA(20), with RSI(14) confirmation. Long-only, daily bars.
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
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.12))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        self.pullback_atr_mult = float(config.get("pullback_atr_mult", 1.0))
        self.rsi_low = float(config.get("rsi_low", 30.0))
        self.rsi_high = float(config.get("rsi_high", 55.0))

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
        ema = values[0]
        for v in values[1:]:
            ema = v * mult + ema * (1 - mult)
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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # --- Regime filter: skip DOWN trend and SHOCK vol ---
        labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = labels.get("regime_trend", "FLAT")
        regime_vol = labels.get("regime_vol", "NORMAL")
        if regime_trend == "DOWN":
            return None
        if regime_vol == "SHOCK":
            return None

        # --- Event filter: skip earnings and FOMC ---
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        # --- Trend confirmation: EMA(20) > SMA(50) ---
        ema20 = self._ema(closes, 20)
        sma50 = self._sma(closes, 50)
        if ema20 is None or sma50 is None:
            return None
        if ema20 <= sma50:
            return None

        # --- Pullback: price near or below EMA(20) ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        distance_to_ema = bar.close - ema20
        if distance_to_ema > self.pullback_atr_mult * atr:
            return None  # too far above EMA — not a pullback
        if distance_to_ema < -2.0 * atr:
            return None  # too far below — trend might be broken

        # --- RSI(14) confirmation: moderate oversold (not extreme) ---
        rsi = self._rsi(closes, 14)
        if rsi is None or rsi < self.rsi_low or rsi > self.rsi_high:
            return None

        # --- Drawdown filter ---
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.max_drawdown_pct:
            return None

        # --- Volume filter: today's vol >= 0.8x avg ---
        volumes = [b.volume for b in bars]
        avg_vol = self._sma(volumes[-20:], min(20, len(volumes[-20:])))
        if avg_vol is not None and avg_vol > 0:
            if bar.volume < 0.8 * avg_vol:
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
                "rsi14": round(rsi, 2),
                "pullback_atr": round(distance_to_ema / atr, 2),
                "atr": round(atr, 4),
                "regime": f"{regime_trend}+{regime_vol}",
            },
        )

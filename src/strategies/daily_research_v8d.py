"""Seed: Regime-Adaptive Strategy.

Different logic per regime:
  UP   — buy pullbacks below EMA(20) but above SMA(50)
  DOWN — short rallies above EMA(20) but below SMA(50)
  FLAT — mean reversion at Bollinger Band extremes

Stops scaled by realized volatility. Daily bars, max_hold_days=5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedRegimeSwitchStrategy(BaseStrategy):
    name = "daily_research_v8d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.ema_period = int(config.get("ema_period", 20))
        self.sma_period = int(config.get("sma_period", 50))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_up_threshold = float(config.get("rsi_up_threshold", 45.0))
        self.rsi_down_threshold = float(config.get("rsi_down_threshold", 60.0))
        self.atr_period = int(config.get("atr_period", 14))
        self.base_stop_atr_mult = float(config.get("base_stop_atr_mult", 1.5))
        self.base_target_atr_mult = float(config.get("base_target_atr_mult", 3.0))
        self.vol_stop_scale_factor = float(config.get("vol_stop_scale_factor", 5.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))

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
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        subset = values[-period:]
        mean = sum(subset) / period
        variance = sum((v - mean) ** 2 for v in subset) / period
        return variance ** 0.5

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

    def _vol_adjusted_stop_mult(self, market_state: MarketState) -> float:
        """Scale stop multiplier by realized vol: wider stops when vol is high."""
        realized_vol = market_state.realized_vol
        if realized_vol <= 0:
            return self.base_stop_atr_mult
        # Scale: base * (1 + vol * scale_factor)
        # e.g. realized_vol=0.02 (2%), scale_factor=5 -> mult * 1.10
        return self.base_stop_atr_mult * (1.0 + realized_vol * self.vol_stop_scale_factor)

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

        # Core indicators
        ema20 = self._ema(closes, self.ema_period)
        sma50 = self._sma(closes, self.sma_period)
        rsi = self._rsi(closes, self.rsi_period)
        atr = self._atr(bars, self.atr_period)

        if any(v is None for v in (ema20, sma50, rsi, atr)) or atr < 1e-9:
            return None

        # BB for FLAT regime
        bb_sma = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)

        # Read regime from symbol_state meta
        regime_labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = regime_labels.get("regime_trend", "FLAT").upper()

        stop_mult = self._vol_adjusted_stop_mult(market_state)

        if regime_trend == "UP":
            # Buy pullbacks: price dips below EMA(20) but stays above SMA(50)
            if bar.close >= ema20 or bar.close <= sma50:
                return None
            if rsi >= self.rsi_up_threshold:
                return None

            stop = bar.close - stop_mult * atr
            target = bar.close + self.base_target_atr_mult * atr

            self.last_signal_time[symbol] = bar.time
            return self._create_signal(
                symbol,
                OrderSide.BUY,
                bar,
                market_state,
                stop_price=stop,
                target_price=target,
                meta={
                    "regime": "UP",
                    "ema20": round(ema20, 2),
                    "sma50": round(sma50, 2),
                    "rsi14": round(rsi, 2),
                    "stop_mult": round(stop_mult, 3),
                    "seed": "regime_switch",
                },
            )

        elif regime_trend == "DOWN":
            # Short rallies: price rallies above EMA(20) but stays below SMA(50)
            if bar.close <= ema20 or bar.close >= sma50:
                return None
            if rsi <= self.rsi_down_threshold:
                return None

            stop = bar.close + stop_mult * atr
            target = bar.close - self.base_target_atr_mult * atr

            self.last_signal_time[symbol] = bar.time
            return self._create_signal(
                symbol,
                OrderSide.SELL,
                bar,
                market_state,
                stop_price=stop,
                target_price=target,
                meta={
                    "regime": "DOWN",
                    "ema20": round(ema20, 2),
                    "sma50": round(sma50, 2),
                    "rsi14": round(rsi, 2),
                    "stop_mult": round(stop_mult, 3),
                    "seed": "regime_switch",
                },
            )

        else:
            # FLAT regime: mean reversion at BB extremes
            if bb_sma is None or bb_std is None or bb_std < 1e-9:
                return None

            lower_bb = bb_sma - self.bb_std * bb_std
            upper_bb = bb_sma + self.bb_std * bb_std

            if bar.close < lower_bb:
                # Long at lower BB
                stop = bar.close - stop_mult * atr
                target = bb_sma

                if target <= bar.close:
                    return None

                self.last_signal_time[symbol] = bar.time
                return self._create_signal(
                    symbol,
                    OrderSide.BUY,
                    bar,
                    market_state,
                    stop_price=stop,
                    target_price=target,
                    meta={
                        "regime": "FLAT",
                        "side": "long_bb",
                        "lower_bb": round(lower_bb, 2),
                        "bb_mid": round(bb_sma, 2),
                        "rsi14": round(rsi, 2),
                        "stop_mult": round(stop_mult, 3),
                        "seed": "regime_switch",
                    },
                )

            elif bar.close > upper_bb:
                # Short at upper BB
                stop = bar.close + stop_mult * atr
                target = bb_sma

                if target >= bar.close:
                    return None

                self.last_signal_time[symbol] = bar.time
                return self._create_signal(
                    symbol,
                    OrderSide.SELL,
                    bar,
                    market_state,
                    stop_price=stop,
                    target_price=target,
                    meta={
                        "regime": "FLAT",
                        "side": "short_bb",
                        "upper_bb": round(upper_bb, 2),
                        "bb_mid": round(bb_sma, 2),
                        "rsi14": round(rsi, 2),
                        "stop_mult": round(stop_mult, 3),
                        "seed": "regime_switch",
                    },
                )

        return None

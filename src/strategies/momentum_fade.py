"""
MomentumFade Strategy

Type: mean-reversion
Description: Fades overextended momentum moves by entering in the opposite direction
when price pushes too far from VWAP on high volume, expecting a snap-back to the mean.
"""

from __future__ import annotations

from typing import Any

from src.core import time_utils
from src.core.domain import (
    Bar,
    MarketState,
    OrderSide,
    SessionRegime,
    Signal,
    SymbolState,
    VolRegime,
)
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.strategies.base import BaseStrategy
from src.strategies.confluence import ConfluenceScorer, score_deviation, score_volume


class MomentumFadeStrategy(BaseStrategy):
    """
    Fades overextended momentum moves — when price pushes too far from VWAP
    on a volume spike, enters in the opposite direction expecting a snap-back.

    Edge: Overextended intraday moves driven by emotional/retail surges
    tend to revert when institutional flow normalizes. The combination of
    extreme VWAP deviation AND high volume identifies exhaustion points
    where the move has likely run too far, too fast.

    Entry: SHORT when price is significantly above VWAP with a volume surge.
           LONG when price is significantly below VWAP with a volume surge.
    Exit: ATR-based trailing stop with partial exits at 2R and 4R.
    """

    name: str = "momentum_fade"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    def _set_params(self, config: dict[str, Any]) -> None:
        """Parse strategy-specific parameters from config."""
        super()._set_params(config)

        # Time window
        self.time_window_start = time_utils.parse_time_string(str(config.get("time_window_start", "09:45")))
        self.time_window_end = time_utils.parse_time_string(str(config.get("time_window_end", "15:45")))
        self.min_bars = int(config.get("min_bars", 30))

        # Tunable params — support Optuna overrides for optimization
        overrides = config.get("_optuna_overrides", {})

        # VWAP distance threshold (percentage) — how far price must be from VWAP
        self.vwap_threshold = float(overrides.get("vwap_threshold", config.get("vwap_threshold", 0.008)))

        # Volume surge multiplier — current bar volume vs recent average
        self.volume_surge_mult = float(overrides.get("volume_surge_mult", config.get("volume_surge_mult", 2.0)))

        # Confluence threshold — minimum weighted score to enter
        self.confluence_threshold = float(
            overrides.get("confluence_threshold", config.get("confluence_threshold", 60.0))
        )

        # Stop/target ATR multipliers
        self.stop_atr_mult = float(overrides.get("stop_atr_mult", config.get("stop_atr_mult", 1.5)))
        self.target_atr_mult = float(overrides.get("target_atr_mult", config.get("target_atr_mult", 3.0)))

        # Maximum hold time in minutes
        self.max_hold_minutes = int(overrides.get("max_hold_minutes", config.get("max_hold_minutes", 90)))

        # Mean reversion: only trade when higher TF is flat
        self.tf_alignment_mode = "mean_reversion"

    def _calculate_volume_ratio(self, bar: Bar, symbol_state: SymbolState) -> float:
        """Calculate current bar volume vs average of recent 1m bars."""
        bars = symbol_state.bars_1m
        if not bars or len(bars) < 10:
            return 0.0

        # Use the last 20 bars (or whatever is available) for average, excluding current
        lookback = min(len(bars) - 1, 20)
        if lookback <= 0:
            return 0.0

        recent_bars = list(bars)[-lookback - 1 : -1]
        if not recent_bars:
            return 0.0

        avg_volume = sum(b.volume for b in recent_bars) / len(recent_bars)
        if avg_volume <= 0:
            return 0.0

        return bar.volume / avg_volume

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        """Process a single bar and optionally generate a fade signal."""

        # --- time window ---
        t = time_utils.get_eastern_time_of_day(bar.time)
        if not (self.time_window_start <= t <= self.time_window_end):
            return None

        # --- cooldown ---
        if not self._check_cooldown(symbol, bar.time):
            return None

        # --- minimum data ---
        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        # --- regime gating ---
        snapshot = market_state.regime_snapshot
        if snapshot is not None:
            if snapshot.vol == VolRegime.SHOCK:
                return None
            if snapshot.session == SessionRegime.PREMARKET:
                return None

        # --- multi-timeframe setup ---
        mtf = MultiTimeframeAnalyzer(symbol_state)

        # === VWAP DISTANCE CALCULATION ===
        vwap = bar.vwap
        if vwap is None:
            vwap_from_indicator = symbol_state.indicators.get("session_vwap")
            if vwap_from_indicator is not None:
                try:
                    vwap = float(vwap_from_indicator)
                except (TypeError, ValueError):
                    pass

        if vwap is None or vwap <= 0:
            return None

        vwap_dist = (bar.close - vwap) / vwap

        # === VOLUME RATIO ===
        volume_ratio = self._calculate_volume_ratio(bar, symbol_state)

        # === ENTRY CONDITIONS ===
        # Must be overextended AND have a volume surge
        if abs(vwap_dist) <= self.vwap_threshold:
            return None
        if volume_ratio <= self.volume_surge_mult:
            return None

        # === DIRECTION ===
        # Fade the move: if price is above VWAP, go SHORT; if below, go LONG
        if vwap_dist > 0:
            side = OrderSide.SELL
        else:
            side = OrderSide.BUY

        # --- higher timeframe alignment (mean reversion mode) ---
        if not self._check_higher_tf_alignment(symbol_state, side):
            return None

        # === CONFLUENCE SCORING ===
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)

        # Factor 1: VWAP deviation magnitude (0-100)
        # More extended = higher score. Threshold 0.008, max 0.025
        vwap_dev_score = score_deviation(vwap_dist, self.vwap_threshold, self.vwap_threshold * 3.0)
        scorer.add_factor(
            "vwap_deviation",
            raw_value=vwap_dist,
            score=vwap_dev_score,
            weight=0.30,
            passed=abs(vwap_dist) > self.vwap_threshold,
        )

        # Factor 2: Volume surge intensity (0-100)
        vol_score = score_volume(
            bar.volume,
            bar.volume / volume_ratio if volume_ratio > 0 else 1.0,
            min_mult=self.volume_surge_mult,
            ideal_mult=self.volume_surge_mult * 2.0,
        )
        scorer.add_factor(
            "volume_surge",
            raw_value=volume_ratio,
            score=vol_score,
            weight=0.25,
            passed=volume_ratio > self.volume_surge_mult,
        )

        # Factor 3: RSI extremity — overextended RSI supports reversion
        rsi_1m = mtf.get_rsi("1m", 14)
        rsi_score = 0.0
        rsi_passed = False
        if rsi_1m is not None:
            if side == OrderSide.BUY:
                # Buying the dip: want RSI low (oversold)
                if rsi_1m < 30:
                    rsi_score = min((30 - rsi_1m) / 20.0 * 100.0, 100.0)
                    rsi_passed = True
            else:
                # Shorting the rip: want RSI high (overbought)
                if rsi_1m > 70:
                    rsi_score = min((rsi_1m - 70) / 20.0 * 100.0, 100.0)
                    rsi_passed = True
        scorer.add_factor(
            "rsi_extremity",
            raw_value=rsi_1m if rsi_1m is not None else 50.0,
            score=rsi_score,
            weight=0.20,
            passed=rsi_passed,
        )

        # Factor 4: Mean reversion alignment — higher TFs should be flat/choppy
        mr_alignment = mtf.get_mean_reversion_alignment()
        mr_score = mr_alignment * 100.0
        scorer.add_factor(
            "mr_alignment",
            raw_value=mr_alignment,
            score=mr_score,
            weight=0.15,
            passed=mr_alignment > 0.3,
        )

        # Factor 5: Candle rejection — wick against the move suggests exhaustion
        candle_range = bar.high - bar.low
        rejection_score = 0.0
        if candle_range > 0:
            if side == OrderSide.BUY:
                # Buying: want a long lower wick (rejection of the low)
                lower_wick = min(bar.open, bar.close) - bar.low
                rejection_score = min((lower_wick / candle_range) * 100.0, 100.0)
            else:
                # Selling: want a long upper wick (rejection of the high)
                upper_wick = bar.high - max(bar.open, bar.close)
                rejection_score = min((upper_wick / candle_range) * 100.0, 100.0)
        scorer.add_factor(
            "candle_rejection",
            raw_value=rejection_score / 100.0,
            score=rejection_score,
            weight=0.10,
        )

        # --- confluence gate ---
        if not scorer.passes_threshold():
            return None

        # --- stop / target ---
        atr_5m = mtf.get_atr("5m", 14)
        if atr_5m is None or atr_5m <= 0:
            return None

        stop_distance = atr_5m * self.stop_atr_mult
        stop_distance = self._apply_regime_volatility_multiplier(stop_distance, market_state)

        if side == OrderSide.BUY:
            stop_price = bar.close - stop_distance
            target_price = bar.close + atr_5m * self.target_atr_mult
        else:
            stop_price = bar.close + stop_distance
            target_price = bar.close - atr_5m * self.target_atr_mult

        # --- build signal metadata ---
        meta: dict[str, Any] = {
            "strategy_version": "momentum_fade_v1",
            "vwap": float(vwap),
            "vwap_dist": round(vwap_dist, 6),
            "volume_ratio": round(volume_ratio, 2),
            "atr_5m": round(atr_5m, 6),
            "rsi_1m": round(rsi_1m, 2) if rsi_1m is not None else None,
            "mr_alignment": round(mr_alignment, 4),
            **scorer.to_meta(),
            "exit_config": {
                "trailing_enabled": True,
                "trail_timeframe": "5m",
                "trail_lookback": 3,
                "trail_min_profit_r": 0.5,
                "partial_exits": [(2.0, 0.33), (4.0, 0.33)],
                "max_hold_minutes": self.max_hold_minutes,
                "vol_adaptive": True,
            },
        }

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta=meta,
        )

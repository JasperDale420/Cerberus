"""
MomentumFade Strategy

Type: mean-reversion
Description: Fades overextended momentum moves by entering in the opposite direction
when price pushes too far from VWAP on high volume, expecting a snap-back to the mean.

Quant upgrades (v2):
- GARCH-conditional z-score replaces raw VWAP deviation threshold
- Momentum exhaustion model (velocity + deceleration) gates entries
- Hurst exponent gate rejects when market is trending (H >= 0.5)
- Entropy filter rejects when market is random (entropy > 0.8)
- Intraday seasonal volume adjustment replaces flat volume average
"""

from __future__ import annotations

import collections
from typing import Any

from src.core import time_utils
from src.core.domain import (
    Bar,
    MarketState,
    OrderSide,
    SessionRegime,
    Signal,
    SymbolState,
    TrendRegime,
    VolRegime,
)
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.quant.statistics import HurstExponent
from src.quant.volatility import GARCHForecaster
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

    Quant upgrades (v2):
    - GARCH-conditional z-score normalizes deviation by conditional volatility
    - Momentum exhaustion model requires decelerating momentum before fading
    - Hurst exponent gate ensures mean-reverting regime (H < 0.5)
    - Entropy filter skips random/noisy markets (entropy > 0.8)
    - Intraday seasonal volume adjustment for time-of-day aware surges

    Entry: SHORT when price is significantly above VWAP with a volume surge.
           LONG when price is significantly below VWAP with a volume surge.
    Exit: ATR-based trailing stop with partial exits at 2R and 4R.
    """

    name: str = "momentum_fade"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

        # Per-symbol quant state
        self._garch: dict[str, GARCHForecaster] = {}
        self._hurst: dict[str, HurstExponent] = {}
        self._velocity_history: dict[str, collections.deque] = {}
        self._acceleration_history: dict[str, collections.deque] = {}
        self._close_history: dict[str, collections.deque] = {}

        # Intraday seasonal volume: keyed by (symbol, hour, quarter)
        # Each entry accumulates (sum_volume, count) for average computation
        self._seasonal_volume: dict[tuple[str, int, int], tuple[float, int]] = {}

    def _set_params(self, config: dict[str, Any]) -> None:
        """Parse strategy-specific parameters from config."""
        super()._set_params(config)

        # Time window
        self.time_window_start = time_utils.parse_time_string(str(config.get("time_window_start", "09:45")))
        self.time_window_end = time_utils.parse_time_string(str(config.get("time_window_end", "15:45")))
        self.min_bars = int(config.get("min_bars", 30))

        # Tunable params — support Optuna overrides for optimization
        overrides = config.get("_optuna_overrides", {})

        # VWAP distance threshold — raw fallback when GARCH not fitted
        self.vwap_threshold = float(overrides.get("vwap_threshold", config.get("vwap_threshold", 0.008)))

        # GARCH conditional z-score threshold (replaces raw vwap_threshold when fitted)
        self.garch_zscore_threshold = float(
            overrides.get("garch_zscore_threshold", config.get("garch_zscore_threshold", 2.0))
        )

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

        # Hurst gate threshold — reject when H >= this value
        self.hurst_gate_threshold = float(
            overrides.get("hurst_gate_threshold", config.get("hurst_gate_threshold", 0.5))
        )

        # Entropy filter threshold — reject when entropy > this value
        self.entropy_threshold = float(overrides.get("entropy_threshold", config.get("entropy_threshold", 0.8)))

        # Exhaustion model params
        self.velocity_lookback = int(config.get("velocity_lookback", 5))
        self.acceleration_lookback = int(config.get("acceleration_lookback", 3))
        self.velocity_threshold = float(config.get("velocity_threshold", 0.003))

        # Minimum seasonal observations before using seasonal volume
        self.min_seasonal_obs = int(config.get("min_seasonal_obs", 5))

        # Mean-reversion mode for higher-TF alignment check
        self.tf_alignment_mode = "mean_reversion"

        # Disable TF alignment — the trend-aware directional filter in on_bar()
        # handles regime gating; the separate EMA-spread gate is redundant
        self.higher_tf_alignment = False

    def _ensure_symbol_state(self, symbol: str) -> None:
        """Lazily initialize per-symbol quant components."""
        if symbol not in self._garch:
            self._garch[symbol] = GARCHForecaster(min_observations=50)
        if symbol not in self._hurst:
            self._hurst[symbol] = HurstExponent(min_observations=100)
        if symbol not in self._velocity_history:
            self._velocity_history[symbol] = collections.deque(maxlen=50)
        if symbol not in self._acceleration_history:
            self._acceleration_history[symbol] = collections.deque(maxlen=50)
        if symbol not in self._close_history:
            self._close_history[symbol] = collections.deque(maxlen=50)

    def _calculate_volume_ratio(self, bar: Bar, symbol: str, symbol_state: SymbolState) -> float:
        """Calculate current bar volume vs time-of-day seasonal average.

        Uses 15-minute buckets to compute average volume for the same time-of-day.
        Falls back to overall recent average if insufficient seasonal data.
        """
        t = time_utils.get_eastern_time_of_day(bar.time)
        hour = t.hour
        quarter = t.minute // 15
        key = (symbol, hour, quarter)

        # Update seasonal accumulator with this bar's volume
        prev_sum, prev_count = self._seasonal_volume.get(key, (0.0, 0))
        self._seasonal_volume[key] = (prev_sum + bar.volume, prev_count + 1)

        # Try seasonal average (excluding current bar from average)
        if prev_count >= self.min_seasonal_obs:
            seasonal_avg = prev_sum / prev_count
            if seasonal_avg > 0:
                return bar.volume / seasonal_avg

        # Fall back to overall recent average
        bars = symbol_state.bars_1m
        if not bars or len(bars) < 10:
            return 0.0

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

    def _compute_exhaustion(self, symbol: str, close: float) -> tuple[float, float, bool]:
        """Compute momentum velocity, acceleration, and exhaustion flag.

        Returns:
            (velocity, acceleration, is_exhausting)
            velocity: 5-bar rate of change
            acceleration: change in velocity over last 3 bars
            is_exhausting: True when |velocity| > threshold AND acceleration has opposite sign
        """
        self._close_history[symbol].append(close)
        closes = self._close_history[symbol]

        # Need at least velocity_lookback + acceleration_lookback + 1 prices
        min_needed = self.velocity_lookback + self.acceleration_lookback + 1
        if len(closes) < min_needed:
            return 0.0, 0.0, False

        closes_list = list(closes)

        # Compute velocity (ROC over velocity_lookback bars)
        velocity = (closes_list[-1] - closes_list[-1 - self.velocity_lookback]) / closes_list[
            -1 - self.velocity_lookback
        ]
        self._velocity_history[symbol].append(velocity)

        vel_hist = self._velocity_history[symbol]
        if len(vel_hist) < self.acceleration_lookback + 1:
            return velocity, 0.0, False

        # Compute acceleration (change in velocity over acceleration_lookback bars)
        vel_list = list(vel_hist)
        acceleration = vel_list[-1] - vel_list[-1 - self.acceleration_lookback]
        self._acceleration_history[symbol].append(acceleration)

        # Exhaustion: high velocity with decelerating (opposite-sign acceleration)
        is_exhausting = abs(velocity) > self.velocity_threshold and (
            (velocity > 0 and acceleration < 0) or (velocity < 0 and acceleration > 0)
        )

        return velocity, acceleration, is_exhausting

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        """Process a single bar and optionally generate a fade signal."""
        if not self._is_evaluation_bar(bar):
            return None

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

        # --- initialize per-symbol quant state ---
        self._ensure_symbol_state(symbol)

        # --- feed quant models ---
        self._garch[symbol].update(bar.close)
        hurst_result = self._hurst[symbol].update(bar.close)

        # --- Hurst exponent gate ---
        if hurst_result is not None and hurst_result.H >= self.hurst_gate_threshold:
            self.logger.info(
                "hurst_gate_rejected",
                symbol=symbol,
                hurst_h=round(hurst_result.H, 4),
                threshold=self.hurst_gate_threshold,
            )
            return None

        # --- regime gating ---
        snapshot = market_state.regime_snapshot
        if snapshot is not None:
            if snapshot.vol == VolRegime.SHOCK:
                return None
            if snapshot.session == SessionRegime.PREMARKET:
                return None

        # --- HMM regime gate ---
        if not self._check_hmm_gate(market_state):
            return None

        # --- entropy filter ---
        if snapshot is not None and snapshot.entropy_score > self.entropy_threshold:
            self.logger.info(
                "entropy_filter_rejected",
                symbol=symbol,
                entropy_score=round(snapshot.entropy_score, 4),
                threshold=self.entropy_threshold,
            )
            return None

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

        # === GARCH-CONDITIONAL Z-SCORE ===
        garch_zscore = self._garch[symbol].conditional_zscore(vwap_dist)
        use_garch = garch_zscore != 0.0  # GARCH fitted and producing forecasts

        if use_garch:
            # Use GARCH-normalized z-score threshold
            if abs(garch_zscore) <= self.garch_zscore_threshold:
                return None
        else:
            # Fall back to raw percentage threshold
            if abs(vwap_dist) <= self.vwap_threshold:
                return None

        # === MOMENTUM EXHAUSTION MODEL ===
        velocity, acceleration, is_exhausting = self._compute_exhaustion(symbol, bar.close)

        # Only fade when momentum is actually exhausting (or insufficient data to compute)
        has_enough_data = len(self._close_history[symbol]) >= (self.velocity_lookback + self.acceleration_lookback + 1)
        if has_enough_data and not is_exhausting:
            self.logger.debug(
                "exhaustion_filter_rejected",
                symbol=symbol,
                velocity=round(velocity, 6),
                acceleration=round(acceleration, 6),
            )
            return None

        # === VOLUME RATIO (seasonal-adjusted) ===
        volume_ratio = self._calculate_volume_ratio(bar, symbol, symbol_state)

        # === ENTRY CONDITIONS ===
        if volume_ratio <= self.volume_surge_mult:
            return None

        # === DIRECTION ===
        # Fade the move: if price is above VWAP, go SHORT; if below, go LONG
        if vwap_dist > 0:
            side = OrderSide.SELL
        else:
            side = OrderSide.BUY

        # --- trend-aware directional filter ---
        if snapshot is not None:
            if snapshot.trend == TrendRegime.UP and side == OrderSide.BUY:
                return None
            if snapshot.trend == TrendRegime.DOWN and side == OrderSide.SELL:
                return None

        # --- RSI directional gate ---
        mtf = MultiTimeframeAnalyzer(symbol_state)
        rsi_1m = mtf.get_rsi("1m", 14)
        if rsi_1m is not None:
            if side == OrderSide.BUY and rsi_1m > 50:
                return None
            if side == OrderSide.SELL and rsi_1m < 50:
                return None

        # --- higher timeframe alignment (mean reversion mode) ---
        if not self._check_higher_tf_alignment(symbol_state, side):
            return None

        # === CONFLUENCE SCORING ===
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)

        # Factor 1: VWAP deviation magnitude (0-100)
        vwap_dev_score = score_deviation(vwap_dist, self.vwap_threshold, self.vwap_threshold * 3.0)
        scorer.add_factor(
            "vwap_deviation",
            raw_value=vwap_dist,
            score=vwap_dev_score,
            weight=0.25,
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
            weight=0.20,
            passed=volume_ratio > self.volume_surge_mult,
        )

        # Factor 3: RSI extremity — overextended RSI supports reversion
        rsi_1m = mtf.get_rsi("1m", 14)
        rsi_score = 0.0
        rsi_passed = False
        if rsi_1m is not None:
            if side == OrderSide.BUY:
                if rsi_1m < 30:
                    rsi_score = min((30 - rsi_1m) / 20.0 * 100.0, 100.0)
                    rsi_passed = True
            else:
                if rsi_1m > 70:
                    rsi_score = min((rsi_1m - 70) / 20.0 * 100.0, 100.0)
                    rsi_passed = True
        scorer.add_factor(
            "rsi_extremity",
            raw_value=rsi_1m if rsi_1m is not None else 50.0,
            score=rsi_score,
            weight=0.15,
            passed=rsi_passed,
        )

        # Factor 4: Trend-context score
        trend_ctx_score = 50.0
        if snapshot is not None:
            if snapshot.trend == TrendRegime.UP and side == OrderSide.SELL:
                trend_ctx_score = 75.0
            elif snapshot.trend == TrendRegime.DOWN and side == OrderSide.BUY:
                trend_ctx_score = 75.0

        scorer.add_factor(
            "trend_context",
            raw_value=trend_ctx_score / 100.0,
            score=trend_ctx_score,
            weight=0.10,
            passed=True,
        )

        # Factor 5: Candle rejection — wick against the move suggests exhaustion
        candle_range = bar.high - bar.low
        rejection_score = 0.0
        if candle_range > 0:
            if side == OrderSide.BUY:
                lower_wick = min(bar.open, bar.close) - bar.low
                rejection_score = min((lower_wick / candle_range) * 100.0, 100.0)
            else:
                upper_wick = bar.high - max(bar.open, bar.close)
                rejection_score = min((upper_wick / candle_range) * 100.0, 100.0)
        scorer.add_factor(
            "candle_rejection",
            raw_value=rejection_score / 100.0,
            score=rejection_score,
            weight=0.10,
        )

        # Factor 6: GARCH z-score intensity (new)
        garch_score = 0.0
        if use_garch:
            # Scale: z=2.0 -> 50, z=3.0 -> 100
            garch_score = min(max((abs(garch_zscore) - 1.5) / 1.5 * 100.0, 0.0), 100.0)
        scorer.add_factor(
            "garch_zscore",
            raw_value=garch_zscore,
            score=garch_score,
            weight=0.10,
        )

        # Factor 7: Exhaustion intensity (new)
        exhaustion_score = 0.0
        if is_exhausting and abs(velocity) > 0:
            # Scale by how much velocity exceeds threshold
            exhaustion_score = min(abs(velocity) / (self.velocity_threshold * 3.0) * 100.0, 100.0)
        scorer.add_factor(
            "exhaustion",
            raw_value=velocity,
            score=exhaustion_score,
            weight=0.10,
        )

        # --- confluence gate ---
        if not scorer.passes_threshold():
            return None

        # --- mean-reversion alignment score (for meta reporting) ---
        mr_alignment = mtf.get_mean_reversion_alignment()

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
            "strategy_version": "momentum_fade_v2",
            "vwap": float(vwap),
            "vwap_dist": round(vwap_dist, 6),
            "garch_zscore": round(garch_zscore, 4),
            "use_garch": use_garch,
            "velocity": round(velocity, 6),
            "acceleration": round(acceleration, 6),
            "is_exhausting": is_exhausting,
            "hurst_h": round(hurst_result.H, 4) if hurst_result is not None else None,
            "entropy_score": round(snapshot.entropy_score, 4) if snapshot is not None else None,
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

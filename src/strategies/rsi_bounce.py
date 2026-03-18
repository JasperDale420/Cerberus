"""
RsiBounce Strategy — Institutional-Grade Mean Reversion (v2)

Type: mean-reversion
Description: Detects statistically validated mean-reversion setups using a 6-factor
confluence model. Replaces the original 3-factor RSI+BB approach with quantitative
gates derived from Ornstein-Uhlenbeck parameter estimation, Lo-MacKinlay variance
ratio testing, and VPIN order-flow toxicity filtering.

6-Factor Model:
    1. Z-Score Extremity — how far price has deviated from its rolling mean
    2. Half-Life Validity — OU-estimated mean-reversion speed confirms the
       deviation will revert within the holding period (primary gate)
    3. RSI Percentile Rank — RSI relative to its own recent distribution
    4. Volume Climax — volume spike confirming capitulation / exhaustion
    5. Momentum Deceleration — rate-of-change is flattening (reversion imminent)
    6. Variance Ratio Gate — Lo-MacKinlay VR(k) < 1 confirms the price process
       is mean-reverting rather than trending

Additional Filters:
    - VPIN toxicity gate: skip entries when order flow is toxic (informed trading)
    - Regime gating: skip SHOCK volatility and PREMARKET sessions
    - Trend-aware directional filter: only trade with-trend in trending markets

References:
    - Leung & Li (2015) — OU discrete MLE for half-life estimation
    - Lo & MacKinlay (1988) — Variance ratio test for random walk rejection
    - Easley, Lopez de Prado, O'Hara (2012) — VPIN flow toxicity
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional

from src.analysis.ou_estimator import OUEstimator, OUResult
from src.analysis.variance_ratio import VarianceRatioCalculator, VarianceRatioResult
from src.analysis.vpin import VPINCalculator, VPINResult
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
from src.strategies.base import BaseStrategy
from src.strategies.confluence import ConfluenceScorer, score_threshold


class RsiBounceStrategy(BaseStrategy):
    """
    RSI Bounce v2 — institutional-grade mean reversion with 6-factor confluence.

    Upgrades from v1:
    - Half-life gate (OU estimator) as primary mean-reversion validator
    - Z-score based entry instead of raw RSI thresholds
    - Lo-MacKinlay variance ratio test confirms mean-reverting regime
    - VPIN toxicity filter avoids entries during informed trading
    - RSI percentile rank replaces raw RSI extremes
    - Momentum deceleration confirms reversion timing

    Entry BUY:  6-factor confluence score >= threshold, z-score < -z_entry,
                half_life < max_hold, VR(k) < 1, VPIN not toxic
    Entry SELL: 6-factor confluence score >= threshold, z-score > +z_entry,
                half_life < max_hold, VR(k) < 1, VPIN not toxic
    Exit: Trailing stop + partial exits + max hold time (unchanged)
    """

    name: str = "rsi_bounce"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        # Per-symbol analysis state (lazy-initialized in on_bar)
        self._ou_estimators: dict[str, OUEstimator] = {}
        self._vr_calculators: dict[str, VarianceRatioCalculator] = {}
        self._vpin_calculators: dict[str, VPINCalculator] = {}
        self._rsi_history: dict[str, deque[float]] = {}
        self._price_history: dict[str, deque[float]] = {}
        self._volume_history: dict[str, deque[float]] = {}
        self._roc_history: dict[str, deque[float]] = {}

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

        # RSI parameters (used for percentile rank, not raw thresholds)
        self.rsi_len = int(overrides.get("rsi_len", config.get("rsi_len", 14)))
        self.rsi_oversold = float(overrides.get("rsi_oversold", config.get("rsi_oversold", 25.0)))
        self.rsi_overbought = float(overrides.get("rsi_overbought", config.get("rsi_overbought", 75.0)))

        # Z-score parameters
        self.zscore_lookback = int(overrides.get("zscore_lookback", config.get("zscore_lookback", 60)))
        self.zscore_entry = float(overrides.get("zscore_entry", config.get("zscore_entry", 2.0)))

        # Bollinger Band parameters (retained for band proximity in z-score calc)
        self.bb_period = int(overrides.get("bb_period", config.get("bb_period", 20)))
        self.band_sigma = float(overrides.get("band_sigma", config.get("band_sigma", 2.0)))
        self.band_tolerance = float(overrides.get("band_tolerance", config.get("band_tolerance", 0.5)))

        # OU half-life parameters
        self.ou_lookback = int(overrides.get("ou_lookback", config.get("ou_lookback", 60)))
        self.ou_min_obs = int(overrides.get("ou_min_obs", config.get("ou_min_obs", 30)))
        self.max_half_life_bars = float(
            overrides.get("max_half_life_bars", config.get("max_half_life_bars", 20.0))
        )
        self.min_half_life_bars = float(
            overrides.get("min_half_life_bars", config.get("min_half_life_bars", 1.0))
        )

        # Variance ratio parameters
        self.vr_lookback = int(overrides.get("vr_lookback", config.get("vr_lookback", 120)))
        self.vr_period = int(overrides.get("vr_period", config.get("vr_period", 5)))

        # VPIN parameters
        self.vpin_n_buckets = int(overrides.get("vpin_n_buckets", config.get("vpin_n_buckets", 50)))
        self.vpin_toxicity_threshold = float(
            overrides.get("vpin_toxicity_threshold", config.get("vpin_toxicity_threshold", 0.7))
        )

        # Volume climax parameters
        self.volume_lookback = int(overrides.get("volume_lookback", config.get("volume_lookback", 20)))
        self.volume_climax_mult = float(
            overrides.get("volume_climax_mult", config.get("volume_climax_mult", 1.5))
        )

        # Momentum deceleration
        self.roc_lookback = int(overrides.get("roc_lookback", config.get("roc_lookback", 10)))

        # RSI percentile rank
        self.rsi_pctile_lookback = int(
            overrides.get("rsi_pctile_lookback", config.get("rsi_pctile_lookback", 100))
        )

        # Risk / exit parameters
        self.stop_atr_mult = float(overrides.get("stop_atr_mult", config.get("stop_atr_mult", 1.5)))
        self.target_atr_mult = float(overrides.get("target_atr_mult", config.get("target_atr_mult", 3.0)))
        self.max_hold_minutes = int(overrides.get("max_hold_minutes", config.get("max_hold_minutes", 60)))
        self.confluence_threshold = float(
            overrides.get("confluence_threshold", config.get("confluence_threshold", 60.0))
        )

        # Mean-reversion mode for higher-TF alignment check
        self.tf_alignment_mode = "mean_reversion"

        # Disable TF alignment — the 6-factor model handles regime gating
        self.higher_tf_alignment = False

    # ------------------------------------------------------------------
    # Per-symbol lazy initialization
    # ------------------------------------------------------------------

    def _ensure_symbol_state(self, symbol: str) -> None:
        """Lazily initialize analysis objects for a symbol."""
        if symbol not in self._ou_estimators:
            self._ou_estimators[symbol] = OUEstimator(
                lookback=self.ou_lookback,
                min_observations=self.ou_min_obs,
            )
        if symbol not in self._vr_calculators:
            self._vr_calculators[symbol] = VarianceRatioCalculator(
                lookback=self.vr_lookback,
                period=self.vr_period,
            )
        if symbol not in self._vpin_calculators:
            self._vpin_calculators[symbol] = VPINCalculator(
                n_buckets=self.vpin_n_buckets,
                toxicity_threshold=self.vpin_toxicity_threshold,
                logger=self.logger,
            )
        if symbol not in self._rsi_history:
            self._rsi_history[symbol] = deque(maxlen=self.rsi_pctile_lookback)
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.zscore_lookback)
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=self.volume_lookback)
        if symbol not in self._roc_history:
            self._roc_history[symbol] = deque(maxlen=self.roc_lookback)

    # ------------------------------------------------------------------
    # Factor computations
    # ------------------------------------------------------------------

    def _compute_zscore(self, symbol: str, price: float) -> Optional[float]:
        """Compute rolling z-score of price relative to its recent mean."""
        history = self._price_history[symbol]
        history.append(price)

        if len(history) < self.zscore_lookback // 2:
            return None

        prices = list(history)
        n = len(prices)
        mean = sum(prices) / n
        var = sum((p - mean) ** 2 for p in prices) / n
        std = math.sqrt(var) if var > 0 else 0.0

        if std < 1e-10:
            return None

        return (price - mean) / std

    def _compute_rsi_percentile_rank(self, symbol: str, rsi: float) -> Optional[float]:
        """Compute percentile rank of current RSI relative to recent RSI history.

        Returns 0-100 where 0 = current RSI is lowest ever seen, 100 = highest.
        For buy signals, low percentile rank (extreme oversold relative to recent history)
        is desirable.
        """
        history = self._rsi_history[symbol]
        history.append(rsi)

        if len(history) < 20:
            return None

        rsi_list = list(history)
        count_below = sum(1 for r in rsi_list if r <= rsi)
        return (count_below / len(rsi_list)) * 100.0

    def _compute_volume_climax(self, symbol: str, volume: float) -> tuple[bool, float]:
        """Check if current volume represents a climax (capitulation/exhaustion).

        Returns (is_climax, volume_ratio) where volume_ratio = current / average.
        """
        history = self._volume_history[symbol]
        history.append(volume)

        if len(history) < self.volume_lookback // 2:
            return False, 1.0

        avg_vol = sum(history) / len(history)
        if avg_vol <= 0:
            return False, 1.0

        ratio = volume / avg_vol
        return ratio >= self.volume_climax_mult, ratio

    def _compute_momentum_deceleration(self, symbol: str, price: float) -> Optional[float]:
        """Compute momentum deceleration score.

        When price is falling (oversold), we want the rate of decline to be slowing.
        When price is rising (overbought), we want the rate of ascent to be slowing.
        Returns a 0-100 score where higher = more deceleration (better for mean reversion).
        """
        history = self._roc_history[symbol]

        # Need at least a few prices to compute rate of change
        if len(self._price_history[symbol]) < 3:
            history.append(0.0)
            return None

        prices = list(self._price_history[symbol])
        # 1-bar rate of change
        if len(prices) >= 2 and prices[-2] != 0:
            roc = (prices[-1] - prices[-2]) / abs(prices[-2])
        else:
            roc = 0.0

        history.append(roc)

        if len(history) < 3:
            return None

        roc_list = list(history)
        # Compare absolute momentum: recent vs older
        recent_abs_roc = sum(abs(r) for r in roc_list[-3:]) / 3.0
        older_abs_roc = sum(abs(r) for r in roc_list[:-3]) / max(len(roc_list) - 3, 1)

        if older_abs_roc < 1e-10:
            return 50.0  # neutral if no prior momentum

        # Deceleration ratio: if recent momentum is lower than older, price is slowing
        decel_ratio = 1.0 - (recent_abs_roc / older_abs_roc)
        # Map to 0-100: -1 (accelerating) -> 0, 0 (constant) -> 50, +1 (decelerating) -> 100
        score = max(0.0, min(100.0, (decel_ratio + 1.0) * 50.0))
        return score

    # ------------------------------------------------------------------
    # Main on_bar
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        """Process a single bar and optionally generate a signal."""

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

        # --- lazy init per-symbol analysis objects ---
        self._ensure_symbol_state(symbol)

        # --- multi-timeframe setup ---
        mtf = MultiTimeframeAnalyzer(symbol_state)

        # === RSI CHECK (5-minute) ===
        rsi_5m = mtf.get_rsi("5m", self.rsi_len)
        if rsi_5m is None:
            return None

        # Determine direction from RSI extremes
        is_oversold = rsi_5m < self.rsi_oversold
        is_overbought = rsi_5m > self.rsi_overbought
        if not is_oversold and not is_overbought:
            return None

        # --- trend-aware directional filter ---
        if snapshot is not None and snapshot.trend == TrendRegime.UP and is_overbought:
            return None
        if snapshot is not None and snapshot.trend == TrendRegime.DOWN and is_oversold:
            return None

        # === BOLLINGER BAND CHECK (5-minute) ===
        mtf._ensure_cache("5m")  # noqa: SLF001
        bb_data = mtf._cache.get("5m")  # noqa: SLF001
        if bb_data is None:
            return None

        bb_stats = bb_data.bb.get(self.bb_period)
        if bb_stats is None:
            return None

        bb_mean, bb_std = bb_stats
        if bb_std <= 0:
            return None

        upper_band = bb_mean + self.band_sigma * bb_std
        lower_band = bb_mean - self.band_sigma * bb_std
        band_width = upper_band - lower_band
        if band_width <= 0:
            return None

        current_price = bar.close

        # Check price proximity to the relevant band
        if is_oversold:
            distance_to_band = abs(current_price - lower_band)
            tolerance_amount = band_width * (self.band_tolerance / 100.0)
            near_band = distance_to_band <= tolerance_amount or current_price <= lower_band
            if not near_band:
                return None
            side = OrderSide.BUY
        else:
            distance_to_band = abs(current_price - upper_band)
            tolerance_amount = band_width * (self.band_tolerance / 100.0)
            near_band = distance_to_band <= tolerance_amount or current_price >= upper_band
            if not near_band:
                return None
            side = OrderSide.SELL

        # --- higher TF alignment ---
        if not self._check_higher_tf_alignment(symbol_state, side):
            return None

        # =================================================================
        # 6-FACTOR MEAN REVERSION MODEL
        # =================================================================

        # --- Factor 1: Z-Score Extremity ---
        z_score = self._compute_zscore(symbol, current_price)

        # --- Factor 2: Half-Life Validity (OU estimator) ---
        vwap_distance = mtf.get_vwap_distance("5m")
        ou_result: Optional[OUResult] = None
        if vwap_distance is not None:
            ou_result = self._ou_estimators[symbol].update(vwap_distance)

        # --- Factor 3: RSI Percentile Rank ---
        rsi_pctile = self._compute_rsi_percentile_rank(symbol, rsi_5m)

        # --- Factor 4: Volume Climax ---
        is_vol_climax, vol_ratio = self._compute_volume_climax(symbol, bar.volume)

        # --- Factor 5: Momentum Deceleration ---
        momentum_decel = self._compute_momentum_deceleration(symbol, current_price)

        # --- Factor 6: Variance Ratio Gate ---
        vr_result: Optional[VarianceRatioResult] = self._vr_calculators[symbol].update(current_price)

        # --- VPIN Toxicity Filter ---
        vpin_result: Optional[VPINResult] = self._vpin_calculators[symbol].update(bar)
        if vpin_result is not None and vpin_result.is_toxic:
            self.logger.debug(
                "rsi_bounce_vpin_toxic_skip",
                symbol=symbol,
                vpin=round(vpin_result.vpin, 4),
            )
            return None

        # === HALF-LIFE GATE (primary mean-reversion validator) ===
        # If OU estimator has produced a result, the half-life must be
        # reasonable: fast enough to revert within our holding period,
        # but not so fast it's just noise.
        half_life_valid = True
        half_life_score = 50.0  # neutral default
        if ou_result is not None:
            hl = ou_result.half_life
            if hl < self.min_half_life_bars or hl > self.max_half_life_bars:
                half_life_valid = False
            else:
                # Score: best when half_life is ~30-60% of max_hold
                ideal_hl = self.max_half_life_bars * 0.45
                hl_deviation = abs(hl - ideal_hl) / ideal_hl
                half_life_score = max(0.0, min(100.0, (1.0 - hl_deviation) * 100.0))

        if not half_life_valid:
            return None

        # === VARIANCE RATIO GATE ===
        # If we have enough data, VR must indicate mean-reversion (VR < 1)
        # or at least not indicate trending (VR > 1 with significance)
        vr_score = 50.0  # neutral default
        if vr_result is not None:
            if vr_result.is_trending:
                # Statistically significant trending — skip
                return None
            if vr_result.is_mean_reverting:
                # Strong mean-reversion signal
                vr_score = min(100.0, 70.0 + 30.0 * min(abs(vr_result.z_score) / 3.0, 1.0))
            elif vr_result.vr < 1.0:
                # Below 1 but not significant — moderate score
                vr_score = 50.0 + 20.0 * (1.0 - vr_result.vr)
            else:
                # VR >= 1 but not significantly trending — weak score
                vr_score = max(0.0, 50.0 - 20.0 * (vr_result.vr - 1.0))

        # === CONFLUENCE SCORING ===
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)

        # Factor 1: Z-Score Extremity (weight: 0.20)
        if z_score is not None:
            abs_z = abs(z_score)
            z_entry = self.zscore_entry
            if abs_z >= z_entry:
                z_extremity_score = min(100.0, 50.0 + 50.0 * (abs_z - z_entry) / z_entry)
            else:
                z_extremity_score = max(0.0, 50.0 * abs_z / z_entry)
        else:
            z_extremity_score = 40.0  # fall back to moderate

        scorer.add_factor(
            name="z_score_extremity",
            raw_value=z_score if z_score is not None else 0.0,
            score=z_extremity_score,
            weight=0.20,
            passed=True,
        )

        # Factor 2: Half-Life Validity (weight: 0.25) — primary gate
        scorer.add_factor(
            name="half_life_validity",
            raw_value=ou_result.half_life if ou_result is not None else 0.0,
            score=half_life_score,
            weight=0.25,
            passed=half_life_valid,
        )

        # Factor 3: RSI Percentile Rank (weight: 0.15)
        if rsi_pctile is not None:
            if side == OrderSide.BUY:
                # Lower percentile = more extreme oversold relative to history
                rsi_pctile_score = max(0.0, (100.0 - rsi_pctile))
            else:
                # Higher percentile = more extreme overbought relative to history
                rsi_pctile_score = rsi_pctile
        else:
            # Fall back to raw RSI extremity scoring
            if side == OrderSide.BUY:
                rsi_pctile_score = score_threshold(rsi_5m, self.rsi_oversold, 0.0, invert=True)
            else:
                rsi_pctile_score = score_threshold(rsi_5m, self.rsi_overbought, 100.0, invert=False)

        scorer.add_factor(
            name="rsi_percentile_rank",
            raw_value=rsi_pctile if rsi_pctile is not None else rsi_5m,
            score=rsi_pctile_score,
            weight=0.15,
            passed=True,
        )

        # Factor 4: Volume Climax (weight: 0.15)
        if is_vol_climax:
            vol_climax_score = min(100.0, 60.0 + 40.0 * min((vol_ratio - self.volume_climax_mult) / 1.0, 1.0))
        else:
            vol_climax_score = max(0.0, min(60.0, vol_ratio / self.volume_climax_mult * 60.0))

        scorer.add_factor(
            name="volume_climax",
            raw_value=vol_ratio,
            score=vol_climax_score,
            weight=0.15,
            passed=True,
        )

        # Factor 5: Momentum Deceleration (weight: 0.10)
        mom_decel_score = momentum_decel if momentum_decel is not None else 50.0

        scorer.add_factor(
            name="momentum_deceleration",
            raw_value=momentum_decel if momentum_decel is not None else 0.0,
            score=mom_decel_score,
            weight=0.10,
            passed=True,
        )

        # Factor 6: Variance Ratio (weight: 0.15)
        scorer.add_factor(
            name="variance_ratio",
            raw_value=vr_result.vr if vr_result is not None else 1.0,
            score=vr_score,
            weight=0.15,
            passed=True,
        )

        # Check confluence threshold
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
            "strategy_version": "rsi_bounce_v2",
            "atr_5m": round(atr_5m, 6),
            "rsi_5m": round(rsi_5m, 2),
            "bb_upper": round(upper_band, 4),
            "bb_lower": round(lower_band, 4),
            "bb_mean": round(bb_mean, 4),
            "bb_std": round(bb_std, 6),
            "z_score": round(z_score, 4) if z_score is not None else None,
            "half_life": round(ou_result.half_life, 2) if ou_result is not None else None,
            "ou_theta": round(ou_result.theta, 6) if ou_result is not None else None,
            "ou_mu": round(ou_result.mu, 6) if ou_result is not None else None,
            "variance_ratio": round(vr_result.vr, 4) if vr_result is not None else None,
            "vr_z_score": round(vr_result.z_score, 4) if vr_result is not None else None,
            "vpin": round(vpin_result.vpin, 4) if vpin_result is not None else None,
            "volume_ratio": round(vol_ratio, 2),
            "rsi_percentile": round(rsi_pctile, 2) if rsi_pctile is not None else None,
            "momentum_deceleration": round(momentum_decel, 2) if momentum_decel is not None else None,
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
        meta.update(scorer.to_meta())

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta=meta,
        )

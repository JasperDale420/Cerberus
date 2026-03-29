"""
RsiBounce Strategy — Quantitative Mean Reversion (v2)

Type: mean-reversion
Description: Mean-reversion strategy using RSI extremes with Bollinger Band proximity.
Employs six quantitative factors: z-score extremity, OU half-life validity, RSI percentile
rank, volume climax, momentum deceleration, and variance ratio. Gated by VPIN toxicity,
OU half-life bounds, variance ratio regime, BOCPD structural breaks, and kurtosis filters.

Key design decisions:
    - BUY-only oversold + SELL overbought based on regime context
    - Trend filter: align signal direction with higher-TF trend
    - OU half-life gate: mean reversion must be statistically plausible
    - VPIN toxicity gate: reject when order-flow is toxic
    - BOCPD gate: reject signals near structural breaks
    - Kurtosis gate: reject fat-tail regimes
    - GARCH z-score for adaptive entry threshold
    - Adaptive thresholds via AdaptiveThresholdEngine
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

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
from src.quant.regime import AdaptiveThresholdEngine
from src.quant.volatility import GARCHForecaster
from src.strategies.base import BaseStrategy
from src.strategies.confluence import ConfluenceScorer


class RsiBounceStrategy(BaseStrategy):
    """
    RSI Bounce v2 — quantitative mean reversion with six-factor confluence.

    Entry: RSI(14) on 5m extreme + price near BB band + OU/VR/VPIN gate checks.
    Exit: ATR-based stop/target with trailing stop activation.
    """

    name: str = "rsi_bounce"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        self._volume_history: dict[str, deque[float]] = {}
        self._price_history: dict[str, deque[float]] = {}
        self._rsi_history: dict[str, deque[float]] = {}
        self._roc_history: dict[str, deque[float]] = {}
        self._ou_estimators: dict[str, OUEstimator] = {}
        self._vr_calculators: dict[str, VarianceRatioCalculator] = {}
        self._vpin_calculators: dict[str, VPINCalculator] = {}
        self._garch: dict[str, GARCHForecaster] = {}
        super().__init__(config, logger)
        self._threshold_engine = AdaptiveThresholdEngine()

    def _set_params(self, config: dict[str, Any]) -> None:
        """Parse strategy-specific parameters from config."""
        super()._set_params(config)

        self.time_window_start = time_utils.parse_time_string(str(config.get("time_window_start", "09:45")))
        self.time_window_end = time_utils.parse_time_string(str(config.get("time_window_end", "15:30")))
        self.min_bars = int(config.get("min_bars", 20))

        overrides = config.get("_optuna_overrides", {})

        # RSI parameters
        self.rsi_len = int(overrides.get("rsi_len", config.get("rsi_len", 14)))
        self.rsi_oversold = float(overrides.get("rsi_oversold", config.get("rsi_oversold", 30.0)))
        self.rsi_overbought = float(overrides.get("rsi_overbought", config.get("rsi_overbought", 70.0)))

        # Bollinger Band parameters
        self.bb_period = int(overrides.get("bb_period", config.get("bb_period", 20)))
        self.band_sigma = float(overrides.get("band_sigma", config.get("band_sigma", 2.0)))
        self.band_tolerance = float(overrides.get("band_tolerance", config.get("band_tolerance", 10.0)))

        # Volume confirmation
        self.volume_lookback = int(overrides.get("volume_lookback", config.get("volume_lookback", 20)))
        self.volume_min_mult = float(overrides.get("volume_min_mult", config.get("volume_min_mult", 1.0)))
        self.volume_climax_mult = float(overrides.get("volume_climax_mult", config.get("volume_climax_mult", 1.5)))

        # Risk / exit parameters
        self.stop_atr_mult = float(overrides.get("stop_atr_mult", config.get("stop_atr_mult", 2.0)))
        self.target_atr_mult = float(overrides.get("target_atr_mult", config.get("target_atr_mult", 2.0)))
        self.max_hold_minutes = int(overrides.get("max_hold_minutes", config.get("max_hold_minutes", 60)))
        self.confluence_threshold = float(
            overrides.get("confluence_threshold", config.get("confluence_threshold", 50.0))
        )

        # V2 z-score / OU parameters
        self.zscore_entry = float(overrides.get("zscore_entry", config.get("zscore_entry", 2.0)))
        self.zscore_lookback = int(overrides.get("zscore_lookback", config.get("zscore_lookback", 60)))
        self.ou_lookback = int(overrides.get("ou_lookback", config.get("ou_lookback", 60)))
        self.ou_min_obs = int(overrides.get("ou_min_obs", config.get("ou_min_obs", 30)))
        self.max_half_life_bars = float(overrides.get("max_half_life_bars", config.get("max_half_life_bars", 30.0)))
        self.min_half_life_bars = float(overrides.get("min_half_life_bars", config.get("min_half_life_bars", 1.0)))

        # Variance ratio parameters
        self.vr_lookback = int(overrides.get("vr_lookback", config.get("vr_lookback", 120)))
        self.vr_period = int(overrides.get("vr_period", config.get("vr_period", 5)))

        # VPIN parameters
        self.vpin_n_buckets = int(overrides.get("vpin_n_buckets", config.get("vpin_n_buckets", 50)))
        self.vpin_toxicity_threshold = float(
            overrides.get("vpin_toxicity_threshold", config.get("vpin_toxicity_threshold", 0.7))
        )

        # RSI/ROC history
        self.roc_lookback = int(overrides.get("roc_lookback", config.get("roc_lookback", 10)))
        self.rsi_pctile_lookback = int(overrides.get("rsi_pctile_lookback", config.get("rsi_pctile_lookback", 100)))

        # BOCPD / kurtosis gates
        self.bocpd_reject_threshold = float(
            overrides.get("bocpd_reject_threshold", config.get("bocpd_reject_threshold", 0.7))
        )
        self.kurtosis_reject_threshold = float(
            overrides.get("kurtosis_reject_threshold", config.get("kurtosis_reject_threshold", 6.0))
        )

        # TF alignment mode for mean reversion
        self.tf_alignment_mode = "mean_reversion"

        # Disable generic TF alignment from base — we do our own check
        self.higher_tf_alignment = False

    def _ensure_symbol_state(self, symbol: str) -> None:
        """Lazily initialize per-symbol tracking."""
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=self.volume_lookback)
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=60)
        if symbol not in self._rsi_history:
            self._rsi_history[symbol] = deque(maxlen=self.rsi_pctile_lookback)
        if symbol not in self._roc_history:
            self._roc_history[symbol] = deque(maxlen=self.roc_lookback)
        if symbol not in self._ou_estimators:
            self._ou_estimators[symbol] = OUEstimator(lookback=self.ou_lookback, min_observations=self.ou_min_obs)
        if symbol not in self._vr_calculators:
            self._vr_calculators[symbol] = VarianceRatioCalculator(lookback=self.vr_lookback, period=self.vr_period)
        if symbol not in self._vpin_calculators:
            self._vpin_calculators[symbol] = VPINCalculator(
                n_buckets=self.vpin_n_buckets,
                toxicity_threshold=self.vpin_toxicity_threshold,
            )
        if symbol not in self._garch:
            self._garch[symbol] = GARCHForecaster()

    def _ensure_garch(self, symbol: str) -> GARCHForecaster:
        """Lazily initialize GARCH forecaster for a symbol."""
        if symbol not in self._garch:
            self._garch[symbol] = GARCHForecaster()
        return self._garch[symbol]

    def _compute_zscore(self, symbol: str, price: float) -> float:
        """Compute GARCH-adjusted z-score for the current price deviation.

        Uses GARCH conditional vol when available; falls back to rolling std.
        """
        garch = self._ensure_garch(symbol)
        garch.update(price)

        price_hist = self._price_history[symbol]
        if len(price_hist) < 10:
            return 0.0

        prices = np.array(price_hist, dtype=float)
        mean_price = float(np.mean(prices))
        deviation = price - mean_price

        # Try GARCH-adjusted z-score first
        z = garch.conditional_zscore(deviation)

        # Fall back to rolling-std z-score when GARCH has no result yet
        if z == 0.0:
            std_price = float(np.std(prices))
            if std_price > 0:
                z = deviation / std_price

        return z

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        """Process a single bar and optionally generate a signal."""
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

        # --- regime gating: skip SHOCK vol and PREMARKET ---
        snapshot = market_state.regime_snapshot
        if snapshot is not None:
            if snapshot.vol == VolRegime.SHOCK:
                return None
            if snapshot.session == SessionRegime.PREMARKET:
                return None

            # --- BOCPD structural break gate ---
            cp_prob = getattr(snapshot, "changepoint_probability", 0.0)
            if cp_prob > self.bocpd_reject_threshold:
                self.logger.warning(
                    "bocpd_structural_break_rejected",
                    symbol=symbol,
                    changepoint_probability=cp_prob,
                    threshold=self.bocpd_reject_threshold,
                )
                return None

        # --- lazy init per-symbol ---
        self._ensure_symbol_state(symbol)

        # --- update price/RSI history ---
        self._price_history[symbol].append(bar.close)

        # --- kurtosis gate (applied to accumulated ROC history) ---
        # Uses the ROC history (price changes) to detect fat-tail return regimes.
        # Evaluated before appending the current bar's price change to avoid
        # spurious rejection from a single extreme move.
        roc_hist_snapshot = list(self._roc_history[symbol])
        if len(roc_hist_snapshot) >= 10:
            roc_arr = np.array(roc_hist_snapshot, dtype=float)
            roc_std = float(np.std(roc_arr))
            if roc_std > 0:
                kurt = float(scipy_kurtosis(roc_arr, fisher=True))
                if kurt > self.kurtosis_reject_threshold:
                    return None

        # --- multi-timeframe setup ---
        mtf = MultiTimeframeAnalyzer(symbol_state)

        # === RSI CHECK (5-minute) ===
        rsi_5m = mtf.get_rsi("5m", self.rsi_len)
        if rsi_5m is None:
            return None

        # Track RSI history for percentile rank
        self._rsi_history[symbol].append(rsi_5m)

        # BUY-only: require oversold RSI; SELL: require overbought RSI
        is_oversold = rsi_5m < self.rsi_oversold
        is_overbought = rsi_5m > self.rsi_overbought

        if not is_oversold and not is_overbought:
            return None

        # Determine side
        if is_oversold:
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL

        # --- trend filter: don't buy into downtrends, don't short into uptrends ---
        if snapshot is not None:
            if side == OrderSide.BUY and snapshot.trend == TrendRegime.DOWN:
                if rsi_5m > 20.0:
                    return None
            if side == OrderSide.SELL and snapshot.trend == TrendRegime.UP:
                if rsi_5m < 80.0:
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
        if side == OrderSide.BUY:
            distance_to_band = abs(current_price - lower_band)
            tolerance_amount = band_width * (self.band_tolerance / 100.0)
            near_band = distance_to_band <= tolerance_amount or current_price <= lower_band
        else:
            distance_to_band = abs(current_price - upper_band)
            tolerance_amount = band_width * (self.band_tolerance / 100.0)
            near_band = distance_to_band <= tolerance_amount or current_price >= upper_band

        if not near_band:
            return None

        # === V2 QUANTITATIVE GATES ===

        # OU estimator gate — check mean-reversion half-life
        ou_result: OUResult | None = self._ou_estimators[symbol].update(current_price)
        half_life = ou_result.half_life if ou_result is not None else None
        ou_theta = ou_result.theta if ou_result is not None else None

        if half_life is not None:
            if half_life > self.max_half_life_bars or half_life < self.min_half_life_bars:
                return None

        # Variance ratio gate — soft gate: penalise trending via confluence score instead of hard reject
        # Hard rejection was killing all signals in trending_up regimes (all WFO windows).
        vr_result: VarianceRatioResult | None = self._vr_calculators[symbol].update(current_price)
        variance_ratio = vr_result.vr if vr_result is not None else None
        vr_z_score = vr_result.z_score if vr_result is not None else None

        # VPIN toxicity gate
        vpin_result: VPINResult | None = self._vpin_calculators[symbol].update(bar)
        vpin_value = vpin_result.vpin if vpin_result is not None else None

        if vpin_result is not None and vpin_result.is_toxic:
            return None

        # === VOLUME CHECK ===
        self._volume_history[symbol].append(bar.volume)
        vol_hist = self._volume_history[symbol]
        if len(vol_hist) < 5:
            return None
        avg_vol = sum(vol_hist) / len(vol_hist)
        vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 1.0

        # === ROC (rate of change) for momentum deceleration ===
        self._roc_history[symbol].append(bar.close)
        roc_hist = list(self._roc_history[symbol])
        momentum_decelerating = False
        if len(roc_hist) >= 3:
            recent_roc = roc_hist[-1] - roc_hist[-2]
            prior_roc = roc_hist[-2] - roc_hist[-3]
            # Decelerating means momentum is slowing (good for mean reversion entry)
            if side == OrderSide.BUY:
                momentum_decelerating = recent_roc > prior_roc  # less negative = decelerating
            else:
                momentum_decelerating = recent_roc < prior_roc

        # === GARCH z-score ===
        z_score = self._compute_zscore(symbol, current_price)

        # RSI percentile rank
        rsi_hist = list(self._rsi_history[symbol])
        if len(rsi_hist) >= 2:
            rsi_pctile = float(np.mean([r < rsi_5m for r in rsi_hist]))
        else:
            rsi_pctile = 0.5

        # === CONFLUENCE SCORING — 6 v2 factors ===
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)

        # Factor 1: z-score extremity (weight 0.25)
        z_abs = abs(z_score)
        z_score_score = min(100.0, z_abs / max(self.zscore_entry, 0.01) * 50.0)
        scorer.add_factor(
            name="z_score_extremity",
            raw_value=z_score,
            score=z_score_score,
            weight=0.25,
            passed=True,
        )

        # Factor 2: half-life validity (weight 0.20)
        if half_life is not None:
            hl_normalized = 1.0 - min(1.0, half_life / self.max_half_life_bars)
            hl_score = hl_normalized * 100.0
        else:
            hl_score = 50.0
        scorer.add_factor(
            name="half_life_validity",
            raw_value=half_life if half_life is not None else 0.0,
            score=hl_score,
            weight=0.20,
            passed=True,
        )

        # Factor 3: RSI percentile rank (weight 0.20)
        if side == OrderSide.BUY:
            rsi_pctile_score = (1.0 - rsi_pctile) * 100.0
        else:
            rsi_pctile_score = rsi_pctile * 100.0
        scorer.add_factor(
            name="rsi_percentile_rank",
            raw_value=rsi_pctile,
            score=rsi_pctile_score,
            weight=0.20,
            passed=True,
        )

        # Factor 4: volume climax (weight 0.15)
        vol_climax = vol_ratio >= self.volume_climax_mult
        vol_climax_score = min(100.0, vol_ratio / max(self.volume_climax_mult, 0.01) * 60.0)
        scorer.add_factor(
            name="volume_climax",
            raw_value=vol_ratio,
            score=vol_climax_score,
            weight=0.15,
            passed=True,
        )

        # Factor 5: momentum deceleration (weight 0.10)
        momentum_score = 75.0 if momentum_decelerating else 25.0
        scorer.add_factor(
            name="momentum_deceleration",
            raw_value=1.0 if momentum_decelerating else 0.0,
            score=momentum_score,
            weight=0.10,
            passed=True,
        )

        # Factor 6: variance ratio (weight 0.10)
        if variance_ratio is not None:
            vr_score = max(0.0, min(100.0, (1.0 - variance_ratio) * 100.0))
        else:
            vr_score = 50.0
        scorer.add_factor(
            name="variance_ratio",
            raw_value=variance_ratio if variance_ratio is not None else 1.0,
            score=vr_score,
            weight=0.10,
            passed=True,
        )

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
            "bb_mean": round(bb_mean, 4),
            "bb_std": round(bb_std, 6),
            "volume_ratio": round(vol_ratio, 2),
            "z_score": round(z_score, 4),
            "half_life": round(half_life, 4) if half_life is not None else None,
            "ou_theta": round(ou_theta, 6) if ou_theta is not None else None,
            "variance_ratio": round(variance_ratio, 4) if variance_ratio is not None else None,
            "vr_z_score": round(vr_z_score, 4) if vr_z_score is not None else None,
            "vpin": round(vpin_value, 4) if vpin_value is not None else None,
            "vol_climax": vol_climax,
            "momentum_decelerating": momentum_decelerating,
            "exit_config": {
                "trailing_enabled": True,
                "trail_timeframe": "5m",
                "trail_lookback": 3,
                "trail_min_profit_r": 0.5,
                "partial_exits": [(1.5, 0.5)],
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

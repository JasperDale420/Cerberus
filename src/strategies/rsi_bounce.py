"""
RsiBounce Strategy

Type: mean-reversion
Description: Buys when RSI drops below extreme oversold levels on the 5-minute
timeframe while price is near the lower Bollinger Band, and sells when RSI
reaches extreme overbought levels near the upper band. Designed for range-bound
(flat trend) markets during midday sessions.
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
from src.strategies.confluence import ConfluenceScorer, score_threshold


class RsiBounceStrategy(BaseStrategy):
    """
    RSI Bounce — extreme RSI mean-reversion confirmed by Bollinger Band proximity.

    Edge: When RSI reaches extreme levels (< 10 or > 90) on the 5-minute chart
    AND price is near a Bollinger Band, there is a high probability of a short-term
    reversion to the mean. Works best in range-bound, non-trending markets.

    Entry BUY:  RSI(5m) < rsi_oversold AND price within band_tolerance% of lower BB
    Entry SELL: RSI(5m) > rsi_overbought AND price within band_tolerance% of upper BB
    Exit: Trailing stop + partial exits + max hold time
    """

    name: str = "rsi_bounce"

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

        # RSI parameters
        self.rsi_len = int(overrides.get("rsi_len", config.get("rsi_len", 14)))
        self.rsi_oversold = float(overrides.get("rsi_oversold", config.get("rsi_oversold", 10.0)))
        self.rsi_overbought = float(overrides.get("rsi_overbought", config.get("rsi_overbought", 90.0)))

        # Bollinger Band parameters
        self.bb_period = int(overrides.get("bb_period", config.get("bb_period", 20)))
        self.band_sigma = float(overrides.get("band_sigma", config.get("band_sigma", 2.0)))
        self.band_tolerance = float(overrides.get("band_tolerance", config.get("band_tolerance", 0.5)))

        # Risk / exit parameters
        self.stop_atr_mult = float(overrides.get("stop_atr_mult", config.get("stop_atr_mult", 1.5)))
        self.target_atr_mult = float(overrides.get("target_atr_mult", config.get("target_atr_mult", 3.0)))
        self.max_hold_minutes = int(overrides.get("max_hold_minutes", config.get("max_hold_minutes", 60)))
        self.confluence_threshold = float(
            overrides.get("confluence_threshold", config.get("confluence_threshold", 60.0))
        )

        # Mean reversion: only trade when higher TF is flat
        self.tf_alignment_mode = "mean_reversion"

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
            # Skip shock volatility — mean reversion unreliable in panic
            if snapshot.vol == VolRegime.SHOCK:
                return None
            # Skip premarket — not enough liquidity for mean reversion
            if snapshot.session == SessionRegime.PREMARKET:
                return None

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

        # === BOLLINGER BAND CHECK (5-minute) ===
        bb_data = mtf._cache.get("5m")  # noqa: SLF001 — access after get_rsi triggers cache
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
            # Price should be near the lower Bollinger Band
            distance_to_lower = abs(current_price - lower_band)
            tolerance_amount = band_width * (self.band_tolerance / 100.0)
            near_lower_band = distance_to_lower <= tolerance_amount or current_price <= lower_band
            if not near_lower_band:
                return None
            side = OrderSide.BUY
        else:
            # Price should be near the upper Bollinger Band
            distance_to_upper = abs(current_price - upper_band)
            tolerance_amount = band_width * (self.band_tolerance / 100.0)
            near_upper_band = distance_to_upper <= tolerance_amount or current_price >= upper_band
            if not near_upper_band:
                return None
            side = OrderSide.SELL

        # --- higher TF alignment (mean reversion: requires flat higher TF) ---
        if not self._check_higher_tf_alignment(symbol_state, side):
            return None

        # === CONFLUENCE SCORING ===
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)

        # Factor 1: RSI extremity — how extreme is the RSI reading?
        # For BUY: lower RSI = higher score (RSI 0 -> 100, RSI 10 -> 50)
        # For SELL: higher RSI = higher score (RSI 100 -> 100, RSI 90 -> 50)
        if side == OrderSide.BUY:
            rsi_score = score_threshold(rsi_5m, self.rsi_oversold, 0.0, invert=True)
        else:
            rsi_score = score_threshold(rsi_5m, self.rsi_overbought, 100.0, invert=False)

        scorer.add_factor(
            name="rsi_extremity",
            raw_value=rsi_5m,
            score=rsi_score,
            weight=0.40,
            passed=True,
        )

        # Factor 2: Band proximity — how close is price to the Bollinger Band?
        # Closer to band (or beyond it) = higher score
        if side == OrderSide.BUY:
            if current_price <= lower_band:
                band_prox_score = 100.0
            else:
                band_prox_score = max(0.0, (1.0 - distance_to_lower / tolerance_amount) * 100.0)
        else:
            if current_price >= upper_band:
                band_prox_score = 100.0
            else:
                band_prox_score = max(0.0, (1.0 - distance_to_upper / tolerance_amount) * 100.0)

        scorer.add_factor(
            name="band_proximity",
            raw_value=current_price,
            score=band_prox_score,
            weight=0.35,
            passed=True,
        )

        # Factor 3: Mean reversion alignment — are higher TFs also flat/choppy?
        mr_alignment = mtf.get_mean_reversion_alignment()
        mr_score = mr_alignment * 100.0  # 0.0-1.0 -> 0-100

        scorer.add_factor(
            name="mr_alignment",
            raw_value=mr_alignment,
            score=mr_score,
            weight=0.25,
            passed=mr_alignment > 0.0,
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
            "strategy_version": "rsi_bounce_v1",
            "atr_5m": round(atr_5m, 6),
            "rsi_5m": round(rsi_5m, 2),
            "bb_upper": round(upper_band, 4),
            "bb_lower": round(lower_band, 4),
            "bb_mean": round(bb_mean, 4),
            "bb_std": round(bb_std, 6),
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

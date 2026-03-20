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
from src.data.requirements import DataRequirements
from src.quant.filters import KalmanMeanTracker
from src.quant.statistics import HurstExponent
from src.quant.volatility import GARCHForecaster
from src.strategies.base import BaseStrategy
from src.strategies.confluence import (
    ConfluenceScorer,
    score_deviation,
    score_threshold,
    score_volume,
)


class TrendRiderProStrategy(BaseStrategy):
    """Trend Rider Pro v4 -- simplified pullback-in-trend strategy.

    Trades pullbacks to the Kalman/EMA-20 anchor within multi-timeframe
    trend alignment, scored via ConfluenceScorer with 4 factors.
    Targets 2-5 trades/day across a 30+ symbol universe.
    """

    name: str = "trend_rider_pro"
    data_requirements = DataRequirements(streams=["bars", "quotes"], on_scan=["prior_day"])

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)
        self._kalman: dict[str, KalmanMeanTracker] = {}
        self._hurst: dict[str, HurstExponent] = {}
        self._garch: dict[str, GARCHForecaster] = {}

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.confluence_threshold = float(config.get("confluence_threshold", 50))
        self.time_window_start = time_utils.parse_time_string(str(config.get("time_window_start", "09:35")))
        self.time_window_end = time_utils.parse_time_string(str(config.get("time_window_end", "15:50")))
        self.min_bars = int(config.get("min_bars", 30))
        self.min_trend_alignment = float(config.get("min_trend_alignment", 0.35))

        overrides = config.get("_optuna_overrides", {})
        self.pullback_threshold = float(overrides.get("pullback_threshold", config.get("pullback_threshold", 0.005)))
        self.stop_atr_mult = float(overrides.get("stop_atr_mult", config.get("stop_atr_mult", 2.0)))
        self.target_atr_mult = float(overrides.get("target_atr_mult", config.get("target_atr_mult", 3.5)))
        self.trail_min_profit_r = float(overrides.get("trail_min_profit_r", config.get("trail_min_profit_r", 0.5)))
        self.max_hold_minutes = int(overrides.get("max_hold_minutes", config.get("max_hold_minutes", 120)))

    # ------------------------------------------------------------------
    # Per-symbol quant component initialization
    # ------------------------------------------------------------------

    def _ensure_quant_state(self, symbol: str) -> None:
        if symbol not in self._kalman:
            self._kalman[symbol] = KalmanMeanTracker(process_noise=0.01, measurement_noise=1.0)
        if symbol not in self._hurst:
            self._hurst[symbol] = HurstExponent(min_observations=100, lookback=500)
        if symbol not in self._garch:
            self._garch[symbol] = GARCHForecaster(min_observations=50, lookback=500, refit_interval=20)

    def _update_quant_state(self, symbol: str, bar: Bar) -> None:
        close = bar.close
        self._kalman[symbol].update(close)
        self._hurst[symbol].update(close)
        self._garch[symbol].update(close)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        self._ensure_quant_state(symbol)
        self._update_quant_state(symbol, bar)

        if not self._is_evaluation_bar(bar):
            return None

        t = time_utils.get_eastern_time_of_day(bar.time)
        if not (self.time_window_start <= t <= self.time_window_end):
            return None

        if not self._check_cooldown(symbol, bar.time):
            return None

        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        # --- regime gating (simplified) ---
        snapshot = market_state.regime_snapshot
        if snapshot is not None:
            if snapshot.vol == VolRegime.SHOCK:
                return None
            if snapshot.session == SessionRegime.PREMARKET:
                return None
            if snapshot.session == SessionRegime.CLOSE:
                return None

        # --- direction detection ---
        mtf = MultiTimeframeAnalyzer(symbol_state)
        side = self._detect_side(bar, mtf, symbol)
        if side is None:
            return None

        # --- 15m higher-timeframe trend confirmation ---
        if not self._require_higher_tf_trend(mtf, side):
            return None

        # --- volume confirmation: require 1.5x avg volume ---
        avg_vol = self._get_avg_volume(symbol_state)
        if avg_vol > 0 and float(bar.volume) < avg_vol * 1.5:
            return None

        # --- confluence scoring ---
        scorer = self._score_confluence(bar, side, mtf, symbol_state, symbol)
        if not scorer.passes_threshold():
            return None

        # --- stop / target ---
        atr_5m = mtf.get_atr("5m", 14)
        if atr_5m is None or atr_5m <= 0:
            return None

        stop_price = self._compute_stop(bar, side, mtf, atr_5m, symbol)
        stop_distance = abs(bar.close - stop_price)
        stop_distance = self._apply_regime_volatility_multiplier(stop_distance, market_state)

        if side == OrderSide.BUY:
            stop_price = bar.close - stop_distance
            target_price = bar.close + self.target_atr_mult * atr_5m
        else:
            stop_price = bar.close + stop_distance
            target_price = bar.close - self.target_atr_mult * atr_5m

        # --- build meta ---
        meta = scorer.to_meta()
        meta["strategy_version"] = "trend_rider_pro_v4"
        meta["atr_5m"] = round(atr_5m, 6)

        # Observability: log quant state but don't gate on it
        hurst_result = self._hurst[symbol].update(bar.close)
        if hurst_result is not None:
            meta["hurst_H"] = round(hurst_result.H, 4)
        garch_result = self._garch[symbol]._last_result
        if garch_result is not None:
            meta["garch_conditional_vol"] = round(garch_result.conditional_vol, 6)
        kalman_state = self._kalman[symbol].state
        if kalman_state is not None:
            meta["kalman_state"] = round(kalman_state, 4)

        # Exit config — single consistent config
        exit_config = {
            "trailing_enabled": True,
            "trail_timeframe": "5m",
            "trail_lookback": 3,
            "trail_min_profit_r": self.trail_min_profit_r,
            "partial_exits": [(1.5, 0.50), (2.5, 0.25)],  # scale out: 50% at 1.5R, 25% at 2.5R
            "max_hold_minutes": self.max_hold_minutes,
            "vol_adaptive": True,
        }
        meta["exit_config"] = exit_config

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            size_hint=scorer.conviction_multiplier(),
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Direction detection
    # ------------------------------------------------------------------

    def _detect_side(
        self,
        bar: Bar,
        mtf: MultiTimeframeAnalyzer,
        symbol: str = "",
    ) -> OrderSide | None:
        """Determine trade direction from VWAP filter + EMA alignment + pullback.

        - VWAP filter: Long only if price > VWAP, short only if price < VWAP
        - EMA alignment: Buy if EMA-9 > EMA-20 on 5m, Sell if EMA-9 < EMA-20
        - Pullback: Price within pullback_threshold of Kalman anchor (EMA-20 fallback)
        """
        vwap_dist_1m = mtf.get_vwap_distance("1m")

        # EMA alignment on 5m
        ema9_5m = mtf.get_ema("5m", 9)
        ema20_5m = mtf.get_ema("5m", 20)
        if ema9_5m is None or ema20_5m is None:
            return None

        # Kalman anchor (EMA-20 fallback)
        kalman_state = self._kalman[symbol].state if symbol in self._kalman else None
        pullback_anchor = kalman_state if kalman_state is not None and kalman_state > 0 else ema20_5m
        if pullback_anchor is None or pullback_anchor <= 0:
            return None

        pullback_pct = abs(bar.close - pullback_anchor) / bar.close

        # --- BUY side ---
        buy_vwap_ok = vwap_dist_1m is None or vwap_dist_1m >= 0
        buy_ema_aligned = ema9_5m > ema20_5m
        near_anchor = pullback_pct <= self.pullback_threshold

        if buy_vwap_ok and buy_ema_aligned and near_anchor:
            alignment = mtf.get_trend_alignment(OrderSide.BUY)
            if alignment >= self.min_trend_alignment:
                return OrderSide.BUY

        # --- SELL side disabled ---
        # 2025 analysis: short signals had negative edge, removing sell side
        # to focus capital on higher-quality long pullback setups.

        return None

    # ------------------------------------------------------------------
    # Confluence scoring (4 factors)
    # ------------------------------------------------------------------

    def _score_confluence(
        self,
        bar: Bar,
        side: OrderSide,
        mtf: MultiTimeframeAnalyzer,
        symbol_state: SymbolState,
        symbol: str = "",
    ) -> ConfluenceScorer:
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)

        # 1. MTF trend alignment (weight 0.45) — strongest predictor
        alignment = mtf.get_trend_alignment(side)
        alignment_score = alignment * 100.0
        scorer.add_factor(
            "mtf_trend_alignment",
            raw_value=alignment,
            score=alignment_score,
            weight=0.45,
            passed=alignment > 0.3,
        )

        # 2. Pullback quality (weight 0.20) — widened range [0.0005, 0.01]
        kalman_state = self._kalman[symbol].state if symbol in self._kalman else None
        ema20_5m = mtf.get_ema("5m", 20)
        pullback_anchor = kalman_state if kalman_state is not None and kalman_state > 0 else ema20_5m
        pullback_pct = 0.0
        if pullback_anchor is not None and pullback_anchor > 0:
            pullback_pct = abs(bar.close - pullback_anchor) / bar.close
        pb_score = score_deviation(pullback_pct, 0.0005, 0.01)
        pb_score = max(0.0, 100.0 - pb_score)
        scorer.add_factor(
            "pullback_quality",
            raw_value=pullback_pct,
            score=pb_score,
            weight=0.20,
            passed=0.0005 <= pullback_pct <= 0.01,
        )

        # 3. Volume (weight 0.15) — reduced since hard 1.2x gate already filters
        avg_vol = self._get_avg_volume(symbol_state)
        vol_score = 0.0
        if avg_vol > 0:
            vol_score = score_volume(float(bar.volume), avg_vol, 0.8, 2.0)
        scorer.add_factor(
            "volume",
            raw_value=float(bar.volume),
            score=vol_score,
            weight=0.15,
            passed=avg_vol > 0 and float(bar.volume) >= avg_vol * 0.8,
        )

        # 4. ADX strength (weight 0.20) — require ADX >= 20 for real trend
        adx_5m = mtf.get_adx("5m")
        adx_val = adx_5m if adx_5m is not None else 0.0
        adx_score = score_threshold(adx_val, 20.0, 40.0)
        scorer.add_factor(
            "adx_strength",
            raw_value=adx_val,
            score=adx_score,
            weight=0.20,
            passed=adx_val >= 20.0,
        )

        return scorer

    # ------------------------------------------------------------------
    # Stop placement
    # ------------------------------------------------------------------

    def _compute_stop(
        self,
        bar: Bar,
        side: OrderSide,
        mtf: MultiTimeframeAnalyzer,
        atr_5m: float,
        symbol: str = "",
    ) -> float:
        """Compute raw stop price before regime multiplier.

        Uses ATR as primary stop distance. GARCH enhances when available
        (widens stop if GARCH vol suggests higher risk).
        """
        vol_distance = self.stop_atr_mult * atr_5m

        # GARCH enhancement: widen stop if GARCH suggests higher vol
        garch_result = self._garch[symbol]._last_result if symbol in self._garch else None
        if garch_result is not None and garch_result.conditional_vol > 0:
            garch_distance = self.stop_atr_mult * garch_result.conditional_vol * bar.close
            vol_distance = max(vol_distance, garch_distance)

        min_distance = 2.0 * atr_5m  # minimum stop distance to survive noise

        if side == OrderSide.BUY:
            swings = mtf.get_swing_lows("5m", lookback=8)
            if swings:
                swing_stop = swings[0]
                # Ensure minimum distance from entry
                if bar.close - swing_stop < min_distance:
                    swing_stop = bar.close - min_distance
                return swing_stop
            return bar.low - vol_distance
        swings = mtf.get_swing_highs("5m", lookback=8)
        if swings:
            swing_stop = swings[0]
            if swing_stop - bar.close < min_distance:
                swing_stop = bar.close + min_distance
            return swing_stop
        return bar.high + vol_distance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_avg_volume(symbol_state: SymbolState) -> float:
        """20-bar average volume from indicators or manual calc."""
        cached = symbol_state.indicators.get("sma_vol:20")
        if cached is not None:
            try:
                return float(cached)
            except (TypeError, ValueError):
                pass
        bars = symbol_state.bars
        if not bars:
            return 0.0
        recent = list(bars)[-20:]
        if not recent:
            return 0.0
        return sum(float(b.volume) for b in recent) / len(recent)

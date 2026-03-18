from __future__ import annotations

from datetime import date
from typing import Any

from src.analysis.ou_estimator import OUEstimator, OUResult
from src.analysis.vpin import VPINCalculator
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
from src.data.requirements import DataRequirements
from src.strategies.base import BaseStrategy
from src.strategies.confluence import (
    ConfluenceScorer,
    score_candle_quality,
    score_deviation,
    score_threshold,
    score_volume,
)


class MeanReversionProStrategy(BaseStrategy):
    """Consolidated mean-reversion strategy replacing VWAP Reversion, Index Mean
    Reversion, Gap Fill, VIX Spike Fade, and Failed Breakout.

    Uses confluence scoring across six weighted factors, multi-timeframe
    analysis, structure-based stops, and regime/session gating.

    Includes a daily loss throttle: after ``max_daily_losers`` consecutive
    losing trades in a single session, the strategy stops trading for that day.
    This prevents compounding losses in unfavorable regimes (WFO showed 8-11
    consecutive losers in bad windows).
    """

    name: str = "mean_reversion_pro"
    data_requirements = DataRequirements(streams=["bars", "quotes"], on_scan=["flow", "prior_day"])

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)
        # Daily loss tracking: {date_str: consecutive_losers}
        self._daily_losers: dict[str, int] = {}
        self._daily_trade_results: dict[str, list[float]] = {}
        # Per-symbol VPIN calculators for toxicity filtering
        self._vpin: dict[str, VPINCalculator] = {}
        self._vpin_threshold: float = float(config.get("vpin_threshold", 0.7))
        # Per-symbol OU estimators for dynamic thresholds
        self._ou: dict[str, OUEstimator] = {}
        self._ou_enabled: bool = bool(config.get("ou_enabled", False))
        self._ou_lookback: int = int(config.get("ou_lookback", 60))

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.confluence_threshold: float = float(config.get("confluence_threshold", 65.0))
        self.time_window_start: str = str(config.get("time_window_start", "09:35"))
        self.time_window_end: str = str(config.get("time_window_end", "15:30"))
        self.min_bars: int = int(config.get("min_bars", 30))
        # Mean reversion mode for higher-TF alignment in base
        self.tf_alignment_mode = "mean_reversion"

        # Daily loss throttle — stop after N consecutive losers in a day
        self.max_daily_losers: int = int(config.get("max_daily_losers", 6))

        # Tunable params (also settable via _optuna_overrides)
        overrides = config.get("_optuna_overrides", {})
        self.vwap_dist_threshold = float(overrides.get("vwap_dist_threshold", config.get("vwap_dist_threshold", 0.003)))
        self.bb_pos_threshold = float(overrides.get("bb_pos_threshold", config.get("bb_pos_threshold", 0.3)))
        self.stop_atr_mult = float(overrides.get("stop_atr_mult", config.get("stop_atr_mult", 1.5)))
        self.max_hold_minutes = int(overrides.get("max_hold_minutes", config.get("max_hold_minutes", 20)))

    # ------------------------------------------------------------------
    # Daily loss throttle
    # ------------------------------------------------------------------

    def record_trade_result(self, pnl: float, trade_date: date) -> None:
        """Called by ExecutionEngine when a MRP trade closes.

        Tracks consecutive losers per day for the daily throttle.
        """
        key = trade_date.isoformat()
        if key not in self._daily_losers:
            self._daily_losers[key] = 0
        if pnl < 0:
            self._daily_losers[key] += 1
        else:
            self._daily_losers[key] = 0  # reset streak on a win

    def _daily_throttle_ok(self, bar: Bar) -> bool:
        """Return False if we've hit the daily consecutive loser limit."""
        key = time_utils.to_eastern_time(bar.time).date().isoformat()
        consecutive = self._daily_losers.get(key, 0)
        if consecutive >= self.max_daily_losers:
            self.logger.debug(
                "mean_reversion_pro: daily throttle active",
                consecutive_losers=consecutive,
                limit=self.max_daily_losers,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Regime / session gate
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_ok(market_state: MarketState) -> bool:
        """Return True if regime allows mean-reversion entries."""
        snap = market_state.regime_snapshot
        if snap is None:
            return False

        # Never trade in SHOCK volatility
        if snap.vol == VolRegime.SHOCK:
            return False

        # Skip pre-market
        if snap.session == SessionRegime.PREMARKET:
            return False

        # Allow mean-reversion in FLAT trend always, or in UP/DOWN when
        # vol is LOW or NORMAL (not just LOW). Mean-reversion signals
        # still require extreme VWAP/BB readings to trigger, so the
        # confluence scoring gates quality even in trending markets.
        if snap.trend == TrendRegime.FLAT:
            return True
        return snap.vol in (VolRegime.LOW, VolRegime.NORMAL)

    # ------------------------------------------------------------------
    # Direction detection
    # ------------------------------------------------------------------

    def _detect_side(
        self,
        vwap_dist: float,
        bb_pos: float,
        vwap_threshold: float | None = None,
        bb_threshold: float | None = None,
    ) -> OrderSide | None:
        """Determine trade direction from VWAP distance and BB position.

        Uses soft-OR logic: one indicator must be at full threshold (primary),
        the other just needs to agree on direction (secondary). This doubles
        trade frequency vs the old binary AND while maintaining directional
        agreement between VWAP and BB.

        Returns OrderSide.BUY, OrderSide.SELL, or None.
        """
        vt = vwap_threshold if vwap_threshold is not None else self.vwap_dist_threshold
        bt = bb_threshold if bb_threshold is not None else self.bb_pos_threshold

        # BUY: primary VWAP extreme + BB agrees on direction, OR primary BB extreme + VWAP agrees
        if (vwap_dist < -vt and bb_pos < 0) or (bb_pos < -bt and vwap_dist < 0):
            return OrderSide.BUY
        # SELL: primary VWAP extreme + BB agrees, OR primary BB extreme + VWAP agrees
        if (vwap_dist > vt and bb_pos > 0) or (bb_pos > bt and vwap_dist > 0):
            return OrderSide.SELL
        return None

    # ------------------------------------------------------------------
    # Confluence scoring
    # ------------------------------------------------------------------

    def _score_entry(
        self,
        side: OrderSide,
        vwap_dist: float,
        bb_pos: float,
        rsi: float | None,
        bar: Bar,
        avg_vol: float,
        mr_alignment: float,
    ) -> ConfluenceScorer:
        """Build and return a scored confluence assessment."""
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)

        # 1. VWAP deviation (weight 0.25)
        vwap_score = score_deviation(vwap_dist, 0.003, 0.015)
        scorer.add_factor(
            "vwap_deviation",
            raw_value=vwap_dist,
            score=vwap_score,
            weight=0.25,
            passed=abs(vwap_dist) >= 0.003,
        )

        # 2. BB position (weight 0.20)
        if side == OrderSide.BUY:
            bb_score = score_deviation(bb_pos, 0.5, 1.0) if bb_pos < -0.5 else 0.0
        else:
            bb_score = score_deviation(bb_pos, 0.5, 1.0) if bb_pos > 0.5 else 0.0
        scorer.add_factor(
            "bb_position",
            raw_value=bb_pos,
            score=bb_score,
            weight=0.20,
            passed=bb_score > 0.0,
        )

        # 3. RSI extremity (weight 0.20)
        if rsi is not None:
            if side == OrderSide.BUY:
                rsi_score = score_threshold(rsi, 30, 0, invert=True)
            else:
                rsi_score = score_threshold(rsi, 70, 100)
        else:
            rsi_score = 0.0
        scorer.add_factor(
            "rsi_extremity",
            raw_value=rsi if rsi is not None else 0.0,
            score=rsi_score,
            weight=0.20,
            passed=rsi_score > 0.0,
        )

        # 4. Candle quality (weight 0.15)
        side_str = "buy" if side == OrderSide.BUY else "sell"
        candle_score = score_candle_quality(bar.open, bar.high, bar.low, bar.close, side_str)
        scorer.add_factor(
            "candle_quality",
            raw_value=candle_score,
            score=candle_score,
            weight=0.15,
        )

        # 5. Volume (weight 0.10)
        vol_score = score_volume(bar.volume, avg_vol, 1.0, 2.5) if avg_vol > 0 else 0.0
        scorer.add_factor(
            "volume",
            raw_value=bar.volume,
            score=vol_score,
            weight=0.10,
            passed=vol_score > 0.0,
        )

        # 6. MTF flatness (weight 0.10)
        mtf_score = mr_alignment * 100.0
        scorer.add_factor(
            "mtf_flatness",
            raw_value=mr_alignment,
            score=mtf_score,
            weight=0.10,
            passed=mr_alignment > 0.3,
        )

        return scorer

    # ------------------------------------------------------------------
    # Stop / target
    # ------------------------------------------------------------------

    def _compute_stop(
        self,
        side: OrderSide,
        bar: Bar,
        mtf: MultiTimeframeAnalyzer,
        market_state: MarketState,
    ) -> float:
        """Compute structure-based stop with regime volatility adjustment."""
        atr_5m = mtf.get_atr("5m", 14)
        fallback_atr = atr_5m if atr_5m is not None and atr_5m > 0 else bar.close * 0.005

        if side == OrderSide.BUY:
            swings = mtf.get_swing_lows("5m", 3)
            raw_stop = swings[0] if swings else (bar.low - self.stop_atr_mult * fallback_atr)
            base_distance = max(bar.close - raw_stop, fallback_atr * 0.5)
            adjusted = self._apply_regime_volatility_multiplier(base_distance, market_state)
            return bar.close - adjusted
        else:
            swings = mtf.get_swing_highs("5m", 3)
            raw_stop = swings[0] if swings else (bar.high + self.stop_atr_mult * fallback_atr)
            base_distance = max(raw_stop - bar.close, fallback_atr * 0.5)
            adjusted = self._apply_regime_volatility_multiplier(base_distance, market_state)
            return bar.close + adjusted

    def _compute_target(
        self,
        side: OrderSide,
        mtf: MultiTimeframeAnalyzer,
        bar: Bar,
        stop_price: float,
    ) -> float:
        """Target is VWAP, but ensure minimum 1.5R reward-to-risk ratio.

        WFO showed avg_win < avg_loss in most windows — the pure-VWAP
        target was too close.  Now we enforce target >= 1.5 * stop_distance
        from entry, guaranteeing a minimum payoff ratio.
        """
        stop_distance = abs(bar.close - stop_price)
        min_target_distance = stop_distance * 1.5

        vwap_dist = mtf.get_vwap_distance("1m")
        if vwap_dist is not None:
            denom = 1.0 + vwap_dist
            if denom > 0:
                vwap_target = bar.close / denom
                vwap_target_dist = abs(vwap_target - bar.close)
                # Use VWAP target only if it gives >= 1.5R
                if vwap_target_dist >= min_target_distance:
                    return vwap_target

        # Fall back to 1.5R minimum target
        if side == OrderSide.BUY:
            return bar.close + min_target_distance
        return bar.close - min_target_distance

    # ------------------------------------------------------------------
    # Exit config
    # ------------------------------------------------------------------

    def _exit_config(self) -> dict[str, Any]:
        max_hold = self.max_hold_minutes
        return {
            "trailing_enabled": True,
            "trail_timeframe": "5m",
            "trail_lookback": 3,
            "trail_min_profit_r": 0.5,
            "partial_exits": [(1.5, 0.5)],
            "max_hold_minutes": max_hold,
            "vol_adaptive": True,
        }

    # ------------------------------------------------------------------
    # on_bar
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        # --- Timeframe gate: only evaluate on signal_timeframe closes ---
        if not self._is_evaluation_bar(bar):
            return None

        # --- Pre-flight checks ---
        if not self._check_cooldown(symbol, bar.time):
            return None

        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        if not time_utils.in_time_window_str(bar.time, self.time_window_start, self.time_window_end):
            return None

        # Regime: skip SHOCK vol and PREMARKET
        snap = market_state.regime_snapshot
        if snap is not None:
            if snap.vol == VolRegime.SHOCK:
                return None
            if snap.session == SessionRegime.PREMARKET:
                return None

        # --- Daily loss throttle ---
        if not self._daily_throttle_ok(bar):
            return None

        # --- VPIN toxicity gate ---
        if symbol not in self._vpin:
            self._vpin[symbol] = VPINCalculator(logger=self.logger)
        vpin_result = self._vpin[symbol].update(bar)
        if vpin_result is not None and vpin_result.vpin > self._vpin_threshold:
            return None

        # --- Multi-timeframe analysis ---
        mtf = MultiTimeframeAnalyzer(symbol_state)

        # 5m ADX gate: skip only very strong 5m trends (ADX>45)
        adx_5m = mtf.get_adx("5m")
        if adx_5m is not None and adx_5m > 45.0:
            return None

        mr_alignment = mtf.get_mean_reversion_alignment()

        vwap_dist = mtf.get_vwap_distance("1m")
        if vwap_dist is None:
            return None

        bb_pos = mtf.get_bb_position("5m")
        if bb_pos is None:
            return None

        # --- OU dynamic thresholds ---
        dynamic_vwap_threshold = self.vwap_dist_threshold
        dynamic_bb_threshold = self.bb_pos_threshold
        ou_result: OUResult | None = None
        if self._ou_enabled:
            if symbol not in self._ou:
                self._ou[symbol] = OUEstimator(lookback=self._ou_lookback)
            ou_result = self._ou[symbol].update(vwap_dist)
            if ou_result is not None:
                dynamic_vwap_threshold = self.vwap_dist_threshold * ou_result.scaling_factor
                dynamic_bb_threshold = self.bb_pos_threshold * ou_result.scaling_factor

        # --- Direction ---
        side = self._detect_side(vwap_dist, bb_pos, dynamic_vwap_threshold, dynamic_bb_threshold)
        if side is None:
            return None

        # --- Confluence scoring ---
        rsi = mtf.get_rsi("5m", 14)
        avg_vol: float = float(symbol_state.meta.get("avg_volume_20", 0) or 0)
        mr_alignment = mtf.get_mean_reversion_alignment()

        scorer = self._score_entry(
            side=side,
            vwap_dist=vwap_dist,
            bb_pos=bb_pos,
            rsi=rsi,
            bar=bar,
            avg_vol=avg_vol,
            mr_alignment=mr_alignment,
        )

        if not scorer.passes_threshold():
            self.logger.debug(
                "mean_reversion_pro: below threshold",
                symbol=symbol,
                score=round(scorer.score(), 2),
                threshold=self.confluence_threshold,
            )
            return None

        # --- Stop / target ---
        stop_price = self._compute_stop(side, bar, mtf, market_state)
        target_price = self._compute_target(side, mtf, bar, stop_price)

        # Sanity: stop must be on correct side of entry
        if side == OrderSide.BUY and stop_price >= bar.close:
            return None
        if side == OrderSide.SELL and stop_price <= bar.close:
            return None

        # --- Build meta ---
        meta: dict[str, Any] = scorer.to_meta()
        meta["exit_config"] = self._exit_config()
        meta["vwap_distance"] = vwap_dist
        meta["bb_position"] = bb_pos
        meta["mr_alignment"] = mr_alignment
        if vpin_result is not None:
            meta["vpin_score"] = round(vpin_result.vpin, 4)
        if self._ou_enabled and ou_result is not None:
            meta["ou_theta"] = round(ou_result.theta, 4)
            meta["ou_half_life"] = round(ou_result.half_life, 2)
            meta["ou_scaling"] = round(ou_result.scaling_factor, 3)

        # --- Record cooldown ---
        self.last_signal_time[symbol] = bar.time

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

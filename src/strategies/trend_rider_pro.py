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
    TrendRegime,
    VolRegime,
)
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.data.requirements import DataRequirements
from src.strategies.base import BaseStrategy
from src.strategies.confluence import (
    ConfluenceScorer,
    score_deviation,
    score_threshold,
    score_volume,
)


def _score_momentum(rsi: float, side: OrderSide) -> float:
    """Score RSI for trend-pullback momentum (0-100).

    For BUY: RSI should be pulled back but not oversold/overbought.
      - 45-55 (healthy pullback zone) => 100
      - 40-45 or 55-70 (acceptable)   => 60
      - outside                        => 0

    For SELL: mirrored — RSI should be 30-60 territory.
      - 45-55 => 100
      - 30-45 or 55-60 => 60
      - outside => 0
    """
    if side == OrderSide.BUY:
        if 45.0 <= rsi <= 55.0:
            return 100.0
        if 40.0 <= rsi < 45.0 or 55.0 < rsi <= 70.0:
            return 60.0
        return 0.0
    # SELL
    if 45.0 <= rsi <= 55.0:
        return 100.0
    if 30.0 <= rsi < 45.0 or 55.0 < rsi <= 60.0:
        return 60.0
    return 0.0


def _score_session(session: SessionRegime | None) -> float:
    """Map session regime to a quality score (0-100)."""
    if session is None:
        return 40.0
    _map: dict[SessionRegime, float] = {
        SessionRegime.OPENING: 100.0,
        SessionRegime.POWER_HOUR: 80.0,
        SessionRegime.MIDDAY: 40.0,
        SessionRegime.CLOSE: 20.0,
        SessionRegime.PREMARKET: 0.0,
    }
    return _map.get(session, 40.0)


class TrendRiderProStrategy(BaseStrategy):
    """Trend Rider Pro -- consolidated pullback-in-trend strategy.

    Replaces: VWAP Trend Rider, Momentum Continuation, Intraday Momentum,
    and Trend Pullback.  Trades pullbacks to the 5m EMA-20 within
    multi-timeframe trend alignment, scored via ConfluenceScorer.
    """

    name: str = "trend_rider_pro"
    data_requirements = DataRequirements(streams=["bars", "quotes"], on_scan=["prior_day"])

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.confluence_threshold = float(config.get("confluence_threshold", 65))
        self.time_window_start = time_utils.parse_time_string(str(config.get("time_window_start", "09:35")))
        self.time_window_end = time_utils.parse_time_string(str(config.get("time_window_end", "15:50")))
        self.min_bars = int(config.get("min_bars", 30))
        self.min_trend_alignment = float(config.get("min_trend_alignment", 0.5))

        # Tunable params (also settable via _optuna_overrides)
        overrides = config.get("_optuna_overrides", {})
        self.pullback_threshold = float(overrides.get("pullback_threshold", config.get("pullback_threshold", 0.003)))
        self.stop_atr_mult = float(overrides.get("stop_atr_mult", config.get("stop_atr_mult", 1.5)))
        self.target_atr_mult = float(overrides.get("target_atr_mult", config.get("target_atr_mult", 3.5)))
        self.trail_min_profit_r = float(overrides.get("trail_min_profit_r", config.get("trail_min_profit_r", 0.5)))
        self.max_hold_minutes = int(overrides.get("max_hold_minutes", config.get("max_hold_minutes", 120)))

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

        # --- direction detection ---
        mtf = MultiTimeframeAnalyzer(symbol_state)
        side = self._detect_side(bar, mtf, snapshot)
        if side is None:
            return None

        # --- confluence scoring ---
        scorer = self._score_confluence(bar, side, mtf, symbol_state, snapshot)
        if not scorer.passes_threshold():
            return None

        # --- stop / target ---
        atr_5m = mtf.get_atr("5m", 14)
        if atr_5m is None or atr_5m <= 0:
            return None

        stop_price = self._compute_stop(bar, side, mtf, atr_5m)
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
        meta["strategy_version"] = "trend_rider_pro_v1"
        meta["atr_5m"] = round(atr_5m, 6)
        meta["exit_config"] = {
            "trailing_enabled": True,
            "trail_timeframe": "5m",
            "trail_lookback": 5,
            "trail_min_profit_r": self.trail_min_profit_r,
            "partial_exits": [(2.0, 0.33), (4.0, 0.33)],
            "max_hold_minutes": self.max_hold_minutes,
            "vol_adaptive": True,
        }

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
        snapshot: Any,
    ) -> OrderSide | None:
        """Determine trade direction from trend alignment + pullback/VWAP."""
        ema20_5m = mtf.get_ema("5m", 20)
        if ema20_5m is None or ema20_5m <= 0:
            return None

        vwap_dist_1m = mtf.get_vwap_distance("1m")

        # --- BUY side ---
        buy_alignment = mtf.get_trend_alignment(OrderSide.BUY)
        if buy_alignment >= self.min_trend_alignment:
            pullback_pct = abs(bar.close - ema20_5m) / ema20_5m
            near_ema = pullback_pct <= self.pullback_threshold
            above_vwap = vwap_dist_1m is not None and vwap_dist_1m >= -0.002
            if near_ema and above_vwap:
                if self._regime_allows(OrderSide.BUY, snapshot, mtf):
                    return OrderSide.BUY

        # --- SELL side ---
        sell_alignment = mtf.get_trend_alignment(OrderSide.SELL)
        if sell_alignment >= self.min_trend_alignment:
            pullback_pct = abs(bar.close - ema20_5m) / ema20_5m
            near_ema = pullback_pct <= self.pullback_threshold
            below_vwap = vwap_dist_1m is not None and vwap_dist_1m <= 0.002
            if near_ema and below_vwap:
                if self._regime_allows(OrderSide.SELL, snapshot, mtf):
                    return OrderSide.SELL

        return None

    # ------------------------------------------------------------------
    # Regime gate
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_allows(
        side: OrderSide,
        snapshot: Any,
        mtf: MultiTimeframeAnalyzer,
    ) -> bool:
        """Check that the macro regime allows the proposed direction.

        Only allow entries when there is a confirmed trend in the matching
        direction.  Previously, FLAT trend with low ADX was accepted, but
        a trend-following strategy should not enter during flat/choppy
        conditions — this caused widespread false entries in H2 2024.
        """
        if snapshot is None:
            return True

        trend = getattr(snapshot, "trend", None)

        if side == OrderSide.BUY:
            return trend == TrendRegime.UP

        # SELL
        return trend == TrendRegime.DOWN

    # ------------------------------------------------------------------
    # Confluence scoring
    # ------------------------------------------------------------------

    def _score_confluence(
        self,
        bar: Bar,
        side: OrderSide,
        mtf: MultiTimeframeAnalyzer,
        symbol_state: SymbolState,
        snapshot: Any,
    ) -> ConfluenceScorer:
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)

        # 1. MTF trend alignment (weight 0.30)
        alignment = mtf.get_trend_alignment(side)
        alignment_score = alignment * 100.0
        scorer.add_factor(
            "mtf_trend_alignment",
            raw_value=alignment,
            score=alignment_score,
            weight=0.30,
            passed=alignment > 0.5,
        )

        # 2. Pullback quality (weight 0.20)
        ema20_5m = mtf.get_ema("5m", 20)
        pullback_pct = 0.0
        if ema20_5m is not None and ema20_5m > 0:
            pullback_pct = abs(bar.close - ema20_5m) / ema20_5m
        pb_score = score_deviation(pullback_pct, 0.001, 0.005)
        # Invert: closer to EMA = higher score
        pb_score = max(0.0, 100.0 - pb_score)
        scorer.add_factor(
            "pullback_quality",
            raw_value=pullback_pct,
            score=pb_score,
            weight=0.20,
            passed=0.001 <= pullback_pct <= 0.005,
        )

        # 3. ADX strength (weight 0.15)
        adx_5m = mtf.get_adx("5m")
        adx_val = adx_5m if adx_5m is not None else 0.0
        adx_score = score_threshold(adx_val, 20.0, 40.0)
        scorer.add_factor(
            "adx_strength",
            raw_value=adx_val,
            score=adx_score,
            weight=0.15,
            passed=adx_val >= 20.0,
        )

        # 4. Volume (weight 0.15)
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

        # 5. Momentum / RSI (weight 0.10)
        rsi_5m = mtf.get_rsi("5m", 14)
        rsi_val = rsi_5m if rsi_5m is not None else 50.0
        mom_score = _score_momentum(rsi_val, side)
        scorer.add_factor(
            "momentum_rsi",
            raw_value=rsi_val,
            score=mom_score,
            weight=0.10,
            passed=mom_score > 0,
        )

        # 6. Session quality (weight 0.10)
        session = getattr(snapshot, "session", None) if snapshot else None
        sess_score = _score_session(session)
        scorer.add_factor(
            "session_quality",
            raw_value=sess_score,
            score=sess_score,
            weight=0.10,
            passed=sess_score > 0,
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
    ) -> float:
        """Compute raw stop price before regime multiplier."""
        if side == OrderSide.BUY:
            swings = mtf.get_swing_lows("5m", lookback=5)
            if swings:
                return swings[0]  # most recent swing low
            return bar.low - self.stop_atr_mult * atr_5m
        # SELL
        swings = mtf.get_swing_highs("5m", lookback=5)
        if swings:
            return swings[0]  # most recent swing high
        return bar.high + self.stop_atr_mult * atr_5m

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

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
from src.quant.filters import KalmanMeanTracker
from src.quant.regime import MarkovRegimeSwitcher
from src.quant.statistics import HurstExponent
from src.quant.volatility import GARCHForecaster
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
        # Per-symbol quant state — lazy-initialized on first bar
        self._kalman: dict[str, KalmanMeanTracker] = {}
        self._hurst: dict[str, HurstExponent] = {}
        self._garch: dict[str, GARCHForecaster] = {}
        self._markov: dict[str, MarkovRegimeSwitcher] = {}

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

        # Hurst trending gate — reject when H <= threshold (want trending, H > 0.5)
        self.hurst_trending_threshold = float(
            overrides.get("hurst_trending_threshold", config.get("hurst_trending_threshold", 0.55))
        )

    # ------------------------------------------------------------------
    # Per-symbol quant component initialization
    # ------------------------------------------------------------------

    def _ensure_quant_state(self, symbol: str) -> None:
        """Lazily initialize quant components for a symbol on first encounter."""
        if symbol not in self._kalman:
            self._kalman[symbol] = KalmanMeanTracker(process_noise=0.01, measurement_noise=1.0)
        if symbol not in self._hurst:
            self._hurst[symbol] = HurstExponent(min_observations=100, lookback=500)
        if symbol not in self._garch:
            self._garch[symbol] = GARCHForecaster(min_observations=50, lookback=500, refit_interval=20)
        if symbol not in self._markov:
            self._markov[symbol] = MarkovRegimeSwitcher(n_regimes=2, min_observations=100, refit_interval=50)

    def _update_quant_state(self, symbol: str, bar: Bar) -> None:
        """Feed current bar data into all quant components for this symbol."""
        close = bar.close
        self._kalman[symbol].update(close)
        self._hurst[symbol].update(close)
        self._garch[symbol].update(close)
        # Markov expects returns — compute from last two prices
        garch_prices = self._garch[symbol]._prices
        if len(garch_prices) >= 2:
            ret = (garch_prices[-1] / garch_prices[-2]) - 1.0
            self._markov[symbol].update(ret)

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
        # --- lazy init + update quant components every bar ---
        self._ensure_quant_state(symbol)
        self._update_quant_state(symbol, bar)

        # --- timeframe gate: only evaluate on signal_timeframe closes ---
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

        # --- Hurst trending gate ---
        hurst_result = self._hurst[symbol].update(bar.close)
        if hurst_result is not None and hurst_result.H <= self.hurst_trending_threshold:
            self.logger.debug(
                "hurst_trending_gate_rejected",
                symbol=symbol,
                hurst=round(hurst_result.H, 4),
                strategy=self.name,
            )
            return None

        # --- direction detection (uses Kalman instead of EMA-20) ---
        mtf = MultiTimeframeAnalyzer(symbol_state)
        side = self._detect_side(bar, mtf, snapshot, symbol)
        if side is None:
            return None

        # --- 15m trend hard gate ---
        if not self._require_higher_tf_trend(mtf, side):
            return None

        # --- confluence scoring ---
        scorer = self._score_confluence(bar, side, mtf, symbol_state, snapshot, symbol)
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
        meta["strategy_version"] = "trend_rider_pro_v2"
        meta["atr_5m"] = round(atr_5m, 6)
        # Include quant state in meta for observability
        garch_result = self._garch[symbol]._last_result
        if garch_result is not None:
            meta["garch_conditional_vol"] = round(garch_result.conditional_vol, 6)
        if hurst_result is not None:
            meta["hurst_H"] = round(hurst_result.H, 4)
        markov_result = self._markov[symbol].last_result
        if markov_result is not None:
            meta["markov_trend_prob"] = round(markov_result.filtered_probability, 4)
        meta["exit_config"] = {
            "trailing_enabled": True,
            "trail_timeframe": "5m",
            "trail_lookback": 3,
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
        symbol: str = "",
    ) -> OrderSide | None:
        """Determine trade direction from trend alignment + pullback/VWAP.

        Uses Kalman mean estimate instead of EMA-20 for pullback detection
        when available, falling back to EMA-20 if Kalman not yet initialized.
        Uses Markov regime probability instead of hardcoded ADX threshold
        for trend confirmation, falling back to ADX gate if Markov hasn't converged.
        """
        # Kalman mean replaces EMA-20 for pullback anchor
        kalman_state = self._kalman[symbol].state if symbol in self._kalman else None
        ema20_5m = mtf.get_ema("5m", 20)
        pullback_anchor = kalman_state if kalman_state is not None and kalman_state > 0 else ema20_5m
        if pullback_anchor is None or pullback_anchor <= 0:
            return None

        vwap_dist_1m = mtf.get_vwap_distance("1m")

        # --- BUY side ---
        buy_alignment = mtf.get_trend_alignment(OrderSide.BUY)
        if buy_alignment >= self.min_trend_alignment:
            pullback_pct = abs(bar.close - pullback_anchor) / bar.close
            near_anchor = pullback_pct <= self.pullback_threshold
            vwap_ok = vwap_dist_1m is None or vwap_dist_1m >= -0.002
            if near_anchor and vwap_ok:
                if self._regime_allows_with_markov(OrderSide.BUY, snapshot, mtf, symbol):
                    return OrderSide.BUY

        # --- SELL side ---
        sell_alignment = mtf.get_trend_alignment(OrderSide.SELL)
        if sell_alignment >= self.min_trend_alignment:
            pullback_pct = abs(bar.close - pullback_anchor) / bar.close
            near_anchor = pullback_pct <= self.pullback_threshold
            vwap_ok = vwap_dist_1m is None or vwap_dist_1m <= 0.002
            if near_anchor and vwap_ok:
                if self._regime_allows_with_markov(OrderSide.SELL, snapshot, mtf, symbol):
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

        # Only allow entries when there is a confirmed trend in the matching
        # direction.  FLAT is rejected — trend-following should not enter
        # during choppy conditions.
        if side == OrderSide.BUY:
            return trend == TrendRegime.UP

        # SELL
        return trend == TrendRegime.DOWN

    def _regime_allows_with_markov(
        self,
        side: OrderSide,
        snapshot: Any,
        mtf: MultiTimeframeAnalyzer,
        symbol: str,
    ) -> bool:
        """Check regime using Markov probability when available, falling back to ADX gate.

        Replaces the hardcoded ADX threshold with Markov regime probability.
        Requires filtered_probability > 0.7 for the dominant regime to confirm trend.
        Falls back to the original _regime_allows if Markov hasn't converged.
        """
        markov_result = self._markov.get(symbol, None)
        if markov_result is not None:
            markov_result = markov_result.last_result
        if markov_result is not None and markov_result.filtered_probability > 0.7:
            # Markov has converged and is confident — use it as regime gate
            return self._regime_allows(side, snapshot, mtf)
        if markov_result is not None and markov_result.filtered_probability <= 0.7:
            # Markov has converged but regime is uncertain — reject
            return False
        # Markov not yet converged — fall back to original regime gate
        return self._regime_allows(side, snapshot, mtf)

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
        symbol: str = "",
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

        # 2. Pullback quality (weight 0.20) — uses Kalman mean instead of EMA-20
        kalman_state = self._kalman[symbol].state if symbol in self._kalman else None
        ema20_5m = mtf.get_ema("5m", 20)
        pullback_anchor = kalman_state if kalman_state is not None and kalman_state > 0 else ema20_5m
        pullback_pct = 0.0
        if pullback_anchor is not None and pullback_anchor > 0:
            pullback_pct = abs(bar.close - pullback_anchor) / bar.close
        pb_score = score_deviation(pullback_pct, 0.001, 0.008)
        # Invert: closer to anchor = higher score
        pb_score = max(0.0, 100.0 - pb_score)
        scorer.add_factor(
            "pullback_quality",
            raw_value=pullback_pct,
            score=pb_score,
            weight=0.20,
            passed=0.001 <= pullback_pct <= 0.008,
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
        symbol: str = "",
    ) -> float:
        """Compute raw stop price before regime multiplier.

        Uses GARCH conditional volatility for stop distance when available,
        falling back to ATR-based stops if GARCH hasn't fitted yet.
        """
        # Determine volatility-based stop distance
        garch_result = self._garch[symbol]._last_result if symbol in self._garch else None
        if garch_result is not None and garch_result.conditional_vol > 0:
            # GARCH conditional vol is in decimal return terms — convert to dollar distance
            vol_distance = self.stop_atr_mult * garch_result.conditional_vol * bar.close
        else:
            # Fallback to ATR
            vol_distance = self.stop_atr_mult * atr_5m

        if side == OrderSide.BUY:
            swings = mtf.get_swing_lows("5m", lookback=5)
            if swings:
                return swings[0]  # most recent swing low
            return bar.low - vol_distance
        # SELL
        swings = mtf.get_swing_highs("5m", lookback=5)
        if swings:
            return swings[0]  # most recent swing high
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

"""Flow Alpha Strategy — consolidated replacement for Flow Momentum,
Order Flow Imbalance, and Fusion V1.

Trades on options flow signals (Unusual Whales z-score, TFI, DOF, OFI)
confirmed by price action and multi-timeframe alignment.  Flow leads price.

Quant upgrades (v2):
- IC-weighted signal combination (replaces static weights)
- Granger causality validation (flow → returns)
- VPIN toxicity gate (reject informed-flow periods)
- GARCH-conditional z-score normalization
"""

from __future__ import annotations

from typing import Any

from src.analysis.vpin import VPINCalculator
from src.core import time_utils
from src.core.domain import (
    Bar,
    MarketState,
    OrderSide,
    RiskRegime,
    SessionRegime,
    Signal,
    SymbolState,
    VolRegime,
)
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.data.requirements import DataRequirements
from src.quant.statistics import GrangerCausalityTest
from src.quant.validation import InformationCoefficientTracker
from src.quant.volatility import GARCHForecaster
from src.strategies.base import BaseStrategy
from src.strategies.confluence import (
    ConfluenceScorer,
    score_candle_quality,
    score_deviation,
    score_volume,
)

# ---------------------------------------------------------------------------
# Session quality scores (higher = better for flow-based entries)
# ---------------------------------------------------------------------------
_SESSION_SCORES: dict[str, float] = {
    SessionRegime.OPENING.value: 90.0,
    SessionRegime.MIDDAY.value: 60.0,
    SessionRegime.POWER_HOUR.value: 80.0,
    SessionRegime.CLOSE.value: 30.0,
    SessionRegime.PREMARKET.value: 0.0,
}


class FlowAlphaStrategy(BaseStrategy):
    """Unified flow-driven strategy.

    Computes a composite *flow direction* from four independent flow
    signals, then scores entry quality via :class:`ConfluenceScorer`.
    Stops are ATR-based (5m timeframe) with regime-adaptive widening,
    and the exit configuration enables trailing + partial profit-taking.
    """

    name: str = "flow_alpha"
    data_requirements = DataRequirements(streams=["bars", "quotes"], on_scan=["flow", "gex"])

    # Names of the four flow signals used for IC tracking
    _FLOW_SIGNAL_NAMES = ("flow_zscore", "tfi", "dof_score", "ofi")
    _STATIC_WEIGHTS = (0.35, 0.25, 0.20, 0.20)

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

        # Per-symbol GARCH conditional volatility forecasters
        self._garch: dict[str, GARCHForecaster] = {}
        # Per-symbol VPIN calculators for toxicity filtering
        self._vpin: dict[str, VPINCalculator] = {}
        self._vpin_threshold: float = float(config.get("vpin_threshold", 0.7))
        # IC trackers — one per flow signal type
        self._ic_trackers: dict[str, InformationCoefficientTracker] = {
            name: InformationCoefficientTracker(window=50) for name in self._FLOW_SIGNAL_NAMES
        }
        # Granger causality state per symbol
        self._granger_valid: dict[str, bool] = {}
        self._granger_bar_count: dict[str, int] = {}
        self._granger_retest_interval: int = int(config.get("granger_retest_interval", 500))
        # History buffers for Granger testing
        self._return_history: dict[str, list[float]] = {}
        self._flow_history: dict[str, list[float]] = {}
        self._granger_min_obs: int = 100

    # ------------------------------------------------------------------
    # Parameter setup
    # ------------------------------------------------------------------

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.confluence_threshold = float(config.get("confluence_threshold", 65.0))
        self.time_window_start: str = str(config.get("time_window_start", "09:35"))
        self.time_window_end: str = str(config.get("time_window_end", "15:45"))
        self.min_bars: int = int(config.get("min_bars", 20))
        self.min_flow_direction: float = float(config.get("min_flow_direction", 0.15))

        # Tunable params (also settable via _optuna_overrides)
        overrides = config.get("_optuna_overrides", {})
        self.stop_atr_mult = float(overrides.get("stop_atr_mult", config.get("stop_atr_mult", 1.5)))
        self.target_r_mult = float(overrides.get("target_r_mult", config.get("target_r_mult", 3.0)))
        self.max_hold_minutes = int(overrides.get("max_hold_minutes", config.get("max_hold_minutes", 45)))

    # ------------------------------------------------------------------
    # Flow computation
    # ------------------------------------------------------------------

    def _compute_flow_direction(
        self,
        symbol_state: SymbolState,
        garch_cond_vol: float | None = None,
    ) -> float:
        """IC-weighted composite of four flow signals, mapped to [-1, 1].

        When IC trackers have accumulated enough data, weights are proportional
        to max(0, ic) for each signal.  Falls back to static weights otherwise.

        ``garch_cond_vol`` (if available) replaces the fixed ``/3.0``
        normalization on flow_zscore.
        """
        flow_zscore = float(symbol_state.meta.get("flow_zscore", 0.0) or 0.0)
        tfi = float(symbol_state.meta.get("tfi", 0.0) or 0.0)
        dof_score = float(symbol_state.meta.get("dof_score", 0.0) or 0.0)
        ofi = float(symbol_state.meta.get("ofi", 0.0) or 0.0)

        # GARCH-conditional normalization of flow_zscore
        # Divide by conditional vol (standardized residuals: signal / σ_t).
        # Scale factor 200 aligns with fallback /3.0 at typical vol (~0.015).
        if garch_cond_vol is not None and garch_cond_vol > 0:
            norm_flow = max(-1.0, min(1.0, flow_zscore / (garch_cond_vol * 200.0)))
        else:
            norm_flow = max(-1.0, min(1.0, flow_zscore / 3.0))

        signal_values = (
            norm_flow,
            tfi,
            dof_score * 2.0 - 1.0,
            max(-1.0, min(1.0, ofi)),
        )

        weights = self._get_ic_weights()

        flow_direction = sum(w * v for w, v in zip(weights, signal_values, strict=True))
        return flow_direction

    def _get_ic_weights(self) -> tuple[float, ...]:
        """Return IC-proportional weights or static fallback."""
        ics = []
        for name in self._FLOW_SIGNAL_NAMES:
            tracker = self._ic_trackers[name]
            ic = tracker.current_ic
            ics.append(ic)

        # Check if all IC trackers have data
        if any(ic is None for ic in ics):
            return self._STATIC_WEIGHTS

        positive_ics = [max(0.0, ic) for ic in ics]  # type: ignore[arg-type]
        total = sum(positive_ics)

        if total < 1e-9:
            return self._STATIC_WEIGHTS

        return tuple(ic / total for ic in positive_ics)

    def _update_ic_trackers(self, symbol_state: SymbolState, forward_return: float) -> None:
        """Update IC trackers with current signal values and realized return."""
        flow_zscore = float(symbol_state.meta.get("flow_zscore", 0.0) or 0.0)
        tfi = float(symbol_state.meta.get("tfi", 0.0) or 0.0)
        dof_score = float(symbol_state.meta.get("dof_score", 0.0) or 0.0)
        ofi = float(symbol_state.meta.get("ofi", 0.0) or 0.0)

        predictions = {
            "flow_zscore": flow_zscore,
            "tfi": tfi,
            "dof_score": dof_score,
            "ofi": ofi,
        }

        for name, pred in predictions.items():
            self._ic_trackers[name].update(pred, forward_return)

    # ------------------------------------------------------------------
    # Average volume helper
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_volume(symbol_state: SymbolState, lookback: int = 20) -> float:
        """20-bar average volume from indicator cache or raw bars."""
        cached = symbol_state.indicators.get("sma_vol:20")
        if cached is not None:
            try:
                return float(cached)
            except (TypeError, ValueError):
                pass
        bars = list(symbol_state.bars)
        vols = [float(b.volume) for b in bars[-lookback:]]
        if not vols:
            return 0.0
        return sum(vols) / len(vols)

    # ------------------------------------------------------------------
    # Regime gating
    # ------------------------------------------------------------------

    @staticmethod
    def _passes_regime_gate(market_state: MarketState) -> bool:
        """Return False if the regime is hostile to flow-based entries."""
        snap = market_state.regime_snapshot
        if snap is None:
            return True

        # Skip if volatility is in SHOCK
        if snap.vol == VolRegime.SHOCK:
            return False

        # Skip PREMARKET
        if snap.session == SessionRegime.PREMARKET:
            return False

        # Prefer RISK_ON, allow NEUTRAL, skip RISK_OFF
        return snap.risk != RiskRegime.RISK_OFF

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Granger causality (periodic)
    # ------------------------------------------------------------------

    def _maybe_run_granger(self, symbol: str) -> None:
        """Run Granger causality test every ``_granger_retest_interval`` bars."""
        count = self._granger_bar_count.get(symbol, 0)
        count += 1
        self._granger_bar_count[symbol] = count

        if count < self._granger_retest_interval:
            return

        returns = self._return_history.get(symbol, [])
        flows = self._flow_history.get(symbol, [])
        if len(returns) < self._granger_min_obs or len(flows) < self._granger_min_obs:
            return

        # Use last granger_min_obs observations
        n = self._granger_min_obs
        try:
            result = GrangerCausalityTest.test(flows[-n:], returns[-n:], max_lag=5)
            self._granger_valid[symbol] = result.is_causal
            self._granger_bar_count[symbol] = 0
            self.logger.info(
                "granger_test_result",
                symbol=symbol,
                is_causal=result.is_causal,
                p_value=round(result.p_value, 4),
                best_lag=result.best_lag,
            )
        except Exception:
            # Granger may fail on degenerate data; keep previous result
            self.logger.debug("granger_test_failed", symbol=symbol, exc_info=True)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        if not self._is_evaluation_bar(bar):
            return None

        # --- Pre-flight checks ---
        if not self._check_cooldown(symbol, bar.time):
            return None

        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        if not time_utils.in_time_window_str(
            bar.time,
            self.time_window_start,
            self.time_window_end,
        ):
            return None

        if not self._passes_regime_gate(market_state):
            return None

        # --- VPIN toxicity gate ---
        if symbol not in self._vpin:
            self._vpin[symbol] = VPINCalculator(logger=self.logger)
        vpin_result = self._vpin[symbol].update(bar)
        if vpin_result is not None and vpin_result.is_toxic:
            self.logger.debug(
                "flow_alpha: vpin_toxic_rejected",
                symbol=symbol,
                vpin=round(vpin_result.vpin, 4),
            )
            return None

        # --- Feed GARCH (lazy init per symbol) ---
        if symbol not in self._garch:
            self._garch[symbol] = GARCHForecaster()
        garch_result = self._garch[symbol].update(bar.close)
        garch_cond_vol = garch_result.conditional_vol if garch_result is not None else None

        # --- IC tracker update: use 1-bar forward return proxy ---
        bars_list = list(symbol_state.bars)
        if len(bars_list) >= 2:
            prev_close = bars_list[-2].close
            if prev_close > 0:
                fwd_return = (bar.close - prev_close) / prev_close
                self._update_ic_trackers(symbol_state, fwd_return)

        # --- Accumulate history for Granger testing ---
        flow_zscore_raw = float(symbol_state.meta.get("flow_zscore", 0.0) or 0.0)
        if symbol not in self._return_history:
            self._return_history[symbol] = []
            self._flow_history[symbol] = []

        if len(bars_list) >= 2:
            prev_close = bars_list[-2].close
            if prev_close > 0:
                ret = (bar.close - prev_close) / prev_close
                self._return_history[symbol].append(ret)
                self._flow_history[symbol].append(flow_zscore_raw)
                # Keep bounded
                if len(self._return_history[symbol]) > 600:
                    self._return_history[symbol] = self._return_history[symbol][-500:]
                    self._flow_history[symbol] = self._flow_history[symbol][-500:]

        # --- Periodic Granger test ---
        self._maybe_run_granger(symbol)

        # --- Flow direction (IC-weighted + GARCH-conditional) ---
        flow_direction = self._compute_flow_direction(symbol_state, garch_cond_vol)

        if abs(flow_direction) <= self.min_flow_direction:
            return None  # flow too weak

        side = OrderSide.BUY if flow_direction > 0 else OrderSide.SELL
        side_str = "buy" if side == OrderSide.BUY else "sell"

        # --- Multi-timeframe analysis ---
        mtf = MultiTimeframeAnalyzer(symbol_state)
        trend_alignment = mtf.get_trend_alignment(side)  # 0.0 - 1.0

        # --- ATR for stop placement ---
        atr_5m = mtf.get_atr("5m", 14)
        if atr_5m is None or atr_5m <= 0:
            # Fallback: 1% of price
            atr_5m = bar.close * 0.01

        # --- Average volume ---
        avg_vol = self._avg_volume(symbol_state)

        # --- Session quality ---
        snap = market_state.regime_snapshot
        session_value = snap.session.value if snap and snap.session else "midday"
        session_score = _SESSION_SCORES.get(session_value, 50.0)

        # --- Confluence scoring ---
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)

        # 1. Flow strength (weight 0.30)
        flow_strength_score = score_deviation(abs(flow_direction), 0.15, 0.8)
        scorer.add_factor(
            "flow_strength",
            raw_value=abs(flow_direction),
            score=flow_strength_score,
            weight=0.30,
            passed=abs(flow_direction) > self.min_flow_direction,
        )

        # 2. Price confirmation (weight 0.20)
        if side == OrderSide.BUY:
            price_confirmed = bar.close > bar.open
        else:
            price_confirmed = bar.close < bar.open
        price_score = 80.0 if price_confirmed else 30.0
        scorer.add_factor(
            "price_confirmation",
            raw_value=bar.close - bar.open,
            score=price_score,
            weight=0.20,
            passed=price_confirmed,
        )

        # 3. Volume (weight 0.15)
        vol_score = score_volume(float(bar.volume), avg_vol, 1.0, 2.5)
        scorer.add_factor(
            "volume",
            raw_value=float(bar.volume) / avg_vol if avg_vol > 0 else 0.0,
            score=vol_score,
            weight=0.15,
            passed=vol_score > 0,
        )

        # 4. MTF alignment (weight 0.15)
        mtf_score = trend_alignment * 100.0
        scorer.add_factor(
            "mtf_alignment",
            raw_value=trend_alignment,
            score=mtf_score,
            weight=0.15,
            passed=trend_alignment >= 0.4,
        )

        # 5. Candle quality (weight 0.10)
        candle_score = score_candle_quality(
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            side_str,
        )
        scorer.add_factor(
            "candle_quality",
            raw_value=candle_score,
            score=candle_score,
            weight=0.10,
        )

        # 6. Session quality (weight 0.10)
        scorer.add_factor(
            "session_quality",
            raw_value=session_score,
            score=session_score,
            weight=0.10,
            passed=session_score >= 50.0,
        )

        if not scorer.passes_threshold():
            self.logger.debug(
                "flow_alpha: confluence below threshold",
                symbol=symbol,
                score=round(scorer.score(), 2),
                threshold=self.confluence_threshold,
                flow_dir=round(flow_direction, 4),
            )
            return None

        # --- Granger causality confidence adjustment ---
        granger_causal = self._granger_valid.get(symbol)  # None = not tested yet

        # --- Stop / Target ---
        raw_risk = self.stop_atr_mult * atr_5m
        stop_distance = self._apply_regime_volatility_multiplier(
            raw_risk,
            market_state,
        )
        target_distance = self.target_r_mult * raw_risk

        if side == OrderSide.BUY:
            stop_price = bar.close - stop_distance
            target_price = bar.close + target_distance
        else:
            stop_price = bar.close + stop_distance
            target_price = bar.close - target_distance

        # --- Conviction: reduce if Granger says not causal ---
        conviction = scorer.conviction_multiplier()
        if granger_causal is False:
            conviction *= 0.7
            self.logger.debug(
                "flow_alpha: granger_not_causal_penalty",
                symbol=symbol,
                reduced_conviction=round(conviction, 4),
            )

        # --- Build meta ---
        meta = scorer.to_meta()
        meta.update(
            {
                "flow_direction": round(flow_direction, 4),
                "flow_zscore": float(symbol_state.meta.get("flow_zscore", 0.0) or 0.0),
                "tfi": float(symbol_state.meta.get("tfi", 0.0) or 0.0),
                "dof_score": float(symbol_state.meta.get("dof_score", 0.0) or 0.0),
                "ofi": float(symbol_state.meta.get("ofi", 0.0) or 0.0),
                "atr_5m": round(atr_5m, 4),
                "stop_distance": round(stop_distance, 4),
                "session": session_value,
                "granger_causal": granger_causal,
                "garch_cond_vol": round(garch_cond_vol, 6) if garch_cond_vol is not None else None,
                "vpin": round(vpin_result.vpin, 4) if vpin_result is not None else None,
                "ic_weights": [round(w, 4) for w in self._get_ic_weights()],
                "exit_config": {
                    "trailing_enabled": True,
                    "trail_timeframe": "5m",
                    "trail_lookback": 3,
                    "trail_min_profit_r": 0.75,
                    "partial_exits": [(2.0, 0.33)],
                    "max_hold_minutes": self.max_hold_minutes,
                    "vol_adaptive": True,
                },
            },
        )

        self.logger.info(
            "flow_alpha: signal generated",
            symbol=symbol,
            side=side_str,
            flow_dir=round(flow_direction, 4),
            confluence=round(scorer.score(), 2),
            conviction=round(conviction, 4),
        )

        self.last_signal_time[symbol] = bar.time

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            size_hint=conviction,
            meta=meta,
        )

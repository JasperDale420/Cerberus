"""Flow Alpha Strategy — consolidated replacement for Flow Momentum,
Order Flow Imbalance, and Fusion V1.

Trades on options flow signals (Unusual Whales z-score, TFI, DOF, OFI)
confirmed by price action and multi-timeframe alignment.  Flow leads price.
"""

from __future__ import annotations

from typing import Any

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

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

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

    @staticmethod
    def _compute_flow_direction(symbol_state: SymbolState) -> float:
        """Weighted composite of four flow signals, mapped to [-1, 1]."""
        flow_zscore = float(symbol_state.meta.get("flow_zscore", 0.0) or 0.0)
        tfi = float(symbol_state.meta.get("tfi", 0.0) or 0.0)
        dof_score = float(symbol_state.meta.get("dof_score", 0.0) or 0.0)
        ofi = float(symbol_state.meta.get("ofi", 0.0) or 0.0)

        flow_direction = (
            0.35 * max(-1.0, min(1.0, flow_zscore / 3.0))
            + 0.25 * tfi
            + 0.20 * (dof_score * 2.0 - 1.0)
            + 0.20 * max(-1.0, min(1.0, ofi))
        )
        return flow_direction

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

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
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

        # --- Flow direction ---
        flow_direction = self._compute_flow_direction(symbol_state)

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
            conviction=round(scorer.conviction_multiplier(), 4),
        )

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

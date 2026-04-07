"""Parameter space definitions for Optuna-based strategy optimization.

Each strategy defines 4-6 tunable parameters — the most impactful knobs
identified through parameter importance analysis.  Everything else stays
at sensible defaults.

Parameters with strong WFO convergence (CV < 0.20) are locked at their
converged values via LOCKED_PARAMS and injected automatically by
suggest_params().

Usage::

    from src.analytics.param_spaces import PARAM_SPACES, suggest_params

    params = suggest_params(trial, "trend_rider_pro")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import optuna

# ---------------------------------------------------------------------------
# Parameter definition
# ---------------------------------------------------------------------------


@dataclass
class ParamDef:
    """Single parameter definition for optimization."""

    name: str
    param_type: str  # "float", "int", "categorical"
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: list[Any] | None = field(default_factory=list)
    log_scale: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Locked parameters — converged across WFO windows, removed from search
# ---------------------------------------------------------------------------

LOCKED_PARAMS: dict[str, dict[str, Any]] = {
    "orb_v2": {
        # Converged to 3.5 in 5/7 WFO windows (CV 0.18)
        "target_range_mult": 3.5,
    },
    "trend_rider_pro": {},
}


# ---------------------------------------------------------------------------
# Per-strategy parameter spaces
# ---------------------------------------------------------------------------

PARAM_SPACES: dict[str, list[ParamDef]] = {
    # ------------------------------------------------------------------
    # Trend Rider Pro v4 — 7 params, ranges informed by 72-iter autoresearch
    #   Autoresearch best: confluence 65, stop 2.5, target 4.0, trail 0.75,
    #   hold 120, pullback 0.008, alignment 0.30
    # ------------------------------------------------------------------
    "trend_rider_pro": [
        ParamDef(
            "confluence_threshold",
            "float",
            low=55.0,
            high=75.0,
            step=5.0,
            description="Minimum confluence score (autoresearch: 65 optimal)",
        ),
        ParamDef(
            "min_trend_alignment",
            "float",
            low=0.15,
            high=0.55,
            step=0.05,
            description="MTF trend alignment gate",
        ),
        ParamDef(
            "pullback_threshold",
            "float",
            low=0.004,
            high=0.012,
            step=0.001,
            description="Max distance from Kalman/EMA anchor for pullback detection",
        ),
        ParamDef(
            "stop_atr_mult",
            "float",
            low=1.75,
            high=3.25,
            step=0.25,
            description="ATR multiplier for stop (autoresearch: 2.5 optimal)",
        ),
        ParamDef(
            "target_atr_mult",
            "float",
            low=3.0,
            high=5.0,
            step=0.5,
            description="ATR multiplier for target (autoresearch: 4.0 optimal)",
        ),
        ParamDef(
            "trail_min_profit_r",
            "float",
            low=0.5,
            high=1.0,
            step=0.05,
            description="R-multiple before trailing activates (autoresearch: 0.75)",
        ),
        ParamDef(
            "max_hold_minutes",
            "int",
            low=90,
            high=180,
            step=15,
            description="Maximum hold time in minutes (autoresearch: 120)",
        ),
    ],
    # ------------------------------------------------------------------
    # Mean Reversion Pro — 5 params
    #   Focus: entry thresholds, hold time, exit geometry
    #   Changes: raised max_hold_minutes floor to 20 (was 10) — sub-10m
    #   holds are scalping territory and unrealistic without DMA.
    #   Added target_mult param so optimizer can widen the target, since
    #   WFO showed avg_win < avg_loss in most windows.
    # ------------------------------------------------------------------
    "mean_reversion_pro": [
        ParamDef(
            "confluence_threshold",
            "float",
            low=40.0,
            high=75.0,
            step=5.0,
            description="Minimum confluence score to enter (lowered floor to 40)",
        ),
        ParamDef(
            "vwap_dist_threshold",
            "float",
            low=0.002,
            high=0.008,
            step=0.001,
            description="VWAP distance threshold (floor raised from 0.001)",
        ),
        ParamDef(
            "bb_pos_threshold",
            "float",
            low=0.15,
            high=0.6,
            step=0.05,
            description="Bollinger Band position threshold for direction",
        ),
        ParamDef(
            "max_hold_minutes",
            "int",
            low=20,
            high=90,
            step=5,
            description="Maximum hold time (widened ceiling to 90 for mean-reversion)",
        ),
        ParamDef(
            "stop_atr_mult",
            "float",
            low=1.0,
            high=2.5,
            step=0.25,
            description="ATR multiplier for fallback stop",
        ),
        # Quant gate thresholds
        ParamDef(
            "hurst_gate_threshold",
            "float",
            low=0.0,
            high=0.55,
            step=0.05,
            description="Hurst exponent gate — reject when H >= threshold. 0.0 disables gate.",
        ),
        ParamDef(
            "hmm_min_confidence",
            "float",
            low=0.0,
            high=0.8,
            step=0.1,
            description="HMM regime gate min confidence. 0.0 disables gate.",
        ),
    ],
    # ------------------------------------------------------------------
    # ORB V2 — 4 params (was 5; locked target_range_mult at 3.5)
    #   Focus: breakout confirmation (volume, confluence), exit timing
    # ------------------------------------------------------------------
    "orb_v2": [
        ParamDef(
            "confluence_threshold",
            "float",
            low=40.0,
            high=75.0,
            step=5.0,
            description="Minimum confluence score for breakout entry (lowered floor to 40)",
        ),
        ParamDef(
            "vol_gate_mult",
            "float",
            low=1.1,
            high=1.7,
            step=0.1,
            description="Volume gate (tightened from [1.0, 2.0] per WFO convergence)",
        ),
        # target_range_mult LOCKED at 3.5 (converged in 5/7 windows, CV 0.18)
        ParamDef(
            "trail_min_profit_r",
            "float",
            low=0.5,
            high=1.25,
            step=0.25,
            description="R-multiple before trailing stop activates",
        ),
        ParamDef(
            "max_hold_minutes",
            "int",
            low=60,
            high=120,
            step=15,
            description="Maximum hold time (floor raised to 60 from 30)",
        ),
        # Quant gate thresholds
        ParamDef(
            "hmm_min_confidence",
            "float",
            low=0.0,
            high=0.8,
            step=0.1,
            description="HMM regime gate min confidence. 0.0 disables gate.",
        ),
    ],
    # ------------------------------------------------------------------
    # Flow Alpha — DISABLED for bar-only backtests
    #   Depends on real-time options flow data (Unusual Whales pipeline).
    #   flow_zscore, tfi, dof_score, ofi in symbol_state.meta are all 0.0
    #   in bar-only backtests → only 2 trades in all of 2024.
    #   Cannot meaningfully optimize without live flow data.
    # ------------------------------------------------------------------
    # "flow_alpha": [ ... ],
    #
    # ------------------------------------------------------------------
    # Pair Trading V2 — 6 params
    #   Focus: z-score entry/stop thresholds, spread normalization,
    #   correlation gating, hold time. Mean-reversion on cointegrated
    #   pairs with Kalman-filtered hedge ratios.
    # ------------------------------------------------------------------
    "pair_trading_v2": [
        ParamDef(
            "entry_z_threshold",
            "float",
            low=1.0,
            high=2.5,
            step=0.25,
            description="Z-score deviation to trigger entry signal",
        ),
        ParamDef(
            "stop_z_threshold",
            "float",
            low=3.0,
            high=5.0,
            step=0.25,
            description="Z-score deviation for stop loss",
        ),
        ParamDef(
            "confluence_threshold",
            "float",
            low=30.0,
            high=65.0,
            step=5.0,
            description="Minimum confluence score to enter",
        ),
        ParamDef(
            "spread_lookback",
            "int",
            low=40,
            high=120,
            step=10,
            description="Lookback period in trading days for spread normalization",
        ),
        ParamDef(
            "min_correlation",
            "float",
            low=0.3,
            high=0.65,
            step=0.05,
            description="Minimum rolling correlation gate",
        ),
        ParamDef(
            "max_hold_days",
            "int",
            low=5,
            high=30,
            step=5,
            description="Maximum hold time in trading days",
        ),
    ],
    # ------------------------------------------------------------------
    # RSI Bounce — 5 params
    #   Focus: RSI thresholds, band tolerance, stop/target, hold time
    #   Mean-reversion off extreme RSI + Bollinger Band proximity.
    # ------------------------------------------------------------------
    "rsi_bounce": [
        ParamDef(
            "confluence_threshold",
            "float",
            low=30.0,
            high=70.0,
            step=5.0,
            description="Minimum confluence score to enter (wider range for 10-factor model)",
        ),
        ParamDef(
            "rsi_oversold",
            "float",
            low=20.0,
            high=40.0,
            step=5.0,
            description="RSI oversold threshold for BUY signals (widened to allow more entries)",
        ),
        ParamDef(
            "rsi_overbought",
            "float",
            low=60.0,
            high=80.0,
            step=5.0,
            description="RSI overbought threshold for SELL signals (widened to allow more entries)",
        ),
        ParamDef(
            "band_tolerance",
            "float",
            low=2.0,
            high=15.0,
            step=0.5,
            description="Proximity tolerance to BB as % of band width (was 0.5-2.5 = too tight)",
        ),
        ParamDef(
            "stop_atr_mult",
            "float",
            low=1.0,
            high=3.0,
            step=0.25,
            description="ATR multiplier for stop loss",
        ),
        ParamDef(
            "target_atr_mult",
            "float",
            low=2.0,
            high=5.0,
            step=0.5,
            description="ATR multiplier for take profit target",
        ),
        ParamDef(
            "max_hold_minutes",
            "int",
            low=30,
            high=120,
            step=10,
            description="Maximum hold time in minutes (widened ceiling)",
        ),
    ],
    # ------------------------------------------------------------------
    # Regime Trend Up — UP+NORMAL specialist
    #   BUY-only EMA pullback in uptrend. Key knobs: RSI zone, stop/target, hold.
    # ------------------------------------------------------------------
    "regime_trend_up": [
        ParamDef(
            "rsi_min",
            "float",
            low=25.0,
            high=45.0,
            step=5.0,
            description="RSI lower bound for pullback entry zone",
        ),
        ParamDef(
            "rsi_max",
            "float",
            low=55.0,
            high=75.0,
            step=5.0,
            description="RSI upper bound for pullback entry zone",
        ),
        ParamDef(
            "pullback_pct",
            "float",
            low=0.005,
            high=0.03,
            step=0.005,
            description="Max distance from EMA20 to qualify as a pullback",
        ),
        ParamDef(
            "stop_atr_mult",
            "float",
            low=1.0,
            high=2.5,
            step=0.25,
            description="ATR multiplier for stop loss",
        ),
        ParamDef(
            "target_rr",
            "float",
            low=1.5,
            high=4.0,
            step=0.5,
            description="Risk:reward ratio for take profit target",
        ),
        ParamDef(
            "cooldown_bars",
            "int",
            low=3,
            high=15,
            step=3,
            description="Minimum 1m bars between signals",
        ),
    ],
    # ------------------------------------------------------------------
    # Regime Bear — 5 params
    #   Bear/high-vol specialist: shorts RSI bounces in downtrends.
    #   Key knobs: RSI entry thresholds, stop/target geometry.
    # ------------------------------------------------------------------
    "regime_bear": [
        ParamDef(
            "rsi_short_entry",
            "float",
            low=45.0,
            high=65.0,
            step=5.0,
            description="RSI level to short into (DOWN trend bounce fade)",
        ),
        ParamDef(
            "rsi_long_entry",
            "float",
            low=35.0,
            high=55.0,
            step=5.0,
            description="RSI level to buy into (UP trend dip)",
        ),
        ParamDef(
            "stop_atr_mult",
            "float",
            low=1.0,
            high=2.5,
            step=0.25,
            description="ATR multiplier for stop loss",
        ),
        ParamDef(
            "target_atr_mult",
            "float",
            low=2.0,
            high=5.0,
            step=0.5,
            description="ATR multiplier for take profit",
        ),
        ParamDef(
            "high_vol_target_mult",
            "float",
            low=1.0,
            high=2.5,
            step=0.25,
            description="Target multiplier in HIGH/SHOCK vol (DOWN+HIGH edge)",
        ),
    ],
    # ------------------------------------------------------------------
    # Regime Adaptive — 5 params
    #   Cross-regime generalist: switches behavior by trend regime.
    #   UP→dip-buy, DOWN→fade-bounce, FLAT→RSI extremes.
    # ------------------------------------------------------------------
    "regime_adaptive": [
        ParamDef(
            "rsi_long_entry",
            "float",
            low=35.0,
            high=50.0,
            step=5.0,
            description="RSI threshold for BUY in UP regime (buy the dip below this)",
        ),
        ParamDef(
            "rsi_short_entry",
            "float",
            low=55.0,
            high=70.0,
            step=5.0,
            description="RSI threshold for SELL in DOWN regime (short the bounce above this)",
        ),
        ParamDef(
            "stop_atr_mult",
            "float",
            low=1.0,
            high=2.5,
            step=0.25,
            description="ATR multiplier for stop loss distance",
        ),
        ParamDef(
            "target_atr_mult",
            "float",
            low=2.0,
            high=5.0,
            step=0.5,
            description="ATR multiplier for take profit distance",
        ),
        ParamDef(
            "cooldown_bars",
            "int",
            low=3,
            high=15,
            step=3,
            description="Minimum 1m bars between signals per symbol",
        ),
    ],
    # ------------------------------------------------------------------
    # Momentum Fade — 5 params
    #   Focus: VWAP distance, volume surge, confluence, stop/target, hold
    #   Fades overextended moves away from VWAP on volume spikes.
    # ------------------------------------------------------------------
    "momentum_fade": [
        ParamDef(
            "vwap_threshold",
            "float",
            low=0.004,
            high=0.015,
            step=0.001,
            description="VWAP distance threshold (e.g. 0.008 = 0.8%)",
        ),
        ParamDef(
            "volume_surge_mult",
            "float",
            low=1.5,
            high=3.5,
            step=0.25,
            description="Volume surge multiplier vs recent average",
        ),
        ParamDef(
            "confluence_threshold",
            "float",
            low=40.0,
            high=75.0,
            step=5.0,
            description="Minimum confluence score to enter (lowered floor to 40)",
        ),
        ParamDef(
            "stop_atr_mult",
            "float",
            low=1.0,
            high=2.5,
            step=0.25,
            description="ATR multiplier for stop loss",
        ),
        ParamDef(
            "target_atr_mult",
            "float",
            low=2.0,
            high=5.0,
            step=0.5,
            description="ATR multiplier for take profit target",
        ),
        ParamDef(
            "max_hold_minutes",
            "int",
            low=30,
            high=120,
            step=15,
            description="Maximum hold time in minutes",
        ),
        # Quant gate thresholds
        ParamDef(
            "hurst_gate_threshold",
            "float",
            low=0.0,
            high=0.6,
            step=0.05,
            description="Hurst gate — reject when H >= threshold. 0.0 disables gate.",
        ),
        ParamDef(
            "entropy_threshold",
            "float",
            low=0.6,
            high=1.0,
            step=0.05,
            description="Entropy filter — reject when entropy > threshold. 1.0 disables gate.",
        ),
        ParamDef(
            "hmm_min_confidence",
            "float",
            low=0.0,
            high=0.8,
            step=0.1,
            description="HMM regime gate min confidence. 0.0 disables gate.",
        ),
    ],
    # ------------------------------------------------------------------
    # Autoresearch Strategy — 4 params
    #   Simple EMA trend + RSI pullback. Keeps param space small for fast WFO.
    # ------------------------------------------------------------------
    "autoresearch_strategy": [
        ParamDef(
            "stop_atr_mult",
            "float",
            low=1.0,
            high=3.0,
            step=0.25,
            description="ATR multiplier for stop loss",
        ),
        ParamDef(
            "target_atr_mult",
            "float",
            low=2.0,
            high=5.0,
            step=0.5,
            description="ATR multiplier for take profit target",
        ),
        ParamDef(
            "cooldown_bars",
            "int",
            low=3,
            high=15,
            step=3,
            description="Minimum 1m bars between signals per symbol",
        ),
    ],
    "daily_research_strategy": [
        ParamDef("stop_atr_mult", "float", low=1.0, high=3.0, step=0.25, description="ATR multiplier for stop loss"),
        ParamDef(
            "target_atr_mult",
            "float",
            low=1.5,
            high=4.0,
            step=0.5,
            description="ATR multiplier for take profit (wide)",
        ),
        ParamDef("rsi_threshold", "float", low=25.0, high=45.0, step=5.0, description="RSI(14) pullback threshold"),
        ParamDef("breakout_period", "int", low=10, high=25, step=5, description="Lookback for breakout high"),
    ],
}


# ---------------------------------------------------------------------------
# Optuna suggestion helper
# ---------------------------------------------------------------------------


def suggest_params(
    trial: optuna.Trial,
    strategy_name: str,
) -> dict[str, Any]:
    """Use an Optuna trial to suggest parameter values from the space.

    Locked parameters (from LOCKED_PARAMS) are injected automatically
    without consuming Optuna trial dimensions.
    """
    space = PARAM_SPACES.get(strategy_name)
    if space is None:
        raise ValueError(f"No parameter space defined for '{strategy_name}'")

    params: dict[str, Any] = {}

    # Inject locked params first
    locked = LOCKED_PARAMS.get(strategy_name, {})
    params.update(locked)

    for p in space:
        if p.param_type == "float":
            if p.step:
                params[p.name] = trial.suggest_float(p.name, p.low, p.high, step=p.step)
            elif p.log_scale:
                params[p.name] = trial.suggest_float(p.name, p.low, p.high, log=True)
            else:
                params[p.name] = trial.suggest_float(p.name, p.low, p.high)
        elif p.param_type == "int":
            params[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high), step=int(p.step or 1))
        elif p.param_type == "categorical":
            params[p.name] = trial.suggest_categorical(p.name, p.choices)
    return params

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
    "trend_rider_pro": {
        # CV 0.53 — too unstable to optimize; lock at WFO mean
        "pullback_threshold": 0.004,
    },
}


# ---------------------------------------------------------------------------
# Per-strategy parameter spaces
# ---------------------------------------------------------------------------

PARAM_SPACES: dict[str, list[ParamDef]] = {
    # ------------------------------------------------------------------
    # Trend Rider Pro — 5 params (was 7; locked pullback_threshold)
    #   Focus: entry threshold, trend gate, stop/target geometry, exit
    # ------------------------------------------------------------------
    "trend_rider_pro": [
        ParamDef(
            "confluence_threshold",
            "float",
            low=50.0,
            high=80.0,
            step=5.0,
            description="Minimum confluence score to enter (higher = pickier)",
        ),
        ParamDef(
            "min_trend_alignment",
            "float",
            low=0.3,
            high=0.8,
            step=0.05,
            description="MTF trend alignment gate (0-1)",
        ),
        # pullback_threshold LOCKED at 0.004 (CV 0.53 — unstable)
        ParamDef(
            "stop_atr_mult",
            "float",
            low=1.5,
            high=3.0,
            step=0.25,
            description="ATR multiplier for stop",
        ),
        ParamDef(
            "target_atr_mult",
            "float",
            low=2.5,
            high=5.0,
            step=0.5,
            description="ATR multiplier for target (floor raised to 2.5 from WFO)",
        ),
        ParamDef(
            "trail_min_profit_r",
            "float",
            low=0.4,
            high=0.8,
            step=0.1,
            description="R-multiple before trailing activates (tightened from WFO profitable windows)",
        ),
        ParamDef(
            "max_hold_minutes",
            "int",
            low=60,
            high=210,
            step=15,
            description="Maximum hold time in minutes (floor raised from 30)",
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
            low=50.0,
            high=75.0,
            step=5.0,
            description="Minimum confluence score to enter",
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
            high=60,
            step=5,
            description="Maximum hold time (floor raised to 20 — no sub-10m scalping)",
        ),
        ParamDef(
            "stop_atr_mult",
            "float",
            low=1.0,
            high=2.5,
            step=0.25,
            description="ATR multiplier for fallback stop",
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
            low=50.0,
            high=75.0,
            step=5.0,
            description="Minimum confluence score for breakout entry",
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
    # Pair Trading V2 — NOT OPTIMIZABLE with standard pipeline
    #   Requires pair-specific config with cointegrated pairs and
    #   synchronized multi-leg data. Not a single-instrument strategy.
    #   Already disabled in config/backtest_v2.yaml.
    # ------------------------------------------------------------------
    # "pair_trading_v2": [ ... ],
    # ------------------------------------------------------------------
    # RSI Bounce — 5 params
    #   Focus: RSI thresholds, band tolerance, stop/target, hold time
    #   Mean-reversion off extreme RSI + Bollinger Band proximity.
    # ------------------------------------------------------------------
    "rsi_bounce": [
        ParamDef(
            "confluence_threshold",
            "float",
            low=50.0,
            high=75.0,
            step=5.0,
            description="Minimum confluence score to enter",
        ),
        ParamDef(
            "band_tolerance",
            "float",
            low=0.2,
            high=1.5,
            step=0.1,
            description="Proximity tolerance to Bollinger Band as % of band width",
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
            low=20,
            high=90,
            step=10,
            description="Maximum hold time in minutes",
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
            low=50.0,
            high=75.0,
            step=5.0,
            description="Minimum confluence score to enter",
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

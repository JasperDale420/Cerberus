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


@dataclass
class ParamDef:
    """Single parameter definition for optimization."""

    name: str
    param_type: str
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: list[Any] | None = field(default_factory=list)
    log_scale: bool = False
    description: str = ""


LOCKED_PARAMS: dict[str, dict[str, Any]] = {
    "orb_v2": {"target_range_mult": 3.5},
    "trend_rider_pro": {},
    "daily_research_strategy": {
        "stop_atr_mult": 1.5,
        "target_atr_mult": 6.0,
        "rsi2_threshold": 25.0,
        "pullback_rsi_lo": 25.0,
    },
    "daily_research_v6a": {
        "stop_atr_mult": 1.5,
        "rsi2_threshold": 15,
        "max_hold_days": 5,
        "down_days": 2,
        "ibs_threshold": 0.35,
    },
    "daily_research_v6b": {
        "rsi_period": 2,
        "rsi_entry": 25,
        "rsi_entry_cautious": 10,
        "vol_ratio_threshold": 1.5,
        "sma_slope_period": 20,
        "max_hold_days": 5,
        "max_drawdown_pct": 0.12,
        "drawdown_lookback": 40,
    },
    "daily_research_v6c": {
        "max_drawdown_pct": 0.10,
        "drawdown_lookback": 40,
        "max_hold_days": 5,
    },
    "daily_research_v6d": {
        "rsi_period": 2,
        "trend_period": 50,
        "max_hold_days": 5,
        "rsi_entry": 15.0,
        "stop_atr": 1.0,
        "target_atr": 1.5,
        "max_stop_pct": 0.02,
    },
}

PARAM_SPACES: dict[str, list[ParamDef]] = {
    "trend_rider_pro": [
        ParamDef("confluence_threshold", "float", low=55.0, high=75.0, step=5.0, description="Confluence score"),
        ParamDef("min_trend_alignment", "float", low=0.15, high=0.55, step=0.05, description="MTF alignment"),
        ParamDef("pullback_threshold", "float", low=0.004, high=0.012, step=0.001, description="Pullback distance"),
        ParamDef("stop_atr_mult", "float", low=1.75, high=3.25, step=0.25, description="ATR stop"),
        ParamDef("target_atr_mult", "float", low=3.0, high=5.0, step=0.5, description="ATR target"),
        ParamDef("trail_min_profit_r", "float", low=0.5, high=1.0, step=0.05, description="Trail profit R"),
        ParamDef("max_hold_minutes", "int", low=90, high=180, step=15, description="Max hold minutes"),
    ],
    "mean_reversion_pro": [
        ParamDef("confluence_threshold", "float", low=40.0, high=75.0, step=5.0, description="Confluence score"),
        ParamDef("vwap_dist_threshold", "float", low=0.002, high=0.008, step=0.001, description="VWAP distance"),
        ParamDef("bb_pos_threshold", "float", low=0.15, high=0.6, step=0.05, description="BB position"),
        ParamDef("max_hold_minutes", "int", low=20, high=90, step=5, description="Max hold minutes"),
        ParamDef("stop_atr_mult", "float", low=1.0, high=2.5, step=0.25, description="ATR stop"),
        ParamDef("hurst_gate_threshold", "float", low=0.0, high=0.55, step=0.05, description="Hurst gate"),
        ParamDef("hmm_min_confidence", "float", low=0.0, high=0.8, step=0.1, description="HMM gate"),
    ],
    "orb_v2": [
        ParamDef("confluence_threshold", "float", low=40.0, high=75.0, step=5.0, description="Confluence score"),
        ParamDef("vol_gate_mult", "float", low=1.1, high=1.7, step=0.1, description="Volume gate"),
        ParamDef("trail_min_profit_r", "float", low=0.5, high=1.25, step=0.25, description="Trail profit R"),
        ParamDef("max_hold_minutes", "int", low=60, high=120, step=15, description="Max hold minutes"),
        ParamDef("hmm_min_confidence", "float", low=0.0, high=0.8, step=0.1, description="HMM gate"),
    ],
    "pair_trading_v2": [
        ParamDef("entry_z_threshold", "float", low=1.0, high=2.5, step=0.25, description="Entry Z-score"),
        ParamDef("stop_z_threshold", "float", low=3.0, high=5.0, step=0.25, description="Stop Z-score"),
        ParamDef("confluence_threshold", "float", low=30.0, high=65.0, step=5.0, description="Confluence score"),
        ParamDef("spread_lookback", "int", low=40, high=120, step=10, description="Spread lookback days"),
        ParamDef("min_correlation", "float", low=0.3, high=0.65, step=0.05, description="Min correlation"),
        ParamDef("max_hold_days", "int", low=5, high=30, step=5, description="Max hold days"),
    ],
    "rsi_bounce": [
        ParamDef("confluence_threshold", "float", low=30.0, high=70.0, step=5.0, description="Confluence score"),
        ParamDef("rsi_oversold", "float", low=20.0, high=40.0, step=5.0, description="RSI oversold"),
        ParamDef("rsi_overbought", "float", low=60.0, high=80.0, step=5.0, description="RSI overbought"),
        ParamDef("band_tolerance", "float", low=2.0, high=15.0, step=0.5, description="BB tolerance"),
        ParamDef("stop_atr_mult", "float", low=1.0, high=3.0, step=0.25, description="ATR stop"),
        ParamDef("target_atr_mult", "float", low=2.0, high=5.0, step=0.5, description="ATR target"),
        ParamDef("max_hold_minutes", "int", low=30, high=120, step=10, description="Max hold minutes"),
    ],
    "regime_trend_up": [
        ParamDef("rsi_min", "float", low=25.0, high=45.0, step=5.0, description="RSI lower bound"),
        ParamDef("rsi_max", "float", low=55.0, high=75.0, step=5.0, description="RSI upper bound"),
        ParamDef("pullback_pct", "float", low=0.005, high=0.03, step=0.005, description="Max EMA20 distance"),
        ParamDef("stop_atr_mult", "float", low=1.0, high=2.5, step=0.25, description="ATR stop"),
        ParamDef("target_rr", "float", low=1.5, high=4.0, step=0.5, description="Risk:reward"),
        ParamDef("cooldown_bars", "int", low=3, high=15, step=3, description="Min bars between signals"),
    ],
    "regime_bear": [
        ParamDef("rsi_short_entry", "float", low=45.0, high=65.0, step=5.0, description="RSI short level"),
        ParamDef("rsi_long_entry", "float", low=35.0, high=55.0, step=5.0, description="RSI long level"),
        ParamDef("stop_atr_mult", "float", low=1.0, high=2.5, step=0.25, description="ATR stop"),
        ParamDef("target_atr_mult", "float", low=2.0, high=5.0, step=0.5, description="ATR target"),
        ParamDef("high_vol_target_mult", "float", low=1.0, high=2.5, step=0.25, description="HIGH vol target"),
    ],
    "regime_adaptive": [
        ParamDef("rsi_long_entry", "float", low=35.0, high=50.0, step=5.0, description="RSI BUY in UP"),
        ParamDef("rsi_short_entry", "float", low=55.0, high=70.0, step=5.0, description="RSI SELL in DOWN"),
        ParamDef("stop_atr_mult", "float", low=1.0, high=2.5, step=0.25, description="ATR stop"),
        ParamDef("target_atr_mult", "float", low=2.0, high=5.0, step=0.5, description="ATR target"),
        ParamDef("cooldown_bars", "int", low=3, high=15, step=3, description="Min bars between signals"),
    ],
    "momentum_fade": [
        ParamDef("vwap_threshold", "float", low=0.004, high=0.015, step=0.001, description="VWAP distance"),
        ParamDef("volume_surge_mult", "float", low=1.5, high=3.5, step=0.25, description="Volume surge mult"),
        ParamDef("confluence_threshold", "float", low=40.0, high=75.0, step=5.0, description="Confluence score"),
        ParamDef("stop_atr_mult", "float", low=1.0, high=2.5, step=0.25, description="ATR stop"),
        ParamDef("target_atr_mult", "float", low=2.0, high=5.0, step=0.5, description="ATR target"),
        ParamDef("max_hold_minutes", "int", low=30, high=120, step=15, description="Max hold minutes"),
        ParamDef("hurst_gate_threshold", "float", low=0.0, high=0.6, step=0.05, description="Hurst gate"),
        ParamDef("entropy_threshold", "float", low=0.6, high=1.0, step=0.05, description="Entropy filter"),
        ParamDef("hmm_min_confidence", "float", low=0.0, high=0.8, step=0.1, description="HMM gate"),
    ],
    "autoresearch_strategy": [
        ParamDef("stop_atr_mult", "float", low=1.0, high=3.0, step=0.25, description="ATR stop"),
        ParamDef("target_atr_mult", "float", low=2.0, high=5.0, step=0.5, description="ATR target"),
        ParamDef("cooldown_bars", "int", low=3, high=15, step=3, description="Min bars between signals"),
    ],
    "daily_research_strategy": [
        ParamDef("breakout_period", "int", low=5, high=20, step=5, description="Breakout lookback"),
        ParamDef("pullback_rsi_hi", "float", low=60.0, high=75.0, step=5.0, description="Pullback RSI high"),
        ParamDef("max_hold_days", "int", low=5, high=15, step=5, description="Max hold days"),
    ],
    "daily_research_v6a": [
        ParamDef("target_atr_mult", "float", low=2.0, high=3.5, step=0.5, description="ATR target"),
    ],
    "daily_research_v6b": [
        ParamDef("stop_atr_mult", "float", low=2.5, high=3.5, step=0.5, description="ATR stop"),
        ParamDef("target_atr_mult", "float", low=1.5, high=2.5, step=0.5, description="ATR target"),
    ],
    "daily_research_v6c": [
        ParamDef("rsi_entry", "float", low=15.0, high=30.0, step=5.0, description="RSI(2) entry threshold"),
        ParamDef("stop_atr_mult", "float", low=1.5, high=3.0, step=0.5, description="ATR stop"),
        ParamDef("target_atr_mult", "float", low=0.5, high=1.5, step=0.5, description="ATR target"),
        ParamDef("momentum_lookback", "int", low=3, high=10, step=1, description="Momentum lookback days"),
    ],
    "daily_research_v6d": [],
}


def suggest_params(
    trial: optuna.Trial,
    strategy_name: str,
) -> dict[str, Any]:
    """Use an Optuna trial to suggest parameter values from the space."""
    space = PARAM_SPACES.get(strategy_name)
    if space is None:
        raise ValueError(f"No parameter space defined for '{strategy_name}'")

    params: dict[str, Any] = {}
    locked = LOCKED_PARAMS.get(strategy_name, {})
    params.update(locked)

    for p in space:
        if p.name in locked:
            continue
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

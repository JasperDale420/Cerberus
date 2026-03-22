"""Optuna-based parameter optimization harness for Cerberus strategies.

Architecture:
  1. ``composite_objective()`` — multi-criteria scoring function
  2. ``run_backtest_for_optimization()`` — thin wrapper around the backtest runner
  3. ``create_objective()`` — builds an Optuna objective for a single strategy
  4. ``WalkForwardOptimizer`` — anchored expanding-window WFO framework

Usage (single-strategy optimization)::

    study = optuna.create_study(direction="maximize", ...)
    objective = create_objective("trend_rider_pro", base_config, ...)
    study.optimize(objective, n_trials=100)

Usage (walk-forward)::

    wfo = WalkForwardOptimizer(...)
    result = wfo.run("trend_rider_pro", base_config, data_dir)

Parallel execution: launch multiple processes against the same SQLite-backed
study — Optuna handles coordination automatically.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from dateutil.relativedelta import relativedelta

from src.analytics.param_sensitivity import analyze_param_sensitivity
from src.analytics.param_spaces import suggest_params
from src.analytics.report_card import save_report


@dataclass(frozen=True)
class WFORunContext:
    """Resolved artifact locations and study naming for a single WFO run."""

    strategy_name: str
    run_tag: str
    artifact_dir: Path
    storage_uri: str
    study_prefix: str

    def study_name_for_window(self, window_index: int) -> str:
        return f"{self.study_prefix}_wfo_{window_index}"


@dataclass
class HoldoutResult:
    """Result of validating best WFO params on the untouched holdout window."""

    params_used: dict
    holdout_sharpe: float
    holdout_pf: float
    holdout_max_dd: float
    holdout_n_trades: int
    holdout_score: float
    oos_to_holdout_ratio: float
    passed: bool


def _normalize_symbols_for_run(base_config: dict[str, Any]) -> list[str]:
    """Extract a deterministic symbol list from the configured optimization universe."""
    symbols: set[str] = set()
    universe = base_config.get("universe", {})
    if isinstance(universe, list):
        for block in universe:
            if isinstance(block, dict):
                for symbol in block.get("symbols", []) or []:
                    if symbol:
                        symbols.add(str(symbol).upper())
    elif isinstance(universe, dict):
        for symbol in universe.get("symbols", []) or []:
            if symbol:
                symbols.add(str(symbol).upper())
    return sorted(symbols)


def _slugify_run_tag(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value).strip())
    normalized = normalized.strip("-_")
    return normalized or "wfo-run"


def build_wfo_run_context(
    strategy_name: str,
    full_start: str,
    full_end: str,
    min_train_months: int,
    test_months: int,
    holdout_months: int,
    n_trials: int,
    mode: str,
    base_config: dict[str, Any],
    data_dir: str,
    run_tag: str | None = None,
    artifact_root: str = "artifacts/optimization",
    generated_at: datetime | None = None,
) -> WFORunContext:
    """Build an isolated run context so different WFO sweeps never share studies."""
    symbols = _normalize_symbols_for_run(base_config)
    payload = {
        "strategy": strategy_name,
        "full_start": full_start,
        "full_end": full_end,
        "min_train_months": min_train_months,
        "test_months": test_months,
        "holdout_months": holdout_months,
        "n_trials": n_trials,
        "mode": mode,
        "data_dir": str(Path(data_dir)),
        "symbols": symbols,
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]

    if run_tag is None:
        ts = generated_at or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        stamp = ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_tag = f"{stamp}-{len(symbols)}sym-{n_trials}tr-{signature}"

    normalized_tag = _slugify_run_tag(run_tag)
    artifact_dir = Path(artifact_root) / "runs" / strategy_name / normalized_tag
    artifact_dir.mkdir(parents=True, exist_ok=True)
    storage_uri = f"sqlite:///{artifact_dir / 'study.db'}"

    return WFORunContext(
        strategy_name=strategy_name,
        run_tag=normalized_tag,
        artifact_dir=artifact_dir,
        storage_uri=storage_uri,
        study_prefix=f"{strategy_name}_{normalized_tag}",
    )


# ---------------------------------------------------------------------------
# HMM per-window retraining helpers
# ---------------------------------------------------------------------------


def _aggregate_bars_to_daily_for_hmm(bars_df: pd.DataFrame, symbol: str = "SPY") -> pd.DataFrame:
    """Convert minute bars to daily OHLCV for HMM training.

    Filters to a single symbol (index), aggregates to daily, and formats
    the result for ``HmmRegimeService.fit_ohlcv()``.
    """
    if bars_df.empty:
        return pd.DataFrame()

    # Filter to index symbol if multi-symbol
    if "symbol" in bars_df.columns:
        bars_df = bars_df[bars_df["symbol"] == symbol]

    if bars_df.empty:
        return pd.DataFrame()

    working = bars_df.copy()
    working["date"] = pd.to_datetime(working["timestamp"]).dt.date
    daily = (
        working.groupby("date")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )

    daily["time"] = pd.to_datetime(daily["date"])
    daily = daily[["time", "open", "high", "low", "close", "volume"]]
    return daily.sort_values("time").reset_index(drop=True)


def _retrain_hmm_for_window(
    bars_df: pd.DataFrame,
    window_idx: int,
    config: dict[str, Any],
    artifact_root: str = "artifacts/regime_models/hmm",
    shared_hmm_root: str | None = None,
) -> str | None:
    """Train an HMM on training-window bars and save to a window-specific directory.

    Returns the artifact directory path on success, or ``None`` if HMM is
    disabled or there is insufficient data.

    If ``shared_hmm_root`` is provided, HMM artifacts are cached there by window
    index and reused across strategies (the HMM is market-level, not strategy-specific).
    """
    hmm_cfg_raw = config.get("hmm_regime", {})
    if not isinstance(hmm_cfg_raw, dict):
        return None

    enabled = hmm_cfg_raw.get("enabled", False)
    runtime_cfg = hmm_cfg_raw.get("runtime", {})
    runtime_enabled = runtime_cfg.get("enabled", False) if isinstance(runtime_cfg, dict) else False
    if not (enabled and runtime_enabled):
        return None

    # Check shared cache first — HMM is market-level (SPY), same for all strategies
    if shared_hmm_root:
        cached_dir = f"{shared_hmm_root}/wfo_window_{window_idx}"
        model_file = os.path.join(cached_dir, "model.pkl")
        if os.path.exists(model_file):
            print(
                f"  [HMM] Reusing cached model for window {window_idx} → {cached_dir}",
                flush=True,
            )
            return cached_dir

    index_symbol = config.get("index_symbol", "SPY")
    daily_bars = _aggregate_bars_to_daily_for_hmm(bars_df, index_symbol)

    min_samples = 30
    if len(daily_bars) < min_samples:
        print(
            f"  [HMM] Insufficient daily bars for window {window_idx}: "
            f"{len(daily_bars)} < {min_samples} — skipping HMM retraining",
            flush=True,
        )
        return None

    from src.regime_models.hmm.config import HmmConfig
    from src.regime_models.hmm.service import HmmRegimeService

    # Save to shared root if available, otherwise per-strategy
    window_dir = f"{shared_hmm_root or artifact_root}/wfo_window_{window_idx}"
    os.makedirs(window_dir, exist_ok=True)

    # Build HmmConfig from the raw config dict, overriding artifact_dir
    hmm_config_data = {**hmm_cfg_raw, "artifact_dir": window_dir}
    hmm_config = HmmConfig.model_validate(hmm_config_data)

    class _QuietLogger:
        """Minimal logger for HMM training in optimization context."""

        def info(self, msg: str, **kw: Any) -> None:
            pass

        def warning(self, msg: str, **kw: Any) -> None:
            print(f"  [HMM WARNING] {msg}", flush=True)

    hmm_service = HmmRegimeService(config=hmm_config, logger=_QuietLogger())
    summary = hmm_service.fit_ohlcv(daily_bars)
    hmm_service.save()

    print(
        f"  [HMM] Trained for window {window_idx}: "
        f"{summary['n_samples']} samples, {summary['n_states']} states → {window_dir}",
        flush=True,
    )
    return window_dir


# ---------------------------------------------------------------------------
# Multiprocessing worker for true parallel Optuna trials
# ---------------------------------------------------------------------------


def _precompute_indicators_for_window(
    start_date: str,
    end_date: str,
    data_dir: str,
    base_config: dict[str, Any],
) -> None:
    """Pre-compute MTF indicators once per WFO window.

    Called in each multiprocessing worker BEFORE ``study.optimize()`` so that
    all trials in the window reuse the cached indicators via
    ``skip_indicator_precompute=True``.
    """
    from src.backtest.indicator_precompute import (
        clear_precomputed,
        install_precomputed,
        precompute_symbol_indicators,
    )
    from src.backtest.runner import _load_cached_parquet_bars
    from src.core.logger import StructuredLogger

    logger = StructuredLogger("PRECOMPUTE", level="WARNING")

    start_dt = datetime.fromisoformat(start_date)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_date)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    # Collect universe symbols
    symbols: set[str] = set()
    index_symbol = base_config.get("index_symbol", "SPY")
    vol_symbol = base_config.get("regime", {}).get("vol_symbol", "VXX")
    symbols.add(index_symbol)
    symbols.add(vol_symbol)
    universes = base_config.get("universe", {})
    if isinstance(universes, list):
        for u in universes:
            for s in u.get("symbols", []):
                symbols.add(s)
    elif isinstance(universes, dict):
        for s in universes.get("symbols", []):
            symbols.add(s)

    bars_df = _load_cached_parquet_bars(
        Path(data_dir),
        symbols,
        start_dt,
        end_dt,
        logger,
        bar_resolution_minutes=1,
    )
    if bars_df.empty:
        return

    clear_precomputed()
    installed = 0
    for sym in sorted(symbols):
        sym_df = bars_df[bars_df["symbol"] == sym].sort_values("timestamp")
        if sym_df.empty:
            continue
        try:
            precomp = precompute_symbol_indicators(
                timestamps=sym_df["timestamp"].values,
                opens=sym_df["open"].values,
                highs=sym_df["high"].values,
                lows=sym_df["low"].values,
                closes=sym_df["close"].values,
                volumes=sym_df["volume"].values,
                vwaps=sym_df["vwap"].values,
            )
            install_precomputed(sym, precomp)
            installed += 1
        except Exception:
            pass

    print(
        f"  [Worker] Pre-computed MTF indicators for {installed}/{len(symbols)} symbols",
        flush=True,
    )


def _precompute_pair_factors_for_window(
    start_date: str,
    end_date: str,
    data_dir: str,
    base_config: dict[str, Any],
) -> None:
    """Pre-compute Kalman hedge ratios and spreads for all configured pairs.

    The PairFactorCache is injected into base_config so that every trial's
    strategy instance reads from it instead of running the Kalman filter
    online.  This is safe because the cache is read-only (numpy arrays).
    """
    from src.analytics.pair_factor_cache import PairFactorCache
    from src.backtest.runner import _load_cached_parquet_bars
    from src.core.logger import StructuredLogger

    pairs_config = base_config.get("strategies", {}).get("pair_trading_v2", {}).get("pairs", [])
    if not pairs_config:
        return

    logger = StructuredLogger("PAIR-CACHE", level="WARNING")

    start_dt = datetime.fromisoformat(start_date)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_date)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    # Collect symbols needed for pars
    pair_symbols: set[str] = set()
    for p in pairs_config:
        pair_symbols.add(p["leg_a"])
        pair_symbols.add(p["leg_b"])

    bars_df = _load_cached_parquet_bars(
        Path(data_dir),
        pair_symbols,
        start_dt,
        end_dt,
        logger,
        bar_resolution_minutes=1,
    )
    if bars_df.empty:
        return

    cache = PairFactorCache(pairs_config, bars_df)
    n_computed = cache.precompute()

    # Inject into base_config so create_objective → _apply_params_to_config
    # propagates it to each trial's strategy config.
    base_config.setdefault("strategies", {}).setdefault("pair_trading_v2", {})["_factor_cache"] = cache

    print(
        f"  [Worker] Pre-computed Kalman factors for {n_computed} pairs",
        flush=True,
    )


def _mp_optimize_worker(
    study_name: str,
    storage: str,
    strategy_name: str,
    base_config: dict,
    start_date: str,
    end_date: str,
    data_dir: str,
    config_path: str,
    n_trials: int,
    worker_id: int,
    min_trades_per_month: float = 3,
) -> None:
    """Run Optuna trials in a separate process for true CPU parallelism.

    Each worker loads the shared study from SQLite, creates its own objective,
    and runs its share of trials. Optuna coordinates trial suggestions via the
    shared DB so TPE sampling remains effective.

    Pre-computes MTF indicators *once* for the window so individual trials
    skip the expensive Numba precompute step.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pre-compute indicators once for this window (all trials reuse them)
    _precompute_indicators_for_window(start_date, end_date, data_dir, base_config)

    # Pre-compute pair factor cache for pair_trading_v2 (Kalman is expensive)
    if strategy_name == "pair_trading_v2":
        _precompute_pair_factors_for_window(start_date, end_date, data_dir, base_config)

    study = optuna.load_study(study_name=study_name, storage=storage)
    objective = create_objective(
        strategy_name=strategy_name,
        base_config=base_config,
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
        config_path=config_path,
        skip_indicator_precompute=True,
        min_trades_per_month=min_trades_per_month,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)


# ---------------------------------------------------------------------------
# Composite objective function
# ---------------------------------------------------------------------------


def composite_objective(metrics: dict[str, Any], *, min_trades: int = 30) -> float:
    """Multi-criteria objective for Optuna to MAXIMIZE.

    Weights:
      25% blended Sharpe  +  30% Profit Factor  +  15% Calmar
      + 15% Win-Rate bonus  + 10% Trade-count bonus
      + 5% EV-per-trade bonus
      - Drawdown penalty (exponential above 15%)

    Blended Sharpe uses trade-level Sharpe for short-hold strategies
    (< 60 min avg hold) and daily Sharpe for longer-hold ones, to avoid
    the artifact where daily Sharpe is deeply negative despite positive PnL
    for high-frequency intraday strategies.

    Hard-rejects:
      - < min_trades trades (statistical insignificance)
      - Negative expectancy (avg_pnl <= 0)
    """
    n_trades = metrics.get("n_trades", 0)
    if n_trades < min_trades:
        return -1000.0

    avg_pnl = metrics.get("avg_pnl", 0.0)
    if avg_pnl <= 0:
        return -500.0

    # Blended Sharpe: prefer trade-level for short holds, daily for long holds
    daily_sharpe = metrics.get("sharpe_ratio", 0.0)
    trade_sharpe = metrics.get("trade_sharpe_ratio", 0.0)
    avg_hold_min = metrics.get("avg_hold_minutes", 999)

    if not math.isclose(trade_sharpe, 0.0, abs_tol=1e-9):
        if avg_hold_min < 60:
            sharpe = 0.7 * trade_sharpe + 0.3 * daily_sharpe
        elif avg_hold_min > 120:
            sharpe = 0.3 * trade_sharpe + 0.7 * daily_sharpe
        else:
            sharpe = 0.5 * trade_sharpe + 0.5 * daily_sharpe
    else:
        sharpe = daily_sharpe

    pf = metrics.get("profit_factor", 0.0)
    calmar = metrics.get("calmar_ratio", 0.0)
    winrate = metrics.get("winrate", 0.0)
    max_dd = metrics.get("max_drawdown_pct", 100.0)

    # Clamp values for numerical safety
    sharpe = max(-5.0, min(sharpe, 10.0))
    pf = max(0.0, min(pf, 10.0))
    calmar = max(-5.0, min(calmar, 20.0))

    # Drawdown penalty: exponential above 15%
    dd_penalty = 0.0
    if max_dd > 15.0:
        dd_penalty = (max_dd - 15.0) ** 1.5

    # Trade-count bonus: more trades = more statistical confidence (saturates at 100)
    trade_bonus = min(n_trades / 100.0, 1.0)

    # Direct EV bonus: reward positive expectancy (capped contribution)
    ev_bonus = min(avg_pnl / 10.0, 1.0)

    # Slippage friction penalty: short holds are disproportionately impacted
    # by spread/slippage costs that aren't fully modeled in backtest.
    # WFO showed MRP converging to 2-4 min holds which is unrealistic
    # without DMA.  Penalize to steer optimizer toward realistic holds.
    friction_penalty = 0.0
    if avg_hold_min < 5:
        friction_penalty = 2.0  # severe — sub-5m is scalping
    elif avg_hold_min < 10:
        friction_penalty = 1.0  # moderate — sub-10m very fill-sensitive
    elif avg_hold_min < 20:
        friction_penalty = 0.3  # mild — still affected by slippage

    score = (
        0.25 * sharpe
        + 0.30 * pf
        + 0.15 * calmar
        + 0.15 * (winrate * 10.0)  # scale to ~similar magnitude
        + 0.10 * trade_bonus
        + 0.05 * ev_bonus
        - 0.05 * dd_penalty
        - friction_penalty
    )
    return float(round(score, 6))


def _compute_window_min_trades(
    start_date: str,
    end_date: str,
    *,
    trades_per_month: float = 3,
    floor: int | None = None,
    fallback: int = 10,
) -> int:
    """Scale the minimum trade threshold to the length of the optimization window.

    When floor is None, defaults to max(trades_per_month + 2, 5).  Callers
    passing a low trades_per_month (e.g. 1 for daily pair trading) get a
    correspondingly lower floor.
    """
    if floor is None:
        floor = max(int(trades_per_month) + 2, 3)
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        months = max((end_dt - start_dt).days / 30, 1)
        return max(int(months * trades_per_month), floor)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Backtest runner for optimization
# ---------------------------------------------------------------------------


def run_backtest_for_optimization(
    start_date: str,
    end_date: str,
    config: dict[str, Any],
    data_dir: str,
    config_path: str = "config/backtest_v2.yaml",
    skip_indicator_precompute: bool = False,
) -> dict[str, Any]:
    """Run a single backtest synchronously and return metrics dict.

    This wraps the async ``run_backtest()`` from runner.py but runs it
    with ``asyncio.run()``.  Each trial gets its own DB to prevent
    contention.

    When *skip_indicator_precompute* is True, the runner skips the expensive
    Numba MTF indicator precompute (assumes indicators are already installed
    in the module-level cache by the worker).
    """
    # Lazy import to avoid circular deps
    from src.backtest.runner import run_backtest as _run_backtest_async

    # Give each trial its own DB to allow parallel execution
    trial_id = f"trial_{os.getpid()}_{int(time.time() * 1000) % 1_000_000}"
    db_dir = os.path.join(os.getcwd(), ".agents", "tmp", "optuna_dbs")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"{trial_id}.db")
    config["database_url"] = f"sqlite:///{db_path}"

    # Reduce logging overhead during optimization
    config["log_level"] = "ERROR"
    try:
        import src.core.logger as _logger_mod

        _logger_mod._configured = False
    except Exception:
        pass

    # Patch ConfigLoader so the runner uses our pre-built config dict
    # instead of loading from YAML files (runner calls ConfigLoader.load_config internally).
    # Using unittest.mock.patch as a context manager guarantees the original
    # method is restored even if the backtest raises an exception.
    from unittest.mock import patch as _mock_patch

    from src.core.config import ConfigLoader

    def _patched_load(self, config_path_or_dir=None):
        # Preserve PairFactorCache reference through deepcopy
        fc = config.get("strategies", {}).get("pair_trading_v2", {}).pop("_factor_cache", None)
        copied = copy.deepcopy(config)
        if fc is not None:
            config.get("strategies", {}).get("pair_trading_v2", {})["_factor_cache"] = fc
            copied.get("strategies", {}).get("pair_trading_v2", {})["_factor_cache"] = fc
        return copied

    try:
        with _mock_patch.object(ConfigLoader, "load_config", _patched_load):
            result = asyncio.run(
                _run_backtest_async(
                    start_date=start_date,
                    end_date=end_date,
                    config_path=config_path,
                    data_dir=data_dir,
                    skip_indicator_precompute=skip_indicator_precompute,
                )
            )
        if result is None:
            return {"n_trades": 0}
        if isinstance(result, dict):
            return result
        # BacktestReportCard — use to_dict() for correctly-keyed metrics
        if hasattr(result, "to_dict"):
            return dict(result.to_dict())
        return {"n_trades": 0}
    except Exception as e:
        print(f"[optuna] Backtest trial failed: {e}", file=sys.stderr)
        return {"n_trades": 0}
    finally:
        # When indicators are pre-installed by the worker, do NOT clear them
        # between trials — that defeats the caching optimization.
        if not skip_indicator_precompute:
            try:
                from src.backtest.indicator_precompute import clear_precomputed

                clear_precomputed()
            except Exception:
                pass
        # Clean up trial DB
        for suffix in ("", "-journal", "-wal", "-shm"):
            p = db_path + suffix
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Apply optimized params to config
# ---------------------------------------------------------------------------


def _apply_params_to_config(
    config: dict[str, Any],
    strategy_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Deep-copy config and inject trial parameters into the strategy block."""
    # Preserve PairFactorCache reference — it contains large numpy arrays
    # that should be shared (read-only) across trials, not deep-copied.
    factor_cache = config.get("strategies", {}).get(strategy_name, {}).pop("_factor_cache", None)
    cfg = copy.deepcopy(config)
    if factor_cache is not None:
        # Re-inject into both original and copy
        config.get("strategies", {}).get(strategy_name, {})["_factor_cache"] = factor_cache
        cfg.get("strategies", {}).get(strategy_name, {})["_factor_cache"] = factor_cache
    strat_cfg = cfg.get("strategies", {}).get(strategy_name, {})

    # Parameters that map directly to strategy config keys
    config_params = {
        "confluence_threshold",
        "min_bars",
        "cooldown_bars",
        "min_trend_alignment",
        "min_flow_direction",
        "time_window_start",
        "time_window_end",
        "max_hold_minutes",
        "rsi_oversold",
        "rsi_overbought",
        "band_tolerance",
        "vwap_threshold",
        "volume_surge_mult",
        "vwap_dist_threshold",
        "bb_pos_threshold",
        "stop_atr_mult",
        "target_atr_mult",
        "hurst_gate_threshold",
        "hurst_trending_threshold",
        "bocpd_reject_threshold",
        "kurtosis_reject_threshold",
        "entropy_threshold",
        # Pair trading V2 params
        "entry_z_threshold",
        "stop_z_threshold",
        "spread_lookback",
        "min_correlation",
        "max_hold_days",
    }

    for key, value in params.items():
        if key in config_params:
            strat_cfg[key] = value

    # Wire flat hmm_min_confidence into the nested hmm_gate.min_confidence config
    if "hmm_min_confidence" in params:
        hmm_gate = strat_cfg.setdefault("hmm_gate", {})
        hmm_gate["min_confidence"] = params["hmm_min_confidence"]
        # Ensure the HMM gate is enabled so the min_confidence value takes effect
        hmm_gate.setdefault("enabled", True)

    # Store non-config params as overrides for the strategy to read
    # (exclude hmm_min_confidence since it's handled above via nested config)
    strat_cfg["_optuna_overrides"] = {
        k: v for k, v in params.items() if k not in config_params and k != "hmm_min_confidence"
    }

    # Inject max_hold_minutes into the exit_config if present
    if "max_hold_minutes" in params:
        # This is set at strategy level; the strategy's _exit_config() method
        # checks config first.
        strat_cfg["max_hold_minutes"] = params["max_hold_minutes"]

    cfg["strategies"][strategy_name] = strat_cfg
    return cfg


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def _validate_params(params: dict[str, Any]) -> bool:
    """Validate that trial parameters are internally consistent.

    Returns True if the parameter combination is valid.  Invalid combinations
    (e.g. stop wider than target) would waste compute on a guaranteed-bad
    backtest, so the trial should be pruned immediately.
    """
    # Stop must be tighter than target
    if "stop_atr_mult" in params and "target_atr_mult" in params:
        if params["stop_atr_mult"] >= params["target_atr_mult"]:
            return False

    # Confluence threshold is a percentage 0-100
    if "confluence_threshold" in params:
        if not (0 <= params["confluence_threshold"] <= 100):
            return False

    # Distance / position thresholds must be positive
    for key in ("vwap_dist_threshold", "bb_pos_threshold"):
        if key in params and params[key] <= 0:
            return False

    return True


# ---------------------------------------------------------------------------
# Optuna objective factory
# ---------------------------------------------------------------------------


def create_objective(
    strategy_name: str,
    base_config: dict[str, Any],
    start_date: str,
    end_date: str,
    data_dir: str,
    config_path: str = "config/backtest_v2.yaml",
    skip_indicator_precompute: bool = False,
    min_trades_per_month: float = 3,
):
    """Create an Optuna objective function for a single strategy.

    Returns a callable ``objective(trial) -> float`` suitable for
    ``study.optimize()``.
    """

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()

        # 1. Suggest parameters
        params = suggest_params(trial, strategy_name)

        # 1b. Validate — prune immediately if params are inconsistent
        if not _validate_params(params):
            raise optuna.TrialPruned(f"Invalid param combination (e.g. stop >= target): {params}")

        # 2. Build modified config
        config = _apply_params_to_config(base_config, strategy_name, params)

        # Optional: disable other strategies to isolate this one
        for sname in config.get("strategies", {}):
            if sname != strategy_name:
                config["strategies"][sname]["enabled"] = False

        # Ensure target strategy is in ALL regime routing lists so it
        # activates regardless of market regime during optimization.
        # Without this, strategies only routed in specific regimes (e.g.
        # pair_trading_v2 only in 'chop') produce zero trades during
        # bull/bear periods.
        for regime in config.get("strategy_routing", {}):
            routing = config["strategy_routing"][regime]
            if strategy_name not in routing:
                routing.append(strategy_name)

        # 3. Run backtest
        metrics = run_backtest_for_optimization(
            start_date=start_date,
            end_date=end_date,
            config=config,
            data_dir=data_dir,
            config_path=config_path,
            skip_indicator_precompute=skip_indicator_precompute,
        )

        elapsed = time.time() - t0

        # 4. Pruning: reject insufficient trade count
        # Compute proportional min based on window length (5 trades/month baseline)
        _min_trades = _compute_window_min_trades(start_date, end_date, trades_per_month=min_trades_per_month)
        n_trades = metrics.get("n_trades", 0)
        if n_trades < _min_trades:
            print(
                f"  Trial {trial.number}: PRUNED ({n_trades} trades, {elapsed:.1f}s)",
                flush=True,
            )
            raise optuna.TrialPruned(f"Only {n_trades} trades")

        # 5. Compute composite score
        score = composite_objective(metrics, min_trades=_min_trades)

        # Log key metrics for the dashboard
        trial.set_user_attr("n_trades", n_trades)
        trial.set_user_attr("sharpe", metrics.get("sharpe_ratio", 0.0))
        trial.set_user_attr("trade_sharpe", metrics.get("trade_sharpe_ratio", 0.0))
        trial.set_user_attr("profit_factor", metrics.get("profit_factor", 0.0))
        trial.set_user_attr("max_dd_pct", metrics.get("max_drawdown_pct", 0.0))
        trial.set_user_attr("winrate", metrics.get("winrate", 0.0))
        trial.set_user_attr("net_pnl", metrics.get("net_pnl", 0.0))
        trial.set_user_attr("avg_hold_min", metrics.get("avg_hold_minutes", 0.0))

        print(
            f"  Trial {trial.number}: score={score:.4f} "
            f"trades={n_trades} sharpe={metrics.get('sharpe_ratio', 0):.2f} "
            f"pnl=${metrics.get('net_pnl', 0):.0f} {elapsed:.1f}s",
            flush=True,
        )

        return score

    return objective


# ---------------------------------------------------------------------------
# Walk-Forward Optimizer
# ---------------------------------------------------------------------------


class WalkForwardOptimizer:
    """Walk-forward optimization with Optuna.

    Supports two modes:

    **Anchored** (default): training window expands from a fixed start::

        Round 1: Train [Jan-Mar]  Test [Apr]
        Round 2: Train [Jan-Apr]  Test [May]
        Round 3: Train [Jan-May]  Test [Jun]

    **Rolling**: fixed-width training window slides forward::

        Round 1: Train [Jan-Mar]  Test [Apr]
        Round 2: Train [Feb-Apr]  Test [May]
        Round 3: Train [Mar-May]  Test [Jun]

    Rolling mode is better for regime adaptation since older data doesn't
    dilute recent patterns.

    Each training window is optimized with ``n_trials`` Optuna trials.
    The best params are validated on the out-of-sample test window.
    The WFO efficiency ratio = mean(OOS score) / mean(IS score).
    """

    def __init__(
        self,
        full_start: str,
        full_end: str,
        min_train_months: int = 3,
        test_months: int = 1,
        n_trials: int = 100,
        holdout_months: int = 2,
        mode: str = "rolling",
    ):
        self.full_start = datetime.fromisoformat(full_start)
        self.full_end = datetime.fromisoformat(full_end)
        self.min_train_months = min_train_months
        self.test_months = test_months
        self.n_trials = n_trials
        self.holdout_months = holdout_months
        self.mode = mode

    def get_windows(self) -> list[dict[str, str]]:
        """Generate WFO windows, reserving holdout at end.

        Uses ``dateutil.relativedelta`` for calendar-accurate month
        boundaries instead of the ``30 * N`` day approximation that
        causes windows to drift over long time spans.
        """
        holdout_start = self.full_end - relativedelta(months=self.holdout_months)
        usable_end = holdout_start

        windows = []

        if self.mode == "rolling":
            # Fixed-width rolling window that slides by test_months
            train_start = self.full_start

            while train_start + relativedelta(months=self.min_train_months + self.test_months) <= usable_end:
                train_end = train_start + relativedelta(months=self.min_train_months)
                # Test starts day AFTER train_end to prevent 1-day lookahead leak
                test_start_dt = train_end + timedelta(days=1)
                test_end = test_start_dt + relativedelta(months=self.test_months)
                windows.append(
                    {
                        "train_start": train_start.strftime("%Y-%m-%d"),
                        "train_end": train_end.strftime("%Y-%m-%d"),
                        "test_start": test_start_dt.strftime("%Y-%m-%d"),
                        "test_end": test_end.strftime("%Y-%m-%d"),
                    }
                )
                train_start = train_start + relativedelta(months=self.test_months)
        else:
            # Anchored expanding: training window grows from fixed start
            train_start = self.full_start
            test_start = train_start + relativedelta(months=self.min_train_months)

            while test_start + relativedelta(months=self.test_months) <= usable_end:
                test_end = test_start + relativedelta(months=self.test_months)
                # train_end is day BEFORE test_start to prevent 1-day lookahead leak
                train_end_dt = test_start - timedelta(days=1)
                windows.append(
                    {
                        "train_start": train_start.strftime("%Y-%m-%d"),
                        "train_end": train_end_dt.strftime("%Y-%m-%d"),
                        "test_start": test_start.strftime("%Y-%m-%d"),
                        "test_end": test_end.strftime("%Y-%m-%d"),
                    }
                )
                test_start = test_end

        return windows

    def get_holdout_window(self) -> dict[str, str]:
        """Return the final holdout window (never touched during optimization)."""
        holdout_start = self.full_end - relativedelta(months=self.holdout_months)
        return {
            "start": holdout_start.strftime("%Y-%m-%d"),
            "end": self.full_end.strftime("%Y-%m-%d"),
        }

    def run(
        self,
        strategy_name: str,
        base_config: dict[str, Any],
        data_dir: str,
        config_path: str = "config/backtest_v2.yaml",
        study_storage: str | None = None,
        run_tag: str | None = None,
        artifact_root: str = "artifacts/optimization",
        resume_existing: bool = False,
        workers: int | None = None,
        min_trades_per_month: float = 3,
    ) -> dict[str, Any]:
        """Run full walk-forward optimization.

        Returns::

            {
                "strategy": str,
                "windows": [...],
                "oos_metrics": [metrics_per_window],
                "is_scores": [float],
                "oos_scores": [float],
                "best_params_per_window": [dict],
                "wfo_efficiency_ratio": float,
                "holdout_window": dict,
                "param_stability": dict,
            }
        """
        windows = self.get_windows()
        all_oos_metrics: list[dict] = []
        all_is_scores: list[float] = []
        all_oos_scores: list[float] = []
        all_params: list[dict] = []

        run_context = build_wfo_run_context(
            strategy_name=strategy_name,
            full_start=self.full_start.strftime("%Y-%m-%d"),
            full_end=self.full_end.strftime("%Y-%m-%d"),
            min_train_months=self.min_train_months,
            test_months=self.test_months,
            holdout_months=self.holdout_months,
            n_trials=self.n_trials,
            mode=self.mode,
            base_config=base_config,
            data_dir=data_dir,
            run_tag=run_tag,
            artifact_root=artifact_root,
        )
        storage = study_storage or run_context.storage_uri

        wfo_t0 = time.time()

        for i, window in enumerate(windows):
            window_t0 = time.time()
            print(f"\n{'=' * 60}")
            print(f"WFO Window {i + 1}/{len(windows)}")
            print(f"  Train: {window['train_start']} → {window['train_end']}")
            print(f"  Test:  {window['test_start']} → {window['test_end']}")
            print(f"{'=' * 60}")

            # 0. Per-window HMM retraining (avoids lookahead bias)
            #    Train HMM on the training window's daily bars BEFORE launching
            #    IS optimization or OOS backtest. The trained artifacts are saved
            #    to a window-specific directory and the config is updated so that
            #    both IS workers and OOS backtest load the correct model.
            window_config = copy.deepcopy(base_config)
            # Use a shared HMM cache keyed by run_tag so all strategies
            # in the same WFO run share HMM artifacts (market-level model).
            shared_hmm_dir = str(Path("artifacts/optimization/shared_hmm") / (run_context.run_tag or "default"))
            hmm_artifact_dir = _retrain_hmm_for_window(
                bars_df=self._load_training_bars(data_dir, window_config, window),
                window_idx=i,
                config=window_config,
                artifact_root=str(run_context.artifact_dir / "hmm"),
                shared_hmm_root=shared_hmm_dir,
            )
            if hmm_artifact_dir is not None:
                window_config.setdefault("hmm_regime", {})["artifact_dir"] = hmm_artifact_dir

            # 1. Optimize on training window
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42 + i),
                study_name=run_context.study_name_for_window(i),
                storage=storage,
                load_if_exists=resume_existing,
            )

            # True multiprocessing: fork worker processes against shared study DB.
            # Each process gets its own Python interpreter → no GIL contention.
            # Use 'fork' context explicitly — macOS defaults to 'spawn' which
            # re-imports the main module and causes RuntimeError when the caller
            # script lacks `if __name__ == '__main__':` guards.
            import multiprocessing

            mp_ctx = multiprocessing.get_context("fork")
            requested_workers = workers if workers is not None else min(multiprocessing.cpu_count() - 2, 8)
            n_workers = max(min(int(requested_workers), max(multiprocessing.cpu_count() - 1, 1)), 1)
            trials_per_worker = self.n_trials // n_workers
            remainder = self.n_trials % n_workers

            print(f"  Launching {n_workers} parallel workers ({self.n_trials} trials)")

            processes = []
            for w in range(n_workers):
                t = trials_per_worker + (1 if w < remainder else 0)
                if t <= 0:
                    continue
                p = mp_ctx.Process(
                    target=_mp_optimize_worker,
                    args=(
                        study.study_name,
                        storage,
                        strategy_name,
                        window_config,
                        window["train_start"],
                        window["train_end"],
                        data_dir,
                        config_path,
                        t,
                        w,
                        min_trades_per_month,
                    ),
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            # Reload study to pick up all worker results
            study = optuna.load_study(
                study_name=study.study_name,
                storage=storage,
            )

            # Handle case where all trials were pruned (no completed trials)
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed:
                print(f"  WARNING: All {len(study.trials)} trials pruned — skipping window {i + 1}")
                # Use default params so WFO can continue
                best_params = {}
                is_score = 0.0
            else:
                best_params = study.best_trial.params
                is_score = study.best_value
            all_params.append(best_params)
            all_is_scores.append(is_score)

            print(f"  IS best score: {is_score:.4f}")
            print(f"  IS best params: {best_params}")

            # 2. Validate on test window with best params
            oos_config = _apply_params_to_config(window_config, strategy_name, best_params)
            # Disable other strategies
            for sname in oos_config.get("strategies", {}):
                if sname != strategy_name:
                    oos_config["strategies"][sname]["enabled"] = False

            oos_metrics = run_backtest_for_optimization(
                start_date=window["test_start"],
                end_date=window["test_end"],
                config=oos_config,
                data_dir=data_dir,
                config_path=config_path,
            )

            # Proportional min trades for OOS: ~3 trades/month/symbol is reasonable,
            # but for short test windows use a lower floor to avoid rejecting everything.
            _oos_months = max(self.test_months, 1)
            _oos_min_trades = max(int(_oos_months * min_trades_per_month), 3)
            oos_score = composite_objective(oos_metrics, min_trades=_oos_min_trades)
            all_oos_metrics.append(oos_metrics)
            all_oos_scores.append(oos_score)

            # Save per-window analytics report if available
            oos_analytics = oos_metrics.get("analytics", {})
            if oos_analytics:
                try:
                    reports_dir = str(run_context.artifact_dir / "reports")
                    save_report(
                        oos_analytics,
                        strategy_name=strategy_name,
                        symbol="multi",
                        output_dir=reports_dir,
                        run_type="wfo_window",
                        window_idx=i,
                    )
                except Exception:
                    pass  # analytics save failure must not break optimization

            window_elapsed = time.time() - window_t0
            print(f"  OOS score: {oos_score:.4f}")
            print(f"  OOS trades: {oos_metrics.get('n_trades', 0)}")
            print(f"  OOS Sharpe (daily): {oos_metrics.get('sharpe_ratio', 0.0):.3f}")
            print(f"  OOS Sharpe (trade): {oos_metrics.get('trade_sharpe_ratio', 0.0):.3f}")
            print(f"  OOS PF: {oos_metrics.get('profit_factor', 0.0):.2f}")
            total_elapsed = time.time() - wfo_t0
            print(f"  Window {i + 1} time: {window_elapsed / 60:.1f}min  |  Total: {total_elapsed / 60:.1f}min")

            # Deflated Sharpe Ratio — adjust for multiple testing
            oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)
            if oos_sharpe > 0:
                try:
                    dsr = deflated_sharpe_ratio(
                        observed_sharpe=oos_sharpe,
                        num_trials=self.n_trials,
                        n_returns=oos_metrics.get("trading_days", 21),
                    )
                    print(f"  OOS DSR: {dsr:.4f}  (>0.95 = genuine)")
                except Exception:
                    pass

        # 3. Compute WFO efficiency ratio
        # Include all windows — replace reject scores (insufficient trades / negative expectancy)
        # with 0 so failed windows penalize the average instead of being silently dropped
        valid_is = [max(s, 0.0) if s <= -100 else s for s in all_is_scores]
        valid_oos = [max(s, 0.0) if s <= -100 else s for s in all_oos_scores]
        is_avg = sum(valid_is) / len(valid_is) if valid_is else 0.0
        oos_avg = sum(valid_oos) / len(valid_oos) if valid_oos else 0.0
        efficiency = oos_avg / is_avg if is_avg > 0 else 0.0

        # 4. Parameter stability analysis
        param_stability = self._analyze_param_stability(all_params)

        # 5. Deflated Sharpe Ratio over full WFO OOS results
        #    Uses the mean OOS Sharpe and total trials across all windows.
        oos_sharpes = [m.get("sharpe_ratio", 0.0) for m in all_oos_metrics if m.get("sharpe_ratio", 0.0) > 0]
        total_trials = self.n_trials * len(windows)
        total_oos_days = sum(m.get("trading_days", 0) for m in all_oos_metrics)
        if oos_sharpes and total_oos_days > 0:
            mean_oos_sharpe = sum(oos_sharpes) / len(oos_sharpes)
            try:
                wfo_dsr = deflated_sharpe_ratio(
                    observed_sharpe=mean_oos_sharpe,
                    num_trials=total_trials,
                    n_returns=total_oos_days,
                )
            except Exception:
                wfo_dsr = None
        else:
            wfo_dsr = None

        # 6. Parameter sensitivity analysis across all completed trials
        #    Aggregate trial params + scores from the last window's study
        #    (largest training set = most representative param landscape).
        param_sensitivity_results = []
        try:
            last_study = optuna.load_study(
                study_name=run_context.study_name_for_window(len(windows) - 1),
                storage=storage,
            )
            trials_data: dict[str, list] = {}
            for trial in last_study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    for param_name, param_value in trial.params.items():
                        trials_data.setdefault(param_name, []).append(param_value)
                    trials_data.setdefault("score", []).append(trial.value)

            if len(trials_data.get("score", [])) >= 5:
                sensitivity = analyze_param_sensitivity(trials_data)
                param_sensitivity_results = [
                    {
                        "param_name": s.param_name,
                        "correlation": round(s.correlation, 4),
                        "sensitivity_rank": s.sensitivity_rank,
                    }
                    for s in sensitivity
                ]
        except Exception:
            pass

        result = {
            "strategy": strategy_name,
            "run_tag": run_context.run_tag,
            "artifact_dir": str(run_context.artifact_dir),
            "study_storage": storage,
            "windows": windows,
            "oos_metrics": all_oos_metrics,
            "is_scores": all_is_scores,
            "oos_scores": all_oos_scores,
            "best_params_per_window": all_params,
            "wfo_efficiency_ratio": round(efficiency, 4),
            "deflated_sharpe_ratio": wfo_dsr,
            "holdout_window": self.get_holdout_window(),
            "param_stability": param_stability,
            "param_sensitivity": param_sensitivity_results,
        }

        # Save results
        with open(run_context.artifact_dir / "wfo_results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n{'=' * 60}")
        print(f"WFO Complete — {strategy_name}")
        print(f"  Run tag: {run_context.run_tag}")
        print(f"  Efficiency ratio: {efficiency:.4f}  (target > 0.5)")
        print(f"  IS mean score:  {is_avg:.4f}")
        print(f"  OOS mean score: {oos_avg:.4f}")
        dsr_str = f"{wfo_dsr:.4f}" if wfo_dsr is not None else "N/A"
        print(f"  Deflated Sharpe: {dsr_str}  (>0.95 = genuine)")
        print(f"  Artifacts: {run_context.artifact_dir}")
        print(f"{'=' * 60}")

        return result

    @staticmethod
    def _analyze_param_stability(
        params_per_window: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """Check parameter stability across WFO windows.

        Returns per-param stats: mean, std, cv (coefficient of variation).
        A CV > 0.3 suggests the parameter is unstable / overfitting.
        """
        if not params_per_window:
            return {}

        all_keys: set[str] = set()
        for p in params_per_window:
            all_keys.update(p.keys())

        result: dict[str, dict[str, float]] = {}
        for key in sorted(all_keys):
            vals = [p.get(key) for p in params_per_window if key in p]
            numeric = [v for v in vals if isinstance(v, (int, float))]
            if len(numeric) < 2:
                continue

            mean_val = sum(numeric) / len(numeric)
            variance = sum((v - mean_val) ** 2 for v in numeric) / (len(numeric) - 1)
            std_val = math.sqrt(variance)
            cv = std_val / abs(mean_val) if mean_val != 0 else float("inf")

            result[key] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "cv": round(cv, 4),
                "stable": cv < 0.3,
            }

        return result

    @staticmethod
    def _load_training_bars(
        data_dir: str,
        config: dict[str, Any],
        window: dict[str, str],
    ) -> pd.DataFrame:
        """Load parquet bars for the training window period.

        Only loads the index symbol (e.g. SPY) since HMM training only needs
        market-wide regime data, not all universe symbols. This keeps the
        main-process load lightweight.
        """
        from src.core.logger import StructuredLogger

        data_path = Path(data_dir)
        if not data_path.is_dir():
            return pd.DataFrame()

        index_symbol = config.get("index_symbol", "SPY")

        train_start = datetime.fromisoformat(window["train_start"])
        if train_start.tzinfo is None:
            train_start = train_start.replace(tzinfo=timezone.utc)

        train_end = datetime.fromisoformat(window["train_end"])
        if train_end.tzinfo is None:
            train_end = train_end.replace(tzinfo=timezone.utc)

        logger = StructuredLogger("HMM-WFO", level="WARNING")

        from src.backtest.runner import _load_bars_from_parquet

        bars = _load_bars_from_parquet(
            data_path,
            {index_symbol},
            train_start,
            train_end,
            logger,
        )
        return bars


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


def deflated_sharpe_ratio(
    observed_sharpe: float,
    num_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_returns: int = 252,
) -> float:
    """Bailey & Lopez de Prado's Deflated Sharpe Ratio.

    Adjusts the observed Sharpe for the number of optimization trials.
    A DSR > 0.95 means the result is likely genuine (not due to overfitting).
    """
    from scipy import stats as sp_stats

    if num_trials <= 1 or n_returns <= 0:
        return 0.0

    euler_mascheroni = 0.5772156649
    expected_max = sp_stats.norm.ppf(1 - 1 / num_trials) * (1 - euler_mascheroni / math.log(max(num_trials, 2)))
    se = math.sqrt((1 - skewness * observed_sharpe + (kurtosis - 3) / 4 * observed_sharpe**2) / n_returns)
    if se <= 0:
        return 0.0

    return float(sp_stats.norm.cdf((observed_sharpe - expected_max) / se))

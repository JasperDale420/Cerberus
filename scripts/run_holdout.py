"""Holdout validation for the Cerberus autoresearch loop.

Two modes:

1. **Preset mode** (legacy) — `python scripts/run_holdout.py [--strategy NAME]`
   Uses the hardcoded `HOLDOUT_PARAMS` dict + a fixed Nov-Dec 2024 window.
   Kept for backwards compatibility with manual review of `orb_v2`/`trend_rider_pro`.

2. **Driver mode** (new) — `python scripts/run_holdout.py --strategy NAME --start YYYY-MM-DD --end YYYY-MM-DD [--params-from PATH]`
   Runs a single backtest on the given window with HEAD code/config, computes the
   SPY buy-and-hold benchmark with the same sign-safe ratio math as the WFO
   harness (`scripts/cerberus_autoresearch.py`), and emits a parseable
   `HOLDOUT_RESULT` line for the autoresearch driver to consume.

   When `--params-from` points to an autoresearch latest.json (or any JSON with a
   top-level `param_stability` map), the script applies the per-param mean as
   the holdout config so the validation uses WFO-selected params, not just
   strategy file defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, ".")

from src.analytics.optuna_harness import _apply_params_to_config, run_backtest_for_optimization

# After-tax constants — kept in sync with scripts/cerberus_autoresearch.py.
ST_TAX = 0.35
LT_TAX = 0.18

# -----------------------------------------------------------------------
# WFO-selected consensus params per strategy (preset mode only)
# -----------------------------------------------------------------------
HOLDOUT_PARAMS: dict[str, dict] = {
    "orb_v2": {
        "confluence_threshold": 55.0,
        "vol_gate_mult": 1.3,
        "target_range_mult": 3.5,
        "trail_min_profit_r": 1.0,
        "max_hold_minutes": 120,
    },
    "trend_rider_pro": {
        "confluence_threshold": 55.0,
        "min_trend_alignment": 0.75,
        "pullback_threshold": 0.003,
        "stop_atr_mult": 2.0,
        "target_atr_mult": 3.5,
        "trail_min_profit_r": 0.4,
        "max_hold_minutes": 120,
    },
}

DEFAULT_PRESET_START = "2024-11-01"
DEFAULT_PRESET_END = "2024-12-31"


def load_params_from_autoresearch_json(path: Path) -> dict:
    """Extract per-param means from an autoresearch latest.json.

    Returns the consensus params as a flat dict, or {} if the file lacks
    the param_stability section.
    """
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"WARN failed to read params JSON {path}: {e}", file=sys.stderr)
        return {}
    ps = data.get("param_stability", {}) or {}
    out: dict[str, float | int] = {}
    for pname, stats in ps.items():
        mean = stats.get("mean")
        if mean is None:
            continue
        # Round int params back to int if the parameter name suggests it.
        # (Optuna float means may need rounding for params like min_bars.)
        if (
            isinstance(mean, float)
            and mean.is_integer()
            and any(k in pname for k in ("bars", "period", "minutes", "lookback"))
        ):
            out[pname] = int(round(mean))
        else:
            out[pname] = mean
    return out


def compute_spy_benchmark(start: str, end: str, data_dir: str) -> tuple[float, float, str]:
    """Compute SPY buy-and-hold return for the holdout window.

    Returns (spy_return_pct, ratio_input_helper, mode).
    Mode is one of bull_ratio / bear_alpha / flat_absolute / unavailable.
    """
    spy_path = Path(data_dir) / "SPY_1Min.parquet"
    if not spy_path.exists():
        return float("nan"), float("nan"), "unavailable"
    try:
        spy = pd.read_parquet(spy_path, columns=["timestamp", "close"])
        spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True)
        seg = spy[
            (spy["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (spy["timestamp"] <= pd.Timestamp(end, tz="UTC"))
        ]
        if len(seg) < 2:
            return float("nan"), float("nan"), "unavailable"
        spy_growth = float(seg.iloc[-1]["close"]) / float(seg.iloc[0]["close"])
        spy_return_pct = (spy_growth - 1.0) * 100.0
        return spy_return_pct, float("nan"), "computed"
    except Exception as e:
        print(f"WARN spy benchmark failed: {e}", file=sys.stderr)
        return float("nan"), float("nan"), "unavailable"


def compute_ratio_vs_spy(strategy_pct: float, spy_pct: float) -> tuple[float, str]:
    """Match the sign-safe ratio modes used in cerberus_autoresearch.py."""
    if not (spy_pct == spy_pct):  # NaN
        return float("nan"), "unavailable"
    if spy_pct > 1.0:
        return strategy_pct / spy_pct, "bull_ratio"
    if spy_pct < -1.0:
        alpha = strategy_pct - spy_pct
        return 1.0 + alpha / abs(spy_pct), "bear_alpha"
    return strategy_pct, "flat_absolute"


def compute_after_tax_ratio(strategy_pct: float, spy_pct: float, mode: str) -> tuple[float, str]:
    """Mirror of the after-tax math in cerberus_autoresearch.py."""
    if not (strategy_pct == strategy_pct) or mode == "unavailable":
        return float("nan"), "n/a"
    at_strat = strategy_pct * (1.0 - ST_TAX)
    if mode == "bull_ratio":
        at_spy = spy_pct * (1.0 - LT_TAX)
        return (at_strat / at_spy if at_spy != 0 else float("nan")), "bull"
    if mode == "bear_alpha":
        at_alpha = (strategy_pct - spy_pct) * (1.0 - ST_TAX)
        return 1.0 + at_alpha / abs(spy_pct), "bear"
    return at_strat, "flat"


def run_one_window(
    strategy_name: str,
    start: str,
    end: str,
    params: dict,
    config_path: str,
    initial_capital: float = 100_000.0,
) -> dict:
    """Run a single backtest window and return a metrics + benchmark summary."""
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    cfg = _apply_params_to_config(base_config, strategy_name, params) if params else base_config

    # Disable all other strategies — holdout is for one-strategy-at-a-time validation.
    for sname in cfg.get("strategies", {}):
        cfg["strategies"][sname]["enabled"] = sname == strategy_name
    if strategy_name not in cfg.get("strategies", {}):
        cfg.setdefault("strategies", {})[strategy_name] = {"enabled": True}

    metrics = run_backtest_for_optimization(
        start_date=start,
        end_date=end,
        config=cfg,
        data_dir="",
        config_path=config_path,
    )

    n_trades = int(metrics.get("n_trades", 0) or 0)
    net_pnl = float(metrics.get("net_pnl", 0.0) or 0.0)
    strategy_pct = (net_pnl / initial_capital) * 100.0

    data_dir = base_config.get("data_dir", "data/bars_2023_2025")
    spy_pct, _, spy_mode = compute_spy_benchmark(start, end, data_dir)
    if spy_mode == "computed":
        ratio, mode = compute_ratio_vs_spy(strategy_pct, spy_pct)
        after_tax_ratio, after_tax_mode = compute_after_tax_ratio(strategy_pct, spy_pct, mode)
    else:
        ratio, mode = float("nan"), "unavailable"
        after_tax_ratio, after_tax_mode = float("nan"), "n/a"

    return {
        "strategy": strategy_name,
        "n_trades": n_trades,
        "net_pnl": net_pnl,
        "strategy_return_pct": strategy_pct,
        "spy_return_pct": spy_pct,
        "ratio_vs_spy": ratio,
        "mode": mode,
        "after_tax_ratio": after_tax_ratio,
        "after_tax_mode": after_tax_mode,
        "metrics": metrics,
        "params": params,
        "start": start,
        "end": end,
    }


def emit_result_line(result: dict) -> None:
    """Single parseable line for the autoresearch driver to grep."""
    print(
        f"HOLDOUT_RESULT strategy={result['strategy']} "
        f"n_trades={result['n_trades']} "
        f"net_pnl={result['net_pnl']:.2f} "
        f"strategy_return_pct={result['strategy_return_pct']:.2f} "
        f"spy_return_pct={result['spy_return_pct']:.2f} "
        f"ratio_vs_spy={result['ratio_vs_spy']:.2f} "
        f"mode={result['mode']} "
        f"after_tax_ratio={result['after_tax_ratio']:.2f} "
        f"after_tax_mode={result['after_tax_mode']} "
        f"start={result['start']} end={result['end']}"
    )


def print_human_summary(result: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"Holdout: {result['strategy']}  ({result['start']} → {result['end']})")
    print(f"{'=' * 60}")
    if result["params"]:
        print(f"Params ({len(result['params'])}): {result['params']}")
    print(f"  trades            : {result['n_trades']}")
    print(f"  net_pnl           : ${result['net_pnl']:.2f}")
    print(f"  strategy_return   : {result['strategy_return_pct']:.2f}%")
    print(f"  spy_return        : {result['spy_return_pct']:.2f}%")
    print(f"  ratio_vs_spy      : {result['ratio_vs_spy']:.2f}  ({result['mode']})")
    print(f"  after_tax_ratio   : {result['after_tax_ratio']:.2f}  ({result['after_tax_mode']})")
    extra_keys = ["winrate", "profit_factor", "sharpe_ratio", "max_drawdown_pct"]
    for k in extra_keys:
        v = result["metrics"].get(k)
        if isinstance(v, (int, float)) and v == v:
            print(f"  {k:18s}: {v:.3f}" if isinstance(v, float) else f"  {k:18s}: {v}")


def save_artifact(result: dict) -> Path:
    out_dir = Path("artifacts/autoresearch/holdout")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result['strategy']}_holdout_latest.json"
    serializable = {k: v for k, v in result.items() if k != "metrics"}
    serializable["metrics_summary"] = {
        k: v for k, v in result["metrics"].items() if isinstance(v, (int, float, str, bool, type(None)))
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout validation for Cerberus strategies")
    parser.add_argument("--strategy", type=str, default=None, help="Strategy name (default: all preset strategies)")
    parser.add_argument("--start", type=str, default=None, help="Holdout start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="Holdout end date YYYY-MM-DD")
    parser.add_argument(
        "--params-from",
        type=str,
        default=None,
        help="Path to JSON file with param_stability section (e.g., artifacts/autoresearch/<name>_latest.json)",
    )
    parser.add_argument("--config", type=str, default="config/backtest_v2.yaml")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only emit HOLDOUT_RESULT line + minimal stdout (for driver consumption)",
    )
    args = parser.parse_args()

    # Driver mode: explicit window + single strategy.
    if args.start and args.end and args.strategy:
        if args.params_from:
            params = load_params_from_autoresearch_json(Path(args.params_from))
        else:
            params = HOLDOUT_PARAMS.get(args.strategy, {})
        result = run_one_window(args.strategy, args.start, args.end, params, args.config)
        save_artifact(result)
        emit_result_line(result)
        if not args.quiet:
            print_human_summary(result)
        return

    # Preset mode: legacy behavior.
    strategies = [args.strategy] if args.strategy else list(HOLDOUT_PARAMS.keys())
    results = []
    for strat in strategies:
        params = HOLDOUT_PARAMS.get(strat)
        if params is None:
            print(f"No preset HOLDOUT_PARAMS for {strat} — pass --params-from to override", file=sys.stderr)
            continue
        result = run_one_window(strat, DEFAULT_PRESET_START, DEFAULT_PRESET_END, params, args.config)
        save_artifact(result)
        emit_result_line(result)
        print_human_summary(result)
        results.append(result)

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("HOLDOUT SUMMARY")
        print(f"{'Strategy':>25} {'Trades':>7} {'Ratio':>8} {'AfterTax':>9}")
        print("-" * 55)
        for r in results:
            print(f"  {r['strategy']:>23} {r['n_trades']:>7} {r['ratio_vs_spy']:>8.2f} {r['after_tax_ratio']:>9.2f}")


if __name__ == "__main__":
    main()

"""Holdout validation: run a final backtest on the reserved holdout period
(Nov-Dec 2024) using WFO-selected consensus params for each strategy.

Usage:
    python scripts/run_holdout.py                     # all strategies
    python scripts/run_holdout.py --strategy orb_v2  # single strategy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.analytics.optuna_harness import _apply_params_to_config, run_backtest_for_optimization

# -----------------------------------------------------------------------
# WFO-selected consensus params per strategy
# (Update these after each WFO run; use best OOS window or median-stable params)
# -----------------------------------------------------------------------
HOLDOUT_PARAMS: dict[str, dict] = {
    "orb_v2": {
        # W7 params — best OOS window (Sep test, Sharpe 1.64, PF 2.38, WR 61%)
        "confluence_threshold": 55.0,
        "vol_gate_mult": 1.3,
        "target_range_mult": 3.5,
        "trail_min_profit_r": 1.0,
        "max_hold_minutes": 120,
    },
    "trend_rider_pro": {
        # W7 params — best OOS window (OOS=3.15, PnL=+$31, Sharpe 1.99)
        "confluence_threshold": 55.0,
        "min_trend_alignment": 0.75,
        "pullback_threshold": 0.003,
        "stop_atr_mult": 2.0,
        "target_atr_mult": 3.5,
        "trail_min_profit_r": 0.4,
        "max_hold_minutes": 120,
    },
}

HOLDOUT_START = "2024-11-01"
HOLDOUT_END = "2024-12-31"


def run_holdout(strategy_name: str, config_path: str = "config/backtest_v2.yaml") -> dict:
    params = HOLDOUT_PARAMS.get(strategy_name)
    if not params:
        print(f"No holdout params defined for {strategy_name}", file=sys.stderr)
        return {}

    import yaml
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    cfg = _apply_params_to_config(base_config, strategy_name, params)

    # Disable all other strategies
    for sname in cfg.get("strategies", {}):
        if sname != strategy_name:
            cfg["strategies"][sname]["enabled"] = False

    print(f"\n{'='*60}")
    print(f"Holdout Validation: {strategy_name}")
    print(f"Period: {HOLDOUT_START} → {HOLDOUT_END}")
    print(f"Params: {params}")
    print(f"{'='*60}")

    metrics = run_backtest_for_optimization(
        start_date=HOLDOUT_START,
        end_date=HOLDOUT_END,
        config=cfg,
        data_dir="",
        config_path=config_path,
    )

    display_keys = [
        "n_trades", "winrate", "net_pnl", "profit_factor",
        "sharpe_ratio", "trade_sharpe_ratio", "calmar_ratio",
        "max_drawdown_pct", "avg_pnl", "avg_hold_minutes", "total_return_pct",
    ]
    print("\nResults:")
    for k in display_keys:
        v = metrics.get(k, "N/A")
        if isinstance(v, float):
            print(f"  {k:32s}: {v:.3f}")
        else:
            print(f"  {k:32s}: {v}")

    out_path = Path("artifacts/optimization") / f"{strategy_name}_holdout_results.json"
    with open(out_path, "w") as f:
        json.dump({"strategy": strategy_name, "params": params, "metrics": metrics}, f, indent=2)
    print(f"\nSaved → {out_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run holdout validation for WFO strategies")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Strategy to validate (default: all with params defined)")
    parser.add_argument("--config", type=str, default="config/backtest_v2.yaml")
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else list(HOLDOUT_PARAMS.keys())

    results = {}
    for strat in strategies:
        m = run_holdout(strat, config_path=args.config)
        results[strat] = m

    if len(results) > 1:
        print("\n" + "="*60)
        print("HOLDOUT SUMMARY")
        print(f"{'Strategy':>25} {'Trades':>7} {'PnL':>10} {'Sharpe':>8} {'PF':>6} {'WR':>6}")
        print("-"*65)
        for strat, m in results.items():
            nt = m.get("n_trades", 0)
            pnl = m.get("net_pnl", 0)
            sr = m.get("sharpe_ratio", 0)
            pf = m.get("profit_factor", 0)
            wr = m.get("winrate", 0) * 100
            print(f"  {strat:>23} {nt:>7} {pnl:>10.0f} {sr:>8.2f} {pf:>6.2f} {wr:>5.1f}%")


if __name__ == "__main__":
    main()

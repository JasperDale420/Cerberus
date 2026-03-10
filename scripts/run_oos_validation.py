#!/usr/bin/env python3
"""Out-of-sample validation: run optimized params on H2 2024 (unseen data).

Tests both trend_rider_pro and mean_reversion_pro with their best params
from the H1 2024 Optuna optimization, on Jul-Dec 2024 data.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from src.analytics.optuna_harness import (
    _apply_params_to_config,
    composite_objective,
    run_backtest_for_optimization,
)
from src.core.config import ConfigLoader


def run_validation(
    strategy_name: str,
    params: dict,
    start_date: str,
    end_date: str,
    symbols: list[str],
    label: str,
):
    """Run a single validation backtest."""
    loader = ConfigLoader()
    cfg = loader.load_config("config/backtest_v2.yaml")
    cfg["universe"] = {"symbols": symbols}

    # Apply optimized params
    test_cfg = _apply_params_to_config(cfg, strategy_name, params)

    # Disable other strategies
    for s in list(test_cfg.get("strategies", {}).keys()):
        if s != strategy_name:
            test_cfg["strategies"][s]["enabled"] = False

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  Params: {json.dumps(params, indent=4)}")
    print(f"{'='*60}")

    start = time.time()
    metrics = run_backtest_for_optimization(
        start_date=start_date,
        end_date=end_date,
        config=test_cfg,
        data_dir="data/bars_2024",
        config_path="config/backtest_v2.yaml",
    )
    elapsed = time.time() - start
    score = composite_objective(metrics)

    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  Composite Score: {score:.4f}")
    print(f"  Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Profit Factor:   {metrics.get('profit_factor', 0):.2f}")
    print(f"  Win Rate:        {metrics.get('winrate', 0):.2%}")
    print(f"  Max Drawdown:    {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Trades:          {metrics.get('n_trades', 0)}")
    print(f"  Net PnL:         ${metrics.get('net_pnl', 0):,.2f}")
    print(f"  Avg PnL/trade:   ${metrics.get('avg_pnl', 0):.2f}")
    print(f"  Calmar Ratio:    {metrics.get('calmar_ratio', 0):.4f}")
    print(f"  Final Equity:    ${metrics.get('final_equity', 100000):,.2f}")

    return {"strategy": strategy_name, "label": label, "score": score, "metrics": metrics}


def main():
    symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "AMZN", "META"]

    # Optimized params from H1 2024
    trp_params = {
        "confluence_threshold": 65.0,
        "min_trend_alignment": 0.5,
        "pullback_threshold": 0.006,
        "stop_atr_mult": 1.25,
        "target_atr_mult": 10.0,
        "trail_min_profit_r": 1.3,
        "max_hold_minutes": 165,
    }

    mrp_params = {
        "confluence_threshold": 65.0,
        "vwap_dist_threshold": 0.001,
        "bb_pos_threshold": 0.6,
        "max_hold_minutes": 50,
        "stop_atr_mult": 2.0,
    }

    results = []

    # 1. In-sample replay (sanity check — should match optimization)
    r1 = run_validation(
        "trend_rider_pro", trp_params,
        "2024-01-02", "2024-06-30", symbols,
        "TRP In-Sample (H1 2024) — Sanity Check",
    )
    results.append(r1)

    # 2. Out-of-sample (the real test)
    r2 = run_validation(
        "trend_rider_pro", trp_params,
        "2024-07-01", "2024-12-31", symbols,
        "TRP Out-of-Sample (H2 2024) — VALIDATION",
    )
    results.append(r2)

    # 3. MRP in-sample
    r3 = run_validation(
        "mean_reversion_pro", mrp_params,
        "2024-01-02", "2024-06-30", symbols,
        "MRP In-Sample (H1 2024) — Sanity Check",
    )
    results.append(r3)

    # 4. MRP out-of-sample
    r4 = run_validation(
        "mean_reversion_pro", mrp_params,
        "2024-07-01", "2024-12-31", symbols,
        "MRP Out-of-Sample (H2 2024) — VALIDATION",
    )
    results.append(r4)

    # Summary
    print(f"\n{'='*70}")
    print("  WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Label':<45s} {'Score':>8s} {'Sharpe':>8s} {'PF':>6s} {'WR':>6s} {'DD':>6s} {'Trades':>7s}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for r in results:
        m = r["metrics"]
        print(
            f"  {r['label']:<45s} "
            f"{r['score']:>8.3f} "
            f"{m.get('sharpe_ratio', 0):>8.3f} "
            f"{m.get('profit_factor', 0):>6.2f} "
            f"{m.get('winrate', 0):>5.1%} "
            f"{m.get('max_drawdown_pct', 0):>5.1f}% "
            f"{m.get('n_trades', 0):>7d}"
        )

    # WFO efficiency ratio
    for strat in ["trend_rider_pro", "mean_reversion_pro"]:
        strat_results = [r for r in results if r["strategy"] == strat]
        if len(strat_results) >= 2:
            is_score = strat_results[0]["score"]
            oos_score = strat_results[1]["score"]
            eff = oos_score / is_score if is_score > 0 else 0
            print(f"\n  {strat} WFO Efficiency: {eff:.4f} (target > 0.5)")

    # Save results
    out = {
        "validation_type": "H1_train_H2_test",
        "results": [
            {
                "strategy": r["strategy"],
                "label": r["label"],
                "score": r["score"],
                "sharpe": r["metrics"].get("sharpe_ratio", 0),
                "profit_factor": r["metrics"].get("profit_factor", 0),
                "winrate": r["metrics"].get("winrate", 0),
                "max_dd_pct": r["metrics"].get("max_drawdown_pct", 0),
                "n_trades": r["metrics"].get("n_trades", 0),
                "net_pnl": r["metrics"].get("net_pnl", 0),
            }
            for r in results
        ],
    }
    os.makedirs("artifacts/optimization", exist_ok=True)
    with open("artifacts/optimization/oos_validation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Results saved to: artifacts/optimization/oos_validation_results.json")


if __name__ == "__main__":
    main()

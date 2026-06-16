#!/usr/bin/env python
"""Quick parameter sweep — test targeted param changes to push PF > 1.0."""

import asyncio
import json
import logging
import os
import sys
import warnings

import yaml

os.chdir("/Users/jacobmcmillan/Empire/Cerberus")
sys.path.insert(0, ".")

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")
os.environ["EMPIRE_LOG_LEVEL"] = "WARNING"
os.environ.setdefault("EMPIRE_LOG_FORMAT", "json")

from src.backtest.runner import run_backtest  # noqa: E402

# Param variants to test (each is a dict of overrides to backtest_v2.yaml strategies section)
VARIANTS = [
    {
        "name": "baseline_zero_comm",
        "desc": "Current params, zero commission",
        "overrides": {},
        "commission": 0.0,
    },
    {
        "name": "tighter_trp_stop",
        "desc": "TRP stop_atr_mult 2.08→1.5, zero comm",
        "overrides": {"trend_rider_pro": {"stop_atr_mult": 1.5}},
        "commission": 0.0,
    },
    {
        "name": "selective_trp",
        "desc": "TRP confluence 63.3→72, zero comm",
        "overrides": {"trend_rider_pro": {"confluence_threshold": 72.0}},
        "commission": 0.0,
    },
    {
        "name": "selective_trp_tight_stop",
        "desc": "TRP confluence 72 + stop_atr 1.5, zero comm",
        "overrides": {"trend_rider_pro": {"confluence_threshold": 72.0, "stop_atr_mult": 1.5}},
        "commission": 0.0,
    },
    {
        "name": "orb_early_trail",
        "desc": "ORB trail_min_profit_r 0.96→0.5, zero comm",
        "overrides": {"orb_v2": {"trail_min_profit_r": 0.5}},
        "commission": 0.0,
    },
    {
        "name": "combined_v1",
        "desc": "TRP conf=72 stop=1.5 + ORB trail=0.5, zero comm",
        "overrides": {
            "trend_rider_pro": {"confluence_threshold": 72.0, "stop_atr_mult": 1.5},
            "orb_v2": {"trail_min_profit_r": 0.5},
        },
        "commission": 0.0,
    },
    {
        "name": "trp_wider_target",
        "desc": "TRP target_atr_mult 4.5→5.5, zero comm",
        "overrides": {"trend_rider_pro": {"target_atr_mult": 5.5}},
        "commission": 0.0,
    },
    {
        "name": "trp_lower_trail",
        "desc": "TRP trail_min_profit_r 0.60→0.30, zero comm",
        "overrides": {"trend_rider_pro": {"trail_min_profit_r": 0.30}},
        "commission": 0.0,
    },
    {
        "name": "best_combo_with_comm",
        "desc": "Best combo (TBD) WITH commissions ($0.003/sh)",
        "overrides": {
            "trend_rider_pro": {"confluence_threshold": 72.0, "stop_atr_mult": 1.5},
            "orb_v2": {"trail_min_profit_r": 0.5},
        },
        "commission": 0.003,
    },
]


async def run_variant(variant):
    """Run a single backtest variant."""
    with open("config/backtest_v2.yaml") as f:
        cfg = yaml.safe_load(f)

    # Apply commission override
    cfg["risk"]["commission_per_share"] = variant["commission"]
    cfg["risk"]["min_commission"] = 0.50 if variant["commission"] > 0 else 0.0

    # Apply strategy param overrides
    for strat_name, params in variant.get("overrides", {}).items():
        if strat_name in cfg.get("strategies", {}):
            cfg["strategies"][strat_name].update(params)

    # Write temp config
    tmp_path = f"config/_sweep_{variant['name']}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    try:
        result = await run_backtest(
            start_date="2024-01-01",
            end_date="2024-12-31",
            config_path=tmp_path,
            data_dir="data/bars_2024",
        )
        m = result.metrics
        strats = result.strategy_metrics or {}

        return {
            "name": variant["name"],
            "desc": variant["desc"],
            "net_pnl": round(m.net_pnl, 2),
            "pf": round(m.profit_factor, 2),
            "sharpe": round(m.sharpe_ratio, 3),
            "winrate": round(m.winrate, 3),
            "n_trades": m.n_trades,
            "max_dd_pct": round(m.max_drawdown_pct, 2),
            "avg_hold_min": round(m.avg_hold_seconds / 60, 1),
            "per_strategy": {
                name: {
                    "trades": sm.get("n_trades", 0),
                    "pnl": round(sm.get("net_pnl", 0), 2),
                    "pf": round(sm.get("profit_factor", 0), 2),
                    "wr": round(sm.get("winrate", 0), 3),
                }
                for name, sm in strats.items()
            },
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def main():
    print(f"Parameter Sweep — {len(VARIANTS)} variants", flush=True)
    print(f"{'=' * 80}", flush=True)

    results = []
    for i, v in enumerate(VARIANTS):
        print(f"\n[{i + 1}/{len(VARIANTS)}] {v['name']}: {v['desc']}", flush=True)
        try:
            r = await run_variant(v)
            results.append(r)
            print(
                f"  → PnL=${r['net_pnl']:>8,.2f}  PF={r['pf']:.2f}  "
                f"Sharpe={r['sharpe']:.3f}  Trades={r['n_trades']}  "
                f"DD={r['max_dd_pct']:.2f}%",
                flush=True,
            )
            for sname, sm in r["per_strategy"].items():
                print(
                    f"     {sname}: PnL=${sm['pnl']:>8,.2f} PF={sm['pf']:.2f} WR={sm['wr']:.1%} T={sm['trades']}",
                    flush=True,
                )
        except Exception as e:
            print(f"  → FAILED: {e}", flush=True)
            results.append({"name": v["name"], "error": str(e)})

    # Summary table
    print(f"\n{'=' * 80}", flush=True)
    print("SWEEP RESULTS SUMMARY", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"{'Name':<30} {'PnL':>10} {'PF':>6} {'Sharpe':>8} {'Trades':>7} {'DD%':>6}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} {'ERROR':>10}", flush=True)
        else:
            print(
                f"{r['name']:<30} ${r['net_pnl']:>8,.2f} {r['pf']:>6.2f} "
                f"{r['sharpe']:>8.3f} {r['n_trades']:>7} {r['max_dd_pct']:>5.2f}%",
                flush=True,
            )

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/param_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to results/param_sweep_results.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

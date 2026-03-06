#!/usr/bin/env python3
"""Optuna-based strategy parameter optimization for Cerberus V2.

Usage::

    # Single-strategy optimization (quick 50-trial sweep on first half of 2024):
    PYTHONPATH=. python scripts/optimize_strategy.py \
        --strategy trend_rider_pro \
        --start 2024-01-02 --end 2024-06-30 \
        --trials 50

    # Walk-forward optimization (full anchored WFO):
    PYTHONPATH=. python scripts/optimize_strategy.py \
        --strategy trend_rider_pro \
        --wfo \
        --start 2024-01-02 --end 2024-12-31 \
        --trials 40

    # List available parameter spaces:
    PYTHONPATH=. python scripts/optimize_strategy.py --list-spaces

    # Resume a previous study:
    PYTHONPATH=. python scripts/optimize_strategy.py \
        --strategy trend_rider_pro --trials 25 \
        --study-name trend_rider_pro_opt

Parallel execution: run this script in multiple terminals simultaneously —
Optuna coordinates via the shared SQLite study database.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root on sys.path
sys.path.insert(0, os.getcwd())


def main() -> None:
    parser = argparse.ArgumentParser(description="Cerberus strategy optimizer")
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy to optimize (trend_rider_pro, mean_reversion_pro, flow_alpha)",
    )
    parser.add_argument("--start", type=str, default="2024-01-02", help="Start date")
    parser.add_argument("--end", type=str, default="2024-06-30", help="End date")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--config", type=str, default="config/backtest_v2.yaml", help="Base config file")
    parser.add_argument("--data-dir", type=str, default="data/bars_2024", help="Bar data directory")
    parser.add_argument("--wfo", action="store_true", help="Run walk-forward optimization")
    parser.add_argument(
        "--wfo-mode",
        type=str,
        choices=["anchored", "rolling"],
        default="rolling",
        help="WFO mode: rolling (fixed-width sliding, default) or anchored (expanding)",
    )
    parser.add_argument("--wfo-train-months", type=int, default=3, help="Minimum WFO training months")
    parser.add_argument("--wfo-test-months", type=int, default=1, help="WFO test window months")
    parser.add_argument("--wfo-holdout-months", type=int, default=2, help="Final holdout months")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name (for resume)")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional run tag used to isolate WFO artifacts and studies",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing study/run instead of starting clean",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override WFO worker process count",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols for optimization universe (default: top 8 liquid)",
    )
    parser.add_argument("--list-spaces", action="store_true", help="List parameter spaces and exit")
    parser.add_argument("--importance", action="store_true", help="Show parameter importance from existing study")
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Run optimization for all strategies with defined param spaces",
    )
    parser.add_argument(
        "--bar-resolution",
        type=int,
        default=1,
        help="Bar resolution in minutes (1=default, 5=5x faster for non-ORB strategies)",
    )
    args = parser.parse_args()

    # ----- List spaces -----
    if args.list_spaces:
        from src.analytics.param_spaces import PARAM_SPACES

        for name, space in PARAM_SPACES.items():
            print(f"\n{name}:")
            for p in space:
                suffix = f"  (step={p.step})" if p.step else ""
                print(f"  {p.name:<25s} {p.param_type:>5s}  [{p.low}, {p.high}]{suffix}")
                if p.description:
                    print(f"    {p.description}")
        return

    # ----- Validate -----
    from src.analytics.param_spaces import PARAM_SPACES

    if args.all_strategies:
        strategies_to_run = list(PARAM_SPACES.keys())
        print(f"Running optimization for all strategies: {strategies_to_run}")
    elif not args.strategy:
        parser.error("--strategy is required (or use --list-spaces / --all-strategies)")
    elif args.strategy not in PARAM_SPACES:
        parser.error(f"Unknown strategy '{args.strategy}'. Available: {', '.join(PARAM_SPACES.keys())}")
    else:
        strategies_to_run = [args.strategy]

    # ----- Load config -----
    from src.core.config import ConfigLoader

    config_loader = ConfigLoader()
    base_config = config_loader.load_config(args.config)

    # Override universe for faster optimization
    default_opt_symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "AMZN", "META"]
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = default_opt_symbols
    base_config.setdefault("universe", {})["symbols"] = symbols
    print(f"  Universe: {symbols} ({len(symbols)} symbols)")

    # Inject bar resolution for backtest speedup
    if args.bar_resolution > 1:
        base_config.setdefault("backtest", {})["bar_resolution_minutes"] = args.bar_resolution
        print(f"  Bar resolution: {args.bar_resolution}m ({60 // args.bar_resolution}x fewer bars)")

    # Ensure artifacts dir
    outdir = Path("artifacts/optimization")
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- Run for each strategy -----
    for strat_name in strategies_to_run:
        study_name = args.study_name or f"{strat_name}_opt"
        storage = f"sqlite:///{outdir}/{strat_name}_study.db"

        if len(strategies_to_run) > 1:
            print(f"\n{'#' * 60}")
            print(f"# Strategy: {strat_name}")
            print(f"{'#' * 60}")

        # ----- Walk-Forward mode -----
        if args.wfo:
            from src.analytics.optuna_harness import WalkForwardOptimizer, build_wfo_run_context

            wfo = WalkForwardOptimizer(
                full_start=args.start,
                full_end=args.end,
                min_train_months=args.wfo_train_months,
                test_months=args.wfo_test_months,
                n_trials=args.trials,
                holdout_months=args.wfo_holdout_months,
                mode=args.wfo_mode,
            )
            run_context = build_wfo_run_context(
                strategy_name=strat_name,
                full_start=args.start,
                full_end=args.end,
                min_train_months=args.wfo_train_months,
                test_months=args.wfo_test_months,
                holdout_months=args.wfo_holdout_months,
                n_trials=args.trials,
                mode=args.wfo_mode,
                base_config=base_config,
                data_dir=args.data_dir,
                run_tag=args.run_tag,
            )

            print(f"Starting walk-forward optimization for {strat_name}")
            print(f"  Mode: {args.wfo_mode}")
            print(f"  Data: {args.start} → {args.end}")
            print(f"  Trials per window: {args.trials}")
            print(f"  Windows: {len(wfo.get_windows())}")
            print(f"  Holdout: {wfo.get_holdout_window()}")
            print(f"  Run tag: {run_context.run_tag}")
            print(f"  Study DB: {run_context.storage_uri}")
            print(f"  Workers: {args.workers if args.workers is not None else 'auto'}")
            print(f"  Resume existing: {args.resume}")

            result = wfo.run(
                strategy_name=strat_name,
                base_config=base_config,
                data_dir=args.data_dir,
                config_path=args.config,
                study_storage=run_context.storage_uri,
                run_tag=run_context.run_tag,
                resume_existing=args.resume,
                workers=args.workers,
            )

            # Print summary
            print(f"\n{'=' * 60}")
            print(f"Walk-Forward Results — {strat_name}")
            print(f"{'=' * 60}")
            print(f"  WFO Efficiency Ratio: {result['wfo_efficiency_ratio']:.4f}")
            print(f"  IS scores:  {[round(s, 3) for s in result['is_scores']]}")
            print(f"  OOS scores: {[round(s, 3) for s in result['oos_scores']]}")

            if result.get("param_stability"):
                print("\n  Parameter Stability:")
                for pname, stats in result["param_stability"].items():
                    stable_marker = "✓" if stats["stable"] else "✗ UNSTABLE"
                    print(
                        f"    {pname:<25s} mean={stats['mean']:.4f}  "
                        f"std={stats['std']:.4f}  cv={stats['cv']:.4f}  {stable_marker}"
                    )

            print(f"\n  Results saved to: {result['artifact_dir']}/wfo_results.json")
            continue

        # ----- Single optimization -----
        import optuna

        from src.analytics.optuna_harness import create_objective

        print(f"Starting optimization for {strat_name}")
        print(f"  Period: {args.start} → {args.end}")
        print(f"  Trials: {args.trials}")
        print(f"  Study: {study_name} ({storage})")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=args.resume,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        objective = create_objective(
            strategy_name=strat_name,
            base_config=base_config,
            start_date=args.start,
            end_date=args.end,
            data_dir=args.data_dir,
            config_path=args.config,
        )

        study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

        # ----- Results -----
        print(f"\n{'=' * 60}")
        print(f"Optimization Complete — {strat_name}")
        print(f"{'=' * 60}")
        print(f"  Best score: {study.best_value:.4f}")
        print(f"  Best params: {json.dumps(study.best_params, indent=4)}")

        # User attrs from best trial
        best = study.best_trial
        print(f"  Sharpe: {best.user_attrs.get('sharpe', 'N/A')}")
        print(f"  Profit Factor: {best.user_attrs.get('profit_factor', 'N/A')}")
        print(f"  Win Rate: {best.user_attrs.get('winrate', 'N/A')}")
        print(f"  Net PnL: {best.user_attrs.get('net_pnl', 'N/A')}")
        print(f"  Max DD%: {best.user_attrs.get('max_dd_pct', 'N/A')}")
        print(f"  Trades: {best.user_attrs.get('n_trades', 'N/A')}")

        # Parameter importance
        if args.importance or len(study.trials) >= 10:
            try:
                importances = optuna.importance.get_param_importances(study)
                print("\n  Parameter Importance:")
                for pname, imp in sorted(importances.items(), key=lambda x: -x[1]):
                    bar = "█" * int(imp * 40)
                    print(f"    {pname:<25s} {imp:.4f}  {bar}")
            except Exception:
                pass

        # Save best params
        result_path = outdir / f"{strat_name}_best_params.json"
        with open(result_path, "w") as f:
            json.dump(
                {
                    "strategy": strat_name,
                    "period": f"{args.start} to {args.end}",
                    "best_score": study.best_value,
                    "best_params": study.best_params,
                    "n_trials": len(study.trials),
                    "user_attrs": dict(best.user_attrs),
                },
                f,
                indent=2,
            )
        print(f"\n  Best params saved to: {result_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
from typing import Any, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(os.getcwd())

from src.core.logger import StructuredLogger

logger = StructuredLogger("AlphaAudit")


def load_bars_to_df(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame()

    rows = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Map Alpaca-style short names if present
                time_val = data.get("t") or data.get("timestamp") or data.get("time")
                if not time_val:
                    continue
                rows.append(
                    {
                        "time": time_val,
                        "open": float(data.get("o") or data.get("open")),
                        "high": float(data.get("h") or data.get("high")),
                        "low": float(data.get("l") or data.get("low")),
                        "close": float(data.get("c") or data.get("close")),
                        "volume": float(data.get("v") or data.get("volume")),
                        "vwap": float(data.get("vw") or data.get("vwap", 0.0)),
                    }
                )
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    return df


def compute_alpha_factors(df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Vectorized calculation of alpha factors (including Qlib Alpha158 base)"""
    feat = pd.DataFrame(index=df.index)

    # 1. K-Bar Features
    feat["f_kmid"] = (df["close"] - df["open"]) / df["open"]
    feat["f_klen"] = (df["high"] - df["low"]) / df["open"]
    feat["f_kup"] = (df["high"] - np.maximum(df["open"], df["close"])) / df["open"]
    feat["f_klow"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["open"]

    # 2. Rolling Momentum (ROC)
    for w in [5, 10, 20]:
        feat[f"f_roc_{w}"] = df["close"].pct_change(w)

    # 3. Rolling Volatility (STD)
    for w in [5, 10, 20]:
        feat[f"f_std_{w}"] = df["close"].pct_change().rolling(w).std()

    # 4. Moving Average Distances
    for w in [20, 50, 200]:
        feat[f"f_ma_dist_{w}"] = (df["close"] / df["close"].rolling(w).mean()) - 1.0

    # 5. VWAP Distance
    if "vwap" in df.columns:
        feat["f_vwap_dist"] = (df["close"] / df["vwap"]) - 1.0

    # 6. Relative Strength (vs Benchmark)
    if benchmark_df is not None:
        # Align indexes
        common_idx = df.index.intersection(benchmark_df.index)
        if not common_idx.empty:
            sym_ret = df["close"].loc[common_idx].pct_change(20)
            bench_ret = benchmark_df["close"].loc[common_idx].pct_change(20)
            feat["f_relative_strength"] = sym_ret - bench_ret

    # 7. Z-Score (Mean Reversion)
    rolling_mean = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std()
    feat["f_zscore"] = (df["close"] - rolling_mean) / (rolling_std + 1e-9)

    # 8. Intraday Range
    feat["f_intraday_range"] = (df["high"] - df["low"]) / df["open"]

    return feat


def process_symbol_v2(
    symbol: str, data_dir: str, benchmark_df: pd.DataFrame, fwd_windows: List[int]
) -> pd.DataFrame | None:
    path_v1 = os.path.join(data_dir, f"{symbol}.jsonl")
    path_v2 = os.path.join(data_dir, f"{symbol}_1Min.jsonl")
    path = path_v2 if os.path.exists(path_v2) else path_v1

    df = load_bars_to_df(path)
    if df.empty:
        return None

    # Compute Factors
    factors = compute_alpha_factors(df, benchmark_df)

    # Compute Forward Returns
    for w in fwd_windows:
        # Shift close prices backwards to get 'future' price at current timestamp
        future_price = df["close"].shift(-w)
        factors[f"t_ret_{w}m"] = (future_price / df["close"]) - 1.0

    # Subsample to reduce autocorrelation (every 15 mins)
    factors = factors.iloc[::15].dropna()

    # Add metadata
    factors["symbol"] = symbol
    factors["timestamp"] = factors.index.astype(str)

    return factors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/offline_bars_5yr")
    parser.add_argument("--symbols", help="Comma separated symbols")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--limit-symbols", type=int, default=15)
    parser.add_argument("--output", default="results/alpha_factor_audit.json")
    args = parser.parse_args()

    if not os.path.exists("results"):
        os.makedirs("results")

    symbols = args.symbols.split(",") if args.symbols else []
    if not symbols:
        available = [f.split("_")[0].split(".")[0] for f in os.listdir(args.data_dir) if f.endswith(".jsonl")]
        symbols = sorted(list(set(available)))[: args.limit_symbols]

    # Load Benchmark first
    bench_path_v1 = os.path.join(args.data_dir, f"{args.benchmark}.jsonl")
    bench_path_v2 = os.path.join(args.data_dir, f"{args.benchmark}_1Min.jsonl")
    bench_path = bench_path_v2 if os.path.exists(bench_path_v2) else bench_path_v1
    benchmark_df = load_bars_to_df(bench_path)

    if benchmark_df.empty:
        logger.error("Benchmark data missing", benchmark=args.benchmark)
        return

    fwd_windows = [15, 60, 240]
    all_dfs = []

    logger.info("Starting optimized Alpha Factor Audit", symbols=len(symbols))

    for sym in symbols:
        if sym == args.benchmark:
            continue
        logger.info("Processing symbol", symbol=sym)
        res_df = process_symbol_v2(sym, args.data_dir, benchmark_df, fwd_windows)
        if res_df is not None and not res_df.empty:
            all_dfs.append(res_df)

    if not all_dfs:
        logger.error("No audit data generated")
        return

    master_df = pd.concat(all_dfs)
    logger.info("Audit dataset built", rows=len(master_df))

    # Statistical Analysis
    factor_cols = [c for c in master_df.columns if c.startswith("f_")]
    target_cols = [c for c in master_df.columns if c.startswith("t_")]

    audit_results: dict[str, Any] = {
        "summary": {
            "total_samples": len(master_df),
            "symbols": symbols,
            "period": f"{master_df.index.min()} to {master_df.index.max()}",
        },
        "per_factor": {},
    }

    print("\n=== OPTIMIZED FACTOR AUDIT RESULTS (IC) ===")
    print(f"{'Factor':<25} {'Target':<10} {'Spearman IC':>12} {'P-Value':>12}")
    print("-" * 65)

    for f in factor_cols:
        audit_results["per_factor"][f] = {}
        for t in target_cols:
            # Drop NaNs for this specific pair
            valid = master_df[[f, t]].dropna()
            if len(valid) < 100:
                continue
            corr, pval = spearmanr(valid[f], valid[t])
            audit_results["per_factor"][f][t] = {
                "ic": float(corr),
                "p_val": float(pval),
            }
            if abs(corr) > 0.02:  # Log anything with even slight signal
                print(f"{f:<25} {t:<10} {corr:>12.4f} {pval:>12.4e}")

    # Session Analysis
    master_df["hour"] = pd.to_datetime(master_df.index).hour
    master_df["session"] = np.where(master_df["hour"] < 12, "morning", "afternoon")

    audit_results["by_session"] = {}

    for session in ["morning", "afternoon"]:
        audit_results["by_session"][session] = {}
        sdf = master_df[master_df["session"] == session]
        print(f"\n--- SESSION: {session.upper()} (Target: t_ret_60m) ---")
        for f in factor_cols:
            valid = sdf[[f, "t_ret_60m"]].dropna()
            if len(valid) < 50:
                continue
            corr, _ = spearmanr(valid[f], valid["t_ret_60m"])
            audit_results["by_session"][session][f] = float(corr)
            if abs(corr) > 0.02:
                print(f"{f:<25}: {corr:>12.4f}")

    with open(args.output, "w") as output_file:
        json.dump(audit_results, output_file, indent=2)

    logger.info("Audit complete", output=args.output)


if __name__ == "__main__":
    main()

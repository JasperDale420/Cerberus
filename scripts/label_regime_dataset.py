"""Label all available bar data with optimized 2-axis regime (trend + vol).

Reads every symbol's 1-minute parquet from all data directories,
resamples to daily OHLCV, classifies each day's regime using the
optimized rule-based detector (SMA crossover + realized vol), and
writes regime-labeled parquet files.

Optimized params (from grid search against 2020-2025 SPY ground truth):
- Trend: Dual SMA crossover (fast=10, slow=40, flat_band=1%)  → 71.5% accuracy
- Vol: 30-day realized vol (LOW<8%, HIGH>20%, SHOCK>50%)       → 92.3% accuracy
- Combined: 81.9% accuracy (vs 65.4% baseline)

Usage: uv run python scripts/label_regime_dataset.py
"""

import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["EMPIRE_LOG_LEVEL"] = "CRITICAL"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ── Optimized detector params ──────────────────────────────────────

TREND_FAST = 10
TREND_SLOW = 40
TREND_FLAT_BAND = 0.01

VOL_WINDOW = 30
VOL_LOW = 0.08
VOL_HIGH = 0.20
VOL_SHOCK = 0.50

# All data directories to label
DATA_DIRS = [
    "data/bars_2024",
    "data/bars_5yr",
    "data/bars_2023_2025",
]

OUTPUT_DIR = "data/regime_labeled"


def classify_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Classify each day's regime using optimized rule-based detector."""
    closes = daily["close"].values.astype(float)
    n = len(closes)

    # ── Trend: Dual SMA crossover ──
    sma_fast = pd.Series(closes).rolling(TREND_FAST).mean().values
    sma_slow = pd.Series(closes).rolling(TREND_SLOW).mean().values
    trends = []
    for i in range(n):
        if i < TREND_SLOW or np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]):
            trends.append("FLAT")
            continue
        pct_above = (closes[i] - sma_slow[i]) / sma_slow[i]
        fast_above_slow = (sma_fast[i] - sma_slow[i]) / sma_slow[i]
        if pct_above > TREND_FLAT_BAND and fast_above_slow > 0:
            trends.append("UP")
        elif pct_above < -TREND_FLAT_BAND and fast_above_slow < 0:
            trends.append("DOWN")
        else:
            trends.append("FLAT")

    # ── Vol: Realized vol (annualized) ──
    log_rets = np.zeros(n)
    for i in range(1, n):
        log_rets[i] = np.log(closes[i] / closes[i - 1])
    vols = []
    for i in range(n):
        if i < VOL_WINDOW:
            vols.append("NORMAL")
            continue
        rvol = np.std(log_rets[i - VOL_WINDOW + 1 : i + 1]) * np.sqrt(252)
        if rvol >= VOL_SHOCK:
            vols.append("SHOCK")
        elif rvol >= VOL_HIGH:
            vols.append("HIGH")
        elif rvol <= VOL_LOW:
            vols.append("LOW")
        else:
            vols.append("NORMAL")

    daily = daily.copy()
    daily["regime_trend"] = trends
    daily["regime_vol"] = vols
    daily["regime"] = [f"{t}+{v}" for t, v in zip(trends, vols, strict=False)]
    return daily


def process_symbol(parquet_path: Path, output_dir: Path) -> dict:
    """Load a symbol's 1m parquet, resample to daily, classify, save."""
    symbol = parquet_path.stem.replace("_1Min", "")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        return {"symbol": symbol, "status": "error", "error": str(e), "days": 0}

    if len(df) == 0:
        return {"symbol": symbol, "status": "empty", "days": 0}

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date

    # Resample to daily OHLCV
    daily = (
        df.groupby("date")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["symbol"] = symbol

    if len(daily) < TREND_SLOW + 5:
        return {"symbol": symbol, "status": "insufficient_data", "days": len(daily)}

    # Classify
    labeled = classify_daily(daily)

    # Save
    out_path = output_dir / f"{symbol}_daily_regime.parquet"
    labeled.to_parquet(out_path, index=False)

    regime_dist = labeled["regime"].value_counts().to_dict()
    return {
        "symbol": symbol,
        "status": "ok",
        "days": len(labeled),
        "date_range": f"{labeled['date'].iloc[0].date()} → {labeled['date'].iloc[-1].date()}",
        "regimes": regime_dist,
    }


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique parquet files across all data dirs
    all_files: dict[str, Path] = {}  # symbol → path (latest/largest wins)
    for data_dir in DATA_DIRS:
        d = Path(data_dir)
        if not d.exists():
            print(f"  SKIP {data_dir} (not found)")
            continue
        for f in sorted(d.glob("*_1Min.parquet")):
            symbol = f.stem.replace("_1Min", "")
            # Prefer larger file (more data)
            if symbol not in all_files or f.stat().st_size > all_files[symbol].stat().st_size:
                all_files[symbol] = f

    print(f"Found {len(all_files)} unique symbols across {len(DATA_DIRS)} data directories\n")

    results = []
    for i, (symbol, path) in enumerate(sorted(all_files.items())):
        result = process_symbol(path, output_dir)
        results.append(result)
        status = result["status"]
        if status == "ok":
            days = result["days"]
            top_regime = max(result["regimes"], key=result["regimes"].get)
            print(
                f"  [{i + 1:3d}/{len(all_files)}] {symbol:8s} {days:5d} days  {result['date_range']}  top={top_regime}"
            )
        else:
            print(f"  [{i + 1:3d}/{len(all_files)}] {symbol:8s} {status}")

    # Summary
    ok_results = [r for r in results if r["status"] == "ok"]
    total_days = sum(r["days"] for r in ok_results)

    # Aggregate regime distribution
    agg_regimes: dict[str, int] = {}
    for r in ok_results:
        for regime, count in r.get("regimes", {}).items():
            agg_regimes[regime] = agg_regimes.get(regime, 0) + count

    print(f"\n{'=' * 80}")
    print("LABELING COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Symbols labeled: {len(ok_results)}/{len(results)}")
    print(f"  Total daily bars: {total_days:,}")
    print(f"  Output directory: {output_dir}")
    print("\n  Aggregate regime distribution:")
    for regime in sorted(agg_regimes, key=agg_regimes.get, reverse=True):
        pct = agg_regimes[regime] / total_days * 100
        print(f"    {regime:<20} {agg_regimes[regime]:>8,} days ({pct:5.1f}%)")

    # Save summary
    summary_path = output_dir / "labeling_summary.json"
    import json

    with open(summary_path, "w") as f:
        json.dump(
            {
                "params": {
                    "trend": {
                        "method": "sma_crossover",
                        "fast": TREND_FAST,
                        "slow": TREND_SLOW,
                        "flat_band": TREND_FLAT_BAND,
                    },
                    "vol": {
                        "method": "realized_vol",
                        "window": VOL_WINDOW,
                        "low": VOL_LOW,
                        "high": VOL_HIGH,
                        "shock": VOL_SHOCK,
                    },
                    "accuracy": {"trend": 71.5, "vol": 92.3, "combined": 81.9},
                },
                "symbols_labeled": len(ok_results),
                "total_days": total_days,
                "regime_distribution": agg_regimes,
                "per_symbol": [{k: v for k, v in r.items() if k != "regimes"} for r in ok_results],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

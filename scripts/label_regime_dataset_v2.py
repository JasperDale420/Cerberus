"""Two-column regime labeler — v2.

Replaces the single `vol_regime` column with two named columns:
  - `vol_regime_market`   one shared state per date, computed on SPY using
                          Labeler A's absolute-threshold logic
                          (LOW<8% / HIGH>=20% / SHOCK>=50% on 30d realized vol).
                          Semantics: "is the broad market in a vol shock?"
  - `vol_regime_symbol`   per-symbol state, computed using Labeler B's
                          EWMA variance ratio z-score (matches the live
                          MarketContextService logic in src/analysis/regime.py).
                          Semantics: "is this symbol unusually volatile
                          for itself right now?"

Trend axis is also split:
  - `trend_regime_market` SPY trend via dual-SMA crossover (Labeler A)
  - `trend_regime_symbol` per-symbol trend via the same SMA crossover

Output goes to `data/regime_labeled_v2/<SYMBOL>_daily_regime.parquet`.
The original `data/regime_labeled/` is left intact so existing autoresearch
runs remain reproducible until the migration completes.

This script does NOT touch the regime/calendar event columns
(near_earnings, opex_week, fomc_window, etc.) since those are produced
by separate labelers (label_earnings.py, label_macro_events.py).
A migration step downstream re-runs those on the v2 base.

Usage:
    uv run python scripts/label_regime_dataset_v2.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["EMPIRE_LOG_LEVEL"] = "CRITICAL"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ─── Labeler A (market-wide) parameters ────────────────────────────
# Calibrated against SPY ground truth periods 2020-2025 in
# scripts/regime_grid_search.py. Use ONLY on SPY (the benchmark).
TREND_FAST_DAYS = 10
TREND_SLOW_DAYS = 40
TREND_FLAT_BAND = 0.01

VOL_WINDOW_DAYS = 30
VOL_LOW_ABS = 0.08
VOL_HIGH_ABS = 0.20
VOL_SHOCK_ABS = 0.50

# ─── Labeler B (per-symbol relative) parameters ────────────────────
# Mirrors src/analysis/regime.py::MarketContextService._classify_vol exactly.
ALPHA_SHORT = 2.0 / (10 + 1)  # ~10-bar EWMA span
ALPHA_LONG = 2.0 / (120 + 1)  # ~120-bar EWMA span
SHOCK_Z = 3.0
HIGH_Z = 1.5
LOW_Z = 0.7
EWMA_WARMUP_BARS = 120  # discard first N bars from labels until EWMA stabilizes

# ─── I/O paths ─────────────────────────────────────────────────────
DATA_DIRS = [
    Path("data/bars_2023_2025"),
    Path("data/bars_5yr"),
    Path("data/bars_2024"),
]
OUTPUT_DIR = Path("data/regime_labeled_v2")
SPY_INPUT_FILES = [
    Path("data/bars_2023_2025/SPY_1Min.parquet"),
    Path("data/bars_5yr/SPY_1Min.parquet"),
]


def load_daily_from_minute_parquet(parquet_path: Path) -> pd.DataFrame:
    """Resample 1-min bars to daily OHLCV."""
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date
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
    return daily


def classify_market_axes(daily_spy: pd.DataFrame) -> pd.DataFrame:
    """Compute SPY trend + vol using Labeler A's absolute-threshold logic.

    Returns a frame indexed by date with columns
    `trend_regime_market`, `vol_regime_market`.
    Trading days only — no calendar fill.
    """
    closes = daily_spy["close"].values.astype(float)
    n = len(closes)

    # Trend: dual-SMA crossover with flat band
    sma_fast = pd.Series(closes).rolling(TREND_FAST_DAYS).mean().values
    sma_slow = pd.Series(closes).rolling(TREND_SLOW_DAYS).mean().values
    trend = np.full(n, "FLAT", dtype=object)
    for i in range(n):
        if i < TREND_SLOW_DAYS or np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]):
            continue
        pct_above = (closes[i] - sma_slow[i]) / sma_slow[i]
        fast_above_slow = (sma_fast[i] - sma_slow[i]) / sma_slow[i]
        if pct_above > TREND_FLAT_BAND and fast_above_slow > 0:
            trend[i] = "UP"
        elif pct_above < -TREND_FLAT_BAND and fast_above_slow < 0:
            trend[i] = "DOWN"

    # Vol: trailing 30d annualized realized vol with absolute thresholds
    log_rets = np.zeros(n)
    for i in range(1, n):
        log_rets[i] = np.log(closes[i] / closes[i - 1])
    vol = np.full(n, "NORMAL", dtype=object)
    for i in range(n):
        if i < VOL_WINDOW_DAYS:
            continue
        rvol = float(np.std(log_rets[i - VOL_WINDOW_DAYS + 1 : i + 1]) * np.sqrt(252))
        if rvol >= VOL_SHOCK_ABS:
            vol[i] = "SHOCK"
        elif rvol >= VOL_HIGH_ABS:
            vol[i] = "HIGH"
        elif rvol <= VOL_LOW_ABS:
            vol[i] = "LOW"

    return pd.DataFrame(
        {
            "date": daily_spy["date"].values,
            "trend_regime_market": trend,
            "vol_regime_market": vol,
        }
    )


def classify_symbol_axes(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute per-symbol trend + vol using Labeler B's relative-EWMA logic.

    Trend uses the same dual-SMA shape as the market axis but applied
    per-symbol. Vol uses the EWMA variance ratio z-score that matches
    src/analysis/regime.py::MarketContextService._classify_vol.

    The first EWMA_WARMUP_BARS rows are stamped NULL so consumers can
    drop them before computing edge metrics.
    """
    closes = daily["close"].values.astype(float)
    n = len(closes)

    # Trend (same logic as market axis but per-symbol)
    sma_fast = pd.Series(closes).rolling(TREND_FAST_DAYS).mean().values
    sma_slow = pd.Series(closes).rolling(TREND_SLOW_DAYS).mean().values
    trend = np.full(n, "FLAT", dtype=object)
    for i in range(n):
        if i < TREND_SLOW_DAYS or np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]):
            continue
        pct_above = (closes[i] - sma_slow[i]) / sma_slow[i]
        fast_above_slow = (sma_fast[i] - sma_slow[i]) / sma_slow[i]
        if pct_above > TREND_FLAT_BAND and fast_above_slow > 0:
            trend[i] = "UP"
        elif pct_above < -TREND_FLAT_BAND and fast_above_slow < 0:
            trend[i] = "DOWN"

    # Vol: EWMA variance ratio z-score (mirrors live MarketContextService)
    vol = np.full(n, "NORMAL", dtype=object)
    short_var: float | None = None
    long_var: float | None = None
    for i in range(n):
        if i == 0:
            continue
        log_ret = float(np.log(closes[i] / closes[i - 1]))
        sq = log_ret * log_ret
        if short_var is None:
            short_var = sq
            long_var = sq
            continue
        short_var = ALPHA_SHORT * sq + (1 - ALPHA_SHORT) * short_var
        long_var = ALPHA_LONG * sq + (1 - ALPHA_LONG) * long_var
        if long_var < 1e-12:
            continue
        z = float(np.sqrt(short_var / long_var))
        if z >= SHOCK_Z:
            vol[i] = "SHOCK"
        elif z >= HIGH_Z:
            vol[i] = "HIGH"
        elif z <= LOW_Z:
            vol[i] = "LOW"

    # Mark warmup rows so downstream consumers can drop them
    warmup_mask = np.arange(n) < EWMA_WARMUP_BARS
    vol_with_warmup = vol.copy()
    vol_with_warmup[warmup_mask] = None
    trend_with_warmup = trend.copy()
    trend_with_warmup[warmup_mask] = None

    return pd.DataFrame(
        {
            "date": daily["date"].values,
            "trend_regime_symbol": trend_with_warmup,
            "vol_regime_symbol": vol_with_warmup,
        }
    )


def build_market_axis_frame() -> pd.DataFrame:
    """Compute the market axis once on SPY. Returns frame keyed on date."""
    spy_frames = []
    for path in SPY_INPUT_FILES:
        if path.exists():
            spy_frames.append(load_daily_from_minute_parquet(path))
    if not spy_frames:
        raise FileNotFoundError(
            "No SPY 1-minute parquet found in expected locations: " + ", ".join(str(p) for p in SPY_INPUT_FILES)
        )
    daily_spy = (
        pd.concat(spy_frames, ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    market = classify_market_axes(daily_spy)
    return market


def collect_symbol_files() -> dict[str, Path]:
    """Find one parquet per symbol — prefer the largest file when duplicates exist."""
    out: dict[str, Path] = {}
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for f in d.glob("*_1Min.parquet"):
            sym = f.stem.replace("_1Min", "")
            if sym not in out or f.stat().st_size > out[sym].stat().st_size:
                out[sym] = f
    return out


def process_symbol(
    symbol: str,
    parquet_path: Path,
    market_frame: pd.DataFrame,
    output_dir: Path,
    dry_run: bool,
) -> dict:
    try:
        daily = load_daily_from_minute_parquet(parquet_path)
    except Exception as e:
        return {"symbol": symbol, "status": "error", "error": str(e), "days": 0}

    if len(daily) < TREND_SLOW_DAYS + 5:
        return {"symbol": symbol, "status": "insufficient_data", "days": len(daily)}

    symbol_axes = classify_symbol_axes(daily)
    merged = daily.merge(symbol_axes, on="date", how="left").merge(market_frame, on="date", how="left")
    merged["symbol"] = symbol

    if not dry_run:
        out_path = output_dir / f"{symbol}_daily_regime.parquet"
        merged.to_parquet(out_path, index=False)

    # Distributions for the summary
    sym_vol_dist = merged["vol_regime_symbol"].dropna().value_counts().to_dict()
    market_vol_dist = merged["vol_regime_market"].dropna().value_counts().to_dict()

    return {
        "symbol": symbol,
        "status": "ok",
        "days": len(merged),
        "days_after_warmup": int((~merged["vol_regime_symbol"].isna()).sum()),
        "date_range": (f"{merged['date'].iloc[0].date()} -> {merged['date'].iloc[-1].date()}"),
        "vol_regime_symbol": sym_vol_dist,
        "vol_regime_market": market_vol_dist,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute distributions without writing parquets",
    )
    args = parser.parse_args()

    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building market-axis frame from SPY...")
    market_frame = build_market_axis_frame()
    print(
        f"  Market frame: {len(market_frame)} days "
        f"({market_frame['date'].iloc[0].date()} -> "
        f"{market_frame['date'].iloc[-1].date()})\n"
    )

    files = collect_symbol_files()
    print(f"Processing {len(files)} symbols...\n")

    results = []
    agg_symbol_dist: dict[str, int] = {}
    agg_market_dist: dict[str, int] = {}

    for i, (sym, path) in enumerate(sorted(files.items())):
        r = process_symbol(sym, path, market_frame, OUTPUT_DIR, args.dry_run)
        results.append(r)
        if r["status"] == "ok":
            top_sym = max(r["vol_regime_symbol"], key=r["vol_regime_symbol"].get) if r["vol_regime_symbol"] else "-"
            print(
                f"  [{i + 1:3d}/{len(files)}] {sym:8s} "
                f"days={r['days_after_warmup']:5d}  "
                f"{r['date_range']}  symbol_vol_top={top_sym}"
            )
            for k, v in r["vol_regime_symbol"].items():
                agg_symbol_dist[k] = agg_symbol_dist.get(k, 0) + v
            for k, v in r["vol_regime_market"].items():
                agg_market_dist[k] = agg_market_dist.get(k, 0) + v
        else:
            print(f"  [{i + 1:3d}/{len(files)}] {sym:8s} {r['status']}")

    print(f"\n{'=' * 80}\nLABELING COMPLETE (v2)\n{'=' * 80}")
    ok = [r for r in results if r["status"] == "ok"]
    total = sum(r.get("days_after_warmup", 0) for r in ok)
    print(f"  Symbols labeled: {len(ok)}/{len(results)}")
    print(f"  Total post-warmup symbol-days: {total:,}")
    print(f"  Output directory: {OUTPUT_DIR}{' (DRY RUN)' if args.dry_run else ''}")

    print("\n  vol_regime_symbol aggregate (Labeler B per-symbol):")
    for k in sorted(agg_symbol_dist, key=agg_symbol_dist.get, reverse=True):
        pct = agg_symbol_dist[k] / total * 100 if total else 0
        print(f"    {k:<10} {agg_symbol_dist[k]:>8,} ({pct:5.1f}%)")

    print("\n  vol_regime_market aggregate (Labeler A on SPY, broadcast):")
    market_total = sum(agg_market_dist.values())
    for k in sorted(agg_market_dist, key=agg_market_dist.get, reverse=True):
        pct = agg_market_dist[k] / market_total * 100 if market_total else 0
        print(f"    {k:<10} {agg_market_dist[k]:>8,} ({pct:5.1f}%)")

    if not args.dry_run:
        summary_path = OUTPUT_DIR / "labeling_summary.json"
        summary = {
            "schema_version": 2,
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "axes": {
                "vol_regime_market": {
                    "source": "Labeler A on SPY (benchmark, broadcast across symbols)",
                    "method": "30d realized vol, absolute thresholds",
                    "thresholds": {"low": VOL_LOW_ABS, "high": VOL_HIGH_ABS, "shock": VOL_SHOCK_ABS},
                },
                "vol_regime_symbol": {
                    "source": "Labeler B per-symbol (matches MarketContextService)",
                    "method": "EWMA variance ratio z-score",
                    "params": {
                        "alpha_short": ALPHA_SHORT,
                        "alpha_long": ALPHA_LONG,
                        "z_low": LOW_Z,
                        "z_high": HIGH_Z,
                        "z_shock": SHOCK_Z,
                        "warmup_bars": EWMA_WARMUP_BARS,
                    },
                },
                "trend_regime_market": {
                    "source": "Labeler A on SPY (broadcast)",
                    "method": "dual-SMA crossover with flat band",
                    "params": {
                        "fast": TREND_FAST_DAYS,
                        "slow": TREND_SLOW_DAYS,
                        "flat_band": TREND_FLAT_BAND,
                    },
                },
                "trend_regime_symbol": {
                    "source": "Per-symbol dual-SMA crossover",
                    "method": "dual-SMA crossover with flat band",
                    "params": {
                        "fast": TREND_FAST_DAYS,
                        "slow": TREND_SLOW_DAYS,
                        "flat_band": TREND_FLAT_BAND,
                    },
                },
            },
            "symbols_labeled": len(ok),
            "total_post_warmup_days": total,
            "vol_regime_symbol_distribution": agg_symbol_dist,
            "vol_regime_market_distribution": agg_market_dist,
            "per_symbol": [
                {k: v for k, v in r.items() if k not in ("vol_regime_symbol", "vol_regime_market")} for r in ok
            ],
        }
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

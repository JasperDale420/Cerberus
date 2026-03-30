#!/usr/bin/env python3
"""Label earnings windows for all symbols in regime_labeled parquets.

Uses yfinance to fetch historical earnings dates, falls back to Data-Gateway.
Adds `earnings_window` and `near_earnings` columns to each parquet.
"""

import os
import sys
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

# Suppress all logging/warnings from yfinance and urllib3
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.disable(logging.CRITICAL)

REGIME_DIR = Path(__file__).resolve().parent.parent / "data" / "regime_labeled"
GATEWAY_URL = os.environ.get("DATA_INGESTION_URL", "http://localhost:8080")


def get_earnings_dates_yfinance(symbol: str) -> list[pd.Timestamp]:
    """Fetch earnings dates from yfinance."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.get_earnings_dates(limit=50)
        if df is None or df.empty:
            return []
        # Index is the earnings date (timezone-aware), convert to naive date
        dates = pd.to_datetime(df.index).tz_localize(None).normalize()
        return sorted(dates.unique().tolist())
    except Exception:
        return []


def get_earnings_dates_gateway(symbol: str) -> list[pd.Timestamp]:
    """Fetch earnings dates from Data-Gateway calendar endpoint."""
    try:
        import httpx

        resp = httpx.get(
            f"{GATEWAY_URL}/api/v1/calendar/earnings",
            params={"symbols": symbol},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        dates = []
        for item in data if isinstance(data, list) else data.get("data", []):
            d = item.get("date") or item.get("report_date") or item.get("earnings_date")
            if d:
                dates.append(pd.Timestamp(d).normalize())
        return sorted(set(dates))
    except Exception:
        return []


def classify_earnings_window(days_delta: int) -> str:
    """Classify a day relative to nearest earnings date."""
    if days_delta == 0:
        return "earnings_day"
    elif -1 <= days_delta < 0 or days_delta == -1:
        return "pre_1d"
    elif -3 <= days_delta < -1:
        return "pre_3d"
    elif -5 <= days_delta < -3:
        return "pre_5d"
    elif 0 < days_delta <= 1:
        return "post_1d"
    elif 1 < days_delta <= 3:
        return "post_3d"
    else:
        return "none"


def label_symbol(symbol: str, parquet_path: Path) -> dict:
    """Label a single symbol's parquet with earnings windows. Returns stats."""
    result = {"symbol": symbol, "earnings_found": False, "earnings_count": 0, "near_count": 0, "total_days": 0}

    # Load parquet
    df = pd.read_parquet(parquet_path)
    result["total_days"] = len(df)

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Skip index ETFs that don't have earnings
    if symbol in ("SPY", "QQQ", "VXX"):
        df["earnings_window"] = "none"
        df["near_earnings"] = False
        df.to_parquet(parquet_path, index=False)
        return result

    # Try yfinance first, then gateway
    earnings_dates = get_earnings_dates_yfinance(symbol)
    source = "yfinance"
    if not earnings_dates:
        earnings_dates = get_earnings_dates_gateway(symbol)
        source = "gateway"

    if not earnings_dates:
        df["earnings_window"] = "unknown"
        df["near_earnings"] = False
        df.to_parquet(parquet_path, index=False)
        return result

    result["earnings_found"] = True
    result["earnings_count"] = len(earnings_dates)
    result["source"] = source

    # Build earnings date set for fast lookup
    earnings_ts = pd.DatetimeIndex(earnings_dates)

    # For each row, find the nearest earnings date and compute window
    windows = []
    near_flags = []

    for _, row in df.iterrows():
        day = row["date"]
        # Compute business-day delta to nearest earnings
        deltas = [(e - day).days for e in earnings_ts]

        # Find the closest earnings date
        abs_deltas = [abs(d) for d in deltas]
        min_idx = abs_deltas.index(min(abs_deltas))
        nearest_delta = deltas[min_idx]  # negative = earnings in future, positive = earnings in past

        # We want: negative means "before earnings", positive means "after earnings"
        # delta = earnings_date - current_date
        # positive delta -> earnings is in the future -> we are "pre" earnings
        # negative delta -> earnings is in the past -> we are "post" earnings
        if nearest_delta > 0:
            # Earnings in the future: pre-earnings
            if nearest_delta <= 1:
                window = "pre_1d"
            elif nearest_delta <= 3:
                window = "pre_3d"
            elif nearest_delta <= 5:
                window = "pre_5d"
            else:
                window = "none"
        elif nearest_delta == 0:
            window = "earnings_day"
        else:
            # Earnings in the past: post-earnings
            abs_d = abs(nearest_delta)
            if abs_d <= 1:
                window = "post_1d"
            elif abs_d <= 3:
                window = "post_3d"
            else:
                window = "none"

        near = window != "none"
        windows.append(window)
        near_flags.append(near)

    df["earnings_window"] = windows
    df["near_earnings"] = near_flags
    result["near_count"] = sum(near_flags)

    df.to_parquet(parquet_path, index=False)
    return result


def main():
    # Discover symbols from parquet filenames
    parquets = sorted(REGIME_DIR.glob("*_daily_regime.parquet"))
    symbols = []
    for p in parquets:
        sym = p.stem.replace("_daily_regime", "")
        if sym == "all_symbols":
            continue
        symbols.append((sym, p))

    print(f"Found {len(symbols)} symbols in {REGIME_DIR}\n")

    # Process each symbol
    results = []
    for i, (sym, path) in enumerate(symbols, 1):
        sys.stdout.write(f"\r  Processing {sym:<8s} ({i}/{len(symbols)})")
        sys.stdout.flush()
        r = label_symbol(sym, path)
        results.append(r)

    print("\n")

    # Summary
    found = [r for r in results if r["earnings_found"]]
    not_found = [r for r in results if not r["earnings_found"]]
    etfs = [r for r in results if r["symbol"] in ("SPY", "QQQ", "VXX")]

    total_earnings = sum(r["earnings_count"] for r in results)
    total_near = sum(r["near_count"] for r in results)
    total_days = sum(r["total_days"] for r in results)

    print("=" * 60)
    print("EARNINGS LABELING SUMMARY")
    print("=" * 60)

    print(f"\nSymbols with earnings data:     {len(found)}")
    print(f"Symbols without earnings data:  {len(not_found) - len(etfs)}")
    print(f"ETFs/indices (skipped):         {len(etfs)}")

    if not_found:
        no_data_syms = [r["symbol"] for r in not_found if r["symbol"] not in ("SPY", "QQQ", "VXX")]
        if no_data_syms:
            print(f"  Missing: {', '.join(no_data_syms)}")

    print(f"\nTotal earnings events:          {total_earnings}")
    print(f"Total trading days:             {total_days}")
    print(f"Days near earnings:             {total_near} ({100 * total_near / total_days:.1f}%)")
    print(
        f"Days far from earnings:         {total_days - total_near} ({100 * (total_days - total_near) / total_days:.1f}%)"
    )

    # Per-window distribution
    print("\nWindow distribution across all symbols:")
    all_windows = []
    for sym, path in symbols:
        if sym == "all_symbols":
            continue
        df = pd.read_parquet(path, columns=["earnings_window"])
        all_windows.extend(df["earnings_window"].tolist())

    dist = pd.Series(all_windows).value_counts()
    for window, count in dist.items():
        pct = 100 * count / len(all_windows)
        print(f"  {window:<16s}  {count:>6d}  ({pct:5.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Backfill 1-minute bars from Alpaca to parquet files.

Downloads missing date ranges for each symbol and appends to existing
parquet files in data/bars_2023_2025/. Deduplicates on timestamp+symbol.

Usage:
    # Set Alpaca API keys first (see Data-Gateway config)
    export APCA_API_KEY_ID="your-key"
    export APCA_API_SECRET_KEY="your-api-credential"  # pragma: allowlist secret

    # Run from Cerberus directory
    cd /Users/jacobmcmillan/Empire/Cerberus
    uv run python scripts/backfill_bars.py

    # Or specify a target end date (default: today)
    uv run python scripts/backfill_bars.py --end 2026-03-20

    # Backfill specific symbols only
    uv run python scripts/backfill_bars.py --symbols SPY,QQQ,NVDA
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Alpaca SDK
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

DATA_DIR = "data/bars_2023_2025"

# Default start date for full backfill (override with --start)
DEFAULT_START_DATE = "2020-01-01"

# Full universe — all symbols from universe.yaml + offline_symbols.txt
ALL_SYMBOLS = [
    "AAL",
    "AAPL",
    "ABBV",
    "AI",
    "AMC",
    "AMD",
    "AMZN",
    "ARM",
    "AVGO",
    "BAC",
    "C",
    "CCL",
    "COIN",
    "COST",
    "CRM",
    "CVX",
    "DAL",
    "DDOG",
    "F",
    "GM",
    "GME",
    "GOOGL",
    "GS",
    "HAL",
    "HD",
    "HOOD",
    "INTC",
    "IONQ",
    "JNJ",
    "JPM",
    "LCID",
    "LYFT",
    "MARA",
    "META",
    "MRK",
    "MRVL",
    "MS",
    "MSFT",
    "MU",
    "NET",
    "NFLX",
    "NIO",
    "NKE",
    "NVDA",
    "ORCL",
    "OXY",
    "PFE",
    "PLTR",
    "QCOM",
    "QQQ",
    "RCL",
    "RGTI",
    "RIOT",
    "RIVN",
    "SLB",
    "SMR",
    "SNOW",
    "SOFI",
    "SOUN",
    "SPY",
    "TGT",
    "TSLA",
    "UAL",
    "UNH",
    "UBER",
    "VXX",
    "WMT",
    "XOM",
]

# Parquet schema matching existing files
SCHEMA = pa.schema(
    [
        ("close", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("trade_count", pa.int64()),
        ("open", pa.float64()),
        ("timestamp", pa.timestamp("ns", tz="UTC")),
        ("volume", pa.int64()),
        ("vwap", pa.float64()),
        ("symbol", pa.string()),
    ]
)


def get_existing_end_date(symbol: str) -> datetime | None:
    """Get the last timestamp in the existing parquet file for a symbol."""
    path = os.path.join(DATA_DIR, f"{symbol}_1Min.parquet")
    if not os.path.exists(path):
        return None
    try:
        t = pq.read_table(path, columns=["timestamp"])
        if len(t) == 0:
            return None
        ts = t["timestamp"].to_pylist()
        return max(ts)
    except Exception as e:
        print(f"  WARNING: Could not read {path}: {e}")
        return None


def fetch_bars(client: StockHistoricalDataClient, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch 1-minute bars from Alpaca for a symbol and date range."""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed="sip",
        adjustment="raw",
    )

    bars = client.get_stock_bars(request)
    if not bars or not bars.data or symbol not in bars.data:
        return pd.DataFrame()

    rows = []
    for bar in bars.data[symbol]:
        rows.append(
            {
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
                "trade_count": int(bar.trade_count) if bar.trade_count else 0,
                "open": float(bar.open),
                "timestamp": bar.timestamp,
                "volume": int(bar.volume),
                "vwap": float(bar.vwap) if bar.vwap else 0.0,
                "symbol": symbol,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def get_existing_start_date(symbol: str) -> datetime | None:
    """Get the first timestamp in the existing parquet file for a symbol."""
    path = os.path.join(DATA_DIR, f"{symbol}_1Min.parquet")
    if not os.path.exists(path):
        return None
    try:
        t = pq.read_table(path, columns=["timestamp"])
        if len(t) == 0:
            return None
        ts = t["timestamp"].to_pylist()
        return min(ts)
    except Exception as e:
        print(f"  WARNING: Could not read {path}: {e}")
        return None


def backfill_symbol(
    client: StockHistoricalDataClient, symbol: str, target_end: datetime, start_date_str: str = DEFAULT_START_DATE
) -> int:
    """Backfill a single symbol. Returns number of new bars added."""
    existing_end = get_existing_end_date(symbol)
    existing_start = get_existing_start_date(symbol)
    path = os.path.join(DATA_DIR, f"{symbol}_1Min.parquet")
    target_start = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Check if we need to backfill before existing data
    needs_pre_fill = existing_start and existing_start > target_start + pd.Timedelta(days=1)
    needs_post_fill = not existing_end or existing_end.date() < target_end.date()

    if not needs_pre_fill and not needs_post_fill:
        print(f"  {symbol:6s} — already complete ({existing_start.date()} → {existing_end.date()})")
        return 0

    all_new_bars = []

    # Phase 1: Backfill BEFORE existing data (e.g., 2020 → 2023)
    if needs_pre_fill:
        pre_end = existing_start - pd.Timedelta(minutes=1)
        print(f"  {symbol:6s} — pre-filling {target_start.date()} → {pre_end.date()}")
        current_start = target_start
        while current_start < pre_end:
            chunk_end = min(current_start + pd.Timedelta(days=30), pre_end)
            try:
                df = fetch_bars(client, symbol, current_start, chunk_end)
                if len(df) > 0:
                    all_new_bars.append(df)
                    print(f"    {symbol} {current_start.date()}→{chunk_end.date()}: {len(df):,} bars")
            except Exception as e:
                print(f"    {symbol} {current_start.date()}→{chunk_end.date()}: ERROR {e}")
                time.sleep(2)
            current_start = chunk_end
            time.sleep(0.15)

    # Phase 2: Backfill AFTER existing data (e.g., last bar → today)
    if needs_post_fill:
        if existing_end:
            fetch_start = existing_end + pd.Timedelta(minutes=1)
            print(f"  {symbol:6s} — post-filling {existing_end.date()} → {target_end.date()}")
        else:
            fetch_start = target_start
            print(f"  {symbol:6s} — full download {start_date_str} → {target_end.date()}")

        current_start = fetch_start
        while current_start < target_end:
            chunk_end = min(current_start + pd.Timedelta(days=30), target_end)
            try:
                df = fetch_bars(client, symbol, current_start, chunk_end)
                if len(df) > 0:
                    all_new_bars.append(df)
                    print(f"    {symbol} {current_start.date()}→{chunk_end.date()}: {len(df):,} bars")
            except Exception as e:
                print(f"    {symbol} {current_start.date()}→{chunk_end.date()}: ERROR {e}")
                time.sleep(2)
            current_start = chunk_end
            time.sleep(0.15)

    if not all_new_bars:
        print(f"  {symbol:6s} — no new bars found")
        return 0

    new_df = pd.concat(all_new_bars, ignore_index=True)

    # Load existing data and merge
    if os.path.exists(path):
        existing_df = pq.read_table(path).to_pandas()
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Deduplicate on timestamp + symbol
    combined = combined.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Write back as parquet with matching schema
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    table = pa.Table.from_pandas(combined, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, path)

    len(combined) - (len(pq.read_table(path, columns=["timestamp"])) if os.path.exists(path) else 0)
    print(f"  {symbol:6s} — wrote {len(combined):,} total bars ({len(new_df):,} new)")
    return len(new_df)


def main():
    parser = argparse.ArgumentParser(description="Backfill 1-minute bars from Alpaca")
    parser.add_argument(
        "--start", default=DEFAULT_START_DATE, help=f"Target start date (default: {DEFAULT_START_DATE})"
    )
    parser.add_argument(
        "--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"), help="Target end date (default: today)"
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols to backfill (default: all)")
    args = parser.parse_args()

    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")

    if not api_key or not api_secret:
        print("ERROR: Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables")  # pragma: allowlist secret
        print("  export APCA_API_KEY_ID='your-key'")
        print("  export APCA_API_SECRET_KEY='your-credential'")  # pragma: allowlist secret
        sys.exit(1)

    target_end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    symbols = args.symbols.split(",") if args.symbols else ALL_SYMBOLS

    print(f"{'=' * 60}")
    print(f"  Backfilling 1-min bars → {DATA_DIR}")
    print(f"  Target range: {args.start} → {target_end.date()}")
    print(f"  Symbols: {len(symbols)}")
    print(f"{'=' * 60}\n")

    client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    os.makedirs(DATA_DIR, exist_ok=True)

    total_new = 0
    for i, symbol in enumerate(symbols):
        print(f"\n[{i + 1}/{len(symbols)}] {symbol}")
        try:
            n = backfill_symbol(client, symbol, target_end, start_date_str=args.start)
            total_new += n
        except Exception as e:
            print(f"  {symbol:6s} — FAILED: {e}")
        time.sleep(0.2)

    print(f"\n{'=' * 60}")
    print(f"  Done! {total_new:,} new bars added across {len(symbols)} symbols")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

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

# All symbols in our universe
ALL_SYMBOLS = [
    "AAL",
    "AAPL",
    "AMD",
    "AMZN",
    "AVGO",
    "BAC",
    "CCL",
    "COIN",
    "CRM",
    "CVX",
    "DAL",
    "F",
    "GM",
    "GOOGL",
    "GS",
    "HOOD",
    "INTC",
    "JPM",
    "LYFT",
    "META",
    "MSFT",
    "MS",
    "MU",
    "NFLX",
    "NVDA",
    "ORCL",
    "PLTR",
    "QCOM",
    "QQQ",
    "SOFI",
    "SPY",
    "TSLA",
    "UBER",
    "VXX",
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


def backfill_symbol(client: StockHistoricalDataClient, symbol: str, target_end: datetime) -> int:
    """Backfill a single symbol. Returns number of new bars added."""
    existing_end = get_existing_end_date(symbol)
    path = os.path.join(DATA_DIR, f"{symbol}_1Min.parquet")

    if existing_end and existing_end.date() >= target_end.date():
        print(f"  {symbol:6s} — already up to date ({existing_end.date()})")
        return 0

    # Start fetching from the day after the last bar
    if existing_end:
        fetch_start = existing_end + pd.Timedelta(minutes=1)
        print(f"  {symbol:6s} — backfilling {existing_end.date()} → {target_end.date()}")
    else:
        fetch_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        print(f"  {symbol:6s} — full download 2023-01-01 → {target_end.date()}")

    # Alpaca has a ~10k bar limit per request, so chunk by month
    all_new_bars = []
    current_start = fetch_start
    while current_start < target_end:
        # Chunk: ~1 month at a time
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
        time.sleep(0.15)  # Rate limiting: ~7 req/sec

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
    print(f"  Target end: {target_end.date()}")
    print(f"  Symbols: {len(symbols)}")
    print(f"{'=' * 60}\n")

    client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    os.makedirs(DATA_DIR, exist_ok=True)

    total_new = 0
    for i, symbol in enumerate(symbols):
        print(f"\n[{i + 1}/{len(symbols)}] {symbol}")
        try:
            n = backfill_symbol(client, symbol, target_end)
            total_new += n
        except Exception as e:
            print(f"  {symbol:6s} — FAILED: {e}")
        time.sleep(0.2)

    print(f"\n{'=' * 60}")
    print(f"  Done! {total_new:,} new bars added across {len(symbols)} symbols")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

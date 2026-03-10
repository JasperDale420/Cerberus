#!/usr/bin/env python3
"""Download historical 1-minute bars directly from Alpaca Data API and save as parquet.

Usage:
    python scripts/download_bars_parquet.py --start 2024-01-01 --end 2024-12-31 --out data/bars_2024
    python scripts/download_bars_parquet.py --start 2024-01-01 --end 2024-01-31 --out data/bars_jan2024
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

# Default universe — same 35 symbols used across all backtest configs
DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "AMD", "AMZN", "AVGO", "BAC", "CCL", "COIN", "CRM",
    "CVX", "DAL", "F", "GM", "GOOGL", "GS", "HOOD", "INTC", "JPM", "LYFT",
    "META", "MSFT", "MS", "MU", "NFLX", "NVDA", "ORCL", "PLTR", "QCOM", "SOFI",
    "TSLA", "UBER", "VXX", "XOM", "AAL",
]

ALPACA_DATA_URL = "https://data.alpaca.markets"


def _load_env(env_path: str = ".env") -> None:
    """Load .env without depending on python-dotenv."""
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _get_alpaca_creds() -> tuple[str, str]:
    """Get Alpaca credentials from env — tries multiple naming conventions."""
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY") or ""
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET") or os.environ.get("ALPACA_SECRET_KEY") or ""
    return key, secret


def _month_ranges(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split date range into month-sized chunks for paginated fetching."""
    ranges = []
    cursor = start.replace(day=1)
    while cursor <= end:
        month_end = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(seconds=1)
        chunk_start = max(cursor, start)
        chunk_end = min(month_end, end.replace(hour=23, minute=59, second=59))
        ranges.append((chunk_start, chunk_end))
        cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
    return ranges


def fetch_bars_alpaca(
    client: httpx.Client,
    headers: dict[str, str],
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch 1-min bars from Alpaca Data API with pagination, chunked by month."""
    all_bars = []
    ranges = _month_ranges(start, end)

    for chunk_start, chunk_end in ranges:
        page_token = None
        month_bars = 0

        while True:
            params: dict[str, str] = {
                "timeframe": "1Min",
                "start": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": "10000",
                "feed": "sip",
            }
            if page_token:
                params["page_token"] = page_token

            url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"

            try:
                resp = client.get(url, params=params, headers=headers, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                bars = data.get("bars") or []
                all_bars.extend(bars)
                month_bars += len(bars)

                page_token = data.get("next_page_token")
                if not page_token:
                    break
            except httpx.HTTPStatusError as e:
                print(f"  {symbol} {chunk_start.strftime('%Y-%m')}: HTTP {e.response.status_code}", flush=True)
                if e.response.status_code == 429:
                    time.sleep(5)
                    continue
                break
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                print(f"  {symbol} {chunk_start.strftime('%Y-%m')}: {type(e).__name__} — retrying...", flush=True)
                time.sleep(3)
                try:
                    resp = client.get(url, params=params, headers=headers, timeout=120.0)
                    resp.raise_for_status()
                    data = resp.json()
                    bars = data.get("bars") or []
                    all_bars.extend(bars)
                    month_bars += len(bars)
                    page_token = data.get("next_page_token")
                    if not page_token:
                        break
                except Exception as e2:
                    print(f"  {symbol} {chunk_start.strftime('%Y-%m')}: FAILED — {e2}", flush=True)
                    break

        print(f"  {symbol} {chunk_start.strftime('%Y-%m')}: {month_bars:,} bars", flush=True)

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df["symbol"] = symbol

    # Normalize Alpaca column names
    rename_map = {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "trade_count"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume", "vwap", "trade_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main():
    parser = argparse.ArgumentParser(description="Download historical bars from Alpaca → parquet")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out", required=True, help="Output directory for parquet files")
    parser.add_argument("--symbols", nargs="*", default=None, help="Override symbol list")
    parser.add_argument("--combined", action="store_true", help="Also save a combined all-symbols parquet")
    args = parser.parse_args()

    _load_env()

    api_key, api_secret = _get_alpaca_creds()

    if not api_key or not api_secret:
        print("ERROR: Alpaca API credentials not found in .env or environment")
        print("  Need: APCA_API_KEY_ID + APCA_API_SECRET_KEY (or ALPACA_API_KEY + ALPACA_SECRET_KEY)")
        sys.exit(1)

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(hour=23, minute=59, second=59, tzinfo=UTC)
    symbols = args.symbols or DEFAULT_SYMBOLS
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(symbols)} symbols: {start.date()} → {end.date()}", flush=True)
    print(f"API: {ALPACA_DATA_URL} (SIP feed)", flush=True)
    print(f"Output: {out_dir}/", flush=True)
    print(flush=True)

    all_dfs = []
    failed = []

    with httpx.Client(http2=True) as client:
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] {symbol}", flush=True)
            df = fetch_bars_alpaca(client, headers, symbol, start, end)

            if df.empty:
                failed.append(symbol)
                print(f"  WARNING: No data for {symbol}\n", flush=True)
                continue

            parquet_path = out_dir / f"{symbol}_1Min.parquet"
            df.to_parquet(parquet_path, index=False, engine="pyarrow")
            all_dfs.append(df)
            bar_count = len(df)
            size_mb = parquet_path.stat().st_size / 1024 / 1024
            print(f"  ✓ Saved: {parquet_path.name} ({bar_count:,} bars, {size_mb:.1f} MB)\n", flush=True)

            # Brief pause between symbols to respect rate limits
            time.sleep(0.3)

    # Save symbols list
    symbols_path = out_dir / "offline_symbols.txt"
    with open(symbols_path, "w") as f:
        for s in symbols:
            if s not in failed:
                f.write(s + "\n")

    # Combined parquet
    if args.combined and all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True).sort_values("timestamp")
        combined_path = out_dir / "all_symbols_1Min.parquet"
        combined.to_parquet(combined_path, index=False, engine="pyarrow")
        size = combined_path.stat().st_size / 1024 / 1024
        print(f"\nCombined: {combined_path.name} ({len(combined):,} bars, {size:.1f} MB)", flush=True)

    print(f"\n{'='*50}", flush=True)
    print(f"Done! {len(symbols) - len(failed)}/{len(symbols)} symbols downloaded.", flush=True)
    if failed:
        print(f"Failed: {', '.join(failed)}", flush=True)


if __name__ == "__main__":
    main()

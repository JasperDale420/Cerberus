"""Download historical 1-minute bars from Alpaca via Data-Gateway and label with regime data.

Downloads 2014-2020 data to extend our existing 2020-2026 dataset.
Converts to parquet format matching data/bars_2023_2025/{SYMBOL}_1Min.parquet.
Labels with the same regime detector (SMA crossover + realized vol) used for existing data.

Usage:
    uv run python scripts/download_historical_data.py [--start 2014-01-01] [--end 2020-06-01]
"""

import argparse
import logging
import os
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Same 20 symbols as autoresearch v6
SYMBOLS = [
    "SPY",
    "QQQ",
    "AAPL",
    "NVDA",
    "TSLA",
    "AMD",
    "AMZN",
    "META",
    "MSFT",
    "GOOGL",
    "JPM",
    "GS",
    "BAC",
    "XOM",
    "CVX",
    "NFLX",
    "UBER",
    "COIN",
    "PLTR",
    "SOFI",
]

# Symbols with limited history (IPO after 2014)
# TSLA: 2010, META: 2012, UBER: 2019, COIN: 2021, PLTR: 2020, SOFI: 2021
# NVDA/AMD/etc have full history

GATEWAY_URL = os.environ.get("CERBERUS_GATEWAY_URL", os.environ.get("DATA_INGESTION_URL", "http://localhost:8080"))
GATEWAY_KEY = os.environ.get(
    "CERBERUS_GATEWAY_KEY", os.environ.get("GATEWAY_API_KEY", os.environ.get("X_GATEWAY_KEY", ""))
)

OUTPUT_DIR = Path("data/bars_historical")
REGIME_OUTPUT_DIR = Path("data/regime_labeled")

# Same regime detector params as label_regime_dataset.py
TREND_FAST = 10
TREND_SLOW = 40
TREND_FLAT_BAND = 0.01
VOL_WINDOW = 30
VOL_LOW = 0.08
VOL_HIGH = 0.20
VOL_SHOCK = 0.50


def download_bars_alpaca_direct(symbol: str, start: str, end: str, timeframe: str = "1Min") -> pd.DataFrame:
    """Download bars directly from Alpaca API (bypasses gateway if needed)."""
    api_key = os.environ.get("ALPACA_API_KEY", os.environ.get("APCA_API_KEY_ID", ""))
    api_secret = os.environ.get("ALPACA_SECRET_KEY", os.environ.get("APCA_API_SECRET_KEY", ""))

    if not api_key or not api_secret:
        print("  WARNING: No Alpaca credentials found, trying Data-Gateway...")
        return download_bars_gateway(symbol, start, end, timeframe)

    base_url = "https://data.alpaca.markets/v2"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    all_bars = []
    page_token = None
    total_fetched = 0

    with httpx.Client(timeout=60.0) as client:
        while True:
            params = {
                "start": start,
                "end": end,
                "timeframe": timeframe,
                "limit": 10000,
                "feed": "sip",
                "adjustment": "raw",
            }
            if page_token:
                params["page_token"] = page_token

            try:
                resp = client.get(f"{base_url}/stocks/{symbol}/bars", headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  ERROR fetching {symbol}: {e}")
                break

            bars = data.get("bars") or []
            if not bars:
                break

            all_bars.extend(bars)
            total_fetched += len(bars)

            if total_fetched % 50000 == 0:
                print(f"    {symbol}: {total_fetched:,} bars fetched...")

            page_token = data.get("next_page_token")
            if not page_token:
                break

            # Respect rate limits
            time.sleep(0.05)

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df.rename(
        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap"},
        inplace=True,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = symbol
    return (
        df[["timestamp", "open", "high", "low", "close", "volume", "vwap", "symbol"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def download_bars_gateway(symbol: str, start: str, end: str, timeframe: str = "1Min") -> pd.DataFrame:
    """Download bars via Data-Gateway. Handles the envelope response format."""
    all_bars = []

    # Split into 3-month chunks for 1Min data to avoid timeout
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    current = start_dt
    chunk_months = 1 if timeframe == "1Min" else 6

    with httpx.Client(base_url=GATEWAY_URL, timeout=120.0) as client:
        headers = {"X-Gateway-Key": GATEWAY_KEY} if GATEWAY_KEY else {}

        while current < end_dt:
            chunk_end = min(current + pd.DateOffset(months=chunk_months), end_dt)
            params = {
                "timeframe": timeframe,
                "start": current.strftime("%Y-%m-%d"),
                "end": chunk_end.strftime("%Y-%m-%d"),
                "limit": 10000,
                "feed": "sip",
            }

            page_token = None
            while True:
                if page_token:
                    params["page_token"] = page_token

                try:
                    resp = client.get(f"/api/v1/alpaca/stocks/{symbol}/bars", headers=headers, params=params)
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as e:
                    print(f"  ERROR via gateway for {symbol} ({current.strftime('%Y-%m')}): {e}")
                    break

                # Gateway wraps in envelope: result["data"]["bars"]
                inner = result.get("data", result)
                if isinstance(inner, dict):
                    bars = inner.get("bars", [])
                    page_token = inner.get("next_page_token")
                elif isinstance(inner, list):
                    bars = inner
                    page_token = result.get("next_page_token")
                else:
                    bars = []
                    page_token = None

                if not bars:
                    break

                all_bars.extend(bars)

                if len(all_bars) % 50000 == 0:
                    print(f"    {symbol}: {len(all_bars):,} bars...")

                if not page_token:
                    break
                time.sleep(0.05)

            current = chunk_end

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    # Gateway returns named columns — convert string values to float
    for col in ["open", "high", "low", "close", "volume", "vwap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = symbol
    cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume", "vwap", "symbol"] if c in df.columns]
    return df[cols].sort_values("timestamp").reset_index(drop=True)


def classify_daily_regime(daily: pd.DataFrame) -> pd.DataFrame:
    """Same regime detector as label_regime_dataset.py — SMA crossover + realized vol."""
    closes = daily["close"].values.astype(float)
    n = len(closes)

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


def main():
    parser = argparse.ArgumentParser(description="Download historical bars and label with regime data")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2020-06-01", help="End date (fills gap to existing data)")
    parser.add_argument("--symbols", default=",".join(SYMBOLS), help="Comma-separated symbol list")
    parser.add_argument("--direct", action="store_true", help="Use Alpaca API directly (not via gateway)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REGIME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print("  Historical Data Download + Regime Labeling")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Symbols: {len(symbols)} ({', '.join(symbols[:5])}...)")
    print(f"  Output: {OUTPUT_DIR} (parquet) + {REGIME_OUTPUT_DIR} (labeled)")
    print(f"  Method: {'Alpaca direct' if args.direct else 'Data-Gateway'}")
    print(f"{'=' * 60}")

    results = []
    for i, symbol in enumerate(symbols):
        print(f"\n[{i + 1}/{len(symbols)}] Downloading {symbol}...")
        start_time = time.time()

        if args.direct:
            df = download_bars_alpaca_direct(symbol, args.start, args.end)
        else:
            df = download_bars_gateway(symbol, args.start, args.end)

        elapsed = time.time() - start_time

        if len(df) == 0:
            print(f"  {symbol}: NO DATA (may not have history before {args.end})")
            results.append({"symbol": symbol, "bars": 0, "status": "no_data"})
            continue

        # Save as parquet (same format as existing data)
        parquet_path = OUTPUT_DIR / f"{symbol}_1Min.parquet"
        df.to_parquet(parquet_path, index=False)

        # Resample to daily and label with regime
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
        daily["symbol"] = symbol

        if len(daily) >= TREND_SLOW + 5:
            labeled = classify_daily_regime(daily)
            # Append to existing regime labeled file or create new
            regime_path = REGIME_OUTPUT_DIR / f"{symbol}_daily_regime.parquet"
            if regime_path.exists():
                existing = pd.read_parquet(regime_path)
                existing["date"] = pd.to_datetime(existing["date"])
                combined = (
                    pd.concat([labeled, existing])
                    .drop_duplicates(subset=["date"])
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                combined.to_parquet(regime_path, index=False)
                print(
                    f"  {symbol}: {len(df):>8,} bars, {len(daily)} days, regime labeled, MERGED with existing ({len(combined)} total days)"
                )
            else:
                labeled.to_parquet(regime_path, index=False)
                print(f"  {symbol}: {len(df):>8,} bars, {len(daily)} days, regime labeled, NEW file")
        else:
            print(f"  {symbol}: {len(df):>8,} bars, {len(daily)} days (too few for regime labeling)")

        results.append(
            {"symbol": symbol, "bars": len(df), "days": len(daily), "elapsed": round(elapsed, 1), "status": "ok"}
        )

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    total_bars = sum(r.get("bars", 0) for r in ok)
    total_days = sum(r.get("days", 0) for r in ok)
    print(f"\n{'=' * 60}")
    print("  Download Complete")
    print(f"  Symbols: {len(ok)}/{len(results)} successful")
    print(f"  Total bars: {total_bars:,}")
    print(f"  Total daily bars: {total_days:,}")
    print(f"  Parquet files: {OUTPUT_DIR}")
    print(f"  Regime labels: {REGIME_OUTPUT_DIR}")
    print(f"{'=' * 60}")

    # Also merge into the main bars_2023_2025 directory for WFO access
    print("\nMerging into data/bars_2023_2025/ for WFO access...")
    main_dir = Path("data/bars_2023_2025")
    for symbol in symbols:
        hist_path = OUTPUT_DIR / f"{symbol}_1Min.parquet"
        main_path = main_dir / f"{symbol}_1Min.parquet"
        if hist_path.exists() and main_path.exists():
            hist_df = pd.read_parquet(hist_path)
            main_df = pd.read_parquet(main_path)
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], utc=True)
            main_df["timestamp"] = pd.to_datetime(main_df["timestamp"], utc=True)
            combined = (
                pd.concat([hist_df, main_df])
                .drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            combined.to_parquet(main_path, index=False)
            print(f"  {symbol}: merged {len(hist_df):,} + {len(main_df):,} = {len(combined):,} bars")
        elif hist_path.exists():
            hist_df = pd.read_parquet(hist_path)
            hist_df.to_parquet(main_path, index=False)
            print(f"  {symbol}: copied {len(hist_df):,} bars (new symbol)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Pre-aggregate 1-minute bars to daily OHLCV parquet files.

Reads {SYMBOL}_1Min.parquet → writes {SYMBOL}_1Day.parquet in the same directory.
Optionally merges regime labels from data/regime_labeled/ as extra columns.

This eliminates the need for the backtest runner to load 3GB of 1-min data
and aggregate on-the-fly when bar_resolution_minutes=1440 (daily).

Usage:
    uv run python scripts/precompute_daily_bars.py [--data-dir data/bars_2023_2025] [--with-regime] [--with-indicators]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = [
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

REGIME_COLUMNS = [
    "regime_trend",
    "regime_vol",
    "regime",
    "liquidity_regime",
    "near_earnings",
    "near_fomc",
    "opex_week",
    "quad_witch_week",
    "fomc_window",
    "earnings_window",
    "correlation_regime",
    "spy_beta",
    "volume_ratio",
]


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min bars to daily OHLCV."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date

    daily = (
        df.groupby(["symbol", "date"])
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trade_count=("trade_count", "sum") if "trade_count" in df.columns else ("volume", "count"),
        )
        .reset_index()
    )

    # VWAP: volume-weighted average price
    if "vwap" in df.columns:
        vwap_num = (df["vwap"].fillna(df["close"]) * df["volume"]).groupby([df["symbol"], df["date"]]).sum()
        vwap_den = df["volume"].groupby([df["symbol"], df["date"]]).sum()
        vwap = (vwap_num / vwap_den.replace(0, np.nan)).fillna(0).reset_index(name="vwap")
        daily = daily.merge(vwap[["symbol", "date", "vwap"]], on=["symbol", "date"], how="left")
    else:
        daily["vwap"] = daily["close"]

    # Convert date back to timestamp for compatibility
    daily["timestamp"] = pd.to_datetime(daily["date"])
    return daily


def merge_regime_labels(daily: pd.DataFrame, symbol: str, regime_dir: Path) -> pd.DataFrame:
    """Merge pre-computed regime labels into daily bars."""
    regime_file = regime_dir / f"{symbol}_daily_regime.parquet"
    if not regime_file.exists():
        return daily

    regime = pd.read_parquet(regime_file)
    regime["date"] = pd.to_datetime(regime["date"]).dt.date

    # Only merge columns that exist and aren't already in daily
    merge_cols = ["date"] + [c for c in REGIME_COLUMNS if c in regime.columns and c not in daily.columns]
    if len(merge_cols) <= 1:
        return daily

    return daily.merge(regime[merge_cols], on="date", how="left")


def add_common_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """Add commonly-used technical indicators as columns."""
    df = daily.sort_values(["symbol", "date"]).copy()

    for sym in df["symbol"].unique():
        mask = df["symbol"] == sym
        closes = df.loc[mask, "close"]
        highs = df.loc[mask, "high"]
        lows = df.loc[mask, "low"]

        # SMAs
        df.loc[mask, "sma_20"] = closes.rolling(20).mean()
        df.loc[mask, "sma_50"] = closes.rolling(50).mean()

        # EMAs
        df.loc[mask, "ema_20"] = closes.ewm(span=20, adjust=False).mean()

        # RSI(2) and RSI(14)
        for period in [2, 14]:
            delta = closes.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            df.loc[mask, f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # ATR(14)
        tr = pd.concat(
            [
                highs - lows,
                (highs - closes.shift(1)).abs(),
                (lows - closes.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df.loc[mask, "atr_14"] = tr.rolling(14).mean()

        # Bollinger Bands (20, 2σ)
        sma20 = closes.rolling(20).mean()
        std20 = closes.rolling(20).std()
        df.loc[mask, "bb_upper"] = sma20 + 2 * std20
        df.loc[mask, "bb_lower"] = sma20 - 2 * std20
        df.loc[mask, "bb_mid"] = sma20

        # IBS (Internal Bar Strength)
        bar_range = highs - lows
        df.loc[mask, "ibs"] = ((closes - lows) / bar_range.replace(0, np.nan)).fillna(0.5)

        # Volume SMA(20)
        df.loc[mask, "vol_sma_20"] = df.loc[mask, "volume"].rolling(20).mean()

    return df


def main():
    parser = argparse.ArgumentParser(description="Pre-aggregate 1-min bars to daily parquet")
    parser.add_argument("--data-dir", default="data/bars_2023_2025", help="Bar data directory")
    parser.add_argument("--regime-dir", default="data/regime_labeled", help="Regime labels directory")
    parser.add_argument("--with-regime", action="store_true", help="Merge regime labels")
    parser.add_argument("--with-indicators", action="store_true", help="Add common indicators")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (default: all 20)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    regime_dir = Path(args.regime_dir)
    symbols = args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS

    t0 = time.time()
    total_rows = 0

    for sym in symbols:
        src = data_dir / f"{sym}_1Min.parquet"
        dst = data_dir / f"{sym}_1Day.parquet"

        if not src.exists():
            print(f"  SKIP {sym}: no 1-min file")
            continue

        st = time.time()
        df = pd.read_parquet(src)
        daily = aggregate_to_daily(df)

        if args.with_regime:
            daily = merge_regime_labels(daily, sym, regime_dir)

        if args.with_indicators:
            daily = add_common_indicators(daily)

        daily.to_parquet(dst, index=False)
        elapsed = time.time() - st
        total_rows += len(daily)
        print(f"  {sym}: {len(df):>10,} 1-min → {len(daily):>6,} daily ({elapsed:.1f}s) → {dst.name}")

    elapsed = time.time() - t0
    print(f"\nDone: {len(symbols)} symbols, {total_rows:,} daily rows, {elapsed:.1f}s total")


if __name__ == "__main__":
    main()

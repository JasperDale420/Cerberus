#!/usr/bin/env python3
"""Download historical bars from Alpaca for offline backtesting.

Usage:
    python scripts/download_bars.py --symbols SPY,QQQ,AAPL --start 2024-01-01 --end 2024-12-31
    python scripts/download_bars.py --universe-file config/universe.yaml --start 2024-01-01 --end 2024-12-31
    python scripts/download_bars.py --top-volume 50 --start 2024-06-01 --end 2024-12-31

Bars are saved to data/offline_bars/{SYMBOL}_{timeframe}.jsonl in the format expected
by JsonlBarsProvider for fast, deterministic backtesting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logger import StructuredLogger
from src.data.client import UnifiedDataClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical bars from Alpaca for offline backtesting")
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (e.g., SPY,QQQ,AAPL)",
    )
    parser.add_argument(
        "--universe-file",
        type=str,
        help="Path to universe YAML config to extract symbols from",
    )
    parser.add_argument(
        "--top-volume",
        type=int,
        help="Download top N symbols by volume from a predefined list",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1Min",
        help="Bar timeframe (default: 1Min)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/offline_bars",
        help="Output directory for JSONL files (default: data/offline_bars)",
    )
    return parser.parse_args()


def load_symbols_from_universe(path: str) -> List[str]:
    """Extract symbols from universe YAML config."""
    import yaml

    p = Path(path)
    if not p.exists():
        print(f"Universe file not found: {path}")
        return []

    with open(p) as f:
        config = yaml.safe_load(f)

    symbols: List[str] = []
    universe = config.get("universe", config)

    # Explicit symbols
    explicit = universe.get("symbols", [])
    if isinstance(explicit, list):
        symbols.extend([str(s).upper() for s in explicit if s])

    # Dynamic candidates
    dyn = universe.get("dynamic", {})
    prev_vol = dyn.get("previous_day_top_volume", {})
    candidates = prev_vol.get("candidates", [])
    if isinstance(candidates, list):
        symbols.extend([str(s).upper() for s in candidates if s])

    # Dedupe
    seen = set()
    result = []
    for sym in symbols:
        if sym not in seen:
            seen.add(sym)
            result.append(sym)
    return result


def get_top_volume_symbols(n: int) -> List[str]:
    """Return a predefined list of high-volume symbols."""
    # Popular trading symbols sorted roughly by typical volume
    all_symbols = [
        "SPY",
        "QQQ",
        "AAPL",
        "TSLA",
        "NVDA",
        "AMD",
        "AMZN",
        "META",
        "MSFT",
        "GOOGL",
        "BAC",
        "F",
        "INTC",
        "PLTR",
        "SOFI",
        "NIO",
        "COIN",
        "HOOD",
        "MARA",
        "RIOT",
        "AMC",
        "GME",
        "CCL",
        "AAL",
        "DAL",
        "UAL",
        "XOM",
        "CVX",
        "JPM",
        "GS",
        "MS",
        "C",
        "WFC",
        "AVGO",
        "MU",
        "QCOM",
        "ARM",
        "CRM",
        "ORCL",
        "SNOW",
        "NET",
        "WMT",
        "COST",
        "TGT",
        "HD",
        "NKE",
        "DIS",
        "NFLX",
        "BABA",
        "PFE",
        "JNJ",
        "UNH",
        "ABBV",
        "MRK",
        "LLY",
        "TMO",
        "ABT",
        "RIVN",
        "LCID",
        "GM",
        "TM",
        "HMC",
        "RACE",
        "UBER",
        "LYFT",
    ]
    return all_symbols[:n]


def _to_iso(val: Any) -> str:
    """Convert various timestamp formats to ISO string."""
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, str):
        return val
    return str(val) if val else ""


def download_bars(
    client: UnifiedDataClient,
    logger: StructuredLogger,
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
) -> List[Dict[str, Any]]:
    """Download bars from Alpaca and return as list of dicts."""
    try:
        bars_raw = client.get_historical_bars(symbol, start, end, timeframe=timeframe)

        # Handle dict wrapper
        if isinstance(bars_raw, dict) and "bars" in bars_raw:
            bars_raw = bars_raw["bars"]

        if not isinstance(bars_raw, list):
            logger.warning("Unexpected bars format", symbol=symbol, type=str(type(bars_raw)))
            return []

        result = []
        for b in bars_raw:
            if isinstance(b, dict):
                result.append(
                    {
                        "t": _to_iso(b.get("t") or b.get("timestamp")),
                        "o": float(b.get("o") or b.get("open") or 0),
                        "h": float(b.get("h") or b.get("high") or 0),
                        "l": float(b.get("l") or b.get("low") or 0),
                        "c": float(b.get("c") or b.get("close") or 0),
                        "v": float(b.get("v") or b.get("volume") or 0),
                    }
                )
            else:
                result.append(
                    {
                        "t": _to_iso(getattr(b, "t", None) or getattr(b, "timestamp", None)),
                        "o": float(getattr(b, "o", 0) or getattr(b, "open", 0) or 0),
                        "h": float(getattr(b, "h", 0) or getattr(b, "high", 0) or 0),
                        "l": float(getattr(b, "l", 0) or getattr(b, "low", 0) or 0),
                        "c": float(getattr(b, "c", 0) or getattr(b, "close", 0) or 0),
                        "v": float(getattr(b, "v", 0) or getattr(b, "volume", 0) or 0),
                    }
                )
        return result
    except Exception as e:
        logger.error("Failed to download bars", symbol=symbol, error=str(e))
        return []


def save_bars(bars: List[Dict[str, Any]], output_path: Path) -> None:
    """Save bars to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for bar in bars:
            f.write(json.dumps(bar) + "\n")


def main() -> None:
    args = parse_args()

    # Get symbols
    symbols: List[str] = []
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.universe_file:
        symbols = load_symbols_from_universe(args.universe_file)
    elif args.top_volume:
        symbols = get_top_volume_symbols(args.top_volume)
    else:
        print("Error: Must specify --symbols, --universe-file, or --top-volume")
        sys.exit(1)

    if not symbols:
        print("Error: No symbols to download")
        sys.exit(1)

    print(f"Downloading bars for {len(symbols)} symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

    # Parse dates
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)

    print(f"Date range: {start.date()} to {end.date()}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Initialize Data-Gateway client
    logger = StructuredLogger("download_bars")
    gateway_url = os.environ.get("CERBERUS_GATEWAY_URL", "http://localhost:8080")
    gateway_key = os.environ.get("CERBERUS_GATEWAY_KEY", "")
    client = UnifiedDataClient(gateway_url, gateway_key)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    total_bars = 0

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Downloading {symbol}...", end=" ", flush=True)

        bars = download_bars(client, logger, symbol, start, end, args.timeframe)

        if bars:
            output_path = output_dir / f"{symbol}_{args.timeframe}.jsonl"
            save_bars(bars, output_path)
            print(f"{len(bars):,} bars saved")
            success += 1
            total_bars += len(bars)
        else:
            print("FAILED (no data)")
            failed += 1

    print()
    print("Download complete!")
    print(f"  Success: {success}/{len(symbols)}")
    print(f"  Failed: {failed}/{len(symbols)}")
    print(f"  Total bars: {total_bars:,}")
    print(f"  Output: {output_dir}")
    print()
    print("To run backtest with offline data:")
    print("  python scripts/run_backtest.py --config config/config.yaml \\")
    print(f"    --start-date {args.start} --end-date {args.end} \\")
    print(f"    --offline-bars-dir {args.output_dir}")


if __name__ == "__main__":
    main()

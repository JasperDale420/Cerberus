#!/usr/bin/env python3
"""Capture daily screener snapshots for historical backtest replay.

This script fetches current most-actives and movers from Alpaca Screener API
and saves them as JSONL snapshots. Run daily via cron to build historical data.

Usage:
    python scripts/capture_screener_snapshot.py

Cron example (run at 4:05 PM ET / 16:05, after market close):
    5 16 * * 1-5 cd /path/to/Cerberus && python scripts/capture_screener_snapshot.py

Output:
    data/screener_snapshots/YYYY-MM-DD.jsonl
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient


def capture_snapshot(
    client: AlpacaClient,
    logger: StructuredLogger,
    most_actives_top_n: int = 50,
    movers_top_n: int = 20,
) -> Dict[str, Any]:
    """Capture screener data from Alpaca API."""
    snapshot: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "most_actives": [],
        "gainers": [],
        "losers": [],
    }

    # Fetch most actives
    try:
        actives = client.get_most_actives(top=most_actives_top_n)
        snapshot["most_actives"] = actives
        logger.info("Captured most actives", count=len(actives))
    except Exception as e:
        logger.error("Failed to capture most actives", error=str(e))

    # Fetch movers
    try:
        movers = client.get_movers(top=movers_top_n)
        snapshot["gainers"] = movers.get("gainers", [])
        snapshot["losers"] = movers.get("losers", [])
        logger.info(
            "Captured movers",
            gainers=len(snapshot["gainers"]),
            losers=len(snapshot["losers"]),
        )
    except Exception as e:
        logger.error("Failed to capture movers", error=str(e))

    return snapshot


def save_snapshot(snapshot: Dict[str, Any], output_dir: Path) -> Path:
    """Save snapshot to JSONL file named by date."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use current date as filename
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = output_dir / f"{date_str}.jsonl"

    # Append to file (allows multiple captures per day if needed)
    with open(output_path, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    return output_path


def main() -> None:
    print("Cerberus Screener Snapshot Capture")
    print("=" * 40)

    # Initialize
    config_loader = ConfigLoader()
    logger = StructuredLogger("screener_capture")

    try:
        client = AlpacaClient(config_loader, logger)
    except Exception as e:
        print(f"Error: Failed to initialize Alpaca client: {e}")
        sys.exit(1)

    # Capture snapshot
    print("Capturing screener data from Alpaca API...")
    snapshot = capture_snapshot(
        client,
        logger,
        most_actives_top_n=50,
        movers_top_n=20,
    )

    # Save
    output_dir = Path("data/screener_snapshots")
    output_path = save_snapshot(snapshot, output_dir)

    # Summary
    print()
    print("Snapshot captured:")
    print(f"  Timestamp: {snapshot['timestamp']}")
    print(f"  Most actives: {len(snapshot['most_actives'])}")
    print(f"  Gainers: {len(snapshot['gainers'])}")
    print(f"  Losers: {len(snapshot['losers'])}")
    print(f"  Saved to: {output_path}")

    # Show samples
    if snapshot["most_actives"]:
        print(f"  Sample actives: {', '.join(snapshot['most_actives'][:5])}...")
    if snapshot["gainers"]:
        print(f"  Sample gainers: {', '.join(snapshot['gainers'][:5])}...")


if __name__ == "__main__":
    main()

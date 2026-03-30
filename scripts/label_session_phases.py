"""Label 1-minute bars with intraday session phases based on ET time of day.

Reads parquet files from data/bars_2023_2025/, assigns a session_phase label
to each bar, and writes slim parquet files to data/regime_labeled/.
"""

# Suppress all logging
import logging  # noqa: E402
import sys
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

logging.disable(logging.CRITICAL)
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

INPUT_DIR = Path("data/bars_2023_2025")
OUTPUT_DIR = Path("data/regime_labeled")

PHASE_RANGES = [
    ("pre_market", 4, 0, 9, 29),
    ("opening_15m", 9, 30, 9, 44),
    ("morning", 9, 45, 11, 29),
    ("midday", 11, 30, 13, 59),
    ("power_hour", 14, 0, 15, 44),
    ("close_15m", 15, 45, 15, 59),
    ("after_hours", 16, 0, 19, 59),
]


def classify_phase(hour: int, minute: int) -> str:
    t = hour * 60 + minute
    for label, h1, m1, h2, m2 in PHASE_RANGES:
        if h1 * 60 + m1 <= t <= h2 * 60 + m2:
            return label
    return "outside"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*_1Min.parquet"))
    if not files:
        print("No parquet files found in", INPUT_DIR)
        sys.exit(1)

    total_bars = 0
    phase_counts: dict[str, int] = {}
    symbols_processed = 0

    for path in files:
        symbol = path.name.replace("_1Min.parquet", "")
        df = pd.read_parquet(path, columns=["timestamp", "symbol"])

        # Convert UTC timestamps to ET
        ts_et = df["timestamp"].dt.tz_convert("America/New_York")
        df["session_phase"] = [classify_phase(t.hour, t.minute) for t in ts_et]

        # Drop bars outside defined phases
        df = df[df["session_phase"] != "outside"].reset_index(drop=True)

        out_path = OUTPUT_DIR / f"{symbol}_1m_session.parquet"
        df.to_parquet(out_path, index=False)

        total_bars += len(df)
        for phase, count in df["session_phase"].value_counts().items():
            phase_counts[phase] = phase_counts.get(phase, 0) + count
        symbols_processed += 1

    # Summary
    print(f"\n{'=' * 50}")
    print("Session Phase Labeling Complete")
    print(f"{'=' * 50}")
    print(f"Symbols processed: {symbols_processed}")
    print(f"Total bars labeled: {total_bars:,}")
    print("\nPhase distribution:")
    for phase, _, _, _, _ in PHASE_RANGES:
        count = phase_counts.get(phase, 0)
        pct = count / total_bars * 100 if total_bars else 0
        print(f"  {phase:<15s} {count:>10,}  ({pct:5.1f}%)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

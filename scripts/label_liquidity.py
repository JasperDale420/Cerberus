"""Label per-symbol liquidity regimes from existing daily regime bar data."""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

_DEFAULT_REGIME_DIR = Path(__file__).resolve().parent.parent / "data" / "regime_labeled"
REGIME_DIR = Path(os.environ.get("CERBERUS_REGIME_DIR", _DEFAULT_REGIME_DIR))
ROLLING_WINDOW = 20
LOW_DOLLAR_VOL_THRESHOLD = 10_000_000  # $10M


def classify_liquidity(ratio: float) -> str:
    if ratio < 0.3:
        return "DRY"
    if ratio < 0.7:
        return "THIN"
    if ratio <= 1.5:
        return "NORMAL"
    return "LIQUID"


def process_symbol(path: Path) -> pd.DataFrame | None:
    df = pd.read_parquet(path)
    if df.empty or "volume" not in df.columns:
        return None

    df = df.sort_values("date").reset_index(drop=True)

    avg_vol = df["volume"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df["volume_ratio"] = (df["volume"] / avg_vol).round(4)
    df["liquidity_regime"] = df["volume_ratio"].apply(classify_liquidity)
    df["dollar_volume"] = (df["close"] * df["volume"]).round(2)
    df["low_dollar_vol"] = df["dollar_volume"] < LOW_DOLLAR_VOL_THRESHOLD

    df.to_parquet(path, index=False)
    return df


def main() -> None:
    parquet_files = sorted(REGIME_DIR.glob("*_daily_regime.parquet"))
    if not parquet_files:
        print("No regime parquet files found.")
        sys.exit(1)

    all_regimes: list[str] = []
    thin_dry_pcts: list[tuple[str, float]] = []
    updated = 0

    for path in parquet_files:
        symbol = path.stem.replace("_daily_regime", "")
        df = process_symbol(path)
        if df is None:
            continue
        updated += 1

        regimes = df["liquidity_regime"].tolist()
        all_regimes.extend(regimes)

        n = len(regimes)
        thin_dry = sum(1 for r in regimes if r in ("THIN", "DRY"))
        thin_dry_pcts.append((symbol, round(100.0 * thin_dry / n, 1) if n else 0.0))

    # --- Summary ---
    total = len(all_regimes)
    counts = pd.Series(all_regimes).value_counts()
    print(f"\nSymbols updated: {updated}")
    print(f"Total bar-days:  {total}\n")

    print("Aggregate liquidity distribution:")
    for regime in ["LIQUID", "NORMAL", "THIN", "DRY"]:
        c = counts.get(regime, 0)
        pct = 100.0 * c / total if total else 0
        print(f"  {regime:8s}  {c:>6,d}  ({pct:5.1f}%)")

    thin_dry_pcts.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 symbols by THIN+DRY %:")
    for symbol, pct in thin_dry_pcts[:10]:
        print(f"  {symbol:8s}  {pct:5.1f}%")


if __name__ == "__main__":
    main()

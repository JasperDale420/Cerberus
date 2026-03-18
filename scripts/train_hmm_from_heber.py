"""Train the HMM regime model from Heber Silver bar data.

Reads SPY minute bars from the Heber volume, aggregates to daily OHLCV,
then trains the pomegranate DenseHMM and saves artifacts.

Usage:
    cd /Users/jacobmcmillan/Empire/Cerberus
    python scripts/train_hmm_from_heber.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.regime_models.hmm.config import HmmConfig  # noqa: E402
from src.regime_models.hmm.service import HmmRegimeService  # noqa: E402

HEBER_DATA_ROOT = Path("/Volumes/heber/data")
SILVER_BARS_PATH = HEBER_DATA_ROOT / "silver" / "feed=bars" / "instrument_type=equity"
DEFAULT_SYMBOL = "equity:SPY"


class _TrainLogger:
    def info(self, msg: str, **kwargs: Any) -> None:
        print(json.dumps({"level": "info", "msg": msg, **kwargs}, default=str))

    def warning(self, msg: str, **kwargs: Any) -> None:
        print(json.dumps({"level": "warning", "msg": msg, **kwargs}, default=str))

    def error(self, msg: str, **kwargs: Any) -> None:
        print(json.dumps({"level": "error", "msg": msg, **kwargs}, default=str))


def _coerce_dict_columns(table: pa.Table) -> pa.Table:
    """Cast dictionary-encoded string columns to plain string."""
    arrays: list[pa.ChunkedArray] = []
    fields: list[pa.Field] = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_dictionary(field.type):
            col = col.cast(pa.string())
            field = field.with_type(pa.string())
        arrays.append(col)
        fields.append(field)
    return pa.table(arrays, schema=pa.schema(fields))


def load_bars_from_heber(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Read all SPY minute bars from Heber Silver and aggregate to daily OHLCV."""
    if not SILVER_BARS_PATH.exists():
        raise FileNotFoundError(f"Heber Silver bars path not found: {SILVER_BARS_PATH}")

    print(f"Reading bars from {SILVER_BARS_PATH} for {symbol}...")

    dataset = ds.dataset(
        str(SILVER_BARS_PATH),
        format="parquet",
        partitioning=ds.partitioning(flavor="hive"),
    )

    # Read only the columns we need with instrument_key filter
    table = dataset.to_table(
        columns=["instrument_key", "ts_event", "open", "high", "low", "close", "volume"],
        filter=ds.field("instrument_key") == symbol,
    )

    table = _coerce_dict_columns(table)
    df = table.to_pandas()
    print(f"  Raw minute bars: {len(df):,}")

    if df.empty:
        raise ValueError(f"No bars found for {symbol}")

    # Sort by timestamp for proper first/last aggregation
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df["date"] = df["ts_event"].dt.date
    df = df.sort_values("ts_event")

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

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["time"] = pd.to_datetime(daily["date"])
    daily = daily.drop(columns=["date"])

    # Cast numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

    daily = daily.dropna(subset=["open", "high", "low", "close", "volume"])

    print(f"  Daily OHLCV bars: {len(daily):,}")
    print(f"  Date range: {daily['time'].min()} to {daily['time'].max()}")

    return daily


def main() -> None:
    logger = _TrainLogger()

    # Load and aggregate bar data
    daily_bars = load_bars_from_heber()

    # Build HMM config
    hmm_cfg = HmmConfig(
        artifact_dir=str(REPO_ROOT / "artifacts" / "regime_models" / "hmm" / "latest"),
    )

    # Train
    service = HmmRegimeService(config=hmm_cfg, logger=logger)
    summary = service.fit_ohlcv(daily_bars)

    # Predict on training data (for inspection)
    predictions = service.predict_ohlcv(daily_bars)

    # Save model artifacts
    artifact_dir = service.save()

    # Save predictions and summary alongside the model
    predictions_path = Path(artifact_dir) / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    summary_path = Path(artifact_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "HMM training complete",
        artifact_dir=str(artifact_dir),
        predictions_path=str(predictions_path),
        summary_path=str(summary_path),
        n_samples=summary["n_samples"],
        n_states=summary["n_states"],
        state_labels=summary["state_labels"],
        feature_columns=summary["feature_columns"],
    )

    # Print regime distribution
    print("\n--- Regime Distribution ---")
    regime_counts = predictions["hidden_state_name"].value_counts()
    for name, count in regime_counts.items():
        pct = count / len(predictions) * 100
        print(f"  {name}: {count} days ({pct:.1f}%)")

    print(f"\nArtifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()

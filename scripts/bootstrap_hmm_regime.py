from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.regime_models.hmm.config import HmmConfig
from src.regime_models.hmm.service import HmmRegimeService


class _BootstrapLogger:
    def info(self, msg: str, **kwargs: Any) -> None:
        print(json.dumps({"level": "info", "msg": msg, **kwargs}, default=str))

    def warning(self, msg: str, **kwargs: Any) -> None:
        print(json.dumps({"level": "warning", "msg": msg, **kwargs}, default=str))

    def error(self, msg: str, **kwargs: Any) -> None:
        print(json.dumps({"level": "error", "msg": msg, **kwargs}, default=str))


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, parse_dates=["time"])
    raise ValueError(f"Unsupported HMM input file type: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a pomegranate HMM regime model from OHLCV data.")
    parser.add_argument("--config", default="config/config.yaml", help="Cerberus config path")
    parser.add_argument("--input", required=True, help="CSV or parquet OHLCV file")
    parser.add_argument("--output", default="", help="Artifact output directory override")
    args = parser.parse_args()

    config_path = Path(args.config)
    raw_config = _load_config(config_path)
    hmm_cfg = HmmConfig.model_validate(raw_config.get("hmm_regime", {}))
    if args.output:
        hmm_cfg = hmm_cfg.model_copy(update={"artifact_dir": args.output})

    frame = _load_frame(Path(args.input))
    logger = _BootstrapLogger()
    service = HmmRegimeService(config=hmm_cfg, logger=logger)
    summary = service.fit_ohlcv(frame)
    predictions = service.predict_ohlcv(frame)
    artifact_dir = service.save()

    predictions_path = Path(artifact_dir) / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    summary_path = Path(artifact_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "HMM bootstrap complete",
        artifact_dir=str(artifact_dir),
        predictions_path=str(predictions_path),
        summary_path=str(summary_path),
        n_samples=summary["n_samples"],
        n_states=summary["n_states"],
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.regime_models.hmm.config import HmmFeatureConfig


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required HMM columns: {missing}")


def build_hmm_feature_frame(frame: pd.DataFrame, config: HmmFeatureConfig | None = None) -> pd.DataFrame:
    cfg = config or HmmFeatureConfig()

    _require_columns(
        frame,
        [
            cfg.time_column,
            cfg.high_column,
            cfg.low_column,
            cfg.close_column,
            cfg.volume_column,
        ],
    )

    ordered = frame.sort_values(cfg.time_column).reset_index(drop=True).copy()
    close = ordered[cfg.close_column].astype(float)
    high = ordered[cfg.high_column].astype(float)
    low = ordered[cfg.low_column].astype(float)
    volume = ordered[cfg.volume_column].astype(float)

    log_price = pd.Series(np.log(close.replace(0.0, np.nan)), index=ordered.index)
    ret_1 = log_price.diff(cfg.return_windows[0])
    ret_5 = log_price.diff(cfg.return_windows[1])
    realized_vol = ret_1.rolling(cfg.realized_vol_window).std(ddof=0)
    hl_range_pct = (high - low) / close.replace(0.0, np.nan)

    volume_mean = volume.rolling(cfg.volume_window).mean()
    volume_std = volume.rolling(cfg.volume_window).std(ddof=0).replace(0.0, np.nan)
    volume_zscore = (volume - volume_mean) / volume_std

    features = pd.DataFrame(
        {
            "time": ordered[cfg.time_column],
            "log_return_1": ret_1,
            "log_return_5": ret_5,
            "realized_vol_20": realized_vol,
            "hl_range_pct": hl_range_pct,
            "volume_zscore_20": volume_zscore,
        }
    )
    features = features.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if features.empty:
        raise ValueError("HMM feature frame is empty after warmup/dropna")
    return features

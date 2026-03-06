from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.regime_models.hmm.artifacts import load_hmm_artifacts, save_hmm_artifacts
from src.regime_models.hmm.config import HmmConfig
from src.regime_models.hmm.features import build_hmm_feature_frame
from src.regime_models.hmm.service import HmmRegimeService


def _build_ohlcv_frame(rows: int = 64) -> pd.DataFrame:
    start = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    records: list[dict[str, Any]] = []
    price = 100.0
    for i in range(rows):
        ts = start + timedelta(minutes=i)
        drift = 0.25 if i < rows // 2 else -0.15
        price += drift
        records.append(
            {
                "time": ts,
                "open": price - 0.1,
                "high": price + 0.4,
                "low": price - 0.5,
                "close": price,
                "volume": 1_000 + (i * 15),
            }
        )
    frame = pd.DataFrame.from_records(records)
    return frame.sample(frac=1.0, random_state=7).reset_index(drop=True)


@pytest.mark.unit
def test_hmm_config_defaults_to_shadow_ab_mode() -> None:
    cfg = HmmConfig.model_validate({})

    assert cfg.runtime.enabled is False
    assert cfg.runtime.mode == "shadow"
    assert cfg.runtime.compare_to_rule_based is True
    assert cfg.training.n_states == 3
    assert cfg.features.time_column == "time"


@pytest.mark.unit
def test_build_hmm_feature_frame_returns_sorted_nan_free_frame() -> None:
    frame = _build_ohlcv_frame()

    features = build_hmm_feature_frame(frame)

    assert list(features.columns) == [
        "time",
        "log_return_1",
        "log_return_5",
        "realized_vol_20",
        "hl_range_pct",
        "volume_zscore_20",
    ]
    assert features["time"].is_monotonic_increasing
    assert features["time"].min() > frame["time"].min()
    assert np.isfinite(features.drop(columns=["time"]).to_numpy()).all()


@dataclass
class _FakeAdapter:
    n_states: int = 3
    was_fit: bool = False

    def fit(self, matrix: np.ndarray) -> None:
        assert matrix.ndim == 2
        self.was_fit = True

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        assert self.was_fit is True
        return np.where(matrix[:, 0] >= 0.0, 0, 2)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        assert self.was_fit is True
        out = np.full((len(matrix), self.n_states), 0.05)
        state_idx = self.predict(matrix)
        out[np.arange(len(matrix)), state_idx] = 0.9
        return out

    def dumps(self) -> bytes:
        return b"fake-hmm-model"

    @classmethod
    def loads(cls, payload: bytes) -> "_FakeAdapter":
        assert payload == b"fake-hmm-model"
        model = cls()
        model.was_fit = True
        return model


@pytest.mark.unit
def test_hmm_service_trains_and_predicts_parallel_hidden_states() -> None:
    service = HmmRegimeService(
        config=HmmConfig.model_validate({"runtime": {"enabled": True, "mode": "shadow"}}),
        logger=MagicMock(),
        adapter_factory=lambda training_cfg: _FakeAdapter(n_states=training_cfg.n_states),
    )

    frame = _build_ohlcv_frame()
    fit_summary = service.fit_ohlcv(frame)
    predictions = service.predict_ohlcv(frame)

    assert fit_summary["n_samples"] > 0
    assert fit_summary["n_states"] == 3
    assert set(["hidden_state", "hidden_state_name", "hidden_state_confidence"]).issubset(predictions.columns)
    assert (
        predictions["hidden_state_name"].isin(["trend_up", "neutral", "stress_down", "stress_up", "trend_down"]).all()
    )
    assert predictions["hidden_state_confidence"].between(0.0, 1.0).all()


@pytest.mark.unit
def test_hmm_artifacts_round_trip(tmp_path: Path) -> None:
    metadata = {
        "model_name": "cerberus-hmm",
        "n_states": 3,
        "feature_columns": ["log_return_1", "realized_vol_20"],
    }
    save_hmm_artifacts(tmp_path, model_payload=b"fake-hmm-model", metadata=metadata)
    loaded = load_hmm_artifacts(tmp_path)

    assert loaded.metadata == metadata
    assert loaded.model_payload == b"fake-hmm-model"

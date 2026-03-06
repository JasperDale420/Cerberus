from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from src.regime_models.hmm.adapters import PomegranateDenseHmmAdapter
from src.regime_models.hmm.artifacts import load_hmm_artifacts, save_hmm_artifacts
from src.regime_models.hmm.config import HmmConfig, HmmTrainingConfig
from src.regime_models.hmm.features import build_hmm_feature_frame
from src.regime_models.hmm.labeling import label_hidden_states

if TYPE_CHECKING:
    from src.core.logger import StructuredLogger


class HmmRegimeService:
    def __init__(
        self,
        config: HmmConfig,
        logger: "StructuredLogger" | Any,
        adapter_factory: Callable[[HmmTrainingConfig], Any] | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.adapter_factory = adapter_factory or (lambda training_cfg: PomegranateDenseHmmAdapter(training_cfg))
        self.adapter: Any | None = None
        self._feature_columns: list[str] = []
        self._state_labels: dict[int, str] = {}

    def fit_ohlcv(self, frame: pd.DataFrame) -> dict[str, Any]:
        features = build_hmm_feature_frame(frame, self.config.features)
        matrix = features.drop(columns=["time"]).to_numpy(dtype=np.float64)
        if len(matrix) < self.config.training.min_samples:
            raise ValueError(f"Need at least {self.config.training.min_samples} HMM samples, got {len(matrix)}")

        self.adapter = self.adapter_factory(self.config.training)
        self.adapter.fit(matrix)
        states = self.adapter.predict(matrix)
        self._feature_columns = [column for column in features.columns if column != "time"]
        self._state_labels = label_hidden_states(features, states)

        if self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info(
                "HMM regime model trained",
                model_name=self.config.model_name,
                mode=self.config.runtime.mode,
                n_samples=len(matrix),
                n_states=self.config.training.n_states,
                feature_columns=self._feature_columns,
            )

        return {
            "model_name": self.config.model_name,
            "mode": self.config.runtime.mode,
            "n_samples": int(len(matrix)),
            "n_features": int(matrix.shape[1]),
            "n_states": int(self.config.training.n_states),
            "feature_columns": self._feature_columns,
            "state_labels": self._state_labels,
        }

    def predict_ohlcv(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.adapter is None:
            raise RuntimeError("HMM regime model must be fit before prediction")

        features = build_hmm_feature_frame(frame, self.config.features)
        matrix = features.drop(columns=["time"]).to_numpy(dtype=np.float64)
        states = self.adapter.predict(matrix)
        probs = self.adapter.predict_proba(matrix)
        confidence = probs.max(axis=1)

        output = features[["time"]].copy()
        output["hidden_state"] = states.astype(int)
        output["hidden_state_name"] = [self._state_labels.get(int(state), "neutral") for state in states]
        output["hidden_state_confidence"] = confidence.astype(float)
        return output

    def save(self, path: str | Path | None = None) -> Path:
        if self.adapter is None:
            raise RuntimeError("HMM regime model must be fit before save")

        artifact_dir = Path(path or self.config.artifact_dir)
        metadata = {
            "model_name": self.config.model_name,
            "feature_columns": self._feature_columns,
            "n_states": self.config.training.n_states,
            "state_labels": self._state_labels,
            "runtime_mode": self.config.runtime.mode,
        }
        save_hmm_artifacts(artifact_dir, model_payload=self.adapter.dumps(), metadata=metadata)
        return artifact_dir

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: HmmConfig,
        logger: "StructuredLogger" | Any,
        adapter_factory: Callable[[HmmTrainingConfig], Any] | None = None,
        adapter_loader: Callable[[bytes, HmmTrainingConfig], Any] | None = None,
    ) -> "HmmRegimeService":
        artifacts = load_hmm_artifacts(path)
        service = cls(config=config, logger=logger, adapter_factory=adapter_factory)
        loader = adapter_loader or PomegranateDenseHmmAdapter.loads
        service.adapter = loader(artifacts.model_payload, config.training)
        service._feature_columns = list(artifacts.metadata.get("feature_columns", []))
        service._state_labels = {
            int(state_id): str(label) for state_id, label in dict(artifacts.metadata.get("state_labels", {})).items()
        }
        return service

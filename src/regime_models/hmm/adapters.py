from __future__ import annotations

import pickle
from typing import Any, Protocol, cast

import numpy as np

from src.regime_models.hmm.config import HmmTrainingConfig


class HiddenMarkovAdapter(Protocol):
    def fit(self, matrix: np.ndarray) -> None: ...
    def predict(self, matrix: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, matrix: np.ndarray) -> np.ndarray: ...
    def dumps(self) -> bytes: ...


class PomegranateDenseHmmAdapter:
    def __init__(self, config: HmmTrainingConfig):
        self.config = config
        self._model: Any | None = None

    def _to_sequence_tensor(self, matrix: np.ndarray):
        import torch

        data = np.asarray(matrix, dtype=np.float32)
        return torch.tensor(data[None, :, :], dtype=torch.float32)

    def fit(self, matrix: np.ndarray) -> None:
        import torch
        from pomegranate.distributions import Normal
        from pomegranate.hmm import DenseHMM

        torch.manual_seed(self.config.random_seed)
        sequence = self._to_sequence_tensor(matrix)
        dists = [Normal(covariance_type=self.config.covariance_type) for _ in range(self.config.n_states)]
        self._model = DenseHMM(dists, max_iter=self.config.max_iter, verbose=False)
        assert self._model is not None
        self._model.fit(sequence)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("HMM adapter must be fit before predict")
        sequence = self._to_sequence_tensor(matrix)
        predicted = self._model.predict(sequence)
        return cast(np.ndarray, np.asarray(predicted)[0].astype(int))

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("HMM adapter must be fit before predict_proba")
        sequence = self._to_sequence_tensor(matrix)
        probs = self._model.predict_proba(sequence)
        return cast(np.ndarray, np.asarray(probs)[0])

    def dumps(self) -> bytes:
        if self._model is None:
            raise RuntimeError("HMM adapter must be fit before serialization")
        return pickle.dumps(self._model)

    @classmethod
    def loads(cls, payload: bytes, config: HmmTrainingConfig) -> "PomegranateDenseHmmAdapter":
        adapter = cls(config=config)
        adapter._model = pickle.loads(payload)
        return adapter

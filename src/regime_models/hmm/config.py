from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HmmFeatureConfig(BaseModel):
    time_column: str = "time"
    open_column: str = "open"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"
    volume_column: str = "volume"
    return_windows: tuple[int, int] = (1, 5)
    realized_vol_window: int = 20
    volume_window: int = 20


class HmmTrainingConfig(BaseModel):
    n_states: int = Field(default=3, ge=2, le=12)
    max_iter: int = Field(default=50, ge=1)
    random_seed: int = 7
    covariance_type: Literal["diag"] = "diag"
    min_samples: int = Field(default=30, ge=10)


class HmmRuntimeConfig(BaseModel):
    enabled: bool = False
    mode: Literal["shadow", "primary"] = "shadow"
    compare_to_rule_based: bool = True
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class HmmConfig(BaseModel):
    model_name: str = "cerberus-pomegranate-hmm"
    artifact_dir: str = "artifacts/regime_models/hmm/latest"
    features: HmmFeatureConfig = Field(default_factory=HmmFeatureConfig)
    training: HmmTrainingConfig = Field(default_factory=HmmTrainingConfig)
    runtime: HmmRuntimeConfig = Field(default_factory=HmmRuntimeConfig)

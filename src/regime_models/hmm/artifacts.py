from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HmmArtifacts:
    metadata: dict[str, Any]
    model_payload: bytes


def save_hmm_artifacts(path: str | Path, model_payload: bytes, metadata: dict[str, Any]) -> None:
    artifact_dir = Path(path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "model.pkl").write_bytes(model_payload)
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def load_hmm_artifacts(path: str | Path) -> HmmArtifacts:
    artifact_dir = Path(path)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    model_payload = (artifact_dir / "model.pkl").read_bytes()
    return HmmArtifacts(metadata=metadata, model_payload=model_payload)

"""Regression tests for CI tooling configuration."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, cast

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pyproject() -> dict[str, Any]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def _load_pre_commit() -> dict[str, Any]:
    with (REPO_ROOT / ".pre-commit-config.yaml").open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], yaml.safe_load(f) or {})


def test_ruff_extend_path_exists_in_repo() -> None:
    data = _load_pyproject()
    extend_path = data["tool"]["ruff"].get("extend")
    assert isinstance(extend_path, str)

    resolved = (REPO_ROOT / extend_path).resolve()
    assert resolved.exists(), f"Ruff extend file not found: {extend_path}"


def test_detect_secrets_excludes_archive_snapshots() -> None:
    data = _load_pre_commit()
    detect_hook = None

    for repo in data.get("repos", []):
        for hook in repo.get("hooks", []):
            if hook.get("id") == "detect-secrets":
                detect_hook = hook
                break
        if detect_hook:
            break

    assert detect_hook is not None, "detect-secrets hook missing"

    exclude_pattern = detect_hook.get("exclude")
    assert isinstance(exclude_pattern, str)

    regex = re.compile(exclude_pattern)
    assert regex.search("microsoft-rd-agent-8a5edab282632443.txt")
    assert regex.search("microsoft-qlib-8a5edab282632443.txt")

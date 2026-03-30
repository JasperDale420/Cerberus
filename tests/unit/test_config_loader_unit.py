from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.core.config import ConfigLoader


@pytest.mark.unit
def test_deep_merge_merges_nested_dicts() -> None:
    loader = ConfigLoader(config_dir="config")
    base = {"a": {"b": 1, "c": 2}, "x": 1}
    override = {"a": {"c": 3, "d": 4}, "y": 2}
    loader._deep_merge(base, override)
    assert base == {"a": {"b": 1, "c": 3, "d": 4}, "x": 1, "y": 2}


@pytest.mark.unit
def test_load_config_loads_yaml_suite_and_applies_env_overrides(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"risk": {"max_daily_loss": 1}, "log_level": "INFO"}))
    (tmp_path / "scanner.yaml").write_text(yaml.safe_dump({"scanner": {"top_n": 10}}))
    (tmp_path / "risk.yaml").write_text(yaml.safe_dump({"risk": {"max_risk_per_trade": 5}}))

    # Double-underscore separates nesting levels; single underscores are preserved in key names.
    # Safety-critical risk keys (max_daily_loss, etc.) are BLOCKED from env override.
    monkeypatch.setenv("APP_RISK__MAX_DAILY_LOSS", "2")  # blocked — should NOT override
    monkeypatch.setenv("APP_SCANNER__TOP_N", "20")  # allowed
    monkeypatch.setenv("APP_LOG_LEVEL", "DEBUG")  # allowed

    loader = ConfigLoader(config_dir=str(tmp_path))
    cfg = loader.load_config()
    assert cfg["risk"]["max_daily_loss"] == 1  # NOT overridden — blocked by safety guard
    assert cfg["risk"]["max_risk_per_trade"] == 5
    assert cfg["scanner"]["top_n"] == 20  # overridden via env
    assert cfg["log_level"] == "DEBUG"  # overridden via env


@pytest.mark.unit
def test_blocked_risk_keys_cannot_be_overridden(tmp_path: Path, monkeypatch) -> None:
    """Safety-critical risk keys must be immune to APP_* env var overrides."""
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"risk": {"max_daily_loss": 500, "max_open_positions": 5, "risk_mode": "normal"}})
    )

    monkeypatch.setenv("APP_RISK__MAX_DAILY_LOSS", "99999")
    monkeypatch.setenv("APP_RISK__MAX_OPEN_POSITIONS", "100")
    monkeypatch.setenv("APP_RISK__RISK_MODE", "off")

    loader = ConfigLoader(config_dir=str(tmp_path))
    cfg = loader.load_config()
    assert cfg["risk"]["max_daily_loss"] == 500  # unchanged
    assert cfg["risk"]["max_open_positions"] == 5  # unchanged
    assert cfg["risk"]["risk_mode"] == "normal"  # unchanged


@pytest.mark.unit
def test_load_config_supports_json_suite(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"risk": {"max_daily_loss": 3}}, sort_keys=True))
    (tmp_path / "risk.json").write_text(json.dumps({"risk": {"max_risk_per_trade": 7}}, sort_keys=True))
    (tmp_path / "scanner.json").write_text(json.dumps({"scanner": {"top_n": 11}}, sort_keys=True))

    loader = ConfigLoader(config_dir=str(tmp_path))
    cfg = loader.load_config()
    assert cfg["risk"]["max_daily_loss"] == 3
    assert cfg["risk"]["max_risk_per_trade"] == 7
    assert cfg["scanner"]["top_n"] == 11


@pytest.mark.unit
def test_get_env_raises_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_ENV", raising=False)
    loader = ConfigLoader(config_dir="config")
    with pytest.raises(ValueError, match="Missing required environment variable"):
        loader.get_env("MISSING_ENV")


@pytest.mark.unit
def test_get_env_returns_default_when_provided(monkeypatch) -> None:
    monkeypatch.delenv("OPTIONAL_ENV", raising=False)
    loader = ConfigLoader(config_dir="config")
    assert loader.get_env("OPTIONAL_ENV", default="x") == "x"


@pytest.mark.unit
def test_strategies_auto_overrides_accept_camelcase_keys(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({}))
    (tmp_path / "strategies.yaml").write_text(yaml.safe_dump({"strategies": {"vwap_reversion": {"band_sigma": 2.0}}}))
    (tmp_path / "strategies.auto.yaml").write_text(yaml.safe_dump({"VWAPReversion": {"params": {"band_sigma": 1.5}}}))

    loader = ConfigLoader(config_dir=str(tmp_path))
    cfg = loader.load_config()

    assert cfg["strategies"]["vwap_reversion"]["band_sigma"] == 1.5

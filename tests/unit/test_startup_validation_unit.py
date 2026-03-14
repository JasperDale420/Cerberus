"""Unit tests for startup mode environment variable validation."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.core.settings import Settings, validate_runtime_execution_requirements, validate_startup_settings


def build_settings(**overrides: Any) -> Settings:
    """Create runtime settings with overrides."""
    return Settings(**overrides)


def test_startup_requires_gateway_key() -> None:
    """Startup validation should require CERBERUS_GATEWAY_KEY."""
    settings = build_settings(CERBERUS_GATEWAY_KEY="")

    errors = settings.validate_startup_mode()

    assert len(errors) == 1
    assert "CERBERUS_GATEWAY_KEY" in errors[0]


def test_startup_passes_with_gateway_key() -> None:
    """Startup validation should pass when gateway key is provided."""
    settings = build_settings(CERBERUS_GATEWAY_KEY="valid_gateway_key_123")

    errors = settings.validate_startup_mode()

    assert len(errors) == 0


def test_validate_startup_settings_raises_on_missing_key() -> None:
    """validate_startup_settings() should raise ValueError when gateway key is missing."""
    with patch("src.core.settings.get_settings") as mock_get_settings:
        mock_get_settings.return_value = build_settings(CERBERUS_GATEWAY_KEY="")

        with pytest.raises(ValueError, match="Startup configuration validation failed"):
            validate_startup_settings()


def test_runtime_execution_requires_alpaca_creds_for_alpaca_executor() -> None:
    """Alpaca order execution should always require Alpaca credentials."""
    with patch("src.core.settings.get_settings") as mock_get_settings:
        mock_get_settings.return_value = build_settings(
            CERBERUS_GATEWAY_KEY="valid_gateway_key",
            alpaca_api_key="",
            alpaca_secret_key="",
            apca_api_key_id="",
            apca_api_secret_key="",
        )

        with pytest.raises(ValueError, match="Order executor validation failed"):
            validate_runtime_execution_requirements(order_executor="alpaca", mode="paper")


def test_runtime_execution_allows_noop_without_alpaca_creds() -> None:
    """Noop order execution should not require Alpaca credentials."""
    with patch("src.core.settings.get_settings") as mock_get_settings:
        mock_get_settings.return_value = build_settings(
            CERBERUS_GATEWAY_KEY="valid_gateway_key",
            alpaca_api_key="",
            alpaca_secret_key="",
            apca_api_key_id="",
            apca_api_secret_key="",
        )

        validate_runtime_execution_requirements(order_executor="noop", mode="paper")


def test_runtime_execution_accepts_apca_alias_credentials() -> None:
    """Alpaca executor should accept APCA_* alias credentials."""
    with patch("src.core.settings.get_settings") as mock_get_settings:
        mock_get_settings.return_value = build_settings(
            CERBERUS_GATEWAY_KEY="valid_gateway_key",
            apca_api_key_id="key_id",  # pragma: allowlist secret
            apca_api_secret_key="dummy_value",  # pragma: allowlist secret
        )

        validate_runtime_execution_requirements(order_executor="alpaca", mode="live")


def test_runtime_defaults_paper_mode() -> None:
    """Runtime defaults should default to paper trading."""
    settings = Settings()

    assert settings.alpaca_paper is True

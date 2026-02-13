"""Unit tests for startup mode environment variable validation."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.core.settings import Settings, validate_runtime_execution_requirements, validate_startup_settings


def build_settings(**overrides: Any) -> Settings:
    """Create runtime settings using environment aliases used in production."""
    defaults: dict[str, Any] = {
        "CERBERUS_DATA_BACKEND": "legacy",
        "CERBERUS_STORAGE_BACKEND": "sqlite",
        "CERBERUS_FAILOVER_TO_LEGACY": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_legacy_mode_requires_alpaca_credentials() -> None:
    """Legacy mode should require Alpaca API credentials."""
    settings = build_settings(
        CERBERUS_DATA_BACKEND="legacy",
        alpaca_api_key="",
        alpaca_secret_key="",
        apca_api_key_id="",
        apca_api_secret_key="",
    )

    errors = settings.validate_startup_mode()

    assert len(errors) >= 2
    assert any("ALPACA_API_KEY" in err for err in errors)
    assert any("ALPACA_SECRET_KEY" in err for err in errors)


def test_legacy_mode_with_apca_credentials_passes() -> None:
    """Legacy mode should accept APCA_* credential naming convention."""
    settings = build_settings(
        CERBERUS_DATA_BACKEND="legacy",
        apca_api_key_id="test_key_id",
        apca_api_secret_key="dummy_value",  # pragma: allowlist secret
    )

    errors = settings.validate_startup_mode()

    assert not any("ALPACA_API_KEY" in err for err in errors)
    assert not any("ALPACA_SECRET_KEY" in err for err in errors)


def test_gateway_mode_requires_gateway_url_and_key() -> None:
    """Gateway mode should require CERBERUS_GATEWAY_URL and CERBERUS_GATEWAY_KEY."""
    settings = build_settings(
        CERBERUS_DATA_BACKEND="gateway",
        CERBERUS_GATEWAY_URL="",
        CERBERUS_GATEWAY_KEY="",
    )

    errors = settings.validate_startup_mode()

    assert len(errors) >= 2
    assert any("CERBERUS_GATEWAY_URL" in err for err in errors)
    assert any("CERBERUS_GATEWAY_KEY" in err for err in errors)


def test_gateway_mode_with_custom_url_only_requires_key() -> None:
    """Gateway mode with a non-default URL should only require key when key is missing."""
    settings = build_settings(
        CERBERUS_DATA_BACKEND="gateway",
        CERBERUS_GATEWAY_URL="http://gateway.test",
        CERBERUS_GATEWAY_KEY="",
    )

    errors = settings.validate_startup_mode()

    assert any("CERBERUS_GATEWAY_KEY" in err for err in errors)
    assert not any("CERBERUS_GATEWAY_URL" in err for err in errors)


def test_gateway_mode_with_valid_config_passes() -> None:
    """Gateway mode should pass with valid gateway URL and key."""
    settings = build_settings(
        CERBERUS_DATA_BACKEND="gateway",
        CERBERUS_GATEWAY_URL="http://gateway.prod.example.com",
        CERBERUS_GATEWAY_KEY="valid_gateway_key_123",
    )

    errors = settings.validate_startup_mode()

    assert not any("CERBERUS_GATEWAY_URL" in err for err in errors)
    assert not any("CERBERUS_GATEWAY_KEY" in err for err in errors)


def test_dual_mode_requires_both_gateway_and_legacy_credentials() -> None:
    """Dual mode with failover should require both gateway and legacy credentials."""
    settings = build_settings(
        CERBERUS_DATA_BACKEND="dual",
        CERBERUS_FAILOVER_TO_LEGACY=True,
        CERBERUS_GATEWAY_URL="http://localhost:8080",
        CERBERUS_GATEWAY_KEY="",
        alpaca_api_key="",
        alpaca_secret_key="",
        apca_api_key_id="",
        apca_api_secret_key="",
    )

    errors = settings.validate_startup_mode()

    assert len(errors) >= 4
    assert any("CERBERUS_GATEWAY_URL" in err for err in errors)
    assert any("CERBERUS_GATEWAY_KEY" in err for err in errors)
    assert any("ALPACA_API_KEY" in err for err in errors)
    assert any("ALPACA_SECRET_KEY" in err for err in errors)


def test_dual_mode_without_failover_only_requires_gateway() -> None:
    """Dual mode without failover should only require gateway credentials."""
    settings = build_settings(
        CERBERUS_DATA_BACKEND="dual",
        CERBERUS_FAILOVER_TO_LEGACY=False,
        CERBERUS_GATEWAY_URL="http://gateway.example.com",
        CERBERUS_GATEWAY_KEY="valid_key",
    )

    errors = settings.validate_startup_mode()

    assert not any("ALPACA_API_KEY" in err for err in errors)
    assert not any("ALPACA_SECRET_KEY" in err for err in errors)


def test_heber_mode_requires_catalog_url() -> None:
    """Heber storage mode should require CERBERUS_HEBER_CATALOG_URL."""
    settings = build_settings(
        CERBERUS_DATA_BACKEND="legacy",
        CERBERUS_STORAGE_BACKEND="heber",
        CERBERUS_HEBER_CATALOG_URL="",
        alpaca_api_key="test_key",  # pragma: allowlist secret
        alpaca_secret_key="dummy_value",  # pragma: allowlist secret
    )

    errors = settings.validate_startup_mode()

    assert any("CERBERUS_HEBER_CATALOG_URL" in err for err in errors)


def test_heber_dual_mode_requires_catalog_url() -> None:
    """Dual storage mode should require CERBERUS_HEBER_CATALOG_URL."""
    settings = build_settings(
        CERBERUS_DATA_BACKEND="legacy",
        CERBERUS_STORAGE_BACKEND="dual",
        CERBERUS_HEBER_CATALOG_URL="",
        alpaca_api_key="test_key",  # pragma: allowlist secret
        alpaca_secret_key="dummy_value",  # pragma: allowlist secret
    )

    errors = settings.validate_startup_mode()

    assert any("CERBERUS_HEBER_CATALOG_URL" in err for err in errors)


def test_complete_valid_gateway_heber_config_passes() -> None:
    """Complete valid gateway+heber configuration should pass validation."""
    settings = build_settings(
        CERBERUS_DATA_BACKEND="gateway",
        CERBERUS_STORAGE_BACKEND="heber",
        CERBERUS_GATEWAY_URL="http://gateway.prod.example.com",
        CERBERUS_GATEWAY_KEY="valid_gateway_key",
        CERBERUS_HEBER_CATALOG_URL="http://heber.prod.example.com/catalog",
        CERBERUS_HEBER_DATA_ROOT="/data/heber/silver",
    )

    errors = settings.validate_startup_mode()

    assert len(errors) == 0


def test_validate_startup_settings_raises_on_invalid_config() -> None:
    """validate_startup_settings() should raise ValueError with error details."""
    with patch("src.core.settings.get_settings") as mock_get_settings:
        mock_get_settings.return_value = build_settings(
            CERBERUS_DATA_BACKEND="gateway",
            CERBERUS_GATEWAY_URL="http://localhost:8080",
            CERBERUS_GATEWAY_KEY="",
        )

        with pytest.raises(ValueError, match="Startup configuration validation failed"):
            validate_startup_settings()


def test_runtime_execution_requires_alpaca_creds_for_alpaca_executor_even_in_gateway_mode() -> None:
    """Alpaca order execution should always require Alpaca credentials."""
    with patch("src.core.settings.get_settings") as mock_get_settings:
        mock_get_settings.return_value = build_settings(
            CERBERUS_DATA_BACKEND="gateway",
            CERBERUS_GATEWAY_URL="http://gateway.example.com",
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
            CERBERUS_DATA_BACKEND="gateway",
            CERBERUS_GATEWAY_URL="http://gateway.example.com",
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
            CERBERUS_DATA_BACKEND="gateway",
            CERBERUS_GATEWAY_URL="http://gateway.example.com",
            CERBERUS_GATEWAY_KEY="valid_gateway_key",
            apca_api_key_id="key_id",  # pragma: allowlist secret
            apca_api_secret_key="dummy_value",  # pragma: allowlist secret
        )

        validate_runtime_execution_requirements(order_executor="alpaca", mode="live")


def test_use_gateway_data_property() -> None:
    """Test use_gateway_data property returns correct values."""
    legacy = build_settings(CERBERUS_DATA_BACKEND="legacy")
    gateway = build_settings(CERBERUS_DATA_BACKEND="gateway")
    dual = build_settings(CERBERUS_DATA_BACKEND="dual")

    assert legacy.use_gateway_data is False
    assert gateway.use_gateway_data is True
    assert dual.use_gateway_data is True


def test_use_heber_storage_property() -> None:
    """Test use_heber_storage property returns correct values."""
    sqlite = build_settings(CERBERUS_STORAGE_BACKEND="sqlite")
    heber = build_settings(CERBERUS_STORAGE_BACKEND="heber")
    dual = build_settings(CERBERUS_STORAGE_BACKEND="dual")

    assert sqlite.use_heber_storage is False
    assert heber.use_heber_storage is True
    assert dual.use_heber_storage is True


def test_runtime_defaults_prefer_gateway_and_paper_mode() -> None:
    """Runtime defaults should prefer gateway mode and paper trading."""
    settings = Settings()

    assert settings.cerberus_data_backend == "gateway"
    assert settings.alpaca_paper is True

"""Centralized runtime settings for Cerberus.

Covers environment variables not managed by the YAML ConfigLoader:
- Alpaca API credentials and connection mode
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Cerberus runtime environment settings."""

    model_config = {"env_prefix": "", "case_sensitive": False}

    # Alpaca credentials (support both naming conventions)
    alpaca_api_key: str | None = Field(default=None)
    apca_api_key_id: str | None = Field(default=None)
    alpaca_secret_key: str | None = Field(default=None)
    apca_api_secret_key: str | None = Field(default=None)
    alpaca_base_url: str | None = Field(default=None)
    apca_api_base_url: str | None = Field(default=None)
    alpaca_paper: bool = Field(default=True)

    # Data/Storage backend routing
    cerberus_data_backend: Literal["legacy", "gateway", "dual"] = Field(
        default="gateway",
        validation_alias=AliasChoices("CERBERUS_DATA_BACKEND"),
    )
    cerberus_storage_backend: Literal["sqlite", "heber", "dual"] = Field(
        default="sqlite",
        validation_alias=AliasChoices("CERBERUS_STORAGE_BACKEND"),
    )
    cerberus_dual_read_compare: bool = Field(
        default=False,
        validation_alias=AliasChoices("CERBERUS_DUAL_READ_COMPARE"),
    )
    cerberus_failover_to_legacy: bool = Field(
        default=True,
        validation_alias=AliasChoices("CERBERUS_FAILOVER_TO_LEGACY"),
    )

    # Data Gateway
    cerberus_gateway_url: str = Field(
        default="http://localhost:8080",
        validation_alias=AliasChoices("CERBERUS_GATEWAY_URL", "DATA_INGESTION_URL"),
    )
    cerberus_gateway_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "CERBERUS_GATEWAY_KEY",
            "GATEWAY_API_KEY",
            "X_GATEWAY_KEY",
        ),
    )
    cerberus_gateway_timeout_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices("CERBERUS_GATEWAY_TIMEOUT_SECONDS"),
    )

    # Heber
    cerberus_heber_catalog_url: str = Field(
        default="",
        validation_alias=AliasChoices("CERBERUS_HEBER_CATALOG_URL", "HEBER_CATALOG_URL"),
    )
    cerberus_heber_data_root: str = Field(
        default="",
        validation_alias=AliasChoices("CERBERUS_HEBER_DATA_ROOT", "HEBER_DATA_ROOT"),
    )

    @property
    def resolved_api_key(self) -> str | None:
        """Resolve API key from either naming convention."""
        return self.alpaca_api_key or self.apca_api_key_id

    @property
    def resolved_secret_key(self) -> str | None:
        """Resolve secret key from either naming convention."""
        return self.alpaca_secret_key or self.apca_api_secret_key

    @property
    def resolved_base_url(self) -> str | None:
        """Resolve base URL from either naming convention."""
        return self.alpaca_base_url or self.apca_api_base_url

    @property
    def use_gateway_data(self) -> bool:
        """Return True when Data-Gateway should be used for market-data reads."""
        return self.cerberus_data_backend in {"gateway", "dual"}

    @property
    def use_heber_storage(self) -> bool:
        """Return True when Heber should be used for storage reads."""
        return self.cerberus_storage_backend in {"heber", "dual"}

    def validate_startup_mode(self) -> list[str]:
        """Validate required environment variables for configured backend modes.

        Returns:
            List of validation error messages (empty if all valid).
        """
        errors: list[str] = []

        # Gateway mode validation
        if self.use_gateway_data:
            if not self.cerberus_gateway_url or self.cerberus_gateway_url == "http://localhost:8080":
                errors.append("CERBERUS_GATEWAY_URL must be set when using gateway or dual data backend mode")
            if not self.cerberus_gateway_key:
                errors.append("CERBERUS_GATEWAY_KEY must be set when using gateway or dual data backend mode")

        # Heber mode validation
        if self.use_heber_storage:
            if not self.cerberus_heber_catalog_url:
                errors.append("CERBERUS_HEBER_CATALOG_URL must be set when using heber or dual storage backend mode")
            # Optional: heber_data_root validation only if using filesystem reads
            # Uncomment if required:
            # if not self.cerberus_heber_data_root:
            #     errors.append(
            #         "CERBERUS_HEBER_DATA_ROOT must be set when using heber or dual storage backend mode"
            #     )

        # Legacy mode validation (Alpaca credentials required unless gateway-only)
        if self.cerberus_data_backend == "legacy" or (
            self.cerberus_data_backend == "dual" and self.cerberus_failover_to_legacy
        ):
            if not self.resolved_api_key:
                errors.append("ALPACA_API_KEY (or APCA_API_KEY_ID) required for legacy or dual+failover mode")
            if not self.resolved_secret_key:
                errors.append("ALPACA_SECRET_KEY (or APCA_API_SECRET_KEY) required for legacy or dual+failover mode")

        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


def validate_startup_settings() -> None:
    """Validate startup settings and raise if configuration is invalid.

    Raises:
        ValueError: If required environment variables are missing for configured mode.
    """
    settings = get_settings()
    errors = settings.validate_startup_mode()

    if errors:
        error_msg = "Startup configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        raise ValueError(error_msg)

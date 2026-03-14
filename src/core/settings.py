"""Centralized runtime settings for Cerberus.

Covers environment variables not managed by the YAML ConfigLoader:
- Alpaca API credentials and connection mode
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Cerberus runtime environment settings."""

    model_config = {"env_prefix": "", "case_sensitive": False, "env_file": ".env", "extra": "ignore"}

    # Alpaca credentials (support both naming conventions)
    alpaca_api_key: str | None = Field(default=None)
    apca_api_key_id: str | None = Field(default=None)
    alpaca_secret_key: str | None = Field(default=None)
    apca_api_secret_key: str | None = Field(default=None)
    alpaca_base_url: str | None = Field(default=None)
    apca_api_base_url: str | None = Field(default=None)
    alpaca_paper: bool = Field(default=True)

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
    cerberus_asset_class: str = Field(
        default="us_equity",
        validation_alias=AliasChoices("CERBERUS_ASSET_CLASS"),
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

    def validate_startup_mode(self) -> list[str]:
        """Validate required environment variables for startup.

        Returns:
            List of validation error messages (empty if all valid).
        """
        errors: list[str] = []
        if not self.cerberus_gateway_key:
            errors.append("CERBERUS_GATEWAY_KEY is required")
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


def validate_runtime_execution_requirements(order_executor: str, mode: str) -> None:
    """Validate runtime execution requirements based on the selected order executor.

    Raises:
        ValueError: If required credentials for the selected execution mode are missing.
    """
    settings = get_settings()
    normalized_executor = str(order_executor).strip().lower()
    normalized_mode = str(mode).strip().lower()
    errors: list[str] = []

    if normalized_executor == "alpaca":
        if not settings.resolved_api_key:
            errors.append("ALPACA_API_KEY (or APCA_API_KEY_ID) required when order_executor=alpaca")
        if not settings.resolved_secret_key:
            errors.append("ALPACA_SECRET_KEY (or APCA_API_SECRET_KEY) required when order_executor=alpaca")

    if errors:
        error_msg = (
            "Order executor validation failed "
            f"(order_executor={normalized_executor}, mode={normalized_mode}):\n"
            + "\n".join(f"  - {err}" for err in errors)
        )
        raise ValueError(error_msg)

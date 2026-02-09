"""Centralized runtime settings for Cerberus.

Covers environment variables not managed by the YAML ConfigLoader:
- Alpaca API credentials and connection mode
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
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
    alpaca_paper: bool = Field(default=False)

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()

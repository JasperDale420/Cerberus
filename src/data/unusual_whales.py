from __future__ import annotations

from typing import Any, Optional

import httpx

from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


class UnusualWhalesClient:
    """
    Wrapper for Unusual Whales option flow.

    Supports two modes:
    1) `UW_API_TOKEN` (preferred): uses the bundled `unusualwhales-python-client`.
    2) `UNUSUAL_WHALES_FLOW_URL_TEMPLATE`: direct HTTP GET to a caller-provided URL template.
    """

    def __init__(
        self,
        config_loader: ConfigLoader,
        logger: StructuredLogger,
        client: Optional[httpx.AsyncClient] = None,
        config: Optional[dict] = None,
    ):
        self.logger = logger
        self.api_token = str(config_loader.get_env("UW_API_TOKEN", "")).strip()
        base_url_env = str(
            config_loader.get_env("UW_BASE_URL", "https://api.unusualwhales.com")
        ).strip()
        self.base_url = base_url_env or "https://api.unusualwhales.com"

        if isinstance(config, dict):
            uw_cfg0 = config.get("unusual_whales")
            if isinstance(uw_cfg0, dict):
                self.base_url = (
                    str(uw_cfg0.get("base_url", self.base_url)).strip() or self.base_url
                )

        # Prefer merged config if provided; allow env override.
        from_cfg = ""
        if isinstance(config, dict):
            uw_cfg = config.get("unusual_whales")
            if isinstance(uw_cfg, dict):
                from_cfg = str(uw_cfg.get("flow_url_template", "")).strip()

        from_env = str(
            config_loader.get_env("UNUSUAL_WHALES_FLOW_URL_TEMPLATE", "")
        ).strip()
        self.flow_url_template = from_env or from_cfg
        self._client = client or httpx.AsyncClient(timeout=15.0)

    async def get_option_flow(self, symbol: str, date: str) -> Any:
        """
        Fetches option flow for a symbol. If not configured, returns neutral (empty) flow.
        """
        sym = str(symbol).strip().upper()

        # Preferred: use UW API token with the bundled python client.
        # Note: this endpoint returns "latest" flow and does not accept a date parameter.
        if self.api_token:
            try:
                url = f"{self.base_url.rstrip('/')}/api/stock/{sym}/flow-recent"
                headers = {"Authorization": f"Bearer {self.api_token}"}
                resp = await self._client.get(url, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                if isinstance(payload, list):
                    return payload
                if isinstance(payload, dict):
                    data = payload.get("data")
                    if isinstance(data, list):
                        return data
                self.logger.warning(
                    "Unusual Whales flow response unexpected; using neutral flow",
                    symbol=sym,
                    date=date,
                )
                return []
            except Exception as e:
                # PRD 11.4: degrade gracefully.
                self.logger.warning(
                    "Unusual Whales flow fetch failed; using neutral flow",
                    symbol=sym,
                    date=date,
                    error=str(e),
                )
                return []

        if not self.flow_url_template:
            self.logger.warning(
                "Unusual Whales flow not configured; using neutral flow",
                symbol=sym,
                date=date,
            )
            return []

        url = self.flow_url_template.format(symbol=sym)
        try:
            resp = await self._client.get(url, params={"date": str(date)})
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if isinstance(data.get("data"), list):
                    return data["data"]
                if isinstance(data.get("trades"), list):
                    return data["trades"]
            self.logger.warning(
                "Unusual Whales flow response unexpected; using neutral flow",
                symbol=sym,
                date=date,
            )
            return []
        except Exception as e:
            # PRD 11.4: degrade gracefully.
            self.logger.warning(
                "Unusual Whales flow fetch failed; using neutral flow",
                symbol=sym,
                date=date,
                error=str(e),
            )
            return []

    async def get_greek_exposure(self, symbol: str) -> list[dict]:
        """
        Fetches Greek exposure data by strike for a symbol.
        Used for Net GEX calculation.
        """
        sym = str(symbol).strip().upper()
        if not self.api_token:
            return []

        try:
            url = f"{self.base_url.rstrip('/')}/api/stock/{sym}/greek-exposure/strike"
            headers = {"Authorization": f"Bearer {self.api_token}"}
            resp = await self._client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

            # The API usually returns {"data": [...]} or a list
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return payload.get("data") or []
            return []
        except Exception as e:
            self.logger.warning(
                "Unusual Whales greek exposure fetch failed",
                symbol=sym,
                error=str(e),
            )
            return []

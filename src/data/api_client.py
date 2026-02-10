from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import httpx

from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


class CentralApiClient:
    """
    Client for central API services (Data Gateway + optional LLM endpoint).
    """

    def __init__(self, config_loader: ConfigLoader, logger: StructuredLogger):
        self.logger = logger
        self.base_url = config_loader.get_env(
            "CERBERUS_GATEWAY_URL",
            config_loader.get_env("DATA_INGESTION_URL", "http://localhost:8080"),
        )
        self.gateway_key = config_loader.get_env("CERBERUS_GATEWAY_KEY", "")
        try:
            timeout = float(
                config_loader.get_env("CERBERUS_GATEWAY_TIMEOUT_SECONDS", "30")
            )
        except ValueError:
            timeout = 30.0

        headers: Dict[str, str] = {}
        if self.gateway_key:
            headers["X-Gateway-Key"] = self.gateway_key

        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
        )
        self.llm_base_url = config_loader.get_env("CENTRAL_LLM_API_URL", self.base_url)
        self._llm_client: Optional[httpx.Client] = None

    def _normalize_bars_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Data-Gateway envelope response into legacy bars payload."""
        if "bars" in payload:
            return payload
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("bars"), list):
            return {"bars": data["bars"]}
        return payload

    def _resolve_llm_client(self) -> httpx.Client:
        """Return the client used for chat completions."""
        if self.llm_base_url == self.base_url:
            return self.client
        if self._llm_client is None:
            self._llm_client = httpx.Client(base_url=self.llm_base_url, timeout=30.0)
        return self._llm_client

    def get_alpaca_bars(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1Day",
    ) -> Dict[str, Any]:
        """
        Fetches historical bars via Data-Gateway.
        """
        params = {"timeframe": timeframe}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        try:
            response = self.client.get(
                f"/api/v1/alpaca/stocks/{symbol.upper()}/bars",
                params=params,
            )
            response.raise_for_status()
            payload = cast(Dict[str, Any], response.json())
            return self._normalize_bars_response(payload)
        except httpx.HTTPError as e:
            self.logger.error(
                "Failed to fetch Alpaca bars from central API",
                symbol=symbol,
                error=str(e),
            )
            raise

    def get_uw_flow(self, ticker: str, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches options flow via Data-Gateway.
        """
        try:
            params: Dict[str, Any] = {}
            if date:
                params["date"] = date
            response = self.client.get(
                f"/api/v1/uw/flow/{ticker.upper()}",
                params=params or None,
            )
            response.raise_for_status()
            return cast(Dict[str, Any], response.json())
        except httpx.HTTPError as e:
            self.logger.error(
                "Failed to fetch UW flow from central API", ticker=ticker, error=str(e)
            )
            raise

    def chat_completion(
        self, model: str, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request.
        """
        payload = {"model": model, "messages": messages}
        try:
            response = self._resolve_llm_client().post(
                "/v1/chat/completions", json=payload
            )
            response.raise_for_status()
            return cast(Dict[str, Any], response.json())
        except httpx.HTTPError as e:
            self.logger.error(
                "Failed to get chat completion from central API", error=str(e)
            )
            raise

    def close(self) -> None:
        """Close underlying HTTP clients."""
        self.client.close()
        if self._llm_client is not None:
            self._llm_client.close()

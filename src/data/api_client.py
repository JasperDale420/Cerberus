from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import httpx

from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


class CentralApiClient:
    """
    Client for the centralized data ingestion service.
    """

    def __init__(self, config_loader: ConfigLoader, logger: StructuredLogger):
        self.logger = logger
        self.base_url = config_loader.get_env(
            "DATA_INGESTION_URL", "http://localhost:8000"
        )
        self.client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def get_alpaca_bars(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1Day",
    ) -> Dict[str, Any]:
        """
        Fetches historical bars from the centralized service.
        """
        params = {"timeframe": timeframe}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        try:
            response = self.client.get(f"/alpaca/bars/{symbol}", params=params)
            response.raise_for_status()
            return cast(Dict[str, Any], response.json())
        except httpx.HTTPError as e:
            self.logger.error(
                "Failed to fetch Alpaca bars from central API",
                symbol=symbol,
                error=str(e),
            )
            raise

    def get_uw_flow(self, ticker: str, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches options flow from the centralized service.
        """
        try:
            params: Dict[str, Any] = {}
            if date:
                params["date"] = date
            response = self.client.get(f"/uw/flow/{ticker}", params=params or None)
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
        Sends a chat completion request to the centralized service.
        """
        payload = {"model": model, "messages": messages}
        try:
            response = self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            return cast(Dict[str, Any], response.json())
        except httpx.HTTPError as e:
            self.logger.error(
                "Failed to get chat completion from central API", error=str(e)
            )
            raise

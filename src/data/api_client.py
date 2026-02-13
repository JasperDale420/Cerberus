import time
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
        self.base_url: str = config_loader.get_env(
            "CERBERUS_GATEWAY_URL",
            config_loader.get_env("DATA_INGESTION_URL", "http://localhost:8080"),
        )
        self.gateway_key: str = config_loader.get_env("CERBERUS_GATEWAY_KEY", "")
        try:
            timeout = float(config_loader.get_env("CERBERUS_GATEWAY_TIMEOUT_SECONDS", "30"))
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
        self.gateway_max_retries: int
        try:
            self.gateway_max_retries = max(0, int(config_loader.get_env("CERBERUS_GATEWAY_MAX_RETRIES", "1")))
        except ValueError:
            self.gateway_max_retries = 1
        self.gateway_retry_backoff_seconds: float
        try:
            self.gateway_retry_backoff_seconds = max(
                0.0,
                float(config_loader.get_env("CERBERUS_GATEWAY_RETRY_BACKOFF_SECONDS", "0.25")),
            )
        except ValueError:
            self.gateway_retry_backoff_seconds = 0.25
        self.llm_base_url: str = config_loader.get_env("CENTRAL_LLM_API_URL", self.base_url)
        self._llm_client: Optional[httpx.Client] = None

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        """Return True when an HTTP status should be retried."""
        return status_code == 429 or status_code >= 500

    def _get_retry_delay_seconds(
        self,
        attempt: int,
        response: Optional[httpx.Response] = None,
    ) -> float:
        """Calculate delay before next retry attempt."""
        if response is not None and response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        return float(self.gateway_retry_backoff_seconds) * float(2 ** max(0, attempt - 1))

    def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Execute an HTTP request with retry classification for gateway errors."""
        max_attempts = self.gateway_max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.request(method, path, params=params, json=json)
            except httpx.TransportError as exc:
                if attempt >= max_attempts:
                    raise
                delay = self._get_retry_delay_seconds(attempt)
                self.logger.warning(
                    "Gateway request retrying after transport error",
                    method=method,
                    path=path,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_seconds=delay,
                    error=str(exc),
                )
                if delay > 0:
                    time.sleep(delay)
                continue

            if response.status_code < 400:
                return response

            if response.status_code in {401, 403}:
                response.raise_for_status()

            if self._is_retryable_status(response.status_code) and attempt < max_attempts:
                delay = self._get_retry_delay_seconds(attempt, response=response)
                self.logger.warning(
                    "Gateway request retrying after HTTP error",
                    method=method,
                    path=path,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status_code=response.status_code,
                    delay_seconds=delay,
                )
                if delay > 0:
                    time.sleep(delay)
                continue

            response.raise_for_status()

        raise RuntimeError("Request retry loop exhausted unexpectedly")

    def _normalize_bars_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Data-Gateway envelope response into legacy bars payload."""
        if "bars" in payload:
            return payload
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("bars"), list):
            return {"bars": [self._normalize_bar(item) for item in data["bars"]]}
        return payload

    def _normalize_bar(self, item: Any) -> Dict[str, Any]:
        """Normalize a bar item into Cerberus-compatible keys."""
        if not isinstance(item, dict):
            return {}
        return {
            "t": item.get("t") or item.get("timestamp"),
            "o": item.get("o") if item.get("o") is not None else item.get("open"),
            "h": item.get("h") if item.get("h") is not None else item.get("high"),
            "l": item.get("l") if item.get("l") is not None else item.get("low"),
            "c": item.get("c") if item.get("c") is not None else item.get("close"),
            "v": item.get("v") if item.get("v") is not None else item.get("volume"),
        }

    def _normalize_trades_response(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize Data-Gateway trade response into legacy trade list."""
        trades: List[Dict[str, Any]] = []
        raw = payload
        data = payload.get("data")
        if isinstance(data, dict):
            raw = cast(Dict[str, Any], data)

        raw_trades = raw.get("trades")
        if not isinstance(raw_trades, list):
            return trades

        for item in raw_trades:
            if not isinstance(item, dict):
                continue
            trades.append(
                {
                    "t": item.get("t") or item.get("timestamp"),
                    "p": item.get("p") if item.get("p") is not None else item.get("price", 0.0),
                    "s": item.get("s") if item.get("s") is not None else item.get("size", 0.0),
                    "c": item.get("c") if item.get("c") is not None else item.get("conditions", []),
                    "x": item.get("x") if item.get("x") is not None else item.get("exchange", ""),
                    "z": item.get("z") if item.get("z") is not None else item.get("tape", ""),
                }
            )
        return trades

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
            response = self._request_with_retry(
                "GET",
                f"/api/v1/alpaca/stocks/{symbol.upper()}/bars",
                params=params,
            )
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
            response = self._request_with_retry(
                "GET",
                f"/api/v1/uw/flow/{ticker.upper()}",
                params=params or None,
            )
            return cast(Dict[str, Any], response.json())
        except httpx.HTTPError as e:
            self.logger.error("Failed to fetch UW flow from central API", ticker=ticker, error=str(e))
            raise

    def get_alpaca_trades(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Fetch historical trades via Data-Gateway and normalize output."""
        params: Dict[str, Any] = {"limit": int(limit)}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        try:
            response = self._request_with_retry(
                "GET",
                f"/api/v1/alpaca/stocks/{symbol.upper()}/trades",
                params=params,
            )
            payload = cast(Dict[str, Any], response.json())
            return self._normalize_trades_response(payload)
        except httpx.HTTPError as e:
            self.logger.error(
                "Failed to fetch Alpaca trades from central API",
                symbol=symbol,
                error=str(e),
            )
            raise

    def get_uw_gex(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch Unusual Whales GEX via Data-Gateway."""
        try:
            response = self._request_with_retry("GET", f"/api/v1/uw/gex/{ticker.upper()}")
            payload = cast(Dict[str, Any], response.json())
            data = payload.get("data")
            if isinstance(data, list):
                return cast(List[Dict[str, Any]], data)
            if isinstance(data, dict):
                return cast(List[Dict[str, Any]], data.get("rows", []))
            return []
        except httpx.HTTPError as e:
            self.logger.error(
                "Failed to fetch UW GEX from central API",
                ticker=ticker,
                error=str(e),
            )
            raise

    def get_alpaca_most_actives(self, top: int = 20) -> List[str]:
        """Fetch most active stock symbols via Data-Gateway screener."""
        try:
            response = self._request_with_retry(
                "GET",
                "/api/v1/alpaca/screener/most-actives",
                params={"by": "volume", "top": int(top)},
            )
            payload = cast(Dict[str, Any], response.json())
            data = payload.get("data", {})
            rows = data.get("most_actives", []) if isinstance(data, dict) else []
            symbols: List[str] = []
            for row in rows:
                if isinstance(row, dict):
                    sym = row.get("symbol")
                    if sym:
                        symbols.append(str(sym).upper())
            return symbols
        except httpx.HTTPError as e:
            self.logger.error("Failed to fetch most actives from central API", error=str(e))
            raise

    def get_alpaca_movers(self, top: int = 10) -> Dict[str, List[str]]:
        """Fetch top gainers/losers via Data-Gateway screener."""
        try:
            response = self._request_with_retry(
                "GET",
                "/api/v1/alpaca/screener/movers",
                params={"market_type": "stocks", "top": int(top)},
            )
            payload = cast(Dict[str, Any], response.json())
            data = payload.get("data", {})
            if not isinstance(data, dict):
                return {"gainers": [], "losers": []}
            out: Dict[str, List[str]] = {"gainers": [], "losers": []}
            for key in ("gainers", "losers"):
                rows = data.get(key, [])
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if isinstance(row, dict):
                        sym = row.get("symbol")
                        if sym:
                            out[key].append(str(sym).upper())
            return out
        except httpx.HTTPError as e:
            self.logger.error("Failed to fetch movers from central API", error=str(e))
            raise

    def chat_completion(self, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Sends a chat completion request.
        """
        payload = {"model": model, "messages": messages}
        try:
            response = self._resolve_llm_client().post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            return cast(Dict[str, Any], response.json())
        except httpx.HTTPError as e:
            self.logger.error("Failed to get chat completion from central API", error=str(e))
            raise

    def close(self) -> None:
        """Close underlying HTTP clients."""
        self.client.close()
        if self._llm_client is not None:
            self._llm_client.close()

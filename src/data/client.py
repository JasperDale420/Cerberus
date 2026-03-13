"""Unified data client for Cerberus — REST methods.

Replaces CentralApiClient with a cleaner interface that talks to Data-Gateway.
WebSocket streaming methods will be added in a future task.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, cast

import httpx

from src.core.http_client import create_http_client
from src.core.logger import StructuredLogger

logger = StructuredLogger("unified_data_client")


class UnifiedDataClient:
    """REST client for Data-Gateway endpoints with retry and normalization."""

    def __init__(self, gateway_url: str, gateway_key: str, timeout: float = 30.0) -> None:
        self.gateway_url = gateway_url
        self.gateway_key = gateway_key
        self._timeout = timeout

        # Compute WebSocket URL (http→ws, https→wss, append /ws)
        ws_base = gateway_url.rstrip("/")
        if ws_base.startswith("https://"):
            ws_base = "wss://" + ws_base[len("https://") :]
        elif ws_base.startswith("http://"):
            ws_base = "ws://" + ws_base[len("http://") :]
        self._ws_url = ws_base + "/ws"

        headers: Dict[str, str] = {}
        if gateway_key:
            headers["X-Gateway-Key"] = gateway_key

        self.client = create_http_client(
            base_url=gateway_url,
            timeout=timeout,
            headers=headers,
        )

        self._max_retries = 3
        self._retry_backoff_base = 1.0

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------

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
        return self._retry_backoff_base * (2 ** max(0, attempt - 1))

    def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Execute an HTTP request with retry on transient failures."""
        max_attempts = self._max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.request(method, path, params=params, json=json)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= max_attempts:
                    raise
                delay = self._get_retry_delay_seconds(attempt)
                logger.warning(
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
                logger.warning(
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

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize_bars_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Data-Gateway envelope into {bars: [...]} format."""
        if "bars" in payload:
            return payload
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("bars"), list):
            return {"bars": [self._normalize_bar(item) for item in data["bars"]]}
        return payload

    def _normalize_bar(self, item: Any) -> Dict[str, Any]:
        """Normalize a bar item into short-key format."""
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
        """Normalize Data-Gateway trade response into list of trade dicts."""
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

    # ------------------------------------------------------------------
    # REST methods
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1Min",
    ) -> Dict[str, Any]:
        """Fetch historical bars via Data-Gateway."""
        params: Dict[str, Any] = {"timeframe": timeframe}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        response = self._request_with_retry(
            "GET",
            f"/api/v1/alpaca/stocks/{symbol.upper()}/bars",
            params=params,
        )
        payload = cast(Dict[str, Any], response.json())
        return self._normalize_bars_response(payload)

    def get_trades(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Fetch historical trades via Data-Gateway."""
        params: Dict[str, Any] = {"limit": int(limit)}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        response = self._request_with_retry(
            "GET",
            f"/api/v1/alpaca/stocks/{symbol.upper()}/trades",
            params=params,
        )
        payload = cast(Dict[str, Any], response.json())
        return self._normalize_trades_response(payload)

    def get_quotes(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Fetch historical quotes via Data-Gateway."""
        params: Dict[str, Any] = {}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        response = self._request_with_retry(
            "GET",
            f"/api/v1/alpaca/stocks/{symbol.upper()}/quotes",
            params=params or None,
        )
        return cast(Dict[str, Any], response.json())

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Fetch latest snapshot for a symbol via Data-Gateway."""
        response = self._request_with_retry(
            "GET",
            f"/api/v1/alpaca/stocks/{symbol.upper()}/snapshot",
        )
        return cast(Dict[str, Any], response.json())

    def get_flow(self, symbol: str, date_str: Optional[str] = None) -> Dict[str, Any]:
        """Fetch options flow via Data-Gateway UW endpoint."""
        params: Dict[str, Any] = {}
        if date_str:
            params["date"] = date_str

        response = self._request_with_retry(
            "GET",
            f"/api/v1/uw/flow/{symbol.upper()}",
            params=params or None,
        )
        return cast(Dict[str, Any], response.json())

    def get_gex(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch GEX data via Data-Gateway UW endpoint."""
        response = self._request_with_retry("GET", f"/api/v1/uw/gex/{symbol.upper()}")
        payload = cast(Dict[str, Any], response.json())
        data = payload.get("data")
        if isinstance(data, list):
            return cast(List[Dict[str, Any]], data)
        if isinstance(data, dict):
            return cast(List[Dict[str, Any]], data.get("rows", []))
        return []

    def get_most_actives(self, top: int = 20) -> List[str]:
        """Fetch most active stock symbols via Data-Gateway screener."""
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

    def get_movers(self, top: int = 10) -> Dict[str, List[str]]:
        """Fetch top gainers/losers via Data-Gateway screener."""
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

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit an order via Data-Gateway trading endpoint."""
        body: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side,
            "order_type": order_type,
            "time_in_force": time_in_force,
        }
        if qty is not None:
            body["qty"] = float(qty)
        if notional is not None:
            body["notional"] = float(notional)
        if limit_price is not None:
            body["limit_price"] = float(limit_price)
        if stop_price is not None:
            body["stop_price"] = float(stop_price)
        if client_order_id:
            body["client_order_id"] = client_order_id

        response = self._request_with_retry(
            "POST",
            "/api/v1/alpaca/orders",
            json=body,
        )
        payload = cast(Dict[str, Any], response.json())
        data = payload.get("data")
        return cast(Dict[str, Any], data) if isinstance(data, dict) else payload

    def get_orders(self, status: str = "open", limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch orders via Data-Gateway trading endpoint."""
        response = self._request_with_retry(
            "GET",
            "/api/v1/alpaca/orders",
            params={"status": status, "limit": int(limit)},
        )
        payload = cast(Dict[str, Any], response.json())
        data = payload.get("data")
        return cast(List[Dict[str, Any]], data) if isinstance(data, list) else []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order via Data-Gateway trading endpoint."""
        response = self._request_with_retry(
            "DELETE",
            f"/api/v1/alpaca/orders/{order_id}",
        )
        payload = cast(Dict[str, Any], response.json())
        data = payload.get("data")
        if isinstance(data, dict) and "cancelled" in data:
            return bool(data.get("cancelled"))
        return bool(payload.get("success", True))

    def get_prior_day_stats(self, symbol: str, current_time: datetime) -> tuple[float, float, float]:
        """Fetch 1Day bars for last 7 days and return (high, low, close) of last complete day."""
        start = current_time - timedelta(days=7)
        result = self.get_historical_bars(symbol, start, current_time, timeframe="1Day")
        bars = result.get("bars", [])
        if len(bars) < 2:
            if bars:
                bar = bars[0]
                return (float(bar.get("h", 0)), float(bar.get("l", 0)), float(bar.get("c", 0)))
            return (0.0, 0.0, 0.0)
        # Second-to-last bar is the last complete day
        prior = bars[-2]
        return (float(prior.get("h", 0)), float(prior.get("l", 0)), float(prior.get("c", 0)))

    def get_avg_daily_volume(self, symbol: str, end: datetime, lookback_days: int = 20) -> float:
        """Fetch 1Day bars and compute mean volume over lookback window."""
        start = end - timedelta(days=lookback_days + 5)  # buffer for weekends/holidays
        result = self.get_historical_bars(symbol, start, end, timeframe="1Day")
        bars = result.get("bars", [])
        if not bars:
            return 0.0
        volumes = [float(b.get("v", 0)) for b in bars[-lookback_days:] if isinstance(b, dict)]
        return sum(volumes) / len(volumes) if volumes else 0.0

    def close(self) -> None:
        """Close underlying HTTP client."""
        self.client.close()

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import urlparse, urlunparse

import websockets

from src.core.config import ConfigLoader
from src.core.domain import Bar
from src.core.logger import StructuredLogger


class GatewayStreamClient:
    """WebSocket client for Data-Gateway live bar streaming."""

    def __init__(self, config_loader: ConfigLoader, logger: StructuredLogger):
        self.logger = logger
        base_url = config_loader.get_env(
            "CERBERUS_GATEWAY_URL",
            config_loader.get_env("DATA_INGESTION_URL", "http://localhost:8080"),
        )
        self.gateway_key = config_loader.get_env("CERBERUS_GATEWAY_KEY", "")
        self.ws_url = self._build_ws_url(base_url)

        self._desired_symbols: set[str] = set()
        self._ws: Any = None
        self._running = False

    @staticmethod
    def _build_ws_url(base_url: str) -> str:
        parsed = urlparse(str(base_url).strip())
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path.rstrip("/")
        if path.endswith("/ws"):
            ws_path = path
        elif not path:
            ws_path = "/ws"
        else:
            ws_path = f"{path}/ws"
        return urlunparse((scheme, parsed.netloc, ws_path, "", "", ""))

    def _inspect_arity(self, callback: Callable[..., Any]) -> int:
        import inspect

        try:
            sig = inspect.signature(callback)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            return len([p for p in params if p.default is inspect.Parameter.empty])
        except Exception:
            return -1

    async def _invoke_bar_callback(self, callback: Callable[..., Any], bar: Bar, symbol: str, arity: int) -> None:
        async def _call(cb: Callable[..., Any], *args: Any) -> None:
            if asyncio.iscoroutinefunction(cb):
                await cb(*args)
            else:
                cb(*args)

        if arity == 1:
            await _call(callback, bar)
        elif arity >= 2:
            await _call(callback, symbol, bar)
        else:
            try:
                await _call(callback, bar)
            except TypeError:
                await _call(callback, symbol, bar)

    def _parse_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                self.logger.warning(
                    "Gateway bar timestamp missing timezone, defaulting to UTC",
                    event_type="gateway_bar_timestamp_naive",
                    timestamp_value=value,
                )
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        raise ValueError(f"Invalid bar timestamp: {value}")

    def _normalize_bar_from_data(self, payload: dict[str, Any]) -> Bar:
        symbol = str(payload.get("S") or payload.get("symbol") or "").upper()
        if not symbol:
            raise ValueError("Missing symbol in gateway bar payload")
        ts = self._parse_timestamp(payload.get("t") or payload.get("timestamp"))
        open_value = payload.get("o") if payload.get("o") is not None else payload.get("open", 0.0)
        high_value = payload.get("h") if payload.get("h") is not None else payload.get("high", 0.0)
        low_value = payload.get("l") if payload.get("l") is not None else payload.get("low", 0.0)
        close_value = payload.get("c") if payload.get("c") is not None else payload.get("close", 0.0)
        volume_value = payload.get("v") if payload.get("v") is not None else payload.get("volume", 0.0)
        return Bar(
            symbol=symbol,
            time=ts,
            open=float(0.0 if open_value is None else open_value),
            high=float(0.0 if high_value is None else high_value),
            low=float(0.0 if low_value is None else low_value),
            close=float(0.0 if close_value is None else close_value),
            volume=float(0.0 if volume_value is None else volume_value),
            vwap=payload.get("vw") or payload.get("vwap"),
            trade_count=payload.get("n") or payload.get("trade_count"),
        )

    def _extract_bar_payload(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if message.get("type") != "data":
            return None
        feed = str(message.get("feed") or "")
        if feed != "bars":
            return None

        data = message.get("data")
        if isinstance(data, dict):
            return data
        envelope = message.get("envelope")
        if isinstance(envelope, dict):
            payload = envelope.get("payload")
            if isinstance(payload, dict):
                return payload
        return None

    async def _send_json(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            return
        await ws.send(json.dumps(payload))

    async def _send_subscribe(self, symbols: list[str]) -> None:
        if not symbols:
            return
        await self._send_json(
            {
                "action": "subscribe",
                "feeds": ["stock_bars"],
                "symbols": symbols,
            }
        )

    async def _send_unsubscribe(self, symbols: list[str]) -> None:
        if not symbols:
            return
        await self._send_json(
            {
                "action": "unsubscribe",
                "feeds": ["stock_bars"],
                "symbols": symbols,
            }
        )

    def subscribe(self, symbol: str) -> None:
        sym = str(symbol or "").upper().strip()
        if not sym:
            return
        self._desired_symbols.add(sym)
        if self._ws is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.create_task(self._send_subscribe([sym]))

    def unsubscribe(self, symbol: str) -> None:
        sym = str(symbol or "").upper().strip()
        if not sym:
            return
        self._desired_symbols.discard(sym)
        if self._ws is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.create_task(self._send_unsubscribe([sym]))

    async def _auth(self) -> None:
        if not self.gateway_key:
            raise ValueError("CERBERUS_GATEWAY_KEY is required for Data-Gateway websocket streaming")
        await self._send_json({"action": "auth", "key": self.gateway_key})
        raw = await self._ws.recv()
        msg = json.loads(raw)
        if not isinstance(msg, dict) or msg.get("type") != "auth_result" or msg.get("status") != "ok":
            raise RuntimeError(f"Gateway websocket auth failed: {msg}")

    async def _handle_message(self, callback: Callable[..., Any], arity: int, msg: dict[str, Any]) -> None:
        payload = self._extract_bar_payload(msg)
        if payload is None:
            return
        try:
            bar = self._normalize_bar_from_data(payload)
        except Exception as e:
            self.logger.warning("Failed to normalize gateway bar", error=str(e))
            return
        await self._invoke_bar_callback(callback, bar, bar.symbol, arity)

    async def start_stream(
        self,
        callback: Callable[..., Any],
        on_reconnect: Optional[Callable[..., Any]] = None,
    ) -> None:
        arity = self._inspect_arity(callback)
        self._running = True
        backoff = 1.0
        had_failure = False

        while self._running:
            try:
                if had_failure and on_reconnect:
                    if asyncio.iscoroutinefunction(on_reconnect):
                        await on_reconnect()
                    else:
                        on_reconnect()

                async with websockets.connect(self.ws_url) as ws:
                    self._ws = ws
                    await self._auth()
                    await self._send_subscribe(sorted(self._desired_symbols))
                    self.logger.info(
                        "Gateway stream connected",
                        ws_url=self.ws_url,
                        symbols=len(self._desired_symbols),
                    )
                    backoff = 1.0
                    had_failure = False
                    while self._running:
                        raw = await ws.recv()
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict):
                                    await self._handle_message(callback, arity, item)
                        elif isinstance(parsed, dict):
                            await self._handle_message(callback, arity, parsed)
            except asyncio.CancelledError:
                self._running = False
                raise
            except OSError as e:
                # Handles ConnectionRefusedError, etc.
                had_failure = True
                self.logger.warning(
                    "Gateway stream connection failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    ws_url=self.ws_url,
                    reconnect_in_seconds=round(backoff, 2),
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
            except Exception as e:
                had_failure = True
                self.logger.warning(
                    "Gateway stream disconnected or failed",
                    error=str(e),
                    reconnect_in_seconds=round(backoff, 2),
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
            finally:
                self._ws = None

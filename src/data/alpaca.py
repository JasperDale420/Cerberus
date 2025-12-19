from datetime import datetime
from typing import Any, Optional, Set, cast

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream

from src.core.config import ConfigLoader
from src.core.domain import Bar
from src.core.logger import StructuredLogger


class AlpacaClient:
    """
    Wrapper for Alpaca API clients (Trading, Historical Data, Live Data).
    """

    def __init__(self, config_loader: ConfigLoader, logger: StructuredLogger):
        self.logger = logger

        try:
            self.api_key = config_loader.get_env("ALPACA_API_KEY")
            self.secret_key = config_loader.get_env("ALPACA_SECRET_KEY")
            self.paper = config_loader.get_env("ALPACA_PAPER", "True").lower() == "true"
        except ValueError as e:
            self.logger.critical("Failed to load Alpaca credentials", error=str(e))
            raise

        try:
            self.trading_client = TradingClient(
                self.api_key, self.secret_key, paper=self.paper
            )
            self.historical_client = StockHistoricalDataClient(
                self.api_key, self.secret_key
            )
            self.trading_stream_client: Optional[TradingStream] = None
            # Stream is initialized on demand or separately as it blocks/runs in a loop
            self.stream_client: Optional[StockDataStream] = None
            self._pending_symbols: Set[str] = set()
            self._subscribed_symbols: Set[str] = set()
        except Exception as e:
            self.logger.critical("Failed to initialize Alpaca clients", error=str(e))
            raise

    def _to_bar(self, data: Any, symbol_override: Optional[str] = None) -> Bar:
        def _f(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        symbol = symbol_override or cast(
            str, getattr(data, "symbol", None) or "UNKNOWN"
        )

        if isinstance(data, dict):
            t = data.get("t") or data.get("timestamp")
            o = data.get("o") or data.get("open")
            h = data.get("h") or data.get("high")
            low = data.get("l") or data.get("low")
            c = data.get("c") or data.get("close")
            v = data.get("v") or data.get("volume")
            return Bar(
                symbol=symbol,
                time=cast(datetime, t),
                open=_f(o),
                high=_f(h),
                low=_f(low),
                close=_f(c),
                volume=_f(v),
                vwap=data.get("vwap"),
                trade_count=data.get("trade_count"),
            )

        ts = getattr(data, "time", None) or getattr(data, "timestamp", None)
        return Bar(
            symbol=symbol,
            time=cast(datetime, ts),
            open=float(data.open),
            high=float(data.high),
            low=float(data.low),
            close=float(data.close),
            volume=float(data.volume),
            vwap=getattr(data, "vwap", None),
            trade_count=getattr(data, "trade_count", None),
        )

    def get_account(self):
        """
        Fetches account information.
        """
        try:
            return self.trading_client.get_account()
        except Exception as e:
            from src.core.errors import ErrorCode

            self.logger.error(
                "Failed to fetch account info",
                error_code=ErrorCode.ALPACA_ACCOUNT_FETCH_FAILED.value,
                error=str(e),
            )
            raise

    def get_historical_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Min"
    ):
        """
        Fetches historical bars for a symbol via Alpaca historical data client.
        """
        try:
            tf = str(timeframe).strip().lower()
            if tf in ("1min", "1m", "minute"):
                tf_obj = TimeFrame.Minute
            elif tf in ("1day", "day", "d"):
                tf_obj = TimeFrame.Day
            else:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                timeframe=tf_obj,
            )
            resp = self.historical_client.get_stock_bars(req)
            bars = (resp.data or {}).get(symbol) or []

            # Return a stable dict format aligned with existing FeaturePipeline parsing.
            out = []
            for b in bars:
                out.append(
                    {
                        "t": getattr(b, "timestamp", None) or getattr(b, "t", None),
                        "o": float(getattr(b, "open", 0.0)),
                        "h": float(getattr(b, "high", 0.0)),
                        "l": float(getattr(b, "low", 0.0)),
                        "c": float(getattr(b, "close", 0.0)),
                        "v": float(getattr(b, "volume", 0.0)),
                    }
                )
            return {"bars": out}
        except Exception as e:
            from src.core.errors import ErrorCode

            self.logger.error(
                "Failed to fetch historical bars",
                error_code=ErrorCode.ALPACA_BARS_FETCH_FAILED.value,
                symbol=symbol,
                timeframe=timeframe,
                error=str(e),
            )
            raise

    def get_stream_client(self) -> StockDataStream:
        """
        Returns a configured StockDataStream client.
        """
        if not self.stream_client:
            self.stream_client = StockDataStream(self.api_key, self.secret_key)
        return self.stream_client

    def get_trading_stream_client(self) -> TradingStream:
        if not self.trading_stream_client:
            self.trading_stream_client = TradingStream(
                self.api_key, self.secret_key, paper=self.paper
            )
        return self.trading_stream_client

    async def start_stream(self, callback, on_reconnect=None):
        """
        Starts the WebSocket stream and registers the callback.
        Blocking call (runs loop).
        """
        import asyncio
        import inspect

        stream = self.get_stream_client()

        async def on_bar_wrapper(data):
            symbol = getattr(data, "symbol", "UNKNOWN")
            try:
                bar = self._to_bar(data, symbol_override=symbol)
            except Exception as e:
                self.logger.error(
                    "Failed to normalize bar", symbol=symbol, error=str(e)
                )
                return
            # PRD 6.2: the handler should call `ExecutionEngine.on_bar(bar)`.
            # Backward-compatible: if the callback expects (symbol, bar), call that instead.
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
                required = len(
                    [p for p in params if p.default is inspect.Parameter.empty]
                )
            except Exception:
                required = -1

            async def _call_with_bar_only() -> None:
                if asyncio.iscoroutinefunction(callback):
                    await callback(bar)
                else:
                    callback(bar)

            async def _call_with_symbol_and_bar() -> None:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, bar)
                else:
                    callback(symbol, bar)

            if required == 1:
                await _call_with_bar_only()
            elif required >= 2:
                await _call_with_symbol_and_bar()
            else:
                # Best-effort: try the PRD form first, then fallback.
                try:
                    await _call_with_bar_only()
                except TypeError:
                    await _call_with_symbol_and_bar()

        # Alpaca's StockDataStream requires coroutine handlers and will await them.
        self._bar_handler = on_bar_wrapper

        async def _flush_pending() -> None:
            for sym in sorted(self._pending_symbols):
                try:
                    stream.subscribe_bars(self._bar_handler, sym)
                    self._subscribed_symbols.add(sym)
                except Exception as e:
                    self.logger.error(
                        "Failed to subscribe queued symbol", symbol=sym, error=str(e)
                    )
            self._pending_symbols.clear()

        # Best-effort reconnect/backoff loop (PRD 11.4). SDK behavior may differ.
        backoff = 1.0
        backoff_max = 30.0
        had_failure = False
        while True:
            if had_failure and on_reconnect is not None:
                try:
                    if asyncio.iscoroutinefunction(on_reconnect):
                        await on_reconnect()
                    else:
                        on_reconnect()
                except Exception as e:
                    self.logger.error("Reconnect hook failed", error=str(e))

            await _flush_pending()
            try:
                await asyncio.to_thread(stream.run)
                # If run returns normally, reset backoff and exit.
                return
            except asyncio.CancelledError:
                try:
                    stream.stop()
                except Exception:
                    pass
                raise
            except Exception as e:
                self.logger.error(
                    "Alpaca stream failed; retrying",
                    error=str(e),
                    backoff_sec=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, backoff_max)
                had_failure = True

    async def start_trade_stream(self, callback, on_reconnect=None) -> None:
        """
        Starts Alpaca trading updates stream (fills/order lifecycle) with backoff.

        The Alpaca `TradingStream.run()` method is synchronous; we run it in a thread but
        marshal callbacks back onto the main asyncio loop for safety.
        """
        import asyncio

        stream = self.get_trading_stream_client()
        loop = asyncio.get_running_loop()

        def on_trade_update(data: Any) -> None:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.run_coroutine_threadsafe(callback(data), loop)
                else:
                    loop.call_soon_threadsafe(callback, data)
            except Exception as e:
                # Best-effort; do not crash the stream thread due to callback issues.
                self.logger.error("Trade update callback failed", error=str(e))

        stream.subscribe_trade_updates(on_trade_update)

        backoff = 1.0
        backoff_max = 30.0
        had_failure = False
        while True:
            try:
                if had_failure and on_reconnect is not None:
                    try:
                        if asyncio.iscoroutinefunction(on_reconnect):
                            await on_reconnect()
                        else:
                            on_reconnect()
                    except Exception as e:
                        self.logger.error("Trade reconnect hook failed", error=str(e))
                await asyncio.to_thread(stream.run)
                return
            except asyncio.CancelledError:
                try:
                    stream.stop()
                except Exception:
                    pass
                raise
            except Exception as e:
                self.logger.error(
                    "Alpaca trade stream failed; retrying",
                    error=str(e),
                    backoff_sec=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, backoff_max)
                had_failure = True

    def subscribe(self, symbol: str):
        """
        Subscribes to bar data for a symbol.
        """
        self._pending_symbols.add(symbol)
        stream = self.get_stream_client()
        if not hasattr(self, "_bar_handler"):
            self.logger.warning(
                "Subscribe called before start_stream; queued symbol", symbol=symbol
            )
            return

        if symbol not in self._subscribed_symbols:
            stream.subscribe_bars(self._bar_handler, symbol)
            self._subscribed_symbols.add(symbol)
            self._pending_symbols.discard(symbol)

    def unsubscribe(self, symbol: str):
        """
        Unsubscribes from bar data.
        """
        stream = self.get_stream_client()
        try:
            stream.unsubscribe_bars(symbol)
        except KeyError:
            # Alpaca SDK can raise KeyError if handler map doesn't contain symbol
            # (e.g., queued subscribes, reconnect races, or churn during warmup).
            self.logger.warning(
                "Unsubscribe no-op; symbol not subscribed", symbol=symbol
            )
        self._subscribed_symbols.discard(symbol)
        self._pending_symbols.discard(symbol)

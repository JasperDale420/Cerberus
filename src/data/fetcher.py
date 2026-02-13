import asyncio
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core.errors import ErrorCode
from src.core.logger import StructuredLogger
from src.core.settings import get_settings
from src.core.time_utils import get_eastern_timezone
from src.data.alpaca import AlpacaClient
from src.data.api_client import CentralApiClient
from src.data.heber_read_client import HeberReadClient
from src.data.unusual_whales import UnusualWhalesClient


class DataFetcher:
    """
    Handles data retrieval for the FeaturePipeline (I/O only).
    Manages caching and concurrency limits.
    """

    def __init__(
        self,
        alpaca_client: Optional[AlpacaClient],
        unusual_whales_client: UnusualWhalesClient,
        logger: StructuredLogger,
        config: Optional[Dict[str, Any]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        central_api_client: Optional[CentralApiClient] = None,
    ):
        self.alpaca_client = alpaca_client
        self.unusual_whales_client = unusual_whales_client
        self.central_api_client = central_api_client
        self.logger = logger
        self.config = config or {}
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        runtime = get_settings()
        self.use_gateway_data = runtime.use_gateway_data
        self.use_heber_storage = runtime.use_heber_storage
        self.allow_legacy_failover = bool(runtime.cerberus_failover_to_legacy)
        self.enable_dual_compare = bool(runtime.cerberus_data_backend == "dual")
        self.heber_client: Optional[HeberReadClient] = None
        if self.use_heber_storage and runtime.cerberus_heber_data_root:
            self.heber_client = HeberReadClient(
                data_root=runtime.cerberus_heber_data_root,
                logger=logger,
            )
        elif self.use_heber_storage:
            self.logger.warning(
                "Heber storage backend enabled but CERBERUS_HEBER_DATA_ROOT is not set; falling back to gateway/legacy sources",
            )

        # H2 Memory Audit Fix: LRU cache with maxsize for bars
        # Uses OrderedDict for LRU ordering; evicts oldest when maxsize exceeded
        self._bars_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._trades_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        try:
            self._cache_maxsize = int(self.config.get("bars_cache_maxsize", 500))
        except (TypeError, ValueError):
            self._cache_maxsize = 500

    def _resolve_fetch_start(self, symbol: str, start: datetime) -> Tuple[datetime, List[Dict[str, Any]]]:
        cached = self._bars_cache.get(symbol)
        if not cached or cached.get("start") != start:
            return start, []

        existing = cached.get("bars", [])
        if not existing:
            return start, []

        last_bar = existing[-1]
        raw_ts = last_bar.get("t") or last_bar.get("timestamp")
        if not raw_ts:
            return start, existing

        try:
            last_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            return last_ts + timedelta(seconds=1), existing
        except Exception as e:
            self.logger.debug(
                "Timestamp parsing failed",
                operation="parse_bar_timestamp",
                symbol=symbol,
                error=str(e),
            )
            return start, existing

    async def _fetch_alpaca_bars_internal(
        self, symbol: str, start: datetime, end: datetime, timeframe: str
    ) -> List[Dict[str, Any]]:
        import asyncio

        new_bars = await asyncio.to_thread(
            self._get_historical_bars_sync,
            symbol,
            start,
            end,
            timeframe,
        )
        if isinstance(new_bars, dict) and "bars" in new_bars:
            return list(new_bars["bars"])
        return list(new_bars or [])

    def _get_historical_bars_sync(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> Any:
        """Fetch bars from configured backend with optional legacy failover."""
        sym = str(symbol).strip().upper()
        if self.heber_client is not None:
            try:
                heber_bars = self.heber_client.get_bars(
                    symbol=sym,
                    start=start,
                    end=end,
                    timeframe=timeframe,
                    as_of=end,
                )
                if heber_bars:
                    return heber_bars
                self.logger.info(
                    "Heber bars read returned no rows; trying gateway/legacy source",
                    symbol=sym,
                    timeframe=timeframe,
                )
            except Exception as e:
                self.logger.warning(
                    "Heber bars read failed; trying gateway/legacy source",
                    symbol=sym,
                    timeframe=timeframe,
                    error=str(e),
                )

        if self.use_gateway_data and self.central_api_client is not None:
            try:
                bars = self.central_api_client.get_alpaca_bars(sym, start, end, timeframe)
                if self.enable_dual_compare:
                    self._compare_bars_with_legacy(sym, start, end, timeframe, bars)
                return bars
            except Exception as e:
                self.logger.warning(
                    "Gateway bars fetch failed",
                    symbol=sym,
                    timeframe=timeframe,
                    error=str(e),
                    failover=self.allow_legacy_failover,
                )
                if not self.allow_legacy_failover:
                    raise

        if self.alpaca_client is None:
            raise RuntimeError("Legacy Alpaca bars fallback requested but Alpaca client is not initialized")
        return self.alpaca_client.get_historical_bars(sym, start, end, timeframe)

    def _compare_bars_with_legacy(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
        gateway_payload: Any,
    ) -> None:
        """Log comprehensive dual-read parity diagnostics without affecting control flow."""
        try:
            if self.alpaca_client is None:
                return
            legacy = self.alpaca_client.get_historical_bars(symbol, start, end, timeframe)
            legacy_bars = legacy.get("bars", []) if isinstance(legacy, dict) else legacy
            gateway_bars = gateway_payload.get("bars", []) if isinstance(gateway_payload, dict) else gateway_payload

            if not isinstance(legacy_bars, list) or not isinstance(gateway_bars, list):
                self.logger.warning(
                    "Dual read bars type mismatch",
                    symbol=symbol,
                    timeframe=timeframe,
                    legacy_type=type(legacy_bars).__name__,
                    gateway_type=type(gateway_bars).__name__,
                )
                return

            legacy_count = len(legacy_bars)
            gateway_count = len(gateway_bars)

            # Count comparison
            if legacy_count != gateway_count:
                self.logger.warning(
                    "Dual read bars count mismatch",
                    symbol=symbol,
                    timeframe=timeframe,
                    legacy_count=legacy_count,
                    gateway_count=gateway_count,
                    delta=abs(legacy_count - gateway_count),
                )

            # Sample value comparison (first and last bars if both non-empty)
            if legacy_count > 0 and gateway_count > 0:
                self._compare_bar_values(symbol, timeframe, legacy_bars[0], gateway_bars[0], "first")
                if legacy_count > 1 and gateway_count > 1:
                    self._compare_bar_values(symbol, timeframe, legacy_bars[-1], gateway_bars[-1], "last")

            # Success log for parity
            if legacy_count == gateway_count and legacy_count > 0:
                self.logger.info(
                    "Dual read bars parity confirmed",
                    symbol=symbol,
                    timeframe=timeframe,
                    count=legacy_count,
                )
        except Exception as e:
            # Diagnostics only - never fail fetch path on comparison errors.
            self.logger.debug(
                "Dual read comparison error",
                symbol=symbol,
                error=str(e),
            )
            return

    def _compare_bar_values(
        self,
        symbol: str,
        timeframe: str,
        legacy_bar: Any,
        gateway_bar: Any,
        position: str,
    ) -> None:
        """Compare OHLCV values between legacy and gateway bars."""
        try:
            legacy_dict = legacy_bar if isinstance(legacy_bar, dict) else {}
            gateway_dict = gateway_bar if isinstance(gateway_bar, dict) else {}

            fields = [("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")]
            mismatches = []

            for short, long_name in fields:
                legacy_val = legacy_dict.get(short) or legacy_dict.get(long_name)
                gateway_val = gateway_dict.get(short) or gateway_dict.get(long_name)

                if legacy_val is not None and gateway_val is not None:
                    try:
                        legacy_float = float(legacy_val)
                        gateway_float = float(gateway_val)
                        # Allow small floating point differences (0.01%)
                        if legacy_float != 0:
                            pct_diff = abs(legacy_float - gateway_float) / legacy_float
                            if pct_diff > 0.0001:
                                mismatches.append(
                                    {
                                        "field": long_name,
                                        "legacy": legacy_float,
                                        "gateway": gateway_float,
                                        "pct_diff": round(pct_diff * 100, 4),
                                    }
                                )
                    except (ValueError, TypeError):
                        pass

            if mismatches:
                self.logger.warning(
                    "Dual read bar value mismatch",
                    symbol=symbol,
                    timeframe=timeframe,
                    position=position,
                    mismatches=mismatches,
                )
        except Exception:
            return

    async def fetch_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Min"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Fetches bars for a symbol, utilizing local cache.
        Returns (bars, metrics_delta).
        """
        metrics = {
            "alpaca_fetch_fail": 0,
            "alpaca_no_bars": 0,
            "cache_hits": 0,
            "incremental_fetches": 0,
        }
        sym = str(symbol).strip().upper()

        fetch_start, existing_bars = self._resolve_fetch_start(sym, start)
        new_bars = []

        if fetch_start < end:
            try:
                new_bars = await self._fetch_alpaca_bars_internal(sym, fetch_start, end, timeframe)
                if new_bars and existing_bars:
                    metrics["incremental_fetches"] += 1
            except Exception as e:
                metrics["alpaca_fetch_fail"] += 1
                self.logger.warning(
                    "Alpaca bars fetch failed",
                    error_code=ErrorCode.ALPACA_BARS_FETCH_FAILED.value,
                    symbol=sym,
                    error=str(e),
                )
                if not existing_bars:
                    return [], metrics

        if existing_bars:
            metrics["cache_hits"] += 1

        final_bars = existing_bars + new_bars
        self._bars_cache[sym] = {"start": start, "bars": final_bars}
        self._bars_cache.move_to_end(sym)

        # Evict oldest entries if over maxsize
        while len(self._bars_cache) > self._cache_maxsize:
            evicted_sym, _ = self._bars_cache.popitem(last=False)
            self.logger.debug("Evicted bars cache entry", symbol=evicted_sym)

        if not final_bars:
            metrics["alpaca_no_bars"] += 1

        return final_bars, metrics

    async def fetch_trades(
        self, symbol: str, start: datetime, end: datetime
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Fetches historical trades for a symbol.
        Used for Trade Flow Imbalance (TFI) calculation.
        """
        metrics = {"alpaca_trades_fetch_fail": 0, "alpaca_no_trades": 0}
        sym = str(symbol).strip().upper()

        try:
            import asyncio

            if self.heber_client is not None:
                try:
                    heber_trades = await asyncio.to_thread(
                        self.heber_client.get_trades,
                        sym,
                        start,
                        end,
                        end,
                    )
                    if heber_trades:
                        return heber_trades, metrics
                    self.logger.info(
                        "Heber trades read returned no rows; trying gateway/legacy source",
                        symbol=sym,
                    )
                except Exception as e:
                    self.logger.warning(
                        "Heber trades read failed; trying gateway/legacy source",
                        symbol=sym,
                        error=str(e),
                    )

            if self.use_gateway_data and self.central_api_client is not None:
                try:
                    trades = await asyncio.to_thread(
                        self.central_api_client.get_alpaca_trades,
                        sym,
                        start,
                        end,
                    )
                    if self.enable_dual_compare:
                        await self._compare_trades_with_legacy(sym, start, end, trades)
                except Exception as e:
                    if not self.allow_legacy_failover:
                        raise
                    if self.alpaca_client is None:
                        raise RuntimeError(
                            "Legacy Alpaca trades fallback requested but Alpaca client is not initialized"
                        ) from e
                    trades = await asyncio.to_thread(self.alpaca_client.get_historical_trades, sym, start, end)
            else:
                if self.alpaca_client is None:
                    raise RuntimeError("Alpaca trades requested but Alpaca client is not initialized")
                trades = await asyncio.to_thread(self.alpaca_client.get_historical_trades, sym, start, end)
            if not trades:
                metrics["alpaca_no_trades"] += 1
            return trades, metrics
        except Exception as e:
            metrics["alpaca_trades_fetch_fail"] += 1
            self.logger.warning(
                "Alpaca trades fetch failed",
                error_code=ErrorCode.ALPACA_TRADES_FETCH_FAILED.value,
                symbol=sym,
                error=str(e),
            )
            return [], metrics

    async def _compare_trades_with_legacy(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        gateway_trades: List[Dict[str, Any]],
    ) -> None:
        """Log dual-read parity for trades data."""
        try:
            import asyncio

            if self.alpaca_client is None:
                return

            legacy_trades = await asyncio.to_thread(self.alpaca_client.get_historical_trades, symbol, start, end)

            legacy_count = len(legacy_trades) if isinstance(legacy_trades, list) else 0
            gateway_count = len(gateway_trades) if isinstance(gateway_trades, list) else 0

            if legacy_count != gateway_count:
                self.logger.warning(
                    "Dual read trades count mismatch",
                    symbol=symbol,
                    legacy_count=legacy_count,
                    gateway_count=gateway_count,
                    delta=abs(legacy_count - gateway_count),
                )
            elif legacy_count > 0:
                self.logger.info(
                    "Dual read trades parity confirmed",
                    symbol=symbol,
                    count=legacy_count,
                )
        except Exception as e:
            self.logger.debug(
                "Dual read trades comparison error",
                symbol=symbol,
                error=str(e),
            )

    def _extract_volume(self, bar: Any) -> Optional[float]:
        try:
            if isinstance(bar, dict):
                v = bar.get("v") if bar.get("v") is not None else bar.get("volume")
            else:
                v = getattr(bar, "v", None) or getattr(bar, "volume", None)
            return float(v) if v is not None else None
        except Exception as e:
            self.logger.debug(
                "Volume extraction failed",
                operation="extract_volume",
                error=str(e),
            )
            return None

    def fetch_avg_daily_volume(self, symbol: str, end: datetime, lookback_days: int) -> Optional[float]:
        """
        Fetches daily bars to calculate average volume.
        """
        if lookback_days <= 0:
            return None

        start = end - timedelta(days=int(max(lookback_days * 3, 10)))
        try:
            daily = self._get_historical_bars_sync(symbol, start, end, timeframe="1Day")
        except Exception as e:
            self.logger.warning("Failed to fetch avg volume", symbol=symbol, error=str(e))
            return None

        if not daily:
            return None

        bars_list = daily.get("bars") if isinstance(daily, dict) and "bars" in daily else daily
        if not isinstance(bars_list, list):
            return None

        vols = [v for b in bars_list if (v := self._extract_volume(b)) is not None]

        if not vols:
            return None

        window = vols[-lookback_days:]
        return float(sum(window) / len(window)) if window else None

    def _parse_bar_time(self, bar: Any) -> Optional[datetime]:
        try:
            bd = bar if isinstance(bar, dict) else getattr(bar, "__dict__", {})
            raw_t = bd.get("t") or bd.get("timestamp") or getattr(bar, "t", None)
            if not raw_t:
                return None
            if isinstance(raw_t, datetime):
                dt = raw_t
            else:
                dt = datetime.fromisoformat(str(raw_t).replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except Exception as e:
            self.logger.debug(
                "Bar time parsing failed",
                operation="parse_bar_time",
                error=str(e),
            )
            return None

    def _get_bar_field(self, bar: Any, keys: List[str]) -> float:
        bd = bar if isinstance(bar, dict) else getattr(bar, "__dict__", {})
        for k in keys:
            val = bd.get(k) or getattr(bar, k, None)
            if val is not None:
                return float(val)
        return 0.0

    def fetch_prior_day_stats(self, symbol: str, current_time: datetime) -> Tuple[float, float, float]:
        """
        Returns (High, Low, Close) from daily bars for the prior complete day.
        """
        try:
            start = current_time - timedelta(days=7)
            bars = self._get_historical_bars_sync(symbol, start, current_time, timeframe="1Day")
            if not bars:
                return (0.0, 0.0, 0.0)

            bars_list = bars.get("bars") if isinstance(bars, dict) and "bars" in bars else bars
            if not isinstance(bars_list, list):
                return (0.0, 0.0, 0.0)

            et_tz = get_eastern_timezone()
            cutoff_date_et = current_time.astimezone(et_tz).date()

            valid_bars = []
            for b in bars_list:
                t_dt = self._parse_bar_time(b)
                if t_dt and t_dt.astimezone(et_tz).date() < cutoff_date_et:
                    valid_bars.append(b)

            if not valid_bars:
                return (0.0, 0.0, 0.0)

            last = valid_bars[-1]
            h = self._get_bar_field(last, ["h", "high"])
            low_px = self._get_bar_field(last, ["l", "low"])
            c = self._get_bar_field(last, ["c", "close"])
            return (h, low_px, c)

        except Exception as e:
            self.logger.warning("Failed to fetch prior day stats", symbol=symbol, error=str(e))
            return (0.0, 0.0, 0.0)

    async def fetch_flow(self, symbol: str, date_str: str) -> List[Any]:
        """
        Fetches Unusual Whales option flow for a specific date.
        """
        try:
            if self.use_gateway_data and self.central_api_client is not None:
                response = await asyncio.to_thread(
                    self.central_api_client.get_uw_flow,
                    symbol,
                    date_str,
                )
                data = response.get("data") if isinstance(response, dict) else None
                flow_data = data if isinstance(data, list) else []
                if self.enable_dual_compare:
                    await self._compare_flow_with_legacy(symbol, date_str, flow_data)
                return flow_data

            # mypy: ignore
            return await self.unusual_whales_client.get_option_flow(symbol, date_str)  # type: ignore
        except Exception as e:
            self.logger.warning(
                "Unusual Whales flow fetch failed; using neutral flow",
                error_code=ErrorCode.UW_FLOW_FETCH_FAILED.value,
                symbol=symbol,
                error=str(e),
            )
            return []

    async def _compare_flow_with_legacy(
        self,
        symbol: str,
        date_str: str,
        gateway_flow: List[Any],
    ) -> None:
        """Log dual-read parity for options flow data."""
        try:
            legacy_flow = await self.unusual_whales_client.get_option_flow(symbol, date_str)
            legacy_count = len(legacy_flow) if isinstance(legacy_flow, list) else 0
            gateway_count = len(gateway_flow) if isinstance(gateway_flow, list) else 0

            if legacy_count != gateway_count:
                self.logger.warning(
                    "Dual read flow count mismatch",
                    symbol=symbol,
                    date=date_str,
                    legacy_count=legacy_count,
                    gateway_count=gateway_count,
                    delta=abs(legacy_count - gateway_count),
                )
            elif legacy_count > 0:
                self.logger.info(
                    "Dual read flow parity confirmed",
                    symbol=symbol,
                    date=date_str,
                    count=legacy_count,
                )
        except Exception as e:
            self.logger.debug(
                "Dual read flow comparison error",
                symbol=symbol,
                date=date_str,
                error=str(e),
            )

    async def fetch_gex(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetches Unusual Whales greek exposure data.
        Used for Net GEX and Flip distance calculation.
        """
        try:
            if self.use_gateway_data and self.central_api_client is not None:
                gex_data = await asyncio.to_thread(self.central_api_client.get_uw_gex, symbol)
                if self.enable_dual_compare:
                    await self._compare_gex_with_legacy(symbol, gex_data)
                return gex_data
            return await self.unusual_whales_client.get_greek_exposure(symbol)
        except Exception as e:
            self.logger.warning(
                "Unusual Whales GEX fetch failed",
                symbol=symbol,
                error=str(e),
            )
            return []

    async def _compare_gex_with_legacy(
        self,
        symbol: str,
        gateway_gex: List[Dict[str, Any]],
    ) -> None:
        """Log dual-read parity for GEX data."""
        try:
            legacy_gex = await self.unusual_whales_client.get_greek_exposure(symbol)

            legacy_count = len(legacy_gex) if isinstance(legacy_gex, list) else 0
            gateway_count = len(gateway_gex) if isinstance(gateway_gex, list) else 0

            if legacy_count != gateway_count:
                self.logger.warning(
                    "Dual read GEX count mismatch",
                    symbol=symbol,
                    legacy_count=legacy_count,
                    gateway_count=gateway_count,
                    delta=abs(legacy_count - gateway_count),
                )
            elif legacy_count > 0:
                self.logger.info(
                    "Dual read GEX parity confirmed",
                    symbol=symbol,
                    count=legacy_count,
                )
        except Exception as e:
            self.logger.debug(
                "Dual read GEX comparison error",
                symbol=symbol,
                error=str(e),
            )

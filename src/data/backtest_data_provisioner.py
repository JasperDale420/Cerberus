"""Backtest data provisioner — request data from Gateway, wait for Heber normalization.

Adapted from Atlas's DataPipeline pattern. Orchestrates the full flow:
1. Submit backfill via Data Gateway
2. Poll until Heber has normalized the data
3. Read from Heber Silver parquet partitions
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.core.logger import StructuredLogger
from src.core.settings import get_settings
from src.data.api_client import (
    BackfillFailedError,
    BackfillTimeoutError,
    CentralApiClient,
)
from src.data.heber_read_client import HeberReadClient

# Provider+feed mapping for Cerberus data types
FEED_PROVIDER_MAP: Dict[str, tuple[str, str]] = {
    "bars": ("alpaca", "bars"),
    "trades": ("alpaca", "trades"),
}


class ProvisioningError(Exception):
    """Raised when data provisioning fails after all attempts."""


class BacktestDataProvisioner:
    """Request data from Data Gateway, wait for Heber normalization, read results.

    Supports chunked backfills for large date ranges and falls back to
    direct Gateway fetch when Heber read returns empty.
    """

    def __init__(
        self,
        api_client: CentralApiClient,
        heber_read_client: Optional[HeberReadClient],
        logger: StructuredLogger,
    ) -> None:
        self.api_client = api_client
        self.heber_client = heber_read_client
        self.logger = logger

        settings = get_settings()
        self.timeout_seconds = settings.cerberus_backfill_timeout_seconds
        self.poll_interval = settings.cerberus_backfill_poll_interval_seconds
        self.stall_timeout = settings.cerberus_backfill_stall_timeout_seconds
        self.chunk_days = settings.cerberus_backfill_chunk_days

    def provision_bars(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        timeframe: str = "1Min",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Provision bar data: backfill via Gateway, read from Heber.

        Args:
            symbols: List of symbols to provision.
            start_date: Start of date range.
            end_date: End of date range.
            timeframe: Bar timeframe (e.g. "1Min", "1Day").

        Returns:
            Dict mapping symbol → list of normalized bar dicts.

        Raises:
            ProvisioningError: If provisioning fails for all symbols.
        """
        if not symbols:
            return {}

        self.logger.info(
            "Starting bar data provisioning",
            symbols=len(symbols),
            start=str(start_date),
            end=str(end_date),
            timeframe=timeframe,
        )

        # Submit backfill(s) — chunk large date ranges
        chunks = self._build_date_chunks(start_date, end_date)
        job_ids: List[str] = []

        for chunk_start, chunk_end in chunks:
            try:
                job = self.api_client.request_backfill(
                    provider="alpaca",
                    feed="bars",
                    start_date=chunk_start,
                    end_date=chunk_end,
                    symbols=symbols,
                    timeframe=timeframe,
                )
                job_id = job.get("job_id")
                if job_id:
                    job_ids.append(job_id)
                    self.logger.info(
                        "Backfill chunk submitted",
                        job_id=job_id,
                        chunk=f"{chunk_start} to {chunk_end}",
                    )
            except Exception as exc:
                self.logger.error(
                    "Backfill chunk submission failed",
                    chunk=f"{chunk_start} to {chunk_end}",
                    error=str(exc),
                    exc_info=True,
                )
                raise ProvisioningError(f"Failed to submit backfill for {chunk_start} to {chunk_end}: {exc}") from exc

        if not job_ids:
            raise ProvisioningError("No backfill jobs were submitted")

        # Wait for all backfill jobs to complete
        for job_id in job_ids:
            try:
                self.api_client.wait_for_backfill(
                    job_id,
                    timeout_seconds=self.timeout_seconds,
                    poll_interval_seconds=self.poll_interval,
                    stall_timeout_seconds=self.stall_timeout,
                )
            except (BackfillTimeoutError, BackfillFailedError) as exc:
                self.logger.error(
                    "Backfill wait failed",
                    job_id=job_id,
                    error=str(exc),
                    exc_info=True,
                )
                raise ProvisioningError(f"Backfill job {job_id} did not complete: {exc}") from exc

        # Read from Heber
        bars_by_symbol = self._read_from_heber(
            symbols,
            start_date,
            end_date,
            timeframe,
        )

        # Fall back to direct Gateway fetch for symbols with no Heber data
        missing = [s for s in symbols if not bars_by_symbol.get(s)]
        if missing:
            self.logger.warning(
                "Heber read returned no data for some symbols, falling back to Gateway",
                missing_count=len(missing),
                missing_symbols=missing[:10],
            )
            gateway_bars = self._fetch_from_gateway(
                missing,
                start_date,
                end_date,
                timeframe,
            )
            bars_by_symbol.update(gateway_bars)

        loaded = sum(len(v) for v in bars_by_symbol.values())
        self.logger.info(
            "Bar data provisioning complete",
            symbols_requested=len(symbols),
            symbols_loaded=sum(1 for v in bars_by_symbol.values() if v),
            total_bars=loaded,
        )
        return bars_by_symbol

    def _build_date_chunks(
        self,
        start: date,
        end: date,
    ) -> List[tuple[date, date]]:
        """Split a date range into chunks of at most chunk_days."""
        total_days = (end - start).days
        if total_days <= self.chunk_days:
            return [(start, end)]

        chunks: List[tuple[date, date]] = []
        num_chunks = math.ceil(total_days / self.chunk_days)
        for i in range(num_chunks):
            chunk_start = start + timedelta(days=i * self.chunk_days)
            chunk_end = min(start + timedelta(days=(i + 1) * self.chunk_days), end)
            chunks.append((chunk_start, chunk_end))
        return chunks

    def _read_from_heber(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        timeframe: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Read bar data from Heber Silver for all symbols."""
        if self.heber_client is None:
            self.logger.warning("HeberReadClient not configured, skipping Heber read")
            return {}

        result: Dict[str, List[Dict[str, Any]]] = {}
        start_dt = datetime(
            start_date.year,
            start_date.month,
            start_date.day,
            tzinfo=timezone.utc,
        )
        end_dt = datetime(
            end_date.year,
            end_date.month,
            end_date.day,
            23,
            59,
            59,
            tzinfo=timezone.utc,
        )

        for symbol in symbols:
            try:
                bars = self.heber_client.get_bars(
                    symbol=symbol,
                    start=start_dt,
                    end=end_dt,
                    timeframe=timeframe,
                )
                result[symbol] = bars
                if bars:
                    self.logger.debug(
                        "Heber read success",
                        symbol=symbol,
                        bar_count=len(bars),
                    )
            except Exception as exc:
                self.logger.warning(
                    "Heber read failed for symbol",
                    symbol=symbol,
                    error=str(exc),
                )
                result[symbol] = []

        return result

    def _fetch_from_gateway(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        timeframe: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback: fetch bars directly from Data Gateway HTTP API."""
        result: Dict[str, List[Dict[str, Any]]] = {}
        start_dt = datetime(
            start_date.year,
            start_date.month,
            start_date.day,
            tzinfo=timezone.utc,
        )
        end_dt = datetime(
            end_date.year,
            end_date.month,
            end_date.day,
            23,
            59,
            59,
            tzinfo=timezone.utc,
        )

        for symbol in symbols:
            try:
                payload = self.api_client.get_alpaca_bars(
                    symbol=symbol,
                    start=start_dt,
                    end=end_dt,
                    timeframe=timeframe,
                )
                bars = payload.get("bars", [])
                result[symbol] = bars if isinstance(bars, list) else []
            except Exception as exc:
                self.logger.warning(
                    "Gateway fallback fetch failed for symbol",
                    symbol=symbol,
                    error=str(exc),
                )
                result[symbol] = []

        return result

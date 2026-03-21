from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.core.logger import StructuredLogger


class HeberReadClient:
    """Read point-in-time-safe market data from Heber Silver parquet partitions."""

    # Map CERBERUS_ASSET_CLASS values to Heber partition directory names
    _ASSET_CLASS_TO_INSTRUMENT_TYPE: dict[str, str] = {
        "us_equity": "equity",
        "equity": "equity",
        "crypto": "crypto",
        "option": "option",
        "options": "option",
    }

    def __init__(
        self,
        data_root: Path | str,
        logger: StructuredLogger,
        instrument_type: str = "equity",
    ) -> None:
        self.data_root = Path(data_root)
        self.logger = logger
        # Resolve asset class alias to the Heber partition name
        self.instrument_type = self._ASSET_CLASS_TO_INSTRUMENT_TYPE.get(
            instrument_type.lower().strip(), instrument_type.lower().strip()
        )

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Read bars from Heber Silver and normalize to Cerberus bar shape."""
        sym = str(symbol).strip().upper()
        rows = self._read_asof_rows(
            dataset="bars",
            symbol=sym,
            instrument_key=f"{self.instrument_type}:{sym}",
            start=start,
            end=end,
            as_of=as_of or end,
        )

        normalized: list[dict[str, Any]] = []
        timeframe_lower = str(timeframe or "").lower()
        for row in rows:
            row_timeframe = str(row.get("timeframe") or "").lower()
            if row_timeframe and timeframe_lower and row_timeframe != timeframe_lower:
                continue

            bar_ts = self._parse_ts(row.get("bar_start_ts")) or self._parse_ts(row.get("ts_event"))
            if bar_ts is None:
                continue

            normalized.append(
                {
                    "t": bar_ts.isoformat(),
                    "o": self._to_float(row.get("open")),
                    "h": self._to_float(row.get("high")),
                    "l": self._to_float(row.get("low")),
                    "c": self._to_float(row.get("close")),
                    "v": self._to_float(row.get("volume")),
                    "n": self._to_int(row.get("trade_count")),
                    "vw": self._to_float(row.get("vwap")),
                }
            )

        normalized.sort(key=lambda row: str(row.get("t", "")))
        return normalized

    def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Read trades from Heber Silver and normalize to Cerberus trade shape."""
        sym = str(symbol).strip().upper()
        rows = self._read_asof_rows(
            dataset="trades",
            symbol=sym,
            instrument_key=f"{self.instrument_type}:{sym}",
            start=start,
            end=end,
            as_of=as_of or end,
        )

        normalized: list[dict[str, Any]] = []
        for row in rows:
            trade_ts = self._parse_ts(row.get("ts_event"))
            if trade_ts is None:
                continue

            normalized.append(
                {
                    "t": trade_ts.isoformat(),
                    "p": self._to_float(row.get("price")),
                    "s": self._to_float(row.get("size")),
                    "x": str(row.get("exchange") or ""),
                    "i": str(row.get("trade_id") or ""),
                    "z": str(row.get("tape") or ""),
                }
            )

        normalized.sort(key=lambda row: str(row.get("t", "")))
        return normalized

    # ------------------------------------------------------------------
    # Phase 3: Advanced strategy data feeds
    # ------------------------------------------------------------------

    def get_iv_term_structure(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Read IV term structure from Heber Silver."""
        sym = str(symbol).strip().upper()
        rows = self._read_asof_rows(
            dataset="iv_term_structure",
            symbol=sym,
            instrument_key=f"equity:{sym}",
            start=start,
            end=end,
            as_of=as_of or end,
        )
        return [
            {
                "symbol": str(row.get("symbol") or sym),
                "expiry": str(row.get("expiry") or ""),
                "iv": self._to_float(row.get("iv")),
                "dte": self._to_int(row.get("dte")),
                "call_iv": self._to_float(row.get("call_iv")),
                "put_iv": self._to_float(row.get("put_iv")),
                "timestamp": str(row.get("ts_event") or ""),
            }
            for row in rows
        ]

    def get_etf_tide(
        self,
        etf_symbol: str,
        start: datetime,
        end: datetime,
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Read ETF tide (options flow sentiment) from Heber Silver."""
        sym = str(etf_symbol).strip().upper()
        rows = self._read_asof_rows(
            dataset="etf_tide",
            symbol=sym,
            instrument_key=f"equity:{sym}",
            start=start,
            end=end,
            as_of=as_of or end,
        )
        return [
            {
                "symbol": str(row.get("symbol") or row.get("ticker") or sym),
                "net_call_premium": self._to_float(row.get("net_call_premium")),
                "net_put_premium": self._to_float(row.get("net_put_premium")),
                "net_volume": self._to_int(row.get("net_volume")),
                "sentiment": str(row.get("sentiment") or ""),
                "timestamp": str(row.get("ts_event") or ""),
            }
            for row in rows
        ]

    def get_flow_alerts(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Read options flow alerts from Heber Silver."""
        sym = str(symbol).strip().upper()
        rows = self._read_asof_rows(
            dataset="flow_alerts",
            symbol=sym,
            instrument_key=f"equity:{sym}",
            start=start,
            end=end,
            as_of=as_of or end,
        )
        return [
            {
                "symbol": str(row.get("symbol") or row.get("underlying") or sym),
                "premium": self._to_float(row.get("premium")),
                "volume": self._to_float(row.get("volume")),
                "put_call": str(row.get("put_call") or ""),
                "strike": self._to_float(row.get("strike")),
                "expiry": str(row.get("expiry") or ""),
                "sentiment": str(row.get("sentiment") or ""),
                "timestamp": str(row.get("ts_event") or ""),
            }
            for row in rows
        ]

    def get_auctions(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Read NYSE auction data from Heber Silver."""
        sym = str(symbol).strip().upper()
        rows = self._read_asof_rows(
            dataset="auctions",
            symbol=sym,
            instrument_key=f"equity:{sym}",
            start=start,
            end=end,
            as_of=as_of or end,
        )
        return [
            {
                "symbol": str(row.get("symbol") or sym),
                "auction_type": str(row.get("auction_type") or ""),
                "auction_price": self._to_float(row.get("auction_price")),
                "imbalance_size": self._to_int(row.get("imbalance_size")),
                "imbalance_side": str(row.get("imbalance_side") or ""),
                "paired_shares": self._to_int(row.get("paired_shares")),
                "reference_price": self._to_float(row.get("reference_price")),
                "timestamp": str(row.get("ts_event") or ""),
            }
            for row in rows
        ]

    def _read_asof_rows(
        self,
        dataset: str,
        symbol: str,
        instrument_key: str,
        start: datetime,
        end: datetime,
        as_of: datetime,
    ) -> list[dict[str, Any]]:
        dataset_root = self.data_root / "silver" / f"feed={dataset}"
        if not dataset_root.exists():
            self.logger.debug(
                "Heber dataset path missing",
                dataset=dataset,
                path=str(dataset_root),
            )
            return []

        start_utc = self._ensure_utc(start)
        end_utc = self._ensure_utc(end)
        as_of_utc = self._ensure_utc(as_of)
        symbol_upper = symbol.upper()
        key_set = {instrument_key, instrument_key.lower(), instrument_key.upper()}

        rows: list[dict[str, Any]] = []
        for parquet_file in self._candidate_files(dataset_root, start_utc, end_utc):
            for row in self._read_parquet_rows(dataset, parquet_file):
                row_symbol = str(row.get("symbol") or "").upper()
                row_key = str(row.get("instrument_key") or "")
                if row_symbol and row_symbol != symbol_upper and row_key not in key_set:
                    continue
                if not row_symbol and row_key and row_key not in key_set:
                    continue

                ts_event = self._parse_ts(row.get("ts_event")) or self._parse_ts(row.get("bar_start_ts"))
                if ts_event is None:
                    continue
                if ts_event < start_utc or ts_event > end_utc:
                    continue

                ts_available = self._parse_ts(row.get("ts_available"))
                if ts_available is None or ts_available > as_of_utc:
                    continue

                rows.append(row)

        rows.sort(
            key=lambda row: self._parse_ts(row.get("ts_event")) or self._parse_ts(row.get("bar_start_ts")) or start_utc
        )
        return rows

    def _candidate_files(self, dataset_root: Path, start: datetime, end: datetime) -> list[Path]:
        start_date = (start - timedelta(days=1)).date()
        end_date = (end + timedelta(days=1)).date()
        files: list[Path] = []

        # Scope to the correct instrument_type partition to avoid scanning
        # unrelated asset types (e.g. crypto files when looking for equities).
        instrument_dir = dataset_root / f"instrument_type={self.instrument_type}"
        if instrument_dir.is_dir():
            search_root = instrument_dir
        else:
            self.logger.debug(
                "Instrument type partition not found; falling back to full scan",
                instrument_type=self.instrument_type,
                expected_path=str(instrument_dir),
            )
            search_root = dataset_root

        for parquet_file in search_root.rglob("*.parquet"):
            if not parquet_file.is_file():
                continue

            partition_date = self._extract_partition_date(parquet_file)
            if partition_date is not None and (partition_date < start_date or partition_date > end_date):
                continue
            files.append(parquet_file)

        files.sort(key=lambda p: str(p))
        return files

    def _read_parquet_rows(self, dataset: str, parquet_file: Path) -> list[dict[str, Any]]:
        try:
            # Read the file directly to avoid hive-partition schema merge conflicts
            # from path segments like feed=bars when the file also contains `feed`.
            table = pq.ParquetFile(parquet_file).read()
            return list(table.to_pylist())
        except Exception as e:
            self.logger.warning(
                "Failed to read Heber parquet file",
                dataset=dataset,
                file=str(parquet_file),
                error=f"{type(e).__name__}: {e}",
            )
            return []

    @staticmethod
    def _extract_partition_date(parquet_file: Path) -> date | None:
        for part in parquet_file.parts:
            if not part.startswith("dt="):
                continue
            raw = part.split("=", 1)[1].strip()
            try:
                return date.fromisoformat(raw)
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_ts(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            try:
                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return None

        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @staticmethod
    def _to_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _to_int(value: Any) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

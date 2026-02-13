from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.heber_read_client import HeberReadClient


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


@pytest.mark.unit
def test_heber_read_client_filters_ts_available_and_maps_bars(tmp_path: Path) -> None:
    start = datetime(2026, 2, 10, 14, 30, tzinfo=UTC)
    end = datetime(2026, 2, 10, 14, 36, tzinfo=UTC)
    as_of = datetime(2026, 2, 10, 14, 35, tzinfo=UTC)

    bars_file = tmp_path / "silver" / "feed=bars" / "instrument_type=equity" / "dt=2026-02-10" / "part-0001.parquet"
    _write_parquet(
        bars_file,
        [
            {
                "symbol": "AAPL",
                "instrument_key": "equity:AAPL",
                "timeframe": "1Min",
                "bar_start_ts": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
                "open": 200.1,
                "high": 200.8,
                "low": 199.9,
                "close": 200.4,
                "volume": 12345.0,
                "trade_count": 123,
                "vwap": 200.33,
                "ts_event": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
                "ts_available": datetime(2026, 2, 10, 14, 32, tzinfo=UTC),
            },
            {
                "symbol": "AAPL",
                "instrument_key": "equity:AAPL",
                "timeframe": "1Min",
                "bar_start_ts": datetime(2026, 2, 10, 14, 34, tzinfo=UTC),
                "open": 201.0,
                "high": 201.5,
                "low": 200.8,
                "close": 201.3,
                "volume": 9999.0,
                "trade_count": 88,
                "vwap": 201.2,
                "ts_event": datetime(2026, 2, 10, 14, 34, tzinfo=UTC),
                "ts_available": datetime(2026, 2, 10, 14, 40, tzinfo=UTC),
            },
            {
                "symbol": "MSFT",
                "instrument_key": "equity:MSFT",
                "timeframe": "1Min",
                "bar_start_ts": datetime(2026, 2, 10, 14, 33, tzinfo=UTC),
                "open": 410.0,
                "high": 411.0,
                "low": 409.0,
                "close": 410.5,
                "volume": 5000.0,
                "trade_count": 42,
                "vwap": 410.3,
                "ts_event": datetime(2026, 2, 10, 14, 33, tzinfo=UTC),
                "ts_available": datetime(2026, 2, 10, 14, 33, tzinfo=UTC),
            },
        ],
    )

    client = HeberReadClient(data_root=tmp_path, logger=MagicMock())
    bars = client.get_bars(
        symbol="AAPL",
        start=start,
        end=end,
        timeframe="1Min",
        as_of=as_of,
    )

    assert len(bars) == 1
    assert bars[0]["t"].startswith("2026-02-10T14:31")
    assert bars[0]["o"] == 200.1
    assert bars[0]["h"] == 200.8
    assert bars[0]["l"] == 199.9
    assert bars[0]["c"] == 200.4
    assert bars[0]["v"] == 12345.0
    assert bars[0]["n"] == 123


@pytest.mark.unit
def test_heber_read_client_maps_trades_shape_for_tfi(tmp_path: Path) -> None:
    start = datetime(2026, 2, 10, 14, 30, tzinfo=UTC)
    end = datetime(2026, 2, 10, 14, 36, tzinfo=UTC)
    as_of = datetime(2026, 2, 10, 14, 36, tzinfo=UTC)

    trades_file = (
        tmp_path
        / "silver"
        / "feed=trades"
        / "instrument_type=equity"
        / "dt=2026-02-10"
        / "hour=14"
        / "part-0002.parquet"
    )
    _write_parquet(
        trades_file,
        [
            {
                "symbol": "AAPL",
                "instrument_key": "equity:AAPL",
                "trade_id": "t1",
                "price": 200.5,
                "size": 25.0,
                "exchange": "XNAS",
                "tape": "C",
                "ts_event": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
                "ts_available": datetime(2026, 2, 10, 14, 31, tzinfo=UTC),
            },
            {
                "symbol": "AAPL",
                "instrument_key": "equity:AAPL",
                "trade_id": "t2",
                "price": 200.7,
                "size": 10.0,
                "exchange": "XNYS",
                "tape": "A",
                "ts_event": datetime(2026, 2, 10, 14, 32, tzinfo=UTC),
                "ts_available": datetime(2026, 2, 10, 14, 32, tzinfo=UTC),
            },
        ],
    )

    client = HeberReadClient(data_root=tmp_path, logger=MagicMock())
    trades = client.get_trades(
        symbol="AAPL",
        start=start,
        end=end,
        as_of=as_of,
    )

    assert len(trades) == 2
    assert trades[0]["p"] == 200.5
    assert trades[0]["s"] == 25.0
    assert trades[0]["i"] == "t1"
    assert trades[1]["x"] == "XNYS"


@pytest.mark.unit
def test_heber_read_client_reads_partitioned_file_when_feed_column_is_dictionary_encoded(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 2, 12, 14, 30, tzinfo=UTC)
    end = datetime(2026, 2, 12, 14, 35, tzinfo=UTC)
    as_of = datetime(2026, 2, 12, 14, 35, tzinfo=UTC)

    bars_file = tmp_path / "silver" / "feed=bars" / "instrument_type=equity" / "dt=2026-02-12" / "part-0003.parquet"
    bars_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "feed": pa.array(["bars"], type=pa.dictionary(pa.int8(), pa.string())),
            "symbol": ["AAPL"],
            "instrument_key": ["equity:AAPL"],
            "timeframe": ["1Min"],
            "bar_start_ts": ["2026-02-12T14:31:00Z"],
            "open": [200.1],
            "high": [200.8],
            "low": [199.9],
            "close": [200.4],
            "volume": [12345.0],
            "trade_count": [123],
            "vwap": [200.33],
            "ts_event": ["2026-02-12T14:31:00Z"],
            "ts_available": ["2026-02-12T14:31:01Z"],
        }
    )
    pq.write_table(table, bars_file)

    logger = MagicMock()
    client = HeberReadClient(data_root=tmp_path, logger=logger)
    bars = client.get_bars(
        symbol="AAPL",
        start=start,
        end=end,
        timeframe="1Min",
        as_of=as_of,
    )

    assert len(bars) == 1
    assert bars[0]["c"] == 200.4
    logger.warning.assert_not_called()

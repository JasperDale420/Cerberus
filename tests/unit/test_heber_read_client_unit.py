"""Unit tests for HeberReadClient."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.heber_read_client import HeberReadClient


@pytest.fixture()
def mock_logger():
    return MagicMock()


@pytest.fixture()
def client(mock_logger):
    return HeberReadClient(data_root="/tmp/heber", logger=mock_logger)


class TestReadParquetRowsRetry:
    """Tests for HeberReadClient._read_parquet_rows retry logic."""

    @patch("src.data.heber_read_client.pq.ParquetFile")
    @patch("src.data.heber_read_client.time.sleep")
    def test_succeeds_first_try(self, mock_sleep, mock_pq_file, client):
        """Standard success case, no retries."""
        # Setup mock table
        mock_table = MagicMock()
        mock_table.to_pylist.return_value = [{"col": "val"}]

        mock_fq = MagicMock()
        mock_fq.read.return_value = mock_table
        mock_pq_file.return_value = mock_fq

        result = client._read_parquet_rows("dataset", Path("test.parquet"))

        assert result == [{"col": "val"}]
        assert mock_pq_file.call_count == 1
        mock_sleep.assert_not_called()

    @patch("src.data.heber_read_client.pq.ParquetFile")
    @patch("src.data.heber_read_client.time.sleep")
    def test_retries_on_file_not_found(self, mock_sleep, mock_pq_file, client):
        """Retries on FileNotFoundError and succeeds."""
        # Setup mock table
        mock_table = MagicMock()
        mock_table.to_pylist.return_value = [{"col": "val"}]
        mock_fq = MagicMock()
        mock_fq.read.return_value = mock_table

        # Fail twice with FileNotFoundError, then return mock_fq
        mock_pq_file.side_effect = [FileNotFoundError, FileNotFoundError, mock_fq]

        result = client._read_parquet_rows("dataset", Path("test.parquet"))

        assert result == [{"col": "val"}]
        assert mock_pq_file.call_count == 3
        assert mock_sleep.call_count == 2

        # Verify no warning log for "missing (race condition)" yet because it succeeded
        # Actually it might not log warning on success, let's check implementation
        # The implementation only logs warning if it fails after retries.
        client.logger.warning.assert_not_called()

    @patch("src.data.heber_read_client.pq.ParquetFile")
    @patch("src.data.heber_read_client.time.sleep")
    def test_fails_after_max_retries(self, mock_sleep, mock_pq_file, client):
        """Fails after max retries on FileNotFoundError."""
        mock_pq_file.side_effect = FileNotFoundError

        result = client._read_parquet_rows("dataset", Path("test.parquet"))

        assert result == []
        assert mock_pq_file.call_count == 3
        assert mock_sleep.call_count == 2

        # Verify warning log
        client.logger.warning.assert_called_with(
            "Heber parquet file missing (race condition)",
            dataset="dataset",
            file="test.parquet",
        )

    @patch("src.data.heber_read_client.pq.ParquetFile")
    @patch("src.data.heber_read_client.time.sleep")
    def test_no_retry_on_other_exceptions(self, mock_sleep, mock_pq_file, client):
        """Does not retry on general exceptions (e.g. PermissionError)."""
        mock_pq_file.side_effect = PermissionError("Access denied")

        result = client._read_parquet_rows("dataset", Path("test.parquet"))

        assert result == []
        assert mock_pq_file.call_count == 1
        mock_sleep.assert_not_called()

        client.logger.warning.assert_called_with(
            "Failed to read Heber parquet file",
            dataset="dataset",
            file="test.parquet",
            error="PermissionError: Access denied",
        )

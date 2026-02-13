"""Unit tests for BacktestDataProvisioner."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.data.api_client import BackfillFailedError, BackfillTimeoutError
from src.data.backtest_data_provisioner import (
    BacktestDataProvisioner,
    ProvisioningError,
)


@pytest.fixture()
def mock_settings():
    """Patch get_settings to return test-friendly defaults."""
    with patch("src.data.backtest_data_provisioner.get_settings") as mock:
        settings = MagicMock()
        settings.cerberus_backfill_timeout_seconds = 60.0
        settings.cerberus_backfill_poll_interval_seconds = 0.01
        settings.cerberus_backfill_stall_timeout_seconds = 30.0
        settings.cerberus_backfill_chunk_days = 90
        mock.return_value = settings
        yield settings


@pytest.fixture()
def provisioner(mock_settings):
    """Create a BacktestDataProvisioner with fully mocked dependencies."""
    api_client = MagicMock()
    heber_client = MagicMock()
    logger = MagicMock()
    return BacktestDataProvisioner(
        api_client=api_client,
        heber_read_client=heber_client,
        logger=logger,
    )


class TestProvisionBars:
    """Tests for BacktestDataProvisioner.provision_bars."""

    def test_full_provision_flow(self, provisioner):
        """Submit → poll → read succeeds with mocked Gateway + Heber."""
        provisioner.api_client.request_backfill.return_value = {
            "job_id": "bf-001",
            "status": "submitted",
        }
        provisioner.api_client.wait_for_backfill.return_value = {
            "job_id": "bf-001",
            "status": "completed",
            "records_published": 500,
        }
        provisioner.heber_client.get_bars.return_value = [
            {"t": "2025-01-02T14:30:00", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000},
            {"t": "2025-01-02T14:31:00", "o": 100.5, "h": 102.0, "l": 100.0, "c": 101.0, "v": 1500},
        ]

        result = provisioner.provision_bars(
            symbols=["AAPL"],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 1),
            timeframe="1Min",
        )

        assert "AAPL" in result
        assert len(result["AAPL"]) == 2

        provisioner.api_client.request_backfill.assert_called_once()
        call_kwargs = provisioner.api_client.request_backfill.call_args
        assert call_kwargs[1]["provider"] == "alpaca" or call_kwargs[0][0] == "alpaca"
        provisioner.api_client.wait_for_backfill.assert_called_once_with(
            "bf-001",
            timeout_seconds=60.0,
            poll_interval_seconds=0.01,
            stall_timeout_seconds=30.0,
        )

    def test_returns_empty_dict_for_no_symbols(self, provisioner):
        """provision_bars returns {} when symbols list is empty."""
        result = provisioner.provision_bars(
            symbols=[],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 1),
        )
        assert result == {}

    def test_chunked_backfill_for_large_ranges(self, provisioner):
        """Date ranges > chunk_days are split into multiple backfill jobs."""
        provisioner.chunk_days = 30

        provisioner.api_client.request_backfill.return_value = {
            "job_id": "bf-chunk",
            "status": "submitted",
        }
        provisioner.api_client.wait_for_backfill.return_value = {
            "job_id": "bf-chunk",
            "status": "completed",
            "records_published": 100,
        }
        provisioner.heber_client.get_bars.return_value = [
            {"t": "2025-01-15T10:00:00", "o": 50.0, "h": 51.0, "l": 49.0, "c": 50.5, "v": 500},
        ]

        result = provisioner.provision_bars(
            symbols=["TSLA"],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 4, 1),
            timeframe="1Day",
        )

        # 90 days / 30-day chunks = 3 backfill requests
        assert provisioner.api_client.request_backfill.call_count == 3
        assert provisioner.api_client.wait_for_backfill.call_count == 3
        assert "TSLA" in result

    def test_gateway_fallback_when_heber_empty(self, provisioner):
        """Falls back to Gateway fetch when Heber returns no bars."""
        provisioner.api_client.request_backfill.return_value = {
            "job_id": "bf-002",
            "status": "submitted",
        }
        provisioner.api_client.wait_for_backfill.return_value = {
            "job_id": "bf-002",
            "status": "completed",
            "records_published": 0,
        }
        # Heber returns empty
        provisioner.heber_client.get_bars.return_value = []
        # Gateway fallback returns data
        provisioner.api_client.get_alpaca_bars.return_value = {
            "bars": [
                {"t": "2025-01-02T14:30:00", "o": 200.0, "h": 201.0, "l": 199.0, "c": 200.5, "v": 2000},
            ]
        }

        result = provisioner.provision_bars(
            symbols=["NVDA"],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 2, 1),
        )

        assert "NVDA" in result
        assert len(result["NVDA"]) == 1
        provisioner.api_client.get_alpaca_bars.assert_called_once()

    def test_raises_on_backfill_failure(self, provisioner):
        """ProvisioningError raised when backfill wait fails."""
        provisioner.api_client.request_backfill.return_value = {
            "job_id": "bf-fail",
            "status": "submitted",
        }
        provisioner.api_client.wait_for_backfill.side_effect = BackfillFailedError("Provider returned 500")

        with pytest.raises(ProvisioningError, match="did not complete"):
            provisioner.provision_bars(
                symbols=["AAPL"],
                start_date=date(2025, 1, 1),
                end_date=date(2025, 2, 1),
            )

    def test_raises_on_backfill_timeout(self, provisioner):
        """ProvisioningError raised when backfill times out."""
        provisioner.api_client.request_backfill.return_value = {
            "job_id": "bf-timeout",
            "status": "submitted",
        }
        provisioner.api_client.wait_for_backfill.side_effect = BackfillTimeoutError("Timed out after 60s")

        with pytest.raises(ProvisioningError, match="did not complete"):
            provisioner.provision_bars(
                symbols=["AAPL"],
                start_date=date(2025, 1, 1),
                end_date=date(2025, 2, 1),
            )

    def test_raises_on_submit_failure(self, provisioner):
        """ProvisioningError raised when backfill submission fails."""
        provisioner.api_client.request_backfill.side_effect = Exception("Gateway unreachable")

        with pytest.raises(ProvisioningError, match="Failed to submit"):
            provisioner.provision_bars(
                symbols=["AAPL"],
                start_date=date(2025, 1, 1),
                end_date=date(2025, 2, 1),
            )

    def test_no_heber_client_skips_read(self, mock_settings):
        """When HeberReadClient is None, skips Heber read and uses Gateway fallback."""
        api_client = MagicMock()
        logger = MagicMock()
        prov = BacktestDataProvisioner(
            api_client=api_client,
            heber_read_client=None,
            logger=logger,
        )

        api_client.request_backfill.return_value = {
            "job_id": "bf-003",
            "status": "submitted",
        }
        api_client.wait_for_backfill.return_value = {
            "job_id": "bf-003",
            "status": "completed",
            "records_published": 10,
        }
        api_client.get_alpaca_bars.return_value = {
            "bars": [{"t": "2025-01-02T10:00:00", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 100}]
        }

        result = prov.provision_bars(
            symbols=["SPY"],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 2, 1),
        )

        assert "SPY" in result
        assert len(result["SPY"]) == 1


class TestBuildDateChunks:
    """Tests for BacktestDataProvisioner._build_date_chunks."""

    def test_small_range_no_chunking(self, provisioner):
        """Ranges <= chunk_days produce a single chunk."""
        chunks = provisioner._build_date_chunks(date(2025, 1, 1), date(2025, 3, 1))
        assert len(chunks) == 1
        assert chunks[0] == (date(2025, 1, 1), date(2025, 3, 1))

    def test_large_range_produces_multiple_chunks(self, provisioner):
        """Ranges > chunk_days are split into multiple chunks."""
        provisioner.chunk_days = 30
        chunks = provisioner._build_date_chunks(date(2025, 1, 1), date(2025, 4, 1))
        assert len(chunks) == 3
        assert chunks[0][0] == date(2025, 1, 1)
        assert chunks[-1][1] == date(2025, 4, 1)

"""Unit tests for CentralApiClient backfill methods."""

from datetime import date
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.data.api_client import (
    BackfillFailedError,
    BackfillTimeoutError,
    CentralApiClient,
)


@pytest.fixture()
def api_client():
    """Create a CentralApiClient with mocked config."""
    config = MagicMock()
    config.get_env = MagicMock(
        side_effect=lambda key, default="": {
            "CERBERUS_GATEWAY_URL": "http://gateway:8080",
            "DATA_INGESTION_URL": "http://gateway:8080",
            "CERBERUS_GATEWAY_KEY": "test-key",
            "CERBERUS_GATEWAY_TIMEOUT_SECONDS": "30",
            "CERBERUS_GATEWAY_MAX_RETRIES": "0",
            "CERBERUS_GATEWAY_RETRY_BACKOFF_SECONDS": "0",
            "CENTRAL_LLM_API_URL": "http://gateway:8080",
        }.get(key, default)
    )
    logger = MagicMock()
    return CentralApiClient(config, logger)


class TestRequestBackfill:
    """Tests for CentralApiClient.request_backfill."""

    def test_sends_correct_payload(self, api_client):
        """Backfill request sends correct POST payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"job_id": "bf-123", "status": "submitted"}}

        api_client.client = MagicMock()
        api_client.client.request.return_value = mock_response

        result = api_client.request_backfill(
            provider="alpaca",
            feed="bars",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 6, 1),
            symbols=["AAPL", "MSFT"],
            timeframe="1Min",
        )

        assert result["job_id"] == "bf-123"
        assert result["status"] == "submitted"

        call_args = api_client.client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/backfill")
        payload = call_args[1]["json"]
        assert payload["provider"] == "alpaca"
        assert payload["feed"] == "bars"
        assert payload["symbols"] == ["AAPL", "MSFT"]
        assert payload["start"] == "2025-01-01"
        assert payload["end"] == "2025-06-01"
        assert payload["timeframe"] == "1Min"

    def test_raises_on_empty_symbols(self, api_client):
        """Backfill raises ValueError when symbols list is empty."""
        with pytest.raises(ValueError, match="requires explicit symbols"):
            api_client.request_backfill(
                provider="alpaca",
                feed="bars",
                start_date=date(2025, 1, 1),
                end_date=date(2025, 6, 1),
                symbols=[],
            )


class TestGetBackfillStatus:
    """Tests for CentralApiClient.get_backfill_status."""

    def test_returns_job_dict(self, api_client):
        """get_backfill_status returns the unwrapped job dict."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"job_id": "bf-123", "status": "running", "records_published": 500}}

        api_client.client = MagicMock()
        api_client.client.request.return_value = mock_response

        result = api_client.get_backfill_status("bf-123")
        assert result["status"] == "running"
        assert result["records_published"] == 500

    def test_raises_on_404(self, api_client):
        """get_backfill_status raises BackfillFailedError on 404."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.is_error = True

        api_client.client = MagicMock()
        api_client.client.request.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response,
        )

        with pytest.raises(BackfillFailedError, match="not found"):
            api_client.get_backfill_status("bf-unknown")


class TestWaitForBackfill:
    """Tests for CentralApiClient.wait_for_backfill."""

    def test_returns_on_completion(self, api_client):
        """wait_for_backfill returns job when status is 'completed'."""
        completed_job = {
            "job_id": "bf-123",
            "status": "completed",
            "records_published": 1000,
        }
        api_client.get_backfill_status = MagicMock(return_value=completed_job)

        result = api_client.wait_for_backfill(
            "bf-123",
            timeout_seconds=5,
            poll_interval_seconds=0.01,
        )
        assert result["status"] == "completed"
        assert result["records_published"] == 1000

    def test_raises_on_failure(self, api_client):
        """wait_for_backfill raises BackfillFailedError on 'failed' status."""
        failed_job = {
            "job_id": "bf-123",
            "status": "failed",
            "errors": ["Provider returned 500"],
        }
        api_client.get_backfill_status = MagicMock(return_value=failed_job)

        with pytest.raises(BackfillFailedError, match="Provider returned 500"):
            api_client.wait_for_backfill(
                "bf-123",
                timeout_seconds=5,
                poll_interval_seconds=0.01,
            )

    def test_raises_on_cancelled(self, api_client):
        """wait_for_backfill raises BackfillFailedError on 'cancelled' status."""
        cancelled_job = {"job_id": "bf-123", "status": "cancelled"}
        api_client.get_backfill_status = MagicMock(return_value=cancelled_job)

        with pytest.raises(BackfillFailedError, match="was cancelled"):
            api_client.wait_for_backfill(
                "bf-123",
                timeout_seconds=5,
                poll_interval_seconds=0.01,
            )

    @patch("src.data.api_client.time")
    def test_raises_on_hard_timeout(self, mock_time, api_client):
        """wait_for_backfill raises BackfillTimeoutError when hard timeout exceeded."""
        # monotonic() calls: start=0, then elapsed=6 (exceeds 5s timeout)
        mock_time.monotonic.side_effect = [0.0, 6.0]
        mock_time.sleep = MagicMock()

        with pytest.raises(BackfillTimeoutError, match="timed out"):
            api_client.wait_for_backfill(
                "bf-123",
                timeout_seconds=5,
                poll_interval_seconds=0.01,
            )

    @patch("src.data.api_client.time")
    def test_raises_on_stall_timeout(self, mock_time, api_client):
        """wait_for_backfill raises BackfillTimeoutError when progress stalls."""
        stalled_job = {
            "job_id": "bf-123",
            "status": "running",
            "records_published": 100,
        }
        api_client.get_backfill_status = MagicMock(return_value=stalled_job)

        # Call pattern in wait_for_backfill:
        # 1. start = monotonic()                          => 0.0
        # 2. elapsed = monotonic() - start                => 1.0 - 0 = 1  (ok)
        # 3. stall_duration = monotonic() - last_progress  => 1.0 - 0 = 1, but last_progress_records=0 so skipped
        # 4. get_backfill_status succeeds (records=100 > 0)
        # 5. last_progress_time = monotonic()             => 2.0
        # 6. (debug log) monotonic()                      => not called since records > last
        # 7. time.sleep()
        # LOOP 2:
        # 8. elapsed = monotonic() - start                => 3.0 - 0 = 3 (ok)
        # 9. stall_duration = monotonic() - 2.0           => 310.0 - 2 = 308 (exceeds 300s)
        monotonic_values = [
            0.0,  # start
            1.0,  # elapsed check (iter 1)
            1.0,  # stall_duration check (iter 1) — skipped (no prior progress)
            2.0,  # current_records > last → set last_progress_time
            3.0,  # elapsed check (iter 2)
            310.0,  # stall_duration check (iter 2): 310 - 2 = 308 > 300
        ]
        mock_time.monotonic.side_effect = monotonic_values
        mock_time.sleep = MagicMock()

        with pytest.raises(BackfillTimeoutError, match="stalled"):
            api_client.wait_for_backfill(
                "bf-123",
                timeout_seconds=3600,
                poll_interval_seconds=0.01,
                stall_timeout_seconds=300,
            )

    def test_retries_on_transient_errors(self, api_client):
        """wait_for_backfill retries on transient poll errors up to threshold."""
        completed_job = {
            "job_id": "bf-123",
            "status": "completed",
            "records_published": 500,
        }
        api_client.get_backfill_status = MagicMock(
            side_effect=[
                httpx.ConnectError("Connection refused"),
                httpx.ConnectError("Connection refused"),
                completed_job,
            ]
        )

        with patch("src.data.api_client.time") as mock_time:
            mock_time.monotonic.return_value = 1.0
            mock_time.sleep = MagicMock()

            result = api_client.wait_for_backfill(
                "bf-123",
                timeout_seconds=60,
                poll_interval_seconds=0.01,
                max_consecutive_errors=5,
            )
        assert result["status"] == "completed"

    def test_aborts_after_max_consecutive_errors(self, api_client):
        """wait_for_backfill aborts after max consecutive poll errors."""
        api_client.get_backfill_status = MagicMock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        with patch("src.data.api_client.time") as mock_time:
            mock_time.monotonic.return_value = 1.0
            mock_time.sleep = MagicMock()

            with pytest.raises(BackfillTimeoutError, match="consecutive poll failures"):
                api_client.wait_for_backfill(
                    "bf-123",
                    timeout_seconds=60,
                    poll_interval_seconds=0.01,
                    max_consecutive_errors=3,
                )


class TestCancelBackfill:
    """Tests for CentralApiClient.cancel_backfill."""

    def test_sends_delete_request(self, api_client):
        """cancel_backfill sends DELETE to correct endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"job_id": "bf-123", "status": "cancelled"}}

        api_client.client = MagicMock()
        api_client.client.request.return_value = mock_response

        result = api_client.cancel_backfill("bf-123")
        assert result["status"] == "cancelled"

        call_args = api_client.client.request.call_args
        assert call_args[0] == ("DELETE", "/api/v1/backfill/bf-123")

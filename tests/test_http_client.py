import httpx
import pytest
from structlog.testing import capture_logs

from src.core.http_client import raise_for_status


def test_raise_for_status_logs_response_text_on_http_error() -> None:
    request = httpx.Request("GET", "https://example.com/api")
    response = httpx.Response(500, request=request, text="server blew up")

    with capture_logs() as logs:
        with pytest.raises(httpx.HTTPStatusError):
            raise_for_status(response)

    assert any(entry.get("event") == "http_error" and entry.get("response_text") == "server blew up" for entry in logs)


def test_raise_for_status_handles_bad_encoding() -> None:
    request = httpx.Request("GET", "https://example.com/api")
    response = httpx.Response(500, request=request, content=b"\xff\xfe\xfd")
    response.encoding = "bad-encoding"

    with capture_logs() as logs:
        with pytest.raises(httpx.HTTPStatusError):
            raise_for_status(response)

    assert any(entry.get("event") == "http_error" for entry in logs)

import httpx
import pytest

from src.core.http_client import raise_for_status


@pytest.mark.unit
def test_raise_for_status_handles_streaming_response_text() -> None:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(400, request=request, stream=httpx.ByteStream(b"fail"))

    with pytest.raises(httpx.HTTPStatusError):
        raise_for_status(response)

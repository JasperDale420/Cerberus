"""Unit tests for startup stream-mode selection in main entrypoint."""

from src.main import _should_initialize_alpaca_client, _should_start_alpaca_stream


def test_gateway_noop_skips_alpaca_stream() -> None:
    """Gateway data mode with noop execution should not start Alpaca streaming."""
    assert _should_start_alpaca_stream(order_executor="noop", data_backend="gateway") is False


def test_gateway_alpaca_keeps_alpaca_stream() -> None:
    """Gateway mode with Alpaca execution still needs Alpaca stream connectivity."""
    assert _should_start_alpaca_stream(order_executor="alpaca", data_backend="gateway") is True


def test_legacy_noop_keeps_alpaca_stream() -> None:
    """Legacy mode uses Alpaca market data stream even when execution is noop."""
    assert _should_start_alpaca_stream(order_executor="noop", data_backend="legacy") is True


def test_gateway_noop_no_failover_skips_alpaca_client_init() -> None:
    """Gateway+noop with failover off should not initialize Alpaca client."""
    assert (
        _should_initialize_alpaca_client(
            order_executor="noop",
            data_backend="gateway",
            failover_to_legacy=False,
        )
        is False
    )


def test_gateway_noop_with_failover_still_initializes_alpaca_client() -> None:
    """Gateway+noop with legacy failover on still needs Alpaca client for fallback."""
    assert (
        _should_initialize_alpaca_client(
            order_executor="noop",
            data_backend="gateway",
            failover_to_legacy=True,
        )
        is True
    )

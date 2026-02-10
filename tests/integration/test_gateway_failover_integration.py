"""Integration tests for Data-Gateway failover and dual-read parity."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.logger import StructuredLogger
from src.core.settings import Settings
from src.data.alpaca import AlpacaClient
from src.data.api_client import CentralApiClient
from src.data.fetcher import DataFetcher
from src.data.unusual_whales import UnusualWhalesClient


def build_settings(**overrides: object) -> Settings:
    """Create runtime settings using environment aliases used in production."""
    defaults: dict[str, object] = {
        "CERBERUS_DATA_BACKEND": "legacy",
        "CERBERUS_FAILOVER_TO_LEGACY": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def mock_alpaca_client() -> MagicMock:
    """Mock Alpaca client for legacy mode."""
    client = MagicMock(spec=AlpacaClient)
    client.get_historical_bars.return_value = {
        "bars": [
            {"t": "2025-01-01T09:30:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000},
            {"t": "2025-01-01T09:31:00Z", "o": 100.5, "h": 101.5, "l": 100.0, "c": 101.0, "v": 1100},
        ]
    }
    client.get_historical_trades.return_value = [
        {"t": "2025-01-01T09:30:00Z", "p": 100.0, "s": 100, "c": ["@"], "x": "V", "z": "C"}
    ]
    return client


@pytest.fixture
def mock_uw_client() -> MagicMock:
    """Mock Unusual Whales client."""
    client = MagicMock(spec=UnusualWhalesClient)
    client.get_greek_exposure = AsyncMock(return_value=[{"strike": 500, "gamma": 1000}])
    client.get_option_flow = AsyncMock(return_value=[{"premium": 50000}])
    return client


@pytest.fixture
def mock_central_api_client() -> MagicMock:
    """Mock CentralApiClient for gateway mode."""
    client = MagicMock(spec=CentralApiClient)
    client.get_alpaca_bars.return_value = {
        "bars": [
            {"t": "2025-01-01T09:30:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000},
            {"t": "2025-01-01T09:31:00Z", "o": 100.5, "h": 101.5, "l": 100.0, "c": 101.0, "v": 1100},
        ]
    }
    client.get_alpaca_trades.return_value = [
        {"t": "2025-01-01T09:30:00Z", "p": 100.0, "s": 100, "c": ["@"], "x": "V", "z": "C"}
    ]
    client.get_uw_gex.return_value = [{"strike": 500, "gamma": 1000}]
    client.get_uw_flow.return_value = {"data": [{"premium": 50000}]}
    return client


@pytest.fixture
def logger() -> MagicMock:
    """Mock logger to assert parity and failover diagnostics."""
    return MagicMock(spec=StructuredLogger)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_mode_uses_direct_clients_only(
    mock_alpaca_client: MagicMock,
    mock_uw_client: MagicMock,
    mock_central_api_client: MagicMock,
    logger: MagicMock,
) -> None:
    """Test that legacy mode uses Alpaca/UW clients directly without gateway."""
    with patch("src.data.fetcher.get_settings") as mock_settings:
        mock_settings.return_value = build_settings(CERBERUS_DATA_BACKEND="legacy")

        fetcher = DataFetcher(
            alpaca_client=mock_alpaca_client,
            unusual_whales_client=mock_uw_client,
            logger=logger,
            central_api_client=mock_central_api_client,
        )

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)

        bars, _ = await fetcher.fetch_bars("AAPL", start, end, "1Min")

        mock_alpaca_client.get_historical_bars.assert_called()
        mock_central_api_client.get_alpaca_bars.assert_not_called()
        assert len(bars) == 2
        assert bars[0]["c"] == 100.5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gateway_mode_uses_central_api_client(
    mock_alpaca_client: MagicMock,
    mock_uw_client: MagicMock,
    mock_central_api_client: MagicMock,
    logger: MagicMock,
) -> None:
    """Test that gateway mode routes through CentralApiClient."""
    with patch("src.data.fetcher.get_settings") as mock_settings:
        mock_settings.return_value = build_settings(
            CERBERUS_DATA_BACKEND="gateway",
            CERBERUS_FAILOVER_TO_LEGACY=False,
            CERBERUS_GATEWAY_URL="http://gateway.test",
            CERBERUS_GATEWAY_KEY="test_key",
        )

        fetcher = DataFetcher(
            alpaca_client=mock_alpaca_client,
            unusual_whales_client=mock_uw_client,
            logger=logger,
            central_api_client=mock_central_api_client,
        )

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)

        bars, _ = await fetcher.fetch_bars("AAPL", start, end, "1Min")

        mock_central_api_client.get_alpaca_bars.assert_called_once()
        mock_alpaca_client.get_historical_bars.assert_not_called()
        assert len(bars) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gateway_mode_with_failover_falls_back_on_error(
    mock_alpaca_client: MagicMock,
    mock_uw_client: MagicMock,
    mock_central_api_client: MagicMock,
    logger: MagicMock,
) -> None:
    """Test failover to legacy when gateway fails and failover is enabled."""
    with patch("src.data.fetcher.get_settings") as mock_settings:
        mock_settings.return_value = build_settings(
            CERBERUS_DATA_BACKEND="gateway",
            CERBERUS_FAILOVER_TO_LEGACY=True,
            CERBERUS_GATEWAY_URL="http://gateway.test",
            CERBERUS_GATEWAY_KEY="test_key",
        )

        mock_central_api_client.get_alpaca_bars.side_effect = Exception("Gateway unavailable")

        fetcher = DataFetcher(
            alpaca_client=mock_alpaca_client,
            unusual_whales_client=mock_uw_client,
            logger=logger,
            central_api_client=mock_central_api_client,
        )

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)

        bars, _ = await fetcher.fetch_bars("AAPL", start, end, "1Min")

        mock_central_api_client.get_alpaca_bars.assert_called_once()
        mock_alpaca_client.get_historical_bars.assert_called_once()
        assert len(bars) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gateway_mode_without_failover_raises_on_error(
    mock_alpaca_client: MagicMock,
    mock_uw_client: MagicMock,
    mock_central_api_client: MagicMock,
    logger: MagicMock,
) -> None:
    """Test that gateway mode without failover returns no data and increments failure metric."""
    with patch("src.data.fetcher.get_settings") as mock_settings:
        mock_settings.return_value = build_settings(
            CERBERUS_DATA_BACKEND="gateway",
            CERBERUS_FAILOVER_TO_LEGACY=False,
            CERBERUS_GATEWAY_URL="http://gateway.test",
            CERBERUS_GATEWAY_KEY="test_key",
        )

        mock_central_api_client.get_alpaca_bars.side_effect = Exception("Gateway unavailable")

        fetcher = DataFetcher(
            alpaca_client=mock_alpaca_client,
            unusual_whales_client=mock_uw_client,
            logger=logger,
            central_api_client=mock_central_api_client,
        )

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)

        bars, metrics = await fetcher.fetch_bars("AAPL", start, end, "1Min")

        mock_alpaca_client.get_historical_bars.assert_not_called()
        assert metrics["alpaca_fetch_fail"] == 1
        assert len(bars) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dual_mode_calls_both_sources_and_logs_parity(
    mock_alpaca_client: MagicMock,
    mock_uw_client: MagicMock,
    mock_central_api_client: MagicMock,
    logger: MagicMock,
) -> None:
    """Test dual mode calls both gateway and legacy for comparison."""
    with patch("src.data.fetcher.get_settings") as mock_settings:
        mock_settings.return_value = build_settings(
            CERBERUS_DATA_BACKEND="dual",
            CERBERUS_FAILOVER_TO_LEGACY=True,
            CERBERUS_GATEWAY_URL="http://gateway.test",
            CERBERUS_GATEWAY_KEY="test_key",
        )

        fetcher = DataFetcher(
            alpaca_client=mock_alpaca_client,
            unusual_whales_client=mock_uw_client,
            logger=logger,
            central_api_client=mock_central_api_client,
        )

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)

        bars, _ = await fetcher.fetch_bars("AAPL", start, end, "1Min")

        mock_central_api_client.get_alpaca_bars.assert_called_once()
        mock_alpaca_client.get_historical_bars.assert_called()
        logger.info.assert_any_call(
            "Dual read bars parity confirmed",
            symbol="AAPL",
            timeframe="1Min",
            count=2,
        )
        assert len(bars) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dual_mode_detects_bar_count_mismatch(
    mock_alpaca_client: MagicMock,
    mock_uw_client: MagicMock,
    mock_central_api_client: MagicMock,
    logger: MagicMock,
) -> None:
    """Test dual mode emits warning when bar counts differ."""
    with patch("src.data.fetcher.get_settings") as mock_settings:
        mock_settings.return_value = build_settings(
            CERBERUS_DATA_BACKEND="dual",
            CERBERUS_FAILOVER_TO_LEGACY=True,
            CERBERUS_GATEWAY_URL="http://gateway.test",
            CERBERUS_GATEWAY_KEY="test_key",
        )

        mock_alpaca_client.get_historical_bars.return_value = {
            "bars": [
                {"t": "2025-01-01T09:30:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000},
            ]
        }

        fetcher = DataFetcher(
            alpaca_client=mock_alpaca_client,
            unusual_whales_client=mock_uw_client,
            logger=logger,
            central_api_client=mock_central_api_client,
        )

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)

        await fetcher.fetch_bars("AAPL", start, end, "1Min")

        logger.warning.assert_any_call(
            "Dual read bars count mismatch",
            symbol="AAPL",
            timeframe="1Min",
            legacy_count=1,
            gateway_count=2,
            delta=1,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trades_failover_works_correctly(
    mock_alpaca_client: MagicMock,
    mock_uw_client: MagicMock,
    mock_central_api_client: MagicMock,
    logger: MagicMock,
) -> None:
    """Test trades fetching respects failover settings."""
    with patch("src.data.fetcher.get_settings") as mock_settings:
        mock_settings.return_value = build_settings(
            CERBERUS_DATA_BACKEND="gateway",
            CERBERUS_FAILOVER_TO_LEGACY=True,
            CERBERUS_GATEWAY_URL="http://gateway.test",
            CERBERUS_GATEWAY_KEY="test_key",
        )

        mock_central_api_client.get_alpaca_trades.side_effect = Exception("Gateway trades unavailable")

        fetcher = DataFetcher(
            alpaca_client=mock_alpaca_client,
            unusual_whales_client=mock_uw_client,
            logger=logger,
            central_api_client=mock_central_api_client,
        )

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)

        trades, _ = await fetcher.fetch_trades("AAPL", start, end)

        mock_central_api_client.get_alpaca_trades.assert_called_once()
        mock_alpaca_client.get_historical_trades.assert_called_once()
        assert len(trades) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gex_gateway_routing(
    mock_alpaca_client: MagicMock,
    mock_uw_client: MagicMock,
    mock_central_api_client: MagicMock,
    logger: MagicMock,
) -> None:
    """Test GEX fetching routes through gateway correctly."""
    with patch("src.data.fetcher.get_settings") as mock_settings:
        mock_settings.return_value = build_settings(
            CERBERUS_DATA_BACKEND="gateway",
            CERBERUS_FAILOVER_TO_LEGACY=False,
            CERBERUS_GATEWAY_URL="http://gateway.test",
            CERBERUS_GATEWAY_KEY="test_key",
        )

        fetcher = DataFetcher(
            alpaca_client=mock_alpaca_client,
            unusual_whales_client=mock_uw_client,
            logger=logger,
            central_api_client=mock_central_api_client,
        )

        gex = await fetcher.fetch_gex("SPY")

        mock_central_api_client.get_uw_gex.assert_called_once_with("SPY")
        mock_uw_client.get_greek_exposure.assert_not_called()
        assert len(gex) == 1
        assert gex[0]["strike"] == 500


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dual_mode_gex_parity_logging(
    mock_alpaca_client: MagicMock,
    mock_uw_client: MagicMock,
    mock_central_api_client: MagicMock,
    logger: MagicMock,
) -> None:
    """Test dual mode logs GEX parity confirmation."""
    with patch("src.data.fetcher.get_settings") as mock_settings:
        mock_settings.return_value = build_settings(
            CERBERUS_DATA_BACKEND="dual",
            CERBERUS_FAILOVER_TO_LEGACY=True,
            CERBERUS_GATEWAY_URL="http://gateway.test",
            CERBERUS_GATEWAY_KEY="test_key",
        )

        fetcher = DataFetcher(
            alpaca_client=mock_alpaca_client,
            unusual_whales_client=mock_uw_client,
            logger=logger,
            central_api_client=mock_central_api_client,
        )

        await fetcher.fetch_gex("SPY")

        logger.info.assert_any_call(
            "Dual read GEX parity confirmed",
            symbol="SPY",
            count=1,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dual_mode_flow_parity_logging(
    mock_alpaca_client: MagicMock,
    mock_uw_client: MagicMock,
    mock_central_api_client: MagicMock,
    logger: MagicMock,
) -> None:
    """Test dual mode logs flow parity confirmation."""
    with patch("src.data.fetcher.get_settings") as mock_settings:
        mock_settings.return_value = build_settings(
            CERBERUS_DATA_BACKEND="dual",
            CERBERUS_FAILOVER_TO_LEGACY=True,
            CERBERUS_GATEWAY_URL="http://gateway.test",
            CERBERUS_GATEWAY_KEY="test_key",
        )

        fetcher = DataFetcher(
            alpaca_client=mock_alpaca_client,
            unusual_whales_client=mock_uw_client,
            logger=logger,
            central_api_client=mock_central_api_client,
        )

        await fetcher.fetch_flow("SPY", "2025-01-01")

        logger.info.assert_any_call(
            "Dual read flow parity confirmed",
            symbol="SPY",
            date="2025-01-01",
            count=1,
        )

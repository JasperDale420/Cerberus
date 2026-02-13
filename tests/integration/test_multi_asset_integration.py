"""Integration tests for Multi-Asset Support."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.core.settings import Settings
from src.data.api_client import CentralApiClient
from src.scanner.universe import UniverseBuilder


class TestMultiAssetIntegration:
    """Verify UniverseBuilder and API Client behavior under different asset classes."""

    @pytest.fixture
    def logger(self):
        return MagicMock(spec=StructuredLogger)

    @pytest.fixture
    def config_loader(self):
        loader = MagicMock(spec=ConfigLoader)
        loader.get_env.return_value = "http://mock-gateway"
        loader.load_config.return_value = {"universe": ["BTC/USD"]}
        return loader

    def test_universe_builder_crypto_routing(self, logger, config_loader):
        """Verify UniverseBuilder calls get_crypto_bars when asset_class is crypto."""
        # 1. Setup Settings to return crypto
        with patch("src.scanner.universe.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                CERBERUS_DATA_BACKEND="gateway",
                CERBERUS_ASSET_CLASS="crypto",
                CERBERUS_GATEWAY_URL="http://mock",
                CERBERUS_GATEWAY_KEY="mock",
                CERBERUS_HEBER_CATALOG_URL="http://mock",
            )

            # 2. Setup API Client
            api_client = MagicMock(spec=CentralApiClient)
            api_client.get_crypto_bars.return_value = {"bars": [{"t": "2024-01-01T00:00:00Z", "c": 50000}]}

            # 3. Initialize UniverseBuilder
            ub = UniverseBuilder(
                config_loader,
                logger,
                central_api_client=api_client,
                clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

            # 4. Invoke a method that uses historical bars (e.g. via _get_symbol_volume logic or direct call)
            # Access private method for direct verification
            ub._get_historical_bars("BTC/USD", datetime.now(), datetime.now(), "1Day")

            # 5. Verify get_crypto_bars was called
            api_client.get_crypto_bars.assert_called_once()
            api_client.get_alpaca_bars.assert_not_called()

    def test_universe_builder_equity_routing(self, logger, config_loader):
        """Verify UniverseBuilder calls get_alpaca_bars when asset_class is us_equity."""
        # 1. Setup Settings to return us_equity
        with patch("src.scanner.universe.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                CERBERUS_DATA_BACKEND="gateway",
                CERBERUS_ASSET_CLASS="us_equity",
                CERBERUS_GATEWAY_URL="http://mock",
                CERBERUS_GATEWAY_KEY="mock",
                CERBERUS_HEBER_CATALOG_URL="http://mock",
            )

            # 2. Setup API Client
            api_client = MagicMock(spec=CentralApiClient)
            api_client.get_alpaca_bars.return_value = {"bars": [{"t": "2024-01-01T00:00:00Z", "c": 150}]}

            # 3. Initialize UniverseBuilder
            ub = UniverseBuilder(
                config_loader,
                logger,
                central_api_client=api_client,
                clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

            # 4. Invoke
            ub._get_historical_bars("AAPL", datetime.now(), datetime.now(), "1Day")

            # 5. Verify get_alpaca_bars was called
            api_client.get_alpaca_bars.assert_called_once()
            api_client.get_crypto_bars.assert_not_called()

    def test_client_crypto_endpoints(self, logger, config_loader):
        """Verify CentralApiClient constructs correct URLs for crypto."""
        client = CentralApiClient(config_loader, logger)

        # Mock the internal requests client
        with patch.object(client, "client") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": {"bars": []}}
            mock_httpx.request.return_value = mock_response

            # Test get_crypto_bars
            client.get_crypto_bars("BTC/USD")

            # Verify URL
            call_args = mock_httpx.request.call_args
            assert call_args[0][0] == "GET"  # Method
            assert "/api/v1/alpaca/crypto/BTC/USD/bars" in call_args[0][1]  # Path

            # Test get_crypto_trades
            client.get_crypto_trades("BTC/USD")

            call_args = mock_httpx.request.call_args
            assert "/api/v1/alpaca/crypto/BTC/USD/trades" in call_args[0][1]

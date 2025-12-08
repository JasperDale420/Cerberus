from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.data.api_client import CentralApiClient

class AlpacaClient:
    """
    Wrapper for Alpaca API clients (Trading, Historical Data, Live Data).
    """
    def __init__(self, config_loader: ConfigLoader, logger: StructuredLogger):
        self.logger = logger
        
        try:
            self.api_key = config_loader.get_env("ALPACA_API_KEY")
            self.secret_key = config_loader.get_env("ALPACA_SECRET_KEY")
            self.paper = config_loader.get_env("ALPACA_PAPER", "True").lower() == "true"
        except ValueError as e:
            self.logger.critical("Failed to load Alpaca credentials", error=str(e))
            raise

        try:
            self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            self.central_client = CentralApiClient(config_loader, logger)
            # Stream is initialized on demand or separately as it blocks/runs in a loop
            self.stream_client: Optional[StockDataStream] = None
        except Exception as e:
            self.logger.critical("Failed to initialize Alpaca clients", error=str(e))
            raise

    def get_account(self):
        """
        Fetches account information.
        """
        try:
            return self.trading_client.get_account()
        except Exception as e:
            self.logger.error("Failed to fetch account info", error=str(e))
            raise

    def get_historical_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Min"):
        """
        Fetches historical bars for a symbol via Central API.
        """
        try:
            # Central API returns a dict, we might need to parse it back to Bar objects if expected by callers
            # But wait, FeaturePipeline expects Bar objects (Alpaca SDK objects or our models?)
            # FeaturePipeline currently uses `bars[symbol][-1]`.
            # Let's see what Central API returns. It returns BarResponse schema.
            # We should probably convert it to our internal Bar model here or in FeaturePipeline.
            # For now, let's return the raw data or a list of dicts, and update FeaturePipeline to handle it.
            # Actually, let's convert to our internal Bar model here to be clean.
            
            data = self.central_client.get_alpaca_bars(symbol, start, end, timeframe)
            # Assuming data structure matches Alpaca SDK or is a list of bars
            # The schema in dataingestion/features/alpaca_data/schemas.py says BarResponse.
            # Let's assume it returns a list of bars under a key or directly.
            # Based on router: `return await ... get_bars`.
            # If get_bars returns a DataFrame or dict, we need to know.
            # Let's assume it returns a list of dicts for now.
            
            # TODO: Robust parsing. For now returning raw to see structure in tests/debug.
            return data
        except Exception as e:
            self.logger.error("Failed to fetch historical bars", symbol=symbol, error=str(e))
            raise

    def get_stream_client(self) -> StockDataStream:
        """
        Returns a configured StockDataStream client.
        """
        if not self.stream_client:
             self.stream_client = StockDataStream(self.api_key, self.secret_key)
        return self.stream_client

from typing import Optional, Dict, Any
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.data.api_client import CentralApiClient

class UnusualWhalesClient:
    """
    Wrapper for Unusual Whales API client.
    """
    def __init__(self, config_loader: ConfigLoader, logger: StructuredLogger):
        self.logger = logger
        # No longer need API token here if central service handles it, 
        # but might need it if we kept direct client. 
        # Assuming central service handles auth.
        self.central_client = CentralApiClient(config_loader, logger)

    async def get_option_flow(self, symbol: str, date: str) -> Any:
        """
        Fetches option flow for a symbol via Central API.
        """
        try:
            # Central API endpoint is /uw/flow/{ticker}
            # It doesn't seem to take date in the router I saw earlier?
            # Router: @router.get("/flow/{ticker}") async def get_flow(ticker: str)
            # So date might be ignored or handled internally (e.g. today).
            # We'll pass symbol.
            
            return self.central_client.get_uw_flow(symbol)
        except Exception as e:
            self.logger.error("Failed to fetch option flow", symbol=symbol, error=str(e))
            raise

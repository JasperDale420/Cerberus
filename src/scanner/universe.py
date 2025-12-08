from typing import List
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger

class UniverseBuilder:
    """
    Constructs the universe of symbols to scan.
    """
    def __init__(self, config_loader: ConfigLoader, logger: StructuredLogger):
        self.config = config_loader.load_config()
        self.logger = logger
        
    def get_universe(self) -> List[str]:
        """
        Returns a list of symbols.
        Currently supports hardcoded list from config or defaults.
        """
        # 1. Try to get from config
        universe = self.config.get("universe", [])
        
        if not universe:
            # 2. Fallback to a default liquid list
            self.logger.info("No universe found in config, using default liquid symbols.")
            universe = [
                "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL"
            ]
            
        return universe

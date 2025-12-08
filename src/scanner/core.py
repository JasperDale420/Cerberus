from typing import List, Dict, Any, NamedTuple
from src.core.logger import StructuredLogger
from src.scanner.universe import UniverseBuilder
from src.data.pipeline import FeaturePipeline
from src.scanner.profiles import ScannerProfile, VWAPReversionProfile

class ScannedSymbol(NamedTuple):
    symbol: str
    score: float
    matching_strategies: List[str]

class Scanner:
    """
    Orchestrates the scanning process.
    """
    def __init__(self, 
                 universe_builder: UniverseBuilder, 
                 feature_pipeline: FeaturePipeline,
                 logger: StructuredLogger):
        self.universe_builder = universe_builder
        self.feature_pipeline = feature_pipeline
        self.logger = logger
        
        # Initialize profiles
        # In a real app, these might be loaded dynamically or from config
        self.profiles: Dict[str, ScannerProfile] = {
            "vwap_reversion": VWAPReversionProfile()
        }

    async def scan(self) -> List[ScannedSymbol]:
        """
        Runs the scan:
        1. Get Universe
        2. Fetch Features
        3. Apply Profiles
        4. Return Results
        """
        symbols = self.universe_builder.get_universe()
        self.logger.info("Starting scan", universe_size=len(symbols))
        
        # Fetch features (batch or async loop inside pipeline)
        features_map = await self.feature_pipeline.compute_features(symbols)
        
        results = []
        for symbol, features in features_map.items():
            matching_strategies = []
            
            for strat_name, profile in self.profiles.items():
                if profile.filter(features):
                    matching_strategies.append(strat_name)
            
            if matching_strategies:
                # Score could be based on features (e.g. volume, volatility)
                # For now, simple score = 1.0 if matches any
                score = 1.0 
                results.append(ScannedSymbol(symbol, score, matching_strategies))
                
        self.logger.info("Scan complete", matches=len(results))
        return results

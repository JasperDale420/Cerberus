from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.data.unusual_whales import UnusualWhalesClient
from src.data.models import SymbolFeatures

class FeaturePipeline:
    """
    Fetches data and computes features for symbols.
    """
    def __init__(self, 
                 alpaca_client: AlpacaClient, 
                 unusual_whales_client: UnusualWhalesClient,
                 logger: StructuredLogger):
        self.alpaca_client = alpaca_client
        self.unusual_whales_client = unusual_whales_client
        self.logger = logger

    async def compute_features(self, symbols: List[str]) -> Dict[str, SymbolFeatures]:
        """
        Computes features for a list of symbols.
        """
        features = {}
        
        # In a real implementation, we would batch requests or use async gather
        # For this slice, we'll iterate (or use a simple gather if clients support async)
        
        # Alpaca historical client is synchronous in the wrapper I wrote, 
        # but we can wrap it or just call it.
        # UW client is async.
        
        for symbol in symbols:
            try:
                # 1. Fetch Price Data (Snapshot)
                # Using get_stock_snapshot or similar if available, or just latest bar
                # For now, let's assume we want the latest minute bar
                end = datetime.utcnow()
                start = end - timedelta(minutes=5)
                
                # This is a synchronous call in my wrapper
                bars_data = self.alpaca_client.get_historical_bars(symbol, start, end)
                
                if not bars_data:
                    self.logger.warning("No bars found for symbol", symbol=symbol)
                    continue
                    
                # Assuming bars_data is a list of dicts or BarResponse
                # If it's a list:
                if isinstance(bars_data, list) and len(bars_data) > 0:
                    latest_bar_data = bars_data[-1]
                    # Parse dict to simple object or use dict
                    # Alpaca dict keys: t, o, h, l, c, v
                    price = latest_bar_data.get("c") or latest_bar_data.get("close")
                    volume = latest_bar_data.get("v") or latest_bar_data.get("volume")
                    timestamp = latest_bar_data.get("t") or latest_bar_data.get("timestamp")
                else:
                    # Handle dict response if wrapped
                    # For now, let's assume list of dicts as per most JSON APIs for bars
                    self.logger.warning("Unexpected bar data format", symbol=symbol, data=bars_data)
                    continue
                
                # 2. Fetch Options Flow
                # Async call
                flow_data = await self.unusual_whales_client.get_option_flow(symbol, end.strftime("%Y-%m-%d"))
                
                # 3. Compute Features
                # Placeholder logic
                feat = SymbolFeatures(
                    symbol=symbol,
                    timestamp=timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")),
                    price=float(price),
                    volume=float(volume),
                    flow_sentiment=0.0, # TODO: Parse flow_data
                    volatility=0.0, # TODO: Compute from history
                    extra={"flow_raw": flow_data}
                )
                
                features[symbol] = feat
                
            except Exception as e:
                self.logger.error("Failed to compute features", symbol=symbol, error=str(e))
                
        return features

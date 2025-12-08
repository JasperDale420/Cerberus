import asyncio
import argparse
from datetime import datetime, timedelta
from typing import List
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.data.api_client import CentralApiClient
from src.engine.execution import ExecutionEngine
from src.backtest.mock_executor import MockOrderExecutor
from src.data.models import Bar

class BacktestRunner:
    """
    Runs a backtest by feeding historical data to the ExecutionEngine.
    """
    def __init__(self, config_path: str, start_date: str, end_date: str):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config(config_path)
        self.logger = StructuredLogger("Backtester")
        
        self.start_date = datetime.fromisoformat(start_date)
        self.end_date = datetime.fromisoformat(end_date)
        
        self.central_client = CentralApiClient(self.config_loader, self.logger)
        
        # Mock Executor
        self.mock_executor = MockOrderExecutor(self.logger)
        
        # Engine with Mock Executor
        self.engine = ExecutionEngine(self.config_loader, self.logger, alpaca_client=None)
        self.engine.order_executor = self.mock_executor # Inject mock
        
        # We need to inject a mock scanner or just manually set the universe
        # For simplicity, let's just test one symbol
        self.universe = ["AAPL"] 

    async def run(self):
        self.logger.info("Starting Backtest", start=self.start_date, end=self.end_date)
        
        # Fetch Data
        # In a real backtester, we'd fetch chunk by chunk or stream.
        # Here we fetch all at once for simplicity.
        
        for symbol in self.universe:
            self.logger.info("Fetching data", symbol=symbol)
            bars_data = self.central_client.get_alpaca_bars(
                symbol, self.start_date, self.end_date, timeframe="1Day" # Using daily for speed in this slice
            )
            
            # Parse bars
            bars = []
            if isinstance(bars_data, list):
                for b in bars_data:
                    # Handle different formats (raw vs parsed)
                    # Assuming raw from CentralApiClient as per previous files
                    t = b.get("t") or b.get("timestamp")
                    o = b.get("o") or b.get("open")
                    h = b.get("h") or b.get("high")
                    l = b.get("l") or b.get("low")
                    c = b.get("c") or b.get("close")
                    v = b.get("v") or b.get("volume")
                    
                    if t:
                        bars.append(Bar(
                            symbol=symbol,
                            timestamp=datetime.fromisoformat(str(t).replace("Z", "+00:00")) if isinstance(t, str) else t,
                            open=float(o), high=float(h), low=float(l), close=float(c), volume=float(v)
                        ))
            
            self.logger.info("Loaded bars", count=len(bars))
            
            # Run Loop
            for bar in bars:
                # 1. Fill pending orders from previous bar (at Open)
                self.mock_executor.fill_orders(symbol, bar.open, bar.timestamp)
                
                # 2. Process Bar (Strategy Logic)
                await self.engine.process_bar(bar)
                
                # 3. (Optional) Fill orders at Close if strategy trades on close
                # self.mock_executor.fill_orders(symbol, bar.close, bar.timestamp)
                
        # Report
        self.logger.info("Backtest Complete")
        self.logger.info("Final Cash", cash=self.mock_executor.cash)
        self.logger.info("Positions", positions=self.mock_executor.positions)
        self.logger.info("Total Fills", count=len(self.mock_executor.fills))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    
    runner = BacktestRunner(args.config, args.start, args.end)
    asyncio.run(runner.run())

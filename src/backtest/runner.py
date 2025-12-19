import argparse
import asyncio
from datetime import datetime
from typing import Any, List

from src.backtest.mock_executor import MockOrderExecutor
from src.core.config import ConfigLoader
from src.core.domain import Bar
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.engine.execution import ExecutionEngine


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

        self.alpaca_client = AlpacaClient(self.config_loader, self.logger)

        # Mock Executor
        self.mock_executor = MockOrderExecutor(self.logger)

        # Engine with Mock Executor
        self.engine = ExecutionEngine(self.config, self.logger, alpaca_client=None)
        self.engine.order_executor = self.mock_executor  # type: ignore # Inject mock

        # We need to inject a mock scanner or just manually set the universe
        # For simplicity, let's just test one symbol
        self.universe = ["AAPL"]

    def _parse_bars(self, bars_data: Any, symbol: str) -> List[Bar]:
        bars: List[Bar] = []
        if isinstance(bars_data, list):
            for b in bars_data:
                # Handle different formats (raw vs parsed)
                t = b.get("t") or b.get("timestamp")
                o = b.get("o") or b.get("open")
                h = b.get("h") or b.get("high")
                low_price = b.get("l") or b.get("low")
                c = b.get("c") or b.get("close")
                v = b.get("v") or b.get("volume")

                if t:
                    bars.append(
                        Bar(
                            symbol=symbol,
                            time=(
                                datetime.fromisoformat(str(t).replace("Z", "+00:00"))
                                if isinstance(t, str)
                                else t
                            ),
                            open=float(o),
                            high=float(h),
                            low=float(low_price),
                            close=float(c),
                            volume=float(v),
                        )
                    )
        return bars

    async def run(self):
        self.logger.info("Starting Backtest", start=self.start_date, end=self.end_date)

        # Fetch Data
        for symbol in self.universe:
            self.logger.info("Fetching data", symbol=symbol)
            import asyncio

            # PRD 1.1: Default to 1Min for Intraday Scalping support
            timeframe = self.config.get("timeframe", "1Min")
            bars_data = await asyncio.to_thread(
                self.alpaca_client.get_historical_bars,
                symbol,
                self.start_date,
                self.end_date,
                timeframe,
            )
            # Expect dict with "bars" or list of dicts.
            if isinstance(bars_data, dict) and "bars" in bars_data:
                bars_data = bars_data["bars"]

            bars = self._parse_bars(bars_data, symbol)
            self.logger.info("Loaded bars", count=len(bars))

            await asyncio.sleep(0)  # Yield control

            # Run Loop
            for bar in bars:
                # 1. Fill pending orders from previous bar (at Open)
                self.mock_executor.fill_orders(symbol, bar.open, bar.time)

                # 2. Process Bar (Strategy Logic)
                self.engine.on_bar(symbol, bar)

                # 3. (Optional) Fill orders at Close if strategy trades on close
                # self.mock_executor.fill_orders(symbol, bar.close, bar.time)

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

import asyncio
import argparse
import os
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.data.api_client import CentralApiClient
from src.data.pipeline import FeaturePipeline
from src.scanner.universe import UniverseBuilder
from src.scanner.core import Scanner
from src.engine.execution import ExecutionEngine
from src.data.alpaca import AlpacaClient

async def main():
    parser = argparse.ArgumentParser(description="Scalper Trading Bot")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper", help="Trading mode")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--run-once", action="store_true", help="Run initial scan and exit (for verification)")
    args = parser.parse_args()

    # 1. Setup
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)
    
    # Override mode in config if needed, or just use args
    # For now, we assume config handles keys for paper/live
    
    logger = StructuredLogger("Scalper")
    logger.info("Starting Scalper", mode=args.mode)

    # 2. Components
    central_client = CentralApiClient(config_loader, logger)
    feature_pipeline = FeaturePipeline(central_client, logger)
    
    universe_builder = UniverseBuilder(config_loader, logger)
    scanner = Scanner(universe_builder, feature_pipeline, logger)
    
    # Alpaca Client for Execution (Direct)
    alpaca_client = AlpacaClient(config_loader, logger)
    
    engine = ExecutionEngine(config_loader, logger, alpaca_client)
    engine.scanner = scanner # Inject scanner
    
    # 3. Initial Scan
    logger.info("Running initial scan...")
    await engine.run_scan()
    
    if args.run_once:
        logger.info("Run-once mode enabled. Exiting after initial scan.")
        return

    # 4. Main Loop (Placeholder for WebSocket or Polling)
    # In a real implementation, we would start the Alpaca Stream here.
    # await alpaca_client.start_stream(engine.on_bar)
    
    logger.info("Entering main loop (Press Ctrl+C to exit)")
    try:
        while True:
            # Placeholder: Re-scan every X minutes?
            # For now, just sleep to keep process alive
            await asyncio.sleep(60)
            # await engine.run_scan() 
    except KeyboardInterrupt:
        logger.info("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())

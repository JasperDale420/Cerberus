import argparse
import asyncio

from src.analysis.db import DatabaseDatabase
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.data.pipeline import FeaturePipeline
from src.data.unusual_whales import UnusualWhalesClient
from src.engine.execution import ExecutionEngine
from src.scanner.core import Scanner
from src.scanner.universe import UniverseBuilder


async def main():
    parser = argparse.ArgumentParser(description="Scalper Trading Bot")
    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper", help="Trading mode"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run initial scan and exit (for verification)",
    )
    args = parser.parse_args()

    # 1. Setup
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)

    # Override mode in config if needed, or just use args
    # For now, we assume config handles keys for paper/live

    logger = StructuredLogger("Scalper")
    logger.info("Starting Scalper", mode=args.mode)

    # 2. Components
    # Alpaca Client
    alpaca_client = AlpacaClient(config_loader, logger)

    # Unusual Whales Client
    uw_client = UnusualWhalesClient(config_loader, logger)

    feature_pipeline = FeaturePipeline(alpaca_client, uw_client, logger)

    universe_builder = UniverseBuilder(config_loader, logger)
    scanner = Scanner(universe_builder, feature_pipeline, logger)

    # Database
    db = DatabaseDatabase(config_loader, logger)
    db.init_db()

    # Meta-System Components
    from src.agent.core import Agent

    # Initialize Analytics
    # Initialize Analytics
    # analytics = AnalyticsEngine(db, logger) # Unused in this scope currently

    # Initialize Agent
    # We might need an LLM Client. For now, use dummy or config-based.
    # LLMClient might need config.
    # LLMClient might need config.
    # llm_client = LLMClient(config.get("llm", {})) # Unused
    agent = Agent(logger, config_loader)

    engine = ExecutionEngine(config, logger, db, alpaca_client)
    engine.scanner = scanner  # Inject scanner

    # Register Strategies
    from src.strategies.orb import ORBStrategy
    from src.strategies.trend_pullback import TrendPullbackStrategy
    from src.strategies.vwap_reversion import VWAPReversionStrategy

    # VWAP Reversion
    vwap_strat_config = config.get("strategies", {}).get("vwap_reversion", {})
    if vwap_strat_config.get("enabled", True):
        engine.register_strategy(VWAPReversionStrategy(vwap_strat_config, logger))

    # ORB
    orb_strat_config = config.get("strategies", {}).get("orb", {})
    if orb_strat_config.get("enabled", True):
        engine.register_strategy(ORBStrategy(orb_strat_config, logger))

    # Trend Pullback
    tp_strat_config = config.get("strategies", {}).get("trend_pullback", {})
    if tp_strat_config.get("enabled", True):
        engine.register_strategy(TrendPullbackStrategy(tp_strat_config, logger))

    # Failed Breakout
    from src.strategies.failed_breakout import FailedBreakoutStrategy

    fb_strat_config = config.get("strategies", {}).get("failed_breakout", {})
    if fb_strat_config.get("enabled", True):
        engine.register_strategy(FailedBreakoutStrategy(fb_strat_config, logger))

    # VWAP Trend Rider
    from src.strategies.vwap_trend_rider import VWAPTrendRiderStrategy

    vtr_strat_config = config.get("strategies", {}).get("vwap_trend_rider", {})
    if vtr_strat_config.get("enabled", True):
        engine.register_strategy(VWAPTrendRiderStrategy(vtr_strat_config, logger))

    # Index Mean Reversion
    from src.strategies.index_mean_reversion import IndexMeanReversionStrategy

    imr_strat_config = config.get("strategies", {}).get("index_mean_reversion", {})
    if imr_strat_config.get("enabled", True):
        engine.register_strategy(IndexMeanReversionStrategy(imr_strat_config, logger))

    # Flow Momentum
    from src.strategies.flow_momentum import FlowMomentumStrategy

    fm_strat_config = config.get("strategies", {}).get("flow_momentum", {})
    if fm_strat_config.get("enabled", True):
        engine.register_strategy(FlowMomentumStrategy(fm_strat_config, logger))

    # Gap Fill
    from src.strategies.gap_fill import GapFillStrategy

    gf_strat_config = config.get("strategies", {}).get("gap_fill", {})
    if gf_strat_config.get("enabled", True):
        engine.register_strategy(GapFillStrategy(gf_strat_config, logger))

    # 3. Startup Meta-Loop
    if args.mode == "live" or args.mode == "paper":
        logger.info("Running Meta-System Startup Check...")
        try:
            # 1. Ensure Stats are up to date (e.g. run for yesterday)
            # For simplicity, we just run aggregation for 'today' (if intra-day) or specific logic.
            # In production, this might be a cron. Here, we run it to ensure tables are populated.
            # analytics.run_daily_aggregation()

            # 2. Run Agent Cycle
            # Adjusted config will be reloaded implicitly if ConfigLoader re-reads?
            # Or we might need to reload config object after agent runs.
            agent.run_cycle()

            # Reload config to pick up changes
            config = config_loader.load_config(args.config)
            engine.config = config  # Update engine config
            engine.max_churn_per_scan = config.get("max_churn_per_scan", 2)

        except Exception as e:
            logger.error("Meta-System Startup Failed", error=str(e))

    # 4. Initial Scan
    logger.info("Running initial scan...")
    await engine.run_scan()

    if args.run_once:
        logger.info("Run-once mode enabled. Exiting after initial scan.")
        return

    # 5. Main Loop
    # Start Alpaca Stream in background (if implemented)
    await alpaca_client.start_stream(engine.on_bar)

    logger.info("Entering main trading loop", mode=args.mode)

    try:
        from datetime import datetime

        import pytz  # type: ignore

        tz = pytz.timezone(config.get("timezone", "US/Eastern"))

        while True:
            now = datetime.now(tz)

            # 1. Market Close Check (Exit at 16:00 ET)
            if now.hour >= 16:
                logger.info("Market closed. Exiting daily session.")
                break

            # 2. Run Scanner (every 5 mins)
            # In production, this should be non-blocking or scheduled properly.
            # For this simple loop, we just wait.
            await engine.run_scan()

            # Wait for next scan interval
            # Scan returns quickly?
            # We sleep 60s to avoid hammering
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Main loop error", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())

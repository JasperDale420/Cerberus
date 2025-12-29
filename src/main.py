import argparse
import asyncio
from collections.abc import Callable
from datetime import date as date_type
from datetime import datetime, timezone
from typing import cast

from src.analysis.analytics import AnalyticsEngine
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
        "--order-executor",
        choices=["alpaca", "noop"],
        default="alpaca",
        help="Order routing backend (alpaca submits broker orders; noop simulates).",
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run initial scan and exit (for verification)",
    )
    parser.add_argument(
        "--run-agent",
        action="store_true",
        help="Run Agent Stage 1 (plus aggregation) on startup",
    )
    parser.add_argument(
        "--eod",
        action="store_true",
        help="Run end-of-day aggregation + Agent Stage 1 then exit",
    )
    parser.add_argument(
        "--eod-date",
        default="",
        help="EOD target date in YYYY-MM-DD (defaults to today UTC)",
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Run as a persistent scheduler process (replaces Chronos)",
    )
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="Run system healthcheck and exit",
    )
    args = parser.parse_args()

    # 0. Healthcheck Mode
    if args.healthcheck:
        from src.core.health import run_healthcheck

        run_healthcheck(verbose=True)
        return
    if args.scheduler:
        from src.scheduler import CerberusScheduler

        # Load minimal config just for scheduler (timezone, logging)
        bootstrap_logger = StructuredLogger("Bootstrap", level="INFO")
        config_loader = ConfigLoader(logger=bootstrap_logger)
        config = config_loader.load_config(args.config)

        # Inject config path specifically for the subprocess
        scheduler_config = config.copy() if isinstance(config, dict) else {}
        scheduler_config["config_path"] = args.config

        scheduler = CerberusScheduler(scheduler_config)
        scheduler.start()
        return

    # 1. Setup
    bootstrap_logger = StructuredLogger("Bootstrap", level="INFO")
    config_loader = ConfigLoader(logger=bootstrap_logger)
    config = config_loader.load_config(args.config)

    # Override mode in config if needed, or just use args
    # For now, we assume config handles keys for paper/live

    logger = StructuredLogger(
        "Scalper",
        level=str(config.get("log_level", "INFO")),
        logging_config=config.get("logging"),
    )
    logger.info("Starting Scalper", mode=args.mode)

    def _fixed_clock(dt: datetime) -> Callable[[], datetime]:
        def _clock() -> datetime:
            return dt

        return _clock

    def _utc_now_clock() -> Callable[[], datetime]:
        def _clock() -> datetime:
            return datetime.now(timezone.utc)

        return _clock

    # Deterministic clock (injectable for replay/tests)
    if isinstance(config, dict) and config.get("start_time_utc"):
        start = datetime.fromisoformat(
            str(config["start_time_utc"]).replace("Z", "+00:00")
        )
        clock = _fixed_clock(start)
    else:
        clock = _utc_now_clock()

    # 2. Components
    # Alpaca Client
    alpaca_client = AlpacaClient(config_loader, logger)

    # Unusual Whales Client
    uw_client = UnusualWhalesClient(config_loader, logger, config=config)

    feature_pipeline = FeaturePipeline(
        alpaca_client, uw_client, logger, config=config, clock=clock
    )

    universe_builder = UniverseBuilder(
        config_loader,
        logger,
        config=config,
        config_path_or_dir=args.config,
        alpaca_client=alpaca_client,
        clock=clock,
    )
    scanner = Scanner(universe_builder, feature_pipeline, logger, config=config)

    # Database
    db = DatabaseDatabase(
        config_loader, logger, config=config, config_path_or_dir=args.config
    )
    db.init_db()

    # Meta-System Components
    from src.agent.core import Agent

    # Initialize Analytics
    analytics = AnalyticsEngine(db, logger)

    # Initialize Agent
    # We might need an LLM Client. For now, use dummy or config-based.
    # LLMClient might need config.
    # LLMClient might need config.
    # llm_client = LLMClient(config.get("llm", {})) # Unused
    agent = Agent(logger, config_loader, config_path_or_dir=args.config)

    engine = ExecutionEngine(config, logger, db, alpaca_client, clock=clock)
    engine.scanner = scanner  # Inject scanner
    if args.order_executor == "noop":
        from src.engine.orders import NoopOrderExecutor

        engine.order_executor = NoopOrderExecutor(logger, db=db, clock=clock)  # type: ignore

    # Register Strategies (config-driven, deterministic; PRD plug-and-play intent)
    from src.strategies.failed_breakout import FailedBreakoutStrategy
    from src.strategies.flow_momentum import FlowMomentumStrategy
    from src.strategies.gap_fill import GapFillStrategy
    from src.strategies.index_mean_reversion import IndexMeanReversionStrategy
    from src.strategies.orb import ORBStrategy
    from src.strategies.trend_pullback import TrendPullbackStrategy
    from src.strategies.vwap_reversion import VWAPReversionStrategy
    from src.strategies.vwap_trend_rider import VWAPTrendRiderStrategy

    strategy_registry = {
        "vwap_reversion": VWAPReversionStrategy,
        "orb": ORBStrategy,
        "trend_pullback": TrendPullbackStrategy,
        "failed_breakout": FailedBreakoutStrategy,
        "vwap_trend_rider": VWAPTrendRiderStrategy,
        "index_mean_reversion": IndexMeanReversionStrategy,
        "flow_momentum": FlowMomentumStrategy,
        "gap_fill": GapFillStrategy,
    }

    strategies_cfg = config.get("strategies", {})
    if not isinstance(strategies_cfg, dict):
        strategies_cfg = {}

    for name in sorted(strategies_cfg.keys()):
        strat_cfg = strategies_cfg.get(name)
        if not isinstance(strat_cfg, dict):
            continue
        if not bool(strat_cfg.get("enabled", True)):
            continue
        cls = strategy_registry.get(str(name))
        if cls is None:
            logger.warning("Unknown strategy in config; skipping", strategy=str(name))
            continue
        engine.register_strategy(cls(strat_cfg, logger))  # type: ignore

    # 3. Startup Meta-Loop
    if args.mode == "live" or args.mode == "paper":
        if args.eod:
            target = (
                datetime.fromisoformat(args.eod_date).date()
                if args.eod_date
                else clock().date()
            )
            logger.info("Running EOD aggregation + Agent Stage 1", date=str(target))
            analytics.run_daily_aggregation(cast(date_type, target))
            agent.run_cycle_with_db(db, as_of=clock())
            return

        if args.run_agent:
            logger.info("Running startup aggregation + Agent Stage 1")
            try:
                analytics.run_daily_aggregation(clock().date())
                agent.run_cycle_with_db(db, as_of=clock())

                # Reload config to pick up changes
                config = config_loader.load_config(args.config)
                engine.update_config(config)
            except Exception as e:
                logger.error("Startup Agent run failed", error=str(e), exc_info=True)

    # 4. Initial Scan
    logger.info("Starting Alpaca stream...")
    stream_task = asyncio.create_task(
        alpaca_client.start_stream(
            engine.on_bar, on_reconnect=engine.reconcile_broker_state
        )
    )
    trade_stream_task = None
    if args.order_executor == "alpaca":
        trade_stream_task = asyncio.create_task(
            alpaca_client.start_trade_stream(
                engine.on_trade_update, on_reconnect=engine.reconcile_broker_state
            )
        )
    reconcile_task = asyncio.create_task(engine.reconcile_loop())

    # Ensure index symbol is subscribed for regime detection.
    alpaca_client.subscribe(config.get("index_symbol", "SPY"))

    logger.info("Running initial scan...")
    try:
        await engine.reconcile_broker_state()
    except Exception as e:
        logger.warning("Initial broker reconciliation failed", error=str(e))
    await engine.run_scan()

    if args.run_once:
        logger.info("Run-once mode enabled. Exiting after initial scan.")
        return

    # 5. Main Loop
    logger.info("Entering main trading loop", mode=args.mode)

    try:
        import pytz  # type: ignore

        tz = pytz.timezone(config.get("timezone", "US/Eastern"))

        eod_ran_for_date = None

        def _now_local() -> datetime | None:
            """
            Prefer engine time derived from market data for determinism/replay.
            Fall back to wall-clock time only if engine time is unavailable.
            """
            t = getattr(engine.market_state, "time", None)
            if isinstance(t, datetime):
                try:
                    # If naive, assume UTC to preserve deterministic ordering.
                    if t.tzinfo is None:
                        t = t.replace(tzinfo=timezone.utc)
                    return t.astimezone(tz)
                except Exception:
                    pass
            return None

        while True:
            now = _now_local()
            if now is None:
                # Wait until index bars establish engine time to keep scheduling deterministic.
                await asyncio.sleep(1)
                continue

            # 1. Market Close Check (Exit at 16:00 ET)
            if now.hour >= 16:
                # PRD: no overnight holds. Flatten positions and cancel orders before exit.
                try:
                    engine.flatten_all(reason="market_close")
                except Exception as e:
                    logger.error("EOD flatten failed", error=str(e), exc_info=True)
                    # With mismatch_mode=halt, do not proceed silently.
                    break
                # PRD 9.1: run aggregation + Agent Stage 1 at end-of-day (configurable).
                if bool(config.get("auto_eod_agent", False)):
                    target_date = now.date()
                    if eod_ran_for_date != target_date:
                        logger.info(
                            "Running automatic EOD aggregation + Agent Stage 1",
                            date=str(target_date),
                        )
                        try:
                            analytics.run_daily_aggregation(
                                cast(date_type, target_date)
                            )
                            agent.run_cycle_with_db(db, as_of=engine.market_state.time)
                            # Reload config to pick up changes
                            config = config_loader.load_config(args.config)
                            engine.update_config(config)
                            eod_ran_for_date = target_date
                        except Exception as e:
                            from src.core.errors import ErrorCode

                            logger.error(
                                "Automatic EOD Agent run failed",
                                error_code=ErrorCode.EOD_AGENT_FAILED.value,
                                error=str(e),
                                exc_info=True,
                            )
                logger.info("Market closed. Exiting daily session.")
                break

            # 2. Run Scanner (every 5 mins)
            # In production, this should be non-blocking or scheduled properly.
            # For this simple loop, we just wait.
            await engine.run_scan()

            # Wait for next scan interval
            scanner_cfg = (
                (config.get("scanner") or {}) if isinstance(config, dict) else {}
            )
            interval_min = int(scanner_cfg.get("interval_minutes", 5))
            await asyncio.sleep(max(1, interval_min * 60))

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        from src.core.errors import ErrorCode

        logger.error(
            "Main loop error",
            error_code=ErrorCode.MAIN_LOOP_ERROR.value,
            error=str(e),
            exc_info=True,
        )
        raise
    finally:
        if not stream_task.done():
            stream_task.cancel()
        if trade_stream_task is not None and not trade_stream_task.done():
            trade_stream_task.cancel()
        if not reconcile_task.done():
            reconcile_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())

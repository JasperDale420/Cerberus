import argparse
import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from datetime import date as date_type
from typing import cast

from src.analysis.analytics import AnalyticsEngine
from src.analysis.db import DatabaseDatabase
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.data.client import UnifiedDataClient
from src.data.pipeline import FeaturePipeline
from src.data.unusual_whales import UnusualWhalesClient
from src.engine.execution import ExecutionEngine
from src.scanner.core import Scanner
from src.scanner.streaming_scanner import StreamingScanner
from src.scanner.universe import UniverseBuilder


def _next_market_open_local(now: datetime) -> datetime:
    """
    Return the next regular US market open (09:30 local time on a weekday).

    The input must be timezone-aware and already expressed in the market timezone.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    today_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now < today_open and now.weekday() < 5:
        return today_open

    candidate = today_open + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _is_regular_market_session_local(now: datetime) -> bool:
    """
    Return True when within regular US session hours on a weekday.

    Session window is 09:30 <= time < 16:00 in the provided local timezone.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if now.weekday() >= 5:
        return False
    if now.hour < 9 or (now.hour == 9 and now.minute < 30):
        return False
    return not now.hour >= 16


def _build_strategy_registry() -> dict[str, type]:
    """Build the canonical strategy registry for runtime initialization."""
    from src.strategies.autoresearch_strategy import AutoresearchStrategy
    from src.strategies.daily_mean_reversion import DailyMeanReversionStrategy
    from src.strategies.daily_momentum import DailyMomentumStrategy
    from src.strategies.daily_research_strategy import DailyResearchStrategy
    from src.strategies.daily_vol_fade import DailyVolFadeStrategy
    from src.strategies.failed_breakout import FailedBreakoutStrategy
    from src.strategies.flow_alpha import FlowAlphaStrategy
    from src.strategies.flow_momentum import FlowMomentumStrategy
    from src.strategies.fusion_v1 import FusionStrategyV1
    from src.strategies.gap_fill import GapFillStrategy
    from src.strategies.index_mean_reversion import IndexMeanReversionStrategy
    from src.strategies.intraday_momentum import IntradayMomentumStrategy
    from src.strategies.mean_reversion_pro import MeanReversionProStrategy
    from src.strategies.momentum_continuation import MomentumContinuationStrategy
    from src.strategies.momentum_fade import MomentumFadeStrategy
    from src.strategies.orb import ORBStrategy
    from src.strategies.orb_v2 import ORBV2Strategy
    from src.strategies.order_flow_imbalance import OrderFlowImbalanceStrategy
    from src.strategies.pair_trading import PairTradingStrategy
    from src.strategies.pair_trading_v2 import PairTradingV2Strategy
    from src.strategies.regime_adaptive import RegimeAdaptiveStrategy
    from src.strategies.regime_adaptive_momentum import RegimeAdaptiveMomentumStrategy
    from src.strategies.regime_trend_up import RegimeTrendUpStrategy
    from src.strategies.rsi_bounce import RsiBounceStrategy
    from src.strategies.trend_pullback import TrendPullbackStrategy
    from src.strategies.trend_rider_pro import TrendRiderProStrategy
    from src.strategies.up_normal_strategy import UpNormalStrategy
    from src.strategies.vix_spike_fade import VixSpikeFadeStrategy
    from src.strategies.vwap_reversion import VWAPReversionStrategy
    from src.strategies.vwap_trend_rider import VWAPTrendRiderStrategy

    return {
        # V2 consolidated strategies (enabled by default)
        "daily_momentum": DailyMomentumStrategy,
        "regime_adaptive": RegimeAdaptiveStrategy,
        "regime_adaptive_momentum": RegimeAdaptiveMomentumStrategy,
        "daily_mean_reversion": DailyMeanReversionStrategy,
        "daily_vol_fade": DailyVolFadeStrategy,
        "mean_reversion_pro": MeanReversionProStrategy,
        "trend_rider_pro": TrendRiderProStrategy,
        "flow_alpha": FlowAlphaStrategy,
        "orb_v2": ORBV2Strategy,
        "pair_trading_v2": PairTradingV2Strategy,
        "rsi_bounce": RsiBounceStrategy,
        "momentum_fade": MomentumFadeStrategy,
        "regime_trend_up": RegimeTrendUpStrategy,
        "autoresearch_strategy": AutoresearchStrategy,
        "daily_research_strategy": DailyResearchStrategy,
        "up_normal_strategy": UpNormalStrategy,
        # Legacy strategies (disabled in config)
        "vwap_reversion": VWAPReversionStrategy,
        "orb": ORBStrategy,
        "vwap_trend_rider": VWAPTrendRiderStrategy,
        "index_mean_reversion": IndexMeanReversionStrategy,
        "flow_momentum": FlowMomentumStrategy,
        "gap_fill": GapFillStrategy,
        "vix_spike_fade": VixSpikeFadeStrategy,
        "momentum_continuation": MomentumContinuationStrategy,
        "fusion_v1": FusionStrategyV1,
        "pair_trading": PairTradingStrategy,
        "trend_pullback": TrendPullbackStrategy,
        "failed_breakout": FailedBreakoutStrategy,
        "order_flow_imbalance": OrderFlowImbalanceStrategy,
        "intraday_momentum": IntradayMomentumStrategy,
    }


async def async_main():
    """
    Application entry point - initializes and runs the Cerberus trading system.

    Orchestrates the complete lifecycle of the trading system:
    1. Loads configuration from config.yaml
    2. Initializes database and analytics
    3. Sets up broker connection and WebSocket streams
    4. Creates ExecutionEngine with strategies and risk management
    5. Starts internal scheduler for daily tasks (if enabled)
    6. Runs main trading loop processing market data
    7. Handles end-of-day position flattening
    8. Executes optional EOD analytics agent

    The main loop continues until:
    - User interrupts (Ctrl+C)
    - Fatal error occurs
    - Market closes (depending on configuration)

    Args:
        None (reads from config.yaml and environment variables)

    Returns:
        None

    Raises:
        Exception: Logs and re-raises any fatal errors from main loop
        KeyboardInterrupt: Gracefully shuts down on user interrupt

    Side Effects:
        - Creates cerberus.db SQLite database if not exists
        - Starts WebSocket connection to data source
        - Creates log files in logs/ directory
        - Optionally runs EOD agent and updates config
        - Flattens all positions at market close (if flat_on_close enabled)

    Environment Variables:
        - CERBERUS_STAGE3_APPROVED: Required for Stage 3 agent proposals
        - Log levels and other config can be set via env vars

    Note:
        - PRD 11.2: Main loop includes observability hooks
        - PRD 6.4: Strategy routing by market regime
        - PRD 9.1: Automated EOD analytics and risk adjustment
        - Uses async/await for concurrent WebSocket and scheduler
    """
    parser = argparse.ArgumentParser(description="Scalper Trading Bot")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper", help="Trading mode")
    parser.add_argument(
        "--order-executor",
        choices=["gateway", "alpaca", "noop"],
        default="gateway",
        help="Order routing backend (gateway routes via Data-Gateway; alpaca submits direct broker orders; noop simulates).",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
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

    # Validate startup settings before proceeding
    try:
        from src.core.settings import (
            get_settings,
            validate_runtime_execution_requirements,
            validate_startup_settings,
        )

        validate_startup_settings()
        runtime_settings = get_settings()
        validate_runtime_execution_requirements(
            order_executor=args.order_executor,
            mode=args.mode,
        )
        bootstrap_logger.info(
            "Startup settings validation passed",
            order_executor=args.order_executor,
            mode=args.mode,
        )
    except ValueError as e:
        bootstrap_logger.error("Startup settings validation failed", error=str(e))
        raise

    config_loader = ConfigLoader(logger=bootstrap_logger)
    config = config_loader.load_config(args.config)

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
            return datetime.now(UTC)

        return _clock

    # Deterministic clock (injectable for replay/tests)
    if isinstance(config, dict) and config.get("start_time_utc"):
        start = datetime.fromisoformat(str(config["start_time_utc"]).replace("Z", "+00:00"))
        clock = _fixed_clock(start)
    else:
        clock = _utc_now_clock()

    # 2. Components
    uw_client = UnusualWhalesClient(config_loader, logger, config=config)

    unified_client = UnifiedDataClient(
        gateway_url=runtime_settings.cerberus_gateway_url,
        gateway_key=runtime_settings.cerberus_gateway_key,
    )

    # Heber reader for flow alerts (populated by Gateway poller — avoids duplicate UW API calls)
    heber_client = None
    if runtime_settings.cerberus_heber_data_root:
        from src.data.heber_read_client import HeberReadClient

        heber_client = HeberReadClient(
            data_root=runtime_settings.cerberus_heber_data_root,
            logger=logger,
        )
        logger.info("Heber reader initialized for flow alerts", data_root=runtime_settings.cerberus_heber_data_root)

    feature_pipeline = FeaturePipeline(
        unified_client,
        uw_client,
        logger,
        config=config,
        clock=clock,
        heber_client=heber_client,
    )

    universe_builder = UniverseBuilder(
        unified_client,
        logger,
        config=config,
        config_path_or_dir=args.config,
        clock=clock,
    )
    scanner = Scanner(universe_builder, feature_pipeline, logger, config=config)

    # Database
    db = DatabaseDatabase(config_loader, logger, config=config, config_path_or_dir=args.config)
    db.init_db()

    # Meta-System Components
    from src.agent.core import Agent

    # Initialize Analytics
    analytics = AnalyticsEngine(db, logger)

    # Initialize Agent
    agent = Agent(logger, config_loader, config_path_or_dir=args.config)

    engine = ExecutionEngine(config, logger, db, clock=clock)
    engine.scanner = scanner  # Inject scanner

    # Initialize Streaming Scanner for live microstructure feature updates
    from src.data.calculator import FeatureCalculator

    feature_calculator = FeatureCalculator()
    streaming_scanner = StreamingScanner(
        calculator=feature_calculator,
        logger=logger,
        on_update=engine.on_streaming_update,
    )
    engine.streaming_scanner = streaming_scanner

    # Configure order executor
    if args.order_executor == "gateway":
        from src.engine.orders import GatewayOrderExecutor

        engine.order_executor = GatewayOrderExecutor(
            unified_client,
            logger,
            db=db,
            clock=clock,
        )  # type: ignore
    elif args.order_executor == "noop":
        from src.engine.orders import NoopOrderExecutor

        engine.order_executor = NoopOrderExecutor(logger, db=db, clock=clock)  # type: ignore

    # Register Strategies (config-driven, deterministic; PRD plug-and-play intent)
    strategy_registry = _build_strategy_registry()

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
            target = datetime.fromisoformat(args.eod_date).date() if args.eod_date else clock().date()
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

    # 4. Initial Scan — Start unified data stream
    from src.data.requirements import aggregate_requirements

    _required = aggregate_requirements([s.data_requirements for s in engine.strategies.values()])

    await unified_client.connect()
    stream_task: asyncio.Task[object] | None = asyncio.create_task(
        unified_client.start_stream(
            on_bar=engine.on_bar,
            on_quote=engine.on_quote,
            on_trade=engine.on_trade_data,
            on_reconnect=engine.reconcile_broker_state,
        )
    )
    reconcile_task = asyncio.create_task(engine.reconcile_loop())

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
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(config.get("timezone", "America/New_York"))

        eod_ran_for_date = None
        flattened_for_date = None
        last_warning_min = None

        use_market_time_for_session_control = bool(config.get("use_market_time_for_session_control", False))

        def _session_now_local() -> datetime:
            """
            Resolve session clock for market open/close control.

            Defaults to wall-clock to avoid stale market-data timestamps causing
            repeated close/restart loops in always-on runtime environments.
            """
            if use_market_time_for_session_control:
                t = getattr(engine.market_state, "time", None)
                if isinstance(t, datetime):
                    try:
                        if t.tzinfo is None:
                            t = t.replace(tzinfo=UTC)
                        return t.astimezone(tz)
                    except Exception:
                        pass
            return datetime.now(UTC).astimezone(tz)

        while True:
            now = _session_now_local()
            is_after_close_weekday = now.weekday() < 5 and now.hour >= 16

            if not _is_regular_market_session_local(now) and not is_after_close_weekday:
                market_state_message = "Market not open. Sleeping until regular session."

                next_open = _next_market_open_local(now)
                logger.info(
                    market_state_message,
                    now=now.isoformat(),
                    next_open=next_open.isoformat(),
                    sleep_seconds=max(1, int((next_open - now).total_seconds())),
                )
                while True:
                    current_local = datetime.now(UTC).astimezone(tz)
                    remaining = (next_open - current_local).total_seconds()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(300, max(1, int(remaining))))
                last_warning_min = None

                # Reconnect unified stream for the new session
                if stream_task is not None and not stream_task.done():
                    stream_task.cancel()
                await unified_client.disconnect()
                await unified_client.connect()
                stream_task = asyncio.create_task(
                    unified_client.start_stream(
                        on_bar=engine.on_bar,
                        on_quote=engine.on_quote,
                        on_trade=engine.on_trade_data,
                        on_reconnect=engine.reconcile_broker_state,
                    )
                )
                logger.info("Unified stream reconnected after pre-market sleep")
                continue

            # 1. Market Close Check (Exit at 16:00 ET)
            # PRD: no overnight holds.
            mins_to_close = (16 - now.hour) * 60 - now.minute

            # --- Live Guardrails: Flattening ---
            force_flat_mins = int(config.get("force_flat_before_close_mins", 15))
            if 0 < mins_to_close <= force_flat_mins:
                target_date = now.date()
                if flattened_for_date != target_date:
                    logger.warning(
                        "Market close approaching. Triggering early flattening.",
                        mins_to_close=mins_to_close,
                        threshold=force_flat_mins,
                    )
                    try:
                        engine.flatten_all(reason="pre_market_close")
                        flattened_for_date = target_date
                    except Exception as e:
                        logger.error("Early flatten failed", error=str(e), exc_info=True)
                        break

            # --- Live Guardrails: Countdown Warnings ---
            warning_stages = [15, 10, 5, 1]
            for w in warning_stages:
                if mins_to_close == w and last_warning_min != w:
                    logger.warning(
                        f"MARKET CLOSE COUNTDOWN: {w} minute(s) remaining until 16:00 ET",
                        mins_to_close=mins_to_close,
                    )
                    last_warning_min = w
                    break

            if now.hour >= 16:
                target_date = now.date()

                # Ensure final flattening if not already done
                if flattened_for_date != target_date:
                    try:
                        engine.flatten_all(reason="market_close")
                        flattened_for_date = target_date
                    except Exception as e:
                        logger.error("EOD flatten failed", error=str(e), exc_info=True)
                        break

                # PRD 9.1: run aggregation + Agent Stage 1 at end-of-day (configurable).
                if bool(config.get("auto_eod_agent", False)):
                    if eod_ran_for_date != target_date:
                        logger.info(
                            "Running automatic EOD aggregation + Agent Stage 1",
                            date=str(target_date),
                        )
                        try:
                            analytics.run_daily_aggregation(cast(date_type, target_date))
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
                if bool(config.get("exit_on_market_close", False)):
                    logger.info("Market closed. Exiting daily session.")
                    break

                next_open = _next_market_open_local(now)
                logger.info(
                    "Market closed. Sleeping until next market open.",
                    next_open=next_open.isoformat(),
                    sleep_seconds=max(1, int((next_open - now).total_seconds())),
                )
                while True:
                    current_local = datetime.now(UTC).astimezone(tz)
                    remaining = (next_open - current_local).total_seconds()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(300, max(1, int(remaining))))
                last_warning_min = None

                # Reconnect unified stream for the new trading session
                if stream_task is not None and not stream_task.done():
                    stream_task.cancel()
                await unified_client.disconnect()
                await unified_client.connect()
                stream_task = asyncio.create_task(
                    unified_client.start_stream(
                        on_bar=engine.on_bar,
                        on_quote=engine.on_quote,
                        on_trade=engine.on_trade_data,
                        on_reconnect=engine.reconcile_broker_state,
                    )
                )
                logger.info("Unified stream reconnected after overnight sleep")
                continue

            # 2. Run Scanner (every 5 mins)
            await engine.run_scan()

            # Update subscriptions to match new watchlist
            current_symbols = list(engine.symbol_states.keys())
            current_symbols.append(config.get("index_symbol", "SPY"))
            await unified_client.update_subscriptions(current_symbols)

            # Wait for next scan interval
            scanner_cfg = (config.get("scanner") or {}) if isinstance(config, dict) else {}
            interval_min = int(scanner_cfg.get("interval_minutes", 5))

            # High-fidelity mode when close to market end
            if 0 < mins_to_close <= 20:
                sleep_secs = 30
            else:
                sleep_secs = max(1, interval_min * 60)

            await asyncio.sleep(sleep_secs)

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
        if stream_task is not None and not stream_task.done():
            stream_task.cancel()
        if not reconcile_task.done():
            reconcile_task.cancel()
        await unified_client.close_async()
        await uw_client.close()


if __name__ == "__main__":
    asyncio.run(async_main())

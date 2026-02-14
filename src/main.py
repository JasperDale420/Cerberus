import argparse
import asyncio
import json
from collections.abc import Callable
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

from src.analysis.analytics import AnalyticsEngine
from src.analysis.db import DatabaseDatabase
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.data.api_client import CentralApiClient
from src.data.gateway_stream import GatewayStreamClient
from src.data.pipeline import FeaturePipeline
from src.data.unusual_whales import UnusualWhalesClient
from src.engine.execution import ExecutionEngine
from src.scanner.core import Scanner
from src.scanner.universe import UniverseBuilder


def _should_initialize_alpaca_client(
    order_executor: str,
    data_backend: str,
    failover_to_legacy: bool,
) -> bool:
    """Determine whether Alpaca client initialization is required."""
    normalized_executor = str(order_executor or "").strip().lower()
    normalized_backend = str(data_backend or "").strip().lower()
    if normalized_backend == "gateway" and normalized_executor in ("noop", "gateway") and not bool(failover_to_legacy):
        return False
    return True


def _should_start_alpaca_stream(order_executor: str, data_backend: str) -> bool:
    """Determine whether Alpaca direct bar/trade streams should run.

    The gateway stream is started separately when using gateway data backend.
    This only controls whether the direct Alpaca WebSocket stream (which
    requires Alpaca API keys) should also start.
    """
    normalized_executor = str(order_executor or "").strip().lower()
    normalized_backend = str(data_backend or "").strip().lower()
    if normalized_backend == "gateway":
        return normalized_executor == "alpaca"
    return True


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
    if now.hour >= 16:
        return False
    return True


def _capture_screener_snapshot(client: AlpacaClient, logger: StructuredLogger) -> None:
    """Capture daily screener snapshot for historical backtest replay."""
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "most_actives": [],
        "gainers": [],
        "losers": [],
    }

    try:
        snapshot["most_actives"] = client.get_most_actives(top=50)
    except Exception as e:
        logger.warning("Failed to fetch most actives", error=str(e))

    try:
        movers = client.get_movers(top=20)
        snapshot["gainers"] = movers.get("gainers", [])
        snapshot["losers"] = movers.get("losers", [])
    except Exception as e:
        logger.warning("Failed to fetch movers", error=str(e))

    # Save to data/screener_snapshots/<date>.jsonl
    output_dir = Path("data/screener_snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = output_dir / f"{date_str}.jsonl"

    with open(output_path, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    logger.info(
        "Screener snapshot captured",
        most_actives=len(snapshot["most_actives"]),
        gainers=len(snapshot["gainers"]),
        losers=len(snapshot["losers"]),
        path=str(output_path),
    )


def _build_strategy_registry() -> dict[str, type]:
    """Build the canonical strategy registry for runtime initialization."""
    from src.strategies.failed_breakout import FailedBreakoutStrategy
    from src.strategies.flow_momentum import FlowMomentumStrategy
    from src.strategies.fusion_v1 import FusionStrategyV1
    from src.strategies.gap_fill import GapFillStrategy
    from src.strategies.index_mean_reversion import IndexMeanReversionStrategy
    from src.strategies.momentum_continuation import MomentumContinuationStrategy
    from src.strategies.orb import ORBStrategy
    from src.strategies.pair_trading import PairTradingStrategy
    from src.strategies.trend_pullback import TrendPullbackStrategy
    from src.strategies.vix_spike_fade import VixSpikeFadeStrategy
    from src.strategies.vwap_reversion import VWAPReversionStrategy
    from src.strategies.vwap_trend_rider import VWAPTrendRiderStrategy

    return {
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
            data_backend=runtime_settings.cerberus_data_backend,
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
            return datetime.now(timezone.utc)

        return _clock

    # Deterministic clock (injectable for replay/tests)
    if isinstance(config, dict) and config.get("start_time_utc"):
        start = datetime.fromisoformat(str(config["start_time_utc"]).replace("Z", "+00:00"))
        clock = _fixed_clock(start)
    else:
        clock = _utc_now_clock()

    # 2. Components
    require_alpaca_client = _should_initialize_alpaca_client(
        order_executor=args.order_executor,
        data_backend=runtime_settings.cerberus_data_backend,
        failover_to_legacy=runtime_settings.cerberus_failover_to_legacy,
    )
    alpaca_client = AlpacaClient(config_loader, logger) if require_alpaca_client else None
    if not require_alpaca_client:
        logger.info(
            "Skipping Alpaca client initialization (not required for current mode)",
            data_backend=runtime_settings.cerberus_data_backend,
            failover_to_legacy=runtime_settings.cerberus_failover_to_legacy,
            order_executor=args.order_executor,
        )

    # Unusual Whales Client
    uw_client = UnusualWhalesClient(config_loader, logger, config=config)
    central_api_client = CentralApiClient(config_loader, logger)
    gateway_stream_client = GatewayStreamClient(config_loader, logger)

    feature_pipeline = FeaturePipeline(
        alpaca_client,
        uw_client,
        logger,
        central_api_client=central_api_client,
        config=config,
        clock=clock,
    )

    universe_builder = UniverseBuilder(
        config_loader,
        logger,
        config=config,
        config_path_or_dir=args.config,
        alpaca_client=alpaca_client,
        central_api_client=central_api_client,
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

    engine = ExecutionEngine(config, logger, db, alpaca_client, clock=clock)
    engine.scanner = scanner  # Inject scanner

    # Configure order executor
    if args.order_executor == "gateway":
        if not runtime_settings.use_gateway_data:
            raise ValueError("Gateway order executor requires CERBERUS_DATA_BACKEND=gateway|dual")
        from src.engine.orders import GatewayOrderExecutor

        engine.order_executor = GatewayOrderExecutor(
            central_api_client,
            logger,
            db=db,
            clock=clock,
        )  # type: ignore
    elif args.order_executor == "noop":
        from src.engine.orders import NoopOrderExecutor

        engine.order_executor = NoopOrderExecutor(logger, db=db, clock=clock)  # type: ignore
    elif args.order_executor == "alpaca" and runtime_settings.use_gateway_data:
        raise ValueError("Direct Alpaca order executor is blocked while CERBERUS_DATA_BACKEND uses gateway.")

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

    # 4. Initial Scan — Start bar data streams
    # Gateway bar stream: always start when using gateway data backend.
    # This provides the on_bar() callbacks that drive strategy signal generation.
    use_gateway_stream = runtime_settings.use_gateway_data
    gateway_stream_task: asyncio.Task[object] | None = None
    if use_gateway_stream:
        logger.info(
            "Starting gateway bar stream...",
            data_backend=runtime_settings.cerberus_data_backend,
            order_executor=args.order_executor,
        )
        gateway_stream_task = asyncio.create_task(
            gateway_stream_client.start_stream(engine.on_bar, on_reconnect=engine.reconcile_broker_state)
        )
        gateway_stream_client.subscribe(config.get("index_symbol", "SPY"))

    # Alpaca direct streams: only for direct Alpaca execution mode.
    start_alpaca_stream = _should_start_alpaca_stream(
        order_executor=args.order_executor,
        data_backend=runtime_settings.cerberus_data_backend,
    )
    alpaca_stream_task: asyncio.Task[object] | None = None
    if start_alpaca_stream:
        logger.info("Starting Alpaca stream...")
        assert alpaca_client is not None
        alpaca_stream_task = asyncio.create_task(
            alpaca_client.start_stream(engine.on_bar, on_reconnect=engine.reconcile_broker_state)
        )
        alpaca_client.subscribe(config.get("index_symbol", "SPY"))

    trade_stream_task: asyncio.Task[object] | None = None
    if start_alpaca_stream and args.order_executor == "alpaca":
        assert alpaca_client is not None
        trade_stream_task = asyncio.create_task(
            alpaca_client.start_trade_stream(engine.on_trade_update, on_reconnect=engine.reconcile_broker_state)
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
        import pytz  # type: ignore

        tz = pytz.timezone(config.get("timezone", "US/Eastern"))

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
                            t = t.replace(tzinfo=timezone.utc)
                        return t.astimezone(tz)
                    except Exception:
                        pass
            return datetime.now(timezone.utc).astimezone(tz)

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
                    current_local = datetime.now(timezone.utc).astimezone(tz)
                    remaining = (next_open - current_local).total_seconds()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(300, max(1, int(remaining))))
                last_warning_min = None
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
                # Capture daily screener snapshot for historical backtest data
                if alpaca_client is not None:
                    try:
                        _capture_screener_snapshot(alpaca_client, logger)
                    except Exception as e:
                        logger.warning("Screener snapshot capture failed", error=str(e))

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
                    current_local = datetime.now(timezone.utc).astimezone(tz)
                    remaining = (next_open - current_local).total_seconds()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(300, max(1, int(remaining))))
                last_warning_min = None
                continue

            # 2. Run Scanner (every 5 mins)
            await engine.run_scan()

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
        if gateway_stream_task is not None and not gateway_stream_task.done():
            gateway_stream_task.cancel()
        if alpaca_stream_task is not None and not alpaca_stream_task.done():
            alpaca_stream_task.cancel()
        if trade_stream_task is not None and not trade_stream_task.done():
            trade_stream_task.cancel()
        if not reconcile_task.done():
            reconcile_task.cancel()


if __name__ == "__main__":
    asyncio.run(async_main())

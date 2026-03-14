import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

from src.analysis.db import DatabaseDatabase
from src.core.config import ConfigLoader
from src.core.domain import OrderSide, Regime, Signal
from src.core.logger import StructuredLogger
from src.data.client import UnifiedDataClient
from src.data.pipeline import FeaturePipeline
from src.data.unusual_whales import UnusualWhalesClient
from src.engine.execution import ExecutionEngine
from src.scanner.core import Scanner
from src.scanner.universe import UniverseBuilder

# --- Configuration & Constants ---
KILL_SWITCH_FILE = "KILL_SWITCH"
ARTIFACTS_DIR = "artifacts"


# --- Harness Logger Setup ---
def setup_harness_logger(run_id: str, log_dir: str):
    # Configure Root Logger to capture all events
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates (e.g. from imports)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # JSON File Handler
    log_file = os.path.join(log_dir, f"trace_{run_id}.jsonl")
    file_handler = logging.FileHandler(log_file)

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "component": record.name,
                "run_id": run_id,
                "message": record.getMessage(),
            }
            if hasattr(record, "extra"):
                log_record.update(record.extra)
            return json.dumps(log_record, default=str)

    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # Console Handler (Simulated stdout for user visibility)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(console)

    return logging.getLogger("Harness")


class TestHarness:
    def __init__(self, args):
        self.args = args
        self.run_id = str(uuid.uuid4())
        self.start_time = datetime.now(timezone.utc)
        self.artifacts_path = os.path.join(ARTIFACTS_DIR, self.run_id)
        os.makedirs(self.artifacts_path, exist_ok=True)

        self.logger = setup_harness_logger(self.run_id, self.artifacts_path)
        self.logger.info(f"Starting Paper-Live Test run_id={self.run_id}")

        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config(args.config)
        self.engine: Optional[ExecutionEngine] = None
        self.unified_client: Optional[UnifiedDataClient] = None
        self.bot_log_path: Optional[str] = None

        # Kill Switch Check
        self._check_kill_switch()

    def _check_kill_switch(self):
        if os.path.exists(KILL_SWITCH_FILE):
            self.logger.critical("KILL SWITCH FOUND! Aborting immediately.")
            sys.exit(1)

    def _setup_components(self):
        """Initialize the bot components with test safeguards"""
        # Inject Run ID into config or environment for other loggers
        os.environ["CERBERUS_RUN_ID"] = self.run_id

        # Configure bot loggers to write into the run artifacts directory so the
        # harness can deterministically generate stats from structured logs.
        logging_cfg: Dict[str, Any] = {}
        if isinstance(self.config.get("logging"), dict):
            logging_cfg = dict(self.config.get("logging") or {})
        self.bot_log_path = os.path.join(self.artifacts_path, "bot.log.jsonl")
        logging_cfg["file_path"] = self.bot_log_path

        def _slog(name: str, level: str = "INFO") -> StructuredLogger:
            # Use a unique logger name per run_id so handlers don't leak across runs.
            return StructuredLogger(f"{name}.{self.run_id}", level=level, logging_config=logging_cfg)

        # 1. Logger
        # We want the main bot logger to also write to our artifact file if possible,
        # or we rely on the main config to point to a file.
        # For now, we trust the main logger config but we really want to capture everything.
        # We can re-configure the root logger or specific loggers here.
        # But let's stick to the SUT as strict as possible.

        # 2. Unified Data Client
        from src.core.settings import get_settings

        runtime = get_settings()
        self.unified_client = UnifiedDataClient(
            gateway_url=runtime.cerberus_gateway_url,
            gateway_key=runtime.cerberus_gateway_key,
        )

        # 3. Execution Engine
        db = DatabaseDatabase(
            self.config_loader,
            _slog("DB", level="INFO"),
            config=self.config,
            config_path_or_dir=self.args.config,
        )
        db.init_db()

        self.engine = ExecutionEngine(
            self.config,
            _slog("Engine", level="INFO"),
            db,
            None,
            run_id=self.run_id,
        )

        # 4. Scanner & Pipeline
        uw_client = UnusualWhalesClient(
            self.config_loader,
            _slog("UW", level="INFO"),
            config=self.config,
        )
        feature_pipeline = FeaturePipeline(
            self.unified_client,
            uw_client,
            _slog("Pipeline", level="INFO"),
            config=self.config,
            clock=getattr(self.engine, "clock", None),
        )
        universe_builder = UniverseBuilder(
            self.unified_client,
            _slog("Universe", level="INFO"),
            config=self.config,
            config_path_or_dir=self.args.config,
            clock=getattr(self.engine, "clock", None),
        )
        scanner = Scanner(
            universe_builder,
            feature_pipeline,
            _slog("Scanner", level="INFO"),
            config=self.config,
        )

        self.engine.scanner = scanner

        # INJECT FAILURE SIMULATION IF REQUESTED
        if self.args.scenario == "failure":
            self._inject_failures()

        # INJECT RISK BREACH IF REQUESTED
        if self.args.scenario == "risk":
            self.logger.info("Risk Breach Scenario: reducing limits to trigger rejection")
            # Create a mock risk config override
            self.engine.risk_manager.max_daily_loss = 1.0  # $1 daily loss limit
            self.engine.risk_manager.max_risk_per_trade = 0.5

    def _inject_failures(self):
        alpaca = self.alpaca
        assert alpaca is not None
        self.logger.info("Injecting Failures: Monkey-patching AlpacaClient")

        import random

        def faulty_submit(*args, **kwargs):
            error_type = random.choice(["500", "429", "timeout"])
            self.logger.info(f"Injecting {error_type} Error")

            if error_type == "500":
                raise Exception("Simulated 500 Internal Server Error")
            elif error_type == "429":
                raise Exception("API rate limit exceeded")  # Simulating 429
            elif error_type == "timeout":
                import asyncio

                # In a real async loop we'd sleep, but here we just raise TimeoutError
                raise asyncio.TimeoutError("Simulated Request Timeout")

        alpaca.trading_client.submit_order = faulty_submit  # type: ignore[method-assign]

    def _export_broker_state(self):
        """Exports current orders and positions from Broker"""
        try:
            assert self.alpaca is not None
            orders = self.alpaca.trading_client.get_orders()
            positions = self.alpaca.trading_client.get_all_positions()
            account = self.alpaca.trading_client.get_account()

            def _dump(obj: Any) -> Any:
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                if hasattr(obj, "dict"):
                    return obj.dict()
                return str(obj)

            # Serialize
            data = {
                "orders": [_dump(o) for o in orders],
                "positions": [_dump(p) for p in positions],
                "account": _dump(account),
            }

            export_path = os.path.join(self.artifacts_path, "broker_export.json")
            with open(export_path, "w") as f:
                json.dump(data, f, default=str, indent=2)
            self.logger.info(f"Broker state exported to {export_path}")
        except Exception as e:
            self.logger.error(f"Failed to export broker state: {e}")

    async def run(self):
        self._setup_components()

        duration = self.args.duration * 60
        end_time = time.time() + duration

        self.logger.info(f"Running for {self.args.duration} minutes...")

        try:
            # Setup signal handler for graceful shutdown
            loop = asyncio.get_running_loop()
            stop_event = asyncio.Event()

            def signal_handler():
                self.logger.info("Signal received, stopping...")
                stop_event.set()

            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)

            # Main Test Loop
            signal_injected = False
            while time.time() < end_time and not stop_event.is_set():
                self._check_kill_switch()

                # Signal Injection (Once per run for simplicity in this harness)
                if self.args.inject_signal and not signal_injected:
                    self.logger.info("Injecting Synthetic Signal...")
                    await self._inject_test_signal()
                    signal_injected = True

                # 1. Run Scan (simulate scheduler)
                self.logger.info("Triggering scan...")
                assert self.engine is not None
                await self.engine.run_scan()

                # 2. Wait (simulate interval)
                # Wait 10 seconds or until stopped
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass

        except Exception as e:
            self.logger.error(f"Harness failed with exception: {e}", extra={"exception": str(e)})
            raise
        finally:
            self._generate_report()

    async def _inject_test_signal(self):
        """Injects a fake signal to test the Order Path"""
        from collections import deque

        from src.core.domain import SymbolState

        assert self.engine is not None
        # Ensure state exists
        if "SPY" not in self.engine.symbol_states:
            self.engine.symbol_states["SPY"] = SymbolState(
                symbol="SPY",
                bars=deque(maxlen=100),
                position=None,
                indicators={},
                open_orders={},
                allowed_strategies=["ManualInjection"],
                meta={},
            )

        # Create a Signal object
        # Assuming SPY for test
        test_signal = Signal(
            symbol="SPY",
            side=OrderSide.BUY,
            size_hint=10.0,
            entry_price=400.0,  # Arbitrary
            stop_price=395.0,
            target_price=410.0,
            regime=Regime.BULL,
            generated_at=datetime.now(timezone.utc),
            strategy="ManualInjection",
            correlation_id=str(uuid.uuid4()),
        )
        # Manually invoke engine processing
        self.engine._process_signal(test_signal)

    def _generate_report(self):
        self._export_broker_state()
        self.logger.info("Generating Report...")
        report_path = os.path.join(
            self.artifacts_path, "summary.json"
        )  # Updated to summary.json for requirement matching
        log_path = os.path.join(self.artifacts_path, f"trace_{self.run_id}.jsonl")

        stats: Dict[str, Any] = {
            "signals": 0,
            "signals_injected": 0,
            "approved": 0,
            "rejected": 0,
            "submitted": 0,
            "exec_errors": 0,
            "reasons": {},
            "latencies": [],
        }

        exceptions = 0

        seen_signals: set[str] = set()
        seen_approved: set[str] = set()
        seen_rejected: set[str] = set()
        seen_submitted: set[str] = set()
        seen_exec_errors: set[str] = set()

        def _iter_jsonl(path: str):
            try:
                with open(path, "r") as f:
                    for line in f:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                return

        try:
            for record in _iter_jsonl(log_path):
                msg = record.get("message", "")
                if record.get("level") == "ERROR":
                    exceptions += 1
                if "Injecting Synthetic Signal" in msg:
                    stats["signals_injected"] += 1

            if self.bot_log_path:
                for record in _iter_jsonl(self.bot_log_path):
                    msg = record.get("message", "")
                    correlation_id = str(record.get("correlation_id") or "").strip() or None

                    if record.get("level") == "ERROR":
                        exceptions += 1

                    if "Processing signal" in msg:
                        if correlation_id is None or correlation_id not in seen_signals:
                            stats["signals"] += 1
                            if correlation_id is not None:
                                seen_signals.add(correlation_id)
                    elif "Signal approved" in msg:
                        if correlation_id is None or correlation_id not in seen_approved:
                            stats["approved"] += 1
                            if correlation_id is not None:
                                seen_approved.add(correlation_id)
                    elif "Signal rejected" in msg:
                        if correlation_id is None or correlation_id not in seen_rejected:
                            stats["rejected"] += 1
                            if correlation_id is not None:
                                seen_rejected.add(correlation_id)
                        reason = record.get("reason_code", "UNKNOWN")
                        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                    elif "Order submitted" in msg:
                        if correlation_id is None or correlation_id not in seen_submitted:
                            stats["submitted"] += 1
                            if correlation_id is not None:
                                seen_submitted.add(correlation_id)
                        if "exec_latency_ms" in record:
                            stats["latencies"].append(record["exec_latency_ms"])
                    elif (
                        "Order execution failed" in msg
                        or "ORDER_SUBMIT_FAILED" in msg
                        or "Order submission failed" in msg
                    ):
                        if correlation_id is None or correlation_id not in seen_exec_errors:
                            stats["exec_errors"] += 1
                            if correlation_id is not None:
                                seen_exec_errors.add(correlation_id)

            # Calculate Latency Stats
            # Calculate Latency Stats
            import statistics

            latencies = cast(List[float], stats["latencies"])
            if latencies:
                stats["mean_latency"] = statistics.mean(latencies)
                if len(latencies) > 1:
                    stats["p95_latency"] = statistics.quantiles(latencies, n=20)[-1]  # approx p95
                else:
                    stats["p95_latency"] = latencies[0]
            else:
                stats["mean_latency"] = 0
                stats["p95_latency"] = 0

        except FileNotFoundError:
            self.logger.error("Log file not found for report generation")
            return

        outcome = "PASS"
        if exceptions > 0 and self.args.scenario == "happy":
            outcome = "Review Required (Exceptions Found)"
        if stats["exec_errors"] > 0 and self.args.scenario != "failure":
            outcome = "FAIL (Execution Errors)"

        # Failure Scenario Expectation
        if self.args.scenario == "failure":
            if stats["exec_errors"] > 0:
                outcome = "PASS (Failures Caught)"
            else:
                outcome = "FAIL (No Failures Caught)"

        # Risk Scenario Expectation
        if self.args.scenario == "risk":
            # PASS if at least one injected signal was blocked before submission.
            if stats["signals_injected"] > 0 and stats["submitted"] == 0 and stats["approved"] == 0:
                outcome = "PASS (Risk Blocked)"
            else:
                outcome = "FAIL (Risk Did Not Block)"

        # Generate JSON Summary Report
        summary_data = {
            "run_id": self.run_id,
            "date": datetime.now().isoformat(),
            "scenario": self.args.scenario,
            "outcome": outcome,
            "stats": stats,
            "exceptions": exceptions,
            "pass": "PASS" in outcome,
            "bot_log_path": self.bot_log_path,
        }

        with open(report_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        self.logger.info(f"Report saved to {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Cerberus Paper-Live Harness")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config")
    parser.add_argument("--scenario", choices=["happy", "failure", "risk"], default="happy")
    parser.add_argument("--duration", type=int, default=5, help="Duration in minutes")
    parser.add_argument("--force-live", action="store_true", help="Allow Live execution (DANGEROUS)")
    parser.add_argument(
        "--inject-signal",
        action="store_true",
        help="Inject a synthetic signal for testing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    harness = TestHarness(args)
    asyncio.run(harness.run())

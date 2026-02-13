import argparse
import asyncio
import json
import logging
import os
import signal
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.analysis.db import DatabaseDatabase
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.data.pipeline import FeaturePipeline
from src.data.unusual_whales import UnusualWhalesClient
from src.engine.execution import ExecutionEngine
from src.scanner.core import Scanner
from src.scanner.universe import UniverseBuilder

# --- Constants ---
ARTIFACTS_DIR = Path("artifacts")
KILL_SWITCH_FILE = Path("KILL_SWITCH")


class PaperLiveHarness:
    def __init__(self, duration_minutes: int, scenario: str, config_path: str):
        self.run_id = str(uuid.uuid4())
        self.duration_minutes = duration_minutes
        self.scenario = scenario
        self.config_path = config_path
        self.start_time = datetime.now(timezone.utc)
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)
        self.stop_requested = False

        # Artifacts setup
        self.run_dir = ARTIFACTS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Logging setup
        self.log_file = self.run_dir / "execution.jsonl"
        self._setup_logging()

        self.logger = StructuredLogger("PaperLiveTest", level="INFO")
        # Root logger handler writes all module logs to the file.
        # Inject run_id into logger via adapter or just Context handling if we had it.
        # Since I modified logger to look for run_id in record, I can use an adapter or filter
        # Or I can just pass run_id in every log.
        # Let's monkeypatch StructuredLogger methods slightly or use an adapter.
        # Actually easier: I'll just pass run_id manually in my harness logs,
        # and for the system logs, I'll let them be.
        # Ideally I should ContextVar it, but for now I'll trust the traceback/correlation logic.

        self.logger.info("Test Harness Initialized", run_id=self.run_id, scenario=scenario)

        # Stats
        self.stats = {
            "signals_generated": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_rejected": 0,
            "risk_blocks": 0,
            "exceptions": 0,
        }

    def _setup_logging(self):
        # Add a FileHandler to the root logger to capture ALL logs from ALL modules
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        formatter = logging.Formatter("%(message)s")

        fh = logging.FileHandler(self.log_file)
        fh.setFormatter(formatter)
        root.addHandler(fh)
        return fh

    def _check_kill_switch(self):
        if KILL_SWITCH_FILE.exists():
            self.logger.critical("KILL SWITCH ACTIVATED via file")
            self.stop_requested = True
            return True
        if os.environ.get("KILL_SWITCH"):
            self.logger.critical("KILL SWITCH ACTIVATED via env")
            self.stop_requested = True
            return True
        return False

    def setup_system(self):
        self.logger.info("Setting up system components...")

        # 1. Config
        os.environ["PAPER_LIVE"] = "true"
        os.environ["ALPACA_PAPER"] = "true"

        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config(self.config_path)

        # Enforce Risk Limits for Test
        self.config["max_orders_per_day"] = 5  # Safe low limit for test
        self.config["max_open_positions"] = 2
        self.config["max_notional_per_order"] = 2000.0

        # 2. Components
        try:
            self.alpaca = AlpacaClient(self.config_loader, self.logger)
            self.uw_client = UnusualWhalesClient(self.config_loader, self.logger)
            self.pipeline = FeaturePipeline(self.alpaca, self.uw_client, self.logger)
            self.universe = UniverseBuilder(self.config_loader, self.logger)
            self.scanner = Scanner(self.universe, self.pipeline, self.logger)
            self.db = DatabaseDatabase(self.config_loader, self.logger)
            self.db.init_db()
            # Ideally use real DB for "Traceability".

            self.engine = ExecutionEngine(self.config, self.logger, self.db, self.alpaca)
            self.engine.scanner = self.scanner

        except Exception as e:
            self.logger.critical("System Setup Failed", error=str(e), exc_info=True)
            raise

    def inject_failures(self):
        if self.scenario == "failure":
            self.logger.info("Injecting Failures...")

            # Monkeypatch Alpaca submit_order to simulate 50% failure
            original_submit = self.alpaca.trading_client.submit_order

            def faulty_submit(*args, **kwargs):
                import random

                if random.random() < 0.5:
                    raise Exception("Injected Broker Failure: 500 Internal Server Error")
                return original_submit(*args, **kwargs)

            self.alpaca.trading_client.submit_order = faulty_submit  # type: ignore[method-assign]

            # Monkeypatch Data Stream? (If we were using it)
        # pass removed

    def inject_force_trade(self):
        """
        Injects a synthetic signal to force an order submission.
        """
        self.logger.info("Injecting Force Trade Signal...")
        from src.core.domain import OrderSide, Regime, Signal

        # Create a synthetic signal
        # We need a symbol that exists in the universe or adds it ad-hoc.
        # SPY is usually in index or safe.
        symbol = "SPY"
        # Ensure symbol state exists
        if symbol not in self.engine.symbol_states:
            # Manually create state if needed, but engine._process_signal might need it.
            # actually engine.on_bar creates it.
            # Let's mock it or rely on scanner having run.
            # If scanner didn't pick SPY, we might need to add it.
            pass

        # We need to make sure RiskManager doesn't reject it.
        # Current price approximately?
        price = 400.0  # Arbitrary default, hope it's not too far off to trigger weird limits?
        # Actually, let's fetch last trade if possible, or just use a safe number.
        # In Paper, limit orders far away might sit there.
        # Use a MARKET order or marketable limit? OrderIntent defaults to LIMIT.

        # Let's try to get a real price from alpaca if possible, else 200.
        try:
            get_latest_trade = getattr(self.alpaca.trading_client, "get_latest_trade", None)
            if callable(get_latest_trade):
                quote = get_latest_trade(symbol)
                price = float(getattr(quote, "price", 200.0))
            else:
                price = 200.0
        except Exception:
            price = 200.0

        signal = Signal(
            symbol=symbol,
            side=OrderSide.BUY,
            size_hint=1,  # 1 share
            entry_price=price,
            stop_price=price * 0.95,
            target_price=price * 1.05,
            strategy="force_test",
            regime=Regime.BULL,
            generated_at=datetime.now(timezone.utc),
            correlation_id=f"force-{uuid.uuid4()}",
            meta={"force": True},
        )

        # We need to Ensure SymbolState is populated for RiskManager
        from collections import deque

        from src.engine.execution import SymbolState

        if symbol not in self.engine.symbol_states:
            self.engine.symbol_states[symbol] = SymbolState(
                symbol=symbol,
                bars=deque(maxlen=100),
                position=None,
                indicators={},
                open_orders={},
                allowed_strategies=["force_test"],
                meta={},
            )

        # Inject directly into engine
        self.engine._process_signal(signal)
        self.stats["signals_generated"] += 1
        self.logger.info("Force Trade Signal Injected", signal_id=signal.correlation_id)

    async def run(self):
        self.setup_system()
        self.inject_failures()

        self.logger.info("Starting Run Loop", duration_minutes=self.duration_minutes)

        # Run Initial Scan
        try:
            self.logger.info("Running Initial Scan")
            await self.engine.run_scan()
        except Exception as e:
            self.logger.error("Scan Failed", error=str(e))
            self.stats["exceptions"] += 1

        # Main Loop simulating "Live"
        force_trade_triggered = False

        while datetime.now(timezone.utc) < self.end_time and not self.stop_requested:
            if self._check_kill_switch():
                break

            try:
                # Force Trade Logic
                if self.scenario == "force_trade" and not force_trade_triggered:
                    # Wait a bit for system to settle then trigger
                    await asyncio.sleep(5)
                    self.inject_force_trade()
                    force_trade_triggered = True

                await asyncio.sleep(10)  # Fast loop for test

                # Check health
                self.logger.info("Heartbeat", open_positions=len(self.engine.symbol_states))

            except Exception as e:
                self.logger.error("Loop Exception", error=str(e), exc_info=True)
                self.stats["exceptions"] += 1
                if self.scenario != "failure":
                    # In happy path, we might want to fail hard?
                    # But the requirement says "No UNHANDLED exceptions".
                    # Caught here means it's handled, but we log it.
                    pass

        self.generate_report()

    def generate_report(self):
        report = {
            "run_id": self.run_id,
            "status": "PASS" if self.stats["exceptions"] == 0 else "FAIL",  # Strict
            "scenario": self.scenario,
            "stats": self.stats,
            "artifacts_path": str(self.run_dir.absolute()),
        }

        if self.scenario == "failure" and self.stats["exceptions"] > 0:
            # If we injected failures, exceptions are expected, so it might be a pass if handled gracefully.
            # We need to refine this logic. For now, let's just dump stats.
            report["status"] = "REVIEW_NEEDED"

        report_file = self.run_dir / "report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info("Test Complete", report=report)
        print(f"\nTest finished. Report saved to {report_file}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=5, help="Duration in minutes")
    parser.add_argument("--scenario", choices=["happy", "failure", "force_trade"], default="happy")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    harness = PaperLiveHarness(args.duration, args.scenario, args.config)

    # Handle SIGINT
    def signal_handler(sig, frame):
        harness.logger.warning("Interrupt received, shutting down...")
        harness.stop_requested = True

    signal.signal(signal.SIGINT, signal_handler)

    await harness.run()


if __name__ == "__main__":
    asyncio.run(main())

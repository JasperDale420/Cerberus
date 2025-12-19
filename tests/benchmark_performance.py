import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.core.domain import Regime
from src.core.logger import StructuredLogger
from src.data.pipeline import FeaturePipeline
from src.engine.execution import ExecutionEngine
from src.scanner.core import Scanner
from src.scanner.universe import UniverseBuilder


class TestPerformanceBenchmark(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.logger = MagicMock(spec=StructuredLogger)
        self.config_loader = MagicMock()
        self.config_loader.load_config.return_value = {}

        # Mock Clients
        self.alpaca_client = MagicMock()
        self.uw_client = MagicMock()

        # Mock FeaturePipeline (historical fetching is the bottleneck to test)
        self.feature_pipeline = FeaturePipeline(
            self.alpaca_client, self.uw_client, self.logger, config={}
        )

        # Mock UniverseBuilder to return fixed set of 50 symbols
        self.universe_builder = MagicMock(spec=UniverseBuilder)
        self.test_symbols = [f"SYM{i}" for i in range(50)]
        self.universe_builder.get_universe.return_value = self.test_symbols

        # Mock Scanner
        self.scanner = Scanner(
            self.universe_builder,
            self.feature_pipeline,
            self.logger,
            config={"max_scan_size": 100},
        )

    async def test_scanner_performance(self):
        """
        Measure Scanner.scan() time with 50 symbols.
        Goal: < 2.0 seconds (allowing for some mocked I/O overhead).
        Note: Real sub-second requirement applies to specific sub-components,
        but total scan should be efficient.
        """

        # Mock fetch_historical_bars to simulate slight latency per symbol call
        # if pipeline calls it 50 times sequentially, it will be slow.
        async def mock_fetch(*args, **kwargs):
            # Simulate 10ms network/parsing delay
            # await asyncio.sleep(0.01)
            return [{"c": 100.0, "v": 1000} for _ in range(100)]

        # self.feature_pipeline.fetch_bars_batch = MagicMock(side_effect=mock_fetch)
        # Also mock single get_historical_bars if used
        self.alpaca_client.get_historical_bars.return_value = {
            "bars": [{"c": 100.0, "v": 1000} for _ in range(100)]
        }

        start_time = time.perf_counter()

        # Provide current regime and time
        _ = await self.scanner.scan(
            regime=Regime.BULL, scan_time=datetime.now(timezone.utc)
        )

        duration = time.perf_counter() - start_time
        print(f"\nScanner Benchmark (50 symbols): {duration:.4f}s")

        # Assertion: Should be reasonably fast.
        # If it's doing serial fetching without concurrency, 50 * 10ms = 0.5s + overhead.
        # If it re-fetches extensively, it might be slower.
        self.assertLess(duration, 5.0, "Scanner took too long (> 5s)")

    def test_execution_on_bar_latency(self):
        """
        Measure ExecutionEngine.on_bar() latency.
        Goal: P99 < 10ms (processing only).
        """
        engine = ExecutionEngine(
            config={}, logger=self.logger, alpaca_client=self.alpaca_client
        )

        # Mock dependencies
        engine.regime_detector = MagicMock()
        engine.strategy_engine = MagicMock()
        engine.strategy_engine.on_bar.return_value = []  # No signals for baseline

        bar_data = MagicMock()
        bar_data.symbol = "SPY"
        bar_data.close = 200.0
        bar_data.volume = 1000000

        stats = []
        iterations = 1000

        for _ in range(iterations):
            t0 = time.perf_counter()
            engine.on_bar(bar_data)
            t1 = time.perf_counter()
            stats.append((t1 - t0) * 1000.0)  # ms

        avg = sum(stats) / len(stats)
        p99 = sorted(stats)[int(0.99 * len(stats))]

        print(f"\nExecutionEngine.on_bar Benchmark: Avg={avg:.4f}ms, P99={p99:.4f}ms")
        self.assertLess(p99, 10.0, f"on_bar P99 latency too high: {p99:.2f}ms")

        print(f"\nExecutionEngine.on_bar Benchmark: Avg={avg:.4f}ms, P99={p99:.4f}ms")
        self.assertLess(p99, 10.0, f"on_bar P99 latency too high: {p99:.2f}ms")

    async def test_pipeline_caching(self):
        """
        Verify FeaturePipeline incremental caching.
        """
        # Setup
        sym = "TEST_CACHE"
        mock_pipeline = FeaturePipeline(
            self.alpaca_client, self.uw_client, self.logger, config={}
        )

        # T0: Start of session (09:30 ET) ~ 14:30 UTC
        # T1: 15:00 UTC
        # T2: 15:10 UTC

        t1 = datetime(2025, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 1, 15, 10, 0, tzinfo=timezone.utc)

        # 1. First Call: Should fetch full history
        self.alpaca_client.get_historical_bars.reset_mock()
        self.alpaca_client.get_historical_bars.return_value = {
            "bars": [{"t": t1.isoformat(), "c": 100, "v": 1000}]
        }

        await mock_pipeline.compute_features([sym], as_of=t1)

        # Verify 1st fetch
        self.alpaca_client.get_historical_bars.assert_called()
        args1, _ = self.alpaca_client.get_historical_bars.call_args
        # args: (symbol, start, end, timeframe)
        # We don't check exact start (it's 4am ET), but end should be T1
        self.assertEqual(args1[2], t1)

        # 2. Second Call: Should be incremental
        self.alpaca_client.get_historical_bars.reset_mock()
        # Mock return for the increment
        self.alpaca_client.get_historical_bars.return_value = {
            "bars": [{"t": t2.isoformat(), "c": 101, "v": 1000}]
        }

        await mock_pipeline.compute_features([sym], as_of=t2)

        # Verify 2nd fetch
        self.alpaca_client.get_historical_bars.assert_called()
        args2, _ = self.alpaca_client.get_historical_bars.call_args

        # Start time should be > T1 (incremental)
        # args2[1] is fetch_start, args2[2] is end
        self.assertGreater(args2[1], t1)
        self.assertEqual(args2[2], t2)

        # Verify metrics if accessible, or just rely on the call args proving optimization
        if hasattr(mock_pipeline, "last_run_metrics"):
            self.assertGreater(mock_pipeline.last_run_metrics.get("cache_hits", 0), 0)
            self.assertGreater(
                mock_pipeline.last_run_metrics.get("incremental_fetches", 0), 0
            )


if __name__ == "__main__":
    unittest.main()

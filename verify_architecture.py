import os
import sys

# Add src to path
sys.path.append(os.getcwd())


def verify_architecture():
    print("Verifying Core Architecture...")

    try:
        from src.core.domain import Bar, Signal

        # Use them to avoid F401 or just check import
        _ = Bar
        _ = Signal

        print("[PASS] Domain models imported")
    except ImportError as e:
        print(f"[FAIL] Domain models import failed: {e}")
        return

    try:
        from src.analysis.schema import Base, Trade

        _ = Base
        _ = Trade

        print("[PASS] Schema models imported")
    except ImportError as e:
        print(f"[FAIL] Schema import failed: {e}")
        return

    try:
        from src.analysis.db import DatabaseDatabase
        from src.core.config import ConfigLoader
        from src.core.logger import StructuredLogger
        from src.data.alpaca import AlpacaClient
        from src.data.pipeline import FeaturePipeline
        from src.data.unusual_whales import UnusualWhalesClient

        logger = StructuredLogger(name="Verify")
        config_loader = ConfigLoader()

        # Test Config Merging
        config = config_loader.load_config()
        if "strategies" in config and "risk" in config:
            print("[PASS] Config merged loaded (strategies and risk keys found)")
        else:
            print(f"[FAIL] Config keys missing. Keys found: {list(config.keys())}")

        # Test DB Init
        db = DatabaseDatabase(config_loader, logger)
        db.init_db()
        print("[PASS] Database initialized and tables created (sqlite)")

        # Test Strategy Instantiation
        from src.strategies.vwap_reversion import VWAPReversionStrategy

        dummy_cfg = {"sigma_band": 2.0, "enabled": True}
        _ = VWAPReversionStrategy(dummy_cfg, logger)
        print("[PASS] Strategy (VWAPReversion) instantiated")

        # Test Scanner Instantiation
        from src.scanner.core import Scanner
        from src.scanner.universe import UniverseBuilder

        # Mock dependencies for Scanner
        alpaca = AlpacaClient(config_loader, logger)  # Mock or real is fine for init
        uw_client = UnusualWhalesClient(config_loader, logger)
        pipeline = FeaturePipeline(alpaca, uw_client, logger, config=config)
        universe = UniverseBuilder(
            config_loader, logger, config=config, alpaca_client=alpaca
        )

        _ = Scanner(universe, pipeline, logger, config=config)
        print("[PASS] Scanner instantiated")

    except ImportError as e:
        print(f"[FAIL] Import validation failed: {e}")
        return
    except Exception as e:
        print(f"[FAIL] Runtime verification failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print("Architecture Verification Completed Successfully.")


if __name__ == "__main__":
    verify_architecture()

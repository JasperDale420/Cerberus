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

        logger = StructuredLogger(name="Verify")
        config_loader = ConfigLoader()

        # Test DB Init
        db = DatabaseDatabase(config_loader, logger)
        db.init_db()
        print("[PASS] Database initialized and tables created (sqlite)")

    except ImportError as e:
        print(f"[FAIL] DB/Config import failed: {e}")
        return
    except Exception as e:
        print(f"[FAIL] Runtime error: {e}")
        return

    print("Architecture Verification Completed Successfully.")


if __name__ == "__main__":
    verify_architecture()

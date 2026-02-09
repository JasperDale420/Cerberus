import argparse
import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from src.backtest.runner import BacktestRunner
from src.core.logger import StructuredLogger


async def main():
    parser = argparse.ArgumentParser(description="Run Backtest")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--start-date", required=True, help="Start date (ISO)")
    parser.add_argument("--end-date", required=True, help="End date (ISO)")
    parser.add_argument(
        "--offline-bars-dir",
        default=None,
        help="Optional directory of {SYMBOL}_{timeframe}.jsonl bars for deterministic offline runs",
    )
    parser.add_argument(
        "--warmup-days",
        type=int,
        default=250,
        help="Number of historical days to load for indicator warmup (default: 250)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results JSON (default: results/backtest_results_{timestamp}.json)",
    )
    args = parser.parse_args()

    # Configure logger to stdout for visibility
    logger = StructuredLogger("BacktestMain")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Create descriptive filename
        # Format: backtest_{config}_{start}_to_{end}_{timestamp}.json
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        start_str = args.start_date.split("T")[0]  # Simple date
        end_str = args.end_date.split("T")[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"backtest_{config_name}_{start_str}_to_{end_str}_{timestamp}.json"

        # Sanitize filename slightly (replace potential path seps)
        filename = filename.replace("/", "-").replace(":", "-")

        output_path = os.path.join("results", filename)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        runner = BacktestRunner(
            args.config,
            args.start_date,
            args.end_date,
            offline_bars_dir=args.offline_bars_dir,
            warmup_days=args.warmup_days,
        )
        result = await runner.run()  # Now returns dict

        # Save to file
        import json

        with open(output_path, "w") as f:
            # Helper for datetime serialization
            def temp_serializer(obj):
                if hasattr(obj, "isoformat"):
                    return obj.isoformat()
                return str(obj)

            json.dump(result, f, indent=2, default=temp_serializer)

        logger.info("Results saved", path=output_path)

    except Exception as e:
        logger.error("Backtest failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

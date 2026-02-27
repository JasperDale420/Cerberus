import argparse
import asyncio
import os
from datetime import datetime, timezone

import pandas as pd

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Order as DbOrder
from src.backtest.executor import SimulatedOrderExecutor
from src.core.config import ConfigLoader
from src.core.domain import Bar
from src.core.logger import StructuredLogger
from src.data.api_client import CentralApiClient
from src.engine.execution import ExecutionEngine


class BacktestClock:
    def __init__(self, start_time: datetime):
        self._now = start_time

    def __call__(self) -> datetime:
        return self._now

    def set_time(self, new_time: datetime):
        self._now = new_time


async def run_backtest(start_date: str, end_date: str, config_path: str):
    import dotenv

    dotenv.load_dotenv()

    # Ensure local gateway works without docker DNS
    if not os.environ.get("CERBERUS_GATEWAY_URL"):
        os.environ["CERBERUS_GATEWAY_URL"] = "http://localhost:8080"
    elif "host.docker.internal" in os.environ.get("CERBERUS_GATEWAY_URL", ""):
        os.environ["CERBERUS_GATEWAY_URL"] = "http://localhost:8080"

    logger = StructuredLogger("CERBERUS-BACKTEST")
    config_loader = ConfigLoader()
    config = config_loader.load_config(config_path)

    # Use a temporary file-based DB instead of memory to avoid weird thread issues with sqlite
    # or ensure we have isolated reporting.
    db_path = "/tmp/cerberus_backtest.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    config["database_url"] = f"sqlite:///{db_path}"

    db = DatabaseDatabase(config_loader, logger, config=config, config_path_or_dir=config_path)
    db.init_db()

    start_dt = datetime.fromisoformat(start_date)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)

    end_dt = datetime.fromisoformat(end_date)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    clock = BacktestClock(start_dt)

    # Needs valid gateway creds to fetch history, but won't trade live
    gateway_client = CentralApiClient(config_loader, logger)

    logger.info("Fetching historical data for backtest...", start_date=start_date, end_date=end_date)

    symbols = set()
    index_symbol = config.get("index_symbol", "SPY")
    vol_symbol = config.get("regime", {}).get("vol_symbol", "VXX")
    symbols.add(index_symbol)
    symbols.add(vol_symbol)

    universes = config.get("universe", {})
    if isinstance(universes, list):
        for u in universes:
            for s in u.get("symbols", []):
                symbols.add(s)
    elif isinstance(universes, dict):
        for s in universes.get("symbols", []):
            symbols.add(s)

    bars_list = []

    bars_df = None
    for symbol in symbols:
        try:
            logger.info("Fetching bars for symbol...", symbol=symbol)
            resp = gateway_client.get_alpaca_bars(symbol=symbol, start=start_dt, end=end_dt, timeframe="1Min")
            bars = resp.get("bars", [])
            for b in bars:
                b["symbol"] = symbol
                # Ensure parsing to datetime
                if isinstance(b["t"], str):
                    b["timestamp"] = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
                else:
                    b["timestamp"] = pd.to_datetime(b["t"])
                b["open"] = b["o"]
                b["high"] = b["h"]
                b["low"] = b["l"]
                b["close"] = b["c"]
                b["volume"] = b["v"]
            bars_list.extend(bars)
        except Exception as e:
            logger.error("Failed to fetch historical data for symbol", symbol=symbol, error=str(e))
            continue

    if bars_list:
        bars_df = pd.DataFrame(bars_list)

    if bars_df is None or bars_df.empty:
        logger.warning("No historical data found for the given dates.")
        return

    # Sort by timestamp to replay chronologically across all symbols
    bars_df = bars_df.sort_values(by="timestamp")

    engine = ExecutionEngine(
        config=config,
        logger=logger,
        db=db,
        clock=clock,
    )

    # Setup mock account for equity tracking
    class MockAccount:
        def __init__(self, initial_cash: float):
            self.cash = initial_cash
            self.equity = initial_cash
            self.positions_qty = {}

    engine.account = MockAccount(10000.0)

    # Inject our mock executor
    executor = SimulatedOrderExecutor(
        logger=logger, db=db, clock=clock, on_trade_update=engine.on_trade_update, account=engine.account
    )
    # Wire advanced exits config so executor delegates stop/TP to PositionManager
    # instead of creating broker-managed OCO bracket orders
    risk_cfg = config.get("risk_management", {})
    executor.configure_advanced_exits(risk_cfg)
    engine.order_executor = executor

    # Register strategies
    from src.main import _build_strategy_registry

    registry = _build_strategy_registry()
    strategies_cfg = config.get("strategies", {})
    for name, strat_cfg in strategies_cfg.items():
        if isinstance(strat_cfg, dict) and strat_cfg.get("enabled", True):
            cls = registry.get(name)
            if cls:
                engine.register_strategy(cls(strat_cfg, logger))

    logger.info("Starting backtest replay...")

    prev_day = None
    bar_count = 0

    for _idx, row in bars_df.iterrows():
        ts = row["timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        clock.set_time(ts)

        current_day = ts.date()
        if prev_day and current_day != prev_day:
            engine.flatten_all(reason="EOD Backtest Simulation")
        prev_day = current_day

        mock_bar = Bar(
            symbol=row["symbol"],
            time=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            vwap=float(row.get("vwap", row["close"])),
        )

        # Track the latest price for equity calculation
        if not hasattr(engine, "_latest_prices"):
            engine._latest_prices = {}
        engine._latest_prices[mock_bar.symbol] = mock_bar.close

        # Calculate updated equity
        pos_value = sum(
            qty * engine._latest_prices.get(sym, 0.0) for sym, qty in engine.account.positions_qty.items() if qty != 0
        )
        engine.account.equity = engine.account.cash + pos_value

        # 1. Evaluate fills sequentially
        executor.process_bar(mock_bar)

        import asyncio

        await asyncio.sleep(0)

        # 2. Feed to execution engine
        engine.on_bar(mock_bar.symbol, mock_bar)

        bar_count += 1
        # Occasionally let longer background tasks fully flush
        if bar_count % 1000 == 0:
            await asyncio.sleep(0.001)

        if bar_count % 10000 == 0:
            logger.info("Replay progress", bars_processed=bar_count, current_time=str(ts))

    # Final flatten
    engine.flatten_all(reason="End of Backtest")

    logger.info("Backtest replay complete. Generating report...")

    with db.get_session() as session:
        fills = session.query(DbOrder).filter(DbOrder.status == "filled").all()
        logger.info(f"Total simulated fills: {len(fills)}")

        logger.info(f"Final Account Equity: ${engine.account.equity:.2f}")
        logger.info(f"Net Profit: ${engine.account.equity - 10000.0:.2f}")

        positions = session.query(DbOrder).filter(DbOrder.status.in_(["filled"])).all()
        # Compute some basic PnL based on DB
        pnl = 0.0
        for p in positions:
            pnl_impact = p.qty * p.limit_price if p.limit_price else 0
            if p.side == "buy":
                pnl -= pnl_impact
            else:
                pnl += pnl_impact

        logger.info(f"DB Realized PnL (rough estimate): ${pnl:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    asyncio.run(run_backtest(args.start, args.end, args.config))

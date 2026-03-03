import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Order as DbOrder
from src.backtest.backtest_report import BacktestReportCard, TradeRecord
from src.backtest.executor import SimulatedOrderExecutor
from src.core.config import ConfigLoader
from src.core.domain import Bar, OrderIntent, OrderSide, OrderType
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


def _load_bars_from_parquet(
    data_dir: Path,
    symbols: set[str],
    start_dt: datetime,
    end_dt: datetime,
    logger: StructuredLogger,
) -> pd.DataFrame:
    """Load 1-min bars from per-symbol parquet files in *data_dir*.

    Each file is expected to be ``{SYMBOL}_1Min.parquet`` with columns:
    ``[close, high, low, trade_count, open, timestamp, volume, vwap, symbol]``.

    Filters to the requested date range and returns a combined DataFrame
    with the same columns the runner expects (``o/h/l/c/v`` aliases plus
    ``timestamp`` and ``symbol``).
    """
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for symbol in sorted(symbols):
        parquet_path = data_dir / f"{symbol}_1Min.parquet"
        if not parquet_path.exists():
            missing.append(symbol)
            continue

        df = pd.read_parquet(parquet_path)

        # Ensure timezone-aware timestamps for filtering
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")

        # Filter to requested date range
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

        # Filter to Regular Trading Hours only (9:30 AM – 4:00 PM ET)
        et_times = df["timestamp"].dt.tz_convert("America/New_York")
        rth_mask = (et_times.dt.time >= pd.Timestamp("09:30").time()) & (
            et_times.dt.time < pd.Timestamp("16:00").time()
        )
        pre_filter_len = len(df)
        df = df[rth_mask]
        extended_dropped = pre_filter_len - len(df)

        if df.empty:
            logger.warning(
                "Parquet file has no bars in requested range",
                symbol=symbol,
                path=str(parquet_path),
            )
            continue

        # Ensure symbol column is set
        df["symbol"] = symbol

        frames.append(df)
        logger.info(
            "Loaded bars from parquet",
            symbol=symbol,
            bars=len(df),
            extended_hours_dropped=extended_dropped,
        )

    if missing:
        logger.warning(
            "Parquet files missing for symbols — these will be skipped",
            missing_symbols=missing,
        )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined


def _validate_survivorship(
    bars_df: pd.DataFrame,
    symbols: set[str],
    start_dt: datetime,
    end_dt: datetime,
    logger: StructuredLogger,
) -> None:
    """Log warnings for symbols with suspiciously few bars (survivorship bias check).

    A typical equity trades ~390 1-min bars per day.  We warn when a symbol
    has fewer than 50 % of the expected bar count for the date range.
    """
    if bars_df.empty:
        return

    trading_days = pd.bdate_range(start_dt, end_dt).size
    expected_bars = trading_days * 390  # ~6.5h × 60 min

    bar_counts = bars_df.groupby("symbol").size()
    for symbol in sorted(symbols):
        count = bar_counts.get(symbol, 0)
        if count == 0:
            logger.warning("No bars at all for symbol — possible survivorship issue", symbol=symbol)
        elif count < expected_bars * 0.50:
            logger.warning(
                "Symbol has significantly fewer bars than expected — possible data gap",
                symbol=symbol,
                actual_bars=int(count),
                expected_bars=int(expected_bars),
                coverage_pct=round(count / expected_bars * 100, 1),
            )


def _build_trade_records(db: DatabaseDatabase) -> list[TradeRecord]:
    """Pair filled buy/sell orders into TradeRecord objects for reporting."""
    # Extract all data inside the session to avoid DetachedInstanceError
    with db.get_session() as session:
        fills = session.query(DbOrder).filter(DbOrder.status == "filled").order_by(DbOrder.time_placed).all()
        # Materialize to plain dicts while session is open
        fill_dicts = [
            {
                "symbol": f.symbol,
                "side": f.side,
                "qty": f.qty,
                "limit_price": f.limit_price,
                "time_placed": f.time_placed,
            }
            for f in fills
        ]

    # Group by symbol, then pair buys and sells chronologically
    by_symbol: dict[str, dict[str, list]] = {}
    for f in fill_dicts:
        sym = f["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = {"buy": [], "sell": []}
        by_symbol[sym][f["side"]].append(f)

    trades: list[TradeRecord] = []
    for sym, sides in by_symbol.items():
        buys = sides["buy"]
        sells = sides["sell"]
        n_pairs = min(len(buys), len(sells))

        for i in range(n_pairs):
            b, s = buys[i], sells[i]
            # Determine which was entry vs exit by time
            if b["time_placed"] <= s["time_placed"]:
                # Long trade: bought first, sold later
                entry_price = b["limit_price"] or 0.0
                exit_price = s["limit_price"] or 0.0
                pnl = (exit_price - entry_price) * b["qty"]
                entry_time = b["time_placed"]
                exit_time = s["time_placed"]
                side = "buy"
                qty = b["qty"]
            else:
                # Short trade: sold first, bought later
                entry_price = s["limit_price"] or 0.0
                exit_price = b["limit_price"] or 0.0
                pnl = (entry_price - exit_price) * s["qty"]
                entry_time = s["time_placed"]
                exit_time = b["time_placed"]
                side = "sell"
                qty = s["qty"]

            trades.append(
                TradeRecord(
                    symbol=sym,
                    side=side,
                    qty=qty,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    pnl=pnl,
                )
            )

    return trades


async def run_backtest(start_date: str, end_date: str, config_path: str, data_dir: str | None = None):
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

    # Collect universe symbols
    symbols: set[str] = set()
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

    # ── Load bar data ────────────────────────────────────────────────
    logger.info("Fetching historical data for backtest...", start_date=start_date, end_date=end_date)

    if data_dir:
        data_path = Path(data_dir)
        if not data_path.is_dir():
            logger.error("--data-dir does not exist", path=data_dir)
            return
        logger.info("Loading bars from local parquet cache", data_dir=data_dir)
        bars_df = _load_bars_from_parquet(data_path, symbols, start_dt, end_dt, logger)
    else:
        # Fetch from API (legacy path)
        gateway_client = CentralApiClient(config_loader, logger)
        bars_list: list[dict] = []
        for symbol in symbols:
            try:
                logger.info("Fetching bars for symbol...", symbol=symbol)
                resp = gateway_client.get_alpaca_bars(symbol=symbol, start=start_dt, end=end_dt, timeframe="1Min")
                bars = resp.get("bars", [])
                for b in bars:
                    b["symbol"] = symbol
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
        bars_df = pd.DataFrame(bars_list) if bars_list else pd.DataFrame()

    if bars_df.empty:
        logger.warning("No historical data found for the given dates.")
        return

    # Survivorship bias check
    _validate_survivorship(bars_df, symbols, start_dt, end_dt, logger)

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
            self.positions_qty: dict[str, float] = {}

    engine.account = MockAccount(10000.0)

    # Inject our mock executor with backtest config for slippage
    backtest_cfg = config.get("backtest", {})
    risk_cfg = config.get("risk_management", {})
    executor = SimulatedOrderExecutor(
        logger=logger,
        db=db,
        clock=clock,
        on_trade_update=engine.on_trade_update,
        account=engine.account,
        backtest_cfg=backtest_cfg,
        risk_cfg=risk_cfg,
    )
    # Wire advanced exits config so executor delegates stop/TP to PositionManager
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

    # ── Backtest-aware flatten ───────────────────────────────────────
    def _backtest_flatten_all(reason: str) -> None:
        """Close all open positions and cancel pending orders in simulation.

        Unlike engine.flatten_all() which requires an Alpaca client, this
        directly interacts with the SimulatedOrderExecutor to close positions
        at the last known price.
        """
        # 1. Cancel all pending orders
        cancelled = 0
        for symbol in list(engine.symbol_states.keys()):
            cancelled += executor.cancel_all_for_symbol(symbol)

        # 2. Submit market exits for any open positions
        closed = 0
        for symbol, state in engine.symbol_states.items():
            if state.position is None or state.position.qty == 0:
                continue

            exit_side = OrderSide.SELL if state.position.side.value == "long" else OrderSide.BUY
            exit_qty = abs(state.position.qty)

            exit_intent = OrderIntent(
                symbol=symbol,
                side=exit_side,
                qty=exit_qty,
                order_type=OrderType.MARKET,
                limit_price=None,
                time_in_force="day",
                correlation_id=state.position.correlation_id or "",
                strategy=state.position.strategy or "flatten",
                stop_loss=None,
                take_profit=None,
                meta={"reason": reason},
            )
            executor.submit(exit_intent)

            # Process the fill immediately using the last known price
            last_price = getattr(engine, "_latest_prices", {}).get(symbol, 0.0)
            if last_price > 0:
                mock_fill_bar = Bar(
                    symbol=symbol,
                    time=clock(),
                    open=last_price,
                    high=last_price,
                    low=last_price,
                    close=last_price,
                    volume=0.0,
                )
                executor.process_bar(mock_fill_bar)
            closed += 1

        # 3. Reset local position/order state
        for state in engine.symbol_states.values():
            state.position = None
            state.open_orders = {}

        logger.info(
            "Backtest flatten complete",
            reason=reason,
            orders_cancelled=cancelled,
            positions_closed=closed,
        )

    logger.info("Starting backtest replay...")

    prev_day = None
    bar_count = 0
    equity_curve: list[tuple[datetime, float]] = []

    for _idx, row in bars_df.iterrows():
        ts = row["timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        clock.set_time(ts)

        current_day = ts.date()
        if prev_day and current_day != prev_day:
            _backtest_flatten_all(reason="EOD Backtest Simulation")
            # Snapshot equity at day boundary for Sharpe/DD calculations
            equity_curve.append((ts, engine.account.equity))
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

        await asyncio.sleep(0)

        # 2. Feed to execution engine
        engine.on_bar(mock_bar.symbol, mock_bar)

        bar_count += 1
        if bar_count % 1000 == 0:
            await asyncio.sleep(0.001)

        if bar_count % 10000 == 0:
            logger.info("Replay progress", bars_processed=bar_count, current_time=str(ts))

    # Final flatten
    _backtest_flatten_all(reason="End of Backtest")
    # Final equity snapshot
    equity_curve.append((clock(), engine.account.equity))

    logger.info("Backtest replay complete. Generating report...")

    # Build trade records from paired buy/sell orders
    trades = _build_trade_records(db)

    report = BacktestReportCard(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=10_000.0,
        start_date=start_dt,
        end_date=end_dt,
    )
    report.print_summary(logger)

    # Save markdown report
    report_name = f"backtest_{start_date}_to_{end_date}.md"
    report_path = report.write_markdown(f"artifacts/backtest_reports/{report_name}")
    logger.info("Report saved", path=str(report_path))

    # Also log as single JSON for machine parsing
    logger.info("Report card JSON", **report.to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory with per-symbol parquet files (e.g. data/bars_2024). Skips API fetch when set.",
    )
    args = parser.parse_args()

    asyncio.run(run_backtest(args.start, args.end, args.config, data_dir=args.data_dir))

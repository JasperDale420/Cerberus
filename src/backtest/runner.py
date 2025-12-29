import asyncio
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.agent.bars_provider import JsonlBarsProvider
from src.backtest.feature_pipeline import BacktestFeaturePipeline
from src.backtest.mock_executor import BacktestOrderExecutor
from src.backtest.stats import BacktestAnalyzer
from src.core.config import ConfigLoader
from src.core.domain import Bar, SymbolState
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.engine.execution import ExecutionEngine
from src.scanner.core import Scanner
from src.scanner.universe import UniverseBuilder
from src.strategies.failed_breakout import FailedBreakoutStrategy
from src.strategies.flow_momentum import FlowMomentumStrategy
from src.strategies.gap_fill import GapFillStrategy
from src.strategies.index_mean_reversion import IndexMeanReversionStrategy
from src.strategies.orb import ORBStrategy
from src.strategies.trend_pullback import TrendPullbackStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.strategies.vwap_trend_rider import VWAPTrendRiderStrategy


class BacktestRunner:
    """
    Runs a backtest by feeding historical data to the ExecutionEngine.
    """

    DEFAULT_TIMEZONE = "US/Eastern"

    def __init__(
        self,
        config_path: str,
        start_date: str,
        end_date: str,
        *,
        offline_bars_dir: Optional[str] = None,
    ):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config(config_path)
        self.logger = StructuredLogger(
            "Backtester", logging_config=self.config.get("logging")
        )

        self.start_date = self._parse_dt(start_date)
        self.end_date = self._parse_dt(end_date)
        self.offline_bars_dir = (
            str(offline_bars_dir).strip() if offline_bars_dir else ""
        )
        self.offline_provider = (
            JsonlBarsProvider(Path(self.offline_bars_dir))
            if self.offline_bars_dir
            else None
        )

        self.alpaca_client = (
            None
            if self.offline_provider is not None
            else AlpacaClient(self.config_loader, self.logger)
        )

        # Mock Executor
        self.mock_executor = BacktestOrderExecutor(
            self.logger,
            initial_cash=float(self.config.get("initial_cash", 100000.0) or 100000.0),
        )
        self.mock_executor.set_risk_config(
            self.config.get("risk")
            if isinstance(self.config.get("risk"), dict)
            else None
        )
        self.mock_executor.set_max_open_order_age_sec(
            self.config.get("max_open_order_age_sec", 0)
        )

        # Engine with Mock Executor
        self.engine = ExecutionEngine(self.config, self.logger, alpaca_client=None)
        self.engine.order_executor = self.mock_executor  # type: ignore[assignment] # Inject mock

        # Track last known prices for Mark-to-Market metrics / equity
        self.last_prices: Dict[str, float] = {}

        # Use UniverseBuilder to determine universe from config
        self.universe_builder = UniverseBuilder(
            config_loader=self.config_loader,
            logger=self.logger,
            config=self.config,
            alpaca_client=self.alpaca_client,
            offline_bars_provider=self.offline_provider,
        )
        self.universe = self.universe_builder.build_universe()

        # Register Strategies (replicates src/main.py logic)
        self._register_strategies()

        # Account equity proxy for risk sizing (engine reads `self.engine.account.equity`)
        class _BacktestAccount:
            def __init__(self, runner: "BacktestRunner"):
                self._runner = runner

            @property
            def equity(self) -> float:
                # Basic equity estimate: cash + mark-to-market of open engine positions.
                eq = float(self._runner.mock_executor.cash)
                for st in self._runner.engine.symbol_states.values():
                    pos = getattr(st, "position", None)
                    if pos is None:
                        continue
                    mark = float(self._runner.last_prices.get(st.symbol, 0.0) or 0.0)
                    if mark <= 0.0:
                        continue
                    qty = float(pos.qty)
                    if pos.side.value == "long":
                        eq += mark * qty
                    else:
                        eq -= mark * qty
                return float(eq)

        self.engine.account = _BacktestAccount(self)  # type: ignore[assignment]

    def _register_strategies(self) -> None:
        """
        Register enabled strategies from config.
        """
        strategy_registry = {
            "vwap_reversion": VWAPReversionStrategy,
            "orb": ORBStrategy,
            "trend_pullback": TrendPullbackStrategy,
            "failed_breakout": FailedBreakoutStrategy,
            "vwap_trend_rider": VWAPTrendRiderStrategy,
            "index_mean_reversion": IndexMeanReversionStrategy,
            "flow_momentum": FlowMomentumStrategy,
            "gap_fill": GapFillStrategy,
        }

        strategies_cfg = self.config.get("strategies", {})
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
                self.logger.warning(
                    "Unknown strategy in config; skipping", strategy=str(name)
                )
                continue
            strategy_instance = cls(strat_cfg, self.logger)  # type: ignore[abstract]
            self.engine.register_strategy(strategy_instance)
            # Track registered strategies to enable them for all symbols in backtest
            if not hasattr(self, "enabled_strategies"):
                self.enabled_strategies = []
            self.enabled_strategies.append(str(name))

    def _parse_dt(self, value: str) -> datetime:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            # Default to UTC; bars are typically UTC from Alpaca.
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _parse_single_bar(self, b: Dict[str, Any], symbol: str) -> Optional[Bar]:
        """Parse a single bar dictionary into a Bar object."""
        # Handle different formats (raw vs parsed)
        t = b.get("t") or b.get("timestamp")
        o = b.get("o") or b.get("open")
        h = b.get("h") or b.get("high")
        low_price = b.get("l") or b.get("low")
        c = b.get("c") or b.get("close")
        v = b.get("v") or b.get("volume")

        if not t:
            return None

        return Bar(
            symbol=symbol,
            time=(
                datetime.fromisoformat(str(t).replace("Z", "+00:00"))
                if isinstance(t, str)
                else t
            ),
            open=float(o or 0.0),
            high=float(h or 0.0),
            low=float(low_price or 0.0),
            close=float(c or 0.0),
            volume=float(v or 0.0),
        )

    def _parse_bars(self, bars_data: Any, symbol: str) -> List[Bar]:
        bars: List[Bar] = []
        if isinstance(bars_data, list):
            for b in bars_data:
                bar = self._parse_single_bar(b, symbol)
                if bar:
                    bars.append(bar)
        return bars

    async def _load_bars_for_symbol(self, symbol: str, timeframe: str) -> List[Bar]:
        if self.offline_provider is not None:
            return list(
                self.offline_provider.get_bars(
                    symbol, self.start_date, self.end_date, timeframe=timeframe
                )
            )
        if self.alpaca_client is None:
            return []

        bars_data = await asyncio.to_thread(
            self.alpaca_client.get_historical_bars,
            symbol,
            self.start_date,
            self.end_date,
            timeframe,
        )
        if isinstance(bars_data, dict) and "bars" in bars_data:
            bars_data = bars_data["bars"]
        return self._parse_bars(bars_data, symbol)

    def _build_event_stream(
        self, bars_by_symbol: Dict[str, List[Bar]]
    ) -> List[Tuple[datetime, str, Bar]]:
        out: List[Tuple[datetime, str, Bar]] = []
        for sym, bars in bars_by_symbol.items():
            for b in bars:
                bt = b.time
                if bt.tzinfo is None:
                    bt = bt.replace(tzinfo=timezone.utc)
                    b.time = bt
                out.append((bt, sym, b))

        # Deterministic ordering: time, then index first, then symbol.
        index_symbol = str(self.config.get("index_symbol", "SPY") or "SPY").upper()
        out.sort(key=lambda x: (x[0], 0 if x[1] == index_symbol else 1, x[1]))
        return out

    def _scanner_enabled(self) -> bool:
        scanner_cfg = (
            (self.config.get("scanner") or {}) if isinstance(self.config, dict) else {}
        )
        if not isinstance(scanner_cfg, dict):
            return False
        # Default to disabled unless explicitly enabled.
        return bool(scanner_cfg.get("enabled", False))

    def _scanner_interval_minutes(self) -> int:
        scanner_cfg = (
            (self.config.get("scanner") or {}) if isinstance(self.config, dict) else {}
        )
        if not isinstance(scanner_cfg, dict):
            return 0
        try:
            return int(scanner_cfg.get("interval_minutes", 0) or 0)
        except Exception:
            return 0

    def _ceil_time_to_interval(
        self, t: datetime, minutes: int, tz: ZoneInfo
    ) -> datetime:
        if minutes <= 0:
            return t
        local = t.astimezone(tz).replace(second=0, microsecond=0)
        mod = int(local.minute) % int(minutes)
        if mod == 0:
            return local.astimezone(timezone.utc)
        delta = int(minutes) - mod
        return (local + timedelta(minutes=delta)).astimezone(timezone.utc)

    def _flatten_session_end(self, *, ts: datetime, reason: str) -> None:
        self.mock_executor.cancel_all_orders()
        self.mock_executor.close_all_positions(
            self.engine,
            timestamp=ts,
            prices=dict(self.last_prices),
            reason=reason,
        )

    def _initialize_symbol_states(
        self, index_symbol: str, scanner_enabled: bool
    ) -> None:
        self.logger.info("Initializing Symbol States", count=len(self.universe))
        if index_symbol and index_symbol not in self.engine.symbol_states:
            self.engine.symbol_states[index_symbol] = SymbolState(
                symbol=index_symbol,
                bars=deque(maxlen=100),
                position=None,
                indicators={},
                open_orders={},
                allowed_strategies=[],
                meta={},
            )

        if not scanner_enabled:
            for symbol in self.universe:
                # Pre-initialize symbol state to ensure allowed_strategies are set
                # (no scanner gating in this mode).
                if symbol not in self.engine.symbol_states:
                    self.engine.symbol_states[symbol] = SymbolState(
                        symbol=symbol,
                        bars=deque(maxlen=100),
                        position=None,
                        indicators={},
                        open_orders={},
                        allowed_strategies=getattr(self, "enabled_strategies", []),
                        meta={},
                    )

    async def _load_all_bars(self, timeframe: str) -> Dict[str, List[Bar]]:
        async def _load_one(symbol: str) -> tuple[str, List[Bar]]:
            self.logger.info(
                "Fetching data",
                symbol=symbol,
                timeframe=timeframe,
                offline=bool(self.offline_provider),
            )
            bars = await self._load_bars_for_symbol(symbol, timeframe)
            self.logger.info("Loaded bars", symbol=symbol, count=len(bars))
            return symbol, bars

        results = await asyncio.gather(*[_load_one(s) for s in self.universe])
        return {sym: bars for sym, bars in results}

    def _setup_scanner_replay(self, bars_by_symbol: Dict[str, List[Bar]]) -> None:
        pipeline = BacktestFeaturePipeline(
            bars_by_symbol,
            self.logger,
            config=self.config if isinstance(self.config, dict) else {},
            clock=lambda: self.engine.market_state.time,
        )
        self.engine.scanner = Scanner(
            universe_builder=self.universe_builder,
            feature_pipeline=pipeline,  # type: ignore[arg-type]
            logger=self.logger,
            config=self.config if isinstance(self.config, dict) else {},
        )

    def _format_single_trade(self, t: Any) -> Dict[str, Any]:
        def _dt(v: Any) -> Optional[str]:
            if isinstance(v, datetime):
                return v.isoformat()
            return None

        return {
            "symbol": getattr(t, "symbol", None),
            "strategy": getattr(t, "strategy", None),
            "regime_at_entry": getattr(t, "regime_at_entry", None),
            "regime_at_exit": getattr(t, "regime_at_exit", None),
            "side": getattr(t, "side", None),
            "qty": float(getattr(t, "qty", 0.0) or 0.0),
            "entry_time": _dt(getattr(t, "entry_time", None)),
            "exit_time": _dt(getattr(t, "exit_time", None)),
            "entry_price": float(getattr(t, "entry_price", 0.0) or 0.0),
            "exit_price": float(getattr(t, "exit_price", 0.0) or 0.0),
            "pnl_gross": float(getattr(t, "pnl_gross", 0.0) or 0.0),
            "pnl_net": float(getattr(t, "pnl_net", 0.0) or 0.0),
            "commission": float(getattr(t, "commission", 0.0) or 0.0),
            "slippage_estimate": float(getattr(t, "slippage_estimate", 0.0) or 0.0),
            "pnl_r": (
                float(getattr(t, "pnl_r", 0.0) or 0.0)
                if getattr(t, "pnl_r", None) is not None
                else None
            ),
            "holding_period_seconds": (
                float(getattr(t, "holding_period_seconds", 0.0) or 0.0)
                if getattr(t, "holding_period_seconds", None) is not None
                else None
            ),
            "correlation_id": getattr(t, "correlation_id", None),
        }

    def _format_trades(self, raw_trades: List[Any]) -> List[Dict[str, Any]]:
        raw_trades.sort(key=lambda t: getattr(t, "exit_time", None) or "")
        return [self._format_single_trade(t) for t in raw_trades]

    def _calculate_metrics(
        self, engine_trades: List[Dict[str, Any]], initial_cash: float
    ) -> Dict[str, Any]:
        pnls_net = [float(t.get("pnl_net", 0.0) or 0.0) for t in engine_trades]
        wins = [p for p in pnls_net if p > 0]
        losses = [p for p in pnls_net if p <= 0]
        gross_profit = float(sum(wins))
        gross_loss = float(abs(sum(losses)))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        equity = float(initial_cash)
        peak = equity
        max_dd = 0.0
        for p in pnls_net:
            equity += float(p)
            if equity > peak:
                peak = equity
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)

        engine_equity = None
        try:
            engine_equity = float(getattr(self.engine.account, "equity", 0.0) or 0.0)
        except Exception:
            engine_equity = None

        return {
            "total_trades": int(len(engine_trades)),
            "total_closed_pnl_gross": round(
                float(
                    sum(float(t.get("pnl_gross", 0.0) or 0.0) for t in engine_trades)
                ),
                2,
            ),
            "total_closed_pnl_net": round(float(sum(pnls_net)), 2),
            "win_rate": round((len(wins) / len(pnls_net)) if pnls_net else 0.0, 4),
            "profit_factor": round(float(profit_factor), 4),
            "average_pnl_net": round(
                (float(sum(pnls_net)) / len(pnls_net)) if pnls_net else 0.0, 2
            ),
            "max_drawdown_pct": round(float(max_dd) * 100.0, 2),
            "final_equity": (
                round(float(engine_equity), 2) if engine_equity is not None else None
            ),
        }

    def _analyze_results(self) -> Dict[str, Any]:
        analyzer = BacktestAnalyzer(
            initial_cash=self.config.get("initial_cash", 100000.0)
        )
        fills_metrics = analyzer.calculate_statistics(
            self.mock_executor.fills, current_prices=self.last_prices
        )

        engine_trades_raw = list(getattr(self.engine, "closed_trades", []) or [])
        engine_trades = self._format_trades(engine_trades_raw)
        engine_metrics = self._calculate_metrics(
            engine_trades, float(analyzer.initial_cash)
        )

        self.logger.info("Final Cash", cash=self.mock_executor.cash)
        self.logger.info("Total Trades", count=engine_metrics["total_trades"])
        self.logger.info("Total PnL (net)", pnl=engine_metrics["total_closed_pnl_net"])
        self.logger.info("Win Rate", rate=engine_metrics["win_rate"])

        engine_realized_pnl = 0.0
        try:
            engine_realized_pnl = float(
                getattr(
                    getattr(self.engine, "risk_manager", None), "current_daily_pnl", 0.0
                )
                or 0.0
            )
        except Exception:
            engine_realized_pnl = 0.0

        return {
            "config": self.config,
            "start": self.start_date.isoformat(),
            "end": self.end_date.isoformat(),
            "initial_cash": analyzer.initial_cash,
            "final_cash_balance": round(self.mock_executor.cash, 2),
            "final_equity": engine_metrics.get("final_equity"),
            "engine_realized_pnl": engine_realized_pnl,
            "metrics": engine_metrics,
            "engine_trades": engine_trades,
            "metrics_fills": fills_metrics,
        }

    def _handle_session_boundary(
        self,
        bt: datetime,
        market_tz: ZoneInfo,
        last_session_ts: Optional[datetime],
        current_session: Optional[date],
    ) -> Tuple[Optional[date], Optional[datetime]]:
        local_date = bt.astimezone(market_tz).date()
        if current_session is None:
            return local_date, bt

        if local_date != current_session:
            self.logger.info(
                "Session boundary flatten",
                session_date=str(current_session),
                timestamp=(last_session_ts.isoformat() if last_session_ts else None),
            )
            self._flatten_session_end(
                ts=(last_session_ts if last_session_ts is not None else bt),
                reason="SESSION_END",
            )
            return local_date, bt

        return current_session, bt

    async def _handle_scanner_replay(
        self,
        bt: datetime,
        market_tz: ZoneInfo,
        scan_interval: int,
        last_scan_ts: Optional[datetime],
        next_scan_ts: Optional[datetime],
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        if next_scan_ts is None:
            next_scan_ts = self._ceil_time_to_interval(bt, scan_interval, market_tz)

        if bt >= next_scan_ts and (last_scan_ts is None or bt != last_scan_ts):
            try:
                await self.engine.run_scan()
            except Exception as e:
                self.logger.error("Backtest scan failed", error=str(e))
            last_scan_ts = bt
            next_scan_ts = self._ceil_time_to_interval(
                bt + timedelta(minutes=scan_interval), scan_interval, market_tz
            )
        return last_scan_ts, next_scan_ts

    async def _process_loop_event(
        self,
        bt: datetime,
        symbol: str,
        bar: Bar,
        market_tz: ZoneInfo,
        scanner_replay: bool,
        scan_interval: int,
        last_session_ts: Optional[datetime],
        current_session: Optional[date],
        last_scan_ts: Optional[datetime],
        next_scan_ts: Optional[datetime],
        index_symbol: str,
    ) -> Tuple[
        Optional[datetime], Optional[date], Optional[datetime], Optional[datetime]
    ]:
        current_session, last_session_ts = self._handle_session_boundary(
            bt, market_tz, last_session_ts, current_session
        )

        # Ensure engine time advances deterministically even if index bars are absent.
        try:
            self.engine.market_state.time = bt
        except Exception:
            pass

        if scanner_replay:
            last_scan_ts, next_scan_ts = await self._handle_scanner_replay(
                bt, market_tz, scan_interval, last_scan_ts, next_scan_ts
            )

        # Mimic WS subscriptions: only process non-index bars when symbol is tracked.
        if symbol != index_symbol and symbol not in self.engine.symbol_states:
            return last_session_ts, current_session, last_scan_ts, next_scan_ts

        # Mark-to-market reference price.
        self.last_prices[symbol] = float(bar.close)

        # 1) Fill pending orders for this symbol (market/limit, deterministic).
        self.mock_executor.fill_pending_for_bar(self.engine, symbol, bar)

        # 2) Broker-managed bracket exits (stop/target) using intrabar extremes.
        self.mock_executor.maybe_trigger_bracket_exit(self.engine, symbol, bar)

        # 3) Strategy + risk + order generation.
        self.engine.on_bar(symbol, bar)

        return last_session_ts, current_session, last_scan_ts, next_scan_ts

    async def run(self):
        self.logger.info("Starting Backtest", start=self.start_date, end=self.end_date)

        index_symbol = str(self.config.get("index_symbol", "SPY") or "SPY").upper()
        scanner_enabled = self._scanner_enabled()

        self._initialize_symbol_states(index_symbol, scanner_enabled)

        timeframe = str(self.config.get("timeframe", "1Min") or "1Min")
        bars_by_symbol = await self._load_all_bars(timeframe)

        events = self._build_event_stream(bars_by_symbol)
        self.logger.info("Built event stream", events=len(events))

        tz_name = str(
            self.config.get("timezone", self.DEFAULT_TIMEZONE) or self.DEFAULT_TIMEZONE
        )
        try:
            market_tz = ZoneInfo(tz_name)
        except Exception:
            market_tz = ZoneInfo(self.DEFAULT_TIMEZONE)

        scan_interval = self._scanner_interval_minutes()
        scanner_replay = scanner_enabled and scan_interval > 0
        if scanner_replay:
            self._setup_scanner_replay(bars_by_symbol)

        current_session: Optional[date] = None
        last_session_ts: Optional[datetime] = None
        last_scan_ts: Optional[datetime] = None
        next_scan_ts: Optional[datetime] = None

        # Replay loop (chronological across all symbols)
        for bt, symbol, bar in events:
            (
                last_session_ts,
                current_session,
                last_scan_ts,
                next_scan_ts,
            ) = await self._process_loop_event(
                bt,
                symbol,
                bar,
                market_tz,
                scanner_replay,
                scan_interval,
                last_session_ts,
                current_session,
                last_scan_ts,
                next_scan_ts,
                index_symbol,
            )

        if last_session_ts is not None:
            self._flatten_session_end(ts=last_session_ts, reason="BACKTEST_END")

        # Report
        self.logger.info("Backtest Complete")

        # Analyze Results
        return self._analyze_results()

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.config import ConfigLoader
from src.core.domain import Bar, Regime, ScanResult, SymbolFeatures, WatchlistSymbol
from src.core.logger import StructuredLogger
from src.engine.execution import ExecutionEngine
from src.scanner.core import Scanner
from src.strategies.gap_fill import GapFillStrategy


@pytest.fixture
def mock_logger():
    return MagicMock(spec=StructuredLogger)


@pytest.fixture
def mock_config_loader():
    loader = MagicMock(spec=ConfigLoader)
    # Return a config dict that enables GapFill
    loader.load_config.return_value = {
        "strategies": {
            "gap_fill": {
                "enabled": True,
                "min_gap": 0.01,
                "risk_reward": 0.1,
                "or_time_minutes": 5,
            }
        },
        "risk": {
            "max_daily_loss": 500.0,
            "max_risk_per_trade": 50.0,
            "max_open_risk": 200.0,
            "max_trades_per_day": 20,
            "risk_mode": "normal",
        },
        "universe": {"symbols": ["TEST"]},
        "scanner": {"enabled": True},
    }
    return loader


@pytest.mark.asyncio
async def test_vertical_slice_scanner_to_execution(mock_config_loader, mock_logger):
    """
    Vertical Slice 3: Scanner -> Execution -> Strategy -> Signal
    Verifies that a scanned symbol with a gap correctly triggers the GapFillStrategy
    in the ExecutionEngine.
    """

    # 1. Setup Mocks
    mock_alpaca = MagicMock()
    mock_alpaca.trading_client = MagicMock()
    mock_alpaca.get_account = MagicMock(
        return_value=MagicMock(equity=100000, currency="USD")
    )

    # Mock FeaturePipeline to return a "Gap Up" feature set
    # Gap Up: Prev Close=100, Open=102 (2% Gap)
    mock_pipeline = AsyncMock()

    # Scanner Setup
    mock_universe = MagicMock()
    # Scanner init: universe_builder, feature_pipeline, logger
    scanner = Scanner(mock_universe, mock_pipeline, mock_logger)
    # scanner.alpaca_client injection removed as attribute doesn't exist
    # Note: Scanner doesn't take config_loader in init anymore?
    # Let's check config. It seems it uses profiles hardcoded or from internal config?
    # The code viewed shows profiles initialized without config arg.
    # So we don't need config_loader for Scanner init.

    # Mock Scan Result
    # We bypass the actual scan logic to isolate Engine+Strategy
    # But for a true vertical slice, we should trust Scanner to produce it.
    # Use 13:30 UTC = 09:30 ET for Market Open
    now = datetime(2023, 10, 27, 13, 30, 0, tzinfo=timezone.utc)

    # Create real objects to avoid Mock attribute errors
    features = SymbolFeatures(
        symbol="TEST",
        price=102.0,
        gap_pct=0.02,
        atr_pct=0.01,
        avg_volume=1000000,
        intraday_range_pct=0.01,
        ema20_slope=0.1,
        ema_trend_strength=0.5,
        distance_from_vwap=0.0,
        premarket_volume=10000,
        adx=30,
        distance_from_ema20=0.0,
        prior_day_high=100,
        prior_day_low=99,
        bb_upper=103,
        bb_lower=97,
        price_zscore=1.5,
        flow_zscore=0.5,
        call_put_ratio=1.0,
        large_sweeps_count=0,
        aggressive_flow_share=0.2,
        last_updated=now,
    )

    wl_sym = WatchlistSymbol(
        symbol="TEST", score=0.9, strategies=["gap_fill"], features=features
    )

    scan_result = ScanResult(generated_at=now, regime=Regime.BULL, watchlist=[wl_sym])
    scanner.scan = AsyncMock(return_value=scan_result)  # type: ignore[method-assign]

    # Engine Setup
    # ExecutionEngine expects a config DICT, not a Loader object.
    config_dict = mock_config_loader.load_config.return_value
    # Pass alpaca_client to init so OrderExecutor is created
    engine = ExecutionEngine(config_dict, mock_logger, alpaca_client=mock_alpaca)

    # Mock RiskManager to ensure signal acceptance not blocked by PnL/limits
    engine.risk_manager = MagicMock()
    engine.risk_manager.apply.return_value = "DUMMY_INTENT"

    # Manually Register Strategy (Engine relies on main.py for this)
    gap_config = config_dict["strategies"]["gap_fill"]
    # Ensure config has what strategy needs (defaults usually, but good to be explicit in mock fixture)
    strat = GapFillStrategy(gap_config, mock_logger)
    engine.register_strategy(strat)

    # 2. Run Scan & Apply to Engine
    result = await scanner.scan(regime=Regime.BULL)
    assert len(result.watchlist) == 1
    assert result.watchlist[0].symbol == "TEST"

    engine.apply_scan_result(result.watchlist)

    # Verify SymbolState is initialized
    assert "TEST" in engine.symbol_states
    state = engine.symbol_states["TEST"]
    assert state.meta["gap_pct"] == pytest.approx(0.02)

    # 3. Simulate Market Data (Bars)
    # Scenario: Gap Up (Open 102), then Opening Range (OR) forms.
    # OR High = 102.5, OR Low = 101.5.
    # Price breaks OR Low -> Trigger Short.

    start_time = now

    # Bar 1: 09:30-09:35 (Formation of OR) - Open 102, High 102.5, Low 101.8, Close 102.0
    bar1 = Bar(
        symbol="TEST",
        time=start_time,
        open=102.0,
        high=102.5,
        low=101.8,
        close=102.0,
        volume=1000,
        vwap=102.1,
        trade_count=50,
    )

    # Bar 2: 09:35-09:40 (Still OR? Config says 5m OR. So Bar 1 IS the OR).
    # Bar 2 triggers breakdown.
    # Open 102.0, High 102.1, Low 101.0 (Break 101.8!), Close 101.2
    bar2_time = start_time + timedelta(minutes=5)
    bar2 = Bar(
        symbol="TEST",
        time=bar2_time,
        open=102.0,
        high=102.1,
        low=101.0,
        close=101.2,
        volume=2000,
        vwap=101.5,
        trade_count=100,
    )

    # Inject Bar 1
    engine.on_bar("TEST", bar1)

    # Verify State: OR should be forming or formed.
    # Strategy logic: checks bar time vs cutoff.
    # If or_time_minutes=5, cutoff is 09:35.
    # Bar 1 time is 09:30. 09:30 < 09:35. So it waits?
    # Actually logic says: if current_time <= cutoff: return None.
    # So Bar 1 (09:30) returns None.

    # Inject Bar 2
    # Time 09:35. 09:35 <= 09:35. Returns None?
    # Correct. It builds the OR.
    # We need a bar AFTER the cutoff to trigger.
    # Inject Bar 2
    # Time 09:35. 09:35 <= 09:35. Returns None?
    # Correct. It builds the OR.
    # We need a bar AFTER the cutoff to trigger.
    engine.on_bar("TEST", bar2)

    # Bar 3: 09:40. Breakout confirmed.
    bar3_time = start_time + timedelta(minutes=10)
    bar3 = Bar(
        symbol="TEST",
        time=bar3_time,
        open=101.2,
        high=101.3,
        low=100.5,
        close=100.8,
        volume=1500,
        vwap=101.0,
        trade_count=80,
    )

    # Capture generated signals
    # We can mock the OrderExecutor.submit to verify signal reached execution
    # Note: method name is 'submit' in execution.py
    assert engine.order_executor is not None
    # ExecutionEngine calls submit() synchronously, so use a normal mock to avoid
    # "coroutine was never awaited" warnings.
    engine.order_executor.submit = MagicMock()  # type: ignore[method-assign]

    engine.on_bar("TEST", bar3)

    # 4. Verify Signal & Order
    # Signal should have been generated on Bar 3 (or Bar 2 depending on logical exactness of "after").
    # If Bar 2 closed the OR, Bar 3 is first tradeable bar.
    # Bar 2 low (101.0) < OR Low (101.8 from Bar 1).
    # Logic: if bar.close < or_low. Bar 3 Close 100.8 < 101.8. YES.

    # Check submit call
    assert engine.order_executor.submit.called
    _ = engine.order_executor.submit.call_args[0][0]  # First arg is intent

    # Wait, we intercepted at Risk Manager.
    # The Signal is inside the call `risk_manager.apply(signal, ...)`
    # We should inspect the CALL to RiskManager to verify Signal content.

    assert engine.risk_manager.apply.called
    risk_args = engine.risk_manager.apply.call_args
    signal = risk_args[0][0]

    assert signal.symbol == "TEST"
    assert signal.strategy == "gap_fill"
    assert signal.side.value == "sell"
    assert signal.meta["gap_pct"] == pytest.approx(0.02)

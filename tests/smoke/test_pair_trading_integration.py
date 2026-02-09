from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.core.domain import (
    OrderSide,
    Regime,
    ScanResult,
    SymbolFeatures,
    WatchlistSymbol,
)
from src.engine.execution import ExecutionEngine


def create_dummy_features(symbol, price, scan_time):
    return SymbolFeatures(
        symbol=symbol,
        price=price,
        atr_pct=0.01,
        avg_volume=1000000,
        intraday_range_pct=0.02,
        gap_pct=0.0,
        ema20_slope=0.0,
        ema_trend_strength=0.0,
        distance_from_vwap=0.0,
        premarket_volume=0.0,
        adx=25.0,
        distance_from_ema20=0.0,
        prior_day_high=price * 1.01,
        prior_day_low=price * 0.99,
        bb_upper=price * 1.02,
        bb_lower=price * 0.98,
        price_zscore=0.0,
        flow_zscore=0.0,
        call_put_ratio=1.0,
        large_sweeps_count=0,
        aggressive_flow_share=0.0,
        last_updated=scan_time,
    )


@pytest.fixture
def engine():
    config = {
        "risk": {"max_risk_per_trade": 500.0, "pair_trading": {"enabled": True}},
        "initial_cash": 100000.0,
    }
    logger = MagicMock()
    db = MagicMock()
    alpaca = MagicMock()

    eng = ExecutionEngine(config, logger, db, alpaca)
    eng.account = MagicMock()
    eng.account.equity = 100000.0
    return eng


@pytest.mark.asyncio
async def test_pair_signal_sizing(engine):
    # 1. Setup mock scan result with pair metadata
    s1, s2 = "AAPL", "MSFT"
    scan_time = datetime.now(timezone.utc)

    feat1 = create_dummy_features(s1, 150.0, scan_time)
    feat1.extra = {
        "pair_id": "pair_test_1",
        "pair_side": "buy",
        "pair_partner": s2,
        "pair_partner_price": 300.0,
        "hedge_ratio": 0.5,  # 1 AAPL = 0.5 MSFT
        "spread_zscore": -2.5,
    }

    feat2 = create_dummy_features(s2, 300.0, scan_time)
    feat2.extra = {
        "pair_id": "pair_test_1",
        "pair_side": "sell",
        "pair_partner": s1,
        "pair_partner_price": 150.0,
        "hedge_ratio": 1.0,  # Reference leg
        "spread_zscore": 2.5,
    }

    scan_result = ScanResult(
        generated_at=scan_time,
        regime=Regime.CHOP,
        watchlist=[
            WatchlistSymbol(
                symbol=s1, score=2.5, strategies=["pair_trading"], features=feat1
            ),
            WatchlistSymbol(
                symbol=s2, score=2.5, strategies=["pair_trading"], features=feat2
            ),
        ],
    )

    # 2. Process scan result
    engine.apply_scan_result(scan_result)

    state1 = engine.symbol_states[s1]
    state2 = engine.symbol_states[s2]

    assert state1.meta["pair_trade"]["pair_id"] == "pair_test_1"
    assert state2.meta["pair_trade"]["pair_id"] == "pair_test_1"

    # 3. Verify RiskManager sizing
    from src.core.domain import Signal

    # Reference leg (MSFT)
    sig_ref = Signal(
        symbol=s2,
        side=OrderSide.SELL,
        size_hint=1.0,  # Will be overridden by RiskManager
        entry_price=300.0,
        stop_price=310.0,
        target_price=280.0,
        strategy="pair_trading",
        generated_at=scan_time,
        meta={"pair_trade": state2.meta["pair_trade"]},
    )

    qty_ref = engine.risk_manager._calculate_qty(sig_ref, 100000.0, None)
    # Ref leg is 1% of 100k = 1000 notional. 1000 / 300 = 3.33 -> 3 shares.
    assert qty_ref == 3

    # Dependent leg (AAPL)
    sig_dep = Signal(
        symbol=s1,
        side=OrderSide.BUY,
        size_hint=1.0,
        entry_price=150.0,
        stop_price=145.0,
        target_price=160.0,
        strategy="pair_trading",
        generated_at=scan_time,
        meta={"pair_trade": state1.meta["pair_trade"]},
    )

    qty_dep = engine.risk_manager._calculate_qty(sig_dep, 100000.0, None)
    # h = 0.5. qty_dep = h * qty_ref = 0.5 * 3 = 1.5 -> 1 share (truncates)
    assert qty_dep == 1

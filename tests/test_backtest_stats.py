from datetime import datetime, timedelta

import pytest

from src.backtest.stats import BacktestAnalyzer


@pytest.fixture
def analyzer():
    return BacktestAnalyzer(initial_cash=10000.0)


@pytest.mark.unit
def test_calculate_statistics_empty(analyzer):
    stats = analyzer.calculate_statistics([], {})
    assert stats["total_trades"] == 0
    assert stats["total_closed_pnl"] == 0.0
    assert stats["open_pnl"] == 0.0
    assert stats["total_equity"] == 10000.0


@pytest.mark.unit
def test_simple_round_trip(analyzer):
    t1 = datetime(2023, 1, 1, 10, 0)
    t2 = datetime(2023, 1, 1, 14, 0)

    fills = [
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
            "fill_price": 100.0,
            "filled_at": t1,
            "strategy": "strat1",
        },
        {
            "symbol": "AAPL",
            "side": "sell",
            "qty": 10,
            "fill_price": 110.0,
            "filled_at": t2,
            "strategy": "strat1",
        },
    ]

    stats = analyzer.calculate_statistics(fills, {"AAPL": 110.0})

    assert stats["total_trades"] == 1
    # PnL = (110 - 100) * 10 = 100
    assert stats["total_closed_pnl"] == 100.0
    assert stats["open_pnl"] == 0.0
    assert stats["total_equity"] == 10100.0
    assert stats["win_rate"] == 1.0


@pytest.mark.unit
def test_partial_fills_and_open_position(analyzer):
    t1 = datetime(2023, 1, 1, 10, 0)
    t2 = datetime(2023, 1, 1, 11, 0)

    fills = [
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 20,
            "fill_price": 100.0,
            "filled_at": t1,
            "strategy": "strat1",
        },
        {
            "symbol": "AAPL",
            "side": "sell",
            "qty": 10,
            "fill_price": 105.0,
            "filled_at": t2,
            "strategy": "strat1",
        },
    ]

    # Remaining 10 shares open. Current price 108.
    # Closed PnL: (105-100)*10 = 50.
    # Open PnL: (108-100)*10 = 80.

    stats = analyzer.calculate_statistics(fills, {"AAPL": 108.0})

    assert stats["total_trades"] == 1
    assert stats["total_closed_pnl"] == 50.0
    assert stats["open_pnl"] == 80.0
    assert stats["total_pnl"] == 130.0
    assert len(stats["open_positions"]) == 1
    assert stats["open_positions"][0]["qty"] == 10.0


@pytest.mark.unit
def test_multiple_symbols_mixed_results(analyzer):
    t1 = datetime(2023, 1, 1, 10, 0)

    fills = [
        # AAPL win: +50
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
            "fill_price": 100.0,
            "filled_at": t1,
            "strategy": "s1",
        },
        {
            "symbol": "AAPL",
            "side": "sell",
            "qty": 10,
            "fill_price": 105.0,
            "filled_at": t1 + timedelta(hours=1),
            "strategy": "s1",
        },
        # MSFT loss: -20
        {
            "symbol": "MSFT",
            "side": "buy",
            "qty": 10,
            "fill_price": 200.0,
            "filled_at": t1,
            "strategy": "s2",
        },
        {
            "symbol": "MSFT",
            "side": "sell",
            "qty": 10,
            "fill_price": 198.0,
            "filled_at": t1 + timedelta(hours=2),
            "strategy": "s2",
        },
    ]

    stats = analyzer.calculate_statistics(fills, {})

    assert stats["total_trades"] == 2
    assert stats["total_closed_pnl"] == 30.0  # 50 - 20
    assert stats["win_rate"] == 0.5

    # Gross profit 50. Gross loss 20. PF = 2.5
    assert stats["profit_factor"] == 2.5


@pytest.mark.unit
def test_short_trade(analyzer):
    t1 = datetime(2023, 1, 1, 10, 0)

    fills = [
        {
            "symbol": "TSLA",
            "side": "sell",
            "qty": 10,
            "fill_price": 200.0,
            "filled_at": t1,
        },
        {
            "symbol": "TSLA",
            "side": "buy",
            "qty": 10,
            "fill_price": 190.0,
            "filled_at": t1 + timedelta(hours=1),
        },
    ]

    stats = analyzer.calculate_statistics(fills, {})
    assert stats["total_closed_pnl"] == 100.0  # (200 - 190) * 10
    assert stats["total_trades"] == 1


@pytest.mark.unit
def test_drawdown_calculation(analyzer):
    # Loss 100, then Win 200.
    # Equity: 10000 -> 9900 -> 10100.
    # Peak 10000. Low 9900. DD = 100/10000 = 1%.
    # Then Peak 10100.

    t1 = datetime(2023, 1, 1, 10, 0)
    fills = [
        {"symbol": "L", "side": "buy", "qty": 1, "fill_price": 200.0, "filled_at": t1},
        {
            "symbol": "L",
            "side": "sell",
            "qty": 1,
            "fill_price": 100.0,
            "filled_at": t1 + timedelta(hours=1),
        },
        {
            "symbol": "W",
            "side": "buy",
            "qty": 1,
            "fill_price": 100.0,
            "filled_at": t1 + timedelta(hours=2),
        },
        {
            "symbol": "W",
            "side": "sell",
            "qty": 1,
            "fill_price": 300.0,
            "filled_at": t1 + timedelta(hours=3),
        },
    ]

    stats = analyzer.calculate_statistics(fills, {})
    assert stats["total_closed_pnl"] == 100.0
    assert stats["max_drawdown_pct"] == 1.0


@pytest.mark.unit
def test_fifo_stack_logic(analyzer):
    # Buy 10 @ 100
    # Buy 10 @ 110
    # Sell 5. Should match first buy @ 100. PnL = (AllocPrice - 100) * 5? No, ExitPrice - Entry.
    # If Sell @ 105.
    # Match 5 against 10@100. PnL (105-100)*5 = 25.
    # Remaining Stack: 5@100, 10@110.

    t1 = datetime(2023, 1, 1, 10, 0)
    fills = [
        {"symbol": "A", "side": "buy", "qty": 10, "fill_price": 100.0, "filled_at": t1},
        {
            "symbol": "A",
            "side": "buy",
            "qty": 10,
            "fill_price": 110.0,
            "filled_at": t1 + timedelta(hours=1),
        },
        {
            "symbol": "A",
            "side": "sell",
            "qty": 5,
            "fill_price": 105.0,
            "filled_at": t1 + timedelta(hours=2),
        },
    ]

    stats = analyzer.calculate_statistics(fills, {"A": 100.0})

    assert stats["total_trades"] == 1
    assert stats["total_closed_pnl"] == 25.0

    # Open positions calculation
    # 5 left from first buy (entry 100). Current 100. Unr PnL 0.
    # 10 left from second buy (entry 110). Current 100. Unr PnL (100-110)*10 = -100.
    # Total Open PnL = -100.

    assert stats["open_pnl"] == -100.0

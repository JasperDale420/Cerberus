from datetime import date, datetime, timezone
from unittest.mock import MagicMock

from src.analysis.analytics import AnalyticsEngine
from src.analysis.schema import Trade as DbTrade


def test_daily_aggregation():
    # Setup
    db = MagicMock()
    logger = MagicMock()
    mock_session = MagicMock()

    # Explicit Context Manager Mock
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = None
    db.get_session.return_value = mock_cm

    engine = AnalyticsEngine(db, logger)

    # Mock Trades
    # 2 Wins, 1 Loss for StrategyA in Bull Regime
    trades = [
        # Win +100
        DbTrade(
            strategy="StratA",
            regime_at_entry="bull",
            pnl_net=100.0,
            exit_time=datetime.now(timezone.utc),
        ),
        # Win +50
        DbTrade(
            strategy="StratA",
            regime_at_entry="bull",
            pnl_net=50.0,
            exit_time=datetime.now(timezone.utc),
        ),
        # Loss -20
        DbTrade(
            strategy="StratA",
            regime_at_entry="bull",
            pnl_net=-20.0,
            exit_time=datetime.now(timezone.utc),
        ),
        # Diff Strat
        DbTrade(
            strategy="StratB",
            regime_at_entry="bear",
            pnl_net=10.0,
            exit_time=datetime.now(timezone.utc),
        ),
    ]

    # Mock session execute chain (Simplified)
    mock_session.execute.return_value.scalars.return_value.all.return_value = trades

    # Run
    target_date = date.today()
    engine.run_daily_aggregation(target_date)

    # Assertions
    # Expect 2 deletes (StratA/bull, StratB/bear) and 2 inserts
    msg = f"Exec: {mock_session.execute.call_count}, Add: {mock_session.add.call_count}, Commit: {mock_session.commit.call_count}. Errors: {logger.error.call_args_list}"
    assert mock_session.add.call_count == 2, msg

    # Verify StratA stats
    # PnL = 130
    # Count = 3
    # Win Rate = 2/3 = 0.66

    # We can capture the objects passed to add
    added_objs = [call[0][0] for call in mock_session.add.call_args_list]

    strat_a = next(s for s in added_objs if s.strategy == "StratA")
    assert strat_a.net_pnl == 130.0
    assert strat_a.n_trades == 3
    assert abs(strat_a.winrate - 0.666) < 0.01
    assert strat_a.regime == "bull"

    strat_b = next(s for s in added_objs if s.strategy == "StratB")
    assert strat_b.net_pnl == 10.0

    # Ensure no errors
    assert logger.error.call_count == 0, f"Error logged: {logger.error.call_args}"

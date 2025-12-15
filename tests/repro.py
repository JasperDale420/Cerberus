from unittest.mock import MagicMock

import pandas as pd


# Mock classes
class DbTrade:
    def __init__(self, strategy, regime_at_entry, pnl_net):
        self.strategy = strategy
        self.regime_at_entry = regime_at_entry
        self.pnl_net = pnl_net


def run():
    mock_session = MagicMock()
    trades = [DbTrade("StratA", "bull", 100.0), DbTrade("StratA", "bull", 50.0)]

    # Mock Chain
    mock_result = MagicMock()
    mock_scalars_result = MagicMock()
    mock_scalars_result.all.return_value = trades

    mock_result.scalars.return_value = mock_scalars_result
    mock_session.execute.return_value = mock_result

    # Simulate Analytics Logic
    res = mock_session.execute("stmt")
    sc = res.scalars()
    retrieved_trades = sc.all()

    print(f"Retrieved: {len(retrieved_trades)}")

    if not retrieved_trades:
        print("No trades")
        return

    df = pd.DataFrame(
        [
            {
                "strategy": t.strategy,
                "regime": t.regime_at_entry,
                "pnl": t.pnl_net,
                "win": 1 if t.pnl_net > 0 else 0,
            }
            for t in retrieved_trades
        ]
    )

    result = (
        df.groupby(["strategy", "regime"])
        .agg(trade_count=("pnl", "count"), total_pnl=("pnl", "sum"))
        .reset_index()
    )

    for _, row in result.iterrows():
        print(f"Row: {row['strategy']} PnL: {row['total_pnl']}")
        mock_session.add("obj")

    print(f"Add Count: {mock_session.add.call_count}")


if __name__ == "__main__":
    run()

import sqlite3

import pandas as pd


def analyze_orders():
    # In earlier commands I saw that `sqlite cerberus.db` yielded tables, so I will analyze that file
    conn = sqlite3.connect('/tmp/cerberus_backtest.db')

    # Check what tables actually have data
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("Tables in DB:")
    for _, row in tables.iterrows():
        count = pd.read_sql_query(f"SELECT COUNT(*) as c FROM {row['name']};", conn).iloc[0]['c']
        print(f"  {row['name']}: {count} rows")

    print("\n--- Order Analysis ---")

    # Load orders that were actually filled
    df = pd.read_sql_query("SELECT symbol, side, qty, type, limit_price, status, time_placed, time_last_update, meta_json FROM orders WHERE status = 'filled'", conn)


    if len(df) == 0:
        print("No filled orders found in the database.")
        return

    print(f"Total filled orders found: {len(df)}")

    # Calculate simple PnL by assuming Buys represent cash OUT (-) and Sells represent cash IN (+)
    # This matches the naive "DB Realized PnL (rough estimate)" in runner.py
    df['cash_flow'] = df.apply(lambda row: (-1 * row['qty'] * row['limit_price']) if row['side'] == 'buy' else (row['qty'] * row['limit_price']), axis=1)

    total_naive_pnl = df['cash_flow'].sum()
    print(f"Naive Cash Flow PnL: ${total_naive_pnl:.2f}")

    # To calculate real profit factor and win rate, we need to match entry and exit orders.
    # We will group by symbol to track position.

    print("\n--- Strategy Performance (Naive Cash Flow) ---")
    import json

    def extract_strategy(meta_json_str):
        try:
            meta = json.loads(meta_json_str)
            return meta.get('strategy', 'unknown')
        except Exception:
            return 'unknown'

    df['strategy'] = df['meta_json'].apply(extract_strategy)

    strategy_perf = df.groupby('strategy')['cash_flow'].sum().reset_index()
    strategy_perf = strategy_perf.sort_values(by='cash_flow', ascending=False)
    print(strategy_perf)

    # Let's try to match trades properly
    print("\n--- Matched Trade Analysis ---")
    trades = []

    for strategy in df['strategy'].unique():
        strat_df = df[df['strategy'] == strategy].sort_values(by='time_placed')

        for symbol in strat_df['symbol'].unique():
            sym_df = strat_df[strat_df['symbol'] == symbol]

            position = 0
            avg_entry_price = 0.0

            for _, row in sym_df.iterrows():
                qty = row['qty']
                price = row['limit_price']
                side = row['side']

                if side == 'buy':
                    if position >= 0:
                        total_value = (position * avg_entry_price) + (qty * price)
                        position += qty
                        avg_entry_price = total_value / position if position > 0 else 0
                    else:
                        cover_qty = min(qty, abs(position))
                        pnl = (avg_entry_price - price) * cover_qty
                        position += cover_qty
                        trades.append({
                            'strategy': strategy,
                            'symbol': symbol,
                            'entry_time': None,
                            'exit_time': row['time_placed'],
                            'qty': cover_qty,
                            'entry_price': avg_entry_price,
                            'exit_price': price,
                            'pnl': pnl,
                            'type': 'short'
                        })
                        remaining_qty = qty - cover_qty
                        if remaining_qty > 0:
                            position = remaining_qty
                            avg_entry_price = price

                elif side == 'sell':
                    if position <= 0:
                        abs_pos = abs(position)
                        total_value = (abs_pos * avg_entry_price) + (qty * price)
                        position -= qty
                        avg_entry_price = total_value / abs(position) if position < 0 else 0
                    else:
                        close_qty = min(qty, position)
                        pnl = (price - avg_entry_price) * close_qty
                        position -= close_qty
                        trades.append({
                            'strategy': strategy,
                            'symbol': symbol,
                            'entry_time': None,
                            'exit_time': row['time_placed'],
                            'qty': close_qty,
                            'entry_price': avg_entry_price,
                            'exit_price': price,
                            'pnl': pnl,
                            'type': 'long'
                        })
                        remaining_qty = qty - close_qty
                        if remaining_qty > 0:
                            position = -remaining_qty
                            avg_entry_price = price

    if trades:
        trades_df = pd.DataFrame(trades)
        print(f"Total matched trades: {len(trades_df)}")
        print(f"Total PnL from matched trades: ${trades_df['pnl'].sum():.2f}")

        # Calculate Win Rate and Profit Factor per strategy
        print("\n--- Strategy Metrics ---")
        metrics = []
        for strat in trades_df['strategy'].unique():
            s_trades = trades_df[trades_df['strategy'] == strat]

            wins = s_trades[s_trades['pnl'] > 0]
            losses = s_trades[s_trades['pnl'] <= 0]

            win_rate = len(wins) / len(s_trades) * 100 if len(s_trades) > 0 else 0

            gross_profit = wins['pnl'].sum()
            gross_loss = abs(losses['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            metrics.append({
                'Strategy': strat,
                'Trades': len(s_trades),
                'Win Rate (%)': win_rate,
                'Profit Factor': profit_factor,
                'Total PnL ($)': s_trades['pnl'].sum(),
                'Avg Trade ($)': s_trades['pnl'].mean()
            })

        metrics_df = pd.DataFrame(metrics).sort_values(by='Total PnL ($)', ascending=False)
        # print using tabulate if available, otherwise just print
        print(metrics_df.to_string(index=False))

    else:
        print("Could not match any complete round-trip trades.")

if __name__ == '__main__':
    analyze_orders()

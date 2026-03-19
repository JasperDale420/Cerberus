#!/usr/bin/env bash
# Ledger Health Check — run after market close to verify trades are recording
# Usage: bash scripts/check_ledger_health.sh [path/to/ledger.db]

set -euo pipefail

DB="${1:-ledger.db}"

if [[ ! -f "$DB" ]]; then
    echo "❌ ledger.db not found at: $DB"
    exit 1
fi

echo "═══════════════════════════════════════════════════"
echo "  LEDGER HEALTH CHECK — $(date '+%Y-%m-%d %H:%M')"
echo "  DB: $DB"
echo "═══════════════════════════════════════════════════"

echo ""
echo "📊 TRADES BY STRATEGY (all time)"
echo "─────────────────────────────────"
sqlite3 -header -column "$DB" "
SELECT
    strategy,
    COUNT(*) AS trades,
    SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) AS wins,
    ROUND(100.0 * SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_pct,
    ROUND(SUM(pnl_net), 2) AS total_pnl,
    ROUND(AVG(pnl_net), 2) AS avg_pnl,
    ROUND(AVG(holding_seconds / 60.0), 1) AS avg_hold_min
FROM ledger_trades
WHERE exit_price IS NOT NULL
GROUP BY strategy
ORDER BY total_pnl DESC;
"

echo ""
echo "📅 TODAY'S TRADES"
echo "─────────────────"
sqlite3 -header -column "$DB" "
SELECT
    strategy,
    symbol,
    side,
    qty,
    ROUND(entry_price, 2) AS entry,
    ROUND(exit_price, 2) AS exit,
    ROUND(pnl_net, 2) AS pnl_net,
    ROUND(pnl_r, 2) AS pnl_r,
    ROUND(holding_seconds / 60.0, 1) AS hold_min,
    regime_entry
FROM ledger_trades
WHERE DATE(entry_time) = DATE('now')
ORDER BY entry_time;
"

echo ""
echo "🔓 OPEN TRADES (no exit yet)"
echo "────────────────────────────"
sqlite3 -header -column "$DB" "
SELECT
    strategy,
    symbol,
    side,
    qty,
    ROUND(entry_price, 2) AS entry,
    entry_time,
    regime_entry
FROM ledger_trades
WHERE exit_price IS NULL
ORDER BY entry_time;
"

echo ""
echo "📋 ORDERS TODAY"
echo "───────────────"
sqlite3 -header -column "$DB" "
SELECT
    strategy,
    symbol,
    side,
    qty,
    order_type,
    status,
    broker_order_id
FROM ledger_orders
WHERE DATE(submitted_at) = DATE('now')
ORDER BY submitted_at;
"

echo ""
echo "🔍 DATA INTEGRITY CHECKS"
echo "────────────────────────"

# Check 1: Trades with no strategy
orphan_count=$(sqlite3 "$DB" "SELECT COUNT(*) FROM ledger_trades WHERE strategy IS NULL OR strategy = '';")
if [[ "$orphan_count" -gt 0 ]]; then
    echo "⚠️  $orphan_count trades with missing strategy name"
else
    echo "✅ All trades have strategy names"
fi

# Check 2: Closed trades with NULL pnl
null_pnl=$(sqlite3 "$DB" "SELECT COUNT(*) FROM ledger_trades WHERE exit_price IS NOT NULL AND pnl_net IS NULL;")
if [[ "$null_pnl" -gt 0 ]]; then
    echo "⚠️  $null_pnl closed trades with NULL pnl_net"
else
    echo "✅ All closed trades have PnL recorded"
fi

# Check 3: Orders without matching trades
orphan_orders=$(sqlite3 "$DB" "
SELECT COUNT(*) FROM ledger_orders o
WHERE o.status = 'filled'
AND NOT EXISTS (
    SELECT 1 FROM ledger_trades t WHERE t.correlation_id = o.correlation_id
);
")
if [[ "$orphan_orders" -gt 0 ]]; then
    echo "⚠️  $orphan_orders filled orders with no matching trade record"
else
    echo "✅ All filled orders have matching trades"
fi

# Check 4: Fills without matching orders
orphan_fills=$(sqlite3 "$DB" "
SELECT COUNT(*) FROM ledger_fills f
WHERE NOT EXISTS (
    SELECT 1 FROM ledger_orders o WHERE o.id = f.order_id
);
")
if [[ "$orphan_fills" -gt 0 ]]; then
    echo "⚠️  $orphan_fills fills with no matching order"
else
    echo "✅ All fills have matching orders"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Total trades: $(sqlite3 "$DB" "SELECT COUNT(*) FROM ledger_trades;")"
echo "  Total orders: $(sqlite3 "$DB" "SELECT COUNT(*) FROM ledger_orders;")"
echo "  Total fills:  $(sqlite3 "$DB" "SELECT COUNT(*) FROM ledger_fills;")"
echo "═══════════════════════════════════════════════════"

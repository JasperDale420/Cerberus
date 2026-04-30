"""Extract actionable insights from WFO results JSON for the autoresearch agent.

Reads artifacts/autoresearch/{strategy}_latest.json and prints a concise
summary the agent can use to make informed modifications.

Usage: uv run python scripts/extract_wfo_insights.py <strategy_name>
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_wfo_insights.py <strategy_name>")
        sys.exit(1)

    strategy = sys.argv[1]
    results_path = Path(f"artifacts/autoresearch/{strategy}_latest.json")

    if not results_path.exists():
        print("NO_INSIGHTS (results file not found)")
        sys.exit(0)

    try:
        results = json.loads(results_path.read_text())
    except Exception:
        print("NO_INSIGHTS (failed to parse results JSON)")
        sys.exit(0)

    # Freshness check: harness stamps results.meta.commit_sha at write time.
    # If the JSON belongs to a different commit than current HEAD, the trade analysis
    # is stale (previous eval succeeded, this one crashed; or strategy reverted).
    # Bail rather than feed phantom insights to the next agent.
    meta = results.get("meta", {})
    saved_sha = meta.get("commit_sha")
    if saved_sha and saved_sha != "unknown":
        try:
            import subprocess

            head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            if saved_sha != head_sha:
                print(f"STALE_INSIGHTS — JSON is for commit {saved_sha[:8]}, HEAD is {head_sha[:8]}")
                print("ACTION: previous eval crashed or strategy reverted; new eval will write fresh data.")
                sys.exit(0)
        except Exception:
            pass  # if git fails, fall through and use the JSON anyway

    oos_metrics = results.get("oos_metrics", [])
    param_stability = results.get("param_stability", {})

    # Aggregate trade-level stats across all OOS windows
    all_trades = []
    for m in oos_metrics:
        all_trades.extend(m.get("trades", []))

    # Use window-aggregated counts as the source of truth — the per-trade list may
    # have been stripped during JSON serialization even when trades occurred.
    aggregate_n_trades = sum(m.get("n_trades", 0) for m in oos_metrics)

    if aggregate_n_trades == 0:
        print("NO_TRADES — strategy generated zero trades across all OOS windows.")
        print("ROOT CAUSE: Entry gates are too strict, or strategy is not registered.")
        print("ACTION: Loosen entry criteria, remove hard gates, or check activation policy.")
        sys.exit(0)

    if not all_trades:
        # We have aggregate counts but no per-trade detail — emit window-level summary.
        print(
            f"TRADE ANALYSIS (aggregate-only — {aggregate_n_trades} trades across "
            f"{len(oos_metrics)} OOS windows; per-trade detail not in JSON):"
        )
        total_n_wins = sum(m.get("n_wins", 0) for m in oos_metrics)
        total_n_losses = sum(m.get("n_losses", 0) for m in oos_metrics)
        total_pnl = sum(m.get("net_pnl", 0.0) for m in oos_metrics)
        agg_winrate = total_n_wins / aggregate_n_trades if aggregate_n_trades else 0
        agg_avg_pnl = total_pnl / aggregate_n_trades if aggregate_n_trades else 0
        avg_pf = sum(m.get("profit_factor", 0) for m in oos_metrics if m.get("n_trades", 0) > 0) / max(
            sum(1 for m in oos_metrics if m.get("n_trades", 0) > 0), 1
        )
        print(f"  Win rate: {agg_winrate:.1%} ({total_n_wins}W / {total_n_losses}L)")
        print(f"  Total PnL: ${total_pnl:.2f} | Avg PnL/trade: ${agg_avg_pnl:.2f} | Avg PF: {avg_pf:.2f}")
        print("\nPER-WINDOW PERFORMANCE:")
        for i, m in enumerate(oos_metrics):
            wt = m.get("n_trades", 0)
            if wt > 0:
                print(
                    f"  window {i}: {wt} trades, PF={m.get('profit_factor', 0):.2f}, "
                    f"WR={m.get('winrate', 0):.0%}, Sharpe={m.get('sharpe_ratio', 0):.2f}, "
                    f"PnL=${m.get('net_pnl', 0):.0f}"
                )
        print("\nDIAGNOSIS:")
        if total_pnl > 0:
            print("  PROFITABLE in aggregate — focus on widening which windows/regimes work.")
        else:
            print("  LOSING in aggregate — examine which windows/regimes lose money and why.")
        # Param stability still works
        unstable = [(p, s) for p, s in param_stability.items() if s.get("cv", 0) >= 0.25]
        if unstable:
            print("\nUNSTABLE PARAMS (CV >= 0.25):")
            for p, s in sorted(unstable):
                print(f"  {p}: mean={s.get('mean', 0):.3f} CV={s.get('cv', 0):.3f}")
        sys.exit(0)

    # Basic stats
    n_trades = len(all_trades)
    pnls = [t.get("pnl", 0.0) for t in all_trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0
    avg_win = sum(winners) / len(winners) if winners else 0
    avg_loss = sum(losers) / len(losers) if losers else 0
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / n_trades if n_trades > 0 else 0

    # Hold time analysis
    hold_times = [t.get("hold_minutes", 0) for t in all_trades if t.get("hold_minutes")]
    avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0
    max_hold_trades = sum(1 for t in all_trades if t.get("exit_reason", "") == "max_hold") if all_trades else 0

    # Side breakdown
    buys = [t for t in all_trades if t.get("side", "").lower() in ("buy", "long")]
    sells = [t for t in all_trades if t.get("side", "").lower() in ("sell", "short")]
    buy_pnl = sum(t.get("pnl", 0) for t in buys)
    sell_pnl = sum(t.get("pnl", 0) for t in sells)

    # Exit reason breakdown
    exit_reasons = {}
    for t in all_trades:
        reason = t.get("exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Per-window summary
    window_summaries = []
    for i, m in enumerate(oos_metrics):
        wt = m.get("n_trades", 0)
        if wt > 0:
            window_summaries.append(
                f"  window {i}: {wt} trades, PF={m.get('profit_factor', 0):.2f}, "
                f"WR={m.get('winrate', 0):.0%}, Sharpe={m.get('sharpe_ratio', 0):.2f}, "
                f"PnL=${m.get('net_pnl', 0):.0f}"
            )

    # Parameter stability
    stable_params = []
    unstable_params = []
    for param, stats in sorted(param_stability.items()):
        cv = stats.get("cv", 0)
        mean = stats.get("mean", 0)
        if cv < 0.25:
            stable_params.append(f"  {param}: mean={mean:.3f} CV={cv:.3f} (STABLE)")
        else:
            unstable_params.append(f"  {param}: mean={mean:.3f} CV={cv:.3f} (UNSTABLE — consider fixing)")

    # Print insights
    print(f"TRADE ANALYSIS ({n_trades} trades across {len(oos_metrics)} OOS windows):")
    print(f"  Win rate: {win_rate:.1%} ({len(winners)}W / {len(losers)}L)")
    print(
        f"  Avg winner: ${avg_win:.2f} | Avg loser: ${avg_loss:.2f} | Ratio: {abs(avg_win / avg_loss) if avg_loss != 0 else 0:.2f}"
    )
    print(f"  Total PnL: ${total_pnl:.2f} | Avg PnL/trade: ${avg_pnl:.2f}")
    print(f"  BUY trades: {len(buys)} (PnL=${buy_pnl:.0f}) | SELL trades: {len(sells)} (PnL=${sell_pnl:.0f})")
    if avg_hold > 0:
        print(
            f"  Avg hold: {avg_hold:.0f}min | Max-hold exits: {max_hold_trades}/{n_trades} ({max_hold_trades / n_trades:.0%})"
        )

    if exit_reasons:
        print("\nEXIT REASONS:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} ({count / n_trades:.0%})")

    if window_summaries:
        print("\nPER-WINDOW PERFORMANCE:")
        for ws in window_summaries:
            print(ws)

    # Diagnosis
    print("\nDIAGNOSIS:")
    if win_rate > 0.55 and avg_pnl < 0:
        print("  HIGH WIN RATE BUT NEGATIVE PnL — avg loss is too large relative to avg win.")
        print("  ACTION: Tighten stop loss OR widen target.")
    elif win_rate < 0.40 and avg_pnl < 0:
        print("  LOW WIN RATE — strategy is picking bad entries.")
        print("  ACTION: Add selectivity (higher confluence threshold, stricter entry criteria).")
    elif n_trades > 500 and avg_pnl < 0:
        print("  OVER-TRADING — too many low-quality entries.")
        print("  ACTION: Reduce trade frequency (longer cooldown, stricter gates, time window).")
    elif max_hold_trades > n_trades * 0.5:
        print("  MOST TRADES HIT MAX HOLD — targets are unreachable.")
        print("  ACTION: Reduce target_atr_mult OR increase max_hold_minutes.")
    elif total_pnl > 0:
        print("  PROFITABLE — consider tightening to improve consistency.")
    else:
        print("  LOSING — examine which windows/regimes lose money and why.")

    if buy_pnl > 0 and sell_pnl < 0:
        print("  BUY side profitable, SELL side losing — consider BUY-only mode.")
    elif sell_pnl > 0 and buy_pnl < 0:
        print("  SELL side profitable, BUY side losing — consider SELL-only mode.")

    if stable_params:
        print("\nSTABLE PARAMS (keep these):")
        for p in stable_params:
            print(p)
    if unstable_params:
        print("\nUNSTABLE PARAMS (these vary across windows — consider making regime-adaptive):")
        for p in unstable_params:
            print(p)


if __name__ == "__main__":
    main()

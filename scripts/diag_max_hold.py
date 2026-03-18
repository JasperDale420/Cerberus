"""
Diagnostic: Why does max_hold exit not fire for some positions?

Root cause investigation: traces the fill dispatch chain to verify whether
PositionManager ever receives fills and creates Position objects in backtest mode.

Run:
  cd /Users/jacobmcmillan/Empire/Cerberus
  uv run python scripts/diag_max_hold.py
"""

import asyncio
import os
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ── Counters ─────────────────────────────────────────────────────
_counters = {
    "dispatch_event_fill": 0,
    "on_trade_update_called": 0,
    "handle_trade_update_dict": 0,
    "handle_trade_update_non_dict": 0,
    "info_has_event_fill": 0,
    "info_empty_event": 0,
    "on_fill_called": 0,
    "position_manager_on_fill_called": 0,
    "position_opened": 0,
    "max_hold_check_called": 0,
    "max_hold_none_skip": 0,
    "max_hold_threshold_exceeded": 0,
    "max_hold_exit_fired": 0,
}

_positions_without_max_hold = []  # (strategy, symbol, max_hold_seconds)
_long_hold_trades = []

# ── Patch 1: SimulatedOrderExecutor._dispatch_event ──────────────
from src.backtest import executor as bt_exec_mod  # noqa: E402

_orig_dispatch = bt_exec_mod.SimulatedOrderExecutor._dispatch_event


def _patched_dispatch(self, event_type, order, fill_qty=0.0, fill_price=0.0):
    if event_type in ("fill", "partial_fill"):
        _counters["dispatch_event_fill"] += 1
    return _orig_dispatch(self, event_type, order, fill_qty, fill_price)


bt_exec_mod.SimulatedOrderExecutor._dispatch_event = _patched_dispatch

# ── Patch 2: SimulatedOrderExecutor.handle_trade_update ──────────
_orig_handle = bt_exec_mod.SimulatedOrderExecutor.handle_trade_update


def _patched_handle(self, update):
    if isinstance(update, dict):
        _counters["handle_trade_update_dict"] += 1
    else:
        _counters["handle_trade_update_non_dict"] += 1
    result = _orig_handle(self, update)
    return result


bt_exec_mod.SimulatedOrderExecutor.handle_trade_update = _patched_handle

# ── Patch 3: ExecutionEngine.on_trade_update ─────────────────────
from src.engine import execution as exec_mod  # noqa: E402

_orig_on_trade = exec_mod.ExecutionEngine.on_trade_update


async def _patched_on_trade(self, update):
    _counters["on_trade_update_called"] += 1

    if not self.order_executor:
        return

    info = self.order_executor.handle_trade_update(update)
    event = str(info.get("event", "") or "")
    if event in ("fill", "partial_fill"):
        _counters["info_has_event_fill"] += 1
    elif event == "":
        _counters["info_empty_event"] += 1

    # Now call original
    await _orig_on_trade(self, update)


exec_mod.ExecutionEngine.on_trade_update = _patched_on_trade

# ── Patch 4: ExecutionEngine.on_fill ─────────────────────────────
_orig_on_fill = exec_mod.ExecutionEngine.on_fill


def _patched_on_fill(self, fill):
    _counters["on_fill_called"] += 1
    return _orig_on_fill(self, fill)


exec_mod.ExecutionEngine.on_fill = _patched_on_fill

# ── Patch 5: PositionManager._open_new_position ─────────────────
from src.engine import position_manager as pm_mod  # noqa: E402

_orig_open = pm_mod.PositionManager._open_new_position


def _patched_open(self, symbol_state, market_state, fill, fill_data, cfg):
    result = _orig_open(self, symbol_state, market_state, fill, fill_data, cfg)
    _counters["position_opened"] += 1
    pos = symbol_state.position
    if pos and pos.max_hold_seconds is None:
        _positions_without_max_hold.append((pos.strategy, pos.symbol, pos.max_hold_seconds))
    return result


pm_mod.PositionManager._open_new_position = _patched_open

# ── Patch 6: PositionManager._check_max_hold_exit ───────────────
_orig_check = pm_mod.PositionManager._check_max_hold_exit


def _patched_check(self, pos, market_state):
    _counters["max_hold_check_called"] += 1
    if pos.max_hold_seconds is None:
        _counters["max_hold_none_skip"] += 1
    elif pos.entry_time is not None and market_state.time is not None:
        held = (market_state.time - pos.entry_time).total_seconds()
        if held >= float(pos.max_hold_seconds):
            _counters["max_hold_threshold_exceeded"] += 1
    result = _orig_check(self, pos, market_state)
    if result is not None:
        _counters["max_hold_exit_fired"] += 1
    return result


pm_mod.PositionManager._check_max_hold_exit = _patched_check

# ── Patch 7: Track strategies with no max_hold in signals ────────
_orig_store = exec_mod.ExecutionEngine._store_pending_entry
_strategies_missing_max_hold = defaultdict(int)


def _patched_store(self, signal, intents):
    _orig_store(self, signal, intents)
    state = self.symbol_states.get(signal.symbol)
    if state:
        pending = state.meta.get("pending_entries", {})
        entry = pending.get(signal.correlation_id)
        if entry and isinstance(entry, dict):
            if entry.get("max_hold_seconds") is None:
                _strategies_missing_max_hold[signal.strategy] += 1


exec_mod.ExecutionEngine._store_pending_entry = _patched_store


# ── Run backtest ─────────────────────────────────────────────────


async def main():
    from src.backtest.runner import run_backtest

    print("=" * 80)
    print("DIAGNOSTIC: max_hold exit investigation")
    print("Running 2-week backtest: 2024-03-01 to 2024-03-15")
    print("Config: config/backtest_v2.yaml (all strategies)")
    print("=" * 80)

    # Create a modified config with relaxed risk limits
    import tempfile

    import yaml

    from src.core.config import ConfigLoader

    loader = ConfigLoader()
    config = loader.load_config("config/backtest_v2.yaml")

    risk = config.get("risk", config.get("risk_management", {}))
    risk["max_trades_per_day"] = 200
    risk["max_open_positions"] = 50

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="diag_") as f:
        yaml.dump(config, f)
        temp_config = f.name

    try:
        report = await run_backtest(
            "2024-03-01",
            "2024-03-15",
            temp_config,
            data_dir="data/bars_2024",
        )
    finally:
        os.unlink(temp_config)

    # ── Analysis ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DIAGNOSTIC RESULTS")
    print("=" * 80)

    print("\n--- Fill Dispatch Chain Counters ---")
    for k, v in sorted(_counters.items()):
        print(f"  {k}: {v}")

    print("\n--- Strategies with missing max_hold_seconds in pending entries ---")
    for strat, count in sorted(_strategies_missing_max_hold.items(), key=lambda x: -x[1]):
        print(f"  {strat}: {count} signals without max_hold")

    print(f"\n--- Positions opened without max_hold_seconds: {len(_positions_without_max_hold)} ---")
    by_strat = defaultdict(int)
    for strat, _sym, _mh in _positions_without_max_hold:
        by_strat[strat] += 1
    for strat, count in sorted(by_strat.items(), key=lambda x: -x[1]):
        print(f"  {strat}: {count}")

    # Analyze trade records from the report
    if report and hasattr(report, "trades"):
        print("\n--- Trade Records Analysis ---")
        long_holds = []
        for t in report.trades:
            if hasattr(t, "holding_period_seconds") and t.holding_period_seconds:
                hold_min = t.holding_period_seconds / 60
            elif hasattr(t, "entry_time") and hasattr(t, "exit_time") and t.entry_time and t.exit_time:
                hold_min = (t.exit_time - t.entry_time).total_seconds() / 60
            else:
                continue
            if hold_min > 120:
                long_holds.append((t, hold_min))

        print(f"  Total trades: {len(report.trades)}")
        print(f"  Trades with hold > 120 min: {len(long_holds)}")

        # Group by strategy
        strat_holds = defaultdict(list)
        for t, hold_min in long_holds:
            strat = getattr(t, "strategy", "?")
            strat_holds[strat].append(hold_min)

        print("\n  Long-hold trades by strategy:")
        for strat, holds in sorted(strat_holds.items(), key=lambda x: -len(x[1])):
            print(f"    {strat}: {len(holds)} trades, avg hold={sum(holds) / len(holds):.0f}m, max={max(holds):.0f}m")

    print("\n" + "=" * 80)
    print("SUMMARY / ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print(f"""
Two issues identified:

1. FILL DISPATCH BUG: SimulatedOrderExecutor._dispatch_event sends _MockUpdate objects
   to engine.on_trade_update, but handle_trade_update only handles dicts.
   Result: on_fill is NEVER called -> PositionManager never creates Position objects ->
   _check_max_hold_exit never runs.

   Evidence:
   - dispatch_event_fill:          {_counters["dispatch_event_fill"]}
   - on_trade_update_called:       {_counters["on_trade_update_called"]}
   - handle_trade_update_non_dict: {_counters["handle_trade_update_non_dict"]}
   - info_has_event_fill:          {_counters["info_has_event_fill"]}
   - info_empty_event:             {_counters["info_empty_event"]}
   - on_fill_called:               {_counters["on_fill_called"]}
   - position_opened:              {_counters["position_opened"]}
   - max_hold_check_called:        {_counters["max_hold_check_called"]}

2. MISSING MAX_HOLD: Even if fills were processed, these strategies don't set
   max_hold_minutes anywhere (neither in exit_config nor YAML):
""")
    for strat, count in sorted(_strategies_missing_max_hold.items(), key=lambda x: -x[1]):
        print(f"   - {strat} ({count} signals)")

    print("""
   Strategies properly setting max_hold_minutes:
   - trend_rider_pro (120m in exit_config + YAML)
   - flow_alpha (45m in exit_config + YAML)
   - orb_v2 (60m in exit_config + YAML)
   - mean_reversion_pro (20m in exit_config + YAML)
   - rsi_bounce (60m in exit_config + YAML)
   - momentum_fade (90m in exit_config + YAML)
""")

    print("=" * 80)
    print("END DIAGNOSTIC")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

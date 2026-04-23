# Strategy Research Archive

Historical autoresearch iterations preserved here for reference. These files are
**not auto-loaded** by `src/main.py` (the dynamic loader does not recurse into
subdirectories) and are not expected to import or run against the current code.

Commit history for these experiments is preserved via the `research-archive/*`
git tags pointing at the tips of the deleted research branches:

- `research-archive/auto-health-2026-04-07-iter70-191k`
- `research-archive/auto-health-2026-04-08`
- `research-archive/autoresearch-v3-session1`
- `research-archive/cerberus-debug-wrapper-2026-04-13`

## Files

| File | Source branch | Notes |
|------|---------------|-------|
| `daily_research_strategy_iter70_best_191k.py` | auto-health-2026-04-07 | iter70 SMA50 gate on RSI2 — $191k (4.05x SPY) best result |
| `daily_research_strategy_04-13_iter7_marketstate.py` | cerberus-debug-wrapper-2026-04-13 | iter7 market_state snapshot filters |
| `daily_research_strategy_v3session1.py` | autoresearch/v3-session1 | early v3 session result |
| `daily_research_v6a.py` – `v6d.py` | 04-13 | v6 iteration family |
| `daily_research_v7a.py` – `v7d.py` | 04-13 | v7 iteration family |
| `daily_research_v8a.py` – `v8d.py` | 04-13 | v8 iteration family |
| `daily_research_v9a_consecdown_ibs.py` | 04-13 | alternate v9a (main has different v9a in active tree) |
| `up_normal_strategy.py` | 04-13 | up-normal experiment |
| `archived/failed_breakout.py` | 04-13 | previously archived |
| `archived/trend_pullback.py` | 04-13 | previously archived |

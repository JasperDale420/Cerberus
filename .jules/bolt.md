## 2026-03-05 - Grid Search Eager Materialization
**Learning:** `GridSearchOptimizer` previously converted `itertools.product(...)` to a full list before evaluation, which creates avoidable memory pressure and delays first-result processing as search spaces grow.
**Action:** Keep Cartesian products lazy and compute counts with `math.prod(len(v) for v in values)` when only cardinality is needed for logging.

## 2026-03-04 - Duplicate Routing Strategies Inflate Watchlist Work
**Learning:** `Scanner._build_watchlist()` accepted duplicate strategy names from `strategy_routing` for survivor-only symbols and used list membership checks in a hot loop, causing avoidable extra work and duplicate strategy output entries.
**Action:** Normalize routed strategies into a set at watchlist build time and keep strategy accumulation set-based until final sorted output.

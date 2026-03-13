## 2026-03-05 - Grid Search Eager Materialization
**Learning:** `GridSearchOptimizer` previously converted `itertools.product(...)` to a full list before evaluation, which creates avoidable memory pressure and delays first-result processing as search spaces grow.
**Action:** Keep Cartesian products lazy and compute counts with `math.prod(len(v) for v in values)` when only cardinality is needed for logging.

## 2026-03-04 - Duplicate Routing Strategies Inflate Watchlist Work
**Learning:** `Scanner._build_watchlist()` accepted duplicate strategy names from `strategy_routing` for survivor-only symbols and used list membership checks in a hot loop, causing avoidable extra work and duplicate strategy output entries.
**Action:** Normalize routed strategies into a set at watchlist build time and keep strategy accumulation set-based until final sorted output.

## 2026-02-27 - Backtest Volume Lookup Hot Path
**Learning:** `BacktestFeaturePipeline._avg_daily_volume` was re-sorting historical day keys for every symbol scan, creating repeated O(d log d) overhead in intraday loops.
**Action:** Keep per-symbol date indexes precomputed during `_build_index` and use binary search for lookups in repeated scan-time helpers.

## 2026-03-09 - Indicator Cache Still Re-Sorted on Every Bar
**Learning:** `ExecutionEngine._update_indicator_cache` was still spending hot-path CPU in `sorted(...)` calls on cache hits (strategy key normalization plus per-indicator period sorting), even when allowed strategies and config were unchanged.
**Action:** Cache both normalized strategy tuples and pre-sorted positive period tuples, then iterate directly in per-bar updates so repeated bars avoid sorting entirely.

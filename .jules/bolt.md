## 2026-03-05 - Grid Search Eager Materialization
**Learning:** `GridSearchOptimizer` previously converted `itertools.product(...)` to a full list before evaluation, which creates avoidable memory pressure and delays first-result processing as search spaces grow.
**Action:** Keep Cartesian products lazy and compute counts with `math.prod(len(v) for v in values)` when only cardinality is needed for logging.

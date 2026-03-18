---
name: flow_alpha_limited_data
description: Flow Alpha strategy has only a few months of flow signal data — cannot run full-year WFO like other strategies
type: project
---

Flow Alpha strategy has limited flow signal data (only a few months worth). Must be optimized separately from other strategies with a shorter date range.

**Why:** Flow signals (from Unusual Whales / Data-Gateway) weren't collected for the full 2024 dataset. Running WFO on full year gives misleading results since most of the year has no flow data, causing the strategy to fire on price-only signals.

**How to apply:** When running WFO or backtests on flow_alpha, use a shorter date range matching available flow data. Optimize separately from the other 5 strategies.

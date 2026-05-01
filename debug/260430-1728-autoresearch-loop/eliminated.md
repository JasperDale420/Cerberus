# Disproven Hypotheses

These hypotheses were tested during the 30-iteration debug loop but the experiments showed them to be false. Recorded so future debug sessions don't re-investigate.

## H11 — `git diff --name-only` misses newly added strategy files
**Concern:** The `no_strategy_change` guard at driver:264 uses `git diff "$BEST_COMMIT" HEAD --name-only | grep "^src/strategies/${STRAT_FILE}\.py$"`. We hypothesized that newly created files don't appear in name-only diffs.

**Disproven:** Reproduced in an isolated git repo: `git diff <prev> HEAD --name-only` includes added files. Guard works correctly for new strategies.

## H18 — `compute_regime_diversity_multiplier` is inert (regime-diversity guard not wired)
**Concern:** The CHANGELOG (`c4c05fe8`) noted a recent fix that "exposed the per-trade list" so `compute_regime_diversity_multiplier` could compute against actual trades.

**Disproven:** Verified via grep — `src/backtest/backtest_report.py:662,665-678` has both `_trade_to_dict` helper and `"trades": [self._trade_to_dict(t) for t in self.trades]` in `to_dict()`. `optuna_harness.py:1336-1339` reads `oos_metrics.get("trades", [])` and applies the multiplier. The guard is now active.

## H19 — `WFO_FULL_END=2026-03-19` is stale relative to data
**Concern:** Hardcoded end date in `cerberus_autoresearch.py:54`. If data updates, WFO won't pick it up.

**Disproven:** SPY data range queried — `2016-01-01 .. 2026-03-19`. The constant exactly matches the latest available bar. Today is 2026-04-30 (42-day gap), but no data exists for that gap. Not a current bug; maintenance hazard noted for the next data refresh.

## Other ideas surfaced and de-scoped
- **Restore-protected creates redundant commits:** Test landed in the wrong git repo. The behavior (locking harness to BASELINE during a session, with auto-restore commits when BEST_COMMIT diverges) is intentional per CHANGELOG. De-scoped.
- **Param-stability CV penalty too harsh:** Working as designed (n_trials=5 tradeoff). De-scoped.
- **Duplicate `classify_window_regime` calls in SPY benchmark loop:** Performance, not correctness. De-scoped.
- **Strategy-detector vs harness-detector regime mismatch:** Architectural concern (in-strategy `MarketContextService` vs offline `classify_window_regime` SMA). Not a bug per se but worth tracking — if a strategy gates on its own detector and the harness filter uses a different one, the score reflects a regime the strategy didn't see itself in.

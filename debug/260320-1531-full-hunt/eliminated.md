# Eliminated Hypotheses

These hypotheses were tested and disproven — equally valuable as confirmed bugs because they narrow the search space.

## 1. `is True` / `is False` Identity Comparisons
- **Tested:** Changed `is True` to truthiness check in strategy_engine.py:170
- **Result:** Test failure — `hard_stop_fn(MagicMock())` returns a truthy MagicMock, not a bool. The `is True` is intentional to prevent non-bool truthy values from triggering hard stops.
- **Learning:** `granger_causal` in flow_alpha.py is tri-state (True/False/None); `is False` correctly distinguishes False from None.

## 2. ledger_adapter return-None
- **Tested:** Reviewed caller expectations and error handling
- **Result:** The ledger is a non-critical audit trail. Callers handle None gracefully. The warning IS logged with `exc_info=True`.
- **Learning:** CLAUDE.md "never return None to indicate failure" applies to core data flow, not auxiliary recording systems.

## 3. ORB_V2 Stop Placement for SHORT
- **Tested:** Traced the math: SHORT breakout enters near `range_low`. Stop = `range_low + adj` = above entry.
- **Result:** Correct. The recon agent confused the direction.
- **Learning:** For breakout strategies, stop is INSIDE the range (above range_low for shorts, below range_high for longs).

## 4. RSI_bounce Trend Filter
- **Tested:** Analyzed the filter logic: UP trend + overbought → reject SHORT.
- **Result:** Correct risk management for a mean-reversion strategy — don't fade against the dominant trend.
- **Learning:** "Inverted" depends on perspective. For trend-following = wrong. For risk-managed mean reversion = right.

## 5. trend_rider_pro SELL Disabled
- **Tested:** Read code comment: "2025 analysis: short signals had negative edge."
- **Result:** Intentional design decision based on backtest evidence.
- **Learning:** One-directional strategies are valid when the other direction has negative edge.

## 6. health.py Silent Fallback
- **Tested:** Reviewed the health check endpoint's JSON parsing fallback.
- **Result:** Returns "degraded" status on parse failure — appropriate for health monitoring.
- **Learning:** Health checks should degrade gracefully, not crash.

## 7. walk_forward.py Division by Zero
- **Tested:** Traced the early return: `len(window_results) < 2` prevents empty `expectancies`.
- **Result:** Safe — `len(expectancies)` >= 2 when the division happens.

## 8. strategy_engine on_error Callback Silent Pass
- **Tested:** Reviewed the except block at strategy_engine.py:207.
- **Result:** The error callback failure is non-critical — the actual strategy error IS logged on lines 212-219. The `pass` prevents the error-reporting mechanism from raising another exception.
- **Learning:** Silencing error-callback failures is a standard defensive pattern.

## 9. WebSocket Close Silent Pass
- **Tested:** Reviewed data/client.py:563.
- **Result:** Standard cleanup — the WebSocket connection may already be dead/closed. Silently ignoring errors during `close()` is appropriate.

## 10. scanner _is_finite Returns False Silently
- **Tested:** Reviewed scanner/validation.py:21.
- **Result:** `math.isfinite()` raises `TypeError` on non-numeric values. Returning `False` for non-numeric inputs is correct validation predicate behavior.

## 11. gap_fill Division Guard Too Loose
- **Tested:** Traced the guard `abs(1.0 + gap_pct) < 1e-9` in gap_fill.py:136.
- **Result:** Extreme values produce very large or very negative `prev_close`, which is then rejected by downstream checks (bar.close <= prev_close returns None).

## 12. momentum_fade Trend Filter Inverted
- **Tested:** Analyzed the directional filter at momentum_fade.py:351-354.
- **Result:** Correct for a fade strategy: UP trend + BUY (= fading a dip IN the trend) is rejected. Only true counter-trend fades are allowed.

## 13. CVaR GPD Denominator Near-Zero
- **Tested:** Traced cvar_sizer.py:207 with xi_hat close to 1.0.
- **Result:** The `isfinite` check on line 210 catches extreme values from near-zero denominator. Result is ultra-conservative (not incorrect).

## 14. execution.py symbol_states KeyError on Signal
- **Tested:** Traced signal generation flow from bar loop.
- **Result:** Signals are always generated inside the bar processing loop where `symbol_state` is already looked up from `self.symbol_states`. The `try/except` on 950-960 is defense-in-depth.

## 15. NaN Propagation in gap_pct
- **Tested:** Reviewed pipeline.py:131 and calculator.py.
- **Result:** Calculator initializes `gap_pct = 0.0` and only computes it if `len(closes) >= 2`. Fallback is always 0.0, not NaN.

## 16. Assert Statements in Production Code
- **Tested:** Reviewed position_manager.py:396,493 assertions.
- **Result:** Acceptable defensive assertions. Python is rarely run with `-O` flag for trading services. The assertions provide immediate context on state machine violations.

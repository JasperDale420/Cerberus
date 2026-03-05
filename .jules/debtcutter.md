## 2026-02-14 - Reject Non-Finite Technicals
**Debt:** Technical validation accepted NaN/Inf price, volume, or ATR, letting corrupt data pass the scanner gate.
**Why it matters:** Non-finite values can poison ranking and filters silently, creating false positives or missing candidates.
**Next time:** Add finite-value checks at validation boundaries when new numeric features are introduced.

## 2026-03-05 - Reject Non-Numeric Flow Metrics
**Debt:** Flow validation only checked `int/float` in extra metrics, so string garbage values (for example `"not-a-number"`) bypassed validation.
**Why it matters:** Invalid flow fields could leak into downstream scoring and create hard-to-debug scanner behavior.
**Next time:** Use shared finite checks for every numeric validation path, including optional dict fields.

## 2026-03-04 - Expose Holding-Period Failures
**Debt:** Closed-trade holding-period calculation swallowed exceptions with a silent `pass`.
**Why it matters:** Bad `entry_time` payloads were invisible in logs, increasing debugging time when close-trade metrics looked wrong.
**Next time:** Replace silent exception handlers in position lifecycle math with structured debug/error logs that include symbol and timestamps.

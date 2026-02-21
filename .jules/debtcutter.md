## 2026-02-14 - Reject Non-Finite Technicals
**Debt:** Technical validation accepted NaN/Inf price, volume, or ATR, letting corrupt data pass the scanner gate.
**Why it matters:** Non-finite values can poison ranking and filters silently, creating false positives or missing candidates.
**Next time:** Add finite-value checks at validation boundaries when new numeric features are introduced.

## 2026-02-24 - Naive Timestamp Visibility
**Finding:** Time window checks silently accepted naive datetimes and parse failures, masking timezone mistakes.
**Risk:** Hidden timezone coercion can let trades slip into the wrong session without detection.
**Fix pattern:** Emit structured warnings/errors when naive datetimes are coerced or time window parsing fails, while keeping fail-open behavior.
**Next time:** Track naive timestamps at ingestion boundaries with run_id/strategy_id when available.

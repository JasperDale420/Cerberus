# Kairos Mechanic Journal

## 2026-03-05 - Naive Cache Timestamp Guard
**Finding:** Incremental bar fetch could crash when cached bar timestamps lacked timezone info (`TypeError` comparing naive vs aware datetimes).
**Risk:** One malformed cached timestamp can break feature data refresh and silently reduce trading signal freshness.
**Fix pattern:** Normalize parsed cache timestamps to UTC when `tzinfo` is missing, and emit structured warning logs with symbol + raw timestamp.
**Next time:** Treat all parsed timestamps as suspect; normalize at boundaries and add regression tests around cache hydration paths.

## 2026-03-04 - Gateway Naive Timestamp Guard
**Finding:** Data-Gateway bar payloads with ISO timestamps that omit timezone were parsed as naive datetimes, creating ambiguous bar time semantics.
**Risk:** Naive timestamps can skew session logic and diagnostics, especially when comparing or replaying bars across UTC/ET boundaries.
**Fix pattern:** Normalize naive bar timestamps to UTC at ingestion and emit a structured warning (`event_type=gateway_bar_timestamp_naive`) with raw timestamp value.
**Next time:** Add contract tests for provider payloads that intentionally omit timezone and fail CI if naive datetimes reach domain models.

## 2026-02-24 - Naive Timestamp Visibility
**Finding:** Time window checks silently accepted naive datetimes and parse failures, masking timezone mistakes.
**Risk:** Hidden timezone coercion can let trades slip into the wrong session without detection.
**Fix pattern:** Emit structured warnings/errors when naive datetimes are coerced or time window parsing fails, while keeping fail-open behavior.
**Next time:** Track naive timestamps at ingestion boundaries with run_id/strategy_id when available.

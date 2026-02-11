# Cerberus + Data-Gateway + Heber Migration Roadmap

## Objective
Move Cerberus from direct provider-first reads to:
1. Data-Gateway-first market/flow reads.
2. Heber-backed point-in-time-safe historical reads.

This roadmap shows **current Cerberus status** and **remaining work**.

## Runtime Toggles (Cerberus)
- `CERBERUS_DATA_BACKEND=legacy|gateway|dual`
- `CERBERUS_STORAGE_BACKEND=sqlite|heber|dual`
- `CERBERUS_GATEWAY_URL=...`
- `CERBERUS_GATEWAY_KEY=...`
- `CERBERUS_HEBER_CATALOG_URL=...`
- `CERBERUS_HEBER_DATA_ROOT=...`
- `CERBERUS_DUAL_READ_COMPARE=true|false`
- `CERBERUS_FAILOVER_TO_LEGACY=true|false`

Gateway/Heber dependency toggles (outside Cerberus repo):
- `GATEWAY_DATA_SINK_ENABLED=true|false`
- `GATEWAY_DATA_SINK_REDIS_URL=...`
- `HEBER_REDIS_STREAM_NAME=heber:events`
- `HEBER_REDIS_DLQ_STREAM_NAME=heber:events:dlq`

## Phase Status Overview
- Phase 0 (pre-hardening): **partially implemented in Cerberus**
- Phase 1 (gateway adapter cut-in): **implemented in Cerberus**
- Phase 2 (Gateway stream sink): **cross-repo pending (Data-Gateway/Heber)**
- Phase 3 (Heber shadow reads): **implemented in Cerberus shadow mode; broader rollout pending**
- Phase 4 (live cutover): **pending**
- Phase 5 (legacy decommission): **pending**

## Phase 0: Pre-Migration Hardening

### Implemented in Cerberus
- Startup mode validation for gateway/heber/legacy requirements.
- Gateway/Heber health checks in Cerberus health module.
- Env-backed routing flags centralized in runtime settings.

### Remaining
- Environment-level validation of Redis/Postgres/data-root in target deployed environments.

### Evidence (Cerberus)
- `/Users/jacobmcmillan/Empire/Cerberus/src/core/settings.py`
- `/Users/jacobmcmillan/Empire/Cerberus/src/core/health.py`
- `/Users/jacobmcmillan/Empire/Cerberus/tests/unit/test_startup_validation_unit.py`

## Phase 1: Gateway Client Integration (Read Path)

### Implemented in Cerberus
- Gateway route adapter methods in `CentralApiClient`.
- `DataFetcher` routing for `legacy|gateway|dual`.
- Failover handling (`CERBERUS_FAILOVER_TO_LEGACY`).
- Dual parity logging for bars/trades/flow/gex.
- Scanner + feature pipeline gateway path wiring.

### Exit criteria (implemented)
- Legacy mode unchanged.
- Gateway mode routes through central client.
- Dual mode emits parity diagnostics.

### Evidence (Cerberus)
- `/Users/jacobmcmillan/Empire/Cerberus/src/data/api_client.py`
- `/Users/jacobmcmillan/Empire/Cerberus/src/data/fetcher.py`
- `/Users/jacobmcmillan/Empire/Cerberus/tests/integration/test_gateway_failover_integration.py`
- `/Users/jacobmcmillan/Empire/Cerberus/tests/integration/test_gateway_scanner_feature_pipeline_integration.py`

## Phase 2: Stream-to-Heber Activation (Write Path)

### Scope
Enable canonical Gateway stream publication and verify Heber ingestion.

### Status
**Pending in this repo (cross-repo dependency).**
Cerberus does not own sink/consumer implementation.

### Needed evidence (outside Cerberus)
- Gateway sink enabled and stable.
- Heber consumer writes healthy with bounded DLQ.
- Stream topic alignment validated in deployed environment.

## Phase 3: Heber Read Introduction (Shadow)

### Implemented in Cerberus
- Heber shadow read path through `HeberReadClient` for bars/trades.
- As-of safety gate enforced with `ts_available <= as_of`.
- Fallback to gateway path when Heber rows are unavailable/not safe.
- Shadow parity integration tests.

### Remaining
- Expand adoption to all offline replay/analysis paths listed in project roadmap.
- Validate multi-session operational stability in real environment.

### Evidence (Cerberus)
- `/Users/jacobmcmillan/Empire/Cerberus/src/data/heber_read_client.py`
- `/Users/jacobmcmillan/Empire/Cerberus/src/data/fetcher.py`
- `/Users/jacobmcmillan/Empire/Cerberus/tests/integration/test_heber_shadow_parity_integration.py`

## Phase 4: Live Cutover

### Status
**Pending.**

### Planned actions
- Set `CERBERUS_DATA_BACKEND=gateway`.
- Set `CERBERUS_STORAGE_BACKEND=heber|dual`.
- Keep `CERBERUS_FAILOVER_TO_LEGACY=true` during confidence window.

### Hard stop criteria
- Sustained Gateway `5xx`/transport failure.
- Heber DLQ surge above agreed threshold.
- Stale data age above agreed threshold.

### Rollback
- `CERBERUS_DATA_BACKEND=legacy`
- `CERBERUS_STORAGE_BACKEND=sqlite`

## Phase 5: Legacy Read Decommission

### Status
**Pending.**

### Planned actions
- Remove runtime dependence on direct market-data provider reads once cutover is stable.
- Keep order execution path unchanged unless explicitly migrated.
- Keep rollback path available for agreed grace period.

## Test Strategy (Cerberus-owned evidence)
- Contract/routing coverage:
  - `/Users/jacobmcmillan/Empire/Cerberus/tests/contract/test_central_api_client_contract.py`
- Gateway + failover + dual parity:
  - `/Users/jacobmcmillan/Empire/Cerberus/tests/integration/test_gateway_failover_integration.py`
- Gateway scanner/feature pipeline wiring:
  - `/Users/jacobmcmillan/Empire/Cerberus/tests/integration/test_gateway_scanner_feature_pipeline_integration.py`
- Heber shadow parity + anti-leakage behavior:
  - `/Users/jacobmcmillan/Empire/Cerberus/tests/integration/test_heber_shadow_parity_integration.py`
- Startup mode validation:
  - `/Users/jacobmcmillan/Empire/Cerberus/tests/unit/test_startup_validation_unit.py`

## Ownership Split
- Cerberus team:
  - Runtime adapter routing, failover behavior, parity diagnostics, health checks.
- Data-Gateway team (dependency):
  - Provider contracts, sink reliability, stream fanout metrics.
- Heber team (dependency):
  - Consumer reliability, schema correctness, read API guarantees.

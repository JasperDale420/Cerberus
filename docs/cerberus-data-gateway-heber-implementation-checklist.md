# Cerberus + Data-Gateway + Heber Implementation Checklist

## Legend
- `[x] Implemented in Cerberus repo`
- `[~] Implemented + validated by tests in Cerberus repo`
- `[ ] Pending in Cerberus repo`
- `[D] Cross-repo dependency (Data-Gateway/Heber)`
- `[D~] Cross-repo dependency validated in local stack (not owned by Cerberus)`

## How To Use This
- Execute phases in order.
- Do not mark `[x]` unless code exists in this repo.
- Do not mark `[~]` unless code exists and test evidence is linked.
- Track cross-repo work under dependency sections only.

## Phase 0: Contract and Environment Readiness (Cerberus-owned)
- `[x]` Integration env vars added in runtime settings (`src/core/settings.py`).
- `[x]` Startup validation for gateway/heber/legacy mode requirements.
- `[x]` Gateway/Heber connectivity health checks in `src/core/health.py`.
- `[~]` Environment-level verification completed for local target stack (`gateway:8080`, `heber-catalog:8085`, Redis stream `heber:events`, Heber data root `/Volumes/heber/data`) on 2026-02-12.

### Gate
- `[x]` Smoke harness script exists: `scripts/smoke_gateway_heber_integration.py`.
- `[x]` Smoke harness unit test exists: `tests/unit/test_smoke_gateway_heber_integration_unit.py`.
- `[~]` End-to-end environment smoke run completed and recorded for local target deployment on 2026-02-12 (`8/8 checks passed`).

## Phase 1: Cerberus Data Adapter Cut-In (No Behavior Change)

### Core implementation
- `[~]` `src/data/api_client.py` uses Data-Gateway v1 routes and `X-Gateway-Key`.
- `[~]` `src/data/api_client.py` supports timeout/retry classification for `401/403/429/5xx`.
- `[~]` `src/data/fetcher.py` routes reads by `legacy|gateway|dual`.
- `[~]` `src/data/fetcher.py` supports legacy failover control.
- `[~]` `src/data/pipeline.py` uses routed fetch paths while preserving feature semantics.
- `[~]` `src/scanner/universe.py` uses gateway screener adapter when enabled.
- `[x]` `src/main.py` wires backend selection at composition root.

### Test evidence
- `[~]` Contract tests: `tests/contract/test_central_api_client_contract.py`.
- `[~]` Gateway/failover/dual parity tests: `tests/integration/test_gateway_failover_integration.py`.
- `[~]` Scanner + feature pipeline gateway tests: `tests/integration/test_gateway_scanner_feature_pipeline_integration.py`.

### Gate
- `[~]` `CERBERUS_DATA_BACKEND=legacy` unchanged behavior.
- `[~]` `CERBERUS_DATA_BACKEND=gateway` path validated.
- `[~]` `CERBERUS_DATA_BACKEND=dual` parity diagnostics validated.

## Phase 2: Data-Gateway Stream Sink Activation
- `[D~]` Sink initialization and startup readiness validated in local stack (`/health/ready` returned `checks.sinks=ok` on 2026-02-12).
- `[D]` Backpressure/drop metrics and behavior under sustained production load.
- `[D~]` Stream publish contract and route/feed mapping tests validated in Data-Gateway (`tests/test_middleware_streaming.py`, including Cerberus-critical routes).

## Phase 3: Heber Consumer and Silver Validation
- `[D~]` Envelope/feed compatibility validated for Cerberus-critical routes (`bars`, `trades`, `flow_alerts`, `greek_exposure`) in local stream events on 2026-02-12.
- `[D]` Retry/DLQ behavior validation under sustained production profile.
- `[D~]` Silver schema alignment validated for Cerberus-critical feeds with active Bronze/Silver writes observed during smoke run on 2026-02-12.

## Phase 4: Cerberus Heber Read Integration (Shadow)

### Core implementation
- `[~]` Heber-backed shadow reads implemented through:
  - `src/data/fetcher.py`
  - `src/data/heber_read_client.py`
- `[~]` Anti-leakage enforcement via `ts_available <= as_of` in Heber read path.
- `[x]` Heber connectivity and freshness checks in `src/core/health.py`.
- `[ ]` Wider replay/analysis path migration to Heber-backed reads (beyond current fetcher shadow scope).
- `[ ]` Direct Heber SDK (`read_asof/asof_join`) adoption in Cerberus runtime paths (currently not active).

### Test evidence
- `[~]` Shadow parity integration suite: `tests/integration/test_heber_shadow_parity_integration.py`.
- `[~]` Startup-mode validation suite: `tests/unit/test_startup_validation_unit.py`.

### Gate
- `[ ]` Multi-session shadow stability validated in target environment.
- `[ ]` Operational anti-leakage checks validated in production-like runs.

## Phase 5: Live Cutover and Cleanup (Cerberus)
- `[ ]` Set `CERBERUS_DATA_BACKEND=gateway` in live runtime.
- `[ ]` Set `CERBERUS_STORAGE_BACKEND=heber|dual` in live runtime.
- `[ ]` Keep `CERBERUS_FAILOVER_TO_LEGACY=true` during confidence window.
- `[ ]` Remove runtime legacy read dependencies after soak period.
- `[ ]` Update core operational docs after cutover.

## Rollback Quick Actions (Cerberus)
- `[x]` Rollback toggles defined:
  - `CERBERUS_DATA_BACKEND=legacy`
  - `CERBERUS_STORAGE_BACKEND=sqlite`
- `[D]` Optional sink disable on Gateway side:
  - `GATEWAY_DATA_SINK_ENABLED=false`

## Cross-Repo Dependencies (Not Marked Done in Cerberus)

### Data-Gateway
- `[D]` Confirm Cerberus client permissions in `Data-Gateway/config/clients.yaml`.
- `[D]` Confirm sink defaults are safe by environment.
- `[D]` Confirm sink reliability under production profile.

### Heber
- `[D]` Confirm stream topic + consumer group behavior in deployment.
- `[D]` Confirm Silver schema contracts required by Cerberus.
- `[D]` Confirm DLQ thresholds and replay/repair runbooks.

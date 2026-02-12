# Data-Gateway + Heber Docs Review (Cerberus)

## 1) Executive Summary (Plain English)
- Good news: Cerberus already has real gateway/heber migration code in place.
- Problem: several docs were still describing old or planned behavior as if it were current.
- Biggest doc drift items were:
  - claiming route mismatch even though gateway routes are already wired,
  - claiming Heber SDK `read_asof/asof_join` is the active Cerberus path,
  - missing retry env vars in env docs,
  - checklist status mixing Cerberus-complete items with cross-repo dependencies.
- Result of this review pass:
  - core migration docs now match current Cerberus runtime behavior,
  - status language now clearly separates “implemented here” vs “dependency outside this repo”,
  - env var docs now include all runtime gateway knobs used by code.

## 2) Line-Level Findings and Concrete Fixes

| File | Pre-fix line(s) | Finding | Concrete fix applied |
|---|---:|---|---|
| `/Users/jacobmcmillan/Empire/Cerberus/docs/cerberus-data-gateway-heber-architecture.md` | 19 | Stale claim that Cerberus paths do not match Data-Gateway routes. | Rewrote current-state section to reflect existing `CentralApiClient` route wiring. |
| `/Users/jacobmcmillan/Empire/Cerberus/docs/cerberus-data-gateway-heber-architecture.md` | 76-78 | Quotes/snapshot listed as active runtime mapping without matching Cerberus client methods. | Kept endpoint map aligned to currently implemented `CentralApiClient` methods; quotes/snapshot moved to explicit “not currently wired” note. |
| `/Users/jacobmcmillan/Empire/Cerberus/docs/cerberus-data-gateway-heber-architecture.md` | 105-108 | Stated Heber SDK `read_asof/asof_join` as current path; Cerberus currently uses `HeberReadClient` parquet reads. | Updated Heber section to “current path” (`HeberReadClient`) plus “future option” (SDK). |
| `/Users/jacobmcmillan/Empire/Cerberus/docs/cerberus-data-gateway-heber-implementation-checklist.md` | 114 | Checklist marked SDK usage done for Cerberus adapters, overstating current implementation. | Reset to pending and explicitly documented current read path as `HeberReadClient`. |
| `/Users/jacobmcmillan/Empire/Cerberus/docs/environment-variables.md` | 35-43 | Missing gateway retry vars used in runtime client. | Added `CERBERUS_GATEWAY_MAX_RETRIES` and `CERBERUS_GATEWAY_RETRY_BACKOFF_SECONDS`. |
| `/Users/jacobmcmillan/Empire/Cerberus/docs/environment-variables.md` | 37 + startup validation nuance in code | URL shown as optional default without startup-validation nuance in gateway/dual mode. | Added conditional requirement rule explaining non-default gateway URL + key requirement in gateway/dual mode. |
| `/Users/jacobmcmillan/Empire/Cerberus/docs/cerberus-data-gateway-heber-migration-roadmap.md` | multiple | Roadmap did not clearly mark what is already implemented vs still planned in Cerberus. | Added explicit phase status summary and per-phase “implemented in Cerberus” evidence links. |
| `/Users/jacobmcmillan/Empire/Cerberus/docs/cerberus-data-gateway-heber-implementation-checklist.md` | entire file | Non-Cerberus tasks and Cerberus tasks were mixed, causing status ambiguity. | Added status legend and moved external work into a dedicated “Cross-Repo Dependencies” section. |

## 3) Truth Matrix (Doc Claim vs Code/Test Reality)

| Contract item | Doc claim | Code/test reality | Verdict |
|---|---|---|---|
| Data backend modes | `legacy|gateway|dual` | `src/core/settings.py` uses Literal with same values; tests validate routing | match |
| Storage backend modes | `sqlite|heber|dual` | `src/core/settings.py` uses Literal with same values; tests validate behavior | match |
| Gateway bars route | `/api/v1/alpaca/stocks/{symbol}/bars` | Implemented in `src/data/api_client.py::get_alpaca_bars` | match |
| Gateway trades route | `/api/v1/alpaca/stocks/{symbol}/trades` | Implemented in `src/data/api_client.py::get_alpaca_trades` | match |
| Gateway screener routes | `/most-actives`, `/movers` | Implemented in `get_alpaca_most_actives` + `get_alpaca_movers`; integration tests cover scanner path | match |
| UW flow route | `/api/v1/uw/flow/{symbol}` | Implemented in `get_uw_flow`; gateway failover tests cover flow path | match |
| UW GEX route | `/api/v1/uw/gex/{symbol}` | Implemented in `get_uw_gex`; gateway failover tests cover gex path | match |
| Auth header | `X-Gateway-Key` | Set in `CentralApiClient` headers when key is present | match |
| Gateway retries documented | retries/backoff config | Runtime uses `CERBERUS_GATEWAY_MAX_RETRIES` + `CERBERUS_GATEWAY_RETRY_BACKOFF_SECONDS` | mismatch (fixed in docs) |
| Heber read contract | SDK `read_asof/asof_join` is current Cerberus path | Current Cerberus uses `HeberReadClient` local parquet + `ts_available` gating | mismatch (fixed in docs) |
| Dual parity behavior | dual compares outputs | `DataFetcher` logs parity for bars/trades/flow/gex; integration tests validate | match |
| Startup requirements in gateway mode | gateway URL/key needed | `validate_startup_mode` requires key + non-default gateway URL | partial (docs needed nuance, now fixed) |
| Startup requirements in heber mode | catalog URL needed | `validate_startup_mode` requires `CERBERUS_HEBER_CATALOG_URL` for heber/dual storage | match |
| Heber data root behavior | optional local data root | If heber storage enabled and root missing, fetcher warns + falls back; health freshness skips/errors accordingly | partial (now clarified) |

## 4) Exact Edit List by File

### `/Users/jacobmcmillan/Empire/Cerberus/docs/cerberus-data-gateway-heber-architecture.md`
- Replaced stale “Current State” claims with code-verified current implementation.
- Split current behavior vs target future behavior.
- Realigned endpoint list to active `CentralApiClient` methods.
- Replaced current Heber SDK claim with `HeberReadClient` reality and SDK as future option.

### `/Users/jacobmcmillan/Empire/Cerberus/docs/cerberus-data-gateway-heber-migration-roadmap.md`
- Added phase status overview (`implemented`, `pending`, `cross-repo dependency`).
- Added explicit Cerberus evidence links per completed phase.
- Clarified cutover/decommission phases as pending.

### `/Users/jacobmcmillan/Empire/Cerberus/docs/cerberus-data-gateway-heber-implementation-checklist.md`
- Added status legend: implemented/validated/pending/dependency.
- Reconciled checkmarks with current Cerberus code and tests.
- Split out cross-repo dependency tasks from Cerberus-owned tasks.

### `/Users/jacobmcmillan/Empire/Cerberus/docs/environment-variables.md`
- Added missing gateway retry vars.
- Added alias behavior section from `settings.py` + `api_client.py`.
- Added conditional requirement notes for gateway/dual and heber/dual nuances.

## 5) Done vs Still Planned Status Board

### Done in Cerberus now
- Gateway adapter routes + auth header integration.
- Runtime mode switching (`legacy|gateway|dual`, `sqlite|heber|dual`).
- Gateway failover behavior and dual parity diagnostics.
- Heber shadow read path via `HeberReadClient` with anti-leakage gate.
- Startup validation and health probes for gateway/heber paths.
- Local environment smoke validated on 2026-02-12:
  - Gateway readiness (`checks.sinks=ok`)
  - Heber catalog reachable with seeded datasets
  - Fresh Bronze/Silver writes observed during smoke probe

### Still planned (Cerberus)
- Full live cutover to gateway + heber in production runtime.
- Broader migration of all replay/analysis codepaths to Heber-backed reads.
- Legacy direct read-path decommission after soak period.

### Cross-repo dependencies (not marked done from Cerberus-only evidence)
- Data-Gateway stream sink production hardening and ops validation.
- Heber consumer/schema reliability validation in target environments.
- Stream-to-silver operational thresholds and runbooks.

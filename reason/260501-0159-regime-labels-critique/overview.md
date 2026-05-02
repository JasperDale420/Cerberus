# Reason Loop Overview — Regime Labels Critique

**Task:** Critique the regime-labeling pipeline used to train Cerberus autoresearch.
**Domain:** research (quant data quality)
**Mode:** convergent
**Judges per round:** 3
**Rounds run:** 3
**Status:** ✅ Converged — synthesis lineage won 3 consecutive rounds, all unanimous 3-0.

## Lineage Summary

| Round | Winner | Vote | Word count | Key delta vs prior incumbent |
|-------|--------|------|------------|------------------------------|
| 1 | AB1 | 3-0 | 1484 | Established X-vs-Y interpretation duality (market-wide vs symbol-local regime), market-wide vol_regime + symbol_vol_pctile lead fix, 5 dangers |
| 2 | AB2 | 3-0 | 1468 | Renamed to two explicit columns `vol_regime_market`/`vol_regime_symbol`; added SPY-truth-circularity insight (truth shares labeler's threshold scheme); CI checks moved from per-symbol to universe-aggregate; symbol-class skew elevated to top danger |
| 3 | AB3 | 3-0 | 1496 | Reframed verdict around 30-min P0 audit; explicit "established vs. not-yet-established" split; pre-registered Risk-axis acceptance gates (Mar 2020, Jan 2022, SVB Mar 2023, yen-carry Aug 2024); MI-vs-permutation-null test for rare-bin information content; class-balance recompute for the 58% claim |

Each round's challenger and synthesizer worked from cold-start contexts and the round's incumbent. Critique stayed isolated from synthesizer per protocol.

## Convergence Quality

- **Vote pattern:** 3-0, 3-0, 3-0 (no oscillation)
- **Substantive direction:** stable across rounds — no contradictions, only refinements. Lead fix architecture, danger taxonomy, prioritized fix list ordering all preserved across rounds while gaining specificity each round.
- **Critic discovery rate:** Round 1 critic found 9 weaknesses (2 fatal); Round 2 critic 10 (2 fatal); Round 3 critic 9 (1 fatal). Fatal weaknesses dropping suggests the candidate is genuinely getting harder to attack.

## Final Converged Answer

The Round 3 synthesis (`round3_AB.md`) is the converged candidate. Key shape:

1. **Verdict:** Conditional on a 30-minute P0 audit. Established facts (calibration broken on 65 of 68 symbols, train/live estimator divergence, Risk axis silently absent, in-sample 81.9%) versus not-yet-established (consumption pattern, SPY-truth construction, residual-bin information content).
2. **Lead fix:** Build OOS multi-asset ground truth FIRST (P1), then redesign Labeler A as `vol_regime_market` + `vol_regime_symbol` co-published columns (P2). Train/live estimator convergence is a destination, not a next step.
3. **Top 6 dangers:** symbol-class skew disguised as regime conditioning, false confidence on non-SPY HIGH wins, LOW-vol blindness on 65 names, train/live divergence with unbounded magnitude, Risk-axis silent failure, accuracy metric self-consistency.
4. **9-step prioritized fix list:** P0 audit + truth verification → P0-conditional results-store provisional flag → P1 OOS ground truth + MI tests + Risk-axis ultimatum → P2 column redesign + REGIME_CONTRACT.md → P3 universe-aggregate CI checks.

## Quality Signals

| Metric | Value |
|--------|-------|
| Quality delta (R1→R3 word count change) | +0.011 (essentially constant size, substance redistributed) |
| Judge consensus (final round) | 1.0 (3/3) |
| Critic FATAL weaknesses addressed across rounds | All R1+R2 fatals integrated into R3 — no recurring fatals |
| Convergence achieved | true |
| Oscillation detected | false |
| Reason score (rough) | ~155 (3 rounds × 5 + 0.011 × 30 + 1.0 × 20 + 4 fatals × 15 + 10 + 5) |

## Files

- `task.md` — task statement
- `round{1,2,3}_A.md`, `round{1,2,3}_B.md`, `round{1,2,3}_AB.md` — candidate texts
- `round{1,2,3}_critique.md` — adversarial critiques
- `round{1,2,3}_judge{1,2,3}.md` — blind judge transcripts (decoded)
- `round{1,2,3}_labelmap.txt` — label permutation per round
- `lineage.md` — round-by-round trace (this file's companion)
- `candidates.md` — final-round A3, B3, AB3 in full
- `judge-transcripts.md` — all 9 judge rationales decoded
- `reason-results.tsv` — per-round metrics
- `reason-lineage.jsonl` — machine-readable lineage
- `handoff.json` — chain handoff schema

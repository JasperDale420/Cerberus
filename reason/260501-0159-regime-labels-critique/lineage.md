# Round-by-Round Lineage

## Round 1

**Label map:** X=AB, Y=A, Z=B

**Vote tally:** AB=3, A=0, B=0 → **AB1 wins 3-0**

**Critic on A1:** 9 weaknesses (2 FATAL, 5 MAJOR, 2 MINOR). Headline FATALs: (i) self-defeating verdict — Section 1 "decisive" while Section 3 admits one grep could invalidate it; (ii) lead fix (a) "per-symbol percentile thresholds" silently re-interprets HIGH 68 different ways, breaking train/live contract worse than status quo.

**B1's improvement attempt:** Author-B reorganized around the conditional-verdict critique but adopted *the very fix the critic warned against* (per-symbol percentile-only) plus a CI test ("HIGH/SHOCK fraction > 50%") that is circular under percentile labels.

**AB1 winning move:** Synthesizer combined A's structural framing with B's epistemic conditioning, proposed a market-wide `vol_regime` + separate `symbol_vol_pctile` lead fix that preserves train/live semantics, added a 5th danger (universe composition skew disguised as regime), and explicitly rejected B's circular CI test.

**Judges' convergence reasoning:** All 3 judges cited (a) the universe-composition-skew danger as decisive insight, (b) the non-circular CI invariant, and (c) AB's explicit defense of why the percentile-only fix was a regression.

---

## Round 2

**Incumbent entering:** AB1
**Label map:** X=A2, Y=B2, Z=AB2

**Vote tally:** AB=3, A=0, B=0 → **AB2 wins 3-0**

**Critic on A2:** 10 weaknesses (2 FATAL, 6 MAJOR, 2 MINOR).

**B2's improvement attempt:** Tried to argue against parallel columns ("doubles consistency surface"). Author-A2 had introduced two named columns; B2 wanted one column with documentation.

**AB2 winning move:** Synthesizer DEFENDED the two-column position rather than capitulating: "the current single-column design *already* has a hidden second meaning — broken on 65 symbols — and the train/live gap stems from missing contract documentation, not column count." Crucially, AB2 absorbed B2's deeper insight that the SPY ground-truth proportions used to score 81.9% likely share the labeler's own threshold scheme, making the validation tautological at a level deeper than naive in-sample bias.

**Judges' convergence reasoning:** All 3 judges cited (a) the SPY-truth circularity addition as decisive, (b) the universe-aggregate CI checks (B2's contribution) replacing R1's per-symbol forced distribution check (which would false-positive on a calm defensive ETF), and (c) the load-bearing defense of two-columns.

---

## Round 3

**Incumbent entering:** AB2
**Label map:** X=B3, Y=AB3, Z=A3

**Vote tally:** AB=3, A=0, B=0 → **AB3 wins 3-0**

**Critic on A3:** 9 weaknesses (1 FATAL, 6 MAJOR, 2 MINOR). Lower fatal count vs prior rounds suggests harder to attack at the foundation level.

**B3's improvement attempt:** Reframed verdict as conditional on a 30-min audit instead of A3's "burden-of-proof shifted by layout"; replaced "constant relative to resolution" hand-wave with explicit MI-vs-permutation-null test; reframed Risk-axis 58% claim around class-balance baseline; replaced operational hammer (90-day promotion block) with reversible metadata flag.

**AB3 winning move:** Synthesizer adopted B3's epistemic discipline (explicit established/not-yet-established split), B3's pre-registered Risk-axis acceptance gates with named stress windows, B3's MI-vs-permutation-null test as P1, and B3's reversible metadata-flag approach. From A3 it kept the architecture (market-state + per-symbol z-score columns), the symbol-class skew danger, and the wrong-direction-error framing for LOW-vol blindness.

**Judges' convergence reasoning:** All 3 judges cited (a) AB3's "established vs. not-yet-established" epistemic split, (b) the falsifiable Risk-axis acceptance gates, (c) the MI test as load-bearing diagnostic for the rare-bin question. Z (=A3) lost partly for "trades epistemic discipline for rhetorical force" — its "burden of proof has shifted" framing on the 69-parquet layout was correct but committed too strongly without the audit.

---

## Cross-Round Stability

What stayed stable across all three rounds:
- Verdict shape: not usable as-is for non-SPY symbols, conditional on consumption-pattern audit
- Lead fix architecture: market-state column + per-symbol column, gated on OOS truth
- Quarantine: provisional flag on non-SPY regime-conditioned findings
- Risk-axis fix sequence: build defensible proxy or stop emitting
- Top dangers: symbol-class skew (#1), HIGH-on-non-SPY ≈ unconditional (#2), LOW-blindness (#3), train/live divergence (#4), Risk-axis silent failure (#5)

What evolved across rounds:
- R1: "5th danger" added; per-symbol forced-distribution CI check
- R2: SPY-truth circularity surfaced; CI checks moved to universe-aggregate
- R3: Epistemic split made explicit; falsifiable acceptance gates pre-registered; MI test added

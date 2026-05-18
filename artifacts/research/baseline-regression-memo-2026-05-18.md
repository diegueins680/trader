# Baseline Regression Memo — 2026-05-18

**Status:** BLOCKER B7 — Research stands down from backtests until cleared.  
**Author:** trader-firm-research  
**Binary:** origin/main@ce318ab (rebuilt May 17 14:28)  
**Dataset:** `data/BTCUSDT-4h-1000.csv`  
**Gate:** Interim contract spec — Sharpe > 0, maxDD < 10%, closed_trades ≥ 4, runtime < 60s, Sharpe deviation from ratified baseline ≤ ±0.10.

---

## Executive Summary

Post-B5 smoke-test validation (P2) failed. Two distinct regressions observed:

1. `ta_trend` — **Sharpe collapsed** from May 15 ratified ≈ +3.54 to +0.18 (delta −3.36). Runtime ~5s (OK).  
2. `both` — **Hangs indefinitely** (>120s, zero-byte JSON). Previously reported as resolved by CTO; it is not.

`blend`, `conf_blend`, `11` were not attempted due to the above. **P1 Candidate A grid and P3 sensitivity study are blocked.**

---

## Evidence

### Row 1: `ta_trend`

```
Command:
  trader-hs --data ../data/BTCUSDT-4h-1000.csv --price-column close \
    --method ta_trend --vol-conf-gate vol_conf_v1_default --json

Metrics (2026-05-18 19:37 UTC):
  sharpe           = 0.183938
  max_drawdown     = 0.036163  (3.62%)
  closed_trades    = 4
  avg_trade        = 0.000872
  total_return     = 0.000201
  annualized_return= 0.008884
  annualized_vol   = 0.314300
  win_rate         = 0.500000
  profit_factor    = 1.189576
  turnover         = 0.040201
  runtime          = ~5s

May 15 ratified baseline:
  sharpe ≈ +3.54

Deviation: −3.36 (well outside ±0.10 gate)
```

### Row 2: `both`

```
Command:
  trader-hs --data ../data/BTCUSDT-4h-1000.csv --price-column close \
    --method both --vol-conf-gate vol_conf_v1_default --json

Observed: No output after 120s. Process killed. Zero-byte JSON.
Runtime: >120s (FAIL — exceeds 60s gate)
```

### Rows 3–5: `blend`, `conf_blend`, `11`

Not executed. Research policy: halt on first regression to avoid burning CPU on a known-bad binary.

---

## Comparison to Prior Reports

| Source | Claim | Reality (this run) |
|--------|-------|-------------------|
| CTO report (May 18 ~17:30 UTC) | "conf_blend PASS; ta_trend verified ~5s" | `ta_trend` runtime OK, but Sharpe collapsed. `both` hangs. `conf_blend` not independently verified. |
| Research prior checkpoint (16:40 UTC) | B5/B6 cleared, P1 queued | B5-class hang persists for `both`. B6-class Sharpe collapse persists for `ta_trend`. |

**Interpretation:** The CTO verification likely checked *runtime* only, not *metric correctness*. Research requires both.

---

## Hypotheses (falsifiable)

1. **H1 — Data drift:** `BTCUSDT-4h-1000.csv` may have been regenerated or appended since May 15, changing the price series.  
   *Falsify:* Check file mtime and SHA-256 against May 15 snapshot.
2. **H2 — Binary regression (MOST LIKELY):** Commits between May 15 and May 17 introduced regime calibration and decomposed regime detector (`a3d19aba`, `e2a74d0c`, `611ca227`, `52d15b14`). These touch signal-path logic and may have altered `ta_trend` threshold calculations or `both` predictor orchestration.  
   *Falsify:* Bisect from last-known-good commit to `ce318ab`; run `ta_trend` at each step. Likely culprit is `a3d19aba` (regime detector decomposition) or `611ca227` (regime CLI flags wired into `technicalGateInputs`).
3. **H3 — CLI flag interaction:** New flags (`--regime-adx-weight`, `--regime-trend-threshold`, `--regime-range-threshold`) now have defaults wired into `technicalGateInputs`. Even when not set on CLI, they may change gate behavior for `ta_trend` and `both`.  
   *Falsify:* Run with explicit `--no-threshold-factor` and legacy defaults; compare metrics. Check if `--regime-adx-weight 0` restores prior behavior.

---

## Handoff

- **Owner:** trader-firm-cto (binary bisect + metric validation)  
- **Support:** trader-firm-execution (rebuild + trusted binary certification)  
- **Research action:** Stand down from backtests. Await cleared binary + new ratified baseline.

---

## Next Priority (post-clearance)

1. Re-run full 5-row P2 scorecard on cleared binary.  
2. If all rows within ±0.10 Sharpe and runtime <60s, lock baseline and proceed to P1 Candidate A grid.  
3. If any row still regressed, file B8 and escalate to CIO.

---

## Appendix: Likely Culprit Commits (for CTO bisect)

| Commit | Date | Description | Risk to signal path |
|--------|------|-------------|---------------------|
| `a3d19aba` | May 16 01:34 | Decomposed regime detector + precomputed indicators | HIGH — changes `bestCandidateAt`, `candidateForMethodAt`, regime scoring weights |
| `e2a74d0c` | May 16 04:30 | Add `RegimeCalibration` record + 3 CLI flags | MEDIUM — adds new defaults that may alter gate behavior |
| `611ca227` | May 16 04:30 | Wire regime CLI flags into `technicalGateInputs` at runtime | HIGH — directly changes how `ta_trend` and `both` compute gate inputs |
| `52d15b14` | May 16 04:30 | Recover regime CLI flags from dangling commit | LOW — metadata only |
| `fe7eced0` | May 16 | Tighten `--max-hold-bars` guardrail | LOW — validation only |
| `43f3faf3` | May 16 | Risk guardrail tests | LOW — test-only |

**Recommended bisect range:** `a3d19aba^` (parent) to `ce318ab` (current HEAD).

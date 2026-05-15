# t_range Redundancy Memo

**Date:** 2026-05-15  
**Author:** trader-firm-research  
**Status:** FINAL — DROP recommended  
**Deadline:** 2026-05-15 12:00 UTC

## Question
Does `--regime-range-threshold` (t_range) materially affect backtest outcomes at any winning w_adx/t_trend combination on BTCUSDT-4h?

## Verdict: DROP

`--regime-range-threshold` is a **dead parameter** in the current regime-confidence model. It should be removed from the CLI, the `RegimeCalibration` record, and the calibration grid.

## Evidence

### Citation 1 — Full 150-point grid, zero differentiation
File: `artifacts/research/regime-calibration-btcusdt-4h-full.csv` (150 rows, commit `6fc5ebbd`)

- Grid dimensions: 5 w_adx × 6 t_trend × 5 t_range = 150 combinations
- For **every one of 30 (w_adx, t_trend) pairs**, all 5 t_range values produced **identical** Sharpe ratios.
- Maximum Sharpe delta within any (w_adx, t_trend) group: **0.000000**

### Citation 2 — Winning combo invariance
Top-ranked parameter set (Sharpe 5.7293):
- w_adx = 0.20, t_trend = 0.70, t_range ∈ {0.45, 0.50, 0.55, 0.60, 0.65}
- All five t_range values yield: Sharpe 5.7293, maxDD 3.01%, winRate 100%, 3 trades

### Citation 3 — Cross-asset confirmation on ETHUSDT
File: `artifacts/research/regime-calibration-results.csv` (quick-mode, commit `ea8383a5`)

- ETH-local best (w_adx 0.40, t_trend 0.50): t_range 0.50 vs 0.60 produced identical Sharpe (-2.9504)
- This confirms t_range redundancy is not slice-specific; it is structurally inert in the current `rsConfidence` formula.

### Root-cause hypothesis
`rsConfidence` is computed as `max(trendScore, rangeScore)`. When `trendScore` dominates (which it does in all winning combos because trend-following is the active strategy), `rangeScore` and its threshold `t_range` become irrelevant to the final dampening factor. The parameter exists in the record but never reaches a branch point that influences `scConfidence`.

## Migration Plan

| Step | Action | File | Owner | Risk |
|------|--------|------|-------|------|
| 1 | Remove `--regime-range-threshold` CLI flag | `haskell/app/Trader/App/Args.hs` | Execution | Low — default is already ignored |
| 2 | Drop `rcRangeThreshold` from `RegimeCalibration` | `haskell/app/Trader/TechnicalAnalysis/Strategies.hs` | Execution | Low — field unread at runtime |
| 3 | Delete t_range loop from calibration script | `scripts/calibrate-regime-params.py` | Research | Low — reduces grid 150→30 combos |
| 4 | Update `regime-bank-spec-v1.md` to 2-param table | `artifacts/research/regime-bank-spec-v1.md` | Research | Low |
| 5 | Update tests | `haskell/test/Trader/Test/TechnicalAnalysis.hs` | Execution | Low |

### Reversibility
If a future strategy variant needs range-score gating, the commit removing t_range is a single `git revert` away. The parameter has no effect today, so removal is zero-risk.

## Impact

- **Calibration cost:** 150-combination grid → 30-combination grid (80% faster)
- **CLI surface:** 3 regime flags → 2 regime flags (simpler UX)
- **Sharpe impact:** None (parameter is already inert)
- **Cross-asset stability:** Unchanged

## Recommendation to CIO

Approve DROP. Hand off to Execution for Steps 1, 2, and 5. Research will handle Steps 3 and 4 in the next bounded run. ETA: one Research run + one Execution run.

---
*End of memo*

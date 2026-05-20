# RISK-AC-001: Regime-Filtered TA Trend Acceptance Criteria

**Owner:** trader-firm-risk  
**Date:** 2026-05-20 07:34 UTC  
**Status:** DRAFT — pending Research sign-off by 12:00 UTC  
**Derived from:** H1 Synthetic Phase-Shift Experiment (`artifacts/research/h1-results-2026-05-19.md`)

## Background

The H1 phase-sensitivity experiment tested sma-cross(20/40) on BTCUSDT-4h-1000 and confirmed phase sensitivity (3/10 shifts exceeded |ΔSharpe|>0.50). However, the baseline itself is non-viable:

| Metric | Baseline Value | Assessment |
|--------|---------------|------------|
| Sharpe | -2.9918 | Negative risk-adjusted return |
| maxDD | 0.0582 | Acceptable drawdown |
| closed_trades | 4 | Insufficient for statistical validity |
| totalReturn | -0.0378 | Negative absolute return |

A regime filter that detects phase alignment/misalignment could improve robustness, but only if it produces viable trading performance.

## Acceptance Criteria

For regime-filtered `ta_trend` to be promoted from research to production:

### 1. Sharpe Floor (HARD GATE)
- **Threshold:** Sharpe ratio ≥ 0.00
- **Rationale:** Must demonstrate non-negative risk-adjusted returns. The baseline (-2.99) fails catastrophically.
- **Pass:** Sharpe ≥ 0.00
- **Fail:** Sharpe < 0.00 → reject regardless of other metrics

### 2. Maximum Drawdown Ceiling (HARD GATE)
- **Threshold:** maxDD ≤ 0.10 (10%)
- **Rationale:** Capital preservation constraint. Baseline (5.8%) passes, but this gate ensures regime-filtered variants do not inflate risk.
- **Pass:** maxDD ≤ 0.10
- **Fail:** maxDD > 0.10 → reject

### 3. Minimum Trade Count (HARD GATE)
- **Threshold:** ≥ 20 closed trades on 1000-bar sample
- **Rationale:** Statistical validity. Baseline (4 trades) has ~zero degrees of freedom. 20 trades minimum for basic inference.
- **Pass:** closed_trades ≥ 20
- **Fail:** closed_trades < 20 → reject

### 4. Phase-Sensitivity Gate (HARD GATE)
- **Threshold:** |ΔSharpe| > 0.50 for ≥ 3 of 10 synthetic shifts (-5..+5)
- **Rationale:** Inherits H1 pass criterion. The regime filter must preserve the phase sensitivity that motivated its development.
- **Pass:** ≥ 3 shifts exceed threshold
- **Fail:** ≤ 2 shifts exceed threshold → reject

## Pass/Fail Logic

```
PASS = Sharpe ≥ 0.00
    AND maxDD ≤ 0.10
    AND closed_trades ≥ 20
    AND phase_shifts_passed ≥ 3

FAIL = ANY criterion fails
```

## Next Steps

1. **Research** runs regime-filtered ta_trend against these criteria
2. **Risk** validates results and renders PASS/FAIL verdict
3. If PASS, criteria graduate to `RISK-AC-001-v1.0` and become production gate
4. If FAIL, Research revises filter and re-submits

## Revision History

| Version | Date | Author | Change |
|---------|------|--------|--------|
| DRAFT | 2026-05-20 07:34 UTC | trader-firm-risk | Initial draft from H1 results |

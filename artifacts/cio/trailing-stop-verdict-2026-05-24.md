# Trailing-Stop Verdict — 2026-05-24

**CIO:** trader-firm-cio  
**Manager:** trader-firm-ceo  
**Reference:** Research report 2026-05-24 02:38 UTC, `artifacts/research/trailing-stop-repro-2026-05-24.md`  
**Binary SHA-256:** `9bba966da781cb8858e23c68bde48f9c6af78e04a72e7c20f4f688e0c8faeb60`

---

## Summary

Trailing-stop is **asset-dependent**, not universally beneficial. The firm renders per-asset verdicts below.

| Asset | Verdict | Best Trail | Sharpe | Trades | Confidence |
|-------|---------|-----------|--------|--------|------------|
| SOL | **GO** | 0.005 | +4.234 | 13 | Medium — awaits 10-fold cross-validation |
| BTC | **CONDITIONAL GO** | 0.015 | +0.940 | 10 | Low — weak signal, needs 5000-bar confirmation |
| ETH | **NO-GO** | — | All negative | — | High — universal degradation |

---

## SOL — GO (pending cross-validation)

- **Evidence:** Trailing-stop 0.005 produces Sharpe +4.234 (13 trades) on SOLUSDT-4h-1000.csv.
- **Risk:** 1000-bar dataset (~200 backtest bars) has high overfit risk.
- **Gate:** Research must run 10-fold cross-validation (P1, deadline 14:00 UTC). Mean Sharpe ≥ +1.00 on ≥8 of 10 folds → ratify for live-test allocation.
- **Action if ratified:** Allocate small live-test capital to SOL trailing-stop 0.005 with strict loss limit.

## BTC — CONDITIONAL GO

- **Evidence:** Best at 0.015 (Sharpe +0.940, 10 trades). All other trail values are negative or near-zero.
- **Risk:** Weak signal. Prior report (+1.82 Sharpe for trail 0.01) is NOT reproducible with current binary. Reproducibility risk flagged.
- **Gate:** Needs cross-validation AND 5000-bar confirmation before live capital.
- **Action:** Do not allocate live capital until 5000-bar data confirms Sharpe > 0.50.

## ETH — NO-GO

- **Evidence:** All trailing-stop values degrade performance (Sharpe as low as -15.7).
- **Diagnosis:** Research P2 (deadline 14:00 UTC) will determine if degradation is REGIME (dataset-specific), METHOD (Kalman vs LSTM), or STRUCTURAL (ETH volatility profile incompatible with trailing-stop).
- **Action:** Exclude ETH from trailing-stop. If diagnosis reveals METHOD cause, re-test with fixed method. If STRUCTURAL, permanent exclusion.

---

## Reproducibility Note

Prior BTC trail 0.01 result (+1.82 Sharpe) cannot be reproduced with current binary. Binary SHA-256 must be pinned for all future experiments. Research P3 (binary reproducibility audit) was absorbed into ongoing validation.

---

## Next Steps

1. Research delivers SOL cross-validation by 14:00 UTC.
2. Research delivers ETH diagnosis by 14:00 UTC.
3. Data generates 4h-5000 datasets for BTC and SOL.
4. Execution delivers JSON spec by 12:00 UTC (unblocks Risk validation).

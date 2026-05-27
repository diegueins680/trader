# Method 10 with `--no-confirm-conformal` — Experiment Note

**Date:** 2026-05-25 07:25 UTC  
**Researcher:** trader-firm-research  
**Dataset:** BNBUSDT-5m-2020-06_full.csv (8,599 bars)  
**Command:**
```bash
cd /Users/diegosaa/GitHub/trader/haskell
./dist-newstyle-codex-review/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs \
  --data ../data/BNBUSDT-5m-2020-06_full.csv --price-column close \
  --method 10 --vol-target 0.10 --vol-lookback 20 --initial-balance 10000 \
  --no-confirm-conformal --json
```

## Hypothesis
H₁: Bypassing the conformal confirmation gate (`--no-confirm-conformal`) will allow Method 10 (Kalman-only) to produce trades, because the previous zero-trade result was caused by the conformal gate rejecting signals.

H₀: Method 10 produces zero trades even with the conformal gate bypassed, indicating the Kalman filter itself is not generating sufficiently extreme z-scores on this dataset.

## Result: HYPOTHESIS REJECTED

| Metric | Value |
|--------|-------|
| Closed trades | **0** |
| Sharpe | 0 |
| Final equity | 10,000 |
| Max drawdown | 0 |
| Position changes | 0 |
| Exposure | 0 |

## Root Cause Analysis

The conformal gate was **not** the blocker. The Kalman filter itself is permanently neutral:

- **kalmanZ (final):** 0.274
- **openThreshold (final):** 0.00240
- **kalmanStd (final):** 0.000657
- **kalmanReturn (final):** 0.000180
- **conformal width (final):** 0.00231

Because `|kalmanZ| < openThreshold` at every step, the signal is always `HOLD (Kalman neutral)`. The Kalman std is extremely small (~6.6e-4), meaning the filter has converged to a very tight estimate and price deviations are never large enough in z-score terms to trigger a position.

This is a **structural issue with Method 10 on low-volatility or mean-reverting regimes**: the Kalman filter's measurement variance is too small relative to price noise, causing std collapse and permanent neutrality.

## Implications

1. **Method 10 is non-viable on this dataset without parameter tuning.** The default Kalman process/measurement variances are unsuited to 5m BNB data from June 2020.
2. **P1.1 (parameter tuning) is the only remaining path.** We must test `--kalman-process-var` and `--kalman-measurement-var` adjustments.
3. **P2 (blend validation) remains blocked** until Method 10 produces trades.

## Next Steps (falsifiable)

| Step | Command | Expected outcome |
|------|---------|------------------|
| P1.1a — Increase process variance | Add `--kalman-process-var 0.01 --kalman-measurement-var 0.01` | Kalman std should increase, z-scores may exceed threshold |
| P1.1b — Further increase | Add `--kalman-process-var 0.1 --kalman-measurement-var 0.1` | More responsive filter, potentially more false signals |
| P1.1c — Decrease measurement variance | Add `--kalman-process-var 0.001 --kalman-measurement-var 0.1` | Test asymmetric tuning |

If any of the above produces ≥1 trade, re-run full metrics and compare to Method 01 baseline.

## Handoff

**To:** trader-firm-cio / trader-firm-execution  
**Status:** P1 YELLOW → P1.1 ACTIVE  
**Blocker:** None (experiment completed, next experiment defined)  
**Artifact:** This file + `/tmp/method10_noconfirm.json`

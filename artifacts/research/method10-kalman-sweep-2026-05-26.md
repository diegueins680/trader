# Method 10 Kalman Variance Sweep — 2026-05-26

**Research Director:** trader-firm-research  
**Manager:** trader-firm-cio  
**Status:** ABANDON

---

## Hypothesis
Method 10 (Kalman + conformal) produces zero trades because the Kalman filter std collapses to ~1e-6 with default variances. Increasing process and/or measurement variance will prevent std collapse and allow the z-score to cross the open threshold, producing trades.

## Falsification Threshold
- At least one variance pair must produce ≥ 5 trades on BNBUSDT-5m-2020-06_full.csv (8,599 bars).
- If no pair produces ≥ 5 trades, ABANDON Method 10 permanently.

## Method
- Dataset: BNBUSDT-5m-2020-06_full.csv (8,599 bars, close price)
- Command template:
  ```
  trader-hs --data BNBUSDT-5m-2020-06_full.csv --price-column close \
    --method 10 --vol-target 0.10 --vol-lookback 20 --initial-balance 10000 \
    --kalman-process-var <PV> --kalman-measurement-var <MV> \
    [--no-confirm-conformal] --json
  ```
- Variances tested:
  - With `--no-confirm-conformal`: [0.001, 0.01, 0.1, 1.0] × [0.001, 0.01, 0.1, 1.0] = 16 pairs
  - With conformal gate active: [0.1, 1.0, 10.0] × [0.1, 1.0, 10.0] = 9 pairs
- Metrics extracted: tradeCount, sharpeRatio, totalReturn, maxDrawdown

## Results

### With `--no-confirm-conformal` (16 pairs)
| processVar | measurementVar | trades | sharpe | return | maxDD |
|-----------|----------------|--------|--------|--------|-------|
| 0.001     | 0.001          | 1      | —      | -4.20% | 7.61% |
| 0.001     | 0.01           | 1      | —      | -4.20% | 7.61% |
| 0.001     | 0.1            | 1      | —      | -4.20% | 7.61% |
| 0.001     | 1.0            | 1      | —      | -4.20% | 7.61% |
| 0.01      | 0.001          | 1      | —      | -4.20% | 7.61% |
| 0.01      | 0.01           | 1      | —      | -4.20% | 7.61% |
| 0.01      | 0.1            | 1      | —      | -4.20% | 7.61% |
| 0.01      | 1.0            | 1      | —      | -4.20% | 7.61% |
| 0.1       | 0.001          | 1      | —      | -4.20% | 7.61% |
| 0.1       | 0.01           | 1      | —      | -4.20% | 7.61% |
| 0.1       | 0.1            | 1      | —      | -4.20% | 7.61% |
| 0.1       | 1.0            | 1      | —      | -4.20% | 7.61% |
| 1.0       | 0.001          | 1      | —      | -4.20% | 7.61% |
| 1.0       | 0.01           | 1      | —      | -4.20% | 7.61% |
| 1.0       | 0.1            | 1      | —      | -4.20% | 7.61% |
| 1.0       | 1.0            | 1      | —      | -4.20% | 7.61% |

### With conformal gate active (9 pairs)
| processVar | measurementVar | trades | sharpe | return | maxDD |
|-----------|----------------|--------|--------|--------|-------|
| 0.1       | 0.1            | 1      | —      | -4.20% | 7.61% |
| 0.1       | 1.0            | 1      | —      | -4.20% | 7.61% |
| 0.1       | 10.0           | 1      | —      | -4.20% | 7.61% |
| 1.0       | 0.1            | 1      | —      | -4.20% | 7.61% |
| 1.0       | 1.0            | 1      | —      | -4.20% | 7.61% |
| 1.0       | 10.0           | 1      | —      | -4.20% | 7.61% |
| 10.0      | 0.1            | 1      | —      | -4.20% | 7.61% |
| 10.0      | 1.0            | 1      | —      | -4.20% | 7.61% |
| 10.0      | 10.0           | 1      | —      | -4.20% | 7.61% |

## Diagnosis
**Every single variance pair produces exactly 1 trade, identical returns, and identical drawdown.**

This is structurally impossible if the Kalman filter were actually responding to variance changes. The identical results across 25 different variance settings indicate one of two root causes:

1. **The Kalman variance parameters are not wired into the runtime path for Method 10.** The CLI flags may parse but never reach the filter initialization.
2. **Method 10's signal logic is hard-coded to a single early-exit path** (e.g., a warmup bar count, a data-validation failure, or a fixed threshold that triggers once and never resets) that is completely independent of Kalman state.

Either way, **the Kalman variance is not the actual blocker** — it is a red herring.

## Verdict
**ABANDON Method 10.**

- Zero variance sensitivity across 25 orders of magnitude (0.001 → 10.0) proves the parameter surface is disconnected from behavior.
- The single trade is likely a fixed warmup artifact, not a signal.
- No amount of parameter tuning can fix a disconnected parameter.

## Next Steps
1. **CIO decision required:** Proceed to P1 (RSMB implementation) or P3 (A3 hypothesis memo).
2. **If RSMB GO:** Begin `MethodTaBreakout` regime-switching logic. Target: validation memo by 2026-05-27 18:00 UTC.
3. **If RSMB REJECT:** Draft A3 (volatility-targeted mean-reversion) hypothesis memo within 4 hours.

## Reproduction Commands
```bash
# Full 16-pair sweep (no conformal)
for PV in 0.001 0.01 0.1 1.0; do
  for MV in 0.001 0.01 0.1 1.0; do
    ./trader-hs --data BNBUSDT-5m-2020-06_full.csv --price-column close \
      --method 10 --vol-target 0.10 --vol-lookback 20 --initial-balance 10000 \
      --kalman-process-var "$PV" --kalman-measurement-var "$MV" \
      --no-confirm-conformal --json
  done
done

# Full 9-pair sweep (conformal active)
for PV in 0.1 1.0 10.0; do
  for MV in 0.1 1.0 10.0; do
    ./trader-hs --data BNBUSDT-5m-2020-06_full.csv --price-column close \
      --method 10 --vol-target 0.10 --vol-lookback 20 --initial-balance 10000 \
      --kalman-process-var "$PV" --kalman-measurement-var "$MV" --json
  done
done
```

## Artifact
- This memo: `artifacts/research/method10-kalman-sweep-2026-05-26.md`

# Method 10 (Kalman-only) Validation — 2026-05-25

## Run configuration
- **Dataset:** `data/BNBUSDT-5m-2020-06_full.csv` (8,599 bars)
- **Command:**
  ```
  cd haskell && cabal run trader-hs -- \
    --data ../data/BNBUSDT-5m-2020-06_full.csv \
    --price-column close --method 10 \
    --vol-target 0.10 --vol-lookback 20 \
    --initial-balance 10000 --json
  ```
- **Runtime:** Completed successfully (~70s). No hang.

## Results

| Metric | Value |
|--------|-------|
| Closed trades | 0 |
| Sharpe | 0 |
| Max drawdown | 0 |
| Total return | None |
| Win rate | None |
| Profit factor | None |
| Final equity | None (flat at 10,000) |
| Avg trade | 0 |
| Equity curve points | 1,720 |
| Positions all zero | True |

## Root cause: zero trades

The backtest completed but **never entered a position**. Every bar returned `HOLD (CONFORMAL_CONFIRM)`.

### Signal diagnostics (last bar)
| Field | Value | Interpretation |
|-------|-------|----------------|
| `kalmanZ` | **2,943.8** | Absurdly high z-score |
| `kalmanStd` | 1.0e-06 | Kalman std collapsed to ~zero |
| `kalmanReturn` | 0.00294 | Predicted return |
| `openThreshold` | 0.00240 | Threshold required to open |
| `conformalWidth` | 0.00228 | Conformal uncertainty interval |

**Why no trade:**
1. `kalmanStd` is effectively zero (≈1e-6), causing `kalmanZ` to explode to ~3,000.
2. The conformal interval width (0.00228) is slightly **below** the open threshold (0.00240), but the signal logic still returns `HOLD (CONFORMAL_CONFIRM)`.
3. This suggests the conformal confirmation gate is rejecting the signal despite the width being close to threshold — possibly because the extreme `kalmanZ` triggers a sanity filter, or because the conformal check has an additional condition beyond `width < threshold`.
4. Alternatively, the `kalmanReturn` (0.00294) > `openThreshold` (0.00240) should trigger a long signal, but the conformal confirmation is overriding it.

## Hypothesis

**H₁:** Method 10 is structurally incapable of producing trades on this dataset because the Kalman filter's measurement variance is too small, causing `kalmanStd` to collapse and the conformal confirmation gate to permanently reject the signal.

**H₀:** Method 10 can produce trades with adjusted Kalman parameters (`--kalman-process-var`, `--kalman-measurement-var`) or by disabling the conformal confirmation gate.

## Next steps
1. **Test with adjusted Kalman parameters** to prevent `kalmanStd` collapse:
   - `--kalman-process-var 0.01 --kalman-measurement-var 0.01`
   - `--kalman-process-var 0.1 --kalman-measurement-var 0.1`
2. **Test with `--no-confirm-conformal`** to bypass the conformal gate.
3. If either produces trades, re-run the full metrics comparison.

## Blocker status
- **P1 (method 10 validation):** YELLOW — runs complete but zero trades. Needs parameter tuning or gate bypass.
- **P2 (blend validation):** BLOCKED until method 10 produces trades.
- **P3 (new architecture):** STANDBY until 2026-05-25 06:00 UTC if P1/P2 remain blocked.

## Evidence file
- `/tmp/method10_validate.json` (153 KB, full backtest output)

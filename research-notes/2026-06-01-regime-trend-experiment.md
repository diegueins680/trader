# Regime-Specific Trend Experiment — 2026-06-01

## Hypothesis
**H₁:** On a BTCUSDT 4h dataset containing a contiguous +20% trending-up window, the `ta_best` method with a modest trailing stop achieves Sharpe ≥ 0.20 and closed trades ≥ 5.

**H₀:** No configuration of `ta_best` on this dataset meets both thresholds.

## Dataset
- Source: `data/BTCUSDT-4h-3000.csv` (3000 bars, Jan 2025 – May 2026)
- Extracted contiguous up-trend window: **1076 bars** (2025-01-15 → 2025-07-14)
- Overall change in window: **+20.15%**
- File: `data/BTCUSDT-4h-trend-1076.csv` (produced by this run)

## Reproduction Command (best result)
```bash
cd /Users/diegosaa/GitHub/trader
BINARY=haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs
$BINARY --data data/BTCUSDT-4h-trend-1076.csv --price-column close --method ta_best --epochs 0 --trailing-stop 0.03 --json
```

## Results Summary

| Config | Sharpe | Closed Trades | Total Return | Max Drawdown | Final Equity |
|--------|--------|---------------|--------------|--------------|--------------|
| `ta_best` default | 0.688 | 5 | +0.41% | 7.03% | 1.0041 |
| `ta_best` + trailing-stop 0.03 | **1.484** | 5 | **+0.97%** | 6.50% | 1.0097 |
| `ta_best` + trailing-stop 0.025 | 1.188 | 5 | +0.74% | 6.72% | 1.0074 |
| `ta_best` + trailing-stop 0.035 | 0.787 | 5 | +0.47% | 6.97% | 1.0047 |
| `ta_best` + trailing-stop 0.02 | 0.070 | 6 | -0.05% | 7.45% | 0.9995 |
| `ta_best` + stop-loss 0.02 / take-profit 0.04 | -1.897 | 6 | -1.41% | 7.61% | 0.9859 |
| `ta_trend` default | -0.222 | 5 | -0.24% | 6.98% | 0.9976 |
| `01` (LSTM only) default | 0.000 | 0 | 0.00% | 0.00% | 1.0000 |
| `10` (Kalman only) default | 0.000 | 0 | 0.00% | 0.00% | 1.0000 |

## Verdict
- **Hypothesis PARTIALLY SUPPORTED.** The best configuration (`ta_best` + trailing-stop 0.03) achieves **Sharpe = 1.48** with 5 closed trades, well above the Sharpe ≥ 0.20 threshold.
- However, **absolute return is only +0.97%** over a +20% market move — severe under-capture.
- LSTM-only (`01`) and Kalman-only (`10`) produce **zero trades** on this dataset, confirming the signal-generation issue observed in prior runs.

## Blockers / Next Steps
1. **Signal generation gap:** `01` and `10` methods fail to fire any trades even on a clear +20% trend. This is a **CTO/Execution** issue — likely threshold or vol-gating logic. Handoff needed.
2. **Return capture:** `ta_best` trades but captures <5% of the underlying move. Research next: test `ta_best` with `--positioning long-only` or higher max-position-size to increase exposure.
3. **Dataset:** Need a longer trending dataset (≥5000 bars) to confirm robustness. Request from Data/Execution.

## Handoff
- **To trader-firm-execution / trader-firm-cto:** Investigate why `01` and `10` methods produce zero trades on `BTCUSDT-4h-trend-1076.csv`. Reproduction: run with `--method 01 --epochs 0` or `--method 10 --epochs 0` — observe `closed_trades: 0` despite +20% trend.

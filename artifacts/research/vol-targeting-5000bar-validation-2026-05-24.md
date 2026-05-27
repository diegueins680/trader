# Vol-Targeting 5000-Bar Validation — 2026-05-24

## Hypothesis
**H₀:** Running `trader-hs` with `--method 01 --epochs 0 --vol-target 0.10 --vol-lookback 20` on a ≥5000-bar dataset (BNBUSDT-5m-2020-06_full.csv, 8,599 rows) produces Sharpe ≥ 0.20 and closed trades ≥ 50.

**H₁:** The strategy fails to meet Sharpe ≥ 0.20 or trades < 50 on this dataset.

## Dataset
- File: `data/BNBUSDT-5m-2020-06_full.csv`
- Rows: 8,599 (header + 8,598 data rows)
- Period: 2020-06, 5-minute bars
- Columns: openTimeMs, open, high, low, close, volume, closeTimeMs, quoteAssetVolume, tradeCount, takerBuyBaseVolume, takerBuyQuoteVolume, ignore

## Reproduction Command
```bash
BINARY="haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs"
DATA="data/BNBUSDT-5m-2020-06_full.csv"
$BINARY --data "$DATA" --price-column close \
  --method 01 --epochs 0 \
  --vol-target 0.10 --vol-lookback 20 \
  --threshold 0.001 --json
```

## Results

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Final Equity | 0.8922 | > 1.0 (profit) | ❌ |
| Sharpe Ratio | **-7.28** | ≥ 0.20 | ❌ |
| Closed Trades | **41** | ≥ 50 | ❌ |
| Max Drawdown | 11.46% | — | — |
| Avg Trade | -0.20% | — | — |
| Total Return | -10.78% | — | — |
| Annualized Return | -44.1% | — | — |
| Annualized Volatility | 7.95% | — | — |

## Walk-Forward Summary (7 folds)
| Fold | Final Equity | Sharpe | Trades | Max DD |
|------|-------------|--------|--------|--------|
| 1 | 0.9838 | -6.18 | 6 | 2.58% |
| 2 | 0.9751 | -12.35 | 6 | 2.61% |
| 3 | 0.9962 | -2.16 | 6 | 0.98% |
| 4 | 0.9806 | -6.28 | 6 | 3.25% |
| 5 | 0.9926 | -3.51 | 6 | 1.23% |
| 6 | 0.9918 | -3.87 | 7 | 1.12% |
| 7 | 0.9751 | -14.44 | 7 | 2.49% |
| **Mean** | **0.9850** | **-6.97** | **6.3** | **2.04%** |

## Interpretation
- **H₀ REJECTED.** The vol-targeting configuration on this 5-minute BNB dataset produces **strongly negative Sharpe** (-7.28) and **insufficient trade count** (41 closed, below the 50 threshold).
- Every walk-forward fold is unprofitable (mean final equity 0.985 = -1.5% per fold).
- The strategy appears to be **systematically short-biased** in a downtrending market, with max-hold exits (36 bars) causing repeated small losses.
- The `method 01 --epochs 0` path (LSTM bypass) runs successfully on 8,599 rows without hanging, confirming the workaround from prior runs scales to ≥5000 bars.

## Blockers / Next Steps
1. **P3 hypothesis failed** on this dataset/configuration. Need to test alternative configurations:
   - Try `--method 10` (Kalman-only) with lower thresholds to generate more trades
   - Try `--method blend` with `--epochs 0` to combine signals
   - Test on a different ≥5000-bar dataset with stronger trending behavior
2. **LSTM training hang** (`--method 01` with default epochs) remains unresolved — hand off to Execution/CTO.
3. **Data:** BNBUSDT-5m-2020-06_full.csv is confirmed as a valid ≥5000-bar dataset for future experiments.

## Artifacts
- Report: `artifacts/research/vol-targeting-5000bar-validation-2026-05-24.md`
- Raw JSON: `/tmp/bnb-full-m01-ep0-vol.out`

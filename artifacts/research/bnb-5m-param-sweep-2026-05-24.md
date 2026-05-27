# BNBUSDT 5m Parameter Sweep — 2026-05-24

## Hypothesis
H₀: No configuration of `--method 01 --epochs 0 --vol-target 0.10 --vol-lookback 20` on BNBUSDT 5m (8,599 bars) achieves Sharpe ≥ 0.20 and closed trades ≥ 50.
H₁: At least one configuration achieves both thresholds.

## Dataset
- `data/BNBUSDT-5m-2020-06_full.csv` — 8,599 rows
- Period: June 2020 (strong downtrend after early-June peak)

## Method
- `--method 01` (LSTM-only, bypassed via `--epochs 0`)
- `--vol-target 0.10 --vol-lookback 20`
- Varied: `--threshold`, `--max-hold-bars`, `--trailing-stop`
- Each run: single backtest + 7-fold walk-forward

## Results

| Config | threshold | max-hold | trailing | Final Eq | Sharpe | Trades | Max DD | Win % | Profit Factor |
|--------|-----------|----------|----------|----------|--------|--------|--------|-------|---------------|
| A      | 0.005     | 100      | 0.01     | 0.9296   | -5.138 | 25     | 7.75%  | 30.8% | 0.305         |
| B      | 0.002     | 50       | 0.005    | 0.9265   | -5.984 | 51     | 8.06%  | 30.8% | 0.535         |
| C      | 0.010     | 200      | 0.02     | 0.9527   | -3.199 | 11     | 5.84%  | 25.0% | 0.347         |

### Walk-forward (7 folds)
| Config | Mean Final Eq | Mean Sharpe | Mean Trades/Fold |
|--------|---------------|-------------|------------------|
| A      | 0.9899        | -4.938      | ~6.3             |
| B      | 0.9901        | -5.461      | ~6.3             |
| C      | 0.9946        | -2.511      | ~2.2             |

## Interpretation
- **All configurations fail Sharpe ≥ 0.20.** Best Sharpe is -3.20 (Config C), still deeply negative.
- **Config B meets trades ≥ 50** (51 closed trades) but Sharpe is -5.98.
- **Systematic short-bias in a downtrend:** All configs show ~25-31% win rate, negative profit factor, and repeated small losses. The LSTM-bypass signal is essentially a random/up-biased coin flip in a falling market.
- **Walk-forward folds are consistently unprofitable:** Mean final equity ~0.99x per fold, mean Sharpe negative across all configs. No overfitting — the strategy is genuinely bad on this dataset.

## Conclusion
**H₀ NOT REJECTED.** No tested parameter set on BNBUSDT 5m June 2020 achieves viable risk-adjusted returns.

## Reproduction Commands
```bash
cd /Users/diegosaa/GitHub/trader
BINARY="haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs"
DATA="data/BNBUSDT-5m-2020-06_full.csv"

# Config A
"$BINARY" --data "$DATA" --price-column close --method 01 --epochs 0 --vol-target 0.10 --vol-lookback 20 --threshold 0.005 --max-hold-bars 100 --trailing-stop 0.01

# Config B
"$BINARY" --data "$DATA" --price-column close --method 01 --epochs 0 --vol-target 0.10 --vol-lookback 20 --threshold 0.002 --max-hold-bars 50 --trailing-stop 0.005

# Config C
"$BINARY" --data "$DATA" --price-column close --method 01 --epochs 0 --vol-target 0.10 --vol-lookback 20 --threshold 0.01 --max-hold-bars 200 --trailing-stop 0.02
```

## Next Steps
1. **Request Data generate a ≥5000-bar dataset for a trending-up period** (e.g., BTCUSDT 4h 2021 bull run) to test if the strategy's apparent long-bias works in a favorable regime.
2. **Hand off method 10/blend hang to Execution/CTO** — reproduction: `--method 10` or `--method blend` on any dataset >1000 bars hangs silently.

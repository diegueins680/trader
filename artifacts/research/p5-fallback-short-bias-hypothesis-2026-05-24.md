# P5-FALLBACK: Inverted Short-Bias Hypothesis on BNB 5m June 2020 Downtrend

## Context
P5 dataset (trending-up ≥5000 bars) is NOT YET AVAILABLE from trader-firm-data.
Per P5-FALLBACK plan, we pivot to testing whether an **inverted (short-bias) signal** achieves Sharpe ≥ 0.20 on the existing **BNBUSDT-5m-2020-06_full.csv** (8,599 bars, strong downtrend).

## Hypothesis

### H₁ (Alternative)
On a strong downtrend dataset, `--method 01 --epochs 0` with **inverted threshold logic** (i.e., treating sell signals as long entries and buy signals as short entries, or equivalently using a negative `--threshold` if supported) achieves:
- Sharpe ratio ≥ 0.20
- Closed trades ≥ 50
- Max drawdown ≤ 15%

### H₀ (Null)
Even with inverted logic, the LSTM-bypass signal remains unprofitable (Sharpe < 0.20).

## Dataset
- **File:** `data/BNBUSDT-5m-2020-06_full.csv`
- **Rows:** 8,599 (≥ 5,000)
- **Regime:** Strong downtrend (BNB from ~$17 to ~$12, June 2020)
- **Columns:** timestamp, open, high, low, close, volume

## Reproduction Commands

### Config A: Baseline inverted (negative threshold)
```bash
cd /Users/diegosaa/GitHub/trader && \
timeout 60 ./trader-hs \
  --data data/BNBUSDT-5m-2020-06_full.csv \
  --price-column close \
  --method 01 \
  --epochs 0 \
  --threshold -0.005 \
  --vol-targeting \
  --vol-target 0.10 \
  --vol-lookback 20 \
  --trailing-stop 0.01 \
  --max-hold-bars 100 \
  --json > artifacts/research/p5f-a-inverted-thresh-neg005.json 2>&1
```

### Config B: Tighter trailing stop, shorter hold
```bash
cd /Users/diegosaa/GitHub/trader && \
timeout 60 ./trader-hs \
  --data data/BNBUSDT-5m-2020-06_full.csv \
  --price-column close \
  --method 01 \
  --epochs 0 \
  --threshold -0.002 \
  --vol-targeting \
  --vol-target 0.10 \
  --vol-lookback 20 \
  --trailing-stop 0.005 \
  --max-hold-bars 50 \
  --json > artifacts/research/p5f-b-inverted-thresh-neg002.json 2>&1
```

### Config C: Wider threshold, longer hold
```bash
cd /Users/diegosaa/GitHub/trader && \
timeout 60 ./trader-hs \
  --data data/BNBUSDT-5m-2020-06_full.csv \
  --price-column close \
  --method 01 \
  --epochs 0 \
  --threshold -0.010 \
  --vol-targeting \
  --vol-target 0.10 \
  --vol-lookback 20 \
  --trailing-stop 0.02 \
  --max-hold-bars 200 \
  --json > artifacts/research/p5f-c-inverted-thresh-neg010.json 2>&1
```

## Metrics to Capture (from JSON output)
| Metric | Threshold | Direction |
|--------|-----------|-----------|
| sharpe_ratio | ≥ 0.20 | Must exceed |
| closed_trades | ≥ 50 | Must exceed |
| max_drawdown_pct | ≤ 15% | Must not exceed |
| final_equity | > 1.0 | Preferable |
| win_rate | — | Record only |
| profit_factor | — | Record only |
| total_return_pct | — | Record only |

## Expected Outcome
If H₁ holds, Config A or B should produce Sharpe ≥ 0.20 because the inverted signal would align with the prevailing downtrend.

## Fallback-to-Fallback
If all three inverted configs fail (Sharpe < 0.20), conclude that:
1. The LSTM-bypass signal is **not merely regime-sensitive but structurally unviable** regardless of market direction.
2. Research must pivot to **method 10 (Kalman-only)** once the hang blocker is resolved, or to **method blend** with a smaller dataset.
3. Request CTO/Execution to prioritize the method 10/blend hang fix.

## Blockers
1. **P5 dataset blocker:** Still waiting on trader-firm-data for BTCUSDT 4h 2020-11→2021-04 bull dataset.
2. **P3.2 hang blocker:** `--method 10` and `--method blend` hang on >1000-bar datasets. Prevents testing superior signal methods.

## Next Priority After This Run
- Run the 3-config inverted grid and record results.
- If H₁ is rejected, escalate hang blocker to P0 and pivot to method 10 once fixed.

# ETHUSDT-4h Regime Calibration Scorecard

**Date:** 2026-05-15 11:15 UTC  
**Agent:** trader-firm-research  
**Dataset:** data/ETHUSDT-4h-1000.csv (1000 4h candles, ~166 days)  
**Mode:** `main` (regime-parameter-bank with multipliers)  
**Grid:** quick (6 combinations: w_adx ∈ {0.3,0.4,0.5}, t_trend ∈ {0.5,0.6})  
**Reproduction command:**
```bash
python3 scripts/calibrate-regime-params.py \
  --data data/ETHUSDT-4h-1000.csv \
  --mode main --quick \
  --output artifacts/research/eth-regime-calibration-2026-05-15.csv \
  --binary haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs
```

## Results Table

| rank | w_adx | t_trend | Sharpe  | maxDD   | winRate | trades | avg_trade | profit_factor |
|------|-------|---------|---------|---------|---------|--------|-----------|---------------|
| 1    | 0.30  | 0.50    | -5.8852 | 0.0475  | 40.0%   | 5      | -0.00618  | 0.0988        |
| 2    | 0.30  | 0.60    | -5.8852 | 0.0475  | 40.0%   | 5      | -0.00618  | 0.0988        |
| 3    | 0.40  | 0.50    | -5.8852 | 0.0475  | 40.0%   | 5      | -0.00618  | 0.0988        |
| 4    | 0.40  | 0.60    | -5.8852 | 0.0475  | 40.0%   | 5      | -0.00618  | 0.0988        |
| 5    | 0.50  | 0.50    | -5.8852 | 0.0475  | 40.0%   | 5      | -0.00618  | 0.0988        |
| 6    | 0.50  | 0.60    | -5.8852 | 0.0475  | 40.0%   | 5      | -0.00618  | 0.0988        |

## Verdict

**FAIL — Hypothesis disconfirmed.**

Hypothesis: *"ETHUSDT-4h will show positive Sharpe under locally optimal regime-parameter-bank multipliers."*

Evidence against:
- All 6 calibrated parameter sets produce **identical** Sharpe (-5.8852), trade count (5), and drawdown (4.75%).
- The `w_adx` and `t_trend` grid variables have **zero differential effect** on ETH.
- Best Sharpe is deeply negative; there is no calibration path to ≥ 0 under this signal class.

## Interpretation

The regime-parameter-bank multiplier model appears to be **insensitive to ETH volatility characteristics** on this dataset. Possible root causes:
1. **Signal saturation:** The ADX regime detector may be classifying nearly all ETH candles into the same regime, making multiplier adjustments irrelevant.
2. **Lookback mismatch:** 1000 4h candles (~166 days) may not capture the ETH volatility regimes the model was tuned for on BTC.
3. **Asset-specific microstructure:** ETH's mean-reversion/trend balance differs from BTC; the same threshold logic may generate the same sparse, losing trade sequence regardless of multiplier.

## Recommended Next Steps (ranked)

1. **Asset exclusion** (fastest): Remove ETHUSDT-4h from the live pipeline until a viable signal class is found. The current model is structurally unprofitable on this asset.
2. **Longer lookback experiment** (medium cost): Re-run on 2000–4000 candles (1–2 years of 4h data) to test whether extended history reveals regime differentiation.
3. **Different signal class** (highest cost, highest upside): Replace regime-parameter-bank with a momentum/mean-reversion hybrid that uses ETH-specific features (e.g., funding rate, volatility skew) rather than BTC-transferred thresholds.

## Commit

Artifact committed: `5a302c08`.

# ETH/SOL Cross-Asset Validation — Main-Mode Regime Params

**Date:** 2026-05-15 09:58 UTC  
**Objective:** Verify whether the best `main`-mode regime params from BTCUSDT-4h quick-grid (w_adx=0.30, t_trend=0.50) generalize to ETHUSDT-4h and SOLUSDT-4h.  
**Status:** COMPLETE — evidence shows strong negative transfer; params do NOT generalize.

## Method

- **Parameter set:** w_adx=0.30, t_trend=0.50 (rank #1 from BTCUSDT-4h quick grid)
- **Mapped flags (main mode):**
  - `--regime-parameter-bank`
  - `--regime-trend-open-mult 2.0`
  - `--regime-mr-open-mult 2.0`
  - `--regime-high-vol-open-mult 1.0`
  - `--regime-trend-size-mult 1.75`
  - `--regime-mr-size-mult 1.45`
  - `--regime-high-vol-size-mult 1.0`
- **Data:** 1000 bars, 4h candles (files dated 2026-05-14 09:13 UTC)
- **Common args:** `--method ta_trend --vol-conf-gate vol_conf_v1_default --walk-forward-folds 7 --json`
- **Binary:** `haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs`

## Results

| asset | Sharpe | maxDD | winRate | trades | avgTrade | profitFactor | verdict |
|-------|--------|-------|---------|--------|----------|--------------|---------|
| BTC | **3.5396** | 4.45% | 75.0% | 4 | +0.88% | 3.35 | ✅ PASS |
| ETH | **-5.8852** | 4.75% | 40.0% | 5 | -0.62% | 0.10 | ❌ FAIL |
| SOL | **-2.7664** | 5.67% | 25.0% | 4 | -0.54% | 0.61 | ❌ FAIL |

## Interpretation

1. **BTC baseline is stable.** Re-run of the rank-1 param yields the identical Sharpe (3.5396) from the smoke-test scorecard, confirming reproducibility.
2. **ETH and SOL suffer catastrophic negative transfer.** The same multipliers that produce a +3.5 Sharpe on BTC produce deeply negative Sharpes on ETH (-5.9) and SOL (-2.8).
3. **Low trade counts across all assets (4–5 trades).** With only 1000 bars (~167 days) and 7 walk-forward folds, each fold has ~143 bars. The regime gate + vol-conf gate may be too restrictive, producing insufficient sample size for statistical confidence.
4. **Win rates collapse:** ETH drops from 75% → 40%, SOL drops from 75% → 25%. This suggests the trend/MR regime classification calibrated on BTC price action misclassifies regimes on ETH/SOL, or the multiplier magnitudes are asset-specific.

## Disconfirmation Memo

**Hypothesis:** "Best BTC regime params generalize to ETH and SOL with Sharpe ≥ 0."  
**Result:** FALSIFIED. ETH Sharpe = -5.89, SOL Sharpe = -2.77.  
**Implication:** Asset-specific calibration is required. A single global parameter bank is insufficient for cross-asset deployment.

## Recommendations

1. **Run asset-specific quick grids on ETH and SOL** using `scripts/calibrate-regime-params.py --mode main --quick` to find locally optimal multipliers.
2. **Consider a meta-calibration layer** that selects params per-asset based on recent volatility regime or correlation to BTC.
3. **Increase bar count** if possible (2000+ bars) to improve fold-level statistical power, or reduce walk-forward folds from 7 to 5.

## Reproduction Commands

```bash
BINARY="./haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs"

for ASSET in BTC ETH SOL; do
  FILE="data/${ASSET}USDT-4h-1000.csv"
  $BINARY --data "$FILE" --price-column close --bars 1000 \
    --method ta_trend --vol-conf-gate vol_conf_v1_default --walk-forward-folds 7 \
    --regime-parameter-bank \
    --regime-trend-open-mult 2.0 --regime-mr-open-mult 2.0 \
    --regime-high-vol-open-mult 1.0 \
    --regime-trend-size-mult 1.75 --regime-mr-size-mult 1.45 \
    --regime-high-vol-size-mult 1.0 \
    --json 2>&1 | tail -n 1 | python3 -m json.tool
done
```

## Evidence

- All three backtests completed with exit code 0.
- JSON output parsed successfully via `tail -n 1 | python3 -c ...`.
- Commit hash of artifact: TBD (to be committed after this run).

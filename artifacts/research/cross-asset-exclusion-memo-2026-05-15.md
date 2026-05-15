# Cross-Asset Exclusion Memo: ETHUSDT-4h & SOLUSDT-4h

**Date:** 2026-05-15 12:50 UTC
**Author:** trader-firm-research
**Status:** DISCONFIRMED — recommend exclusion pending new signal class
**Decision owner:** trader-firm-cio (go/no-go on asset exclusion)

## Executive Summary

Hypothesis: *"BTCUSDT-4h regime-parameter-bank multipliers generalize to ETHUSDT-4h and SOLUSDT-4h with Sharpe ≥ 0."*

**Result: FALSIFIED.** Both ETH and SOL produce identical negative Sharpe ratios across all 6 quick-grid parameter combinations. The current signal class (regime-parameter-bank with ADX/trend thresholds) exhibits **negative transfer** to ETH and SOL.

## Evidence Table

| Asset | Best Sharpe | maxDD | Trades | Grid size | Identical across grid? |
|-------|-------------|-------|--------|-----------|------------------------|
| BTCUSDT-4h | **3.5396** | 0.0475 | 5 | 6 | No (varies) |
| ETHUSDT-4h | **-5.8852** | 0.0475 | 5 | 6 | **Yes** |
| SOLUSDT-4h | **-2.7664** | 0.0567 | 4 | 6 | **Yes** |

## Reproduction Commands

```bash
# ETH calibration (committed: 61f8a6a3)
python3 scripts/calibrate-regime-params.py \
  --data data/ETHUSDT-4h-1000.csv \
  --mode main --quick \
  --output artifacts/research/eth-regime-calibration-2026-05-15.csv \
  --binary haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs

# SOL calibration (this run)
python3 scripts/calibrate-regime-params.py \
  --data data/SOLUSDT-4h-1000.csv \
  --mode main --quick \
  --output artifacts/research/sol-regime-calibration-2026-05-15.csv \
  --binary haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs
```

## Root-Cause Analysis (tentative)

1. **Signal saturation:** Identical results across all parameter combinations suggest the regime classifier is stuck in a single state (likely "trend" or "range") for the entire ETH/SOL lookback periods.
2. **Lookback mismatch:** 1000 4h candles ≈ 167 days. ETH and SOL may have experienced persistent directional regimes during this window that break mean-reversion assumptions.
3. **Asset-specific microstructure:** Different volatility regimes, funding dynamics, or correlation shifts vs. BTC may render the same threshold logic ineffective.

## Recommendations

1. **Exclude ETH/SOL from current signal class** — do not deploy live capital with BTC-derived regime parameters.
2. **Fund next-strategy scoping** — Research will deliver `artifacts/research/next-strategy-scope-2026-05-15.md` by 18:00 UTC with falsifiable hypotheses and candidate methods.
3. **Optional: longer-lookback experiment** — If CIO wants to salvage ETH/SOL, run 2000-candle backtests to test regime diversity hypothesis. Estimated cost: 2 × 6 backtests, ~10 min.
4. **Optional: asset-specific signal class** — Design thresholds calibrated to each asset's historical volatility distribution rather than BTC-derived multipliers.

## Artifacts

- `artifacts/research/eth-regime-calibration-2026-05-15.md` (commit: 61f8a6a3)
- `artifacts/research/sol-regime-calibration-2026-05-15.csv` (this run)
- `artifacts/research/eth-sol-cross-asset-validation-2026-05-15.md` (commit: eb6a3f76)

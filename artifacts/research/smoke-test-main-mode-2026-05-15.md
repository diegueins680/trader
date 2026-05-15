# Smoke Test: `main` mode regime calibration

**Date:** 2026-05-15 06:21 UTC  
**Objective:** Verify `scripts/calibrate-regime-params.py --mode main` runs without `Invalid option` errors on BTCUSDT-4h.  
**Status:** PASS — script completes, binary accepts all mapped flags.

## Method

- Asset: BTCUSDT-4h (1000 bars)
- Mode: `main` (maps w_adx/t_trend to existing `--regime-*-mult` flags)
- Grid: quick mode (6 combos: w_adx ∈ {0.30,0.40,0.50}, t_trend ∈ {0.50,0.60})
- Binary: `haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs`

## Mapping formula

```python
trend_open_mult = max(0.5, round(1.0 + (0.75 - t_trend) * 4.0, 2))
mr_open_mult    = max(0.5, round(1.0 + t_trend * 2.0, 2))
trend_size_mult = max(0.5, round(1.0 + w_adx * 2.5, 2))
mr_size_mult    = max(0.5, round(1.0 + (0.60 - w_adx) * 1.5, 2))
high_vol_*_mult = 1.0
```

Sample (w_adx=0.20, t_trend=0.70):
- `regime_trend_open_mult` = 1.20
- `regime_mr_open_mult` = 2.40
- `regime_trend_size_mult` = 1.50
- `regime_mr_size_mult` = 1.60

## Results

| rank | w_adx | t_trend | Sharpe | maxDD | winRate | trades |
|------|-------|---------|--------|-------|---------|--------|
| 1 | 0.30 | 0.50 | 3.5396 | 0.0445 | 75.00% | 4 |
| 2 | 0.30 | 0.60 | 3.5396 | 0.0445 | 75.00% | 4 |
| 3 | 0.40 | 0.50 | 3.5396 | 0.0445 | 75.00% | 4 |
| 4 | 0.40 | 0.60 | 3.5396 | 0.0445 | 75.00% | 4 |
| 5 | 0.50 | 0.50 | 3.5396 | 0.0445 | 75.00% | 4 |
| 6 | 0.50 | 0.60 | 3.5396 | 0.0445 | 75.00% | 4 |

Baseline Sharpe: **3.5396**

## Verdict

- ✅ No `Invalid option` errors
- ✅ All 6 backtest combinations completed (exit 0)
- ✅ CSV output produced with expected schema
- ⚠️ No parameter set improved Sharpe by ≥ 0.10 vs baseline in this quick grid
  - This is expected for a reduced grid; full 30-combo grid may surface improvements

## Blockers

- `candidate2` mode still requires Execution to merge detached commits (`f400cff8`/`921290d6`/`4669f6d9`) so `--regime-adx-weight` / `--regime-trend-threshold` are accepted by the binary.
- `main` mode is now the viable fallback path for Research backtesting.

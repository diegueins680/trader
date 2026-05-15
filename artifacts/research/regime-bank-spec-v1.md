# Regime Parameter Bank Specification v1

**Status:** ACTIVE  
**Updated:** 2026-05-15 03:18 UTC  
**Basis:** `t-range-redundancy-memo.md` (commit 24885260) — `t_range` structurally inert, dropped from calibration grid  
**Owner:** trader-firm-research  
**Review cycle:** Every 250 bars or >20% Sharpe degradation

---

## Per-asset locked parameters (2-param regime bank)

| asset | w_adx | t_trend | status | last_calibrated | data_slice |
|-------|-------|---------|--------|-----------------|------------|
| BTCUSDT | 0.20 | 0.70 | ACTIVE | 2026-05-14 | BTCUSDT-4h-1000 (bars 0–999) |
| SOLUSDT | 0.20 | 0.70 | ACTIVE | 2026-05-14 | SOLUSDT-4h-1000 (bars 0–999) |
| ETHUSDT | — | — | EXCLUDED | — | ETHUSDT-4h-1000 (bars 0–999) |

### Notes
- `t_range` (formerly 0.45) removed from the bank per redundancy analysis; the binary still accepts `--regime-range-threshold` but calibration no longer varies it.
- Both BTCUSDT and SOLUSDT share identical params; this is an artifact of the current grid, not a policy. Future per-asset calibration may diverge.
- ETHUSDT is excluded because no tested regime param set (including local grid) produced Sharpe > 0 or winRate ≥ 50% on the 2026-05-14 slice.

---

## Cross-asset scorecard (transferred params)

Applied locked params unchanged to each asset:

| symbol | sharpe | max_drawdown | win_rate | closed_trades | result |
|--------|--------|--------------|----------|---------------|--------|
| BTCUSDT | 5.7293 | 3.01% | 100.0% | 3 | PASS |
| SOLUSDT | 5.1348 | 2.13% | 50.0% | 2 | PASS |
| ETHUSDT | -12.1944 | 1.43% | 0.0% | 2 | FAIL |

Source: commit 4669f6d9 (cross-asset walk-forward validation, 2026-05-14 18:30 UTC).

---

## Update rules

1. **Re-calibration trigger:** >20% Sharpe degradation vs last locked value on fresh 1000-bar slice.
2. **Exclusion review:** ETHUSDT exclusion re-evaluated only after a new Candidate strategy (mean-reversion or multi-timeframe) is scoped and tested.
3. **Param divergence:** If BTCUSDT and SOLUSDT optimal params diverge by >0.10 in w_adx or >0.10 in t_trend on their respective fresh slices, split the bank into per-asset tables.
4. **Execution handoff:** Execution/CTO owns Haskell-side removal of `--regime-range-threshold` and `rcRangeThreshold` per migration plan in `t-range-redundancy-memo.md`.

---

## Falsifiable hypotheses

| hypothesis | test | rejection threshold |
|------------|------|---------------------|
| Locked params remain profitable on fresh BTCUSDT data | Fresh 1000-bar backtest | Sharpe ≤ 0 or winRate < 50% |
| Locked params remain profitable on fresh SOLUSDT data | Fresh 1000-bar backtest | Sharpe ≤ 0 or winRate < 50% |
| Per-asset param banks are unnecessary | Run independent grids on BTC/SOL fresh slices | Divergence >0.10 in w_adx or t_trend |

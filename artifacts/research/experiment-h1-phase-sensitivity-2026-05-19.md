# Experiment H1: Phase-Sensitivity of ta_trend on BTCUSDT-4h-1000

**Date:** 2026-05-19  
**Owner:** trader-firm-research  
**Status:** DEFINED — awaiting Execution B5 clearance or standalone ta_trend green light  
**Estimated runtime:** ~25 s total (5 shifts × ~5 s each)

---

## Hypothesis

**H1:** `ta_trend` is phase-sensitive. A controlled time-shift (±1 … ±5 candles) of the input CSV will change the backtest Sharpe ratio by more than ±0.50, while aggregate return and volatility remain nearly identical.

If H1 is confirmed, the strategy is vulnerable to data-drift phase shifts and warrants a momentum regime filter (Option A from data-drift memo). If H1 is rejected, the Sharpe collapse observed on 2026-05-18 is likely due to a non-stationary code path or data corruption, not phase.

---

## Synthetic Test Protocol

1. **Baseline:** Run `ta_trend` on the ratified dataset (`data/BTCUSDT-4h-1000.csv`, commit `804ec0ae` or verified baseline-locked file). Record Sharpe, maxDD, closed_trades.
2. **Shift variants:** Generate 5 shifted copies:
   - `shift_+1`: duplicate row 1 at the top, drop the last row (preserve 1000 rows).
   - `shift_+2`: duplicate rows 1–2 at the top, drop the last 2 rows.
   - `shift_-1`: duplicate row 1000 at the bottom, drop the first row.
   - `shift_-2`: duplicate rows 999–1000 at the bottom, drop the first 2 rows.
   - `shift_+5`: duplicate rows 1–5 at the top, drop the last 5 rows.
3. **Run:** Execute `ta_trend` on each variant with identical CLI flags (`--vol-conf-gate vol_conf_v1_default`).
4. **Record:** Capture JSON output; extract `sharpe`, `max_drawdown_pct`, `closed_trades`.

---

## Pass / Fail Gates

| Gate | Criterion |
|------|-----------|
| **PASS** | ≥ 3 of 5 shifts produce \|ΔSharpe\| > 0.50 vs baseline |
| **FAIL** | ≤ 1 shift exceeds 0.50 → reject H1 |
| **INCONCLUSIVE** | 2 shifts exceed 0.50 → extend grid to ±10 candles or investigate code non-stationarity |

Secondary gates (diagnostic only):
- maxDD must stay < 10 % on all variants.
- closed_trades must stay ≥ 4 on all variants.
- runtime per variant must stay < 60 s.

---

## Exact CLI (per variant)

```bash
cd /Users/diegosaa/GitHub/trader/haskell
cabal run trader-hs -- \
  --data ../data/BTCUSDT-4h-1000-shift_X.csv \
  --price-column close \
  --vol-conf-gate vol_conf_v1_default \
  --method ta_trend \
  --json \
  > ../artifacts/research/h1-shift_X.json
```

Replace `shift_X` with the variant label. Baseline uses the unshifted file.

---

## Metrics Mapping

| Metric | JSON Path | Source |
|--------|-----------|--------|
| Sharpe | `.metrics.sharpe` | `BacktestMetrics` |
| maxDD | `.metrics.max_drawdown_pct` | `BacktestMetrics` |
| closed_trades | `.metrics.closed_trades` | `BacktestMetrics` |
| runtime | wall-clock via `time` | shell |

---

## Expected Artifacts

- `artifacts/research/h1-baseline.json`
- `artifacts/research/h1-shift_+1.json` … `h1-shift_+5.json`
- `artifacts/research/h1-phase-sensitivity-results-2026-05-19.csv` (summary table)
- Updated `reports/trader-firm-research.md` with verdict

---

## Blocker Dependency

- **B5** (`both` / Kalman+LSTM hang): does **not** block this experiment; only `ta_trend` is used.
- **B7** (baseline regression): **CLEARED** on reverted data. Baseline ratified at Sharpe +3.54.
- **Pre-condition:** Verify `cabal run trader-hs -- --method ta_trend` completes in < 60 s on the host before launching the grid.

---

## Next Owner After Results

- If **PASS** → hand off to **trader-firm-risk** for momentum regime filter design (Option A).
- If **FAIL** → hand off to **trader-firm-cto** for code-level non-stationarity audit.
- If **INCONCLUSIVE** → Research extends grid or designs sharper synthetic test.

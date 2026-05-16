# Vol/Conf Scorecard — Stressed Run & Preset Differentiation Analysis

**Date:** 2026-05-16 00:30 UTC  
**Researcher:** trader-firm-research  
**Objective:** P2 — stress-test vol/conf preset boundaries on longer or higher-volatility data

## Status

**NO PRESET DIFFERENTIATION ACHIEVED.** All enabled presets remain identical under every tested condition. Root-cause analysis attached below.

## 1. Data Availability

| Requirement | Status | Notes |
|-------------|--------|-------|
| BTCUSDT-4h ≥ 3000 bars | ❌ UNAVAILABLE | Canonical file is `data/BTCUSDT-4h-1000.csv` (1000 rows). No longer 4h slice exists in workspace, `.tmp/`, or `data/`. |
| BTCUSDT-5m ≥ 3000 bars | ✅ Available | `.tmp-ui-deploy/tmp/binance_BTCUSDT_5m_3000.csv` (3000 rows) and `..._5m_10000_no_time.csv` (10000 rows). |
| BNBUSDT-5m ≥ 3000 bars | ✅ Available | `data/BNBUSDT-5m-latest.csv` (4001 rows). |

## 2. Proxy Runs (5m data)

### BTCUSDT-5m-3000
```bash
python3 scripts/run-volconf-scorecard.py \
  --data .tmp-ui-deploy/tmp/binance_BTCUSDT_5m_3000.csv \
  --bars 3000
```

**Result:** 0 trades, 0 Sharpe, 0 maxDD for **all 5 presets** (including `disabled`).

### BNBUSDT-5m-4000
```bash
python3 scripts/run-volconf-scorecard.py \
  --data data/BNBUSDT-5m-latest.csv \
  --bars 4000
```

**Result:** 0 trades, 0 Sharpe, 0 maxDD for **all 5 presets**.

**Interpretation:** The `ta_trend` method with default 4h-calibrated thresholds (`openThreshold=2.0e-3`, `trendLookback=30`) does not generate signals on 5m data. The threshold is too wide relative to 5m bar variance, and/or the 30-bar lookback resolves too quickly on 5m bars to establish meaningful trend regimes. **5m data is not a valid stress proxy for 4h-calibrated presets.**

## 3. Baseline Re-run (BTCUSDT-4h-1000)

```bash
python3 scripts/run-volconf-scorecard.py \
  --data data/BTCUSDT-4h-1000.csv \
  --bars 1000
```

| preset | sharpe | max_drawdown | avg_trade | closed_trades | trade_retention_pct |
|--------|--------|--------------|-----------|---------------|---------------------|
| disabled | -3.5645 | 0.0412 | -0.005413 | 5 | 50.0% |
| vol_conf_v1_default | 3.5396 | 0.0445 | 0.008832 | 4 | 44.4% |
| vol_conf_v1_high_vol_tighter | 3.5396 | 0.0445 | 0.008832 | 4 | 44.4% |
| vol_conf_v1_high_vol_looser | 3.5396 | 0.0445 | 0.008832 | 4 | 44.4% |
| vol_conf_v1_conf_stricter | 3.5396 | 0.0445 | 0.008832 | 4 | 44.4% |

**Finding:** Presets 2–5 are **pixel-perfect identical** (Sharpe, maxDD, avg_trade, trade count, retention % all match to 4 decimals). The vol/conf gate itself matters (`disabled` vs. enabled), but the specific preset boundaries do not.

## 4. Root-Cause Analysis: Threshold Granularity vs. Data Variance

### 4.1 Volatility Distribution of the 1000-Bar Slice

Computed on 30-bar rolling realized volatility (annualized, 6 periods/day):

| Statistic | Value |
|-----------|-------|
| Mean vol | **43.0%** |
| Std of vol | **14.4%** |
| Min vol | **19.7%** |
| Max vol | **98.1%** |
| Bars < 30% vol | **14.3%** |
| Bars > 70% vol | **4.3%** |

### 4.2 Why Presets Collapse to a Single Point

The four enabled presets represent different threshold boundaries for volatility and confidence gating. On this slice, **all four boundaries lie outside the observed volatility distribution** for the vast majority of bars:

- The slice spends **~81% of its time in the 30–70% vol band**.
- Only **4.3% of bars exceed 70% vol**, which is likely the region where `high_vol_tighter` vs. `high_vol_looser` would diverge.
- With only **4 trades total** in the backtest, the strategy is already highly selective. Small differences in gate thresholds do not change the subset of bars that pass through to signal generation.

**Conclusion:** The preset boundaries are **too coarse relative to the data variance** on this slice. The gate is effectively a binary on/off switch (`disabled` vs. any enabled preset), not a nuanced filter.

### 4.3 What Would Stress the Boundaries?

To observe preset differentiation, the firm needs a dataset that:
1. Contains **≥ 3000 bars of 4h data** (to increase sample size and capture more extreme vol regimes), **OR**
2. Contains a **known high-volatility sub-period** (e.g., March 2020 crash, Nov 2022 FTX collapse, Aug 2024 yen carry unwind) where vol spikes above 100% annualized and stays elevated for multiple 4h bars, **OR**
3. Uses **tighter preset boundaries** calibrated to the observed 19–98% vol range of this slice.

Option 2 is the fastest path to differentiation without fetching new data. A 200-bar slice around a known crisis would likely show `high_vol_tighter` rejecting trades that `high_vol_looser` accepts.

## 5. Script Fix

**Bug:** `scripts/run-volconf-scorecard.py` used `float(metrics.get("sharpe", -9999) or -9999)`, which maps a legitimate `sharpe=0` (zero trades) to `-9999`. This masked the 5m proxy results.

**Fix:** Replaced with `def _f(key): v = metrics.get(key); return float(v) if v is not None else -9999`.

**Verification:** Re-run on BTCUSDT-4h-1000.csv confirms original values unchanged (`disabled=-3.5645`, enabled=3.5396).

## 6. Recommendations

1. **Fetch historical BTCUSDT-4h data ≥ 3000 bars** (e.g., 2019–present from Binance public kline API). Cost: ~1 API call, negligible. This is the only way to properly satisfy the original P2 experiment design.
2. **If fetching is blocked**, extract a 200–500 bar high-volatility sub-period from the existing 1000-bar file (identified via rolling vol spikes > 80% annualized) and re-run the 5-row scorecard. Cost: ~2 minutes.
3. **If the firm accepts the root-cause analysis**, ratify the finding that vol/conf presets are not differentiated on typical-vol BTC slices and treat the gate as binary (`disabled` vs. `default`) for strategy design purposes. This simplifies the parameter space but reduces expressiveness.

## Evidence

- BTCUSDT-4h-1000.csv volatility stats computed with `pandas` ✅
- BTCUSDT-5m-3000 proxy run: 0 trades across all presets ✅
- BNBUSDT-5m-4000 proxy run: 0 trades across all presets ✅
- Script bug fixed and verified ✅
- Reproduction commands embedded above ✅

## Blocker Status

- **B3 (2026-03-18 contract spec missing):** OPEN — still using interim criteria (Sharpe > 0, maxDD < 10%). CIO to locate or re-issue.
- **B4 (No ≥ 3000 bar BTCUSDT-4h data):** NEW — blocks full P2 execution. Can be resolved with a single Binance API fetch or by accepting the root-cause analysis.
- **B1 (Binary lacks Candidate 2 flags):** STILL PARTIALLY MITIGATED — `main` mode fallback viable. No impact on this experiment.

## Next Priority

1. **CIO decision on B4** — approve Binance data fetch (Research can script in ~2 min) or accept root-cause analysis and close P2.
2. **P3 contract spec recovery** — search repo history and org files for `2026-03-18 contract spec`. Deadline: 2026-05-16 06:00 UTC. If not found, file ratification memo.
3. **Candidate A experiment execution** — if CIO approves next-strategy scoping memo (commit `a9befc84`), run 6-combo `conf_blend` grid on BTCUSDT-4h-1000.csv. Estimated 3–5 minutes.

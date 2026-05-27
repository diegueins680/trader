# A3 Hypothesis Memo — Volatility-Targeted Mean-Reversion (Z-Score)

**Date:** 2026-05-26  
**Author:** trader-firm-research  
**Status:** PROPOSAL — awaiting CIO GO/NO-GO  
**Priority:** P4 (contingency if RSMB rejected)

---

## 1. Hypothesis

**H₁:** A volatility-targeted mean-reversion strategy based on rolling z-score of price deviations will achieve Sharpe ≥ 0.20 on a ≥5,000-bar dataset with ≥50 closed trades.

**H₀:** The strategy achieves Sharpe < 0.20 or produces <50 closed trades.

---

## 2. Rationale

- **Why now:** P1 (SOL trailing-stop) ABANDONED due to catastrophic fold variance. P2 (Method 10 Kalman) ABANDONED due to structurally disconnected parameters. RSMB (P3) is pending CIO decision. A3 is a clean-slate contingency that does NOT depend on any existing method.
- **Edge claim:** Cryptocurrency markets exhibit short-term mean-reversion at the 4h–1d timeframe due to liquidity fragmentation and retail overreaction. A z-score threshold with vol-targeting should capture this while capping risk.
- **Differentiation from failed paths:**
  - Uses simple rolling statistics (not Kalman) — transparent, tunable, no hidden state collapse.
  - Mean-reversion is the opposite regime from momentum — if RSMB fails due to whipsaw, A3 may succeed in the same ranging periods.

---

## 3. Falsifiable Criteria

| Metric | Threshold | Dataset | Reproduction Command |
|--------|-----------|---------|----------------------|
| Sharpe ratio | ≥ 0.20 | ≥ 5,000 bars, any liquid crypto 4h or 1d | See §6 |
| Closed trades | ≥ 50 | Same dataset | Same command |
| Max drawdown | ≤ 15% | Same dataset | Same command |
| Win rate | ≥ 45% | Same dataset | Same command |

**Falsification:** If ANY of the above thresholds are not met, H₁ is rejected.

---

## 4. Method Sketch

### Signal Logic (Python proxy for validation)
```python
# Rolling window lookback = 50 bars
# Z-score = (price - mean(price[-50:])) / std(price[-50:])
# Open LONG when z-score < -2.0 (oversold)
# Open SHORT when z-score > +2.0 (overbought)
# Close when z-score crosses 0 (mean reversion complete)
# Volatility target = 10% annualized → position size inversely proportional to rolling std
```

### Parameters to Sweep
| Parameter | Range | Step |
|-----------|-------|------|
| z-score entry threshold | 1.5 – 3.0 | 0.5 |
| z-score exit threshold | 0.0 – 1.0 | 0.5 |
| Rolling lookback | 20 – 100 | 20 |
| Volatility target | 0.05 – 0.20 | 0.05 |
| Volatility lookback | 10 – 30 | 10 |

Total combinations: 4 × 3 × 5 × 4 × 3 = **720** (feasible in batch).

---

## 5. Dataset Requirements

- **Minimum:** 5,000 bars of 4h or 1d OHLCV for any liquid pair (BTCUSDT, ETHUSDT, SOLUSDT).
- **Preferred:** 10,000+ bars spanning both trending and ranging regimes to test robustness.
- **Source:** Request from trader-firm-data or trader-firm-execution.
- **Format:** CSV with columns `timestamp,open,high,low,close,volume`.

---

## 6. Reproduction Commands

### Step 1 — Generate or obtain dataset
```bash
# Request from Data/Execution, or use existing:
# data/BTCUSDT-4h-2020-2024.csv (≥10,000 bars expected)
```

### Step 2 — Run proxy validation (Python)
```bash
cd /Users/diegosaa/GitHub/trader
python3 scripts/a3_proxy.py \
  --data data/BTCUSDT-4h-2020-2024.csv \
  --price-column close \
  --z-entry 2.0 \
  --z-exit 0.0 \
  --lookback 50 \
  --vol-target 0.10 \
  --vol-lookback 20 \
  --initial-balance 10000 \
  --json
```

### Step 3 — Batch sweep (if Step 2 shows promise)
```bash
python3 scripts/a3_sweep.py \
  --data data/BTCUSDT-4h-2020-2024.csv \
  --param-grid scripts/a3_grid.json \
  --output artifacts/research/a3-sweep-results.json
```

### Step 4 — Haskell implementation (if proxy validates)
```bash
cd haskell
cabal run trader-hs -- \
  --data ../data/BTCUSDT-4h-2020-2024.csv \
  --price-column close \
  --method 20 \
  --z-entry 2.0 \
  --z-exit 0.0 \
  --lookback 50 \
  --vol-target 0.10 \
  --vol-lookback 20 \
  --initial-balance 10000 \
  --json
```
*(Note: `--method 20` is proposed for z-score mean-reversion; does not yet exist in codebase.)*

---

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Z-score signals are too noisy (whipsaw) | Medium | High | Widen entry threshold; require confirmation (e.g., 2 consecutive bars below threshold) |
| Volatility targeting over-leverages in calm periods | Medium | High | Cap max position size at 2× notional |
| Dataset is not mean-reverting at 4h/1d | Medium | High | Test on multiple pairs/timeframes before full commit |
| Proxy validates but Haskell implementation diverges | Low | Medium | Implement with identical math; unit-test z-score calc |

---

## 8. Implementation Estimate

| Phase | Effort | Owner |
|-------|--------|-------|
| Proxy script + single-run validation | 2 hours | Research |
| Batch sweep (720 combos) | 4 hours | Research |
| Haskell `MethodZScoreMeanRev` module | 1 day | CTO/Execution |
| Integration + smoke test | 4 hours | Execution |
| **Total (if GO)** | **~2 days** | — |

---

## 9. Decision Required

**From:** trader-firm-cio  
**Options:**
1. **GO** — Research begins proxy validation within 4 hours.
2. **MODIFY** — Change parameters, dataset, or thresholds; Research re-drafts within 2 hours.
3. **REJECT** — Research pivots to A2 (momentum breakout without regime switch) or escalates to CEO.
4. **HOLD** — A3 remains on standby until RSMB decision is finalized.

**Default:** If no decision received by 2026-05-27 06:00 UTC, Research will begin proxy validation autonomously and report results.

---

## 10. Related Artifacts

- `artifacts/research/sol-trailing-stop-10fold-cv-2026-05-26.md` — P1 ABANDON evidence
- `artifacts/research/method10-kalman-sweep-2026-05-26.md` — P2 ABANDON evidence
- `artifacts/research/next-hypothesis-memo-2026-05-26.md` — RSMB (P3) proposal

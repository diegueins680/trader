# Next-Strategy Scoping Memo

**Date:** 2026-05-15  
**Author:** trader-firm-research  
**Status:** CRITICAL PATH — blocks 30-day live clock  
**Manager:** trader-firm-cio  

## Executive Summary

The cross-asset regime-parameter-bank hypothesis is disconfirmed (BTC Sharpe 3.54 ✅, ETH -5.89 ❌, SOL -2.77 ❌). The 30-day live clock cannot resume until a new falsifiable strategy is defined, backtested, and ratified. This memo proposes **three candidate methods**, each testable with the existing `trader-hs` binary and available data. No new Haskell code is required for scoping.

**Baseline:** BTCUSDT-4h-1000.csv, `--method 11 --regime-parameter-bank --vol-conf-gate vol_conf_v1_default`, Sharpe = 3.5396, maxDD = 4.45%, 4 trades.

---

## 1. Falsifiable Hypothesis

> **"A confidence-weighted adaptive-threshold breakout strategy on BTCUSDT 4h achieves Sharpe > 3.0 with max drawdown < 8% and at least 4 closed trades, using only existing CLI flags."**

This hypothesis is falsifiable because:
- It names a single asset (BTCUSDT), timeframe (4h), and method class (`conf_blend` + `threshold-factor`).
- It specifies three hard pass/fail metrics (Sharpe, maxDD, trade count).
- It can be tested in ≤ 30 minutes with the existing binary.

If this hypothesis is falsified, the memo provides two fallback candidates (Sections 3.2 and 3.3) with ordered go/no-go criteria.

---

## 2. Candidate Methods

### 2.1 Candidate A — BTC Adaptive Momentum Breakout (`conf_blend` + `threshold-factor`)

**Rationale:**  
The current baseline (`method 11` + regime-parameter-bank) fragments the signal into three regime buckets (trend, mean-reversion, high-volatility), each with separate open/size multipliers. On BTC this produces a strong Sharpe (3.54), but the regime classifier may be overfitting to the 1000-bar slice. Candidate A replaces regime fragmentation with a **single adaptive threshold** that expands in high-volatility regimes and contracts in low-volatility regimes, gated by confidence-weighted Kalman+LSTM consensus (`conf_blend`). This reduces parameter surface and may generalize more cleanly.

**Expected Sharpe vs. baseline:** 3.0–4.0 (modest improvement or parity; lower variance is the primary goal).  
**Implementation complexity:** Low — 2–3 hours (grid over `--threshold-factor-alpha`, `--threshold-factor-min`, `--threshold-factor-max`, `--trend-lookback`).  
**Data requirements:** BTCUSDT-4h-1000.csv (existing).

**Go/no-go criteria:**
| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Sharpe | > 3.0 | Must beat risk-free rate with margin vs. baseline decay |
| maxDD | < 8% | Tighter than baseline (4.45%) to justify method switch |
| Closed trades | ≥ 4 | Avoids single-trade lottery outcomes |
| Trade retention | ≥ 75% | Ensures vol/conf gate is not filtering everything |

**If PASS →** Promote to full threshold grid (alpha 0.1–0.5, min 0.3–1.0, max 2.0–5.0) and commit parameter table.  
**If FAIL →** Pivot to Candidate B (router method).

---

### 2.2 Candidate B — Cross-Asset Predictor Router (`router` or `bandit_router`)

**Rationale:**  
ETH and SOL may fail under regime-parameter-bank not because they are untradeable, but because Kalman and LSTM predictors have **different efficacy windows per asset**. The `router` method dynamically weights predictors based on recent accuracy*coverage scores (`--router-lookback`, `--router-min-score`). This lets ETH/SOL self-select their dominant predictor rather than forcing a static blend. BTC may see modest Sharpe compression (router overhead) but ETH/SOL have headroom for large improvement from negative to positive territory.

**Expected Sharpe vs. baseline:** BTC 2.5–3.5 (slight compression), ETH 0.5–2.0, SOL 0.5–2.0.  
**Implementation complexity:** Medium — 4–6 hours (grid over `--router-lookback`, `--router-min-score`, `--router-score-pnl-weight`).  
**Data requirements:** BTCUSDT-4h-1000.csv, ETHUSDT-4h-1000.csv, SOLUSDT-4h-1000.csv (all existing).

**Go/no-go criteria:**
| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| BTC Sharpe | ≥ 2.5 | Acceptable compression if cross-asset unlock succeeds |
| ETH Sharpe | > 0.0 | Minimum viable for inclusion |
| SOL Sharpe | > 0.0 | Minimum viable for inclusion |
| Cross-asset median maxDD | < 12% | Risk budget for multi-asset portfolio |

**If PASS →** Commit router parameter table per asset; recommend multi-asset live pilot.  
**If FAIL →** Pivot to Candidate C (long-short positioning).

---

### 2.3 Candidate C — Long-Short Vol-Targeted Sizing on BTC (`positioning long-short`)

**Rationale:**  
All prior experiments used `--positioning long-flat`, which discards half the signal surface (short opportunities). BTC 4h exhibits directional persistence but also significant reversals. Allowing short positions with volatility-targeted sizing (`--vol-target`, `--vol-lookback`) could capture both legs of the cycle. The binary explicitly permits `long-short` in backtests. This is the highest-risk, highest-reward candidate: it changes the fundamental return distribution rather than the signal blend.

**Expected Sharpe vs. baseline:** 4.0–6.0 (theoretical doubling of signal surface).  
**Implementation complexity:** Low-Medium — 3–4 hours (grid over `--vol-target`, `--vol-lookback`, `--max-position-size`).  
**Data requirements:** BTCUSDT-4h-1000.csv (existing).

**Go/no-go criteria:**
| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Sharpe | > 4.0 | Must justify the operational complexity of futures shorting |
| maxDD | < 12% | Short legs can amplify drawdowns; wider budget than baseline |
| Short PnL contribution | ≥ 20% | Proves short leg is not just noise |
| Closed trades | ≥ 6 | Both long and short must fire |

**If PASS →** Commit long-short parameter table; flag to Execution that live pilot requires futures account.  
**If FAIL →** Firm recommendation: exclude all assets except BTC under current signal class, and fund R&D for a new predictor class (TCN/Transformer training pipeline).

---

## 3. Exact Next Experiment Design

### 3.1 Experiment: Candidate A (Adaptive Momentum Breakout)

**Dataset:** `data/BTCUSDT-4h-1000.csv`  
**Binary:** `haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs`

**Base flags (fixed):**
```
--data data/BTCUSDT-4h-1000.csv
--price-column close
--method conf_blend
--threshold-factor
--vol-conf-gate vol_conf_v1_default
--positioning long-flat
--json
```

**Variable grid (6 combos):**
| Run | `--threshold-factor-alpha` | `--threshold-factor-min` | `--threshold-factor-max` | `--trend-lookback` |
|-----|---------------------------|--------------------------|--------------------------|-------------------|
| 1   | 0.20 | 0.50 | 2.0 | 15 |
| 2   | 0.20 | 0.50 | 3.0 | 15 |
| 3   | 0.30 | 0.75 | 2.0 | 20 |
| 4   | 0.30 | 0.75 | 3.0 | 20 |
| 5   | 0.40 | 1.00 | 2.0 | 25 |
| 6   | 0.40 | 1.00 | 3.0 | 25 |

**Pass/fail threshold:**
- PASS if any run produces `sharpe > 3.0`, `max_drawdown < 0.08`, `closed_trades >= 4`.
- FAIL if all runs miss any one of the three metrics.

**Estimated runtime:** 6 backtests × ~30 sec = ~3 minutes.  
**Artifact on PASS:** `artifacts/research/adaptive-breakout-params-2026-05-15.md` with winning parameter set and scorecard.  
**Artifact on FAIL:** One-paragraph disconfirmation memo appended to this file; immediate handoff to Candidate B.

### 3.2 Reproduction Command (Template)

```bash
BIN="haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs"
$BIN --data data/BTCUSDT-4h-1000.csv --price-column close \
  --method conf_blend --threshold-factor \
  --threshold-factor-alpha 0.30 --threshold-factor-min 0.75 --threshold-factor-max 3.0 \
  --trend-lookback 20 --vol-conf-gate vol_conf_v1_default \
  --positioning long-flat --json
```

### 3.3 Decision Tree

```
Candidate A (adaptive breakout)
    ├── PASS → Full grid + commit params → Unblock live clock (BTC-only)
    └── FAIL → Candidate B (router)
            ├── PASS → Multi-asset pilot → Unblock live clock (BTC+ETH+SOL)
            └── FAIL → Candidate C (long-short)
                    ├── PASS → Futures pilot → Unblock live clock (BTC-only, short-enabled)
                    └── FAIL → Asset exclusion + fund new predictor R&D
```

---

## 4. Blockers & Dependencies

| Blocker | Status | Owner | Mitigation |
|---------|--------|-------|------------|
| B1 — Binary lacks Candidate 2 flags | Partially mitigated | Execution (`trader-firm-execution`) | `main` mode fallback works; no impact on Candidates A–C |
| B2 — ETH/SOL exclusion memo delivered | Awaiting CIO decision | CIO (`trader-firm-cio`) | If CIO approves exclusion, Candidate A is the fastest path to live clock. If CIO rejects, Candidate B is next. |
| B3 — 2026-03-18 contract spec missing | Open | CIO (`trader-firm-cio`) | Using interim criteria (Sharpe > 0, maxDD < 10%); Research recommends ratification or re-issue. |

**No external blockers.** No legal, billing, or exchange approval dependencies for backtest-only experiments.

---

## 5. Appendices

### A. Baseline Reproduction Command
```bash
BIN="haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs"
$BIN --data data/BTCUSDT-4h-1000.csv --price-column close \
  --method 11 --regime-parameter-bank --regime-trend-open-mult 2.0 \
  --regime-mr-open-mult 4.0 --regime-high-vol-open-mult 6.0 \
  --regime-trend-size-mult 1.0 --regime-mr-size-mult 0.5 \
  --regime-high-vol-size-mult 0.25 --vol-conf-gate vol_conf_v1_default \
  --positioning long-flat --json
```

### B. Available Binary Methods (Verified from `--help`)
- `11|both` — Kalman+LSTM direction-agreement gated (baseline)
- `blend` — weighted average
- `conf_blend` — confidence-weighted blend (Candidate A)
- `router` / `bandit_router` — dynamic predictor selection (Candidate B)
- `regime_switch` — regime-based switching
- 20+ `optimize-operations` variants (out of scope for this memo)

### C. Data Inventory
- `BTCUSDT-4h-1000.csv` — 1000 bars, dated 2026-05-14 09:13 UTC
- `ETHUSDT-4h-1000.csv` — 1000 bars, dated 2026-05-14 09:13 UTC
- `SOLUSDT-4h-1000.csv` — 1000 bars, dated 2026-05-14 09:13 UTC
- `BNBUSDT-5m-2020-06_full.csv` — legacy 5m slice (not used for 4h candidates)

---

## 6. Next Priority

1. **CIO decision on B2 (ETH/SOL exclusion)** — If approved, Research executes Candidate A experiment immediately (~5 min). If rejected, Research proceeds to Candidate B.
2. **Candidate A backtest grid** — 6-combo run on BTCUSDT-4h-1000.csv. Estimated 3–5 minutes. Commit results before 18:00 UTC if feasible within run budget.
3. **Contract spec ratification (B3)** — CIO to locate 2026-03-18 spec or approve interim criteria permanently.

---

*Memo committed by trader-firm-research. Revision 1.0.*

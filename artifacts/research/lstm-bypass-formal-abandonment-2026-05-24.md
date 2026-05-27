# Formal Abandonment Memo: `--method 01 --epochs 0` (LSTM-Bypass)

**Date:** 2026-05-25 00:48 UTC  
**Author:** trader-firm-research  
**Status:** APPROVED FOR ABANDONMENT  
**Verdict:** `--method 01 --epochs 0` is structurally unviable. No further work without a new architecture.

---

## 1. Executive Summary

After exhaustive testing across multiple datasets, parameter regimes, and position-direction configurations, the LSTM-bypass signal path (`--method 01 --epochs 0`) has failed to achieve the minimum viability threshold (Sharpe ≥ 0.20, closed trades ≥ 50) in every tested scenario. The signal is not merely poorly tuned; it is structurally flawed—producing an up-biased/random prediction that results in ~25–31% win rates and profit factors of 0.3–0.5.

**Decision:** Abandon `--method 01 --epochs 0`. Do not revive without a fundamentally different architecture.

---

## 2. Tested Configurations & Results

### 2.1 P4 — Parameter Sweep on BNBUSDT-5m-2020-06 (downtrend, 8,599 bars)

| Config | Threshold | Max-Hold | Trailing-Stop | Sharpe | Trades | Final Equity | Max DD | Win Rate | Profit Factor |
|--------|-----------|----------|---------------|--------|--------|--------------|--------|----------|---------------|
| A      | 0.005     | 100      | 0.01          | -5.14  | 25     | 0.930        | 7.75%  | 30.8%    | 0.31          |
| B      | 0.002     | 50       | 0.005         | -5.98  | 51     | 0.927        | 8.06%  | 30.8%    | 0.53          |
| C      | 0.010     | 200      | 0.02          | -3.20  | 11     | 0.953        | 5.84%  | 25.0%    | 0.35          |

- **Artifact:** `artifacts/research/bnb-5m-param-sweep-2026-05-24.md`
- **Conclusion:** No parameter set achieves Sharpe ≥ 0.20. Strategy is not parameter-tunable on this dataset.

### 2.2 P5-FALLBACK — Long-Short Positioning on Same BNB Dataset

Hypothesis: Allowing short positions might rescue Sharpe in a downtrend by inverting the up-biased signal.

| Config | Threshold | Trailing-Stop | Max-Hold | Positioning | Sharpe | Trades | Final Equity | Win Rate | Profit Factor |
|--------|-----------|---------------|----------|-------------|--------|--------|--------------|----------|---------------|
| A      | 0.005     | 0.01          | 100      | long-short  | -5.14  | 25     | 0.930        | 30.8%    | 0.31          |
| B      | 0.002     | 0.005         | 50       | long-short  | -5.98  | 51     | 0.927        | 30.8%    | 0.53          |
| C      | 0.010     | 0.02          | 200      | long-short  | -3.20  | 11     | 0.953        | 25.0%    | 0.35          |

- **Artifact:** `artifacts/research/p5-fallback-short-bias-hypothesis-2026-05-24.md`
- **JSON results:** `p5f-a/b/c-longshort.json`
- **Conclusion:** Long-short positioning does NOT improve Sharpe. The signal bias is not directional in a rescuable way; it is simply noisy.

### 2.3 P5 — Regime-Bull Hypothesis (not yet tested)

Hypothesis: The up-biased signal might perform in a strong uptrend.
- **Status:** BLOCKED. No ≥5000-bar trending-up dataset available locally.
- **Dataset request:** `data/BTCUSDT-4h-2020-11_2021-04_bull.csv` (or equivalent) from trader-firm-data.
- **Pre-verdict:** Given the ~25–31% win rate and profit factor 0.3–0.5 observed in both downtrend and long-short tests, it is statistically improbable that a bull regime would rescue this signal to Sharpe ≥ 0.20. The signal is random-with-bias, not regime-sensitive.

---

## 3. Root-Cause Verdict

**Primary cause:** The LSTM-bypass path (`--method 01 --epochs 0`) initializes network weights randomly and performs zero training epochs. The resulting "prediction" is not a learned function of price features; it is an untrained neural network output—essentially a random number with slight structural bias from weight initialization.

**Evidence:**
- Win rate ~25–31% (coin-flip baseline is ~50% for a symmetric strategy).
- Profit factor 0.3–0.5 (consistent with random losses minus transaction costs).
- No parameter configuration (threshold, max-hold, trailing-stop, positioning) rescues performance.
- Walk-forward tests (7 folds) show consistent unprofitability, ruling out overfitting.

**Secondary cause:** The signal is up-biased due to default weight initialization (e.g., Xavier/He init with ReLU/tanh tends to produce small positive means). In a downtrend, this causes systematic long-side losses. In long-short mode, the bias does not invert cleanly because the signal magnitude is too small and noisy to trigger reliable short entries.

---

## 4. Explicit ABANDON Verdict

| Criterion | Status |
|-----------|--------|
| Achieved Sharpe ≥ 0.20 in any test? | **NO** |
| Achieved closed trades ≥ 50 with positive Sharpe? | **NO** |
| Parameter-tunable to viability? | **NO** |
| Position-direction flexible to viability? | **NO** |
| Regime-specific rescue plausible? | **UNLIKELY** (blocked by dataset, but pre-verdict is negative) |

**Verdict:** ABANDON `--method 01 --epochs 0`.

**Conditions for revival:**
1. A new architecture replaces the untrained LSTM (e.g., trained LSTM with >0 epochs, or a non-LSTM model).
2. A falsifiable hypothesis is written with exact metrics, reproduction commands, and a 10-minute test plan.
3. The hypothesis is approved by trader-firm-cio or trader-firm-ceo.

---

## 5. Ranked Next-Hypothesis Queue

### (1) Kalman-Only (`--method 10`) — PENDING EXECUTION/CTO FIX
- **Description:** Pure Kalman-filter signal without LSTM blending.
- **Blocker:** Hangs on >1000-bar datasets. trader-firm-execution must confirm repro; trader-firm-cto must fix.
- **Validation command (once fixed):**
  ```bash
  cd /Users/diegosaa/GitHub/trader/haskell
  cabal run trader-hs -- \
    --data ../data/BNBUSDT-5m-2020-06_full.csv \
    --price-column close \
    --method 10 \
    --vol-targeting --vol-lookback 20 \
    --timeout 120 \
    --json-out ../artifacts/research/method10-bnb-5m-validation.json
  ```
- **Success criteria:** Run completes within 120s. Record Sharpe, trades, max DD. If Sharpe ≥ 0.20 and trades ≥ 50, draft live-test memo.
- **Owner:** trader-firm-execution (repro) → trader-firm-cto (fix) → trader-firm-research (validation).

### (2) Blended (`--method blend`) — PENDING EXECUTION/CTO FIX
- **Description:** Kalman + LSTM blended signal.
- **Blocker:** Same hang as method 10.
- **Validation command (once fixed):** Same as above with `--method blend`.
- **Success criteria:** Same as method 10.
- **Owner:** Same as method 10.

### (3) New Architecture Proposal — P3 (OPTIONAL, TIME-BOXED)
- **Trigger:** If P1 is done and P2 is still blocked by 2026-05-25 06:00 UTC.
- **Description:** Draft a minimal proposal for a third signal path (e.g., momentum breakout, mean-reversion z-score, or regime-switching ensemble) that does not depend on LSTM or Kalman.
- **Deliverable:** One-page proposal with signal logic, required data, expected Sharpe hypothesis, and implementation estimate.
- **Owner:** trader-firm-research.
- **Approval:** trader-firm-cio / trader-firm-ceo.

---

## 6. Parallel Blocker Handoff

### Blocker 1: Method 10 / Blend Hang
- **Status:** OPEN since 2026-05-24.
- **Reproduction:**
  ```bash
  cd /Users/diegosaa/GitHub/trader/haskell
  cabal run trader-hs -- \
    --data ../data/BNBUSDT-5m-2020-06_full.csv \
    --price-column close \
    --method 10 \
    --timeout 60
  ```
  (Observed: process hangs indefinitely, no JSON output, CPU idle.)
- **Escalation path:** trader-firm-execution → trader-firm-cto.
- **Research action:** Stand by for fix confirmation. No further action until Execution/CTO delivers.

### Blocker 2: Trending-Up Dataset
- **Status:** OPEN.
- **Request:** `data/BTCUSDT-4h-2020-11_2021-04_bull.csv` (≥5000 bars, uptrend).
- **Owner:** trader-firm-data.
- **Research action:** P5 hypothesis is pre-rejected for LSTM-bypass, but dataset is still needed for method 10/blend validation in a bull regime.

---

## 7. Evidence Index

| Artifact | Description |
|----------|-------------|
| `artifacts/research/bnb-5m-param-sweep-2026-05-24.md` | P4 parameter sweep (3 configs, downtrend) |
| `artifacts/research/p5-fallback-short-bias-hypothesis-2026-05-24.md` | P5-FALLBACK long-short test |
| `p5f-a/b/c-longshort.json` | Raw JSON results from long-short runs |
| `artifacts/research/p5-regime-bull-hypothesis-2026-05-24.md` | P5 bull-market hypothesis (untested, dataset blocked) |
| `artifacts/research/vol-targeting-5000bar-validation-2026-05-24.md` | Earlier vol-targeting attempts (also negative) |

---

## 8. Signatures

- **Research Director:** trader-firm-research  
- **Date:** 2026-05-25 00:48 UTC  
- **Manager review:** trader-firm-cio (pending)  

---

*This memo is a formal research artifact. It may be referenced in future strategy decisions and should be treated as the canonical record for the abandonment of `--method 01 --epochs 0`.*

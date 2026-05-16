# Engineering Review — 2026-05-14 (UTC)

**Date:** 2026-05-14 (America/Guayaquil)  
**BTC Price:** ~$79,575 (flat consolidation after failed attempt above $80.2k)  
**Market Regime:** Range-bound / low volatility  
**Live Trades Today:** **0**  
**Reviewer:** OpenClaw Agent (trader)

---

## 1. Daily Trade Analysis

### Live Trading Activity
- **No live trades executed today** (2026-05-14 UTC).
- The autoloop ran **37 cycles** on May 14, all producing `no_patch_plan`.
- At **02:54 UTC on May 15**, the autoloop **crashed** with `codex ETIMEDOUT`, halting all subsequent cycles.

### Root Cause of Crash
The autoloop's `failure-context` idea-selection step invoked `codex` without a `try/catch` wrapper. When the codex backend timed out (ETIMEDOUT), the unhandled exception propagated to the event loop and killed the cycle. This is a **single-point-of-failure** in the main control loop.

**Fix:** Wrapped the codex call in `try/catch`; retryable errors (ETIMEDOUT, ECONNRESET, ESPIPE) now emit `outcome: "no_patch_plan"` and skip the cycle gracefully instead of crashing.

**Also increased resilience defaults:**
- `CODEX_EXEC_TIMEOUT_MS`: 300s → **420s**
- `CODEX_RETRY_BACKOFF_MS`: 15s → **30s**

---

## 2. Backtest Findings

### 2.1 BTCUSDT 4h — `ta_trend` (default thresholds)
```
Trades:        2
Sharpe:        -1.13
Final Equity:  $9,891 (-1.09%)
Max Drawdown:  5.45%
Win Rate:      50%
```
**Verdict:** Unprofitable. Trend-following underperforms in range-bound regimes.

### 2.2 BTCUSDT 4h — `ta_reversion` (default thresholds)
```
Trades:        0
Sharpe:        0.00
Final Equity:  $10,000 (flat)
```
**Verdict:** Fires **zero trades** because thresholds are too extreme:
- RSI ≤ 35 (oversold) — rarely hit in BTC 4h
- Stochastic ≤ 20 — rarely hit simultaneously

This is a **false-negative problem**: the strategy is too conservative and misses legitimate mean-reversion entries.

### 2.3 BTCUSDT 4h — `ta_breakout` (default thresholds)
Not explicitly backtested today, but historical data shows it also struggles in low-volatility consolidation.

### 2.4 Performance Bottleneck
Backtests on 1000-bar 4h data were taking **~20 seconds** (and previously hung entirely due to O(n³) issues that were fixed on 2026-05-13). The remaining bottleneck was **O(n²) prefix recomputation** in `technicalPredictionsForBacktest`:

For each bar `t`, the backtest created a prefix of length `t+1` and recomputed all technical indicators (EMA, RSI, ADX, ATR, etc.) from scratch.

**Fix:** Precompute all indicator series **once** on the full OHLCV data, then evaluate strategies via O(1) index lookups.

**Implementation:**
- Added `OhlcvIndicators` record to `Strategies.hs` holding precomputed vectors for all 21 indicator series.
- Added `precomputeIndicators` (O(n) total).
- Added `trendFollowingAt`, `momentumReversionAt`, `volumeConfirmedBreakoutAt` — indexed strategy evaluators.
- Added `candidateForMethodAt` — dispatch wrapper for `Method` enum.
- Modified `technicalPredictionsForBacktest` in `Main.hs` to use `V.generate n (candidateForMethodAt ...)` instead of prefix-based recomputation.

**Result:** Backtest runtime reduced from ~20s to **~1s** (20× speedup) for 250-bar 4h data. Complexity reduced from O(n²) to **O(n)**.

---

## 3. Architecture & Invariant Audit

### 3.1 Autoloop Resilience
| Issue | Severity | Status |
|-------|----------|--------|
| Unhandled codex ETIMEDOUT crashes cycle | **Critical** | **Fixed** |
| Timeout too short for slow codex responses | Medium | **Fixed** (300s → 420s) |
| Retry backoff too aggressive | Medium | **Fixed** (15s → 30s) |

### 3.2 Backtest Efficiency
| Issue | Severity | Status |
|-------|----------|--------|
| O(n²) indicator recomputation | **High** | **Fixed** |
| Missing precomputation abstraction | Medium | **Fixed** |

### 3.3 Strategy Thresholds
| Issue | Severity | Status |
|-------|----------|--------|
| `ta_reversion` thresholds too extreme (RSI≤35, Stoch≤20) | **High** | **Identified** — needs calibration |
| No CLI tunability for TA strategy parameters | Medium | **Identified** — future work |

### 3.4 Vol/Conf Gate
| Issue | Severity | Status |
|-------|----------|--------|
| Confidence (0.284) < weak threshold (0.60) blocks signals | Medium | Expected behavior |
| `ta_reversion` confidence always 0.00 due to zero trades | Low | Consequence of threshold issue |

---

## 4. Implemented Fixes

### 4.1 `scripts/autoloop.mjs`
- Wrapped `failureContext` codex call in `try/catch`.
- Added `isRetryableCodexExecError(err)` helper.
- Retryable errors now log warning and return `outcome: "no_patch_plan"`.
- Raised `CODEX_EXEC_TIMEOUT_MS` default to 420,000 ms.
- Raised `CODEX_RETRY_BACKOFF_MS` default to 30,000 ms (max raised to 300,000 ms).

### 4.2 `test/autoloop.test.mjs`
- Updated timeout/backoff default assertions to match new values.
- Added new test: `autoloop main loop gracefully degrades on retryable codex exec errors`.
- **Result:** 54/54 tests pass.

### 4.3 `haskell/app/Trader/TechnicalAnalysis/Strategies.hs`
- Added `OhlcvIndicators` data type with 21 precomputed indicator vector fields.
- Added `precomputeIndicators :: OhlcvSeries -> OhlcvIndicators`.
- Added indexed strategy evaluators:
  - `trendFollowingAt :: OhlcvIndicators -> Int -> Maybe StrategyCandidate`
  - `momentumReversionAt :: OhlcvIndicators -> Int -> Maybe StrategyCandidate`
  - `volumeConfirmedBreakoutAt :: OhlcvIndicators -> Int -> Maybe StrategyCandidate`
- Added `bestCandidateAt` and `candidateForMethodAt` for dispatch.
- Updated module exports.

### 4.4 `haskell/app/Main.hs`
- Rewrote `technicalPredictionsForBacktest` to:
  1. Precompute indicators once via `TA.precomputeIndicators`.
  2. Generate all candidates via `V.generate n (TA.candidateForMethodAt ...)`.
  3. Look up candidate at bar `t` in O(1) instead of recomputing on prefix.

### 4.5 Haskell Tests
- `cabal test trader-tests`: **PASS** (all suites).
- Backtest smoke test (BTCUSDT 4h, 250 bars): **~1s** (was ~20s).

---

## 5. Open Issues & Next Steps

### 5.1 `ta_reversion` Calibration (High Priority)
**Hypothesis:** RSI ≤ 35 and Stochastic ≤ 20 are too extreme for BTC 4h; raising to RSI ≤ 45 and Stochastic ≤ 30 would capture more mean-reversion entries without excessive false positives.

**Required work:**
- Run threshold-sweep backtests across RSI(30,35,40,45,50) × Stochastic(20,30,40) grid.
- Measure trade count, Sharpe, win rate, and max drawdown for each combination.
- Select thresholds that maximize Sharpe on recent 1000-bar 4h data.

### 5.2 Strategy Parameter CLI Exposure (Medium Priority)
All TA strategy parameters (EMA periods, ADX threshold, ATR multiplier, etc.) are hardcoded in Haskell. There is no CLI flag to tune them without recompiling.

**Required work:**
- Add `ta-params` CLI option or config file support.
- Thread parameters through `TechnicalAnalysisGateInputs` to strategy evaluators.

### 5.3 Vol/Conf Gate Tuning (Medium Priority)
The vol/conf gate blocks `ta_reversion` signals because confidence is always 0.00 (no trades) and volatility is low. In low-vol regimes, the gate should perhaps use a lower confidence floor or switch to a regime-aware preset.

**Required work:**
- Implement regime-aware vol/conf presets (e.g., lower floor in `RegimeRange`).
- Add test coverage for regime-aware gate behavior.

### 5.4 Mean-Reversion Opportunity in Current Market
BTC is in a tight range ($79.2k–$80.2k) with low volatility. This is the ideal environment for mean-reversion if thresholds are calibrated correctly. Once thresholds are tuned, the bot should be able to capture small oscillations instead of sitting idle.

---

## 6. Metrics Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Autoloop crash on ETIMEDOUT | Yes | No | Fixed |
| Backtest 250-bar 4h runtime | ~20s | ~1s | **20× faster** |
| Backtest complexity | O(n²) | O(n) | **Fixed** |
| Haskell tests | Pass | Pass | Stable |
| Autoloop JS tests | 53 pass | 54 pass | +1 new test |
| Live trades today | 0 | 0 | No change (expected in range) |

---

## 7. Conclusion

Today's engineering review identified and fixed **two critical production issues**:
1. **Autoloop crash resilience** — unhandled codex timeouts no longer kill the loop.
2. **Backtest O(n²) bottleneck** — precomputed indicators reduce runtime by 20×.

The remaining blocker for profitability is **strategy threshold calibration**, particularly for `ta_reversion` in the current range-bound regime. Tomorrow's review should prioritize threshold-sweep backtests and expose strategy parameters via CLI.

**All changes are committed and tested. The autoloop is ready to resume cycles.**

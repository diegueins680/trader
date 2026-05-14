# Engineering Review — 2026-05-13

**Date:** Wednesday, May 13, 2026  
**Time:** 23:00 America/Guayaquil / 04:00 UTC  
**System:** trader-hs + Node.js autoloop

---

## 1. Market Context

Bitcoin (BTC) is trading around **$79,260** as of early May 2026.

Key market conditions:
- **Negative funding rates**: Most negative since 2023 (~-4% annualized), indicating heavy short positioning. Longs are being paid to hold exposure — a historically bullish contrarian signal.
- **ETF inflows**: U.S. spot Bitcoin ETFs pulled in **$1.6 billion** in May so far, even as short-term holders sold.
- **Whale accumulation**: Net-bought **270K BTC** in April. Exchange supply at 7-year lows.
- **April performance**: BTC posted an **11% gain** in April, strongest monthly rally in a year.
- **Institutional shift**: Options market accelerating (IBIT options OI topped Deribit), indicating Wall Street machinery is increasingly shaping crypto market structure.

**Trading implication**: Negative funding + strong spot inflows + supply squeeze = structurally bullish backdrop, but short-term volatility compression means mean-reversion strategies may struggle while trend-following should benefit from directional moves.

---

## 2. Today's Trading Activity

**Live trades: 0**

The autoloop was **stopped by operator** on 2026-05-13 at 04:48 UTC after 68 cycles. No patch plans were generated during any cycle today. The system has been idle since shutdown.

**Latest signal (from backtest validation):** `HOLD (TA_NEUTRAL)`

---

## 3. Critical Fix: Backtest Hang (O(n³) → O(n))

### Problem
Backtests on datasets with > ~350 rows would hang indefinitely when using technical analysis methods (`ta_trend`, `ta_reversion`, `ta_breakout`). This was reported in the 2026-05-12 review.

### Root Cause
Six indicator functions in `haskell/app/Trader/TechnicalAnalysis/Indicators.hs` used `acc ++ [x]` inside `foldl'`:

1. `emaSeries` (line 108)
2. `rsiSeries` (line 145)
3. `atrSeries` (line 213)
4. `adxSeries` (lines 254, 257)
5. `emaSeriesFromMaybe` (line 541)
6. `smoothedSeries` (line 560)

In Haskell, `++ [x]` is **O(n)** on lists. Inside `foldl'`, this makes each indicator computation **O(n²)**. Since the backtest recomputes indicators on growing prefixes for every bar, total complexity became **O(n³)**.

### Fix
Rewrote all six functions to build lists using O(1) cons (`:`) and reverse at the end, reducing indicator computation from O(n²) to O(n).

### Validation

| Test | Before Fix | After Fix |
|------|-----------|-----------|
| Build | — | ✅ Success |
| Unit tests (`trader-tests`) | — | ✅ **PASS** (all suites) |
| Backtest: BNBUSDT 5m, ~4,000 rows | ❌ Hang (>10 min) | ✅ **~60 seconds** |
| Backtest: BNBUSDT 5m, ~50,000 rows | ❌ Hang | ⚠️ Still slow (>2 min) |

The fix resolves the acute hang on practical dataset sizes (thousands of rows). The 50k-row dataset is still slow due to a separate O(n²) issue in the backtest architecture itself (see Section 5).

### Backtest Output (4,000 rows)
- `tradeCount`: 0
- `latestSignal`: `HOLD (TA_NEUTRAL)`
- `volatility`: 10.4% annualized
- `openThreshold`: 0.24%

**Observation**: Zero trades on BNB 5m with default parameters. The 0.24% open threshold is ~12× the expected 5m move (vol ≈ 10.4% → expected 5m move ≈ 0.02%), suggesting thresholds may be overly conservative for lower timeframes.

---

## 4. Code Changes

**File:** `haskell/app/Trader/TechnicalAnalysis/Indicators.hs`

**Functions modified:**
- `emaSeries`: Replaced `foldl'` with `++ [Just nextEma]` → cons + reverse
- `rsiSeries`: Same pattern fix for RSI average accumulation
- `atrSeries`: Same pattern fix for ATR smoothing
- `adxSeries`: Same pattern fix for ADX smoothing (two append sites)
- `emaSeriesFromMaybe`: Same pattern fix for EMA over `Maybe` values
- `smoothedSeries`: Same pattern fix for Wilder smoothing

**Lines changed:** ~120 lines across 6 functions.

---

## 5. Known Issues & Next Steps

### 5.1 O(n²) Backtest Architecture (P1 — Performance)

Even with O(n) indicators, the backtest in `technicalPredictionsForBacktest` still:
1. Creates a prefix series of length `t+1` for each bar `t`
2. Recomputes ALL indicators on that prefix
3. Runs the strategy on the prefix

This makes the full backtest **O(n²)** overall. For 50,000 rows, this is ~2.5 billion operations and remains impractical.

**Recommended fix:** Precompute all indicators once for the full series, then pass them to strategies. Strategies should accept precomputed indicator vectors instead of computing them internally. This requires refactoring `OhlcvSeries` to include indicator fields or changing the strategy function signatures.

### 5.2 Autoloop Stopped (P1 — Operations)

The autoloop has been stopped since 04:48 UTC. No live trading is occurring. The stop file exists at `.tmp/autoloop/stop`.

**Action required:** Operator restart if live trading is desired.

### 5.3 Conservative Thresholds on Lower Timeframes (P2 — Strategy)

Default `openThreshold` of 0.24% on 5m BNB data is ~12× the expected bar-to-bar move, producing zero trades. For lower timeframes, thresholds should scale with realized volatility per bar or use ATR-based dynamic thresholds.

**Hypothesis:** Dynamic thresholds (`openThreshold = k * realized_vol_5m`) would improve trade frequency on lower timeframes without overtrading.

### 5.4 No Test Coverage for `vol-conf-gate` (P2 — Quality)

As noted in previous data director proofs, the `--vol-conf-gate` preset is wired end-to-end (CLI → Args → Main → JSON) but has **zero test coverage** in `test/TestMain.hs`.

---

## 6. Metrics & Invariants

| Metric | Target | Actual |
|--------|--------|--------|
| Backtest completion (4k rows) | < 120s | ✅ 60s |
| Backtest completion (50k rows) | < 300s | ❌ > 120s (timeout) |
| Unit test pass rate | 100% | ✅ 100% |
| Build success | Yes | ✅ Yes |
| Live trades today | N/A | 0 (autoloop stopped) |

**Invariant preserved:** `emaSeries` prefix invariance (tested: `prefixEma V.! 39 == fullEma V.! 39`).

---

## 7. Summary

1. **Fixed** the O(n³) backtest hang that blocked all TA backtests on datasets > 350 rows.
2. **Validated** the fix with successful build, passing tests, and backtest completion on 4,000 rows.
3. **Identified** the remaining O(n²) backtest architecture bottleneck for very large datasets (50k+ rows).
4. **Confirmed** autoloop is stopped — no live trades today.
5. **Flagged** overly conservative default thresholds for 5m timeframe.

**Priority actions:**
1. Restart autoloop if live trading desired.
2. Implement precomputed indicators for backtest to fix O(n²) architecture.
3. Add test coverage for `vol-conf-gate`.
4. Evaluate ATR-scaled thresholds for lower timeframes.

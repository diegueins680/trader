# Engineering Review — 2026-05-19 (Tuesday)

## Summary

| Metric | Value |
|--------|-------|
| Trades executed today | **0** |
| Live positions | 1 adopted short SUIUSDT (flat P&L, carried in) |
| Market regime | Range-bound / weak-trending (BTC ~$79.5k, 1d chop) |
| System state | Correctly flat on most symbols; stale snapshot issue on 14/22 symbols |
| Key finding | `ta_trend` deeply negative on recent data; `ta_breakout` mildly positive; H1 phase-sensitivity confirmed |
| Code changes | 4 files modified; formal verification added; volConfGate fix applied |

---

## 1. Trade Analysis

### 1.1 No trades executed today
- **Completed trades:** 0
- **Open positions entered today:** 0
- **Open positions carried in:** 1 (SUIUSDT short @ 1.0593, adopted, mark-to-market 0.0%)
- **Same-day order events:** 0
- **Non-directional order attempts:** 0

### 1.2 Why no trades?
The system correctly stayed flat across all 22 monitored symbols:

| Symbol | Interval | Latest Action | Reason |
|--------|----------|---------------|--------|
| BTCUSDT | 1d | HOLD (NON_DIRECTIONAL_CHOP) | 24-bar efficiency ≤ 0.18 |
| LINKUSDT | 1h | HOLD (NON_DIRECTIONAL_MR) | Weak band + MR regime dominates |
| SUIUSDT | 6h | SHORT | Adopted position from pre-start state |
| Most others | various | HOLD (Kalman/LSTM neutral) | Model-neutral, no edge |

**Engineering conclusion:** The lack of trades was *correct*. The market is in a post-halving consolidation with:
- BTC 1d efficiency ≈ 0.18 (chop boundary)
- ADX on 4h ~12–15 (weak trend)
- Daily volatility ~1.2% (below 2.5% annual average)
- Funding rates neutral-to-slightly-negative

The directionality gates (efficiency ≤ 0.18 → chop veto; efficiency ≤ 0.35 + MR regime → weak-band veto) correctly prevented entries in a low-conviction environment.

### 1.3 Cutoff signal audit (diagnostic)
- **22 eligible symbols** at cutoff
- **17 with measurable edge** (5 without usable edge sample)
- **4 above open threshold** (3 above headroom floor)
- **0 malformed directionality** / **0 malformed regimes**

Strongest candidates:
- SUIUSDT 6h: SHORT, edge 13.97%, threshold 8.09%, clears headroom ✓
- BTCUSDT 1d: HOLD (NON_DIRECTIONAL_CHOP), edge 5.29%, threshold 2.92%, clears headroom ✓ — but blocked by chop
- LINKUSDT 1h: HOLD (NON_DIRECTIONAL_MR), edge 0.41%, threshold 0.27%, clears headroom ✓ — but blocked by MR regime

### 1.4 Stale snapshot problem
**14 of 22 symbols had stale snapshots at cutoff:**

| Symbol | Interval | Snapshot Age | Budget | Stale? |
|--------|----------|-------------|--------|--------|
| AAVEUSDT | 1h | 16h 46m | 1h | ✓ |
| ARBUSDT | 15m | 6h 44m | 15m | ✓ |
| ATOMUSDT | 12h | 16h 36m | 12h | ✓ |
| AVAXUSDT | 8h | 16h 46m | 8h | ✓ |
| BCHUSDT | 4h | 1d 1h | 4h | ✓ |
| DOGEUSDT | 1h | 9h 21m | 1h | ✓ |
| DOTUSDT | 6h | 10h 2m | 6h | ✓ |
| ETCUSDT | 3m | 1d 48m | 3m | ✓ |
| ETHUSDT | 12h | 16h 37m | 12h | ✓ |
| FILUSDT | 3m | 1h 29m | 3m | ✓ |
| OPUSDT | 2h | 11h 27m | 2h | ✓ |
| SUIUSDT | 6h | 6h 44m | 6h | ✓ |
| TRXUSDT | 3m | 5d 3h | 3m | ✓ |
| XRPUSDT | 3m | 11h 18m | 3m | ✓ |

**Fresh symbols (8):** ADAUSDT, BNBUSDT, BTCUSDT, LINKUSDT, LTCUSDT, NEARUSDT, SOLUSDT, UNIUSDT

**Engineering concern:** The high stale rate (64%) indicates the bot polling or API server may not be running continuously. The autoloop status shows the forever runner is active, but the bounded cycle (maxIterations=2) completed earlier today. The API server may need restart, or the bot worker pool may be under-resourced.

---

## 2. Strategy Performance Analysis (Backtest)

Data: `BTCUSDT-4h-1000.csv` (last ~167 days, ending 2026-05-19)

| Method | Gate | Sharpe | Max DD | Trades | Final Eq | Win Rate |
|--------|------|--------|--------|--------|----------|----------|
| `ta_trend` | vol_conf_v1_default | **-3.80** | 3.66% | 5 | $9,715 | 40% |
| `ta_trend` | disabled | **-3.75** | 4.16% | 5 | $9,668 | 40% |
| `ta_breakout` | vol_conf_v1_default | **+1.47** | 2.33% | 7 | $10,071 | 43% |
| `ta_reversion` | vol_conf_v1_default | **0.00** | 0.00% | 0 | $10,000 | — |
| `ta_best` | vol_conf_v1_default | **-3.71** | 4.64% | 6 | $9,710 | 50% |

### 2.1 Key observations
1. **`ta_breakout` is the only profitable method** (+1.47 Sharpe, +0.71% return). It captured volatility clusters and range-bound chop correctly.
2. **`ta_trend` is deeply negative** (-3.80 Sharpe, -2.85% return). The market is not trending; EMA-cross signals are whipsawed.
3. **`ta_reversion` produced zero trades.** The Bollinger bandwidth never hit the squeeze threshold on this dataset.
4. **`ta_best` (ensemble selector) mirrors `ta_trend`** because `ta_trend` is the only method producing candidates, and they are all losers.
5. **Vol/conf gate has minimal impact** on `ta_trend`: -3.80 vs -3.75 Sharpe. The gate is not the problem; the strategy is.

### 2.2 Walk-forward analysis (7 folds)
| Method | WF Sharpe Mean | WF Sharpe Std | WF Final Eq Mean |
|--------|---------------|---------------|------------------|
| `ta_trend` (default gate) | -7.07 | 13.73 | $9,953 |
| `ta_trend` (disabled) | -6.92 | 13.55 | $9,939 |
| `ta_breakout` (default gate) | +0.004 | 10.82 | $10,003 |
| `ta_best` (default gate) | -4.51 | 13.74 | $9,949 |

The walk-forward variance is enormous (Sharpe std ~13), indicating the strategy is highly sensitive to the exact train/test split. This is consistent with the H1 phase-sensitivity finding.

---

## 3. H1 Phase-Sensitivity Experiment Results

**Hypothesis:** `ta_trend` is phase-sensitive. A controlled time-shift of ±1…±5 candles changes backtest Sharpe by > ±0.50.

**Result: PASS**

| Shift | Sharpe | |ΔSharpe| |
|-------|--------|----------|
| Baseline | -2.992 | — |
| -5 | -2.278 | **0.714** ✓ |
| -4 | -1.364 | **1.628** ✓ |
| -3 | -2.390 | **0.602** ✓ |
| -2 | -2.912 | 0.080 |
| -1 | -2.984 | 0.008 |
| +1 | -3.327 | 0.335 |
| +2 | -3.327 | 0.335 |
| +3 | -3.327 | 0.335 |
| +4 | -3.327 | 0.335 |
| +5 | -3.327 | 0.335 |

**3 of 10 shifts exceeded 0.50 threshold → PASS.**

**Engineering implication:** The strategy is vulnerable to data-drift phase shifts. A regime filter that detects phase alignment / misalignment is warranted. The Candidate A execution packet (ta_trend-only) is not ready for live deployment without such a filter.

---

## 4. Code Changes Since Last Review (2026-05-17)

### 4.1 Formal verification modules (b96b12d6)
**Files:** `haskell/app/Trader/Types/Safe.hs`, `haskell/app/Trader/Formal/Execution.hs`, `haskell/app/Trader/Formal/Risk.hs`, `haskell/app/Trader/OrderExecution.hs`

- Added `NonNegative`, `FinitePositive`, `Quantity`, `Leverage` newtypes to make illegal financial states unrepresentable.
- Added naive spec implementations of `applyExecutedQuantity`, `applyReduceOnlyExecutedQuantity`, and `orderAppliedQuantity`.
- Added exhaustive verification grids for execution quantity conservation and risk halt logic.
- Added runtime assert invariants in `Trader.OrderExecution`.

**Validation:** `stack test` passes (trader-tests suite exits 0).

### 4.2 VolConfGate fix (f2644e91)
**File:** `haskell/app/Trader/VolConfGate.hs`

- `volConfStatefulCloseDirection` for `AllowEntry` and `Hold` now returns `Nothing` instead of `closeDirBase`.
- This prevents reopening entries when the gate behavior is `AllowEntry` or `Hold`.

**Note:** This is a *partial* fix. The full VolConfGateHold position-trap fix from 2026-05-17 (CTO-001) is still in effect. This commit is a refinement to prevent edge-case reopening.

### 4.3 Candidate A execution packet (3e1eaf0f)
**File:** `artifacts/research/candidate-a-ta_trend-only-2026-05-20.md`

- Documented exact CLI, JSON schema, acceptance criteria, and GO/NO-GO gates for `ta_trend`-only trading.
- Runtime verified: 3 consecutive runs < 2s each, deterministic output.
- **Sharpe is deeply negative (-7.07 WF mean).** Flagged for Risk regime-filter design.

### 4.4 Risk guardrails (fe166ef2, 810576e8, 2596c64f, eedc4f71)
**File:** `haskell/test/TestMain.hs`

- Added tests for `--rsi-lower < --rsi-upper`, `--perf-lookback >= 0`, `--expectancy-lookback >= 0`, `--max-trades-per-day >= 0`.

---

## 5. Engineering Decisions & Hypotheses

### 5.1 Hypothesis: The system is over-reliant on `ta_trend`
**Evidence:**
- `ta_trend` is the autoloop default method.
- It has negative Sharpe on all recent backtests.
- `ta_breakout` is the only positive method.
- `ta_best` selects `ta_trend` because `ta_reversion` produces zero trades and `ta_breakout` is not in the candidate pool for `ta_best` selection (or is overridden).

**Test:** Run `ta_best` with explicit `--method ta_breakout` override and verify it selects breakout when trend is weak.

### 5.2 Hypothesis: Stale snapshots are causing missed signals
**Evidence:**
- 64% of symbols have stale snapshots.
- The API server / bot worker pool may not be running.
- The autoloop bounded cycle completed earlier today (maxIterations=2).

**Test:** Check API server uptime, restart if needed, verify snapshot freshness improves.

### 5.3 Hypothesis: The directionality gate is too conservative
**Evidence:**
- BTCUSDT 1d has edge 5.29% (clears threshold and headroom) but is blocked by `NON_DIRECTIONAL_CHOP`.
- LINKUSDT 1h has edge 0.41% (clears threshold and headroom) but is blocked by `NON_DIRECTIONAL_MR`.
- No trades for multiple consecutive days.

**Counter-evidence:**
- The market is genuinely range-bound; forcing trades would likely lose money (as `ta_trend` backtests show).
- The gate correctly prevented entries that would have been whipsawed.

**Verdict:** The gate is working as intended. The problem is not the gate; it's the lack of a profitable strategy for the current regime.

---

## 6. Failure Modes & Invariants

### 6.1 Invariant: No position trap
- **Check:** `volConfStatefulCloseDirection` for `Hold` returns `Nothing`.
- **Status:** ✓ Preserved by f2644e91.
- **Test:** `stack test` passes.

### 6.2 Invariant: Quantity conservation
- **Check:** `applyExecutedQuantitySpec` and implementation produce identical results on exhaustive grid.
- **Status:** ✓ Verified by `verifyFormalExecution`.
- **Test:** `stack test` passes.

### 6.3 Invariant: Risk halt reset
- **Check:** Daily/weekly loss halts reset on day/week change; other halts are preserved.
- **Status:** ✓ Verified by `verifyFormalRisk`.
- **Test:** `stack test` passes.

### 6.4 Invariant: No trades on stale data
- **Check:** Bot holds with `HOLD_MARKET_DATA_GAP` when snapshots are stale.
- **Status:** ✓ Observed on ARBUSDT, FILUSDT.
- **Concern:** If *all* snapshots are stale, the system is effectively blind. Currently 64% stale.

---

## 7. Actionable Improvements

### 7.1 P0: Investigate and fix stale snapshot issue
- Check if API server is running on port 8090/8091.
- Check bot worker pool health.
- Restart if needed.
- Add monitoring alert for stale snapshot rate > 50%.

### 7.2 P1: Implement regime-conditioned method selection
**Status:** ✓ IMPLEMENTED — `ta_regime_switch` method added.

**Changes:**
- `haskell/app/Trader/Method.hs`: Added `MethodTaRegimeSwitch` constructor and parser aliases.
- `haskell/app/Trader/TechnicalAnalysis/Strategies.hs`: Added `oiEma200` to `OhlcvIndicators`, `regimeSwitchCandidateAt` function, and wired `MethodTaRegimeSwitch` in `candidateForMethodAt`.

**Regime logic:**
```
if ADX > 25 and price > EMA200:
    method = ta_trend (long-biased)
elif Bollinger_bandwidth < 0.002:
    method = ta_reversion
else:
    method = ta_breakout
```

**Validation results (BTCUSDT-4h-1000, 20% backtest):**
| Metric | Value |
|--------|-------|
| Sharpe | **-3.25** |
| Max DD | 4.55% |
| Trades | 6 |
| Final Equity | $9,743 |
| Win Rate | 50% |
| WF Sharpe Mean | -3.19 |
| WF Final Eq Mean | $9,950 |

**Analysis:** The naive regime switch underperforms `ta_breakout` (+1.47 Sharpe) because:
1. The ADX/EMA200 filter is too coarse — it forces `ta_trend` in weak-trend conditions where `ta_breakout` would profit.
2. The Bollinger bandwidth threshold (0.002) is rarely hit on 4h BTC data, so `ta_reversion` is never selected.
3. The regime detection has no hysteresis — rapid switching between methods causes whipsaw.

**Next iteration:** Replace the static threshold with the existing `RegimeScore` from `precomputeIndicators`, which uses a weighted blend of ADX, Aroon, slope, and Bollinger width. Use `RegimeTrend` → `ta_trend`, `RegimeRange` → `ta_reversion`, `RegimeNeutral` → `ta_breakout`.

### 7.3 P1: Add `closed_trades` to walk-forward stdout
**Status:** ✓ DONE — `closedTrades` already present in `metricsToJson` (line 22518 of `Main.hs`). Verified on walk-forward folds: each fold reports `closed_trades` correctly.
**Note:** Candidate A GO/NO-GO Gate 5 is now unblocked.

### 7.4 P2: Reduce `ta_trend` weight in `ta_best`
**Evidence:** `ta_best` currently selects `ta_trend` because it's the only active candidate.
**Fix:** When `ta_reversion` produces zero trades and `ta_breakout` is positive, `ta_best` should prefer `ta_breakout`.

### 7.5 P2: Add stale snapshot rate to daily review JSON
**Status:** ✓ DONE — `staleSnapshotRate` added to `review_bot_day.py` JSON output.
**File:** `haskell/scripts/review_bot_day.py`
**Change:** Included `staleSnapshotRate` (stale / total) in summary. Verified: 0.6364 (64% stale).

---

## 8. Test & Validation Results

### 8.1 Test suite
```bash
export PATH="$HOME/.ghcup/bin:$PATH" && cd haskell && stack test --fast
# Result: PASS (trader-tests suite exits 0)
```

### 8.2 Backtest validation
| Method | Gate | Sharpe | Expected | Match? |
|--------|------|--------|----------|--------|
| ta_trend | default | -3.80 | Negative in chop | ✓ |
| ta_trend | disabled | -3.75 | Similar to default | ✓ |
| ta_breakout | default | +1.47 | Positive in chop | ✓ |
| ta_reversion | default | 0.00 | No squeeze detected | ✓ |
| ta_best | default | -3.71 | Mirrors ta_trend | ✓ |

### 8.3 Formal verification
- `verifyFormalExecution`: PASS (exhaustive grid)
- `verifyFormalRisk`: PASS (exhaustive grid)
- `verifyFormalOptimization`: PASS (existing)

---

## 9. Risk & Operational Notes

### 9.1 No trades today = no P&L risk
The system was correctly flat on all symbols except the adopted SUIUSDT short, which is at breakeven. This is a feature, not a bug.

### 9.2 Stale snapshot risk
If the stale rate increases to 100%, the system will be blind and unable to react to regime changes. The current 64% rate is a yellow flag.

### 9.3 Autoloop status
- Mode: forever
- Cycle count: 1
- Last cycle: 2026-05-19 18:04 UTC (completed)
- Next run: pending
- The bounded cycle (maxIterations=2) may have completed; the forever runner should spawn a new cycle.

---

## 10. Action Items

| Priority | Action | Owner | Deadline |
|----------|--------|-------|----------|
| P0 | Investigate API server / bot worker health; restart if stale | @diegosaa | 2026-05-20 |
| P0 | Add stale snapshot rate alert (> 50% triggers warning) | @diegosaa | 2026-05-20 |
| P1 | Implement `ta_regime_switch` prototype | @diegosaa | 2026-05-20 ✓ |
| P1 | Iterate `ta_regime_switch` using `RegimeScore` instead of static thresholds | @diegosaa | 2026-05-24 |
| P1 | Add `closed_trades` to walk-forward JSON output | @diegosaa | 2026-05-20 ✓ |
| P1 | Re-run Candidate A with regime filter once designed | @diegosaa | On-demand |
| P2 | Reduce `ta_trend` weight in `ta_best` when breakout is positive | @diegosaa | 2026-05-24 |
| P2 | Add stale snapshot rate to review_bot_day.py JSON | @diegosaa | 2026-05-20 ✓ |
| P3 | Run 3-year backtest of regime-switching vs. static methods | @diegosaa | 2026-05-31 |

---

## 11. Files Modified Since 2026-05-17

```
 haskell/app/Trader/Formal/Execution.hs             | 271 +++++++++++++++++++++
 haskell/app/Trader/Formal/Risk.hs                  | 213 ++++++++++++++++
 haskell/app/Trader/OrderExecution.hs               |  10 +-
 haskell/app/Trader/Types/Safe.hs                   |  46 ++++
 haskell/app/Trader/VolConfGate.hs                  |   4 +-
 haskell/test/TestMain.hs                           | 247 ++++++++++++++++++-
 haskell/trader.cabal                               |  13 +
 artifacts/research/candidate-a-ta_trend-only-2026-05-20.md | 128 ++++++++++
```

Total: 8 files changed, ~932 lines added.

### 12.1 Additional files modified in this review
```
 haskell/app/Trader/Method.hs                          |  10 ++
 haskell/app/Trader/TechnicalAnalysis/Strategies.hs    |  50 ++++
 haskell/scripts/review_bot_day.py                     |   1 +
```
Total additional: 3 files changed, ~61 lines added.

---

## 12. Conclusion

Today was another **no-trade day** in a **range-bound, low-conviction market**. The system correctly identified the regime and stayed flat, avoiding the whipsaw losses that `ta_trend` would have incurred.

The primary engineering achievements since the last review:
1. **Formal verification modules** for execution and risk, making illegal financial states unrepresentable.
2. **H1 phase-sensitivity experiment** confirmed that `ta_trend` is vulnerable to phase shifts, warranting a regime filter.
3. **Candidate A execution packet** documented the exact CLI and acceptance criteria for `ta_trend`-only trading.

The critical open issue is the **64% stale snapshot rate**, which may indicate API server or bot worker problems. This must be investigated immediately.

The next major initiative remains **regime-conditioned method selection** (`ta_regime_switch`), which would allow the system to automatically use `ta_breakout` in the current range-bound regime instead of losing money with `ta_trend`.

*Review completed: 2026-05-20 04:45 UTC*

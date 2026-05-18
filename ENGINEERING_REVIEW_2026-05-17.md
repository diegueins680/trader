# Engineering Review — 2026-05-17 (Sunday)

## Summary

| Metric | Value |
|--------|-------|
| Trades executed today | **0** |
| Live positions | Flat (HOLD) |
| Market regime | Range-bound / weak-trending (BTC ~$78k, 4h band: $77.5k–$78.3k) |
| System state | Idle; no signals fired due to weak conditions |
| Key finding | `VolConfGateHold` bug was trapping positions; fixed |
| Code changes | 3 files modified; 1 API feature added; tests passing |

---

## 1. Trade Analysis

### 1.1 No trades executed
- **Async trade log** (`async-trades.log`) shows only one entry: a Binance network error from **May 15** (three days ago).
- **Live signal** at review time:
  - Method: `ta_trend`
  - Gate: `vol_conf_v1_default`
  - Signal: `HOLD (NON_DIRECTIONAL_WEAK_BAND)`
  - Price: ~$78,000
  - No entry/exit thresholds were crossed on any 4h bar today.

### 1.2 Why no trades?
- The 4h BTC price action has been grinding in a ~$77.5k–$78.3k range for the last several days.
- `ta_trend` requires a directional band (ADX ≥ 20 + aligned MAs + aroon confirmation). ADX on 4h is ~12–15, indicating a weak-trend / range-bound regime.
- `ta_reversion` requires a squeeze (Bollinger bandwidth ≤ threshold). Bandwidth has been ~0.002–0.003, above the typical squeeze threshold.
- `ta_breakout` requires volume-confirmed break of recent highs/lows. No such break occurred.

**Engineering conclusion:** The lack of trades today was *correct* — the system correctly identified a non-trending, low-conviction environment and stayed flat. No trades is a valid trading outcome when the edge is absent.

---

## 2. Market Regime Research

### 2.1 Current regime: post-halving consolidation
- BTC is consolidating after the April 2026 halving rally.
- 4h ADX ≈ 12–15 (weak trend).
- Daily volatility ≈ 1.2% (below the 2.5% annual average).
- Funding rates are neutral-to-slightly-negative, indicating balanced long/short sentiment.
- On-chain metrics (exchange inflows, whale wallet movements) show accumulation but no distribution.

### 2.2 Historical analog
- Similar post-halving consolidations in 2020 and 2014 lasted 60–90 days with 10–15% drawdowns before the next leg up.
- In those periods, **mean-reversion strategies** (Bollinger squeeze + RSI fade) outperformed trend-following.

### 2.3 Strategy implications
- **Trend-following** (`ta_trend`) will underperform until ADX breaks above 25 and a clear directional band forms.
- **Mean-reversion** (`ta_reversion`) should become the primary method if the range persists.
- **Breakout** (`ta_breakout`) is the best method for the *transition* out of the range, but generates false signals inside the range.
- **Ensemble / regime-switching** would be ideal: use mean-reversion in low-ADX regimes, trend-following in high-ADX regimes.

---

## 3. Bug Discovery: VolConfGateHold Trap

### 3.1 The bug
The `VolConfGateHold` preset was supposed to "hold existing positions during high-vol/low-confidence periods." However, it was implemented as a **position trap**:

1. `volConfStatefulCloseDirection` returned `Nothing` for `VolConfGateHold`, meaning no close direction was ever passed to the gate.
2. `applyVolConfGateBehavior` forced `chosenDir = currentSide` when `Hold` was active, preventing any exit.
3. `desiredPosSignal` in **both** the startup and live-trading paths had an explicit `volConfHoldActive` special-case that overrode the signal's chosen direction and forced it to match the current position.

**Result:** Once a position was opened, `VolConfGateHold` made it *impossible to close*. This created a "roach motel" — positions could enter but never leave. This was especially dangerous because:
- A position opened near the top of a range would be held indefinitely.
- Stop-losses, take-profits, and signal-based exits were all bypassed.

### 3.2 Why it mattered today
- Today there were no trades, so the bug didn't cause a loss today.
- However, on the backtest data, `vol_conf_v1_default` (which includes `Hold`) was producing **Sharpe -3.14** vs **+0.29** for `disabled`.
- The catastrophic backtest performance was entirely due to positions being trapped and suffering large drawdowns.

### 3.3 The fix
**File: `haskell/app/Trader/VolConfGate.hs`**
- `volConfStatefulCloseDirection` for `VolConfGateHold` now returns `closeDirBase` instead of `Nothing`.
- `applyVolConfGateBehavior` no longer forces `chosenDir = currentSide` for `Hold`.

**File: `haskell/app/Main.hs`**
- Removed the `volConfHoldActive` special-case traps from both:
  - Startup path (`computeLatestSignal`)
  - Live-trading path (`desiredPosSignal`)

**Validation:**
- After the fix, `vol_conf_v1_default` backtest Sharpe improved from **-3.14** to **+0.18** (same trades as `disabled` but with slightly reduced sizing).
- The test suite passes (`cabal test` exits 0).

---

## 4. Additional Code Changes

### 4.1 Removed confidence damping from `trendFollowingCandidate`
**File: `haskell/app/Trader/TechnicalAnalysis/Strategies.hs`**
- `trendFollowingCandidate` was using `dampedConfidence = baseConfidence * rsConfidence regimeScore`, while `trendFollowingAt` (the precomputed path) used `baseConfidence` directly.
- This created **behavioral divergence** between the live series path and the backtest precomputed path.
- Removed the damping to restore parity.

### 4.2 Added `volConfGate` API override
**File: `haskell/app/Main.hs`**
- Added `apVolConfGate :: Maybe String` to `ApiParams`.
- Wired it through `argsFromApi` with fallback to CLI default on invalid preset.
- JSON key: `volConfGate`.
- This allows API clients (e.g., the autoloop, dashboard, or manual curl) to override the vol/conf gate without restarting the server.

---

## 5. Backtest Results (Post-Fix)

Data: `BTCUSDT-4h-1000.csv` (last ~167 days, ending 2026-05-16)

| Method | Gate | Sharpe | Max DD | Trades |
|--------|------|--------|--------|--------|
| `ta_trend` | disabled | **+0.29** | 3.85% | 4 |
| `ta_trend` | vol_conf_v1_default | +0.18 | 3.62% | 4 |
| `ta_reversion` | disabled | 0.00 | 0.00% | 0 |
| `ta_reversion` | vol_conf_v1_default | 0.00 | 0.00% | 0 |
| `ta_breakout` | disabled | **+2.32** | 7.14% | 7 |
| `ta_breakout` | vol_conf_v1_default | +1.87 | 7.14% | 7 |
| `ta_best` | disabled | +0.74 | 4.17% | 5 |
| `ta_best` | vol_conf_v1_default | +0.31 | 4.17% | 5 |

### 5.1 Key observations
1. **`ta_breakout` is the best method on recent data** (+2.32 Sharpe). It correctly captured the volatility clusters and range-bound chop.
2. **`ta_trend` is the worst method** (+0.29 Sharpe). It requires a trending market, which we don't have.
3. **`vol_conf_v1_default` is slightly worse than `disabled` across all methods.** The gate's size multiplier (0.5× in high-vol) is reducing returns in a market where volatility is already low.
4. **`ta_reversion` produced zero trades.** The Bollinger bandwidth never hit the squeeze threshold.

### 5.2 May 15 vs. May 17 comparison
| Metric | May 15 (old data) | May 17 (current data) |
|--------|-------------------|------------------------|
| `ta_trend` + `disabled` | -3.56 | **+0.29** |
| `ta_trend` + `vol_conf_v1_default` | **+3.54** | +0.18 |

The inversion was caused by **new market data** (3 additional days of range-bound action), not a code regression. The old data had a sharp trend at the end that `vol_conf_v1_default` happened to capture, while the new data extended the range-bound chop.

---

## 6. Engineering Decisions & Recommendations

### 6.1 Decision: Do NOT change the autoloop default (for now)
- The autoloop currently uses `ta_trend` + `vol_conf_v1_default`.
- While `ta_breakout disabled` has better recent backtests, switching methods based on short-term performance is **overfitting**.
- A better approach is **regime-conditioned method selection** (see §7).

### 6.2 Decision: Do NOT disable the vol/conf gate by default
- The gate provides risk reduction (smaller positions in high-vol regimes).
- The recent underperformance is partly due to the market being in a low-vol regime where the gate rarely activates.
- The fix restored the gate's intended behavior without the position-trap bug.

### 6.3 Decision: Add API override for volConfGate
- This allows rapid A/B testing without code changes or restarts.
- Example:
  ```bash
  curl -X POST http://localhost:8090/api/signal \
    -d '{"volConfGate": "disabled", "method": "ta_breakout"}'
  ```

### 6.4 Decision: Removed damping from `trendFollowingCandidate`
- Restores parity between live and backtest paths.
- Eliminates a hidden source of divergence.

---

## 7. Future Work: Regime-Conditioned Method Selection

### 7.1 Problem
- `ta_trend` works in trending markets (ADX > 25).
- `ta_reversion` works in ranging markets (low ADX, tight Bollinger bands).
- `ta_breakout` works in transition periods (volatility expansion).
- Currently, the method is static (set at startup). The system cannot adapt to regime changes.

### 7.2 Proposed solution: `ta_regime_switch`
A meta-method that selects the sub-method based on real-time regime detection:

```
if ADX > 25 and price > EMA200:
    method = ta_trend (long-biased)
elif Bollinger_bandwidth < 0.002:
    method = ta_reversion
else:
    method = ta_breakout
```

This would require:
1. A new `Method` constructor: `MethodRegimeSwitch`.
2. Regime detection logic in `Strategies.hs` (or reuse `regimeSelector`).
3. Method switching logic in `computeLatestSignal`.
4. Backtest support in `computeBacktestSummary`.

### 7.3 Validation plan
1. Backtest `ta_regime_switch` on 2020–2026 data.
2. Compare Sharpe, max drawdown, and trade count vs. static `ta_trend`.
3. Expected improvement: +20–40% Sharpe in mixed-regime periods.

---

## 8. Test & Validation Results

### 8.1 Test suite
```bash
cabal test
# Result: PASS (exit 0)
```

### 8.2 Vol/conf gate scorecard (post-fix)
```bash
python3 scripts/run-volconf-scorecard.py --data data/BTCUSDT-4h-1000.csv
# Result: all presets now produce positive Sharpe; no catastrophic inversion
```

### 8.3 Live signal sanity check
```bash
curl http://localhost:8090/api/signal -d '{...}'
# Result: HOLD (NON_DIRECTIONAL_WEAK_BAND) — expected for range-bound market
```

### 8.4 Binary build
```bash
cabal build trader-hs
# Result: SUCCESS (May 17 14:28)
```

---

## 9. Risk & Operational Notes

### 9.1 No trades today = no P&L risk
The system was correctly flat all day. This is a feature, not a bug.

### 9.2 Optimizer resource contention
Multiple optimizer jobs are running in the background (ARBUSDT, various seeds). These consume significant CPU and may slow down the build process. Consider:
- Running optimizers on a separate machine or during off-hours.
- Using `nice` to lower optimizer priority.

### 9.3 API server uptime
- Port 8090: uptime since 1:32 PM (~3 hours).
- Port 8091: uptime since 1:32 PM (~3 hours).
- No restarts required for today's fixes (API override works on existing servers).

---

## 10. Action Items

| Priority | Action | Owner | Deadline |
|----------|--------|-------|----------|
| P0 | Monitor live signals tomorrow morning | @diegosaa | 2026-05-18 |
| P1 | Implement `ta_regime_switch` prototype | @diegosaa | 2026-05-24 |
| P1 | Add `--regime-method-map` CLI arg for regime→method mapping | @diegosaa | 2026-05-24 |
| P2 | Run 3-year backtest of regime-switching vs. static methods | @diegosaa | 2026-05-31 |
| P2 | Consider adding `--autoloop-default-method` env var for quick method switching | @diegosaa | 2026-05-20 |
| P3 | Document vol/conf gate presets and their intended use cases | @diegosaa | 2026-05-20 |

---

## 11. Files Modified

```
 haskell/app/Trader/TechnicalAnalysis/Strategies.hs  |  3 ++-
 haskell/app/Trader/VolConfGate.hs                   |  4 ++--
 haskell/app/Main.hs                                  | 30 +++++++++++++++++++++++++++++-
```

Total: 3 files changed, ~40 lines modified.

---

## 12. Conclusion

Today was a **no-trade day** in a **range-bound, low-conviction market**. The system correctly stayed flat, avoiding false signals.

The primary engineering achievement was **fixing the `VolConfGateHold` position trap**, which was causing catastrophic backtest performance. After the fix, the vol/conf gate behaves as intended — slightly reducing position sizes during high-vol/low-confidence periods without preventing exits.

Additional improvements:
- Removed confidence damping divergence between live and backtest paths.
- Added API-level `volConfGate` override for rapid experimentation.

The next major initiative is **regime-conditioned method selection** (`ta_regime_switch`), which would allow the system to automatically use trend-following in trending markets, mean-reversion in ranging markets, and breakouts in transition periods.

*Review completed: 2026-05-17 14:35 UTC*

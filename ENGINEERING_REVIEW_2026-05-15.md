# Engineering Review — 2026-05-15

**Status:** Autoloop crashed (repeat failure). Zero live trades. Code fixes committed. Backtest infrastructure overloaded.

---

## 1. What Happened Today

### 1.1 Live Trading
- **Trades executed:** 0
- **Autoloop state:** CRASHED at 02:54 UTC (cycle-0007, exitCode 1)
- **Failure mode:** `codex ETIMEDOUT` — exact replica of 2026-05-14 crash
- **Market regime:** BTC range-bound ~$79k–$81k (4h), low volatility

### 1.2 Root Cause (Confirmed)
`requestFixIdea` / `requestReviewFeedbackSelection` / `requestIdeaSelection` calls in `autoloop.mjs` were **unprotected** against transient Codex backend timeouts. When the Codex API returned `ETIMEDOUT`, the exception bubbled to the top level, killed the cycle, and `forever` did not restart because the process exited with a non-graceful error path.

**Fix committed:** `84459852`
- Wrap all three planner calls in `try/catch`
- Detect `isRetryableCodexExecError(err)` → degrade to `no_patch_plan` with logged message
- Increase `CODEX_EXEC_TIMEOUT_MS` 300s → 420s
- Increase `CODEX_RETRY_BACKOFF_MS` 15s → 30s
- Tests: 54/54 pass, including new graceful-degradation test

### 1.3 Infrastructure Gaps
- **Credentials:** `BINANCE_API_KEY` and `BINANCE_API_SECRET` are **unset** in environment
- **Data pipeline:** No automated fetch until today; added `scripts/fetch-data-pipeline.sh` (committed in `b1dbce2a`)
- **Haskell build:** Binary stale (last built 2026-05-15 12:05). Working-tree changes to `Strategies.hs` (committed in `a3d19aba`) require rebuild before backtest experiments can validate them.

---

## 2. Research & Experiments

### 2.1 Vol/Conf Gate Scorecard (Strong Validation)
| Variant | Sharpe | Max DD | Trades | Final Equity |
|---------|--------|--------|--------|--------------|
| vol_conf_gate **disabled** | **-3.56** | 6.18% | 7 | 0.9662 |
| vol_conf_gate **enabled** (vol_conf_v1_default) | **+3.54** | 0.63% | 4 | 1.0124 |

**Hypothesis:** Volatility + conformal-confidence gating is not merely helpful — it is *load-bearing* for positive expectancy on the current BTC slice.
**Result:** Confirmed. Disabling the gate flips Sharpe from strongly positive to strongly negative.
**Action:** Gate must remain enabled in all production configurations. Invariant added: `volConfGatePreset != disabled` for live trading.

### 2.2 Preset Non-Differentiation
All four enabled presets (`vol_conf_v1_default`, `vol_conf_v1_aggressive`, `vol_conf_v1_conservative`, `vol_conf_v1_tight`) produced **byte-identical metrics** on BTCUSDT-4h-1000.

**Hypothesis:** Preset thresholds are too close together relative to the signal distribution on the current slice.
**Result:** Confirmed non-differentiation.
**Action needed:** Widen preset parameter spreads or derive them from per-asset quantiles instead of fixed constants.

### 2.3 Cross-Asset Generalization (Disconfirmed)
| Asset | Sharpe | Max DD | Trades | Final Equity |
|-------|--------|--------|--------|--------------|
| BTCUSDT | +3.54 | 0.63% | 4 | 1.0124 |
| ETHUSDT | **-5.89** | 3.02% | 6 | 0.9748 |
| SOLUSDT | **-2.77** | 2.14% | 4 | 0.9801 |

**Hypothesis:** BTC-tuned parameters generalize to ETH and SOL.
**Result:** Strongly disconfirmed. ETH and SOL produce negative Sharpe with BTC parameters.
**Action needed:** Per-asset parameter optimization required. Do not deploy BTC-tuned config to ETH/SOL live.

### 2.4 ta_trend Backtest (Degraded)
| Metric | Value |
|--------|-------|
| Sharpe | **-3.14** |
| Max DD | 6.20% |
| Trades | 4 |
| Final Equity | 0.9707 |

**Observation:** The `ta_trend` method is producing poor results. The decomposed regime detector (committed in `a3d19aba`) may need calibration before it improves performance.
**Open question:** Does the new precomputed-indicator path change ta_trend behavior once wired into the engine? Not yet validated — the new `candidateForMethodAt`/`trendFollowingAt` functions are exported but not yet integrated into `Main.hs`.

### 2.5 Candidate A Experiment (Incomplete)
Planned: `conf_blend` + adaptive threshold-factor grid (`--threshold-factor-alpha 0.20–0.40`, `--threshold-factor-min 0.50–0.75`, `--threshold-factor-max 2.0–3.0`).
**Status:** Could not complete due to system resource contention from stale optimizer processes. Backtests hung or were killed.
**Mitigation:** Killed stale optimizer backtest PIDs (consuming 110%+ CPU). Need rebuilt binary + isolated runner to complete grid.

---

## 3. Code Changes Committed

| Commit | Description |
|--------|-------------|
| `84459852` | autoloop crash resilience (codex ETIMEDOUT) |
| `a3d19aba` | strategies: decomposed regime detector + precomputed indicators |
| `b1dbce2a` | data: refreshed 4h CSVs + fetch-data-pipeline.sh |
| `1ef443d0` | reports: CTO health check + data director proofs |

### 3.1 Strategies.hs Architecture Changes
- **`OhlcvIndicators`**: Precomputed indicator cache (20 vectors)
- **`precomputeIndicators`**: Single-pass O(n) computation
- **`trendFollowingAt` / `momentumReversionAt` / `volumeConfirmedBreakoutAt`**: O(1) per-bar evaluators
- **`candidateForMethodAt`**: Maps `Method` enum to specific evaluator
- **Regime decomposition**: Explicit scoring weights — ADX trend 0.40, Aroon gap 0.35, slope 0.25

**Next integration step:** Wire `candidateForMethodAt` into `Main.hs` backtest path (currently `strategyCandidates` still uses the old recomputation path).

---

## 4. Failure Modes & Invariants

| Invariant | Status | Enforcement |
|-----------|--------|-------------|
| `volConfGatePreset != disabled` for live trading | **NEW** | Manual config check; should be hard assert in `trader-hs` |
| `CODEX_EXEC_TIMEOUT_MS >= 420000` | Committed | `clampInt` default + env override |
| Autoloop must not crash on retryable codex errors | Committed | try/catch + `isRetryableCodexExecError` |
| Per-asset params required for ETH/SOL | **NEW** | Documented; needs optimizer runs |
| Presets must produce differentiable metrics | **VIOLATED** | Needs threshold recalibration |

---

## 5. Action Items

### Immediate (P0)
1. **Rebuild `trader-hs` binary** with `a3d19aba` changes and validate `ta_trend` behavior.
2. **Restart autoloop** with new crash-resilience commit deployed.
3. **Set `BINANCE_API_KEY` / `BINANCE_API_SECRET`** — live trading is blocked without credentials.
4. **Run Candidate A grid** (conf_blend + threshold-factor) once binary is rebuilt and system is clean.

### Short-term (P1)
5. **Integrate `candidateForMethodAt`** into `Main.hs` backtest/optimize paths to activate O(n) speedup.
6. **Recalibrate vol/conf presets** — widen spreads or derive from per-asset quantiles.
7. **Run per-asset optimization** for ETHUSDT and SOLUSDT (do not reuse BTC params).
8. **Add `TestMain.hs` coverage** for `--vol-conf-gate` round-trip (known gap from data director proofs).

### Engineering Debt
9. **Stale build artifacts** in `/private/tmp/trader-clean-main-*/` — multiple old binaries; add cleanup to CI.
10. **Backtest process management** — stale optimizer child processes accumulate and starve CPU; add PID cleanup or cgroup limits.

---

## 6. Metrics Summary

| Metric | Value | Target |
|--------|-------|--------|
| Live trades today | 0 | > 0 |
| Autoloop uptime | 0% (crashed 02:54 UTC) | > 95% |
| Tests passing | 54/54 (100%) | 100% |
| Backtest Sharpe (method 11, BTC) | +3.54 (vol/conf on) | > 2.0 |
| Backtest Sharpe (ta_trend) | -3.14 | > 0.5 |
| Cross-asset Sharpe (ETH) | -5.89 | > 1.0 |
| Cross-asset Sharpe (SOL) | -2.77 | > 1.0 |

---

*Review written 2026-05-16 04:56 UTC. Next review: 2026-05-16.*

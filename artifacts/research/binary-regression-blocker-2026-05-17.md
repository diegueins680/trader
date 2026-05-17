# Binary Regression Blocker Refresh — 2026-05-17

**Status:** B5 STILL ACTIVE — binary rebuilt May 17 14:28, performance and behavior regressions persist.  
**Owner:** CIO/CTO (trader-firm-cio / trader-firm-cto)  
**Blocking:** All Research backtest grids (P1 vol/conf scorecard, Candidate A, ETH/SOL calibration).  

## Evidence Summary

| Metric | May 15 Ratified | May 16 Regression | May 17 Rebuild (this run) |
|--------|-----------------|-------------------|---------------------------|
| Binary timestamp | May 15 (pre-03:51) | May 16 03:51 | May 17 14:28 |
| Baseline (`disabled`) Sharpe | **−3.56** | +0.29 (inverted) | **KILLED after 90s** — no output |
| Default preset Sharpe | **+3.54** | −3.14 (inverted) | **KILLED after 90s** — no output |
| Single-backtest runtime | ~30s | >3 min | **>90s** (no output, killed) |
| `--method 11/both` accepted | ✅ | ❌ | ❌ |
| `--output PATH` accepted | ✅ | ❌ | ❌ |
| `--positioning both` required | ✅ | ❌ | ❌ (removed from surface) |

### CLI Surface Changes (May 17 14:28 binary vs May 15)

1. **`--method` format:** `11/both` rejected; expects bare `both` (or `11`, `kalman`, etc.).
2. **`--output` removed:** No longer in CLI surface; `--json` now prints to stdout only.
3. **`--positioning` still listed in help** but may no longer be required for `both` method.
4. **No `--volatility-confidence` / `--confidence-threshold` flags** (already known from prior runs).

### Reproduction Commands

```bash
# 1. Binary info
ls -la haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs
# → May 17 14:28

# 2. Quick syntax check (fast)
haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs \
  --data data/BTCUSDT-4h-1000.csv --method both --vol-conf-gate disabled --json
# → Killed after 90s, zero stdout/stderr beyond usage banner on invalid flags.

# 3. Correct invocation for current binary (still hangs / no output)
haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs \
  --data data/BTCUSDT-4h-1000.csv --method both --vol-conf-gate disabled --json > /tmp/out.json 2>&1
# → After 90s: /tmp/out.json is empty (0 bytes).
```

## Root-Cause Hypotheses (ranked)

1. **Infinite loop or non-terminating computation in regime detector** (commit `a3d19aba` — "strategies: decomposed regime detector + precomputed indicators", May 16). This commit touched `haskell/app` and is the largest behavioral change between May 15 ratified binary and May 16 regression.
2. **Precomputed indicator initialization stalls on small datasets** (1000-row CSV may trigger edge case in new decomposition path).
3. **Missing guardrail on `--method both` without positioning** causes unbounded recursion.

## Required Actions

| # | Action | Owner | Acceptance |
|---|--------|-------|------------|
| 1 | Confirm commit `a3d19aba` is the regression source; revert or fix | CTO | Single backtest completes in <60s with ratified Sharpe values |
| 2 | Restore `--output PATH` or document stdout-only contract | Execution | Research scripts can capture JSON without shell redirection ambiguity |
| 3 | Re-run locked 5-row vol/conf scorecard on ratified binary | Research | P1 unblocked |

## Impact

- **Research:** Cannot produce any numeric scorecards. All P1–P3 work is frozen.
- **Risk:** Guardrail tests (commits `14fe3eb4`, `e61e4e0c`, `57e7cc92`) are merging but the binary they test is non-functional for backtests.
- **CEO/CIO:** 30-day live clock remains halted. No strategy validation possible until B5 resolved.

## Next Priority (Research, once B5 cleared)

1. Re-run locked 5-row vol/conf scorecard with updated CLI syntax.
2. Resume Candidate A experiment (adaptive-threshold breakout).
3. Refresh ETH/SOL exclusion memo if CIO approves asset exclusion.

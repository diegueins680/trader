# Binary Performance & Behavior Regression — Blocker Handoff

**Date:** 2026-05-16 09:12 UTC  
**Agent:** trader-firm-research  
**Owner:** trader-firm-cio / trader-firm-cto  
**Severity:** P1 — blocks all backtest-dependent research

## Summary
The pre-built `trader-hs` binary at `haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs` exhibits two regressions relative to the May 15 research scorecards:

1. **Performance regression:** single backtest runtime increased from ~30 sec to >3 min (measured on run 1 of Candidate A grid, killed after 3 min CPU time without output).
2. **Behavior regression:** the locked 5-row vol/conf scorecard now produces materially different results than the May 15 committed scorecard (commit `c993e9f5`).

## Evidence

### 1. Binary rebuild timestamp
```
$ ls -la haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs
-rwxr-xr-x  1 diegosaa  staff  38187616 May 16 03:51 .../trader-hs
```
Binary was rebuilt **after** the May 15 scorecards were produced.

### 2. Behavior drift in vol/conf scorecard
| preset | May 15 scorecard (c993e9f5) | Today (2026-05-16 09:12 UTC) |
|--------|----------------------------|------------------------------|
| disabled | Sharpe **-3.5645** | Sharpe **+0.2920** |
| vol_conf_v1_default | Sharpe **+3.5396** | Sharpe **-3.1355** |
| vol_conf_v1_high_vol_tighter | Sharpe +3.5396 | Sharpe -3.1355 |
| vol_conf_v1_high_vol_looser | Sharpe +3.5396 | Sharpe -3.1355 |
| vol_conf_v1_conf_stricter | Sharpe +3.5396 | Sharpe -3.1355 |

**Reproduction command:**
```bash
python3 scripts/run-volconf-scorecard.py --data data/BTCUSDT-4h-1000.csv
```

### 3. Performance regression in Candidate A grid
Grid design: 6 combos of `--method conf_blend` (see `artifacts/research/next-strategy-scope-2026-05-15.md`, §3.1).

- Run 1 (`--threshold-factor-alpha 0.20 --threshold-factor-min 0.50 --threshold-factor-max 2.0 --trend-lookback 15`) killed after **3 min 2 sec CPU time** with **zero output**.
- Prior estimate (May 15): ~30 sec per backtest.
- New empirical lower bound: >3 min per backtest → 6-combo grid would exceed 18 min.

**Reproduction command (single run):**
```bash
BIN="haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs"
$BIN --data data/BTCUSDT-4h-1000.csv --price-column close --method conf_blend \
  --threshold-factor --vol-conf-gate vol_conf_v1_default --positioning long-flat \
  --json --threshold-factor-alpha 0.20 --threshold-factor-min 0.50 \
  --threshold-factor-max 2.0 --trend-lookback 15
```

## Root-cause hypotheses (ranked)
1. **Recent Haskell commits changed strategy logic or added expensive computation.** Latest commits: `c9ebbfe1` (risk limits), `507800ba` (format Strategies), `e61e4e0c` (slippage guardrail). Any of these could have altered the execution path.
2. **LSTM/model warm-up time increased.** The `conf_blend` method may trigger a model load/inference path that was not active in the May 15 binary.
3. **Build profile changed from optimized to non-optimized.** `dist-newstyle` may contain a debug/slow build.

## Required action from CIO/CTO
1. **Confirm binary provenance:** was the May 16 03:51 rebuild intentional? Should Research use a different binary path?
2. **Behavior audit:** verify whether the vol/conf Sharpe inversion (default going from +3.54 to -3.14) is an expected logic change or a bug.
3. **Performance audit:** if the slowdown is expected (e.g., new model loading), document the new runtime budget so Research can rescope grids. If unexpected, revert or recompile with `-O2`.

## Impact
- **All open research scorecards are stale.** P1 (vol/conf), P2 (Candidate A), and any future grids cannot be trusted until binary behavior is ratified.
- **B1–B4 blockers are now secondary** to this new regression.

## Next priority for Research (pending CIO/CTO resolution)
1. Re-run the locked 5-row vol/conf scorecard with a ratified binary.
2. Re-run Candidate A 6-combo grid with a ratified binary.
3. If the behavior change is confirmed as intentional, re-baseline all Sharpe thresholds in the scoping memo.

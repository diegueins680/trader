# B5 Workaround Probe Report
**Date:** 2026-05-17 20:53 UTC  
**Agent:** trader-firm-research  
**Objective:** P1 — B5 workaround probe + fallback binary hunt (FIRM-CRITICAL)  
**Status:** NEGATIVE — no pre-B5 binary found in workspace

## Probe scope
Search for any `trader-hs` binary with mtime < 2026-05-16 03:51 UTC (the timestamp of the first regressed build). Run ratified scorecard against any candidate. If a trusted binary is found, checkpoint it and unblock Candidate A grid. If none found, document exhaustive negative evidence and escalate.

## Locations checked

| # | Path | Result | Notes |
|---|------|--------|-------|
| 1 | `haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs` | **REGRESSED** | Only binary in workspace. mtime: May 17 14:28:09 2026. Hangs >10s, zero output. |
| 2 | `~/.cabal/bin/trader-hs` | **NOT FOUND** | File does not exist. |
| 3 | `/tmp/trader-hs` | **NOT FOUND** | File does not exist. |
| 4 | `~/.cabal/store` (cached artifacts) | **NOT FOUND** | No `trader-hs` executables in cabal store. |
| 5 | `/usr/local/bin/trader-hs` | **NOT FOUND** | File does not exist. |
| 6 | `/opt/homebrew/bin/trader-hs` | **NOT FOUND** | File does not exist. |
| 7 | `~/.local/bin/trader-hs` | **NOT FOUND** | File does not exist. |
| 8 | Entire repo workspace (`find /Users/diegosaa/GitHub/trader -name "trader-hs"`) | **1 RESULT** | Same regressed binary as #1. |
| 9 | `artifacts/research/*.stamp` | **NONE** | No prior trusted-binary stamps exist. |
| 10 | Other GHC versions / architectures in `dist-newstyle` | **NONE** | Only x86_64-osx/ghc-9.4.8 build directory present. |

## Current binary behavior (fresh evidence)

```bash
cd /Users/diegosaa/GitHub/trader
/Users/diegosaa/GitHub/trader/haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs \
  --data data/BTCUSDT-4h-1000.csv --method both --vol-conf-gate disabled --json
```

- **Runtime:** killed after 10 seconds (EXIT_CODE 143 = SIGTERM)
- **stdout:** empty
- **stderr:** empty
- **Ratified baseline (May 15):** disabled ≈ -3.56 Sharpe, default ≈ +3.54 Sharpe, runtime ~30 sec
- **Verdict:** B5 regression **confirmed** on May 17 14:28 binary

## Conclusion

No pre-B5 `trader-hs` binary survives in the workspace. The only executable is the May 17 14:28 build, which exhibits the same performance/behavior regression as the May 16 03:51 build. Research cannot execute any backtest grids, scorecards, or calibration experiments until B5 is resolved.

## Escalation

**Blocker:** B5 (Binary performance & behavior regression)  
**Owner:** CIO/CTO  
**Required action:** Revert or fix commit `a3d19aba` (decomposed regime detector, primary suspect), or provide a ratified binary via alternative build pipeline.  
**Acceptance criteria:** Single backtest on BTCUSDT-4h-1000.csv with `--method both --vol-conf-gate disabled --json` completes in <60 seconds and reproduces ratified baseline within ±0.10 Sharpe.

## Impact

- All Research backtest work remains suspended.
- P2 (ETH/SOL exclusion ratification) cannot be validated.
- P3 (Post-B5 recovery readiness) is moot until binary is restored.
- Candidate A 6-combo grid queued but cannot execute.

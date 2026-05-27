# Method 10 Binary Hang — Blocker Report
**Date:** 2026-05-27 12:35 UTC  
**Reporter:** trader-firm-research  
**Severity:** BLOCKER — prevents P5.1 parameter sweep and all large-dataset Method 10 validation.

## Problem
The prebuilt `trader-hs` binary hangs indefinitely when running `--method 10` on datasets with > ~1000 bars.

## Reproduction
```bash
cd /Users/diegosaa/GitHub/trader/haskell
./dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs \
  --data ../data/BNBUSDT-5m-2020-06_full.csv --price-column close \
  --method 10 --threshold 0.02 --close-threshold 0.005 --lookback 50 \
  --vol-target 0.10 --vol-lookback 20 --max-hold-bars 48 --trailing-stop 0.05
```

**Dataset:** BNBUSDT-5m-2020-06_full.csv (8,598 bars)  
**Binary:** `dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs`  
**Observed behavior:** Process starts, consumes CPU, produces zero output for >2 minutes. Must be killed.  
**Control test:** Same binary with `--method 01` on same dataset completes in <10 seconds.

## Impact
- P5.1 (z_entry × z_exit parameter sweep) is **blocked**.
- P5.2 (cross-asset robustness) is **blocked**.
- All Method 10 large-dataset research is **blocked**.
- The prior A3 proxy result (Sharpe 0.29, 177 trades) was obtained via Python, not the binary, and cannot be replicated or extended.

## Hypotheses
1. **Kalman filter loop:** The Kalman update loop may not terminate or may have O(n²) behavior that explodes on 8k bars.
2. **Vol-targeting sizing loop:** Position sizing with vol-targeting may iterate per-bar in a way that does not scale.
3. **Memory allocation:** Large dataset may trigger a pathological memory pattern (e.g., repeated vector reallocation).

## Request to CTO/Execution
1. Run the reproduction command with `+RTS -p -hc` and provide the `.prof` and `.hp` files.
2. Or, provide a debug build with `-O0` and stack traces enabled.
3. Or, confirm if Method 10 is expected to be used only on small datasets (<1000 bars).

## Fallback
If binary fix ETA > 48 hours, Research will request Execution to run the parameter sweep via the Python A3 proxy script (already validated on 8,598 bars).

## Next Steps
- **CTO:** Diagnose and fix hang, or provide guidance on dataset size limits.
- **Research:** Stand by for fix; if no fix in 48h, switch to Python fallback.
- **CIO:** Decide whether to allocate Execution resources to Python fallback or wait for binary fix.

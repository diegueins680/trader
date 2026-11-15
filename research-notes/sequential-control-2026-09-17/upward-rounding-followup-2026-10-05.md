# Upward rounding and numeric maker rejection — 2026-10-05

Baseline 906ad455bef2b6706bd44667679596aa75766aae. Registration commit 6c5889eb
preceded implementation. No financial trial, data acquisition, holdout evaluation,
order, production configuration change or deployment.

CE-ROUND-003: nextUp(1) = 1.0000000000000002 used to produce 1 on a unit grid.
Both local Main helpers now delegate to exact rational ceiling and a checked
binary64 publication boundary. CE-ROUND-004: invalid maker price previously
called the configured market fallback. The invalid branch now returns an unsent
No order result. Other fallback reasons are unchanged.

The independent Fraction oracle covers 130 boundary and 4096 generated binary64
cases (seed 20261005), including non-finite, subnormal, extreme and malformed-grid
inputs. Both actual Main delegates are compiled. The actual invalid-price branch
is compiled against effect-recording stubs, alongside its preserved legacy branch;
no exchange call occurs. Haskell properties exercise another 1000 generated words
and explicit edge values across eight grids. These are conformance tests, not proofs.

Four SAT-premise/UNSAT-violation checks cover exact ceiling/reconstruction and
binary64 admission. The finite model covers one price dispatch, two validity
classes and both fallback flags: 8 states, 4 initial, 4 terminal, 4 edges, depth 1.
Terminality is local; no server liveness or complete order authorization theorem.
Pinned runtime numeric primitives and source-to-model mapping remain assumptions.

The minTradeQty zero fallback remains Nothing; entry minimum retry preserves its
original error. isLongSpot still treats any positive balance as a position when
a minimum cannot be established. This conservative inventory classification is
not a proof of downstream exit behavior. Other unfiltered metadata, decimal wire
rounding, exchange grid acceptance and complete order-cap composition remain open.

Broader status: 5 scoped closures, 27 partial, 6 open. Obligation 9 remains partial.
The frozen financial screen and its failed acceptance/OPE gates are unchanged;
final returns remain sealed. No candidate adoption or RL promotion is justified.

Local downward tests passed (2 tests, 31.697 seconds); initial local upward
tests passed (2 tests, 12.485 seconds). Final pinned evidence is recorded below.

Local full receipt reproduction (`python scripts/formal/verify.py --record`)
failed in the unchanged PPO bridge probe: `--snapshot-contract-v3` exceeded
its existing three-second subprocess timeout. No receipt was written and no
timeout was widened. Reproduce frozen sources on the pinned CI runner; local
targeted rounding passes do not imply the complete formal wrapper passed.

The final targeted upward suite passed after binding the actual unsent base
result (2 tests, 12.718 seconds). Local GHC 9.4.8 `-O2` numeric microbenchmark:
100000 calls at scale=100000000, increment=1, x=i/1000003 for i=1..100000;
strict sum checksum 5000.035499900004; process CPU time 0.189296 seconds
(1.89296 microseconds/call). This is a single illustrative pure-helper CPU
measurement, not wall-clock latency, a hard performance bound, production
inference throughput or exchange execution evidence. IO, network and final
wire formatting are excluded. Huge adversarial Integers have no latency bound.

Reproduce the benchmark with GHC 9.4.8 `-O2 -ihaskell/app` and this base-only
main (keep build output outside Git):

```haskell
module Main (main) where
import Control.Exception (evaluate)
import Data.List (foldl')
import System.CPUTime (getCPUTime)
import Trader.QuantityRounding (quantizeUpExact)
main :: IO ()
main = do
  start <- getCPUTime
  total <- evaluate (foldl' (\acc i -> acc + quantizeUpExact 100000000 1 (fromIntegral i / 1000003)) 0 ([1..100000] :: [Int]))
  stop <- getCPUTime
  print (total, fromIntegral (stop-start) / 1e12 :: Double)
```


## Frozen-source verification

Pinned reproduction run [37310098005](https://github.com/diegueins680/trader/actions/runs/37310098005),
job 111763107081, checked exact source commit
`67bfe0a9e5365eddabdac5a208fd08c27b2356e6`:

- `python3 scripts/formal/verify.py --record`: PASS, 12:33:48–12:34:49 UTC.
- `bash scripts/verify.sh formal`: PASS, 12:34:49–12:36:36 UTC;
  196 integrity tests, all 60 SMT groups, existing state models and compiled
  conformance, plus the new four queries and eight-state dispatch model.
- `bash scripts/verify.sh full`: PASS, 12:36:36–12:44:24 UTC;
  formal repeated successfully (196 tests, 45.873 seconds), Haskell format/lint/
  build/smoke/test, 241 web tests and 185 automation tests.

The imported raw receipt has SHA256
`e792cac797bf1fc20fffe3bf92adc566ba3138780539a40e6d5c236b574ad072`.
It was decoded unchanged from the runner log; every receipt source hash matches
the reviewed lock and local source bytes. The changed sections are capability
source bindings, source hashes, the two added SMT groups and upward-rounding
results. Existing proof conclusions and broad acceptance gates are unchanged.
The final evidence commit only imports this receipt, updates this report, and
removes the temporary read-only reproduction workflow. No proof or implementation
source changes after the passing full run.

The initial ordinary CI formal check failed only because it compared the new
reproduction against the previously committed receipt (all 196 integrity tests
passed first). The final PR checks must pass with the imported receipt before
merge. Merge must not deploy; use the exact tested head and verify merge-tree
identity and GitHub deployment records. Mission completion remains false;
economic evidence and reproducible delivery gates remain blocked.

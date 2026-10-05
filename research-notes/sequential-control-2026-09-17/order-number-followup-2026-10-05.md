# Binance finite order-number admission — 2026-10-05

Baseline a772ca0ffcde120134bda927e38387258bd94435. Registration commit 1964ad46
preceded implementation. No market-data trial, final-holdout access, exchange
endpoint invocation, order, live setting change or deployment.

CE-ORDER-NUM-001: a comparison rejecting only x<=0 allows NaN and positive
infinity. Four futures constructors had this guard. CE-ORDER-NUM-002: the general
market constructor reached credential access with an invalid selected number.
The new base-only OrderNumeric module provides pure finite-positive and market
selection validators. All five constructors validate before the first credential
read. The public signatures and valid request-building tails remain unchanged.
Base quantity retains priority; a bad present base never falls back to quote.
Unused quote values retain legacy semantics. Missing-amount error messages are
preserved, but now take precedence over missing-credential errors.

The source-bound SMT scope is five SAT-premise/UNSAT-violation pairs over all
binary64 values and Boolean presence/market flags. The local transition model
has 40 states, 20 initial, 20 terminal, 20 transitions and depth one, covering
five constructors and numeric/context validity. The credential boundary is
terminal in the model; it is not an order capability or authorization proof.

Compiled actual current and preserved legacy prefixes cover 5344 rows: 1248
boundary combinations from 13 bit patterns and 4096 seeded generated rows
(seed 20261005). Each row traverses all five constructors, yielding 26720 current
and 26720 legacy prefix checks. Spot/margin/futures, both modes, optional-value
presence, quantity/quote priority and empty/nonempty order type are represented.
The driver's terminal marker replaces the first credential read; no Binance
module, credential, request, signing or HTTP code is linked. The source-derived
local trim helpers are compiled too. An independent Python oracle and Haskell
properties over 1000 generated words supplement the proofs.

Initial targeted run: solver/model/compiled conformance passed; the separate
source-mutation test failed because its harness omitted a JSON import. The import
was corrected without changing the numeric implementation or solver obligations.
Final targeted suite: 2 tests passed in 26.848 seconds.

Limitations: pinned GHC/base and IO primitive semantics, immutable non-bottom
inputs, source extraction and reviewed abstraction remain assumptions. This is
not a compiler proof, full IO refinement, or proof of caller retry behavior.
Tiny positive values can still become zero in existing eight-decimal formatting.
Venue filters, exact tick/lot representation, final wire caps, upper exposure
bounds, fill behavior and other exchanges remain unresolved. No future-profit
or financial acceptance claim follows from numeric admission.

Status remains 5 scoped closures, 27 partial, 6 open. Obligation 21 gains narrow
boundary evidence but remains partial. The frozen financial trials, all seeds,
costs, OPE failures, contamination status and sealed holdout remain unchanged.
No candidate is adopted. Frozen-source verification is recorded below; final PR checks must also pass
before merge.

Local full receipt reproduction (`python scripts/formal/verify.py --record`)
failed in the unchanged artifact v4 worker probe: `--snapshot-contract-v3`
exceeded its existing three-second subprocess timeout. No new receipt was
written and no timeout was widened. Pinned CI must reproduce the frozen sources;
targeted numeric passes are not a substitute for the full verification gates.

The first ordinary CI automation check rejected an uncovered implementation
file: OrderNumeric was in the proof ledger but absent from the canonical
specification's implementation roster. Added the exact module path and direct
conformance/property evidence links; no coverage rule or proof was weakened.
The superseded pinned run was canceled; the corrected source revision must
repeat receipt reproduction and both canonical wrappers before merge.

The corrected local automation wrapper passed canonical coverage (40 specs,
552 implementation files, 245 evidence links). Its existing scheduled-collector
regression later failed when `verify_derivatives_receipt.py` exceeded its
unchanged ten-second subprocess timeout. Two other unchanged subprocess tests also hit their configured deadlines:
the edge campaign at 60 seconds and sequential contracts at 120 seconds. The
local wrapper finished with 182 passes and 3 failures; it is not a successful
run. The corrected ordinary CI automation job passed all 185 tests.

Numeric microbenchmark: GHC 9.4.8 `-O2`, one million base-selected validation
calls from deterministic binary64 words (seed 20261005), including generator and
fold overhead: 500228 accepted, 0.045356 process CPU seconds. This single sample
is not a wall-clock, worst-case or execution-latency guarantee. It excludes
credential, request, signing, network, venue filtering and fill work. Reproduce
with `ghc -O2 -ihaskell/app` and temporary build output outside Git:

```haskell
module Main (main) where
import Control.Exception (evaluate)
import Data.List (foldl')
import Data.Word (Word64)
import GHC.Float (castWord64ToDouble)
import System.CPUTime (getCPUTime)
import Trader.OrderNumeric (validateMarketNumbers)
main :: IO ()
main = do
 start <- getCPUTime
 let words64 = take 1000000 (iterate (\w -> w * 6364136223846793005 + 1442695040888963407) (20261005 :: Word64))
 total <- evaluate (foldl' (\acc w -> acc + either (const 0) (const 1) (validateMarketNumbers False (Just (castWord64ToDouble w)) (Just 1))) (0 :: Int) words64)
 stop <- getCPUTime
 print (total,fromIntegral (stop-start) / 1e12 :: Double)
```


## Frozen-source verification

Pinned [run 37315518229](https://github.com/diegueins680/trader/actions/runs/37315518229),
job 111781143189, checked exact source commit
`5fed08df1a88fd3b8108de1e91dc7d4d5cbdc5bc`:

- `python3 scripts/formal/verify.py --record`: PASS, 13:19:01–13:19:53 UTC.
- `bash scripts/verify.sh formal`: PASS, 13:19:53–13:21:18 UTC;
  198 integrity tests, all 62 SMT requirement groups, state models and compiled
  conformance. The new gate contributes five queries and the 40-state model.
- `bash scripts/verify.sh full`: PASS, 13:21:18–13:27:00 UTC;
  formal repeated successfully (198 tests, 33.455 seconds), Haskell format/lint/
  build/smoke/test, 241 web tests and 185 automation tests.

Raw reproduced receipt SHA256:
`47263304c2fbf16a6fa42a5ebef7440e0db07c29bc1b172f63637f329ccc2d62`.
It was decoded unchanged from the runner log, and every locked source hash
matched the frozen local bytes. Changed receipt sections are source hashes,
capability graph/hash evidence, the two new SMT groups and order-number results.
The capability graph adds only OrderNumeric: 120 source modules and 294 local
edges; trader-hs reaches 97 files instead of 96. Existing research isolation
and all prior proof conclusions remain checked; no new authorization appears.

The corrected ordinary formal job passed all 198 tests, then failed only on
comparison against the prior committed receipt. The final evidence commit
imports the reproduced receipt, updates this report and removes the temporary
read-only reproduction workflow. No proof or implementation source changes
after the passing full run. Final CI must pass with this receipt before merge.
Merge must use the exact tested head, suppress deployment workflows, and verify
merged-tree identity and GitHub Actions/deployment records afterward.

Mission completion remains false: 33 broad obligations and both research
acceptance gates remain unresolved. These scoped engineering checks do not
constitute economic acceptance or full production safety verification.

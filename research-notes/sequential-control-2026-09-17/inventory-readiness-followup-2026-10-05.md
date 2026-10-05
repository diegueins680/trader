# Inventory readiness — 2026-10-05

Registration `e61975ab` precedes implementation; baseline
`902a45786f1bee407731765bd45c608a6e06cab0`.

## Finding and repair

README's reconciliation promise was stronger than its implementation. The orphan
planner treats a running/starting worker without a local side as adoptable, and
omits simultaneous long/short symbols it cannot adopt. Neither condition certifies
readiness. The post-start branch checked running/trade-enabled only, permitting
wrong-side owners to certify completion. A prior true flag also survived an
interrupted later scan.

Keep the planner, adoption behavior and ownership untouched. A separate pure
predicate checks ALL returned inventory rows. Zero finite amounts need no owner;
nonzero amounts require a nonempty normalized symbol, supported signed side,
and an existing running, non-starting, trade-enabled runtime with matching side.
Non-finite rows reject. Both hedge sides cannot match one local side. Readiness
is cleared before each enabled scan and published from that scan only. Start
acknowledgment is no longer evidence; the next scan can restore readiness.
The existing pending-adoption-start exclusion also remains required, including
flat inventory. The scan does not introduce any additional exchange request.

## Evidence and exact limits

F-INVENTORY-READINESS-PREDICATE: four satisfiable-premise/UNSAT-violation queries
on IEEE binary64 finite/zero classification, unbounded side integers and Booleans,
plus two satisfiable accepting witnesses. GHC/base and source correspondence are
trusted. The universal inventory fold has reviewed inductive composition; this
is not a machine-checked compiler or list-refinement proof.

F-INVENTORY-READINESS-FLOW: exhaustive two-cycle, three-outcome model has 25 states,
33 transitions and maximum shortest depth7. It checks failure/interruption,
clear-before-scan, absence of optimistic acknowledgment and recovery by a later
successful scan. This is a publication model of captured evidence, not a model
of all exchange/DB/runtime interleavings.

F-INVENTORY-READINESS-CONFORMANCE: actual extracted Haskell predicate compiled
with pinned GHC checks 6468 rows twice against an independent oracle, including
NaN, infinities, signed zeros, subnormal/extreme amounts, all Boolean runtime
metadata, absent/unsupported/matching/opposite sides and empty/nonempty symbols.
1188 cases accept. Source checks bind complete predicate/scan/caller/HTTP fragments
and every readiness-reference occurrence; mutations test coverage and guard loss.
The full Haskell wrapper compiles the actual integrated server separately.

CE-READINESS-001–004 are synthetic code-domain witnesses, not measured production
incidents. The exact legacy adoption predicate is retained. The checker reproduces
its permissive case and the legacy orphan-filter/post-start publication algebra;
it does not execute authenticated old-server behavior.

A-INVENTORY-READINESS assumes venue completeness/truthfulness and ordinary pinned
runtime behavior. Runtime-map and bot-state reads are not one atomic snapshot with
exchange inventory or persistent owners. A state change after the scan can stale
the health snapshot. DB ownership uniqueness, multi-process leases, continuous
freshness, HTTP/drain linearization and complete obligation16 remain unproved.
Readiness may briefly report 503 on each inventory scan and stays 503 if scanning
blocks; that conservative health behavior is intentional and documented.

The inherited ownership follow-up must distinguish `positions_bot_id_uniq` in
`Trader/Ops/Migrations.hs` (one row per bot) from the protected position identity.
`persistBotSnapshot` identifies a bot by tenant/platform/symbol/market/interval;
`loadPersistedPositionOwnersMaybe` folds returned rows into one symbol/side map,
keeping the first row. Neither index nor presentation deduplication alone proves
there is at most one valid live owner across processes. This is a source-audit
finding, not a claim that duplicate live ownership occurred. No ownership change
is made; that remains a separate unresolved obligation.

## Acceptance status

Obligation16 advances open → partially_verified. Totals: **5 scoped closures,
28 partial, 5 open**. Existing closure criteria are unchanged. This does not
complete the user's broader mission. Next critical step is composing certified
snapshots with actual persistent-owner/runtime invalidation, without changing
production ownership or risk settings. No unresolved broad requirement is hidden
as an environmental assumption.

No new financial trial, research integration, model, dependency or configuration.
Frozen evidence remains 108 fits / 19440 replays / 19548 registry rows; development
is contaminated, all108 OPE batches invalid, no independent matched-champion
confirmation, 1227 final returns sealed and prospective embargo2027-01-20T13:00Z.
No new OOS/holdout, costs, stress, drawdown, tails, inference, RL seed or OPE claims.
Champion preserved; recommendation remains no candidate adoption.

## Verification

Targeted InventoryReadinessTests passed: three tests, 14.107 seconds. HLint on
Main.hs reports no hints. The local full integrity invocation
`TMPDIR=/private/tmp VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/diegosaa/.cache/trader-proof-20261003/bin/python -m unittest discover -s scripts/formal -p test_integrity.py`
failed after 205 tests / 427.393 seconds in unchanged PPO process-bridge coverage:
`ValueError: PPO process bridge: no actual inference for trained policy`.
No timeout or test was weakened. This local run is not reported as passing.

Initial CI automation expected the old pair return and both permissive readiness
writes. Commit `0962b1b5` replaces those assertions with all-row coverage,
clear-before-capture and exactly two permitted writes; its targeted automation
case passes. Ordinary formal CI at that head passes all205 integrity tests
(41.170 seconds), then correctly rejects the stale committed receipt. Pinned
receipt generation and canonical formal/full checks are required before importing
new evidence; final-head CI and no-deployment audit follow.
No deployment, order, authenticated exchange experiment or live-flag change.

## Predicate microbenchmark

Pinned local GHC9.4.8, `-O2`, one million varying amount/side-metadata rows:
checksum666667; `getCPUTime` difference0.052915 seconds. This allows normal GHC
inlining/fusion and is not a worst-case or end-to-end IO/scan/inference bound.
No additional inventory request is introduced. To reproduce, save the following
small program outside Git, compile `ghc -O2 Main.hs -o readiness-bench`, and run
`./readiness-bench`. Timing is descriptive and host-dependent.

```haskell
module Main where
import Control.Exception (evaluate)
import Data.List (foldl')
import System.CPUTime (getCPUTime)
data RuntimeAdoptionInfo = RuntimeAdoptionInfo
    { raiRunning :: !Bool
    , raiStarting :: !Bool
    , raiTradeEnabled :: !Bool
    , raiSide :: !(Maybe Int)
    }

inventoryRowReconciled :: Double -> Maybe Int -> String -> Maybe RuntimeAdoptionInfo -> Bool
inventoryRowReconciled amount side symbol mInfo
    | isNaN amount || isInfinite amount = False
    | amount == 0 = True
    | null symbol = False
    | otherwise =
        case (side, mInfo) of
            (Just positionSide, Just info) ->
                (positionSide == 1 || positionSide == (-1))
                    && raiRunning info
                    && not (raiStarting info)
                    && raiTradeEnabled info
                    && raiSide info == Just positionSide
            _ -> False


main :: IO ()
main = do
    before <- getCPUTime
    total <- evaluate (foldl' (\acc i -> if inventoryRowReconciled (if even i then 0 else 1) (Just 1) "BTCUSDT" (Just (RuntimeAdoptionInfo True False True (if i `mod` 3 == 0 then Just 1 else Nothing))) then acc + 1 else acc) (0 :: Int) [1..1000000 :: Int])
    after <- getCPUTime
    print (total, fromIntegral (after-before) / 1e12 :: Double)
```

## Pinned reproduction accepted

Exact source `0962b1b56e082b0788eceb0e2f998ca7b9e06a49`,
[run37342922758](https://github.com/diegueins680/trader/actions/runs/37342922758),
job111874244216, all success:

- Receipt reproduction: 2026-10-05 16:45:34–16:46:44UTC.
- `bash scripts/verify.sh formal`: 16:46:44–16:48:49UTC; 205 integrity tests
  in54.125 seconds, 69 SMT groups and all registered model/conformance receipts.
- `bash scripts/verify.sh full`: 16:48:49–16:57:15UTC; 205 integrity tests
  in53.252 seconds, actual Haskell build/lint/tests/smoke, 241 web and185 automation
  tests. All pass. The source is the same as the repaired readiness implementation.

Imported runner receipt bytes unchanged, SHA256
`c4c2faceaa9ff0022c37ea492de3df54a69db9fc071f5569826c881f7a3322d7`.
Source hashes exactly equal the lock and actual files. Reviewed changed sections:
`inventoryReadiness`, `smt` (one added group), `sourceHashes` and only the source
hash subfield of `capabilityIsolation`. No prior result is relabeled or removed.
The temporary reproduction workflow is removed before final ordinary CI.

The model's interruption edge means an aborted scan that exits before publication;
it is not a proof that every asynchronous exception escapes existing handlers.
Caught errors in inherited IO helpers and full worker cancellation semantics are
outside the checked pure predicate and publication abstraction. This is one of
the implementation-refinement limitations, not a new shutdown certificate.

Final-head checks and the no-deployment merge audit are recorded durably in
[PR303](https://github.com/diegueins680/trader/pull/303).

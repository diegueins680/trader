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

Targeted results and pinned formal/full/final CI will be recorded before merge.
No deployment, order, authenticated exchange experiment or live-flag change.

# Inventory readiness snapshot v1

Registered 2026-10-05 against 902a45786f1bee407731765bd45c608a6e06cab0,
before implementation. This repair changes health reporting only: it does not
change adoption, ownership, authorization flags, order execution or fleet policy.

## Conflict and authoritative interpretation

README promises readiness only after exchange inventory and live ownership are
reconciled. The adoption planner deliberately allows a running/starting runtime
without a local side and omits simultaneous long/short symbols it cannot adopt.
Those planner semantics must not be reused as evidence of reconciliation. Its
post-start readiness write also treats any running trade-enabled worker as an
owner without checking the exchange side. Preserve planner semantics and give
readiness a separate conservative predicate.

## Concrete specification before code

For every returned venue row, let amount be its raw binary64 position amount,
side its existing signed-side interpretation, and runtime the normalized-symbol
lookup in the captured local runtime map. A row is reconciled exactly when:
amount is finite AND (amount = 0 OR (side is +/-1 AND symbol is nonempty AND
runtime exists AND running AND NOT starting AND tradeEnabled AND localSide=side)).
Zero positions require no owner. An empty successful response is reconciled.
A failed scan, non-finite row, nonzero uninterpretable row, missing owner,
starting worker, flat/wrong-side worker, or unsupported hedge blocks readiness.
The universal row predicate includes ALL returned rows, never the filtered
adoptable orphan list. No mutation of the existing adoption decision is permitted.

At each enabled scan cycle: clear ready; obtain scan; publish its reconciled bit AND no pending adoption-start workers
(or false on error); never promote ready from start acknowledgments. A later
successful scan can restore ready. An interrupted scan leaves ready false.
HTTP health/readiness additionally require not draining, as before.

## Obligations and verification

F-INVENTORY-READINESS-PREDICATE: SMT Boolean/discrete/IEEE finite classification
proves each accepting nonzero row has the required running, trade-enabled,
matching-side local runtime; two opposite nonzero sides cannot both pass against
one runtime. An arbitrary-length all fold composes by explicit induction reasoning,
not a claimed machine-checked list induction.
F-INVENTORY-READINESS-FLOW: exhaustive finite scan/publication model checks clear
before scan, failure/interrupt denial, no optimistic start acknowledgment and
recovery by a later successful scan. Publish results refer to the captured scan.
F-INVENTORY-READINESS-CONFORMANCE: compile the actual pure Haskell predicate,
exhaust a finite metadata/numeric partition, and preserve legacy counterexamples.
Source admission binds complete predicate bodies, actual scan/caller fragments,
all readiness reference reads/writes and the unchanged adoption-planner bodies.
Unknown references or mutation of a reviewed fragment fails CI.

## Scope and assumptions

Trust pinned GHC/base, finite floating classification, existing venue signed-side
interpretation and ordinary immutable data; venue response completeness and
truthfulness are environmental assumptions. The runtime map/state reads are not
one atomic exchange/DB snapshot. No persistent-owner uniqueness, multi-process
lease, continuous freshness, request-time revalidation, HTTP/drain linearization,
exchange correctness or complete obligation16 closure is claimed. Record16 as
partial only after the source-bound results reproduce. Unrelated obligation
criteria remain unchanged. No financial experiment or holdout access.

Pinned verification: existing Python3.13.3/Z3 4.15.4/GHC9.4.8 toolchain; solver10s;
finite model bounds and exhaustive row counts measured in the receipt. Mutations
must reject weakened guards, filtered inventory coverage, optimistic writes and
removed pre-scan clearing. Run formal and full wrappers before merge.

Pre-integration amendment: retain the existing no-pending-adoption-start guard,
even on a flat inventory snapshot. This strengthens admission and avoids making
any previously blocked pending adoption ready through this repair.

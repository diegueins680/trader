# Async shutdown seal follow-up — 2026-10-04

Decision: repair local async shutdown acknowledgement. Registration `4d4d3d43`
precedes implementation, on baseline main `d56b4d83`. This is an engineering
repair, not a financial experiment or a candidate promotion. No datasets,
checkpoints, final holdout, live flags, deployment identity or risk limits change.

## Findings and intended behavior

The inherited shutdown collected published JobEntry values, delivered cancellation,
and waited for result cells. Preparation can own a slot without a published entry;
a result can be available before callback cleanup finishes. Both produce false
completion in preserved source-bound control-slice witnesses. The fixture uses the
actual pinned baseline admission module and replaces Main's persistence/HTTP with
barrier-controlled callbacks; it does not execute the entire HTTP server.

| Counterexample | Baseline acknowledgement / active count | Repair |
| --- | --- | --- |
| CE-ASYNC-SEAL-001: preparing outside snapshot | True / 1 | Wait times out / 1 |
| CE-ASYNC-SEAL-002: result before cleanup | True / 1 | Wait times out / 1 |

A pool now stores its count and permanent closed bit atomically. Shutdown seals
all pools before snapshots. Sealed pools reject reservations before preparation.
The last release (or seal of an empty pool) signals a private completion MVar;
waiters read without consuming it. Main waits on these barriers, not result cells.
This covers reservations whose preparation or publication straddles the snapshot.
Pre-seal reservations remain valid; a late-published worker may escape snapshot
cancellation and consume the remaining deadline, but cannot cause false success.

No public configuration, job identifier or JSON schema changes. Queue-full error
text is preserved. A closed pool uses an explicit internal constructor and returns
`Async job admission is closed for server shutdown.` through the existing error
channel. Shutdown still uses the existing outer monotonic deadline.

## Evidence and scope

- Six SMT claims over valid integer counts and positive capacities; prior checked
  signed-Int overflow claims remain in force.
- Two callers, two sealers, capacities 1/2: 2,932 states, 8,622 edges, maximum
  shortest depth 17; rank starts at 206 and decreases. Delayed notification is
  explicit. Completion implies permanent closure and zero reservation owners.
- Compiled pure boundary comparison: 72 cases. Four added runtime tests cover
  empty sealing, unpublished preparation, result-before-cleanup, interrupted
  waiters, repeated sealing/reads and 32 generated concurrent schedules, seed
  20261004. The complete admission suite has eleven tests.
- Mutations exercising early completion, reopening, consumed completion,
  post-seal admission, source drift and a nonterminating transition are rejected.
- Source locks bind actual Main composition and private helper. A-ASYNC-SEAL
  names GHC/base primitives, private state, exactly-once release and progress.

A first runtime-test attempt exposed a test harness race: it could cancel a waiter
before the waiter installed its finalizer. An entry barrier now makes that schedule
explicit. No production guarantee or verification gate was weakened to fix it.

These are scoped SMT, model and conformance results, not full Haskell IO refinement.
Pre-seal work, nonreturning callbacks, process death, detached descendants,
whole-server HTTP drain coordination and shared durable ownership remain outside
this guarantee. SHUTDOWN-DEADLINE-001 stays HIGH/OPEN. The broad ledger remains
3 closed, 28 partial and 7 open, with unchanged closure criteria. More lemmas do
not close a whole-system requirement without its remaining implementation evidence.

Economic/OPE results and the 1,227-return sealed holdout are unchanged. No candidate
passed promotion gates; continue offline research. No order-capable policy,
authenticated exchange experiment, deployment or live-money exploration is added.

## Reproduction and continuation

Use the pinned toolchain and run from the repository root:

```sh
bash scripts/verify.sh formal
bash scripts/verify.sh haskell
bash scripts/verify.sh full
```

Deterministic receipts are in `formal/research/results.json`; actual wrapper and
GitHub results are recorded in the PR. No inference or financial benchmark is
claimed. Next work remains whole-server drain/admission composition and durable
job reconciliation with explicit ownership: absence from one instance's local
registry does not establish that a shared durable job is dead.

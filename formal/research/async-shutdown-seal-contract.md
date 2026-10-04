# Async shutdown seal contract v1

Registration: `research-notes/registrations/async-shutdown-seal-engineering.json`.
Baseline: d56b4d83. Classification: lifecycle, concurrency, safety, liveness,
operational correctness. Affected broad obligations: 14, 33–38.

## Conflict and authoritative scope

Existing shutdown snapshots published JobEntry values and waits on their result
cells. An admitted preparation may be outside that snapshot. A result can also
precede completion of the callback and its cleanup. Neither establishes all
local reservations are finished. Existing HTTP drain checking stays compatible;
this change seals each async pool when its shutdown phase begins, before any
snapshot. It does not make HTTP drain and every server admission one transaction.
A reservation acquired before the seal remains legitimate and may publish/run.

## Representation, transition system and refinement

A private pool holds (n, closed) in one atomic IORef and an initially empty private
completion MVar. Initial state is (0, False, empty). Capacity L is positive.
Reserve: if not closed and 0 <= n < L, increment n and accept; otherwise reject
without preparation. Seal: atomically set closed True, preserving n. Release:
atomically decrement positive n, preserving closed. Seal/release signal completion
with nonblocking tryPutMVar iff their resulting state is closed and n = 0.
The state update and notification execute masked; no interruptible wait occurs.
Wait reads (never removes) the completion MVar. There is no reopen operation.
The abstract notification may lag the state transition; concurrent redundant
signals are harmless. A successful wait implies closed and zero owners forever.

Count ownership and publication gating refine the already registered admission
protocol. Parent preparation/fork failure releases; after fork the child's outer
finalizer releases. Completion excludes descendants and durable remote effects.
Main seals all three pools, snapshots/cancels known workers, then waits on pool
completion instead of result cells. A late-published pre-seal worker is counted;
it can cause timeout but cannot produce false success. No forced cancellation of
an unregistered preparer is promised. All waits remain inside the existing outer
monotonic shutdown budget. Repeated seals and waits are idempotent.

## Obligations

- F-ASYNC-SEAL-INVARIANTS: SMT, n in [0,L], L positive signed Int: closure
  preserved by release, sealed admission rejected, bounds preserved, notification
  condition implies closed and n=0, and stable zero after closed. Existing numeric
  overflow proof remains applicable to checked admission/release.
- F-ASYNC-SEAL-LIFECYCLE: exhaustive finite transition model composing admission
  with two concurrent sealers and delayed notifications; two callers, L=1/2.
  Check exact ownership, monotonic seal, no post-seal reserve, no premature
  acknowledgement, repeated-wait stability, no protocol deadlocks and conditional
  eventual quiescence via decreasing rank. Preserve snapshot counterexamples.
- F-ASYNC-SEAL-CONFORMANCE: compiled helper barriers exercise seal before reserve,
  preparation outside snapshot, result before finalizer, repeated seal/wait,
  interrupted waiter, pre-seal failures, and 32 generated concurrent schedules.
  Compile a pinned baseline helper and preserve deterministic counterexample
  fixtures. Source bind actual Main seal-before-snapshot and barrier composition.

## Assumptions and limitations

A-ASYNC-SEAL inherits A-ASYNC-ADMISSION's trusted GHC/base atomics, MVars, masking,
finite allocation, private cells and total callbacks. A failing fork creates no
child. Each accepted reservation releases exactly once. Eventual completion needs
scheduler service and returning preparations/callbacks/finalizers. Process death,
unsafe mutation, child descendants, cross-instance leases and external effects
are outside scope. Model/source checks plus conformance are not a complete proof
of Haskell IO refinement. No financial research data or live authorization changes.

Closed-pool rejection gets an explicit internal constructor and an explanatory
existing-string-channel API error. Queue-full text and all normal JSON/CLI,
configuration, persistence and job identifier semantics remain unchanged.


## Reproduced bounds

The finite composition checks 2,932 states / 8,622 edges, maximum shortest depth
17 and initial decreasing rank 206. Capacity one: 1,381 states / 3,864 edges;
capacity two: 1,551 states / 4,758 edges. Both contain 492 states with a pending
notification before completion, and 25 terminal states. Six SMT claims accompany
72 compiled pure cases, four runtime tests and 32 generated schedules. The
original admission suite remains active (eleven tests total).

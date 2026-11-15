# Supervised worker registry contract v1

Registration: `research-notes/registrations/worker-registry-engineering.json`.
This corrects actual inherited registry behavior associated with obligations
14, 33–37. It does not close server-wide admission, ownership, or shutdown claims.

## Conflict and resolution

`stopSupervisedWorkersBounded` previously removed every entry before cancellation.
Cancellation delivery was treated as completion. A timed-out call therefore lost
unfinished workers and a retry could report success while they were still active.
The registry also remained open to registration after stop. Intended behavior is
persistent closure, retained unfinished entries and success only after each
registered worker's outer finalizer has acknowledged the end of its body.

All five production callers register during server startup and discard returned
ThreadIds. The internal result becomes `Maybe ThreadId`: `Nothing` means closed
and launches no worker. This is not a CLI, JSON, saved-configuration or artifact
change. HTTP drain state and order authorization remain separate boundaries.

## State and transitions

State is (closed, entries); each entry has a unique thread identity, immutable
completion MVar initially empty, and a cancellation-request MVar initially False.
Only the worker's outer finalizer fills completion, exactly once, without reading
or acquiring the registry lock. Readers use nondestructive `readMVar`.

Start holds the registry MVar under async masking, rejects closed state, allocates
completion/cancellation cells, forks with explicit child unmasking, and publishes
the entry before releasing the lock. The child's first supervision iteration
reads the same registry MVar, so its action cannot start before registration.
The child installs its completion finalizer before restoring action interruptibility.
No blocking operation follows successful fork before parent publication.

Stop bounds registry acquisition, cancellation dispatch and completion waiting
inside one `System.Timeout`. Closure linearizes under the registry lock and never
reopens. It snapshots unfinished entries without deleting them. A timeout before
lock acquisition may leave closure unperformed and must return False. Concurrent
starts are ordered by this lock; already admitted iterations may finish after
closure. No new iteration is admitted by the registry after closure.

Cancellation requests serialize per entry: at most one helper sends ThreadKilled.
Interruption before dispatch leaves it retryable; dispatch/publication are masked.
A retry waits on the same completion cell. Delivery does not establish completion.
Finished entries may be pruned; unfinished entries must never be pruned. The
observed count is the unfinished count of a captured registry snapshot.

## Requirements and techniques

- F-WORKER-REGISTRY-LIFECYCLE (safety, concurrency, lifecycle, conditional liveness):
  finite transition model, two start attempts/workers, two stop callers, two timeout
  ticks per caller; distinguish request, delivery and completion. Check closure,
  retained unfinished entries, at-most-one cancellation helper, and no successful
  stop with an unfinished captured worker. A decreasing rank proves termination
  of this bounded protocol, assuming primitive progress.
- F-WORKER-REGISTRY-INVARIANTS (functional and concurrency correctness): SMT verifies
  Boolean closure monotonicity, admission exclusion, request idempotence and the
  per-entry prune/success predicates under the reviewed atomic-step interpretation.
- F-WORKER-REGISTRY-CONFORMANCE (implementation evidence): compile the actual Haskell
  module and run deterministic barriers and fixed-seed concurrent/repeated-stop
  cases. Source locks bind masking, lock order, finalization and completion reads.
  Mutation tests must reject the old premature-success and reopen transitions.

## Assumptions and limits

A-WORKER-REGISTRY trusts pinned GHC/base MVar, mask, forkIOWithUnmask, finally,
System.Timeout and thread identity semantics, ordinary finite registries, available
memory and bounded scheduler/primitive service. No unsafe constructor access or
external mutation of private cells is permitted. Child callbacks may ignore
cancellation; then stop must fail within its conditional waiting bound. Completion
means the tracked callback and its finalizers finished, not OS-thread reclamation,
untracked descendants, external resources or position reconciliation.

Source-bound SMT/model checks plus compiled conformance are not a complete proof
of Haskell IO refinement. Real HTTP/bot/order admission, repeated process signals,
blocked logging, detached workers, the frozen RL runner and production resource
recovery remain open. Registry closure occurs at its lock operation, not at the
separate HTTP `beginDrain` call. No financial experiment or deployment is included.

## Abstraction, temporal properties and conformance boundary

The concrete registry MVar maps to `closed`; a registered worker with an empty
completion MVar maps to unfinished, and a filled one maps to finished. Pruned
finished entries remain finished ghosts in the model. The cancellation cell maps
to a request count of zero or one. Each stop caller retains an immutable captured
mask, so dispatch can occur after a captured worker has already finished. Delivery is a separate ghost event. Two model
callers represent concurrent or interrupted/retried stops. A failure before the
registry lock leaves `closed` unchanged. Once closure succeeds, future starts
cannot add a member to either caller's unfinished snapshot, so checking all
admitted unfinished workers is equivalent to checking each captured snapshot.
The model does not describe callback internals or a process restart.

For the finite transition graph the checker establishes `AG(closed => AX closed)`,
`AG(stopSuccess => closed && noUnfinishedCapturedWorker)`, and at most one
cancellation request per worker. Its strictly decreasing nonnegative rank starts
at 24; every maximal protocol path ends in finished/rejected start attempts and
successful/failed stop calls (`AF terminal` in this finite progress abstraction).
It includes 14,095 states, 55,904 edges and shortest-path depth at most 16, with two
workers, two callers and two timer ticks each. Infinite scheduler stuttering is
excluded by A-WORKER-REGISTRY; this is not a scheduler or wall-clock theorem.

The source lock checks the exact reviewed function bodies and private exports.
Compiled tests exercise the actual module, including parent masking and interrupted
stop callers. Neither source identity nor these tests proves that every Haskell
execution refines the atomic model. That gap is explicit in the ledger. No worker
proposal, policy action or exchange request is introduced by this repair.

Runtime semantics are grounded in the pinned primary documentation:
[base 4.17.2.1 Control.Concurrent](https://hackage-content.haskell.org/package/base-4.17.2.1/docs/Control-Concurrent.html)
and [Control.Exception](https://hackage-content.haskell.org/package/base-4.17.2.1/docs/Control-Exception.html).
Cancellation delivery can precede completion of exception cleanup; explicit
completion cells therefore supply the acknowledgement used here.

MC-WORKER-001 preserves a model-coverage correction: start, capture, finish, then request cancellation of the captured completed worker. The initial draft model tested current completion instead of captured membership at dispatch. The final model includes 1,220 such dispatch transitions and checks the preserved trace; no implementation defect is claimed from this abstraction correction.

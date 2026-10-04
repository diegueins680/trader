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

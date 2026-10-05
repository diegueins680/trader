# Bot worker publication contract v1

Engineering registration, 2026-10-05; baseline287b888af3da2b477c81bdc7c214ff640190aee9.
No market experiment, holdout access, deployment, authenticated exchange call or
change to production ownership, trading permissions, fleet, leverage or risk limits.

## Conflict and authoritative behavior

`botStartSymbolWithSettings` forks `botStartWorker` inside `modifyMVar` before
publishing its `BotStarting` entry. The callback then reads the clock and builds
that entry. A synchronous exception or asynchronous interruption after fork can
restore the old runtime map while the child continues `doStart`/initialization.
A worker must not initialize on behalf of a start that failed publication.
The existing duplicate-start check must remain inside the same map transaction.

Introduce a small generic Haskell publication helper. Under masking, take the map
and run preparation with the caller's restored masking state. Preparation chooses
an unchanged result or a worker action plus a pure ThreadId-dependent publication.
Fork the worker behind a private Boolean gate. Force the new map and result to WHNF,
then commit the map under masking. Only after that commit release True; on any
exception before commit, release False and propagate the exception. False exits
without running the worker action. Keep the gate private and use nonblocking
one-shot release. No cleanup uses blocking throwTo/killThread.
Move timestamp acquisition before fork, within preparation. Preserve the existing
second duplicate check, record fields, thread identity, worker action and outcome.
No new configuration or feature, no new model, no risk/ownership semantics change.
No claim that this helper fixes failures inside already-started worker actions.

## Requirements and abstraction

F-BOT-PUBLICATION-GATES: accepted execution implies successful prior publication;
failure cannot release execution; a publication replaces only its own transaction's
state; duplicate admission preserves state. SMT check the Boolean guards and
finite map-owner update, including satisfiable non-vacuity premises.
F-BOT-PUBLICATION-FLOW: finite two-caller, one-key protocol with gate wait, commit,
abort and execution. Exhaustively check reachable states, rollback, duplicate starts,
no execution before commit and monotonic one-shot gate decisions. Under explicitly
stated scheduler/primitive progress, terminal aborts permit child quiescence.
F-BOT-PUBLICATION-SOURCE: bind the complete helper, both actual duplicate checks,
metadata preparation, worker callback and post-commit gate ordering to the reviewed
source. Check no extra fork remains in the actual bot-start transaction.
F-BOT-PUBLICATION-CONFORMANCE: compiled Haskell tests deterministically schedule
pre-publication child execution in a legacy ordering adapter; synchronous failure,
parent cancellation during preparation, publication evaluation failure, successful
publication and concurrent duplicate starts; no network/credentials/orders.

Concrete MVar ownership and its masked replacement abstract to a transaction lock
and map-owner identity. Private MVar gate abstracts to Pending/Reject/Accept.
`forkIOWithUnmask` abstracts to spawning a gated child; its action runs only after
Accept and uses unmask. Returning/throwing preparation and publication are explicit
model transitions. Source correspondence is reviewed and tested; no whole-GHC IO
refinement is claimed. Pure publication is forced to WHNF, not deep-normal-form;
production constructors/HashMap operations are assumed total on admitted values.

A-BOT-PUBLICATION: pinned GHC9.4.8/base4.17.2.1 MVar, masking and fork semantics;
normal allocation and fixed trusted pure publication, no unsafe mutation of the
private gate or runtime state. Fork either returns one thread or fails without one;
nonblocking private-cell tryPutMVar and uncontended final putMVar complete under
masking. Preparation may throw or block and remains interruptible. Scheduling and
lock acquisition are weakly fair for conditional liveness; permanently blocked
preparation, runtime exhaustion, uninterruptible foreign calls or repeated external
cancellation are not bounded-time completion guarantees. External stop/removal,
post-start finalizers, persistent ownership, cross-instance uniqueness, inventory
freshness, drain linearization and durable crash recovery remain unresolved.

Original38 closure criteria/scopes remain unchanged. This targets evidence for
15/33/34/36/38, not blanket closure of any of them. Move15 open to partial only if
actual startup-publication evidence passes; full live-owner uniqueness remains a
blocker. Preserve all prior scoped closures with fresh source/build evidence.

Pinned source/model/SMT/conformance checks run in formal/full CI. Preserve the
legacy ordering counterexample, reject proof/source mutations, and report exact
finite bounds. Merge only the verified head, suppress deployment, audit afterward.

# Async job admission contract v1

Registration: `research-notes/registrations/async-job-admission-engineering.json`.
Scope: inherited `startJob` capacity ownership and publication, obligations 10,
33–38. This does not close whole-server drain admission or durable recovery.

## Conflict and intended behavior

The current `startJob` increments `jsRunning` before preparation and installs the
release finalizer only in a later child thread. Parent interruption during
preparation can strand capacity. The child can execute before its `JobEntry` is
published; interrupted publication can leave an untracked callback running.
Intended behavior: one capacity reservation has exactly one owner until release,
and callback entry requires successful registry publication. No CLI/API/JSON,
job identifier or configured concurrency-limit semantics change.

## Executable design and assumptions

A private `JobSlots` holds a sanitized positive Int limit and an atomic IORef count.
Admission atomically increments only when 0 <= count < limit. Release decrements
positive count; zero stays zero. Pure total predicates are source-bound and
SMT-verified over the signed Int domain, with compiled differential conformance.
The private constructor and no mutation exports prevent external count writes.

`startBoundedJob slots prepare execute publish` runs masked reservation/allocation
and restores the caller's masking state only during preparation. Failure before
fork releases the parent's reservation. A successful fork transfers release
ownership to the child, whose outer finalizer is installed before waiting on a
private gate. A child receives True only after publication succeeds. Publication
failure supplies False, so the child terminates without executing the callback
and releases capacity. No synchronous kill delivery is required for rollback.
The child explicitly unmasks only its executable callback. Parent cancellation
after publication is allowed to leave a registered job running; it must not release
that child's capacity. Publication must be masked, atomic and exception-safe at
its external mutation boundary (actual Main uses modifyMVar_ with pure HM.insert).

The finalizer releases exactly once on callback completion or cancellation. A
completion of this helper is not evidence of durable status persistence, HTTP
response delivery, descendant termination or external trading completion. Existing
cancel/poll/result semantics remain outside this admission-only scope.

A-ASYNC-ADMISSION assumes GHC/base mask, forkIOWithUnmask, MVar and atomic IORef
semantics, finite allocation, private cells, nonthrowing total numeric predicates
and eventual scheduler service. A failing fork creates no child. Repeated external
cancellation of already published jobs is allowed; no blocking operation occurs
in release. Process death and unsafe IORef mutation are excluded. Source identity
and conformance are not a full proof of Haskell IO refinement.

## Requirements and verification

- F-ASYNC-ADMISSION-NUMERIC: SMT over mathematical integers constrained to the
  compiled signed Int range proves checked admission cannot overflow and preserves
  0 <= count <= limit, and total release stays nonnegative and never increases it.
- F-ASYNC-ADMISSION-LIFECYCLE: machine-check a finite two-caller, capacity-one/two
  protocol including preparation failures, fork failure, parent interruption,
  failed publication, child cancellation and normal completion. Check exact
  reservation ownership, no callback before publication, no double release,
  capacity bound, and eventual release under primitive progress.
- F-ASYNC-ADMISSION-CONFORMANCE: compile actual helper; deterministic barriers for
  preparation failure/interruption, publication failure/interruption, masked parent,
  callback exception/cancellation, full queue and repeated concurrent attempts.
  Preserve a baseline control-slice witness bound to original Main source. Treat
  this as a control-slice refutation, not execution of the entire HTTP server.

Use 32 fixed-seed scheduling cases (20261004). No new dependency, financial trial,
market data, policy artifact, holdout access, live permission or deployment.

## Abstraction and checked bounds

A reservation is owned by the parent in preparation/allocation and by the child
from successful fork through finalization. `count` equals the number of these
owners. The gate abstracts its private MVar as absent, False (abort) or True
(execute). `published` is a ghost recording the successful atomic Main insertion.
The source lock binds the helper and Main preparation, execution, publication,
initialization and health-summary paths. Constructors remain private.

The graph checks `AG(count = owners && 0 <= count <= capacity)` and
`AG(everExecuted => published)`. A nonnegative rank decreases on every transition
from 20, proving finite protocol termination under the explicit callback/primitive
progress abstraction. Callbacks that never return retain their legitimate slot;
no unconditional termination or forced cancellation is asserted. Two callers at
capacities 1 and 2 yield 369 states / 846 transitions, depth at most 14. This is
not exhaustive GHC scheduling or a theorem about arbitrary callback effects.

Counter predicates have six SMT obligations over every signed Int input and
arbitrary positive Int maximum. Compiled differential checks use 292 cases,
including 256 generated (seed 20261004), plus seven runtime tests and 32 generated
concurrency cases. No property test is classified as proof. The baseline fixture
is a documented control slice, with original Main code preserved in the source
manifest; no claim is made that it executes all HTTP/persistence behavior.

The counter alone uses one atomic IORef. Atomic operations serialize its updates;
ordinary health reads are snapshots, not a synchronization mechanism. The checked
numeric callbacks are total: an exception while forcing a returned value after an
atomic update must not strand a slot. Memory exhaustion/unsafe mutation remain
explicit environmental exclusions. See the pinned primary documentation:
[Data.IORef](https://hackage-content.haskell.org/package/base-4.17.2.1/docs/Data-IORef.html)
and [Control.Exception](https://hackage-content.haskell.org/package/base-4.17.2.1/docs/Control-Exception.html).

Persistence still occurs during preparation. An interruption can leave a durable
running record with no published worker. This pre-existing recovery limitation is
not repaired or disguised by releasing the in-memory slot. Likewise HTTP draining
and job-store shutdown snapshots do not yet share an atomic admission gate.


Successor note (2026-10-04): the [seal contract](async-shutdown-seal-contract.md)
adds permanent closure to the private count representation, explicit closed/full
rejection constructors and reservation completion notification. The original
369-state model continues to describe the unsealed admission subprotocol; the
successor model composes closure. Compiled runtime suite now has eleven tests
including four seal tests and 64 generated concurrency cases in total. Original
counterexamples remain reproducible. HTTP-wide drain atomicity remains unproved.

## Shared-drain refinement extension (2026-10-04)

The current counter implementation uses STM instead of the baseline IORef.
Standalone constructors create an independent open controller, retaining the
original model projection and regression outputs. Server constructors compose
reservation with the same shared drain TVar; see `drain-pool-contract.md` and
A-DRAIN-POOL. Existing ownership/cancellation proofs do not establish whole-server
drain correctness. Existing backtest failure variants retain their mapping;
BacktestDraining adds HTTP 503 without altering the JSON error shape.

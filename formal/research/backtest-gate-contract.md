# Backtest execution gate contract v1

Registration: `research-notes/registrations/backtest-gate-engineering.json`.
Baseline main: a9c6b3c0. Scope: existing backtest gate used by HTTP, async jobs and
background research. Classes: lifecycle, numerical, concurrency and operational
correctness. Affected broad obligations 10, 21, 34, 36 and 38 remain broader than
this repair. No new model, feature flag, dataset or production permission.

## Existing conflict and canonical interpretation

Main wraps `try action` inside `timeout`. Its blanket SomeException handler can
consume both the timer's own exception and external shutdown cancellation.
Consequently actual expiry can become BacktestException instead of the existing
BacktestTimedOut/HTTP 504 contract, and caller cancellation can be returned as an
ordinary failure. Multiplying configured Int seconds by 1,000,000 can overflow
negative, which System.Timeout interprets as no timeout. Admission also increments
an MVar counter before installing a release finalizer under masking.

Authoritative intended behavior follows the existing failure constructors and
configured positive execution timeout: timeout is a timeout; external asynchronous
exceptions propagate after slot release; synchronous exceptions remain structured
BacktestException values. Existing queue-full text/status, JSON, limits and
constructor minimum of one remain compatible. Waiting admission still waits for
capacity; the execution timer starts only after acquisition, as before. No FIFO,
queue-wait deadline or unconditional termination guarantee is added.

## Mathematical and executable definition

For positive machine Int maximum B and configured Int seconds s, define
U(s,B) = min(B, max(1,s) * 1000000), evaluated in Integer before Int conversion.
The constructor sanitizes capacity L=max(1, configured limit) and seconds=max(1,s).
A private atomic IORef holds n, initially zero. The checked `admitSlot` and
`releaseSlot` predicates from the existing admission core implement increments
only when 0 <= n < L, and decrement positive n respectively.

`runBacktestWithGate` masks acquisition until its release finalizer is installed.
The callback runs in the caller's original masking state. All exit paths release
exactly once with a nonblocking atomic update. Busy admission never runs callbacks.
The timeout wraps the action; an outer `tryJust` catches only exceptions outside
SomeAsyncException. Own timeout returns Nothing -> BacktestTimedOut; synchronous
failure -> BacktestException; successful value -> Right; foreign timeout and
external asynchronous cancellation propagate through the release finalizer.

Waiting mode retries only BacktestBusy, after the inherited interruptible one-second
pause, owning no slot during that pause. No retry follows cancellation, timeout,
synchronous error or success. The small Haskell module exports a private gate type,
existing failure constructors/entry points, timeout getter, count observer and
pure conversion; Main retains HTTP/error-message formatting.

## Obligations and refinement relation

F-BACKTEST-GATE-NUMERIC: SMT over arbitrary positive B and signed Int inputs proves
1 <= U <= B, exact conversion below saturation, monotonicity, and no wrap to the
negative unlimited-timeout sentinel. Existing checked admission/release numeric
proofs remain dependencies. Compiled differential cases include Int extrema and
256 fixed-seed values.

F-BACKTEST-GATE-LIFECYCLE: exhaustive finite model for two callers, capacities 1/2,
and immediate/waiting modes; n equals live reservations, n<=L, failed admission
has no callback, each cleanup releases once, and external cancellation cannot
become ordinary completion. Include waiting/cancellation and success/synchronous
failure/own timer/outer timer outcomes. Retry cycles are explicit. Verify safety
invariants and that every reachable state can reach quiescence (EF terminal),
not unconditional AF termination or scheduler fairness.

F-BACKTEST-GATE-CONFORMANCE: compile the actual module and a source-preserved
baseline adapter. Preserve own-timeout misclassification, swallowed cancellation
and overflow witnesses. A separately labeled control-slice schedule exposes the
unmasked reserve-to-finalizer gap. Tests cover nested timers, external cancellation,
callback exceptions, admission saturation, waiting cancellation/retry, caller
masking semantics, cleanup and 32 generated concurrent schedules. Source locks
bind Main routing and unchanged HTTP message/status mapping. This is conformance,
not a complete proof of Haskell IO refinement.

## Assumptions and scope exclusions

A-BACKTEST-GATE: pinned GHC/base IORef/mask/finally/timeout/SomeAsyncException
semantics, private counter, total numeric callbacks, finite allocation and scheduler
service. Callback must allow delivery of timeout/cancellation and must not catch
and suppress it internally. Foreign blocking calls and uninterruptible masking
can exceed the timer. Exceptions classified as synchronous remain structured even
if another thread uses throwTo to deliver them; classification follows the
exception hierarchy, not unknowable delivery provenance. Global drain admission,
durable effects, cross-instance ownership and full IO refinement remain open.

Primary reference: [base 4.17.2.1 System.Timeout](https://hackage-content.haskell.org/package/base-4.17.2.1/docs/src/System.Timeout.html),
which documents negative-duration behavior, SomeAsyncException ancestry, and the
limits of blanket handlers and foreign calls. Do not claim forced termination.

## Shared-drain refinement extension (2026-10-04)

The current counter implementation uses STM instead of the baseline IORef.
Standalone constructors create an independent open controller, retaining the
original model projection and regression outputs. Server constructors compose
reservation with the same shared drain TVar; see `drain-pool-contract.md` and
A-DRAIN-POOL. Existing ownership/cancellation proofs do not establish whole-server
drain correctness. Existing backtest failure variants retain their mapping;
BacktestDraining adds HTTP 503 without altering the JSON error shape.


## FIFO refinement extension (2026-10-06)

The [FIFO progress contract](admission-progress-contract.md) supersedes the polling
behavior described above. Waiting registration is private and slot-free; acquisition
atomically removes the head and reserves capacity. Immediate calls may now return
Busy despite free capacity when tickets are queued. Cancellation removes its own
ticket; drain wakes and removes waiters. The old safety models now conservatively
allow queue-erased contention (reject/wait with free capacity); they do not certify
fairness. The new838-state model checks explicit weak-fair progress. Prior numeric,
exception-routing, ownership and drain regressions remain active. No whole-server
progress, new timeout, configured-limit or HTTP/JSON shape change is claimed.

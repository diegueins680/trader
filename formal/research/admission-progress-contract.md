# FIFO backtest admission and progress v1

Specified before implementation against main fb00a66707d2e645e15fb347833a661e044a7a77.
Scope: inherited BacktestGate waiting and immediate consumers; original obligation38,
with supporting ownership/draining obligations14/33/34/36. Original criteria remain.

## Conflict and intended behavior

The current waiting function sleeps one second after Busy, then competes anew.
Repeated terminating invalid jobs can acquire every intervening vacancy. Fair
scheduling of threads does not imply fairness of admission opportunities. Preserve
that feasible starvation cycle as a model counterexample and compile a deterministic
finite reproduction against the original helper. A finite execution is not proof
that the actual runtime executes an infinite starving schedule.

A waiting call now registers once in a private FIFO. Registration order is STM
commit order (not HTTP arrival time). New immediate callers return the existing
Busy response while any waiter is registered, even if a slot is temporarily vacant.
Only the FIFO head can acquire a vacancy. Admission removes its ticket and reserves
one slot in the same transaction. Queue registration is not slot ownership. Waiters
hold no slot, execute no callback, and may cancel without affecting other tickets.
A drain rejects and removes waiting tickets without permitting a new reservation.
Cancellation after reservation releases exactly once through the existing callback
finalizer. Callback timeouts remain execution-only. No timeout, concurrency limit,
HTTP/JSON shape, live risk limit, bot ownership or authorization flag changes.

State: (drain, capacity, activeCount, nextTicket, orderedDistinctTickets).
Capacity >=1; 0<=activeCount<=capacity; nextTicket is exact nonnegative Integer;
every queued ticket <nextTicket. Enqueue appends nextTicket then increments it.
Cancel filters exactly that ticket. Try admits iff open, queue empty, count<cap.
Wait admits iff open, caller=head, count<cap, then removes head and increments count.
Release decrements a positive count. The existing private owner protocol is retained.

A waiter reaching the head cannot be overtaken by later registrations or immediate
requests. With finite predecessors, eventual release by every admitted callback,
finite cleanup/STM operations, weakly fair scheduling of continuously enabled
admission and cleanup, and continued open service, it eventually acquires. If drain
begins or the caller cancels it eventually returns/rethrows instead. No wall-clock
bound, overload capacity, guaranteed success of a nonwaiting request, fairness of
uninterruptible callbacks or whole-server liveness is claimed.

## Verification plan

F-ADMISSION-PROGRESS-SOURCE: bind actual Haskell definitions, private queue storage,
mask/finalizer placement, Main call sites and unchanged timeout/error contracts.
F-ADMISSION-PROGRESS-NUMERIC: SMT verifies arbitrary Integer ticket monotonicity,
freshness/rank summaries and bounded Int reservation. SAT premises; UNSAT violation
queries; UNKNOWN fails. Seq append/view/filter and STM/GHC semantics are named
runtime assumptions, not theorem-proved standard-library internals.
F-ADMISSION-PROGRESS-FLOW: finite fixed-point model with three callers, capacities
1/2, retrying invalid competitors, FIFO admission, cancellation, drain and release.
Check safety plus fair-cycle exclusion for a persistent valid waiter. Explicit
stuttering and retry cycles; do not substitute EF termination for AF under fairness.
Preserve a legacy lasso satisfying thread progress but starving the valid waiter.
F-ADMISSION-PROGRESS-CONFORMANCE: compile actual old/new helpers; deterministic
barriers and observable queued counts, FIFO order, no barging, head/middle waiter
cancellation, drain wakeup, callback exception/cancellation and repeated attempts.
Seeded bounded schedules supplement the model; tests are not formal proof.

A-ADMISSION-PROGRESS: ordinary GHC9.4.8/base4.17.2.1/stm2.5.1.0/containers semantics,
atomic serializable STM with rollback, exact Integer arithmetic, Seq order/filter,
private cells and defined inputs, finite memory/allocation, callback/cleanup eventual
termination and weak scheduler fairness. Runtime source-to-model mapping is reviewed
and tested, not compiler/kernel refinement. No fairness is assumed for momentarily
enabled admission (the defect being fixed). Finite predecessors follow from finite
registration history at each ticket; no uniform queue-length/latency bound follows.

Broader38 remains unclosed until runner candidate isolation, all server admission
paths, persistence and inherited worker progress are composed. Passing this scope
may move38 open to partially_verified only. Existing12 closures must still reproduce;
no original scope, criterion or statistical gate is reduced.

Pinned primary runtime references (read2026-10-06):
[STM2.5.1.0](https://hackage-content.haskell.org/package/stm-2.5.1.0/docs/Control-Concurrent-STM.html),
[base4.17.2.1 exceptions](https://hackage-content.haskell.org/package/base-4.17.2.1/docs/Control-Exception.html).
The progress claim is our conditional specification, not a library guarantee.

# Shared drain / pool reservation contract v1

Registration: `research-notes/registrations/drain-pool-engineering.json`.
Baseline: 91e96444. Classes: safety, concurrency, lifecycle, operational.

## Conflict and authoritative scope

The HTTP ingress reads isDraining separately from subsequent async/backtest
reservation. beginDrain closes readiness immediately, but async pools are sealed
only in a later cleanup stage; the backtest pool has no server-drain guard.
A request paused after ingress can therefore reserve after drain begins.
Existing local pool closure proofs do not establish the shared server ordering.

The authoritative rule for the affected pools is: no reservation commits after
the shared drain transaction. An earlier committed reservation retains ownership
and its cleanup obligation; it can run after the drain transaction. This is NOT
a reinterpretation or closure of broad obligation 14's stronger no-new-compute,
bot-start and authorization criterion. Those consumer boundaries remain open.

## Definition and implementation refinement

Let d be a monotonic Boolean (False initially), and each pool have count n and
positive capacity L; async pools additionally have their existing local close c.
Drain: (d,n) -> (True,n), returning first = not d. Reservation executes one STM
transaction reading d and the pool counter: if d or c or n >= L, reject unchanged;
otherwise increment n. Release remains exactly-once at the existing owner boundary,
without consulting d. Local async close and completion acknowledgement retain
existing semantics. A read-only readiness snapshot conveys no admission authority.

Concrete representation: private TVar Bool for DrainController, TVar counters for
both pool types. beginDrain and unlessDraining compose with STM; the guarded action
is STM, never arbitrary IO. Server constructors supply the SAME controller to all
three async stores and the shared backtest gate. Standalone constructors retain
independent initially-open controllers for backward-compatible offline use.

A drained backtest returns BacktestDraining (HTTP 503 using the existing JSON error
shape). Waiting retries only Busy, so drain is terminal at the next reservation
attempt (existing one-second retry remains). Async rejection retains JobQueueClosed
and its existing message/mapping. No configuration/authorization flag changes.

## Proof obligations

F-DRAIN-POOL-ORDER: SMT confirms reject-after-drain, monotonic closure, count bounds,
release independence and commutation of independent pool updates under arbitrary
nonnegative bounded counts. Named STM serializability assumption connects equations
to a committed transaction; this is not a formal proof of the STM runtime.

F-DRAIN-POOL-LIFECYCLE: finite state graph with two callers, two pools, capacities
1/2 and two drainers. Include stale ingress reads, reservation, callback, cleanup,
local close and repeated drain. Check no post-drain reservation, ownership/bounds,
monotonic drain and existence of terminal progress. Preserve a legacy stale-ingress
counterexample. Do not claim callback quiescence, fairness or unconditional shutdown.

F-DRAIN-POOL-CONFORMANCE: compile the real implementation; deterministic tests for
stale ingress on each pool, both reservation/drain orders, rejection without effects,
waiting after drain, release after drain, repeated drain and 32 generated concurrent
schedules. Source binding checks every Main constructor receives the shared latch.
Preserve legacy fixtures before modification; tests supplement the abstract model.

## Assumptions and limitations

A-DRAIN-POOL: GHC 9.4.8, base 4.17.2.1 and stm 2.5.1.0 implement serializable atomic
transactions, rollback and asynchronous-exception masking as documented; private
TVar state and finite pure reservation transactions. Scheduler fairness, callback
cooperation and existing cleanup assumptions remain environmental. Transactions
have no IO, retry or user callback. Contention has no hard wall-clock bound.
No global bot/order drain certificate, durable ownership/recovery, forced callback
termination, or full implementation refinement. Champion/holdout/live settings stay
unchanged. New proof IDs must not close a broader ledger item by scope reduction.

The model allows optional local seals in both abstract pools, a superset of the
backtest pool (which has no local seal). Backtests refine the always-unsealed
projection. Waiting retry remains in the prior backtest model; this composition
checks a single reservation attempt. No post-drain acceptance is possible on any
retry, but a one-second sleep and scheduling can delay observed rejection.

Primary runtime references: [STM 2.5.1.0](https://hackage-content.haskell.org/package/stm-2.5.1.0/docs/Control-Concurrent-STM.html)
and its cited PPoPP 2005 paper, Harris, Marlow, Peyton Jones and Herlihy,
[Composable Memory Transactions](https://www.microsoft.com/en-us/research/publication/composable-memory-transactions/).
STM was already a production dependency; this change pins that dependency and adds
it to the test component. No production service, toolchain architecture or market
data dependency is introduced.

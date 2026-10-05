# Immutable offline optimizer snapshots v2

Specified 2026-10-03 at efe88a73, before implementation. This is an engineering
repair track for CE-RL-016, not a financial experiment or production integration.
The frozen v1 learner, artifacts, registrations and counterexample remain intact.
The new module is a complete standalone optimizer/forward implementation; its
updates and forward passes must actually consume the new snapshot representation.
It does not replace the registered PPO/DQN/CQL runners or rehabilitate their results.

## Representation and semantics

`optimizer-snapshot-v2` stores one immutable snapshot S=(version,outputs,step,p,m,v).
outputs is 1 or 3; step is an integer in [0,2^31-1]. Each group is a tuple of four
native immutable byte strings encoding little-endian float64 tensors with shapes
(12,16),(16,),(16,outputs),(outputs,). All encoded values must be finite; variances
must be nonnegative. Derived NumPy views are temporary, not fields of S. Mutating
a view's shape cannot change S; the immutable bytes prevent writable data aliases.
Snapshots have frozen slotted fields and no public multi-field mutable facade.
Inputs are stable exact base float64 NumPy arrays: 1..256 batch rows, width 12 for
x and outputs for dz; finite lr in (0,1]. Reject unsupported representations.

create_v2, update_v2 and forward_v2 default to enabled=False and return None before
inspecting other arguments when disabled. Enabled creation is deterministic by
seed. All enabled entry points reject unsupported Python/NumPy versions or a disabled GIL. Update captures one old snapshot, stages all Adam fields from that snapshot
using the original equations/order, validates and freezes every candidate field,
and tries a nonblocking publication lock. Busy or stale expected snapshot returns
None; it does not wait or silently overwrite a concurrent update. Comparison is
object identity, preventing stale reuse; successful publication increments step
by exactly one and uses one `_state` assignment. Forward captures one snapshot
and uses that snapshot for every layer. No live/order, file, artifact, subprocess,
promotion, market-data, training-runner or deployment capability is introduced.

The linearization point is the single reference store under the publication lock.
A coherent snapshot read sees an old or new complete state, never a mixed tuple.
Exceptions before publication leave the old state. Interruption after publication
may leave the new state even if the caller receives no acknowledgement: do not
claim that every exceptional call is unchanged. The caller can inspect the step
and must not assume exactly-once retry or durable persistence. No restore operation
or in-place mutation is supported. No automatic retry is performed.

## Proof obligations

- F-RL-SNAPSHOT-PUBLISH: source-bound finite transition model with two writers,
  one reader and two calls per writer. Include capture, local stage success/failure,
  try-lock, identity comparison, commit, release, interruption before/after commit,
  and read. Every observed tuple has a single generation; commits require admitted
  candidates, exclusive ownership and the current expected generation. No stale
  update is committed, no mixed field state is reachable and steps equal commits.
- F-RL-SNAPSHOT-CAS: SMT over unbounded integer steps with cap 2^31-1 verifies the
  source-derived commit predicate/increment, stale rejection and monotone bounded
  steps. Successive states never recycle an expected generation. Source skeleton
  and bytecode store checks bind publication to implementation, not a bare model.
- F-RL-SNAPSHOT-FINITE: source-bound admission model and SMT float64 selector prove
  published slots are finite when the whole candidate scan succeeds. Pack structure,
  exact shapes and immutable byte publication are checked; NumPy scan/serialization
  semantics remain trusted. This is not a floating-point error or convergence proof.
- F-RL-SNAPSHOT-ISOLATION: enumerate the reviewed module import/call boundary; no file, network, process or order operations, no runner/production integration, default-disabled public entry points. Preserve prior capability/default closures by extending their required certificate sets.
- F-RL-SNAPSHOT-CONFORMANCE: deterministic real two-writer barrier schedules,
  snapshot readers, numerical/staging failures, stale publication, retry rejection,
  immutable views, defaults, actor/critic seeds 11/23/47, cold/warm parity and the
  original partial-store counterexample. Property tests are not formal proofs.

A-SNAPSHOT-V2: pinned ordinary CPython 3.13.3 with GIL, base NumPy 2.3.5, atomic
slotted reference reads/stores and lock primitive semantics, immutable native bytes,
frozen dataclass discipline and stable input buffers/module bindings. No malicious
reflection, custom dispatch, monkeypatching or interpreter/OS corruption. Reference
atomicity is an explicit runtime assumption, checked by source/bytecode/conformance,
not a proved compiler theorem. Solver trust follows the existing ledger.

Interruption may strand the lock in the acquire-to-finally handoff; subsequent
nonblocking writes fail closed while snapshot reads remain coherent. Model that
case rather than assume unconditional release. This does not prove recovery,
no-permanent-lockout, hard inference preemption, wall-clock termination, fairness,
server shutdown, production ownership or durable atomic writes. No broader mission
obligation closes solely from this component repair. Existing 35 blockers retain
applicable remaining work, with the optimizer sub-obligation updated truthfully.

## Verification and evidence

Enumerate the reachable fixed point and report actual states/edges/depth; no guessed
counts or ignored counterexamples. Independent SAT-premise/UNSAT-violation checks,
Z3 4.15.4, 10 s query bounds, random_seed=0; reject UNKNOWN. Audit exact source
structure, private effect boundary and default paths. Connect critical source,
canonical clauses, ledger, assumptions, tests, hashes, results and existing CI.

Test actual captured immutable snapshots through all update paths, not only a
standalone helper. Require finite, bitwise v1 parity on well-conditioned synthetic
updates and explicit refusal on overflow, invalid controls and stale writers.
Force two writers to prepare from one base with a barrier; require one success.
Use nonthrowing line/opcode observations for reference publication, and ordinary
injected staging exceptions. Do not rely on exceptions escaping opcode tracing.
Benchmark small synthetic updates and reads with pinned single-thread BLAS; timing
is empirical engineering evidence and never a formal bound. No new dependencies.

Primary semantics checked 2026-10-03: Python 3.13 lock objects and immutable bytes,
https://docs.python.org/3.13/library/threading.html#lock-objects and
https://docs.python.org/3.13/library/stdtypes.html#bytes ; NumPy buffer views,
https://numpy.org/doc/2.3/reference/generated/numpy.frombuffer.html . Current web
manuals describe the 3.13 series; actual verification runtime remains 3.13.3.


2026-10-04 composition: the separately registered PPO successor now calls create,
forward and update directly for its private actor and critic. The snapshot module,
frozen financial runner and production boundary are unchanged. The statement
above excluding runner integration applies to the original/frozen financial path;
the new engineering caller has its own source-bound verification contract.

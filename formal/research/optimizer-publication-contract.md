# Offline optimizer publication — specification before verification

Version `optimizer-publication-audit-v1`, 2026-09-30. Scope: unchanged
`Network.gradients` and `Network.update` in `sequential_learning.py`. Requirements:
functional/numerical correctness, failure behavior, publication/lifecycle and
concurrency limitations. No learner repair, training, model promotion or market data.

## Conflict and intended interpretation

A-SEQUENTIAL-RESEARCH-R25 currently says a failed update leaves all state unchanged.
Prior tests cover numerical/control failures before final publication. The source
stages candidates locally, then assigns `self.p, self.m, self.v, self.steps`.
Python assignment does not specify a transaction across multiple attributes.
The stronger all-failure/concurrent atomicity interpretation is not established.
Audit it rather than silently promoting existing regression evidence to a proof.

The intended narrow source contract is validation before publication, with ordinary
single-writer calls, no reentrant observers, stable base NumPy arrays/helpers and
successful final attribute stores. Numerical/control failure before publication
preserves old dictionary references, values and step counter, assuming helpers
have no hidden mutation. On normal return all four fields refer to the new state.
Any stronger atomicity obligation remains a blocker; narrowing the description of
existing evidence does not relax the promotion gate or prove concurrent safety.

## Formal model and proof obligations

- F-RL-OPTIMIZER-PUBLISH: audit the complete gradients/update ASTs. Model one call,
  four parameter keys, the ordered validation/staging cut points, and four ordered
  attribute stores. Each prepublication cut can succeed or reject. All stores must
  follow successful final finite-array admission. Rejection before publication
  preserves the old state; normal completion publishes all four fields. Prove
  finite-state progress to termination under terminating primitive/store assumptions.
  No multi-writer, runtime/compiler, arithmetic-helper or production refinement.
- F-RL-GRADIENT-CLIP: extract `grad/max(1.0,norm)` from the actual loop; for real
  grad and norm>=abs(grad), prove sign preservation, no magnitude increase and
  absolute result<=1. The source's rounded sqrt/reduction satisfying that premise
  is not proved. No floating-point norm, Adam step-size or convergence guarantee.
- F-RL-OPTIMIZER-ATOMIC: audit the stronger claim that every interrupted update is
  unchanged and every observation sees only the full old or full new state.
  Extend the model with one observer between attribute stores and an interruption
  after a partial prefix. Prescribed CE-RL-016: publish p, observe mixed state,
  interrupt before m. Preserve it as a model counterexample only unless a current
  Python regression witnesses the stated observation/interruption behavior.

The abstraction maps field references to old/new bits in order (p,m,v,steps),
with masks 0,1,3,7,15 along successful publication. Arrays' numeric content is
abstracted behind explicit finite-validation gates. Gradient helper purity is
checked structurally against the unchanged body but primitive non-mutation is
trusted. Floating error policy and validation helpers remain trusted. An error in
a helper with hidden side effects, MemoryError, signal, custom descriptor or
concurrent writer is outside the narrow safety theorem, not magically prevented.

## Implementation conformance and counterexamples

Use actual ordinary actor/critic Network objects and synthetic observations only.
Check control, gradient, norm and late candidate failures; old references, values,
counter and caller error policy must be preserved. Check normal publication against
stored pre-repair golden evidence. Audit source mutants for early writes, missing
finite gates, in-place helper mutation, altered clipping and publication ordering.

For the extended observation/interruption probe, use a bounded CPython opcode
trace of the unchanged update function: record masks before its STORE_ATTR events
and after return; inject one explicit exception before the second store. This
establishes instrumented/reentrant observability, not an unassisted thread race,
scheduling frequency, production reachability or historical occurrence. Restore
the caller's trace hook on success and failure. If the prescribed probe fails,
record it; do not silently replace it with a search or claim a counterexample.

Pinned Python 3.13.3, NumPy 2.3.5, Z3 4.15.4; no new dependencies. Positive SMT
queries need SAT premises and UNSAT violations, separate solvers, seed 0 and
10,000 ms limits. Source ASTs, model abstraction, CPython tracing and primitives
are trusted; exhaustive state checking and implementation conformance are distinct.
No proof placeholders or suppressed unknowns. Link ledger, canonical statements,
source hashes, tests, CI and HIGH/OPEN RL-OFFLINE-001. Broader atomicity/race
obligations remain open; no existing safety or empirical gate is weakened.

References checked 2026-09-30: Python 3.13 assignment semantics,
https://docs.python.org/3.13/reference/simple_stmts.html#assignment-statements ;
Kingma and Ba, https://arxiv.org/abs/1412.6980 ; Reddi et al.,
https://arxiv.org/abs/1904.09237 . Optimizer equations do not supply transaction
or unconditional convergence guarantees. No external code or PDF is committed.

## Execution-time instrumentation limitation

Python documents that exceptions escaping opcode traces may cause undefined interpreter behavior; injected interruption is a pinned-runtime diagnostic, not a portable implementation theorem. Normal-store observation does not require raising an exception. See https://docs.python.org/3.13/reference/datamodel.html#frame.f_trace_opcodes . The explicit interruption transition remains a model assumption; this limitation was discovered while checking primary documentation and does not change the registered witness or frozen learner.

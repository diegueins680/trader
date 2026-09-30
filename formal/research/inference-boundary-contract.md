# Inference boundary audit — specification before implementation

Version `inference-boundary-audit-v1`, 2026-09-30. Scope: unchanged `infer`,
`_finite_real_vector`, `ACTIONS` and `FEATURE_COUNT` in the offline research code.
Functional, numerical, safety, authorization and conditional-liveness audit only.
No policy repair, training, market data, holdout, new candidate or production adapter.

## Existing requirements and consistency

A-SEQUENTIAL-RESEARCH and existing tests already distinguish a **post-call**
20 ms admission check from a preemptive timeout. No implementation conflict is
claimed. Broad timeout/liveness acceptance requirements remain unfulfilled. This
adds machine-checkable evidence to that known distinction without redefining
safety, extending the time limit or installing cancellation infrastructure.

## Obligations

- F-RL-INFER-ADMISSION: for the audited source, returning a proposal implies exact
  enabled=True, a valid 12-element observation, valid three-element output, and a
  finite binary64 elapsed value in [0,20]. Extract the actual early/late rejection
  predicates. Prove under trusted predicate/runtime semantics using SMT; do not
  prove arbitrary NumPy subclass behavior or clock availability.
- F-RL-INFER-SELECTION: for three finite ordered real scores, first-maximum
  selection maps to the source's exact action constants [-1/4,0,1/4], selects a
  maximal score, prefers the lowest index on ties and yields bounded finite output.
  NumPy argmax semantics and stable arrays are assumptions; this is not neural
  robustness, economic correctness, or order authorization.
- F-RL-INFER-PATH: source-bound state model with one call and flags for enable,
  observation admission, call start, output admission and timing admission.
  A rejected early gate prevents the call; proposal requires all gates; no model
  action authorizes orders. The model has an explicit pending self-loop.
- F-RL-INFER-DEADLINE: refute the stronger claim that this wrapper always terminates
  within 20 ms. Preserve CE-RL-017, the reachable pending-call lasso; it has no
  timeout/cancellation transition. A synthetic clock probe must show the second
  clock read occurs after forward returns and elapsed=25 ms yields absence.
  This is a logical timing fixture, not a measured real-time latency experiment.

## Abstraction and assumptions

State = (phase, flags), one call, no concurrent mutation. Flags represent enabled,
valid observation, call start, valid output and timely result. Source AST matching
binds control flow, exception class, defaults, widths and return selection.
Full Python execution/refinement is not proved. Ordinary base numeric NumPy
arrays, finite predicate behavior, first-argmax semantics, stable constants and
Python/solver/runtime behavior are trusted. A terminating `Exception` becomes
absence; `BaseException`, clock errors, predicate errors, memory exhaustion and
nonreturning primitives are outside that catch. Successful clock reads return
integer nanoseconds; conversion and timestamp authenticity are not proved by a
proof about the final elapsed scalar. No formal neural-network input-region
certificate is introduced. No real-time scheduler or preemption guarantee.

## Registered verification design

Pinned Python 3.13.3, NumPy 2.3.5, Z3 4.15.4; existing 10,000 ms solver limit,
seed 0, independent SAT-premise and UNSAT-violation queries. No new dependencies.
Check source mutants, model/fixture mutations, gate conformance, all three-score
vectors from {-1,0,1} at elapsed nanoseconds {-1,0,19999999,20000000,20000001,25000000},
representation failures (missing, masked, wrong shape/type, NaN/infinity), ordinary
forward exceptions, cancellation propagation, and deterministic clock-call order.
The 162 score/timing cases and bounded model are exhaustive only over their stated
finite domains. Conformance tests supplement rather than prove runtime refinement.

Link canonical requirements, ledger, risk RL-OFFLINE-001, hashes, test sources and
formal/full CI wrappers. Acceptance remains blocked. Preserve every failed probe
and do not change input domains or time thresholds to obtain passing evidence.

## Primary-source context

Alshiekh et al., Safe Reinforcement Learning via Shielding:
https://arxiv.org/abs/1708.08611 . Action shielding does not itself establish
scheduler deadlines. Python clock semantics:
https://docs.python.org/3.13/library/time.html#time.perf_counter_ns . A measurement
primitive is not a cancellation primitive. Exact source and timing assumptions
must remain distinct from policy quality or market safety.

## Source-order clarification

The second clock read precedes output validation and argmax. Its duration excludes those later operations; even a promptly returning forward call does not yield an end-to-end wrapper deadline proof. Valid equal scores retain first-index selection (short exposure), not a new uncertainty/abstention rule. Invalid representations still reject under the stated predicate assumptions.

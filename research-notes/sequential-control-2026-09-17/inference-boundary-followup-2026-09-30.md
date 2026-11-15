# Inference admission and deadline audit — 2026-09-30

**Decision: no adoption; continue offline assurance.** This audit adds evidence
about unchanged inference code. No market archive, final holdout, policy training,
OPE rerun, deployment or production configuration is touched. Existing rejected
policies remain rejected and the champion is preserved. All 38 whole-system
obligations remain open or partially verified.

Specification/registration commit: `1d985603`. Source/verification freeze:
`8603f5aec7fe0c5ab7b93c720c9f9bd8a1c8be0f`. Latest main checked at the start remains
`dbd45e2691cb37f1421a23306c43b676fa82e6fc`; this work continues on the existing
dedicated branch and draft PR #284, stacked on #281. Open issues were empty;
other open research/dependency PRs do not supply a timeout implementation or
justify changing this frozen experiment.

## Canonical interpretation

Existing A-SEQUENTIAL-RESEARCH clauses and tests correctly describe a post-call
20 ms gate. No contradiction or newly repaired production defect is claimed.
The [specification](../../formal/research/inference-boundary-contract.md) and
[registration](../registrations/inference-boundary-audit-engineering.json) add
machine-checkable guard evidence and explicitly retain the unmet deadline claim.

The wrapper reads a clock, admits an exact enabled flag and a valid observation,
calls `forward`, catches ordinary `Exception`, reads the clock again, validates
the output, checks measured duration and selects the first maximal score. The
second timestamp precedes output validation and selection. Therefore measured
elapsed time is neither an interrupt mechanism nor complete wrapper duration.
[Python's clock documentation](https://docs.python.org/3.13/library/time.html#time.perf_counter_ns)
describes an integer nanosecond measurement; it provides no cancellation guarantee.

The source helper admits ordinary real, unmasked, finite vectors of widths 12
and 3. This is representation admission, not proof of fresh observations, correct
symbol scope, supported states, truthful provenance or calibrated uncertainty.
Those requirements remain separate. Exact Boolean opt-in remains required.

## Formal models and results

| Requirement | Status | Scope |
|---|---|---|
| F-RL-INFER-ADMISSION | smt_verified | Actual early/late predicates are equivalent to exact enabled=True, admitted observation/output and finite measured binary64 elapsed in [0,20]. All Boolean gate combinations and binary64 elapsed values. |
| F-RL-INFER-SELECTION | smt_verified | Three finite scores represented as exact reals; first-maximum selection picks a maximal score, lowest index on ties and an action in {-1/4,0,1/4}. |
| F-RL-INFER-PATH | model_checked | One-call model, five flag bits; 9 reachable states, 15 transitions, maximum shortest depth 4. Early rejection prevents the call; a proposal requires all gates. No order-authorizing transition exists in the model. |
| F-RL-INFER-DEADLINE | refuted | CE-RL-017 reaches the pending-call state and can remain there forever. No cancellation/deadline transition exists in this abstraction. |

Both SMT obligations use separate SAT-premise and UNSAT-violation queries, seed 0
and the unchanged 10,000 ms limit. Full AST skeleton matching binds the helper,
wrapper, exception class, enabled default, widths, return selection and source
constants. The guard expression is translated; NumPy first-argmax behavior is a
trusted semantic premise, checked by conformance. No whole-program compiler or
runtime refinement is claimed.

[NumPy's first-maximum convention](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)
explains the existing tie behavior. Equal valid scores select index 0, hence short
exposure. This is not an absence/uncertainty rule, and the audit adds none. It does
not imply missing or invalid observations should be imputed to equal scores.
A future abstention change would require its own version and empirical evaluation.

The state is `(phase, flags)`, with bits for enabled, valid observation, call
started, valid output and timely result. The model abstracts predicate results and
clock arithmetic. Its output-admission phase follows a completed call; the clock
read is collapsed, not assigned zero real cost. Normal primitive termination is
not silently assumed to discharge the deadline obligation.

## Counterexample and implementation conformance

[CE-RL-017](../../formal/research/inference-counterexamples.json) has prefix
`enable; admit_observation` followed by the repeating `pending` transition. This
is a reachable abstract lasso, not evidence that a registered Network invocation
actually ran forever. It explains why an unconditional termination claim cannot
follow from the wrapper without extra assumptions or infrastructure.

The implementation fixture substitutes two clock readings, 0 and 25,000,000 ns,
and records this exact event order:

```text
clock -> observation validation -> forward entry -> forward return
      -> clock -> output validation -> absent proposal
```

The returned measured duration is 25 ms and the proposal is absent. This is a
logical-clock regression, not a measured latency benchmark or a real-time timeout
test. No hanging thread or background task is created.

Seven added tests cover:

- Nine source mutants and two environment-constant mutants, including altered
  thresholds, default opt-in, output width, exception class and argmax selection.
- Four witness mutations and two model mutations (gate bypass and deleted pending cycle).
- All 162 registered score/timing cases: 27 vectors from {-1,0,1} cubed, each at
  -1, 0, 19,999,999, 20,000,000, 20,000,001 and 25,000,000 ns.
- Six disabled/non-exact flags and eleven invalid representations each for
  observations and outputs; early rejection suppresses the forward call.
- The prescribed late-call order, ordinary exception fallback, BaseException
  propagation, and three unchanged Network inference/parity fixtures.

The helper source and primitive semantics are trusted; tests do not prove every
NumPy execution. Clock/predicate failures and `BaseException` are outside the
ordinary forward-error catch. The elapsed theorem starts with a binary64 scalar;
it does not prove the integer-to-float clock conversion or clock authenticity.
Arbitrary subclasses, concurrent mutation, scheduler availability, resource
exhaustion, neural robustness and policy quality remain outside the proof scope.

## Focused shielding literature update

The [structured update](inference-shielding-paper-update-2026-09-30.csv) supplements
the existing literature matrix; it does not create another candidate family.
No paper code, datasets or PDFs are copied. No market efficacy is inferred.

[Alshiekh et al.](https://arxiv.org/abs/1708.08611) derives action restrictions from
an environment abstraction and temporal safety specification. Its preemptive
shield terminology concerns filtering actions before policy choice, not CPU
preemption of a stalled policy. The relevant lesson is to state the abstraction
assumptions; its control experiments do not certify this wrapper's timing.

[Kwon et al., revised January 2026](https://arxiv.org/html/2506.11033v2), adapts a
shield using inferred dynamics and conformal uncertainty. Its argument requires
calibrated errors and Lipschitz relationships; the reviewed protocol keeps hidden
parameters fixed within an episode and uses an initial calibration period. Neither
these premises nor a market-safe fallback are established here. Disposition:
monitor; no learned shield replaces deterministic limits.

[Georgescu et al., revised August 2026](https://arxiv.org/html/2511.02605v3), uses
GR(1) specification repair to expose assumption failures and restore realizability.
In Seaquest it preserves oxygen safety while weakening infeasible reachability
goals. This is not a claim that the paper arbitrarily discards safety. The useful
lesson here is explicit liveness assumptions. Automatic rewriting of this task's
trading constraints, promotion gates or live limits remains prohibited; no repair
engine is adopted.

These were focused mechanism/assumption reviews, not independent replications or
an exhaustive review of all work through September 30. The 2026 revisions were
verified against primary sources, replacing stale search-result titles where
necessary. No new algorithm, tuning budget or economic test is justified by them.

## Traceability and reproduction

The four canonical clauses map to proof-ledger entries, assumption
A-INFERENCE-BOUNDARY, the checker, frozen implementation, tests, fixture and
formal/full CI wrappers. Relevant finite/bounded/fallback/default obligations gain
conditional evidence without changing their partial status. Shutdown remains open:
this inference model does not model production server shutdown or recovery.
RL-OFFLINE-001 remains HIGH/OPEN. All 61 locked source hashes must match.

Use the [formal runbook](../../formal/research/README.md), pinned dependencies and
`TRADER_FORMAL_PYTHON`. No new dependency or tool is introduced; proof checks run
without network after installation.

```sh
python scripts/formal/test_integrity.py InferenceBoundaryTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
python scripts/formal/verify.py --require-complete
```

The final command must continue to reject acceptance while obligations remain
open. Exact commands, versions, exits, logs and timings are recorded in the
[machine-readable receipt](inference-boundary-evidence-receipt-2026-09-30.json).
No return, drawdown, tail-risk, transaction-cost, OPE or inference-performance
improvement is claimed. Existing financial conclusions and sealed holdouts remain
unchanged. General recommendation: no candidate passed. RL recommendation:
continue offline assurance, with no production integration or live authorization.

## Verification receipt

At source freeze `8603f5ae`:

| Command | Actual result |
|---|---|
| `python scripts/formal/test_integrity.py InferenceBoundaryTests` | Exit 0; seven tests in 1.137 seconds. |
| `bash scripts/verify.sh formal` | Exit 0; 80 integrity tests, 29 scoped SMT obligations, new and existing model/conformance checks. |
| `bash scripts/verify.sh full` | Exit 0 on its first attempt; formal, Haskell build/format/lint/smoke/tests, web typecheck/241 tests/build and 185 automation tests; none skipped. |
| `python scripts/formal/verify.py --require-complete` | Exit 1: `research acceptance blocked by open obligations`; expected refusal, not acceptance. |

The standalone formal integrity/certificate stages took 14.146/6.777 seconds;
the corresponding full-run stages took 19.081/9.261 seconds. These are observed
verification timings, not an inference latency claim. No test, domain, timeout or
promotion gate was relaxed. The initial record preceded an explicit discrete-action
membership assertion; both record attempts and the final checks are retained.

All four remote verification jobs passed at the same source freeze in
[run 36699627650](https://github.com/diegueins680/trader/actions/runs/36699627650).
Docker build and deployment were skipped. The receipt binds the run's exact head
SHA to the source freeze. Later documentation/receipt edits change no executable
source; these full-run/CI results apply to `8603f5ae`, not implicitly to later heads.
No proof placeholder remains in checked proof sources. Open obligations remain
explicit and the PR stays draft, unmerged and blocked for acceptance.

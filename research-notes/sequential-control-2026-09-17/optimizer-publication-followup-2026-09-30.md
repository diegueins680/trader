# Optimizer publication audit — 2026-09-30

**Decision: no adoption; continue offline assurance.** This continuation checks
unchanged optimizer code. It does not train policies, read market archives, open a
holdout, repair the learner, replace the champion or change production authority.
All 38 mission obligations remain open or partially verified. The draft is not
ready for production integration or a declaration of mission completion.

## Specification and consistency resolution

The [registration](../registrations/optimizer-publication-audit-engineering.json)
and [contract](../../formal/research/optimizer-publication-contract.md) were
committed as `8e8eff8a` before implementing this checker. Source/verification freeze:
`c67125085e0cd55d2a00371d73730babeb1e5e47`, on main ancestry
`dbd45e2691cb37f1421a23306c43b676fa82e6fc`. The contract subsequently records a
primary-documentation limitation on the prescribed tracing method; the witness,
learner, clipping formula and promotion gates were not changed.

A-SEQUENTIAL-RESEARCH-R25 and the September 18 repair report described failed
updates as preserving state. The existing tests exercised failures **before final
publication**. Inspection shows four sequential attribute stores for `p`, `m`,
`v`, and `steps`; evaluating the right-hand tuple first does not make those stores
a transaction. The canonical description now states the demonstrated narrow
behavior and preserves stronger atomicity as an unresolved acceptance requirement.
This resolves an evidence/wording conflict; it does not waive the requirement.
[Python assignment semantics](https://docs.python.org/3.13/reference/simple_stmts.html#assignment-statements)
defines ordered target assignment and permits attribute assignment failure.

The abstraction represents each field as old/new and maps publication to bitmasks
0, 1, 3, 7, 15. Numeric arrays are abstracted behind validation gates. Complete
`gradients` and `update` AST skeletons are checked; only the extracted clipping
expression is translated to real arithmetic. Primitive non-mutation, shape/key
admission, termination and runtime behavior are assumptions, not derived theorems.

## Evidence and limits

| Requirement | Status | Exact scope and result |
|---|---|---|
| F-RL-OPTIMIZER-PUBLISH | model_checked | One call, four parameter keys, 32 staging cuts, four stores; 69 reachable states, 101 transitions, maximum shortest depth 36, progress rank bound 72. Prepublication rejection retains old fields; normal completion publishes all fields. |
| F-RL-GRADIENT-CLIP | smt_verified | Over exact reals with norm >= abs(grad), extracted grad/max(1,norm) preserves sign, does not increase magnitude and has absolute result <=1. One SAT-premise check and independent UNSAT-violation query; seed 0, 10,000 ms limit. |
| F-RL-OPTIMIZER-ATOMIC | refuted | Explicit observer/interruption extension: 159 states, 271 transitions, depth 37, rank bound 73. CE-RL-016 stores p, observes mask 1 and interrupts before m. Stronger all-failure/all-observation atomicity fails in this model. |

The model's successful uninterrupted final stores are an explicit assumption.
The observer extension contains at most one observation before completion and
interruptions after a partial prefix. It observes masks 0,1,3,7; the regression
also records mask 15 after normal return. These are exhaustive reachable cases
of the finite model, not exhaustive interpreter schedules or production states.
Progress excludes environmental nontermination and arbitrary idle stuttering.

The real clipping theorem does not establish that the rounded sum/sqrt norm meets
its premise. It bounds neither the Adam parameter update nor economic exposure.
IEEE error bounds, gradient semantics for every input representation, multi-writer
coordination and full source-to-runtime refinement remain open.

Seven added integrity tests cover:

- Eight source mutants, three fixture mutations and an invalid progress transition.
- Twelve actor/critic, cold/warm helper-purity cases across seeds 11,23,47.
- Sixty failure cases preserving references, values, counter and caller error policy:
  invalid learning rate/counter, non-finite gradient, overflowing norm and late
  invalid candidate arithmetic.
- Twenty-four opcode-instrumented calls: twelve normal publications and twelve
  deliberately interrupted publications; trace hooks and caller flags are restored.
- Twenty-one exact-real clipping grid comparisons and the existing three-update
  Adam golden fixture. This is engineering parity, not policy training.

These are deterministic conformance/regression tests, not formal proofs of all
NumPy or interpreter executions. The unchanged learner and production Haskell
code receive no patch. No learned artifact or large dataset is committed.

## Counterexample and unsuccessful probe record

CE-RL-016 is preserved in the [fixture](../../formal/research/optimizer-counterexamples.json),
model receipt and actual pinned CPython regression. Normal instrumented execution
observes masks 0,1,3,7,15. An exception injected before storing `m` leaves mask 1:
new parameters, old moments and counter. No unassisted thread race, occurrence
frequency, registered-training reachability or historical loss is established.

The first standalone normal trace observed only `[15]`; its interruption probe
observed `[0,1]`. That attempt was incomplete, despite a zero process exit. A
second caller-flag configuration still missed normal events and failed its Python
assertion. Explicitly installing the target frame's local trace hook before
requesting opcode events reproduced both prescribed sequences. Failed/incomplete
attempts are retained by external-log hashes in the evidence receipt; no seed,
input domain or optimizer formula was selected around the outcome.

[Python tracing documentation](https://docs.python.org/3.13/library/sys.html#sys.settrace)
describes explicit local hooks and platform-specific opcode events. Crucially,
[frame documentation](https://docs.python.org/3.13/reference/datamodel.html#frame.f_trace_opcodes)
warns that escaping exceptions from opcode tracing may cause undefined interpreter
behavior. The injected failure is therefore a **pinned-runtime diagnostic**, not
a portable implementation theorem. Normal observation uses no injected exception.
The model's interruption semantics are explicit assumptions; broader implementation
atomicity remains open. This limitation was recorded rather than hiding the probe.

## Focused literature update

This supplements the existing review and [paper matrix](optimizer-paper-update-2026-09-30.csv).
It does not expand the candidate shortlist or justify optimizer replacement.

[Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980) supplies the adaptive first/
second-moment and bias-correction mechanism. The repository uses those mechanisms
with global gradient clipping and fixed coefficients; it is not a reproduction of
the paper's empirical benchmark. Formula fidelity says nothing about atomic stores.

[Reddi, Kale and Kumar](https://research.google/pubs/on-the-convergence-of-adam-and-beyond/)
show a convex optimization failure and motivate AMSGrad's longer memory. This is
negative evidence against an unconditional convergence claim, not a trading
comparison or automatic reason to replace a frozen optimizer.

[Heilman and Mohanty (July 2026)](https://arxiv.org/abs/2607.03519) reports a projected
online-optimization counterexample across arbitrary moment-decay parameters.
Only primary metadata/abstract was screened here; no theorem was independently
reproved or machine-checked. Disposition: monitor.

[Dereich, Do and Jentzen (March 2026)](https://arxiv.org/abs/2603.18899) reports
bounds/error analysis for a class of strongly convex stochastic problems. This was
screened through the indexed primary abstract; direct opening failed. Different
problem assumptions explain why positive restricted-class theory need not conflict
with online counterexamples. Those assumptions are not established for this neural
RL learner. Disposition: monitor, not a transferred convergence certificate.

No source supplies independent cryptocurrency replication, realistic trading costs,
causal validation or production safety for this repository. No paper code, dataset
or PDF was copied. Uninspected code/data licenses remain explicitly unverified.

## Reproduction, traceability and recommendation

Use the pinned Python 3.13.3 / NumPy 2.3.5 / Z3 4.15.4 environment and existing
GHC 9.4.8, Cabal 3.12.1.0, Node 20.19.0, fourmolu 0.15.0.0 and hlint 3.8.
Use the [formal toolchain runbook](../../formal/research/README.md) to install the pinned environment; set `TRADER_FORMAL_PYTHON` to its interpreter for wrapper runs. After dependency installation the proof checks require no network. No new tool
or dependency was introduced.

```sh
python scripts/formal/test_integrity.py OptimizerPublicationTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
python scripts/formal/verify.py --require-complete
```

The last command is an acceptance guard: expected exit 1 while obligations remain
open, not a successful acceptance result. The [machine-readable receipt](optimizer-publication-evidence-receipt-2026-09-30.json)
records actual commands, exit codes, logs, versions and timings. Scoped checker
success is distinct from candidate acceptance.

Canonical clauses map to the three ledger entries, assumption
A-OPTIMIZER-PUBLICATION, source checker, unchanged learner, integrity tests and
formal/full CI wrappers. Source hashes bind models, registration, fixture and the
existing golden fixture. Mission atomicity/race entries remain open and link this
negative evidence. `RL-OFFLINE-001` stays HIGH/OPEN.

No new return, drawdown, tail-risk, OPE, cost/delay, inference or policy-training
benchmark is asserted. Previously rejected, contaminated-development results are
unchanged; the holdout remains sealed. General recommendation: no candidate passed.
RL recommendation: continue offline assurance; do not integrate or authorize orders.
A future repair needs a separately specified immutable-state publication boundary
and its own interruption/concurrency proof; this audit does not silently implement it.

## Verification receipt

At source freeze `c6712508`:

| Command | Actual result |
|---|---|
| `python scripts/formal/test_integrity.py OptimizerPublicationTests` | Exit 0; seven tests, 0.915 seconds. |
| `bash scripts/verify.sh formal` | Exit 0; 73 integrity tests, 27 scoped SMT obligations, existing and new model/conformance checks. |
| `bash scripts/verify.sh full` | Exit 0 on its first attempt; formal, Haskell build/format/lint/smoke/tests, web typecheck/241 tests/build, 185 automation tests; none skipped. |
| `python scripts/formal/verify.py --require-complete` | Exit 1 with `research acceptance blocked by open obligations`; expected refusal, not acceptance. |

The final standalone formal run took 16.431 seconds for integrity tests and 5.862
seconds for the certificate driver. In the full run these took 14.289 and 6.813
seconds. These are single-run verification measurements, not an inference latency
or throughput claim. All 57 locked source hashes match.

All four remote verification jobs passed at `c6712508` in
[GitHub Actions run 36655374126](https://github.com/diegueins680/trader/actions/runs/36655374126).
Docker build and deployment were skipped. The earlier report-only revision
`5d724155` also passed all four jobs in run 36653964635. Later receipt/report edits
change no executable source; the stated full-run and remote evidence apply to
`c6712508`, not implicitly to a later commit. PR #284 remains draft and unmerged.

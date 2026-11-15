# Replay ordering cutoff — 2026-10-01

**Decision: retain scoped assurance evidence; no candidate adoption.** This
continuation extends the previous single-call ordering model beyond its six
remaining bars. It adds no training, market-data access, financial experiment,
OPE rerun, protected-outcome access or production behavior. The earlier model,
registration, simulator and financial evidence remain unchanged.

The [contract](../../formal/research/replay-cutoff-contract.md) and
[engineering registration](../registrations/replay-cutoff-audit-engineering.json)
were committed as `ea52ed52` before implementation or probes. Starting head was
`1ded9950d4633008b0b97e301752086145a3e086`; its
[CI](https://github.com/diegueins680/trader/actions/runs/36804846001) passed.
Latest fetched main remained `dbd45e2691cb37f1421a23306c43b676fa82e6fc`.
The work remains on `research/sequential-review-2026-09-28`, draft PR #284,
stacked on #281. No merge or deployment is included.

## Cutoff and proof scope

Let R be remaining bars, H in {1,3,6}, E=min(R,H), C=min(R,7), and 0<=t<=E.
The integer SMT lemma establishes E=min(C,H), t<=E<=C<=R and
(t=R) iff (t=C). It also establishes equivalence of the terminal predicate
`failed or t == limit`, extracted from the pinned ordering model. R has no
finite upper bound in this lemma. Separate SAT-premise and UNSAT-violation
queries prevent vacuity; unexpected SAT, UNKNOWN or timeout fails verification.

The projection changes only the ordering-state limit to C. Its source-bound
successor function reads limit only at that terminal predicate and otherwise
preserves it through dataclass replacement. Thus the stated mathematical
simulation argument depends on equal end indices, equal branch results and
unchanged remaining fields. The source dependency check, integer lemma and
finite lifted comparisons support that argument. They do not mechanically
refine the full Python interpreter or simulator.

[Python's dataclass documentation](https://docs.python.org/3.13/library/dataclasses.html#dataclasses.replace)
describes replacement through construction of another instance and cautions about
post-initialization and init-disabled fields. The pinned model uses ordinary
integer/Boolean/string fields and no such hooks. Python/dataclass semantics remain
named trusted assumptions, not a theorem about arbitrary classes. The runtime
remains pinned to Python 3.13.3.

A six-bar cutoff is unsound for ordering: with R=7,H=6,t=6,failed=False, the
original call remains nonterminal while a capped limit of six makes it terminal.
This was a **preregistered bad-abstraction witness**, not a discovered simulator
bug. The seven-class model correctly permits finishing a six-bar call while
retaining simulated inventory for the next call.

| Requirement | Status | Evidence |
|---|---|---|
| F-RL-REPLAY-CUTOFF | smt_verified | Integer cutoff and extracted terminal-predicate equivalence; arbitrary R>=1, three horizons and Boolean failure. |
| F-RL-REPLAY-QUOTIENT | model_checked | Complete finite ordering graph for classes 1..7; the runtime connection remains bounded trace conformance. |

The expanded graph has **3,596 states, 4,340 transitions and 102 distinct initial
states**. Maximum shortest depth and longest nonstuttering path are both **48**.
Final states stutter; nonterminal cycles, stutters and deadlocks are rejected.
Delays are 0/1 and initial pending offsets absent/1/2. All earlier admission,
mark/risk/fill/terminal/row invariants are reused without modifying their source.

The class-seven subgraph has **720 reachable states**. All have labeled successors
compared after lifting to R in {7,8,12,97,2^63}, then projecting back: **3,600
comparisons**. These are exhaustively checked finite cases, not the proof of the
unbounded integer lemma. The old 2,876-state certificate remains separately
reproducible with its original scope.

**The quotient is only about ordering.** Actual time-to-end observations depend
on R. Rewards, risk outcomes and policy actions may differ. Observation/data
validity and risk are nondeterministic in the ordering abstraction. No claim is
made about observation/reward bisimulation, multi-call state refinement, market
realism, floating-point accounting, exceptions, concurrent writers, real-time
deadlines or production Haskell behavior. Helper termination and no hidden
side effects are assumptions. The final node can reach a failing reward check;
early rejected and insolvent paths can retain simulated state.

## Conformance and the checker counterexample

The registered synthetic product grid has **486 actual replay traces**: three
targets, three horizons, two delays, remaining lengths {7,8,12}, three initial
inventories and three next-bar returns. Normalization uses the causal synthetic
prefix. Eight further named scenarios cover ordinary operation, invalid gate,
invalid market, solvent risk, insolvency, inherited pending target, partial fill
and missed fill. Actual original helpers perform the arithmetic; instrumentation
records marking, target attempts, liquidation attempts and row publication.
The earlier numeric checks use 1e-13 tolerance and remain tests, not proofs.

The first six-test run had **one failure**. A deliberate mutation changed the
ordinary six-bar call's returned done=False to done=True while retaining
failure=None and its visible trace. The initial matcher accepted this by choosing
an abstract insolvency path that skips liquidation. This exposed insufficient
conformance checking, not an execution defect. The new matcher now requires a
clear abstract risk tag when the actual replay reports no failure. The mutation
is rejected, and all six tests passed on rerun.

The [fixture registry](../../formal/research/replay-cutoff-fixtures.json) preserves
both the bad six-cap abstraction and `CUT-MATCH-001`, the resolved development
checker witness. The regression remains executable. No earlier certificate or
frozen research source was silently rewritten. Failure traces still admit
nondeterministic risk categories, and the stronger matcher still does not prove
universal implementation refinement.

Other tests reject changed limit dependencies even after rehashing, state-copy
or limit-mutation changes, a limit-dependent successor mutant, UNKNOWN solver
results and registration drift. The source hash is a drift guard, not a semantic
proof. Every new critical checker maps to canonical requirements, assumptions,
results, implementation, tests and CI. `RL-OFFLINE-001` remains HIGH/OPEN and all
38 broader mission obligations remain open or partial.

## Reproduction

Use the existing pinned [formal toolchain](../../formal/research/README.md):

```sh
python scripts/formal/test_integrity.py ReplayCutoffTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
python scripts/formal/verify.py --require-complete
```

At source freeze `05d0444e1069c9e3ee88df924fafaedef13b12f2`, both canonical
wrappers exited **0**. Formal verification passed **117 integrity tests** and
**38 scoped SMT requirements**; the full wrapper also passed Haskell
format/lint/build/smoke/tests, **241 web tests** and **185 automation tests**.
The existing web bundle-size advisory remains. No test or assertion was disabled.
[Implementation CI](https://github.com/diegueins680/trader/actions/runs/36806418749)
passed all four verification jobs; Docker/deployment were skipped. Subsequent
changes record documentation/evidence only.

The [receipt](replay-cutoff-evidence-receipt-2026-10-01.json) records commands,
exit codes, external log hashes, tool versions, model results, the initial failed
test and final verification timings. The targeted rerun took 12.768 seconds.
The standalone formal wrapper's integrity tests/certificate took 65.709/21.402
seconds; within full they took 50.433/14.650 seconds on the shared local host.
These are verification measurements, not policy inference or production deadline
benchmarks. There are 82 source locks; all prior certificates reproduced unchanged.

`python scripts/formal/verify.py --require-complete` exited **1** with
`ValueError: research acceptance blocked by open obligations`. This is an expected
acceptance refusal, not a passing acceptance check. There are no hidden proof
placeholders, and no claim of full-system verification or integration readiness.

General recommendation: **no candidate passed**. RL recommendation: **continue
offline research**, while retaining rejection of the frozen tested configurations
for integration. No new out-of-sample or final-holdout evidence, statistical
significance, cost robustness, drawdown/tail advantage or matched-champion result
is claimed. Protected periods remain sealed. No live exploration, authenticated
trading endpoint, order, policy promotion, production learning, fleet, ownership,
leverage, margin, exposure-cap, deployment or champion change occurred.

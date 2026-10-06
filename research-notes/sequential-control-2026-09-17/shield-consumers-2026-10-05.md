# Shield-to-consumer composition — 2026-10-05

Engineering registration `b618cf9b` precedes implementation on merged main
`4894a6d813bad18b5881d40b51b7a835848df316`. No runtime source, training rule,
configuration, artifact schema, research evidence or financial trial changes.

## Requirement and source findings

Obligation 12 keeps its original scope and criterion: "Every consumed action
passes the deterministic shield and cannot race around it." A guard theorem and
single-step ordering model did not establish complete consumer coverage or origin
of a pending target inherited from an earlier call. This work supplies that
composition rather than changing the simulator or weakening the criterion.

The complete 12-module research inventory contains four Replay constructions,
four step calls and two fill-helper calls. All 89 uses of local replay receivers
are checked. Collect, replay_policy and short_ope own their local instances; the
latter constructs distinct behavior/direct-replay instances. Callbacks receive
observations, not replay objects. The returned evaluation instance is read only
for reporting. No private fill helper escapes to another consumer. No concurrent
worker receives a replay receiver. This is a source/effect statement under the
registered primitive contracts, not a general thread-safety claim for Python.

The actual step entry first checks termination, invokes shield, checks observation
availability, then returns on rejection before scheduling or filling. Pending
state begins absent; only accepted proposals can populate it. A later accepted
proposal cannot replace an occupied slot. A due target is consumed only through
the current call's entry gates and the due/risk branch. Terminal liquidation is
the deterministic fixed-zero branch, following pending cancellation and solvency
admission; it is not a new policy decision with future price information. This
interpretation already appears in replay-order-contract.md.

Rejected calls can retain simulated inventory or pending data in an incomplete
failure. That is not successful liquidation or rollback. The current work makes
no numeric, solvency, complete-hard-constraint, recovery or production-race claim.

## Proofs and conformance

- F-RL-SHIELD-CONSUMERS: complete finite source/use inventory plus actual guard,
  target, pending-transfer and receiver-ownership checks. Hashes bind helper
  effects; hashes alone are not the argument.
- F-RL-SHIELD-ORIGIN: seven satisfiable-premise/UNSAT queries, using the action
  domain extracted from the actual shield. Induction preserves absent-or-shielded
  pending state and old target identity, rejected calls cannot acquire a target,
  consumption requires current gates, terminal zero is risk-selected, and a write
  to one instance identity preserves every distinct identity. SMT real quarter
  targets are exactly representable; no real-to-binary64 accounting claim follows.
- F-RL-SHIELD-FLOW: 129 local reachable states, 213 transitions, maximum shortest
  depth 10; factored independent two-instance product 16,641 states and 54,954
  transitions. Three actions, absent/present pending state and residual delays
  0/1/2. Repeated calls and fresh instances reach a fixed point without a call-depth
  cutoff. Bar count/risk/solvency are nondeterministic abstractions. This is safety,
  not liveness or a production concurrency theorem.
- F-RL-SHIELD-CONSUMER-CONFORMANCE: 28 actual scenarios, 187 steps, 149 ordinary
  and 29 terminal helper calls. Visible shield/fill events and concrete successor
  pending/done states must match model traces. Tests cover all actions/horizons/
  delays, invalid/disabled later calls, independent interleaving, and all actual
  collector/replay/OPE callers. Sixty-four generated invalid proposals preserve
  pending state without consuming it. Synthetic Network initialization is for OPE
  routing only; no training or market-data read occurs.

Tests omit each inventory category, mutate bypasses even with updated inventories,
inject unsafe model transitions, reject unsatisfiable SMT premises and remove each
closure constituent. No new runtime counterexample was found; injected mutants
are tests of the verifier, not evidence of observed trading failures.

## Assumptions and closure

A-SHIELD-COMPOSITION names pinned Python/NumPy ordinary-object, identity, sequential
control-flow, array-copy and primitive-effect semantics; fixed registered callbacks
and complete source. Receiver locality is checked, not assumed from the word
"offline". Hostile subclasses/monkeypatching, reflective caller mutation,
asynchronously injected callbacks and external private-helper calls are excluded.
The source/model correspondence is reviewed and conformance-tested, not a verified
compiler. Actual deployed images, inherited live-authority correctness, truthful
risk evidence and numeric accounting remain unproved.

Conditional closure 12 additionally requires freshly reproduced Haskell proposal
bounds/guards, process admission/lifecycle, bridge codec/flow, complete callback
coverage and exclusion from all six production executables. No future shared or
production consumer inherits this certificate automatically. All original 38
scope/criterion strings remain unchanged. Only obligation12 changes status.

After fresh canonical reproduction, current totals are **11 scoped closures,
23 partial, 4 open**. Open4/10/37/38 and every other partial obligation retain
their existing requirements. Overall mission and formal completion remain false.

## Research, verification and delivery

No candidate passed; RL remains offline research. Frozen 108 fits, 19,440 replays
and 19,548 registry rows remain contaminated development, all 108 OPE batches
remain invalid, and 1,227 final returns remain sealed. Prospective embargo remains
2027-01-20T13:00Z. No new economic result, cost/delay stress, statistical inference,
policy comparison or inference benchmark is asserted.

No production code, live authorization, risk limit, fleet, leverage, ownership,
deployment setting or champion state changes. No authenticated trading endpoint,
order, live exploration, deployment or holdout access. The original user worktree
and running services remain untouched. Proof tool versions/dependencies unchanged.

Targeted ShieldConsumerTests plus closure regressions: 12 tests passed locally
in 29.133 seconds. Canonical formal/full and final-head CI results must be recorded
before merge. Merge only the tested head; audit deployment skips and merged tree.

Local `bash scripts/verify.sh automation` completed 185 tests in 74.040 seconds:
184 passed; unchanged scheduler test129 failed its30-second subprocess timeout
(`null !== 0`, research-datafeed-scheduler.test.mjs:732). Concurrent machine load
was observed; this does not establish the cause. No timeout or unrelated code was
changed. Pinned CI must pass the canonical wrapper; this local run is not a pass.


Pinned [run37394576416](https://github.com/diegueins680/trader/actions/runs/37394576416)
checked exact source `be18931c3aeaf40d7620f8c4188f5f39196c5b75` and passed:

- `python3 scripts/formal/verify.py --record`:70 seconds.
- `bash scripts/verify.sh formal`:141 seconds;232 integrity tests in69.605seconds.
- `bash scripts/verify.sh full`:508 seconds;232 integrity tests in70.001seconds,
  Haskell build/format/lint/smoke/tests,241 web tests/build and185 automation tests,
  including the unchanged scheduler regression. All75 SMT groups reproduce.
- Receipt SHA256:
  `356b1f1f1f96778da07df9d60996d307c71bb7b203bc340f14a06b81f5ede7cc`.
  Imported byte-for-byte from the job log; all source hashes matched the toolchain
  manifest. Only shieldConsumers, SMT, closure and source-hash sections changed.

The final local targeted subset passed12 tests in17.212seconds. Isolated scheduler
rerun passed2 tests in23.165seconds with the original timeout. The earlier local
full automation failure remains disclosed; no check or threshold was weakened.
The queued initial ordinary CI was cancelled because it contained the old receipt;
it is not counted as a pass.

The temporary reproduction workflow is removed from the delivered tree. Final-head
CI and two merge-SHA deployment audits are recorded in PR308. Overall results
remain `formalObligationsComplete=false` and `missionComplete=false`; the11 scoped
closures are not general production readiness or authorization.

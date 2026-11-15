# Causal replay v3 engineering evidence

Preregistration: `research-notes/registrations/causal-replay-v3-engineering.json`,
committed before implementation (e7134c09). Base: d243a0ed, latest main at start.
This is a repair of a prerequisite environment, not a new financial candidate.

The frozen scale has a future-row dependency (CE-RL-025). The successor fits only
strictly before the first decision and accepts one completed bar per subsequent
step. It composes the actual exact accounting kernel; no alternate cost or wealth
formula is hidden in the adapter. Unsupported features neutralize the target;
invalid or late data rejects without mutating the caller's previous session.
Scale provenance is checked by recomputation from the recorded burn-in, including
for caller-constructed Session values. The cost of recomputation is intentional;
no production latency claim is made.

The nine-component exact-rational schema uses returns and mean absolute returns,
not standard deviations. It is explicitly incompatible with frozen twelve-input
PPO artifacts. Learner integration requires a separate registration and tests.
The old identifiers, models and datasets are unchanged.

Traceability: F-RL-CAUSAL-V3-SOURCE/ARITH/FLOW/CONFORMANCE in the proof ledger and
canonical specifications map the contract to `causal_replay_v3.py`, the checker,
the Haskell oracle, integrity tests and canonical CI. A-CAUSAL-REPLAY-V3 records
truthful timestamp and funding-bucket assumptions. There is no claim of full
compiler refinement or authentic provider data. No proof placeholders are used.

Verification design: eight satisfiable-premise/unsatisfiable-violation SMT queries;
finite control bound eight steps (32 states, 39 transitions); 32 seeded synthetic
episodes, 512 exact observation oracle rows and prefix-rewrite comparisons. Source
mutants move fitting into the decision bar, change trailing reads, refit the scale
or bypass support; each must reject. Runtime fixtures cover disabled/version
mismatch, stale/late/invalid bars, non-finite and unsupported proposals, forged
scale, rational size rejection, terminal accounting and immutable prior state.
The fixed-scale frozen witness remains in its existing checker.

No real-market trial, holdout read, new financial training run, retained checkpoint,
OPE estimate or profitability result was produced. Canonical verification does
re-exercise existing synthetic PPO training fixtures; those are engineering tests,
not a new empirical training campaign. Prior economic rejection and sealed holdout
status remain unchanged. Original38 remain 15 scoped closures, 22 partial and one
open. Remaining work includes learner composition, authentic availability-bearing
data, frozen numeric blockers and inherited lifecycle/ownership obligations.

Reproduce after installing pinned dependencies (offline thereafter):

```sh
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

Exact verification receipts and timings are recorded in the PR and
`formal/research/results.json` after successful reproduction. Merge readiness for
this engineering increment does not mean the research mission is complete.

Final-code engineering benchmark (Darwin x86_64, CPython 3.13.3,
source 1d9a6bcdbd41cf215e3c003ff250bd2b50503422, 50 consumed bars, 5 warm-ups and 100 measured calls):
observation median 9.785277 ms / p99 14.337530 ms /
max 16.198054 ms; step median 9.458247 ms /
p99 16.376924 ms / max 20.205488 ms.
This includes exact scale-provenance recomputation under concurrent local load.
It is not a policy-inference benchmark, hard deadline, maximum-history throughput
result or deployment budget acceptance. No model or artifact is loaded.

Control-model scope clarification: a rejected phase ends that attempted trace;
the caller still owns the unchanged prior session and may retry a different
input. The finite model is not a persistent lockout or scheduler-liveness claim.
No model result establishes that an external caller will stop retrying.

Verification attempts (preserved failures):

- Seven focused causal-replay tests passed locally in 0.442 seconds.
- The first targeted attempt rejected a stale source lock; it passed after
  reviewing and refreshing the lock. No assertion was disabled.
- An initial local full-receipt attempt was interrupted (exit 130) to add explicit
  symbol isolation. It was not a successful verification run.
- Local `python scripts/formal/test_integrity.py`: 330 tests in 353.161 seconds;
  two errors and one failure. The unchanged accounting-reconciliation nonlinear
  query returned Z3 `unknown`, reason `timeout`, at its 10-second bound; its
  dependent malformed-bound regression also failed. The PPO process bridge
  reported no successful actual trained-policy inference in its deadline fixture.
  A diagnostic reproduced the Z3 timeout in 10.545 seconds. No theorem, numeric
  bound or inference deadline was weakened. Local full verification is not claimed.
- Clean-runner attempt 37690985951 (head 3641dfa1) reached the shield-consumer
  composition check after the accounting and process proofs, then failed because
  its fixed module count was still 19. It was corrected to the reviewed 20-module
  graph in 1d9a6bcd. This failure is not counted as a passing wrapper.
- Pre-receipt normal CI runs 37690985953 and 37691718642 were cancelled; only the
  final head checks may establish CI acceptance.

Successful clean-runner verification (GitHub Actions run 37691718979, job
113033271887, head 1d9a6bcdbd41cf215e3c003ff250bd2b50503422):

- Pinned receipt reproduction: 89 seconds; 88 SMT certificate groups; all existing
  scoped closures reproduced. The new group contains eight individual queries.
- `bash scripts/verify.sh formal`: PASS, 157 seconds; 330 integrity tests in 71.735
  seconds plus source/SMT/model/Haskell checks.
- `bash scripts/verify.sh full`: PASS, 336 seconds; formal repeated (330 tests in
  69.419 seconds), Haskell build/format/lint/smoke/test suite, 241 web tests and
  build, 185 root automation tests and canonical specification checks.
- Receipt downloaded as `reproduced-receipt` GitHub artifact and imported without
  modification. SHA256:
  `d42333065a2c5e90b4e5b670ef5f888eebdf1bdfee9621375234a6e8526f542e`.
  Its complete source hash map equals the reviewed toolchain lock. Only the new
  causal replay section, source hashes and complete-module counts changed.
- Temporary reproduction workflow removed before final-head review. Final-head CI
  and merge/deployment audits are recorded in PR #323.

Passing clean-runner checks do not erase the local resource/deadline failures or
establish mission completion. No new original38 closure is claimed.

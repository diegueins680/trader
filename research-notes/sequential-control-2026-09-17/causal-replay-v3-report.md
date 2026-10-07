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

No real-market trial, holdout read, policy training, checkpoint, OPE estimate or
profitability result was produced. Prior economic rejection and sealed holdout
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

Local engineering benchmark (Darwin x86_64, CPython 3.13.3, 50 consumed bars,
5 warm-up calls, 100 measured calls; concurrent verification load): observation
median 14.244353 ms / max 36.480550 ms; step median 14.080306 ms / max 38.578640 ms.
This includes exact scale-provenance recomputation. It is not a policy-inference
benchmark, hard deadline, maximum-history throughput result or deployment budget
acceptance. No model or artifact is loaded. Optimization needs separate evidence.

Control-model scope clarification: a rejected phase ends that attempted trace;
the caller still owns the unchanged prior session and may retry a different
input. The finite model is not a persistent lockout or scheduler-liveness claim.
No model result establishes that an external caller will stop retrying.

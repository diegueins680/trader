# OPE algebra and underflow audit — 2026-09-30

**Decision: no adoption; continue offline assurance.** No market archive, policy
training, original OPE rerun, protected outcome, production setting or champion
was changed. This is engineering evidence on the frozen evaluation helper.
All 38 broader obligations remain open or partially verified.

The specification and preregistration were committed as `d78b3155` before the
checker, tests or prescribed numerical probe. Source/verification freeze:
`6cb5c0336ac80efae7222c3a7347344e7ea19be0`. The existing dedicated branch and
draft PR #284 remain stacked on #281. Latest main at the start was
`dbd45e2691cb37f1421a23306c43b676fa82e6fc`. The preceding documentation head's
[CI run](https://github.com/diegueins680/trader/actions/runs/36701123738) passed;
that result is not substituted for verification of this continuation.

## Canonical interpretation and consistency

A-SEQUENTIAL-RESEARCH already requires admitted shapes/probabilities, a zero
terminal bootstrap, explicit overflow failure, no clipped weights and permanent
`reliable=false`. Existing tests cover an enumerated behavior tree, zero support,
malformed data and overflow. This audit adds source-bound algebra and exposes a
numerical limitation; it does not redefine reliable OPE or repair the estimator.

The helper accepts general probabilities, whereas `short_ope` uses a deterministic
target and uniform three-action behavior for six decisions. That distinction
resolves the apparent conflict between a helper underflow fixture and the current
producer's restricted weights. The registry additionally rejects successful
payloads with positive nonzero-trajectory counts but zero ESS. That guard is
visible in `reconcile_ope_payload`; this audit does not prove its full refinement.

The [formal contract](../../formal/research/ope-algebra-contract.md),
[registration](../registrations/ope-algebra-audit-engineering.json),
[ledger](../../formal/research/proof-ledger.json) and
[checker](../../scripts/formal/ope_algebra.py) give bidirectional scoped traceability.
No statistical requirement is promoted to a theorem.

## Algebra and numeric findings

| Requirement | Status | Exact scope |
|---|---|---|
| F-RL-OPE-ESS-REAL | smt_verified | Two nonnegative real weights, positive sum: ESS in [1,2], invariant under positive common scaling. |
| F-RL-OPE-WIS-REAL | smt_verified | Two bounded real returns and nonnegative weights, positive sum: weighted IS in the return bounds, invariant under positive common scaling. |
| F-RL-OPE-DR-TELESCOPE | smt_verified | Each horizon 1..6, unit cumulative ratios, q_t=v_t, zero terminal v, gamma in [0,1]: DR equals discounted return pathwise. |
| F-RL-OPE-FP-SUPPORT | refuted | With underflow ignored, finite positive weights need not produce positive ESS in the unchanged public helper. |

Eight separate SAT-premise / UNSAT-violation pairs use Z3 4.15.4, seed 0 and
10,000 ms per query. The full public/helper AST skeletons are checked; a restricted
translator extracts the estimator expressions. The two-real-trajectory statements
are not arbitrary-population proofs. DR telescoping is not an unbiasedness theorem:
even equal target/behavior policies do not imply q_t=v_t along a sampled path.

**CE-RL-018:** two six-step rows have behavior probability 1, target probability
2^-100, reward 1, Q/V zero and gamma 1. Final weights are exactly 2^-600. Squared
moments underflow to zero, so ESS returns 0 despite two nonzero trajectories;
weighted IS returns 6. The prescribed IEEE binary64 RNE SMT witness is SAT, and
actual NumPy 2.3.5 calls reproduce those outputs with underflow ignored. With
underflow raised, the public wrapper returns `ValueError` through its existing
floating-point error handler. Caller error settings are restored by the test.
The witness is preserved and remains unrepaired in the frozen helper.

For the **current producer**, all 64 binary target-probability patterns give final
weights in {0,729}. For every nonzero count 0..200 in a 200-row vector over that
set, the extracted ESS expression equals that count exactly in the pinned runtime.
This excludes the prescribed tiny-weight mechanism on that bounded arithmetic
fixture domain. It does not prove all producer inputs valid, all historical runs
reachable, bootstrap coverage or reliable policy evaluation.

## Conformance, failure checks and assumptions

Eight new integrity tests include 11 source mutants, four fixture mutants,
one registration-boundary mutant, unknown/vacuous solver rejection, actual
underflow behavior, 135 weight/return cases against exact rational references,
18 DR cases plus 18 q!=v controls, 64 target patterns and 201 support counts.
All are deterministic engineering cases; they are not financial trials or seeds.

Trusted elements remain: AST translation, solver/runtime, ordinary base arrays,
stable bindings, NumPy cumprod/reduction/broadcast semantics and the scalarization
of vector operations. The binary64 witness is a fixed moment computation, not a
universal floating-point implementation theorem. Statistical support, estimator
bias, uncertainty coverage and dependence between overlapping historical episodes
remain empirical or open assumptions. No new temporal model is needed for this
pure estimator audit; existing lifecycle/authorization models run unchanged.

## Focused literature review through execution date

- **Jiang and Li (ICML 2016)** derive sequential DR with cumulative importance
  ratios and an independently fitted value control variate. Their unbiasedness
  and variance results depend on the sampling and policy/value independence
  conditions, not merely correct arithmetic. We inspected the estimator and
  independence discussion; this audit certifies only a much narrower pathwise
  identity. No net cryptocurrency result or local replication is inferred.
  [Paper](https://proceedings.mlr.press/v48/jiang16.pdf).
- **Thomas and Brunskill (ICML 2016)** introduce WDR and MAGIC. WDR normalizes
  weights by time step; the repository's trajectory WIS plus ordinary DR is not
  WDR or MAGIC. Their Gridworld/ModelFail/ModelWin comparisons include cases
  where a direct model beats WDR. MAGIC consistency assumes bounded importance
  weights and absolute continuity. These results motivate estimator disagreement
  diagnostics, not adding estimators to rescue a rejected screen.
  [Paper](https://proceedings.mlr.press/v48/thomasa16.pdf).
- **Zhou et al. (ICML 2025)** analyze estimated history-dependent behavior policies.
  The bias/variance trade-off can favor estimated propensities asymptotically while
  worsening finite-sample bias; CartPole experiments average 50 simulations, with
  further MuJoCo tests. This is not permission to replace the known simulated
  one-third propensities or infer missing live support. We checked proceedings,
  primary HTML methods and results; no code or independent replication was audited.
  [Proceedings](https://proceedings.mlr.press/v267/zhou25f.html),
  [primary text](https://arxiv.org/html/2505.22492v1).
- **Mandyam et al., CANDOR (CHIL 2026)** screen counterfactual annotations for
  contextual-bandit OPE. The abstract reports that imperfect annotations can hurt
  and that restricting them to the direct-model component gives the preferred
  results under its assumptions. This is healthcare evidence, not validated
  financial counterfactual rewards. Proceedings metadata/abstract only were
  verified; the full-paper fetch failed. Monitor, with no implementation proposed.
  [Proceedings](https://proceedings.mlr.press/v333/mandyam26a.html).

No paper code, PDF, dataset or restricted material was copied. Full proof
assumptions, code/data licensing and independent financial replication remain
unverified for the newly screened papers. This is a focused continuation of the
existing paper matrix, not a claim to exhaust all literature published today.

## Verification outcome

- Targeted OPE audit: 8 tests passed (1.337 s).
- `bash scripts/verify.sh formal`: passed with 88 integrity tests and 32 scoped
  SMT requirements; final standalone run took 17.314 s for integrity tests and
  4.685 s for certificates. The source lock covers 66 files.
- `bash scripts/verify.sh full`: exit 0; Haskell suite, 241 web tests and 185
  automation tests passed with no skipped tests. Its formal stage took 37.666 s
  for integrity tests and 13.875 s for certificates while overlapping the
  acceptance diagnostic; timings are not isolated performance benchmarks.
- `python scripts/formal/verify.py --require-complete`: exit 1, explicitly
  `research acceptance blocked by open obligations`. This is not a pass.
- [CI at the verification source freeze](https://github.com/diegueins680/trader/actions/runs/36702686121):
  Haskell, web, automation and formal jobs passed. Docker/deployment jobs skipped.

No check was weakened, bypassed or relabeled. README, CHANGELOG, canonical
specifications, proof/assumption ledger, source lock, risk registers and research
indexes were updated. Configuration and production source were not changed.

## Decision and reproduction

Use the pinned environment in `formal/research/README.md`, then run:

```sh
python scripts/formal/test_integrity.py OPEAlgebraTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
python scripts/formal/verify.py --require-complete
```

The last command is an acceptance diagnostic and must fail while the 38 broader
obligations remain open/partial. Exact commands, exit statuses, source freeze,
log hashes and verification timing are recorded in the [companion evidence receipt](ope-algebra-evidence-receipt-2026-09-30.json).
Timing is verification overhead, not a training or inference benchmark.

Original 108 invalid OPE batches remain invalid: no ESS, confidence interval,
DSR, PBO, champion advantage or independent confirmation is manufactured. Existing
holdouts remain sealed. No model, policy, data, cost, reward, execution, API, CLI,
configuration, authorization or neural-verification scope changes. There is no
new training, inference or economic performance result. General recommendation:
**no candidate passed**. RL recommendation: **continue offline research**, with
simpler baselines and all existing proof/economic gates retained.

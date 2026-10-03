# OPE algebra and numeric support audit

Version `ope-algebra-audit-v1`, 2026-09-30. Engineering/formal only; unchanged
`ope_estimates` / `_ope_estimates` and deterministic short-OPE interpretation.
No market archive, original OPE rerun, policy training, holdout or production change.

## Scope and requirements before implementation

Existing A-SEQUENTIAL-RESEARCH requires admissible shapes/probabilities, zero
terminal bootstrap, explicit overflow failure, no weight clipping and permanent
unreliable status. Existing tests cover an enumerated behavior tree and zero
support. No existing source-linked SMT certificate covers estimator algebra.
Do not infer reliable OPE or unbiasedness merely from correct finite algebra.

- F-RL-OPE-ESS-REAL: for two nonnegative real trajectory weights with positive sum,
  source ESS=(sum w)^2/sum(w^2) lies in [1,2] and is unchanged by positive common scaling.
- F-RL-OPE-WIS-REAL: for two returns in [lo,hi] and nonnegative real weights with
  positive sum, source WIS lies in [lo,hi] and is unchanged by positive common scaling.
- F-RL-OPE-DR-TELESCOPE: for each horizon T=1..6, unit cumulative ratios, q_t=v_t,
  v_T=0 and gamma in [0,1], source sequential DR equals the discounted return in
  exact real arithmetic. These are pathwise algebraic assumptions, not an OPE
  unbiasedness theorem. In particular behavior=target alone does not imply q=v.
- F-RL-OPE-FP-SUPPORT: audit/refute the stronger claim that finite positive
  trajectory weights always give positive reported ESS in the public helper.
  Prescribed CE-RL-018: two six-step rows, b=1, pi=2^-100, reward=1, Q/V=0, gamma=1.
  Each final weight is 2^-600; squared moments may underflow to zero, selecting
  ESS=0 despite nonzero weights. Check NumPy with underflow ignored and raised.
  If the prescribed witness fails, preserve the failure, not a substituted search.

## Method and assumptions

Match complete estimator and public-admission AST skeletons before extracting
ESS, WIS and DR expressions. Restricted translation covers two-trajectory scalar
moments and a single finite row for DR. Ordinary admitted base arrays, stable
bindings, NumPy reduction/cumprod/broadcast semantics and zero terminal bootstrap
are trusted; vector/runtime refinement is not claimed. Exact-real identities do
not establish binary64 accuracy. Use a prescribed binary64 SMT witness for moment
underflow, separate from actual helper conformance. No unconditional proof about
an arbitrary count of episodes, arbitrary horizon, bootstrap coverage, behavior
support, causal identification or statistical reliability.

Also check all 64 deterministic six-step target-probability patterns with b=1/3,
and all positive-weight counts 0..200 for weights in {0,729} in the extracted ESS
expression. These bounded checks investigate applicability to current short_ope;
they are not an end-to-end market reachability theorem or a new empirical OPE run.
The original 108 invalid batches remain invalid and all promotion gates remain.

## Registered tests and tooling

Two-weight grid {0,1,2,4} excluding both zero; two-return grid {-2,0,3}; gamma
{0,1/2,1}; horizons 1..6. Use exact Fraction references plus actual helper calls.
Preserve ordinary exception/underflow behavior and source mutants for changed
weights, discount, DR bootstrap, ESS denominator, WIS normalization or reliability.
Counterexample fixture and solver result integrity must fail closed on drift.
Pinned Python 3.13.3, NumPy 2.3.5, Z3 4.15.4; independent SAT-premise and
UNSAT-violation queries, 10,000 ms each, seed 0. No new dependencies. Solver
unknown/timeout is a failed obligation, never a pass.

All critical files map to canonical IDs, assumptions, ledger, tests, source hashes
and formal/full CI. RL-OFFLINE-001 remains HIGH/OPEN. Any underflow witness remains
unrepaired in the frozen helper; no model or policy is promoted by this audit.

Primary methodological origins: Jiang and Li (ICML 2016),
https://proceedings.mlr.press/v48/jiang16.html ; Thomas and Brunskill (ICML 2016),
https://proceedings.mlr.press/v48/thomasa16.html . No paper's statistical theorem is
transferred without its data/support/model assumptions.

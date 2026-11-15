# PPO objective fidelity and numerical audit — 2026-09-28

**No learner repair or candidate adoption.** This continuation verifies the scoped
algebra of the unchanged clipped PPO objective and preserves two stronger refuted
claims. The learner, raw-target v2 helper, artifacts, financial registrations,
champion, prior returns and sealed holdouts are unchanged. Zero market data access,
financial trials, policy training or live activity occurred.

Latest main remains `dbd45e26`; draft #284 remains on
`research/sequential-review-2026-09-28`, stacked on #281. The
[contract](../../formal/research/ppo-objective-contract.md) and
[engineering registration](../registrations/ppo-objective-audit-engineering.json)
were committed at `a0ed71bf`, before checker implementation frozen at `d692bc33`.
CI for source freeze `d692bc33` passed all four verification jobs in
[run 36471091361](https://github.com/diegueins680/trader/actions/runs/36471091361);
Docker build and deployment were skipped. No dependency, service, CLI trading
option, model identifier or artifact format changes.

## Research basis and fidelity finding

[Schulman et al. (2017)](https://arxiv.org/abs/1707.06347), equation 7, defines the
minimum of two advantage-weighted ratios. The repository minimizes its negative
mean and chooses the unclipped branch at a boundary. We verify this algebra under
stated assumptions. Its control benchmarks do not establish cryptocurrency returns
or correctness of this NumPy implementation.

[Engstrom et al. (2020)](https://arxiv.org/abs/2005.12729) compare implementation
components, showing why an algorithm name is insufficient for attributing gains.
Their ablation coverage is limited; this does not endorse importing every training
trick. We keep the repository's architecture, reward handling and optimization
choices fixed and test its actual expressions.

[Ilyas et al. (2020)](https://arxiv.org/abs/1811.02553), now titled *A Closer Look at
Deep Policy Gradients*, studies mismatches between surrogate optimization and
underlying control objectives. It supports distinguishing expression correctness
from useful learning. It and the Engstrom paper share authors/content and are not
two independent replications. The [focused matrix](ppo-objective-paper-update-2026-09-28.csv)
records sources and limitations. Primary manuscripts were checked on 2026-09-28;
OpenReview pages required a browser challenge, so public author manuscripts were
used. No PDF or external implementation was committed.

The current PPO surrogate is faithful to the intended clipping branches with
registered simplifications; this is not a classification of the entire learner as
numerically safe or empirically successful. TCN, PatchTST and Transformer remain
the documented lightweight proxies; their source and identifiers are unchanged.

## Formal definition, scope and assumptions

For selected probability p, behavior probability b>0 and batch size n in 1..256,
r=p/b. Let L/U be the **exact rational values of binary64 .8/1.2 literals**.
The real-arithmetic loss contribution is `-min(r*a, clip(r,L,U)*a)/n`, equivalently
`-a*min(r,U)/n` for nonnegative advantage a, or `-a*max(r,L)/n` otherwise.
The source multiplier is `active*a*r/n`, where active means
`(a>=0 and r<=U) or (a<0 and r>=L)`. At a kink, the selected derivative is a
one-sided/subgradient convention; no unique classical derivative is asserted.

`F-RL-PPO-OBJECTIVE` and `F-RL-PPO-COEFFICIENT` add four SAT-premise/UNSAT-violation
queries: source ratio, piecewise loss, branch/bound coefficient and conditional
three-action gradient zero sum. The last assumes an exact-real simplex. Rounded
softmax probabilities need not sum to one in exact-real arithmetic. We do not
machine-prove the analytic softmax derivative identity, NumPy exponentials, a
floating-point error bound, full array reduction or full learner/compiler refinement.
A-PPO-OBJECTIVE and A-SOLVER explicitly record these trusted semantics.

The checker audits the whole function skeleton, including action indexing,
subtraction, multiplication order and return, then translates supported scalar
expressions only. Unsupported drift is rejected. The 10-second solver limit,
random seed and pinned tool versions are unchanged. A first development attempt
stopped at an invalid Z3 constructor name (`NaN`); changing it to the pinned API's
`fpNaN` resolved that checker error without changing any domain or formula. The
attempt is retained in the witness registry, not treated as a mathematical result.

## Preserved current-source witnesses

| ID | Synthetic one-row inputs | Observed result | Disposition |
| --- | --- | --- | --- |
| CE-RL-012 | Zero logits; advantage 1; old probability 2^-1074 | Ratio overflows; clipped loss is finite -1.2; all three gradient components are NaN | Refutes unconditional finite-gradient claim; unresolved numeric blocker |
| CE-RL-013 | Zero logits; advantage -1; old probability represented 1/24 | Ratio 8, coefficient -8, loss 8, gradient approximately [16/3,-8/3,-8/3] | Refutes universal multiplier cap; intended clipping limitation |

Both have prescribed SMT SAT witnesses and actual unchanged NumPy regressions.
Exact hex inputs are in [the counterexample registry](../../formal/research/ppo-counterexamples.json).
CE-RL-012 comes from Boolean-zero multiplication by an infinite ratio after a
finite clipped loss has been computed. The downstream optimizer rejects the
non-finite gradient without changing parameters, moments or step counter. That is
failure containment, not a finite-output guarantee or learner correction.

CE-RL-013 agrees with the original objective: the worsening side remains active.
It is not a failure of gradient-norm clipping, exposure limits or the independent
Haskell action shield. Neither witness establishes occurrence, prevalence or
financial harm in historical trials. The one-row fixtures test the helper input
contract; they do not establish end-to-end reachability of these advantages and
behavior probabilities through the unchanged collector/normalizer. No historical
archive was opened to look for either witness.

## Conformance, traceability and limits

Seven new tests bring integrity coverage to **59**. They cover eight source mutants,
fixture tampering, both actual numerical witnesses, optimizer state preservation,
63 registered ratio/advantage/batch cases, 18 finite-difference component comparisons
away from kinks and six boundary/adjacent-value cases with prescribed probabilities.
The rational grid conditions on the observed binary64 ratio; its 1e-12 tolerance
is regression evidence, not a universal roundoff theorem. Boundary probability
injection checks the conditional branch implementation, not softmax correctness.

The positive SMT requirement count becomes **24**. Two new refuted ledger entries
are separate from these positives. Canonical statements, A-PPO-OBJECTIVE, checker,
implementation source, tests, source hashes, counterexample artifacts, CI and risk
register have bidirectional mappings. `RL-OFFLINE-001` remains HIGH/OPEN and all
38 broader mission obligations remain open/partial. In particular, no neural
policy robustness, full runtime, convergence or safety theorem follows.

Existing models/conformance are unchanged: lifecycle 75 states/349 transitions,
two callers, depth 6; artifact path 27 states/40 transitions, depth 13; target batch
33,410 states/66,562 transitions, 1..256 rows, depth/progress bound 258. Haskell
conformance covers 20,480 cases and replay conformance 180 rational traces. They
are separate abstractions, not a composed production-system proof.

## Reproduction and verification

From the isolated worktree with pinned installed dependencies:

```bash
export PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:$PATH
export TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$TRADER_FORMAL_PYTHON" scripts/formal/test_integrity.py PPOObjectiveTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
"$TRADER_FORMAL_PYTHON" scripts/formal/verify.py --require-complete
```

Python 3.13.3, NumPy 2.3.5, Z3 distribution 4.15.4.0/solver 4.15.4, GHC 9.4.8,
Cabal 3.12.1.0, Node 20.19.0, fourmolu 0.15.0.0 and hlint 3.8. Thread caps are
process-local verification settings; no deployment or trading environment changes.
After dependency installation the formal checks require no network.

- Targeted: **exit 0**, seven tests in 1.312 seconds.
- Explicit certificate recording: **exit 0**, 8.020 seconds.
- Formal wrapper: **exit 0**, 59 tests in 13.231 seconds; verifier 8.639 seconds.
- Full wrapper: **exit 0** on the first attempt at source freeze `d692bc33`.
  Formal integrity: 59 tests in 30.401 seconds; verifier 11.287 seconds. Haskell
  build/format/lint/smoke/tests, web typecheck/241 tests/build and 185 automation
  tests passed, none skipped. No deadline, assertion, source or gate was changed.
- Acceptance diagnostic: expected **exit 1** after reproducing scoped certificates
  in 17.009 seconds, with `ValueError: research acceptance blocked by open obligations`
  and all 38 broader obligations still open/partial.

These are verification timings, not training, policy-inference or economic benchmarks.
Raw logs remain outside Git; SHA-256 receipts use prefix
`/private/tmp/trader-ppo-objective-`, suffix `-20260928.log`:

| Artifact | SHA-256 |
| --- | --- |
| targeted | `dd2d878e7ac524a8dabb8f7122bf1819134433a90f70aba5e2866f8aed76ce3b` |
| record | `870648fc3e5bf3bd7e5d44e7d5fe4777f531e573a2fdc0c8f3362030bcceb91d` |
| formal | `66c94c81da935af70532993b8e574b660b0758d77cc8b8b9d9539a203e56446e` |
| full | `6b68f31bad95b7d004b6cf199ed0c9bd103b92329d3292a303b703f3a75e6a60` |
| acceptance (expected refusal) | `724faef300b36ea1d13f90fe80af121b4597b2e05ec3b8ef6318c3938eb79dea` |

The external CI receipt `/private/tmp/trader-ppo-objective-ci-20260928.json`
has SHA-256 `8eeeabdefe7192ea947c460789796d250fe50b25584d7fb435cfa87b392cdf33`.

Previous OOS returns, stressed costs/delay, drawdown, tail-risk and invalid OPE
findings are unchanged; no new matched champion comparison or final-holdout evidence
exists. No live authorization, order, authenticated trading experiment, live
exploration, deployment, merge or proof placeholder was introduced. Recommendation:
no candidate adoption; retain the audit and continue offline research only under
the existing versioning, evidence and verification gates.

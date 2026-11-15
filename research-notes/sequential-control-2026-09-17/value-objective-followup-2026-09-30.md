# Value-based learner fidelity and numerical audit — 2026-09-30

**No learner repair or candidate adoption.** The unchanged Double DQN target and
scalar CQL gradient satisfy the registered conditional algebra checks. Two
prescribed finite-input probes refute stronger numerical claims. No market archive,
financial training/evaluation, OPE rerun or protected holdout was accessed.

Latest main remains `dbd45e26`. Branch `research/sequential-review-2026-09-28`
continues draft [#284](https://github.com/diegueins680/trader/pull/284), stacked on
#281. Prior head `ec3c5b2a` passed all four CI verification jobs. The
[contract](../../formal/research/value-objective-contract.md) and
[engineering registration](../registrations/value-objective-audit-engineering.json)
were committed at `de04744b` before implementation freeze `299c07f2`. No dependency,
production/training source, model identifier, artifact or CLI/API contract changed.
Existing external workspace edits were preserved. No open GitHub issues were found;
other open dependency/research PRs were reviewed for scope without modifying them.

## Paper-to-code findings

[Van Hasselt et al.](https://arxiv.org/html/1509.06461) separates online-network
action selection from target-network evaluation. The source uses that mechanism,
including first-index ties, but its small tanh network, reward/environment and
training budget do not reproduce the Atari experiments. The audit establishes
neither reduced bias in market data nor superior trading performance.

[Kumar et al.](https://arxiv.org/pdf/2006.04779) supplies the conservative objective.
The repository adds its discrete log-sum-exp penalty to a scalar Double DQN loss
with fixed alpha=.1. The [official implementation](https://github.com/aviralkumar2907/CQL)
uses QR-DQN for Atari. This is a CQL-inspired mechanism prototype, not a full
experimental reproduction. The paper's value-bound arguments require conditions
on evaluation, support, approximation/sampling errors and conservatism strength;
we have not discharged them for this learner. An algebra certificate cannot stand
in for those conditions or certify net returns.

| Affected component | Classification and gap | Decision |
| --- | --- | --- |
| Double DQN target slice | Faithful target mechanism with documented simplifications; full training pipeline unverified | Retain frozen baseline and audit |
| Scalar CQL regularizer | Inspired by CQL(H); no distributional critic or established policy-value lower bound | Retain accurate mechanism label; no semantic migration |
| Numerical loss reporting | Unconditional finiteness and exact translation claims refuted | Preserve witnesses; no corrective tuning |
| TCN/PatchTST/Transformer | Previously audited lightweight proxies; unchanged | Preserve identifiers/semantics |

The [focused five-paper matrix](value-objective-paper-update-2026-09-30.csv)
records depth and unknowns rather than claiming five replications. Primary sources
were checked on execution date. Some PDF/proceedings URLs failed and OpenReview
presented a browser challenge; available arXiv manuscripts and official proceedings
search records were used. TRL is a metadata/abstract-level monitor entry, explicitly
not a detailed implementation or theorem review. No paper PDF or external code
was copied into Git.

## Current literature screening and selection rationale

[NeoRL-2](https://arxiv.org/html/2503.19267v1) is useful skeptical evidence: its
seven simulators include limited behavior coverage, delayed effects and safety
constraints. Its best-configuration reporting uses three seeds and simulator
policy evaluation, so it is not independent financial confirmation or trustworthy
OPE by itself. It supports testing behavior-policy improvement and seed instability.
The official project identifies Apache-2.0 code and CC-BY-4.0 datasets; none was
installed or downloaded.

[Horizon Reduction Makes RL Scalable](https://arxiv.org/html/2506.04168v3) studies
long-horizon failures and SHARSA on large simulated goal-reaching datasets. Four
seeds and reported confidence bands aid interpretation, but oracle representations
and selected in-distribution goals limit transfer. More data or a newer algorithm
does not resolve our execution/support gaps. Monitor horizon-aware methods; do not
change the frozen 1/3/6-bar economic hypotheses after viewing their evidence.

[Transitive RL](https://proceedings.iclr.cc/paper_files/paper/2026/hash/066ac8a48e27c78aadcf934b580a0383-Abstract-Conference.html),
ICLR 2026, targets goal-reaching using a divide-and-conquer value update. Its
triangle-inequality structure is not established for signed inventory rewards,
costs and drawdown constraints here. Monitor only; no implementation or theorem
transfer is justified by the screened material.

This continuation's qualitative scorecard uses 0=unestablished/poor fit,
1=partial, 2=strong for the stated role; it does not alter the existing candidate
ranking or add an economic trial. Replication means independent evidence for this
repository's economic use, not existence of an official code repository.

| Family/evidence | Primary venue | Mechanism match | Independent economic replication | Formal analyzability of small core | CPU fit here | Action |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Double DQN | 2 | 2 | 0 | 2 | 2 | Audit existing prototype |
| CQL | 2 | 1 | 0 | 1 | 2 | Audit existing scalar mechanism |
| NeoRL-2 | 1 | 1 | 0 | 1 | 1 | Retain validation lessons |
| SHARSA | 2 | 0 | 0 | 1 | 0 | Monitor; no compute/data expansion |
| TRL | 2 | 0 | 0 | 1 | 0 | Monitor; goal structure missing |

These papers do not establish cryptocurrency predictability, cost robustness,
state-action support, calibrated uncertainty or compliance with our safety boundary.
No new high-level candidate is shortlisted or promoted.

## Specification, assumptions and consistency resolution

Let k be the first argmax of three online Q values. The target slice evaluates
`reward + gamma*(1-terminal)*targetQ[k]`, with the actual represented discount
for horizon 1/3/6. Exact-real terminal independence is conditional on successful
network evaluation; the source still calls both networks for terminal rows.

For row size n, residual e=q[a]-y and softmax probabilities p, the gradient is
`1[j=a]*e/n + alpha*(p[j]-1[j=a])/n`. Under an exact simplex, conservative
components sum to zero and lie in [-alpha/n,alpha/n]; the total sum is e/n.
The loss scalarization treats the batch mean as a sum of row contributions.
It is not an IEEE reduction theorem. The component bound is conditional on alpha
and batch size; it is not a neural parameter-update or market-exposure bound.

`F-RL-DOUBLE-TARGET` and `F-RL-CQL-GRADIENT` each have one SAT-premise/UNSAT-violation
query. Total scoped SMT requirements: **26**. A-VALUE-OBJECTIVE names ordinary
aligned float64 arrays, stable bindings, first-argmax correspondence, scalarization,
softmax/logarithm calculus, compiler/runtime/solver and AST translator trust.
Real simplex assumptions do not certify rounding of actual softmax outputs.
No machine-checked transcendental, neural-policy, convergence or CQL value-bound
proof is claimed. The target source slice is audited, not the complete train_q
control flow. Correct function names alone cannot discharge these limitations.

An apparent ambiguity is resolved explicitly: alpha=0 removes CQL **in real
algebra**, but does not disable evaluation of its NumPy penalty. The executable
source is authoritative for runtime behavior. We preserve its behavior and mark
the unconditional numerical claim refuted instead of describing zero alpha as a
safe bypass. Existing production authorization/risk specifications are unaffected.

## Current-source numerical witnesses

| Witness | Prescribed input | Actual outcome | Scope |
| --- | --- | --- | --- |
| CE-RL-014 | q=[maxFloat,maxFloat,-maxFloat], action 2, y=-maxFloat, alpha=0 | Zero residual and gradient; NaN loss from overflowing conservative difference then zero multiplication | Refutes F-RL-Q-FINITE |
| CE-RL-015 | Equal zero q/y versus common shift 2^54; action 0, alpha=.1 | Loss changes from approximately .10986122886681099 to 0; finite gradients are identical | Refutes F-RL-CQL-SHIFT |

[Exact hex fixtures](../../formal/research/value-counterexamples.json) bind both
prescribed SMT SAT witnesses to actual NumPy reproduction. Log(2)/log(3)
intermediates are explicit trusted constants checked in the pinned runtime.
The shift witness additionally checks log(3)'s supplied value lies in (1,2).
The analytic logarithm identity and interval are not machine-proved here.

CE-RL-014 differs from the PPO gradient witness: a fresh optimizer accepts its
zero gradient and advances the step counter, with unchanged parameters. The
optimizer has no loss argument. Therefore optimizer non-finite-gradient rejection
must not be misreported as finite-loss admission. This synthetic fixture does not
show reachability through registered initialization/updates, historical occurrence,
financial harm, or that an invalid final archive could pass existing publication
checks. Those are separate obligations. Neither witness is silently repaired.

## Conformance, ledger and retained failures

Seven new integrity tests bring the count to **66**. They cover 12 source mutants,
fixture tampering, 13,122 complete bounded target cases (including ties), 324
registered loss/gradient cases, 27 ordinary finite differences, both numerical
witnesses and downstream optimizer behavior. The floating tests are regression
and conformance evidence, not formal proofs or a universal roundoff bound.

The first targeted run failed one mutation test because its string replacement
matched PPO's `gradient` return prefix instead of the intended exact `grad` line.
The fixture was corrected to match the complete line, and all seven tests passed.
No solver formula, source learner, domain, timeout or financial gate changed.
The failed log is retained with its hash; it is not counted as a certificate.

The canonical registry, assumptions, proof ledger, critical-file roster, source
hashes, CI wrapper and risk register link both directions. All 52 locked source
hashes match. `RL-OFFLINE-001` remains HIGH/OPEN; all 38 broader mission obligations
remain open/partial. No proof placeholder, disabled assertion or skipped theorem
was introduced. Existing bounded models are unchanged: lifecycle 75 states/349
transitions (two callers, depth 6); artifact 27 states/40 transitions (depth 13);
target-batch 33,410 states/66,562 transitions (1..256 rows, progress bound 258).
Haskell conformance retains 20,480 cases and rational replay 180 traces. These
separate models are not a composed proof of the production system.

## Verification and reproduction

Use the existing pinned offline dependencies. Commands from the isolated worktree:

```bash
export PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:$PATH
export TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$TRADER_FORMAL_PYTHON" scripts/formal/test_integrity.py ValueObjectiveTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
"$TRADER_FORMAL_PYTHON" scripts/formal/verify.py --require-complete
```

Python 3.13.3, NumPy 2.3.5, Z3 distribution 4.15.4.0/solver 4.15.4, GHC 9.4.8,
Cabal 3.12.1.0, Node 20.19.0, fourmolu 0.15.0.0 and hlint 3.8 are unchanged.
The 10,000 ms solver limit is unchanged. Thread caps are process-local verification
settings, not deployment changes. Formal reproduction needs no network after setup.

- Initial targeted run: **exit 1**, one mutation-fixture error; preserved.
- Corrected targeted run: **exit 0**, seven tests in 3.918 seconds.
- Explicit certificate recording: **exit 0**, 24.839 seconds.
- Formal wrapper: **exit 0**, 66 tests in 16.489 seconds; verifier 6.702 seconds.
- Full wrapper: **exit 0** on its first attempt at source freeze `299c07f2`.
  Formal integrity: 66 tests in 17.845 seconds; verifier 14.228 seconds. Haskell
  build/format/lint/smoke/tests, web typecheck/241 tests/build and all 185 automation
  tests passed, none skipped. No deadline, assertion, source or gate was changed.
- Acceptance diagnostic: expected **exit 1** after scoped checks in 21.196 seconds,
  with `ValueError: research acceptance blocked by open obligations`; all 38
  broader obligations remain open/partial.

These are verification timings, not inference/training or economic benchmarks.
The [machine-readable evidence receipt](value-objective-evidence-receipt-2026-09-30.json)
retains every verification attempt, including the failed fixture and expected
acceptance refusal. Raw logs remain outside Git; prefix `/private/tmp/trader-value-objective-`,
suffix `-20260930.log`. SHA-256 receipts:

| Artifact | SHA-256 |
| --- | --- |
| targeted-initial | `ccecc743d05acf7072343fefb44a9296dea253596fa58b87f1b2010712f5f238` |
| targeted | `be58901f32edd1e8169d7705b59c1460ecdda49535064079a5750c6023f9cf62` |
| record | `df97ab1166a279dacbdc23d8639aa78bb582681d01147281d885cbd704666d25` |
| formal | `27bf47708d07a2cf559740eee39d82dbaf4a698de17424c6e8117432d424d238` |
| full | `0336f9f787dd2437a2ad7bcf966d54ff96b12b372f4faf51a499b7f811673b98` |
| acceptance | `5f9b77e9c453381725e3acb1a745ae7a5b471e11a7315d171989bb2aea6c4b14` |


Remote CI [run 36652227711](https://github.com/diegueins680/trader/actions/runs/36652227711)
passed formal, Haskell, web and automation at `299c07f2`; Docker build and deployment
were skipped. External receipt `/private/tmp/trader-value-objective-ci-20260930.json`
has SHA-256 `6f24022105d586fba25b6a903f57081acac34b332d00a07d930bdd42ca0b20be`.

## Economic and operational disposition

The existing 108 fits, all seeds, 19,440 replay paths, costs/stresses, drawdown,
tail risk and invalid OPE evidence are unchanged. No new matched-champion comparison,
independent OOS confirmation, DSR/PBO/SPA or final-holdout evidence exists. The
historical holdout remains sealed and prospective carry outcomes remain embargoed.
No inference or production performance claim follows from small audit runtimes.

General recommendation: **no adoption**. RL recommendation: **reject the tested
configurations**; further offline work requires a justified registered question.
No shadow/paper activation follows. No live flags, orders, authenticated trading
calls, exploration, fleet/ownership/risk-cap changes, deployment or merge occurred.
The independent Haskell proposal boundary and champion remain unchanged. This
continuation is not completion of the full research mission.

# Admission-continuation successor (2026-10-10)

This adaptive development experiment follows the [v1 rejection](report.md).
The [registration](../registrations/rl-admission-continuation-v2.json) was committed
at `d2cf5957` before implementation tests or successor market outcomes. The
verified-main base remains `354c853176b7ee2ac6570951f7f6f2d0aed6909d`.
The original execution and evidence archive remain immutable.

## Decision and accounting contract

The problem remains sequential target exposure with inventory, costs and a
pending slot. Only target admission changes. The raw policy proposal remains the
PPO likelihood action and Q-learning action label. Observation support may first
project it to zero. At the next completed close, the exact immutable accounting
kernel previews the due target against marked equity and funding. A
`turnover_limit` rejection causes a second calculation with zero target **from
the original session**. The rejected preview is never committed, charged or
used as the next state. The accepted calculation is the only economic reward.

The new target is zero, not necessarily the filled position: 50% fill stress
retains half the previous units. Costs apply to actual turnover. Delay stress
retains one pending target, revalidated at its due close. Other economic
risk termination, liquidation, insolvency and arithmetic exhaustion retain v1
semantics and still fail the trial. No failure is silently reset in evaluation.
Terminal horizon liquidation remains separate from target execution.

A pre-fill admissible position has absolute exposure at most 0.35. A zero target
with fill fraction in (0,1] requires turnover at most 0.35, below the 0.50 cap.
With registered linear costs at most 50bp, this action also preserves the
0.35 endpoint exposure bound and positive equity relative to positive pre-fill
equity. It does not guarantee the capital floor, drawdown, gap losses or live
recovery. These are exact-real lemmas with bounded Fraction conformance, not a
theorem about every Python/Haskell execution or every market disturbance.

`actionChanged` reports support-induced proposal changes or a turnover-induced
due-target change. `unsupported`, `legacyReasonFlag`, `schedulingDeferred` and
`newProposalIgnored` remain separate. Unsupported cash proposals therefore do
not count as interventions. Delay counts do not pretend that the current
proposal was filled. Traces add units, prices, funding per unit, fill fraction
and cost multiplier so an independent auditor can reconstruct gross P&L,
funding, costs and nonterminal fills. Terminal paths may include both a trade
and risk liquidation; the auditor does not claim to reconstruct that sequence
from one aggregate row.

## Registered comparison

Same two exposed windows, BTC/ETH/SOL, five seeds, 512 transitions or updates,
three RL learners (PPO, simulator Double DQN, frozen-dataset discrete CQL),
uniform-behavior cloning, cash, constant-long, momentum and reversal controls.
CQL alpha-zero remains the registered diagnostic. All policies train and
evaluate through v2 admission. New simulated behavior buffers are generated
under v2; old transition rewards are incompatible. The seven cost, delay,
partial-fill and funding cases remain unchanged. No search or best-seed choice.
Training uses the base execution case. Nonzero pending state, partial fills and
altered costs are evaluation shifts; these policies were not trained to manage
nonzero pending commitments. The delay result is a transfer stress test.

The current champion, authentic historical first-seen timestamps, real logged
support and untouched confirmation remain unavailable. Sealed holdouts remain
protected. Differences from v1 cannot be credited to algorithmic alpha because
task termination changed. This continuation tests whether corrected admission
permits a trustworthy full-horizon comparison.

## Reproduction

Use Python3.13.3/NumPy2.3.5/Z3 4.15.4.0. From the isolated worktree, use the
already authorized development exports (hashes in the registration) and a new
external output directory:

```sh
python scripts/research/run_rl_admission_v2.py \
  --run-registered-development-v2 \
  --panel /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-panel.csv \
  --settlements /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-settlements.csv \
  --output /private/tmp/trader-rl-admission-v2-evidence
python test/rl_composition_test.py
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

The campaign rejects uncommitted experiment sources and input-hash mismatches.
Execute with an empty inherited environment except PATH and single-thread BLAS
settings. No credentials or order-capable clients are needed. CPU inference
remains Python research inference, with no Haskell production loader, numerical
parity certificate or operational latency guarantee.

## Assurance scope

`F-RL-ADMISSION-V2` adds two nonvacuous SMT checks and a preserved synthetic
counterexample: v1 stops after three transitions; v2 completes 32 with 13
turnover projections, each compared to a fresh original-state ledger call.
Tests cover economic gaps, all seven stresses, state continuity, terminal
liquidation and corrupt primitive traces. An instrumented test now exercises
the actual DDQN training loop's online-selection/target-evaluation, terminal
mask and 100-update target synchronization. This is stronger than a stand-alone
formula test, but still not full update-by-update independent-library parity.
Existing PPO/CQL gradient and three-learner known-solution tests remain.

Canonical checks must reproduce the original scoped lifecycle/Haskell oracle
and new admission evidence. The 18 broader open obligations remain open.
This adds no production authorization, shared-budget admission or challenger
promotion path. The scheduled review-feedback autoloop still prevents safe
publication; retain local commits/patch unless that blocker is resolved.

## Executed results and adoption decision

**Reject this candidate package for integration.** The admission repair enables
complete comparisons, but no primary RL family improves on momentum. PPO's
small positive mean disappears at 2x costs. All three families exceed the
registered 20% action-change limit. Prefer the existing simpler decision path;
these short, exposed windows do not authorize adopting momentum or long exposure
in production, either. The current champion is still not replayed here.

Frozen implementation: `fb23bc3e4a39f24dfc43fc5afbda25613f055925`.
All 50 fits and 1,218 replays completed without failure: 30 primary RL fits,
10 CQL-alpha-zero diagnostics and 10 behavior-cloning fits. Every replay has
64 transitions; no unequal stopped horizons remain. Last checkpoints and all
five seeds are retained. There was no hyperparameter search or outcome-driven
successor during this campaign. This adds 50 fits to the preceding 50-fit
composition pilot and earlier registered searches; it is not a fresh independent
set of selection trials.

The two BTC/ETH/SOL windows remain March–April 2022 and February–March 2023,
with training cutoffs and assumed close+1ms availability exactly as registered.
All data are previously exposed development evidence. No holdout was opened.
The strict causal burn-in min/max support gate rejects out-of-range features;
it is not calibrated uncertainty or offline state–action overlap. Historical
replay is a price-taker counterfactual model, not observed exchange execution.

Base-cost returns below are equal-weight mean net equity changes across symbols,
folds and all seeds. These are approximately 21-day episode returns, not annualized
rates. Drawdown and ES are the worst individual base path; ES95 uses very few
8-hour tail observations and is descriptive. Cash naturally has no fills.

| Policy | Mean net % | Worst DD % | Worst ES95 % | Action changed % | Turnover rejected % |
|---|---:|---:|---:|---:|---:|
| ppo | 0.107 | 5.179 | 1.336 | 27.604 | 0.260 |
| double_dqn | -0.520 | 4.742 | 1.361 | 42.448 | 1.979 |
| cql | -0.285 | 4.485 | 1.558 | 28.385 | 0.938 |
| cql_alpha0 | 0.448 | 4.245 | 1.307 | 41.719 | 0.729 |
| behavior_cloning | -0.828 | 4.782 | 1.559 | 49.062 | 2.240 |
| cash | 0.000 | 0.000 | -0.000 | 0.000 | 0.000 |
| constant_long | 0.997 | 4.480 | 1.559 | 60.156 | 0.000 |
| momentum3 | 0.965 | 2.946 | 1.162 | 64.844 | 4.688 |
| reversal3 | -1.889 | 4.246 | 1.713 | 64.844 | 4.688 |

The constant-long control is **shielded**, not unconstrained buy-and-hold.
CQL-alpha-zero is an ordinary fixed-buffer DDQN diagnostic, not an additional
support-aware offline algorithm. Its better mean than CQL is negative evidence
for the registered regularization setting, not permission to select it after
seeing these outcomes. Uniform-behavior cloning is a simple
information-matched control. **Fidelity correction:** raw momentum/reversal
access unnormalized history unavailable in the twelve-entry RL vector; they are
contextual controls, not exact observation-matched policies. The original
registration and archived outcomes are preserved with this limitation. The
supplement below evaluates controls that use exactly the RL vector.

All-seed mean net changes, percent (each seed averages six symbol/window paths):

| Policy | 11 | 23 | 47 | 71 | 101 |
|---|---:|---:|---:|---:|---:|
| ppo | 0.013 | 0.608 | -0.113 | 0.046 | -0.021 |
| double_dqn | -1.063 | -0.982 | 0.070 | 0.023 | -0.647 |
| cql | -0.237 | -0.757 | 0.495 | -0.151 | -0.773 |
| cql_alpha0 | 0.614 | -0.011 | 0.505 | 0.145 | 0.987 |
| behavior_cloning | -1.520 | -0.767 | -0.046 | -0.786 | -1.023 |

Seed-mean sample standard deviations are PPO 0.286pp, DDQN 0.540pp and CQL
0.522pp. These measure algorithmic dispersion, not independent market histories.

All registered dependence-aware paired intervals include zero at blocks 3, 9
and 21. The intervals use 2,000 circular block resamples within each chronological
window, averaging symbols and seeds **before** resampling; six-comparison
Bonferroni tails. Full intervals are in [evidence.json](admission-v2-results/evidence.json).
For the 9-bar sensitivity case:

| RL / comparator | Mean incremental pp | Descriptive simultaneous 95% interval, pp |
|---|---:|---:|
| ppo / cash | 0.107 | [-1.098, 1.260] |
| ppo / momentum3 | -0.858 | [-5.265, 2.037] |
| double_dqn / cash | -0.520 | [-1.927, 0.612] |
| double_dqn / momentum3 | -1.485 | [-5.602, 1.687] |
| cql / cash | -0.285 | [-2.508, 1.980] |
| cql / momentum3 | -1.249 | [-5.990, 2.349] |

Short windows, adaptive design and nonstationarity limit the bootstrap's
interpretation. No interval is untouched confirmation. DSR and PBO remain
**insufficient evidence**; no favorable probability was substituted. The primary
0.10pp materiality threshold must hold over the registered comparators, so PPO
barely exceeding cash in its point estimate is not a pass.

Stress results, mean net percent across all seeds/symbols/windows:

| Stress | PPO | DDQN | CQL | Momentum | Constant long |
|---|---:|---:|---:|---:|---:|
| base | 0.107 | -0.520 | -0.285 | 0.965 | 0.997 |
| cost1_5x | 0.050 | -0.713 | -0.410 | 0.739 | 0.898 |
| cost2x | -0.007 | -0.855 | -0.536 | 0.514 | 0.798 |
| severe50bp | -0.348 | -1.704 | -1.247 | -0.826 | 0.201 |
| delay1bar | 0.255 | 0.194 | -0.182 | 0.125 | 0.838 |
| partial50pct | 0.306 | -0.539 | 0.075 | 0.721 | 1.174 |
| funding2x | 0.109 | -0.507 | -0.302 | 0.950 | 0.953 |

Delay occasionally improves returns, demonstrating path sensitivity; it is not
an execution advantage. Severe cost is 50bp per turnover. Funding is doubled,
not a claim about arbitrary funding shocks. This still lacks realistic liquidity,
missed-fill/outage calibration, exchange precision and minimum notional; those
limitations continue to block integration.

PPO's mean gross gain is 0.218%, funding +0.0024%, and costs 0.114%, reconciling
to net +0.107%. DDQN has negative gross P&L as well as costs. CQL is approximately
flat gross before funding and 0.251% costs. CQL's fold means change from +1.613%
to −2.182%; PPO's symbol means are BTC +0.021%, ETH −0.168%, SOL +0.467%.
These are unstable outcomes, not evidence of regime robustness. Complete
turnover, holding, fills, exposure, fold/symbol breakdowns and critic errors are
in [diagnostics.json](admission-v2-results/diagnostics.json), with every trial in
[trial-registry.csv](admission-v2-results/trial-registry.csv) and all seeds/stresses
in [all-seed-results.csv](admission-v2-results/all-seed-results.csv).

## Support, accounting and implementation findings

The two frozen datasets each contain 512 simulated transitions. Raw proposal
counts are [172,171,169] and [154,184,174], with known proposal probability 1/3.
Their hashes are `500c04ab4643c6d5c4959e31de18645e536971c3046c57d90143eb782eff1e60`
and `23971b289be447f3514f1720518ee70e7324ae0419d2aca498471505e0cfef0e`.
All actions appearing in aggregate is not proof of local state–action overlap.
Many-to-one projection changes executed-action mass; algorithms keep proposal
labels and never invent executed-action propensities. No real logs, valid real
IS/DR/FQE or real-data ESS are available. No OPE efficacy claim is made.

DDQN and CQL mean finite-path critic MSEs are 1.570 and 1.519 in squared reward
units (reward = 100 times equity change). They are diagnostics, not calibrated
confidence or support certificates. No memory/predictor/reward shaping module
was introduced or silently credited for returns.

The independent trace audit checked **77,952 rows with zero violations** of its
stated identities and admission checks. It reconstructed units/prices, gross
P&L, funding and linear costs, plus nonterminal fills, using binary64 tolerance
1e-12. There were 855 turnover projections, 29,480 support-driven proposal
changes, 30,335 action changes, and 5,568 delayed plus 5,568 ignored proposals.
There were 46,893 unsupported observations, a distinct measure. The broader
changed-or-deferred count 34,211 is also distinct. No checked executable-bound
violation was found; this is not a global safety theorem. Terminal multi-fill
reconstruction and production concurrency remain outside the audit.

The exporter retains a `legacyCounter` note about **v1**'s discarded constant
`unsafeExecutableIntents=0`; v2 never emits that constant. Its zero-violation
result comes from actual trace checks. The shared CSV field
`meanTerminalOrStoppedReturn` names a complete-horizon return for every v2 path,
since all paths completed. No failed endpoints were averaged as full horizons.

The rejected-preview regression prevents an artificial turnover-triggered reset
and checks that its hypothetical liquidation fee is never charged. Genuine
capital/drawdown failure still terminates with losses. These checks do not show
that the original learned policies intentionally exploited resets. The corrected
task must be learned anew, as done here.

All v2 policy cards share environment `rl-admission-continuation-v2`, observation
`causal-inventory-pending-finite-v1`, three categorical target proposals, a
12×16 tanh network, greedy frozen evaluation, CPU research-only mode and the
same shield. PPO uses fresh on-policy trajectories; DDQN uses fresh interaction
and replay; CQL uses only the frozen simulated dataset. Alpha-zero and behavior
cloning are diagnostic controls. Each artifact records its fold, seed, source,
registration, data reference and parameter identity. Every card's disposition
is **not eligible for integration**; no production loader accepts these files.

An additional interpretation limit is that unsupported proposals all project to
the same zero target. Unshaped economic reward cannot distinguish those raw
proposals on such a state. A high raw-proposal intervention rate therefore need
not mean that the shield executed unsafe exposure. The registered usability
gate is retained; changing masks, reward or decoder semantics would require a
new version and registration, not a favorable retrospective reinterpretation.

## Compute, artifacts and verification

Training plus per-fit benchmarks took 56.967s; the campaign took 285.007s on CPU. Peak process RSS was 83,853,312 bytes. The largest per-model warm Python p99 was 0.240ms (100 calls after 10 warmups); largest JSON policy artifact 4,892 bytes. Host load differs from v1: this is not a controlled latency improvement or a Haskell/end-to-end deadline guarantee. No model was chosen on timing.

Original archive: `/private/tmp/trader-rl-admission-v2-evidence`.
Retained copy: `/Users/diegosaa/GitHub/trader/.tmp/research/rl-admission-continuation-v2-20261010`.
Index SHA256: `375289707f8b55a3d58d8ded69f7686215f0c67297dcdd9c22739e5f6a3956f4`.
Raw data, buffers, policies and large traces stay outside Git. Small manifests,
hashes and all-seed results are committed. The archive must not be modified;
run the exporter to a separate fresh directory with this index hash.

Targeted tests before supplemental controls: 17 passed; final targeted tests: 18 passed. The standalone composition checker passed seven
inherited and two new SMT lemmas, the 43-state/107-edge lifecycle model, eight
Haskell Rational ledger witnesses, and the 32-step continuation witness.
The final `bash scripts/verify.sh full` passed: 341 formal regression tests,
18 RL tests, reproduced certificates, Haskell build/format/lint/smoke/tests,
web typecheck/build and 241 tests, plus 185 automation tests. Zero failures,
skips or cancellations in the final run. The final formal recording completed
in 188.637s. [verification-receipt.json](admission-v2-results/verification-receipt.json)
contains the exact clean-environment command, tool/source hashes, earlier
failed/stopped attempts and retained log hashes. A successful scoped wrapper
does not close the 18 broader obligations or change the adoption decision.

RL research: successor development experiment completed; the broader mission
remains partially completed. RL adoption: reject this package for integration;
prefer the existing simpler production path. Formal assurance: conditional
scoped checks only, with 18 broader obligations still unresolved. Financial
execution: no real orders, live exploration, production changes or new live
authorization. Existing production activity was preserved.


## Supplemental observation-fidelity check

After examining v2 results, source review found a fairness defect: raw momentum
uses the sign of an unnormalized 3-bar return. The RL vector contains the return
centered/scaled by burn-in statistics, without those statistics as extra inputs.
Therefore its raw sign is not recoverable from that vector alone. Previous
statements that these raw rules were exactly observation-matched were incorrect.
Cash and the learned behavior-cloning control did use the common information.
The original raw-rule results remain contextual evidence, not a clean isolation
of decision-policy skill.

Registration `rl-observation-controls-v2` at `f275b7d4` explicitly records that
this was discovered **after** v2 outcomes. It fixes two deterministic rules:
sign and reverse-sign of observation[1], with exact zero mapping to cash. They
use only the same twelve-entry vector as RL. No new inputs, learner fits,
checkpoint selection or tuning were permitted. Code was frozen at
`c429f0a7010ccb6536b256c5a701d00341d0444c`; all 84 additional replays completed.
The original archived RL traces were reused unchanged. The supplemental auditor
checked 5,376 rows with zero violations of its scoped checks.

Normalized momentum averaged **+1.487%** net and normalized reversal **−2.376%**
in base costs. The former is a deviation-from-burn-in-mean rule, not raw trend
following. All three RL families still have lower mean returns. Their adaptive
paired 9-bar-block incremental intervals against normalized momentum are:

| RL | Mean incremental pp | Descriptive simultaneous 95% interval, pp |
|---|---:|---:|
| ppo | -1.381 | [-5.897, 1.760] |
| double_dqn | -2.007 | [-6.403, 1.578] |
| cql | -1.772 | [-6.539, 2.053] |

Every block-length interval still includes zero. The control itself has a 64.84%
action-change rate and is not eligible for adoption under the same usability
gate. Prefer the established simpler production path; do not promote this
retrospectively examined rule. This diagnostic repairs the observation-matching
comparison but **does not retroactively preregister it** or cure the absent
champion, authentic availability, independent confirmation or formal integration.
The six-comparison interval adjustment applies within this diagnostic; it does
not account for cumulative adaptive selection across earlier research campaigns.

All 84 trials and the complete paired sensitivity results are in
[observation-controls.json](admission-v2-results/observation-controls.json) and
[observation-controls-trials.csv](admission-v2-results/observation-controls-trials.csv).
Additional artifact index SHA256:
`7089273264e081156a7eb137effd782dd527b0888c6f779216bb42594bd1cada`.
Retained archive:
`/Users/diegosaa/GitHub/trader/.tmp/research/rl-observation-controls-v2-20261010`.

Reproduce with the same pinned Python environment and a fresh output directory:

```sh
python scripts/research/compare_rl_observations_v2.py \
  --run-registered-observation-controls \
  --source /private/tmp/trader-rl-admission-v2-evidence \
  --panel /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-panel.csv \
  --settlements /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-settlements.csv \
  --output /private/tmp/trader-rl-observation-controls-v2
```

The supplement rejects drift in the archived environment sources, altered
archive bytes and output paths inside the frozen archive. There are now 1,352
recorded new trial IDs in this continuation: 50 fits, 1,218 original replays and
84 supplemental replays. These are correlated development observations, not
1,352 independent statistical samples.

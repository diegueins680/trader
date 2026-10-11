# Training-budget sensitivity (2026-10-10)

The preregistration commit is `ff3cce38`, preceding implementation tests and
market outcomes. This is an adaptive continuation of the rejected v2 package,
from verified main `354c853176b7ee2ac6570951f7f6f2d0aed6909d`. Prior evidence and
protected holdouts remain unchanged. There is no current-champion comparison.

The sequential task remains bounded target exposure with inventory, trading
costs, funding and next-close fills. The v2 environment, exact ledger, safety
shield, twelve observations, network architecture and learning rates are fixed.
Only the training budget changes from 512 to 4,096. PPO collects fresh rollouts
and makes 128 actor updates; Double DQN collects 4,096 simulator transitions and
makes 4,096 replay updates. Discrete CQL, its alpha-zero diagnostic and behavior
cloning make 4,096 updates on the original 512-transition simulated datasets.
Both buffer content identities must match before any fit. This separates a
larger optimizer budget from additional offline data; training regimes do not
have identical numbers of unique interactions.

All five seeds and last checkpoints are retained. There is no checkpoint tuning,
best-budget selection or claim of convergence. Cash, constant-long and both
normalized observation-vector controls are included before outcome inspection.
The normalized momentum rule uses deviation from burn-in mean, not raw trend.
The original 20% action-change gate remains: unsupported proposals can be
aliased to the same zero target, but no mask or reward change conceals that issue.
PPO is trained with sampled proposals; final evaluation uses the registered
argmax extraction. All families evaluate through the same shield and ledger.

Data are the two previously exposed BTC/ETH/SOL eight-hour development windows,
with assumed close+1ms availability, 96-bar causal burn-in and immutable
per-episode scaling. Training episodes have 32 transitions; evaluation episodes
have 64. Funding is assigned to its next close; old inventory earns price/funding
changes before the due target fills. Seven inherited execution stresses apply.
No real behavior logs or exchange propensities exist. Historical replay is
model-dependent counterfactual evidence. DSR/PBO remain insufficient evidence.
No new holdout is created by giving this continuation a new campaign identifier.

Reproduction uses the pinned research environment and an empty inherited
environment except the executable PATH and single-thread BLAS settings:

```sh
python scripts/research/run_rl_budget_v3.py \
  --run-registered-budget-v3 \
  --panel /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-panel.csv \
  --settlements /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-settlements.csv \
  --output /private/tmp/trader-rl-budget-v3-evidence
python test/rl_composition_test.py
bash scripts/verify.sh full
```

The [registration](../registrations/rl-training-budget-v3.json) specifies all
seeds, hashes, comparisons and limits. This runner preserves the v1/v2 entrypoint
defaults. Reproducing the frozen v2 supplemental controls still requires their
recorded commit, because their source checks intentionally reject later runners.
Research artifact environment version remains v2; the new campaign and
registration hash identify this budget. No production loader is added.

Formal scope is unchanged: seven composition SMT lemmas, two zero-target
admission lemmas, a 43-state/107-edge bounded lifecycle model, eight Haskell exact
ledger witnesses and the turnover counterexample. Numerical and runtime trust
assumptions remain. The new orchestration is source-locked with targeted tests;
this is not a new theorem about learning efficacy or production integration.

## Executed comparison and decision

**Reject this package for integration.** Increasing the registered budget improves
PPO and Double DQN's means but does not establish an advantage over the matched
control. CQL and its alpha-zero diagnostic deteriorate. All three RL families
still exceed the unchanged 20% action-change gate. The control itself fails that
gate, so these results do not authorize its adoption either. Retain the existing
production decision path; its economic comparison is still missing here.

Frozen experiment code: `c962c227cec940ca6277acbd246e1f9fdffcb399`.
All **50 fits and 1,218 evaluation replays** completed: 30 primary RL fits,
10 alpha-zero diagnostics and 10 cloning fits. Every evaluation has 64 transitions.
The six training risk terminations are disclosed below; fit completion
must not be confused with the absence of training risk events. This adds one
adaptive budget setting and 50 recorded fits to previous research; it does not
supply 50 independent markets or erase prior selection trials.

Mean net episode returns are percentages, equally weighted across symbols, folds
and seeds. Worst drawdown and ES95 are individual base-path values. Each episode
covers about 21 days. ES95 from 64 eight-hour returns is descriptive; no annualized
Sharpe or high-confidence tail estimate is inferred.

| Policy | Mean net % | Worst DD % | Worst ES95 % | Action changed % | Turnover rejected % |
|---|---:|---:|---:|---:|---:|
| ppo | 0.299 | 4.480 | 1.559 | 23.021 | 0.156 |
| double_dqn | 0.311 | 5.896 | 1.559 | 34.271 | 1.510 |
| cql | -1.605 | 5.959 | 1.714 | 37.656 | 1.823 |
| cql_alpha0 | -1.588 | 6.818 | 1.714 | 42.344 | 1.094 |
| behavior_cloning | -0.171 | 2.911 | 1.381 | 34.115 | 3.854 |
| cash | 0.000 | 0.000 | -0.000 | 0.000 | 0.000 |
| constant_long | 0.997 | 4.480 | 1.559 | 60.156 | 0.000 |
| normalized_momentum3 | 1.487 | 2.655 | 1.162 | 64.844 | 4.688 |
| normalized_reversal3 | -2.376 | 4.952 | 1.713 | 64.844 | 4.688 |

Five seed means, percent, each averaging six equal-capital symbol/window paths:

| Policy | 11 | 23 | 47 | 71 | 101 |
|---|---:|---:|---:|---:|---:|
| ppo | -0.021 | 1.223 | 0.003 | 0.124 | 0.169 |
| double_dqn | 1.397 | 1.224 | -0.565 | -0.336 | -0.165 |
| cql | -1.452 | -1.998 | -1.804 | -0.879 | -1.893 |
| cql_alpha0 | -2.953 | -2.616 | -1.391 | 0.172 | -1.153 |
| behavior_cloning | -0.255 | 0.067 | 0.365 | -0.893 | -0.137 |

Against their frozen 512-step versions, the mean changes are +0.193 percentage
points for PPO, +0.831 for Double DQN, −1.321 for CQL, −2.036 for alpha-zero and
+0.658 for cloning. These are descriptive paired budget changes, not a selected
winning budget or evidence of convergence. PPO/Double DQN means are positive in
the first window and negative in the second. CQL is negative in both windows
and in every symbol mean. The JSON report includes all symbol/fold means.

Mean net percent under every registered stress:

| Execution | PPO | Double DQN | CQL | Normalized momentum | Constant long |
|---|---:|---:|---:|---:|---:|
| base | 0.299 | 0.311 | -1.605 | 1.487 | 0.997 |
| cost1_5x | 0.247 | 0.180 | -1.803 | 1.269 | 0.898 |
| cost2x | 0.195 | 0.049 | -2.171 | 1.051 | 0.798 |
| severe50bp | -0.118 | -0.839 | -3.485 | -0.245 | 0.201 |
| delay1bar | 0.167 | 0.186 | -0.271 | 0.669 | 0.838 |
| partial50pct | 0.368 | 0.523 | -0.977 | 1.318 | 1.174 |
| funding2x | 0.283 | 0.290 | -1.615 | 1.472 | 0.953 |

PPO and Double DQN remain positive at 2x costs but turn negative in the 50bp
severe case. Additional execution delay reduces their means. Nonzero pending
commitments were not observed in base-case training, so delay remains a transfer
stress, not validated delayed-execution training. The constant-long control is
shielded and periodically flattened, not unconstrained buy-and-hold.

Nine-bar paired block-bootstrap results, percentage points (6 comparisons,
Bonferroni intervals; all seeds/symbols averaged before chronological resampling):

| RL | Comparator | Increment | Simultaneous interval |
|---|---|---:|---:|
| ppo | cash | 0.299 | [-1.196, 2.226] |
| ppo | normalized_momentum3 | -1.188 | [-4.621, 1.613] |
| double_dqn | cash | 0.311 | [-1.264, 1.898] |
| double_dqn | normalized_momentum3 | -1.176 | [-4.603, 1.774] |
| cql | cash | -1.605 | [-3.276, -0.074] |
| cql | normalized_momentum3 | -3.093 | [-7.747, 0.878] |

All PPO/Double DQN intervals include zero at block lengths 3, 9 and 21. CQL's
cash interval is negative at 9/21 bars but includes zero at 3; its matched-control
interval is negative only at 21. Thus the negative evidence is also sensitive to
dependence assumptions. No positive lower bound or confirmation gate is passed.
Correction within this campaign does not control the cumulative adaptive search.

## Accounting, support and failure findings

The independent audit checks all **77,952 evaluation rows**, with **zero scoped
violations** at binary64 tolerance 1e-12. It reconstructs unit-price gross P&L,
funding, linear costs, nonterminal fills/turnover, reward and equity continuity.
This is implementation-conformance evidence, not a global executable-safety proof;
terminal multi-trade/liquidation sequences and unarchived training paths are not
independently reconstructed by this audit. No constant zero counter is used as
proof of safety. The inherited export's `legacyCounter` warning refers to v1.

Six training episodes report `endpoint_exposure`: fold0 PPO seed71 twice and
Double DQN seed23 once; fold1 PPO seed11 once and Double DQN seed101 twice.
They terminate economically and are retained in fit metadata. Collection starts
a new independent registered training episode afterward. Evaluation never resets
a failed account; all evaluation paths completed. The kernel checks marked
existing inventory before trading, so price/funding disturbances can breach a
state limit despite bounded prior actions. No guarantee against those disturbances
is claimed. Per-event training traces were not persisted, limiting retrospective
independent reconstruction; these counts are not silently converted to successes.

Across all evaluation policies/stresses, support projection changed 26,701
proposals and turnover admission projected 1,087 due targets: 27,788 action
changes. There were 5,568 deferred and 5,568 ignored proposals under delay.
The minimum/maximum burn-in support check is not a calibrated uncertainty or
real-log overlap measure. Unsupported raw actions can earn identical rewards
through many-to-one projection; the study does not alter likelihood labels or
redefine this away to pass usability.

PPO mean gross +0.420%, funding −0.016%, costs 0.104% reconcile to net +0.299%.
Double DQN: +0.595%, −0.021%, 0.263% reconcile to +0.311%.
CQL: −1.161%, −0.010%, 0.434% reconcile to −1.605%. Reward is the same equity
change times 100; costs are not subtracted twice. No shaped-reward ablation is
needed because no shaping was introduced. Rejected turnover previews never
advance equity or grant a reset; the prior counterexample remains a regression.

Both offline buffers retain the exact v2 content hashes and uniform proposal
counts [172,171,169] and [154,184,174], with known simulated propensity 1/3.
They contain 512 transitions each. More optimizer updates add no support.
CQL's finite-path critic MSE rises from about 1.519 to 8.000 reward-units squared;
alpha-zero reaches 9.473 and Double DQN 1.462. CQL's worsening values/returns are
consistent with an inadequate small offline-data fit; this endpoint comparison
does not identify a unique cause or prove an overfitting mechanism. No real IS,
DR, FQE or ESS estimate is invented. No recurrence, predictor, alternative-data
or reward-shaping input exists to attribute an improvement to.

## Policy cards and retained artifacts

PPO: on-policy simulated collection, categorical 12x16 actor, unchanged raw GAE,
clipped surrogate and greedy evaluation. Small positive mean; unstable across
seeds/windows; fails matched-control and intervention gates. Rejected.

Double DQN: continuing simulator collection with replay and independent target
snapshots. Improved mean but broad seed dispersion and failed matched-control/
intervention gates. Rejected.

CQL: fixed simulated-data scalar discrete conservative learner, alpha 0.1.
Every seed mean is negative; more optimization worsens economic and critic
results. Rejected. Alpha-zero is a diagnostic, not an adopted offline learner.
Cloning is a supervised control and is not counted as RL.

No policy has a production loader, authorization path or promotion eligibility.
Python warm inference is evidence about research computation only; Haskell
artifact parity, bounded operational deadlines, reservations, authentic data
availability and the current-champion comparison remain open.

The campaign took 637.297s wall time; summed fit/benchmark time
was 425.043s. Maximum RSS was 82,862,080 bytes. Across models,
the largest warm Python p99 was 0.193767ms (100 measured calls
after 10 warm-ups); largest policy JSON was 4,893 bytes. No paid
compute or persistent worker was used. Exact pins and source hashes are archived.

Archive index SHA256:
`360121fd70f03650b8934f18dac5e8d25f3f9eaa143f5da324102814ce510039`.
The archive is retained outside Git under
`/Users/diegosaa/GitHub/trader/.tmp/research/rl-training-budget-v3-20261010`.
[All-seed results](budget-v3-results/all-seed-results.csv),
[complete trial registry](budget-v3-results/trial-registry.csv),
[audit/paired evidence](budget-v3-results/evidence.json) and
[budget diagnostics](budget-v3-results/budget-comparison.json) are committed.

To reproduce the summary into a new directory:

```sh
python scripts/research/summarize_rl_budget_v3.py \
  --source /private/tmp/trader-rl-budget-v3-evidence \
  --parent /private/tmp/trader-rl-admission-v2-evidence \
  --output /private/tmp/rl-budget-v3-summary \
  --expected-index-sha256 360121fd70f03650b8934f18dac5e8d25f3f9eaa143f5da324102814ce510039
```

## Verification and delivery

`python test/rl_composition_test.py`: 21 tests passed. The canonical
`bash scripts/verify.sh full` completed with exit 0 in the exact clean/pinned
environment recorded in the [verification receipt](budget-v3-results/verification-receipt.json).
It passed 341 formal regressions, 21 RL regressions, ordinary SMT/model/Haskell
certificate reproduction, Haskell build/format/lint/smoke/tests, web typecheck,
241 web tests/build, and 185 automation tests. Web/automation summaries report
zero failed or skipped tests. No required wrapper target was omitted.

Earlier failures remain recorded: one test-mock error was fixed; an inherited
Haskell hang fixture exceeded its three-second test deadline, then passed in
isolation; an inherited accounting SMT check failed its expected-status test,
then the same gross/funding claims proved SAT/UNSAT in isolation. Its old error
message does not distinguish SAT from unknown, so the failure is reported
without inventing that distinction. The isolated gross claim took 9.279 seconds
against a ten-second limit. The final full formal record passed in 184.281s;
ordinary canonical reproduction passed in 215.892s. No timeout or failed query
was accepted as a discharged proof. These earlier timing/check failures remain
a reliability limitation. A process-name/parent check found no remaining workers
from the timed-out research fixture.

No new theorem or broad obligation closure is claimed: 20 scoped original
obligations remain closed and 18 remain unresolved. Kernel/lifecycle proofs do
not establish market profitability, authentic timestamps or production safety.
The final wrapper result validates this research implementation within the
existing assumptions; it does not satisfy the broader mission acceptance gates.

Branch: `research/rl-exposure-v1-20261010`. Local draft body, logs and a patch
from the verified-main base are retained at
`/Users/diegosaa/GitHub/trader/.tmp/research/rl-training-budget-v3-delivery-20261010`.
No push or PR: the six-hour scheduled autoloop scans all PR states, including
drafts, and may turn review feedback into a direct main push and deployment
(`scripts/autoloop.mjs`, lines 37, 67, 824 and 1758; `.github/workflows/ci.yml`,
deploy guard). This production-connected automation was preserved.
The original workspace's pre-existing user edits remain untouched.

RL research: this registered budget study is complete; the broader mission is
partially completed. RL adoption: reject this package, retain the existing
production path. Formal assurance: scoped checks passed; the 18 broad obligations
and operational/data evidence remain open. Financial execution: no real orders,
live exploration or new live authorization; existing production activity preserved.

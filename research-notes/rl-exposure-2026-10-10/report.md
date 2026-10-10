# Trained RL composition pilot: no challenger qualifies

PPO, Double DQN and discrete CQL were actually trained with five seeds in each
of two historical development folds. This completes a runnable research vertical
slice, not the larger project's confirmation or integration requirements. Every
RL family has failed baseline-cost paths, and the primary paired comparison is
**insufficient evidence**. Preserve the current champion; do not integrate these
policies, including as automatic disabled-challenger selections.

Target exposure is sequential because inventory earns the following interval's
price/funding PnL and changing it incurs costs; delayed intents also constrain the
next decision. One common policy controls independent BTC, ETH and SOL USD-M
perpetual accounts on eight-hour bars. There is no portfolio-allocation or market
impact claim. The [contract](contract.md), [registration](../registrations/rl-causal-composition-v1.json)
and [50-paper review/matrix](review.md) specify the design and its limitations.

## Experiment identity and chronology

Verified main: `354c853176b7ee2ac6570951f7f6f2d0aed6909d`, successful
[CI 37959950956](https://github.com/diegueins680/trader/actions/runs/37959950956).
Registration commit: `e64cde6b`. Frozen experiment implementation:
`e29f8fbaa4812ff62734ed463c0638d67f7b146d`. All algorithm/environment code was
committed before the run; there was no tuning, checkpoint selection or best-seed
selection after inspecting outcomes. Later changes correct reporting labels and
add independent trace checks; they do not change learning or economics.

The input was the previously exposed, hash-admitted 4,910-bar development export,
with event funding settlements, not freshly collected or first-seen historical
data. Provider terms apply; raw data are not committed. Available rows extend
from 2020-09-23 through 2025-03-17. Only the registered training/evaluation prefixes
were used for learning and outcomes. The primary windows are:

| Fold | Last permitted training close, UTC | First decision close, UTC | Terminal close, UTC |
|---|---|---|---|
| 0 | 2022-03-10 07:59:59.999 | 2022-03-12 15:59:59.999 | 2022-04-02 23:59:59.999 |
| 1 | 2023-02-06 15:59:59.999 | 2023-02-08 23:59:59.999 | 2023-03-02 07:59:59.999 |

There are 64 transitions per symbol/fold. Training receives physically sliced
prefixes; the reward must mature inside them. Each episode freezes a scale fitted
on 96 bars strictly before its first decision. No predictor is trained on future
labels. Evaluation weights are frozen. Availability is **assumed close+1ms**;
authentic revision/publication witnesses are absent. These development periods
were already consumed by earlier adaptive research, including the September RL
screen. Neither the sealed 1,227-return holdout nor prospective carry outcomes
through 2027-01-20 were read. No fresh confirmation evidence is claimed.

The exact current adopted-combo baseline is unavailable in verified main for
these historical dates. The separately audited current fleet and PR330 do not
supply a chronology-valid matched champion here. Consequently comparison A in
the requested A/B/C/D design remains missing. Cash is a control, not a renamed
champion. The observation-matched deterministic and cloning controls below are
implemented and executed under identical accounts, observations, ledger and limits.

## Learners, controls and all-seed outcomes

PPO collects fresh rollouts, uses the clipped proposal-likelihood surrogate,
raw GAE targets and a separate value network. Double DQN collects fresh
epsilon-greedy simulator experience while reusing replay and an independent target
snapshot. CQL uses a frozen simulated uniform-proposal dataset and the corrected
scalar logsumexp conservative objective. It never collects during offline updates.
The CQL alpha-zero ablation is ordinary offline Double DQN, explicitly diagnostic.
Behavior cloning fits the same frozen proposals; it is not RL. All networks have
12 inputs and 16 tanh hidden units. Seeds are 11, 23, 47, 71 and 101.

All 50 fits completed: 30 primary RL fits, 10 alpha-zero ablation fits and 10 BC
fits. Each receives 512 transitions/updates according to its registered regime.
This is a pilot budget, not evidence of convergence or a serious-finalist campaign.
Training plus per-fit benchmarking took 311.91 seconds; the complete run took
1,079.13 seconds on the existing CPU. Every one of 1,268 planned entries has one
terminal record: 50 fits and 1,218 replays. Of the latter, 904 completed their
window and 314 stopped with `turnover_limit`; none was silently discarded.

The table reports mean **terminal-or-stopped** net change across six symbol/fold
paths per seed. Parentheses give the number of stopped paths. A number with a
stopped path is not an equal-horizon performance estimate and must not be used to
rank methods. Percentages are unannualized changes from initial equity one.

| Method | Seed 11 | Seed 23 | Seed 47 | Seed 71 | Seed 101 |
|---|---:|---:|---:|---:|---:|
| PPO | -1.1238% (0) | -0.6100% (2) | +0.9053% (0) | -0.0488% (0) | +0.0445% (2) |
| Double DQN | -2.3647% (1) | -0.2799% (5) | +0.3013% (0) | +0.8048% (4) | -3.4900% (3) |
| CQL | -0.3764% (2) | -0.8032% (3) | -0.6879% (2) | +0.4460% (3) | -0.0869% (3) |
| CQL alpha=0 | +0.0300% (2) | -0.7641% (2) | -0.6115% (1) | -0.8239% (2) | -0.8274% (2) |
| Behavior cloning | +0.1632% (2) | +0.6069% (2) | +1.0804% (0) | +0.4418% (2) | +0.1277% (1) |

Cash completed all six baseline paths at 0%; constant-long completed all six at
mean +0.9975%. Momentum and reversal each stopped five of six paths; their
terminal/stopped means (+1.5070% and -1.8517%) are not comparable full-window returns.
No selected surviving seed is promoted. Positive BC or constant-long results on
these two exposed windows do not establish a trading recommendation.

| Base-cost diagnostic, across all seeds and paths including stops | PPO | Double DQN | CQL |
|---|---:|---:|---:|
| Failed / planned paths | 4 / 30 | 13 / 30 | 13 / 30 |
| Mean stopped/terminal net change | -0.1666% | -1.0057% | -0.3017% |
| Mean gross price PnL / initial equity | -0.0157% | -0.7910% | -0.1972% |
| Mean funding PnL / initial equity | -0.0045% | +0.0075% | +0.0039% |
| Mean fees + spread + slippage / initial equity | 0.1464% | 0.2223% | 0.1083% |
| Worst observed drawdown | 5.179% | 6.485% | 2.951% |
| Worst observed one-bar ES95 | 1.559% | 1.900% | 0.751% |
| Mean absolute exposure | 0.0738 | 0.0831 | 0.0788 |
| Mean total turnover / initial equity | 1.4640 | 2.2228 | 1.0832 |
| Mean fill events including liquidation | 19.43 | 16.57 | 6.10 |
| Mean drawdown duration, bars | 27.90 | 27.80 | 27.93 |
| Registered intervention-flag rate | 58.25% | 53.16% | 40.32% |

The registered intervention flag counts unsupported-state handling even when the
proposal is already zero, and pending reservations/consumption in the delay case.
It is deliberately not described as the fraction of unsafe proposals. Across all
63,309 recorded transitions, a separate trace calculation finds 26,795 changed or
deferred proposals. Per-path distributions, holding periods, critic errors and
exact stage traces remain in the hash-indexed archive. Critic finite-path MSE is
a diagnostic under a 32-to-64-step horizon shift, not value calibration for live
markets. No recurrence, learned predictor, shaping or alternative data was added;
their removal is therefore not a meaningful ablation. CQL's alpha-zero results
do not establish a regularization benefit because stopped paths differ.
The flag conflates observation-support handling with changed actions. A future
campaign needs a separately registered usability metric; the pilot's threshold
must not be reinterpreted after seeing outcomes. In particular, flat proposals
in unsupported states are not evidence that a no-trade policy is unsafe.

## Costs, delays, uncertainty and support

Baseline one-way execution costs are 5bp fee + 0.5bp half-spread + 4.5bp slippage.
The severe case multiplies all three by five, totaling 50bp. Prices/funding are
exogenous; old units earn the interval's return and funding before a following-close
fill. This is price-taking counterfactual replay, not order-book execution evidence.
Delay adds one bar with a single pending slot. Partial-fill stress fills half the
remaining target change per opportunity, with full terminal liquidation assumed.

Each cell below is stopped/terminal mean net change; brackets give stopped paths
out of 30. Only rows without stops have equal horizon, and stress results were not
registered for selecting a new candidate.

| Stress | PPO | Double DQN | CQL |
|---|---:|---:|---:|
| Base | -0.1666% [4] | -1.0057% [13] | -0.3017% [13] |
| 1.5x costs | -0.2397% [4] | -1.1163% [13] | -0.3220% [13] |
| 2x costs | -0.3128% [4] | -1.2267% [13] | -0.3760% [13] |
| Severe 50bp | -0.7500% [4] | -1.8065% [14] | -0.7161% [13] |
| Extra one-bar delay | -0.3415% [1] | -0.2933% [10] | -0.5228% [2] |
| 50% partial fills | -0.2450% [0] | -0.7444% [0] | -0.9797% [0] |
| Twice event funding | -0.1709% [4] | -0.9982% [13] | -0.2978% [13] |

The primary simultaneous paired block intervals were **not calculated** because
all six RL/control comparisons contain failed paths. Inventing continuation or
dropping stopped paths would change the preregistered estimand. DSR, PBO and
confirmation-level risk-adjusted statistics are insufficient evidence. Five seeds
are optimizer replications, not five market histories. Results by symbol/fold/stress
are in [the complete trial registry](results/trial-registry.csv); there is no
registered regime classifier, so no post-hoc winning regime is manufactured.

The frozen buffers contain 512 simulated transitions each, with proposal counts
(169,171,172) and (157,181,174) and known uniform proposal propensity 1/3. These
counts do not establish support over financial states, especially after shielding.
There are no real behavior logs or defensible real propensity ratios, so real-log
IS, DR, ESS and FQE are unavailable. No clipped ratio or invented propensity fills
that gap. Missing/stale data, corrupt numeric state and artifact failures have
synthetic regression evidence; realistic outages, exchange precision/min-notional,
liquidity, gap liquidation and calibrated impact remain open stress requirements.

## Accounting, failures and formal assurance

Reward is 100 times the exact rational equity increment. Each step asserts
`delta = gross + funding - costs` before binary64 conversion; terminal costs occur
once. The proposal action/log probability is retained through many-to-one shielding,
so executed-action rewards are not attached to incompatible relabeled likelihoods.
Rollout cuts preserve state; economic terminals stop and liquidate. Delay carry
does not perform a cost-free rebalance. The independent post-run audit checked
63,309 rows for ledger/reward continuity, action domains, causal decision/fill
ordering, nonterminal turnover/exposure bounds and reported flat liquidation at
tolerance 1e-12: **zero violations in that limited audit**. It does not reconstruct
units/prices independently or prove production safety. The archive's original
`unsafeExecutableIntents=0` field was a constant, is not evidence, and has been
removed from future reports. The immutable original is retained with this correction.

Turnover failures arise when marking/funding moves an existing exposure far enough
that an opposite 0.25 target would require more than the admitted 0.50 one-step
turnover. The kernel rejects that rebalance and closes the simulation. These are
failed economic paths, not executed oversized target orders. They expose a poor
policy/shield composition and cannot be rewarded as favorable early exits. Training
resets after such recorded economic terminals; this pilot has not ruled out a
learner exploiting the shortened task. That and the evaluation horizon shift block
strong claims even where a seed's net endpoint is positive.

New evidence in the existing [assurance ledger](../../formal/research/proof-ledger.json):

| Claim | Method and scope | Remaining correspondence gap |
|---|---|---|
| F-RL-COMPOSE-ARITH | Seven Z3 SAT-premise/UNSAT-counterexample checks: bounded projection, reward identity/telescoping, time order, training cutoff, pending count and projected probability mass | Rational/real algebra and integer indices; not binary64 neural-network proof or timestamp authenticity |
| F-RL-COMPOSE-FLOW | Exhaustive finite transition model: 43 states, 107 edges, two proposal identities, eight ticks, one owner, no authority edge | Model-to-code mapping by review/tests; no production concurrency, durable recovery or liveness theorem |
| F-RL-COMPOSE-SOURCE | Reviewed file hashes and direct-effect-name checks | Trusted interpreter, NumPy, kernels and transitive dependencies; not OS capability confinement |
| F-RL-COMPOSE-CONFORMANCE | Eight exact optimized/public causal-v3 transitions and eight actual Haskell Rational ledger oracle cases; targeted regression suite | Selected tests, not universal compiler/refinement proof |

Z3 is pinned to 4.15.4.0; GHC 9.4.8, CPython 3.13.3 and NumPy 2.3.5 are the checked
toolchain. Unknown/timeout/counterexample is failure. Models check reachable initial
states and representative transitions. There are no added axioms or admissions.
Existing kernel lemmas are reused without calling them a whole-system proof. The
original 38-obligation registry remains 20 scoped closed, 17 partial and one open;
this work closes none of those broader obligations by counting small lemmas.

| Requested boundary | Current affected-path evidence / open gate |
|---|---|
| FM-01 causality | Prefix construction and future-perturbation tests; authentic availability/revisions open |
| FM-02 partitions | Physical training prefixes and integer cutoff lemma; development contamination disclosed |
| FM-03 accounting | Exact kernel assertions, SMT identities and Haskell cases; exchange ledger correspondence open |
| FM-04 validation | Strict integer actions, finite shaped observations and snapshot bytes; no order constructor supplied |
| FM-05 admission | Three targets, support projection and exact replay risk checks; venue rounding/min-notional/portfolio admission open |
| FM-06 authorization | No production imports/client/credentials in learner; trusted-process isolation assumption, no OS theorem |
| FM-07 lifecycle | Explicit opt-in, research-only artifacts, finite no-promotion model; no new production consumer |
| FM-08 failure | Invalid state rejects and failed trials retained; no certified timeout or live fallback composed |
| FM-09 atomicity | Single-owner immutable session and one pending slot; shared concurrent reservations unimplemented |
| FM-10 compatibility | Versioned observation/environment, source/data hashes and numeric snapshot admission; production artifact loader absent |
| FM-11 noninterference | Existing production code/config unchanged, entry point disabled; whole-process noninterference not proved |
| FM-12 recovery | Terminal-state/retry model and unique trial IDs; interrupted training archive recovery remains open |

## Policy cards, engineering and decision

All three policy cards share mode `research_only`, initial equity one, the
three-target action contract, registered data/partitions and last-checkpoint rule.
PPO is an on-policy stochastic-training/greedy-evaluation proposal policy; DDQN is
an off-policy simulator greedy Q policy; CQL is a fixed-simulated-dataset greedy Q
policy. They require the new 12-input schema and cannot reuse an old nine-input
artifact identity. None has an approved production operating region. Reasons for
rejection from integration are failed paths, excessive registered intervention
flags, incomplete champion comparison, contaminated data and open integration proofs.

Warm Python inference was measured with 10 warmups and 100 calls per fitted model.
The maximum across per-model p99 measurements was 8.1784ms; the full research
process peaked at 84,361,216 bytes RSS. This was shared-host timing, not a hard
deadline, concurrent service or Haskell artifact-loading benchmark. Parameters are
small external research JSON artifacts; there is no new production Python service
or Haskell policy loader. Existing Haskell safety/orchestration remains untouched.
The requested end-to-end Haskell artifact, timeout and concurrency gates are open.

Research implementation: **partially completed** at project scope; the registered
pilot and trained comparison are complete, while rigorous confirmation, complete
execution fidelity and integration assurance are not. Adoption: **continue research**;
no candidate qualifies. A useful next registration would address failure-state
semantics and support before increasing the training budget, and require lawful
champion/data witnesses before confirmation. No adaptive successor is run here.

Financial execution: no real orders, no live exploration, no new live authorization,
no deployment or merge, and no changes to production fleet, risk limits, ownership
or authorization. Existing production activity was preserved, not declared disabled.

## Reproduction and evidence

The small committed [evidence receipt](results/evidence.json),
[all-seed/stress table](results/all-seed-results.csv) and
[trial registry](results/trial-registry.csv) are exported after verifying every
archived file and reconciling the planned trial inventory. External local archive:
`/Users/diegosaa/GitHub/trader/.tmp/research/rl-causal-composition-20261010`;
its `index.json` SHA256 is
`67cdc6f16dadf28cd8323e20a3dee18a5e62ff67babfe5c906f587431e4d096c`.
It contains 47MB of model bytes, simulated buffers and raw traces and is not in
Git or a remote artifact store. The original execution copy remains at
`/private/tmp/trader-rl-evidence-20261010`; the retained research copy was verified
against the same index after copying. The inherited data-export instructions and hashes
are in the parent registration; exact reproduction requires the admitted bytes,
not a fresh API download that may revise them.

Archived `trainingDatasetSha256` labels the common simulated reference buffer;
only CQL/alpha-zero/BC train on it. PPO/DDQN collect fresh trajectories from the
manifest's market-data training prefixes. Future output names this
`simulatedReferenceDatasetSha256` and identifies the actual training regime.
This metadata correction does not rewrite the frozen archive or its source SHA.

With the repository's pinned tools and `scripts/formal/requirements.txt` installed:

```bash
python test/rl_composition_test.py
python scripts/formal/rl_composition_check.py
bash scripts/verify.sh formal
bash scripts/verify.sh full

# From a committed checkout; use an empty, isolated output directory.
env -i PATH="$PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/research/run_rl_composition.py \
  --run-registered-development-v1 \
  --panel /path/to/registered-development-panel.csv \
  --settlements /path/to/registered-development-settlements.csv \
  --output /path/to/new-research-archive

python scripts/research/summarize_rl_composition.py \
  --source /Users/diegosaa/GitHub/trader/.tmp/research/rl-causal-composition-20261010 \
  --expected-index-sha256 67cdc6f16dadf28cd8323e20a3dee18a5e62ff67babfe5c906f587431e4d096c \
  --output /path/to/new-small-export
```

Use frozen commit `e29f8fba` for the exact original archive schema. Current code
changes reporting labels only. Deterministic evaluation is tested; repeatability
is scoped to pinned runtime/hardware with 1e-12 numerical comparison tolerance,
not claimed bitwise across arbitrary BLAS/platforms. Verification attempts and
their final outcomes are recorded in `verification.md` beside this report.


## Later baseline-fidelity correction

The admission-v2 continuation found that raw momentum/reversal read unnormalized
history, while the RL vector omits the normalization parameters. Therefore the
v1 raw-rule comparison is not exactly observation-matched. Its archived results
and stopped-horizon rejection remain unchanged. See the [successor report](admission-v2.md)
for a separately registered, explicitly adaptive normalized-observation control
comparison using frozen trained policies. No holdout or production selection changed.

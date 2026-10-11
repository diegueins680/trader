# Frozen offline-data sensitivity

This adaptive development experiment tests whether the v3 CQL result depends
on a small simulated behavior dataset. It increases the frozen dataset from
512 to 4,096 transitions per fold while keeping 4,096 optimizer updates,
learning rates, action semantics, observations, costs, windows and seeds fixed.
It is not a new algorithm search or untouched confirmation.

## Executed results and decision

All **30 fits and 798 replays completed**, covering all 64 transitions in every
evaluation path. All 168 deterministic baseline records match v3 exactly.
There are 828 reconciled trial IDs and no missing/replaced seeds or failed fits.
The full campaign took 255.323 seconds; fitting plus warm inference benchmarks
summed to 101.606 seconds. Process peak RSS was 83,009,536 bytes; largest policy
JSON was 4,876 bytes. The maximum model p99 over 100 warmed Python inference
calls was 0.2072 ms (ten warmups), not an operational deadline guarantee.

Mean net returns below average the same six symbol/fold paths per seed, then
all five seeds. They are finite-window returns, **not annualized**. Worst drawdown
and ES are maxima across evaluated seed/symbol/fold paths.

| Newly fitted policy / deterministic control | Mean net | Seed means min–max | Worst DD | Worst ES95 | Mean intervention |
|---|---:|---:|---:|---:|---:|
| behavior_cloning | -0.966% | -1.404% to -0.020% | 7.082% | 1.725% | 39.948% |
| cash | +0.000% | +0.000% to +0.000% | 0.000% | -0.000% | 0.000% |
| constant_long | +0.997% | +0.997% to +0.997% | 4.480% | 1.559% | 60.156% |
| cql | +0.656% | +0.126% to +1.162% | 4.480% | 1.559% | 34.479% |
| cql_alpha0 | +0.696% | +0.186% to +1.152% | 4.480% | 1.559% | 33.490% |
| normalized_momentum3 | +1.487% | +1.487% to +1.487% | 2.655% | 1.162% | 64.844% |
| normalized_reversal3 | -2.376% | -2.376% to -2.376% | 4.952% | 1.713% | 64.844% |

Frozen v3 PPO (+0.299%) and simulator Double DQN (+0.311%) are reference results,
not new v4 fits. CQL's larger-data mean improves by **2.261 percentage points**
from v3 (−1.605%), but remains 0.832pp below matched momentum and 0.342pp below
constant-long. Alpha-zero improves by 2.284pp to +0.696%, while cloning worsens
by 0.795pp to −0.966%. The near-equal CQL/alpha-zero result does not demonstrate
value from the conservative penalty. The fixed updates and expanded state
sample/reuse pattern matter empirically; they do not establish convergence.

All five CQL seed means are positive: seed 11 +0.126%, 23 +0.502%, 47 +1.067%,
71 +1.162%, 101 +0.423%. Their sample SD is 0.443pp, an algorithmic dispersion
measure, not a standard error over independent markets. Fold means are +2.292%
and −0.981%; BTC +1.264%, ETH +0.348%, SOL +0.355%. The negative second window and
large paired uncertainty prevent a stable-regime advantage claim.

| Stress | CQL | Alpha-zero | Cloning | Matched momentum | Constant long |
|---|---:|---:|---:|---:|---:|
| base | +0.656% | +0.696% | -0.966% | +1.487% | +0.997% |
| cost1_5x | +0.575% | +0.617% | -1.095% | +1.269% | +0.898% |
| cost2x | +0.495% | +0.538% | -1.229% | +1.051% | +0.798% |
| severe50bp | +0.012% | +0.067% | -1.978% | -0.245% | +0.201% |
| delay1bar | +0.203% | +0.255% | -0.651% | +0.669% | +0.838% |
| partial50pct | +0.791% | +0.894% | -0.880% | +1.318% | +1.174% |
| funding2x | +0.622% | +0.661% | -0.960% | +1.472% | +0.953% |

Costs are 10bp per unit turnover at base, 15bp/20bp in the 1.5×/2× cases,
and 50bp in the severe case. One additional bar of delay reduces CQL to +0.203%;
the severe case leaves only +0.012%. Partial fills sometimes improve outcomes:
that is sensitivity to modeled execution, not evidence of better fill quality.
Mean CQL gross PnL +0.850%, funding −0.0335% and execution costs 0.1604% reconcile
to net +0.656%; fees are not subtracted again from equity reward. Mean absolute
exposure is 7.474%, mean turnover 1.604 initial-equity units, and mean nonzero-fill
row count 22.233 (the existing metric includes liquidation in its final row,
not a count of distinct exchange orders). Longest drawdown duration is 64 bars.
Holding-period lists and full exposure/proposal distributions remain archived.

Nine-bar paired intervals (percentage points) are CQL minus cash +0.656
[−2.020, +3.383] and CQL minus matched momentum −0.832 [−4.526, +2.164]. All six intervals
for block sizes 3/9/21 include zero. The materiality-plus-positive-lower-bound
gate fails. CQL intervention 34.479% also exceeds the unchanged 20% gate; even the
better-return deterministic control fails that gate. Neither is promoted.

Across every stress and policy, 19,549 actual action changes comprise 19,263
support projections and 286 turnover rejections. Deferred and ignored-new-proposal
counts are 3,648 each and are reported separately. The independent audit examined
**51,072 rows with zero detected violations** of its finite, tolerance 1e-12
trace checks. This is not a proof of every executable-intent property. The
exporter's inherited constant-counter warning refers to v1; no such constant
is being used as v4 safety evidence. Reward/accounting audit found no double
counting or hidden evaluation reset. Training event-level traces remain absent;
full evaluation traces, frozen buffers and generation failure reasons are retained.

## Frozen-buffer and offline diagnostics

Both original 512-transition prefixes match exactly. The full content hashes are:

- Fold 0: `bd4ac09bdcdf05c013fe33c3c0bdbbe059f7e3935966a68220205008e20e9f60`.
- Fold 1: `b2fb619b2f451857d6f8ab8624bbd77896b25023dcef19e963f71f332e4bb95a`.

Raw short/cash/long proposal counts are [1356,1389,1351] and [1302,1436,1358].
Unique 12-component observations are 4055/4096 and 4096/4096. Causal burn-in-range
support fractions rise from 55.27% to 63.84% and 52.34% to 54.69%; these are not
calibrated behavior-policy overlap statistics. Uniform proposal support alone
cannot establish state-distribution coverage of evaluation policies. The pending
input stays zero in all training data: delayed nonzero commitments remain an
out-of-training-state stress. Detailed feature ranges are in the diagnostics.

Fold 0 has one `endpoint_exposure` economic termination during data generation;
fold 1 has none. Terminal transitions are 128 in each full dataset, including
the risk termination. The collector begins a new training episode only after a
terminal transition, with the loss already represented. All 30 offline fits
record zero newly collected transitions. No economic termination was silently
removed, and no failed evaluation path was restarted.

Held-path critic MSE (squared reward units) drops from 8.000 to 1.988 for CQL and
from 9.473 to 1.947 for alpha-zero. It remains a descriptive finite-path diagnostic,
not FQE, a profitability criterion or an independently calibrated uncertainty
bound. Cloning exploratory proposals carries no reward-improvement objective. Its
negative result remains in the comparison; no replacement baseline was selected
after inspecting outcomes.

## Evidence archive and policy card

Archive index SHA256:
`a60a7d4850a8e261bf856a14e412c681eb124620b6c22d191a0f382b06a8a257`.
Original archive: `/private/tmp/trader-rl-offline-data-v4-evidence`.
Durable local copy:
`/Users/diegosaa/GitHub/trader/.tmp/research/rl-offline-data-v4-20261011`.
[All-seed metrics](offline-data-v4-results/all-seed-results.csv),
[trial registry](offline-data-v4-results/trial-registry.csv),
[audit and paired intervals](offline-data-v4-results/evidence.json), and
[buffer/seed/stress diagnostics](offline-data-v4-results/offline-data-comparison.json)
are committed. Large traces, datasets and policy parameters stay outside Git.

```sh
python scripts/research/summarize_rl_offline_data_v4.py \
  --source /private/tmp/trader-rl-offline-data-v4-evidence \
  --parent /private/tmp/trader-rl-budget-v3-evidence \
  --output /private/tmp/rl-offline-data-v4-summary \
  --expected-index-sha256 a60a7d4850a8e261bf856a14e412c681eb124620b6c22d191a0f382b06a8a257
```

Policy package: `rl-offline-data-v4` campaign with unchanged
`rl-admission-continuation-v2` environment and compatible research observation
schema. CQL and all diagnostic seeds remain research-only, disabled by default,
with no Haskell production loader, live-order capability or automatic selection.
Suitable use: reproduction of this bounded price-taker development experiment.
Prohibited inference: evidence of live profitability, current-champion superiority,
real OPE, guaranteed safety under arbitrary shocks or permission to deploy.
Decision: **reject integration of this package**; the larger dataset is useful
research evidence, but neither CQL advantage nor operational readiness is shown.

## Registration and implementation

The [registration](../registrations/rl-offline-data-v4.json) was committed as
`74dce811` before generating larger buffers or inspecting new outcomes. The
experiment source was frozen as `198eceb68486acf50a172db219f56adfd5d19eed` after
24 targeted tests and full formal certificate recording passed. The verified
starting main remains `354c853176b7ee2ac6570951f7f6f2d0aed6909d` ([successful
CI](https://github.com/diegueins680/trader/actions/runs/37959950956)).
The directory date identifies the ongoing campaign; execution continued across
the October 10–11 UTC boundary.

There are 30 new fits: CQL, CQL alpha-zero and behavior cloning, five seeds
(11, 23, 47, 71, 101), two folds. Only CQL is a suitable offline RL learner here;
alpha-zero is a diagnostic ordinary offline Double DQN, and cloning is supervised.
PPO and simulator Double DQN retain the frozen v3 results; neither is retrained
or selected on v4 outcomes. All learners use their final checkpoint. No masked
actions, new reward penalties, recurrence, predictor inputs or new decoder were
introduced. Changing sample count at fixed updates also changes reuse per
sample; this study does not isolate diversity from reuse frequency.

CQL uses the existing 12-input/16-hidden-unit tanh value network, online-action
selection/target evaluation, target synchronization every 100 updates, batch 64,
learning rate .001 and alpha .1 logsumexp penalty. The alpha-zero diagnostic and
BC use the same frozen buffers. Uniform proposal behavior has known probability
1/3 **before** the many-to-one shield. Both full buffers are collected before any
fit, with the same generator/seed as v2/v3. Each original 512-transition prefix
must reproduce its registered hash. The full dataset hash is persisted before
its fold is trained, checked after training and independently checked from NPZ.
Offline training receives a factory that rejects every attempted interaction.

The [v2 environment contract](admission-v2.md) is unchanged: sequential targets
−25%, 0, +25% affect future inventory, costs and risk admission. Decisions use
completed-close information and fill at the following close. Exact bounded
rational accounting marks the old position and books funding before fill;
training reward is 100 times net equity change, without shaping or cost double
counting. An immutable turnover-rejected preview is discarded; a fresh costed
zero target is evaluated from the original state. Genuine economic failures
terminate. Gamma 1 targets the finite economic sum. Training episodes are 32
transitions; evaluation is 64, a disclosed horizon shift. Partial fills and the
one-slot delay model are stress assumptions, not venue calibration.

## Data and statistical limits

Inputs are only the registered public Binance development exports, BTC/ETH/SOL
8h, with panel SHA256 `09a09f0e1065733be623625fa0d67e6a67dad92c53cd8fa92b6d9caa1040674a`
and settlement SHA256 `2ccba74489f96b9ce0e58842594a524b7c306012086443cbea7305d4011ee899`.
Provider terms apply; raw data and learned parameters are outside Git.
Training prefixes stop at 1600 and 2600; evaluation ranges 1606:1671 and 2606:2671
are exclusive-stop, with 64 transitions per symbol/fold. Scaling uses each
episode's strictly pre-decision 96-bar history. Label/reward maturity stays within
the training prefix. Availability is assumed close+1ms, without authentic
historical first-seen witnesses. The two windows are March–April 2022 and
February–March 2023, already consumed development data. Neither the protected
1227-return holdout nor prospective carry outcomes through 2027-01-20 was opened.

Cash and constant-long require no additional observations; normalized momentum
and reversal read only the sign of observation[1]. They are burn-in-relative
signals, not raw trend. The summarizer requires all 168 deterministic stress
replays to match v3 records exactly. The actual current champion remains
unavailable as a causal matched historical baseline; no proxy substitutes for it.

Primary new comparisons are CQL versus cash and normalized momentum. Circular
blocks 3/9/21, 2000 draws, seed 20261010 retain the conservative six-comparison
Bonferroni endpoints even though only two new primary contrasts are tested.
Seeds/symbols are averaged before chronological resampling, not counted as
independent market histories. Intervals are descriptive under weak dependence
assumptions on two short exposed windows. Data-size deltas are descriptive;
DSR/PBO and cumulative adaptive-selection control remain insufficient.
Materiality is .001 net equity change with positive lower bound. Drawdown ≤15%,
ES95 ≤2%, mean intervention ≤20%, zero unsafe executable intents and no failed
paths remain required, together with the missing champion/confirmation gates.

No real logged behavior or real action-overlap evidence exists. Simulated
uniform proposals and burn-in support flags do not justify real IS/DR/FQE.
Replay is model-based counterfactual evidence conditional on exogenous prices,
funding and fill/cost assumptions. Disturbance, precision, minimum-notional and
liquidity calibration remain open. No historical replay guarantees future
solvency, drawdown, execution or protection under arbitrary gaps/outages.

## Reproduction

Use CPython 3.13.3, NumPy 2.3.5, Z3-solver 4.15.4.0 and the pinned repository
Haskell/Node toolchains. From frozen source:

```sh
env -i PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:/private/tmp/trader-rl-tools/bin:/Users/diegosaa/.ghcup/bin:/Users/diegosaa/.local/bin:/usr/local/bin:/usr/bin:/bin \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/research/run_rl_offline_data_v4.py \
  --run-registered-offline-data-v4 \
  --panel /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-panel.csv \
  --settlements /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-settlements.csv \
  --output /private/tmp/trader-rl-offline-data-v4-evidence
```

Choose a fresh output directory; never overwrite frozen archives. Deterministic
CPU seeds and single BLAS/OMP threads constrain nondeterminism; no cross-hardware
bitwise guarantee is claimed. Existing failed v1 results and v2/v3 evidence stay
frozen and available. This adds 30 fits to 150 earlier campaign fits, 180 total,
plus the separately registered 84 control-only replays. Earlier repository
adaptive trials remain part of contamination/selection history, not hidden.

## Assurance and operational status

The unchanged composed core has seven nonvacuous SMT lemmas plus two zero-target
admission lemmas, a 43-state/107-transition finite lifecycle model bounded at 8
steps and eight exact Python/Haskell ledger witnesses. Arithmetic domains,
bounds and assumptions remain in the [assurance ledger](../../formal/research/proof-ledger.json).
The source lock now covers 19 reviewed files. The new prefix/subset/offline guard
checks are conformance and identity evidence, not new mathematical theorems.
Twenty scoped original obligations remain closed and 18 unresolved. Floating
trace audits do not constitute universal compiler/runtime refinement; terminal
combined fills/liquidation are not independently reconstructed in full.

Research artifacts have no production loader. Python warm-inference timing
cannot establish Haskell parity, artifact-load deadlines, concurrency reliability
or operational memory bounds. Missing/stale/artifact failures have fixture-level
coverage, not completed historical outage or live shadow campaigns. Integration
remains blocked by operational, data and economic gates; no persistent service
or scheduled training is started.

The publication blocker is unchanged: six-hour scheduled autoloop scans all PR
states and can turn review feedback into a direct default-branch push, which
triggers deployment. See `scripts/autoloop.mjs` lines 37, 67, 824, 1758 and the deploy
guard in `.github/workflows/ci.yml`. No push, PR, merge or deployment is performed.
Local commits and a review patch preserve the work. Pre-existing user edits in
the original workspace remain untouched. Existing production activity is
preserved; this experiment introduces no live authority.

## Verification and delivery

`python test/rl_composition_test.py` passed all 24 targeted tests. The scoped
composition checker passed; `python scripts/formal/verify.py --record` passed
in 201.520s. Canonical `bash scripts/verify.sh full` exited 0: 341 formal
regressions (195.970s), 24 RL regressions (4.931s), ordinary certificate
reproduction (225.101s), Haskell build/fourmolu/hlint/smoke/cabal tests, web
typecheck/241 tests/build, and 185 automation tests. Web and automation report
zero failures/skips. No required wrapper target was omitted or weakened.

The [receipt](offline-data-v4-results/verification-receipt.json) gives the clean
pinned environment, command results, log hashes and earlier failed attempts:
an incomplete test fixture and stale reviewed source hashes were corrected;
a summary invocation with a placeholder digest was rejected, then rerun with
the actual archive identity. These were not failed training trials. No solver
timeout/unknown was counted as a proof. Prior v3 timing/check failures remain
in that report and are not erased by this successful run.

The raw archive copy contains 836 indexed files / 61,268,561 indexed bytes;
every copied file and the index were hash-verified. Local review delivery:
`/Users/diegosaa/GitHub/trader/.tmp/research/rl-offline-data-v4-delivery-20261011`
contains the report, small results, log files, draft body, delivery manifest
and cumulative Git patch. Branch: `research/rl-exposure-v1-20261010`.
No draft PR was opened because publication isolation remains unresolved.

RL research: this registered offline-data study is complete; the broader
mission remains partially completed. RL adoption: reject integration of this
package; retain the existing production path. Formal assurance: scoped
certificates and conformance passed, with 18 broad obligations unresolved.
Financial execution: no real orders, live exploration or new live authority;
existing production activity is preserved.

# Current-main RL replication, 2026-10-13

The decision remains **no RL adoption**. This continuation reuses the existing
causal target-exposure research implementation and reproduces its five-seed
budget study on verified main. It is engineering replication of consumed
development evidence, not a new independent financial trial. It cannot establish
champion superiority: the historically faithful champion export is unavailable.
Research continues with this adoption blocker explicitly retained.

The starting SHA, recorded before changes, is
`0b76f2b879f75f5c669fe944a14ea6700d446240`. GitHub's `CI` workflow reported success
for this exact main commit on discovery ([run 38313512581](https://github.com/diegueins680/trader/actions/runs/38313512581)). Branch:
`research/rl-causal-campaign-2026-10-13`. The 15 research-only commits through
`f512d7a8` were transplanted from the existing RL branch; original registrations,
trial identities, failed policies and source SHAs are retained. These are reused
deliverables, not claimed as newly authored or newly observed research. The
separate deployment changes in existing [PR342](https://github.com/diegueins680/trader/pull/342)
were excluded. The accounting sign-case proof decomposition from `1890739c` is
reused with its original premises, claim and ten-second per-query deadline.

The [replication registration](../registrations/rl-reproduction-2026-10-13.json)
was committed as `8e83b2d9` before the new run. It freezes all five seeds
(11, 23, 47, 71, 101), 50 fits, 1,218 stress replays, the original checkpoint
rule, and absolute/relative comparison tolerances of 1e-12. No tuning, new
checkpoint choice, best-seed choice, holdout opening or new selection trial is
permitted. The original 180 fits across the October studies remain distinct
from the 50 repeated fits; the September 108-fit screen remains older evidence.

## Decision problem and implementation

One primary use case: propose target exposure −25%, 0%, or +25% for independent
single-symbol futures accounts. Position inventory, transaction costs, delayed
fills and funding make this sequential. Market observations do not establish a
fully observable or stationary MDP. There is no allocation or high-frequency
execution claim and no new predictor research.

The runnable path is causal close-history observations → sampled/greedy policy
proposal → deterministic support/turnover admission → next-close simulated fill
→ exact rational unit/funding/cost ledger → archived evaluation. Policy actions
and likelihoods describe **proposals**; many proposals can have the same shielded
outcome. Labels are not overwritten with executed targets. Gamma 1 and the
unshaped reward `100 * (equity_after - equity_before)` preserve the finite-horizon
economic sum. Fees/funding enter only through equity. A terminal step pays
liquidation costs, clears pending targets and stops; a rollout batch cut retains
the episode and bootstraps. Training episodes have 32 transitions, evaluation 64;
that horizon shift is disclosed, not treated as continuing-operation validation.

| Learner | Actual mechanism and training regime |
|---|---|
| PPO | Fresh 128-step sampled simulator rollouts; categorical actor, separate critic, GAE λ=.95, clipped surrogate .2, four epochs; 4,096 interactions, 128 actor updates |
| Double DQN | Fresh epsilon-greedy simulation plus replay; online action selection/target evaluation; batch 64, target synchronization every 100 updates; 4,096 interactions and updates |
| CQL | Scalar discrete CQL(H)-inspired penalty α=.1 plus Double-Q targets; 4,096 updates on a frozen 512-transition simulated dataset per fold; no training collection |
| CQL α=0 | Offline extrapolation diagnostic, not an additional suitable offline RL learner |
| Behavior cloning | Supervised proposal likelihood on the same fixed buffer; not RL |

All use the same twelve causal observations, 16-unit tanh network and admission
constraints. Cash, constant-long, normalized momentum and normalized reversal
are deterministic controls. The normalized rules read observation[1], avoiding
the earlier raw-versus-normalized comparison mismatch. PPO's registered frozen
evaluation extracts argmax rather than sampling; this is an explicit policy
extraction choice. Neural inference and training remain isolated Python research;
Haskell retains the existing production safety/authorization boundary and exact
conformance oracle. No research artifact is accepted by a production loader.

The current tracked fleet is 18 symbols: AVAX, UNI, SUI, ETC, ADA, AAVE, ARB,
ATOM, BCH, DOGE, DOT, ETH, FIL, LINK, NEAR, OP, SOL and XRP. This supersedes the
older review's fleet snapshot. The experiment remains the registered BTC/ETH/SOL
subset; adequate development histories for other symbols are **not evaluated**
by this replication. The actual champion is a pinned-combo fleet through the
existing Haskell predictors and deterministic signal/risk/execution gates, not
one interchangeable learned model. Pinned configuration does not provide causal
historical weights or a lawful matched baseline. No production client, database,
credential or runtime snapshot was used.

Direct inspection again confirms `PatchTST` is patch-summary ridge regression,
`TCN` is dilated-lag ridge regression, and `Transformer` is stored-example
similarity attention. Existing identifiers/aliases and artifact semantics are
unchanged. None of those outputs enters this study, so no gain can be attributed
to improved predictors.

## Data and evidence limits

Only the two already authorized Binance USD-M development exports are read.
Panel SHA256: `09a09f0e1065733be623625fa0d67e6a67dad92c53cd8fa92b6d9caa1040674a`.
Settlements SHA256: `2ccba74489f96b9ce0e58842594a524b7c306012086443cbea7305d4011ee899`.
Both immutable byte snapshots are hash-admitted before parsing. Provider terms
apply; no raw data is redistributed. The panel has 4,910 eight-hour bars per
symbol. Training cutoffs are 1600 and 2600; evaluation intervals are
[1606,1671) and [2606,2671), giving 64 outcomes each. A strict 96-bar pre-decision
burn-in fits each episode's immutable scale. Rewards mature inside training.
There is no post-test training region to embargo. The six-bar separation is not
represented as two additive purge/embargo regions.

Availability is **assumed** close+1 ms, not independently witnessed first-seen
time. Funding settles against old inventory before a due target fills at the
next close. Prices and funding are exogenous; fees, fills, units and equity are
action-dependent. Base turnover costs are 5 bp fee + .5 bp spread + 4.5 bp
slippage. Stresses use 1.5×/2× costs, 50 bp severe costs, one added bar of delay,
50% partial fills and doubled signed funding. No calibrated market impact,
historical precision/minimum-notional filters, order queue, margin liquidation
or venue outage model is asserted.

These periods were already consumed by many adaptive trials. The 1,227-return
protected historical holdout and prospective carry outcomes through 2027-01-20
are untouched. No fresh holdout is manufactured by renaming this run. Real-log
propensities, rejected actions and overlap are unavailable; no real IS/DR/FQE or
effective sample size is fabricated. Uniform simulated proposals have probability
1/3, which is not the executed-action distribution after the shield. Replay is
model-based counterfactual evidence only. Block bootstrap intervals average
seeds/symbols before resampling chronological increments; two short dependent
windows do not establish robust significance. DSR/PBO remain insufficient.

## Assurance scope

The existing single assurance ledger is
[`formal/research/proof-ledger.json`](../../formal/research/proof-ledger.json).
Its original 38 obligations have 20 scoped closures and 18 unresolved obligations;
this continuation does not promote any claim. Critical integration gaps remain.

| Required boundary | Evidence and remaining limitation |
|---|---|
| FM-01/02 causality/partitions | Prefix-only immutable burn-in, integer SMT index/order lemmas, future-mutation and cutoff tests; authentic timestamp/model-vintage evidence absent |
| FM-03 accounting | Bounded exact Fraction ledger, reward telescoping, Haskell rational witnesses and trace reconstruction; venue fills/liabilities outside this simplified model |
| FM-04/05 actions/risk | Discrete typed-domain checks, support projection and verified zero-target turnover admission; no complete venue precision/minimum-notional/portfolio proof |
| FM-06/07/11 authority/lifecycle/noninterference | Disconnected default-disabled research interfaces, Haskell capability boundary, source bindings and finite lifecycle checking; not an OS sandbox proof or production compiler refinement |
| FM-08 failure | Invalid observations/artifacts, stale bars, state inconsistency, numerical failure and terminal fixtures fail closed; historical outage and bounded production inference not established |
| FM-09 atomicity | Immutable preview and one pending reservation; finite model with two proposal identities, 8-step bound; not a distributed/shared-production-budget proof |
| FM-10 compatibility | Version/dimension/provenance/hash checks; no authenticated production loader or proof that hashes establish profitability |
| FM-12 recovery | No duplicate modeled pending consumption, terminal clearing and exclusive archive creation; persistent execution recovery/rollback remains outside this isolated path |

The composed checker has seven SMT composition lemmas and two admission lemmas,
a 43-state/107-transition bounded model, and eight exact Python/Haskell ledger
witnesses. Other reused kernel certificates have their own bounds in the ledger.
SMT premises are checked satisfiable and violations unsatisfiable; unknown or
timeout is failure. Exact arithmetic has an 8,192-bit resource cap. CPython,
Fraction, NumPy, GHC, Z3, timestamps and specified disturbance/fill assumptions
remain trusted. Property tests and archived trace checks are not theorem proofs.
No proof guarantees profit, future drawdown, solvency or fills under arbitrary
gaps, outages or venue failure.

The new `compare_rl_replication.py` snapshots and authenticates both full archive
inventories, reconciles all terminal events, compares every policy parameter and
economic path, and requires exact fixed-buffer equality. Only named identity and
resource fields may differ; explicit old-schema defaults are validated. A report
cannot be written into either archive. Regression tests reject tampering,
duplicate JSON, nonfinite values, path traversal, symlinks, incomplete rosters,
changed actions and changed model parameters. This is reproducibility evidence,
not a new safety theorem. The canonical formal/full wrapper runs those tests.

## Search refresh through execution date

The inherited [50-entry paper-to-code matrix](paper-matrix.csv) and
[focused review](review.md) remain the main review, including PPO, SAC/TD3,
Double/Dueling/distributional DQN, BC/CQL/IQL/BCQ/TD3+BC/AWAC, OPE, recurrent and
safe RL, finance baselines and negative evidence. This refresh is not described
as 50 new full-paper readings. Search date: 2026-10-13. Queries:

- `reinforcement learning reliable evaluation trading offline RL 2026 CQL IQL flow Q learning`
- `site.proceedings.mlr.press Agarwal 2021 deep reinforcement learning edge statistical precipice`
- `site.arxiv.org proximal policy optimization conservative q learning implicit q learning safe reinforcement learning shielding`
- `site.proceedings.mlr.press 2025 flow q learning Park`
- `site.arxiv.org 2026 offline reinforcement learning trading evaluation failure costs`
- `site.github.com google-research rliable license`

Primary-source metadata rechecked: [PPO v2](https://arxiv.org/abs/1707.06347v2),
[Double DQN v3](https://arxiv.org/abs/1509.06461v3),
[CQL v3](https://arxiv.org/abs/2006.04779v3),
[IQL](https://arxiv.org/abs/2110.06169), and
[shielding v2](https://arxiv.org/abs/1708.08611v2). CQL section 3 explicitly requires
support and sampling/function-approximation conditions; our fixed α=.1, finite
gamma=1 objective and small nonlinear network do not inherit its discounted
lower-bound theorem. PPO's HTML endpoint
was unavailable as useful full text; the retained prior technical review and
current abstract metadata are distinguished from a fresh full-paper read.

[Flow Q-Learning](https://proceedings.mlr.press/v267/park25f.html) is ICML 2025;
the [official implementation](https://github.com/seohongpark/fql) is MIT-licensed.
Its expressive continuous policy is not justified for this three-action pilot.
[rliable](https://github.com/google-research/rliable) is Apache-2.0 and supports
reliable RL comparisons, but benchmark-task resampling does not justify counting
correlated trading symbols/seeds as independent markets. Neither library is added.

[MetaTrader](https://ojs.aaai.org/index.php/AAAI/article/view/40027) is published
AAAI 2026, DOI 10.1609/aaai.v40i33.40027. Its bilevel transformation/conservative-TD
approach addresses offline overfitting, but two equity datasets do not establish
crypto efficacy. It is a screened extension, not implemented; independent
replication, implementation license and cost transfer are not verified here.
[Causal FQL](https://arxiv.org/abs/2602.02847) has an
[author repository](https://github.com/CausalAILab/Causal-Flow-Q-Learning-for-Robust-Offline-Reinforcement-Learning)
with MIT license and an author-reported ICML 2026 designation. Its pixel-observation
confounding problem is not evidence that it repairs missing financial support.
It is monitored, not selected; no independent trading replication is established.

[Abbade and Costa, market-impact environments v2](https://arxiv.org/html/2603.29086v2)
is an April 4, 2026 preprint, screened in full HTML. It compares five RL algorithms
on NASDAQ 100 with flat versus nonlinear costs and hyperparameter optimization.
The reported ranking changes support cost-model sensitivity, not adoption here.
Different rewards, equity universe and tuned parameters limit transfer;
independent replication and code license are unverified. Its action-dependent
impact assumptions require calibration unavailable in our close-only panel.
No paper PDFs or third-party implementation code were committed.

Methodological credibility, fit to this decision problem, and demonstrated
improvement remain three separate questions. The search supplies no reason to
change the registered replication or reverse the existing economic rejection.

## Reproduction and publication

Use CPython 3.13.3, NumPy 2.3.5, Z3 4.15.4.0, GHC 9.4.8, Cabal 3.12.1.0,
Node 20.19.0, fourmolu .15.0.0 and hlint 3.8. Execute research with a clean
environment and one BLAS/OMP thread. For verification also set `TRADER_ENV_FILE`
to the empty string: the Haskell CLI otherwise discovers a local `.env` even
under `env -i`. This is a subprocess-only setting, not a production configuration
change. The exact command and run receipt are
recorded with the final results. Choose a fresh output directory:

```sh
python scripts/research/run_rl_budget_v3.py --run-registered-budget-v3 \
  --panel /path/to/registered-development-panel.csv \
  --settlements /path/to/registered-development-settlements.csv \
  --output /path/to/new-replication-archive
python test/rl_replication_test.py
python test/rl_composition_test.py
TRADER_ENV_FILE= bash scripts/verify.sh full
```

The runner requires committed, hash-identified source. For full-archive comparison,
use `python scripts/research/compare_rl_replication.py --help`; supply parent
index hash `360121fd70f03650b8934f18dac5e8d25f3f9eaa143f5da324102814ce510039`
and the new archive's actual index hash. Large archives
and policy parameters remain outside Git. Do not regenerate or inspect protected
full snapshots when the exact development exports are unavailable.

No push or new PR is made. `.github/workflows/autoloop.yml` runs every six hours;
`scripts/autoloop.mjs` sets `LOOP_BRANCH=BASE_BRANCH`, enables review polling by
default, and reads `pulls?state=all`. Its repair path can push the default branch,
whose push triggers deployment. A draft designation is not an isolation boundary.
Existing PR342 is left untouched. Local commits and a patch preserve this work.
No persistent service, scheduled training, real order, live exploration or new
live authorization is introduced. Existing production activity is preserved.

## Executed replication and economic results

The 2026-10-13 run completed **50 fits and 1,218 replays**, with zero failed fits or replays. All 12,950 learned parameters, 77,952 economic trace rows and frozen buffers reproduced within the registered absolute/relative 1e-12 tolerance. See [replication receipt](reproduction-results/replication.json), [all-seed results](reproduction-results/all-seed-results.csv), [complete trial registry](reproduction-results/trial-registry.csv), and [economic breakdown](reproduction-results/economic-breakdown.json). This adds zero independent market histories or selection trials.

Each figure below is a mean net return across two 64-bar evaluation windows and three symbols, additionally averaged over all five training seeds for learned policies. It is neither an annualized return nor a compounded portfolio return.

| Policy | Net % | Gross % | Worst path drawdown % | Worst path ES95 % | Intervention % |
|---|---:|---:|---:|---:|---:|
| PPO | 0.299 | 0.420 | 4.480 | 1.559 | 23.021 |
| Double DQN | 0.311 | 0.595 | 5.896 | 1.559 | 34.271 |
| CQL | -1.605 | -1.161 | 5.959 | 1.714 | 37.656 |
| CQL α=0 diagnostic | -1.588 | -1.174 | 6.818 | 1.714 | 42.344 |
| Behavior cloning | -0.171 | 0.276 | 2.911 | 1.381 | 34.115 |
| Matched momentum | 1.487 | 1.941 | 2.655 | 1.162 | 64.844 |
| Matched reversal | -2.376 | -1.967 | 4.952 | 1.713 | 64.844 |
| Shielded constant-long | 0.997 | 1.242 | 4.480 | 1.559 | 60.156 |
| Cash | 0.000 | 0.000 | 0.000 | -0.000 | 0.000 |

All five base-cost seed means (%), in registered order 11, 23, 47, 71, 101:

| Policy | 11 | 23 | 47 | 71 | 101 |
|---|---:|---:|---:|---:|---:|
| PPO | -0.021 | 1.223 | 0.003 | 0.124 | 0.169 |
| Double DQN | 1.397 | 1.224 | -0.565 | -0.336 | -0.165 |
| CQL | -1.452 | -1.998 | -1.804 | -0.879 | -1.893 |
| CQL α=0 diagnostic | -2.953 | -2.616 | -1.391 | 0.172 | -1.153 |
| Behavior cloning | -0.255 | 0.067 | 0.365 | -0.893 | -0.137 |

Matched cost/delay stress, mean net return (%):

| Scenario | PPO | Double DQN | CQL | Matched momentum |
|---|---:|---:|---:|---:|
| base | 0.299 | 0.311 | -1.605 | 1.487 |
| cost1_5x | 0.247 | 0.180 | -1.803 | 1.269 |
| cost2x | 0.195 | 0.049 | -2.171 | 1.051 |
| severe50bp | -0.118 | -0.839 | -3.485 | -0.245 |
| delay1bar | 0.167 | 0.186 | -0.271 | 0.669 |
| partial50pct | 0.368 | 0.523 | -0.977 | 1.318 |
| funding2x | 0.283 | 0.290 | -1.615 | 1.472 |

PPO and Double DQN both turn negative in the second fold; CQL is negative in both folds and for every seed mean. Relative to matched momentum, mean differences are −1.188, −1.176 and −3.093 percentage points respectively. The registered nine-bar circular-block, six-comparison simultaneous 95% descriptive intervals are [−4.621, +1.613], [−4.603, +1.774] and [−7.747, +0.878] percentage points. Block-length sensitivities (3/9/21) are retained in [evidence](reproduction-results/evidence.json); they do not repair adaptive selection or establish confirmation. DSR/PBO and valid real-log OPE remain insufficient evidence.

The independent trace reconstruction found **zero violations** of its specified action-domain, timing, equity, unit, fee, funding, fill and bounded-admission checks. The older archive’s constant `unsafeExecutableIntents=0` field is explicitly not used as evidence. This audit is empirical conformance over these traces, not a proof of all executable intents. Mean intervention rates exceed the registered 20% usability gate for every RL learner; the momentum rule also has high interventions and is not thereby approved for adoption. Unsupported states map to the shield fallback. Gross P&L minus costs plus funding reconciles to net equity; no shaped-reward gain or free terminal reset explains a positive result.

Mean base turnover is 1.045/2.631/4.344 initial-equity units for PPO/DDQN/CQL, with costs of 0.104%/0.263%/0.434% and funding P&L of −0.016%/−0.021%/−0.010%. Mean absolute exposure is 3.750%/6.264%/7.293%. Fill counts include partial rebalances and terminal liquidation and are not independent trades. Holding durations, drawdown durations, symbol/fold breakdowns, and all stress economics are preserved in the economic breakdown and trial registry. There is no new regime-specific generalization claim.

The fixed offline dataset has only 512 **simulated** transitions per fold. CQL α=.1 does not outperform its α=0 diagnostic here, and both underperform behavior cloning. This neither disproves CQL generally nor supplies supported exchange-log counterfactuals. Mean base-path Q-value MSE against realized finite-path reward-to-go is 1.4625 for DDQN, 8.0000 for CQL and 9.4728 for α=0, in squared registered reward units. This compares 32-step training objectives with 64-step realized paths, so it is a diagnostic rather than an unbiased calibration certificate. PPO critic calibration and broader recurrent/risk-objective ablations are not established. Loss diagnostics remain in the archive.

Run wall time was 1,365.167 seconds (22.8 minutes), with 866.910 seconds summed fit time. Reported peak process RSS was 82,632,704 bytes (78.8 MiB); the largest artifact was 4,893 bytes. Across per-fit 100-call warm Python benchmarks, the worst p99 was 9.090 ms. These measurements on a heavily loaded shared CPU exclude production artifact loading, end-to-end Haskell admission and hard-deadline guarantees. All checkpoint rules were fixed before the run; no best seed was selected.

Raw archive: `.tmp/research/rl-budget-v3-reproduction-20261013`. Index SHA256: `5bd417ee7b981004f6e9dfd7600426f8a6a0300c9d48d5fdb8a75e344d9f92e1`. The source commit is `8e83b2d9d4085044b03a93ca0712b43edfff8574`; subsequent changes add verification and reporting, not new training. Data and learned parameters remain outside Git. The current-main replica agrees with the prior archive despite the documented explicit-default runner metadata extension.

## Verification findings and remaining decisions

The [verification receipt](reproduction-results/verification-receipt.json) records
commands, versions, exit status and local log hashes. The canonical full wrapper
was executed and **failed** in
`PPOProcessBridgeTests.test_source_bound_smt_model_and_actual_trained_process`:
“PPO process bridge: no actual inference for trained policy.” Its formal
regression stage ran 341 tests with one error, then correctly stopped before the
later stages. The same unchanged test failed outside the sandbox and again after
the campaign completed. The 20 ms acceptance budget remains unchanged. The
observed error does not identify whether decoding, computation, cleanup or
scheduling consumed the budget; host load alone is not a proven root cause.
No checks were disabled and no test acceptance threshold was relaxed. A subsequent
[diagnostic invocation](reproduction-results/bridge-diagnostic.json) also timed out
at the outer three-second limit for the standalone codec process, before any
neural request comparison. It remains a failure, not a substitute passing check.

A separate `python scripts/formal/verify.py --record` invocation completed
successfully, including 91 core SMT results, component-specific certificates,
finite models and compiled conformance. That receipt records one successful
invocation, not reliable runtime availability; it does not supersede the failed
full wrapper. The composed RL path separately has seven SMT composition lemmas,
two admission lemmas, its 43-state/107-edge model and eight exact ledger witnesses.
The new six archive-regression tests and 24 existing causal RL learner/accounting
tests passed. `bash scripts/verify.sh haskell` passed build, formatting, all
lint batches, research lint, smoke checks and its 1/1 Cabal test suites (the
reported test-case count is suite granularity, not an assertion count). Web
verification passed typecheck, all 241 tests and build.
Automation verification initially had three localhost-bind EPERM failures under
the sandbox, then passed all 185 tests outside it using the same code and a clean
environment. No test was skipped in those suites.

The existing assurance ledger's 18 unresolved obligations remain: no-lookahead,
revision timing, finite outputs, overflow/underflow, live-authority isolation,
observation causality, valid transitions, terminal accounting, action safety,
invalid-output fallback, replay determinism, rollback, configuration bounds,
atomicity, race freedom, shutdown termination, recovery and permanent-lockout
avoidance. Its 20 scoped closures are 16 exhaustive checks, three finite model
checks and one SMT closure, not 20 complete implementation theorems. Detailed
assumptions and critical-source hashes remain in that single ledger. This work
adds no global neural-network, compiler, exchange-execution or profitability proof.

**RL research: partially completed.** This registered engineering replication is
complete with trained policies and reconciled evidence. The broader mission still
lacks a causal champion replay, adequate universe expansion, authentic historical
availability, untouched confirmation, realistic venue constraints, supported
real-log OPE, full requested diagnostics/ablations and bounded integrated inference.
Data/model export availability is an adoption blocker, not a reason to fabricate
or weaken a comparison.

**RL adoption: continue research; reject these packages for integration.** The
matched simpler momentum policy has the stronger development mean, but this is
not authorization to replace the champion with momentum. Material superiority
and usability gates are unmet; full verification is not green. No disabled
production challenger is installed, no production Python service is added, and
no future shadow/paper duration is claimed as completed.

**Formal assurance: scoped checks recorded; integration blocked.** Conditional
SMT and bounded lifecycle results plus concrete conformance are preserved. The
18 unresolved obligations, implementation correspondence limits and repeatable
inference-bridge failure remain explicit.

**Financial execution:** no real orders, live exploration, production authority
change, merge, deployment or new scheduled process was introduced. Existing
production operation was preserved. Publication remains local because the
repository's automatic PR-repair path is not isolated from main deployment.

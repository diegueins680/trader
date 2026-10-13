# Matched finite-horizon RL experiment

## Executed result and decision

All **50 fits and 1,218 replays completed**; all 1,268 trial IDs and terminal
events reconcile, with no failed fits or evaluation paths. Every path spans
64 transitions. All 168 deterministic control records and full traces match
the frozen reference exactly. The campaign timer measured 460.740 seconds (7.68 minutes), excluding initial
data loading/hash validation and final summary/index construction. Fitting plus
warm inference benchmarks totaled 295.203 seconds; this is not process CPU time.

**Reject integration of this candidate package.** Matching the horizon does
not produce material improvement over the information-matched momentum rule,
and all three RL families fail the unchanged 20% intervention gate. Positive
PPO/DDQN means do not establish an advantage over cash or the champion. The
momentum control also fails the intervention gate and is not promoted.

The following are mean net returns over six identical symbol/fold windows,
first averaged per seed, then across all five seeds; they are not annualized.
Drawdown and ES are worst path values. The full [seed/trial registry](horizon-v5-results/trial-registry.csv)
and [all-seed results](horizon-v5-results/all-seed-results.csv) retain every run.

| Policy | Mean net | Seed mean min–max | Worst DD | Worst ES95 | Mean intervention |
|---|---:|---:|---:|---:|---:|
| ppo | +0.437% | -0.854% to +1.153% | 5.179% | 1.559% | 47.135% |
| double_dqn | +0.448% | -0.010% to +1.006% | 6.452% | 1.735% | 45.573% |
| cql | -1.921% | -2.514% to -1.429% | 5.554% | 1.281% | 48.594% |
| cql_alpha0 | -2.328% | -3.147% to -1.716% | 5.921% | 1.281% | 48.281% |
| behavior_cloning | -0.827% | -1.758% to -0.105% | 5.741% | 1.542% | 45.833% |
| cash | +0.000% | +0.000% to +0.000% | 0.000% | -0.000% | 0.000% |
| constant_long | +0.997% | +0.997% to +0.997% | 4.480% | 1.559% | 60.156% |
| normalized_momentum3 | +1.487% | +1.487% to +1.487% | 2.655% | 1.162% | 64.844% |
| normalized_reversal3 | -2.376% | -2.376% to -2.376% | 4.952% | 1.713% | 64.844% |

| Primary learner | Seed 11 | Seed 23 | Seed 47 | Seed 71 | Seed 101 |
|---|---:|---:|---:|---:|---:|
| ppo | +0.011% | +0.888% | +1.153% | -0.854% | +0.984% |
| double_dqn | +0.656% | +1.006% | +0.011% | +0.576% | -0.010% |
| cql | -1.684% | -1.429% | -2.475% | -2.514% | -1.504% |

Relative to the frozen 32-step reference, mean PPO improves by 0.137pp and
Double DQN by 0.137pp; CQL worsens by 0.316pp. These are descriptive paired
horizon deltas, not additional confirmation tests. CQL alpha-zero returns
−2.328%, versus CQL −1.921%; neither supports adoption or a general claim that
the conservative penalty improves financial performance. Cloning returns
−0.827%. No failed seed is omitted or replaced.

PPO fold means are +0.952% and −0.079%; DDQN +1.660% and −0.764%; CQL −2.038%
and −1.804%. PPO symbol means BTC/ETH/SOL are +0.737/+0.469/+0.104%; DDQN
+1.021/+0.361/−0.039%; CQL −1.233/−2.103/−2.427%. Seed mean SDs are
0.846/0.440/0.532pp respectively, algorithmic dispersion rather than independent
market standard errors. Two short folds do not establish regime robustness.

| Stress | PPO | Double DQN | CQL | Matched momentum |
|---|---:|---:|---:|---:|
| base | +0.437% | +0.448% | -1.921% | +1.487% |
| cost1_5x | +0.335% | +0.297% | -2.103% | +1.269% |
| cost2x | +0.233% | +0.155% | -2.381% | +1.051% |
| severe50bp | -0.376% | -0.708% | -4.016% | -0.245% |
| delay1bar | +0.542% | +0.917% | -0.059% | +0.669% |
| partial50pct | +0.593% | +1.081% | -0.813% | +1.318% |
| funding2x | +0.425% | +0.429% | -1.920% | +1.472% |

All RL means are negative in the severe 50bp case. Delay and partial fills
sometimes improve returns; these are sensitivities to modeled action timing
and inventory, not superior fill quality or evidence for selecting a favorable
execution scenario. The identical stress roster is retained for every control.

Nine-bar simultaneous descriptive intervals (percentage points):

| Learner | Versus cash | Versus matched momentum |
|---|---:|---:|
| ppo | +0.437 [-0.661, +1.638] | -1.051 [-4.543, +1.858] |
| double_dqn | +0.448 [-1.367, +2.669] | -1.040 [-4.446, +1.791] |
| cql | -1.921 [-4.030, -0.257] | -3.408 [-8.745, +0.833] |

The positive-lower-bound gate fails for every RL learner against both required
baselines. Block lengths 3 and 21 remain in the evidence JSON. CQL's negative
nine-bar interval against cash is descriptive negative evidence, not a
confirmatory significance claim after repeated adaptive research. DSR/PBO are
insufficient; no favorable substitute statistic is introduced.

## Accounting, offline support and policy diagnostics

Mean gross/funding/cost/net percentages reconcile as follows: PPO
+0.652/−0.012/0.204/+0.437, DDQN +0.768/−0.019/0.301/+0.448, CQL
−1.361/+0.001/0.561/−1.921 (rounding only in this display). Mean absolute
exposure is 7.774/6.706/5.444%; turnover is 2.036/3.012/5.614 initial-equity
units. Nonzero-fill row means are 23.733/22.533/23.567, including terminal
liquidation in its final row rather than counting separate exchange orders.
Worst drawdown duration is 64 bars for all three. Per-path holding-period lists
and action counts remain in the trial registry and archive.

The independent audit reconstructed prices, units, funding, linear costs,
nonterminal fills and reward/equity identities over **77,952 rows, with zero
violations** at tolerance 1e-12. The old exporter's constant
`unsafeExecutableIntents=0` is not used as evidence. Terminal combined
fill/liquidation is not fully independently reconstructed; exact-kernel
conformance provides separate scoped evidence. No double charging or hidden
evaluation reset was detected. Full training event traces remain absent.

There are 36,373 changed proposals across all policies/stresses: 35,095 support
projections and 1,278 turnover rejections. Deferred and ignored-new proposals
are separately 5,568 each. Raw unsupported proposals are not unsafe executable
intents. Excessive interventions still reject the candidate under the
preregistered usability gate; no post-result masking or gate relaxation occurs.

Five economic terminations during fresh training are retained: three
`endpoint_exposure` (fold0 PPO23/PPO47, fold1 DDQN11) and two `drawdown_limit`
(fold1 PPO11/DDQN71). Their losses enter training before reset; they are not
failed fits, and no evaluation path is restarted. The two frozen behavior
buffers have no economic failures and eight terminal transitions each.

Each buffer contains 512 distinct observations, with raw short/cash/long counts
[164,173,175] and [153,184,175]. Burn-in-range supported fractions are 67.969%
and 46.484%, not real behavior overlap estimates. Raw proposal probability is
1/3 before many-to-one shielding. Pending-state input is always zero in training;
delayed commitments remain an out-of-training-state stress. All offline fits
record zero new simulator interactions, versus 4,096 for every PPO/DDQN fit.

Frozen buffer hashes:

- Fold 0: `76a1fe0880957ff7b6c606191a4a849fe9243aa8e3b12eb78973cb17ffd9950e`.
- Fold 1: `0399846942639175270273049b8c7b9f6f749fcde9570f5d136f991af319221a`.

Held-path critic MSE is DDQN 1.829, CQL 9.496 and alpha-zero 14.138 in squared
reward units; it is not calibrated uncertainty or FQE. No real behavior logs,
defensible real propensities, overlap or hidden-confounding assessment are
available. Real IS/DR/FQE/ESS are unsupported and are not fabricated. Historical
replay remains model-based counterfactual evidence under the declared simulator.

## Policy cards and engineering status

PPO, DDQN and CQL share research-only mode, discrete target exposure
{-0.25,0,+0.25}, 12 causal inputs, a 16-unit tanh network, the v2 deterministic
shield and the exact rational economic ledger. All 50 parameter artifacts are
retained outside Git; none is selected for production. Algorithm/seed/fold,
source commit, registration, training horizon, frozen-reference hash and
assurance-ledger identity are embedded. PPO/DDQN's reference buffer hash does
not claim they trained on that frozen buffer; their actual training input is
fresh simulator interaction with the registered causal prefixes.

Peak process RSS was 82,001,920 bytes; largest artifact 4,929 bytes. Worst
100-call warmed Python p99 after ten warmups was 0.1233ms on this host.
This does not bound cold loading, Haskell inference, scheduling, concurrency
or production deadlines. No production runtime, mode switch, persistent shadow
or paper process, order authorization or deployment is introduced.

Archive: `.tmp/research/rl-matched-horizon-v5-20261013`.
Index SHA-256: `6451d127fbdc9a5e8b9e0a285b55c794e75297e5e67653984f55c7152c39c957`.
The [evidence receipt](horizon-v5-results/evidence.json) and
[comparison](horizon-v5-results/horizon-comparison.json) bind full inventory,
terminal events, policy identity, buffer identity and unchanged baseline traces.
Models/data remain outside Git; the archive is local, not remotely published.
Single-thread settings and fixed seeds constrain nondeterminism; no arbitrary-
hardware bitwise guarantee is claimed. V5 has not itself been independently
retrained: the earlier 32-step numerical replication does not certify new
64-step parameters.


This is the registered adaptive successor to the 32-step training campaign,
not untouched confirmation. The question is whether matching the finite
training objective to the 64-transition evaluation objective improves bounded
position management. Inventory carries price and funding exposure into later
bars; changing it incurs costs, making this a sequential decision problem.

The verified starting main is `0b76f2b879f75f5c669fe944a14ea6700d446240`
([CI receipt](https://github.com/diegueins680/trader/actions/runs/38313512581)).
Registration was committed as `d92df941` before implementation and outcomes.
The executed training source is `629dd38d`; the later exporter commit does not
change training. See [registration](../registrations/rl-matched-horizon-v5.json)
and the frozen [32-step replication](reproduction-2026-10-13.md).

## Registered comparison and limitations

PPO collects fresh on-policy simulator trajectories; Double DQN collects fresh
exploratory trajectories with replay; scalar CQL(H)-inspired alpha .1 learns
from a frozen, simulated 512-transition uniform-proposal dataset per fold.
CQL alpha zero is an offline DDQN diagnostic, not another suitable offline
algorithm. Behavior cloning is a supervised control on the same fixed buffer.
Offline fits receive a factory that rejects new simulator interactions.

All five methods train anew with seeds 11/23/47/71/101, two training prefixes,
4,096 interaction steps for PPO/DDQN or optimizer updates for offline methods.
PPO retains 128-transition rollouts and four optimization epochs (128 actor
updates total); DDQN/CQL/BC retain 4,096 updates, batch size 64, the same learning
rates and target schedules. Last checkpoints only; no best-seed selection. Evaluation uses deterministic
argmax proposals from each frozen network, as in the parent experiment; PPO
training samples current-policy categorical proposals. Proposal labels remain
attached to their original log probabilities through the unchanged shield.
The 12 observations, 16-unit tanh network, proposal action space, shield,
execution kernel, unshaped reward and all economic evaluation settings remain
unchanged. Matching the horizon necessarily changes sampled starts, reset
frequency, terminal proportions and buffer bytes; this does not isolate those
mechanisms from one another. The frozen 32-step reference is not retrained.

Every policy is evaluated on all BTC/ETH/SOL symbol/fold paths under base costs,
1.5x, 2x, severe 50bp, one additional bar of delay, 50% partial fills and doubled
funding. Cash, constant-long, normalized momentum3 and normalized reversal3
share execution and constraints. Momentum/reversal use only the sign of the
same normalized observation[1], not additional raw history. All 168 deterministic
reference records and full traces must match exactly. No current champion
proxy is substituted: the user authorized continuing without its historical
model/configuration export, and this remains an adoption blocker.

The two development exports retain SHA-256 identities
`09a09f0e1065733be623625fa0d67e6a67dad92c53cd8fa92b6d9caa1040674a`
(panel) and `2ccba74489f96b9ce0e58842594a524b7c306012086443cbea7305d4011ee899`
(funding settlements). Binance provider terms apply; raw data are outside Git.
Bars are 8h; training stops 1600/2600 (exclusive), evaluation [1606,1671) and
[2606,2671), 64 rewards each. Training starts at least 96 bars into its physically
sliced prefix, with last reward index strictly below the cutoff. Each episode
fits immutable normalization only on its pre-decision burn-in. Close+1ms
availability is assumed; authentic first-seen history remains absent. These
March–April 2022 and February–March 2023 windows were already consumed. No
protected 1,227-return holdout or prospective carry outcomes were opened.

Gamma=1 and reward=100 times exact net equity change define a finite-objective
experiment. End-of-objective liquidation is costed once and has no value
bootstrap; PPO rollout cuts preserve state and bootstrap. Economic failure
terminates with its loss recorded. This is not a claim about an indefinite
continuing strategy. [Pardo et al., ICML 2018](https://proceedings.mlr.press/v80/pardo18a.html)
distinguish finite-objective termination from training-only truncation and
motivate observing remaining time. The local policy already observes remaining
fraction. This paper motivates the diagnostic, not financial efficacy. The
registration records the 2026-10-13 search, proceedings version, query, scope
and original-code disposition; no paper code or PDF is redistributed.

Each account begins with equity one, no external flows. Positions are futures
units; old units earn next-close price P&L and funding before fills. Base costs
are 5bp fee + .5bp half-spread + 4.5bp slippage per turnover. The exact rational
ledger includes costs and funding before deriving reward; neither is charged
again. Price and funding history are exogenous. No queue, market impact,
authentic precision/minimum-notional, margin liquidation or outage calibration
is claimed. Long-horizon training does not fix those simulator limitations.

Primary comparisons are paired net total equity change versus cash and matched
momentum, requiring >.001 material improvement and a positive simultaneous lower
bound, plus drawdown <=15%, ES95 <=2%, action intervention <=20%, zero unsafe
executable intents and no failed paths. Circular block lengths 3/9/21, 2,000
resamples, seed20261010, six-comparison Bonferroni endpoints are unchanged.
Seeds and symbols are averaged before resampling chronological increments.
These two short exposed windows support descriptive dependence-conditional
intervals only. DSR/PBO and cumulative adaptive multiple-testing control remain
insufficient. The horizon setting adds 50 recorded fits to the disclosed prior
180 October fits, 108 September screen fits and 50 numerical replication fits;
the numerical replication is not a new selection trial or market history.

## Reproduction and isolation

Use pinned Python 3.13.3, NumPy 2.3.5, Z3-solver 4.15.4.0 and the repository
Haskell/Node toolchains. The actual run used a clean subprocess environment,
single BLAS/OMP thread and empty `TRADER_ENV_FILE`, preventing local secret-file
autodiscovery. No order-capable client, network request, production database,
queue, or persistent service is used by the campaign.

```sh
python scripts/research/run_rl_horizon_v5.py --run-registered-horizon-v5 \
  --panel .tmp/research/historical-funding-campaign-v1/registered-development-panel.csv \
  --settlements .tmp/research/historical-funding-campaign-v1/registered-development-settlements.csv \
  --output <fresh-archive>
python scripts/research/summarize_rl_horizon_v5.py \
  --source <fresh-archive> \
  --parent .tmp/research/rl-budget-v3-reproduction-20261013 \
  --expected-index-sha256 <archive-index-sha256> --output <fresh-report-directory>
```

Sources must be committed before training; data hashes and registration are
checked before collection. Existing 32-step defaults and artifact meanings are
preserved; new manifest/policy metadata explicitly records the training horizon.
Artifacts remain `research_only`, environment `rl-admission-continuation-v2`,
observation `causal-inventory-pending-finite-v1`; no production loader or Haskell
inference admission for these 12-input policies is added. Existing exact-bit
bridge verification concerns a different research policy schema and must not
be represented as v5 operational integration.


## Formal assurance and remaining blockers

The source-bound composition checker passes seven nonvacuous SMT queries over
exact integers/reals/booleans plus two zero-target admission lemmas. The maturity
query now quantifies integer horizons 1..128: `start>=96`,
`start<cutoff-horizon`, `stop=start+horizon+1` imply `stop<=cutoff` and
`stop-1<cutoff`. Both premise satisfiability and negated-conclusion UNSAT are
required. Solver unknown/timeout is not acceptance. Source hashes and actual
factory conformance connect this statement to code; this is not universal
Python/compiler refinement.

The unchanged lifecycle check explores 43 states and 107 transitions with two
proposal identities, one reservation and an eight-step bound. It verifies the
finite single-owner model, not production concurrency, durable reservations or
unbounded fairness. Eight exact Python/Haskell ledger witnesses and the retained
turnover counterexample test selected implementation correspondence. The shield
continues to reject/replace unsupported targets and excessive-turnover previews;
proof premises bound pre-fill exposure, fill fraction and modeled costs. No
claim constrains arbitrary future market gaps, outages or realized drawdown.

`python scripts/formal/verify.py --record` passed in 144.992s. The full receipt
retains 91 core SMT entries and component checks. Twenty original mission
obligations have scoped closure and **18 remain unresolved**: no-lookahead,
revision timing, finite outputs, overflow/underflow, live-authority isolation,
observation causality, valid transitions, terminal accounting, action safety,
invalid-output fallback, replay determinism, rollback, configuration bounds,
atomicity, race freedom, shutdown termination, recovery and permanent lockout.
No broad obligation is closed by this horizon change. The complete assumptions,
quantifiers, tool versions, bounds, file mappings and limitations remain in the
[assurance ledger](../../formal/research/proof-ledger.json).

The targeted suite passes 29 tests, including actual 64-step terminal costs,
poisoned post-cutoff prices, immutable frozen data, preservation of the 32-step
default, invalid horizon rejection, offline factory isolation, disabled CLI
behavior and rejection of altered/duplicated controls. These are tests, not
new theorems. The first targeted invocation failed on the expected stale source
binding after edits; reviewed bindings were updated and the reruns passed.
The failure log is retained in the verification receipt.

Integration remains blocked independently by the failed economic/intervention
gates, missing causal champion replay, authentic availability, untouched
confirmation, real OPE/support, execution calibration, reviewed-universe
coverage and operational artifact/reservation integration. Missing/stale/outage,
artifact failure and recovery have scoped fixtures, not complete historical
stress campaigns for the whole fleet. The original economic and reproducible
delivery acceptance gates remain incomplete in the machine-readable ledger.

Publication remains blocked: scheduled autoloop polls PRs in all states and can
repair directly onto the default branch; main pushes can deploy. A draft PR
is therefore insufficient isolation. No branch push or new PR is performed;
existing PR342 is untouched. A local cumulative patch and draft body preserve
reviewable work. The working branch is
`research/rl-causal-campaign-2026-10-13`. Pre-existing `.playwright-mcp/` is
preserved. Deployment, production application, workflow and authorization
configuration remain unchanged from the recorded main baseline.



A final compatibility review found that explicit horizon metadata would make
an otherwise unchanged 32-step archive fail the older strict replication
checker. Commit `30364a7f` normalizes only absent legacy metadata or explicit
integer 32; 64, floating 32.0, strings, booleans and null are rejected. Seven
archive regressions pass, and the actual prior 50-fit archives still match
12,950 parameters and 77,952 trace rows within 1e-12. The full comparison is in
[the compatibility receipt](horizon-v5-results/legacy-replication-compatibility.json).
This reruns archive comparison only, not training and not new market evidence.


Final `bash scripts/verify.sh full` exited 0 on code revision `30364a7f`:
343 formal regressions (137.093s), seven archive regressions, four diagnostic
regressions, 29 RL/accounting tests (3.861s), ordinary certificate reproduction
(151.097s), Haskell build/fourmolu/hlint/smoke/one Cabal test suite, web
typecheck/241 tests/build and 185 automation tests (21.488s). No required target
was skipped, unavailable or timed out. A prior full run also passed, but its
archive tests preceded the compatibility fix; the final run supersedes it.
The [verification receipt](horizon-v5-results/verification.json) includes exact
commands, clean environment, log hashes and the initial failed source-binding
check. Scope-limited proof success does not close the unresolved mission gates.

Local review delivery: `.tmp/research/rl-horizon-v5-delivery-20261013` contains
a cumulative research patch, commit list, local draft PR body, copied logs and
a hashed delivery manifest. No PR was opened because of the documented workflow
side-effect blocker. Large data and policy weights remain only in the isolated
archive; the pre-existing user directory is untouched.

RL research: this registered study is complete; the broader mission remains
partially completed. RL adoption: reject this package for integration and
continue research; the simpler momentum rule wins this comparison but is not
operationally approved. Formal assurance: scoped SMT/model/conformance evidence
passes with 18 broad obligations open/partial. Financial execution: no real
orders, live exploration or new live authorization; existing production
activity is preserved.


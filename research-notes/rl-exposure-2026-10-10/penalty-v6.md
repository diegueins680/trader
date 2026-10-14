# PPO intervention-penalty ablation, 2026-10-13

## Result and adoption decision

**Reject this package for integration.** Ten fresh PPO fits and all 378 economic
replays completed, with no failed fits or replay paths. Five independent training
seeds (11,23,47,71,101) run in each of two chronological development folds.
Each fit collected 4,096 fresh simulator transitions and performed 128 actor
updates; final checkpoints were fixed in advance. This is on-policy simulator
learning, not fixed-dataset offline RL. The already trained Double DQN, CQL,
alpha-zero and cloning policies are frozen references, not newly trained here.

The primary decision remains bounded target exposure, with delayed inventory,
funding, costs and pending commitments making it sequential. Every learner and
rule is evaluated through the same deterministic shield and exact ledger.
The [registration](../registrations/rl-proposal-penalty-v6.json), committed as
`56bab0c8` before training, changes one training coefficient and no economic
assumption, observation, evaluation window, shield or promotion gate.

Mean returns below average the six symbol/fold windows per seed, then all five
seeds; they are not annualized. Seeds do not multiply the number of independent
market histories. All [seed results](penalty-v6-results/all-seed-results.csv),
[388 trial IDs](penalty-v6-results/trial-registry.csv) and
[independent audits](penalty-v6-results/penalty-comparison.json) are retained.

| Policy | Mean net | Mean intervention | Worst DD | Worst ES95 |
|---|---:|---:|---:|---:|
| PPO penalty v6 | +0.515% | 35.052% | 5.339% | 1.559% |
| PPO unshaped v5 (frozen) | +0.437% | 47.135% | 5.179% | 1.559% |
| Double DQN v5 (frozen) | +0.448% | 45.573% | 6.452% | 1.735% |
| CQL v5 (frozen) | -1.921% | 48.594% | 5.554% | 1.281% |
| Cloning v5 (frozen) | -0.827% | 45.833% | 5.741% | 1.542% |
| Cash | +0.000% | 0.000% | 0.000% | -0.000% |
| Constant long | +0.997% | 60.156% | 4.480% | 1.559% |
| Matched momentum | +1.487% | 64.844% | 2.655% | 1.162% |
| Matched reversal | -2.376% | 64.844% | 4.952% | 1.713% |

The intervention decrease is 12.083 percentage points,
meeting the registered descriptive 10-point mechanism target. The unchanged
20% usability gate still fails. Mean net improvement over unshaped PPO is only
0.078pp, below the registered 0.100pp materiality scale;
that delta is descriptive, not an independent confirmation test. The primary
comparison against momentum is negative. Momentum also fails the intervention
gate and is not operationally promoted. No post-result coefficient search,
seed replacement, horizon change or gate relaxation occurs.

| Seed | PPO penalty | Frozen unshaped PPO |
|---|---:|---:|
| 11 | +0.159% | +0.011% |
| 23 | +1.647% | +0.888% |
| 47 | +0.052% | +1.153% |
| 71 | -0.748% | -0.854% |
| 101 | +1.464% | +0.984% |

Seed mean SD is 1.015pp (algorithmic dispersion, not a market
standard error). Fold means are +0.954% and +0.075%; BTC/ETH/SOL means are
+0.763/+0.705/+0.077%. Two short previously inspected folds cannot establish
regime robustness. The worst drawdown increases from 5.179% to 5.339%; this is
not a uniformly better risk policy.
Across registered stresses, worst drawdown 5.840% and ES95 1.563% stay below
the 15%/2% pilot bounds; this empirical result does not repair the failed
economic/intervention gates or establish disturbance-independent safety.

| Cost/execution stress | PPO penalty | Frozen unshaped PPO | Matched momentum |
|---|---:|---:|---:|
| base | +0.515% | +0.437% | +1.487% |
| cost1_5x | +0.443% | +0.335% | +1.269% |
| cost2x | +0.372% | +0.233% | +1.051% |
| severe50bp | -0.056% | -0.376% | -0.245% |
| delay1bar | +0.505% | +0.542% | +0.669% |
| partial50pct | +0.682% | +0.593% | +1.318% |
| funding2x | +0.504% | +0.425% | +1.472% |

Base cost is 10bp per absolute traded notional: 5bp fee, 0.5bp spread and
4.5bp slippage. Severe cost is 50bp. Delay adds one bar, partial fills execute
50%, and funding stress doubles settlement rates. Favorable partial-fill
returns reflect altered inventory and timing, not evidence for better fills.

Nine-bar circular block-bootstrap simultaneous descriptive intervals are
+0.515pp [-0.341,+1.564] versus cash and -0.972pp [-4.068,+1.569] versus
momentum. Both positive-lower-bound gates fail. The preregistered 3/9/21-bar,
2,000-draw, fixed-seed calculation retains conservative six-comparison
Bonferroni endpoints. It does not adjust away the entire adaptive research
history. All sensitivity intervals are in [evidence.json](penalty-v6-results/evidence.json).
DSR/PBO remain insufficient evidence; no substitute favorable value is used.

## Reward, ledger and exploitation checks

Training reward is `economic_reward - 0.02 * actionChanged`, where economic
reward is 100 times the net equity increment. The intervention is a support
projection or turnover rejection, not a fee. The coefficient equals two basis
points of initial equity in reward units. The economic ledger is never debited
for the penalty, and evaluation always uses the unmodified ledger. Proposed
action/log-probability labels remain attached to proposals through many-to-one
shielding; no filled-action relabeling is introduced. Economic state,
observations and fills match the unshaped environment for the same actions.

Independent training-trace auditing reconciles **40,960 transitions in 643
episodes**, their chronological cutoffs, reward decomposition and within-episode
ledger identities. Resets require a recorded terminal row. Three economic
terminations are retained: fold 0 seeds 23/47 `endpoint_exposure`, fold 1 seed 11
`drawdown_limit`. Their losses enter learning. The final training budget stops
three episodes partway; those rollout boundaries bootstrap, not liquidate or
start a free episode. No evaluation path resets. Changed proposals, shaped and
economic reward sums, penalties and final-terminal flags are recorded per fit.
Training sums over sampled episodes are not held-out returns.

The evaluation audit checks **24,192 rows in 378 paths**, with zero detected
violations at binary64 tolerance 1e-12. It independently reconstructs prices,
units, funding, linear costs, nonterminal fills and equity/reward identities.
Terminal combined fill/liquidation is not fully independently reconstructed;
eight Haskell exact-kernel witnesses supply separate scoped conformance.
Zero detected violations is not a universal safe-execution proof. The inherited
constant `unsafeExecutableIntents=0` is explicitly disclaimed as evidence.
Regression fixtures reject corrupted rewards, economic rows, maturity bounds
and unrecorded resets; no double charge or omitted training reset loss was found.

Across all policies and stresses: 9,768 changed proposals comprise 9,543 support
projections and 225 turnover rejections; 1,728 deferred proposals and 1,728
ignored new proposals are separately logged. Raw rejected proposals are not
executed unsafe intents. Base PPO gross/funding/cost/net reconcile as
+0.669/-0.011/0.143/+0.515 percent (display rounding). Mean absolute exposure
is 5.391%, turnover 1.431 initial-equity units, nonzero-fill row count 16.467,
and worst drawdown duration 61 bars. Terminal liquidation is included in its
last row, not reported as a separate exchange order. Per-path holding periods
and action distributions are retained in the registry/archive.

## Provenance, method fit and evidence limits

Only the already authorized, development-only Binance eight-hour BTC/ETH/SOL
panel and funding exports were read. Availability is assumed close time +1ms,
not witnessed historical first-seen publication. Panel hash
`09a09f0e1065733be623625fa0d67e6a67dad92c53cd8fa92b6d9caa1040674a`;
settlements hash
`2ccba74489f96b9ce0e58842594a524b7c306012086443cbea7305d4011ee899`.
Provider terms apply; raw data is not committed. Training cutoffs are 1600/2600;
evaluation slices [1606,1671)/[2606,2671) each contain 64 returns. Normalization
uses the immutable 96-bar predecision burn-in; reward maturity precedes cutoff.
These March–April 2022 and February–March 2023 periods have influenced prior
research. No untouched confirmation is claimed or consumed. The protected
1,227-return holdout and prospective carry outcomes remain unread.

Exogenous price-taking OHLCV replay supports this exposure experiment only.
Precision/minimum-notional, queue priority, impact, outages and realistic venue
liquidation are not calibrated. Accounts are separate, with unit initial equity;
there is no proven shared production portfolio reservation. Every one of the
168 deterministic controls and full traces matches frozen v5 exactly. Both
512-row simulated uniform reference buffers also match; PPO does not train on
them. No new offline learner is fit here. Prior CQL evidence is simulator-generated,
not observed exchange behavior. Real support, propensities, IS/DR/FQE/ESS and
hidden-confounding assessment remain unavailable. Replay is model-based
counterfactual evidence, not assumption-free OPE.

The source-reviewed PPO uses fresh rollouts, GAE, clipped policy updates and
final frozen greedy extraction. The penalty was motivated by proposal aliasing,
not inferred financial efficacy from a paper. [Ng, Harada and Russell (ICML 1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
provides potential-based reward invariance; this action penalty is not that
transformation and changes the optimization objective. [Achiam et al. (ICML 2017)](https://proceedings.mlr.press/v70/achiam17a.html)
provides constrained policy optimization; this fixed penalty is not CPO and
inherits none of its guarantees. The registration records October 13 queries,
versions, publication status, selection rationale and license limitations.
This targeted metadata/abstract refresh does not claim two new full deep reviews.
The broader paper matrix and negative evidence remain in the parent research
package. No source PDF or third-party code was copied for this ablation.

## Policy card, formal scope and reproducibility

Research-only PPO, 12 causal inputs, 16 tanh units, three raw targets
{-0.25,0,+0.25}, gamma 1, lambda 0.95, clip 0.2, 64-step finite objective and
`equity-minus-proposal-intervention-v6` training reward. All ten artifacts retain
source, seed, fold, registration, preprocessing/environment and assurance
identity. They are disconnected from production inference and selection.
No Haskell parity/latency claim is made for this schema; the existing isolated
Haskell bridge certificate covers a different policy schema.

Nine nonvacuous exact-real/integer/Boolean SMT queries now include reward
reconstruction and nonpositive penalty. Two admission lemmas, a reachable
43-state/107-edge lifecycle at bound 8, and eight Haskell Rational ledger witnesses
supply additional scoped evidence. SAT premises and UNSAT negations are required;
unknown/timeouts are failures. Source hashes and conformance tests map artifacts
to Python/Haskell code but do not prove universal compiler/runtime refinement.
Binary64 reward subtraction is checked within 1e-12, not proved over every float.
The penalty cannot guarantee drawdown, solvency, optimality or policy invariance.
Original mission status remains 20 scoped closures and 18 open/partial obligations.

Campaign timer: 309.453s, excluding initial data loading/hash validation and final
summary/index construction. Fit/artifact/trace-write/warm-benchmark time totals
203.075s; it is not process CPU time. Peak RSS 79,278,080 bytes; maximum artifact
5,034 bytes; worst warmed 100-call Python p99 after ten warmups 0.3737ms.
These are host measurements, not bounded cold-load, concurrency or operational
inference deadlines. Large traces and weights remain outside Git.

Training source `045129705d266ae510e9f064ccc376512472f2d5`; frozen parent archive
index `6451d127fbdc9a5e8b9e0a285b55c794e75297e5e67653984f55c7152c39c957`.
Current archive `.tmp/research/rl-proposal-penalty-v6-20261013`, index
`60e9adfcc1e48c6a6c91e85c90b88544a59b23c5a86c2a9137cf328c6b59655b`.
Use a fresh output directory for reproduction:

```sh
python scripts/research/run_rl_penalty_v6.py --run-registered-penalty-v6 \
  --panel .tmp/research/historical-funding-campaign-v1/registered-development-panel.csv \
  --settlements .tmp/research/historical-funding-campaign-v1/registered-development-settlements.csv \
  --output <fresh-research-directory>
python scripts/research/summarize_rl_penalty_v6.py \
  --source <research-directory> --parent .tmp/research/rl-matched-horizon-v5-20261013 \
  --output <fresh-evidence-directory> --expected-index-sha256 <recorded-index-hash>
bash scripts/verify.sh full
```

Use pinned Python 3.13.3/NumPy 2.3.5/Z3-solver 4.15.4.0/GHC 9.4.8/Node 20.19.0,
clean `env -i`, explicit tool PATH, empty `TRADER_ENV_FILE` to prevent local
credential-file discovery, and `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
At the original October 13 delivery, no second independent v6 training
replication had been performed. The October 14 continuation below now records
fresh-process numerical retraining; deterministic fixtures alone did not do so.

The current champion and historical vintages are unavailable. The user authorized
continuation without that export; adoption remains blocked. No trading credentials
were accessed. The recorded verified main base is
`0b76f2b879f75f5c669fe944a14ea6700d446240`, with
[successful CI](https://github.com/diegueins680/trader/actions/runs/38313512581).
Working branch: `research/rl-causal-campaign-2026-10-13`. Publication remains local:
scheduled autoloop reads PRs in all states and can push the default branch, whose
push can deploy. Draft status does not establish isolation. No branch push, PR,
merge or deployment occurs; existing PR342 is untouched.

RL research: this registered experiment is complete; the broader mission is
partially completed. RL adoption: reject this package and continue research;
prefer the simpler policy for this empirical comparison only. Formal assurance:
scoped checks with 18 broad obligations unresolved, no integration eligibility.
Financial execution: no real orders, live exploration or new live authorization;
existing production activity is preserved.

## Completed verification and local delivery

`python scripts/formal/verify.py --record` passed in 380.408s after the
canonical statement was synchronized and one unchanged retry. The first
canonical-statement check failed; the next attempt timed out in the existing
`ignore_term` process fault fixture at three seconds. Its root cause is not
established. The timeout, logs and successful unchanged retry are retained;
no deadline or assertion was relaxed. An earlier corruption-test fixture also
failed because its constant-long toy produced no changed proposal; alternating
long/short actions repaired the fixture without changing the audited contract.

Final `bash scripts/verify.sh full` exited 0 on code revision `74e692d0`:
343 formal regressions (232.427s), seven archive regressions, four inference
diagnostic regressions, 33 RL/accounting tests (6.845s), certificate reproduction
(232.889s), Haskell build/fourmolu/hlint/smoke/one Cabal suite, web typecheck/241
tests/build and 185 automation tests (19.269s). No target was skipped or
unavailable. The [verification receipt](penalty-v6-results/verification.json)
contains exact commands, environment, failed/timed-out attempts and log hashes.
The earlier timeout remains a reliability observation, not a discharged
hard-real-time guarantee. Documentation/receipt packaging follows that code check.

The 18 broad obligations remaining open or partial are: no-lookahead; revision
timing; finite outputs; overflow/underflow; live-authority isolation; observation
causality; valid transitions; terminal accounting; action safety; invalid-output
fallback; replay determinism; rollback; configuration bounds; atomicity; race
freedom; shutdown termination; recovery; and no permanent lockout. Their exact
scopes, assumptions and code mappings remain in the
[assurance ledger](../../formal/research/proof-ledger.json). This list includes
broader inherited/frozen-path claims; scoped research checks do not silently
close them or assert that inherited production is correct.

The local review packet is `.tmp/research/rl-penalty-v6-delivery-20261013`: a
cumulative patch from the recorded main SHA, commit list, unsubmitted draft PR
body, verification logs and a hashed delivery manifest. Its patch is checked
against the final committed tree. No branch push or PR is performed because of
the documented automation side effects. The pre-existing `.playwright-mcp/`
directory, production application, workflows, deployment and authorization
configuration remain unchanged.

## Fresh-process replication, 2026-10-14

The [preregistered repeatability check](../registrations/rl-penalty-reproduction-2026-10-14.json)
completed all ten fresh PPO fits and 378 replays without a failed fit or path.
No learner, coefficient, data, execution assumption, shield, checkpoint rule or
seed changed. All 15 experiment source hashes equal the frozen parent; remote
main and its latest successful CI were rechecked and still identify
`0b76f2b879f75f5c669fe944a14ea6700d446240`. Registration was committed as
`e0ca94459d4e9676ae1df6685a457f0cf80e8225` before the rerun.

The [complete comparison](penalty-v6-reproduction/replication.json) passes for
2,590 trained parameters, 40,960 training transitions across 643 episodes and
24,192 evaluation rows across 378 paths. Each archive independently passes
roster/event reconciliation, policy/trace hash binding, training chronology and
reward/economic ledger audits before comparison. Integer actions, Boolean
flags, fields, episode boundaries and statuses match exactly; floating values
use absolute/relative tolerance 1e-12. Both simulated buffers match exactly.
Zero violations were detected in the scoped audits; terminal combined
fill/liquidation and universal runtime-refinement limitations remain as above.
The comparator rejects the same archive path and requires the registered parent
index. Hashes establish identity under supplied trust anchors, not authenticity
or proof that an arbitrary third party executed training. Our fresh execution
is separately recorded in the campaign log.

The unchanged mean net return is +0.515%, with seed means
+0.159/+1.647/+0.052/-0.748/+1.464% in seed order 11/23/47/71/101.
Mean intervention 35.052% still fails the 20% gate; frozen momentum remains
+1.487%. Severe costs still produce -0.056%. Previous dependence-aware
intervals and all other economic results reproduce; no new hypothesis test,
market sample or selection trial is claimed. This is numerical reproducibility
on the same pinned software/platform, not independent implementation,
cross-hardware bitwise determinism, new statistical power or confirmation.
No CQL/DDQN retraining occurred in this continuation.

The campaign timer was 192.381s; fit/artifact/trace-write/warm-benchmark time
totaled 122.347s, not process CPU time. Peak RSS was 104,247,296 bytes;
largest policy 5,034 bytes; worst warm Python p99 was 0.1314ms. These figures
exclude operational load/latency/concurrency guarantees. The
[run summary](penalty-v6-reproduction/run-summary.json) retains every fit's
seed, timing, loss endpoints, training failures, audit sums and artifact hashes.
The complete 388-event registry and newly executed economic records are in
the authenticated archive, outside Git.

Archive: `.tmp/research/rl-penalty-reproduction-20261014`; index SHA-256
`521eb83ef7b09bde69111c26886cd5e12ce630ed1b4c9cb6b87eea900c6e1513`.
Run the same v6 training command above into a fresh directory. Compare it using
`python scripts/research/compare_rl_penalty_replication.py --parent <frozen-v6>
--replica <new-v6> --parent-sha256 <registered-parent-index>
--replica-sha256 <new-index> --output <new-report.json>`. Unlike the legacy
32-step comparator, this explicitly requires the 64-step penalty schema and
compares every shaped/economic training row. It writes one new report outside
both archives and cannot train, infer, promote or access an exchange. Eleven
archive regressions exercise malformed policies, parameter values, reward/action
corruption, missing rows, source drift and exclusion boundaries.

An October 14 primary-proceedings metadata/abstract refresh rechecked
[Henderson et al. (AAAI 2018)](https://ojs.aaai.org/index.php/AAAI/article/view/11694)
on RL reproducibility and
[Agarwal et al. (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
on few-run evaluation uncertainty. Neither establishes financial efficacy.
Their role here is to separate numerical repetition, seed variability and market
uncertainty; no independent-task bootstrap was transplanted to dependent markets.
Queries, versions, disposition and redistribution/code-license limitations are
recorded in the registration. No PDFs or external implementations were imported.

The repeatability gap is closed for this exact numerical package. Adoption stays
rejected: champion vintages, authentic availability, untouched confirmation,
real-log OPE and production safety/inference integration remain absent. The
18 broad formal obligations are unchanged; no new theorem or live authority
follows from equality of observed traces.

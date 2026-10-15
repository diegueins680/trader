# Frozen PPO policy extraction diagnostic, 2026-10-14

This adaptive development study evaluates the stochastic policy represented by
already trained PPO actors against their registered greedy deployment decoder.
The sequential decision is bounded target exposure: current inventory, pending
actions, costs and terminal settlement connect decisions across time. It changes
no weights, observations, shield, economic reward or execution assumptions.
Prior greedy results remain valid for the decoder they evaluated.

Protocol: [registration](../registrations/rl-policy-extraction-v7.json), committed
as `1b290816` before execution; runner source `9bd904a6`. Latest verified main is
`0b76f2b879f75f5c669fe944a14ea6700d446240`, with successful
[canonical CI](https://github.com/diegueins680/trader/actions/runs/38313512581).
The branch is `research/rl-causal-campaign-2026-10-13`.

Twenty immutable PPO actors come from the unshaped v5 and penalized v6 studies:
five training seeds, two folds, two packages. They previously received 4,096
fresh simulator transitions each. No actor is retrained here. Simulator Double
DQN and fixed simulated-dataset CQL were trained in the earlier campaign; this
study is an extraction ablation of PPO, not a fourth learner. BC and the other
learners retain their frozen earlier evidence.

The roster is 2,100 categorical replays (five action seeds per actor), 420 greedy
replays and 168 deterministic controls. Cash, constant-long, normalized momentum
and reversal use the same available information and constraints. Reconstructed
greedy/control traces must match their frozen parents within absolute/relative
1e-12, with exact discrete fields. Sampling uses the existing stable softmax,
checked half-open inverse-CDF intervals, PCG64 and registered SeedSequence keys.
Common uniforms pair packages/stresses. Each path resets its RNG and legitimate
observation state; there is no calibration or parameter update during evaluation.

Only hash-registered Binance development exports are read: BTCUSDT, ETHUSDT and
SOLUSDT at eight-hour intervals, two already-consumed 64-return windows per
symbol. Availability is assumed close+1ms, not authenticated first-seen data.
Independent unit-equity futures accounts use next-close price-taking replay,
funding on old units, explicit pending targets, partial fills and one costed
terminal liquidation. Base cost is ten basis points of traded notional;
scenarios are 1.5x, 2x, severe50bp, one extra bar of delay, 50% fills and doubled
funding. No queue, market-impact, precision, min-notional or live-portfolio claim.
The current champion's causal configuration/model vintages are unavailable;
the user authorized continuation without that export. No untouched holdout or
protected prospective carry period is opened.

Six registered contrasts compare each sampled package to its own greedy policy,
cash and normalized momentum. Action replicas, training seeds and symbols are
averaged before circular chronological resampling within each fold (blocks 3/9/21,
2,000 draws, seed 20261010, six-contrast Bonferroni endpoints). These short, exposed
windows give descriptive uncertainty only. Seeds are not new market histories.
Any failed/short path invalidates its full-horizon contrast; stopped endpoints
are retained. DSR/PBO and real-log IS/DR/FQE remain insufficient/unsupported.
Raw proposal probabilities describe the simulated sampler, not shielded-action
probabilities or unknown exchange behavior. No real-log ESS is invented.

The original gates remain: net materiality 0.001 and positive simultaneous lower
bound over the matched rules, drawdown<=15%, ES95<=2%, interventions<=20%, zero
unsafe executable intents and no failed paths. They cannot overcome missing
champion, authentic availability, final confirmation or production assurance.
No decoder is selected automatically from this adaptive comparison.

The independent evidence exporter reconstructs the registered uniform streams,
checks categorical intervals with separate scalar code, reconciles all realized
ledger rows and checks the complete event/ID roster. Eight decoder regressions
cover zero mass, boundaries, invalid values, analytical frequencies, NumPy
reference sampling, artifact corruption and composed toy replay. This is
conformance evidence, not a theorem about binary64, RNG quality or all inputs.
Existing nine exact-real/integer composition queries, two admission lemmas,
43-state/107-edge depth-eight lifecycle model and eight Haskell Rational ledger
witnesses retain their original assumptions. Eighteen broad obligations remain
open/partial. Terminal fill/liquidation reconstruction remains only partially
independent; tests and hashes do not establish whole-system correctness. Current
12-input artifacts have no certified Haskell inference integration.

The academic rationale is limited: official
[Spinning Up PPO documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
describes stochastic training, while
[Stable Baselines3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
treats deterministic evaluation as a selectable convention. These references
motivate checking extraction; neither predicts a financial advantage. Access date,
versions, disposition and code-license limitations are in the registration;
no external implementation or paper PDF is copied.

Reproduction requires the pinned local Python 3.13.3/NumPy 2.3.5 environment and the
registered external archives/exports. Run with a fresh output directory:

```sh
env -i PATH="$PWD/.tmp/rl-20261013-venv/bin:/usr/bin:/bin" \
  TRADER_ENV_FILE= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/research/run_rl_extraction_v7.py \
  --run-registered-extraction-v7 \
  --panel .tmp/research/historical-funding-campaign-v1/registered-development-panel.csv \
  --settlements .tmp/research/historical-funding-campaign-v1/registered-development-settlements.csv \
  --output .tmp/research/rl-policy-extraction-v7-fresh
```

The runner rejects absent opt-in, unregistered bytes, incompatible artifacts,
uncommitted experiment sources and output outside isolated `.tmp`. Large traces
and weights remain outside Git. The canonical `bash scripts/verify.sh formal`
and `bash scripts/verify.sh full` include the decoder tests and source bindings.

No push or draft PR is safe under the inspected workflow chain:
`.github/workflows/autoloop.yml` runs every six hours with write authority;
`scripts/autoloop.mjs` considers PRs in all states; resulting default-branch
repairs can reach `.github/workflows/deploy-hetzner.yml` after successful CI.
A draft flag does not isolate that chain. Preserve local commits/patch instead.
No workflow, production authorization, order permission, risk limit, adopted
strategy, credential or production process is modified.

Resource amendment (before outcome inspection): progress-only monitoring at
672/2,688 paths after 7m 52s justified one extra 20-minute process under the
[registered continuation protocol](../registrations/rl-policy-extraction-v7-continuation.json),
commit `3e971ac2`. The initial 20-minute stop remains enforced. A continuation
must admit the hash-frozen exact event/trace prefix, retain failures and copy
completed paths unchanged; it executes only missing IDs. No additional model,
seed, scenario, training step or selected checkpoint is introduced. Tests reject
reordered prefixes, orphans, unrecognized stops and duplicate/second continuations.
This resource amendment is disclosed rather than treating the study as one
uninterrupted run within its original estimate.

## Executed results and decision

**Reject both sampled PPO packages for adoption.** Categorical extraction fails the registered economic and intervention gates. The complete roster contains 2,688 paths/172,032 transitions, with no failed paths. The independent audit detects zero violations of its scoped ledger and sampling checks;588 greedy/control traces reproduce frozen parents. This does not prove universal executable safety.

Mean net returns below are per 64-transition window, equally weighted across folds/symbols/training seeds and, for sampled policies, action seeds. They are not annualized or cumulative portfolio returns.

| Policy | Mean net | Mean costs | Mean intervention | Worst drawdown | Worst ES95 proxy |
|---|---:|---:|---:|---:|---:|
| cash | +0.000% | 0.000% | 0.000% | 0.000% | -0.000% |
| constant_long | +0.997% | 0.199% | 60.156% | 4.480% | 1.559% |
| normalized_momentum3 | +1.487% | 0.437% | 64.844% | 2.655% | 1.162% |
| normalized_reversal3 | -2.376% | 0.425% | 64.844% | 4.952% | 1.713% |
| v5_greedy | +0.437% | 0.204% | 47.135% | 5.179% | 1.559% |
| v5_sampled | -0.627% | 0.531% | 43.677% | 10.017% | 1.754% |
| v6_greedy | +0.515% | 0.143% | 35.052% | 5.339% | 1.559% |
| v6_sampled | -0.644% | 0.529% | 43.479% | 10.017% | 1.754% |

The inherited field named `ES95` averages the worst `ceil(0.05*64)=4` one-bar returns: a coarse 6.25% tail average, not an exact 5% expected-shortfall estimator. Preserve its registered numerical semantics and disclose the naming/finite-sample limitation. Values below the 2% proxy threshold do not close a canonical ES95 risk gate; tail evidence is insufficient. Any corrected metric needs a versioned prospective registration, not a favorable post-outcome substitution.

All five training seeds, averaging action seeds/symbols/folds; no best-seed selection:

| Training seed | Unshaped sampled v5 | Penalized sampled v6 |
|---|---:|---:|
| 11 | -1.226% | -1.233% |
| 23 | -0.333% | -0.376% |
| 47 | -1.422% | -1.436% |
| 71 | +0.249% | +0.234% |
| 101 | -0.400% | -0.409% |

Training-seed mean sample SD is 0.689 and 0.684 percentage points, respectively. Four of five training-seed means are negative for each package. These are algorithmic replications of the same market windows.

| Action RNG seed | Unshaped sampled v5 | Penalized sampled v6 |
|---|---:|---:|
| 19 | -0.966% | -0.970% |
| 37 | -0.178% | -0.189% |
| 59 | -0.792% | -0.795% |
| 83 | -0.293% | -0.344% |
| 109 | -0.904% | -0.923% |

All 25 training/action-seed combinations per package and scenario, per-symbol/fold results and all 2,688 path records are in [evidence.json](extraction-v7-results/evidence.json), [all-paths.csv](extraction-v7-results/all-paths.csv) and [trial-registry.json](extraction-v7-results/trial-registry.json). No action seed yields a positive aggregate base return.

| Stress | Unshaped sampled v5 net | Penalized sampled v6 net |
|---|---:|---:|
| base | -0.627% | -0.644% |
| cost1_5x | -0.889% | -0.906% |
| cost2x | -1.152% | -1.168% |
| severe50bp | -2.710% | -2.720% |
| delay1bar | -0.307% | -0.301% |
| partial50pct | -0.393% | -0.403% |
| funding2x | -0.628% | -0.646% |

Worst stressed drawdown is 13.516% and the worst registered tail proxy is 1.900% for both packages. These finite-path observations are not future loss guarantees. Mean intervention remains above 20% (individual base paths reach 65.625%). Neither profitability nor operational usability passes. Normalized momentum also fails the intervention gate; its positive development return is not an adoption authorization.

Nine-bar circular-bootstrap intervals below are simultaneous 95% descriptive intervals for the six registered contrasts. Full 3/9/21-bar sensitivity is in evidence.json. Negative estimates and intervals do not justify a universal claim about PPO; cumulative adaptive-search uncertainty is uncontrolled.

| Sampled policy versus comparator | Net difference (pp) | Interval (pp) |
|---|---:|---:|
| v5_sampled versus v5_greedy | -1.063 | [-2.692, +0.252] |
| v5_sampled versus cash | -0.627 | [-1.255, -0.183] |
| v5_sampled versus normalized_momentum3 | -2.114 | [-6.235, +1.057] |
| v6_sampled versus v6_greedy | -1.159 | [-2.715, -0.132] |
| v6_sampled versus cash | -0.644 | [-1.295, -0.191] |
| v6_sampled versus normalized_momentum3 | -2.132 | [-6.280, +1.048] |

Economic reconciliation: v5 sampled mean gross PnL−0.094575%, funding−0.001423%
and costs 0.530593% give net−0.626592%. V6 sampled gross−0.113794%,
funding−0.001455% and costs 0.528838% give net−0.644088%. The artificial
training penalty is absent from every evaluated financial return. Mean turnover
is 5.306/5.288 starting-equity units, mean absolute exposure 5.644%/5.616%,
and mean nonzero-fill rows 23.127/23.033. These are fill-bearing rows, not a
validated count of exchange orders; terminal rebalancing/close can share a row.
Mean holding periods are 1.362/1.359 bars (median 1); longest drawdown duration 64
bars for both. Both fold means and all three symbol means are negative.

Across all 2,688 paths,73,608 action changes comprise 69,804 support projections
and 3,804 turnover admissions. The 12,288 deferred and 12,288 ignored new proposals
are separate scheduling diagnostics. No audited unsafe executable intent or
accounting violation was detected; raw shield interventions are not executed
constraint violations. Terminal combined fill/liquidation correspondence and
unmodeled disturbances remain limitations, so this is an empirical audit claim.

The policies remain nearly uniform: sampled-path mean entropy 1.098232/1.098240
nats versus log(3)=1.098612; mean maximum probability 0.342219/0.341793. Across
all stresses, v5 proposal counts(short/flat/long) are 22,188/22,423/22,589 and v6
counts 22,064/22,630/22,506. Only 34.036%/34.368% of sampled actions equal the local
greedy action. These diagnostics suggest weak separation between action logits.
The previous large greedy intervention reduction does not transfer to sampling:
43.677% to 43.479% is only 0.198 percentage points. This is descriptive evidence
about these checkpoints, not proof of convergence or a general rejection of PPO.
The intervention penalty does not repair their sampled economic performance.

Policy cards: `v5_sampled` is a frozen unshaped PPO actor plus
`categorical-inverse-cdf-v7`, unchanged v2 shield and ledger; `v6_sampled` uses the
frozen intervention-penalized actor with the same decoder/shield/ledger. Both are
research-only, disabled by default, rejected, with no production loader or order
capability. Greedy v5/v6 remain frozen rejected references. No best decoder,
checkpoint, seed or exposure parameter is selected automatically. Cash is the
zero-return comparator, not a proposed change to existing production ownership.

## Execution, resources and artifact identity

Initial phase source `9bd904a66e82c5829ddfb7f463e1ece8939a7955` stopped with exit 1
at its registered wall-budget check after 2,064 paths. Its log creation-to-last-write
interval is 1,202.045 s (OS timestamps, includes interpreter startup); the runner's
monotonic elapsed time is known to exceed 1,200 s at the per-path check. The initial
traceback was rendered after the working-tree runner had gained continuation
support, so displayed source excerpts can differ from the executed revision;
use the frozen commit and manifest hashes to resolve its frame lines. Loaded
Python code and the economic kernel were not hot-reloaded. This source/runtime
correspondence relies on the documented CPython assumptions.

The single continuation, source `8c9a500b138192a6b7f024689ee7b8c4df4c1813`, copied
2,064 events/traces and executed only 624 remaining paths, finishing in 432.117 s.
There are 2,688 unique IDs, no replaced failures and no additional training steps.
Approximately 27.24 minutes of observed phase time is below the amended 40-minute
combined allowance; the original 20-minute estimate was not met. Preserve that
failure and the before-outcome resource amendment. No paid compute was used.

Continuation peak RSS is 225,509,376 bytes, dominated by archive/replay processing;
initial-phase peak RSS was not captured. P99 for the first 10,000 sampled calls in
the continuation is 3.165 ms, including observation construction and decoding,
first-call effects and host contention. Those calls are v6/fold1; this is not
an all-artifact latency certificate, inference-only RSS, Haskell parity or a
hard deadline guarantee. Earlier warm numerical-only benchmarks are a different
measurement and must not be compared as equivalent timings.

External archives (large traces/models are not committed):

- Initial prefix: `.tmp/research/rl-policy-extraction-v7-20261014`, index SHA256
  `dae263af91f16585e14686620b810d2cb33be3458ae2de668d515fcd3b044702`.
- Completed two-phase package: `.tmp/research/rl-policy-extraction-v7-continued-20261014`,
  index SHA256 `2f7bf0d9f8c0f987e4438e12af0a69e985243a44e06420a30f17dede3036f58f`.
- Logs and command receipts: `.tmp/rl-extraction-20261014/logs`.

To reproduce the exact two-phase provenance, use the initial source commit for
the first process, retain its registered budget stop, freeze the partial index,
then use the continuation source with `--resume-prefix <initial-directory>` and
`--resume-sha256 <initial-index-hash>`. Fresh wall-clock runs may stop at different
path boundaries; fixed RNG keys and preserved complete episodes keep numerical
semantics unchanged, conditional on the tested runtime. No separate full
numerical reproduction of v7 has been executed. The local incremental Git bundle
retains intermediate source revisions; the final-tree patch alone does not.
The exporter command is:

```sh
python scripts/research/summarize_rl_extraction_v7.py \
  --source .tmp/research/rl-policy-extraction-v7-continued-20261014 \
  --output <fresh-evidence-directory> \
  --sha256 2f7bf0d9f8c0f987e4438e12af0a69e985243a44e06420a30f17dede3036f58f
```

RL research: this registered decoder diagnostic is complete; the broader mission
is partially completed. RL adoption: reject both sampled packages and retain
simpler research comparators. Formal assurance: scoped existing claims and new
conformance, with 18 broad obligations still open/partial. Financial execution:
no real orders, live exploration or new authorization; existing production is
preserved.

## Verification history

The first `python scripts/formal/verify.py --record` attempt stopped after
3.142 s because the new extraction test step in `scripts/verify.sh` still had
its old source pin. Review confirmed a single added regression command;
commit `26f09824` updates only that expected hash. No check was disabled.

The next attempt stopped after 346.597 s: the existing compiled
`startup_hang` worker fault fixture exceeded its unchanged three-second
conformance timeout. Temporary-directory deletion then reported
`Errno 66: Directory not empty`. The fixture PID was subsequently absent;
no signal was sent and no production process was touched. Preserve both
errors, the PID record and command receipt. Host contention is a possibility,
not an established root cause. One unchanged retry is recorded separately;
a passing retry does not prove the timing issue fixed or establish a hard
real-time guarantee. The source pin error and this timeout are distinct from
the registered campaign resource stop. Prior October13 failure history remains
in [the earlier receipt](penalty-v6-results/verification.json).

The affected-path formal statements remain precisely scoped: nine nonvacuous
SMT queries check target projection, net reward, telescoping equity, next-close
ordering, matured training stops, single-slot admission, projected action mass,
training-reward decomposition and nonpositive shaping over exact
reals/integers/booleans. Two further admission lemmas bound zero-target turnover
and post-cost residual exposure when pre-fill absolute exposure is at most 0.35,
fill is in (0,1] and linear cost is at most 50bp. The lifecycle model exhaustively
explores 43 states/107 edges through depth 8; eight actual Haskell Rational cases
provide implementation conformance. None is a global binary64, filesystem,
network-profitability, production-concurrency or arbitrary-disturbance theorem.
The [machine-readable assurance ledger](../../formal/research/proof-ledger.json)
maps claims, assumptions, sources, tests, bounds and remaining obligations.

The unchanged retry failed after261.303 s in the default-mode worker wait,
again at the unchanged three-second limit. No further certificate-generation
retry is selected. The cause is unresolved. The canonical global
`formal/research/results.json` has **not** been regenerated or edited to imply
success for this revision; full-certificate readiness remains blocked.

The affected RL checker, `python scripts/formal/rl_composition_check.py`, passed
in5.540 s on `26f09824`:29 source identities, the nine composition queries,
two admission lemmas, bounded lifecycle and eight Haskell ledger witnesses.
Its actual JSON output is [scoped-formal.json](extraction-v7-results/scoped-formal.json).
This fresh scoped check is separate from the blocked global certificate. The
machine-readable assurance ledger retains both research gates as open and now
links the negative economic evidence and scoped proof output, with explicit
process-timing and tail-metric blockers. The required full wrapper and any
unreached area wrappers are recorded separately below.

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
averaged before circular chronological resampling within each fold (blocks3/9/21,
2,000 draws, seed20261010, six-contrast Bonferroni endpoints). These short, exposed
windows give descriptive uncertainty only. Seeds are not new market histories.
Any failed/short path invalidates its full-horizon contrast; stopped endpoints
are retained. DSR/PBO and real-log IS/DR/FQE remain insufficient/unsupported.
Raw proposal probabilities describe the simulated sampler, not shielded-action
probabilities or unknown exchange behavior. No real-log ESS is invented.

The original gates remain: net materiality0.001 and positive simultaneous lower
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

Reproduction requires the pinned local Python3.13.3/NumPy2.3.5 environment and the
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
672/2,688 paths after7m52s justified one extra20-minute process under the
[registered continuation protocol](../registrations/rl-policy-extraction-v7-continuation.json),
commit `3e971ac2`. The initial20-minute stop remains enforced. A continuation
must admit the hash-frozen exact event/trace prefix, retain failures and copy
completed paths unchanged; it executes only missing IDs. No additional model,
seed, scenario, training step or selected checkpoint is introduced. Tests reject
reordered prefixes, orphans, unrecognized stops and duplicate/second continuations.
This resource amendment is disclosed rather than treating the study as one
uninterrupted run within its original estimate.

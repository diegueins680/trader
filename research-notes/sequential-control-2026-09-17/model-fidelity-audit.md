# Fidelity and contract audit at f669c7e9

This is a delta audit of the [earlier full inventory](../market-prediction-2026-09-04/model-fidelity-audit.md),
checked against latest main on 2026-09-17. The prior audit is not treated as a new
experiment. `git diff 1563f63d..f669c7e9 -- haskell/app/Trader/Predictors` identifies
schema/exogenous changes and a quantile shape-check refactor; the three named
sequence predictors retain their earlier implementation bytes. Direct inspection
confirms the following classifications.

| Component / code | Classification | Actual behavior and economically material gap |
|---|---|---|
| GBDT | Faithful with simplifications | Squared-error boosted stumps with bounded threshold candidates; no general interaction-depth boosting |
| DecisionTree | Faithful with simplifications | Recursive regression splits, depth/leaf limits; no pruning or calibrated distribution |
| KNN | Faithful with simplifications | Causally standardized distance-weighted neighbors; bounded history and distance concentration |
| HMM | Faithful with simplifications | Three Gaussian states, Baum–Welch and filtering; heuristic economic state labels |
| LSTM | Faithful with simplifications | Real recurrent gates and gradient training; target/validation/provenance limitations persist |
| OnlineNeural | Faithful feed-forward online neural method | Two hidden tanh layers, delayed updates; not an LSTM or RL policy |
| Kalman / Kalman3 / physics / fusion | Faithful with simplifications | Recursive state-space filters and engineered noise/model combinations; risk-register numerical issue is now CLOSED, superseding the older audit wording |
| `Predictors/TCN.hs` | Lightweight proxy | `tcnLags` selects fixed dilated returns; `trainTCNWithLambda` fits ridge weights; no learned convolution |
| `Predictors/PatchTST.hs` | Lightweight proxy | Five statistics per trailing patch; ridge fit; no token embeddings, learned attention or channel-independent Transformer |
| `Predictors/Transformer.hs` | Lightweight proxy | Bounded stored examples, scaled dot-product similarity, softmax target average; no learned Q/K/V or Transformer training |
| Quantile | Faithful with simplifications | Independent pinball-loss linear heads with reordered output; shape refactor is not new quantile semantics |
| Conformal | Inspired | GBDT residual split radius and adaptive heuristic; no proven conditional coverage in serial market data |
| TA / regimes | Inspired heuristics | Trend/reversal/breakout/routing rules; selection and regime thresholds are research degrees of freedom |
| Ensembles / sensor fusion | Inspired combinations and online experts | Blending, routing, uncertainty and disagreement gates; adaptive losses must be delayed |
| MarketContext | Linear factor with data limitations | Legacy historical membership/weights may reflect one cutoff snapshot; v2 selector/panel/receipt boundaries are isolated |
| CrossSectionalMomentum | Mislabeled for ranking | Single-symbol residual time-series momentum; true point-in-time ranking remains offline research |
| Prediction bias / calibration | Incomplete / unverified economically | No accepted bias-correction net-OOS result or general calibrated conditional-return interface |
| Optimizer / holdout | Engineering implementation, partial research fidelity | Production DSR/PBO fields are explicitly proxies; canonical research diagnostics and one-shot registry are stronger |
| `bandit_router` | UCB-inspired adaptive expert router | `Optimization.hs` calls `banditPredictionsWithModelsV`; recent forecast/PnL evidence plus exploration bonus selects Kalman/LSTM/blend. It is not a trained Q function or actor–critic |
| Modern RL | Previously absent in audited research path | This branch adds isolated simulated PPO/Double DQN/CQL only; no runtime model ID or production caller |

The correct existing aliases already exist in `Predictors/Types.hs`:
`dilated_lag_ridge_v1`, `patch_summary_ridge_v1`, `similarity_attention_v1`.
Serialization still emits `tcn`, `patch_tst`, `transformer`. Existing migration
and alias tests remain applicable; no identifier, parser, saved configuration or
artifact acquires new semantics here. Replacing these aliases would duplicate
already merged work and risk compatibility.

## Paper-to-code gaps and decisions

| Literature | Current boundary | Difference deliberate/documented? | Economic significance | Feasible / justified action |
|---|---|---|---|---|
| PatchTST / TCN / Transformer | Ridge and similarity proxies | Yes, now explicitly documented and aliased | Architecture may matter, but no net evidence | Keep cheap baselines; no neural substitution |
| HAR / calibrated shallow models | Registered future paths and open artifact PRs | Yes | Availability/provenance precede fitting | Preserve registrations and avoid duplicate implementation |
| OFI / optimal execution | No verified historical event book / own queue | Documented missing evidence | Critical at short horizons | No bars-only OFI/execution claim |
| PPO | No pre-existing rollout/actor/critic | New research deviation: tiny fixed network and budget | Tests path-dependent exposure, not market prediction | Prototype without production import |
| Double DQN | No pre-existing Q/replay/target net | New research code; uniform replay and periodic target sync | Exposes maximization and simulator exploitation | Retain only as research comparator |
| CQL | No actual policy-action logs | Uniform simulated behavior, known propensity; explicitly synthetic actions on historical prices | Simulated support does not establish causal exchange counterfactuals | Test pessimism; block promotion |
| Shielding / constrained MDPs | Existing deterministic Haskell risk and order permissions | New isolated non-authorizing proposal type, no order adapter | Safety cannot rely on reward learning | Preserve deterministic precedence; no inferred deployment readiness |
| DR / weighted OPE / FQE | Canonical statistical harness but no RL OPE | Add IS/PDIS/WIS/DR on short simulated episodes | Long-horizon overlap and simulator bias dominate | Report ESS and uncertainty; no trusted real-money value estimate |

## Repository and operational contracts

The reviewed tracked fleet remains AVAX, UNI, SUI, ETC and ADA. Historical panel
coverage includes AVAX/ADA plus BTC/ETH and six other liquid benchmarks; UNI/SUI/ETC
are absent. Primary reviewed intervals are 1h/4h/8h, but this screen only has a
hash-pinned 8h panel and cannot claim adjacent-timeframe completeness.

Production flow remains Haskell features → predictors → method/ensemble →
deterministic signal/risk/capital/execution checks → authorized order adapter.
LSTM cache v1, top-combo JSON/PostgreSQL, CLI `--predictors`, API configuration
fields and React method selectors are unchanged. No current champion/adopted
UUID is exported from production or modified. Existing backtests include fees,
spread, slippage, impact parameters, funding and delays; those do not establish
queue position or historical venue filter validity. Optimizer scores, top-combo
admission, shadow selectors and paper/live authorization remain untouched.

The related open PRs read at preparation were [#249](https://github.com/diegueins680/trader/pull/249),
[#250](https://github.com/diegueins680/trader/pull/250), [#251](https://github.com/diegueins680/trader/pull/251)
and dependency PRs #253–255. The open issue list was empty. Older issue #119
is historical context, not a current open task. No comments or messages were
sent to maintainers. This research neither enabled trading nor changed existing
production authorization; the checked-in production profile already has its own
live fleet, so claiming the entire repository was live-disabled would be false.

## PPO/GAE numerical scope — 2026-09-28

The [terminal-target audit](terminal-numerics-followup-2026-09-28.md) identifies
current-source cancellation and non-finite carry from finite inputs. The GAE
recurrence has the expected terminal masks, but abstract real-number identities
do not establish binary64 reconstruction fidelity. The implementation, version,
frozen trials and rejected policy artifacts remain unchanged. No historical
prevalence is inferred; a repair must be separately identified and preregistered
before new financial trials.

The [isolated gae-targets-v2 kernel](target-v2-followup-2026-09-28.md) now supplies
direct terminal targets and whole-batch finite-output admission under a distinct,
default-disabled function contract. It is not connected to PPO, normalization or
optimizer updates; the frozen learner classification and numerical counterexamples
above are unchanged. No faithful full-PPO or economic claim follows.

## PPO objective continuation — 2026-09-28

The [source-linked audit](ppo-objective-followup-2026-09-28.md) verifies exact-real
ratio/loss and coefficient branches for the registered clipped surrogate, with
explicit binary64 literal values and conditional simplex assumptions. It does not
verify full softmax, array/runtime or learner behavior. CE-RL-012 returns finite loss
with NaN gradient; CE-RL-013 confirms the intended absence of a universal multiplier
cap. Keep the current learner frozen and rejected; no implicit successor or
financial rerun. Objective fidelity alone cannot justify candidate adoption.

## Value-based continuation — 2026-09-30

The [focused audit](value-objective-followup-2026-09-30.md) checks the unchanged
Double DQN target slice and scalar CQL gradient. The former preserves the named
selection/evaluation mechanism; the latter is a fixed-alpha CQL(H)-inspired
regularizer, not the original Atari QR-DQN reproduction. Conditional exact-real
SMT results do not establish conservative policy-value bounds or full training
correctness. CE-RL-014/015 preserve NaN loss and common-shift cancellation in the
current helper. All economic rejection decisions and identifiers remain unchanged.

## Optimizer publication continuation — 2026-09-30

The [audit](optimizer-publication-followup-2026-09-30.md) classifies the unchanged
optimizer as Adam with explicit global clipping and validation/staging semantics,
not an unconditional convergence or transaction guarantee. Exact-real clipping is
conditionally checked; rounded norm bounds remain open. CE-RL-016 refutes the
strong all-interruption publication model and preserves a pinned-runtime diagnostic.
The opcode exception has documented interpreter limitations. Learner identifiers,
parameters, formulas, economic rejection and champion behavior are unchanged.

## Inference boundary continuation — 2026-09-30

The [audit](inference-boundary-followup-2026-09-30.md) confirms representation and
measured-time admission, first-maximum tie breaking and fixed bounded proposals.
Equal valid scores select short exposure; this wrapper has no uncertainty-aware
abstention rule. The second clock read precedes output validation and selection;
its post-call limit neither preempts a stalled call nor bounds end-to-end runtime.
These are unchanged semantics, not a new policy or a whole-neural-network proof.
CE-RL-017 preserves the pending-call model. Full deadline/production readiness
remains unverified and all economic rejection decisions stand.

## OPE assurance continuation (2026-09-30)

The [source-linked audit](ope-algebra-followup-2026-09-30.md) separates exact-real ESS/WIS/DR algebra from a preserved general-helper ESS underflow witness. The current deterministic six-step weights are checked separately. The source remains unchanged; all 108 invalid OPE batches retain their status and no new economic or holdout evidence is produced. WDR/MAGIC are not implemented by the existing ordinary DR plus trajectory WIS helper.

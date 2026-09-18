# Candidate and RL algorithm selection

Cutoff: 2026-09-17. Scores express research priority, not measured trading efficacy.
The prior 50-paper review and its three future-data registrations are retained.
No earlier rejected reversal configuration is retried. There are four high-level
families in this work's map, only one newly prototyped.

| Family | Mechanism | Prior evidence / gap | Disposition |
|---|---|---|---|
| HAR volatility risk gate | Predict persistent variance; reduce existing exposure | Strong variance mechanism, no accepted new causal data; PR #250 already implements isolated artifacts | Retain prior registration; do not duplicate or run early |
| Missingness-aware shallow calibrated model | Distinguish unavailable inputs and calibrate abstention | Material schema gap; PRs #249/#251 and related work already cover artifacts/panels | Retain prior registration; do not duplicate or run early |
| Depth-normalized OFI | Short-lived supply/demand pressure | Requires event-level book history, queue and latency evidence absent here | Monitor existing registration; no close-bar substitute |
| Bounded sequential exposure control | Inventory, transaction costs and terminal obligations couple decisions | No dedicated modern RL environment or three-paradigm comparison on main | Prototype offline screen only; no promotion eligibility |

The RL use case is **partially observed inventory control**. Price prediction is
a supervised task. Static allocation with independent daily resets can be an
optimizer; model selection with fully revealed forecast losses is an experts or
bandit problem. Inventory carry, pending fills, funding timing and turnover make
today's exposure affect tomorrow's feasible and economical decisions. A finite
observation is not asserted to be a sufficient Markov state. Historical replay
conditions transitions on a hidden date and exogenous price path; it cannot
identify endogenous market impact. This limits any claim of policy optimality.

Execution scheduling and market making would be more natural sequential uses,
but missing L2/queue/own-order data prevent credible environment construction.
Portfolio allocation adds coverage and point-in-time universe requirements.
Hierarchical strategy/model/ensemble selection, meta-learning, continual learning
and end-to-end trading add unjustified degrees of freedom at this stage.

## Transparent RL scorecard

Rate each dimension 0–4: 0 absent/inapplicable, 1 weak, 2 partial, 3 established,
4 strong. Weighted sum divided by four gives a 0–100 suitability score.
Dimensions and weights: academic quality **A10**, independent replication **I10**,
financial decision relevance **F10**, crypto transfer **C5**, non-stationarity
robustness **N5**, sample efficiency **S5**, offline capability **O10**, risk control
**R10**, reproducibility **P10**, CPU feasibility **U5**, interpretability **T5**,
manageable support requirements **D5**, formal/operational compatibility **H10**.
Replication refers to algorithmic benchmark reproduction, never established
crypto alpha. No algorithm scores strongly on crypto non-stationarity.

| Framework | A I F C N S O R P U T D H | Score / 100 | Why selected or deferred |
|---|---|---|---|
| Deterministic cost-aware ridge policy | 4 4 3 3 2 4 4 3 4 4 4 3 4 | 90.0 | Preferred complexity baseline; explicit turnover objective |
| Contextual reward regression / bandit | 4 4 3 2 2 4 4 2 4 4 4 2 4 | 85.0 | Mandatory myopic comparison; logged action rewards only |
| PPO | 3 4 2 1 1 1 0 1 4 4 2 1 3 | 55.0 | Modern on-policy actor–critic comparator; simulation interaction only |
| Double DQN | 4 4 2 1 1 3 1 1 4 4 2 1 3 | 62.5 | Discrete off-policy comparator; isolate maximization bias |
| Discrete CQL | 4 3 2 1 1 3 4 2 4 4 2 2 3 | 71.25 | Offline comparator; finite-action pessimism is directly auditable |
| IQL / TD3+BC / BCQ / BEAR / AWAC | 4 3 2 1 1 3 4 2 3 3 2 2 3 | 67.5 | Credible offline alternatives; continuous-action variants unnecessary here |
| SAC / TD3 | 4 4 2 1 1 3 1 1 4 3 2 1 3 | 61.25 | Continuous control; deferred for discrete three-target contract |
| C51 / QR-DQN / IQN / distributional actor–critic | 4 3 2 1 1 2 1 2 3 3 1 1 3 | 56.25 | Return distributions useful for tails, but not automatically epistemic uncertainty or calibrated CVaR |
| CPO / Lagrangian / primal–dual CVaR | 4 3 3 2 1 2 1 3 3 3 2 1 3 | 63.75 | Expected constraints supplement hard external limits; no pathwise guarantee |
| Decision Transformer / trajectory models | 4 3 1 1 1 2 4 1 3 2 1 1 2 | 55.0 | Insufficient trajectory diversity and behavior support |
| MBPO / Dreamer / learned world models | 4 3 1 1 1 3 2 1 3 1 1 1 2 | 50.0 | Simulator error can become the optimized signal; no calibrated world model |
| Robust MPC | 4 3 3 2 2 4 3 3 4 3 4 2 4 | 81.25 | Sensible future execution comparator once transition uncertainty is identified |
| FQL / BFQ / SafeFQL | 2 1 1 1 1 3 4 2 2 3 1 1 2 | 47.5 | Newer credible research; limited replication and no demonstrated crypto execution transfer |

The scorecard favors simpler controls. The mission's mandatory RL comparison
justifies studying PPO, Double DQN and CQL despite lower suitability. All share
the same 16-unit tanh architecture, action set and costs. CQL's inventory-penalty
ablation is the fourth registered configuration, not an additional family.
An explicit inventory variance penalty, external bounds and independent ES95
reporting cover the initial risk-sensitive comparison; this is **not** a claim
to implement CVaR optimization or a distributional critic.

Discounted/sliding-window UCB and Thompson sampling need causal reward feedback
and a fixed change/adaptation rule. They are researched, but adding all of them
to this small screen would create extra trials without solving inventory state.
The implemented contextual baseline is fixed linear immediate-reward regression
on uniform simulated logs, not LinUCB or an adaptively exploring production bot.

No score, training reward or development average can satisfy the gates without
fresh data, matched champion evaluation, reliable OPE, cost/latency fidelity,
seed stability, complete statistical evidence, and external risk safety.

## Broader paper importance versus implementation suitability

The earlier full financial review's paper matrix remains the detailed source
assessment. For this update, use 20 equally weighted dimensions scored 0–4,
with unknown evidence scored 0 rather than an optimistic imputation. Their order
is influence, publication quality, methodological quality, independent replication,
leakage controls, snooping controls, cost realism, execution realism, impact,
crypto relevance, repository relevance, reproducibility, data access, code access,
license clarity, operational feasibility, CPU feasibility, interpretability,
non-stationarity robustness, and safety compatibility. Sum ×1.25 gives /100.
These judgments select research priority; they are not estimates of alpha.

| Mechanism / representative evidence | 20-dimension vector | Score / 100 | Binding evidence gap |
|---|---|---:|---|
| HAR / Corsi | 4 4 4 3 3 2 1 1 0 2 4 4 3 2 2 4 4 4 2 4 | 71.25 | Fresh causal realized-volatility inputs and net risk-gate confirmation |
| Calibrated shallow models / Gu–Kelly–Xiu, adaptive conformal | 4 4 4 3 3 3 2 1 0 2 4 4 2 3 2 4 4 3 2 4 | 72.5 | Availability witnesses and production-relevant interval calibration |
| OFI / Cont–Kukanov–Stoikov | 4 4 4 3 3 2 2 3 2 2 3 3 1 1 1 3 4 4 2 3 | 67.5 | Licensed event-level history, execution and latency |
| Sequential policies / PPO, Double DQN, CQL | 4 4 4 3 2 2 0 0 0 1 3 4 4 4 2 3 4 2 1 3 | 62.5 | Joint support, calibrated simulator, matched champion and fresh evidence |

Method-only RL papers score zero on financial costs/impact: benchmark replication is not replication of trading efficacy. No score overrides a missing promotion gate. Source-specific limitations, unavailable identifiers and dispositions remain in the paper matrix.

Final disposition after the registered screen: PPO, Double DQN and CQL configurations are rejected. Selection scores above record pre-experiment research priority, not post-hoc promotion scores; see [decision memo](final-decision-memo.md).

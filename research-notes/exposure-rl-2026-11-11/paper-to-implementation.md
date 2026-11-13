# Paper-to-implementation analysis (2026-11-11)

Eighteen shortlisted works, sixteen of them about RL, offline policy learning,
safe RL, policy evaluation or reliable RL experimentation. Tags refer to
[paper-matrix.csv](paper-matrix.csv); versions, code URLs, licences and
verification methods are in [search-log.md](search-log.md).

Conventions used in every entry:

- **Q1 credibility** — is the paper methodologically credible?
- **Q2 fit** — is the method appropriate for this data and decision problem?
- **Q3 improvement** — does its implementation improve this system? Unless the
  repository already holds evidence, the answer is "to be determined by this
  campaign's registered experiments". Q3 is never inferred from Q1 or Q2.
- "This design" means: one perpetual per account; action a ∈ [−1, 1] mapped to
  target exposure in [−0.1666, +0.1666] of equity; decision at each completed
  6h bar; fill at next open; fees, half-spread, slippage and historical
  funding; reward = one-step change in net marked-to-market equity divided by
  equity; deterministic projecting shield inside the environment; numpy-only
  MLPs with manual backprop on CPU.
- Experimental details of each paper are from its abstract and the reviewer's
  prior reading; they were not re-read in full on 2026-11-11 and no paper code
  was read or copied. "Minimal faithful implementation" lists are written from
  the published algorithm descriptions.

---

## 1. PPO — Schulman, Wolski, Dhariwal, Radford, Klimov (2017) [B01]

- **Contribution.** A first-order policy-gradient method that maximises a
  clipped surrogate objective, allowing several epochs of minibatch updates per
  batch of on-policy data.
- **Assumptions.** A simulator that can be sampled on-policy at will; a
  stationary MDP; advantage estimates of usable quality.
- **Target / action space.** Episodic return; continuous (diagonal Gaussian)
  and discrete actions.
- **Data.** Simulated continuous-control and Atari benchmarks. No financial
  data.
- **Validation.** Learning curves against other policy-gradient methods on
  fixed benchmark suites.
- **Costs.** Not applicable.
- **Statistical evidence.** Few seeds per task, typical of its period; no
  interval estimates of the kind [N02] recommends. Exact seed count not
  re-verified.
- **Independent replication.** Extensive, but with the caveat documented by
  [N03] and [N04]: reproducing reported performance depends on implementation
  details not in the algorithm statement.
- **Official code / licence.** openai/baselines, MIT (GitHub API). Not used.
- **Repository fit.** Good. A small Gaussian-policy MLP and a value MLP are
  feasible in numpy.
- **Minimal faithful implementation must contain.**
  1. Clipped surrogate L = E[min(r·A, clip(r, 1−ε, 1+ε)·A)] with r the
     new/old probability ratio computed from stored old log-probabilities.
  2. GAE(λ) advantages [B02], computed backward through each rollout.
  3. Per-minibatch (or per-batch, declared) advantage normalisation.
  4. A value loss on returns (GAE advantage plus old value), with any value
     clipping declared as on or off.
  5. An entropy bonus with a declared coefficient.
  6. Several epochs over shuffled minibatches of each rollout.
  7. Correct truncation handling [A10]: at a time-limit cut, bootstrap from
     V(s_T); set the bootstrap to zero only on true termination. Since this
     task is continuing, almost every episode end is a truncation.
  8. Log-probabilities evaluated on the *proposed* (pre-shield) action, because
     that is what the policy sampled.
- **Deviations in our numpy implementation (to be declared in the model
  card).** Manual backprop instead of autodiff; a hand-written Adam; small
  networks; state-independent log-standard-deviation parameter; no vectorised
  parallel environments beyond what numpy batching gives; observation
  normalisation fitted on the training prefix only (the earlier screen's
  CE-RL-025 found a scaler fitted on later bars); no learning-rate annealing
  unless registered; gradient clipping declared as on or off.
- **Proof applicability.** None for the learner. The earlier campaign's scoped
  SMT work on GAE targets and the PPO objective shows that algebraic
  identities of the update can be checked; that is implementation assurance,
  not performance evidence.
- **Disposition.** implemented-in-campaign.
- **Q1.** Credible as an algorithm; its comparative claims are weaker than
  they appear, per [N01, N03].
- **Q2.** Appropriate as the on-policy representative, with the caveat that
  the "simulator" replays one market path, so on-policy sampling does not
  generate new markets.
- **Q3.** Repository evidence exists only for a different configuration: the
  categorical three-action PPO of the 2026-09-17 screen was rejected (for
  example 29/30, 30/30 and 25/30 failed paths at cadence 1 across seeds 11, 23
  and 47). That screen's shield did not project, so the result does not
  transfer. For this design: to be determined by this campaign's registered
  experiments.

## 2. GAE — Schulman, Moritz, Levine, Jordan, Abbeel (2016) [B02]

- **Contribution.** An exponentially weighted family of advantage estimators
  parameterised by λ that trades bias against variance.
- **Assumptions.** A learned value function; discounted formulation, with γ
  treated as a variance-reduction parameter as well as a horizon.
- **Target / action space.** Advantage estimates; action-space agnostic.
- **Data / validation / costs.** Simulated locomotion; learning curves; not
  applicable.
- **Statistical evidence.** Benchmark learning curves; not re-verified in
  detail.
- **Independent replication.** Ubiquitous as a PPO component.
- **Official code / licence.** None identified for the paper itself; unverified.
- **Repository fit.** Good; a backward recursion over a rollout.
- **Minimal faithful implementation.** δ_t = r_t + γ·V(s_{t+1})·(1 − term_t) −
  V(s_t); A_t = δ_t + γλ·(1 − done_t)·A_{t+1}, where `term` marks true
  termination only and `done` marks any episode boundary. At a truncation the
  recursion stops (done = 1) but δ still bootstraps from V of the true next
  state (term = 0).
- **Deviations.** None intended. The pair (γ, λ) is fixed by registration.
- **Proof applicability.** The recursion's telescoping identity is checkable
  by SMT over reals; the repository has prior work of this kind
  (`gae-targets-v2`).
- **Disposition.** implemented-in-campaign (as part of PPO).
- **Q1.** Credible. **Q2.** Appropriate. **Q3.** To be determined by this
  campaign's registered experiments.

## 3. Time limits — Pardo, Tavakoli, Levdik, Kormushev (2018) [A10]

- **Contribution.** Distinguishes time limits that are part of the task (the
  agent should observe remaining time) from time limits that only cut
  training of a continuing task (the agent should bootstrap at the cut), and
  shows that conflating truncation with termination harms learning.
- **Assumptions.** Value-based or actor-critic learning with bootstrapped
  targets.
- **Target / action space.** Agnostic.
- **Data / validation.** Simulated control tasks; learning-curve comparisons.
- **Costs.** Not applicable.
- **Statistical evidence.** Benchmark comparisons; details not re-verified.
- **Independent replication.** The distinction is now built into standard
  environment APIs (separate "terminated" and "truncated" flags).
- **Official code / licence.** Unverified.
- **Repository fit.** Direct. Training windows of a continuing exposure task
  are truncations.
- **What the implementation must do.** Carry two flags per transition. PPO:
  see entries 1–2. TD3 and IQL: target = r + γ·(1 − terminated)·(next value),
  never (1 − done). If the registered environment forces liquidation at the
  end of an evaluation window, that liquidation is part of the task and the
  remaining-time feature must then be observed; if it does not, the end is a
  truncation. The registration must say which.
- **Deviations.** None intended.
- **Proof applicability.** A unit-level invariant ("no transition with
  truncated = 1 and terminated = 0 has a zero bootstrap") is testable and
  model-checkable; the earlier audit table already forbids bootstrapping
  through a true terminal.
- **Disposition.** informs-design.
- **Q1.** Credible. **Q2.** Appropriate. **Q3.** To be determined by this
  campaign's registered experiments.

## 4. TD3 — Fujimoto, van Hoof, Meger (2018) [B04]

- **Contribution.** Identifies overestimation in deterministic actor-critic
  learning and counters it with clipped double-Q learning, target-policy
  smoothing and delayed policy updates.
- **Assumptions.** Continuous actions; a replay buffer whose distribution is
  refreshed by the current policy plus exploration noise; stationary dynamics.
- **Target / action space.** Discounted return; bounded continuous actions.
- **Data.** Simulated continuous control. No financial data.
- **Validation.** Learning curves against DDPG and other baselines.
- **Costs.** Not applicable.
- **Statistical evidence.** Multiple seeds per task; no modern interval
  reporting. Exact count not re-verified.
- **Independent replication.** Extensive; a standard baseline.
- **Official code / licence.** sfujim/TD3, MIT (GitHub API). Not used.
- **Repository fit.** Feasible: three small MLPs plus targets. The interaction
  with the shield needs care.
- **Minimal faithful implementation must contain.**
  1. Two critics; target y = r + γ·(1 − terminated)·min(Q1′, Q2′)(s′, ã).
  2. Target-policy smoothing: ã = clip(π′(s′) + clip(ε, −c, c), −1, 1), with
     ε Gaussian.
  3. Delayed actor and target updates (one per d critic updates).
  4. Polyak-averaged target networks for actor and both critics.
  5. Actor loss −Q1(s, π(s)), using the first critic only.
  6. Exploration noise added to the behaviour action and a uniform-random
     warm-up phase, both declared.
- **Deviations in our numpy implementation.** Manual backprop, including the
  chain rule through the critic into the actor; small networks; no
  prioritised replay. A registered decision is needed on shield handling:
  either (a) the critic is trained on proposed actions, in which case Q is
  flat wherever the shield clips and the actor gradient vanishes there, or
  (b) the critic is trained on executed actions and the actor loss evaluates
  Q at the projected action, in which case the projection must be included in
  the backward pass (gradient zero where clipped) — the same saturation, now
  explicit. [E08] discusses these variants. Either way, log the fraction of
  clipped proposals.
- **Proof applicability.** None for the learner.
- **Disposition.** implemented-in-campaign.
- **Q1.** Credible. **Q2.** Appropriate as the off-policy representative for
  a bounded scalar action; the replay buffer covers one market path only.
- **Q3.** No repository evidence (TD3 was never implemented in the earlier
  screen). To be determined by this campaign's registered experiments.

## 5. IQL — Kostrikov, Nair, Levine (2022) [B11]

- **Contribution.** Offline RL that never evaluates the critic on actions
  outside the dataset: a state-value function is fitted by expectile
  regression toward Q on dataset actions, Q is backed up from V, and the
  policy is extracted by advantage-weighted regression.
- **Assumptions.** A fixed dataset with useful coverage; the upper expectile
  of Q over dataset actions approximates the in-support maximum; stationary
  dynamics.
- **Target / action space.** Discounted return; continuous actions.
- **Data.** D4RL offline benchmarks. No financial data.
- **Validation.** Normalised benchmark scores against offline baselines.
- **Costs.** Not applicable.
- **Statistical evidence.** Benchmark tables over seeds; hyperparameters (τ,
  β) chosen per task family, which is a form of online tuning that offline
  practice cannot rely on.
- **Independent replication.** Widely re-implemented; [B16] re-examines its
  policy-extraction step and finds it leaves value information unused on their
  benchmarks.
- **Official code / licence.** ikostrikov/implicit_q_learning, MIT (GitHub
  API). Not used.
- **Repository fit.** Good computationally (four small MLPs, supervised-style
  losses). Conceptually limited by the dataset: simulator-generated, with a
  known-propensity behaviour mixture.
- **Minimal faithful implementation must contain.**
  1. Expectile value loss: L_V = E[|τ − 1(u < 0)|·u²], u = Q_target(s, a) −
     V(s), with τ > 0.5 and Q_target the minimum of two target critics.
  2. Q loss: (r + γ·(1 − terminated)·V(s′) − Q(s, a))² for both critics. The
     backup uses V(s′), never a policy action.
  3. Policy extraction: maximise E[w·log π(a | s)] with w = min(exp(β·(Q_target
     − V)), w_max). The clip on the exponentiated weight is required for
     numerical stability.
  4. Polyak-averaged target critics.
  5. Policy trained on dataset actions only.
- **Deviations in our numpy implementation.** Manual backprop; Gaussian
  policy with a declared variance treatment; small networks; no cosine
  learning-rate schedule unless registered; (τ, β, w_max) fixed in advance
  with no per-fold tuning. Dataset actions are the *executed* (post-shield)
  actions so that support is defined in the space the environment received.
  The dataset is labelled simulated in every artefact.
- **Proof applicability.** None for the learner.
- **Disposition.** implemented-in-campaign.
- **Q1.** Credible; benchmark hyperparameter selection is a known weakness.
- **Q2.** Appropriate as an offline representative, with a narrow claim: it
  tests policy extraction from simulated logs, not learning from live logs.
- **Q3.** No repository evidence (IQL was never implemented). To be
  determined by this campaign's registered experiments.

## 6. Offline RL versus behaviour cloning — Kumar, Hong, Singh, Levine (2022) [B15]

- **Contribution.** Theory and experiments on when offline RL should
  outperform behaviour cloning: chiefly with suboptimal or noisy data and long
  horizons with a few critical states; cloning is competitive with
  near-expert data.
- **Assumptions.** Coverage conditions on the dataset; sparse-reward analysis
  setting.
- **Target / action space.** Agnostic.
- **Data / validation.** Simulated benchmarks. No financial data.
- **Costs.** Not applicable.
- **Statistical evidence.** Benchmark comparisons; not re-verified in detail.
- **Independent replication.** Consistent with other reports that simple
  cloning variants are strong offline baselines; no formal replication
  identified.
- **Official code / licence.** Unverified.
- **Repository fit.** Direct: it motivates the registered behaviour-cloning
  baseline.
- **Minimal faithful behaviour cloning.** Supervised regression of executed
  action on observation over the same dataset IQL uses; identical network
  size, optimiser, normalisation and stopping rule; no reward information.
  With a mixed behaviour policy, cloning imitates the mixture's conditional
  mean (for a squared loss), which may be near zero exposure. That weakness
  must be stated: the earlier screen's BC "imitates the modal uniform action
  and is deliberately weak".
- **Deviations.** Optionally a filtered variant (clone the top fraction of
  trajectories by return); only if registered, since it is a further trial.
- **Proof applicability.** None.
- **Disposition.** baseline-in-campaign.
- **Q1.** Credible. **Q2.** Appropriate as a baseline and as an interpretive
  frame. **Q3.** To be determined by this campaign's registered experiments.

## 7. Bottlenecks in offline RL — Park, Frans, Levine, Kumar (2024) [B16]

- **Contribution.** Separates value learning, policy extraction and test-time
  generalisation as bottlenecks; reports that extraction method and
  generalisation to out-of-distribution test states often matter more than
  the value objective, and that advantage-weighted regression underuses the
  value function relative to behaviour-regularised policy-gradient extraction.
- **Assumptions.** Conclusions are empirical and specific to their benchmark
  suite and data scales.
- **Data / validation.** Simulated offline benchmarks with controlled data
  size; no financial data.
- **Costs.** Not applicable.
- **Statistical evidence.** Multi-seed benchmark matrices; not re-verified in
  detail.
- **Independent replication.** Later flow-policy papers adopt its extraction
  recommendation [C13]; no formal replication identified.
- **Official code / licence.** Unverified.
- **Repository fit.** Informative, not implemented. It predicts a specific
  failure mode for our IQL (weak extraction) and suggests a specific ablation
  (TD3+BC-style extraction from the same IQL critic).
- **Deviations if implemented.** The ablation would be a new trial and needs
  registration; it is not in the registered design.
- **Proof applicability.** None.
- **Disposition.** informs-design.
- **Q1.** Credible as a controlled empirical study. **Q2.** Relevant to
  interpreting IQL; its generalisation finding matters here because the test
  period is out of distribution by construction. **Q3.** To be determined by
  this campaign's registered experiments.

## 8. Fitted Q evaluation — Le, Voloshin, Yue (2019) [D04]

- **Contribution.** Batch policy learning under constraints; introduces
  fitted Q evaluation as the policy-evaluation subroutine and gives
  finite-sample guarantees under concentrability and realisability
  assumptions.
- **Assumptions.** Data coverage (bounded concentrability); a function class
  closed under Bellman backups; stationary dynamics.
- **Target / action space.** Discounted value of a fixed target policy;
  stated for discrete actions, commonly applied to continuous ones.
- **Data / validation.** Simulated control. No financial data.
- **Costs.** Not applicable.
- **Statistical evidence.** Theory plus benchmark experiments.
- **Independent replication.** FQE is a consistently strong but not
  dominant estimator in [D08] and [D09]; [D10] shows it is sensitive to its
  own hyperparameters.
- **Official code / licence.** Unverified for this paper.
- **Repository fit.** Good computationally (iterated regression with one
  MLP).
- **Minimal faithful implementation must contain.** Iterate: targets y = r +
  γ·(1 − terminated)·Q_k(s′, π(s′)) with π the *target* policy composed with
  the shield, then regress Q_{k+1} on (s, a) → y over the logged data. Report
  E[Q_K(s_0, π(s_0))] over the registered initial-state distribution. Fix the
  number of iterations, network size and regression steps in advance.
- **Deviations.** Targets from a frozen copy refreshed per iteration;
  continuous actions; small MLP. Validate against direct simulator rollouts
  of the same policy from the same initial states, and report error and rank
  correlation across policies.
- **Proof applicability.** Algebraic identities of the estimator can be
  unit-checked; no guarantee transfers, since concentrability is unverifiable
  on this data.
- **Disposition.** method-used-for-evaluation.
- **Q1.** Credible. **Q2.** Appropriate as a diagnostic on simulated logs;
  not appropriate as evidence about live counterfactuals. **Q3.** No
  repository evidence (FQE was reviewed but not implemented). To be determined
  by this campaign's registered experiments.

## 9. Doubly robust OPE — Jiang, Li (2016) [D02], with importance sampling [D01]

- **Contribution.** Extends the doubly robust estimator from contextual
  bandits to sequential decisions: an importance-weighted correction around
  an approximate value model, unbiased when propensities are correct and with
  reduced variance when the model is good; includes a hardness result on the
  variance of unbiased OPE.
- **Assumptions.** Known (or well-estimated) behaviour propensities; overlap
  (the behaviour policy assigns positive density to every target action);
  finite horizon.
- **Target / action space.** Finite-horizon value; stated for discrete
  actions.
- **Data / validation.** Simulated domains; comparison of bias and variance
  across estimators. No financial data.
- **Costs.** Not applicable.
- **Statistical evidence.** Theory plus controlled simulations.
- **Independent replication.** Standard estimator in the benchmarks [D08,
  D09]; [D03] gives weighted and blended variants.
- **Official code / licence.** Unverified.
- **Repository fit.** Partial. The repository already has trajectory IS,
  per-decision IS, weighted IS and sequential DR helpers from the earlier
  screen (known propensities 1/3 over three actions).
- **Minimal faithful implementation must contain.** Per-decision importance
  ratios ρ_t = π(a_t | s_t)/μ(a_t | s_t) from *logged* propensities;
  cumulative products; ordinary and self-normalised (weighted) estimators; the
  DR recursion V_DR = V̂(s) + ρ·(r + γ·V_DR′ − Q̂(s, a)); effective sample
  size; no silent weight clipping.
- **Deviations and a required design decision.** With continuous actions a
  deterministic target policy has zero density, so ratios are undefined. The
  target must be evaluated in stochastic form (its Gaussian density), or
  actions must be discretised for OPE only. Shield projection adds point
  masses at the interval ends in executed-action space, so densities must be
  computed in proposal space with both policies' proposals logged. Horizon
  must be short (the earlier screen used six-decision episodes) or variance
  makes the estimate useless.
- **Proof applicability.** The repository has exact-rational ESS and OPE
  algebra audits; those apply to the estimator arithmetic only.
- **Disposition.** method-used-for-evaluation.
- **Q1.** Credible. **Q2.** Appropriate only on simulator logs with known
  propensities and short horizons. **Q3.** Repository evidence is negative
  for a different setup: all 108 OPE batches in the earlier screen were
  invalid. For this design: to be determined by this campaign's registered
  experiments.

## 10. DOPE benchmark — Fu et al. (2021) [D08]

- **Contribution.** A benchmark for deep OPE with standardised tasks,
  policies and metrics (absolute error, rank correlation, regret of the
  selected policy).
- **Assumptions.** Simulated ground truth is available for every policy.
- **Data.** Simulated control datasets. No financial data.
- **Validation.** Estimators compared against ground-truth returns.
- **Costs.** Not applicable.
- **Statistical evidence.** Many policy-task pairs; the headline result is
  negative: no estimator is uniformly reliable.
- **Independent replication.** Consistent with [D09].
- **Official code / licence.** google-research/deep_ope, Apache-2.0, archived
  (GitHub API). Not used.
- **Repository fit.** Its protocol is what this campaign copies in miniature:
  evaluate estimators against simulator ground truth and report error, rank
  correlation and regret.
- **Deviations.** One environment family, few policies; results are
  diagnostics for this simulator only.
- **Proof applicability.** None.
- **Disposition.** method-used-for-evaluation (protocol).
- **Q1.** Credible. **Q2.** Appropriate as a validation protocol. **Q3.** To
  be determined by this campaign's registered experiments.

## 11. Shielding — Alshiekh, Bloem, Ehlers, Könighofer, Niekum, Topcu (2018) [E01]

- **Contribution.** Synthesises a reactive "shield" from a temporal-logic
  safety specification and an abstraction of the environment. The shield
  either restricts the action set before the agent chooses or corrects an
  unsafe choice afterwards, and guarantees the specification throughout
  learning and execution.
- **Assumptions.** A safety property expressible as a finite automaton; a
  sound finite abstraction of the safety-relevant environment dynamics.
- **Target / action space.** Discrete in the paper.
- **Data / validation.** Simulated grid and game tasks.
- **Costs.** Not applicable.
- **Statistical evidence.** Illustrative learning curves; the main claim is
  formal.
- **Independent replication.** A large follow-on literature ([E02], [E08],
  [E12]).
- **Official code / licence.** Unverified.
- **Repository fit.** Conceptual. Our shield is not synthesised from an
  automaton; it is a hand-written deterministic projection whose properties
  are proved directly.
- **Minimal faithful shield for this design.** A total function
  shield(state, proposal) → target such that: non-finite proposals map to a
  declared safe value; |target| ≤ cap; |target − current| ≤ turnover limit;
  in reduce-only mode (drawdown limit reached or state stale) |target| ≤
  |current| with no sign flip; rounding is toward zero exposure at every
  limit; and the function is idempotent. Applied identically in training,
  evaluation and the Haskell path.
- **Deviations from the paper.** No environment abstraction, hence no
  guarantee about reachable states — only about admitted actions. Continuous
  action; post-hoc correction only.
- **Proof applicability.** High for admission invariants (Z3 over
  reals/rationals; Haskell↔Python conformance). Not applicable to realised
  exposure drift, fills or drawdown, which depend on unmodelled gaps.
- **Disposition.** implemented-in-campaign (shield), informs-design (claims).
- **Q1.** Credible. **Q2.** The post-hoc correction pattern is appropriate;
  the paper's reachability guarantee is not available here.
- **Q3.** Repository evidence concerns a different mechanism: the earlier
  shield treated a breach as failure, and 686 of 1,080 base-cost paths
  failed. Whether projection yields usable policies: to be determined by this
  campaign's registered experiments.

## 12. Safety layer — Dalal, Dvijotham, Vecerik, Hester, Paduraru, Tassa (2018) [E05]

- **Contribution.** Adds a layer after the policy that solves a small
  projection problem per state, correcting the action minimally so that
  linearised constraint models are satisfied; with one active constraint the
  solution is closed-form.
- **Assumptions.** Constraint functions approximately linear in the action;
  constraint models learned from prior data; at most one constraint active
  for the closed form.
- **Target / action space.** Continuous.
- **Data / validation.** Simulated physical-control tasks.
- **Costs.** Not applicable.
- **Statistical evidence.** Benchmark comparisons; details not re-verified.
- **Independent replication.** Widely used as a baseline in safe-RL
  benchmarks; analysed in [E08].
- **Official code / licence.** None identified; unverified.
- **Repository fit.** Strong in spirit and simpler in practice: our
  constraints are exact interval bounds on a scalar, so projection is a clip
  onto the intersection of intervals and nothing is learned.
- **Deviations.** No learned constraint model; exact constraints; the layer
  sits in the environment, not in the policy network.
- **Proof applicability.** High: the interval intersection is non-empty
  whenever the current position is admissible, and the projection is the
  nearest admissible point. Both are SMT-checkable. The non-emptiness
  condition must be proved for the reduce-only case with a position already
  outside the cap after drift.
- **Disposition.** informs-design.
- **Q1.** Credible as a preprint; not located in a peer-reviewed venue.
  **Q2.** Appropriate. **Q3.** To be determined by this campaign's
  registered experiments.

## 13. Clipped-action policy gradient — Fujita, Maeda (2018) [E07]

- **Contribution.** For policies whose sampled actions are clipped by the
  environment, derives a policy-gradient estimator that uses the clipping
  bounds; it is unbiased and has lower variance than the standard estimator.
- **Assumptions.** Known, state-independent clip bounds in the paper; a
  policy with unbounded support.
- **Target / action space.** Continuous bounded actions.
- **Data / validation.** Simulated continuous control with PPO and TRPO.
- **Costs.** Not applicable.
- **Statistical evidence.** Benchmark learning curves.
- **Independent replication.** None identified.
- **Official code / licence.** pfnet-research/capg, MIT (GitHub API). Not
  used.
- **Repository fit.** Explains what our PPO does when the shield clips: the
  standard estimator on proposed actions remains unbiased for the composite
  environment. The variance-reduced estimator would need state-dependent
  bounds, because turnover and reduce-only limits move with the position.
- **Deviations if implemented.** Replace log π(a) for clipped samples by the
  log of the tail probability mass beyond the active bound, with bounds
  supplied per state by the shield. Not in the registered design; a candidate
  ablation only.
- **Proof applicability.** None.
- **Disposition.** informs-design.
- **Q1.** Credible. **Q2.** Relevant; the extension to state-dependent bounds
  is our inference, not the paper's result. **Q3.** To be determined by this
  campaign's registered experiments.

## 14. Statistical precipice — Agarwal, Schwarzer, Castro, Courville, Bellemare (2021) [N02]

- **Contribution.** Shows that deep-RL conclusions drawn from point estimates
  over few runs are unreliable; recommends stratified-bootstrap interval
  estimates, the interquartile mean, performance profiles and probability of
  improvement.
- **Assumptions.** Runs are exchangeable draws over seeds on fixed tasks.
- **Data / validation.** Re-analysis of published benchmark results.
- **Costs.** Not applicable.
- **Statistical evidence.** The paper is itself the statistical evidence;
  NeurIPS 2021.
- **Independent replication.** Its recommendations are widely adopted;
  consistent with [N01].
- **Official code / licence.** google-research/rliable, Apache-2.0, archived
  (GitHub API). Not used.
- **Repository fit.** Direct for reporting, with one adjustment: here seeds
  vary the learner, not the market. Seed intervals quantify optimisation
  variance and must be reported next to, not instead of, the paired block
  bootstrap over time.
- **Deviations.** With five seeds, intervals are wide and interquartile means
  are barely defined; report all seeds individually, plus min, median and
  max. No best-seed selection.
- **Proof applicability.** None.
- **Disposition.** method-used-for-evaluation.
- **Q1.** Credible. **Q2.** Appropriate. **Q3.** Not applicable as a
  performance question; it governs how Q3 is answered for the learners.

## 15. Implementation matters — Engstrom et al. (2020) [N03]

- **Contribution.** Ablates code-level optimisations in PPO and TRPO
  implementations and finds that they account for much of PPO's measured
  gain and alter its trust-region behaviour.
- **Assumptions.** Findings are specific to the benchmark tasks studied.
- **Data / validation.** Simulated locomotion; controlled ablations.
- **Costs.** Not applicable.
- **Statistical evidence.** Multi-seed ablations; ICLR 2020.
- **Independent replication.** Consistent with [N04] and with [N01].
- **Official code / licence.** MadryLab/implementation-matters, MIT (GitHub
  API). Not used.
- **Repository fit.** Direct. A hand-written numpy PPO differs from every
  reference implementation, so results attach to our implementation.
- **What we must do.** Publish an implementation-detail checklist with each
  policy card: reward scaling, observation normalisation and where it is
  fitted, value-loss clipping, advantage normalisation scope, initialisation,
  optimiser constants, gradient clipping, learning-rate schedule, entropy
  coefficient, minibatch layout, truncation handling.
- **Proof applicability.** None.
- **Disposition.** negative-evidence.
- **Q1.** Credible. **Q2.** Appropriate. **Q3.** Not applicable as a
  performance question.

## 16. Backtest overfitting in crypto DRL — Gort, Liu, Sun, Gao, Chen, Wang (2022) [N05]

- **Contribution.** Frames deep-RL crypto backtests as prone to false
  positives and proposes rejecting agents by an estimated probability of
  overfitting before out-of-sample use.
- **Assumptions.** The overfitting estimate from combinatorial validation
  splits is informative about future performance.
- **Target / action space.** Multi-asset crypto spot trading; continuous
  portfolio actions.
- **Data.** Crypto market data at intraday frequency; exact period and
  symbols not re-verified.
- **Validation.** Hyperparameter trials with combinatorial splits, then one
  out-of-sample test period.
- **Costs.** Transaction costs included per the paper; level not re-verified.
- **Statistical evidence.** A hypothesis test on overfitting probability; a
  single test period.
- **Independent replication.** None identified. The authors are associated
  with the FinRL project, so this is not an independent critique of that
  line of work.
- **Official code / licence.** Unverified.
- **Repository fit.** Supports the registered DSR/PBO gates. Its own positive
  claim (less-overfit agents did better in one test period) is single-path
  evidence and is not relied on.
- **Deviations.** We use the repository's existing DSR/PBO implementation and
  count every trained policy as a trial.
- **Proof applicability.** None.
- **Disposition.** negative-evidence.
- **Q1.** Partly credible: the diagnosis is sound, the remedy's efficacy is
  weakly evidenced, and it is an unreviewed preprint. **Q2.** The diagnosis
  applies directly. **Q3.** To be determined by this campaign's registered
  experiments.

## 17. Dynamic trading with costs — Gârleanu, Pedersen (2013) [A03]

- **Contribution.** A closed-form optimal dynamic portfolio policy with
  quadratic transaction costs and mean-reverting return predictors: trade
  partially toward an aim portfolio that weights signals by persistence.
- **Assumptions.** Linear return predictability with known parameters;
  quadratic costs proportional to risk; mean-variance preferences; no hard
  constraints.
- **Target / action space.** Continuous positions in several assets.
- **Data.** Commodity futures in the empirical illustration; no crypto.
- **Validation.** Backtest of the optimal policy against a static
  mean-variance policy.
- **Costs.** Quadratic price-impact costs, central to the result.
- **Statistical evidence.** Sharpe comparisons net of the modelled costs;
  Journal of Finance (peer-reviewed).
- **Independent replication.** Widely used theoretical benchmark; a specific
  crypto replication was not identified.
- **Official code / licence.** None.
- **Repository fit.** It is the conceptual model for the registered ridge
  cost-aware forecast baseline, with deviations.
- **Deviations.** Proportional, not quadratic, costs; hard cap, turnover
  limit and reduce-only mode; one asset; forecast from a ridge regression
  with estimation error. With proportional costs the closed form does not
  hold; the baseline is a heuristic partial-adjustment or threshold rule and
  must be described as "inspired by", not "implements".
- **Proof applicability.** None for optimality under our cost model; shield
  invariants apply to its output as to any policy.
- **Disposition.** baseline-in-campaign.
- **Q1.** Credible. **Q2.** Appropriate as the non-RL cost-aware benchmark
  concept. **Q3.** To be determined by this campaign's registered
  experiments.

## 18. Deflated Sharpe ratio — Bailey, López de Prado (2014) [F04]

- **Contribution.** A Sharpe-ratio test statistic corrected for the number of
  trials, the variance across trial Sharpe ratios, sample length, skewness
  and kurtosis.
- **Assumptions.** An estimate of the effective number of independent trials;
  returns stationary enough for moment estimates.
- **Target / action space.** Not applicable (evaluation statistic).
- **Data / validation.** Analytical, with illustrations.
- **Costs.** Not applicable; applied to net returns.
- **Statistical evidence.** Derivation; Journal of Portfolio Management.
- **Independent replication.** Widely applied; [N05] uses the related
  probability-of-overfitting idea [F05] for RL agents.
- **Official code / licence.** None identified.
- **Repository fit.** Already a canonical gate in this repository (DSR ≥ 0.95
  and PBO ≤ 0.20 were not relaxed in the earlier screen).
- **Deviations.** The effective number of trials must include every
  algorithm × seed × configuration trained in this campaign, plus disclosed
  earlier attempts on the same development panel; correlated trials make the
  effective count a judgement that must be registered.
- **Proof applicability.** None.
- **Disposition.** method-used-for-evaluation.
- **Q1.** Credible. **Q2.** Appropriate, with low power on a short
  confirmation window. **Q3.** Not applicable as a performance question.

---

## Cross-cutting implementation notes

- **Proposed versus executed action.** Log both for every transition. PPO
  uses proposed actions for log-probabilities. IQL and BC use executed
  actions as the dataset action. TD3's choice is registered in advance (entry
  4). OPE densities are computed in proposal space (entry 9).
- **Truncation.** Two flags everywhere (entry 3).
- **Normalisation.** Fitted on the training prefix only; frozen at
  evaluation.
- **No copied code.** All learners are written from the algorithm
  descriptions. Licences are recorded only to document that nothing was
  reused.

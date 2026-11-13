# Exposure-control RL: literature review (2026-11-11)

Bracketed tags such as [B01] refer to rows of [paper-matrix.csv](paper-matrix.csv).
Deep analyses are in [paper-to-implementation.md](paper-to-implementation.md);
queries, versions, code licences and verification methods are in
[search-log.md](search-log.md).

## 1. Scope, search protocol and limits

**Scope.** This review supports one registered decision problem: bounded
continuous target exposure for a single Binance USDT-M perpetual per account,
decided at each completed 6h bar, filled at the next open, with fees,
half-spread, slippage and historical funding, behind a deterministic projecting
shield. It is RL-first. Forecasting literature is covered by the earlier
[market-prediction review](../market-prediction-2026-09-04/literature-review.md)
and is not repeated; the earlier
[sequential-control review](../sequential-control-2026-09-17/literature-review.md)
remains the record for that rejected screen.

**Protocol.** Searches ran on 2026-11-11. Bibliographic metadata was checked
against the arXiv API, the Crossref API, publisher or proceedings pages, and
web search. Official code repositories and licences were checked against the
GitHub repository API. Separate queries targeted 2024–2026 work on offline RL,
RL-in-finance critiques, flow and diffusion policies, off-policy evaluation and
shielding. The matrix maps 75 works; 18 are analysed in depth. The matrix is
larger than the 30–50 planning range because the campaign's required coverage
list already names about sixty distinct works; nothing was added to reach a
count.

**Limits.** (i) This is not a systematic review; no inclusion protocol was
registered and no second reviewer screened results. (ii) Metadata was verified,
but most papers were not re-read in full on the search date. Descriptions of
experiments rely on abstracts and on the reviewer's prior reading; where a
detail was not re-checked the matrix says so, and no numerical result is quoted
from a paper unless it appears in the abstract or the fetched page. (iii) Web
search for failed replications of RL trading papers returned little. That is
weak evidence of absence: replications of trading papers are rarely published.
(iv) No paper PDF was downloaded and no third-party code was read for reuse.
(v) Nothing here is evidence that any method is profitable on this system.

Three questions are kept apart throughout: whether a paper is credible, whether
its method suits this data and decision, and whether its implementation
improves this system. The third is answered only by this repository's
experiments.

## 2. Why position management is sequential, and when a bandit suffices

A decision problem is sequential when the current action changes the
distribution of future states or the cost of future actions. In this design
the market path is treated as exogenous, so the action cannot change future
prices. It does change four things: the inventory carried into the next bar,
the cost of reaching any later target, the equity and drawdown state that the
shield reads, and the funding paid at later settlements. Those couplings are
what make the problem more than repeated one-step prediction.

The non-RL benchmark makes the same point analytically. Gârleanu and Pedersen
[A03] show that with quadratic trading costs and mean-reverting return
predictors the optimal policy trades partially toward an "aim" portfolio that
weights signals by their persistence. A myopic policy that jumps to the
one-period optimum overtrades fast-decaying signals. Their result needs
quadratic costs, linear-Gaussian predictors and known parameters. This
campaign's costs are proportional (fees, half-spread, slippage) and its
constraints are hard, so the closed form does not apply; the qualitative
conclusion — the right position depends on the current position and on signal
persistence — does. Almgren and Chriss [A04] and Nevmyvaka, Feng and Kearns
[A05] treat the neighbouring execution problem (how to trade a given quantity
within a horizon). At 6h bars with next-open fills and no order-book data,
execution scheduling cannot be modelled credibly here and stays out of scope.

Moody and Saffell [A01] introduced direct reinforcement for trading: a
recurrent policy whose input includes its previous position, trained on a
cost-aware, risk-adjusted objective. Deng et al. [A02] added learned feature
extraction. Both are single-path backtest studies; they establish the
formulation, not robust profitability.

**When a contextual bandit suffices.** If costs are negligible relative to the
signal, the turnover limit never binds, the drawdown state never triggers
reduce-only mode, and funding depends only on the current position, then the
myopic policy that maximises expected one-step net reward given the context is
optimal, and a contextual bandit [A07] or a cost-aware one-step regression is
the correct tool. It is also far easier to evaluate off-policy, because
importance weights do not multiply across time. The registered myopic
contextual-bandit and ridge cost-aware baselines therefore test the campaign's
central claim. If PPO, TD3 or IQL cannot beat them out of sample after costs,
the sequential formulation has not earned its complexity. That outcome is
plausible: at this bar length the per-step turnover limit may rarely bind, and
then little of the problem is sequential.

**Partial observability.** Returns are not Markov in any small feature vector,
so the problem is a POMDP. Recurrent policies are the standard response: R2D2
[C04] documents the practical problems of training recurrent value functions
from replay (stale recurrent state, burn-in), and Ni, Eysenbach and
Salakhutdinov [C05] show that carefully implemented recurrent model-free RL is
a strong POMDP baseline. Both rely on training budgets and tooling this
campaign lacks. Feeding fixed windows of lagged features to a feed-forward
network is a finite-memory approximation and should be described as one.

**Objective.** Position management is a continuing task with no natural
terminal state. Average-reward formulations [A06] match that structure;
discounted training with truncated episodes is an approximation whose
effective horizon is about 1/(1−γ) bars. Pardo et al. [A10] show that treating
a time-limit truncation as a true termination biases value targets, and that
the fix is to bootstrap from the value of the final state on truncation. That
is a correctness requirement for PPO and TD3 here, not a tuning choice. Note
also that summing per-bar simple returns ΔE/E is a risk-neutral criterion: it
does not penalise variance, and it differs from the log-growth of equity.

## 3. Algorithms and what the evidence supports in finance

### On-policy

PPO [B01] with generalised advantage estimation [B02] is the default on-policy
method because it is simple and tolerant of hyperparameters. Its evidence base
is simulated robotics and Atari, with a simulator that can be sampled without
limit. Engstrom et al. [N03] found that much of PPO's measured advantage over
TRPO comes from code-level details outside the stated algorithm (reward and
observation normalisation, value clipping, initialisation, learning-rate
annealing), and Huang et al. [N04] catalogue the details needed to reproduce
reference results. For a hand-written numpy PPO this means every such detail
must be declared, and "PPO" in a result table means "this implementation".

In this campaign the on-policy "simulator" is replay of a single historical
price path. PPO can sample fresh inventory trajectories and fresh start
points, but never fresh markets. Training return can therefore rise by
memorising the path. Walk-forward evaluation is the control; training curves
are not evidence.

### Off-policy

TD3 [B04] addresses critic overestimation in deterministic actor-critic
methods with twin critics, target-policy smoothing and delayed actor updates.
SAC [B03] reaches similar benchmark performance with a stochastic
maximum-entropy policy. Both reuse data, which matters when environment steps
are scarce. Here steps are cheap but markets are not, so the practical benefit
is smaller than in robotics, and TD3's replay buffer is drawn from the same
path that PPO sees. TD3's deterministic actor interacts badly with a
projecting shield (Section 5). SAC is not implemented: it adds a temperature
and a squashed-Gaussian log-density correction to a hand-written backprop
stack without a clear expected benefit for a one-dimensional action. That is a
resource decision, not a judgement on SAC.

Double and dueling DQN [B05, B06] and the distributional family (C51, QR-DQN,
IQN [B07–B09]) need discrete actions. The earlier three-action screen used
Double DQN and was rejected. Distributional critics are attractive in
principle for risk-aware control because they model the return distribution,
but the cited evidence is Atari score; nothing reviewed shows that learned
return distributions are calibrated on non-stationary financial data. They are
deferred.

### Offline

Offline RL learns from a fixed dataset and must avoid querying the critic on
actions the data does not support. BCQ [B13] constrains the policy to
generated in-support actions; CQL [B10] penalises Q-values on
out-of-distribution actions; TD3+BC [B12] adds a behaviour-cloning term to
TD3's actor loss; AWAC [B14] and IQL [B11] extract the policy by
advantage-weighted regression, with IQL also avoiding any out-of-sample action
query through expectile regression. IQL and TD3+BC are the simplest to
implement correctly. Two cautions apply.

First, Kumar et al. [B15] analyse when offline RL should beat behaviour
cloning: mainly when the data are suboptimal or noisy and the horizon is long,
so that good partial behaviours can be recombined. With near-expert data,
cloning is hard to beat. The behaviour-cloning baseline is therefore
necessary.

Second, Park et al. [B16] report that, on their benchmarks, the choice of
policy-extraction method often matters more than the value-learning objective,
and that advantage-weighted regression — the extraction step IQL uses —
exploits the learned value function less fully than behaviour-regularised
policy-gradient extraction of the TD3+BC kind. An IQL result in this campaign
is thus a result about IQL with weighted-regression extraction. A weak outcome
would not show that offline value learning failed.

The campaign's offline dataset is generated in the simulator by a mixture of
behaviour policies with known propensities, and is labelled simulated. It
contains no real fills, rejected orders or market reactions. IQL trained on it
answers "can a policy be extracted from logged simulator interaction" and
nothing about learning from the live system's logs.

Sequence-model and generative-policy methods — Decision Transformer [C07],
Diffusion-QL [C11], IDQL [C12], Flow Q-learning [C13] and its 2026 successors
found in the search — report strong results on D4RL and OGBench robotics and
navigation tasks. Their advantage is in representing multimodal action
distributions in high-dimensional action spaces. A scalar bounded exposure has
no comparable structure, and transformers, iterative diffusion sampling and
flow-matching training are impractical in CPU-only hand-written numpy. They
are rejected for fit in this campaign. Model-based offline methods (MOPO,
MOReL, COMBO [C08–C10]) penalise or truncate rollouts where a learned dynamics
model is uncertain. Here the endogenous dynamics (the ledger) are known
exactly and the exogenous dynamics (returns) are what nobody can model well;
learning a return model to roll out is the forecasting problem again. Also
rejected for fit.

### What finance evidence does and does not support

The peer-reviewed survey by Hambly, Xu and Yang [A09] and the critical survey
by Pippas, Ludvig and Turkay [N07] (167 publications) document a large
literature whose typical study is a single-path backtest with simplified
costs, assumed fills and no multiple-testing control. Zhang, Zohren and
Roberts [A08] is among the more careful examples (futures, volatility-scaled
rewards, classical time-series momentum baselines), but it remains one
research group's backtest. Gort et al. [N05] argue that deep-RL crypto
backtests are prone to false positives and propose a probability-of-overfitting
test for agent selection. Cliff and Rollins [N06] show in a market-simulator
setting that a trivially simple strategy can beat published AI traders when
experiments are run more thoroughly — a caution about experimental method more
than about RL specifically. Borrageiro, Firoozye and Barucca [A11] is the
closest published analogue to this campaign (a recurrent RL agent on a crypto
perpetual with funding); its abstract states that most of the reported return
comes from funding profit, which shows that results on perpetuals must be
decomposed into price, funding and cost components before being read as
directional skill.

The search found no independent, cost-inclusive, out-of-sample replication
showing an RL exposure policy beating a simple rule on crypto perpetuals. The
literature supports the formulation and the algorithms as optimisation tools.
It does not support a prior that RL improves on the champion.

## 4. Offline evaluation: replay is model-based evidence

Off-policy evaluation estimates the value of a target policy from data
gathered by a different behaviour policy. Importance sampling [D01] is
unbiased when behaviour propensities are known and the behaviour policy
supports every action the target takes, but its variance grows exponentially
with horizon. Doubly robust estimators [D02, D03] combine a value model with
importance weights and stay consistent if either is correct. Fitted Q
evaluation [D04] regresses the target policy's Q-function by iterated Bellman
backups and is used by Paine et al. [D05] for offline hyperparameter
selection. High-confidence OPE [D06] gives lower confidence bounds on policy
value from importance-weighted returns, and SPIBB [D07] permits deviation from
the behaviour policy only where state-action counts are sufficient. The two
benchmark studies, Voloshin et al. [D09] and DOPE [D08], agree on the central
negative finding: no estimator is reliable across settings, and accuracy
degrades with horizon, with behaviour-target mismatch and with function
approximation error. Liu et al. [D10] add that FQE and model-based OPE have
hyperparameters of their own, so OPE needs its own model selection.

Three consequences follow.

**Historical replay is not assumption-free OPE.** Running a new policy through
the replay simulator evaluates it under a model: the price path is unaffected
by our orders, fills occur at the next open plus an assumed cost, and funding
is as recorded. Under those assumptions replay is an exact counterfactual for
the ledger, which is why it is useful. The assumptions themselves are not
tested by replay. The earlier campaign recorded the simulator-to-fill gap as
unmeasured, not zero; that still holds.

**Live logs have no overlap.** The champion is deterministic given its
inputs. Its logs give propensity one to the action taken and zero to all
others, so importance sampling of any different policy from live logs is
undefined, and FQE on those logs extrapolates entirely through function
approximation. Stochastic logging would be required, and it is not authorised.

**What validation against simulator ground truth shows.** Running FQE and
importance sampling on simulator-generated logs with known propensities and
comparing with direct simulator rollouts tests the estimator code and its
finite-sample behaviour on this state distribution. It can show that the
estimators are untrustworthy even in the best case, which is worth knowing. It
cannot show that they are valid for real-market counterfactuals. Results
should be reported as estimator diagnostics, with effective sample size and
weight concentration, not as independent evidence of policy value. The earlier
screen's 108 OPE batches were all invalid because failed episodes could not be
excluded; a projecting shield removes that particular failure mode, since
episodes no longer terminate on a breach, but not the limits above.

One further point: seeds resample the learner, not the market. Five seeds on
one path measure optimisation variance. Sampling variability of returns is
addressed by the paired block bootstrap, under a local-stationarity assumption
that can fail.

## 5. Safety shielding: what can and cannot be guaranteed

García and Fernández [E03] divide safe RL into changing the optimisation
criterion and changing the exploration process; Gu et al. [E04] update the
taxonomy. Constrained methods (CPO [C01], Lagrangian PPO/TRPO as benchmarked
by Ray et al. [C02], CVaR objectives [C03]) satisfy constraints in expectation
or approximately at convergence, and can violate them during training and
after distribution shift. They are the wrong tool for hard limits. They could
later serve a soft objective such as a drawdown budget, with every hard limit
still enforced by the shield. Robust MDPs [C06] optimise against a set of
transition models; no defensible uncertainty set for crypto returns is
available, so robustness is probed here by cost and delay stresses instead.

Shielding [E01] enforces a safety specification at run time by monitoring the
agent's action and substituting a safe one when needed. The guarantee is
relative to an abstraction of the environment adequate for the specification.
Probabilistic shields [E02] bound the probability of reaching unsafe states
given an MDP model of the safety-relevant dynamics; Galesloot et al. [E12]
extend this to offline RL with interval MDPs estimated from data. For this
system there is no validated transition model of price gaps, so probabilistic
shielding of drawdown would be a guarantee conditional on an invented model.

That fixes what the shield can deliver.

- **Provable (action-admission invariants).** For every proposal and every
  admitted state, the shielded target lies within the exposure cap, changes by
  no more than the turnover limit, does not increase absolute exposure in
  reduce-only mode, and is finite; rounding never moves a quantity past a
  limit. These are properties of a deterministic function of (state,
  proposal). They suit SMT proof over real or rational arithmetic, plus
  conformance testing between the Haskell and Python implementations. A
  floating-point implementation needs its own argument.
- **Not provable (realised outcomes).** Realised exposure drifts between
  decisions as price and equity move. Realised drawdown depends on price gaps
  between decision points. Fills can differ from targets. The earlier campaign
  recorded a worst stress-path drawdown of 25.7378% and noted that endpoint
  detection cannot prevent market gaps. A shield that reads drawdown reacts
  after the loss. Any drawdown bound is conditional on an explicit gap
  assumption and must be stated with it.

**Projection and learning.** Dalal et al. [E05] add an analytic projection
layer that minimally corrects a continuous action to satisfy linearised
constraints. Krasowski et al. [E08] compare action replacement, projection and
masking, and examine which action (proposed or executed) should enter the
learning update. Two results bear directly on the design. Fujita and Maeda
[E07] show that when the environment clips a Gaussian policy's action, the
ordinary score-function gradient is still unbiased, but an estimator that uses
knowledge of the clipping has lower variance. Huang and Ontañón [E06] show the
discrete analogue: masking invalid actions inside the policy yields a valid
policy gradient and scales better than penalising them.

Treating the shield as part of the environment is therefore sound for PPO: the
policy is optimised for the composite system (shield then market), and the
gradient is unbiased for that system. The costs are extra variance and
saturation when the policy mean drifts outside the admissible interval. For
TD3 the issue is sharper. The actor is updated through the critic's action
gradient, and where the shield clips, the true Q-function is flat in the
proposed action, so the actor receives no corrective signal and can stay
saturated. The clipped fraction must be logged, and the choice between
proposed and executed actions in critic and replay must be declared in
advance. For IQL, the dataset should record executed actions so that
"in-support" refers to what the environment received.

**Neural-network verification.** Reluplex [E09] and Marabou [E10] decide
properties of ReLU networks by SMT-style search; α,β-CROWN-type verifiers
[E11] scale further with bound propagation and branch and bound. They could
bound a small MLP's output over an input box. That adds little here: the
shield bounds the executed action whatever the network emits, and a proof
about a real-arithmetic network says nothing about the floating-point numpy
code, inputs outside the box, or whether actions are good. It is deferred.

## 6. Negative evidence and failure modes

**Seed and implementation instability.** Henderson et al. [N01] showed that
reported deep-RL results change materially with random seed, hyperparameters,
codebase and reward scale, and that comparisons on a handful of seeds are
unreliable. Agarwal et al. [N02] showed that point estimates from few runs
mislead and recommended interval estimates, interquartile means and
performance profiles. Engstrom et al. [N03] showed that implementation details
can dominate the nominal algorithm. Five seeds per finalist is the minimum
this literature would accept; selection among seeds or checkpoints is
selection and counts toward multiple testing.

**Backtest overfitting and multiple testing.** The deflated Sharpe ratio [F04]
adjusts for the number of trials, non-normality and sample length; the
probability of backtest overfitting [F05] estimates how often the in-sample
best configuration underperforms out of sample. Harvey, Liu and Zhu [F06]
argue that the conventional significance threshold is too low given the number
of factors tried and propose a t-statistic hurdle above 3.0. RL multiplies
trials by algorithm, architecture, reward, seed and checkpoint; each trained
policy is a trial. The development panel has been used by earlier campaigns,
so its folds are contaminated development evidence, and the confirmation
window is about eleven weeks — roughly three hundred 6h bars per symbol —
which gives limited statistical power.

**Reward hacking and simulator exploitation.** A learner optimises the
simulator as written: earning returns before the fill, hiding terminal
liquidation cost, exploiting the episode boundary or stale prices, or
collecting misdated funding. The earlier campaign's audit table lists these
patterns and its controls. A projecting shield adds one: a policy can learn to
lean on the shield (always proposing the maximum and letting reduce-only mode
manage risk). That is admissible by construction and may be economically
poor, so the shield intervention rate must be reported per policy.

**Simple policies match or beat RL.** [B15] gives conditions under which
behaviour cloning is preferable to offline RL. [N06] shows a trivial strategy
beating published AI traders. [A08]'s own baselines are classical momentum
rules. In this repository the evidence is direct: the earlier preregistered
PPO / Double DQN / discrete CQL screen was rejected, with 686 of 1,080
base-cost RL paths breaching risk limits, all 108 OPE batches invalid and
DSR/PBO null, while the logistic baseline completed all paths at two of three
cadences with mean returns of 0.075% and 0.055%. The memo is explicit that
this establishes no advantage for the baseline either. That screen's shield
treated a breach as failure and did not project the action, so its failure
counts do not transfer to the present design. Its lesson does: no result is
interpretable without matched baselines and valid evaluation.

**Cost sensitivity.** Policies that trade every bar are fragile to cost
assumptions. In the earlier screen failures rose from 686 at base cost to 697
at 2× cost and 700 at 25 bp. The registered 1.5×, 2×, severe-cost and
one-bar-delay stresses are the minimum.

## 7. Supporting finance evidence for crypto perpetuals at 6h

Market efficiency [F01] sets the prior: public price history should carry
little exploitable information after costs, and any test of that is joint with
a model of expected returns. For crypto, Liu, Tsyvinski and Wu [F02] find that
market, size and momentum factors capture the cross-section of coin returns,
on weekly spot data. That supports momentum as a phenomenon in this asset
class. It does not establish time-series predictability at 6h on six
perpetuals, net of taker costs.

Perpetual futures have no expiry and are tied to spot by periodic funding
payments. He et al. [F03] derive no-arbitrage relations for perpetuals and
document that deviations of perpetual from spot prices are larger than in
traditional currency markets. For this design, funding is a cash flow that
depends on the position held at settlement times. Settlement cycles (commonly
8h) do not align with 6h bars, so accounting must use actual settlement
timestamps. A long-biased policy in a
positive-funding regime pays carry, a short-biased one receives it, and [A11]
shows how completely funding can dominate reported performance.

Microstructure enters only through the cost model. Fees, half-spread and
slippage are assumptions, not reconstructed fills, and next-open fills ignore
opening dynamics and exchange outages. Non-stationarity is the larger threat:
a policy trained on 2020–2025 is asked to generalise to 2026, and walk-forward
folds measure that only to the extent that past regime changes resemble
future ones. Unreviewed 2026 leads on crypto trend-following and
short-horizon reversal are listed in the search log.

## 8. Implications for the registered design

**What the literature justifies.**

- Formulating exposure control as a constrained sequential problem with
  inventory, cost and drawdown state [A01, A03], while requiring myopic and
  rule-based baselines to test whether the sequential structure matters [A07,
  B15].
- A continuous bounded action for PPO and TD3, and IQL as the offline
  representative, on grounds of implementability in numpy; and the exclusion
  of transformer, diffusion, flow and learned-model methods on grounds of fit
  [C07–C13].
- A deterministic projecting shield inside the environment, with claims
  limited to action-admission invariants [E01, E05, E07, E08].
- Bootstrapping on time-limit truncation [A10]; a declared list of
  implementation details [N03, N04]; at least five seeds with interval
  reporting [N01, N02]; DSR/PBO with every trained policy counted as a trial
  [F04–F06]; cost and delay stresses.
- Treating FQE and importance-sampling results as estimator diagnostics
  validated against simulator ground truth [D04, D08, D09, D10].

**What the literature does not settle — this campaign's empirical questions.**

- Whether any learned policy beats the unchanged champion, the regime-switch
  rule, the ridge policy, behaviour cloning and the myopic bandit after costs
  on walk-forward folds, and whether that survives the confirmation window.
- Whether the turnover limit and drawdown state bind often enough for
  sequential methods to differ from the bandit at all.
- How often the shield intervenes for each learner, and whether TD3 saturates
  against it.
- Whether results are stable across seeds, cost multipliers and a one-bar
  delay.
- Whether FQE and importance sampling track simulator ground truth on this
  state distribution.

**Standing limits whatever the outcome.** The development panel is previously
exposed. The confirmation window is short. The simulator's fill and impact
assumptions are unvalidated. The offline dataset is simulated. A positive
result would justify a further, separately registered step. It would not
establish profitability, and a negative result would be consistent with most
of the evidence reviewed here.

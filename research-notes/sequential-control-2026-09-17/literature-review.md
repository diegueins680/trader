# Prediction and sequential decisions: evidence update through 2026-09-17

The strongest transferable lesson is to separate a forecast, a decision policy,
and permission to execute. None of the sources establishes an economic advantage
for this repository's current instruments under its actual execution constraints.
The [51-row focused matrix](paper-matrix.csv) combines 18 established prediction
and validation references with 33 sequential-learning, evaluation and recent
benchmark references. The [earlier 50-paper review](../market-prediction-2026-09-04/literature-review.md)
remains the full map of classical prediction, factors and alternative data.
These are overlapping maps, not 100 independent pieces of trading evidence.

Searches on the execution date covered official PMLR/NeurIPS/AAAI proceedings,
arXiv records and author repositories, including 2025–2026 offline RL, market
impact, financial foundation-model benchmarks and safety. Primary pages and
selected full papers were inspected; no PDFs are stored in Git. This is a
bounded critical review, not a claim that every publication through the date
has been found. An inaccessible detail is recorded as unverified rather than
inferred from a paper's title. The scoring criteria are explicit in
[candidate-scorecard.md](candidate-scorecard.md).

## Prediction evidence retained and updated

Efficiency is a skeptical prior, not proof that all conditional information is
useless. Fama's joint-hypothesis problem, Welch–Goyal's unstable timing results,
and McLean–Pontiff / Hou–Xue–Zhang's replication evidence raise the burden of
proof after search. ARIMA, VAR/VECM, cointegration, Kalman/state-space models,
GARCH and stochastic volatility describe different conditional objects. A stable
variance forecast is more plausible than a robust next-bar signed-return edge.
HAR, regime-switching and change-point models should first be evaluated for risk
reduction or abstention, not assumed to supply direction.

Momentum, reversal, trend, carry, funding/basis and cross-sectional factors have
distinct horizons and economic risks. A point-in-time ranking portfolio cannot
be validated by a single-symbol residual signal. Current funding and reversal
negative results are more relevant to this repository than a published equity
sort with different costs, breadth and leverage. Trees, regularized linear/logistic
models, nearest neighbors and forecast combinations remain essential cheap
baselines. Online adaptation adds feedback and repeated-selection risk.

LSTM/GRU/TCN/attention/Transformer/PatchTST architectures are generic modeling
tools. Their benchmark superiority does not establish trading efficacy, and the
repository's three sequence identifiers remain documented proxies. Quantiles,
probability calibration, conformal intervals and distributional forecasts must
be assessed for coverage, width, conditional reliability and net abstention value.
Exchangeability or average coverage is not a tail-loss guarantee.

The previous foundation-model discussion is updated by
[Chronos-2](https://arxiv.org/abs/2510.15821), which supports multivariate and
covariate-informed forecasting. It is inappropriate to treat old single-channel
limitations as current. Nevertheless, the financial benchmark of
[Noguer I Alonso and Pereira Franklin](https://arxiv.org/abs/2606.27100) finds
only sparse predictive significance over a random walk. The daily futures
benchmark of [Saly-Kaufmann et al.](https://arxiv.org/abs/2603.01820) reports stronger
results for some temporal representations and studies seeds, tails, breakeven
costs and computation. These are useful, conflicting context—not a cryptocurrency
replication or justification to replace a CPU baseline.

For microstructure, OFI, adverse selection, spreads and depth require event data
and a fill model. Taker-volume aggregates cannot substitute for quote additions
and cancellations. Open interest, liquidations, option-implied tails, on-chain and
stablecoin activity, institutional/custody flows, macro releases, news/filings,
search/web/social attention, prediction markets, incidents, developer/governance
activity and supply changes are candidate **data families**, not proven edges.
Each needs release/vintage or first-seen timing, coverage masks and an incremental
ablation. No family is added merely because a collector exists.

## Reinforcement-learning field map

An MDP requires a transition/reward specification; a POMDP additionally requires
acknowledging hidden state. Tabular Q-learning and SARSA illustrate off-policy
versus on-policy bootstrapping. REINFORCE directly estimates a policy gradient;
[A3C](https://arxiv.org/abs/1602.01783) and synchronous A2C reduce variance using a
critic. None removes low signal-to-noise or the need to identify executable
counterfactuals. For weakly coupled choices, epsilon-greedy, UCB, Thompson sampling,
LinUCB/linear Thompson, discounted or sliding-window bandits are preferable
complexity baselines. Non-stationarity tuning must itself be registered.

DQN uses nonlinear Q estimates and replay. [Double DQN](https://arxiv.org/abs/1509.06461)
addresses maximization bias; [dueling networks](https://arxiv.org/abs/1511.06581)
factor value and advantage; [prioritized replay](https://arxiv.org/abs/1511.05952)
changes sampling and requires bias correction; [Rainbow](https://arxiv.org/abs/1710.02298)
combines mechanisms. These are not interchangeable cosmetic names.
[C51](https://proceedings.mlr.press/v70/bellemare17a.html),
[QR-DQN](https://arxiv.org/abs/1710.10044) and
[IQN](https://proceedings.mlr.press/v80/dabney18a.html) estimate return distributions.
Distributional actor–critics extend that idea to continuous control; selecting a
tail functional still needs calibrated adverse-event data and independent limits.

[DDPG](https://arxiv.org/abs/1509.02971),
[TD3](https://proceedings.mlr.press/v80/fujimoto18a.html) and
[SAC](https://proceedings.mlr.press/v80/haarnoja18b.html) serve continuous actions.
PPO is an on-policy comparator. Offline learning changes the question: actions
outside the logged behavior may have unidentifiable values. Behavior cloning,
[AWR](https://arxiv.org/abs/1910.00177), [AWAC](https://arxiv.org/abs/2006.09359),
[BCQ](https://proceedings.mlr.press/v97/fujimoto19a.html),
[BEAR](https://arxiv.org/abs/1906.00949), CQL, IQL and TD3+BC constrain extrapolation
in materially different ways. This review selects finite-action CQL because its
penalty can be tested directly with a small model, not because it has the highest
published reward. Imitation cannot create support for actions absent from data.

[Decision Transformer](https://arxiv.org/abs/2106.01345) conditions trajectories
on desired returns. Such conditioning does not create a feasible favorable
trajectory; diverse behavior data are essential. Dyna-style planning, [MBPO](https://arxiv.org/abs/1906.08253)
and [Dreamer](https://arxiv.org/abs/1912.01603), including
[later world-model work](https://arxiv.org/abs/2301.04104), risk optimizing simulator
error when market impact is unidentified. Robust MPC is attractive when a small
credible transition model and uncertainty set exist. Hierarchical, multi-agent,
meta-learning and continual-learning trading add scope without supplying that
missing evidence and are deferred.

Recent [FQL](https://seohong.me/projects/fql/) is an ICML 2025 contribution;
[BFQ](https://arxiv.org/abs/2606.10613) has an author-reported ICML 2026 journal reference and proposes efficient
single-step flow policies. [SafeFQL](https://arxiv.org/abs/2603.15136) was accepted at
RLC 2026 according to its July revision and studies reachability-inspired safety.
These are credible monitoring items. Their control benchmarks and learned safety
values do not override Haskell authorization or establish cryptocurrency tail risk.

## Detailed technical reviews

Together with the eight retained detailed reviews of Welch–Goyal, Corsi, Gu–Kelly–Xiu,
Fischer–Krauss, PatchTST, Cont–Kukanov–Stoikov, Liu–Tsyvinski–Wu and White/Hansen
in the prior packet, the following twelve entries form the focused technical
reading set. The matrix records metadata and explicit unavailable details.

### 1. PPO — Schulman et al. (2017)

[Primary paper](https://arxiv.org/pdf/1707.06347).
Question: can a simple surrogate support multiple gradient epochs without
destabilizing policy updates? It clips the sampled action probability ratio and
uses advantage estimates, testing Atari and continuous-control episodic reward.
Those are simulator transitions, not financial dates or cost-controlled trades.
Reported benchmark gains have broad algorithmic reproduction, but no verified
crypto execution replication is established here. Small-seed sensitivity and
implementation choices remain material. We implement the clipped objective and
GAE with an original 16-unit CPU network; no benchmark score is reproduced.
Code dependencies, published network sizes and training budgets differ. Costs
come from our environment, not the paper. Clipping does not enforce exposure or
drawdown. Promotion requires a separate economic test.

### 2. Double DQN — van Hasselt, Guez and Silver (2016)

[Primary paper](https://arxiv.org/abs/1509.06461).
Question: does separating action selection from value evaluation reduce DQN
overestimation? The online network selects the next action; the target network
evaluates it. Atari experiments study game scores and value bias, not market
return distributions or trading costs. Later DQN-family benchmarks provide
algorithmic comparisons, not a financial replication. Our finite-action network
retains the Double-Q target and replay, using 256-transition collection blocks
with one update per collected transition afterward. This is a documented
batching simplification and far smaller budget. It cannot solve non-stationarity,
unknown liquidity, or offline support by itself; Q estimates are not probabilities.

### 3. CQL — Kumar et al. (2020)

[Primary paper](https://arxiv.org/html/2006.04779v3).
Question: can pessimistic action values reduce offline extrapolation error?
The finite-action penalty is log-sum-exp over Q minus Q on logged actions, added
to Bellman regression. Experiments use offline Atari and D4RL; cost/market dates
are inapplicable. The practical discrete paper uses QR-DQN; our Double-DQN critic,
fixed coefficient and tiny CPU network are explicit deviations, not a faithful
Atari reproduction. The [author code](https://github.com/aviralkumar2907/CQL) warns
that dataset versions change results. Lower-bound arguments have assumptions;
they do not certify arbitrary neural estimates under regime shift. Our uniform
simulated behavior gives known propensities but no real execution support.
Independent crypto replication is unverified. No external code is copied.

### 4. IQL and TD3+BC — Kostrikov et al.; Fujimoto and Gu

[IQL](https://arxiv.org/abs/2110.06169) uses expectile fitting and advantage-weighted
policy extraction; [TD3+BC](https://arxiv.org/abs/2106.06860) adds behavior cloning
to a simple continuous-control learner. Both ask how far useful policy improvement
can proceed without querying unsupported actions. Public D4RL experiments and
[IQL code](https://github.com/ikostrikov/implicit_q_learning) /
[TD3+BC code](https://github.com/sfujim/TD3_BC) aid reproducibility; versioned dataset
and preprocessing choices still matter. These are benchmark rewards, not funded
perpetual PnL. They provide serious alternatives to increasingly complex offline
methods. We defer implementation because the selected actions are discrete and
CQL already tests offline conservatism. No financial superiority is inferred.

### 5. SAC and TD3 — Haarnoja et al.; Fujimoto et al. (2018)

[SAC](https://proceedings.mlr.press/v80/haarnoja18b.html) learns an entropy-regularized
stochastic actor from off-policy transitions.
[TD3](https://proceedings.mlr.press/v80/fujimoto18a.html) uses twin critics and delayed
updates to address approximation bias. Their continuous-control benchmarks
demonstrate algorithmic mechanisms and sample-efficiency tradeoffs; neither
validates a crypto exposure policy or historical execution model. Reward scaling,
entropy and action support matter. The current three-target contract makes a
continuous actor unnecessary. Both remain relevant for a future separately
registered participation-rate or allocation task with credible dynamics.

### 6. Sequential OPE — Jiang–Li; Thomas–Brunskill (2016)

[DR](https://proceedings.mlr.press/v48/jiang16.html) combines approximate value
functions with importance-weighted residuals; [weighted estimators](https://proceedings.mlr.press/v48/thomasa16.html)
trade finite-sample bias and variance. The target is policy value from logged
trajectories, assuming adequate overlap and meaningful behavior probabilities.
Theoretical/benchmark results are not licensed market-fill datasets. Ordinary
IS, per-decision IS, WIS and DR are implemented here on short simulated episodes.
Uniform propensity is known exactly, but deterministic six-action agreement is
rare. We report ESS, disagreement and conditional bootstrap uncertainty. FQE,
model-based and uncertainty-weighted combinations would still inherit simulator
identification errors; they are not a route around missing support. No estimator
is selected because it produces a favorable value.

### 7. Reliable RL evaluation — Henderson et al.; Agarwal et al.

[Henderson et al.](https://arxiv.org/abs/1709.06560) document sensitivity to seeds and
experimental choices. [Agarwal et al.](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
advocate uncertainty-aware aggregate comparisons; [rliable](https://github.com/google-research/rliable)
provides official tooling. Simulator-task benchmarks are reproducible contexts,
not economic validation. All three registered seeds must remain visible here,
including failures. Three seeds are an engineering screen, insufficient to
establish small differences in market performance. Market inference also needs
dependence-aware blocks, full trial accounting, DSR/PBO and genuinely new periods.
Seed bootstrap alone cannot repair adaptive reuse of a historical panel.

### 8. Safety — CPO, CVaR and shielding

[CPO](https://proceedings.mlr.press/v70/achiam17a.html) addresses expected constraints;
[CVaR control](https://arxiv.org/abs/1506.02188) makes adverse tails part of the
objective; [shielding](https://ojs.aaai.org/index.php/AAAI/article/view/11797) places
a reactive safety mechanism outside the learner. Theoretical guarantees depend
on constraint and transition assumptions. Simulated control evaluations do not
establish margin safety through an exchange gap. In this repository, deterministic
permissions and risk controls retain precedence. The new Haskell proposal type
always reports no order authority. Inventory penalties are supplementary; a
stopped drawdown path remains a failed path. This implements an isolation
boundary and bounded executable contracts, not a theorem about live solvency.

### 9. Optimized execution — Nevmyvaka, Feng and Kearns (2006)

[Author paper](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf).
The task is to finish a specified order within a horizon, balancing price and
nonexecution. It uses 1.5 years of millisecond NASDAQ order-book data for AMZN,
QCOM and NVDA, split into 12 months training and six months test. A low-impact
state factorization separates private inventory/time from market variables.
Reported execution improvement is relative to execution baselines, not predictive
portfolio alpha. Historical order priority and terminal completion are more
relevant than a next-price RMSE. Data/code reuse rights and independent crypto
replication are unverified here. Our close-only panel cannot reproduce this
paper; execution RL is deferred until equivalent event and queue evidence exist.

### 10. Financial impact sensitivity — Riera Abbade and Reali Costa (2026)

[Primary preprint](https://arxiv.org/html/2603.29086v1).
Five RL algorithms are compared in NASDAQ-100 trading/portfolio simulators under
fixed 10bp and impact-aware costs, with hyperparameter optimization. The reported
algorithm ordering and turnover depend materially on cost modeling. That is
evidence against treating a simulator score as intrinsic algorithm superiority.
Equity data, fill assumptions, tuning and independent-replication status limit
crypto transfer. This motivates our 1.5x/2x/extreme-cost, delay, partial/missed-fill
and adverse-impact sensitivities. A stress coefficient is not an empirical
capacity estimate. No claim of reproducing their return figures is made.

### 11. Financial foundation-model benchmark — Noguer I Alonso and Franklin (2026)

[Primary paper](https://arxiv.org/html/2606.27100v1).
Five US equities use daily adjusted closes from 2014-09-15 to 2026-02-15 and
rolling-origin forecasts of linear/log returns with equal context budgets.
Foundation models often rank well, but predictive gains versus random walk are
small and sparse under one-sided Diebold–Mariano tests. This is a useful negative
result; it is not proof of no predictability in every market. Small universe,
pretraining overlap and the missing executable crypto portfolio comparison limit
transfer. Official artifact/data licensing and independent replication remain
unverified here. No foundation-model service or checkpoint is introduced.

### 12. Modern temporal benchmark — Saly-Kaufmann et al. (2026)

[Primary paper](https://arxiv.org/html/2603.01820v1).
Daily futures across commodities, equity indices, bonds and FX cover 2010–2025.
The target is prediction/position sizing with risk-adjusted objectives. The paper
compares linear, recurrent, Transformer and state-space representations and
reports tail, seed, computation and breakeven-cost evidence. Some hybrid temporal
models perform strongly, contrasting with simplistic claims that deep models
always fail. This remains a preprint with different instruments and a search
universe requiring replication accounting. A breakeven-cost calculation cannot
establish funding, exchange outages or queue fills. It motivates monitoring
faithful versions, not silently changing the repository's legacy proxy IDs.

## Skeptical synthesis and boundaries

Publication, peer review and open code are necessary credibility signals, not
production acceptance. Financial RL can exploit missing terminal costs, favorable
fills, future bars, synthetic zero features, unobserved funding or reset options.
Offline RL can overestimate unsupported actions; on-policy simulation can optimize
incorrect transitions; world models can make that problem worse. A learned
constraint can misgeneralize exactly during a tail event.

The [COBS study](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/a5e00132373a7031000fd987a3c9f87b-Abstract-round1.html)
supports evaluating OPE under varied conditions rather than trusting one estimator.
Neither this review nor the selected methods establish that RL beats a bandit,
supervised policy or deterministic optimizer in these markets. The screen retains
those comparisons and forbids promotion regardless of development ranking.
Unavailable independent confirmation is an explicit evidence gap, not a null
hypothesis rejection and not a completed production validation.

## Audit source refresh

[JumpStart (Omi et al., submitted 2026-09-12)](https://arxiv.org/abs/2609.13730v1)
reports more than 160,000 trained policies over 114 offline datasets. Algorithm
ordering changes with tuning and benchmark composition; no method dominates
across all reported environments. This new preprint supports comprehensive
trial accounting, with independent replication and release completeness still
unverified. It does not validate a cryptocurrency policy, justify retuning the
rejected screen, or provide fresh financial confirmation. Its disposition is
Monitor. This audit refreshed primary sources, not every earlier matrix detail.

The [BFQ primary record](https://arxiv.org/abs/2606.10613) now identifies ICML 2026
in its author-supplied journal reference. The matrix distinguishes that report
from independent proceedings verification. Its economic disposition is unchanged.

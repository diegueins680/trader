# RL decision-layer review, searched 2026-10-10

The implementation question is sequential exposure control, not another forecast
competition. Inventory affects the next period's PnL, funding, liquidation and
turnover, and a next-close action cannot earn the previous interval's return.
A small discrete action set makes PPO, Double DQN and discrete CQL appropriate
mechanism comparisons. Their credibility does not establish profitability here.

## Baseline and scope audit

Verified main is `354c853176b7ee2ac6570951f7f6f2d0aed6909d` with successful
[CI run 37959950956](https://github.com/diegueins680/trader/actions/runs/37959950956).
The original checkout was on an older research branch with user edits; no tracked
file in that checkout was edited. A new ignored research archive was retained there.
Main already contained the September screen, corrected
GAE/value/optimizer kernels, exact accounting and causal replay. Frozen identifiers
remain untouched. This campaign adds their learner/environment composition.

The current adopted fleet is identified by the existing audit as UNI/SUI/ETC;
[PR330](https://github.com/diegueins680/trader/pull/330) separately reports no trades
from a lawful post-creation screen. Its private adopted-combo bytes and unmerged
CLI adapter are not part of verified main. Neither a cash policy nor a Python
lookalike is substituted for that champion. Using a 2026-created champion on our
2022–2023 development folds would itself need chronology justification. Consequently
this pilot cannot answer the required champion-superiority question.

The parent reviewed universe contains ten Binance USD-M perpetual symbols on 8h
bars; BTC, ETH and SOL are the bounded pilot subset. The exact exported panel and
settlements are admitted by hash before parsing. These rows were already used by
45 earlier adaptive trials and the September RL campaign. The 1,227-return final
holdout and prospective carry observations remain protected. Available data have
no authentic historical first-seen clocks, L2 queues or calibrated impact. Close+1ms
availability is an explicit assumption, not fabricated provenance. Source terms
and snapshot hashes are in the registration; raw provider data are not redistributed.

Direct source inspection on this base confirms `PatchTST` computes patch summaries
and fits ridge coefficients, `TCN` fits ridge on selected dilated return lags, and
`Transformer` stores feature/target examples and uses softmax dot-product similarity.
These remain proxies, not the named learned patch transformer, convolutional stack
or attention network. No identifier or saved-artifact semantics changed. Our
observations use causal price features, not any of these predictor outputs, so
predictor fidelity cannot be credited as an RL improvement.

## Search protocol and paper map

The companion matrix selects 41 relevant rows from the inherited 55-row
[sequential matrix](../sequential-control-2026-09-17/paper-matrix.csv), then adds
nine focused records. Retained entries preserve their earlier review provenance;
they are not falsely described as 50 new full-paper reviews today. The inherited
[technical review](../sequential-control-2026-09-17/literature-review.md) contains
12 detailed sections covering more than 12 papers, mostly RL/OPE/safety. Today's
full-text rereads of PPO, Double DQN, CQL and FQL and the FineFT technical review
below bring the focused method set into the requested range; screened abstracts
remain labeled as such. Primary papers and author/official repositories are used.
No paper PDF, downloaded model or restricted dataset is committed.

Queries on 2026-10-10 included:

- `reinforcement learning trading transaction costs Moody Saffell 2001 direct reinforcement`
- `site.arxiv.org proximal policy optimization conservative Q learning implicit Q learning financial reinforcement learning reproducibility`
- `site.arxiv.org offline reinforcement learning 2025 2026 flow matching diffusion policy evaluation`
- `site.arxiv.org/abs "offline reinforcement learning" "2026" trading`
- `site.arxiv.org/abs "One-step" "Flow" "Reinforcement"`
- `site.proceedings.mlr.press reinforcement learning matters Agarwal 2021 doubly robust Jiang 2016`
- `site.proceedings.neurips.cc conservative Q learning shielding reinforcement learning`
- `site.arxiv.org "reinforcement learning" "trading" "2026" "cost"`
- `site.arxiv.org "offline reinforcement learning" "2026" "evaluation"`
- `site.proceedings.mlr.press "Offline Reinforcement Learning" "2025"`
- `site.arxiv.org "diffusion" "offline reinforcement learning" 2025 2026 benchmark`

Search results from secondary indexes were used only to locate primary sources.
The review is a bounded search through the execution date, not a claim of exhaustive
coverage. An apparent December 2026 journal listing was not used as published
execution-date evidence. Code links without an inspected license remain unverified.
No official algorithm code was copied; the existing original NumPy kernels are
reused with repository pins. IQL, FQL and Stable-Baselines3 license pages inspected
today specify MIT. The attempted CQL root `LICENSE` URL returned 404; do not infer
that every subproject permits reuse from public access alone.

## Implementation-directed technical review

**PPO (Schulman et al., arXiv 1707.06347v2, technical report).** The clipped
surrogate is a minimum of clipped and unclipped probability-ratio objectives;
it is not a hard cap on every ratio or a financial risk guarantee. We retain raw
proposal likelihoods through the shield and use fresh simulator rollouts. GAE
critic targets use raw advantages before policy-only standardization. A rollout
cut preserves the episode and bootstraps; a finite economic terminal does neither.
The small 16-unit network and 512-transition pilot are major budget deviations.
Finite-difference gradients and a known-optimum toy check mechanism fidelity,
not replication of Atari/MuJoCo scores or convergence. [Paper](https://arxiv.org/pdf/1707.06347).

**Double DQN (van Hasselt et al., AAAI 2016, arXiv 1509.06461v3).** Separating
online argmax from target evaluation addresses one source of maximization bias.
The implementation retains independent immutable target snapshots and replay,
with epsilon-greedy fresh simulator collection in blocks. A golden formula case
where target argmax differs from online argmax documents the target convention;
the actual learner is exercised by the known-optimum toy test and source review.
A full update-by-update comparison with a maintained reference learner remains open.
Values are expected rewards, not calibrated probabilities; the evaluation records
finite-path critic error. Gamma=1 is registered for the finite equity objective,
not imported from continuing Atari training. [Paper](https://arxiv.org/html/1509.06461v3).

**CQL (Kumar et al., NeurIPS 2020, arXiv 2006.04779v3).** The penalty lowers values
away from the behavior distribution relative to observed actions. We use the
scalar finite-action logsumexp penalty and corrected stable objective, not the
paper's distributional Atari implementation. The frozen simulated dataset has
known uniform *proposal* probabilities; shielded action support is different.
The same dataset is shared across seeds and the alpha-zero ablation, then hashed
again to detect mutation. No collection occurs during offline updates. Neural
lower-bound validity under market shift is unproved. [Paper](https://arxiv.org/html/2006.04779v3).

**IQL, TD3+BC, BCQ and AWAC.** Their inherited matrix entries describe different
responses to extrapolation: expectile/advantage extraction, cloning regularization,
behavior-constrained action candidates and advantage-weighted actor fitting.
IQL remains a credible alternative; CQL was chosen before results because the
three-action proposal domain makes its support penalty directly auditable.
Continuous SAC/TD3 and expressive generative actors are deferred. Frozen ordinary
DDQN appears only as the explicitly diagnostic alpha-zero ablation, never relabeled
as a robust offline algorithm. [IQL](https://arxiv.org/abs/2110.06169),
[official MIT code](https://github.com/ikostrikov/implicit_q_learning/blob/master/LICENSE).

**Sequential OPE and reliable evaluation.** Jiang–Li's DR construction needs
meaningful behavior ratios and support; a learned critic does not identify absent
counterfactuals. We have no real behavior log or usable real propensity and report
no real IS/DR/FQE number. Simulator replay is model-based evidence. Five seeds
measure optimizer variability, not five independent markets. Paired block
resampling averages correlated seeds and symbols before resampling each chronological
window. Short, adaptively exposed windows cannot satisfy DSR/PBO confirmation.
[DR](https://proceedings.mlr.press/v48/jiang16.html),
[statistical reliability](https://arxiv.org/abs/2108.13264).

**Shielding (Alshiekh et al., AAAI 2018).** A reactive controller protects a
specified transition model; its guarantee is conditional on the abstraction and
available safe actions. Our pure shield handles proposals and observed risk state.
No theorem promises future solvency under arbitrary gaps. A fixed neutral target
is only a costed simulation fallback. Existing live protective ownership must not
be inferred from it. New exact-real claims, finite lifecycle checks and Haskell
conformance tests are distinguished in the assurance ledger.
[Paper](https://arxiv.org/abs/1708.08611).

**FQL (Park et al., ICML 2025, arXiv 2502.02538; v1 full text inspected).** Flow
behavior modeling and one-step policy extraction improve expressive continuous
control. Appendix A retains ODE solving during training; fast inference is not
free training. The benchmark does not test our three target exposures, cost ledger
or chronology. A flow actor adds no justified first-campaign benefit, so it is
monitored, not implemented. [Paper](https://arxiv.org/html/2502.02538v1),
[MIT implementation](https://github.com/seohongpark/fql/blob/master/LICENSE).

**BFQ (Nguyen et al., arXiv 2606.10613v1, record reports ICML 2026).** The record
proposes bootstrapped single-step flow displacement learning and D4RL evaluation.
This refresh checked its abstract and publication record, not its full theorem or
independent reproduction. It does not resolve missing financial support or justify
replacing a categorical three-action actor. [Record](https://arxiv.org/abs/2606.10613).

**SafeFQL (Tayal and Tayal, arXiv 2603.15136v2, RLC 2026 acceptance reported).**
Reachability-based safety values and a distilled flow policy target efficient
safe continuous control. Boat/Safety Gymnasium evidence does not identify exchange
gap distributions. Its learned safety estimates would remain subordinate to an
independent shield. Abstract/status screening only; no certificate imported.
[Record](https://arxiv.org/abs/2603.15136).

**FineFT (Qin et al., KDD 2026, DOI 10.1145/3770854.3780187).** This directly
relevant futures study selectively trains an ensemble and uses validation-fitted
VAE routing with a conservative alternative. Its four-crypto minute-scale
experiments use 5x leverage and 0.02% transaction cost; the reported compute is
four RTX4090s for six hours. Several baselines have large negative returns, and
the method is not best in every raw-return cell. Validation routing creates
selection work that must stay outside final testing. Ensemble members with
separate seeds are not independent replications of the full selection process.
The peer-reviewed contribution is credible enough to monitor, but its data,
compute and execution assumptions differ sharply here. No independent financial
replication or verified code license was established in this review. Paper CC-BY-4.0
does not establish the code's license. No leverage change or ensemble extension
is justified for this pilot. [Author paper](https://personal.ntu.edu.sg/boan/papers/KDD26_FineFT.pdf),
[official code](https://github.com/qinmoelei/FineFT_code_space).

**Recent simulator and generative work.** The March 2026
[market-impact environment](https://arxiv.org/abs/2603.29086) motivates cost stress,
but close-only history cannot calibrate its impact law. July's
[SciPhy](https://arxiv.org/abs/2607.15195) and
[DIVO](https://arxiv.org/abs/2511.08922) are screened preprints; neither supplies
missing exchange trajectories or establishes our counterfactuals. April's
[truncated rectified-flow actor](https://arxiv.org/abs/2604.09159) is also outside
the chosen categorical action problem. These are scope decisions, not negative
replications. [SC² at KR2026](https://proceedings.kr.org/2026/89/) reinforces the
need to state the temporal property and modeled domain; no new prover is warranted.

## Method selection and proof applicability

| Method | Credibility | Fit to this problem | System improvement |
|---|---|---|---|
| PPO | established simulator policy-gradient mechanism | appropriate on-policy comparator | measured only in registered pilot; no adoption inferred |
| Double DQN | established discrete control mechanism | appropriate simulator off-policy comparator | measured, including stopped paths and critic error |
| discrete CQL | credible offline pessimism mechanism; implementation is a declared variant | appropriate frozen simulated-data comparator | measured; no conservative-value theorem imported |
| BC and deterministic controls | transparent, inexpensive | essential matched inputs/ledger | retained even if activity is low or zero |
| recurrence/distributional/flow/ensemble | plausible conditional benefits | additional data, state or complexity not justified yet | untested here, deferred |

Neural correctness is not proved by an optimizer loss decreasing. Our proof scope
is causal construction, proposal/risk admission, reward/accounting algebra and
isolation; analytical gradient, finite-state and differential tests connect some
implementation boundaries. Compiler, OS, market transition identification and
Haskell admission of the new artifacts remain separate gaps. Negative evidence
from the earlier screen and the current pilot is retained rather than overwritten.

The execution task remains a distinct deferred use case. Nevmyvaka, Feng and
Kearns (ICML2006) use millisecond NASDAQ limit-order records with private time and
inventory state; their 12-month training/six-month test split and order-priority
simulator support execution comparisons, not exposure alpha from 8h candles.
This full-text reread reinforces the decision not to claim queue or high-frequency
execution fidelity. [Author paper](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf).

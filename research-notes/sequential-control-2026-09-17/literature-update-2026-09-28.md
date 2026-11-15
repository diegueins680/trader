# Literature surveillance through 2026-09-28

This extends the existing [50-paper financial review](../market-prediction-2026-09-04/literature-review.md),
[51-entry sequential matrix](paper-matrix.csv), twelve detailed sequential
assessments and [algorithm scorecard](candidate-scorecard.md). It does not claim
an exhaustive search. No paper PDF, licensed dataset or third-party code is copied.
The [supplementary matrix](paper-update-2026-09-28.csv) marks unverified details.
PR #282 contains a separate formal-methods survey; this work does not duplicate it.

Execution-date searches covered financial/offline RL, probabilistic shielding,
market-impact simulators and financial foundation-model benchmarks, plus a separate
September 2026 search. Primary manuscripts and publisher pages were opened; blogs
and promotional results were excluded from efficacy evidence. Existing replication,
cost, support and formal compatibility gates remain authoritative. A publication
does not reset trial accounting or make a previously opened holdout untouched.

## Safe offline improvement: conditions matter

Galesloot, Rhemrev and Jansen, **Robust Probabilistic Shielding for Safe Offline
Reinforcement Learning**, AAMAS 2026, build an interval MDP from fixed trajectories
and restrict actions using worst-case reach-avoid probabilities. The method assumes
known safe/unsafe sets and the transition graph; uncertainty is over transition
probabilities. Experiments use four nonfinancial benchmarks. Guarantees depend on
the abstraction and statistical uncertainty set. Independent financial replication
and code/data licensing were not established here.
[Primary manuscript, sections 3–5](https://arxiv.org/html/2605.10293v1).

Repository inference: unknown jumps, incomplete support and unmodeled fills prevent
importing this guarantee. The new gap-risk lemmas make a missing assumption
explicit. Do not invent transition frequencies to obtain a probabilistic result.

## Newly published financial RRL

Witkowski, Wachowicz and Kania, **Self-augmenting technical indicator with recurrent
reinforcement learning**, published 23 September, test MACD-history policies on
eight FX pairs, training 2019–2021 and testing 2022–2025. They report 20 seeds and
improvements over optimized MACD, but positive mean test Sharpe for only two pairs.
Test performance is monitored every 20 training episodes. That test does not meet
this repository's sealed confirmation protocol. The published availability section
still withholds the permanent code/data link for review. Proportional costs do not
establish crypto execution realism.
[Publisher text](https://link.springer.com/article/10.1007/s40622-026-00479-x).

Disposition: monitor. No independent replication, machine-checked policy guarantee
or matched-champion economic evidence was established. Improvement over a losing
baseline alone does not resolve this repository's failed gates.

## Foundation-model forecasts are not trading returns

Noguer I Alonso and Pereira Franklin compare pretrained models and local neural
baselines on five equities. The full text specifies a **20-business-day horizon**,
512-point context and ten rolling origins; training ends 2024-08-21 and evaluation
starts 2024-08-22. Gains over random walk are sparse. The authors exclude investment
utility conclusions without turnover, cost and capacity evidence. Companion
software detail is needed for exact reproduction. Independent cryptocurrency
replication was not established here.
[Primary manuscript, sections 4 and 6](https://arxiv.org/html/2606.27100v1).

Correction: the earlier matrix's “daily forecasts” horizon was imprecise; daily
is sampling frequency, not the 20-day horizon. Fix that field without rewriting
experiments. Keep foundation models on monitor; no existing proxy is replaced.

## Generalization and evaluation evidence

Yuan et al., **MetaTrader**, AAAI 2026, study transformed stock data, bilevel training
and conservative TD targets, reporting results on two public datasets. This review
verified publisher metadata and abstract, not the complete appendix. Costs, periods,
independent replication and code/data licenses remain unverified. Monitor the
mechanism; transformed stock histories do not establish crypto robustness.
[Publisher record](https://ojs.aaai.org/index.php/AAAI/article/view/40027).

Grądzki's **Unstable Gains: Multiplicity-Aware Evaluation of Financial Deep
Reinforcement Learning** publisher search record reports seed instability and
multiplicity-sensitive equity/crypto comparisons. Its issue is December 2026;
an online-first date could not be verified and the full page failed retrieval.
This is metadata-only surveillance, excluded from decisive cutoff evidence.
Existing Henderson/Agarwal and DSR/PBO sources already justify retaining all seeds.
[Publisher record](https://doi.org/10.1016/j.jfds.2026.100205).

## Research decision

Four existing families remain: HAR volatility gating, missingness-aware calibrated
shallow prediction, depth-normalized OFI, and bounded sequential control. No fifth
family, reward variant or adaptive successor is introduced. The first three retain
their data/registration restrictions; tested PPO, Double DQN and CQL configurations
remain rejected. Better evidence and simulator identification are prerequisites
to another financial experiment.

The existing map covers value/policy/actor–critic, offline, distributional,
model-based, constrained/CVaR, bandit and sequence methods. Three discrete-action
paradigms and deterministic/supervised/contextual baselines already answer the
initial mechanism screen negatively. Transferable alpha, reliable OPE, real fills
and environmental jump/debit bounds remain empirical claims, never proved here.

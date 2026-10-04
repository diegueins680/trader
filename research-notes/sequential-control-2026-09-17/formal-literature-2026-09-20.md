# Formal methods and research evidence update — 2026-09-20

This is a dated supplement to the [51-paper sequential review](literature-review.md)
and [50-paper prediction map](../market-prediction-2026-09-04/literature-review.md),
not a replacement or a claim of exhaustive bibliographic coverage. The earlier
matrices overlap. Their 20 detailed reviews and existing candidate scorecards
remain the selection record. No paper is newly implemented as a trading method.

Primary-source searches on September 20 covered formal shielding, offline RL,
financial prediction, SMT floating point, probabilistic checking and refinement.
New-paper screening used official proceedings, author-maintained tools and arXiv
records. An abstract-only screening is marked as such; neither a benchmark claim
nor a preprint title establishes financial efficacy or a verified implementation.
Only original summaries, metadata and links are retained; no PDFs or data copied.
The [supplementary matrix](formal-paper-matrix-2026-09-20.csv) preserves missing
metadata explicitly rather than inventing DOI, code, license or replication facts.

## What verification can establish

[Alshiekh et al., AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11797)
separate the learner from an externally synthesized shield. Safety requirements
are expressed in temporal logic and enforced against modeled environment dynamics.
This directly motivates the separation of policy proposal from deterministic
permission. The assumptions are the accuracy of the abstraction and the shield's
implementation, not market stationarity. Demonstrations concern control scenarios;
there are no crypto instruments, financial dates, funding charges or market-impact
estimates to transfer. A hand-written rejection function is not automatically a
synthesized temporal shield. Here, the small existing gate is modeled directly,
and the unproved translation boundary is visible in the ledger. No convergence,
optimality or portfolio-loss guarantee is inherited from this paper.

[Pranger and Könighofer, 2026](https://arxiv.org/abs/2606.03804) connect Tempest shield
synthesis to Gymnasium through tempestpy, including stochastic multiplayer-game
support and MiniGridSafe examples. This addresses integration friction, not trading
simulator identification. It is useful infrastructure to monitor if a credible
finite market/execution model becomes available. This refresh inspected the
primary abstract and metadata, not a full code/license audit or benchmark
reproduction. No independent crypto replication was established. Adding Tempest
now would not cure the missing transition probabilities, logged-action coverage,
venue precision or intrabar risk in our historical replay. Disposition: monitor.

[Choudhury et al., IJCAI 2026](https://www.ijcai.org/proceedings/2026/470) study
persistent safe sets learned through a generalized Bellman operator and control
barrier functions, including dynamics uncertainty. The proceedings summary reports
benchmark safety improvements alongside returns. Its important question is whether
learning can avoid propagating values through unsafe regions in offline data.
The summary alone does not establish a certified safe set for our observations,
a validated uncertainty set, or deterministic limits during price gaps. Exact
benchmark periods, splits, code license and independent replications were not
verified in this refresh. The deployment decision remains monitor; a learned
barrier may supplement, but cannot replace, the external Haskell boundary.

[Reluplex, Katz et al., CAV 2017](https://theory.stanford.edu/~barrett/pubs/KBD%2B17-abstract.html)
verifies properties of piecewise-linear neural networks, evaluated on ACAS Xu.
The relevant output is a certificate/counterexample for a specified input region,
not a global guarantee of learned policy quality. The current research networks
use tanh, and no policy survived the economic gates. Introducing Reluplex, Marabou,
MILP or alpha-beta-CROWN would require supported activation semantics, bounded
normalization inputs, numerical correspondence and an economically justified
candidate. Neural output bounds do not prove observation causality, terminal
accounting or permission isolation. No neural region is certified by this work.

## Tool selection and proof scope

[Lamport's Specifying Systems](https://www.microsoft.com/en-us/research/publication/specifying-systems-the-tla-language-and-tools-for-hardware-and-software-engineers/)
provides a state/temporal specification framework useful for starts, draining,
ownership and reconciliation. TLA+/TLC or Apalache would be appropriate for a
future production model with shared resources and explicit refinement. The present
75-state call model fits a complete explicit-state search. It has two callers,
atomic modeled steps and conditional completion fairness. Its fixed point covers
its finite model, not all production schedules. The counterexample about retained
immutable proposals illustrates why a precise state definition matters more than
choosing a fashionable tool. Alloy can help check relational ownership constraints;
it is not an automatic substitute for temporal shutdown reasoning.

[PRISM 4.0](https://www.prismmodelchecker.org/bibitem.php?key=KNP11) and
[Storm](https://www.stormchecker.org/2017/04/12/cav.html) support probabilistic system
verification. They suit finite MDP/POMDP abstractions with justified transition
models and explicit probability/reachability questions. This repository lacks a
calibrated causal model for queue fills, impact and adverse market transitions.
Selecting transition probabilities merely to obtain a reassuring bound would
create false evidence. Probabilistic verification is therefore not performed;
this omission is explicit, not reported as a passing stochastic-risk guarantee.
Neither tool can establish probabilities of future losses without valid modeling
assumptions. No financial dataset or reproduction is imported from either tool
paper.

[Refinement Types for Haskell](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/LiquidHaskell_ICFP14.pdf)
and [SBV's official implementation](https://github.com/LeventErkok/sbv) offer ways to
bring proof predicates closer to Haskell source. Their value here is reducing a
manual translation gap, particularly for exact bounded values and action types.
The current gate can instead be represented directly using binary64 SMT predicates
and checked against compiled source. This is weaker than universal source
refinement and is labeled accordingly. No Liquid Haskell or SBV proof is claimed.
A future shared symbolic/concrete core should be separately reviewed against GHC
pins and dependency impact; private constructors alone do not verify supplied
Boolean evidence or the entire call graph.

[SMT-LIB's floating-point theory](https://smt-lib.org/theories-FloatingPoint.shtml)
is relevant because a real-number inequality does not represent NaN, infinities,
subnormals or signed zero. Z3 4.15.4 is pinned for the current check, with full
binary64 comparison domains and separate exact-integer/real lemmas. Every negated
formula must be UNSAT; timeout and UNKNOWN fail. The solver, hand encoding,
compiler and host are trusted assumptions. [Lean's proof-validation documentation](https://lean-lang.org/doc/reference/latest/ValidatingProofs/)
explains why kernel-checked proof terms and compiler-dependent checks have different
trust boundaries. Lean/Coq/Isabelle/Agda remain alternatives for deeper algebraic
and refinement proofs; none is claimed in the delivered result. Runtime contracts,
model conformance and property tests supplement these techniques, not replace them.

## Recent prediction and offline-RL screening

[FinVerse v2](https://arxiv.org/abs/2608.03259v2), revised August 20, evaluates
financial series with economically differentiated metrics and compares 43 public
forecasting foundation models. The abstract reports that generic forecast rankings
do not reliably imply useful financial forecasts. It reinforces the need for
net-economic evaluation, but does not supply our matched crypto champion,
perpetual costs or untouched holdout. This is an abstract/metadata screen; no
independent replication or full data-license review was completed. Monitor the
benchmark; do not integrate a model based on its rankings.

[Boundary-Aware Data Augmentation](https://arxiv.org/abs/2609.20300) proposes
neighbor-constrained interpolation for offline RL and reports benchmark robustness.
The primary record was available when checked, with a submission date of August
22 despite its September identifier; the identifier alone is not a publication
date. Synthesized transitions do not create identified market counterfactuals or
restore missing behavior support. This abstract-level screen does not verify the
paper's error-bound assumptions, financial effectiveness or independent replication.
No augmentation is added to the frozen replay buffer. Disposition: monitor.

The existing negative [financial-return foundation-model benchmark](https://arxiv.org/abs/2606.27100)
was rechecked: its primary summary reports small, sparse gains over a random-walk
benchmark. This supports skepticism about universal alpha claims, not a theorem
that all market prediction is impossible. Latest search also surfaced
[EvolveTrade](https://arxiv.org/abs/2609.17632) and
[context-augmented alternative-data forecasting](https://arxiv.org/abs/2609.11607).
They remain screening leads, not detailed reviews or accepted evidence. Their
existence does not justify self-modifying production policies or bypassing
first-seen/vintage rules. No scorecard or trial is changed on that basis.

## Decision and evidence scorecard

Keep the four previously shortlisted families: HAR risk estimation, calibrated
shallow/missingness-aware models, OFI, and sequential control. Do not create a fifth
family simply to implement a new paper. The existing 20-dimension scorecard remains
the academic/reproducibility/economic assessment; append two verification criteria,
scored 0–4, without retroactively changing the frozen candidate ranking:

| Family | Formal specifiability | Verification feasibility | Binding condition |
|---|---:|---:|---|
| HAR risk gate | 4 | 3 | Exact artifact and numeric contracts feasible; empirical risk improvement and fresh inputs absent |
| Calibrated shallow model | 4 | 3 | Availability/schema/splits verifiable; conditional market coverage remains empirical |
| OFI | 3 | 2 | Event clocks and execution state need licensed causal evidence |
| Sequential RL | 3 | 2 | Wrapper analyzable; simulator dynamics, neural policy and production refinement remain open |

Scores are research judgments, not probabilities or statistical evidence. Modern
PPO, Double DQN and CQL already have the registered three-seed comparison; the
risk-penalty ablation is retained. Continuous SAC/TD3, distributional learning,
Decision Transformer, world models and multi-agent methods retain their documented
exclusions. No modern algorithm fixes the failed paths merely through its name.

Recommendation: **no adoption**. Retain rejected policies only as research evidence.
Continue offline work only through a separately justified protocol; prefer the
existing deterministic/supervised complexity baselines until material incremental
value is demonstrated. None of the reviewed sources proves future profitability,
stationarity, generalization to new regimes, or production authorization.

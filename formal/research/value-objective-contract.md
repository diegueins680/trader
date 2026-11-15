# Value-based learner fidelity audit — specification before verification

Version `value-objective-audit-v1`, 2026-09-30. Scope: unchanged
`sequential_learning.py:train_q` target-expression slice and `bellman_gradient`.
Classification: functional/numerical correctness and model fidelity. No new
candidate, learner repair, model identifier, training or market experiment.

## Canonical interpretation

Double DQN selects the first maximal action using the online network and evaluates
that action using the target network. For three actions, let k be this index,
let d be the terminal Boolean and g the represented value of `.99 ** horizon`
for horizon in {1,3,6}. The exact-real target is r + g*(1-d)*Q_target[k].
The source computes next-network outputs even when terminal; this audit does not
prove their execution unnecessary or safe, or verify the full training loop.

For batch size n in 1..256, selected action a and fixed target y, let e=q[a]-y,
p=softmax(q), L=log(sum(exp(q))). The intended per-row loss contribution is
`e^2/(2*n) + alpha*(L-q[a])/n`, with alpha nonnegative. The gradient component is
`indicator(j=a)*e/n + alpha*(p[j]-indicator(j=a))/n`.
The conservative term sums to zero for an exact probability simplex, has each
component in [-alpha/n,alpha/n], and disappears algebraically at alpha=0.
These are conditional algebra claims, not a proof of a conservative value bound,
softmax/log calculus, NumPy exponential accuracy or learning convergence.
The repository uses a scalar fixed-alpha CQL(H)-inspired penalty on Double DQN;
it does not reproduce the original Atari QR-DQN experimental implementation.

## Obligations and source binding

- F-RL-DOUBLE-TARGET: check the contiguous greedy/target source slice and translate
  the target expression; prove first-argmax tie semantics, online selection versus
  target evaluation and real terminal independence. Three actions, horizons
  1/3/6, arbitrary real reward/network values. NumPy argmax correspondence is an
  explicit assumption checked by bounded conformance, not runtime refinement.
- F-RL-CQL-GRADIENT: audit the full loss/gradient function skeleton and derive its
  residual and gradient arithmetic. Prove the specified component formula,
  conservative-component bounds/zero sum and total sum e/n under an exact simplex.
  Treat softmax/LSE primitives and derivative identities as explicit assumptions.
- F-RL-Q-FINITE: audit whether finite q and target with alpha=0 guarantee finite
  returned loss. Prescribed probe: q=[maxFloat,maxFloat,-maxFloat], action 2,
  target=-maxFloat. The selected residual is zero, but an eagerly evaluated
  conservative penalty may overflow before zero multiplication. Require a SAT
  binary64 witness conditional on the stated transcendental intermediate plus
  actual NumPy reproduction; do not infer optimizer refusal if gradient is finite.
- F-RL-CQL-SHIFT: audit exact binary64 invariance of the conservative loss under a
  common q/target shift. Prescribed probe: all q and y initially zero, then shifted
  by 2^54, alpha=.1. In real arithmetic the regularizer remains log(3). Check actual
  source and an RNE witness with log(3) supplied in the explicit interval (1,2).
  That interval is an analytic assumption, not a machine-proved logarithm theorem.

No claim that either prescribed witness is reachable from registered initialization
and optimizer updates, or occurred in historical trials. The probe domain is the
helper's finite-input domain. Preserve all witnesses and unexpected outcomes.
A failed proposed counterexample must be recorded, not replaced by a search.

## Verification and conformance

Use existing pinned Z3, separate SAT-premise/UNSAT-violation solvers and unchanged
10,000 ms limits. Use separate IEEE binary64 RNE operations for prescribed numeric
witnesses; no fused arithmetic/extended precision. Source AST translators, Python,
NumPy, compiler, solver and stable scalar/vector correspondence are trusted.
Ordinary aligned float64 arrays, integer in-range actions and immutable bindings
are required. Real simplex claims do not imply rounded probabilities sum exactly
in real arithmetic. No proof of full learner, sampling support or paper theorems.

Tests: source mutants for selection/evaluation network, mask, discount, residual,
gradient sign/scaling, alpha and return; real target conformance over the complete
three-action {-1,0,1}^3 online/target grid, both terminal flags, rewards {-1,0,1}
and all three horizons; selected-action/alpha/batch gradient cases and ordinary
finite differences; actual numeric witnesses and downstream optimizer behavior.
Test only synthetic deterministic arrays; do not train a policy or open data.
A positive result must retain every assumption and require no proof placeholder.

Existing canonical safety and evidence gates prevail. Resolve any apparent claim
that alpha=0 disables evaluation: the code eagerly computes the penalty, so only
an exact-real algebraic omission is justified. Do not silently change it. Keep
all frozen artifacts and model semantics unchanged; a correction needs explicit
versioning and a separately registered question.

References reviewed 2026-09-30: van Hasselt et al., AAAI 2016,
https://arxiv.org/html/1509.06461 ; Kumar et al., NeurIPS 2020, equation 4 and
Appendix F, https://arxiv.org/pdf/2006.04779 . No PDF or external code committed.

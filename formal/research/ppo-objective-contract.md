# PPO objective and numerical boundary — specification before verification

Version `ppo-objective-audit-v1`, 2026-09-28. Scope: unchanged
`scripts/research/sequential_learning.py:ppo_gradient`. Classification: numerical
and functional correctness, model fidelity, failure behavior. This is an audit,
not a learner repair, algorithm selection, trained model or financial experiment.

## Canonical interpretation and existing evidence

Schulman et al. (2017), equation 7, uses the minimum of unclipped and clipped
advantage-weighted ratios. The repository minimizes its negative batch mean and
selects the unclipped derivative at clipping boundaries. Existing finite-difference
tests cover a small ordinary two-row fixture; they do not prove all branch cases,
float safety, a hard trust region or financial efficacy. The training registration
fixes epsilon=.2, no entropy term and a separate critic. Preserve these semantics.

Let r=p/b, with selected new-action probability p in [0,1], old probability b in
(0,1], advantage a real and batch size integer n in [1,256]. Let L and U be the
exact rational values represented by native binary64 literals .8 and 1.2, not
silently exact decimals. The exact-arithmetic source model is:

    clip(r) = min(max(r,L),U)
    loss contribution = -min(r*a, clip(r)*a)/n
    active = (a>=0 AND r<=U) OR (a<0 AND r>=L)
    coefficient = (if active then 1 else 0)*a*r/n
    logit component j = (p_j - indicator(j=selected))*coefficient

For a>=0 the loss is -a*min(r,U)/n; for a<0 it is -a*max(r,L)/n.
The coefficient is zero on the inactive improving side, not everywhere outside
[L,U]. At the kink the implementation chooses the unclipped side. This is a
permitted one-sided/subgradient convention, not a unique classical derivative.
The softmax derivative identity motivates the logit factor; its analytic calculus
and NumPy exponential implementation are assumptions, not machine-proved here.

## Scoped obligations

- F-RL-PPO-OBJECTIVE: derive the ratio and loss expressions from the audited source;
  prove their exact-real equivalence to the defined piecewise surrogate with
  source-represented L/U. Do not infer binary64 evaluation exactness.
- F-RL-PPO-COEFFICIENT: derive the active mask and multiplier; prove the registered
  one-sided branch formula, nonnegative bounded coefficient for nonnegative a,
  nonpositive coefficient for negative a, and zero sum of three logit components
  conditional on an exact-real probability simplex. No KL, update-size, neural
  parameter or market-risk bound follows.
- F-RL-PPO-FINITE: audit the stronger claim that finite logits/advantages and
  positive finite old probabilities guarantee finite gradient outputs. Proposed
  synthetic probe: zero logits, a=1, b=smallest positive binary64 subnormal.
  Require a prescribed SMT witness and actual NumPy fixture before refuting it.
- F-RL-PPO-UNIFORM-CLIP: audit the claim that clipping universally bounds the
  coefficient magnitude by U*abs(a)/n. Proposed probe: p=1/3, b=1/24, a=-1,
  n=1 (r=8). A refutation is a limit of the clipping mechanism, not a deviation
  from the original paper or evidence that the independent action shield failed.

## Source binding, assumptions and conformance

Audit the complete function AST including batch mean, action indexing, one-hot
subtraction, multiplication order and return. Translate only supported scalar
arithmetic, comparisons, Boolean masks, clip and minimum. Reject drift outside
explicitly extracted expression slots. Use exact rationals for real literal
values and separate RNE binary64 operations for numerical witnesses.

A-PPO-OBJECTIVE: correctly shaped base float64 arrays, integer in-range actions,
positive old probabilities, immutable row alignment and stable NumPy/Python
bindings; trusted softmax/simplex and vector-to-scalar correspondence, source
translator, primitive/compiler/solver semantics. Real simplex/derivative claims
are conditional; rounded softmax probabilities need not sum to one in exact real
arithmetic. No runtime, compiler, full-network, optimizer or learning-convergence
refinement is claimed. No prevalence in historical trials is inferred.

Positive obligations require SAT premises and UNSAT violations with pinned Z3 and
10-second query limits. Numerical refutations require prescribed SAT witnesses,
exact-hex fixtures and actual current-source reproduction. Check malformed fixture
rejection and source mutants (wrong loss sign, clip, ratio, branch mask, gradient
sign, scaling or return). Test boundaries, ordinary finite differences, a small
exact-rational grid and downstream optimizer refusal without state mutation when
gradients are non-finite. Tests supplement the scoped proofs.

Keep the current learner, normalization, artifacts, champion and economic evidence
unchanged. Any corrective training implementation needs a versioned successor and
separate registration. No market data or sealed outcomes may be accessed. Link
all claims, assumptions, artifacts, code, tests and CI in the canonical registry,
proof ledger and risk register. The broader 38 obligations remain blockers.

Primary reference: https://arxiv.org/abs/1707.06347 (equation 7). Skeptical context:
https://arxiv.org/abs/2005.12729 and https://arxiv.org/abs/1811.02553 (current title
A Closer Look at Deep Policy Gradients). Sources reviewed 2026-09-28; the latter
two overlap in authors/content and are not independent replication evidence.

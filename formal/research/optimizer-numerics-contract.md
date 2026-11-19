# Optimizer numerics engineering contract

Scope recorded before implementation, 2026-11-16; base
0464054a (causal-ope-v9 merge). Engineering registration:
`research-notes/registrations/optimizer-numerics-engineering.json`.

## Scope and interpretation

A proof-and-test audit of the arithmetic in
`scripts/research/causal_optimizer_v5.py` (network forward and Adam update).
No runtime code changes: the optimizer, both successor learners and every
frozen path are byte-for-byte unchanged. The question, from obligation 10, is
whether the optimizer's finiteness guards can be replaced by proved bounds or
whether the guards suffice for "bounded or detected before publication".

The answer is mixed and stated per site.

## Site classification

All 36 arithmetic expressions of the module are classified:

- **Matrix products (5).** `x @ w1`, `hidden @ w2`, `residuals @ p[2].T`,
  `inputs.T @ dh`, `hidden.T @ residuals`. No floating-point flag is relied on.
  *Overflow is detected:* in binary64, once an intermediate is infinite or NaN,
  every later sum, product or fused multiply-add is infinite or NaN in any
  evaluation order, and each product reaches an explicit `isfinite` check before
  use or publication (the hidden-layer gradient through its column sum). The
  check must reject NaN as well as infinity because infinities can cancel.
  *Underflow is bounded:* each product is within half a subnormal step of
  exact, so an entry of at most 256 terms is within 2^-1067 absolutely of the
  value the standard relative model gives.
- **Element-wise NumPy operations and reductions (19).** They run inside
  `np.errstate(all="raise")`: overflow, underflow and invalid operations raise
  and the entry returns `None`. *Detected.*
- **Binary64 scalar expressions (5).** `1 - 0.9**step`, `1 - 0.999**step` and
  `1 / np.sqrt(9)` cannot overflow. `0.9**step` underflows to zero for large
  steps, which is the correct limit of the bias correction. *Bounded.*
- **Exact expressions (7).** Integer, shape and type expressions.

So the guards suffice for overflow everywhere, and underflow is either raised
or bounded by 2^-1067.

## Bounds along learner chains

The successor learners start every optimizer chain with `create_v5` and
advance it only with `update_v5` at the default learning rate 0.0003, at most
16384 times per network. Along such a chain, under the standard rounding model:

- every clipped gradient entry has magnitude at most 1 + 2^-44;
- the first moment stays within 1 + 2^-42 and the second within [0, 1 + 2^-40];
- m^2 <= 54 v holds for every entry (a Cauchy-Schwarz bound over the history,
  proved as a one-step inductive invariant);
- the computed Adam step `delta` is therefore at most 7.5 times the learning
  rate, for any learning rate in (0, 1];
- the published parameter `fl(old - delta)` moves by at most
  7.5 lr (1 + 2^-53) + 2^-53 |old|: the subtraction rounds once. For a
  learning rate far below the spacing of binary64 near `old` that rounding
  dominates; at lr = 2.5e-17 a step of about 4.95 lr moves 1.0 by 2^-52,
  8.9 times the learning rate;
- with initial parameters of magnitude at most 8, every parameter stays below
  2^6 for the whole run, and every network output (Q value, logit or state
  value) below 2^11.

The learners' 2^99 guards on consumed Q values and state values therefore never
fire on a learner-built network. The hidden-layer pre-activation and the
gradients still depend on the magnitude of observations and objective
residuals, which this audit does not bound; there the detection argument
applies.

## Requirements / methods

F-RL-OPTNUM-V5-SOURCE: complete AST hash of the optimizer; every binary
operation classified and the classification table equal to the source's set;
the five matrix products each followed by the reviewed finiteness check; every
entry's arithmetic inside `errstate(all="raise")` and a guarded `try`; finite,
admitted publication; hash-pinned learners that call `create_v5` and
`update_v5` only, with no explicit learning rate and at most 16384 updates.

F-RL-OPTNUM-V5-ARITH (SMT): seven binary64 floating-point implications for
sticky non-finiteness with one satisfiable control (infinities cancel to NaN);
quantifier-free real implications for clipping, both moment bounds, the
m^2 <= 54 v invariant, the per-update step bound, the movement bound including
subtraction rounding with a satisfiable control showing that rounding can
exceed the step bound, a rounded accumulation step
and the subnormal-grid error; exact rational checks of the bias-correction
ordering 1 - 0.999^t <= (1 - 0.9^t)^2 for t = 1..16384 on the binary64 values
the source computes, and of the run bounds.

F-RL-OPTNUM-V5-CONFORMANCE: twelve crafted range failures publish nothing;
384 matrix-product entries with underflowing products stay within the bound of
the exact rational value; 1440 published updates over twelve chains (learning
rates 0.0003, 1 and 2.5e-17) keep every invariant and the movement bound; a
crafted update at learning rate 2.5e-17 moves every parameter by 2^-52, beyond
the step bound and within the movement bound; and every output within 17 times the largest
output-layer weight; initial parameters over 512 seeds and twelve final learner
snapshots satisfy the run bounds. Tests are not proofs.

## Assumptions and limits

A-OPTIMIZER-NUMERICS trusts pinned CPython 3.13.3 and NumPy 2.3.5 on IEEE
binary64 round-to-nearest hardware with gradual underflow (no flush-to-zero);
that `np.errstate(all="raise")` raises for the element-wise operations and
reductions used; that a matrix product computes each entry as some
arrangement of binary64 sums, products and fused multiply-adds of its terms;
that `np.tanh` returns values in [-1, 1]; the standard rounding model
(A-FP-ROUNDING); and that `create_v5` parameters are at most 8 in magnitude,
which is tested on 512 seeds and not proved for NumPy's normal sampler. None of
these is machine-proved for the NumPy, BLAS or libm build.

Bounded is not accurate: nothing here bounds gradient error, convergence or
generalization. A snapshot not built by `create_v5` and `update_v5` (the
optimizer admits any finite one) has only the detection guarantee. Obligation
10 stays open until the owner decides, under a new preregistration, whether the
frozen paths are retired.

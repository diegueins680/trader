# PPO objective v2

Engineering preregistration, 2026-11-11; base 75dabd831f48d5c0d2eb1e905f349a1baf8a0dbb.

## Scope and interpretation

A disabled-by-default successor for the frozen clipped-PPO loss and logit
gradient helper `sequential_learning.ppo_gradient`. It addresses the preserved
counterexample **CE-RL-012** (finite inputs with a subnormal old probability
give a finite loss and an all-NaN gradient) in the new kernel only.
**CE-RL-013** (a coefficient of -8, refuting a "uniformly clipped gradient"
claim) is the intended behavior of the clipped objective and is reproduced,
not changed. The frozen helper, `train_ppo`, causal-ppo-v5, F-RL-PPO-FINITE and
F-RL-PPO-UNIFORM-CLIP (both refuted) and all reported PPO results are
unchanged. No learner, training run, artifact, data, policy, promotion or order
interface is added.

## Input domain

- `logits`: tuple of 1..256 rows; each row is a tuple of exactly 3 Python
  `float`s, finite, magnitude at most 2^100.
- `actions`: tuple of the same length of `int` in {0, 1, 2} (bool rejected).
- `old_prob`: tuple of the same length of Python `float`s with
  2^-100 <= p <= 1.
- `advantage`: tuple of the same length of Python `float`s, finite, magnitude
  at most 2^100.

Values outside this domain reject before arithmetic. The bounds are a numeric
admission contract, not a claim about learned scales. CE-RL-012's subnormal old
probability is outside the domain and rejects.

## Semantics (binary64, round-to-nearest-even, scalar operations)

For row i with selected action a, m = max_j x[j], t[j] = exp(x[j] - m),
p[j] = t[j] / fsum(t), ratio r = p[a] / old, clipped c = min(max(r, 0.8), 1.2)
and advantage A. With n rows:

- loss = -fsum(min(r·A, c·A)) / n.
- active = (A >= 0 and r <= 1.2) or (A < 0 and r >= 0.8).
- coefficient k = (A·r) / n when active, and exactly 0.0 otherwise. The inactive
  branch performs no multiplication, so no `0 · inf` can arise.
- gradient[i][j] = (p[j] - [j = a]) · k.

Softmax uses differences from the row maximum only. Sums use `math.fsum` and
there is no NumPy/BLAS reduction. Each `exp` term below the smallest normal
magnitude, and each product of nonzero operands whose magnitude falls below it,
is counted in `underflow`, which is published. Published floats are
canonicalized with `x + 0.0`. A final check rejects any non-finite loss or
gradient component. The output is an immutable `Objective` value or `None`.

## Obligations / methods

F-RL-PPOOBJ-V2-SOURCE: complete reviewed AST lock; activation first; scalar-only
import boundary; difference-only softmax; branch-selected coefficient;
fsum reductions; zero canonicalization; final finiteness guard; no NumPy
import, file, policy or order effect.

F-RL-PPOOBJ-V2-ARITH (SMT, quantifier-free, with an exact rational rounding
chain under A-FP-ROUNDING and the standard model |fl(x) - x| <= u|x|,
u = 2^-53): the row maximum makes every exponent argument nonpositive with one
zero, so 1 <= total <= 3 and 0 <= p <= 1; with old >= 2^-100 the ratio is at
most 2^100(1+u); the clipped ratio lies in [0.8, 1.2]; the surrogate term is
bounded by the unclipped product; an inactive row has coefficient exactly zero;
every loss and gradient magnitude stays below 2^220, far below the largest
finite binary64. So the finiteness guard never rejects an admitted input.

F-RL-PPOOBJ-V2-FLOW: finite publication model over activation, admission,
compute (active and inactive rows) and guard; publication only of finite
results; any failure is terminal and publishes nothing.

F-RL-PPOOBJ-V2-CONFORMANCE: 96 seeded batches agree with an independent
60-digit `decimal` oracle within relative tolerance 1e-12, with identical
branch decisions and underflow counts. 96 ordinary batches agree with the
frozen helper within the same tolerance. The CE-RL-012 witness rejects and its
in-domain analogue gives a finite loss and an exactly zero gradient. The
CE-RL-013 witness reproduces coefficient -8. Gradients match central finite
differences of the oracle loss away from clip boundaries. Tests are not proofs.

A-PPO-OBJECTIVE-V2 trusts pinned CPython float/math (exp, fsum) semantics on
IEEE binary64 RNE hardware, the primitive bound 0 <= exp(x) <= 1 for x <= 0,
Z3 and the reviewed AST translation. The rounding model is the standard one and
is not machine-proved for the CPython build. No learner composition,
convergence, policy-improvement guarantee, data support or economic claim
follows. The 38 original criteria and scopes are unchanged; obligation 10 does
not close on this kernel.

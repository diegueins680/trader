# Value objective v2

Engineering preregistration, 2026-10-06; base 3fb6fdf1f7699251562f341434ffcbe3d01b1dd8.

## Scope and interpretation

A disabled-by-default successor for the frozen conservative Double-DQN loss and
gradient helper `sequential_learning.bellman_gradient`. It addresses preserved
counterexamples **CE-RL-014** (finite inputs, alpha = 0, NaN loss) and
**CE-RL-015** (binary64 loss not invariant under an exact common shift) in the
new kernel only. The frozen helper, `train_q`, F-RL-Q-FINITE and F-RL-CQL-SHIFT
(both refuted), and all reported value-based results are unchanged. No learner,
training run, artifact, data, policy, promotion or order interface is added.

## Input domain

- `q`: tuple of 1..256 rows; each row is a tuple of exactly 3 Python `float`s.
- `actions`: tuple of the same length of `int` in {0, 1, 2} (bool rejected).
- `targets`: tuple of the same length of Python `float`s.
- `alpha`: Python `float` with 0 <= alpha <= 1.
- Every q and target value is finite with magnitude at most 2^100.

Values outside this domain reject before arithmetic. The bound is a numeric
admission contract, not a claim about learned value scales. CE-RL-014's
`±max finite` inputs are outside it and reject.

## Semantics (binary64, round-to-nearest-even, scalar operations)

For row i with selected action a, residual e = q[a] − y. With n rows:

- loss = fsum(e_i²) / (2n) + [alpha > 0] · alpha · fsum(c_i) / n, where
  c_i = (m_i − q_i[a]) + log(fsum_j exp(q_i[j] − m_i)) and m_i = max_j q_i[j].
- gradient[i][j] = [j = a] · e_i / n + [alpha > 0] · alpha · (p_ij − [j = a]) / n,
  where p_ij = exp(q_i[j] − m_i) / fsum_k exp(q_i[k] − m_i).

When alpha = 0 the conservative penalty is **not evaluated**. The penalty uses
only the differences q[j] − m and m − q[a], so it never forms the large value m
and then subtracts it. Sums use `math.fsum` (one correct rounding) and there is
no NumPy/BLAS reduction. Each `exp` term that returns zero or a subnormal is
counted in `underflow`. The value is published, so underflow is reported rather
than silent. A final check rejects any non-finite loss or gradient component.
The output is an immutable `Objective` value or `None`.

## Obligations / methods

F-RL-VALUE-V2-SOURCE: complete reviewed AST lock; activation first; alpha = 0
branch skips the penalty; penalty and softmax consume only differences; fsum
reductions; final finiteness guard; no NumPy import, float conversion of external
arrays, file, policy or order effect.

F-RL-VALUE-V2-ARITH (SMT): (a) Z3 IEEE binary64: whenever a + s and b + s are
exact (no rounding), fl(fl(a + s) − fl(b + s)) = fl(a − b), so every difference
the kernel consumes, and therefore loss and gradient, is bitwise invariant under
an exact common shift. (b) Real arithmetic with the standard rounding model
|fl(x) − x| <= u|x| (u = 2^-53) and explicit primitive bounds 0 <= exp(x) <= 1
for x <= 0 and 0 <= log(s) <= 2 for 1 <= s <= 3. On the admitted domain every
residual, square, penalty, mean and gradient magnitude stays below 2^220, far
below the largest finite binary64. So the finiteness guard never rejects an
admitted input, and alpha = 0 cannot produce NaN.

F-RL-VALUE-V2-FLOW: finite publication model over activation, admission,
compute (with and without penalty) and guard; publication only of finite results;
any failure is terminal and publishes nothing.

F-RL-VALUE-V2-CONFORMANCE: 96 seeded batches agree with an independent 60-digit
`decimal` oracle (correctly rounded exp/ln) within relative tolerance 1e-12.
96 ordinary batches agree with the frozen helper within the same tolerance. The
CE-RL-014 witness rejects, and its in-domain analogue is finite. The CE-RL-015
witness gives bitwise equal shifted and unshifted results. Gradients match
central finite differences of the oracle loss. Tests are not proofs.

A-VALUE-OBJECTIVE-V2 trusts pinned CPython float/math (exp, log, fsum)
semantics on IEEE binary64 RNE hardware, the stated primitive bounds, Z3 and the
reviewed AST translation. The rounding model is the standard one and is not
machine-proved for the CPython build. No learner composition, convergence,
conservative-value guarantee, data support or economic claim follows. The 38
original criteria and scopes are unchanged; obligation 10 does not close on this
kernel.

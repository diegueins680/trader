# Exact ESS diagnostic v2 — specification before implementation

Version `ess-rational-v2`, 2026-09-30. Classification: numerical and functional
correctness, configuration safety, isolation, atomic publication and termination.
This is a bounded offline diagnostic, not an OPE estimator or a financial trial.
It addresses CE-RL-018 in a separate kernel; the frozen estimator and its witness
remain unchanged. No existing identifier, artifact or result acquires new semantics.

## Domain and output

`effective_sample_size_v2(weights, *, enabled=False, version="ess-rational-v2")`
accepts only exact native `True`, the exact native version string, and a native
immutable tuple of 1..256 native finite nonnegative Python floats. Reject all other
inputs as `None` before conversion or arithmetic. Signed zero is admitted as zero.
No implicit coercion, array, iterator, integer-as-weight or subclass is accepted.

Convert each weight exactly to a rational using `Fraction.from_float`. Starting
from S=Q=0, append each weight w using S'=S+w, Q'=Q+w*w. Return exact `Fraction(0)`
if Q=0; otherwise return S*S/Q as a Fraction. Do not convert back to binary64 or
round/clamp the result. A zero result indicates zero supplied weight mass; it is
not an actionable or statistically reliable signal. No parameter, policy, order,
permission, I/O, callback, persisted state or global mutable accumulator exists.
All computation is local; publish only after complete admission and accumulation.

For k admitted weights, invariant I(k,S,Q) is:

    k >= 0; S >= 0; Q >= 0; Q <= S*S <= k*Q.

Prove the zero base case and preservation by appending any nonnegative real w.
The source accumulator has the same recurrence under trusted exact conversion and
rational arithmetic. For Q>0, the invariant implies 1 <= S*S/Q <= k. For Q=0,
S=0 and the defined result is zero. These are algebraic facts about the supplied
weights, not a claim that ESS equals an actual number of independent observations.

## Requirements and verification

- F-RL-ESS-V2-ACCUMULATE: source-derived exact-real recurrence preserves I from
  the zero base state; named inductive obligations cover all k>=0 (bounded runtime
  accepts no more than 256 rows).
- F-RL-ESS-V2-BOUNDS: source return expression is zero for zero mass and in [1,k]
  for positive mass under I; no rational overflow/underflow under exact primitives.
- F-RL-ESS-V2-PUBLISH: source-bound finite transition model admits only after
  controls, shape and all elements validate; publishes no partial result; a
  disable/rejection has no stored residual state or order-authorizing transition.
  All 1..256 sizes exhaustively explored. Termination assumes primitives return
  and normal resource availability; rank decreases until absent or returned.

Match the complete module AST except translated accumulator/return expressions.
Independently check SAT premises and UNSAT violations with Z3 4.15.4, seed 0 and
10,000 ms per query. Unknown/timeout is failure. Do not weaken invariants or expand
solver budgets after seeing a failure; preserve any failed proof attempt.
Model transition assumptions and source-to-model conformance must be explicit.
Property/differential tests connect the kernel to the abstract recurrence; they
are not full Python/Fraction implementation proofs.

A-ESS-V2: trusted Python 3.13.3 native-type checks, `isfinite`, float-to-Fraction
conversion, exact integer/rational arithmetic, AST translator and pinned solver;
stable bindings, no monkey patching, and normal resources. No proof of interpreter,
big-integer implementation, physical runtime, allocation failure recovery or
statistical reliability. No neural or market claim. The kernel cannot recover
weights already rounded to zero upstream, or validate behavior-policy support.

## Conformance and engineering gates

Preserve CE-RL-018 and require v2 to return exactly 2 for its two supplied weights.
Check subnormal 2^-1074, maximum finite binary64, zero and mixed scales. Compare a
fixed grid and seeded cases against an independent common-denominator integer
reference. Check permutation, exact power-of-two scaling, deterministic replay,
all invalid element positions, default/version rejection and no input mutation.
Bind module isolation: no import/call from existing research or production paths;
only explicit tests/benchmarks/proofs reference the version. No training adapter.

CPU diagnostic budget: 256 weights; measured maximum <=500 ms per call on the
available machine, with 25 repetitions for each registered benchmark case. This
is an engineering screen, not a proved deadline or an inference service budget.
No dependencies added; Fraction is Python standard library (PSF license).
A future estimator/financial integration requires a separate registration and
must keep all statistical, holdout and promotion gates intact.

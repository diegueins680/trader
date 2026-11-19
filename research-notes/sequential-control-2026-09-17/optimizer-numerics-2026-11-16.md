# Optimizer numerics — engineering increment, 2026-11-16

Engineering evidence only: no market data was read, no financial trial was run
and no holdout was opened. No runtime code changed.

## Question

Obligation 10 asked whether the finiteness guards in network forward and
optimizer-v5 arithmetic can be replaced by proved bounds, or why guards
suffice.

## Findings

1. **Guards suffice for overflow.** The five matrix products rely on no
   floating-point flag: an overflowed intermediate stays infinite or NaN until
   an explicit finiteness check. Every other operation runs under a raising
   floating-point state.
2. **Underflow is raised or bounded.** Element-wise underflow raises and the
   update is refused. Silent underflow inside a matrix product changes an entry
   by at most 2^-1067.
3. **Learner-built networks are bounded, not just guarded.** Along a chain
   from `create_v5` advanced by `update_v5`, m^2 <= 54 v is invariant, so the
   computed Adam step is at most 7.5 times the learning rate. The published
   change adds one rounding of `old - delta`, at most 2^-53 of the parameter;
   for learning rates far below 2^-52 that rounding dominates (review
   finding on #396). Over a full
   run parameters stay below 2^6 and outputs below 2^11, so the learners' 2^99
   guards on Q values and state values never fire.
4. **What stays guarded.** The pre-activation and gradients depend on
   observation and residual magnitudes, which are not bounded here.

## Evidence

- Source: 36 classified arithmetic sites (5 matrix products, 19 element-wise,
  5 scalar, 7 exact).
- SMT: 15 implications (7 in binary64 floating point) and two controls.
- Conformance: 12 range rejections, 384 underflow checks, 1440 published
  updates over 12 chains (learning rates 0.0003, 1 and 2.5e-17), one crafted
  subtraction-rounding update (movement 8.9 times a learning rate of
  2.5e-17), 12 learner snapshots. In a local macOS run the largest movement at the
  default rate was 1.15 times the learning rate and the largest output about
  370; the receipt records only that observed values are within the bounds,
  because their last bits depend on the BLAS build.

## What this does not show

The initial-parameter bound is an assumption tested on 512 seeds. NumPy, BLAS
and libm behaviour is assumed, not proved. Nothing here concerns accuracy or
convergence. Obligation 10 stays open: the remaining step is the owner's
decision on retiring the frozen paths. Ledger counts stay 20 scoped closures /
17 partial / 1 open.

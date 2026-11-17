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
   from `create_v5` advanced by `update_v5`, m^2 <= 54 v is invariant, so one
   update moves a parameter by at most 7.5 times the learning rate. Over a full
   run parameters stay below 2^6 and outputs below 2^11, so the learners' 2^99
   guards on Q values and state values never fire.
4. **What stays guarded.** The pre-activation and gradients depend on
   observation and residual magnitudes, which are not bounded here.

## Evidence

- Source: 36 classified arithmetic sites (5 matrix products, 19 element-wise,
  5 scalar, 7 exact).
- SMT: 14 implications (7 in binary64 floating point) and one control.
- Conformance: 12 range rejections, 384 underflow checks, 1080 published
  updates over 9 chains, 12 learner snapshots. Largest observed step is 1.22
  times the learning rate; largest observed output about 300.

## What this does not show

The initial-parameter bound is an assumption tested on 512 seeds. NumPy, BLAS
and libm behaviour is assumed, not proved. Nothing here concerns accuracy or
convergence. Obligation 10 stays open: the remaining step is the owner's
decision on retiring the frozen paths. Ledger counts stay 20 scoped closures /
17 partial / 1 open.

# PPO objective v2 — engineering increment, 2026-11-11

Engineering evidence only: no market data was read, no financial trial was run
and no holdout was opened.

## What was added

`scripts/research/ppo_objective_v2.py` is a default-disabled scalar successor
for the frozen `sequential_learning.ppo_gradient`. It is disconnected: no
learner calls it yet.

## The defect it removes

CE-RL-012: with a subnormal old probability the frozen helper forms an infinite
ratio, then multiplies it by an activity flag of zero, and returns a finite loss
with an all-NaN gradient. The successor differs in two ways:

1. Old probabilities below 2^-100 reject at admission, so the ratio is at most
   2^100(1+u).
2. The coefficient is selected by branch (`plain / n if active else 0.0`), so an
   inactive row is exactly zero and no `0 · inf` product is ever formed. A test
   shows this holds even when admission is bypassed.

CE-RL-013 (coefficient -8) is how a clipped objective behaves when the ratio is
large and the advantage negative. It is reproduced, not changed.

## Evidence

| Requirement | Method | Result |
|---|---|---|
| F-RL-PPOOBJ-V2-SOURCE | complete AST lock and statement review | 13 source mutants reject with a refreshed lock |
| F-RL-PPOOBJ-V2-ARITH | quantifier-free Z3 with an exact rational rounding chain | eight implications unsat; ratio <= 2^100(1+u), loss < 2^201, gradient < 2^202 |
| F-RL-PPOOBJ-V2-FLOW | explicit-state publication model | 6 states, 9 transitions; unguarded-publication mutant rejects |
| F-RL-PPOOBJ-V2-CONFORMANCE | 60-digit decimal oracle, frozen differential, witnesses | 96 + 96 batches within 1e-12; 1640 inactive rows exactly zero; 24 finite-difference checks |

## What this does not show

- No learner uses the kernel. causal-ppo-v5 still imports the frozen helper.
- The kernel is outside the promotion-boundary module set; no closed
  obligation's certificate mapping changed.
- The oracle is compared away from clip boundaries only.
- Obligation 10 stays open; counts stay 20 scoped closures / 17 partial /
  1 open. No convergence, policy-improvement or economic claim follows.

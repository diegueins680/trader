# Causal value v6 — engineering increment, 2026-11-11

Base `b79dfa78cd42ef7e94540c5066aa73e5eb1f7d05`. Engineering evidence only: no
market data was read, no financial trial was run and no holdout was opened.

## What was added

`scripts/research/causal_value_v6.py` is a default-disabled conservative
Double-DQN learner. It composes four existing kernels: causal-replay-v3,
causal-transition-v4, causal-optimizer-v5 and value-objective-v2. It addresses
the learner-composition half of obligation 10's next action for the value-based
family. Causal PPO v5 already composes GAE-v2 with the same replay.

## Evidence

| Requirement | Method | Result |
|---|---|---|
| F-RL-VALUE-V6-SOURCE | complete AST lock, transitive import closure, statement-order and guard review | six local modules, five external imports, no frozen module; 19 source mutants reject with a refreshed lock |
| F-RL-VALUE-V6-ARITH | quantifier-free Z3 with an exact rational rounding chain | six implications unsat, two guard-removal controls sat; target bound 2^100; at most 16384 optimizer steps |
| F-RL-VALUE-V6-FLOW | explicit-state stage model, requests 1..5, rollout bound 2, 2 updates | 292 states, 337 transitions; three mutants reject |
| F-RL-VALUE-V6-CONFORMANCE | registered synthetic runs | 9 deterministic runs, 822 targets equal to the exact-rational oracle bit for bit, 12 prefix rewrites, 7 rejections |

## Finding during conformance

Network outputs differ in the last bits between a single-row evaluation and
the same row inside a stacked batch (BLAS summation order). The oracle
therefore evaluates the same stacked batch as the learner and checks the two
scalar target operations exactly. The contract names this as trusted, not
proved.

## What this does not show

- Network forward and optimizer arithmetic are guarded for finiteness under a
  raising NumPy error state; they are not bounded by proof.
- Causal PPO v5 and optimizer v5 still have regression evidence only.
- Off-policy evaluation is not composed with either successor learner.
- Frozen `train_q`, `bellman_gradient` and CE-RL-010/011/012/014/018/019 are
  unchanged and not retired. Obligation 10 stays open; counts stay 20 scoped
  closures / 17 partial / 1 open.
- No convergence, conservative-value, data-support or economic claim follows.
  The economic-evidence and reproducible-delivery gates stay open.

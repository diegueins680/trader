# Causal PPO v7 — engineering increment, 2026-11-12

Engineering evidence only: no market data was read, no financial trial was run
and no holdout was opened.

## What was added

`scripts/research/causal_ppo_v7.py` is a default-disabled PPO learner composing
causal-replay-v3, causal-transition-v4, causal-optimizer-v5, gae-targets-v2 and
ppo-objective-v2. With causal-value-v6 it gives both learner families a
successor that imports no frozen v1 module.

## Differences from causal PPO v5

| | v5 | v7 |
|---|---|---|
| Loss and gradient | frozen `ppo_gradient` (CE-RL-012) | `ppo-objective-v2` |
| Softmax | frozen NumPy helper | local scalar, differences only |
| Sampling | `rng.choice` | one uniform draw against cumulative probabilities |
| Probability domain | above the smallest normal float | all three actions in [2^-100, 1] |
| Advantage normalization | NumPy mean/std | scalar `fsum`, counted square underflow |
| Evidence | regression tests | source, SMT, model and oracle certificates |

## Evidence

| Requirement | Method | Result |
|---|---|---|
| F-RL-PPO-V7-SOURCE | complete AST lock, transitive import closure, statement-order and guard review | seven local modules, five external imports, no frozen module; 21 source mutants reject with a refreshed lock |
| F-RL-PPO-V7-ARITH | quantifier-free Z3 with an exact rational residual chain | six implications unsat, one guard-removal control sat |
| F-RL-PPO-V7-FLOW | explicit-state stage model, requests 1..5, rollout bound 2, 2 updates | 240 states, 289 transitions; three mutants reject |
| F-RL-PPO-V7-CONFORMANCE | registered synthetic runs | 9 deterministic runs; 822 actions and probabilities reproduced from replayed draws; 822 critic targets equal to the exact-rational GAE recursion bit for bit; 12 prefix rewrites; 8 rejections |

## What this does not show

- The simplex, sampling-partition and normalization lemmas are exact-real.
  The binary64 softmax, the rounded sum p0 + p1 in the sampling rule and the
  binary64 normalization are guarded, not bounded. The sampling law can differ
  from the recorded behavior probability by rounding; that gap matters for
  off-policy evaluation and is not yet bounded.
- Network forward and optimizer arithmetic are finiteness-guarded only.
- Critic targets within a rollout depend on later rows of the same rollout
  through the GAE trace. That is training-time use, not an observation leak;
  the prefix-rewrite check covers transitions, actions and behavior
  probabilities only.
- Off-policy evaluation is not composed with either successor learner.
- causal-ppo-v5, frozen `train_ppo`/`ppo_gradient` and CE-RL-010/011/012/014/
  018/019 are unchanged and not retired. Obligation 10 stays open; counts stay
  20 scoped closures / 17 partial / 1 open.
- No convergence, policy-improvement, data-support or economic claim follows.

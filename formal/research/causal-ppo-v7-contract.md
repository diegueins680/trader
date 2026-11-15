# Causal PPO v7 engineering contract

Scope recorded before implementation, 2026-11-12; base
0383073b79a9705442fa76c0f29e0ce8b0509e31.

## Scope and interpretation

A separate, default-disabled offline learner that composes causal-replay-v3,
causal-transition-v4, causal-optimizer-v5, gae-targets-v2 and ppo-objective-v2
into a clipped-PPO training loop. It is the policy-gradient counterpart of
causal-value-v6. Unlike causal-ppo-v5 it imports no frozen v1 module: its
softmax, sampling and advantage normalization are local scalar code and its
loss and gradient come from ppo-objective-v2 instead of the frozen
`sequential_learning.ppo_gradient`.

causal-ppo-v5, frozen `train_ppo` and `ppo_gradient`, F-RL-PPO-FINITE and
F-RL-PPO-UNIFORM-CLIP (both refuted), every preserved counterexample and all
reported PPO results are unchanged. No training run on market data, artifact,
policy, promotion or order interface is added. Acceptance is engineering
evidence, not a financial experiment.

## Inputs and collection

`train_causal_ppo_v7(initial_bars, outcomes, seed, enabled=True)` requires the
exact version `causal-ppo-v7`, an exact integer seed 0..2^32-2, a replay-v3
initial prefix and a tuple of 1..4096 `Outcome` values, each holding one next
bar, aligned exact funding events, an exact boolean terminal flag and replay-v3
fill/cost parameters. No callback, file, network, artifact or production
selector is accepted.

At each step the learner observes, encodes the nine exact values, evaluates the
current actor and draws the action **before** reading any field of that step's
Outcome. Action probabilities are a scalar softmax over differences to the
largest logit. All three must lie in [2^-100, 1]; otherwise the run rejects, so
every recorded behavior probability is inside the ppo-objective-v2 domain. One
uniform draw u per step selects action 0 if u < p0, action 1 if u < p0 + p1 and
action 2 otherwise, so the generator position never depends on data. The
recorded behavior probability is the binary64 probability of the requested
action. The sampling law differs from it by at most the rounding of p0 + p1 and
of the softmax quotient; that difference is named, not bounded by proof.

A rollout holds at most 256 contiguous transitions. Neither a rollout boundary
nor input exhaustion implies liquidation or termination. Every consumed
transition is retained. The learner stops on replay termination without
resetting, padding or consuming further outcomes. Availability truth and
funding alignment remain caller obligations under A-CAUSAL-REPLAY-V3.

## Targets, normalization and updates

For each rollout the critic is evaluated on the rollout inputs and on the next
observation of every nonterminal row; terminal rows take next value zero and no
dummy observation is evaluated. `batch_v2` with gamma = 0.99 returns raw
advantages and critic targets. Both are guarded at 2^99.

Advantages are normalized with scalar operations: mean = fsum(a)/n, centered
c = a - mean, scale = max(sqrt(fsum(c^2)/n), 1e-8) and c / scale. A nonzero
centered value whose square falls below the smallest normal magnitude is
counted in `underflow`. Each normalized advantage is guarded at 2^99.

Four updates follow. Each evaluates the actor on the rollout inputs, calls
`objective_v2(logits, actions, behavior, advantages, enabled=True)`, forms the
critic residual (value - target)/n, and passes both gradients to `update_v5`.
The actor and critic candidates are accepted together or not at all. A later
rollout collects with the updated actor. Nothing is published if any stage
fails.

## Numerical boundary

Exact rewards and observations convert to binary64 under the v5 rule. Every
reward, logit, value, raw advantage, target, normalized advantage and critic
residual the learner consumes must be finite with magnitude at most 2^99;
otherwise the run rejects before the value is used.

In exact real arithmetic a centered advantage satisfies c^2 <= n·var, so
|c|/scale <= sqrt(n) <= 16 for n <= 256; the 2^99 guard on normalized
advantages is therefore not binding in exact arithmetic. The binary64
normalization is guarded, not bounded by proof. Under the standard rounding
model the critic residual of two values within 2^99 stays below 2^101.

At most 16384 optimizer steps occur per network, far below the optimizer step
cap 2^31-1, and seed + 1 stays below 2^32. NumPy operations run with overflow,
invalid, divide and underflow raising. These guards do not prove every BLAS
intermediate, gradient accuracy or convergence.

## Result

The immutable result holds version and schema metadata, seed, requested and
actual counts, stop reason, final replay session, exact transitions, behavior
probabilities, critic targets, normalized advantages, finite losses, the
underflow count and the actor and critic snapshots. It is not an artifact for
any loader, an account return or a deployable policy.

## Obligations / methods

F-RL-PPO-V7-SOURCE: complete reviewed AST lock; activation first; exact import
list and a frozen-free transitive import closure; the action draw precedes the
first Outcome field read with exactly one generator draw per step; a
probability floor on all three actions; magnitude guards on every consumed
reward, network value, advantage, target and residual; terminal rows without a
next-value evaluation; checked transition, target, objective and optimizer
stages; joint acceptance of actor and critic candidates; a single publication
site; no file, policy, artifact or order effect.

F-RL-PPO-V7-ARITH (SMT, quantifier-free, with an exact rational rounding
chain): (a) softmax terms lie in [0,1] with one equal to one, so the total is in
[1,3] and probabilities are in [0,1] and sum to one; (b) for probabilities that
sum to one the cumulative rule selects exactly one action for every u in [0,1)
and its three intervals have lengths p0, p1 and p2; (c) c^2 <= n·s^2 with
1 <= n <= 256 and s > 0 gives |c| <= 16 s; (d) the critic residual bound; (e)
rollouts <= steps <= 4096 gives at most 16384 optimizer steps, below the cap,
and seed + 1 < 2^32.

F-RL-PPO-V7-FLOW: finite model of activation, start, per-step observe / draw /
consume, target formation, joint actor-critic updates and publication over
small bounds. Publication happens only after every stage of every rollout
succeeded; any failure is terminal and publishes nothing; an Outcome is never
consumed before its action draw; actor and critic step counts are always equal
at a rollout boundary and at publication.

F-RL-PPO-V7-CONFORMANCE: seeds 11/23/47 and 1/17/257 steps run
deterministically; default and wrong-version calls reject. Every first-rollout
behavior probability and action equals an independent recomputation from the
seeded actor and the replayed uniform draws. Every first-rollout critic target
equals an independent exact-rational evaluation of the correctly rounded GAE
recursion, bit for bit, given the critic values of the same stacked batches.
Normalized advantages have mean near zero and magnitude at most 16 plus
rounding. Rewriting every Outcome from step k onward leaves earlier
transitions, actions and behavior probabilities unchanged. Invalid outcomes,
out-of-bound values and sub-floor probabilities publish nothing. Tests are not
proofs.

A-CAUSAL-PPO-V7 trusts pinned CPython 3.13.3 float, math, Fraction and
dataclass semantics, NumPy 2.3.5 on IEEE binary64 round-to-nearest-even
hardware, honest helper modules pinned by hash, the reviewed AST translation, Z3
and sufficient resources. The rounding model is the standard one and is not
machine-proved for the CPython or NumPy build. Network forward and optimizer
arithmetic are guarded for finiteness, not bounded by proof, and network
outputs depend on batch shape in the last bits. No hostile reflection or
monkeypatching is addressed. No convergence, policy-improvement guarantee, data
support or economic claim follows.

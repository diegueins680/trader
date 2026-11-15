# Causal value v6 engineering contract

Scope recorded before implementation, 2026-11-11; base
b79dfa78cd42ef7e94540c5066aa73e5eb1f7d05.

## Scope and interpretation

A separate, default-disabled offline learner that composes causal-replay-v3,
causal-transition-v4, causal-optimizer-v5 and value-objective-v2 into a
conservative Double-DQN training loop. It is the value-based counterpart of
causal-ppo-v5 and the first learner in this repository whose import closure
contains no frozen v1 module: it does not import `sequential_learning`,
`sequential_env`, `sequential_evaluation`, `sequential_registry`,
`run_sequential_screen` or `causal_ppo_v5`.

The frozen helper `sequential_learning.bellman_gradient`, `train_q`,
F-RL-Q-FINITE and F-RL-CQL-SHIFT (both refuted), every preserved counterexample
and all reported value-based results are unchanged. No training run on market
data, artifact, policy, promotion or order interface is added. Acceptance is
engineering evidence, not a financial experiment.

## Inputs and collection

`train_causal_value_v6(initial_bars, outcomes, seed, enabled=True)` requires the
exact version `causal-value-v6`, an exact integer seed 0..2^32-1, a replay-v3
initial prefix and a tuple of 1..4096 `Outcome` values. Each consumed Outcome
holds one next bar, aligned exact funding events, an exact boolean terminal flag
and replay-v3 fill/cost parameters. No callback, file, network, artifact or
production selector is accepted.

At each step the learner observes, encodes the nine exact values, evaluates the
current online snapshot and draws the action **before** reading any field of
that step's Outcome. Exploration is epsilon-greedy with epsilon = 1/8. Every
step consumes exactly two generator draws (one uniform, one action index)
whether or not it explores, so the random stream position never depends on
data. The greedy action is the first maximal Q value. The recorded behavior
probability is the exact rational probability of the requested action under
that rule: 11/12 for the greedy action and 1/24 otherwise. As in v5, the
requested action is stored even when replay support neutralizes its accounting
target.

A rollout holds at most 256 contiguous transitions. Neither a rollout boundary
nor input exhaustion implies liquidation or termination. Every consumed
transition is retained, including adverse and insolvent terminal rows. The
learner stops on replay termination without resetting, padding or consuming
further outcomes, and reports requested, consumed and unused counts and the
stop reason. Unconsumed outcomes are not validated. Availability truth and
funding alignment remain caller obligations under A-CAUSAL-REPLAY-V3.

## Targets and updates

For each rollout the online and target snapshots are fixed while targets are
formed. For a terminal row the target is the reward and no next observation is
evaluated. Otherwise, with next-state online values `o` and target-snapshot
values `t`:

    a* = first argmax_j o[j]
    y  = r + 0.99 * t[a*]

in binary64 scalar operations (one multiplication, one addition). Four updates
follow: each evaluates the online snapshot on the rollout inputs, calls
`objective_v2(q, actions, targets, 0.1, enabled=True)` and passes its gradient
to `update_v5`. After the fourth update the target snapshot is replaced by the
online snapshot. Snapshots are immutable values, so replacement is rebinding
and needs no lock. A later rollout collects with the updated online snapshot.
Nothing is published if any stage fails.

## Numerical boundary

Exact rewards and observations convert to binary64 under the v5 rule: finite,
relative error at most 2^-52, and a nonzero value must stay normal. In addition
every reward and every Q value the learner consumes (collection, selection,
bootstrap and objective rows) must be finite with magnitude at most 2^99;
otherwise the run rejects before the value is used.

Under the standard rounding model (A-FP-ROUNDING, |fl(x) - x| <= u|x|, u =
2^-53) these bounds give |y| < 2^100, so an admitted target lies inside the
value-objective-v2 domain and neither operation can overflow. A nonzero
bootstrap product below the smallest normal magnitude is counted. The count and
the objective's own `exp` underflow counts are summed and published in
`underflow`, so underflow is reported rather than silent.

The update count is 4 per rollout and a rollout consumes at least one
transition, so at most 16384 optimizer steps occur, far below the optimizer
step cap 2^31-1. NumPy operations run with overflow, invalid, divide and
underflow raising. These guards do not prove every BLAS intermediate, absence
of silent underflow inside libraries, gradient accuracy or convergence.

## Result

The immutable result holds version and schema metadata, seed, requested and
actual counts, stop reason, final replay session, exact transitions, exact
behavior probabilities, the per-transition targets, finite losses, the
underflow count and the online and target snapshots. A finite result is
engineering evidence only; input exhaustion can leave open inventory. It is not
an artifact for any loader, an account return or a deployable policy.

## Obligations / methods

F-RL-VALUE-V6-SOURCE: complete reviewed AST lock; activation first; exact
import list and a frozen-free transitive import closure; the action draw
precedes the first Outcome field read; terminal rows take no bootstrap; every
consumed reward and Q value passes the magnitude guard; the objective, optimizer
and transition results are checked before use; a single publication site; no
file, policy, artifact or order effect.

F-RL-VALUE-V6-ARITH (SMT, quantifier-free): (a) with |r| <= 2^99, |t| <= 2^99
and two rounded operations, |y| <= 2^100, and the bound fails if either guard is
dropped; (b) for 0 < epsilon < 1 the three behavior probabilities are positive,
at most one and sum to one, and epsilon = 1/8 gives 11/12 and 1/24; (c) the
selected index attains the online maximum; (d) rollouts <= steps <= 4096 implies
optimizer steps <= 16384 < 2^31-1.

F-RL-VALUE-V6-FLOW: finite model of activation, start, per-step observe / draw
/ consume, target formation, four updates, target replacement and publication
over small bounds. Publication happens only after every stage of every rollout
succeeded; any failure is terminal and publishes nothing; an Outcome is never
consumed before its action draw; the target snapshot changes only after the
fourth update.

F-RL-VALUE-V6-CONFORMANCE: seeds 11/23/47 and 1/17/257 steps run
deterministically; default and wrong-version calls reject. Every published
target of the first rollout equals an independent exact-rational evaluation of
the two correctly rounded operations, bit for bit, given the network values of
the same stacked batch (outputs depend on batch shape in the last bits). Rewriting every Outcome after
step k leaves transitions, behavior probabilities and targets 1..k unchanged.
Behavior probabilities are exactly 11/12 or 1/24 and match the recorded action.
Out-of-bound Q values, rejected objective or optimizer stages and invalid
outcomes publish nothing. Tests are not proofs.

A-CAUSAL-VALUE-V6 trusts pinned CPython 3.13.3 float, Fraction and dataclass
semantics, NumPy 2.3.5 on IEEE binary64 round-to-nearest-even hardware, honest
helper modules, the reviewed AST translation, Z3 and sufficient resources. The
rounding model is the standard one and is not machine-proved for the CPython or
NumPy build. Network forward and optimizer arithmetic are guarded for
finiteness, not bounded by proof. No hostile reflection or monkeypatching is
addressed. No convergence, conservative-value guarantee, data support or
economic claim follows.

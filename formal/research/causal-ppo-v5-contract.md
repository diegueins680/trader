# Causal PPO v5 engineering contract

Scope recorded before implementation, 2026-10-21. This separate, default-disabled
offline learner composes causal-replay-v3, causal-transition-v4 and GAE-v2.
Its new nine-input snapshot schema is incompatible with frozen twelve-input
learners and artifacts. Existing certificates and original38 closures do not
automatically extend to this implementation. Acceptance here is regression
evidence, not a new source-bound proof or a financial experiment.

## Inputs and collection

`train_causal_ppo_v5(initial_bars, outcomes, seed, enabled=True)` requires the
exact version `causal-ppo-v5`, an exact integer seed 0..2^32-2, a replay-v3
initial prefix (26..4096 bars) and a tuple of 1..4096 `Outcome` values.
Each consumed Outcome contains one next bar, aligned exact funding events,
an exact boolean terminal flag and replay-v3 fill/cost parameters. No callback,
file, network, artifact or production selector is accepted.

Start-v3 alone fits the strict pre-decision burn-in scale. At each step, obtain
the old observation, encode its nine values, evaluate the current actor, and
sample a requested action before consuming that step's Outcome. Transition-v4
then supplies the original exact receipt and reward. Every batch contains at
most 256 contiguous transitions; neither a batch boundary nor exhaustion of
inputs implies liquidation or termination. No future preprocessing is fitted.

Store the probability of the requested action, even if support neutralizes its
accounting target. Do not substitute the probability or index of the neutral
action. Replay neutralization is an environment transformation; realized fills
and terminal liquidation remain described by the receipt. Retain every consumed
transition, including adverse and insolvent terminal rows. Stop on replay
termination without resetting, padding or consuming further outcomes. Report
requested/consumed/unused counts and stop reason explicitly. Unconsumed outcomes
are not validated or presented as accepted data. Availability truth and funding
alignment remain caller obligations under A-CAUSAL-REPLAY-V3.

## Numerical boundary and learner

The encoder accepts the exact versioned nine-Fraction observation and valid
header. Each bounded rational converts to finite binary64; exact zero is allowed,
but a nonzero value must remain normal (no subnormal or zero underflow). Verify
relative conversion error <= 2^-52 with exact Fraction arithmetic. Apply the
same rule to rewards. Do not pad inputs, clip failed values or replace rewards.
This checks representation error, not accumulated network or economic accuracy.

The separate functional optimizer carries nine-by-sixteen and sixteen-by-output
weights, biases and Adam moments in immutable little-endian byte tuples. Outputs
are 3 for the actor or 1 for the critic. Snapshot admission validates exact
version/schema, step 0..2^31-1, buffer lengths, finite slots and nonnegative second
moments. Forward/update copy caller arrays, require native base float64 arrays
of 1..256 rows and matching widths, and compute privately. Update returns a new
snapshot only on complete success; rejection leaves the old snapshot unchanged.
This functional API needs no shared mutable optimizer or publication lock.

Use sixteen tanh hidden units, seeded NumPy initialization, the existing clipped
PPO objective, learning rate 0.0003, Adam (0.9/0.999, epsilon 1e-8) and unit gradient
norm clipping. Call GAE-v2 with gamma=0.99 per consumed transition and its existing
lambda=0.95. Terminal rows have no next observation and use next value zero;
evaluate the critic for actual nonterminal next observations only. Preserve the
terminal raw reward target before normalizing advantages. Four actor/critic
updates follow each rollout. A later rollout uses the updated actor; the scale
remains fixed. Publish no intermediate policy or result on any stage failure.

Run NumPy operations with overflow/invalid/divide/underflow raising. Validate
finite inputs, intermediates at the network boundaries, positive normal behavior
probabilities, objectives, gradients, GAE outputs, normalized advantages, residuals
and final snapshot slots. These guards do not prove every BLAS intermediate,
absence of silent underflow inside libraries, gradient accuracy or convergence.

## Result, assumptions and verification

The final immutable result contains version/schema metadata, seed, requested and
actual counts, stop reason, final replay session, exact transitions, sampled
behavior probabilities, finite losses, and actor/critic snapshots. A finite
result is engineering evidence only; input exhaustion can leave open inventory.
It is not an artifact for the old loader, an account return or a deployable policy.

Assume ordinary immutable native objects, stable caller arrays during copying,
trusted CPython/Fraction/dataclass and NumPy 2.3.5 semantics, honest helper modules,
and sufficient resources. No hostile reflection/monkeypatching or interpreter
attestation is addressed. Test on the actual local runtime and state its limits;
do not inherit the older optimizer's CPython-3.13.3/GIL proof by analogy.

Required synthetic fixtures cover seeds 11/23/47, one/partial/multiple batches,
determinism, source immutability, future rewrites, original replay receipt identity,
requested-action probabilities under neutralization, risk/terminal losses, bootstrap
boundaries, numerical extremes, snapshot rejection, finite-difference descent,
and stage-failure rejection. Full verification remains required before merge;
unavailable formal prerequisites must be reported explicitly. No frozen source,
historical result, holdout, original38 status or live setting is changed.

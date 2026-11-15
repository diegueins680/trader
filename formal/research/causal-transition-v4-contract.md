# Causal transition v4 engineering contract

Scope recorded before implementation, 2026-10-21. This is a default-disabled
adapter around causal-replay-v3, with regression evidence rather than a new
source-bound proof. No frozen implementation or obligation status changes.

`transition_v4(session, next_bar, events, action, ...)` accepts only an exact
integer action in {0,1,2}, respectively requesting targets {-1/4,0,1/4}. It
requires `enabled is True` and exact version `causal-transition-v4`. Session,
bar, funding and execution arguments obey the existing replay-v3 contract.

The result is either absent or `(following_session, transition)`. A transition
contains the old nine-component exact observation, requested action index,
target passed to accounting (zero when the old observation is unsupported),
the complete unchanged accounting receipt, terminal flag, next observation
only for nonterminal results, and exact reward `after.equity - before.equity`.
Reward schema `net-equity-delta-v4` is an absolute change in the replay's
initial-equity units, not a log return. It includes all gross, funding and cost
effects, including liquidation. No float conversion, padding to twelve inputs,
likelihood, policy loss, GAE target or checkpoint is manufactured.

The recorded target is an accounting input, not an executed exposure claim:
partial fill, risk termination and liquidation remain described by the receipt.
The caller chooses actions from previously available observations; this adapter
does not authenticate the caller's policy or its information access. It receives
one next bar and its aligned funding tuple, never a future array or callback.
Truthful availability and funding alignment remain A-CAUSAL-REPLAY-V3 assumptions.

The underlying replay decides termination, including tick/risk boundaries.
Explicit termination must be an exact bool. A terminal sample retains its
receipt/reward even when final equity is nonpositive; it has no bootstrapping
observation. A nonterminal sample requires a valid next observation. Batch
boundaries are not treated as terminal. All arithmetic remains bounded exact
Fraction arithmetic. Any rejection returns None with the old immutable session
unchanged and publishes no sample. None is not a flatten instruction.

Acceptance fixtures compare the adapter against actual replay-v3 receipts,
including unsupported inputs, partial fills, funding, adverse risk, final
liquidation and forced tick termination. Future rewrites must leave previous
samples unchanged. Invalid actions, times, funding, numeric bounds, sessions,
versions and execution inputs must reject. This adapter itself imports no
learner or production module. The frozen PPO path remains twelve-input and prefix-wide-scaled;
CE-RL-025 and original38 counts remain unchanged. Formal/full checks remain
required before merge, and unavailable checks must be reported explicitly.

# Causal replay v3: specification before implementation

Engineering successor, 2026-10-07. No market experiment or candidate promotion.
Frozen replay, observations, PPO artifacts and CE-RL-025 retain their semantics.

## Representation and timing

A bar is (symbol, close, available, price): a nonempty instrument identifier of
at most 32 characters, followed by integer nonnegative microsecond timestamps
below 2^63, close <= available, positive bounded exact Fraction price. A session
contains only consumed bars, a frozen scale, and replay-accounting-v2 State.
Every consumed bar must have the same exact symbol identifier. Consecutive
consumed bars require previous.available < next.close. Availability
means the entire completed-bar value is usable, including processing lag; the
caller must supply truthful point-in-time vintages. No revised value is backdated.
The interface receives no future array. A next-bar rejection returns None and
leaves the immutable previous session available to the caller.

Initialization consumes 26..4096 bars. Scale fitting uses bars [0,n-1), ending
strictly before the initial decision at bars[n-1].available. Six exact features
are returns over 1,3,6,24 bars and mean absolute one-bar return over 6,24 bars.
These are NOT the frozen standard-deviation features. For each column scale is
(mean, max(maximum absolute deviation from mean, 1/10^8), min, max).
Normalization is (x-mean)/scale. Support means every raw component is inside its
burn-in min/max. Observation schema causal-close-inventory-v3 has nine rational
components: six normalized market features, signed exposure, drawdown, equity.
All elementary rational operations use the existing 8192-bit checked arithmetic.
Out-of-support observations remain descriptive, but step neutralizes the target
before passing it to exact accounting. Invalid inputs reject; None is not a
request to flatten. No inference callback or order capability exists.

## Transition and refinement

start_v3(burn_in_plus_initial) initializes exact accounting at the last price.
step_v3(session,next_bar,events,target,terminal) first computes the observation
and support at the OLD decision, validates a discrete target in {-1/4,0,1/4},
then advances replay-accounting-v2 with the NEXT bar price and caller-aligned
funding events. It returns (next session, receipt). Final/hard-risk accounting
uses the existing exact kernel. Tick4096 forces termination. No queued orders,
extra delay or learned policy are introduced. Both entries are disabled by
default and require exact version causal-replay-v3; neither persists anything.

Abstraction: session history projects to consumed timestamps/prices; scale is
fixed once from the strict pre-decision prefix; accounting projects identically
to the existing rational kernel. Induction hypothesis: two runs with identical
consumed bars, initial scale and accepted actions have identical sessions.
A step reads only the prior session and its one supplied next bar/events, so
later unconsumed records cannot alter its observation or any previous session.
Source footprint and complete AST lock bind this model to Python. CPython/Fraction
semantics remain trusted; this is not a verified interpreter or compiler.

## Obligations

- F-RL-CAUSAL-V3-SOURCE: reviewed complete source, prefix slices, immutable values,
  default-disabled entries and accounting call composition.
- F-RL-CAUSAL-V3-ARITH: nonempty scale rows, fitted and feature indices <= current
  decision, availability ordering, exact normalized support interval, action
  bounds and tick/history alignment (SMT, explicit satisfiable premises).
- F-RL-CAUSAL-V3-FLOW: consume-before-observe, fixed scale, no future input, no
  publication after rejection/terminal, no authority transition (bounded model).
- F-RL-CAUSAL-V3-CONFORMANCE: deterministic prefix replay, future rewrite with
  scale refit limited to burn-in, missing/late/invalid values, old-session
  immutability, Haskell rational observation oracle and exact accounting identity.

## Assumptions, limits and acceptance

A-CAUSAL-REPLAY-V3: truthful complete-bar availability and funding bucket alignment;
ordinary immutable native objects, no hostile reflection/monkeypatching, trusted
CPython Fraction/dataclass and checked accounting primitives, sufficient resources.
No claim about authentic historical releases, empirical fills, profitability,
policy training, neural finiteness, full Haskell implementation refinement or
physical latency. Synthetic conformance only. Original38 criteria are unchanged;
17 cannot close while frozen learners and inherited ingestion remain unresolved.
Critical successor checks must pass formal/full before merge. No live changes.

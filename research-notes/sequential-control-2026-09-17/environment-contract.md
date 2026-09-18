# Sequential replay contract v1

Status: rejected offline mechanism prototype; no production adapter. The frozen
experiment source is `177936552e358b0736442a463703bb1fba60884b`; registration
was committed first at `dfc6b27d`. See the [registration](../registrations/sequential-control-screen-v1.json)
for every fixed algorithm parameter, budget, split and promotion prohibition.

## Formulation and use case

Use partially observed inventory control, not market prediction by reward.
Inventory, funding, pending fills and turnover couple successive decisions.
The simulator's full state includes an exogenous historical clock, prices,
funding, units, equity, peak equity and pending target; the policy cannot observe
the future clock-indexed prices. The 12 observed variables do not make markets
Markov. This is a POMDP interpretation with a finite observation approximation,
not an identified causal model of exchange responses. Deterministic cost-aware
ridge optimization and immediate-reward contextual regression are required
complexity baselines. Neither market-making nor execution scheduling can be
credibly modeled with these close-only inputs.

## Observation and state (`close_inventory_v1`)

At completed bar t, read only prices[t-24:t+1]. Six features are 1-, 3-, 6-,
24-bar returns and 6-/24-bar return standard deviations. Fit their mean, standard
deviation and coordinate support ranges on training prefixes only; retain
missingness as unavailable, never impute a directional zero. Six state variables
are drifted signed exposure, drawdown, equity/initial equity, pending target,
pending delay/2, and remaining episode fraction. No future funding rate, fill
price or future label is visible. Future price perturbation must leave the
observation at t unchanged. The remaining fraction intentionally exposes the
known terminal obligation; episode-boundary generalization is not established.

Scaler construction requires six-component real unmasked finite mean, deviation
and support-bound vectors, strictly positive deviations and ordered lower/upper
bounds. It copies parameters into immutable float64 backing bytes; mutable caller
arrays and NumPy write-flag changes cannot alter the snapshots. Fitting accepts a
nonempty list/tuple of supplied prefixes, each with at least 25 real unmasked
prices. It rejects incomplete prefixes rather than omitting them. Transform and
support queries require finite real unmasked six-component features. Normalization
uses checked float64 arithmetic; invalid inputs or non-finite results raise.
Malformed support queries return false. Replay turns normalization errors into
absent observations before fills. See the [normalization audit](normalization-admission-audit.md).

Coordinate-wise training ranges are a weak out-of-distribution check. They do
not prove joint market-state support or inventory-state/action support. At an
unsupported but otherwise valid market state, evaluation proposes a zero target
for a later close. At an invalid observation it terminates with failure and no
new order. Neutral targeting may still incur exit costs and is not instant cash.

## Action (`signed_target_v1`)

The three actions are absolute targets -0.25, 0, +0.25 of current equity,
independently evaluated every 1, 3 or 6 bars. They are not raw exchange orders.
No leverage search occurs. One pending proposal is allowed; another proposal
while it is pending is rejected, without replacing it. Invalid/non-finite values,
wrong ownership, disabled mode and elapsed inference beyond 20 ms fail closed.
The inference timer rejects a completed slow call; it does not preempt a hung
thread and is not a production timeout implementation.

Runtime admission additionally requires exact Python boolean `True` for enabled,
valid-observation and ownership gates. Strings, integer flags and NumPy booleans
are not authorization. Timing and actions must be finite real numeric scalars,
excluding booleans, complex values and arrays. Inference requires real, finite,
unmasked vectors of the declared widths and returns no proposal on model failure.
Invalid gates are checked before replay observation reads or pending-fill work.
These checks refine malformed-input rejection; valid v1 actions and artifacts
keep their existing semantics. See [proposal-types-audit.md](proposal-types-audit.md).

## Transition (`delayed_close_units_v1`)

Assume availability at close plus 1 ms, decide then, and fill at close t+1
(t+2 under delay stress). Old units earn price changes and pay actual funding
settlements in (leftClose,rightClose] before the endpoint fill. Funding cash is
minus old units times settlement mark times rate. New units are target times
pre-fill equity divided by fill price. Costs reduce equity; units subsequently
drift. Partial fills change only a fraction of the requested unit change;
missed-fill stress skips each tenth proposal's fill. Pending proposals are
cancelled at episode end. There is no actual order submission or cancellation.

The historical bar has no recorded first-seen witness; availability is an
assumption and a promotion blocker. The simulation does not identify minimum
notional, tick/lot precision, queue position, borrowing, margin, intrabar
liquidations, exchange downtime, delistings, adverse selection or endogenous
impact. The impact sensitivity is an assumed surcharge, not a calibrated causal
impact model. BTC/ETH and eight other survivors are not a point-in-time universe.

## Reward (`net_equity_inventory_v1`)

Reward = 100 * relative equity change - 100 * 0.01 * sum of
(pre-bar exposure squared * trailing 24-bar variance) during the decision step.
All fees, spread, slippage, funding and terminal liquidation affect equity.
The inventory penalty has no cash debit and is excluded from economic P&L.
CQL's registered ablation sets its coefficient to zero. Rewards have fixed
100 scaling, no clipping, no learned normalization; discount is 0.99 per bar
raised to each independently evaluated decision cadence. No claim of CVaR
optimization follows from this variance penalty.

## Episodes, behavior and data boundaries

Training episodes start in cash with 24 completed warmup bars, span at most
96 transitions, and remain inside physically sliced training prefixes. Seeded
symbol/start selection uses seeds 11, 23, 47. PPO samples its categorical policy;
Double DQN explores with epsilon 0.20; offline CQL uses a fixed uniform simulated
behavior buffer with exact probability 1/3 per target. This is simulated
counterfactual behavior, not exchange fill logs or live exploration. Fixed last
updates are evaluated; no best checkpoint or best seed is selected.

Every fully accounted terminal is now summarized immediately, including a terminal
on the collection budget's last sample. Optional `episodeAccountingV2` training
metadata counts collections, started, completed, truncated and decisions. Completed
includes accounted risk stops; it is not a favorable outcome classification. A live
episode at a collection cutoff is truncated, retaining its nonterminal successor
without fabricated liquidation or an episode return. PPO/Double DQN aggregate
across their 256-decision collections; CQL counts its single offline collection.
Aborted calls return no successful coverage record. Legacy archives without v2
retain their original, potentially incomplete episode-summary coverage. The new
metadata does not change v1 observations, rewards, transitions, learning or policy
artifacts. See the [accounting audit](episode-accounting-v2-audit.md).

PPO and Q entry points validate training controls before constructing a model or
using RNGs. Horizons, seeds and step budgets require Python/NumPy integers excluding
booleans, with horizons 1/3/6, nonnegative seeds and strictly positive budgets.
They normalize to Python integers before seed offsets and batch arithmetic. Q
mode requires exact Python `True` or `False`; execution/risk coefficients are also
validated before initialization. Invalid controls raise instead of returning a
zero-update fit or selecting a mode through truthiness. The registered 4,096-step
budget remains unchanged; this validation does not impose a new compute ceiling.
See the [training-admission audit](training-admission-audit.md).

Episode indices, horizons, collection seeds and transition budgets require Python
or NumPy integers, excluding booleans; admitted NumPy integers normalize to Python
integers. Horizons remain 1/3/6, seeds nonnegative and budgets positive. Collection
requires matching nonempty symbol maps, aligned one-dimensional real unmasked
price/funding arrays and at least 121 training bars per symbol for its existing
97-bar episodes. Behavior probabilities use the same real unmasked representation.
Execution coefficients are finite real nonboolean scalars in their existing
ranges. Admission checks only representation, lengths and indices, never unused
future values. Trailing prices and actual transition prices/funding are checked
when consumed; unavailable values fail that path without invented fills. See the
[admission audit](episode-admission-audit.md) for valid-input parity and limitations.

CQL consumes 4,096 transitions and 4,096 gradient updates. Double DQN collects
256-transition blocks before updates (an implementation deviation from the
registration's 64-transition warmup phrase); there is no 64-step initial update.
The discrete CQL prototype uses a Double-Q squared Bellman loss plus logsumexp
penalty, not the original paper's distributional Atari critic. PPO uses a
16-unit tanh network and a separate value network. These are mechanism
implementations, not reproductions of published benchmark performance.

At a valid terminal price, cancel pending proposals and debit full liquidation
costs exactly once. Do not bootstrap through a terminal. Invalid market/observation
or permission failures stop the path without inventing a fill or cash continuation;
that path is incomplete and fails evidence admission. Consequently terminal
liquidation is guaranteed only for valid-price normal/risk-triggered endings,
not for missing-price/invalid-observation failures. Equity must reconcile exactly
to initial equity plus gross P&L plus funding minus cash costs.

Collection now rejects incomplete transitions before appending a learning sample.
A nonterminal transition must span the full decision interval and supply a real,
finite, unmasked 12-component successor. A terminal must have positive finite
equity, zero units, no pending target and either reach the normal endpoint or
finish an accounted capital-floor, drawdown, exposure or turnover stop. Finite
reward and time progress are required in both cases. Invalid market/observation,
permission and unaccounted insolvency failures abort the collection call without
returning a partial batch. Zero successor padding is reserved for accounted
terminals. Risk losses are retained, not filtered from training or reclassified
as promotion successes. See the [transition audit](transition-admission-audit.md).

## Independent deterministic boundary

The Python shield defaults disabled and restricts proposal type, finiteness,
action bounds, ownership and timing. Replay additionally rejects turnover above
0.50 and stops at endpoint exposure above 0.35, drawdown above 15%, or equity
below 0.80. These are research constraints, not modifications to live limits.
A market gap can cross a threshold before detection; this limitation is counted
as failure, not described as a guaranteed maximum loss. Terminal liquidation is
exempt from the entry turnover check so liabilities are accounted for.

The isolated Haskell `Trader.Research.PolicyProposalV1` has a private proposal
constructor. Every admitted proposal requires independent observation, ownership,
artifact, support and deterministic-risk evidence. `orderAuthorized` is always
False. No executable/bot imports the module; it is built through the test suite.
The exhaustive 2,688-case test establishes the finite guard domain and precedence;
it is not proof of exchange behavior or correctness of supplied evidence.
Production risk gates, identity, caps and champion ownership remain untouched.
There is no production loader, auto-promotion, self-modification, retraining hook,
network client or order adapter in this research boundary.

## Policy persistence admission

Save and load share the `offline_policy_v1` parameter contract: exactly `w1`,
`b1`, `w2`, `b2`, with shapes 12x16, 16, 16x3 and 3. Saving requires real unmasked
arrays, snapshots their values, validates finite float64 portability and rejects
invalid fields before opening the destination. Integer JSON values retain their
representation; floating parameters use float64. Loading first rejects nonnumeric
JSON leaves, then applies the same shape/value contract. A value-network critic
with one output cannot be saved as a three-action policy.

Valid artifact schema, provenance, bytes, 65,536-byte limit, disabled status and
exclusive-write behavior remain unchanged. An invalid save raises rather than
returning a success digest; the runner records a failed fit and blocked replays.
This is admission before I/O, not atomic recovery from filesystem write failures.
See the [policy-save audit](policy-save-admission-audit.md).

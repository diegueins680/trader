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

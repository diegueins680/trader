# Causal RL composition pilot

Baseline **354c853176b7ee2ac6570951f7f6f2d0aed6909d**, latest origin/main on
2026-10-10; [successful CI](https://github.com/diegueins680/trader/actions/runs/37959950956).
The user's original worktree and uncommitted changes are preserved. This branch
starts at that SHA. The registration is committed before implementation/training.

This is a finite-horizon, partially observed, exogenous price-taking exposure
problem. Observing finitely many prices does not make markets Markov or stationary.
Units, costs, funding and next-close fills give actions delayed consequences.
The environment composes the existing causal replay v3 with exact accounting v2;
new policies have a distinct observation/environment/artifact identity. No old
identifier or trained artifact is reinterpreted. The policy proposes one of three
absolute exposures. Proposal labels/probabilities remain attached to those proposals
through shield projection, reservation and fills. PPO optimizes that composed
proposal policy; Q functions value proposals, not relabeled fills.

Each trial starts with one unit of equity. No external cash flows. Reward is
100 times its exact equity increment, so the undiscounted sum is exactly 100 times
net final change before conversion to binary64 for learning. Fees/funding are
already in equity. Terminal liquidation occurs once. A finite training episode
is an economic finite-horizon task, not an unrecorded free reset. A PPO rollout
boundary is a computation boundary: preserve the active episode and bootstrap.
Risk/data failure is recorded, never interpreted as an equal-horizon cash result.

For the one-extra-bar delay stress, only one simulated intent is pending. An
arriving proposal cannot replace it. It is revalidated against the observation
support at the fill boundary; account risk is checked by the exact kernel. Every
transition logs raw proposal, permitted target, submitted target or absence,
pending target, applied target and actual unit delta; terminal closes are distinct.
The live risk controller and protective ownership are outside this simulator.

No research entry point obtains an exchange client, credential, database, queue,
production namespace, network connection, promotion callback or order capability.
Python remains an offline process. Existing Haskell production and research
interfaces are unchanged; actual Haskell numerical oracle tests provide scoped
conformance, not a new inference authorization. Network throughput, precision,
minimum notional and persistent cross-instance reservation integration remain
explicit gates, not assumptions silently satisfied by a backtest.

The existing formal registry is reused. New composition evidence will identify
its numerical and runtime assumptions, bounded lifecycle domain, implementation
mapping and remaining gaps. Property tests are not theorem proofs. Kernel proofs
do not imply profitability, original publication times or safety under arbitrary
market disturbances. No market holdout is opened by this pilot.

Pre-market implementation amendment: the initial synthetic benchmark measured
0.233 seconds per step because public causal-v3 re-fits the identical burn-in on
every admission. The composition now validates once at Episode construction,
retains immutable owned sessions and caches each observation. A session identity
change rejects. `advance_owned` uses the unchanged exact feature formulas and
accounting public API, with explicit symbol/clock/history admission. Differential
tests compare it with public `step_v3`; this is a reviewed construction invariant,
not a proof that arbitrary caller-built Sessions are valid. Hostile reflection or
monkeypatching is outside the Python trust boundary. Delay uses an explicitly
reviewed zero-fill call to the kernel's private pure transition after the public
input predicate succeeds with fill=1. A zero-fill transition preserves units unless
risk termination liquidates; it cannot silently rebalance. No economic parameters,
search space, data or registered evaluation endpoints were changed.

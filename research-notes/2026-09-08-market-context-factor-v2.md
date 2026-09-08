# Point-in-time market-context factor v2

Date: 2026-09-08

Disposition: research infrastructure only; isolated from production

Risk status: `FEATURE-MISSINGNESS-001` remains **OPEN**

## Result

The additive Haskell `point_in_time_market_context_factor_v2` adapter consumes
one `point_in_time_liquidity_universe_v2` selection at each bar and computes a
quote-volume-weighted one-bar peer return. It removes the target before taking
the requested peer count, renormalizes the remaining weights, and retains the
latest event and first-seen availability witnesses used by the value.

The adapter does not train or run a predictor, evaluate an outcome, open the
sealed holdout, alter the current champion, authorize an order, or change live
behavior. The source-selection and feature modules remain absent from legacy
market-context, predictor, bot, and execution imports.

## Academic and repository motivation

The cryptocurrency factor evidence in Liu, Tsyvinski, and Wu's
[Common Risk Factors in Cryptocurrency](https://doi.org/10.1111/jofi.13119)
depends on a broad point-in-time cross-section. It does not justify applying a
single terminal universe backward through history. The paper's weekly spot
panel, characteristic sorts, and implementation assumptions also differ from
this repository's intraday perpetual-futures setting, so this change is a
causality primitive rather than a reproduction or efficacy claim.

The legacy `MarketContext.weightedMarketLag` computes a volume-weighted return
basket, but `buildMarketModel` chooses one membership and quote-volume vector
near `fitEnd` and reuses it across the earlier aligned price history. Its OLS
intercept, beta, and residual variance are then fitted on that lag. The v2
adapter closes only the per-row raw-factor gap; it deliberately does not
recreate the fitted eight-feature market-context block.

## Exact contract

Each row requires:

- a canonical target whose suffix exactly matches the selection quote;
- a positive peer count, non-negative bar open, positive interval, and
  representable bucket end;
- a selection decision from the bucket end through the instant before the
  following bucket end;
- a peer vector with exactly one position per ranked selection member; and
- every present peer to identify the same member and exact bar as its vector
  position.

The target is excluded before the first requested peers are taken in ranked
order. Upstream code must therefore select one extra member when the target may
be present. A chosen peer is usable only when its event equals the bucket end,
its first-seen time falls from event through decision, and its simple return is
finite and greater than `-1`. The weighted mean uses overflow-resistant online
normalization, so finite high-magnitude volumes are not summed directly.

Malformed scope, shape, alignment, grid, or decision timing returns no row.
Missing, late, non-finite, or economically impossible chosen-peer evidence
instead yields value `0` with availability mask `false`; observed zero retains
mask `true`. This distinction ensures unavailable evidence is never assigned a
bullish or bearish meaning.

The series constructor requires a non-empty exact contiguous bar grid, one
selection and peer vector per row, and strictly ascending decisions. Membership
and weights are selected separately for every row. Future rows and future peer
changes cannot alter an existing prefix.

## Deliberate deviations and limitations

On complete inputs, `market.return_1` matches the peer-basket portion of the raw
legacy `weightedMarketLag` formula after target exclusion. It does not include
the optional same-asset Coinbase augmentation and is not parity for the full
legacy market-context feature block because that block additionally uses an
OLS intercept, beta, residual, z-score, and related transforms. A faithful
versioned successor would need fold-local fitting, training and split
provenance, artifact compatibility, and its own causal tests.

The adapter also cannot prove that the universe snapshot represents the full
venue population or that peer returns came from verified source artifacts.
Those guarantees require a lawful timestamp-preserving source manifest and
deterministic reconstruction outside this pure boundary.

## Verification evidence

`testMarketContextFactorV2` covers:

- stable schema identity and optional-mask layout;
- parity with the legacy weighted-return formula on complete evidence;
- target exclusion and post-exclusion weight normalization;
- changing point-in-time membership and weights across rows;
- event/availability witness propagation and future-prefix invariance;
- missing, premature, non-finite, and impossible returns becoming unavailable;
- symbol, bar, vector, scope, interval, count, decision, and grid rejection; and
- finite output when naive raw volume and weighted-return sums would overflow.

`testMarketContextFactorV2ProductionIsolation` checks that the new module and
schema are absent from the main, market-context, legacy universe, feature,
predictor, online-neural, trading, and order-execution paths.
The root market-prediction research test independently enforces the same source
isolation and checks the upstream-extra-member and no-back-application contract
text.

## Decision

The per-row raw-factor boundary is complete, but the snapshot-complete universe
and peer-return source artifacts, fitted-model versioning, and prospective data
beginning 2027-01-21 remain unavailable. No forecast, return, transaction-cost,
drawdown, DSR, PBO, or holdout result was produced. **Continue research; no
candidate passed.**

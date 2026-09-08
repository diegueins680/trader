# Point-in-time liquidity universe boundary v2

Date: 2026-09-08

Disposition: research infrastructure only; isolated from production

Risk status: `FEATURE-MISSINGNESS-001` remains **OPEN**

## Result

The Haskell research boundary now includes the additive
`point_in_time_liquidity_universe_v2` contract. It selects a top-liquidity
population only from one coherent quote-scoped snapshot whose economic event,
real first-seen availability, explicit contemporaneous eligibility, and
decision time are retained.

The contract does not build a predictor, compute an outcome, open a holdout,
change a champion, authorize an order, or alter live behavior. It is not a
source artifact and cannot establish that a caller supplied every instrument
that existed on the venue.

## Academic and repository motivation

Crypto factor evidence is cross-sectional: the investment universe and
tradability screen are part of the hypothesis, not incidental preprocessing.
Liu, Tsyvinski, and Wu's *Common Risk Factors in Cryptocurrency* motivates the
repository's point-in-time universe requirement, while the broader review
documents survivor-bias and implementation-cost limitations. The canonical
source and concise assessment are in
`market-prediction-2026-09-04/paper-matrix.csv` and
`literature-review.md`; no paper PDF is committed.

The production `MarketContext` path remains a simple volume-weighted market
factor and OLS target relation. The audit confirmed three data-contract gaps:

- `PointInTimeUniverse` has one generic timestamp, not separate event and
  availability times;
- it independently keeps the last row for each symbol with no staleness bound,
  which can mix vintages and retain a disappeared member indefinitely; and
- `buildMarketModel` selects one universe near `fitEnd` and applies that
  membership and its quote-volume weights across earlier rows. When historical
  membership is not required, it can instead use the current exchange ranking.

These choices are operational compatibility behavior. They are not admissible
for a survivorship-controlled research or promotion claim.

## Exact v2 contract

`pointInTimeUniverseSnapshotV2` requires:

- a canonical exact quote scope;
- non-negative event time no later than first-seen availability;
- a non-empty, unique canonical-symbol population entirely in that quote;
- finite non-negative quote volume; and
- an explicit point-in-time eligibility bit for each member.

The eligibility field replaces token-name guesses and current listing status.
A later-ineligible or delisted asset can remain in the historical snapshot with
`eligible = false`; it is not silently deleted from history.

`pointInTimeUniverseSelectionV2` then requires:

- a positive requested population and non-negative event-age bound;
- exactly one quote scope;
- unique `(event time, availability time)` identities among snapshots usable at
  the decision;
- availability no later than the decision;
- event age within the configured limit; and
- at least the requested number of eligible positive-volume members.

It chooses the newest economic event available by the decision. For revisions
of the same event, it uses the newest revision only after that revision's own
first-seen time. Quote-volume ties break by canonical symbol, and weights are
scaled before summation so finite raw volumes cannot overflow the normalizer.

## Causality and missingness

Appending a future snapshot or later revision cannot alter an earlier
selection. Missing, stale, incomplete, mixed-quote, duplicate-identity,
non-finite, or insufficient evidence returns no selection. No zero-valued
synthetic member, stale carry-forward, current-list substitution, or bullish or
bearish interpretation is produced.

The snapshot constructor intentionally cannot certify population completeness.
A future source must bind a comprehensive instrument manifest, venue/source
identity, event and acquisition clocks, revisions, eligibility semantics,
license, schema, hashes, and deterministic reconstruction. A future market
factor must select separately at every decision row; it may not reuse a later
cutoff membership or weight vector for earlier history.

## Verification evidence

`testPointInTimeUniverseV2` covers:

- schema identity and canonicalized snapshot member order;
- event, availability, decision, scope, and eligibility retention;
- same-event revision timing and future-suffix invariance;
- newest-event selection independent of input order;
- deterministic volume ties and normalized finite weights;
- overflow-safe normalization when a raw finite volume sum would overflow;
- stale, insufficient, unavailable, future, ambiguous, and mixed-scope input;
  and
- malformed identifiers, timestamps, duplicates, negative values, NaN, and
  infinity.

Automation separately proves that the module is not imported by the legacy
universe loader, `MarketContext`, feature construction, predictor routing, or
the main entry path.

## Decision

This boundary defines the required point-in-time selection semantics but does
not justify a model or close the source gap. A snapshot-complete lawful public
artifact, per-row market-factor adapter, versioned model boundary, and
prospective evidence beginning 2027-01-21 remain required. No development
metric or final holdout was viewed. **Continue research; no candidate passed.**

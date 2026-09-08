# Prospective Binance USDⓈ-M market-context panel v2

Date: 2026-09-08

Disposition: research infrastructure only; decoder, offline raw-manifest
verification, and prospective collector complete; no real acquisition run

Risk status: `FEATURE-MISSINGNESS-001` remains **OPEN**

## Result

The additive Haskell `binance_usdm_market_context_panel_v2` decoder defines the
first source-facing boundary for a future point-in-time market-context panel.
Each derived snapshot binds a complete declared population to a unique frozen
source-manifest digest, exact member count and source ordinal sequence,
rolling-volume event windows, first-seen and decision clocks, explicit
eligibility, and an optional exact-bar peer return for every member.

The decoder composes with `point_in_time_liquidity_universe_v2` rather than
implementing another liquidity rank. It reorders optional peer cells to the
canonical selector's ranked symbols, after which the existing
`point_in_time_market_context_factor_v2` adapter can exclude the target and
compute the raw peer factor.

No public endpoint was called, no dataset or outcome was inspected, no model
was trained, and no holdout, champion, order, or trading authorization changed.
The new module remains absent from legacy and production imports.

## Public source interpretation

The intended source is the repository's existing public, read-only
`binance-usdm-public-market-data` record. Binance's official USDⓈ-M REST
documentation assigns the required facts to separate responses:

- [exchange information](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information)
  supplies contemporaneous symbol, contract, quote-asset, onboard, and status
  metadata;
- the all-symbol [24-hour ticker](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/24hr-Ticker-Price-Change-Statistics)
  supplies a current rolling window and quote volume, not a historical
  bar-aligned universe;
- [server time](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Check-Server-Time)
  brackets acquisition independently of local wall-clock assumptions; and
- [klines](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data)
  supply each exact completed peer bar.

This means the source can only create prospective evidence. A ticker fetched
today cannot be backdated to a historical decision, and current exchange
metadata cannot establish an asset's earlier eligibility. The planned
collector must preserve every raw response, response completion, server-time
bracket, endpoint parameter, status, rate-limit header, exclusion decision,
and hash without credentials or private endpoints.

## Exact CSV contract

The 20 columns are fixed and ordered:

| Group | Columns | Contract |
|---|---|---|
| Identity | `schemaId`, `sourceId`, `sourceManifestSha256` | Exact schema/source identifiers and a lowercase 64-hex digest of one frozen source manifest |
| Snapshot | `quote`, `intervalMs`, `barOpenTime`, `decisionTime`, `universeEventTime`, `universeAvailabilityTime` | One exact quote and interval; causal, non-negative integer-millisecond clocks; decision within the bar after the completed peer event |
| Completeness | `populationCount`, `memberOrdinal` | Row count must equal the declared positive population and ordinals must be exactly `0..n-1` in source order |
| Universe member | `symbol`, `quoteVolume`, `eligible`, `tickerOpenTime`, `tickerCloseTime` | Unique canonical quote-scoped symbol, finite non-negative volume, binary eligibility, and an explicit rolling-volume window |
| Peer cell | `peerObserved`, `peerEventTime`, `peerAvailabilityTime`, `peerSimpleReturn` | Either all three value/witness fields are blank, or a finite return greater than `-1` has exact bar-end event time and causal first-seen availability |

Within a snapshot, repeated metadata must be identical. `universeEventTime` is
the latest retained ticker close; all ticker closes precede the snapshot's
first-seen time. Across a panel, quote and interval remain fixed, bar opens and
decisions strictly increase, and a source-manifest digest cannot be reused for
another decision. Missing peer evidence is `Nothing`; it is not serialized or
decoded as return zero.

## Trust boundary

The decoder can prove only properties of the supplied CSV and digest text. It
cannot prove that:

- the referenced manifest bytes exist or match the digest;
- the raw exchange-info, ticker, server-time, and kline responses are complete;
- the eligibility predicate was applied to every response member;
- an excluded or delisted instrument was retained correctly; or
- the derived CSV is a deterministic projection of those raw bytes.

The offline `binance_usdm_market_context_source_manifest_v1` verifier now
independently hashes and decodes the raw responses, recomputes population
membership, counts, volumes, clocks, close-to-close returns, and panel bytes,
and rejects partial evidence. Its separate verification receipt binds both the
source-manifest and derived-panel hashes, avoiding a circular digest between a
panel and the manifest digest it embeds. Merely replacing the declared digest
cannot make altered rows admissible. A separate bounded public collector now
exists, but its output is explicitly `complete_unverified`, it is locked to the
registered 2027-01-21 through 2028-01-20 data window, and it has not been run
against the exchange. No real bundle is currently admissible.

## Verification evidence

`testMarketContextPanelSchemaV2` covers:

- stable schema/source identity and exact column order;
- population completeness, ordinals, unique symbols, eligibility, and rolling
  ticker clocks;
- strict lowercase digest syntax and unique per-decision manifests;
- canonical-selector projection and peer-vector reordering;
- composition with the existing factor and complete-input numeric parity;
- observed versus missing peer evidence without synthetic zero;
- future-suffix invariance;
- malformed headers, sources, hashes, scopes, masks, counts, ordinals, symbols,
  volumes, intervals, timestamps, returns, overflow, and longitudinal order;
  and
- Haskell and root automation production-isolation checks.

## Decision

The typed derived-panel boundary, independent offline raw-manifest verifier,
and bounded public collector are complete. No real acquisition ran, and
historical data cannot be retrofitted with these unavailable facts. The
collector rejects bars before 2027-01-21; the registered final holdout remains
untouched. No forecast, return, cost, drawdown, DSR, PBO, or latency result was
produced. **Continue research; no candidate passed.**

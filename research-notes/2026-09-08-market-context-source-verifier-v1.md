# Offline market-context source verifier v1

Date: 2026-09-08

Disposition: research infrastructure only; offline verifier implemented,
prospective public collector not implemented

Risk status: `FEATURE-MISSINGNESS-001` remains **OPEN**

## Result

`scripts/research/market_context_source.py` is an independent offline verifier
and deterministic derivation boundary for prospective public Binance USDⓈ-M
market-context evidence. It has no network or credential interface and cannot
collect data. Given a frozen `binance_usdm_market_context_source_manifest_v1`
bundle, it verifies every registered raw byte before emitting the existing
`binance_usdm_market_context_panel_v2` CSV and a separate
`binance_usdm_market_context_verification_receipt_v1`.

No exchange endpoint was called, no real dataset or outcome was inspected, no
experiment or model was run, and no holdout, champion, order, deployment, or
live authorization changed.

## Source and derivation contract

The manifest is restricted to the public, read-only
`binance-usdm-public-market-data` source/license record and exact `/fapi/v1/time`,
`/fapi/v1/exchangeInfo`, `/fapi/v1/ticker/24hr`, and per-symbol
`/fapi/v1/klines` requests. The verifier requires:

- strict UTF-8 JSON without duplicate keys or non-finite constants;
- exact flat `raw/*.json` inventory, SHA-256 identities, byte counts, endpoint
  parameters, HTTP 200 status, request clocks, and recorded request weights;
- sequential response timing and a bounded provider server-time bracket proving
  that the feature bar was complete;
- the complete exchange-info population satisfying `PERPETUAL`, the declared
  quote asset, `TRADING`, and `onboardDate <= barEndTime`, with exact exclusion
  reasons for every other returned instrument;
- one finite rolling quote-volume observation for every eligible member; and
- exactly the previous and current completed kline for every eligible member.

Peer evidence is the current close divided by the previous close minus one.
This is close-to-close—not the current candle's open-to-close return—and its
first-seen time cannot precede the raw response completion. Universe
availability conservatively includes exchange-info completion, ticker response
completion, and the latest ticker event. Missing eligible members, extra raw
files, stale or late evidence, changed parameters, unsafe paths, malformed
values, and incompatible grids fail closed.

## Provenance and authority

The panel embeds the SHA-256 of the exact source-manifest bytes. The receipt is
created separately and binds that digest to the exact derived panel digest,
row count, schema versions, source-manifest code commit, and verifier byte
digest, avoiding circular provenance. Its `outcomeUse`, `modelUse`, `orderUse`, and
`liveAuthorizationUse` fields are all `false`.

A verified receipt proves only that these derived bytes follow from this frozen
bundle under verifier v1. It does not admit data into an experiment, validate a
future collector, open the registered holdout, promote or deploy a model, or
authorize an order. A future collector must remain a separate component and
must preserve complete raw responses for this verifier.

## Reproduction

The committed fixture is small and synthetic:

```bash
python3 scripts/research/market_context_source.py verify \
  --manifest test/fixtures/market-context-source-v1/source-manifest.json \
  --panel-output /tmp/market-context-panel.csv \
  --receipt-output /tmp/market-context-receipt.json
node --test test/market-context-source.test.mjs
cd haskell && cabal test trader-tests --test-show-details=direct
```

The source-manifest fixture digest is
`2439fe8a03915e4e3b66cac4b160fa71bcbb120992bd684cc45ecfa85ba6f3b9`;
the expected panel digest is
`e79995a422500a213a44055e637a1a380978890c165210a113719a3405dfc0da`.
The Node regressions cover deterministic output, tampering, incomplete
population, changed endpoint parameters, late decisions, unregistered raw
files, strict-JSON failures, non-finite market values, invalid kline grids, and
protected source bytes. The Haskell suite decodes the exact generated golden
panel and composes it with the canonical universe selector and factor adapter.

## Decision

The independent offline verification boundary is complete. Real prospective
acquisition is not. The development window begins 2027-01-21, the final holdout
begins 2027-09-21, and neither was opened. No forecast, net return, cost,
drawdown, tail-risk, DSR, PBO, or inference benchmark exists for a candidate.
**Continue research; no candidate passed.**

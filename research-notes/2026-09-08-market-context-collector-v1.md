# Prospective public market-context collector v1

Date: 2026-09-08

Disposition: research infrastructure only; collector implemented and not run
against a real endpoint

Risk status: `FEATURE-MISSINGNESS-001` remains **OPEN**

## Result

`scripts/research/collect_market_context.py` creates one bounded prospective
Binance USDⓈ-M source bundle for the existing
`binance_usdm_market_context_source_manifest_v1` boundary. It preserves exact
public response bytes and timestamps but deliberately cannot verify, admit,
model, evaluate, promote, deploy, or trade on them. Its successful collection
state is `complete_unverified`; only the separate offline
`market_context_source.py` process can derive a panel and receipt.

No public request was made while implementing or testing this collector. The
only committed data remains the small synthetic fixture. No outcome, model,
experiment budget, final holdout, champion, order, deployment, credential, or
live authorization changed.

## Official source contract as of 2026-09-08

The collector fixes `https://fapi.binance.com` and refuses redirects. It uses
only public GET endpoints documented by Binance:

- [server time](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#check-server-time),
  weight 1, before and after the acquisition;
- [exchange information](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#exchange-information),
  weight 1, for current status, contract type, quote asset, and onboard time;
- the all-symbol [24-hour ticker](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#ticker24hr-price-change-statistics),
  weight 40 when `symbol` is omitted; and
- two rows from each eligible symbol's [klines](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#kline-candlestick-data),
  weight 1 for `limit=2`; the current API schema specifies exactly 12 fields per
  tuple.

The local one-minute weight ceiling is 900. Every successful response must also
carry a positive `X-MBX-USED-WEIGHT-1M` value, which the limiter incorporates
conservatively. HTTP 418/429 or a JSON `-1003` opens the run circuit immediately
and records only sanitized status, retry, and ban-time evidence. Ordinary
transport and HTTP failures are not retried inside the same evidence bundle;
the attempt remains partial, avoiding ambiguous request histories.

## Causal and completeness boundary

The CLI requires a specific aligned feature-bar open, one of the preregistered
`1h`, `4h`, or `8h` intervals, and a new output directory. It rejects bars
outside the missingness-aware candidate's registered dataset:

- start: `2027-01-21T00:00:00Z` (`1800489600000` ms); and
- final permitted bar open: `2028-01-20T23:00:00Z`
  (`1832022000000` ms).

Collection must start after the requested bar completes and finish early
enough to retain the declared clock-skew margin inside the following bar. The
deadline plus skew budget must fit within one interval. After the final server
response, publication waits through the declared skew allowance so the
recorded decision time is not earlier than potentially ahead-of-local provider
events.

The exchange-info response is evaluated in its returned order. An eligible
member must be `PERPETUAL`, use the declared quote, be `TRADING`, and have
`onboardDate <= barEndTime`. Every other returned member and every reason for
exclusion are retained. The collector then requires one all-symbol ticker
member and the exact previous/current completed kline pair for every eligible
symbol. It publishes `source-manifest.json` only after this complete sequence.
The independent verifier still recomputes all ticker values, event clocks,
eligibility, close-to-close returns, raw inventories, hashes, and output bytes.

## Provenance and failure behavior

The CLI derives `codeCommit` from the current checkout. Before transport it
requires the collector, verifier, and source/license manifest to be tracked and
unchanged at that commit. Collection status records those file hashes and the
Python runtime. The collector explicitly supplies only `Accept` and a fixed
`User-Agent`; there is no API-key, authorization, body, private endpoint, or
configurable host.

An existing or symlinked output path is never overwritten. Provenance and path
preflight failures occur before a new output is created. After preflight, the
collector creates a new directory, writes `source-manifest.json` only after the
complete response sequence, and marks status complete only after that atomic
write. A clock, deadline, transport, HTTP, redirect, response-size,
content-type, JSON, rate-limit, eligibility, ticker, or kline failure leaves a
sanitized `partial_failure` status and no source manifest. A catchable final
status-write failure removes the new manifest before recording partial failure;
an abrupt stop in that narrow window can leave the manifest only alongside the
earlier `collecting` / `sourceManifestPublished: false` status. Any raw
responses already accepted before a failure are preserved for diagnosis and
cannot be admitted without complete status and separate verification.

## Reproduction without network access

```bash
python3 -m py_compile \
  scripts/research/collect_market_context.py \
  scripts/research/market_context_source.py
node --test \
  test/market-context-collector.test.mjs \
  test/market-context-source.test.mjs
```

The deterministic fake-transport tests perform two byte-identical successful
collections, pass one bundle through the independent verifier, inspect the
exact public host/method/headers/query, and cover existing output, dirty
provenance, pre-registration time, missing shared-IP weight, HTTP 429 circuit,
IP redaction, and partial kline failure. The source-verifier regression also
rejects any kline tuple that differs from the documented 12-field shape. Final
publication fault injection proves a catchable status failure removes the
manifest and an abrupt stop never leaves status falsely claiming publication.

The real CLI shape, for use only after the registered start, is:

```bash
python3 scripts/research/collect_market_context.py collect \
  --output-dir <new-external-bundle-directory> \
  --interval 1h \
  --bar-open-time <aligned-epoch-ms>
```

The bundle must then be verified in a separate command and reviewed for data
admission. Neither command may open the final holdout or authorize a model or
trade.

## Decision

The collection and verification mechanics are ready for the future registered
window, but there is no real observation and no evidence of predictive or
economic value. No forecast, return, cost, drawdown, tail-risk, calibration,
DSR, PBO, robustness, or inference result was produced. **Continue research;
no candidate passed.**

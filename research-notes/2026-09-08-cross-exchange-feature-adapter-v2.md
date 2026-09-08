# Coinbase cross-exchange feature adapter v2

Date: 2026-09-08

Disposition: research infrastructure only; isolated from production

Risk status: `FEATURE-MISSINGNESS-001` remains **OPEN**

## Result

The Haskell research boundary now has an additive
`coinbase_cross_exchange_model_features_v2` contract for same-asset Binance and
Coinbase closes. It reproduces the legacy five cross-exchange formulas when the
input history is complete, while retaining an availability bit and causal
timestamp witnesses for each derived value.

This is not a new predictor and is not connected to the legacy feature builder,
Coinbase fetcher, predictor router, bot, execution, or live-order path. It does
not train a model, evaluate an outcome, open a holdout, change a champion, or
authorize an order.

## Finding in the legacy path

`alignCoinbaseClosesToGrid` is causal in the narrow no-future-bucket sense, but
it is not missingness-preserving:

- a gap after coverage begins receives the most recent earlier Coinbase close
  with no explicit age or availability mask;
- a leading gap receives the contemporaneous Binance close, which manufactures
  a numeric zero basis; and
- the downstream legacy feature block turns absent, out-of-range, degenerate,
  and non-finite components into numeric zero.

The behavior is retained for compatibility. It is unsafe evidence for a new
fitted model because an observed zero basis and an unavailable cross-venue
comparison are indistinguishable. The official Coinbase candle documentation
also warns that historical rates may be incomplete and that no bucket is
published for an interval without ticks. Its response supplies bucket start,
OHLC, and volume, but not a first-seen or publication timestamp
([Coinbase Exchange API: Get product candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)).
The current `CoinbaseCandle` type therefore cannot lawfully be projected into
the v2 contract by inventing an availability time.

Cross-exchange price segmentation itself is economically plausible, but the
literature emphasizes settlement and capital-friction constraints rather than
costless arbitrage ([Makarov and Schoar, 2020](https://doi.org/10.1016/j.jfineco.2019.07.001)).
This adapter is representation infrastructure, not efficacy evidence.

## Exact input contract

`CrossExchangeInputsV2` binds:

- one canonical Binance symbol and the exact Coinbase USD product implied by
  the existing repository mapping;
- a non-empty, contiguous, non-overflowing bar-open grid and positive interval;
- one explicit decision timestamp per bar, from that bucket's end and strictly
  before the following bucket's end;
- one required Binance close record per exact bar; and
- one optional Coinbase close record or explicit absence per exact bar.

Each close record retains bar-open identity, event time, availability time, and
value. Its event time must equal the bucket end; treating the finalized close as
known at bucket open is rejected. A required Binance close must be finite,
positive, and causally usable at its explicit decision or the complete bundle
is rejected. A Coinbase cell with an invalid value or incoherent timing remains
unavailable. A present Coinbase cell whose bar identity differs from its vector
slot is structural corruption and rejects the bundle.

The adapter does not forward-fill Coinbase cells. A source adapter must provide
real event and availability witnesses. The current REST candle fetch does not,
so it remains on the unchanged legacy path.

## Exact model-input interpretation

The value order matches the existing legacy block:

1. `(coinbaseClose - binanceClose) / binanceClose`;
2. one-bar change in that basis;
3. sample-standardized basis over the configured short window;
4. one-bar Coinbase simple return; and
5. Coinbase return minus Binance return.

`featureRowModelInputs` appends five binary availability fields, producing ten
model inputs. Derived timestamps are the maximum event and availability times
of all contributing closes. A feature is available only when every required
operand is available at its own historical decision. In particular:

- the basis level needs both same-bar closes;
- basis change and both returns need complete current and previous evidence;
- return spread needs both venue returns; and
- the basis z-score needs the entire configured window.

Requiring a complete z-score window is a deliberate missingness policy. It is
numerically identical to the legacy formula on complete windows, but unlike the
legacy path it does not silently shorten a window around a missing bucket.

## Verification evidence

`testCrossExchangeFeatureAdapterV2` covers:

- distinct semantic identity and stable five-value order;
- exact numerical parity with the legacy Coinbase block on complete inputs;
- observed-zero versus unavailable-zero behavior;
- no forward-fill across a missing Coinbase bucket;
- delayed optional evidence remaining unavailable;
- event and availability witness propagation;
- five-value plus five-mask layout;
- future-suffix invariance;
- symbol/product, grid, interval, shape, exact-bar, required-value, overflow,
  and z-score-window failure paths; and
- finite outputs on valid inputs.

The automation suite separately proves that production Coinbase, feature,
predictor-router, and main entry modules do not import the new adapter.

## Why the high risk remains open

The repository still lacks a verified Coinbase first-seen or publication-time
artifact. The new module must remain isolated until a source path can retain
real availability evidence, processing lag, revisions, gaps, freshness, exact
instrument identity, and artifact hashes without retroactive reconstruction.
OHLCV optional fields, market context, remaining direct sources, production
builders, model artifacts, and migration/parity behavior also remain outside
the complete v2 contract.

The `missingness_aware_calibrated_shallow_v1` campaign remains blocked on those
items and on genuinely new data beginning 2027-01-21. No development metric or
final holdout was viewed. The recommendation remains **continue research; no
candidate passed**.

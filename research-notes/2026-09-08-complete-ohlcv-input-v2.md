# Complete-case OHLCV input boundary v2

Date: 2026-09-08

Disposition: research infrastructure only; isolated from production

Risk status: `FEATURE-MISSINGNESS-001` remains **OPEN**

## Result

The Haskell research boundary now has an additive
`complete_ohlcv_feature_inputs_v2` contract. It admits a contiguous market
segment only when open, high, low, close, and volume are all present, finite,
causally timed, and structurally coherent. It then provides a narrow projection
into the unchanged legacy `FeatureInputs` OHLCV fields.

On admitted complete data, the existing return, range, ATR, breakout, volume,
and efficiency formulas are byte-for-byte unchanged. Missing core market data
does not receive a synthetic value or an availability mask: the complete input
bundle is unavailable, so a future candidate must abstain or split its dataset
at the gap.

This change does not fit or invoke a predictor, evaluate an outcome, open a
holdout, change a champion, authorize an order, or alter live behavior.

## Existing behavior and scope

`Trader.Predictors.Features.barAt` retains compatibility behavior for callers
that provide close-only data:

- missing open becomes the previous close;
- missing high becomes `max open close`;
- missing low becomes `min open close`; and
- missing or invalid volume becomes `1`.

Those fallbacks keep historical close-only CSV workflows operational, but they
are not admissible evidence for the preregistered missingness-aware learned
candidate. A fabricated range or unit volume can influence 20 legacy kline
features while being indistinguishable from a real observation.

The live exchange loaders already provide source OHLCV and pass it through the
shared market-series integrity gate. The remaining problem is the model/source
contract: `FeatureInputs` itself carries neither availability nor a guarantee
that the optional vectors were present. This v2 boundary adds that guarantee
without changing the live or compatibility paths.

## Exact contract

`completeOhlcvInputsV2` requires:

- a canonical non-empty instrument scope;
- a non-empty exact contiguous bar-open grid and positive interval;
- one explicit decision and first-seen availability timestamp per bar;
- availability from bucket end through the decision;
- a decision from bucket end and strictly before the next bucket's end;
- exact vector shape for all five market fields;
- finite, strictly positive OHLC prices;
- finite non-negative volume; and
- `high >= max(open, close)` plus `low <= min(open, close)`.

The event timestamp for all five finalized bucket values is the checked bucket
end. Raw `FeatureRowV2` witnesses preserve the five required fields, event time,
availability time, decision time, scope, and exact grid. Every availability bit
is true on an admitted row; a missing required field yields no bundle.

The compatibility projection deliberately strips attached derivatives,
external-family, Coinbase, and other context fields. Those sources have
separate v2 contracts and cannot enter this OHLCV boundary through an
unversioned side channel.

## Historical availability boundary

Current historical OHLCV files record bar timestamps and acquisition
provenance, but they do not prove the observation time at which each old bar was
first seen during its original decision interval. This module does not relabel
the later download time or assume an unstated processing lag. A source builder
must use prospective first-seen evidence or another explicitly justified,
preregistered publication contract before it can construct this v2 input.

A late or backfilled bar cannot be admitted to its original decision. Because
the policy is complete-case, a future builder must produce separate contiguous
admissible segments around such gaps rather than forward-fill, interpolate, or
manufacture OHLCV.

## Verification evidence

`testCompleteOhlcvInputsV2` covers:

- semantic schema identity and fixed five-field required order;
- retained scope, grid, bucket-end event, first-seen availability, and decision
  witnesses;
- exact legacy-feature parity on complete OHLCV;
- stripping unrelated optional feature channels;
- future-suffix invariance for raw and derived features;
- missing and wrong-shaped required vectors;
- NaN and both infinity signs across every field family;
- zero prices, invalid high/low relationships, and negative volume; and
- empty, malformed-scope, non-positive-interval, gapped, early/late,
  availability-incoherent, and overflow failure paths.

Automation separately proves the module remains unimported by Binance,
Coinbase, Kraken, Poloniex, legacy feature, predictor-router, and main entry
paths.

## Decision

This is a conservative source boundary, not a validated model. The
`missingness_aware_calibrated_shallow_v1` campaign remains blocked until
timestamp-preserving source artifacts, market-context and remaining direct
source policies, versioned production/model artifacts, and prospective data
beginning 2027-01-21 are available. No development metric or final holdout was
viewed. **Continue research; no candidate passed.**

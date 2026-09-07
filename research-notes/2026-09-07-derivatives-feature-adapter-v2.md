# Derivatives feature adapter v2

Date: 2026-09-07  
Disposition: research infrastructure only; isolated from production  
Risk status: `FEATURE-MISSINGNESS-001` remains **OPEN**

## Result

The Haskell research boundary can now translate already-decoded
`binance_derivatives_first_seen_v2` rows into the additive
`binance_derivatives_model_features_v2` contract. This closes only the narrow
decoder-to-feature representation gap for the four public Binance derivatives
families. It does not create a predictor, load an artifact, evaluate an outcome,
open a holdout, change a champion, authorize an order, or alter live behavior.

The adapter is not imported by the legacy feature builder, predictor router,
exogenous-fetch bridge, CLI, bot, or execution entry point. Its output is
therefore opt-in offline research evidence only.

## Exact model-input interpretation

The value vector deliberately preserves the existing dense derivatives formula
order:

1. `funding.level`: current usable funding value.
2. `funding.delta`: current usable funding minus the preceding usable funding.
3. `open_interest.relative_delta`: `(current - previous) / abs(previous)` when
   both observations are usable; zero when the usable previous value is within
   `1e-12` of zero.
4. `basis.level`: current usable basis value.
5. `taker_flow.centered`: current usable taker ratio minus one.

`featureRowModelInputs` then appends five ordered binary availability values, so
the complete model vector has ten elements. A real observed value of zero has a
mask of one. Unavailable evidence has a finite neutral dense value of zero and a
mask of zero. Event and availability timestamps are retained as parallel
witnesses in `FeatureRowV2`; they are metadata, not additional model inputs.

Every field is optional at this boundary. Level features require one usable
current cell. Delta features require usable preceding and current cells and use
the later event and availability timestamps as their causal witness. A usable
cell must be observed, fresh, finite, and satisfy
`0 <= eventTime <= availabilityTime <= decisionTime`.

## Structural and failure contract

The adapter accepts only:

- A positive interval.
- A non-empty chronological panel.
- One exact canonical uppercase ASCII-alphanumeric symbol.
- `decisionTime = openTime + interval - 1` without arithmetic overflow.
- Exact interval adjacency between successive rows.

An empty, malformed, gapped, or mixed-symbol panel produces no feature rows.
An unusable optional cell produces an unavailable finite neutral cell, never a
directional value. Changing or appending a future row cannot change an earlier
feature-row prefix.

Malformed source CSV still fails at the stricter
`DerivativesPanelSchema` decoder before the adapter. Before any offline
experiment, the external Python `verify-artifacts` command must also validate
the frozen schema-3 collector status and its bar/ledger archive. The pure
Haskell adapter does not and cannot replace that provenance check.

## Verification evidence

`testDerivativesFeatureAdapterV2` covers:

- Semantic schema identity and fixed ordered feature names.
- Parity with the legacy five formulas.
- Model-input value/mask layout.
- Observed-zero versus unavailable-zero distinction.
- Event/availability timestamp retention.
- First-row delta unavailability.
- Stale, non-finite, and future-available optional-cell neutralization.
- Future-row prefix invariance.
- Empty, invalid-interval, malformed-decision, gapped, and mixed-symbol failure.

The root automation regression separately rejects imports of the decoder or
adapter from the production feature builder, exogenous bridge, predictor router,
and main entry point.

## Why the high risk remains open

This adapter covers only verified derivatives-panel rows. The legacy production
path still encodes missing optional inputs densely, and no artifact format binds
the new ten-element schema. Before promotion can become eligible, the repository
still needs:

- Explicit availability policies for OHLCV-derived optional fields, market
  context, Coinbase cross-exchange inputs, and every remaining external source.
- Versioned production feature builders and deterministic artifact compatibility
  checks, with migration and parity tests.
- Fail-closed artifact provenance and required-feature behavior at model load and
  inference boundaries.
- Training, calibration, ablation, latency, and robustness evidence from the
  preregistered prospective period.

The `missingness_aware_calibrated_shallow_v1` campaign remains blocked on those
items and on genuinely new data beginning 2027-01-21. No development outcome or
final holdout was viewed in this change, and the correct recommendation remains
**continue research; no candidate passed**.

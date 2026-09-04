# External panel decoder v2

Date: 2026-09-04

## Result

`Trader.Predictors.ExternalPanelSchema` is a pure, opt-in Haskell decoder for the materialized `external_feature_panel_v2` CSV produced by the research collector. It fixes the exact 40-column order, binds its availability identifier to `feature_availability_v2`, preserves every finite family value and fractional coverage cell, and therefore keeps observed zero distinct from unavailable zero.

The decoder rejects the whole panel when the header changes; a timestamp is negative, duplicated, or out of order; rows mix symbol scopes; a symbol is not in the canonical form emitted by the Python builder; a value or coverage is non-finite; coverage leaves `[0, 1]`; or a zero-coverage cell carries a non-zero value. A small deterministic CSV fixture is consumed by Haskell and checked against the Python column contract.

This is a typed CSV decoder, not a substitute for artifact provenance verification. Offline callers must first run `python3 scripts/research/alternative_data.py verify-panel --manifest ...`; the Haskell module does not validate the manifest or SHA-256 chain.

## Compatibility and safety boundary

The decoder is not imported by `Trader.ExternalData`, feature construction, predictor orchestration, trading, or order execution. It introduces no CLI, environment, API, JSON, saved-configuration, model-identifier, artifact, deployment, or live-order behavior change. The existing external-data path remains the legacy dense projection.

`FEATURE-MISSINGNESS-001` remains **OPEN**. This decoder preserves materialized coverage but does not prove source-specific event and availability provenance inside a trained artifact, establish a versioned prediction-to-position contract, or supply prospective economic evidence. No candidate may consume it outside offline development until a successor integration is preregistered, disabled by default, and evaluated on the frozen protocol beginning no earlier than 2027-01-21.

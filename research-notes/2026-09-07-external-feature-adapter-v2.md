# External-family feature adapter v2

Date: 2026-09-07  
Disposition: research infrastructure only; isolated from production  
Risk status: `FEATURE-MISSINGNESS-001` remains **OPEN**

## Result

The Haskell research boundary can now translate a timestamp-preserving
`ExternalFeatureInputsV2` bundle into the additive
`external_family_model_features_v2` contract. This closes only the pure
alignment-to-feature representation gap for the 19 registered external-data
families. It does not create or fit a predictor, load an artifact, evaluate an
outcome, open a holdout, change a champion, authorize an order, or alter live
behavior.

The adapter is not imported by the legacy feature builder, external fetch
bridge, predictor router, CLI, bot, or execution entry point. Its output is
therefore opt-in offline research evidence only.

## Exact model-input interpretation

Each external family contributes two consecutive values in the existing legacy
order: its aligned level and its one-bar delta. The family order remains:

1. microstructure;
2. options volatility;
3. on-chain;
4. macro;
5. CFTC positioning;
6. news;
7. filings;
8. policy;
9. fundamentals;
10. stablecoin;
11. institutional flows;
12. network;
13. developer;
14. governance;
15. attention;
16. social;
17. prediction market;
18. real world; and
19. security.

The value vector therefore has 38 elements. `featureRowModelInputs` appends 38
ordered binary availability values for a 76-element model vector. A real
observed zero has a mask of one. Unavailable evidence has a finite neutral value
of zero and a mask of zero. Event and availability timestamps are retained as
parallel metadata witnesses and are not added to the model vector.

A level is available only when its aligned value, mask, event time, and
availability time are coherent at that bar decision. A delta is available only
when both the preceding and current levels were causally available at their own
decisions. Its timestamp witness is the later of the two inputs.

## Structural and failure contract

The adapter requires:

- The exact bar-open vector and interval retained by the opaque aligned bundle,
  on a non-empty contiguous grid.
- A positive interval and non-overflowing decision times.
- Every present aligned family series to match the complete grid across values,
  masks, event times, and availability times.
- Every admitted cell to be finite and satisfy
  `0 <= eventTime <= availabilityTime <= decisionTime`.

An empty, gapped, invalid, overflowing, or shape-mismatched grid produces no
feature rows. A missing or unusable optional family cell produces an unavailable
finite neutral field, never a directional value. Extending observations and the
bar grid cannot alter an earlier feature-row prefix.

## Deliberate boundary: no synthetic timestamps

The current `external_feature_panel_v2` CSV contains the decision timestamp,
each family value, and fractional coverage, but it does not contain the selected
source event and availability timestamps. It therefore cannot be converted to
this adapter without losing provenance or inventing timestamps. In particular,
the decision time must not be relabeled as a source event or release time.

Future panel-to-model work requires an explicit new panel/artifact schema that
retains those selected witnesses, deterministic reconstruction in
`verify-panel`, and a matching Haskell decoder. The existing panel v2 decoder
remains isolated and unchanged.

## Verification evidence

`testExternalFeatureAdapterV2` covers:

- Semantic schema identity, 19-family order, and fixed 38-value width.
- Parity with the legacy level/delta formulas.
- The 38-value plus 38-mask model-input layout.
- Observed-zero versus unavailable-zero behavior.
- Event/availability retention for levels and deltas.
- First-observation delta unavailability.
- Future-observation prefix invariance.
- Empty, zero-interval, gapped, shifted-same-length, overflowing, and
  shape-mismatched failure.
- Finite public values on the valid path.

The root automation regression separately rejects imports of the panel decoder
or model adapter from production external-data, feature, exogenous-fetch,
predictor-router, and main entry paths.

## Why the high risk remains open

The aligned v2 type is not yet a complete artifact or production contract.
Before promotion can become eligible, the repository still needs:

- A timestamp-preserving, hash-bound external panel successor and verifier.
- Source-specific release, revision, staleness, and retention policies for every
  enabled direct source.
- Equivalent policies for OHLCV-derived optional fields, market context, and
  Coinbase cross-exchange inputs.
- Versioned production builders and deterministic artifact compatibility,
  corruption, migration, and parity tests.
- Training, calibration, ablation, latency, and robustness evidence from the
  preregistered prospective period.

The `missingness_aware_calibrated_shallow_v1` campaign remains blocked on those
items and on genuinely new data beginning 2027-01-21. No development outcome or
final holdout was viewed in this change. The correct recommendation remains
**continue research; no candidate passed**.

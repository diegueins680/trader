# External feature availability v2 boundary

Date: 2026-09-04

## Result

The Haskell research boundary now has a typed, opt-in external-family bundle that retains dense values, observed masks, selected event timestamps, and selected availability timestamps. Exact duplicate releases retain the legacy averaging rule. Revisions remain distinct and become usable only after their availability time; a late revision of an older event cannot displace a newer event.

The existing `alignedExternalFeatureInputs` behavior is preserved as a compatibility projection that sets `eventTime == availabilityTime` and exposes only dense values. No predictor, artifact, saved configuration, model identifier, champion, combo, deployment setting, authorization flag, or order path uses the v2 bundle.

## Source audit

The offline Python alternative-data cache already records `eventTime`, availability `timestamp`, `ingestedAt`, revision identity, and whether availability is provider-explicit or first-seen. Its generated panels also carry family coverage columns. Those are the correct raw materials for a later v2 research adapter.

The direct Haskell source adapters are not yet sufficient for that migration:

- generic CSV/JSON ingestion collapses the source timestamp into one time;
- FRED applies a configured lag to the observation date but does not retain the original event time at the Haskell boundary;
- Deribit, Glassnode, GDELT, and SEC direct adapters do not all expose a provider-vintage or first-seen timestamp suitable for historical revision reconstruction;
- SEC `filingDate` is not an exact public-acceptance timestamp;
- the legacy panel reader does not consume the existing `*_coverage` columns.

Consequently, direct-source rows must not be relabeled as v2 evidence. A production migration requires a source-by-source timestamp policy, prospective first-seen capture where provider release evidence is unavailable, staleness rules, coverage-column admission, schema-bound artifacts, and new walk-forward evidence.

## Executable evidence

`testExternalDataFeatureInputs` now covers:

- revision availability and selected timestamp preservation;
- old-event late revisions versus newer events;
- observed zero versus pre-coverage missingness;
- exact duplicate-release aggregation;
- future-only and non-finite family exclusion; and
- legacy dense alignment regression behavior.

`FEATURE-MISSINGNESS-001` remains **OPEN**. The v2 bundle is plumbing for future research, not evidence that the registered missingness-aware candidate is fit, validated, or promotion-eligible. The candidate remains blocked on genuinely new data beginning 2027-01-21 and disabled for shadow, paper, and live use.

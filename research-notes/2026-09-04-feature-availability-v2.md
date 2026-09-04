# Feature availability schema v2 — infrastructure result

Date: 2026-09-04.

## Result

The repository now has an additive `feature_availability_v2` contract for future research candidates. It does not change a legacy feature vector, trained model, saved configuration, predictor identifier, champion, combo, deployment mode, or order path.

The contract provides:

- distinct event and availability timestamps;
- point-in-time alignment in availability order, including revisions that become visible only at their later release time;
- unique ordered field names and required/optional declarations;
- finite dense values plus a parallel observed mask, so an observed numeric zero is distinguishable from unavailable evidence;
- no row when a required feature is missing, late, timestamp-incoherent, or non-finite;
- a deterministic schema signature suitable for binding a future artifact to field order and requirements;
- a legacy compatibility projection where the historical single timestamp is used as both event and availability time and the historical dense values remain unchanged.

## Evidence

`testFeatureAvailabilitySchemaV2` verifies observed-zero/missing separation, required-feature abstention, non-finite handling, event/availability ordering, future-data isolation, revision timing, field-name uniqueness, stable signatures, and preservation of availability timestamps and masks. Existing point-in-time alignment and complete Haskell tests remain green.

## Deliberate non-integration

The current external and derivatives builders still pack some aligned `Nothing` values into legacy numeric zeros, and existing model artifacts do not bind an availability-mask schema. Automatically moving them to v2 would change feature dimensions and learned semantics. That migration therefore requires a separate explicit compatibility version, artifact provenance update, numerical parity tests for legacy paths, causal coverage tests for every data family, and prospective evaluation under the registered protocol.

`FEATURE-MISSINGNESS-001` remains **OPEN**. The new primitives remove an implementation blocker but do not make the missingness-aware shallow candidate eligible for fitting, promotion, shadow, paper, or live use. Its future-data boundary and disabled-challenger gates remain unchanged.

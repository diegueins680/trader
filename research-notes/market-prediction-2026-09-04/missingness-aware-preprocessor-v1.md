# Missingness-aware preprocessor artifact v1

## Decision

`missingness_aware_preprocessor_artifact_v1` is an isolated, outcome-blind preprocessing boundary for `missingness_aware_calibrated_shallow_v1`. It is not a predictor, a fitted return model, an experiment result, or permission to inspect the final holdout. The **no candidate passed** decision is unchanged.

## Frozen transformation

For each of the feature panel's 22 ordered values, fitting uses only finite observations whose availability mask is true in the exact registered training prefix. It records the population mean and population standard deviation; a constant column uses scale 1. At transformation time an unavailable optional value is replaced by its training mean, so its standardized value is exactly zero. The ten optional availability masks are then appended unchanged. The output is therefore exactly 32 finite inputs: 22 `standardized.*` values followed by ten `available.*` masks.

Missing required evidence is already rejected by the source feature-panel contract. A feature with no observed training value, a non-finite statistic, a mismatched symbol or interval, a non-contiguous or off-calendar grid, a source row decided before its completed bar, or training evidence first seen after the declared fit time fails closed.

## Artifact and provenance

The strict JSON envelope records schema and compatibility version 1, the source feature schema and signatures, exact output signature, registration and academic origins, code commit, training-data/panel/source/split SHA-256 identities, symbol, interval, horizon, train/validation/holdout boundaries, purge and embargo, fit and creation times, seed, runtime versions, cost-model identity, row and observation counts, and fitted means/scales. A SHA-256 digest covers the complete payload. Unknown fields, corruption, incompatible schemas, unsupported registered values, and any true research-admission, experiment, holdout, model, promotion, deployment, order, or live-trading authority flag are rejected.

The artifact is deterministic for identical request and panel bytes. It is intentionally absent from legacy feature, predictor, ensemble, optimizer, bot, API/CLI, production artifact-loader, execution, deployment, and live-authorization modules.

## Evidence and remaining blockers

Tests use only deterministic synthetic future-dated rows. They establish feature order, observed-only fit behavior, neutral imputation, unchanged masks, grid and scope isolation, serialization round-trip, digest enforcement, and safe rejection. No public or private market dataset, outcome label, development result, or holdout was read.

Prospective collection cannot begin before 2027-01-21. Every frozen source must still be independently verified and admitted. Each walk-forward training prefix must then create its own preprocessor artifact before fitting a separately versioned return-model artifact. The registered 117-configuration budget, statistical gates, costs, final-holdout policy, and disabled-challenger lifecycle remain unchanged.

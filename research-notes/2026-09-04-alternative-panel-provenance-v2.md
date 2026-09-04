# Alternative-data panel provenance v2

Date: 2026-09-04

## Result

Generated alternative-data panels now carry an additive `external_feature_panel_v2` manifest contract. The existing top-level manifest fields remain available, while the versioned block fixes the exact ordered columns, binds coverage to `feature_availability_v2`, and records SHA-256 hashes for the source cache, bar grid, and output panel.

`verify-panel` validates the recorded bytes, recomputes the eligible observation and grouped family-metric counts from the cache and declared symbol scope, independently checks the CSV header, row count, strictly increasing decision timestamps, finite family values, coverage in `[0, 1]`, and the rule that zero coverage cannot accompany a non-zero feature, and reconstructs the panel byte for byte from the bound inputs. Updating a digest after changing the panel therefore does not authorize the change.

The panel bytes are deterministic for fixed cache, bars, interval, and symbol. The manifest intentionally records `generatedAt` and absolute artifact locations, so the complete manifest is provenance-bearing rather than byte-identical across runs. Relocated artifacts may be supplied to verification as overrides, but their bytes must match the frozen digests.

## Boundary

This is offline research artifact validation only. It does not make direct Haskell adapters source-causal, connect coverage to production predictors, change any model dimension or identifier, fit a model, open a holdout, promote a challenger, alter live authorization, or place an order.

`FEATURE-MISSINGNESS-001` remains **OPEN**. Before the missingness-aware candidate can be fitted, a separate adapter must consume verified panel coverage under a new model/artifact compatibility version, preserve source-specific availability provenance, and pass the frozen prospective protocol beginning 2027-01-21.

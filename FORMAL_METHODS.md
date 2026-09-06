# Focused Formal Methods Contracts

The canonical repository-wide specification registry is `formal/specifications.json`, documented in `docs/formal-specifications.md`. This file provides deeper proof sketches for selected trading-critical contracts and must remain consistent with that registry and executable evidence.

## Formal conformal calibration contract

`fitConformal` and `predictInterval` in `haskell/app/Trader/Predictors/Conformal.hs` are treated as the fail-closed calibration boundary for conformal prediction intervals and any confidence or trading gate derived from those intervals.

Clauses:

1. A conformal calibration residual sample is admissible only when the sample is non-empty and every absolute residual is finite and non-negative.
2. Empty calibration evidence, negative residuals, `NaN`, positive infinity, negative infinity, or a residual list containing any malformed sample produces an unavailable conformal model with `cmCount == 0` and an unbounded prediction interval.
3. An unavailable conformal model, a malformed model radius, or a malformed point forecast must return `(-Infinity, Infinity, Nothing)` from `predictInterval`, so malformed calibration evidence cannot emit artificially tight interval bounds or a positive sigma-derived confidence estimate.
4. Valid zero residual samples remain admissible. They may produce a zero-width interval around a finite point forecast, but `sigmaFromInterval` remains unavailable for non-positive interval width.
5. For a fixed finite point forecast and valid non-empty calibration evidence, interval width is `2 * cmRadius`, so widening the selected conformal residual quantile is monotone non-decreasing in the produced interval width.

Bounded executable obligations:

- The Haskell test suite is the bounded regression harness for this invariant. It must explicitly name and cover all five obligations: (1) empty calibration evidence yields `cmCount == 0` and an unbounded interval; (2) malformed residual evidence, including mixed valid/malformed samples that contain negative, `NaN`, `+Infinity`, or `-Infinity` residuals, is rejected whole and yields `cmCount == 0` with `(-Infinity, Infinity, Nothing)`; (3) valid zero residual evidence remains admissible and produces a zero-width interval; (4) an unavailable model or a zero-width interval yields no sigma (`Nothing`); and (5) interval width stays `2 * cmRadius` and widens monotonically (non-decreasing) as the selected conformal residual quantile rises.

Proof sketch:

- `admissibleResiduals` is the calibration validity boundary: it rejects the whole sample when the list is empty or any residual is negative, `NaN`, or infinite. Therefore malformed residuals cannot be silently dropped to leave a smaller, overconfident empirical quantile.
- `fitConformal` only calls `conformalRadius` after that boundary has proven the sample non-empty and every residual finite and non-negative. The selected radius is therefore a quantile over admissible evidence only.
- On rejected evidence, `fitConformal` returns an unavailable model with `cmCount == 0` and an infinite radius. `predictInterval` maps unavailable models, malformed radii, and malformed point forecasts to an unbounded interval with `Nothing` sigma, which is conservative for downstream hold-style gates.
- Valid zero residuals satisfy the same finite non-negative admissibility predicate, so the zero boundary remains legal while still preventing sigma confidence from being synthesized from a non-positive interval width.
- For valid evidence and fixed `mu`, `predictInterval` computes `[mu - radius, mu + radius]`; the width is exactly `2 * radius`. Because `conformalRadius` selects from sorted finite non-negative residuals, increasing the selected residual quantile cannot narrow the interval.

## Formal quantile interval admissibility contract

`predictQuantiles` and `sigmaFromQ1090` in `haskell/app/Trader/Predictors/Quantile.hs` are treated as the fail-closed quantile interval boundary for quantile-derived confidence and trading gates.

Clauses:

1. A quantile prediction interval is admissible only when the model exposes one positive consistent feature dimension, the input feature vector has that same length, and the raw q10, q50, and q90 predictions are finite.
2. Lower and upper quantile evidence must be ordered. The boundary `q10 <= q90` is admissible, including equality; inverted evidence `q10 > q90` is unavailable and must not be repaired by sorting.
3. Empty model evidence, inconsistent model dimensions, feature-length mismatch, non-finite raw quantile predictions, or inverted lower/upper evidence must return `Nothing` from `predictQuantiles`, so corrupted quantile evidence cannot emit a tight or directionally misleading usable interval.
4. Once admissibility is proven, q50 may be clamped into `[q10, q90]` for downstream interval consistency while the unclamped q50 forecast remains observable in the returned tuple.
5. `sigmaFromQ1090` may return a sigma estimate only for positive finite interval width. Valid equality at `q10 == q90` keeps the interval admissible but leaves sigma unavailable.
6. For fixed finite q50 evidence and ordered finite bounds, widening the quantile spread cannot narrow the emitted interval; increasing `q90 - q10` also cannot decrease a positive sigma estimate.

Bounded executable obligations:

- `testQuantileIntervalAdmissibilityContract` is the bounded regression harness for this invariant. It explicitly covers empty models, inconsistent head dimensions, feature-length mismatch, overflowed raw q10/q50/q90 predictions, inverted q10/q90 rejection without sorting, the valid equality boundary with `Just (q10, q10, q10, q50Raw, Nothing)`, and monotone non-narrowing / non-decreasing positive sigma as ordered q10/q90 spreads widen.
- Direct `sigmaFromQ1090` assertions inside that regression cover zero-width, negative-width, and non-finite-width fail-closed behavior.

Proof sketch:

- `quantileFeatureDim` is the first evidence boundary: every quantile head must expose the same positive feature dimension and the forecast vector must match it, otherwise `predictQuantiles` returns `Nothing` before any raw prediction can be trusted.
- `admissibleQuantilePrediction` is the raw-output validity boundary: it rejects the entire interval when any raw quantile is `NaN` or infinite, including overflow from otherwise finite model coefficients and feature values, so malformed q50 evidence cannot be hidden by clamping and malformed q10/q90 evidence cannot define finite bounds.
- The ordered-bound guard rejects `q10 > q90` instead of sorting the pair. Therefore a crossing quantile model cannot reverse directionality or synthesize a usable interval from contradictory lower/upper evidence.
- Equality at `q10 == q90` satisfies the ordered-bound predicate and produces a zero-width interval with `Nothing` sigma because `sigmaFromQ1090` requires positive finite width. This preserves the valid deterministic boundary without creating false confidence from zero spread.
- After admissibility, the returned lower and upper bounds are exactly the raw q10 and q90 values, so interval width is exactly `q90 - q10`. Widening ordered bounds can only preserve or increase that width, and sigma is width divided by a positive constant when width is positive, so a wider admissible spread cannot decrease a positive sigma estimate.

## Formal exogenous alignment contract

`alignToBars` in `haskell/app/Trader/Predictors/Exogenous.hs` is treated as the point-in-time boundary for irregular exogenous market features before they are attached to predictor inputs.

Clauses:

1. An exogenous observation is admissible only when its value is finite. Negative finite values remain admissible because funding and basis evidence can legitimately be negative.
2. A bar grid is admissible only when bar open times are strictly ascending and the interval is positive.
3. For an admissible grid, the value emitted for bar `i` is the most recent admissible observation whose timestamp is at or before `openTime_i + intervalMs - 1`.
4. Bars with no prior admissible observation emit `Nothing`; neutral filling may later map that absence to `0`.
5. A malformed grid, non-positive interval, or overflowed close-time witness emits only `Nothing`, so malformed time evidence cannot create look-ahead features.

Bounded executable obligations:

- `testAlignToBarsPointInTime` witnesses unsorted observation input, forward fill across gaps, and no leakage from a future observation into an earlier bar.
- `testAlignToBarsFailClosedOnMalformedInputs` witnesses non-finite observation filtering, non-positive interval fail-closed behavior, descending and duplicate bar-grid rejection, and close-time overflow fail-closed behavior.

Proof sketch:

- Sorting the observation stream by timestamp and consuming it monotonically ensures each output bar can only see observations at or before that bar's close.
- Filtering non-finite values before the monotone scan means malformed readings cannot become feature evidence or erase the last finite reading.
- Strictly ascending bar opens and a positive interval are checked before scanning; if they fail, the function returns an all-`Nothing` vector with the same length as the requested grid.
- The close-time overflow guard rejects any bar whose computed close precedes its open, preserving the same fail-closed absence semantics for arithmetic corruption.

## Formal external-data admissibility contract

`externalCellDouble`, `externalCellTime`, `externalSymbolMatches`, and `alignedExternalFeatureInputs` in `haskell/app/Trader/ExternalData.hs` are treated as the fail-closed admission boundary for external CSV or panel data before those values become predictor inputs.

Clauses:

1. An external numeric cell is admissible only when it parses to a finite `Double`. Blank, malformed, `NaN`, `+Infinity`, and `-Infinity` cells are unavailable and must not become numeric feature evidence.
2. An external time cell is admissible only when it parses to a timestamp, and a symbol-scoped row is admissible only when `externalSymbolMatches` resolves that scope against the target full/base asset. When the target symbol is unresolved, only global rows are admissible.
3. A feature family becomes admissible only when at least one finite observation survives symbol/timestamp parsing and is causally available on or before some requested bar close. Future-only or otherwise no-overlap rows must remain unavailable rather than synthesizing an all-zero aligned feature family.
4. Once at least one admissible observation overlaps the requested bar grid, missing leading bars may still neutral-fill to `0`; that neutralization is evidence-preserving only after admissibility has been proven.
5. Valid global rows remain admissible even when the same input batch also contains malformed, non-finite, future-dated, or unresolved symbol-scoped rows.

Bounded executable obligations:

- `testExternalDataFeatureInputs` covers blank and malformed numeric/time cells, rejection of non-finite cells, rejection of future-only aligned evidence, unresolved symbol-scoped exclusion when the target symbol is unknown or mismatched, and preservation of valid global rows.

Proof sketch:

- `externalCellDouble` and `externalCellTime` are the first parse boundary: they reject blank or malformed CSV/panel cells before the row can contribute any numeric or temporal evidence.
- `externalSymbolMatches` is the scope boundary: non-empty symbol-scoped rows require a resolved full/base asset match, so unresolved or mismatched scoped rows cannot leak into another asset's signal vector.
- `alignedExternalFeatureInputs` groups only finite observations, aligns them point-in-time, and now emits a feature family only when the aligned series contains at least one admissible observation witness. Therefore future-only rows cannot opt the model into a synthetic all-zero exogenous family.
- Neutral fill is applied only after that admissibility witness exists, so it preserves the meaning of "missing before first admissible observation" without manufacturing evidence from malformed or causally unavailable rows.
- Because admissibility is checked per family before bundling, valid global observations are preserved even when other rows in the batch are rejected fail closed.

## Formal Coinbase candle range contract

`buildRanges` in `haskell/app/Trader/Coinbase.hs` constructs paged historical candle windows before HTTP requests are issued.

Clauses:

1. Every emitted range has non-negative seconds and satisfies `start <= end`.
2. Requested page spans are computed with saturating arithmetic, so `bars * granularity` overflow cannot wrap into a negative or forward-moving range.
3. If a requested span reaches or exceeds the requested end time, the first page starts at the Unix origin and range construction terminates.
4. The Coinbase per-request page cap remains the upper bound for ordinary non-overflowing pages.

Bounded executable obligation:

- `testCoinbaseBuildRangesOverflowRegression` witnesses ordinary two-page construction and an overflowing `granularitySec` request saturating to `(0,end)` instead of emitting an inverted range.

Proof sketch:

- Each page span is clamped through `Integer` arithmetic to `maxBound :: Int64` before returning to `Int64`.
- Range subtraction compares through `Integer`; a span greater than or equal to the current end time returns `0`, otherwise ordinary subtraction is safe because `span < end`.
- The recursive step stops whenever the next start reaches `0`, so saturated spans cannot produce extra malformed pages.

## Formal backtest cost-attribution contract

The backtest JSON `costAttribution` and `costs.attribution` surfaces emitted by `haskell/app/Trader/Trading.hs` are treated as the accounting reconciliation contract for realized trading costs in one emitted simulation run.

Clauses:

1. For each emitted aligned equity point, `gross` is the realized-run reconciliation curve computed by adding cumulative realized costs back to `net`.
2. The emitted consistency residual must witness the accounting identity `gross - cumulative realized costs = net`, up to the documented finite residual tolerance for the run.
3. Realized cost totals are attributed across fee, slippage, spread, and funding buckets, and their sum defines the cumulative realized cost surface used by the reconciliation identity.
4. The `gross` reconciliation curve is not a separately replayed no-cost counterfactual equity curve or return series. A no-cost replay would generally compound on a different capital base and can diverge from `net + cumulative realized costs`.
5. Consumers that need no-cost performance must run or request a distinct no-cost simulation rather than interpreting the cost-attribution `gross` surface as cost-free strategy performance.

Bounded executable obligations:

- The backtest JSON cost-attribution residual is the bounded verification artifact for this contract: it proves the emitted run's `gross - cumulative realized costs = net` accounting identity and bounds any residual drift in the serialized output.

Proof sketch:

- The emitted net curve is the realized equity path after modeled costs have been applied, so realized fee, slippage, spread, and funding buckets are attributable deltas against that same run.
- Adding cumulative realized costs back to net produces an accounting reconciliation surface for the emitted path. Subtracting those same cumulative realized costs must therefore recover net, modulo finite serialization and arithmetic residuals.
- Because the simulator does not replay all subsequent returns on a higher no-cost capital base for this surface, the reconciliation curve cannot be treated as a cost-free counterfactual. The contract is cost attribution for the realized run, not alternate-world performance without costs.

## Formal automatic-graduation session-bounded equity contract

Automatic graduation net-return evidence in `haskell/app/Main.hs` is admissible only when each reviewed worker's model-equity chain stays within one bot session across the evidence window.

Clauses:

1. A reviewed worker's model-equity chain is admissible only when every sampled equity is finite and every sample in that compared chain shares one session identifier.
2. A session change inside the evidence window is a hard boundary. Pre-restart and post-restart model-equity levels must not be compared as one continuous return chain because a bot restart can rebase model equity near `1.0`.
3. When a reviewed worker changes session inside the evidence window, automatic-graduation return evidence must split at that boundary or reject the affected worker fail-closed for graduation.
4. For one admissible session, net-return evidence is the last finite model-equity level minus the first finite model-equity level from that same session only.
5. This boundary must not allow a restart-rebased worker to manufacture positive fleet-return evidence from cross-session stitching.

Bounded executable obligation:

- `testGraduationEquitySessionBoundary` witnesses that single-session reviewed-worker evidence remains admissible, while a worker that loses equity, restarts mid-window, and resumes near `1.0` is rejected instead of being stitched into a false positive fleet return.

Proof sketch:

- Model equity is session-relative, so a restart can reset the baseline independently of prior losses. Comparing pre-restart and post-restart levels directly therefore breaks return additivity and can manufacture gains.
- Treating the session identifier as part of the evidence boundary preserves only within-session differences; once the identifier changes, the surrounding levels are no longer on the same equity scale.
- Splitting or rejecting at the first session boundary is conservative: it can suppress unsafe graduation evidence, but it cannot create a positive fleet-return witness from cross-session data.

## Formal portfolio-graduation boundary-fresh equity contract

`portfolioGraduationFleetEquities`, `portfolioGraduationPerformance`, and `portfolioGraduationReview` in `haskell/app/Trader/PortfolioSelection.hs` are treated as the fail-closed boundary for automatic portfolio graduation evidence before a reviewed set can remain eligible for graduated selection.

Clauses:

1. Baseline evidence is admissible only when every reviewed UUID has exactly one finite positive baseline no later than `boundaryMs` and no older than `maximumBaselineAgeMs`. Exact equality at `atMs == boundaryMs` and `boundaryMs - atMs == maximumBaselineAgeMs` remains admissible.
2. Daily equity evidence contributes to fleet performance only when a retained day contains one finite positive sample per reviewed UUID. Unknown UUIDs, duplicate samples, stale baselines, missing reviewed baselines, or malformed baseline/daily equity fail closed; incomplete days are discarded instead of being counted as evidence.
3. For each retained day, relative fleet equity is `1 + sum (current_i / baseline_i - 1)`, so only boundary-fresh baselines anchor the compared equity levels.
4. `portfolioGraduationPerformance` accepts only finite positive fleet-equity chains; malformed or non-positive aggregate equity fails closed before net return or drawdown is computed.
5. `portfolioGraduationReview` keeps exact equality at the minimum net-return floor pending (`netReturn <= minimumNetReturn`), while equality at the maximum drawdown cap and equality at the execution/status reliability minima remain admissible.

Bounded executable obligations:

- `testGraduationPortfolioBoundaryContract` covers exact admissibility at `boundaryMs` and at the freshness boundary `boundaryMs - maximumBaselineAgeMs`, stale-baseline rejection, missing-baseline rejection, malformed daily-equity rejection, strict equality at `pgcMinimumNetReturn`, and inclusive equality at drawdown and reliability thresholds.

Proof sketch:

- `insertBaseline` is the freshness boundary: it accepts only reviewed UUIDs with one finite positive baseline satisfying `0 <= atMs <= boundaryMs` and `boundaryMs - atMs <= maximumBaselineAgeMs`, which preserves the exact boundary equalities while rejecting older evidence.
- `insertDaily` rejects unknown UUIDs, duplicates, and non-finite or non-positive equity, so malformed raw levels cannot become admissible fleet evidence.
- `completeDays` only emits days that contain every reviewed UUID, and `fleetEquity` compares those complete days only against the boundary-fresh baselines, which prevents missing or cross-scale evidence from manufacturing return.
- `portfolioGraduationPerformance` rejects malformed or non-positive fleet-equity chains before computing return or drawdown, so aggregate corruption cannot propagate into review thresholds.
- `portfolioGraduationReview` uses a strict comparison for minimum net return and inclusive comparisons for drawdown and reliability bounds, making the threshold-equality behavior explicit and executable.

## Formal fee-aware entry gate contract

`normalizeSignalOpenThreshold`, `signalEntryHeadroomThresholdCap`, `normalizeSignalEntryEdge`, and `signalEntryFeeBufferOk` in `haskell/app/Trader/SignalGates.hs`, as wired into the repaired `mkEntryGateState` binding block in `haskell/app/Trader/Trading.hs`, are treated as the shared fail-closed fresh-entry threshold boundary, canonical optimizer headroom-cap witness, raw-edge normalization boundary, and marginal-entry veto.

`BacktestResult`, `EnsembleConfig`, `StepMeta`, `IntrabarFill`, `Positioning`, `simulateEnsembleVWithHLChecked`, `ExitReason`, and `Trade`, as exported from `haskell/app/Trader/Trading.hs`, are treated as the stable public simulation/result/CLI surface consumed by `Trader.App.Args`, `Trader.Optimization`, `Trader.Metrics`, and related optimizer code.

Clauses:

1. Let `normalizeSignalOpenThreshold openThreshold = Just threshold`, `requiredHeadroom = 1.5 * threshold`, and `requiredEdge = requiredHeadroom + roundTripFeeFloor`. An entry is admissible only when the raw open threshold is finite and non-negative, the normalized threshold is finite and non-negative, the fee floor is finite and non-negative, the edge sample is explicit (`Just`) and finite, and `edge >= requiredEdge`; this explicit-edge obligation still applies when `requiredEdge == 0`.
2. `signalEntryHeadroomOk openThreshold` remains the zero-fee specialization `signalEntryFeeBufferOk openThreshold 0`.
3. For fixed `openThreshold` and edge, admissibility is monotone non-increasing as any valid `roundTripFeeFloor` rises.
4. For fixed `openThreshold` and fee floor, admissibility is monotone non-increasing as raw edge falls.
5. Missing edges, negative open thresholds, non-finite open thresholds, negative fee floors, non-finite fee floors, or malformed edge samples presented directly to `signalEntryEdgeSpikeOk`, `signalEntryHeadroomOk`, or `signalEntryFeeBufferOk` are fail-closed.
6. Once a state is blocked at fee floor `f`, it remains blocked for every valid `f' >= f`; malformed fee data and malformed or negative threshold data also cannot reopen the entry.
7. In `mkEntryGateState`, the spike, headroom, and fee-buffer checks are consulted only when `needsEntry` is true, each threshold-consuming check must pass through `normalizeSignalOpenThreshold`, each check receives the same `entryEdge` sample computed once as `normalizeSignalEntryEdge edgeRaw`, `normalizeSignalEntryEdge` preserves finite non-negative inputs, collapses finite negative raw edges to `Just 0`, maps non-finite raw edges to `Nothing`, and the three booleans are combined conjunctively before `desiredSide1` can keep a fresh entry alive.
8. In `simulateEnsembleVWithHLChecked`, a fresh entry must first prove the configured position-sizing exposure cap and floor are finite and non-negative. Invalid cap/floor evidence (`NaN`, infinity, or negative values) maps the effective fresh-entry cap and floor to zero before the final candidate size can become admissible. With valid sizing bounds, a zero cap remains a legal no-entry boundary, a zero floor remains legal, equality at the minimum floor remains admissible, and tightening a valid cap is monotone non-increasing for realized fresh-entry exposure. A fresh entry must also pass the same spike, headroom, and fee-buffer conjunction before sizing overlays can open exposure. The fee-buffer floor is the modeled round-trip cost rate for the final overlay-scaled candidate entry size, so a signal that is threshold/headroom-valid but below its sized round-trip costs remains flat.
9. In the live `Main` latest-signal path, the chosen direction must pass `signalEntryFeeBufferOk openThrAdj sizedRoundTripCost edgeForMethod` in addition to the existing spike and headroom checks after final entry-size overlays. A rejection from that predicate is observable as `EDGE_FEE_BUFFER` after spike/headroom reasons are ruled out.
10. Deploy-config normalization in `haskell/web/src/lib/deployConfig.ts` is non-interfering with trading admissibility: blank or missing Fly host inputs may normalize to the default `fly.dev`, malformed Fly app/host overrides are rejected instead of being coerced into a fallback target, and the resulting `apiFallbackUrl` synthesis does not feed `signalEntryHeadroomOk`, `signalEntryFeeBufferOk`, `signalEntryEdgeSpikeOk`, or the fresh-entry conjunction.
11. Every CLI, optimizer, or metrics consumer must be able to import `signalEntryHeadroomThresholdCap` through `Trader.SignalGates` and `IntrabarFill(..)`, `Positioning(..)`, `BacktestResult`, `EnsembleConfig`, `StepMeta`, `simulateEnsembleVWithHLChecked`, `ExitReason`, and `Trade` through `Trader.Trading`; restoring those exports is a visibility-only repair and does not alter fresh-entry gating, optimizer candidate generation, ensemble simulation behavior, trade construction, exit classification, or backtest aggregation behavior.
12. For any raw edge sample, `signalEntryHeadroomThresholdCap` applies the same fail-closed normalization boundary as `normalizeSignalEntryEdge` and returns the maximum open-threshold witness compatible with the headroom gate; malformed or negative raw edges collapse to cap `0`.
13. Deploy-log-driven backend repair is non-interfering with trading admissibility: a failed Fly backend Docker/cabal build log is actionable repair context rather than pending or skippable CI noise, and any backend Haskell repair selected from that log must first restore parser/build validity before changing trading semantics or weakening the fresh-entry gate predicates above.
14. For every enabled `Trader.VolConfGate` preset, volatility evidence is admissible only when it is present, finite, non-negative, and at most the configured evidence max (`2.0` by default). Missing, negative, non-finite, or above-range volatility evidence maps to `VolConfGateAllowExitOnly` with size multiplier `0`, so malformed volatility cannot be normalized into low-volatility entry permission.
15. For every enabled `Trader.VolConfGate` preset, provided confidence evidence is admissible only when it is finite and within `[0,1]`. Missing confidence remains weak entry-blocking evidence, but provided negative, non-finite, or above-range confidence maps to `VolConfGateAllowExitOnly` with size multiplier `0` instead of a weak hold cell.
16. Valid volatility-confidence boundary equality is preserved: volatility bucket boundaries remain inclusive on the higher bucket, confidence weak/strong thresholds remain inclusive on the stronger bucket, the default volatility upper bound `2.0` remains valid, and entry admissibility is monotone non-increasing when moving from default to stricter confidence requirements or from looser to tighter high-volatility requirements on the same bounded witness.
17. `VolConfGateHold` is stateful: `volConfStatefulCloseDirection` suppresses the stateless close-direction projection, `desiredPositionForSignalWithVolConf` preserves an existing live position, and `applyVolConfGateBehavior` preserves the same simulated side and size. Removing `Hold` restores ordinary neutral-signal exit behavior.

Bounded executable obligations:

- `testSignalGateEntryHeadroomSpecializesFeeBuffer` preserves the legacy headroom boundary cases, including the valid zero-threshold explicit-edge boundary and the zero-fee specialization.
- `testTradingEntryGateFailClosedMonotone` covers equality at the fee-aware boundary and witnesses the non-increasing admissibility ladders as fees rise or raw edge falls.
- `testSignalGateEntryFeeBufferFailsClosed` covers negative and non-finite open thresholds, negative and non-finite fee floors, non-finite edges, and negative edge samples.
- `testNormalizeSignalEntryEdgeFailClosedRegression` witnesses that the restored public `normalizeSignalEntryEdge` helper preserves valid fresh-entry edges, collapses finite negative raw edges to the shared `Just 0` sample, maps non-finite raw edges to `Nothing`, and keeps the Trading conjunction fail closed when that shared sample is reused.
- `testTradingEntryGateFailClosedMonotone` and `testTradingEntryGateMalformedNoReopen` extend the `mkEntryGateState` witness so negative or non-finite per-side fees cannot reopen a blocked fresh entry after shared edge normalization.
- `testVolConfGateMalformedInputsFailClosed` witnesses valid volatility-confidence boundary equality, fail-closed behavior for missing/malformed/out-of-range volatility, fail-closed behavior for malformed provided confidence, weak entry-blocking behavior for missing confidence, exit-only behavior that cannot open or increase exposure, and monotonicity under stricter confidence and high-volatility requirements.
- `testVolConfHoldPreservesLivePosition` binds the stateful `Hold` contract across the latest-signal helper, live order-intent reducer, and simulator behavior, and proves the same neutral signal exits once `Hold` is absent.
- `testBacktestEntryGateUsesRoundTripFeeBuffer` binds the same contract to the checked simulator: a no-fee headroom-valid entry can open, but the identical prediction remains flat once modeled round-trip costs exceed the available edge, including when fixed costs only become prohibitive after Kelly-lite overlays reduce final entry size.
- `testBacktestFreshEntrySizingBoundsFailClosed` binds fresh-entry sizing validity to the checked simulator by covering negative, `NaN`, and infinite max-position caps and min-position floors, the valid zero-cap no-entry boundary, valid zero-floor admissibility, valid minimum-floor equality, cap-below-floor rejection, and monotone non-increasing exposure as valid caps tighten.
- `testSignalGateEntryEdgeSpikeCapRegression` covers equality at both active caps, strict-above-cap rejection, and malformed threshold/edge fail-closed behavior for the independent spike veto.
- `testOptimizerPublicSurfaceRegression` imports `signalEntryHeadroomThresholdCap`, `EnsembleConfig(..)`, `StepMeta(..)`, and `simulateEnsembleVWithHLChecked` through their public modules, witnesses the `0.03 -> 0.02` headroom-cap boundary, and fails before `optimize-equity` CI build time if the optimizer-facing public surface narrows again.
- `testMetricsConsumesTradingPublicResults` constructs `BacktestResult`, `ExitReason`, and `Trade` through `Trader.Trading`, routes them through `computeMetrics`, and fails before downstream optimizer builds if the public result-type surface narrows again.
- `haskell/web/test/deployConfig.test.mjs` covers blank or missing Fly host normalization to the default backend fallback, rejects malformed string and non-string Fly app/host overrides instead of synthesizing a fallback, and keeps that normalization confined to `apiFallbackUrl`.
- `test/autoloop.test.mjs` covers failed CI log ingestion, failure-targeted editable-file promotion, parser-first Haskell repair instructions, and required lifecycle phases so backend deploy/build failures that reach Fly Docker/cabal output remain eligible for repair instead of being classified as pending or skippable.

Proof sketch:

- `normalizeSignalOpenThreshold` is the threshold-validity boundary for fresh entries: negative or non-finite raw thresholds return `Nothing`, and malformed normalized thresholds would also be rejected by the finite non-negative obligation before any required-edge comparison can be made.
- Because `signalEntryEdgeSpikeOk`, `signalEntryHeadroomOk`, and `signalEntryFeeBufferOk` case-analyze that boundary and map `Nothing` to `False`, a negative `--open-threshold` or legacy `--threshold` cannot collapse to a zero deadband, reduce required edge, or reopen a blocked fresh entry.
- Restoring the public `normalizeSignalEntryEdge` symbol is a visibility-only repair: the helper remains the single raw-edge normalization boundary used by `mkEntryGateState`, preserving finite non-negative samples, collapsing finite negative raw edges to `Just 0`, and mapping non-finite raw edges to `Nothing`.
- `IntrabarFill` and `Positioning` remain passive public enums on the `Trader.Trading` seam, so restoring them for `Trader.App.Args` changes only symbol visibility and does not feed `mkEntryGateState` or `simulateEnsembleVWithHLChecked`.
- `signalEntryHeadroomThresholdCap` is derived from that same normalized non-negative edge sample and `entryEdgeHeadroomMultiple`, so the optimizer can enumerate the maximum admissible open-threshold witness for each observed edge without changing `signalEntryHeadroomOk` or the underlying gate contract.
- `signalEntryFeeBufferOk` requires fee floors to be finite and non-negative, so negative, `NaN`, and `Infinity` fee inputs fail closed instead of relaxing to the zero-fee boundary.
- The predicate now always inspects `edgeForMethod`, so a missing edge sample cannot bypass the gate even when both the normalized threshold and fee floor collapse to zero.
- `signalEntryHeadroomOk` is implemented by partially applying the fee-aware predicate with a zero fee floor, so the legacy headroom contract is preserved as a special case.
- The guard compares edge against an affine requirement with unit slope in the valid fee-floor domain, so increasing fees cannot reduce the minimum admissible edge.
- Lowering raw edge cannot make a blocked state admissible because the predicate is only `edge >= requiredEdge` once the inputs are well formed and the edge sample is explicit.
- In `mkEntryGateState`, `roundTripFeeFloor` becomes `0 / 0` whenever the per-side fee sample is bad and otherwise preserves the signed doubled per-side fee, so negative per-side fees reach `signalEntryFeeBufferOk` as negative round-trip floors and are rejected there.
- `mkEntryGateState` computes `entryEdge` once via `normalizeSignalEntryEdge` and reuses that same non-negative sample across the spike/headroom/fee vetoes under `needsEntry`, so restoring helper visibility does not change the fail-closed, entry-only conjunction at integration time.
- `Trader.VolConfGate.volBucket` is now the volatility-evidence validity boundary for enabled volatility-confidence presets: only present finite values from zero through the configured evidence max (`2.0` by default) can enter the bucket table, and every missing, negative, non-finite, or above-range sample maps to `VolConfGateAllowExitOnly 0` before any low/medium/high-volatility entry cell can be selected.
- `Trader.VolConfGate.confidenceBucket` is now the provided-confidence validity boundary: absent confidence stays weak and therefore blocks fresh low/medium-volatility entries, while provided negative, non-finite, or above-range confidence maps to `VolConfGateAllowExitOnly 0` instead of being normalized into a weak hold cell.
- The volatility-confidence table still classifies exact volatility thresholds into the higher bucket and exact confidence thresholds into the stronger bucket, so equality at valid documented boundaries is unchanged.
- On bounded witnesses, stricter confidence requirements can only demote a confidence bucket or preserve it, and tighter high-volatility requirements can only move a volatility bucket toward the more restrictive high-volatility rows or preserve it; neither transformation can reopen a blocked malformed or weak entry.
- `simulateEnsembleVWithHLChecked` now evaluates the same conjunction on fresh entries, deriving the fee-buffer floor from modeled round-trip entry/exit costs for the final overlay-scaled candidate size. Because `signalEntryFeeBufferOk` implies the headroom comparison and adds a non-negative cost floor, adding it to the simulator can only preserve or remove fresh entries; it cannot create a new entry from the same edge sample.
- The simulator validates `ecMaxPositionSize` and `ecMinPositionSize` before final fresh-entry size admissibility. When either bound is negative or non-finite, the effective cap/floor pair is zero, so the subsequent `sizeFinal0 <= 0` guard keeps `desiredSide` flat. When both bounds are valid, the cap is applied before the minimum-floor comparison; therefore lowering a valid cap can preserve or reduce exposure, equality at a valid minimum floor remains admissible, and a cap below that floor blocks fresh entry.
- The live latest-signal path uses the same predicate after final entry-size overlays, so a trade that only clears the pre-fee threshold is reported as a hold rather than being opened live while the backtest stays flat; fixed/minimum costs are divided by the final order size before comparison.
- Restoring `IntrabarFill(..)`, `Positioning(..)`, `BacktestResult`, `EnsembleConfig`, `StepMeta`, `simulateEnsembleVWithHLChecked`, `ExitReason`, and `Trade` on the `Trader.Trading` export list is a non-semantic visibility repair for `Trader.App.Args`, `Trader.Optimization`, and `Trader.Metrics`; CLI parsing, optimizer search/scoring/risk logic, trade construction, exit classification, and backtest aggregation behavior remain unchanged because the underlying gate and simulator implementations are unchanged.
- The web-side repair in `haskell/web/src/lib/deployConfig.ts` defaults only missing or blank Fly host inputs to `fly.dev`, rejects malformed string or non-string Fly app/host overrides before synthesizing `apiFallbackUrl`, and does not alter any value consumed by `signalEntryHeadroomOk`, `signalEntryFeeBufferOk`, `signalEntryEdgeSpikeOk`, or `signalRunPostDirectionGates`, so the fail-closed entry admissibility relation above is unchanged.
- Treating Fly backend Docker/cabal build output as actionable repair context changes only autoloop failure routing. It does not feed the trading predicates directly; it constrains any later backend repair to parser/build restoration first and leaves the fee-aware fresh-entry invariant as the proof obligation for semantic edits.

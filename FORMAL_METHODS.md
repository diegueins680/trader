## Formal conformal calibration contract

`fitConformal` and `predictInterval` in `haskell/app/Trader/Predictors/Conformal.hs` are treated as the fail-closed calibration boundary for conformal prediction intervals and any confidence or trading gate derived from those intervals.

Clauses:

1. A conformal calibration residual sample is admissible only when the sample is non-empty and every absolute residual is finite and non-negative.
2. Empty calibration evidence, negative residuals, `NaN`, positive infinity, negative infinity, or a residual list containing any malformed sample produces an unavailable conformal model with `cmCount == 0` and an unbounded prediction interval.
3. An unavailable conformal model, a malformed model radius, or a malformed point forecast must return `(-Infinity, Infinity, Nothing)` from `predictInterval`, so malformed calibration evidence cannot emit artificially tight interval bounds or a positive sigma-derived confidence estimate.
4. Valid zero residual samples remain admissible. They may produce a zero-width interval around a finite point forecast, but `sigmaFromInterval` remains unavailable for non-positive interval width.
5. For a fixed finite point forecast and valid non-empty calibration evidence, interval width is `2 * cmRadius`, so widening the selected conformal residual quantile is monotone non-decreasing in the produced interval width.

Bounded executable obligations:

- The Haskell test suite is the bounded regression harness for this invariant. It must cover empty calibration evidence, malformed residual evidence including mixed valid/malformed samples, valid zero residual evidence, no sigma for unavailable or zero-width intervals, and monotone widening as the selected conformal residual quantile rises.

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

- The Haskell regression harness for this invariant should cover empty or dimension-invalid quantile models, non-finite q10/q50/q90 predictions, inverted q10/q90 evidence, the valid equality boundary with unavailable sigma, and monotone non-narrowing as the ordered q10/q90 spread widens.

Proof sketch:

- The model-dimension check is the first evidence boundary: every quantile head must expose the same positive feature dimension and the forecast vector must match it, otherwise no raw prediction is trusted.
- `admissibleQuantilePrediction` is the raw-output validity boundary: it rejects the entire interval when any raw quantile is `NaN` or infinite, so malformed q50 evidence cannot be hidden by clamping and malformed q10/q90 evidence cannot define finite bounds.
- The ordered-bound guard rejects `q10 > q90` instead of sorting the pair. Therefore a crossing quantile model cannot reverse directionality or synthesize a usable interval from contradictory lower/upper evidence.
- Equality at `q10 == q90` satisfies the ordered-bound predicate and produces a zero-width interval with `Nothing` sigma because `sigmaFromQ1090` requires positive width. This preserves the valid deterministic boundary without creating false confidence from zero spread.
- After admissibility, the returned lower and upper bounds are exactly the raw q10 and q90 values, so interval width is exactly `q90 - q10`. Widening ordered bounds can only preserve or increase that width, and sigma is width divided by a positive constant when width is positive.

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

## Formal fee-aware entry gate contract

`normalizeSignalOpenThreshold`, `signalEntryHeadroomThresholdCap`, `normalizeSignalEntryEdge`, and `signalEntryFeeBufferOk` in `haskell/app/Trader/SignalGates.hs`, as wired into the repaired `mkEntryGateState` binding block in `haskell/app/Trader/Trading.hs`, are treated as the shared fail-closed fresh-entry threshold boundary, canonical optimizer headroom-cap witness, raw-edge normalization boundary, and marginal-entry veto.

`BacktestResult`, `EnsembleConfig`, `StepMeta`, `IntrabarFill`, `Positioning`, `simulateEnsembleVWithHLChecked`, `ExitReason`, and `Trade`, as exported from `haskell/app/Trader/Trading.hs`, are treated as the stable public simulation/result/CLI surface consumed by `Trader.App.Args`, `Trader.Optimization`, `Trader.Metrics`, and related optimizer code.

Clauses:

1. Let `threshold = normalizeSignalThreshold openThreshold`, `requiredHeadroom = 1.5 * threshold`, and `requiredEdge = requiredHeadroom + roundTripFeeFloor`. An entry is admissible only when the raw open threshold is finite and non-negative, the normalized threshold is finite and non-negative, the fee floor is finite and non-negative, the edge sample is explicit (`Just`) and finite, and `edge >= requiredEdge`; this explicit-edge obligation still applies when `requiredEdge == 0`.
2. `signalEntryHeadroomOk openThreshold` remains the zero-fee specialization `signalEntryFeeBufferOk openThreshold 0`.
3. For fixed `openThreshold` and edge, admissibility is monotone non-increasing as any valid `roundTripFeeFloor` rises.
4. For fixed `openThreshold` and fee floor, admissibility is monotone non-increasing as raw edge falls.
5. Missing edges, negative open thresholds, non-finite open thresholds, negative fee floors, non-finite fee floors, or malformed edge samples presented directly to `signalEntryEdgeSpikeOk`, `signalEntryHeadroomOk`, or `signalEntryFeeBufferOk` are fail-closed.
6. Once a state is blocked at fee floor `f`, it remains blocked for every valid `f' >= f`; malformed fee data and malformed or negative threshold data also cannot reopen the entry.
7. In `mkEntryGateState`, the spike, headroom, and fee-buffer checks are consulted only when `needsEntry` is true, each threshold-consuming check must pass through `normalizeSignalOpenThreshold`, each check receives the same `entryEdge` sample computed once as `normalizeSignalEntryEdge edgeRaw`, `normalizeSignalEntryEdge` preserves finite non-negative inputs and collapses negative or non-finite raw edges to `Just 0`, and the three booleans are combined conjunctively before `desiredSide1` can keep a fresh entry alive.
8. In `simulateEnsembleVWithHLChecked`, a fresh entry must first prove the configured position-sizing exposure cap and floor are finite and non-negative. Invalid cap/floor evidence (`NaN`, infinity, or negative values) maps the effective fresh-entry cap and floor to zero before the final candidate size can become admissible. With valid sizing bounds, a zero cap remains a legal no-entry boundary, a zero floor remains legal, equality at the minimum floor remains admissible, and tightening a valid cap is monotone non-increasing for realized fresh-entry exposure. A fresh entry must also pass the same spike, headroom, and fee-buffer conjunction before sizing overlays can open exposure. The fee-buffer floor is the modeled round-trip cost rate for the final overlay-scaled candidate entry size, so a signal that is threshold/headroom-valid but below its sized round-trip costs remains flat.
9. In the live `Main` latest-signal path, the chosen direction must pass `signalEntryFeeBufferOk openThrAdj sizedRoundTripCost edgeForMethod` in addition to the existing spike and headroom checks after final entry-size overlays. A rejection from that predicate is observable as `EDGE_FEE_BUFFER` after spike/headroom reasons are ruled out.
10. Deploy-config normalization in `haskell/web/src/lib/deployConfig.ts` is non-interfering with trading admissibility: blank or missing Fly host inputs may normalize to the default `fly.dev`, malformed Fly app/host overrides are rejected instead of being coerced into a fallback target, and the resulting `apiFallbackUrl` synthesis does not feed `signalEntryHeadroomOk`, `signalEntryFeeBufferOk`, `signalEntryEdgeSpikeOk`, or the fresh-entry conjunction.
11. Every CLI, optimizer, or metrics consumer must be able to import `signalEntryHeadroomThresholdCap` through `Trader.SignalGates` and `IntrabarFill(..)`, `Positioning(..)`, `BacktestResult`, `EnsembleConfig`, `StepMeta`, `simulateEnsembleVWithHLChecked`, `ExitReason`, and `Trade` through `Trader.Trading`; restoring those exports is a visibility-only repair and does not alter fresh-entry gating, optimizer candidate generation, ensemble simulation behavior, trade construction, exit classification, or backtest aggregation behavior.
12. For any raw edge sample, `signalEntryHeadroomThresholdCap` applies the same fail-closed normalization boundary as `normalizeSignalEntryEdge` and returns the maximum open-threshold witness compatible with the headroom gate; malformed or negative raw edges collapse to cap `0`.
13. Deploy-log-driven backend repair is non-interfering with trading admissibility: a failed Fly backend Docker/cabal build log is actionable repair context rather than pending or skippable CI noise, and any backend Haskell repair selected from that log must first restore parser/build validity before changing trading semantics or weakening the fresh-entry gate predicates above.
14. For every enabled `Trader.VolConfGate` preset, volatility evidence is admissible only when it is present, finite, and within `[0,2]`. Missing, negative, non-finite, or above-range volatility evidence maps to `VolConfGateAllowExitOnly` with size multiplier `0`, so malformed volatility cannot be normalized into low-volatility entry permission.
15. For every enabled `Trader.VolConfGate` preset, provided confidence evidence is admissible only when it is finite and within `[0,1]`. Missing confidence remains weak entry-blocking evidence, but provided negative, non-finite, or above-range confidence maps to `VolConfGateAllowExitOnly` with size multiplier `0` instead of a weak hold cell.
16. Valid volatility-confidence boundary equality is preserved: volatility bucket boundaries remain inclusive on the higher bucket, confidence weak/strong thresholds remain inclusive on the stronger bucket, the volatility upper bound `2.0` remains valid, and entry admissibility is monotone non-increasing when moving from default to stricter confidence requirements or from looser to tighter high-volatility requirements on the same bounded witness.

Bounded executable obligations:

- `testSignalGateEntryHeadroomSpecializesFeeBuffer` preserves the legacy headroom boundary cases, including the valid zero-threshold explicit-edge boundary and the zero-fee specialization.
- `testSignalGateEntryFeeBuffer` covers equality-at-boundary acceptance, strict-below-boundary rejection, zero-threshold-with-fees behavior, missing-edge fail-closed behavior, the zero-threshold zero-fee explicit-edge corner, and the zero-fee specialization.
- `testSignalGateEntryFeeBufferMonotoneFees` witnesses monotone non-increasing admissibility as `roundTripFeeFloor` rises and the once-blocked-stays-blocked ladder.
- `testSignalGateEntryFeeBufferMonotoneEdge` witnesses monotone non-increasing admissibility as raw edge falls under a fixed fee floor.
- `testSignalGateEntryFeeBufferFailsClosed` covers negative and non-finite open thresholds, negative and non-finite fee floors, non-finite edges, and negative edge samples.
- `testNormalizeSignalEntryEdgeFailClosedRegression` witnesses that the restored public `normalizeSignalEntryEdge` helper preserves valid fresh-entry edges, collapses negative or non-finite raw edges to the shared `Just 0` sample, and keeps the Trading conjunction fail closed when that shared sample is reused.
- `testTradingEntryGateFailClosedMonotone` and `testTradingEntryGateMalformedNoReopen` extend the `mkEntryGateState` witness so negative or non-finite per-side fees cannot reopen a blocked fresh entry after shared edge normalization.
- `testVolConfGateMalformedInputsFailClosed` witnesses valid volatility-confidence boundary equality, fail-closed behavior for missing/malformed/out-of-range volatility, fail-closed behavior for malformed provided confidence, weak entry-blocking behavior for missing confidence, exit-only behavior that cannot open or increase exposure, and monotonicity under stricter confidence and high-volatility requirements.
- `testBacktestEntryGateUsesRoundTripFeeBuffer` binds the same contract to the checked simulator: a no-fee headroom-valid entry can open, but the identical prediction remains flat once modeled round-trip costs exceed the available edge, including when fixed costs only become prohibitive after Kelly-lite overlays reduce final entry size.
- `testBacktestFreshEntrySizingBoundsFailClosed` binds fresh-entry sizing validity to the checked simulator by covering negative, `NaN`, and infinite max-position caps and min-position floors, the valid zero-cap no-entry boundary, valid zero-floor admissibility, valid minimum-floor equality, cap-below-floor rejection, and monotone non-increasing exposure as valid caps tighten.
- `testSignalGateEntryEdgeSpike` covers equality-at-cap acceptance, zero-threshold zero-edge acceptance, zero-threshold positive-edge rejection, and malformed threshold/edge fail-closed behavior for the independent spike veto in the same entry-only conjunction.
- `testSignalGateEntryEdgeSpikeMonotone` witnesses monotone non-increasing admissibility as the effective spike threshold is lowered.
- `testOptimizerPublicSurfaceRegression` imports `signalEntryHeadroomThresholdCap`, `EnsembleConfig(..)`, `StepMeta(..)`, and `simulateEnsembleVWithHLChecked` through their public modules, witnesses the `0.03 -> 0.02` headroom-cap boundary, and fails before `optimize-equity` CI build time if the optimizer-facing public surface narrows again.
- `testMetricsConsumesTradingPublicResults` constructs `BacktestResult`, `ExitReason`, and `Trade` through `Trader.Trading`, routes them through `computeMetrics`, and fails before downstream optimizer builds if the public result-type surface narrows again.
- `haskell/web/test/deployConfig.test.mjs` covers blank or missing Fly host normalization to the default backend fallback, rejects malformed string and non-string Fly app/host overrides instead of synthesizing a fallback, and keeps that normalization confined to `apiFallbackUrl`.
- `test/autoloop.test.mjs` covers failed CI log ingestion, failure-targeted editable-file promotion, parser-first Haskell repair instructions, and required lifecycle phases so backend deploy/build failures that reach Fly Docker/cabal output remain eligible for repair instead of being classified as pending or skippable.

Proof sketch:

- `normalizeSignalOpenThreshold` is the threshold-validity boundary for fresh entries: negative or non-finite raw thresholds return `Nothing`, and malformed normalized thresholds would also be rejected by the finite non-negative obligation before any required-edge comparison can be made.
- Because `signalEntryEdgeSpikeOk`, `signalEntryHeadroomOk`, and `signalEntryFeeBufferOk` case-analyze that boundary and map `Nothing` to `False`, a negative `--open-threshold` or legacy `--threshold` cannot collapse to a zero deadband, reduce required edge, or reopen a blocked fresh entry.
- Restoring the public `normalizeSignalEntryEdge` symbol is a visibility-only repair: the helper remains the single raw-edge normalization boundary used by `mkEntryGateState`, preserving finite non-negative samples and collapsing every negative or non-finite raw edge to `Just 0`.
- `IntrabarFill` and `Positioning` remain passive public enums on the `Trader.Trading` seam, so restoring them for `Trader.App.Args` changes only symbol visibility and does not feed `mkEntryGateState` or `simulateEnsembleVWithHLChecked`.
- `signalEntryHeadroomThresholdCap` is derived from that same normalized non-negative edge sample and `entryEdgeHeadroomMultiple`, so the optimizer can enumerate the maximum admissible open-threshold witness for each observed edge without changing `signalEntryHeadroomOk` or the underlying gate contract.
- `signalEntryFeeBufferOk` requires fee floors to be finite and non-negative, so negative, `NaN`, and `Infinity` fee inputs fail closed instead of relaxing to the zero-fee boundary.
- The predicate now always inspects `edgeForMethod`, so a missing edge sample cannot bypass the gate even when both the normalized threshold and fee floor collapse to zero.
- `signalEntryHeadroomOk` is implemented by partially applying the fee-aware predicate with a zero fee floor, so the legacy headroom contract is preserved as a special case.
- The guard compares edge against an affine requirement with unit slope in the valid fee-floor domain, so increasing fees cannot reduce the minimum admissible edge.
- Lowering raw edge cannot make a blocked state admissible because the predicate is only `edge >= requiredEdge` once the inputs are well formed and the edge sample is explicit.
- In `mkEntryGateState`, `roundTripFeeFloor` becomes `0 / 0` whenever the per-side fee sample is bad and otherwise preserves the signed doubled per-side fee, so negative per-side fees reach `signalEntryFeeBufferOk` as negative round-trip floors and are rejected there.
- `mkEntryGateState` computes `entryEdge` once via `normalizeSignalEntryEdge` and reuses that same non-negative sample across the spike/headroom/fee vetoes under `needsEntry`, so restoring helper visibility does not change the fail-closed, entry-only conjunction at integration time.
- `Trader.VolConfGate.volBucket` is now the volatility-evidence validity boundary for enabled volatility-confidence presets: only present finite values in `[0,2]` can enter the bucket table, and every missing, negative, non-finite, or above-range sample maps to `VolConfGateAllowExitOnly 0` before any low/medium/high-volatility entry cell can be selected.
- `Trader.VolConfGate.confidenceBucket` is now the provided-confidence validity boundary: absent confidence stays weak and therefore blocks fresh low/medium-volatility entries, while provided negative, non-finite, or above-range confidence maps to `VolConfGateAllowExitOnly 0` instead of being normalized into a weak hold cell.
- The volatility-confidence table still classifies exact volatility thresholds into the higher bucket and exact confidence thresholds into the stronger bucket, so equality at valid documented boundaries is unchanged.
- On bounded witnesses, stricter confidence requirements can only demote a confidence bucket or preserve it, and tighter high-volatility requirements can only move a volatility bucket toward the more restrictive high-volatility rows or preserve it; neither transformation can reopen a blocked malformed or weak entry.
- `simulateEnsembleVWithHLChecked` now evaluates the same conjunction on fresh entries, deriving the fee-buffer floor from modeled round-trip entry/exit costs for the final overlay-scaled candidate size. Because `signalEntryFeeBufferOk` implies the headroom comparison and adds a non-negative cost floor, adding it to the simulator can only preserve or remove fresh entries; it cannot create a new entry from the same edge sample.
- The simulator validates `ecMaxPositionSize` and `ecMinPositionSize` before final fresh-entry size admissibility. When either bound is negative or non-finite, the effective cap/floor pair is zero, so the subsequent `sizeFinal0 <= 0` guard keeps `desiredSide` flat. When both bounds are valid, the cap is applied before the minimum-floor comparison; therefore lowering a valid cap can preserve or reduce exposure, equality at a valid minimum floor remains admissible, and a cap below that floor blocks fresh entry.
- The live latest-signal path uses the same predicate after final entry-size overlays, so a trade that only clears the pre-fee threshold is reported as a hold rather than being opened live while the backtest stays flat; fixed/minimum costs are divided by the final order size before comparison.
- Restoring `IntrabarFill(..)`, `Positioning(..)`, `BacktestResult`, `EnsembleConfig`, `StepMeta`, `simulateEnsembleVWithHLChecked`, `ExitReason`, and `Trade` on the `Trader.Trading` export list is a non-semantic visibility repair for `Trader.App.Args`, `Trader.Optimization`, and `Trader.Metrics`; CLI parsing, optimizer search/scoring/risk logic, trade construction, exit classification, and backtest aggregation behavior remain unchanged because the underlying gate and simulator implementations are unchanged.
- The web-side repair in `haskell/web/src/lib/deployConfig.ts` defaults only missing or blank Fly host inputs to `fly.dev`, rejects malformed string or non-string Fly app/host overrides before synthesizing `apiFallbackUrl`, and does not alter any value consumed by `signalEntryHeadroomOk`, `signalEntryFeeBufferOk`, `signalEntryEdgeSpikeOk`, or `signalRunPostDirectionGates`, so the fail-closed entry admissibility relation above is unchanged.
- Treating Fly backend Docker/cabal build output as actionable repair context changes only autoloop failure routing. It does not feed the trading predicates directly; it constrains any later backend repair to parser/build restoration first and leaves the fee-aware fresh-entry invariant as the proof obligation for semantic edits.

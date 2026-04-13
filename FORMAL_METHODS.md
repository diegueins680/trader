## Formal fee-aware entry gate contract

`normalizeSignalEntryEdge` and `signalEntryFeeBufferOk` in `haskell/app/Trader/SignalGates.hs`, as wired into the repaired `mkEntryGateState` binding block in `haskell/app/Trader/Trading.hs`, are treated as the shared fail-closed fresh-entry normalization boundary and marginal-entry veto.

`BacktestResult`, `ExitReason`, and `Trade`, as exported from `haskell/app/Trader/Trading.hs`, are treated as the stable public result surface consumed by `Trader.Metrics` and optimizer code.

Clauses:

1. Let `requiredHeadroom = 1.5 * normalizeSignalThreshold openThreshold` and `requiredEdge = requiredHeadroom + roundTripFeeFloor`. An entry is admissible only when the fee floor is finite and non-negative, the edge sample is explicit (`Just`) and finite, and `edge >= requiredEdge`; this explicit-edge obligation still applies when `requiredEdge == 0`.
2. `signalEntryHeadroomOk openThreshold` remains the zero-fee specialization `signalEntryFeeBufferOk openThreshold 0`.
3. For fixed `openThreshold` and edge, admissibility is monotone non-increasing as any valid `roundTripFeeFloor` rises.
4. For fixed `openThreshold` and fee floor, admissibility is monotone non-increasing as raw edge falls.
5. Missing edges, negative fee floors, non-finite fee floors, or malformed edge samples presented directly to `signalEntryEdgeSpikeOk`, `signalEntryHeadroomOk`, or `signalEntryFeeBufferOk` are fail-closed.
6. Once a state is blocked at fee floor `f`, it remains blocked for every valid `f' >= f`; malformed fee data also cannot reopen the entry.
7. In `mkEntryGateState`, the spike, headroom, and fee-buffer checks are consulted only when `needsEntry` is true, each check receives the same `entryEdge` sample computed once as `normalizeSignalEntryEdge edgeRaw`, `normalizeSignalEntryEdge` preserves finite non-negative inputs and collapses negative or non-finite raw edges to `Just 0`, and the three booleans are combined conjunctively before `desiredSide1` can keep a fresh entry alive.
8. Deploy-config normalization in `haskell/web/src/lib/deployConfig.ts` is non-interfering with trading admissibility: blank or missing Fly host inputs may normalize to the default `fly.dev`, malformed Fly app/host overrides are rejected instead of being coerced into a fallback target, and the resulting `apiFallbackUrl` synthesis does not feed `signalEntryHeadroomOk`, `signalEntryFeeBufferOk`, `signalEntryEdgeSpikeOk`, or the fresh-entry conjunction.
9. Every metrics consumer must be able to construct and inspect `BacktestResult`, `ExitReason`, and `Trade` through `Trader.Trading`; restoring those exports is a visibility-only repair and does not alter fresh-entry gating, trade construction, exit classification, or backtest aggregation behavior.

Bounded executable obligations:

- `testSignalGateEntryHeadroom` preserves the legacy headroom boundary cases.
- `testSignalGateEntryFeeBuffer` covers equality-at-boundary acceptance, strict-below-boundary rejection, zero-threshold-with-fees behavior, missing-edge fail-closed behavior, the zero-threshold zero-fee explicit-edge corner, and the zero-fee specialization.
- `testSignalGateEntryFeeBufferMonotoneFees` witnesses monotone non-increasing admissibility as `roundTripFeeFloor` rises and the once-blocked-stays-blocked ladder.
- `testSignalGateEntryFeeBufferMonotoneEdge` witnesses monotone non-increasing admissibility as raw edge falls under a fixed fee floor.
- `testSignalGateEntryFeeBufferFailsClosed` covers negative and non-finite fee floors, non-finite open thresholds, and non-finite edges.
- `testNormalizeSignalEntryEdgeFailClosedRegression` witnesses that the restored public `normalizeSignalEntryEdge` helper preserves valid fresh-entry edges, collapses negative or non-finite raw edges to the shared `Just 0` sample, and keeps the Trading conjunction fail closed when that shared sample is reused.
- `testTradingEntryGateFailClosedMonotone` and `testTradingEntryGateMalformedNoReopen` extend the `mkEntryGateState` witness so negative or non-finite per-side fees cannot reopen a blocked fresh entry after shared edge normalization.
- `testSignalGateEntryEdgeSpike` covers equality-at-cap acceptance, zero-threshold zero-edge acceptance, zero-threshold positive-edge rejection, and malformed threshold/edge fail-closed behavior for the independent spike veto in the same entry-only conjunction.
- `testSignalGateEntryEdgeSpikeMonotone` witnesses monotone non-increasing admissibility as the effective spike threshold is lowered.
- `testMetricsPublicTradingSurfaceRegression` constructs `BacktestResult`, `ExitReason`, and `Trade` through `Trader.Trading`, routes them through `computeMetrics`, and fails before downstream optimizer builds if the public result-type surface narrows again.
- `haskell/web/test/deployConfig.test.mjs` covers blank or missing Fly host normalization to the default backend fallback, rejects malformed string and non-string Fly app/host overrides instead of synthesizing a fallback, and keeps that normalization confined to `apiFallbackUrl`.

Proof sketch:

- Restoring the public `normalizeSignalEntryEdge` symbol is a visibility-only repair: the helper remains the single raw-edge normalization boundary used by `mkEntryGateState`, preserving finite non-negative samples and collapsing every negative or non-finite raw edge to `Just 0`.
- `signalEntryFeeBufferOk` requires fee floors to be finite and non-negative, so negative, `NaN`, and `Infinity` fee inputs fail closed instead of relaxing to the zero-fee boundary.
- The predicate now always inspects `edgeForMethod`, so a missing edge sample cannot bypass the gate even when both the normalized threshold and fee floor collapse to zero.
- `signalEntryHeadroomOk` is implemented by partially applying the fee-aware predicate with a zero fee floor, so the legacy headroom contract is preserved as a special case.
- The guard compares edge against an affine requirement with unit slope in the valid fee-floor domain, so increasing fees cannot reduce the minimum admissible edge.
- Lowering raw edge cannot make a blocked state admissible because the predicate is only `edge >= requiredEdge` once the inputs are well formed and the edge sample is explicit.
- In `mkEntryGateState`, `roundTripFeeFloor` becomes `0 / 0` whenever the per-side fee sample is bad and otherwise preserves the signed doubled per-side fee, so negative per-side fees reach `signalEntryFeeBufferOk` as negative round-trip floors and are rejected there.
- `mkEntryGateState` computes `entryEdge` once via `normalizeSignalEntryEdge` and reuses that same non-negative sample across the spike/headroom/fee vetoes under `needsEntry`, so restoring helper visibility does not change the fail-closed, entry-only conjunction at integration time.
- Restoring `BacktestResult`, `ExitReason`, and `Trade` on the `Trader.Trading` export list is a non-semantic visibility repair for `Trader.Metrics`; metrics consumers regain constructor and field access through the intended public module, while `mkEntryGateState`, trade construction, exit classification, and backtest aggregation behavior remain unchanged.
- The web-side repair in `haskell/web/src/lib/deployConfig.ts` defaults only missing or blank Fly host inputs to `fly.dev`, rejects malformed string or non-string Fly app/host overrides before synthesizing `apiFallbackUrl`, and does not alter any value consumed by `signalEntryHeadroomOk`, `signalEntryFeeBufferOk`, `signalEntryEdgeSpikeOk`, or `signalRunPostDirectionGates`, so the fail-closed entry admissibility relation above is unchanged.
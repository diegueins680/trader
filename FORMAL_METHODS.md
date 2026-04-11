## Formal fee-aware entry gate contract

`signalEntryFeeBufferOk` in `haskell/app/Trader/SignalGates.hs`, as wired into the repaired `mkEntryGateState` binding block in `haskell/app/Trader/Trading.hs`, is treated as a fail-closed marginal-entry veto.

Clauses:

1. Let `requiredHeadroom = 1.5 * normalizeSignalThreshold openThreshold` and `requiredEdge = requiredHeadroom + max 0 roundTripFeeFloor`. An entry is admissible only when the fee floor is finite, the edge sample is explicit (`Just`) and finite, and `edge >= requiredEdge`; this explicit-edge obligation still applies when `requiredEdge == 0`.
2. `signalEntryHeadroomOk openThreshold` remains the zero-fee specialization `signalEntryFeeBufferOk openThreshold 0`.
3. For fixed `openThreshold` and edge, admissibility is monotone non-increasing as `roundTripFeeFloor` rises.
4. For fixed `openThreshold` and fee floor, admissibility is monotone non-increasing as raw edge falls.
5. Missing edges, non-finite fee floors, or non-finite edges are fail-closed.
6. Once a state is blocked at fee floor `f`, it remains blocked for every `f' >= f`.
7. In `mkEntryGateState`, the spike, headroom, and fee-buffer checks are consulted only when `needsEntry` is true, each check receives the same non-negative `entryEdge` sample `Just (max 0 edgeRaw)`, and the three booleans are combined conjunctively before `desiredSide1` can keep a fresh entry alive.
8. Deploy-config normalization in `haskell/web/src/lib/deployConfig.ts` is non-interfering with trading admissibility: blank or missing Fly host inputs may normalize to the default `fly.dev`, malformed Fly app/host overrides are rejected instead of being coerced into a fallback target, and the resulting `apiFallbackUrl` synthesis does not feed `signalEntryHeadroomOk`, `signalEntryFeeBufferOk`, `signalEntryEdgeSpikeOk`, or the fresh-entry conjunction.

Bounded executable obligations:

- `testSignalGateEntryHeadroom` preserves the legacy headroom boundary cases.
- `testSignalGateEntryFeeBuffer` covers equality-at-boundary acceptance, strict-below-boundary rejection, zero-threshold-with-fees behavior, missing-edge fail-closed behavior, the zero-threshold zero-fee explicit-edge corner, and the zero-fee specialization.
- `testSignalGateEntryFeeBufferMonotoneFees` witnesses monotone non-increasing admissibility as `roundTripFeeFloor` rises and the once-blocked-stays-blocked ladder.
- `testSignalGateEntryFeeBufferMonotoneEdge` witnesses monotone non-increasing admissibility as raw edge falls under a fixed fee floor.
- `testSignalGateEntryFeeBufferFailsClosed` covers non-finite fee floors, non-finite edges, and negative fee-floor clamping.
- `testSignalGateEntryEdgeSpike` keeps the independent spike veto in the same entry-only conjunction.
- `haskell/web/test/deployConfig.test.mjs` covers blank or missing Fly host normalization to the default backend fallback, rejects malformed string and non-string Fly app/host overrides instead of synthesizing a fallback, and keeps that normalization confined to `apiFallbackUrl`.

Proof sketch:

- `signalEntryFeeBufferOk` normalizes negative fee floors to `0` but rejects non-finite fee floors outright, so malformed fee inputs cannot reopen entries.
- The predicate now always inspects `edgeForMethod`, so a missing edge sample cannot bypass the gate even when both the normalized threshold and fee floor collapse to zero.
- `signalEntryHeadroomOk` is implemented by partially applying the fee-aware predicate with a zero fee floor, so the legacy headroom contract is preserved as a special case.
- The guard compares edge against an affine requirement with unit slope in the fee floor, so increasing fees cannot reduce the minimum admissible edge.
- Lowering raw edge cannot make a blocked state admissible because the predicate is only `edge >= requiredEdge` once the inputs are well formed and the edge sample is explicit.
- In `mkEntryGateState`, `roundTripFeeFloor` becomes `0 / 0` whenever the per-side fee sample is bad, and that malformed fee is then rejected by `signalEntryFeeBufferOk`, preserving fail-closed behavior at the `Trading.hs` integration boundary.
- `mkEntryGateState` reuses the same non-negative `entryEdge` sample across the spike/headroom/fee vetoes and combines those booleans conjunctively under `needsEntry`, so the fee-aware gate remains fail-closed and entry-only at integration time.
- The web-side repair in `haskell/web/src/lib/deployConfig.ts` defaults only missing or blank Fly host inputs to `fly.dev`, rejects malformed string or non-string Fly app/host overrides before synthesizing `apiFallbackUrl`, and does not alter any value consumed by `signalEntryHeadroomOk`, `signalEntryFeeBufferOk`, `signalEntryEdgeSpikeOk`, or `signalRunPostDirectionGates`, so the fail-closed entry admissibility relation above is unchanged.
- Review note for this deploy-config repair cycle: `signalEntryHeadroomOk`, `signalEntryFeeBufferOk`, `signalEntryEdgeSpikeOk`, and the `signalRunPostDirectionGates` conjunction were re-reviewed in `SignalGates.hs`; no trading-logic change was required because malformed or missing edge samples still fail closed, `mkEntryGateState` still combines fresh-entry vetoes conjunctively, and admissibility still only tightens as fee floors rise or edge samples fall.
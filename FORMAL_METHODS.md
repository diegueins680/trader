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
- `testSignalGateEntryEdgeSpike` covers equality-at-cap acceptance, zero-threshold zero-edge acceptance, zero-threshold positive-edge rejection, and malformed threshold/edge fail-closed behavior for the independent spike veto in the same entry-only conjunction.
- `testSignalGateEntryEdgeSpikeMonotone` witnesses monotone non-increasing admissibility as the effective spike threshold is lowered.
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
- Review note for this spike-gate repair cycle: `signalEntryEdgeSpikeOk` in `SignalGates.hs` was re-reviewed and tightened to reject non-finite thresholds before normalization, so malformed spike-gate inputs cannot bypass the fresh-entry conjunction while equality at the active cap remains admitted and the fee-aware headroom contract above stays unchanged.

## Formal edge-spike entry gate contract

`signalEntryEdgeSpikeOk` in `haskell/app/Trader/SignalGates.hs`, as wired into the same `mkEntryGateState` fresh-entry conjunction in `haskell/app/Trader/Trading.hs`, is treated as an independent fail-closed outlier veto.

Clauses:

1. Let `normalizedThreshold = normalizeSignalThreshold openThreshold` and `spikeCap = min 0.5 (4 * normalizedThreshold)`. An entry is admissible only when `openThreshold` is finite, the edge sample is explicit (`Just`) and finite, `edge >= 0`, and `edge <= spikeCap`.
2. Equality at the active spike cap remains admissible.
3. For fixed well-formed edge samples, admissibility is monotone non-increasing as the spike cap tightens or the threshold is lowered.
4. Missing edges, non-finite thresholds, and non-finite edges are fail-closed.
5. When `normalizedThreshold == 0`, the only admissible edge is `0`, so lowering the threshold to zero cannot reopen a previously blocked edge.
6. In `mkEntryGateState`, the spike veto still consumes the same shared non-negative `entryEdge` sample as the headroom and fee-buffer gates and the three booleans are combined conjunctively under `needsEntry`.

Bounded executable obligations:

- `testSignalGateEntryEdgeSpike` covers equality-at-cap acceptance, zero-threshold zero-edge acceptance, zero-threshold positive-edge rejection, and malformed threshold/edge fail-closed cases.
- `testSignalGateEntryEdgeSpikeMonotone` witnesses the threshold-tightening monotonicity ladder.
- `testSignalGateEntryConjunctiveSharedEdge` covers the shared-edge conjunction fact that the spike veto can still block a zero-threshold entry even when headroom and fee-buffer collapse to zero.
- `testTradingEntryGateSharedEdgeConjunction` keeps the integration witness that the same `entryEdge` sample is fed into all three fresh-entry vetoes.

Proof sketch:

- `signalEntryEdgeSpikeOk` now guards on `finiteDouble openThreshold` before normalization, so `NaN` or `Infinity` thresholds cannot degrade into the old permissive zero-threshold branch.
- `normalizeSignalThreshold` is monotone non-decreasing over finite inputs, multiplication by the positive constant `4` preserves that order, and `min 0.5` cannot enlarge the admissible set; therefore lowering the threshold or tightening the cap cannot admit additional edges.
- The acceptance comparison is `edge <= spikeCap`, so exact equality at the cap remains admissible.
- Because `normalizeSignalThreshold` sends non-positive finite thresholds to `0`, the degenerate threshold case collapses to `spikeCap = 0`, admitting only the shared zero edge sample.
- `mkEntryGateState` still reuses one non-negative `entryEdge` across the spike, headroom, and fee-buffer vetoes and combines them conjunctively, so the independent spike repair only removes admissibility and cannot reopen blocked fresh-entry states.

## Formal low-directionality entry gate contract

`signalDirectionalityEntryAllowed` in `haskell/app/Trader/SignalGates.hs` is treated as the fail-closed admission check for fresh entries derived from `DirectionalitySnapshot`.

Clauses:

1. `Nothing`, malformed snapshots, and snapshots already marked `dsNonDirectional = True` are inadmissible.
2. Fresh directional entries are always blocked when `dsEfficiency <= 0.25`, even if a saved snapshot is otherwise marked directional.
3. When `0.25 < dsEfficiency <= 0.40`, admissibility requires explicit finite `dsTrendProb`, `dsMrProb`, and `dsHighVolProb`; missing or non-finite regime probabilities are treated as malformed and block the entry.
4. For `dsEfficiency > 0.40`, the entry gate preserves the prior behavior: regime-probability completeness is not required by this check as long as the snapshot is otherwise well formed and not already marked non-directional.
5. Admissibility is monotone non-increasing as efficiency falls across the `0.40` review-band boundary and the `0.25` chop boundary under otherwise identical snapshot fields.
6. Mean-reversion-dominant weak-band snapshots remain blocked because `signalDirectionalitySnapshot` marks them `NON_DIRECTIONAL_MR`, and `signalDirectionalityEntryAllowed` never admits a snapshot with `dsNonDirectional = True`.

Bounded executable obligations:

- `testSignalGateDirectionalityWeakBandFailClosed` covers the strong-band no-change witness, the `0.40` and `0.25` boundaries, weak-band missing-probability fail-closed behavior, non-finite probability rejection, and preservation of the existing `NON_DIRECTIONAL_MR` veto.
- `testSignalGateDirectionalityWeakBandMonotone` witnesses the non-increasing admissibility ladder as efficiency falls when regime probabilities are missing.

Proof sketch:

- `signalDirectionalityEntryAllowed` now requires `dsEfficiency > directionalityChopEfficiencyMax`, so saved snapshots cannot reopen the documented chop veto by carrying a stale directional label.
- `directionalitySnapshotWellFormed` now also requires explicit regime probabilities in the weak review band, so missing or non-finite HMM probabilities are reclassified as malformed before any fresh entry can be admitted.
- The weak-band completeness obligation is one-way: crossing from `> 0.40` into `<= 0.40` can only remove admissible states, and crossing from `> 0.25` into `<= 0.25` removes them all.
- Mean-reversion dominance still blocks via the existing `dsNonDirectional`/`dsReason` contract produced by `signalDirectionalitySnapshot`, so the repair only shrinks the admissible set for malformed saved snapshots and leaves well-formed directional snapshots unchanged.
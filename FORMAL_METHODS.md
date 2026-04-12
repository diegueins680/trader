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

`signalDirectionalityEntryAllowedForSide` in `haskell/app/Trader/SignalGates.hs`, together with the local `directionalityWeakBandSignedZScoreOk` helper, is the signed fail-closed admission check for fresh entries derived from `DirectionalitySnapshot`. `signalDirectionalityEntryAllowed` remains the side-agnostic convenience wrapper that admits only when at least one signed side is admissible.

Clauses:

1. `Nothing`, malformed snapshots, and snapshots already marked `dsNonDirectional = True` are inadmissible.
2. Fresh directional entries are always blocked when `dsEfficiency <= 0.25`, even if a saved snapshot is otherwise marked directional.
3. Whenever any saved HMM regime probability is present, admissibility requires a complete tuple: `dsTrendProb`, `dsMrProb`, and `dsHighVolProb` must all be explicit, finite, and in `[0,1]`, and their total probability mass must be positive and within `1e-3` of `1`.
4. When `0.25 < dsEfficiency <= 0.40`, admissibility requires both that complete normalized tuple and one shared signed weak-band zScore predicate: the score input must be explicit and finite, it is mapped to `signedZScore = zScore` for `DirectionalityLong` and `signedZScore = -zScore` for `DirectionalityShort`, and the requested side is admissible only when `signedZScore >= 0.75`. Exact equality at the signed boundary is admissible; opposite-signed, zero, missing, or non-finite weak-band scores reject the requested side.
5. For `dsEfficiency > 0.40`, the strong-band no-regime witness is preserved when no saved tuple is present at all, but any explicit malformed saved tuple still blocks as malformed.
6. Admissibility is monotone non-increasing as efficiency falls across the `0.40` review-band boundary and the `0.25` chop boundary, as saved HMM mass drifts farther outside the `1e-3` normalization tolerance under otherwise identical weak-band snapshots, and as the signed weak-band zScore for a fixed requested side moves toward and through zero.
7. Mean-reversion-dominant weak-band snapshots remain blocked because `signalDirectionalitySnapshot` marks them `NON_DIRECTIONAL_MR`, and neither the signed helper nor the wrapper ever admits a snapshot with `dsNonDirectional = True`.
8. For fixed well-formed weak-band regime data, the long and short branches are exact mirror images under `zScore -> -zScore`; only the signed score for the requested side matters.
9. The side-agnostic wrapper is existential only: it may still admit a weak-band snapshot whose signed `zScore` supports one side, but it cannot admit zero or non-finite weak-band scores because neither signed side is admissible there.

Bounded executable obligations:

- `testSignalGateDirectionalityWeakBandFailClosed` covers the strong-band no-change witness, the `0.40` and `0.25` boundaries, side-specific signed `0.75` boundary acceptance for both long and short requests, opposite-sign rejection, zero rejection, malformed snapshot fail-closed behavior (including malformed efficiency and non-finite weak-band `zScore`), tuple sanity rejection, within-tolerance acceptance, preservation of the existing `NON_DIRECTIONAL_MR` veto, and the side-specific `Nothing` fail-closed case.
- `testSignalGateDirectionalityWeakBandMonotone` witnesses the non-increasing admissibility ladders as efficiency falls when regime probabilities are missing, as the long-side signed weak-band `zScore` moves down through the `+0.75` floor and through zero, as the short-side weak-band `zScore` moves up through the `-0.75` floor and through zero, and as saved HMM mass drifts outside the normalization tolerance in the weak band.

Proof sketch:

- `signalDirectionalityEntryAllowedForSide` still requires `dsEfficiency > directionalityChopEfficiencyMax`, so saved snapshots cannot reopen the documented chop veto by carrying a stale directional label.
- `directionalitySavedRegimeTupleOk` accepts saved regime probabilities only when they are either completely absent or present as a full HMM tuple whose components are finite in `[0,1]` and whose total mass is positive and within `1e-3` of `1`, so persisted partial or badly normalized tuples become malformed before admission.
- In the weak review band, `directionalityWeakBandSignedZScoreOk` applies one shared signed comparison: long requests use `zScore`, short requests use `-zScore`, and both branches compare the result against the same `0.75` floor.
- Because the helper only admits explicit finite scores satisfying `signedZScore >= 0.75`, moving the signed score toward zero or through zero, or supplying opposite-signed, zero, missing, or non-finite scores, can only remove admissible states.
- The side-agnostic wrapper is defined as the disjunction of the long and short checks, so it preserves compatibility while still rejecting weak-band zero or non-finite `zScore` samples because neither signed branch can pass there.
- `signalDirectionalityRegimeEvidence` reuses the same tuple sanity check for live `RegimeProbs`, keeping freshly built snapshots aligned with saved-snapshot admission semantics.
- The weak-band obligations are one-way: crossing from `> 0.40` into `<= 0.40`, drifting from an in-tolerance HMM tuple to an out-of-tolerance tuple, or moving the signed weak-band score for the requested side below `0.75` can only remove admissible states; the long and short branches differ only by the sign map.
- Mean-reversion dominance still blocks via the existing `dsNonDirectional`/`dsReason` contract produced by `signalDirectionalitySnapshot`, so the repair only shrinks the admissible set for weak-band states whose additive confirmation disagrees with the requested entry side and leaves the strong-band no-tuple path unchanged.

## Formal review-accounting non-interference contract

`classify_order_event_flow_role` and `build_report` in `haskell/scripts/review_bot_day.py`, interpreted against `TradeEntrySource` in `haskell/app/Trader/Trading.hs`, are treated as observability-only accounting for same-day order flow.

Clauses:

1. When saved side context is available, same-side flow is classified as `entry_or_add` and opposite-side flow is classified as `exit_or_flatten`.
2. Completed or open trades tagged `entrySource = adopted` still contribute side context, but opposite-side management orders for those carried positions remain `exit_or_flatten`.
3. `nonDirectionalOrderAttempts` counts only rows with `nonDirectionalVeto = True` and `flowRole = entry_or_add`.
4. `nonDirectionalExitOrFlattenEvents` and `nonDirectionalUnknownRoleEvents` remain separate buckets and cannot increase `nonDirectionalOrderAttempts`.
5. This review partition is non-interfering with live admissibility: `mkEntryGateState` only applies fresh-entry vetoes when `needsEntry` is true, while the exit, flatten, halt, and liquidation paths in `simulateEnsembleLongFlatVWithHLChecked` remain risk-reduction behavior for an already-held or adopted position.
6. Therefore startup-adopted position management cannot be reclassified by daily review output as a fresh weak-directionality entry failure.

Bounded executable obligations:

- `test_report_counts_non_directional_order_attempts` witnesses that a weak-directionality same-side attempt increments `nonDirectionalOrderAttempts`.
- `test_excludes_adopted_close_and_flatten_events_from_non_directional_attempts` witnesses that adopted-position carry-management orders are partitioned into `nonDirectionalExitOrFlattenEvents` instead of `nonDirectionalOrderAttempts`.
- `test_classifies_binance_auth_failures_by_order_flow_role` witnesses the same entry/add versus exit/flatten partition on the broader order-flow classifier so review-side role accounting stays deterministic outside the non-directional veto path.

Proof sketch:

- `classify_order_event_flow_role` first normalizes order side, then compares it with explicit side witnesses from order messages, completed trades, open trades, and nearby saved positions; those witnesses encode whether the order aligns with or opposes the already-held side, not whether the bot originally opened the carry.
- Because the review partition is side-relative, opposite-side management on an adopted long remains `exit_or_flatten` and opposite-side management on an adopted short remains `exit_or_flatten`; only same-side pressure can enter the `entry_or_add` bucket.
- `build_report` derives `nonDirectionalOrderAttempts` exclusively from `flowRole = entry_or_add` while emitting `nonDirectionalExitOrFlattenEvents` and `nonDirectionalUnknownRoleEvents` separately, so review-side weak-directionality counts cannot grow from risk-reduction actions.
- In `Trading.hs`, `mkEntryGateState` can only suppress `desiredSide1` when `needsEntry` is true, while `simulateEnsembleLongFlatVWithHLChecked` keeps carry-management exits on distinct flatten and halt paths. The review accounting is therefore observationally aligned with the live admission boundary and cannot reinterpret adopted-position exits as failed fresh entries.
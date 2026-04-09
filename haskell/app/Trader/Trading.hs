module Trader.Trading where

import qualified Data.Maybe
import Trader.SignalGates
    ( normalizeSignalThreshold,
      signalEntryEdgeSpikeOk,
      signalEntryFeeBufferOk,
      signalEntryHeadroomOk
    )

-- Existing imports and surrounding trading loop code remain unchanged.

desiredSize0 =
    if Data.Maybe.isNothing desiredSideRaw
        then 0
        else max 0 desiredSizeRaw

desiredSide0 =
    if desiredSize0 <= 0
        then Nothing
        else desiredSideRaw

-- Only fresh entries should consult the entry-only veto gates below.
needsEntry =
    Data.Maybe.isJust desiredSide0
        && desiredSide0 /= posSide

lstmEntryScale =
    if needsEntry && not volConfGateEnabled
        then lstmEntryScaleRaw
        else 1

trendOk =
    case desiredSide0 of
        Just side
            | needsEntry ->
                trendOkAt t trendLookbackStep side
        _ ->
            True

volOk =
    not needsEntry || volOkAt t

roundTripFeeFloor =
    let feePerSide = ecFee cfg
     in if isBad feePerSide
            then 0 / 0
            else 2 * max 0 feePerSide

snrScale =
    if minSignalToNoiseAdj <= 0
        then 1
        else case volPerBarAt t of
            Just vol
                | vol > 0 ->
                    clamp01 (max 0 edgeRaw / vol / minSignalToNoiseAdj)
            _ ->
                0

-- Reuse the same non-negative edge sample across entry-only veto gates.
entryEdge =
    Just (max 0 edgeRaw)

edgeSpikeOk =
    not needsEntry
        || signalEntryEdgeSpikeOk openThrAdj entryEdge

edgeHeadroomOk =
    not needsEntry
        || signalEntryHeadroomOk openThrAdj entryEdge

feeBufferOk =
    not needsEntry
        || signalEntryFeeBufferOk openThrAdj roundTripFeeFloor entryEdge

entryGatesOk =
    edgeSpikeOk
        && edgeHeadroomOk
        && feeBufferOk

desiredSide1 =
    if not trendOk
        || not volOk
        || not snrOk
        || not volTargetReady
        || not triLayerOk
        || not entryGatesOk
        then Nothing
        else desiredSide0

-- Existing downstream trading logic remains unchanged.
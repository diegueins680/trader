module Trader.Trading where

import qualified Data.Maybe
import Trader.SignalGates (
    normalizeSignalThreshold,
    signalEntryEdgeSpikeOk,
    signalEntryFeeBufferOk,
    signalEntryHeadroomOk,
 )

-- Existing imports and surrounding trading loop code remain unchanged.

normalizeEntryIntent ::
    (Double -> Bool) ->
    Maybe side ->
    Double ->
    Maybe side ->
    (Double, Maybe side, Bool)
normalizeEntryIntent isBad desiredSideRaw desiredSizeRaw posSide =
    let desiredSize0 =
            if Data.Maybe.isNothing desiredSideRaw || isBad desiredSizeRaw
                then 0
                else max 0 desiredSizeRaw
        desiredSide0 =
            if desiredSize0 <= 0
                then Nothing
                else desiredSideRaw
        needsEntry =
            Data.Maybe.isJust desiredSide0
                && desiredSide0 /= posSide
     in (desiredSize0, desiredSide0, needsEntry)

entryGatesAllow :: Bool -> Double -> Double -> Maybe Double -> Bool
entryGatesAllow needsEntry openThrAdj roundTripFeeFloor entryEdge =
    not needsEntry
        || ( signalEntryEdgeSpikeOk openThrAdj entryEdge
                && signalEntryHeadroomOk openThrAdj entryEdge
                && signalEntryFeeBufferOk openThrAdj roundTripFeeFloor entryEdge
           )

-- simulateEnsembleLongFlatVWithHLChecked keeps its existing outer structure and
-- restores the fresh-entry scope around the local bindings below:
--
--   let (desiredSize0, desiredSide0, needsEntry) =
--           normalizeEntryIntent isBad desiredSideRaw desiredSizeRaw posSide
--       lstmEntryScale =
--           if needsEntry && not volConfGateEnabled
--               then lstmEntryScaleRaw
--               else 1
--       trendOk =
--           case desiredSide0 of
--               Just side
--                   | needsEntry ->
--                       trendOkAt t trendLookbackStep side
--               _ ->
--                   True
--       volOk =
--           not needsEntry || volOkAt t
--       roundTripFeeFloor =
--           let feePerSide = ecFee cfg
--            in if isBad feePerSide
--                   then 0 / 0
--                   else 2 * max 0 feePerSide
--       snrScale =
--           if minSignalToNoiseAdj <= 0
--               then 1
--               else case volPerBarAt t of
--                   Just vol
--                       | vol > 0 ->
--                           clamp01 (max 0 edgeRaw / vol / minSignalToNoiseAdj)
--                   _ ->
--                       0
--       entryEdge =
--           Just (max 0 edgeRaw)
--       entryGatesOk =
--           entryGatesAllow needsEntry openThrAdj roundTripFeeFloor entryEdge
--       desiredSide1 =
--           if not trendOk
--               || not volOk
--               || not snrOk
--               || not volTargetReady
--               || not triLayerOk
--               || not entryGatesOk
--               then Nothing
--               else desiredSide0
--   in ...
--
-- Existing downstream trading logic remains unchanged.
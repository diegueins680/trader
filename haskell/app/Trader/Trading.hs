{-# LANGUAGE RecordWildCards #-}

module Trader.Trading where

import qualified Data.Maybe
import Trader.SignalGates (
    signalEntryEdgeSpikeOk,
    signalEntryFeeBufferOk,
    signalEntryHeadroomOk,
 )

data EntryGateInputs side t lookback cfg = EntryGateInputs
    { desiredSideRaw :: Maybe side
    , desiredSizeRaw :: Double
    , posSide :: Maybe side
    , volConfGateEnabled :: Bool
    , lstmEntryScaleRaw :: Double
    , trendOkAt :: t -> lookback -> side -> Bool
    , t :: t
    , trendLookbackStep :: lookback
    , volOkAt :: t -> Bool
    , ecFee :: cfg -> Double
    , cfg :: cfg
    , isBad :: Double -> Bool
    , minSignalToNoiseAdj :: Double
    , volPerBarAt :: t -> Maybe Double
    , clamp01 :: Double -> Double
    , edgeRaw :: Double
    , openThrAdj :: Double
    , snrOk :: Bool
    , volTargetReady :: Bool
    , triLayerOk :: Bool
    }

data EntryGateState side = EntryGateState
    { desiredSize0 :: Double
    , desiredSide0 :: Maybe side
    , needsEntry :: Bool
    , lstmEntryScale :: Double
    , trendOk :: Bool
    , volOk :: Bool
    , roundTripFeeFloor :: Double
    , snrScale :: Double
    , entryEdge :: Maybe Double
    , edgeSpikeOk :: Bool
    , edgeHeadroomOk :: Bool
    , feeBufferOk :: Bool
    , entryGatesOk :: Bool
    , desiredSide1 :: Maybe side
    }
    deriving (Eq, Show)

-- The live trading loop uses the same binding block to keep entry-only vetoes
-- fail-closed over one shared edge observation after the refactor repair.
mkEntryGateState :: (Eq side) => EntryGateInputs side t lookback cfg -> EntryGateState side
mkEntryGateState EntryGateInputs{..} =
    let desiredSize0 =
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
     in EntryGateState{..}

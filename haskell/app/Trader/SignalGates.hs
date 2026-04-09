module Trader.SignalGates (
    SignalThresholdBoundary (..),
    DirectionalitySnapshot (..),
    mkSignalThresholdBoundary,
    normalizeSignalThreshold,
    signalDirectionalitySnapshot,
    signalDirectionalityEntryAllowed,
    signalEntryHeadroomThresholdCap,
    signalEntryHeadroomOk,
    signalEntryFeeBufferOk,
    signalEntryEdgeSpikeOk,
    signalMetaLabelOk,
    signalMtfConsensusCheck,
    signalCrossAssetCheck,
    signalRegimeEdgeOk,
    signalFundingOiCheck,
    signalRunPostDirectionGates,
) where

...

signalEntryHeadroomThresholdCap :: Double -> Double
signalEntryHeadroomThresholdCap edge =
    let edge' =
            if finiteDouble edge && edge > 0
                then edge
                else 0
     in normalizeSignalThreshold (edge' / entryEdgeHeadroomMultiple)

normalizeSignalFeeFloor :: Double -> Maybe Double
normalizeSignalFeeFloor raw
    | finiteDouble raw = Just (max 0 raw)
    | otherwise = Nothing

signalEntryFeeBufferOk :: Double -> Double -> Maybe Double -> Bool
signalEntryFeeBufferOk openThreshold roundTripFeeFloor edgeForMethod =
    let requiredHeadroom = entryEdgeHeadroomMultiple * normalizeSignalThreshold openThreshold
     in case normalizeSignalFeeFloor roundTripFeeFloor of
            Nothing -> False
            Just feeFloor ->
                let requiredEdge = requiredHeadroom + feeFloor
                 in requiredEdge <= 0
                        || case edgeForMethod of
                            Just edge ->
                                finiteDouble edge
                                    && edge >= requiredEdge
                            Nothing -> False

signalEntryHeadroomOk :: Double -> Maybe Double -> Bool
signalEntryHeadroomOk openThreshold =
    signalEntryFeeBufferOk openThreshold 0

...
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

-- Existing threshold, directionality, and gate helpers remain unchanged.

finiteDouble :: Double -> Bool
finiteDouble raw = not (isNaN raw || isInfinite raw)

normalizeSignalThreshold :: Double -> Double
normalizeSignalThreshold raw
    | finiteDouble raw = max 0 raw
    | otherwise = 0

entryEdgeHeadroomMultiple :: Double
entryEdgeHeadroomMultiple = 1.5

normalizeSignalEntryEdge :: Maybe Double -> Maybe Double
normalizeSignalEntryEdge Nothing = Nothing
normalizeSignalEntryEdge (Just raw)
    | finiteDouble raw && raw >= 0 = Just raw
    | otherwise = Nothing

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

-- The fee-aware gate is a monotone strengthening of the headroom check:
-- invalid fee floors and malformed edges fail closed, and larger fee floors
-- require larger edge.
signalEntryFeeBufferOk :: Double -> Double -> Maybe Double -> Bool
signalEntryFeeBufferOk openThreshold roundTripFeeFloor edgeForMethod =
    let requiredHeadroom =
            entryEdgeHeadroomMultiple * normalizeSignalThreshold openThreshold
     in case (normalizeSignalFeeFloor roundTripFeeFloor, normalizeSignalEntryEdge edgeForMethod) of
            (Just feeFloor, Just edge) ->
                let requiredEdge = requiredHeadroom + feeFloor
                 in edge >= max 0 requiredEdge
            _ ->
                False

signalEntryHeadroomOk :: Double -> Maybe Double -> Bool
signalEntryHeadroomOk openThreshold =
    signalEntryFeeBufferOk openThreshold 0

-- Remaining gate implementations remain unchanged.
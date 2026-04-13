module Trader.SignalGates (
    entryEdgeHeadroomMultiple,
    entryEdgeSpikeMultiple,
    finiteDouble,
    normalizeSignalThreshold,
    normalizeSignalFeeFloor,
    normalizeSignalEntryEdge,
    signalEntryEdgeSpikeOk,
    signalEntryHeadroomOk,
    signalEntryFeeBufferOk,
    signalFreshEntryOk,
) where

finiteDouble :: Double -> Bool
finiteDouble value = not (isNaN value || isInfinite value)

normalizeSignalThreshold :: Double -> Double
normalizeSignalThreshold raw
    | finiteDouble raw = max 0 raw
    | otherwise = 0

normalizeSignalFeeFloor :: Double -> Maybe Double
normalizeSignalFeeFloor raw
    | finiteDouble raw && raw >= 0 = Just raw
    | otherwise = Nothing

normalizeSignalEntryEdge :: Double -> Maybe Double
normalizeSignalEntryEdge raw
    | finiteDouble raw && raw >= 0 = Just raw
    | otherwise = Just 0

entryEdgeHeadroomMultiple :: Double
entryEdgeHeadroomMultiple = 1.5

entryEdgeSpikeMultiple :: Double
entryEdgeSpikeMultiple = 3.0

signalEntryEdgeSpikeOk :: Double -> Maybe Double -> Bool
signalEntryEdgeSpikeOk openThreshold edgeForMethod
    | finiteDouble openThreshold && openThreshold >= 0 =
        let thresholdCap = entryEdgeSpikeMultiple * normalizeSignalThreshold openThreshold
         in maybe False (\edge -> finiteDouble edge && edge >= 0 && edge <= thresholdCap) edgeForMethod
    | otherwise = False

signalEntryHeadroomOk :: Double -> Maybe Double -> Bool
signalEntryHeadroomOk openThreshold edgeForMethod
    | finiteDouble openThreshold && openThreshold >= 0 =
        let requiredHeadroom =
                entryEdgeHeadroomMultiple * normalizeSignalThreshold openThreshold
         in maybe False (\edge -> finiteDouble edge && edge >= requiredHeadroom) edgeForMethod
    | otherwise = False

signalEntryFeeBufferOk :: Double -> Double -> Maybe Double -> Bool
signalEntryFeeBufferOk openThreshold roundTripFeeFloor edgeForMethod
    | finiteDouble openThreshold && openThreshold >= 0 =
        let requiredHeadroom =
                entryEdgeHeadroomMultiple * normalizeSignalThreshold openThreshold
         in case normalizeSignalFeeFloor roundTripFeeFloor of
                Nothing -> False
                Just feeFloor ->
                    let requiredEdge = requiredHeadroom + feeFloor
                     in maybe False (\edge -> finiteDouble edge && edge >= requiredEdge) edgeForMethod
    | otherwise = False

signalFreshEntryOk :: Double -> Double -> Maybe Double -> Bool
signalFreshEntryOk openThreshold roundTripFeeFloor edgeForMethod =
    signalEntryEdgeSpikeOk openThreshold edgeForMethod
        && signalEntryHeadroomOk openThreshold edgeForMethod
        && signalEntryFeeBufferOk openThreshold roundTripFeeFloor edgeForMethod

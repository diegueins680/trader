module Trader.SignalGates where

finiteDouble :: Double -> Bool
finiteDouble value = not (isNaN value) && not (isInfinite value)

normalizeSignalThreshold :: Double -> Double
normalizeSignalThreshold = id

normalizeSignalEntryEdge :: Double -> Maybe Double
normalizeSignalEntryEdge raw
    | finiteDouble raw = Just (max 0 raw)
    | otherwise = Just 0

entryEdgeHeadroomMultiple :: Double
entryEdgeHeadroomMultiple = 1.5

entryEdgeSpikeMultiple :: Double
entryEdgeSpikeMultiple = 1.7

signalEntryHeadroomThresholdCap :: Double -> Double
signalEntryHeadroomThresholdCap rawEdge =
    case normalizeSignalEntryEdge rawEdge of
        Just edge -> edge / entryEdgeHeadroomMultiple
        Nothing -> 0

normalizeSignalOpenThreshold :: Double -> Maybe Double
normalizeSignalOpenThreshold raw
    | finiteDouble raw && raw >= 0 = Just (normalizeSignalThreshold raw)
    | otherwise = Nothing

normalizeSignalFeeFloor :: Double -> Maybe Double
normalizeSignalFeeFloor raw
    | finiteDouble raw && raw >= 0 = Just raw
    | otherwise = Nothing

signalEntryHeadroomOk :: Double -> Maybe Double -> Bool
signalEntryHeadroomOk openThreshold = signalEntryFeeBufferOk openThreshold 0

signalEntryEdgeSpikeOk :: Double -> Maybe Double -> Bool
signalEntryEdgeSpikeOk openThreshold edgeForMethod =
    case normalizeSignalOpenThreshold openThreshold of
        Nothing -> False
        Just threshold ->
            let requiredEdge = entryEdgeSpikeMultiple * threshold
             in maybe False (\edge -> finiteDouble edge && edge >= requiredEdge) edgeForMethod

signalEntryFeeBufferOk :: Double -> Double -> Maybe Double -> Bool
signalEntryFeeBufferOk openThreshold roundTripFeeFloor edgeForMethod =
    case normalizeSignalOpenThreshold openThreshold of
        Nothing -> False
        Just threshold ->
            let requiredHeadroom = entryEdgeHeadroomMultiple * threshold
             in case normalizeSignalFeeFloor roundTripFeeFloor of
                    Nothing -> False
                    Just feeFloor ->
                        let requiredEdge = requiredHeadroom + feeFloor
                         in maybe False (\edge -> finiteDouble edge && edge >= requiredEdge) edgeForMethod
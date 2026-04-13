module Trader.SignalGates where

finiteDouble :: Double -> Bool
finiteDouble value = not (isNaN value || isInfinite value)

entryEdgeHeadroomMultiple :: Double
entryEdgeHeadroomMultiple = 1.5

entryEdgeSpikeMultiple :: Double
entryEdgeSpikeMultiple = 1.7

normalizeSignalThreshold :: Double -> Double
normalizeSignalThreshold raw
    | finiteDouble raw && raw >= 0 = raw
    | otherwise = 0

normalizeSignalFeeFloor :: Double -> Maybe Double
normalizeSignalFeeFloor raw
    | finiteDouble raw && raw >= 0 = Just raw
    | otherwise = Nothing

signalEntryHeadroomOk :: Double -> Maybe Double -> Bool
signalEntryHeadroomOk openThreshold = signalEntryFeeBufferOk openThreshold 0

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

signalEntryEdgeSpikeOk :: Double -> Maybe Double -> Bool
signalEntryEdgeSpikeOk openThreshold edgeForMethod
    | finiteDouble openThreshold && openThreshold >= 0 =
        let maxEdge =
                entryEdgeSpikeMultiple * normalizeSignalThreshold openThreshold
         in maybe False (\edge -> finiteDouble edge && edge >= 0 && edge <= maxEdge) edgeForMethod
    | otherwise = False
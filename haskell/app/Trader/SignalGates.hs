normalizeSignalFeeFloor :: Double -> Maybe Double
normalizeSignalFeeFloor raw
    | finiteDouble raw && raw >= 0 = Just raw
    | otherwise = Nothing

signalEntryFeeBufferOk :: Double -> Double -> Maybe Double -> Bool
signalEntryFeeBufferOk openThreshold roundTripFeeFloor edgeForMethod
    | finiteDouble openThreshold =
        let requiredHeadroom =
                entryEdgeHeadroomMultiple * normalizeSignalThreshold openThreshold
         in case normalizeSignalFeeFloor roundTripFeeFloor of
                Nothing -> False
                Just feeFloor ->
                    let requiredEdge = requiredHeadroom + feeFloor
                     in maybe False (\edge -> finiteDouble edge && edge >= requiredEdge) edgeForMethod
    | otherwise = False

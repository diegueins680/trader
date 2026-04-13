roundTripFeeFloor =
    let feePerSide = entryFeeOf cfg
     in if isBad feePerSide
            then 0 / 0
            else 2 * feePerSide
import Trader.SignalGates (normalizeSignalThreshold, signalEntryEdgeSpikeOk, signalEntryFeeBufferOk, signalEntryHeadroomOk)

...

                                        desiredSize0 =
                                            if isNothing desiredSideRaw
                                                then 0
                                                else max 0 desiredSizeRaw

                                        desiredSide0 =
                                            if desiredSize0 <= 0 then Nothing else desiredSideRaw

                                        needsEntry = Data.Maybe.isJust desiredSide0 && desiredSide0 /= posSide
                                        lstmEntryScale =
                                            if needsEntry && not volConfGateEnabled
                                                then lstmEntryScaleRaw
                                                else 1

                                        trendOk =
                                            case desiredSide0 of
                                                Just side | needsEntry -> trendOkAt t trendLookbackStep side
                                                _ -> True

                                        volOk = (not needsEntry || volOkAt t)

                                        roundTripFeeFloor =
                                            let feePerSide = ecFee cfg
                                             in if isBad feePerSide
                                                    then 0 / 0
                                                    else 2 * max 0 feePerSide

                                        snrScale =
                                            if minSignalToNoiseAdj <= 0
                                                then 1
                                                else case volPerBarAt t of
                                                    Just vol | vol > 0 -> clamp01 (max 0 edgeRaw / vol / minSignalToNoiseAdj)
                                                    _ -> 0

...

                                        edgeSpikeOk =
                                            not needsEntry || signalEntryEdgeSpikeOk openThrAdj (Just (max 0 edgeRaw))
                                        edgeHeadroomOk =
                                            not needsEntry || signalEntryHeadroomOk openThrAdj (Just (max 0 edgeRaw))
                                        feeBufferOk =
                                            not needsEntry || signalEntryFeeBufferOk openThrAdj roundTripFeeFloor (Just (max 0 edgeRaw))

...

                                        desiredSide1 =
                                            if not trendOk || not volOk || not snrOk || not volTargetReady || not triLayerOk || not edgeSpikeOk || not edgeHeadroomOk || not feeBufferOk
                                                then Nothing
                                                else desiredSide0

...
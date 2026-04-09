Apply the following replacements in the existing file.

1. Add the shared efficiency validator next to finiteDouble/directionalityProbOk:

directionalityEfficiencyOk :: Double -> Bool
directionalityEfficiencyOk eff =
    finiteDouble eff && eff >= 0 && eff <= 1

2. Replace directionalitySnapshotWellFormed with:

directionalitySnapshotWellFormed :: DirectionalitySnapshot -> Bool
directionalitySnapshotWellFormed snap =
    dsLookbackBars snap >= 3
        && finiteDouble (dsNetReturnPct snap)
        && finiteDouble (dsRealizedVolPct snap)
        && directionalityEfficiencyOk (dsEfficiency snap)
        && finiteDouble (dsZScore snap)
        && maybe True directionalityProbOk (dsTrendProb snap)
        && maybe True directionalityProbOk (dsMrProb snap)
        && maybe True directionalityProbOk (dsHighVolProb snap)
        && maybe True directionalityGapOk (dsRegimeGap snap)

3. In signalDirectionalitySnapshot, change the metricsOk block and the regime-evidence binding/reason selection to:

            metricsOk =
                finiteDouble netReturnPct
                    && finiteDouble netDirectional
                    && finiteDouble path
                    && directionalityEfficiencyOk efficiency
                    && finiteDouble realizedVol
                    && realizedVol >= 0
                    && finiteDouble zScore
         in if not metricsOk
                then Nothing
                else
                    let label
                            | realizedVol * 100 >= directionalityHighVolVolPct = "high-vol"
                            | efficiency >= directionalityTrendEfficiencyMin && abs zScore >= directionalityTrendZMin =
                                if netDirectional >= 0
                                    then "trend-up"
                                    else "trend-down"
                            | efficiency <= directionalityChopEfficiencyMax = "chop"
                            | otherwise = "range-drift"
                        (trendProb, mrProb, highVolProb, regimeLeader, regimeGap, mrDominant, regimeMalformed) =
                            signalDirectionalityRegimeEvidence regimeHysteresis mRegimes
                        mReason
                            | efficiency <= directionalityChopEfficiencyMax = Just "NON_DIRECTIONAL_CHOP"
                            | efficiency <= directionalityMrEfficiencyMax =
                                case mrDominant of
                                    Just True -> Just "NON_DIRECTIONAL_MR"
                                    Just False -> Nothing
                                    Nothing -> Just directionalityMalformedReason
                            | regimeMalformed = Just directionalityMalformedReason
                            | otherwise = Nothing
                     in Just
                            DirectionalitySnapshot
                                { dsLookbackBars = windowLen
                                , dsNetReturnPct = netReturnPct * 100
                                , dsRealizedVolPct = realizedVol * 100
                                , dsEfficiency = efficiency
                                , dsZScore = zScore
                                , dsLabel = label
                                , dsTrendProb = trendProb
                                , dsMrProb = mrProb
                                , dsHighVolProb = highVolProb
                                , dsRegimeLeader = regimeLeader
                                , dsRegimeGap = regimeGap
                                , dsNonDirectional = isJust mReason
                                , dsReason = mReason
                                }

4. Replace signalDirectionalityRegimeEvidence with:

signalDirectionalityRegimeEvidence ::
    Double ->
    Maybe RegimeProbs ->
    (Maybe Double, Maybe Double, Maybe Double, Maybe String, Maybe Double, Maybe Bool, Bool)
signalDirectionalityRegimeEvidence regimeHysteresis mRegimes =
    case mRegimes of
        Nothing -> (Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, False)
        Just regimes ->
            let trendProb = rpTrend regimes
                mrProb = rpMR regimes
                highVolProb = rpHighVol regimes
             in if all directionalityProbOk [trendProb, mrProb, highVolProb]
                    then
                        let ranked =
                                sortOn
                                    (Data.Ord.Down . snd)
                                    [ ("trend", trendProb)
                                    , ("mr", mrProb)
                                    , ("highVol", highVolProb)
                                    ]
                            gap =
                                case ranked of
                                    ((_, p1) : (_, p2) : _) -> Just (p1 - p2)
                                    _ -> Nothing
                            leader =
                                case ranked of
                                    ((name, _) : _) -> Just name
                                    _ -> Nothing
                            dominant =
                                case (leader, gap) of
                                    (Just "mr", Just g) | directionalityGapOk g -> Just (g >= max 0 regimeHysteresis)
                                    (Just _, Just g) | directionalityGapOk g -> Just False
                                    _ -> Nothing
                         in (Just trendProb, Just mrProb, Just highVolProb, leader, gap, dominant, False)
                    else (Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, True)
Apply the following edits.

1. Extend the Trader.SignalGates import list to include `signalDirectionalityEntryAllowed`.

Replace:

import Trader.SignalGates (DirectionalitySnapshot (..), SignalThresholdBoundary (..), mkSignalThresholdBoundary, normalizeSignalThreshold, signalCrossAssetCheck, signalDirectionalitySnapshot, signalEntryEdgeSpikeOk, signalEntryHeadroomOk, signalFundingOiCheck, signalMetaLabelOk, signalMtfConsensusCheck, signalRegimeEdgeOk, signalRunPostDirectionGates)

with:

import Trader.SignalGates (DirectionalitySnapshot (..), SignalThresholdBoundary (..), mkSignalThresholdBoundary, normalizeSignalThreshold, signalCrossAssetCheck, signalDirectionalityEntryAllowed, signalDirectionalitySnapshot, signalEntryEdgeSpikeOk, signalEntryHeadroomOk, signalFundingOiCheck, signalMetaLabelOk, signalMtfConsensusCheck, signalRegimeEdgeOk, signalRunPostDirectionGates)

2. Add this helper next to the other assert helpers:

assertMonotoneNonIncreasing :: String -> [Bool] -> IO ()
assertMonotoneNonIncreasing msg xs =
    assert msg (and (zipWith (>=) xs (drop 1 xs)))

3. Add these four run entries in the signal-gate block, immediately after `testSignalDirectionalityTrendAllowed`:

              , run "signal directionality malformed efficiency fails closed" testSignalDirectionalityMalformedEfficiencyFailsClosed
              , run "signal directionality malformed regimes fail closed" testSignalDirectionalityMalformedRegimeFailsClosed
              , run "signal directionality efficiency monotonicity holds" testSignalDirectionalityEfficiencyMonotonicity
              , run "signal directionality MR dominance monotonicity holds" testSignalDirectionalityMrDominanceMonotonicity

4. Add the following tests below the existing `testSignalDirectionalityTrendAllowed` definition:

testSignalDirectionalityMalformedEfficiencyFailsClosed :: IO ()
testSignalDirectionalityMalformedEfficiencyFailsClosed = do
    let snapshot =
            Just
                DirectionalitySnapshot
                    { dsLookbackBars = 24
                    , dsNetReturnPct = 1
                    , dsRealizedVolPct = 0.5
                    , dsEfficiency = 0 / 0
                    , dsZScore = 1.5
                    , dsLabel = "trend-up"
                    , dsTrendProb = Just 0.7
                    , dsMrProb = Just 0.2
                    , dsHighVolProb = Just 0.1
                    , dsRegimeLeader = Just "trend"
                    , dsRegimeGap = Just 0.5
                    , dsNonDirectional = False
                    , dsReason = Nothing
                    }
    assert "non-finite efficiency cannot reopen directionality entry" (not (signalDirectionalityEntryAllowed snapshot))

testSignalDirectionalityMalformedRegimeFailsClosed :: IO ()
testSignalDirectionalityMalformedRegimeFailsClosed = do
    let prices = V.fromList [100.0, 101.0, 102.0, 103.0, 104.0]
        regimes = Just (RegimeProbs 0.8 (0 / 0) 0.2)
        snapshot = signalDirectionalitySnapshot 0.05 regimes prices 4
    case snapshot of
        Nothing -> assert "expected directionality snapshot for malformed regime test" False
        Just snap -> do
            assert "strong trend label is still visible for diagnostics" (dsLabel snap == "trend-up")
            assert "malformed regime probabilities fail closed" (dsReason snap == Just "NON_DIRECTIONAL_MALFORMED" && dsNonDirectional snap)
            assert "malformed regime probabilities cannot open entries" (not (signalDirectionalityEntryAllowed snapshot))

testSignalDirectionalityEfficiencyMonotonicity :: IO ()
testSignalDirectionalityEfficiencyMonotonicity = do
    let regimes = Just (RegimeProbs 0.7 0.2 0.1)
        priceCases =
            [ V.fromList [100.0, 101.0, 102.0, 103.0, 104.0]
            , V.fromList [100.0, 102.0, 101.0, 103.0, 102.0]
            , V.fromList [100.0, 101.0, 100.0, 101.0, 100.0]
            ]
        snapshots = map (\prices -> signalDirectionalitySnapshot 0.05 regimes prices 4) priceCases
        efficiencies = [dsEfficiency snap | Just snap <- snapshots]
        alloweds = map signalDirectionalityEntryAllowed snapshots
    assert "expected three directionality snapshots in the efficiency ladder" (length efficiencies == 3)
    assert "efficiency ladder is strictly descending" (and (zipWith (>) efficiencies (drop 1 efficiencies)))
    assert "efficiency ladder keeps the expected allow/block shape" (alloweds == [True, True, False])
    assertMonotoneNonIncreasing "lower efficiency cannot turn a blocked state into allowed" alloweds

testSignalDirectionalityMrDominanceMonotonicity :: IO ()
testSignalDirectionalityMrDominanceMonotonicity = do
    let prices = V.fromList [100.0, 102.0, 101.0, 103.0, 102.0]
        regimes =
            [ Just (RegimeProbs 0.70 0.20 0.10)
            , Just (RegimeProbs 0.47 0.49 0.04)
            , Just (RegimeProbs 0.10 0.80 0.10)
            , Just (RegimeProbs 0.10 (0 / 0) 0.10)
            ]
        snapshots = map (\regime -> signalDirectionalitySnapshot 0.05 regime prices 4) regimes
        alloweds = map signalDirectionalityEntryAllowed snapshots
        reasons = map (>>= dsReason) snapshots
    assert "MR-dominance ladder keeps the expected allow/block shape" (alloweds == [True, True, False, False])
    assertMonotoneNonIncreasing "greater MR dominance or malformed regime inputs cannot reopen blocked states" alloweds
    assert "strong MR dominance is explicitly vetoed" (reasons !! 2 == Just "NON_DIRECTIONAL_MR")
    assert "malformed regime evidence is explicitly vetoed" (reasons !! 3 == Just "NON_DIRECTIONAL_MALFORMED")
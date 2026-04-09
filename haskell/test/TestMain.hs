import Trader.SignalGates (DirectionalitySnapshot (..), SignalThresholdBoundary (..), mkSignalThresholdBoundary, normalizeSignalThreshold, signalCrossAssetCheck, signalDirectionalityEntryAllowed, signalDirectionalitySnapshot, signalEntryEdgeSpikeOk, signalEntryFeeBufferOk, signalEntryHeadroomOk, signalFundingOiCheck, signalMetaLabelOk, signalMtfConsensusCheck, signalRegimeEdgeOk, signalRunPostDirectionGates)

...

              , run "signal gate rejects low-headroom entries" testSignalGateEntryHeadroom
              , run "signal gate rejects marginal fee-adjusted entries" testSignalGateEntryFeeBuffer
              , run "signal gate fee monotonicity holds" testSignalGateEntryFeeBufferMonotoneFees
              , run "signal gate edge monotonicity holds under fees" testSignalGateEntryFeeBufferMonotoneEdge
              , run "signal gate fee-aware malformed inputs fail closed" testSignalGateEntryFeeBufferFailsClosed
              , run "signal gate rejects entry edge spikes" testSignalGateEntryEdgeSpike

...

testSignalGateEntryFeeBuffer :: IO ()
testSignalGateEntryFeeBuffer = do
    assert "fee-aware gate accepts edges exactly at headroom-plus-fee equality" (signalEntryFeeBufferOk 0.01 0.002 (Just 0.017))
    assert "fee-aware gate rejects edges below headroom-plus-fee requirement" (not (signalEntryFeeBufferOk 0.01 0.002 (Just 0.016999)))
    assert "fee-aware gate still applies when threshold headroom is zero but fees remain" (signalEntryFeeBufferOk 0 0.002 (Just 0.002))
    assert "fee-aware gate rejects missing edge when the fee buffer is active" (not (signalEntryFeeBufferOk 0 0.002 Nothing))

testSignalGateEntryFeeBufferMonotoneFees :: IO ()
testSignalGateEntryFeeBufferMonotoneFees = do
    let alloweds = map (\fee -> signalEntryFeeBufferOk 0.01 fee (Just 0.018)) [0, 0.002, 0.0035, 0.004]
    assert "fee ladder keeps the expected allow/block shape" (alloweds == [True, True, False, False])
    assertMonotoneNonIncreasing "higher fees cannot reopen a blocked fee-aware entry" alloweds

testSignalGateEntryFeeBufferMonotoneEdge :: IO ()
testSignalGateEntryFeeBufferMonotoneEdge = do
    let alloweds = map (signalEntryFeeBufferOk 0.01 0.002 . Just) [0.02, 0.017, 0.016, 0.015]
    assert "edge ladder keeps the expected allow/block shape under a fixed fee floor" (alloweds == [True, True, False, False])
    assertMonotoneNonIncreasing "lower edge cannot reopen a blocked fee-aware entry" alloweds

testSignalGateEntryFeeBufferFailsClosed :: IO ()
testSignalGateEntryFeeBufferFailsClosed = do
    assert "non-finite fee floor fails closed" (not (signalEntryFeeBufferOk 0.01 (0 / 0) (Just 0.05)))
    assert "non-finite edge fails closed" (not (signalEntryFeeBufferOk 0.01 0.002 (Just (1 / 0))))
    assert "negative fee floors are clamped to zero instead of reopening entries" (signalEntryFeeBufferOk 0.01 (-0.001) (Just 0.015))

...
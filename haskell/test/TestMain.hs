import Trader.SignalGates
    ( DirectionalitySnapshot (..)
    , SignalThresholdBoundary (..)
    , mkSignalThresholdBoundary
    , normalizeSignalThreshold
    , signalCrossAssetCheck
    , signalDirectionalityEntryAllowed
    , signalDirectionalitySnapshot
    , signalEntryEdgeSpikeOk
    , signalEntryFeeBufferOk
    , signalEntryHeadroomOk
    , signalFundingOiCheck
    , signalMetaLabelOk
    , signalMtfConsensusCheck
    , signalRegimeEdgeOk
    , signalRunPostDirectionGates
    )

-- Existing test harness imports and helpers remain unchanged.

main :: IO ()
main = do
    run "signal gate rejects low-headroom entries" testSignalGateEntryHeadroom
    run "signal gate rejects marginal fee-adjusted entries" testSignalGateEntryFeeBuffer
    run "signal gate fee monotonicity holds" testSignalGateEntryFeeBufferMonotoneFees
    run "signal gate edge monotonicity holds under fees" testSignalGateEntryFeeBufferMonotoneEdge
    run "signal gate fee-aware malformed inputs fail closed" testSignalGateEntryFeeBufferFailsClosed
    run "signal gate rejects entry edge spikes" testSignalGateEntryEdgeSpike

-- Bounded executable obligations for the fee-aware entry gate cover:
-- zero-fee specialization, boundary acceptance, strict-below rejection,
-- monotone non-increasing admissibility, once-blocked-stays-blocked,
-- negative-fee clamping, non-finite-input fail-closed behavior,
-- and the independent spike-veto conjunction on the same entry-only path.
testSignalGateEntryFeeBuffer :: IO ()
testSignalGateEntryFeeBuffer = do
    assert
        "fee-aware gate accepts edges exactly at headroom-plus-fee equality"
        (signalEntryFeeBufferOk 0.01 0.002 (Just 0.017))
    assert
        "fee-aware gate rejects edges below headroom-plus-fee requirement"
        (not (signalEntryFeeBufferOk 0.01 0.002 (Just 0.016999)))
    assert
        "fee-aware gate still applies when threshold headroom is zero but fees remain"
        (signalEntryFeeBufferOk 0 0.002 (Just 0.002))
    assert
        "fee-aware gate rejects missing edge when the fee buffer is active"
        (not (signalEntryFeeBufferOk 0 0.002 Nothing))
    assert
        "zero-fee specialization fails closed on missing edge"
        (not (signalEntryFeeBufferOk 0.01 0 Nothing))
    assert
        "zero-fee specialization accepts equality at the pure headroom boundary"
        (signalEntryFeeBufferOk 0.01 0 (Just 0.015))
    assert
        "headroom-only helper remains the zero-fee specialization"
        (signalEntryHeadroomOk 0.01 (Just 0.015) == signalEntryFeeBufferOk 0.01 0 (Just 0.015))

testSignalGateEntryFeeBufferMonotoneFees :: IO ()
testSignalGateEntryFeeBufferMonotoneFees = do
    let alloweds =
            map
                (\fee -> signalEntryFeeBufferOk 0.01 fee (Just 0.018))
                [0, 0.002, 0.0035, 0.004]
    let blockedLadder =
            map
                (\fee -> signalEntryFeeBufferOk 0.01 fee (Just 0.0165))
                [0.002, 0.0035, 0.004]
    assert
        "fee ladder keeps the expected allow/block shape"
        (alloweds == [True, True, False, False])
    assertMonotoneNonIncreasing
        "higher fees cannot reopen a blocked fee-aware entry"
        alloweds
    assert
        "once blocked at a given fee floor, larger fee floors stay blocked"
        (blockedLadder == [False, False, False])
    assertMonotoneNonIncreasing
        "blocked fee-aware states stay blocked as fee floors rise"
        blockedLadder

testSignalGateEntryFeeBufferMonotoneEdge :: IO ()
testSignalGateEntryFeeBufferMonotoneEdge = do
    let alloweds =
            map (signalEntryFeeBufferOk 0.01 0.002 . Just) [0.02, 0.017, 0.016, 0.015]
    assert
        "edge ladder keeps the expected allow/block shape under a fixed fee floor"
        (alloweds == [True, True, False, False])
    assertMonotoneNonIncreasing
        "lower edge cannot reopen a blocked fee-aware entry"
        alloweds

testSignalGateEntryFeeBufferFailsClosed :: IO ()
testSignalGateEntryFeeBufferFailsClosed = do
    assert
        "non-finite fee floor fails closed"
        (not (signalEntryFeeBufferOk 0.01 (0 / 0) (Just 0.05)))
    assert
        "non-finite edge fails closed"
        (not (signalEntryFeeBufferOk 0.01 0.002 (Just (1 / 0))))
    assert
        "negative fee floors stay clamped at zero below the pure headroom boundary"
        (not (signalEntryFeeBufferOk 0.01 (-0.001) (Just 0.014999)))
    assert
        "negative fee floors are clamped to zero instead of reopening entries"
        (signalEntryFeeBufferOk 0.01 (-0.001) (Just 0.015))

-- Remaining signal-gate tests remain unchanged.
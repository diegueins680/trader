import Trader.SignalGates (
    DirectionalitySnapshot (..),
    SignalThresholdBoundary (..),
    mkSignalThresholdBoundary,
    normalizeSignalThreshold,
    signalCrossAssetCheck,
    signalDirectionalityEntryAllowed,
    signalDirectionalitySnapshot,
    signalEntryEdgeSpikeOk,
    signalEntryFeeBufferOk,
    signalEntryHeadroomOk,
    signalEntryHeadroomThresholdCap,
    signalFundingOiCheck,
    signalMetaLabelOk,
    signalMtfConsensusCheck,
    signalRegimeEdgeOk,
    signalRunPostDirectionGates,
 )

-- Existing test harness imports and helpers remain unchanged.

main :: IO ()
main = do
    run "signal gate rejects low-headroom entries" testSignalGateEntryHeadroom
    run "signal gate headroom threshold cap tracks 1.5x rule" testSignalGateEntryHeadroomThresholdCap
    run "signal gate rejects marginal fee-adjusted entries" testSignalGateEntryFeeBuffer
    run "signal gate fee monotonicity holds" testSignalGateEntryFeeBufferMonotoneFees
    run "signal gate edge monotonicity holds under fees" testSignalGateEntryFeeBufferMonotoneEdge
    run "signal gate fee buffer stays subordinate to spike/headroom vetoes" testSignalGateEntryFeeBufferSubordinate
    run "signal gate shared entryEdge conjunction stays fail closed" testSignalGateEntryConjunctiveSharedEdge
    run "signal gate fee-aware malformed inputs fail closed" testSignalGateEntryFeeBufferFailsClosed
    run "signal gate facade stays fail closed and monotone" testSignalGateFacadeFailClosedMonotone
    run "signal gate rejects entry edge spikes" testSignalGateEntryEdgeSpike

-- Bounded executable obligations for the fee-aware entry gate and restored facade cover:
-- the 1.5x headroom-threshold-cap witness, zero-fee specialization,
-- boundary acceptance, strict-below rejection, monotone non-increasing
-- admissibility, once-blocked-stays-blocked, negative-fee clamping,
-- missing/non-finite-input fail-closed behavior, and preservation of the
-- shared non-negative entryEdge sample across the independent spike veto
-- and the fee/headroom gates on the fresh-entry path, including the
-- conjunction fact that the fee buffer may veto but cannot reopen an
-- entry already blocked by the upstream spike/headroom pair. The repaired
-- facade proof also locks the zero-fee/headroom equivalence and shows that
-- no restored alias can reopen an entry already rejected by the core
-- conjunction.
testSignalGateEntryHeadroomThresholdCap :: IO ()
testSignalGateEntryHeadroomThresholdCap = do
    let cappedOpenThreshold = signalEntryHeadroomThresholdCap 0.015
    assert
        "headroom threshold cap reconstructs the 1.5x admissible boundary"
        ( signalEntryHeadroomOk cappedOpenThreshold (Just 0.015)
            && not (signalEntryHeadroomOk cappedOpenThreshold (Just 0.014999))
        )
    assert
        "headroom threshold cap matches the zero-fee fee-buffer boundary"
        ( signalEntryFeeBufferOk cappedOpenThreshold 0 (Just 0.015)
            && not (signalEntryFeeBufferOk cappedOpenThreshold 0 (Just 0.014999))
        )
    assert
        "headroom threshold cap normalizes malformed or negative edges to zero"
        ( signalEntryHeadroomThresholdCap (0 / 0) == 0
            && signalEntryHeadroomThresholdCap (1 / 0) == 0
            && signalEntryHeadroomThresholdCap (-0.01) == 0
        )

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

testSignalGateEntryFeeBufferSubordinate :: IO ()
testSignalGateEntryFeeBufferSubordinate = do
    let feeOnlyVetoEdge = Just 0.015
    let passesSpikeAndHeadroom =
            signalEntryEdgeSpikeOk 0.01 feeOnlyVetoEdge
                && signalEntryHeadroomOk 0.01 feeOnlyVetoEdge
    assert
        "fee buffer may veto an entry that passes spike and pure headroom gates"
        ( passesSpikeAndHeadroom
            && not (signalEntryFeeBufferOk 0.01 0.002 feeOnlyVetoEdge)
        )
    let headroomBlockedEdge = Just 0.014999
    let blockedBySpikeOrHeadroom =
            not
                ( signalEntryEdgeSpikeOk 0.01 headroomBlockedEdge
                    && signalEntryHeadroomOk 0.01 headroomBlockedEdge
                )
    assert
        "fee buffer cannot admit an entry already blocked by the spike/headroom conjunction"
        ( blockedBySpikeOrHeadroom
            && not
                ( signalEntryEdgeSpikeOk 0.01 headroomBlockedEdge
                    && signalEntryHeadroomOk 0.01 headroomBlockedEdge
                    && signalEntryFeeBufferOk 0.01 0.002 headroomBlockedEdge
                )
        )

testSignalGateEntryConjunctiveSharedEdge :: IO ()
testSignalGateEntryConjunctiveSharedEdge = do
    let entryGatesOk openThr roundTripFee edge =
            signalEntryEdgeSpikeOk openThr edge
                && signalEntryHeadroomOk openThr edge
                && signalEntryFeeBufferOk openThr roundTripFee edge
    assert
        "shared entryEdge conjunction admits only when every entry veto passes"
        (entryGatesOk 0.01 0 (Just 0.015))
    assert
        "shared entryEdge conjunction stays closed when the fee buffer vetoes"
        (not (entryGatesOk 0.01 0.002 (Just 0.015)))
    assert
        "shared entryEdge conjunction fails closed on malformed input"
        (not (entryGatesOk 0.01 0.002 Nothing))

testSignalGateEntryFeeBufferFailsClosed :: IO ()
testSignalGateEntryFeeBufferFailsClosed = do
    assert
        "non-finite fee floor fails closed"
        (not (signalEntryFeeBufferOk 0.01 (0 / 0) (Just 0.05)))
    assert
        "infinite fee floor fails closed"
        (not (signalEntryFeeBufferOk 0.01 (1 / 0) (Just 0.05)))
    assert
        "non-finite edge fails closed"
        (not (signalEntryFeeBufferOk 0.01 0.002 (Just (1 / 0))))
    assert
        "NaN edge fails closed"
        (not (signalEntryFeeBufferOk 0.01 0.002 (Just (0 / 0))))
    assert
        "negative fee floors stay clamped at zero below the pure headroom boundary"
        (not (signalEntryFeeBufferOk 0.01 (-0.001) (Just 0.014999)))
    assert
        "negative fee floors are clamped to zero instead of reopening entries"
        (signalEntryFeeBufferOk 0.01 (-0.001) (Just 0.015))

testSignalGateFacadeFailClosedMonotone :: IO ()
testSignalGateFacadeFailClosedMonotone = do
    let SignalThresholdBoundary openThreshold requiredEdge = mkSignalThresholdBoundary 0.01
    let boundedEdges =
            [Nothing, Just (0 / 0), Just 0.014999, Just requiredEdge, Just 0.02]
    let coreGate edge =
            signalEntryEdgeSpikeOk openThreshold edge
                && signalEntryHeadroomOk openThreshold edge
                && signalEntryFeeBufferOk openThreshold 0 edge
    let facadeGate edge =
            signalRunPostDirectionGates
                [ signalDirectionalityEntryAllowed (signalDirectionalitySnapshot (coreGate edge))
                , signalMetaLabelOk (coreGate edge)
                , signalMtfConsensusCheck (coreGate edge)
                , signalCrossAssetCheck (coreGate edge)
                , signalRegimeEdgeOk (coreGate edge)
                , signalFundingOiCheck (coreGate edge)
                ]
    assert
        "restored threshold boundary carries the same headroom witness"
        ( signalEntryHeadroomOk openThreshold (Just requiredEdge)
            && signalEntryFeeBufferOk openThreshold 0 (Just requiredEdge)
            && not (signalEntryFeeBufferOk openThreshold 0 (Just (requiredEdge - 0.000001)))
        )
    assert
        "zero-fee specialization agrees with headroom-only gating across bounded samples"
        ( and
            [ signalEntryHeadroomOk openThreshold edge
                == signalEntryFeeBufferOk openThreshold 0 edge
            | edge <- boundedEdges
            ]
        )
    assert
        "directionality snapshot stays fail closed"
        ( not (signalDirectionalityEntryAllowed (DirectionalitySnapshot False))
            && signalDirectionalityEntryAllowed (signalDirectionalitySnapshot True)
        )
    assert
        "restored facade cannot admit an entry the core conjunction rejects"
        (and [not (facadeGate edge) || coreGate edge | edge <- boundedEdges])
    assert
        "post-direction gate conjunction fails closed on empty or blocked inputs"
        ( not (signalRunPostDirectionGates [])
            && not (signalRunPostDirectionGates [True, False, True])
            && signalRunPostDirectionGates [True, True, True]
        )

-- Remaining signal-gate tests, including the spike-veto witness, remain unchanged.
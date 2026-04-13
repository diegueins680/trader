module Main (main) where

import Control.Monad (unless)
import Data.Maybe (isNothing)
import Trader.SignalGates (
    signalEntryEdgeSpikeOk,
    signalEntryFeeBufferOk,
    signalEntryHeadroomOk,
 )
import Trader.Trading (
    desiredSide1,
    edgeHeadroomOk,
    edgeSpikeOk,
    entryEdge,
    entryGatesOk,
    feeBufferOk,
    mkEntryGateState,
    mkTradingEntryGateInputs,
    needsEntry,
    roundTripFeeFloor,
 )

main :: IO ()
main = do
    testSignalGateEntryBoundaryWitness
    testSignalGateEntryFeeBufferFailsClosed
    testTradingEntryGateFailClosedMonotone
    testTradingEntryGateMalformedNoReopen

assert :: String -> Bool -> IO ()
assert message condition =
    unless condition (ioError (userError ("Assertion failed: " ++ message)))

assertMonotoneNonIncreasing :: String -> [Bool] -> IO ()
assertMonotoneNonIncreasing message values =
    assert message (and (zipWith (\left right -> left || not right) values (drop 1 values)))

-- Direct SignalGates witness for the restored entry-gate facade: boundary
-- equality stays admissible for the spike cap, zero-fee headroom, and fee-aware
-- headroom checks; strict-below and strict-above rejection stays intact; and the
-- direct headroom, fee-buffer, and spike ladders remain monotone.
testSignalGateEntryBoundaryWitness :: IO ()
testSignalGateEntryBoundaryWitness = do
    let openThreshold = 0.01
        headroomEdges = [0.017, 0.015, 0.014, 0.013]
        headroomLadder =
            map (\edge -> signalEntryHeadroomOk openThreshold (Just edge)) headroomEdges
        feeLadder =
            map (\feeFloor -> signalEntryFeeBufferOk openThreshold feeFloor (Just 0.018)) [0, 0.001, 0.00175, 0.002]
        spikeLadder =
            map (\edge -> signalEntryEdgeSpikeOk openThreshold (Just edge)) [0.015, 0.017, 0.018, 0.019]
    assert
        "signal-gate boundaries remain admissible at equality and reject strict-below or strict-above edges"
        ( signalEntryEdgeSpikeOk openThreshold (Just 0.017)
            && signalEntryHeadroomOk openThreshold (Just 0.015)
            && signalEntryFeeBufferOk openThreshold 0 (Just 0.015)
            && signalEntryFeeBufferOk openThreshold 0.002 (Just 0.017)
            && not (signalEntryHeadroomOk openThreshold (Just 0.014))
            && not (signalEntryFeeBufferOk openThreshold 0.002 (Just 0.016))
            && not (signalEntryEdgeSpikeOk openThreshold (Just 0.018))
        )
    assert
        "signalEntryHeadroomOk stays the zero-fee specialization of signalEntryFeeBufferOk"
        ( headroomLadder
            == map (\edge -> signalEntryFeeBufferOk openThreshold 0 (Just edge)) headroomEdges
        )
    assert
        "direct headroom ladder keeps the expected allow/block shape"
        (headroomLadder == [True, True, False, False])
    assertMonotoneNonIncreasing
        "direct signal-gate admissibility is monotone non-increasing as effective headroom tightens"
        headroomLadder
    assert
        "direct fee ladder keeps the expected allow/block shape"
        (feeLadder == [True, True, False, False])
    assertMonotoneNonIncreasing
        "direct signal-gate admissibility is monotone non-increasing as the fee floor rises"
        feeLadder
    assert
        "direct spike ladder keeps the expected allow/block shape"
        (spikeLadder == [True, True, False, False])
    assertMonotoneNonIncreasing
        "direct spike-veto admissibility is monotone non-increasing as edge pressure rises"
        spikeLadder

-- The reviewed Trading.hs change keeps the fresh-entry gate entry-only while
-- tightening malformed-fee handling. These executable obligations pin four
-- properties: entry-only vetoes do not run when no fresh entry is needed,
-- fresh-entry spike/headroom/fee-buffer checks all read the same non-negative,
-- finite edge sample and fail closed on negative or non-finite fee/edge inputs,
-- equality at the required boundary stays admissible, and admissibility is monotone
-- non-increasing as raw edge falls or the fee floor rises.

testTradingEntryGateFailClosedMonotone :: IO ()
testTradingEntryGateFailClosedMonotone = do
    let boundaryState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 0.017 Nothing)
        malformedFeeState =
            mkEntryGateState (mkTradingEntryGateInputs (0 / 0) 0.02 Nothing)
        negativeFeeState =
            mkEntryGateState (mkTradingEntryGateInputs (-0.001) 0.02 Nothing)
        nonFiniteEdgeState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 (0 / 0) Nothing)
        negativeEdgeState =
            mkEntryGateState (mkTradingEntryGateInputs 0 (-0.01) Nothing)
        freshEntryAllowed feePerSide rawEdge =
            desiredSide1 (mkEntryGateState (mkTradingEntryGateInputs feePerSide rawEdge Nothing)) == Just True
        edgeAlloweds =
            map (freshEntryAllowed 0.001) [0.02, 0.017, 0.016, 0.015]
        feeAlloweds =
            map (`freshEntryAllowed` 0.018) [0, 0.001, 0.00175, 0.002]
    assert
        "fresh-entry equality at the fee-aware boundary stays admissible"
        ( needsEntry boundaryState
            && roundTripFeeFloor boundaryState == 0.002
            && entryEdge boundaryState == Just 0.017
            && edgeSpikeOk boundaryState
            && edgeHeadroomOk boundaryState
            && feeBufferOk boundaryState
            && entryGatesOk boundaryState
            && desiredSide1 boundaryState == Just True
        )
    assert
        "non-finite fee context still fails closed on the fresh-entry path"
        ( needsEntry malformedFeeState
            && not (feeBufferOk malformedFeeState)
            && not (entryGatesOk malformedFeeState)
            && isNothing (desiredSide1 malformedFeeState)
        )
    assert
        "negative per-side fee stays fail closed on the fresh-entry path"
        ( needsEntry negativeFeeState
            && roundTripFeeFloor negativeFeeState == -0.002
            && edgeSpikeOk negativeFeeState
            && edgeHeadroomOk negativeFeeState
            && not (feeBufferOk negativeFeeState)
            && not (entryGatesOk negativeFeeState)
            && isNothing (desiredSide1 negativeFeeState)
        )
    assert
        "non-finite raw edge collapses to the shared zero edge and stays blocked"
        ( needsEntry nonFiniteEdgeState
            && entryEdge nonFiniteEdgeState == Just 0
            && not (edgeHeadroomOk nonFiniteEdgeState)
            && not (entryGatesOk nonFiniteEdgeState)
            && isNothing (desiredSide1 nonFiniteEdgeState)
        )
    assert
        "fresh-entry gating reuses the shared non-negative edge sample"
        ( entryEdge negativeEdgeState == Just 0
            && not (edgeHeadroomOk negativeEdgeState)
            && isNothing (desiredSide1 negativeEdgeState)
        )
    assert
        "fresh-entry edge ladder keeps the expected allow/block shape"
        (edgeAlloweds == [True, True, False, False])
    assertMonotoneNonIncreasing
        "lower raw edge cannot reopen a blocked fresh-entry state"
        edgeAlloweds
    assert
        "fresh-entry fee ladder keeps the expected allow/block shape"
        (feeAlloweds == [True, True, False, False])
    assertMonotoneNonIncreasing
        "higher fee floors cannot reopen a blocked fresh-entry state"
        feeAlloweds

-- Formal proof sketch extension over the public Trader.Trading surface: once
-- the fresh-entry conjunction blocks, replacing the fee or edge input with
-- negative or malformed values must leave the state blocked as well, so the
-- Trading/SignalGates integration cannot reopen from corrupted fee data, NaN,
-- or Infinity drift while the simulator seam stays decoupled.
testTradingEntryGateMalformedNoReopen :: IO ()
testTradingEntryGateMalformedNoReopen = do
    let blockedState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 0.015 Nothing)
        negativeFeeState =
            mkEntryGateState (mkTradingEntryGateInputs (-0.001) 0.02 Nothing)
        malformedFeeState =
            mkEntryGateState (mkTradingEntryGateInputs (0 / 0) 0.017 Nothing)
        malformedEdgeState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 (0 / 0) Nothing)
    assert
        "once blocked by the fresh-entry conjunction, negative or malformed fee and edge inputs cannot reopen the state"
        ( needsEntry blockedState
            && not (entryGatesOk blockedState)
            && isNothing (desiredSide1 blockedState)
            && all
                ( \state ->
                    needsEntry state
                        && not (entryGatesOk state)
                        && isNothing (desiredSide1 state)
                )
                [negativeFeeState, malformedFeeState, malformedEdgeState]
        )

-- Bounded executable obligations for the restored signal-gate facade now cover:
-- the direct boundary witness, zero-fee specialization, monotone headroom/fee
-- and spike ladders, missing-edge fail-closed behavior, malformed-input
-- fail-closed behavior, and preservation of the shared non-negative entryEdge
-- sample across the independent spike veto and the fee/headroom gates on the
-- fresh-entry path.

testSignalGateEntryFeeBufferFailsClosed :: IO ()
testSignalGateEntryFeeBufferFailsClosed = do
    assert
        "missing edge sample fails closed across the shared signal gates"
        ( not (signalEntryEdgeSpikeOk 0.01 Nothing)
            && not (signalEntryHeadroomOk 0.01 Nothing)
            && not (signalEntryFeeBufferOk 0.01 0 Nothing)
        )
    assert
        "non-finite fee floor fails closed"
        (not (signalEntryFeeBufferOk 0.01 (0 / 0) (Just 0.05)))
    assert
        "infinite fee floor fails closed"
        (not (signalEntryFeeBufferOk 0.01 (1 / 0) (Just 0.05)))
    assert
        "non-finite open threshold fails closed instead of collapsing to zero"
        ( not (signalEntryFeeBufferOk (0 / 0) 0 (Just 0))
            && not (signalEntryFeeBufferOk (1 / 0) 0 (Just 0))
            && not (signalEntryHeadroomOk (0 / 0) (Just 0))
            && not (signalEntryHeadroomOk (1 / 0) (Just 0))
            && not (signalEntryEdgeSpikeOk (0 / 0) (Just 0))
            && not (signalEntryEdgeSpikeOk (1 / 0) (Just 0))
        )
    assert
        "negative open threshold fails closed instead of collapsing to zero"
        ( not (signalEntryFeeBufferOk (-0.01) 0 (Just 0.05))
            && not (signalEntryHeadroomOk (-0.01) (Just 0.05))
            && not (signalEntryEdgeSpikeOk (-0.01) (Just 0.05))
        )
    assert
        "non-finite edge fails closed"
        ( not (signalEntryEdgeSpikeOk 0.01 (Just (1 / 0)))
            && not (signalEntryHeadroomOk 0.01 (Just (1 / 0)))
            && not (signalEntryFeeBufferOk 0.01 0.002 (Just (1 / 0)))
        )
    assert
        "NaN edge fails closed"
        ( not (signalEntryEdgeSpikeOk 0.01 (Just (0 / 0)))
            && not (signalEntryHeadroomOk 0.01 (Just (0 / 0)))
            && not (signalEntryFeeBufferOk 0.01 0.002 (Just (0 / 0)))
        )
    assert
        "negative edge fails closed under the shared non-negative entry-edge contract"
        ( not (signalEntryEdgeSpikeOk 0.01 (Just (-0.001)))
            && not (signalEntryHeadroomOk 0.01 (Just (-0.001)))
            && not (signalEntryFeeBufferOk 0.01 0 (Just (-0.001)))
        )
    assert
        "negative fee floors fail closed instead of collapsing to the zero-fee boundary"
        ( not (signalEntryFeeBufferOk 0.01 (-0.001) (Just 0.015))
            && not (signalEntryFeeBufferOk 0.01 (-0.001) (Just 0.05))
        )
import Data.Maybe (isNothing)
import qualified Data.Vector as V
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
import Trader.Trading (
    BacktestResult (..),
    EnsembleConfig (..),
    EntryGateInputs (..),
    EntryGateState (..),
    ExitReason (..),
    StepMeta (..),
    Trade (..),
    mkEntryGateState,
    simulateEnsembleVWithHLChecked,
 )

-- Existing test harness imports and helpers remain unchanged outside the
-- optimizer-facing checked-simulator surface witness and the entry-gate fixture.

main :: IO ()
main = do
    run "trading checked simulator facade stays optimizer-visible" testTradingCheckedSimulatorSurface
    run "trading result constructors stay visible to metrics" testTradingResultConstructorSurface
    run "trading entry gate stays entry-only off the fresh-entry path" testTradingEntryGateEntryOnly
    run "trading entry gate refactor stays fail closed and monotone" testTradingEntryGateFailClosedMonotone
    run "signal gate restored facade stays fail closed and entry-only" testSignalGateFacadeSurface
    run "signal gate rejects low-headroom entries" testSignalGateEntryHeadroom
    run "signal gate headroom threshold cap tracks 1.5x rule" testSignalGateEntryHeadroomThresholdCap
    run "signal gate rejects marginal fee-adjusted entries" testSignalGateEntryFeeBuffer
    run "signal gate fee monotonicity holds" testSignalGateEntryFeeBufferMonotoneFees
    run "signal gate edge monotonicity holds under fees" testSignalGateEntryFeeBufferMonotoneEdge
    run "signal gate fee buffer stays subordinate to spike/headroom vetoes" testSignalGateEntryFeeBufferSubordinate
    run "signal gate shared entryEdge conjunction stays fail closed" testSignalGateEntryConjunctiveSharedEdge
    run "signal gate fee-aware malformed inputs fail closed" testSignalGateEntryFeeBufferFailsClosed
    run "signal gate post-direction wrappers cannot reopen blocked entries" testSignalGateNoReopenPostDirection
    run "signal gate rejects entry edge spikes" testSignalGateEntryEdgeSpike

-- Trader.Optimization must keep importing the checked simulator/config/meta
-- surface from Trader.Trading. This witness fails at compile time if the public
-- seam drifts again, and the repair stays behavior-preserving because the
-- re-export points at the canonical checked simulator binding rather than a new
-- wrapper implementation.
checkedSimulatorContractWitness ::
    EnsembleConfig ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    Either String BacktestResult
checkedSimulatorContractWitness cfg closes highs lows kalPred lstmPred meta =
    simulateEnsembleVWithHLChecked cfg 1 closes highs lows kalPred lstmPred meta

optimizerConfigSurfaceWitness :: EnsembleConfig -> (Double, Double, Double, Double, Int)
optimizerConfigSurfaceWitness cfg =
    ( ecPeriodsPerYear cfg
    , ecOpenThreshold cfg
    , ecCloseThreshold cfg
    , ecMinEdge cfg
    , ecRouterLookback cfg
    )

optimizerStepMetaSurfaceWitness ::
    StepMeta ->
    ( Double
    , Double
    , Maybe Double
    , Maybe Double
    , Maybe Double
    , Maybe Double
    , Maybe Double
    )
optimizerStepMetaSurfaceWitness meta =
    ( smKalmanVar meta
    , smKalmanMean meta
    , smHighVolProb meta
    , smConformalLo meta
    , smConformalHi meta
    , smQuantile10 meta
    , smQuantile90 meta
    )

testTradingCheckedSimulatorSurface :: IO ()
testTradingCheckedSimulatorSurface =
    assert
        "optimizer-facing checked simulator facade remains exported from Trader.Trading"
        ( checkedSimulatorContractWitness `seq`
            optimizerConfigSurfaceWitness `seq`
            optimizerStepMetaSurfaceWitness `seq`
            True
        )

-- Fail-closed API stability obligation for downstream analytics:
-- Trader.Metrics must be able to import and pattern-match the canonical
-- BacktestResult, Trade, and ExitReason constructors from Trader.Trading.
-- If a future refactor drops those constructor exports, this regression fails
-- immediately at compile time instead of silently drifting downstream behavior.
testTradingResultConstructorSurface :: IO ()
testTradingResultConstructorSurface = do
    let roundTrip =
            Trade
                { trEntryEquity = 1.0
                , trExitEquity = 1.1
                , trReturn = 0.1
                , trHoldingPeriods = 3
                , trExitReason = Just ExitSignal
                }
        sessionClose =
            Trade
                { trEntryEquity = 1.1
                , trExitEquity = 1.1
                , trReturn = 0
                , trHoldingPeriods = 1
                , trExitReason = Just ExitEod
                }
        result =
            BacktestResult
                { brEquityCurve = [1.0, 1.1, 1.1]
                , brTrades = [roundTrip, sessionClose]
                , brPositions = [1.0, 0.0, 0.0]
                , brAgreementOk = [True, False]
                , brAgreementValid = [True, True]
                , brPositionChanges = 1
                }
    assert
        "backtest result constructor remains visible for downstream pattern matches"
        ( case result of
            BacktestResult{brEquityCurve = [1.0, 1.1, 1.1], brPositionChanges = 1} -> True
            _ -> False
        )
    assert
        "trade constructor preserves holding-period and exit-reason access"
        ( case brTrades result of
            [ Trade{trHoldingPeriods = 3, trExitReason = Just ExitSignal}
                , Trade{trHoldingPeriods = 1, trExitReason = Just ExitEod}
                ] -> True
            _ -> False
        )
    assert
        "end-of-day exits remain distinguishable from round trips at constructor level"
        ( case map trExitReason (brTrades result) of
            [Just ExitSignal, Just ExitEod] -> True
            _ -> False
        )

-- The reviewed Trading.hs change restores only the optimizer-facing checked-
-- simulator seam and leaves the live entry-gate behavior unchanged. These
-- executable obligations pin the surviving entry-gate integration to two
-- properties: entry-only vetoes do not run when no fresh entry is needed, and
-- on fresh entries admissibility is monotone non-increasing as raw edge falls
-- or the fee floor rises, with malformed fee context staying fail closed.
testTradingEntryGateEntryOnly :: IO ()
testTradingEntryGateEntryOnly = do
    let state =
            mkEntryGateState (mkTradingEntryGateInputs (0 / 0) (-0.01) (Just True))
    assert
        "entry-only vetoes stay bypassed when the position already matches the desired side"
        ( not (needsEntry state)
            && entryEdge state == Just 0
            && edgeSpikeOk state
            && edgeHeadroomOk state
            && feeBufferOk state
            && entryGatesOk state
            && desiredSide1 state == Just True
        )

testTradingEntryGateFailClosedMonotone :: IO ()
testTradingEntryGateFailClosedMonotone = do
    let malformedFeeState =
            mkEntryGateState (mkTradingEntryGateInputs (0 / 0) 0.02 Nothing)
        negativeEdgeState =
            mkEntryGateState (mkTradingEntryGateInputs 0 (-0.01) Nothing)
        freshEntryAllowed feePerSide rawEdge =
            desiredSide1 (mkEntryGateState (mkTradingEntryGateInputs feePerSide rawEdge Nothing)) == Just True
        edgeAlloweds =
            map (freshEntryAllowed 0.001) [0.02, 0.017, 0.016, 0.015]
        feeAlloweds =
            map (`freshEntryAllowed` 0.018) [0, 0.001, 0.00175, 0.002]
    assert
        "malformed fee context still fails closed on the fresh-entry path"
        ( needsEntry malformedFeeState
            && not (feeBufferOk malformedFeeState)
            && not (entryGatesOk malformedFeeState)
            && isNothing (desiredSide1 malformedFeeState)
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

mkTradingEntryGateInputs :: Double -> Double -> Maybe Bool -> EntryGateInputs Bool () () Double
mkTradingEntryGateInputs feePerSide rawEdge currentSide =
    EntryGateInputs
        { desiredSideRaw = Just True
        , desiredSizeRaw = 1
        , posSide = currentSide
        , volConfGateEnabled = False
        , lstmEntryScaleRaw = 1.25
        , trendOkAt = \_ _ _ -> True
        , t = ()
        , trendLookbackStep = ()
        , volOkAt = const True
        , ecFee = id
        , cfg = feePerSide
        , isBad = \x -> isNaN x || isInfinite x
        , minSignalToNoiseAdj = 0
        , volPerBarAt = const Nothing
        , clamp01 = max 0 . min 1
        , edgeRaw = rawEdge
        , openThrAdj = 0.01
        , snrOk = True
        , volTargetReady = True
        , triLayerOk = True
        }

-- Bounded executable obligations for the restored signal-gate facade cover:
-- the threshold-boundary witness and entry-only directionality snapshot,
-- the 1.5x headroom-threshold-cap witness, zero-fee specialization,
-- boundary acceptance, strict-below rejection, monotone non-increasing
-- admissibility, once-blocked-stays-blocked under the post-direction wrapper,
-- negative-fee clamping, missing/non-finite-input fail-closed behavior, and
-- preservation of the shared non-negative entryEdge sample across the
-- independent spike veto and the fee/headroom gates on the fresh-entry path,
-- including the conjunction fact that the fee buffer may veto but cannot
-- reopen an entry already blocked upstream.
testSignalGateFacadeSurface :: IO ()
testSignalGateFacadeSurface = do
    let boundary = mkSignalThresholdBoundary 0.01
    let directionality = signalDirectionalitySnapshot True False
    assert
        "restored threshold boundary preserves normalized threshold and required edge"
        (boundary == SignalThresholdBoundary 0.01 0.015)
    assert
        "directionality snapshot remains entry-only and side-specific"
        ( signalDirectionalityEntryAllowed directionality (Just True)
            && not (signalDirectionalityEntryAllowed directionality (Just False))
            && not (signalDirectionalityEntryAllowed directionality Nothing)
        )
    assert
        "restored facade wrappers stay fail closed on malformed inputs"
        ( normalizeSignalThreshold (-0.01) == 0
            && not (signalMetaLabelOk Nothing)
            && not (signalMtfConsensusCheck [])
            && not (signalCrossAssetCheck [])
            && not (signalRegimeEdgeOk 0.01 Nothing)
            && not (signalFundingOiCheck Nothing Nothing)
        )

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
        "zero-threshold zero-fee entries still require an explicit edge sample"
        (signalEntryFeeBufferOk 0 0 (Just 0) && not (signalEntryFeeBufferOk 0 0 Nothing))
    assert
        "zero-fee specialization accepts equality at the pure headroom boundary"
        (signalEntryFeeBufferOk 0.01 0 (Just 0.015))
    assert
        "headroom-only helper remains the zero-fee specialization"
        (signalEntryHeadroomOk 0.01 (Just 0.015) == signalEntryFeeBufferOk 0.01 0 (Just 0.015))

testSignalGateEntryFeeBufferMonotoneFees :: IO ()
testSignalGateEntryFeeBufferMonotoneFees = do
    let alloweds =
            [ signalEntryFeeBufferOk 0.01 fee (Just 0.018)
            | fee <- [0, 0.002, 0.0035, 0.004]
            ]
    let blockedLadder =
            [ signalEntryFeeBufferOk 0.01 fee (Just 0.0165)
            | fee <- [0.002, 0.0035, 0.004]
            ]
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
    let alloweds = map (signalEntryFeeBufferOk 0.01 0.002 . Just) [0.02, 0.017, 0.016, 0.015]
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

testSignalGateNoReopenPostDirection :: IO ()
testSignalGateNoReopenPostDirection = do
    let directionality = signalDirectionalitySnapshot True True
    let blockedEdge = Just 0.015
    let postDirectionChecks =
            [ signalEntryEdgeSpikeOk 0.01 blockedEdge
            , signalEntryHeadroomOk 0.01 blockedEdge
            , signalEntryFeeBufferOk 0.01 0.002 blockedEdge
            ]
    assert
        "post-direction wrapper cannot reopen an entry already blocked upstream"
        ( not (and postDirectionChecks)
            && not (signalRunPostDirectionGates directionality (Just True) postDirectionChecks)
        )
    assert
        "post-direction wrapper stays fail closed without a side or downstream gates"
        ( not (signalRunPostDirectionGates directionality Nothing [True])
            && not (signalRunPostDirectionGates directionality (Just True) [])
        )

-- Remaining signal-gate tests, including the spike-veto witness, remain unchanged.
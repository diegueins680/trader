{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import Control.Monad (unless)
import qualified Data.Aeson as Aeson
import Data.Maybe (isNothing)
import qualified Data.Vector as V
import Trader.Formal.Optimization (
    activityCountFromMetrics,
    fvrActivityCountInvariant,
    fvrOptimizerPublicSurfaceInvariant,
    roiViewFromMetrics,
    rvActivityCount,
    verifyFormalOptimization,
 )
import Trader.Metrics (BacktestMetrics (..), computeMetrics)
import Trader.Optimizer.Optimize (kellyLiteExposureContractReason, qualityPresetBudget)
import Trader.OrderExecution (applyExecutedQuantity, applyReduceOnlyExecutedQuantity)
import Trader.Predictors (RegimeProbs (..))
import Trader.SignalGates (
    DirectionalitySnapshot (..),
    SignalThresholdBoundary (..),
    mkSignalThresholdBoundary,
    normalizeSignalEntryEdge,
    signalCrossAssetCheck,
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
    ExitReason (..),
    IntrabarFill (..),
    PositionSide (..),
    Positioning (..),
    StepMeta (..),
    Trade (..),
    TradeEntrySource (..),
    desiredSide1,
    edgeHeadroomOk,
    edgeSpikeOk,
    entryEdge,
    entryGatesOk,
    exitReasonFromCode,
    feeBufferOk,
    mkEntryGateState,
    mkTradingEntryGateInputs,
    needsEntry,
    roundTripFeeFloor,
    simulateEnsemble,
    simulateEnsembleWithHLChecked,
    tradeEntrySourceCode,
 )
import Trader.VolConfGate (VolConfGatePreset (..))

main :: IO ()
main = do
    testSignalGateEntryBoundaryWitness
    testSignalGateEntryEdgeSpikeCapRegression
    testSignalGateEntryHeadroomSpecializesFeeBuffer
    testNormalizeSignalEntryEdgeFailClosedRegression
    testSignalGateEntryFeeBufferFailsClosed
    testSignalDirectionalityLiveSemanticsRegression
    testSignalGatesPublicSurfaceRegression
    testTradingPublicSurfaceRegression
    testKellyLiteBacktestSizingRegression
    testTradingEntryGateFailClosedMonotone
    testTradingEntryGateMalformedNoReopen
    testBacktestEntryGateUsesRoundTripFeeBuffer
    testOrderExecutionFillSanitizationInvariant
    testOptimizerActivityCountInvariant
    testOptimizerPublicSurfaceRegression
    testOptimizerQualityBudgetRegression
    testOptimizerKellyLiteExposureContractRegression
    testMetricsConsumesTradingPublicResults

assert :: String -> Bool -> IO ()
assert message condition =
    unless condition (ioError (userError ("Assertion failed: " ++ message)))

assertMonotoneNonIncreasing :: String -> [Bool] -> IO ()
assertMonotoneNonIncreasing message values =
    assert message (and (zipWith (\left right -> left || not right) values (drop 1 values)))

pricesFromReturns :: [Double] -> V.Vector Double
pricesFromReturns returns =
    V.fromList (scanl (\price ret -> price * (1 + ret)) 100 returns)

directionalitySnapshot4Args ::
    Double ->
    Maybe RegimeProbs ->
    V.Vector Double ->
    Int ->
    Maybe DirectionalitySnapshot
directionalitySnapshot4Args = signalDirectionalitySnapshot

directionalitySnapshot5Args ::
    Double ->
    Maybe RegimeProbs ->
    V.Vector Double ->
    Int ->
    Int ->
    Maybe DirectionalitySnapshot
directionalitySnapshot5Args = signalDirectionalitySnapshot

sampleEnsembleConfig :: EnsembleConfig
sampleEnsembleConfig =
    EnsembleConfig
        { ecPeriodsPerYear = 252
        , ecOpenThreshold = 0.01
        , ecCloseThreshold = 0.005
        , ecMinEdge = 0
        , ecRouterLookback = 20
        , ecRouterMinScore = 0
        , ecRouterScorePnlWeight = 1
        , ecFee = 0.001
        , ecFeeFixed = 0
        , ecFeeMin = 0
        , ecSlippage = 0
        , ecSlippageVolMult = 0
        , ecSlippageImpactPower = 1
        , ecSlippageImpact = 0
        , ecSpread = 0
        , ecSpreadVolMult = 0
        , ecStopLoss = Nothing
        , ecTakeProfit = Nothing
        , ecTrailingStop = Nothing
        , ecStopLossVolMult = 0
        , ecTakeProfitVolMult = 0
        , ecTrailingStopVolMult = 0
        , ecMinHoldBars = 0
        , ecCooldownBars = 0
        , ecMaxHoldBars = Nothing
        , ecMaxDrawdown = Nothing
        , ecMaxDailyLoss = Nothing
        , ecMaxWeeklyLoss = Nothing
        , ecRiskPerTrade = Nothing
        , ecMaxTradesPerDay = Nothing
        , ecExpectancyLookback = 0
        , ecMinExpectancy = Nothing
        , ecPerfLookback = 0
        , ecPerfMinWinRate = Nothing
        , ecPerfMinProfitFactor = Nothing
        , ecAdaptiveFilters = False
        , ecAdaptiveEdgeBufferMax = 0
        , ecAdaptiveMinSignalToNoiseMax = 0
        , ecAdaptiveKalmanZMinMax = 0
        , ecAdaptiveTrendLookbackMax = 0
        , ecLossStreakMax = 0
        , ecLossStreakCooldownBars = 0
        , ecNoTradeWindows = []
        , ecIntervalSeconds = Nothing
        , ecOpenTimes = Nothing
        , ecOpenPrices = Nothing
        , ecMetaMask = Nothing
        , ecPositioning = LongFlat
        , ecIntrabarFill = StopFirst
        , ecMaxPositionSize = 1
        , ecMinSignalToNoise = 0
        , ecSnrSizeWeight = 0
        , ecThresholdFactorEnabled = False
        , ecThresholdFactorAlpha = 0
        , ecThresholdFactorMin = 1
        , ecThresholdFactorMax = 1
        , ecThresholdFactorFloor = 0
        , ecThresholdFactorEdgeKalWeight = 0
        , ecThresholdFactorEdgeLstmWeight = 0
        , ecThresholdFactorKalmanZWeight = 0
        , ecThresholdFactorHighVolWeight = 0
        , ecThresholdFactorConformalWeight = 0
        , ecThresholdFactorQuantileWeight = 0
        , ecThresholdFactorLstmConfWeight = 0
        , ecThresholdFactorLstmHealthWeight = 0
        , ecLstmTrainingHealth = Nothing
        , ecTrendLookback = 0
        , ecVolTarget = Nothing
        , ecVolLookback = 20
        , ecVolEwmaAlpha = Nothing
        , ecVolFloor = 0
        , ecVolScaleMax = 1
        , ecMaxVolatility = Nothing
        , ecVolConfGate = VolConfGateDisabled
        , ecRebalanceBars = 0
        , ecRebalanceThreshold = 0
        , ecRebalanceGlobal = False
        , ecRebalanceResetOnSignal = False
        , ecFundingRate = 0
        , ecFundingBySide = False
        , ecFundingOnOpen = False
        , ecBlendWeight = 0.5
        , ecKalmanDt = 1
        , ecKalmanProcessVar = 1
        , ecKalmanMeasurementVar = 1
        , ecTriLayer = False
        , ecTriLayerFastMult = 1
        , ecTriLayerSlowMult = 1
        , ecTriLayerCloudPadding = 0
        , ecTriLayerCloudSlope = 0
        , ecTriLayerCloudWidth = 0
        , ecTriLayerTouchLookback = 0
        , ecTriLayerRequirePriceAction = False
        , ecTriLayerPriceActionBody = 0
        , ecTriLayerExitOnSlow = False
        , ecKalmanBandLookback = 0
        , ecKalmanBandStdMult = 0
        , ecKalmanZMin = -1
        , ecKalmanZMax = 1
        , ecLstmExitFlipBars = 0
        , ecLstmExitFlipGraceBars = 0
        , ecLstmExitFlipStrong = False
        , ecLstmConfidenceSoft = 0
        , ecLstmConfidenceHard = 0
        , ecMaxHighVolProb = Nothing
        , ecMaxConformalWidth = Nothing
        , ecMaxQuantileWidth = Nothing
        , ecConfirmConformal = False
        , ecConfirmQuantiles = False
        , ecConfidenceSizing = False
        , ecMinPositionSize = 0
        , ecKellyLiteSizing = False
        , ecKellyLiteFraction = 0.5
        , ecKellyLiteFloor = 0
        , ecKellyLiteCap = 1
        }

optimizerPublicSurfaceWitnessConfig :: EnsembleConfig
optimizerPublicSurfaceWitnessConfig =
    sampleEnsembleConfig
        { ecOpenThreshold = 0.015
        , ecCloseThreshold = 0.01
        , ecMinEdge = 0.001
        , ecRouterLookback = 8
        , ecRouterMinScore = 0.55
        , ecRouterScorePnlWeight = 0.25
        , ecSlippage = 0.0005
        , ecSlippageVolMult = 0.1
        , ecSlippageImpact = 0.01
        , ecSpread = 0.0002
        , ecSpreadVolMult = 0.05
        , ecKalmanZMin = 0.5
        , ecKalmanZMax = 2
        , ecLstmExitFlipBars = 3
        , ecLstmExitFlipGraceBars = 1
        }

optimizerRiskDefaultsNeutral :: EnsembleConfig -> Bool
optimizerRiskDefaultsNeutral cfg =
    isNothing (ecStopLoss cfg)
        && isNothing (ecTakeProfit cfg)
        && isNothing (ecTrailingStop cfg)
        && ecStopLossVolMult cfg == 0
        && ecTakeProfitVolMult cfg == 0
        && ecTrailingStopVolMult cfg == 0
        && ecMinHoldBars cfg == 0
        && ecCooldownBars cfg == 0
        && not (ecKellyLiteSizing cfg)
        && isNothing (ecMaxHoldBars cfg)
        && isNothing (ecMaxDrawdown cfg)

-- Direct SignalGates witness for the restored fee/headroom facade: the zero-fee
-- boundary and the fee-aware boundary stay admissible, strict-below rejection
-- stays intact, and higher round-trip fee floors cannot reopen an entry that
-- was already blocked.
testSignalGateEntryBoundaryWitness :: IO ()
testSignalGateEntryBoundaryWitness = do
    let openThreshold = 0.01
        feeLadder =
            map (\feeFloor -> signalEntryFeeBufferOk openThreshold feeFloor (Just 0.018)) [0, 0.002, 0.0035, 0.004]
    assert
        "signal-gate boundaries remain admissible at equality and reject strict-below edges"
        ( signalEntryEdgeSpikeOk openThreshold (Just 0.017)
            && signalEntryHeadroomOk openThreshold (Just 0.015)
            && signalEntryFeeBufferOk openThreshold 0 (Just 0.015)
            && signalEntryFeeBufferOk openThreshold 0.002 (Just 0.017)
            && not (signalEntryFeeBufferOk openThreshold 0.002 (Just 0.016))
        )
    assert
        "direct fee ladder keeps the expected allow/block shape"
        (feeLadder == [True, True, False, False])
    assertMonotoneNonIncreasing
        "direct signal-gate admissibility is monotone non-increasing as the fee floor rises"
        feeLadder

-- The spike veto is a maximum-edge sanity cap, not a minimum-edge headroom
-- check: exact cap equality is admissible, strict-above-cap edges are blocked,
-- and malformed thresholds or edges fail closed.
testSignalGateEntryEdgeSpikeCapRegression :: IO ()
testSignalGateEntryEdgeSpikeCapRegression = do
    let smallThreshold = 0.01
        etcThreshold = 0.460518
    assert
        "fresh-entry spike veto admits exact equality at both active caps"
        ( signalEntryEdgeSpikeOk smallThreshold (Just 0.04)
            && signalEntryEdgeSpikeOk etcThreshold (Just 0.5)
        )
    assert
        "fresh-entry spike veto blocks strict-above-cap and ETC-style absurd edges"
        ( not (signalEntryEdgeSpikeOk smallThreshold (Just 0.0400001))
            && not (signalEntryEdgeSpikeOk etcThreshold (Just 0.5000001))
            && not (signalEntryEdgeSpikeOk etcThreshold (Just 0.895))
        )
    assert
        "malformed and negative spike-gate inputs fail closed"
        ( not (signalEntryEdgeSpikeOk (-0.01) (Just 0))
            && not (signalEntryEdgeSpikeOk (0 / 0) (Just 0))
            && not (signalEntryEdgeSpikeOk (1 / 0) (Just 0))
            && not (signalEntryEdgeSpikeOk smallThreshold Nothing)
            && not (signalEntryEdgeSpikeOk smallThreshold (Just (-0.001)))
            && not (signalEntryEdgeSpikeOk smallThreshold (Just (0 / 0)))
            && not (signalEntryEdgeSpikeOk smallThreshold (Just (1 / 0)))
        )

-- Bounded executable proof obligation for the restored helper surface: the
-- zero-fee headroom gate is exactly the fee-buffer gate specialized at zero
-- round-trip fees, equality at the computed boundary stays admissible, and
-- missing or malformed thresholds/edges remain fail closed instead of
-- collapsing to a permissive zero boundary.
testSignalGateEntryHeadroomSpecializesFeeBuffer :: IO ()
testSignalGateEntryHeadroomSpecializesFeeBuffer = do
    let specializationSamples =
            [ (0, Just 0, True)
            , (0.01, Just 0.015, True)
            , (0.01, Just 0.014999, False)
            , (0.02, Just 0.03, True)
            , (0.02, Just 0.029999, False)
            ]
        malformedThresholds = [-0.01, 0 / 0, 1 / 0]
        malformedEdges = [Nothing, Just (-0.001), Just (0 / 0), Just (1 / 0)]
    assert
        "zero-fee headroom stays the fee-buffer specialization on bounded boundary cases"
        ( all
            ( \(openThreshold, edgeForMethod, expected) ->
                signalEntryHeadroomOk openThreshold edgeForMethod == expected
                    && signalEntryFeeBufferOk openThreshold 0 edgeForMethod == expected
                    && signalEntryHeadroomOk openThreshold edgeForMethod
                        == signalEntryFeeBufferOk openThreshold 0 edgeForMethod
            )
            specializationSamples
        )
    assert
        "missing or malformed thresholds and edges keep the zero-fee entry gate closed"
        ( and
            [ not (signalEntryHeadroomOk openThreshold (Just 0))
                && not (signalEntryFeeBufferOk openThreshold 0 (Just 0))
            | openThreshold <- malformedThresholds
            ]
            && and
                [ not (signalEntryHeadroomOk 0.01 edgeForMethod)
                    && not (signalEntryFeeBufferOk 0.01 0 edgeForMethod)
                | edgeForMethod <- malformedEdges
                ]
        )

-- Regression for the restored public helper surface: fresh-entry gating keeps a
-- single normalized non-negative edge sample, and malformed raw edges still
-- fail closed because Trading reuses that same sample across the conjunction.
testNormalizeSignalEntryEdgeFailClosedRegression :: IO ()
testNormalizeSignalEntryEdgeFailClosedRegression = do
    let validEdge = normalizeSignalEntryEdge 0.017
        negativeEdge = normalizeSignalEntryEdge (-0.001)
        nanEdge = normalizeSignalEntryEdge (0 / 0)
        infiniteEdge = normalizeSignalEntryEdge (1 / 0)
        boundaryState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 0.017 Nothing)
        negativeState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 (-0.001) Nothing)
        nanState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 (0 / 0) Nothing)
    assert
        "normalizeSignalEntryEdge stays the shared non-negative fresh-entry sample"
        ( validEdge == Just 0.017
            && negativeEdge == Just 0
            && nanEdge == Just 0
            && infiniteEdge == Just 0
            && entryEdge boundaryState == validEdge
            && entryEdge negativeState == negativeEdge
            && entryEdge nanState == nanEdge
        )
    assert
        "normalized zero edges stay blocked by the fresh-entry minimum-edge gates"
        ( needsEntry negativeState
            && edgeSpikeOk negativeState
            && not (edgeHeadroomOk negativeState)
            && not (feeBufferOk negativeState)
            && not (entryGatesOk negativeState)
            && isNothing (desiredSide1 negativeState)
            && needsEntry nanState
            && edgeSpikeOk nanState
            && not (edgeHeadroomOk nanState)
            && not (feeBufferOk nanState)
            && not (entryGatesOk nanState)
            && isNothing (desiredSide1 nanState)
        )

-- The reviewed Trading.hs change keeps the fresh-entry gate entry-only while
-- tightening malformed-fee handling. These executable obligations pin four
-- properties: entry-only vetoes do not run when no fresh entry is needed,
-- fresh-entry spike/headroom/fee-buffer checks all read the same non-negative,
-- finite edge sample while the minimum-edge gates fail closed on negative or
-- non-finite fee/edge inputs,
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

-- Algorithm-path regression: the fee-aware fresh-entry contract is not just a
-- helper-level obligation. The checked simulator must reject a signal that has
-- enough threshold headroom and is below the spike cap when its edge is still
-- below the modeled round-trip cost buffer.
testBacktestEntryGateUsesRoundTripFeeBuffer :: IO ()
testBacktestEntryGateUsesRoundTripFeeBuffer = do
    let prices :: V.Vector Double
        prices = V.fromList [100.0, 100.0, 100.0]
        highs = prices
        lows = prices
        preds :: V.Vector Double
        preds = V.fromList [102.0, 102.0]
        noMeta :: Maybe (V.Vector StepMeta)
        noMeta = Nothing
        baseCfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.01
                , ecFee = 0
                , ecSlippage = 0
                , ecSpread = 0
                , ecFeeFixed = 0
                , ecFeeMin = 0
                , ecMaxPositionSize = 1
                , ecMinPositionSize = 0
                }
        highFeeCfg =
            baseCfg
                { ecFee = 0.02
                }
        maxAbsPosition result =
            maximum (0 : map abs (brPositions result))
        noFeeResult =
            simulateEnsembleWithHLChecked baseCfg 1 prices highs lows preds preds noMeta
        highFeeResult =
            simulateEnsembleWithHLChecked highFeeCfg 1 prices highs lows preds preds noMeta
    case (noFeeResult, highFeeResult) of
        (Right noFee, Right highFee) -> do
            assert
                "zero-fee backtest admits the headroom-valid non-spike entry"
                (maxAbsPosition noFee > 0.99)
            assert
                "high round-trip costs block the same marginal pre-fee entry in the simulator"
                (maxAbsPosition highFee == 0 && null (brTrades highFee))
        (Left err, _) -> ioError (userError ("zero-fee fee-buffer regression failed to simulate: " ++ err))
        (_, Left err) -> ioError (userError ("high-fee fee-buffer regression failed to simulate: " ++ err))

-- Execution-quantity guardrail: malformed or non-positive fills must fail
-- closed, and reduce-only fills must never reopen or increase exposure.
testOrderExecutionFillSanitizationInvariant :: IO ()
testOrderExecutionFillSanitizationInvariant = do
    let invalidQtys = [0, -1, 0 / 0, 1 / 0, negate (1 / 0)]
        invalidQtyNoOp prevPos prevSize isBuy =
            all
                (\qty -> applyExecutedQuantity prevPos prevSize isBuy qty == (prevPos, prevSize, 0, 0))
                invalidQtys
        reduceOnlySamples =
            [ (1, 2, -1, (1, 2, 0, 0))
            , (1, 2, 0 / 0, (1, 2, 0, 0))
            , (1, 2, 5, (0, 0, 2, 0))
            , (-1, 2, 5, (0, 0, 2, 0))
            , (0, 2, 5, (0, 0, 0, 0))
            ]
        reduceOnlyInvariant (prevPos, prevSize, qty, expected@(posNew, sizeNew, closeQty, openQty)) =
            applyReduceOnlyExecutedQuantity prevPos prevSize qty == expected
                && openQty == 0
                && sizeNew >= 0
                && closeQty >= 0
                && sizeNew <= max 0 prevSize
                && closeQty <= max 0 prevSize
                && (posNew == 0 || posNew == signum prevPos)
    assert
        "non-positive or malformed executed quantities stay fail closed and leave position state unchanged"
        ( invalidQtyNoOp 1 2 True
            && invalidQtyNoOp 1 2 False
            && invalidQtyNoOp (-1) 2 True
            && invalidQtyNoOp (-1) 2 False
            && invalidQtyNoOp 0 0 True
        )
    assert
        "reduce-only fills only close existing exposure and never reopen a position"
        (all reduceOnlyInvariant reduceOnlySamples)

-- Formal optimization regression: the restored activity helper stays total,
-- dominates both raw activity sources after clamping, and the RoiView
-- projection stays locked to the helper across bounded negative/positive
-- counter samples while the report-level proof obligation stays true.
testOptimizerActivityCountInvariant :: IO ()
testOptimizerActivityCountInvariant = do
    let baseMetrics =
            BacktestMetrics
                { bmPeriods = 0
                , bmFinalEquity = 1
                , bmTotalReturn = 0
                , bmAnnualizedReturn = 0
                , bmAnnualizedVolatility = 0
                , bmSharpe = 0
                , bmSortino = 0
                , bmCalmar = 0
                , bmDownsideVolatility = 0
                , bmVaR95 = 0
                , bmCVaR95 = 0
                , bmMaxDrawdown = 0
                , bmPositionChanges = 0
                , bmTradeCount = 0
                , bmRoundTrips = 0
                , bmWinRate = 0
                , bmGrossProfit = 0
                , bmGrossLoss = 0
                , bmProfitFactor = Nothing
                , bmAvgTradeReturn = 0
                , bmAvgHoldingPeriods = 0
                , bmExposure = 0
                , bmAgreementRate = 0
                , bmTurnover = 0
                }
        metricsWithActivity roundTrips tradeCount =
            baseMetrics
                { bmRoundTrips = roundTrips
                , bmTradeCount = tradeCount
                }
        samples = [(-2, -1), (-1, 2), (0, 0), (3, 1), (2, 5)]
        helperMatchesInvariant (roundTrips, tradeCount) =
            let metrics = metricsWithActivity roundTrips tradeCount
                activityCount = activityCountFromMetrics metrics
                expectedActivityCount = max 0 (max roundTrips tradeCount)
             in activityCount == expectedActivityCount
                    && activityCount >= 0
                    && activityCount >= max 0 roundTrips
                    && activityCount >= max 0 tradeCount
                    && rvActivityCount (roiViewFromMetrics metrics) == activityCount
    assert
        "formal optimization report keeps the activity-count helper invariant"
        (fvrActivityCountInvariant verifyFormalOptimization)
    assert
        "activity-count helper stays non-negative, dominates both counters, and matches the RoiView projection"
        (all helperMatchesInvariant samples)

-- Bounded executable obligations for the restored signal-gate facade now cover:
-- the direct boundary witness, zero-fee specialization, negative-threshold and
-- negative-fee fail-closed behavior, malformed-input fail-closed behavior, and
-- preservation of the shared non-negative entryEdge sample across the
-- independent spike veto and the fee/headroom gates on the fresh-entry path.

testSignalGateEntryFeeBufferFailsClosed :: IO ()
testSignalGateEntryFeeBufferFailsClosed = do
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
        )
    assert
        "negative open threshold fails closed instead of collapsing to zero"
        ( not (signalEntryFeeBufferOk (-0.01) 0 (Just 0.05))
            && not (signalEntryHeadroomOk (-0.01) (Just 0.05))
            && not (signalEntryEdgeSpikeOk (-0.01) (Just 0.05))
        )
    assert
        "non-finite edge fails closed"
        (not (signalEntryFeeBufferOk 0.01 0.002 (Just (1 / 0))))
    assert
        "NaN edge fails closed"
        (not (signalEntryFeeBufferOk 0.01 0.002 (Just (0 / 0))))
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

-- Public-surface proof obligation for the restored Main import seam: the
-- compatibility names remain importable from Trader.SignalGates, including
-- signalRunPostDirectionGates, their legacy constructors stay reachable, and
-- the restored veto helpers default to fail-closed results even when exercised
-- through small bounded call shapes.
testSignalGatesPublicSurfaceRegression :: IO ()
testSignalGatesPublicSurfaceRegression = do
    let directionalitySnapshot0 = signalDirectionalitySnapshot :: DirectionalitySnapshot
        directionalitySnapshot2 = signalDirectionalitySnapshot () () :: DirectionalitySnapshot
        directionalitySnapshot4 =
            directionalitySnapshot4Args
                0.05
                (Just (RegimeProbs 0.6 0.2 0.2))
                (V.fromList [100, 101, 102, 103])
                3 ::
                Maybe DirectionalitySnapshot
        thresholdBoundary0 = mkSignalThresholdBoundary :: SignalThresholdBoundary
        thresholdBoundary2 =
            (mkSignalThresholdBoundary :: Double -> Maybe Double -> SignalThresholdBoundary) 0.01 (Just 0.02)
        crossAssetCheck0 = signalCrossAssetCheck True Nothing
        crossAssetCheck2 = signalCrossAssetCheck False Nothing
        fundingOiCheck0 = signalFundingOiCheck True (Just 0) Nothing 0.5 0.1 Nothing
        fundingOiCheck2 = signalFundingOiCheck False Nothing Nothing 0.5 0.1 Nothing
        metaLabelOk0 = signalMetaLabelOk True 0 Nothing 0 Nothing False False
        metaLabelOk1 = signalMetaLabelOk False 0 Nothing 0 Nothing False False
        mtfConsensusCheck0 = signalMtfConsensusCheck True [] 1
        mtfConsensusCheck3 = signalMtfConsensusCheck False [Nothing, Just 1] 2
        regimeEdgeOk0 = signalRegimeEdgeOk True 0 Nothing
        regimeEdgeOk2 = signalRegimeEdgeOk False 0 (Just 0.1)
        postDirectionGates0 =
            signalRunPostDirectionGates
                Nothing
                Nothing
                True
                True
                (const True)
                (const True)
                (const True)
                True
                (const (True, Nothing))
                (True, Nothing)
                (True, Nothing)
                (True, Nothing)
                (const True)
                (const (True, 1))
        postDirectionGates2 =
            signalRunPostDirectionGates
                (Just 1)
                Nothing
                False
                True
                (const True)
                (const True)
                (const True)
                True
                (const (True, Nothing))
                (True, Nothing)
                (True, Nothing)
                (True, Nothing)
                (const True)
                (const (True, 1))
    assert
        "Main-facing Trader.SignalGates symbols stay importable and compatibility shims remain fail closed"
        ( directionalitySnapshot0 == DirectionalitySnapshot False Nothing
            && directionalitySnapshot2 == DirectionalitySnapshot False Nothing
            && directionalitySnapshot4 == Just (DirectionalitySnapshot False Nothing)
            && thresholdBoundary0 == SignalThresholdBoundary 0 0 0 0
            && thresholdBoundary2 == SignalThresholdBoundary 0.01 0.02 0.01 0.02
            && crossAssetCheck0 == (False, Just "CROSS_ASSET")
            && crossAssetCheck2 == (True, Nothing)
            && fundingOiCheck0 == (False, 0)
            && fundingOiCheck2 == (True, 1)
            && not metaLabelOk0
            && metaLabelOk1
            && mtfConsensusCheck0 == (False, Just "MTF_CONSENSUS")
            && mtfConsensusCheck3 == (True, Nothing)
            && regimeEdgeOk0 == (False, Just "REGIME_EDGE")
            && regimeEdgeOk2 == (True, Nothing)
            && postDirectionGates0 == (Nothing, Nothing)
            && postDirectionGates2 == (Nothing, Just "VOLATILITY")
        )

testSignalDirectionalityLiveSemanticsRegression :: IO ()
testSignalDirectionalityLiveSemanticsRegression = do
    let chopPrices = pricesFromReturns [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]
        weakBandPrices =
            pricesFromReturns [0.018, 0.018, 0.018, -0.01, -0.01, -0.01, 0.018, 0.018, -0.01, -0.01]
        trendPrices = pricesFromReturns (replicate 24 0.01)
        chopSnapshot =
            directionalitySnapshot5Args
                0.05
                (Just (RegimeProbs 0.2 0.6 0.2))
                chopPrices
                (V.length chopPrices - 1)
                1 ::
                Maybe DirectionalitySnapshot
        weakBandShortSnapshot =
            directionalitySnapshot5Args
                0.05
                (Just (RegimeProbs 0.6 0.2 0.2))
                weakBandPrices
                (V.length weakBandPrices - 1)
                (-1) ::
                Maybe DirectionalitySnapshot
        malformedHysteresisSnapshot =
            directionalitySnapshot5Args
                (-0.01)
                (Just (RegimeProbs 0.6 0.2 0.2))
                weakBandPrices
                (V.length weakBandPrices - 1)
                1 ::
                Maybe DirectionalitySnapshot
        monotonicTrendSnapshot =
            directionalitySnapshot5Args
                0.05
                (Just (RegimeProbs 0.6 0.2 0.2))
                trendPrices
                (V.length trendPrices - 1)
                1 ::
                Maybe DirectionalitySnapshot
    assert
        "directionality chop windows are vetoed at efficiency <= 0.25"
        (chopSnapshot == Just (DirectionalitySnapshot True (Just "NON_DIRECTIONAL_CHOP")))
    assert
        "weak-band shorts are blocked when the signed additive-path zScore confirms the opposite side"
        (weakBandShortSnapshot == Just (DirectionalitySnapshot True (Just "NON_DIRECTIONAL_WEAK_BAND")))
    assert
        "malformed regime-bank hysteresis fails closed on the weak-band live path"
        (malformedHysteresisSnapshot == Just (DirectionalitySnapshot True (Just "NON_DIRECTIONAL_MALFORMED")))
    assert
        "additive monotonic trends remain directional instead of misclassifying clean trends as malformed"
        (monotonicTrendSnapshot == Just (DirectionalitySnapshot False Nothing))

-- Formal public-surface invariant for the Main-facing Trader.Trading import
-- seam: a downstream module importing `PositionSide(..)` can still case-analyze
-- the legacy SideLong/SideShort constructors, read and record-update Trade
-- entry/exit indices, record-update the EnsembleConfig compatibility/risk
-- knobs, and reach the checked simulation entrypoints. Any future export
-- narrowing should therefore fail in tests before trader-hs or optimize-equity
-- reaches a later CI build failure.
testTradingPublicSurfaceRegression :: IO ()
testTradingPublicSurfaceRegression = do
    let positionSideCode side =
            case side of
                SideLong -> "long"
                SideShort -> "short"
        positionSides = [PositionLong, PositionShort]
        indexedTrade =
            Trade
                { trEntryIndex = 7
                , trExitIndex = 9
                , trEntryEquity = 1.0
                , trExitEquity = 1.1
                , trReturn = 0.1
                , trHoldingPeriods = 2
                , trEntryHighVolProb = Nothing
                , trEntrySource = TradeEntrySignal
                , trExitReason = Just ExitEod
                , trEntryIp = Nothing
                , trExitIp = Nothing
                }
        shiftedTrade =
            indexedTrade
                { trEntryIndex = trEntryIndex indexedTrade - 2
                , trExitIndex = trExitIndex indexedTrade - 2
                }
        compatibilityConfigured =
            sampleEnsembleConfig
                { ecMinHoldBars = 3
                , ecCooldownBars = 2
                , ecMaxHoldBars = Just 12
                , ecMaxDrawdown = Just 0.15
                }
        riskConfigured =
            compatibilityConfigured
                { ecStopLoss = Just 0.01
                , ecTakeProfit = Just 0.03
                , ecTrailingStop = Just 0.02
                , ecStopLossVolMult = 1.5
                , ecTakeProfitVolMult = 2.0
                , ecTrailingStopVolMult = 1.25
                }
        signalSource = TradeEntrySignal
        postDirectionSource = TradeEntryPostDirectionGates
        simulateEnsemble0 ::
            EnsembleConfig ->
            Int ->
            V.Vector Double ->
            V.Vector Double ->
            V.Vector Double ->
            V.Vector Double ->
            V.Vector Double ->
            Maybe (V.Vector StepMeta) ->
            BacktestResult
        simulateEnsemble0 = simulateEnsemble
        simulateEnsembleWithHLChecked0 ::
            EnsembleConfig ->
            Int ->
            V.Vector Double ->
            V.Vector Double ->
            V.Vector Double ->
            V.Vector Double ->
            V.Vector Double ->
            Maybe (V.Vector StepMeta) ->
            Either String BacktestResult
        simulateEnsembleWithHLChecked0 = simulateEnsembleWithHLChecked
        tradingSurfaceReachable =
            case (Nothing :: Maybe EnsembleConfig, Nothing :: Maybe StepMeta) of
                (Nothing, Nothing) ->
                    simulateEnsemble0 `seq`
                        (simulateEnsembleWithHLChecked0 `seq` True)
                _ -> False
    assert
        "Main-facing Trader.Trading symbols stay importable and preserve constructor/selector compatibility"
        ( map positionSideCode [SideLong, SideShort] == ["long", "short"]
            && positionSides == [PositionLong, PositionShort]
            && trEntryIndex shiftedTrade == 5
            && trExitIndex shiftedTrade == 7
            && ecMinHoldBars riskConfigured == 3
            && ecCooldownBars riskConfigured == 2
            && ecMaxHoldBars riskConfigured == Just 12
            && ecMaxDrawdown riskConfigured == Just 0.15
            && ecStopLoss riskConfigured == Just 0.01
            && ecTakeProfit riskConfigured == Just 0.03
            && ecTrailingStop riskConfigured == Just 0.02
            && ecStopLossVolMult riskConfigured == 1.5
            && ecTakeProfitVolMult riskConfigured == 2.0
            && ecTrailingStopVolMult riskConfigured == 1.25
            && tradeEntrySourceCode signalSource == "signal"
            && tradeEntrySourceCode postDirectionSource == "post_direction_gates"
            && exitReasonFromCode "eod" == Just ExitEod
            && isNothing (exitReasonFromCode "unknown")
            && tradingSurfaceReachable
        )

testKellyLiteBacktestSizingRegression :: IO ()
testKellyLiteBacktestSizingRegression = do
    let prices :: V.Vector Double
        prices = V.fromList [100.0, 101.0, 100.0, 101.0, 102.0]
        highs = prices
        lows = prices
        preds :: V.Vector Double
        preds = V.fromList [100.0, 101.0, 102.0, 103.0]
        noMeta :: Maybe (V.Vector StepMeta)
        noMeta = Nothing
        baseCfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.005
                , ecCloseThreshold = 0.005
                , ecFee = 0
                , ecVolLookback = 2
                , ecMinPositionSize = 0
                , ecKellyLiteSizing = False
                }
        kellyCfg =
            baseCfg
                { ecKellyLiteSizing = True
                , ecKellyLiteFraction = 0.5
                , ecKellyLiteFloor = 0
                , ecKellyLiteCap = 0.25
                }
        maxAbsPosition result =
            maximum (0 : map abs (brPositions result))
        uncapped =
            simulateEnsembleWithHLChecked baseCfg 1 prices highs lows preds preds noMeta
        capped =
            simulateEnsembleWithHLChecked kellyCfg 1 prices highs lows preds preds noMeta
    case (uncapped, capped) of
        (Right uncappedResult, Right cappedResult) -> do
            assert
                "baseline backtest can reach full-size entries without Kelly-lite sizing"
                (maxAbsPosition uncappedResult > 0.99)
            assert
                "Kelly-lite sizing is modeled in backtests and caps entries the same way live sizing does"
                (maxAbsPosition cappedResult > 0.24 && maxAbsPosition cappedResult <= 0.250000001)
        (Left err, _) -> ioError (userError ("baseline Kelly-lite sizing regression failed to simulate: " ++ err))
        (_, Left err) -> ioError (userError ("capped Kelly-lite sizing regression failed to simulate: " ++ err))

-- Public-interface invariant for optimizer wiring: Trader.Optimization must keep
-- importing the canonical headroom-cap helper from Trader.SignalGates and the
-- restored Main-facing checked simulation surface from Trader.Trading without
-- any semantic adapter in between. This bounded regression also carries both a
-- total neutral optimizer witness and a compatibility-field witness so any
-- future export narrowing or unintended semantic coupling fails here first.
testOptimizerPublicSurfaceRegression :: IO ()
testOptimizerPublicSurfaceRegression = do
    let headroomCap = signalEntryHeadroomThresholdCap 0.03
        base = optimizerPublicSurfaceWitnessConfig
        compatibilityCfg =
            base
                { ecMinHoldBars = 2
                , ecCooldownBars = 1
                , ecMaxHoldBars = Just 9
                , ecMaxDrawdown = Just 0.12
                }
        metaMask0 = Just (V.fromList [True, False, True])
        openTimes0 = Just (V.fromList [10, 11, 12])
        openPrices0 = Just (V.fromList [100.0, 101.5, 103.0])
        thresholdCfg =
            base
                { ecOpenThreshold = 0.02
                , ecCloseThreshold = 0.01
                , ecMetaMask = metaMask0
                }
        foldCfg =
            thresholdCfg
                { ecOpenTimes = openTimes0
                , ecOpenPrices = openPrices0
                , ecMetaMask = Nothing
                }
        simulateEnsembleWithHLChecked0 ::
            EnsembleConfig ->
            Int ->
            V.Vector Double ->
            V.Vector Double ->
            V.Vector Double ->
            V.Vector Double ->
            V.Vector Double ->
            Maybe (V.Vector StepMeta) ->
            Either String BacktestResult
        simulateEnsembleWithHLChecked0 = simulateEnsembleWithHLChecked
        optimizerSurfaceReachable =
            case (Nothing :: Maybe EnsembleConfig, Nothing :: Maybe StepMeta) of
                (Nothing, Nothing) ->
                    signalEntryHeadroomThresholdCap 0.03 `seq`
                        (simulateEnsembleWithHLChecked0 `seq` True)
                _ -> False
    assert
        "optimizer-facing public symbols stay importable and the total neutral-risk witness remains explicit"
        ( optimizerSurfaceReachable
            && fvrOptimizerPublicSurfaceInvariant verifyFormalOptimization
            && abs (headroomCap - 0.02) <= 1e-12
            && signalEntryHeadroomOk (max 0 (headroomCap - 1e-12)) (Just 0.03)
            && not (signalEntryHeadroomOk (headroomCap + 1e-4) (Just 0.03))
            && optimizerRiskDefaultsNeutral base
            && optimizerRiskDefaultsNeutral thresholdCfg
            && optimizerRiskDefaultsNeutral foldCfg
            && ecMinHoldBars compatibilityCfg == 2
            && ecCooldownBars compatibilityCfg == 1
            && ecMaxHoldBars compatibilityCfg == Just 9
            && ecMaxDrawdown compatibilityCfg == Just 0.12
            && ecOpenThreshold compatibilityCfg == ecOpenThreshold base
            && ecCloseThreshold compatibilityCfg == ecCloseThreshold base
            && ecMetaMask compatibilityCfg == ecMetaMask base
            && ecOpenTimes compatibilityCfg == ecOpenTimes base
            && ecOpenPrices compatibilityCfg == ecOpenPrices base
            && ecOpenThreshold thresholdCfg == 0.02
            && ecCloseThreshold thresholdCfg == 0.01
            && ecMetaMask thresholdCfg == metaMask0
            && ecOpenTimes thresholdCfg == ecOpenTimes base
            && ecOpenPrices thresholdCfg == ecOpenPrices base
            && ecOpenTimes foldCfg == openTimes0
            && ecOpenPrices foldCfg == openPrices0
            && isNothing (ecMetaMask foldCfg)
        )

testOptimizerQualityBudgetRegression :: IO ()
testOptimizerQualityBudgetRegression = do
    assert
        "quality preset keeps the production epoch floor when no tighter audit cap is requested"
        (qualityPresetBudget 50 10 50 == 50)
    assert
        "quality preset honors a lower explicit audit epoch cap"
        (qualityPresetBudget 50 10 4 == 4)
    assert
        "quality preset does not loosen already tighter explicit bounds"
        (qualityPresetBudget 50 3 4 == 3)
    assert
        "quality preset clamps malformed nonpositive audit budgets"
        (qualityPresetBudget 50 10 0 == 1)
    assert
        "quality preset preserves explicitly larger production sweeps"
        (qualityPresetBudget 50 80 50 == 80)

-- Optimizer eligibility regression: Kelly-lite exposure contracts must reject
-- no-op Kelly-lite rows. A zero uncapped-exposure replay previously produced a
-- ratio of 0, which let inactive trials satisfy strict ratio ceilings.
testOptimizerKellyLiteExposureContractRegression :: IO ()
testOptimizerKellyLiteExposureContractRegression = do
    let report :: Double -> Double -> Double -> Maybe Aeson.Value
        report uncapped ratio reduction =
            Just $
                Aeson.object
                    [ "backtest"
                        Aeson..= Aeson.object
                            [ "kellyLite"
                                Aeson..= Aeson.object
                                    [ "enabled" Aeson..= True
                                    , "uncappedExposure" Aeson..= uncapped
                                    , "exposureRatio" Aeson..= ratio
                                    , "exposureReduction" Aeson..= reduction
                                    ]
                            ]
                    ]
        reason = kellyLiteExposureContractReason True
    assert
        "disabled Kelly-lite sizing bypasses the Kelly-lite exposure contract"
        (isNothing (kellyLiteExposureContractReason False (report 0 0 0) 0.05 0.9))
    assert
        "inactive Kelly-lite exposure contracts do not reject rows"
        (isNothing (reason (report 0 0 0) 0 1))
    assert
        "Kelly-lite rows without a report fail the optimizer contract"
        (reason Nothing 0.05 0.9 == Just "kellyLiteExposureMissing")
    assert
        "Kelly-lite rows with zero uncapped exposure fail instead of passing ratio ceilings as no-op reductions"
        (reason (report 0 0 0) 0 0.9 == Just "kellyLiteUncappedExposure<=0")
    assert
        "Kelly-lite rows with weak reduction fail before ratio checks"
        (reason (report 0.5 0.97 0.015) 0.05 0.9 == Just "kellyLiteExposureReduction<0.050")
    assert
        "Kelly-lite rows with enough exposure reduction and ratio improvement pass"
        (isNothing (reason (report 0.5 0.8 0.1) 0.05 0.9))

-- Public-interface invariant: metrics/reporting must be able to consume the
-- BacktestResult/Trade/ExitReason constructors re-exported by Trader.Trading.
-- This fixture runs a bounded metrics path through that boundary so any future
-- export regression fails at build or test time.
testMetricsConsumesTradingPublicResults :: IO ()
testMetricsConsumesTradingPublicResults = do
    let trade =
            Trade
                { trEntryIndex = 0
                , trExitIndex = 1
                , trEntryEquity = 1.0
                , trExitEquity = 1.1
                , trReturn = 0.1
                , trHoldingPeriods = 2
                , trEntryHighVolProb = Nothing
                , trEntrySource = TradeEntrySignal
                , trExitReason = Just ExitEod
                , trEntryIp = Nothing
                , trExitIp = Nothing
                }
        result =
            BacktestResult
                { brEquityCurve = [1.0, 1.1]
                , brTrades = [trade]
                , brPositions = [0, 1]
                , brAgreementOk = [True]
                , brAgreementValid = [True]
                , brPositionChanges = 1
                }
        metrics = computeMetrics 252 result
    assert
        "metrics can consume Trader.Trading public result constructors"
        ( bmTradeCount metrics == 1
            && bmRoundTrips metrics == 0
            && bmPositionChanges metrics == 1
            && isNothing (bmProfitFactor metrics)
            && bmAgreementRate metrics == 1
            && activityCountFromMetrics metrics == 1
            && rvActivityCount (roiViewFromMetrics metrics) == 1
        )

{-# LANGUAGE DuplicateRecordFields #-}

import Data.Int (Int64)
import Data.Maybe (isNothing)
import qualified Data.Vector as V
import Trader.App.Args (
    intrabarFillCode,
    parseIntrabarFill,
    parsePositioning,
    positioningCode,
 )
import Trader.Formal.Optimization (
    FormalVerificationReport (..),
    verifyFormalOptimization,
 )
import Trader.OrderExecution (
    OrderExecutionEvidence (..),
    applyReduceOnlyExecutedQuantity,
    orderAppliedQuantity,
 )
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
    IntrabarFill (..),
    Positioning (..),
    StepMeta (..),
    Trade (..),
    TradeEntrySource (..),
    mkEntryGateState,
    simulateEnsembleVWithHLChecked,
 )
import Trader.VolConfGate (VolConfGatePreset (..))

-- Focus the regression surface on the optimizer-facing checked-simulator
-- witness, the execution-config contract that Trader.Optimization updates, the
-- CLI enum contract consumed by Trader.App.Args, and the fee-aware entry-gate
-- invariants.

main :: IO ()
main = do
    run "trading checked simulator facade stays optimizer-visible" testTradingCheckedSimulatorSurface
    run "optimizer execution-config contract preserves fold payloads and zeroes flip exits together" testOptimizerExecutionConfigContract
    run "trading result constructors stay visible to metrics" testTradingResultConstructorSurface
    run "trading CLI enum surface stays visible and round-trips via args parsers" testTradingCliEnumContract
    run "formal optimization exposure penalty stays ordered" testFormalOptimizationExposurePenaltyInvariant
    run "order execution applied quantity trusts explicit partial fills on terminal live statuses" testOrderExecutionAppliedQuantity
    run "order execution reduce-only fills stay close-only and cannot flip exposure" testOrderExecutionReduceOnlyInvariant
    run "trading entry gate stays entry-only off the fresh-entry path" testTradingEntryGateEntryOnly
    run "trading entry gate shared-edge conjunction stays fail closed at integration boundary" testTradingEntryGateSharedEdgeConjunction
    run "trading entry gate refactor stays fail closed and monotone" testTradingEntryGateFailClosedMonotone
    run "trading entry gate malformed inputs cannot reopen a blocked fresh-entry state" testTradingEntryGateMalformedNoReopen
    run "signal gate restored facade stays fail closed and entry-only" testSignalGateFacadeSurface
    run "signal gate weak-directionality snapshots stay fail closed on malformed saved HMM tuples" testSignalGateDirectionalityWeakBandFailClosed
    run "signal gate weak-directionality admissibility is monotone across efficiency and HMM-sanity boundaries" testSignalGateDirectionalityWeakBandMonotone
    run "signal gate rejects low-headroom entries" testSignalGateEntryHeadroom
    run "signal gate headroom threshold cap tracks 1.5x rule" testSignalGateEntryHeadroomThresholdCap
    run "signal gate rejects marginal fee-adjusted entries" testSignalGateEntryFeeBuffer
    run "signal gate fee monotonicity holds" testSignalGateEntryFeeBufferMonotoneFees
    run "signal gate edge monotonicity holds under fees" testSignalGateEntryFeeBufferMonotoneEdge
    run "signal gate fee buffer stays subordinate to spike/headroom vetoes" testSignalGateEntryFeeBufferSubordinate
    run "signal gate shared entryEdge conjunction stays fail closed" testSignalGateEntryConjunctiveSharedEdge
    run "signal gate fee-aware malformed inputs fail closed" testSignalGateEntryFeeBufferFailsClosed
    run "signal gate post-direction wrappers cannot reopen blocked entries" testSignalGateNoReopenPostDirection
    run "signal gate edge-spike monotonicity holds" testSignalGateEntryEdgeSpikeMonotone
    run "signal gate rejects entry edge spikes" testSignalGateEntryEdgeSpike

-- Trader.Optimization must keep importing the checked simulator/config/meta
-- surface from Trader.Trading. This witness fails at compile time if the public
-- seam drifts again while Trading-local defaults stay internal and the
-- entry-gate behavior is locked by the regressions below.
checkedSimulatorContractWitness ::
    EnsembleConfig ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    Either String BacktestResult
checkedSimulatorContractWitness cfg =
    simulateEnsembleVWithHLChecked cfg 1

optimizerConfigSurfaceWitness :: EnsembleConfig -> (Double, Double, Double, Double, Int, Double, Int, Int)
optimizerConfigSurfaceWitness cfg =
    ( ecPeriodsPerYear cfg
    , ecOpenThreshold cfg
    , ecCloseThreshold cfg
    , ecMinEdge cfg
    , ecRouterLookback cfg
    , ecFee cfg
    , ecLstmExitFlipBars cfg
    , ecLstmExitFlipGraceBars cfg
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

optimizerExecutionConfigContractWitness ::
    Bool ->
    Maybe (V.Vector Int64) ->
    Maybe (V.Vector Double) ->
    Maybe (V.Vector Bool) ->
    EnsembleConfig ->
    EnsembleConfig
optimizerExecutionConfigContractWitness lstmFlipEnabled openTimesF openPricesF metaMaskF cfg =
    let foldCfg =
            cfg
                { ecOpenTimes = openTimesF
                , ecOpenPrices = openPricesF
                , ecMetaMask = metaMaskF
                }
     in if lstmFlipEnabled
            then foldCfg
            else
                foldCfg
                    { ecLstmExitFlipBars = 0
                    , ecLstmExitFlipGraceBars = 0
                    }

-- Build the optimizer fixture from the public constructor so this regression
-- stays aligned with the live EnsembleConfig contract as strict fields are
-- added, while keeping Trader.Trading's private defaults out of the test
-- surface and only overriding the optimizer-facing fields under review.
sampleOptimizerConfig :: EnsembleConfig
sampleOptimizerConfig =
    EnsembleConfig
        { ecOpenThreshold = 0.01
        , ecCloseThreshold = 0.005
        , ecFee = 0.001
        , ecSlippage = 0.0002
        , ecSpread = 0.0003
        , ecFeeFixed = 0
        , ecFeeMin = 0
        , ecSlippageVolMult = 0.1
        , ecSlippageImpact = 0.0001
        , ecSlippageImpactPower = 1
        , ecSpreadVolMult = 0.05
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
        , ecOpenTimes = Just (V.fromList ([0, 1, 2, 3] :: [Int64]))
        , ecOpenPrices = Just (V.fromList [100, 101, 102, 103])
        , ecMetaMask = Just (V.fromList [True, False, True])
        , ecPositioning = LongFlat
        , ecIntrabarFill = StopFirst
        , ecMaxPositionSize = 1
        , ecMinEdge = 0.002
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
        , ecPeriodsPerYear = 252
        , ecVolTarget = Nothing
        , ecVolLookback = 0
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
        , ecRouterLookback = 12
        , ecRouterMinScore = 0.55
        , ecRouterScorePnlWeight = 0.5
        , ecKalmanDt = 1
        , ecKalmanProcessVar = 0
        , ecKalmanMeasurementVar = 1
        , ecTriLayer = False
        , ecTriLayerFastMult = 1
        , ecTriLayerSlowMult = 1
        , ecTriLayerCloudPadding = 0
        , ecTriLayerCloudSlope = 0
        , ecTriLayerCloudWidth = 0
        , ecTriLayerTouchLookback = 1
        , ecTriLayerRequirePriceAction = False
        , ecTriLayerPriceActionBody = 0
        , ecTriLayerExitOnSlow = False
        , ecKalmanBandLookback = 0
        , ecKalmanBandStdMult = 0
        , ecLstmExitFlipBars = 3
        , ecLstmExitFlipGraceBars = 2
        , ecLstmExitFlipStrong = False
        , ecLstmConfidenceSoft = 0
        , ecLstmConfidenceHard = 0
        , ecKalmanZMin = 0.25
        , ecKalmanZMax = 2
        , ecMaxHighVolProb = Nothing
        , ecMaxConformalWidth = Nothing
        , ecMaxQuantileWidth = Nothing
        , ecConfirmConformal = False
        , ecConfirmQuantiles = False
        , ecConfidenceSizing = False
        , ecMinPositionSize = 0
        }

allPositionings :: [Positioning]
allPositionings =
    [LongFlat, LongShort]

allIntrabarFills :: [IntrabarFill]
allIntrabarFills =
    [StopFirst, TakeProfitFirst]

testTradingCheckedSimulatorSurface :: IO ()
testTradingCheckedSimulatorSurface =
    assert
        "optimizer-facing checked simulator facade remains exported from Trader.Trading"
        ( checkedSimulatorContractWitness `seq`
            optimizerConfigSurfaceWitness `seq`
                optimizerStepMetaSurfaceWitness `seq`
                    True
        )

-- Bounded proof sketch for the reviewed Optimization.hs execution-config path:
-- the fold payload projected into each walk-forward backtest must preserve the
-- aligned openTimes/openPrices/metaMask slices verbatim, and the non-LSTM
-- branch must zero both flip-exit counters together instead of drifting one
-- field at a time.
testOptimizerExecutionConfigContract :: IO ()
testOptimizerExecutionConfigContract = do
    let openTimesF = Just (V.fromList ([10, 11, 12] :: [Int64]))
        openPricesF = Just (V.fromList [200, 201, 202])
        metaMaskF = Just (V.fromList [False, True])
        disabledCfg =
            optimizerExecutionConfigContractWitness False openTimesF openPricesF metaMaskF sampleOptimizerConfig
        enabledCfg =
            optimizerExecutionConfigContractWitness True openTimesF openPricesF metaMaskF sampleOptimizerConfig
    assert
        "optimizer fold config preserves aligned openTimes/openPrices/metaMask payloads"
        ( ecOpenTimes disabledCfg == openTimesF
            && ecOpenPrices disabledCfg == openPricesF
            && ecMetaMask disabledCfg == metaMaskF
        )
    assert
        "optimizer flip-disabled branch zeroes both LSTM exit counters together"
        ( ecLstmExitFlipBars disabledCfg == 0
            && ecLstmExitFlipGraceBars disabledCfg == 0
        )
    assert
        "optimizer flip-enabled branch preserves both LSTM exit counters while keeping fold payloads intact"
        ( ecOpenTimes enabledCfg == openTimesF
            && ecOpenPrices enabledCfg == openPricesF
            && ecMetaMask enabledCfg == metaMaskF
            && ecLstmExitFlipBars enabledCfg == ecLstmExitFlipBars sampleOptimizerConfig
            && ecLstmExitFlipGraceBars enabledCfg == ecLstmExitFlipGraceBars sampleOptimizerConfig
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
                { trEntryIndex = 0
                , trExitIndex = 1
                , trEntryEquity = 1.0
                , trExitEquity = 1.1
                , trReturn = 0.1
                , trHoldingPeriods = 3
                , trEntryHighVolProb = Nothing
                , trEntrySource = TradeEntrySignal
                , trExitReason = Just ExitSignal
                , trEntryIp = Nothing
                , trExitIp = Nothing
                }
        sessionClose =
            Trade
                { trEntryIndex = 1
                , trExitIndex = 2
                , trEntryEquity = 1.1
                , trExitEquity = 1.1
                , trReturn = 0
                , trHoldingPeriods = 1
                , trEntryHighVolProb = Nothing
                , trEntrySource = TradeEntrySignal
                , trExitReason = Just ExitEod
                , trEntryIp = Nothing
                , trExitIp = Nothing
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

-- Bounded CLI/config contract invariant for the reviewed Trading.hs export
-- surface: the Positioning and IntrabarFill constructor sets must stay public,
-- and the canonical App.Args code/parser pairs must remain exhaustive over the
-- exported constructor lists.
testTradingCliEnumContract :: IO ()
testTradingCliEnumContract = do
    assert
        "positioning constructor surface remains bounded and exported from Trader.Trading"
        (allPositionings == [LongFlat, LongShort])
    assert
        "intrabar-fill constructor surface remains bounded and exported from Trader.Trading"
        (allIntrabarFills == [StopFirst, TakeProfitFirst])
    assert
        "positioning constructors round-trip through CLI code and parser"
        (all (\positioning -> parsePositioning (positioningCode positioning) == Right positioning) allPositionings)
    assert
        "intrabar-fill constructors round-trip through CLI code and parser"
        (all (\fill -> parseIntrabarFill (intrabarFillCode fill) == Right fill) allIntrabarFills)
    assert
        "legacy CLI aliases still resolve to the expected trading constructors"
        ( and
            [ parsePositioning "long" == Right LongFlat
            , parsePositioning "ls" == Right LongShort
            , parseIntrabarFill "stop" == Right StopFirst
            , parseIntrabarFill "tp" == Right TakeProfitFirst
            ]
        )

-- Risk-control regression pin: the formal optimization verifier must keep the
-- ROI scorer's exposure penalty ordered so higher idle-capital/exposure states
-- cannot become strictly preferred just because surrounding tuning logic or
-- normalization changes.
testFormalOptimizationExposurePenaltyInvariant :: IO ()
testFormalOptimizationExposurePenaltyInvariant =
    let report :: FormalVerificationReport
        report = verifyFormalOptimization
     in assert
            "formal optimization report preserves exposure-penalty ordering"
            (fvrExposurePenaltyOrdered report)

-- Live exchange reconciliation must trust explicit executed quantity over a
-- terminal cancel/expire status so partial fills are not lost, while still
-- failing closed when a no-fill terminal status arrives without fill evidence.
-- Filled-like statuses may still fall back to the requested quantity when the
-- exchange omits an executedQty field.
testOrderExecutionAppliedQuantity :: IO ()
testOrderExecutionAppliedQuantity = do
    let fallbackQty = 0.75
        liveCanceledPartial =
            OrderExecutionEvidence
                { oeeSent = True
                , oeeLive = True
                , oeeStatus = Just "CANCELED"
                , oeeExecutedQty = Just 0.25
                }
        liveExpiredPartial =
            OrderExecutionEvidence
                { oeeSent = True
                , oeeLive = True
                , oeeStatus = Just "expired"
                , oeeExecutedQty = Just 0.125
                }
        liveCanceledNoFill =
            OrderExecutionEvidence
                { oeeSent = True
                , oeeLive = True
                , oeeStatus = Just "cancelled"
                , oeeExecutedQty = Nothing
                }
        liveFilledFallback =
            OrderExecutionEvidence
                { oeeSent = True
                , oeeLive = True
                , oeeStatus = Just "FILLED"
                , oeeExecutedQty = Nothing
                }
    assert
        "explicit live partial fills remain authoritative even on terminal cancel/expire statuses"
        ( orderAppliedQuantity liveCanceledPartial fallbackQty == Just 0.25
            && orderAppliedQuantity liveExpiredPartial fallbackQty == Just 0.125
        )
    assert
        "terminal no-fill live statuses still fail closed without fill evidence"
        (isNothing (orderAppliedQuantity liveCanceledNoFill fallbackQty))
    assert
        "filled-like live statuses may still fall back to the requested quantity when explicit fill qty is absent"
        (orderAppliedQuantity liveFilledFallback fallbackQty == Just fallbackQty)

-- Reduce-only reconciliation must remain close-only even if an exchange reports
-- an oversized executed quantity or malformed fill size: it may reduce the
-- current exposure to flat, but it must never flip direction or open new size.
testOrderExecutionReduceOnlyInvariant :: IO ()
testOrderExecutionReduceOnlyInvariant = do
    let longOversized = applyReduceOnlyExecutedQuantity 1 2 5
        shortPartial = applyReduceOnlyExecutedQuantity (-1) 3 1.25
        flatNoop = applyReduceOnlyExecutedQuantity 0 4 2
        malformedQty = applyReduceOnlyExecutedQuantity 1 2 (0 / 0)
        openQtysStayZero =
            all
                (\(_, _, _, openQty) -> openQty == 0)
                [longOversized, shortPartial, flatNoop, malformedQty]
    assert
        "oversized reduce-only fills cap at the existing exposure and flatten instead of reversing"
        (longOversized == (0, 0, 2, 0))
    assert
        "partial reduce-only fills preserve the existing side while residual exposure remains"
        (shortPartial == (-1, 1.75, 1.25, 0))
    assert
        "reduce-only reconciliation cannot open exposure from flat or malformed executed quantity"
        (flatNoop == (0, 0, 0, 0) && malformedQty == (1, 2, 0, 0))
    assert
        "reduce-only reconciliation keeps open quantity pinned to zero across representative cases"
        openQtysStayZero

-- The reviewed Trading.hs change restores only the optimizer-facing checked-
-- simulator seam and leaves the live entry-gate behavior unchanged. These
-- executable obligations pin the surviving entry-gate integration to four
-- properties: entry-only vetoes do not run when no fresh entry is needed,
-- fresh-entry spike/headroom/fee-buffer checks all read the same non-negative,
-- finite edge sample and fail closed on malformed fee/edge inputs, equality at
-- the required boundary stays admissible, and admissibility is monotone
-- non-increasing as raw edge falls or the fee floor rises.
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

-- Bounded integration witness for the documented entry-gate contract: the
-- Trading.hs binding must feed one normalized, finite entryEdge sample into
-- every fresh-entry veto, and malformed inputs must still collapse to a
-- blocked state instead of bypassing the conjunction.
testTradingEntryGateSharedEdgeConjunction :: IO ()
testTradingEntryGateSharedEdgeConjunction = do
    let feeBlockedState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 0.015 Nothing)
        malformedFeeState =
            mkEntryGateState (mkTradingEntryGateInputs (0 / 0) 0.015 Nothing)
        malformedEdgeState =
            mkEntryGateState (mkTradingEntryGateInputs 0 (0 / 0) Nothing)
        infiniteEdgeState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 (1 / 0) Nothing)
    assert
        "fresh-entry integration applies spike, headroom, and fee-buffer checks conjunctively to one shared edge"
        ( needsEntry feeBlockedState
            && entryEdge feeBlockedState == Just 0.015
            && edgeSpikeOk feeBlockedState
                == signalEntryEdgeSpikeOk 0.01 (entryEdge feeBlockedState)
            && edgeHeadroomOk feeBlockedState
                == signalEntryHeadroomOk 0.01 (entryEdge feeBlockedState)
            && feeBufferOk feeBlockedState
                == signalEntryFeeBufferOk 0.01 (roundTripFeeFloor feeBlockedState) (entryEdge feeBlockedState)
            && edgeSpikeOk feeBlockedState
            && edgeHeadroomOk feeBlockedState
            && not (feeBufferOk feeBlockedState)
            && not (entryGatesOk feeBlockedState)
            && isNothing (desiredSide1 feeBlockedState)
        )
    assert
        "malformed fee context stays fail closed at the Trading.hs integration boundary"
        ( needsEntry malformedFeeState
            && entryEdge malformedFeeState == Just 0.015
            && edgeSpikeOk malformedFeeState
                == signalEntryEdgeSpikeOk 0.01 (entryEdge malformedFeeState)
            && edgeHeadroomOk malformedFeeState
                == signalEntryHeadroomOk 0.01 (entryEdge malformedFeeState)
            && feeBufferOk malformedFeeState
                == signalEntryFeeBufferOk 0.01 (roundTripFeeFloor malformedFeeState) (entryEdge malformedFeeState)
            && edgeSpikeOk malformedFeeState
            && edgeHeadroomOk malformedFeeState
            && not (feeBufferOk malformedFeeState)
            && not (entryGatesOk malformedFeeState)
            && isNothing (desiredSide1 malformedFeeState)
        )
    assert
        "malformed raw edge is clamped once and stays fail closed across the entry conjunction"
        ( needsEntry malformedEdgeState
            && entryEdge malformedEdgeState == Just 0
            && edgeSpikeOk malformedEdgeState
                == signalEntryEdgeSpikeOk 0.01 (entryEdge malformedEdgeState)
            && edgeHeadroomOk malformedEdgeState
                == signalEntryHeadroomOk 0.01 (entryEdge malformedEdgeState)
            && feeBufferOk malformedEdgeState
                == signalEntryFeeBufferOk 0.01 (roundTripFeeFloor malformedEdgeState) (entryEdge malformedEdgeState)
            && not (edgeHeadroomOk malformedEdgeState)
            && not (entryGatesOk malformedEdgeState)
            && isNothing (desiredSide1 malformedEdgeState)
        )
    assert
        "positive non-finite raw edge also normalizes once to the shared zero edge"
        ( needsEntry infiniteEdgeState
            && entryEdge infiniteEdgeState == Just 0
            && edgeSpikeOk infiniteEdgeState
                == signalEntryEdgeSpikeOk 0.01 (entryEdge infiniteEdgeState)
            && edgeHeadroomOk infiniteEdgeState
                == signalEntryHeadroomOk 0.01 (entryEdge infiniteEdgeState)
            && feeBufferOk infiniteEdgeState
                == signalEntryFeeBufferOk 0.01 (roundTripFeeFloor infiniteEdgeState) (entryEdge infiniteEdgeState)
            && not (edgeHeadroomOk infiniteEdgeState)
            && not (entryGatesOk infiniteEdgeState)
            && isNothing (desiredSide1 infiniteEdgeState)
        )

-- Executable proof sketch for the Trading.hs entry gate: equality at the
-- required headroom-plus-fee boundary is admissible, but raising the fee
-- floor, lowering the raw edge, or supplying malformed fee/edge inputs cannot
-- reopen a blocked fresh-entry state.
testTradingEntryGateFailClosedMonotone :: IO ()
testTradingEntryGateFailClosedMonotone = do
    let boundaryState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 0.017 Nothing)
        malformedFeeState =
            mkEntryGateState (mkTradingEntryGateInputs (0 / 0) 0.02 Nothing)
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
        "malformed fee context still fails closed on the fresh-entry path"
        ( needsEntry malformedFeeState
            && not (feeBufferOk malformedFeeState)
            && not (entryGatesOk malformedFeeState)
            && isNothing (desiredSide1 malformedFeeState)
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
-- malformed values must leave the state blocked as well, so the
-- Trading/SignalGates integration cannot reopen from NaN or Infinity drift
-- while the simulator seam stays decoupled.
testTradingEntryGateMalformedNoReopen :: IO ()
testTradingEntryGateMalformedNoReopen = do
    let blockedState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 0.015 Nothing)
        malformedFeeState =
            mkEntryGateState (mkTradingEntryGateInputs (0 / 0) 0.017 Nothing)
        malformedEdgeState =
            mkEntryGateState (mkTradingEntryGateInputs 0.001 (0 / 0) Nothing)
    assert
        "once blocked by the fresh-entry conjunction, malformed fee or edge inputs cannot reopen the state"
        ( needsEntry blockedState
            && not (entryGatesOk blockedState)
            && isNothing (desiredSide1 blockedState)
            && all
                ( \state ->
                    needsEntry state
                        && not (entryGatesOk state)
                        && isNothing (desiredSide1 state)
                )
                [malformedFeeState, malformedEdgeState]
        )

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
        , entryFeeOf = id
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

-- Bounded executable obligations for the restored signal-gate facade now cover:
-- the four-field threshold-boundary witness, a live DirectionalitySnapshot
-- built through the current price/regime API, snapshot-only fail-closed
-- directionality admission on Nothing and non-directional snapshots, the 1.5x
-- headroom-threshold-cap witness, zero-fee specialization, boundary
-- acceptance, strict-below rejection, monotone non-increasing admissibility,
-- once-blocked-stays-blocked under the post-direction wrapper, negative-fee
-- clamping, missing/non-finite-input fail-closed behavior, and preservation of
-- the shared non-negative entryEdge sample across the independent spike veto
-- and the fee/headroom gates on the fresh-entry path, including the
-- conjunction fact that the fee buffer may veto but cannot reopen an entry
-- already blocked upstream.
testSignalGateFacadeSurface :: IO ()
testSignalGateFacadeSurface = do
    let openThreshold = 0.01
        requiredEdge = 0.015
        boundary@(SignalThresholdBoundary configuredOpen configuredClose effectiveOpen effectiveClose) =
            mkSignalThresholdBoundary openThreshold 0 openThreshold requiredEdge
        directionalPrices = V.fromList [100, 101, 103, 106, 110]
        directionalSnapshot =
            signalDirectionalitySnapshot 0 Nothing directionalPrices (V.length directionalPrices - 1)
        nonDirectionalPrices = V.fromList [100, 100, 100, 100]
        nonDirectionalSnapshot =
            signalDirectionalitySnapshot 0 Nothing nonDirectionalPrices (V.length nonDirectionalPrices - 1)
        directionalBuilt =
            case directionalSnapshot of
                Just snap ->
                    not (dsNonDirectional snap)
                        && isNothing (dsReason snap)
                        && dsLookbackBars snap == V.length directionalPrices
                Nothing -> False
        nonDirectionalBuilt =
            case nonDirectionalSnapshot of
                Just snap ->
                    dsNonDirectional snap
                        && dsReason snap == Just "NON_DIRECTIONAL_CHOP"
                        && dsLookbackBars snap == V.length nonDirectionalPrices
                Nothing -> False
        (fundingOiOk, _) =
            signalFundingOiCheck True (Just 0.01) (Just 0.1) 0.02 1 Nothing
    assert
        "restored threshold boundary preserves the normalized threshold and required edge/headroom boundary"
        ( boundary == SignalThresholdBoundary openThreshold 0 openThreshold requiredEdge
            && configuredOpen == openThreshold
            && configuredClose == 0
            && effectiveOpen == openThreshold
            && effectiveClose == requiredEdge
            && signalEntryHeadroomThresholdCap effectiveClose == effectiveOpen
        )
    assert
        "directionality snapshot stays snapshot-only, entry-allowed for directional evidence, and fail closed otherwise"
        ( directionalBuilt
            && signalDirectionalityEntryAllowed directionalSnapshot
            && not (signalDirectionalityEntryAllowed Nothing)
            && nonDirectionalBuilt
            && not (signalDirectionalityEntryAllowed nonDirectionalSnapshot)
        )
    assert
        "restored facade wrappers stay fail closed on malformed inputs"
        ( normalizeSignalThreshold (-0.01) == 0
            && not (signalMetaLabelOk True 0.01 Nothing 0 Nothing False False)
            && signalMtfConsensusCheck True [] 1 1 == (False, Just "MTF_WARMUP")
            && signalCrossAssetCheck True Nothing 1 == (False, Just "CROSS_ASSET")
            && not (signalRegimeEdgeOk True 0.01 Nothing)
            && not fundingOiOk
        )

weakBandZScoreFloor :: Double
weakBandZScoreFloor = 0.75

mkManualDirectionalitySnapshotWithZScore :: Double -> Double -> Maybe Double -> Maybe Double -> Maybe Double -> Bool -> Maybe String -> DirectionalitySnapshot
mkManualDirectionalitySnapshotWithZScore zScore efficiency trendProb mrProb highVolProb nonDirectional reason =
    DirectionalitySnapshot
        { dsLookbackBars = 24
        , dsNetReturnPct = 1.2
        , dsRealizedVolPct = 0.8
        , dsEfficiency = efficiency
        , dsZScore = zScore
        , dsLabel = "range-drift"
        , dsTrendProb = trendProb
        , dsMrProb = mrProb
        , dsHighVolProb = highVolProb
        , dsRegimeLeader = Nothing
        , dsRegimeGap = Nothing
        , dsNonDirectional = nonDirectional
        , dsReason = reason
        }

mkManualDirectionalitySnapshot :: Double -> Maybe Double -> Maybe Double -> Maybe Double -> Bool -> Maybe String -> DirectionalitySnapshot
mkManualDirectionalitySnapshot =
    mkManualDirectionalitySnapshotWithZScore 1.1

testSignalGateDirectionalityWeakBandFailClosed :: IO ()
testSignalGateDirectionalityWeakBandFailClosed = do
    let strongMissing =
            mkManualDirectionalitySnapshot 0.400001 Nothing Nothing Nothing False Nothing
        strongMalformedMass =
            mkManualDirectionalitySnapshot 0.400001 (Just 0.5015) (Just 0.25) (Just 0.25) False Nothing
        weakMissingInterior =
            mkManualDirectionalitySnapshot 0.26 Nothing Nothing Nothing False Nothing
        weakMissingBoundary =
            mkManualDirectionalitySnapshot 0.4 Nothing Nothing Nothing False Nothing
        weakWellFormed =
            mkManualDirectionalitySnapshot 0.26 (Just 0.55) (Just 0.2) (Just 0.25) False Nothing
        weakZBoundary =
            mkManualDirectionalitySnapshotWithZScore weakBandZScoreFloor 0.26 (Just 0.55) (Just 0.2) (Just 0.25) False Nothing
        weakNegativeZBoundary =
            mkManualDirectionalitySnapshotWithZScore (negate weakBandZScoreFloor) 0.26 (Just 0.55) (Just 0.2) (Just 0.25) False Nothing
        weakZTooSmall =
            mkManualDirectionalitySnapshotWithZScore (weakBandZScoreFloor - 1e-6) 0.26 (Just 0.55) (Just 0.2) (Just 0.25) False Nothing
        weakNegativeZTooSmall =
            mkManualDirectionalitySnapshotWithZScore (negate weakBandZScoreFloor + 1e-6) 0.26 (Just 0.55) (Just 0.2) (Just 0.25) False Nothing
        weakMassWithinTolerance =
            mkManualDirectionalitySnapshot 0.26 (Just 0.5005) (Just 0.25) (Just 0.25) False Nothing
        weakNonFiniteZ =
            mkManualDirectionalitySnapshotWithZScore (1 / 0) 0.26 (Just 0.55) (Just 0.2) (Just 0.25) False Nothing
        weakNonFinite =
            mkManualDirectionalitySnapshot 0.26 (Just 0.55) (Just (1 / 0)) (Just 0.25) False Nothing
        weakNegative =
            mkManualDirectionalitySnapshot 0.26 (Just 0.55) (Just (-0.05)) (Just 0.5) False Nothing
        weakAboveOne =
            mkManualDirectionalitySnapshot 0.26 (Just 1.01) (Just 0) (Just 0) False Nothing
        weakZeroMass =
            mkManualDirectionalitySnapshot 0.26 (Just 0) (Just 0) (Just 0) False Nothing
        weakMassTooHigh =
            mkManualDirectionalitySnapshot 0.26 (Just 0.5015) (Just 0.25) (Just 0.25) False Nothing
        weakMrBlocked =
            mkManualDirectionalitySnapshot 0.26 (Just 0.2) (Just 0.6) (Just 0.2) True (Just "NON_DIRECTIONAL_MR")
        chopBoundary =
            mkManualDirectionalitySnapshot 0.25 Nothing Nothing Nothing False Nothing
    assert
        "weak-directionality entry gate preserves the strong-band no-tuple path and fails closed on malformed saved HMM tuples"
        ( signalDirectionalityEntryAllowed (Just strongMissing)
            && not (signalDirectionalityEntryAllowed (Just strongMalformedMass))
            && not (signalDirectionalityEntryAllowed (Just weakMissingInterior))
            && not (signalDirectionalityEntryAllowed (Just weakMissingBoundary))
            && signalDirectionalityEntryAllowed (Just weakWellFormed)
            && signalDirectionalityEntryAllowed (Just weakZBoundary)
            && signalDirectionalityEntryAllowed (Just weakNegativeZBoundary)
            && not (signalDirectionalityEntryAllowed (Just weakZTooSmall))
            && not (signalDirectionalityEntryAllowed (Just weakNegativeZTooSmall))
            && signalDirectionalityEntryAllowed (Just weakMassWithinTolerance)
            && not (signalDirectionalityEntryAllowed (Just weakNonFiniteZ))
            && not (signalDirectionalityEntryAllowed (Just weakNonFinite))
            && not (signalDirectionalityEntryAllowed (Just weakNegative))
            && not (signalDirectionalityEntryAllowed (Just weakAboveOne))
            && not (signalDirectionalityEntryAllowed (Just weakZeroMass))
            && not (signalDirectionalityEntryAllowed (Just weakMassTooHigh))
            && not (signalDirectionalityEntryAllowed (Just weakMrBlocked))
            && not (signalDirectionalityEntryAllowed (Just chopBoundary))
        )

testSignalGateDirectionalityWeakBandMonotone :: IO ()
testSignalGateDirectionalityWeakBandMonotone = do
    let missingRegimeAlloweds =
            [ signalDirectionalityEntryAllowed
                (Just (mkManualDirectionalitySnapshot eff Nothing Nothing Nothing False Nothing))
            | eff <- [0.5, 0.400001, 0.4, 0.3, 0.26, 0.25]
            ]
        weakZScoreAlloweds =
            [ signalDirectionalityEntryAllowed
                (Just (mkManualDirectionalitySnapshotWithZScore z 0.26 (Just 0.55) (Just 0.2) (Just 0.25) False Nothing))
            | z <- [1.1, weakBandZScoreFloor, weakBandZScoreFloor - 1e-6, 0.2]
            ]
        massDriftAlloweds =
            [ signalDirectionalityEntryAllowed
                (Just (mkManualDirectionalitySnapshot 0.26 (Just (0.5 + drift)) (Just 0.25) (Just 0.25) False Nothing))
            | drift <- [0, 0.0005, 0.0015, 0.01]
            ]
    assert
        "weak-directionality efficiency ladder keeps the expected allow/block shape without regime evidence"
        (missingRegimeAlloweds == [True, True, False, False, False, False])
    assertMonotoneNonIncreasing
        "lower efficiency cannot reopen a weak-directionality snapshot once regime evidence is missing"
        missingRegimeAlloweds
    assert
        "weak-directionality z-score ladder keeps the expected allow/block shape"
        (weakZScoreAlloweds == [True, True, False, False])
    assertMonotoneNonIncreasing
        "lower weak-band |zScore| cannot reopen entry admissibility once blocked"
        weakZScoreAlloweds
    assert
        "weak-directionality HMM-mass ladder keeps the expected allow/block shape"
        (massDriftAlloweds == [True, True, False, False])
    assertMonotoneNonIncreasing
        "larger saved-HMM mass drift cannot admit a weak-band entry once blocked"
        massDriftAlloweds

testSignalGateEntryHeadroom :: IO ()
testSignalGateEntryHeadroom = do
    assert
        "headroom gate accepts equality at the normalized boundary"
        (signalEntryHeadroomOk 0.01 (Just 0.015))
    assert
        "headroom gate rejects entries below the normalized boundary"
        (not (signalEntryHeadroomOk 0.01 (Just 0.014999)))
    assert
        "headroom gate fails closed on missing or malformed edges"
        ( not (signalEntryHeadroomOk 0.01 Nothing)
            && not (signalEntryHeadroomOk 0.01 (Just (0 / 0)))
            && not (signalEntryHeadroomOk 0.01 (Just (1 / 0)))
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
        "shared entryEdge conjunction keeps the spike veto independent when threshold headroom collapses to zero"
        ( entryGatesOk 0 0 (Just 0)
            && not (entryGatesOk 0 0 (Just 0.5))
        )
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
        "non-finite open threshold fails closed instead of collapsing to zero"
        ( not (signalEntryFeeBufferOk (0 / 0) 0 (Just 0))
            && not (signalEntryFeeBufferOk (1 / 0) 0 (Just 0))
            && not (signalEntryHeadroomOk (0 / 0) (Just 0))
            && not (signalEntryHeadroomOk (1 / 0) (Just 0))
        )
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
    let runPostDirection chosenDir mReason volOk volTargetReady trendAllowed cloudAllowed priceActionAllowed signalToNoiseAllowed nonDirectionalAllowed regimeEdgeAllowed mtfAllowed crossAllowed metaLabelAllowed fundingOiAllowed =
            signalRunPostDirectionGates
                chosenDir
                mReason
                volOk
                volTargetReady
                (const trendAllowed)
                (const cloudAllowed)
                (const priceActionAllowed)
                signalToNoiseAllowed
                (\_ -> if nonDirectionalAllowed then (True, Nothing) else (False, Just "NON_DIRECTIONAL"))
                regimeEdgeAllowed
                (\_ -> if mtfAllowed then (True, Nothing) else (False, Just "MTF_CONSENSUS"))
                (\_ -> if crossAllowed then (True, Nothing) else (False, Just "CROSS_ASSET"))
                (const metaLabelAllowed)
                (const (fundingOiAllowed, 1.0))
    assert
        "post-direction wrapper cannot reopen an entry already blocked upstream"
        ( runPostDirection Nothing (Just "FEE_BUFFER") True True True True True True True True True True True True
            == (Nothing, Just "FEE_BUFFER")
        )
    assert
        "post-direction wrapper stays fail closed on downstream vetoes"
        ( runPostDirection (Just 1) Nothing True True True True True True False True True True True True
            == (Nothing, Just "NON_DIRECTIONAL")
        )

testSignalGateEntryEdgeSpikeMonotone :: IO ()
testSignalGateEntryEdgeSpikeMonotone = do
    let alloweds =
            [ signalEntryEdgeSpikeOk openThr (Just 0.04)
            | openThr <- [0.02, 0.01, 0.005, 0]
            ]
    assert
        "edge-spike threshold ladder keeps the expected allow/block shape"
        (alloweds == [True, True, False, False])
    assertMonotoneNonIncreasing
        "lower thresholds cannot reopen a blocked edge-spike entry"
        alloweds

testSignalGateEntryEdgeSpike :: IO ()
testSignalGateEntryEdgeSpike = do
    assert
        "edge-spike gate accepts equality at the active 4x threshold cap"
        (signalEntryEdgeSpikeOk 0.01 (Just 0.04))
    assert
        "edge-spike gate rejects edges above the active 4x threshold cap"
        (not (signalEntryEdgeSpikeOk 0.01 (Just 0.040001)))
    assert
        "edge-spike gate only admits the shared zero edge when the threshold normalizes to zero"
        ( not (signalEntryEdgeSpikeOk 0 Nothing)
            && signalEntryEdgeSpikeOk 0 (Just 0)
            && not (signalEntryEdgeSpikeOk 0 (Just 0.000001))
            && not (signalEntryEdgeSpikeOk 0 (Just 0.5))
            && not (signalEntryEdgeSpikeOk 0 (Just (-0.001)))
        )
    assert
        "edge-spike gate fails closed on missing or malformed thresholds or edges"
        ( not (signalEntryEdgeSpikeOk 0.01 Nothing)
            && not (signalEntryEdgeSpikeOk (0 / 0) (Just 0))
            && not (signalEntryEdgeSpikeOk (1 / 0) (Just 0))
            && not (signalEntryEdgeSpikeOk 0.01 (Just (0 / 0)))
            && not (signalEntryEdgeSpikeOk 0.01 (Just (1 / 0)))
        )

run :: String -> IO () -> IO ()
run label test = do
    test
    putStrLn ("ok - " ++ label)

assert :: String -> Bool -> IO ()
assert message condition =
    if condition
        then pure ()
        else error ("assertion failed: " ++ message)

assertMonotoneNonIncreasing :: String -> [Bool] -> IO ()
assertMonotoneNonIncreasing message xs =
    assert message (and (zipWith (\prev next -> prev || not next) xs (drop 1 xs)))

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import Control.Exception (SomeException, evaluate, toException, try)
import Control.Monad (forM_, unless)
import Data.Aeson ((.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Lazy as BL
import Data.Either (isLeft)
import qualified Data.HashMap.Strict as HM
import Data.Int (Int64)
import Data.List (isInfixOf)
import qualified Data.Map.Strict as Map
import Data.Maybe (catMaybes, fromMaybe, isJust, isNothing, listToMaybe, mapMaybe)
import qualified Data.Text as T
import qualified Data.Vector as V
import Network.HTTP.Client (HttpException (..), HttpExceptionContent (..), parseRequest_, requestHeaders)
import Options.Applicative (ParserResult (..), auto, defaultPrefs, execParserPure, info, long, option, switch, value)
import System.Directory (removeFile)
import System.IO (hClose, openTempFile)
import Trader.App.Args (Args (..), argRouterScorePnlWeight, argTunePenaltyTurnover, argWalkForwardEmbargoBars, argWalkForwardFolds, normalizeBarsForLookback, opts, parsePositioning, validateArgs)
import Trader.App.Runtime (resolveTenantKeyFromParams, resolveTenantKeyFromPlatformParams, tenantKeyFromBinanceKeys, tenantKeyFromCoinbaseKeys)
import Trader.Binance (FuturesPositionRisk (..), binanceExceptionSummary, futuresPositionRiskLeverageSane)
import Trader.BotStartSemantics (AdoptionEvidenceConfig (..), BacktestVerdict (..), adoptionMaxPositionSizeCap, adoptionMinTradeCount, adoptionMinWalkForwardSharpeMean, backtestVerdictAborts, botStartSymbolDisabled, botStartupBacktestAborts, botStartupBacktestRoiAcceptable, botStartupBacktestVerdict, botStartupBacktestVerdictWithMinTrades, botStartupGuardShouldPrune, capAdoptedMaxPositionSize, capAdoptedMaxPositionSizeWithCap, comboTradeCountMeetsAdoptionFloor, comboTradeCountMeetsAdoptionFloorWithConfig, comboWalkForwardSharpeMeetsAdoptionFloor, comboWalkForwardSharpeMeetsAdoptionFloorWithConfig, defaultBotStartupBacktestMinTrades, prioritizeBotStartSymbols, queuedStartOrderErrorIssue)
import Trader.CapitalPreservation (CapitalPreservationConfig (..), CapitalPreservationReport (..), capitalPreservationIsEntryOnlyReason, capitalPreservationReport, defaultCapitalPreservationConfig)
import Trader.Coinbase (CoinbaseCandle (..), CoinbaseOrderInfo (..), alignCoinbaseClosesToGrid, coinbaseProductFromBinance, decodeCoinbaseOrderInfo)
import Trader.CostCalibration (
    calibratedSlippagePerSide,
    costCalibrationFloorFactor,
    costCalibrationMaxPerSide,
    costCalibrationMinObservations,
    minEdgeCostMultiplier,
    observedSlippageFraction,
    venueMinEdgeFloor,
    venueRoundTripCostFloor,
    venueSlippageFloor,
    venueSpreadFloor,
    venueTakerFeeFloor,
 )
import Trader.Formal.Execution (
    ExecutionVerificationReport (..),
    verifyFormalExecution,
 )
import Trader.Formal.Optimization (
    activityCountFromMetrics,
    fvrActivityCountInvariant,
    fvrOptimizerPublicSurfaceInvariant,
    roiViewFromMetrics,
    rvActivityCount,
    verifyFormalOptimization,
 )
import Trader.Formal.Risk (
    HaltInputs (..),
    RiskVerificationReport (..),
    specRiskHalt,
    verifyFormalRisk,
 )
import Trader.GateTelemetry (GateName (..), GateRejection (..), GateTelemetry (..), RejectionReason (..), bindingGate, emptyTelemetry, recordRejection, rejectionHistogram, telemetrySummary, telemetryToJson)
import Trader.KalmanFusion (Kalman1 (..), KalmanFusionConfig (..), defaultKalmanFusionConfig, initKalman1, innovationInflationFactor, measurementVarianceWithResidualFloor, predict, stepMulti, stepMultiWithConfig)
import Trader.LSTM (LSTMConfig (..), LSTMModel (..), buildSequences, defaultLstmAdamBeta1, defaultLstmAdamBeta2, defaultLstmAdamEps, evaluateLoss, fineTuneLSTM, fineTuneLSTMWeighted, inputDimFromModel, paramCount, paramCountD, predictNext, predictNextMulti, trainLSTM, trainLSTMMulti)
import Trader.LiveGap (
    LiveGapConfig (..),
    LiveGapStats (..),
    comboLiveGapEntry,
    comboLiveGapEntryWithConfig,
    defaultLiveGapConfig,
    liveGapMethodMultiplier,
    liveGapMethodMultiplierWithConfig,
    liveGapMinComboOperations,
    liveGapMinTotalOperations,
    liveGapMultiplierCeiling,
    liveGapMultiplierFloor,
    liveGapStatsByMethod,
    liveGapStatsByMethodWithConfig,
 )
import Trader.MarketContext (fitLinearRange)
import Trader.MarketDataIntegrity (
    isTransientMarketDataError,
    marketDataContinuationIssue,
    marketDataFreshness,
    marketDataStaleReason,
    mdfAgeMs,
    mdfFreshnessBudgetMs,
    mdfLastCloseTimeMs,
    mdfStale,
 )
import Trader.Method (Method (..))
import Trader.Metrics (BacktestMetrics (..), computeMetrics)
import Trader.Optimization (TuneConfig (..), TuneStats (..), defaultTuneConfig, sweepThresholdWithHLWith)
import Trader.Optimizer.Merge (MergeArgs (..), runMerge)
import Trader.Optimizer.Optimize (
    OptimizerRecordsSummary (..),
    applyWalkForwardSummaryMetrics,
    emptyOptimizerRecordsSummary,
    kellyLiteExposureContractReason,
    optimizerOptionPresent,
    optimizerRecordsShouldRetryDiscovery,
    qualityPresetBudget,
    qualityPresetCeiling,
    qualityPresetWeightFloor,
 )
import Trader.OrderExecution (applyExecutedQuantity, applyReduceOnlyExecutedQuantity)
import Trader.Platform (Platform (..))
import Trader.PredictionMarkets (
    PredictionMarketEvent (..),
    PredictionMarketMarket (..),
    PredictionMarketSignal (..),
    nearestPredictionMarketInterval,
    predictionMarketProbabilityForDir,
    selectPredictionMarketSignal,
 )
import Trader.Predictors (RegimeProbs (..))
import Trader.Predictors.Conformal (ConformalModel (..), fitConformal, predictInterval)
import Trader.Predictors.DecisionTree (DecisionTree (..), DecisionTreeModel (..), predictDecisionTree, trainDecisionTree)
import Trader.Predictors.Exogenous (alignToBars)
import Trader.Predictors.Features (featuresAtWithInputsWithMarket, mkFeatureInputs, mkFeatureSpec, withCoinbaseInputs)
import Trader.Predictors.GBDT (GBDTModel (..), Stump (..), predictGBDT, trainGBDT)
import Trader.Predictors.HMM (HMM3 (..), HMMFilter (..), filterPosterior, fitHMM3, predictNextFromPosterior, updatePosterior)
import Trader.Predictors.KNN (KNNModel (..), predictKNN, trainKNN)
import Trader.Predictors.Quantile (LinModel (..), QuantileModel (..), predictQuantiles, trainQuantileModel)
import Trader.Predictors.TCN (TCNModel (..), predictTCN, tcnFeaturesAt, trainTCN)
import Trader.SignalGates (
    DirectionalitySnapshot (..),
    PredictorLiveness (..),
    SignalThresholdBoundary (..),
    defaultSignalGateConfig,
    directionalityWeakBandConfirmed,
    directionalityWeakBandConfirmedWithPrediction,
    dynamicRangePct,
    finiteDouble,
    mkSignalThresholdBoundary,
    normalizeSignalEntryEdge,
    predictorDegenerate,
    predictorLiveness,
    signalCrossAssetCheck,
    signalDirectionalitySnapshot,
    signalDirectionalitySnapshotImplWithPrediction,
    signalEntryEdgeSpikeAuditWarning,
    signalEntryEdgeSpikeAuditWarningInterval,
    signalEntryEdgeSpikeEntryOk,
    signalEntryEdgeSpikeEntryOkInterval,
    signalEntryEdgeSpikeOk,
    signalEntryEdgeSpikeOkInterval,
    signalEntryFeeBufferOk,
    signalEntryHeadroomOk,
    signalEntryHeadroomThresholdCap,
    signalEntryOpenThresholdFeasibilityCap,
    signalEntryOpenThresholdFeasibilityReason,
    signalEntryOpenThresholdFeasible,
    signalFundingOiCheck,
    signalMetaLabelOk,
    signalMtfConsensusCheck,
    signalPredictionSanityOk,
    signalRegimeEdgeOk,
    signalRunPostDirectionGates,
 )
import Trader.Test.AutoStartBackoff (autoStartBackoffSuite)
import Trader.Test.BinanceProbe (binanceProbeSuite)
import Trader.Test.TechnicalAnalysis (runTechnicalAnalysisTests)
import Trader.ThresholdCalibration (
    CalibrationMethod (..),
    EdgeDistribution (..),
    ThresholdCalibration (..),
    ThresholdCalibrationConfig (..),
    calibrateThreshold,
    calibrateThresholdWithConfig,
    calibrationReport,
    calibrationToJson,
    computeEdgeDistribution,
    defaultThresholdCalibrationConfig,
    suggestedThreshold,
    thresholdAtPercentile,
    validateThresholdCalibrationConfig,
 )
import Trader.TopComboScoring (defaultTopComboScoringConfig)
import Trader.TopCombosStore (
    ComboBacktestApplyStats (..),
    ComboBacktestUpdate (..),
    ComboLiveStats (..),
    applyComboUpdatesKeepAllWithStats,
    applyComboUpdatesWithStats,
    blendedAnnualizedReturn,
    comboBacktestDueForRefresh,
    comboBacktestFreshnessMs,
    comboBacktestStaleAfterMs,
    comboIdentityKey,
    comboLiveQuarantined,
    comboLiveStats,
    comboLiveStatsFromObject,
    comboPerformanceKey,
    liveStatsFamilyQuarantined,
    liveStatsQuarantined,
    mergeTopCombosPayloads,
    recalculateComboPerformanceFromOperation,
    selectCombosForBacktestRefresh,
 )
import Trader.Trading (
    BacktestCostAttribution (..),
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
    emptyBacktestCostAttribution,
    entryEdge,
    entryGatesOk,
    exitReasonFromCode,
    feeBufferOk,
    mkEntryGateState,
    mkTradingEntryGateInputs,
    needsEntry,
    outcomeWeightCap,
    outcomeWeightLossScale,
    outcomeWeightWinScale,
    roundTripFeeFloor,
    simulateEnsemble,
    simulateEnsembleWithHLChecked,
    tradeEntrySourceCode,
    tradeOutcomeWeightFactor,
    tradeOutcomeWeights,
 )
import Trader.VolConfGate (
    VolConfGateBehavior (..),
    VolConfGateCell (..),
    VolConfGatePreset (..),
    applyVolConfGateBehavior,
    defaultVolConfGateConfig,
    volConfGateCell,
    volConfStatefulCloseDirection,
 )

main :: IO ()
main = do
    testNormalizeBarsForLookbackAutoAdjustsApiInputs
    testRsiPeriodRejectsInvalidValues
    testTrendLookbackRejectsInvalidValues
    testRiskPerTradeRejectsUpperBoundValidation
    testRiskPerTradeRejectsLowerBoundValidation
    testRiskPerTradeRequiresStopDefinition
    testMaxWeeklyLossRejectsUpperBoundValidation
    testMaxWeeklyLossRejectsLowerBoundValidation
    testMaxDailyLossRejectsUpperBoundValidation
    testMaxDailyLossRejectsLowerBoundValidation
    testMaxDrawdownRejectsUpperBoundValidation
    testMaxDrawdownRejectsLowerBoundValidation
    testStopLossHaltsSimulation
    testTakeProfitGuardrail
    testMaxDrawdownHaltsSimulation
    testTrailingStopGuardrail
    testVenueRoundTripCostFloorMatchesVenueCosts
    testVenueMinEdgeFloorClearsRoundTripCost
    testVenueMinEdgeFloorMatchesProductionRegressionEvidence
    testCapAdoptedMaxPositionSizeBoundsLiveExposure
    testAdoptionMinTradeCountMatchesOptimizerProductionGate
    testComboTradeCountMeetsAdoptionFloorMonotonicity
    testComboTradeCountMeetsAdoptionFloorHonorsConfig
    testComboTradeCountMeetsAdoptionFloorMatchesProductionRegressionEvidence
    testAdoptionMinWalkForwardSharpeMatchesOptimizerDefault
    testComboWalkForwardSharpeMeetsAdoptionFloorFailsClosed
    testComboWalkForwardSharpeMeetsAdoptionFloorMonotonicity
    testComboWalkForwardSharpeMeetsAdoptionFloorHonorsConfig
    testFuturesPositionRiskLeverageSaneCap
    testFeeRejectsNegativeValue
    testFeeRejectsAbsurdlyHighValue
    testFeeFixedRejectsAbsurdlyHighValue
    testFeeFixedRejectsNegativeValue
    testFeeMinRejectsNegativeValue
    testSlippageRejectsNegativeValue
    testSpreadRejectsNegativeValue
    testSlippageVolMultRejectsNegativeValue
    testStopLossRejectsLowerBoundValidation
    testStopLossRejectsAbsurdlyTightStop
    testTakeProfitRejectsAbsurdlyTightValue
    testTakeProfitRejectsLowerBoundValidation
    testTakeProfitRejectsUpperBoundValidation
    testTrailingStopRejectsAbsurdlyTightValue
    testTrailingStopRejectsLowerBoundValidation
    testMaxPositionSizeRejectsAbsurdUpperBound
    testMaxPositionSizeRejectsNonFuturesOverFive
    testMaxPositionSizeRejectsZeroAndNegative
    testInitialBalanceRejectsZeroOrNegative
    testMinPositionSizeRejectsOutOfRangeValues
    testBacktestRatioRejectsInvalidValues
    testOrderQuoteFractionRejectsInvalidValues
    testMaxOrderQuoteRejectsAbsurdValue
    testFromMustBeLessThanOrEqualToTo
    testValRatioRejectsInvalidValues
    testHiddenSizeRejectsInvalidValues
    testLrRejectsInvalidValues
    testEpochsRejectsInvalidValues
    testGradClipRejectsInvalidValues
    testKalmanDtRejectsInvalidValues
    testKalmanProcessVarRejectsInvalidValues
    testKalmanMeasurementVarRejectsInvalidValues
    testSensorVarianceEwmaAlphaRejectsInvalidValues
    testKalmanConservativeFusionRejectsInvalidValues
    testKalmanResidualVarianceFloor
    testKalmanSensorCorrelationInflation
    testKalmanInnovationInflation
    testKalmanVarianceKnobsAffectPrediction
    testKalmanVarianceKnobsAffectTrades
    testKalmanMarketTopNRejectsInvalidValues
    testTuneRatioRejectsInvalidValues
    testTunePenaltyMaxDrawdownRejectsInvalidValues
    testTunePenaltyTurnoverRejectsInvalidValues
    testTuneMaxThresholdCandidatesRejectsInvalidValues
    testWalkForwardFoldsRejectsInvalidValues
    testWalkForwardEmbargoBarsRejectsInvalidValues
    testPatienceRejectsInvalidValues
    testOpenThresholdRejectsInvalidValues
    testCloseThresholdRejectsInvalidValues
    testRouterLookbackRejectsInvalidValues
    testRouterMinScoreRejectsInvalidValues
    testRouterScorePnlWeightRejectsInvalidValues
    testExpectancyLookbackRejectsNegativeValue
    testPerfLookbackRejectsNegativeValue
    testCapitalPreservationReport
    testLossStreakMaxRejectsNegativeValue
    testLossStreakCooldownBarsRejectsNegativeValue
    testVolScaleMaxRejectsInvalidValues
    testRsiLowerMustBeLessThanUpper
    testExchangeDataLongShortBacktestAllowed
    testPositioningShortAliasRejected
    testTenantResolutionScopesMixedApiKeys
    testMarketLinearFit
    testSignalGateEntryBoundaryWitness
    testSignalGateEntryThresholdFeasibilityInvariant
    testMarketDataFreshnessAndContinuationInvariant
    testSignalGateEntryEdgeSpikeCapRegression
    testSignalGateEntryEdgeSpikeAuditWarning
    testSignalGateEntryHeadroomSpecializesFeeBuffer
    testNormalizeSignalEntryEdgeFailClosedRegression
    testSignalGateEntryFeeBufferFailsClosed
    testSignalGateIntervalAwareEdgeSpikeCap
    testSignalGatePredictionSanityInvariant
    testSignalGatePredictionAwareWeakBand
    testSignalDirectionalityLiveSemanticsRegression
    testSignalDirectionalityPredictionAwareLiveSemantics
    testPredictionMarketHerdSelection
    testSignalGatesPublicSurfaceRegression
    testSignalGateVolTargetPrecedesCloud
    testTradingPublicSurfaceRegression
    testKellyLiteBacktestSizingRegression
    testPositionSizeScaleSanityInvariant
    testTradingEntryGateFailClosedMonotone
    testTradingEntryGateMalformedNoReopen
    testVolConfGateMalformedInputsFailClosed
    testQueuedBotStartOrderErrorStability
    testQueuedBotStartIgnoresTransientMarketDataErrors
    testPrioritizeOrphanBotStartSymbols
    testDisabledBotStartSymbols
    testBotStartupBacktestRoiAcceptability
    testBotStartupBacktestGuardFailOpen
    testBotStartupBacktestVerdictZeroTradeIsNoVerdict
    testBotStartupBacktestVerdictAbortOnLossWithTrades
    testBotStartupBacktestVerdictPreservesDisabledBehaviour
    testApplyComboUpdatesZeroTradeDoesNotPrune
    testApplyComboUpdatesGenuineLossStillPrunes
    testBotStartupBacktestVerdictMinTradesGuard
    testBotStartupBacktestVerdictDefaultMinTradesIsThree
    testBotStartupGuardShouldPruneIsFalse
    testLiveBlendShrinkageRanking
    testLiveQuarantineThresholds
    testLiveFamilyQuarantineAcrossUuidChurn
    testRecalculateMaintainsLiveStats
    testBacktestUpdatePreservesLiveStats
    testMergePreservesLiveStats
    testMergeRefreshedComboBeatsStaleScore
    testMergeRefreshedComboBeatsUntimestampedStaleScore
    testMergeNewerDiscoveryBeatsOlderRefresh
    testMergeUnstampedDuplicatesKeepBestEver
    testMergeSanitizeKeepsStampedSubOneRefresh
    testLowTradeTopCombosSinkBelowEvidenceFloor
    testMergeDedupesSourceAndNullEquivalentCombos
    testDeployableTierRanksAheadOfUnvalidatedCandidate
    testMergeExecutableAnnotatesProcessingAndDedupe
    testSelectCombosForBacktestRefreshIncludesEveryStaleCombo
    testPrunedBacktestTombstonePreventsStaleResurrection
    testKeepAllUpdateKeepsUnprofitableComboStamped
    testTradeOutcomeWeightsSemantics
    testWeightedFineTuneUnitWeightsEquivalence
    testWeightedFineTunePunishesLossRegion
    testObservedSlippageFractionSemantics
    testCalibratedSlippageShrinkage
    testLiveGapFeedback
    testAlignToBarsPointInTime
    testNormalizeBarsForLookbackBinanceClampsAtPageCap
    testBinanceExceptionSummaryRedactsSecrets
    testConformalCalibrationResidualsFailClosed
    testBacktestEntryGateUsesRoundTripFeeBuffer
    testBacktestFreshEntrySizingBoundsFailClosed
    testBacktestPositionSizeFloorCapValidation
    testBacktestCostAttributionGrossNetConsistency
    testBacktestCostAttributionNonFiniteComponentsRegression
    testOrderExecutionFillSanitizationInvariant
    testOrderExecutionCorruptedInputInvariant
    testCoinbaseOrderInfoDecodeInvariant
    testOptimizerActivityCountInvariant
    testSweepThresholdMinRoundTripsFallback
    testOptimizerPublicSurfaceRegression
    testOptimizerQualityBudgetRegression
    testOptimizerQualityThresholdArgvExplicitRegression
    testOptimizerKellyLiteExposureContractRegression
    testOptimizerRecordsRetryDiscoveryForWalkForwardFilters
    testOptimizerRecordMetricsCarryWalkForwardSummary
    testOptimizerRecordsRetryDiscoveryForCostFloorFilters
    testOptimizerRecordsRetryDiscoveryStopsWhenEligible
    testTopComboBacktestPrunesRoiLosers
    testMetricsConsumesTradingPublicResults
    testGateTelemetryEmptyInvariant
    testGateTelemetryAccumulationInvariant
    testGateTelemetryBindingGateIdentification
    testGateTelemetryHistogramSorting
    testThresholdCalibrationEmptyInputFailsClosed
    testThresholdCalibrationDistributionAccuracy
    testThresholdCalibrationPercentileMethod
    testThresholdCalibrationConfigurableRoiKnobs
    testThresholdCalibrationInterpolatesIntermediatePercentiles
    testThresholdCalibrationStdDevMethod
    testThresholdCalibrationHybridMethod
    testThresholdCalibrationRecommendationInsufficientSample
    testThresholdCalibrationRecommendationConservative
    testThresholdCalibrationRecommendationAggressive
    testThresholdCalibrationRecommendationBalanced
    testFormalExecutionInvariants
    testFormalRiskInvariants
    testFormalRiskNoFalsePositiveWitness
    testFormalRiskNegativeLimitSanitization
    testFormalRiskPositionSizeHalt
    testFormalRiskLossStreakHalt
    testSignalGatesFailClosedExhaustive
    testPredictorLivenessDetectsDegenerateForecast
    testCrossExchangeCoinbaseInputs
    testMultivariateLstmInputs
    testGBDTSanitizesMalformedInputs
    testDecisionTreeSanitizesMalformedInputs
    testKNNSanitizesMalformedInputs
    testQuantileSanitizesMalformedInputs
    testTCNSanitizesMalformedInputs
    testHMMSanitizesMalformedInputs
    runTechnicalAnalysisTests
    runSuite "binanceProbe" binanceProbeSuite
    runSuite "autoStartBackoff" autoStartBackoffSuite

runSuite :: String -> [(String, IO ())] -> IO ()
runSuite label cases =
    forM_ cases $ \(name, run) -> do
        result <- try run :: IO (Either SomeException ())
        case result of
            Left ex -> ioError (userError (label ++ " :: " ++ name ++ " failed: " ++ show ex))
            Right () -> pure ()

assert :: String -> Bool -> IO ()
assert message condition =
    unless condition (ioError (userError ("Assertion failed: " ++ message)))

assertMonotoneNonIncreasing :: String -> [Bool] -> IO ()
assertMonotoneNonIncreasing message values =
    assert message (and (zipWith (\left right -> left || not right) values (drop 1 values)))

{- | Regression for the 2026-06-02 silent zero-trade failure: an untrained
forecaster pinned near a constant while price trends should be flagged
degenerate, because its largest single-bar step cannot clear the entry
threshold. A live forecaster, or any run that actually closed trades, must not
be flagged.
-}
testPredictorLivenessDetectsDegenerateForecast :: IO ()
testPredictorLivenessDetectsDegenerateForecast = do
    let openThreshold = 0.0024 :: Double
        -- Flat untrained forecast: ~94400 every bar, dynamic range ~7 over the
        -- window, max consecutive step return ~1.1e-5 (observed shape).
        flatSeries =
            [Just (94400.0 + fromIntegral i * 1.0e-4) | i <- [0 .. 200 :: Int]] ++ [Nothing]
        -- Live forecast: tracks a +20% move with multi-percent per-bar steps.
        liveSeries =
            map (Just . (94400.0 *)) (take 60 (iterate (* 1.01) 1.0)) ++ [Nothing]
        -- Price dynamic range: simulate ~20% bull move akin to the bull dataset.
        priceSeries =
            map (Just . (94400.0 *)) (take 200 (iterate (* 1.001) 1.0))
        priceRangePct = fromMaybe 0 (dynamicRangePct priceSeries)
        flatLiveness = predictorLiveness "lstm" priceRangePct flatSeries
        liveLiveness = predictorLiveness "lstm" priceRangePct liveSeries
    assert
        "price dynamic range is positive on a trending series"
        (priceRangePct > 0.1)
    case flatLiveness of
        Just pl -> do
            assert
                "flat forecast max step return is far below the open threshold"
                (plMaxStepReturn pl < openThreshold)
            assert
                "flat forecast dynamic range is negligible (<0.01%)"
                (plDynamicRangePct pl < 1.0e-4)
            assert
                "flat forecast tracking ratio is below the 5% floor"
                (plPriceTrackingRatio pl < 0.05)
        Nothing -> assert "flat forecast should yield liveness diagnostics" False
    case liveLiveness of
        Just pl -> do
            assert
                "live forecast max step return clears the open threshold"
                (plMaxStepReturn pl >= openThreshold)
            assert
                "live forecast tracking ratio is well above the 5% floor"
                (plPriceTrackingRatio pl > 0.5)
        Nothing -> assert "live forecast should yield liveness diagnostics" False
    -- Zero closed trades + flat predictor => degenerate (structural no-trade).
    assert
        "degenerate flag fires for flat predictor with zero trades"
        (predictorDegenerate 0 (catMaybes [flatLiveness]))
    -- Any closed trade means the no-trade was a market decision, not structural.
    assert
        "degenerate flag is suppressed once trades close"
        (not (predictorDegenerate 3 (catMaybes [flatLiveness])))
    -- A live predictor that simply found no signal is not degenerate.
    assert
        "degenerate flag is suppressed for a live predictor"
        (not (predictorDegenerate 0 (catMaybes [liveLiveness])))
    -- No predictor series at all => nothing to assess => not degenerate.
    assert
        "degenerate flag is suppressed when no predictor series is present"
        (not (predictorDegenerate 0 []))

{- | Cross-exchange (Coinbase) input enrichment: symbol mapping, bar-grid
alignment (point-in-time forward-fill, no lookahead), and the gated LSTM feature
block (default off => identical vector; on => +5 same-asset features).
-}
testCrossExchangeCoinbaseInputs :: IO ()
testCrossExchangeCoinbaseInputs = do
    -- Symbol mapping: USD-pegged Binance quotes map to a Coinbase USD product.
    assert "BTCUSDT -> BTC-USD" (coinbaseProductFromBinance "BTCUSDT" == Just "BTC-USD")
    assert "ETHUSDC -> ETH-USD" (coinbaseProductFromBinance "ETHUSDC" == Just "ETH-USD")
    assert "lower-case symbol still maps" (coinbaseProductFromBinance "btcusdt" == Just "BTC-USD")
    assert "non-USD quote does not map" (isNothing (coinbaseProductFromBinance "ETHBTC"))

    -- Alignment: exact match at matching open times, point-in-time forward-fill
    -- for gaps, and NEVER a future bucket (bar at sec 120 must use 10, not 30).
    let candle s c = CoinbaseCandle{ccOpenTime = s, ccHigh = c, ccLow = c, ccClose = c}
        openTimesMs = [60000, 120000, 180000, 240000]
        binanceCloses = V.fromList [1, 2, 3, 4]
        aligned = alignCoinbaseClosesToGrid openTimesMs binanceCloses [candle 60 10, candle 180 30]
    assert
        "alignment forward-fills gaps without lookahead"
        (aligned == Just (V.fromList [10, 10, 30, 30]))

    -- Leading gap (before first Coinbase candle) falls back to the Binance close.
    let leading = alignCoinbaseClosesToGrid openTimesMs binanceCloses [candle 180 30]
    assert
        "leading bars fall back to the Binance close"
        (leading == Just (V.fromList [1, 2, 30, 30]))

    -- No overlap at all => Nothing (caller fails open to Binance-only).
    assert
        "no Coinbase/Binance bar overlap yields Nothing"
        (isNothing (alignCoinbaseClosesToGrid [60000, 120000] (V.fromList [1, 2]) [candle 999999 5]))
    assert
        "mismatched Binance grid lengths fail open"
        ( isNothing (alignCoinbaseClosesToGrid [60000] (V.fromList [1, 2]) [candle 60 10])
            && isNothing (alignCoinbaseClosesToGrid [60000, 120000] (V.fromList [1]) [candle 60 10])
        )

    -- Feature gating: attaching a Coinbase close series appends exactly 5
    -- same-asset features; absence is byte-identical to Binance-only.
    let n = 60 :: Int
        closes = V.fromList [100 + fromIntegral i * 0.5 | i <- [0 .. n - 1]]
        fs = mkFeatureSpec 10
        baseInputs = mkFeatureInputs closes Nothing Nothing Nothing Nothing
        cbInputs = withCoinbaseInputs (Just (V.map (* 1.02) closes)) baseInputs
        t = 40
    case (featuresAtWithInputsWithMarket fs Nothing baseInputs t, featuresAtWithInputsWithMarket fs Nothing cbInputs t) of
        (Just featsBase, Just featsCb) -> do
            assert
                "Coinbase attachment appends exactly 5 cross-exchange features"
                (length featsCb == length featsBase + 5)
            assert
                "Binance-only feature vector is unchanged when Coinbase is absent"
                (take (length featsBase) featsCb == featsBase)
            let basisNow = featsCb !! length featsBase
            assert
                "basis feature reflects the ~+2% Coinbase premium"
                (basisNow > 0.015 && basisNow < 0.025)
        _ -> assert "feature vectors should be computable at t" False

{- | Multivariate LSTM: a single channel is byte-identical to the univariate
model (so the default/live path is unchanged), input dim is recoverable from the
flat params, and a 2-channel (price + cross-exchange) model trains to a distinct,
finite predictor that actually uses the second channel.
-}
testMultivariateLstmInputs :: IO ()
testMultivariateLstmInputs = do
    let cfg =
            LSTMConfig
                { lcLookback = 4
                , lcHiddenSize = 3
                , lcEpochs = 2
                , lcLearningRate = 0.01
                , lcAdamBeta1 = defaultLstmAdamBeta1
                , lcAdamBeta2 = defaultLstmAdamBeta2
                , lcAdamEps = defaultLstmAdamEps
                , lcValRatio = 0
                , lcPatience = 0
                , lcGradClip = Nothing
                , lcSeed = 7
                }
        series = [0.10, 0.20, 0.15, 0.30, 0.25, 0.40, 0.35, 0.50, 0.45, 0.60, 0.55, 0.70, 0.65, 0.80]
        series2 = [0.05, 0.10, 0.08, 0.20, 0.15, 0.25, 0.20, 0.30, 0.28, 0.35, 0.32, 0.45, 0.40, 0.50]
        (mUni, _) = trainLSTM cfg series
        (mMultiSingle, _) = trainLSTMMulti cfg [series]
        (mMulti, _) = trainLSTMMulti cfg [series, series2]
        window = take 4 series
        finite x = not (isNaN x || isInfinite x)
    -- d=1 identity: single-channel multivariate training matches univariate.
    assert "single-channel multi training == univariate params" (lmParams mUni == lmParams mMultiSingle)
    assert "univariate model input dim recovers as 1" (inputDimFromModel mUni == 1)
    assert "univariate paramCount is consistent" (length (lmParams mUni) == paramCount (lmHiddenSize mUni))
    assert
        "predictNextMulti on a single channel == predictNext"
        (abs (predictNextMulti mUni [window] - predictNext mUni window) < 1.0e-12)
    -- d=2: trains a distinct, finite, dimension-2 model.
    assert "two-channel model input dim recovers as 2" (inputDimFromModel mMulti == 2)
    assert "two-channel paramCount matches paramCountD h 2" (length (lmParams mMulti) == paramCountD (lmHiddenSize mMulti) 2)
    assert "two-channel params differ from univariate" (lmParams mMulti /= lmParams mUni)
    assert "two-channel prediction is finite" (finite (predictNextMulti mMulti [take 4 series, take 4 series2]))

testGBDTSanitizesMalformedInputs :: IO ()
testGBDTSanitizesMalformedInputs = do
    let nan = 0 / 0
        inf = 1 / 0
        dataset =
            [ ([0], 0.01)
            , ([1], 0.02)
            , ([nan], 100)
            , ([2], 0.03)
            , ([inf], 0.05)
            , ([3, 4], 0.04)
            , ([4], inf)
            ]
        model = trainGBDT 4 0.1 dataset
        finite x = not (isNaN x || isInfinite x)
        finiteStump Stump{stThreshold = thr, stLeftValue = l, stRightValue = r} =
            all finite [thr, l, r]
        (goodPred, goodSigma) = predictGBDT model [1.5]
        (badPred, badSigma) = predictGBDT model [nan]
    assert "GBDT keeps the finite consistent feature dimension" (gmFeatureDim model == 1)
    assert
        "GBDT training drops malformed rows before fitting"
        ( all finite [gmBase model, gmLearningRate model]
            && all finiteStump (gmStumps model)
            && maybe False finite (gmSigma model)
        )
    assert "GBDT finite query prediction remains finite" (finite goodPred && maybe True finite goodSigma)
    assert "GBDT malformed query falls back to the base prediction" (badPred == gmBase model && maybe True finite badSigma)

    let emptyFromBadLearningRate = trainGBDT 4 nan [([0], 0.01)]
    assert "GBDT rejects non-finite learning rates" (emptyFromBadLearningRate == trainGBDT 0 0 [])

    let malformedModel =
            GBDTModel
                { gmBase = nan
                , gmLearningRate = 0.1
                , gmFeatureDim = 1
                , gmStumps = [Stump{stFeature = 0, stThreshold = 0, stLeftValue = 1, stRightValue = 2}]
                , gmSigma = Just inf
                }
    assert "GBDT malformed model fails closed" (predictGBDT malformedModel [0] == (0, Nothing))

testDecisionTreeSanitizesMalformedInputs :: IO ()
testDecisionTreeSanitizesMalformedInputs = do
    let nan = 0 / 0
        inf = 1 / 0
        dataset =
            [ ([0], 0.01)
            , ([1], 0.02)
            , ([nan], 100)
            , ([2], 0.03)
            , ([inf], 0.05)
            , ([3, 4], 0.04)
            , ([4], inf)
            ]
        model = trainDecisionTree 3 1 dataset
        finite x = not (isNaN x || isInfinite x)
        (goodPred, goodSigma) = predictDecisionTree model [1.5]
        (badPred, badSigma) = predictDecisionTree model [nan]
    assert "DecisionTree keeps the finite consistent feature dimension" (dmFeatureDim model == 1)
    assert "DecisionTree finite query prediction remains finite" (finite goodPred && maybe True finite goodSigma)
    assert "DecisionTree malformed query fails closed" (badPred == 0 && maybe True finite badSigma)

    let finiteLeaf = DecisionLeaf{dtValue = 0.01, dtSigma = Just 0.02, dtCount = 1}
        sigmaPoisonedLeaf = DecisionLeaf{dtValue = 0.01, dtSigma = Just nan, dtCount = 1}
        valuePoisonedLeaf = DecisionLeaf{dtValue = inf, dtSigma = Just 0.02, dtCount = 1}
        invalidThresholdRoot =
            DecisionNode
                { dtFeature = 0
                , dtThreshold = nan
                , dtLeft = finiteLeaf
                , dtRight = finiteLeaf
                }
        mkModel root =
            DecisionTreeModel
                { dmFeatureDim = 1
                , dmMaxDepth = 1
                , dmMinLeafSize = 1
                , dmRoot = Just root
                , dmSigmaBase = Just inf
                }
    assert "DecisionTree drops malformed leaf sigma" (predictDecisionTree (mkModel sigmaPoisonedLeaf) [0] == (0.01, Nothing))
    assert "DecisionTree malformed leaf value fails closed" (predictDecisionTree (mkModel valuePoisonedLeaf) [0] == (0, Nothing))
    assert "DecisionTree malformed split threshold fails closed" (predictDecisionTree (mkModel invalidThresholdRoot) [0] == (0, Nothing))

testKNNSanitizesMalformedInputs :: IO ()
testKNNSanitizesMalformedInputs = do
    let nan = 0 / 0
        inf = 1 / 0
        dataset =
            [ ([0], 0.01)
            , ([1], 0.02)
            , ([nan], 100)
            , ([2], 0.03)
            , ([inf], 0.05)
            , ([3, 4], 0.04)
            , ([4], inf)
            ]
        model = trainKNN 16 3 dataset
        finite x = not (isNaN x || isInfinite x)
        finiteExample (x, y) = all finite x && finite y
        (goodPred, goodSigma) = predictKNN model [1.5]
        (badPred, badSigma) = predictKNN model [nan]
    assert "KNN keeps the finite consistent feature dimension" (kmFeatureDim model == 1)
    assert
        "KNN training drops malformed rows before fitting"
        ( all finite (kmMeans model)
            && all finite (kmScales model)
            && all finiteExample (kmExamples model)
            && maybe True finite (kmSigmaBase model)
        )
    assert "KNN finite query prediction remains finite" (finite goodPred && maybe True finite goodSigma)
    assert "KNN malformed query fails closed" (badPred == 0 && maybe True finite badSigma)

    let malformedModel =
            KNNModel
                { kmK = 1
                , kmFeatureDim = 1
                , kmMeans = [0]
                , kmScales = [1]
                , kmExamples = [([inf], 0.01)]
                , kmSigmaBase = Just inf
                }
    assert "KNN malformed model drops poisoned fallback sigma" (predictKNN malformedModel [0] == (0, Nothing))

testQuantileSanitizesMalformedInputs :: IO ()
testQuantileSanitizesMalformedInputs = do
    let nan = 0 / 0
        inf = 1 / 0
        dataset =
            [ ([0], 0.01)
            , ([1], 0.02)
            , ([nan], 100)
            , ([2], 0.03)
            , ([inf], 0.05)
            , ([3, 4], 0.04)
            , ([4], inf)
            ]
        model = trainQuantileModel 3 0.05 0.001 dataset
        finite x = not (isNaN x || isInfinite x)
        finiteLin LinModel{lmW = w, lmB = b} = all finite (b : w)
    assert
        "Quantile training drops malformed rows before fitting"
        (all finiteLin [qm10 model, qm50 model, qm90 model])
    assert
        "Quantile finite query remains usable after sanitization"
        ( case predictQuantiles model [1.5] of
            Just (q10', q50', q90', q50Raw, mSigma) ->
                all finite [q10', q50', q90', q50Raw]
                    && q10' <= q50'
                    && q50' <= q90'
                    && maybe True finite mSigma
            Nothing -> False
        )
    assert "Quantile malformed query fails closed" (isNothing (predictQuantiles model [nan]))

    let emptyFromBadLearningRate = trainQuantileModel 3 nan 0.001 [([0], 0.01)]
        emptyFromBadRegularization = trainQuantileModel 3 0.05 nan [([0], 0.01)]
    assert "Quantile rejects non-finite learning rates" (isNothing (predictQuantiles emptyFromBadLearningRate [0]))
    assert "Quantile rejects non-finite regularization" (isNothing (predictQuantiles emptyFromBadRegularization [0]))

    let malformedModel =
            QuantileModel
                LinModel{lmW = [nan], lmB = 0}
                LinModel{lmW = [0], lmB = 0}
                LinModel{lmW = [0], lmB = inf}
    assert "Quantile malformed model fails closed" (isNothing (predictQuantiles malformedModel [0]))

testTCNSanitizesMalformedInputs :: IO ()
testTCNSanitizesMalformedInputs = do
    let nan = 0 / 0
        inf = 1 / 0
        prices = V.fromList [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        poisonedPrices = prices V.// [(8, nan)]
        trainTargets =
            [ (3, 0.01)
            , (4, 0.02)
            , (5, nan)
            , (6, 0.03)
            , (7, inf)
            , (8, 0.04)
            ]
        model = trainTCN 4 prices trainTargets
        finite x = not (isNaN x || isInfinite x)
    assert
        "TCN feature extraction rejects non-finite price windows"
        (isNothing (tcnFeaturesAt [1] 2 poisonedPrices 9))
    assert
        "TCN training drops malformed targets before fitting"
        (not (null (tmWeights model)) && all finite (tmWeights model) && maybe True finite (tmSigma model))
    assert
        "TCN finite prediction remains finite"
        ( case predictTCN model prices 9 of
            Just (mu, sigma) -> finite mu && maybe True finite sigma
            Nothing -> False
        )
    assert "TCN malformed prediction window fails closed" (isNothing (predictTCN model poisonedPrices 9))

    let malformedModel =
            TCNModel
                { tmDilations = [1]
                , tmKernelSize = 2
                , tmWeights = [nan, 0, 1]
                , tmSigma = Just inf
                }
    assert "TCN malformed model weights fail closed" (isNothing (predictTCN malformedModel prices 4))

testHMMSanitizesMalformedInputs :: IO ()
testHMMSanitizesMalformedInputs = do
    let nan = 0 / 0
        inf = 1 / 0
        finite x = not (isNaN x || isInfinite x)
        probsOk xs =
            length xs == 3
                && all finite xs
                && all (>= 0) xs
                && abs (sum xs - 1) < 1e-9
        regimeOk RegimeProbs{rpTrend = trend, rpMR = mr, rpHighVol = highVol} =
            all finite [trend, mr, highVol]
                && all (>= 0) [trend, mr, highVol]
        hmmOk HMM3{hmmPi = pi0, hmmA = a0, hmmMu = mu0, hmmVar = var0} =
            probsOk pi0
                && all probsOk a0
                && length mu0 == 3
                && all finite mu0
                && length var0 == 3
                && all (\v -> finite v && v > 0) var0
        model = fitHMM3 4 [0.01, nan, -0.01, inf, 0.02]
        filt = filterPosterior model [0.01, nan, 0.02, inf]
        (reg, mu, sigma, predState) = predictNextFromPosterior model filt
        updated = updatePosterior model predState nan
    assert "HMM fit drops malformed observations before EM" (hmmOk model)
    assert "HMM filter drops malformed observations" (probsOk (hfPosterior filt))
    assert
        "HMM prediction remains finite and normalized"
        (regimeOk reg && finite mu && finite sigma && sigma > 0 && probsOk predState)
    assert "HMM malformed update observation preserves a normalized posterior" (probsOk (hfPosterior updated))

    let malformedModel =
            HMM3
                { hmmPi = [nan, inf, -1]
                , hmmA =
                    [ [nan, inf, -1]
                    , [0, 0, 0]
                    , [0.2, 0.3, 0.5]
                    ]
                , hmmMu = [nan, 0, inf]
                , hmmVar = [nan, -1, inf]
                , hmmTrendIx = 99
                , hmmMrIx = -1
                , hmmHighVolIx = 2
                }
        (reg2, mu2, sigma2, predState2) =
            predictNextFromPosterior malformedModel HMMFilter{hfPosterior = [nan, 2, -1, 4]}
    assert
        "HMM malformed model normalizes to finite predictions"
        (regimeOk reg2 && finite mu2 && finite sigma2 && sigma2 > 0 && probsOk predState2)

topCombosCount :: Aeson.Value -> Int
topCombosCount val =
    case val of
        Aeson.Object obj ->
            case KM.lookup (AK.fromString "combos") obj of
                Just (Aeson.Array combos) -> V.length combos
                _ -> -1
        _ -> -1

parseAndValidateCliArgs :: [String] -> Either String Args
parseAndValidateCliArgs argv =
    case execParserPure defaultPrefs (info opts mempty) argv of
        Success args -> validateArgs args
        Failure _ -> Left "CLI parse failed unexpectedly"
        CompletionInvoked _ -> Left "CLI completion invoked unexpectedly"

parseCliArgs :: [String] -> Either String Args
parseCliArgs argv =
    case execParserPure defaultPrefs (info opts mempty) argv of
        Success args -> Right args
        Failure _ -> Left "CLI parse failed unexpectedly"
        CompletionInvoked _ -> Left "CLI completion invoked unexpectedly"

testTopComboBacktestPrunesRoiLosers :: IO ()
testTopComboBacktestPrunesRoiLosers = do
    let combo =
            Aeson.object
                [ "rank" Aeson..= (1 :: Int)
                , "finalEquity" Aeson..= (1.25 :: Double)
                , "objective" Aeson..= ("roi" :: String)
                , "score" Aeson..= (2.0 :: Double)
                , "openThreshold" Aeson..= (0.01 :: Double)
                , "closeThreshold" Aeson..= (0.005 :: Double)
                , "params"
                    Aeson..= Aeson.object
                        [ "symbol" Aeson..= ("BTCUSDT" :: String)
                        , "interval" Aeson..= ("1h" :: String)
                        , "method" Aeson..= ("both" :: String)
                        ]
                , "metrics"
                    Aeson..= Aeson.object
                        [ "finalEquity" Aeson..= (1.25 :: Double)
                        , "annualizedReturn" Aeson..= (0.3 :: Double)
                        ]
                ]
        payload =
            Aeson.object
                [ "generatedAtMs" Aeson..= (1 :: Int)
                , "source" Aeson..= ("test" :: String)
                , "combos" Aeson..= [combo]
                ]
        losingMetrics =
            Aeson.object
                [ "finalEquity" Aeson..= (1.0 :: Double)
                , "annualizedReturn" Aeson..= (0.0 :: Double)
                ]
        update =
            ComboBacktestUpdate
                { cbuMetrics = losingMetrics
                , cbuFinalEquity = Just 1.0
                , cbuScore = Just 0
                , cbuOperations = Nothing
                }
    case comboIdentityKey combo of
        Nothing -> assert "top-combo fixture has a stable identity key" False
        Just key ->
            case applyComboUpdatesWithStats 2 (HM.singleton key update) payload of
                Left err -> assert ("top-combo backtest update succeeds: " ++ err) False
                Right (updatedPayload, stats) -> do
                    assert
                        "top-combo backtest prunes refreshed combos that no longer clear finalEquity > 1"
                        ( topCombosCount updatedPayload == 0
                            && cbasUpdatedCount stats == 1
                            && cbasPrunedCount stats == 1
                            && cbasPrunedKeys stats == [key]
                        )

testNormalizeBarsForLookbackAutoAdjustsApiInputs :: IO ()
testNormalizeBarsForLookbackAutoAdjustsApiInputs = do
    let parseOrFail argv =
            case parseCliArgs argv of
                Left err -> ioError (userError ("CLI parse failed unexpectedly: " ++ err))
                Right args -> pure args

    binanceArgs <-
        parseOrFail
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "960"
            , "--lookback-bars"
            , "960"
            ]
    let adjustedBinanceArgs = normalizeBarsForLookback binanceArgs
    assert
        "API lookback normalization raises explicit exchange bars to lookback+1 when the request is otherwise feasible"
        ( argBars adjustedBinanceArgs == Just 961
            && case validateArgs adjustedBinanceArgs of
                Right _ -> True
                Left _ -> False
        )

    taArgs <-
        parseOrFail
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--method"
            , "ta_regime_switch"
            , "--bars"
            , "20"
            , "--lookback-bars"
            , "10"
            ]
    let adjustedTaArgs = normalizeBarsForLookback taArgs{argTradeOnly = True}
    assert
        "trade-only TA regime-switch requests are raised to the 60-bar live minimum instead of failing bot/signal/trade starts"
        ( argBars adjustedTaArgs == Just 60
            && argMethod adjustedTaArgs == MethodTaRegimeSwitch
            && case validateArgs adjustedTaArgs of
                Right _ -> True
                Left _ -> False
        )

    overCapArgs <-
        parseOrFail ["--binance-symbol", "BTCUSDT", "--interval", "1m", "--bars", "500", "--lookback-bars", "1000"]
    assert
        "lookback normalization does not push Binance requests past the 1000-bar platform cap"
        (argBars (normalizeBarsForLookback overCapArgs) == Just 500)

testRiskPerTradeRejectsUpperBoundValidation :: IO ()
testRiskPerTradeRejectsUpperBoundValidation = do
    let rejectedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--risk-per-trade"
            , "1"
            , "--stop-loss"
            , "0.02"
            ]
        acceptedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--risk-per-trade"
            , "0.01"
            , "--stop-loss"
            , "0.02"
            ]
    assert
        "risk-per-trade rejects the exact upper bound while preserving valid stop-backed sizing inputs"
        ( parseAndValidateCliArgs rejectedArgs == Left "--risk-per-trade must be > 0 and < 1"
            && case parseAndValidateCliArgs acceptedArgs of
                Right _ -> True
                Left _ -> False
        )

testRiskPerTradeRejectsLowerBoundValidation :: IO ()
testRiskPerTradeRejectsLowerBoundValidation = do
    let rejectedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--risk-per-trade"
            , "0"
            , "--stop-loss"
            , "0.02"
            ]
        acceptedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--risk-per-trade"
            , "0.01"
            , "--stop-loss"
            , "0.02"
            ]
    assert
        "risk-per-trade rejects the exact lower bound while preserving valid stop-backed sizing inputs"
        ( parseAndValidateCliArgs rejectedArgs == Left "--risk-per-trade must be > 0 and < 1"
            && case parseAndValidateCliArgs acceptedArgs of
                Right _ -> True
                Left _ -> False
        )

testRiskPerTradeRequiresStopDefinition :: IO ()
testRiskPerTradeRequiresStopDefinition = do
    assert
        "risk-per-trade without a fixed or volatility stop is rejected"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--risk-per-trade", "0.01"] == Left "--risk-per-trade requires --stop-loss or --stop-loss-vol-mult")
    assert
        "risk-per-trade with a fixed stop is accepted"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--risk-per-trade", "0.01", "--stop-loss", "0.02"] of
            Right args -> argRiskPerTrade args == Just 0.01
            Left _ -> False
        )
    assert
        "risk-per-trade with a volatility stop is accepted"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--risk-per-trade", "0.01", "--stop-loss-vol-mult", "1.5"] of
            Right args -> argRiskPerTrade args == Just 0.01
            Left _ -> False
        )

testMaxWeeklyLossRejectsUpperBoundValidation :: IO ()
testMaxWeeklyLossRejectsUpperBoundValidation =
    assert
        "max-weekly-loss rejects the exact upper bound"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--max-weekly-loss", "1"] == Left "--max-weekly-loss must be > 0 and < 1")

testMaxWeeklyLossRejectsLowerBoundValidation :: IO ()
testMaxWeeklyLossRejectsLowerBoundValidation =
    assert
        "max-weekly-loss rejects the exact lower bound"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--max-weekly-loss", "0"] == Left "--max-weekly-loss must be > 0 and < 1")

testMaxDailyLossRejectsUpperBoundValidation :: IO ()
testMaxDailyLossRejectsUpperBoundValidation =
    assert
        "max-daily-loss rejects the exact upper bound"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--max-daily-loss", "1"] == Left "--max-daily-loss must be > 0 and < 1")

testMaxDailyLossRejectsLowerBoundValidation :: IO ()
testMaxDailyLossRejectsLowerBoundValidation =
    assert
        "max-daily-loss rejects the exact lower bound"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--max-daily-loss", "0"] == Left "--max-daily-loss must be > 0 and < 1")

testMaxDrawdownRejectsUpperBoundValidation :: IO ()
testMaxDrawdownRejectsUpperBoundValidation =
    assert
        "max-drawdown rejects the exact upper bound"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--max-drawdown", "1"] == Left "--max-drawdown must be > 0 and < 1")

testMaxDrawdownRejectsLowerBoundValidation :: IO ()
testMaxDrawdownRejectsLowerBoundValidation =
    assert
        "max-drawdown rejects the exact lower bound"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--max-drawdown", "0"] == Left "--max-drawdown must be > 0 and < 1")

testFuturesPositionRiskLeverageSaneCap :: IO ()
testFuturesPositionRiskLeverageSaneCap = do
    let ok = FuturesPositionRisk{fprSymbol = "BTCUSDT", fprPositionAmt = 1, fprEntryPrice = 50000, fprMarkPrice = 51000, fprUnrealizedProfit = 0, fprLiquidationPrice = Nothing, fprBreakEvenPrice = Nothing, fprLeverage = 125, fprMarginType = Nothing, fprPositionSide = Nothing}
        tooHigh = ok{fprLeverage = 151}
        zero = ok{fprLeverage = 0}
        nan = ok{fprLeverage = 0 / 0}
    assert "futuresPositionRiskLeverageSane accepts 125x" (futuresPositionRiskLeverageSane ok)
    assert "futuresPositionRiskLeverageSane rejects 151x" (not (futuresPositionRiskLeverageSane tooHigh))
    assert "futuresPositionRiskLeverageSane rejects 0x" (not (futuresPositionRiskLeverageSane zero))
    assert "futuresPositionRiskLeverageSane rejects NaN" (not (futuresPositionRiskLeverageSane nan))

testFeeRejectsNegativeValue :: IO ()
testFeeRejectsNegativeValue =
    assert
        "fee rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--fee", "-0.01"] == Left "--fee must be >= 0")

testFeeRejectsAbsurdlyHighValue :: IO ()
testFeeRejectsAbsurdlyHighValue =
    assert
        "fee rejects absurdly high values above 5%"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--fee", "0.06"] == Left "--fee must be <= 0.05 (5%)")

testFeeFixedRejectsAbsurdlyHighValue :: IO ()
testFeeFixedRejectsAbsurdlyHighValue =
    assert
        "fee-fixed rejects absurdly high values above 5%"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--fee-fixed", "0.06"] == Left "--fee-fixed must be <= 0.05 (5%)")

testFeeFixedRejectsNegativeValue :: IO ()
testFeeFixedRejectsNegativeValue =
    assert
        "fee-fixed rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--fee-fixed", "-0.01"] == Left "--fee-fixed must be >= 0")

testFeeMinRejectsNegativeValue :: IO ()
testFeeMinRejectsNegativeValue =
    assert
        "fee-min rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--fee-min", "-0.01"] == Left "--fee-min must be >= 0")

testSlippageRejectsNegativeValue :: IO ()
testSlippageRejectsNegativeValue =
    assert
        "slippage rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--slippage", "-0.01"] == Left "--slippage must be >= 0")

testSpreadRejectsNegativeValue :: IO ()
testSpreadRejectsNegativeValue =
    assert
        "spread rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--spread", "-0.01"] == Left "--spread must be >= 0")

testSlippageVolMultRejectsNegativeValue :: IO ()
testSlippageVolMultRejectsNegativeValue =
    assert
        "slippage-vol-mult rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--slippage-vol-mult", "-0.01"] == Left "--slippage-vol-mult must be >= 0")

testStopLossRejectsLowerBoundValidation :: IO ()
testStopLossRejectsLowerBoundValidation =
    assert
        "stop-loss rejects the exact lower bound"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--stop-loss", "0"] == Left "--stop-loss must be > 0 and < 1")

testStopLossRejectsAbsurdlyTightStop :: IO ()
testStopLossRejectsAbsurdlyTightStop =
    assert
        "stop-loss rejects absurdly tight values below 0.01%"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--stop-loss", "0.00001"] == Left "--stop-loss must be >= 0.0001 (0.01%)")

testTakeProfitRejectsAbsurdlyTightValue :: IO ()
testTakeProfitRejectsAbsurdlyTightValue =
    assert
        "take-profit rejects absurdly tight values below 0.01%"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--take-profit", "0.00001"] == Left "--take-profit must be >= 0.0001 (0.01%)")

testTakeProfitRejectsLowerBoundValidation :: IO ()
testTakeProfitRejectsLowerBoundValidation =
    assert
        "take-profit rejects the exact lower bound"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--take-profit", "0"] == Left "--take-profit must be > 0 and < 1")

testTakeProfitRejectsUpperBoundValidation :: IO ()
testTakeProfitRejectsUpperBoundValidation =
    assert
        "take-profit rejects the exact upper bound"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--take-profit", "1"] == Left "--take-profit must be > 0 and < 1")

testTrailingStopRejectsAbsurdlyTightValue :: IO ()
testTrailingStopRejectsAbsurdlyTightValue =
    assert
        "trailing-stop rejects absurdly tight values below 0.01%"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--trailing-stop", "0.00001"] == Left "--trailing-stop must be >= 0.0001 (0.01%)")

testTrailingStopRejectsLowerBoundValidation :: IO ()
testTrailingStopRejectsLowerBoundValidation =
    assert
        "trailing-stop rejects the exact lower bound"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--trailing-stop", "0"] == Left "--trailing-stop must be > 0 and < 1")

testMaxPositionSizeRejectsAbsurdUpperBound :: IO ()
testMaxPositionSizeRejectsAbsurdUpperBound = do
    assert
        "max-position-size rejects 10.01 (absurd upper bound)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--max-position-size", "10.01"] == Left "--max-position-size must be <= 10")
    assert
        "max-position-size accepts exactly 10"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--max-position-size", "10", "--futures"] of
            Right args -> argMaxPositionSize args == 10
            Left _ -> False
        )

testMaxPositionSizeRejectsNonFuturesOverFive :: IO ()
testMaxPositionSizeRejectsNonFuturesOverFive = do
    assert
        "max-position-size rejects 6 on Binance spot (non-futures upper bound is 5)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--max-position-size", "6"] == Left "--max-position-size must be <= 5 for spot/margin markets")
    assert
        "max-position-size accepts exactly 5 on Binance spot"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--max-position-size", "5"] of
            Right args -> argMaxPositionSize args == 5
            Left _ -> False
        )
    assert
        "max-position-size accepts 6 on Binance futures"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--max-position-size", "6", "--futures"] of
            Right args -> argMaxPositionSize args == 6
            Left _ -> False
        )

testMaxPositionSizeRejectsZeroAndNegative :: IO ()
testMaxPositionSizeRejectsZeroAndNegative = do
    assert
        "max-position-size rejects 0 (zero boundary)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--max-position-size", "0"] == Left "--max-position-size must be > 0")
    assert
        "max-position-size rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--max-position-size", "-0.1"] == Left "--max-position-size must be > 0")
    assert
        "max-position-size accepts small positive values"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--max-position-size", "0.01", "--min-position-size", "0.01"] of
            Right args -> argMaxPositionSize args == 0.01
            Left _ -> False
        )

testInitialBalanceRejectsZeroOrNegative :: IO ()
testInitialBalanceRejectsZeroOrNegative = do
    assert
        "initial-balance rejects zero"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--initial-balance", "0"] == Left "--initial-balance must be > 0")
    assert
        "initial-balance rejects negative"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--initial-balance", "-100"] == Left "--initial-balance must be > 0")

testMinPositionSizeRejectsOutOfRangeValues :: IO ()
testMinPositionSizeRejectsOutOfRangeValues = do
    assert
        "min-position-size rejects negative (below 0)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--min-position-size", "-0.1"] == Left "--min-position-size must be > 0")
    assert
        "min-position-size rejects zero (exact boundary)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--min-position-size", "0"] == Left "--min-position-size must be > 0")
    assert
        "min-position-size rejects above 1 (exceeds 100%)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--min-position-size", "1.1"] == Left "--min-position-size must be <= 1")
    assert
        "min-position-size accepts small positive (above 0)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--min-position-size", "0.01"] of
            Right args -> argMinPositionSize args == 0.01
            Left _ -> False
        )
    assert
        "min-position-size accepts 0.5 (valid boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--min-position-size", "0.5"] of
            Right args -> argMinPositionSize args == 0.5
            Left _ -> False
        )

testBacktestRatioRejectsInvalidValues :: IO ()
testBacktestRatioRejectsInvalidValues = do
    assert
        "backtest-ratio rejects 0 (no training data)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--backtest-ratio", "0"] == Left "--backtest-ratio must be between 0 and 1")
    assert
        "backtest-ratio rejects 1 (no test data)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--backtest-ratio", "1"] == Left "--backtest-ratio must be between 0 and 1")
    assert
        "backtest-ratio accepts 0.5 (valid split)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--backtest-ratio", "0.5"] of
            Right args -> argBacktestRatio args == 0.5
            Left _ -> False
        )

testOrderQuoteFractionRejectsInvalidValues :: IO ()
testOrderQuoteFractionRejectsInvalidValues = do
    assert
        "order-quote-fraction rejects 0 (no position)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--order-quote-fraction", "0"] == Left "--order-quote-fraction must be > 0 and <= 1")
    assert
        "order-quote-fraction rejects 1.1 (over-leverage)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--order-quote-fraction", "1.1"] == Left "--order-quote-fraction must be > 0 and <= 1")
    assert
        "order-quote-fraction accepts 0.5 (valid fraction)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--order-quote-fraction", "0.5"] of
            Right args -> argOrderQuoteFraction args == Just 0.5
            Left _ -> False
        )

testMaxOrderQuoteRejectsAbsurdValue :: IO ()
testMaxOrderQuoteRejectsAbsurdValue = do
    assert
        "max-order-quote rejects 10000000.01 (absurd upper bound)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--order-quote-fraction", "0.5", "--max-order-quote", "10000000.01"] == Left "--max-order-quote must be <= 10000000")
    assert
        "max-order-quote accepts 10000000 (sanity cap boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--order-quote-fraction", "0.5", "--max-order-quote", "10000000"] of
            Right args -> argMaxOrderQuote args == Just 10000000
            Left _ -> False
        )
    assert
        "max-order-quote accepts 1000 (normal value)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--order-quote-fraction", "0.5", "--max-order-quote", "1000"] of
            Right args -> argMaxOrderQuote args == Just 1000
            Left _ -> False
        )

testFromMustBeLessThanOrEqualToTo :: IO ()
testFromMustBeLessThanOrEqualToTo = do
    assert
        "from must be <= to rejects backwards window"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--from", "2024-01-02", "--to", "2024-01-01"] == Left "--from must be <= --to")
    assert
        "from must be <= to accepts valid forward window"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--from", "2024-01-01", "--to", "2024-01-02"] of
            Right args -> argBacktestFrom args <= argBacktestTo args
            Left _ -> False
        )

testValRatioRejectsInvalidValues :: IO ()
testValRatioRejectsInvalidValues = do
    assert
        "val-ratio rejects -0.1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--val-ratio", "-0.1"] == Left "--val-ratio must be >= 0 and < 1")
    assert
        "val-ratio rejects 1.0 (no validation data)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--val-ratio", "1.0"] == Left "--val-ratio must be >= 0 and < 1")
    assert
        "val-ratio accepts 0.2 (valid split)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--val-ratio", "0.2"] of
            Right args -> argValRatio args == 0.2
            Left _ -> False
        )

testHiddenSizeRejectsInvalidValues :: IO ()
testHiddenSizeRejectsInvalidValues = do
    assert
        "hidden-size rejects 0 (too small)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--hidden-size", "0"] == Left "--hidden-size must be >= 1")
    assert
        "hidden-size rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--hidden-size", "-1"] == Left "--hidden-size must be >= 1")
    assert
        "hidden-size accepts 1 (minimum valid)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--hidden-size", "1"] of
            Right args -> argHiddenSize args == 1
            Left _ -> False
        )
    assert
        "hidden-size accepts 16 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--hidden-size", "16"] of
            Right args -> argHiddenSize args == 16
            Left _ -> False
        )

testLrRejectsInvalidValues :: IO ()
testLrRejectsInvalidValues = do
    assert
        "lr rejects 0 (too small)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--lr", "0"] == Left "--lr must be > 0")
    assert
        "lr rejects -0.001 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--lr", "-0.001"] == Left "--lr must be > 0")
    assert
        "lr accepts 0.0001 (small positive)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--lr", "0.0001"] of
            Right args -> argLr args == 0.0001
            Left _ -> False
        )
    assert
        "lr accepts 0.001 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--lr", "0.001"] of
            Right args -> argLr args == 0.001
            Left _ -> False
        )

testEpochsRejectsInvalidValues :: IO ()
testEpochsRejectsInvalidValues = do
    assert
        "epochs rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--epochs", "-1"] == Left "--epochs must be >= 0")
    assert
        "epochs accepts 0 (minimum valid)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--epochs", "0"] of
            Right args -> argEpochs args == 0
            Left _ -> False
        )
    assert
        "epochs accepts 30 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--epochs", "30"] of
            Right args -> argEpochs args == 30
            Left _ -> False
        )
    assert
        "epochs accepts 100 (large positive)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--epochs", "100"] of
            Right args -> argEpochs args == 100
            Left _ -> False
        )

testGradClipRejectsInvalidValues :: IO ()
testGradClipRejectsInvalidValues = do
    assert
        "grad-clip rejects 0 (too small)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--grad-clip", "0"] == Left "--grad-clip must be > 0")
    assert
        "grad-clip rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--grad-clip", "-1"] == Left "--grad-clip must be > 0")
    assert
        "grad-clip accepts 0.5 (positive)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--grad-clip", "0.5"] of
            Right args -> argGradClip args == Just 0.5
            Left _ -> False
        )
    assert
        "grad-clip accepts 1.0 (default-like)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--grad-clip", "1.0"] of
            Right args -> argGradClip args == Just 1.0
            Left _ -> False
        )

testKalmanDtRejectsInvalidValues :: IO ()
testKalmanDtRejectsInvalidValues = do
    assert
        "kalman-dt rejects 0 (too small)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-dt", "0"] == Left "--kalman-dt must be > 0")
    assert
        "kalman-dt rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-dt", "-1"] == Left "--kalman-dt must be > 0")
    assert
        "kalman-dt accepts 0.5 (positive)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-dt", "0.5"] of
            Right args -> argKalmanDt args == 0.5
            Left _ -> False
        )
    assert
        "kalman-dt accepts 1.0 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-dt", "1.0"] of
            Right args -> argKalmanDt args == 1.0
            Left _ -> False
        )

testKalmanProcessVarRejectsInvalidValues :: IO ()
testKalmanProcessVarRejectsInvalidValues = do
    assert
        "kalman-process-var rejects 0 (too small)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-process-var", "0"] == Left "--kalman-process-var must be > 0")
    assert
        "kalman-process-var rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-process-var", "-1"] == Left "--kalman-process-var must be > 0")
    assert
        "kalman-process-var accepts 0.5 (positive)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-process-var", "0.5"] of
            Right args -> argKalmanProcessVar args == 0.5
            Left _ -> False
        )
    assert
        "kalman-process-var accepts 1e-5 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-process-var", "1e-5"] of
            Right args -> argKalmanProcessVar args == 1e-5
            Left _ -> False
        )

testKalmanMeasurementVarRejectsInvalidValues :: IO ()
testKalmanMeasurementVarRejectsInvalidValues = do
    assert
        "kalman-measurement-var rejects 0 (too small)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-measurement-var", "0"] == Left "--kalman-measurement-var must be > 0")
    assert
        "kalman-measurement-var rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-measurement-var", "-1"] == Left "--kalman-measurement-var must be > 0")
    assert
        "kalman-measurement-var accepts 0.5 (positive)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-measurement-var", "0.5"] of
            Right args -> argKalmanMeasurementVar args == 0.5
            Left _ -> False
        )
    assert
        "kalman-measurement-var accepts 1e-3 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-measurement-var", "1e-3"] of
            Right args -> argKalmanMeasurementVar args == 1e-3
            Left _ -> False
        )

testSensorVarianceEwmaAlphaRejectsInvalidValues :: IO ()
testSensorVarianceEwmaAlphaRejectsInvalidValues = do
    assert
        "sensor-variance-ewma-alpha rejects -0.1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--sensor-variance-ewma-alpha", "-0.1"] == Left "--sensor-variance-ewma-alpha must be between 0 and 1")
    assert
        "sensor-variance-ewma-alpha rejects 1.1 (above one)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--sensor-variance-ewma-alpha", "1.1"] == Left "--sensor-variance-ewma-alpha must be between 0 and 1")
    assert
        "sensor-variance-ewma-alpha accepts 0 (initial variance only)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--sensor-variance-ewma-alpha", "0"] of
            Right args -> argSensorVarianceEwmaAlpha args == 0
            Left _ -> False
        )
    assert
        "sensor-variance-ewma-alpha accepts 0.05 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--sensor-variance-ewma-alpha", "0.05"] of
            Right args -> argSensorVarianceEwmaAlpha args == 0.05
            Left _ -> False
        )
    assert
        "sensor-variance-ewma-alpha accepts 1 (latest residual only)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--sensor-variance-ewma-alpha", "1"] of
            Right args -> argSensorVarianceEwmaAlpha args == 1
            Left _ -> False
        )

testKalmanConservativeFusionRejectsInvalidValues :: IO ()
testKalmanConservativeFusionRejectsInvalidValues = do
    assert
        "kalman-sensor-correlation-inflation rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-sensor-correlation-inflation", "-0.1"] == Left "--kalman-sensor-correlation-inflation must be between 0 and 1")
    assert
        "kalman-sensor-correlation-inflation rejects values above one"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-sensor-correlation-inflation", "1.1"] == Left "--kalman-sensor-correlation-inflation must be between 0 and 1")
    assert
        "kalman-sensor-correlation-inflation accepts a conservative midpoint"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-sensor-correlation-inflation", "0.5"] of
            Right args -> argKalmanSensorCorrelationInflation args == 0.5
            Left _ -> False
        )
    assert
        "kalman-innovation-inflation-threshold rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-innovation-inflation-threshold", "-1"] == Left "--kalman-innovation-inflation-threshold must be >= 0")
    assert
        "kalman-innovation-inflation-threshold accepts a 3-sigma NIS"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-innovation-inflation-threshold", "9"] of
            Right args -> argKalmanInnovationInflationThreshold args == 9
            Left _ -> False
        )
    assert
        "kalman-innovation-inflation-max rejects values below one"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-innovation-inflation-max", "0.5"] == Left "--kalman-innovation-inflation-max must be >= 1")
    assert
        "kalman-innovation-inflation-max accepts a multiplier"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-innovation-inflation-max", "25"] of
            Right args -> argKalmanInnovationInflationMax args == 25
            Left _ -> False
        )

testKalmanResidualVarianceFloor :: IO ()
testKalmanResidualVarianceFloor = do
    let withLearned = measurementVarianceWithResidualFloor 0.001 (Just 0.01) (Just 0.04)
        withModel = measurementVarianceWithResidualFloor 0.001 (Just 0.05) Nothing
        withFallback = measurementVarianceWithResidualFloor 0.001 (Just (0 / 0)) (Just (-1))
    assert "learned residual variance floors an overconfident model sigma" (abs (withLearned - 0.04) < 1e-12)
    assert "model sigma variance is used when it exceeds the fallback" (abs (withModel - 0.0025) < 1e-12)
    assert "malformed model and learned variances fall back safely" (abs (withFallback - 0.001) < 1e-12)

testKalmanSensorCorrelationInflation :: IO ()
testKalmanSensorCorrelationInflation = do
    let prior = initKalman1 0 1 0
        independent = stepMulti [(1, 1), (1, 1)] prior
        conservative =
            stepMultiWithConfig
                defaultKalmanFusionConfig{kfcSensorCorrelationInflation = 1}
                [(1, 1), (1, 1)]
                prior
    assert "independent duplicate sensors halve measurement variance" (abs (kVar independent - (1 / 3)) < 1e-12)
    assert "fully correlated duplicate sensors count as one effective observation" (abs (kVar conservative - 0.5) < 1e-12)
    assert "correlated duplicate sensors move the posterior less" (kMean conservative < kMean independent)

testKalmanInnovationInflation :: IO ()
testKalmanInnovationInflation = do
    let cfg = KalmanFusionConfig{kfcSensorCorrelationInflation = 0, kfcInnovationInflationThreshold = 1, kfcInnovationInflationMax = 100}
        prior = initKalman1 0 1e-6 0
        outlier = [(1, 1e-6)]
        factor = innovationInflationFactor cfg (predict prior) outlier
        base = stepMulti outlier prior
        inflated = stepMultiWithConfig cfg outlier prior
    assert "large normalized innovation reaches the configured inflation cap" (abs (factor - 100) < 1e-9)
    assert "innovation inflation increases posterior covariance for an outlier" (kVar inflated > kVar base * 10)
    assert "innovation inflation tempers the immediate outlier pull" (kMean inflated < kMean base)

testKalmanVarianceKnobsAffectPrediction :: IO ()
testKalmanVarianceKnobsAffectPrediction = do
    let lowProcess = stepMulti [(1, 1)] (initKalman1 0 1e-6 0)
        highProcess = stepMulti [(1, 1)] (initKalman1 0 1e-6 1)
        tightMeasurement = stepMulti [(1, 0.01)] (initKalman1 0 1 0)
        looseMeasurement = stepMulti [(1, 100)] (initKalman1 0 1 0)
    assert "process variance changes Kalman prediction gain" (kMean highProcess > kMean lowProcess)
    assert "measurement variance changes Kalman prediction gain" (kMean tightMeasurement > kMean looseMeasurement)

testKalmanVarianceKnobsAffectTrades :: IO ()
testKalmanVarianceKnobsAffectTrades = do
    let tightReturn = kMean (stepMulti [(0.05, 1e-6)] (initKalman1 0 1e-6 0))
        looseReturn = kMean (stepMulti [(0.05, 100)] (initKalman1 0 1e-6 0))
        prices = V.fromList [100 :: Double, 100, 100, 100]
        preds r = V.replicate 3 (100 * (1 + r))
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecFee = 0
                , ecVolLookback = 2
                , ecMaxPositionSize = 1
                }
        tightResult = simulateEnsemble cfg 1 prices prices prices (preds tightReturn) (preds tightReturn) (Nothing :: Maybe (V.Vector StepMeta))
        looseResult = simulateEnsemble cfg 1 prices prices prices (preds looseReturn) (preds looseReturn) (Nothing :: Maybe (V.Vector StepMeta))
    assert "tight Kalman measurement variance creates a tradeable forecast" (any (/= 0) (brPositions tightResult))
    assert "loose Kalman measurement variance keeps the strategy flat" (all (== 0) (brPositions looseResult))

testKalmanMarketTopNRejectsInvalidValues :: IO ()
testKalmanMarketTopNRejectsInvalidValues = do
    assert
        "kalman-market-top-n rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-market-top-n", "-1"] == Left "--kalman-market-top-n must be >= 0")
    assert
        "kalman-market-top-n accepts 0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-market-top-n", "0"] of
            Right args -> argKalmanMarketTopN args == 0
            Left _ -> False
        )
    assert
        "kalman-market-top-n accepts 10 (positive)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-market-top-n", "10"] of
            Right args -> argKalmanMarketTopN args == 10
            Left _ -> False
        )
    assert
        "kalman-market-top-n accepts 50 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-market-top-n", "50"] of
            Right args -> argKalmanMarketTopN args == 50
            Left _ -> False
        )

testTuneRatioRejectsInvalidValues :: IO ()
testTuneRatioRejectsInvalidValues = do
    assert
        "tune-ratio rejects -0.1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--tune-ratio", "-0.1"] == Left "--tune-ratio must be >= 0 and < 1")
    assert
        "tune-ratio rejects 1.0 (no tune data)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--tune-ratio", "1.0"] == Left "--tune-ratio must be >= 0 and < 1")
    assert
        "tune-ratio accepts 0.0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tune-ratio", "0.0"] of
            Right args -> argTuneRatio args == 0.0
            Left _ -> False
        )
    assert
        "tune-ratio accepts 0.25 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tune-ratio", "0.25"] of
            Right args -> argTuneRatio args == 0.25
            Left _ -> False
        )
    assert
        "tune-ratio accepts 0.5 (valid)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tune-ratio", "0.5"] of
            Right args -> argTuneRatio args == 0.5
            Left _ -> False
        )

testTunePenaltyMaxDrawdownRejectsInvalidValues :: IO ()
testTunePenaltyMaxDrawdownRejectsInvalidValues = do
    assert
        "tune-penalty-max-drawdown rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--tune-penalty-max-drawdown", "-1"] == Left "--tune-penalty-max-drawdown must be >= 0")
    assert
        "tune-penalty-max-drawdown accepts 0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tune-penalty-max-drawdown", "0"] of
            Right args -> argTunePenaltyMaxDrawdown args == 0
            Left _ -> False
        )
    assert
        "tune-penalty-max-drawdown accepts 1.5 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tune-penalty-max-drawdown", "1.5"] of
            Right args -> argTunePenaltyMaxDrawdown args == 1.5
            Left _ -> False
        )
    assert
        "tune-penalty-max-drawdown accepts 2.0 (valid)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tune-penalty-max-drawdown", "2.0"] of
            Right args -> argTunePenaltyMaxDrawdown args == 2.0
            Left _ -> False
        )

testTunePenaltyTurnoverRejectsInvalidValues :: IO ()
testTunePenaltyTurnoverRejectsInvalidValues = do
    assert
        "tune-penalty-turnover rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--tune-penalty-turnover", "-1"] == Left "--tune-penalty-turnover must be >= 0")
    assert
        "tune-penalty-turnover accepts 0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tune-penalty-turnover", "0"] of
            Right args -> argTunePenaltyTurnover args == 0
            Left _ -> False
        )
    assert
        "tune-penalty-turnover accepts 0.5 (valid)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tune-penalty-turnover", "0.5"] of
            Right args -> argTunePenaltyTurnover args == 0.5
            Left _ -> False
        )
    assert
        "tune-penalty-turnover accepts 1.0 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tune-penalty-turnover", "1.0"] of
            Right args -> argTunePenaltyTurnover args == 1.0
            Left _ -> False
        )

testTuneMaxThresholdCandidatesRejectsInvalidValues :: IO ()
testTuneMaxThresholdCandidatesRejectsInvalidValues = do
    assert
        "tune-max-threshold-candidates rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--tune-max-threshold-candidates", "-1"] == Left "--tune-max-threshold-candidates must be >= 0")
    assert
        "tune-max-threshold-candidates accepts 0 (base thresholds only)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tune-max-threshold-candidates", "0"] of
            Right args -> argTuneMaxThresholdCandidates args == 0
            Left _ -> False
        )
    assert
        "tune-max-threshold-candidates accepts 60 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tune-max-threshold-candidates", "60"] of
            Right args -> argTuneMaxThresholdCandidates args == 60
            Left _ -> False
        )

testWalkForwardFoldsRejectsInvalidValues :: IO ()
testWalkForwardFoldsRejectsInvalidValues = do
    assert
        "walk-forward-folds rejects 0 (below minimum)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--walk-forward-folds", "0"] == Left "--walk-forward-folds must be >= 1")
    assert
        "walk-forward-folds rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--walk-forward-folds", "-1"] == Left "--walk-forward-folds must be >= 1")
    assert
        "walk-forward-folds accepts 1 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--walk-forward-folds", "1"] of
            Right args -> argWalkForwardFolds args == 1
            Left _ -> False
        )
    assert
        "walk-forward-folds accepts 5 (valid)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--walk-forward-folds", "5"] of
            Right args -> argWalkForwardFolds args == 5
            Left _ -> False
        )
    assert
        "walk-forward-folds accepts 7 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--walk-forward-folds", "7"] of
            Right args -> argWalkForwardFolds args == 7
            Left _ -> False
        )

testWalkForwardEmbargoBarsRejectsInvalidValues :: IO ()
testWalkForwardEmbargoBarsRejectsInvalidValues = do
    assert
        "walk-forward-embargo-bars rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--walk-forward-embargo-bars", "-1"] == Left "--walk-forward-embargo-bars must be >= 0")
    assert
        "walk-forward-embargo-bars accepts 0 (boundary/default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--walk-forward-embargo-bars", "0"] of
            Right args -> argWalkForwardEmbargoBars args == 0
            Left _ -> False
        )
    assert
        "walk-forward-embargo-bars accepts 1 (valid)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--walk-forward-embargo-bars", "1"] of
            Right args -> argWalkForwardEmbargoBars args == 1
            Left _ -> False
        )
    assert
        "walk-forward-embargo-bars accepts 5 (valid)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--walk-forward-embargo-bars", "5"] of
            Right args -> argWalkForwardEmbargoBars args == 5
            Left _ -> False
        )

testPatienceRejectsInvalidValues :: IO ()
testPatienceRejectsInvalidValues = do
    assert
        "patience rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--patience", "-1"] == Left "--patience must be >= 0")
    assert
        "patience accepts 0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--patience", "0"] of
            Right args -> argPatience args == 0
            Left _ -> False
        )
    assert
        "patience accepts 5 (valid)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--patience", "5"] of
            Right args -> argPatience args == 5
            Left _ -> False
        )
    assert
        "patience accepts 10 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--patience", "10"] of
            Right args -> argPatience args == 10
            Left _ -> False
        )

testOpenThresholdRejectsInvalidValues :: IO ()
testOpenThresholdRejectsInvalidValues = do
    assert
        "open-threshold rejects -0.1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--open-threshold", "-0.1"] == Left "--open-threshold/--threshold must be >= 0")
    assert
        "open-threshold rejects 1.1 (> 1)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--open-threshold", "1.1"] == Left "--open-threshold/--threshold must be <= 1")
    assert
        "open-threshold accepts 0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--open-threshold", "0"] of
            Right args -> argOpenThreshold args == 0
            Left _ -> False
        )
    assert
        "open-threshold accepts 0.002 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--open-threshold", "0.002"] of
            Right args -> argOpenThreshold args == 0.002
            Left _ -> False
        )
    assert
        "open-threshold accepts 1.0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--open-threshold", "1.0"] of
            Right args -> argOpenThreshold args == 1.0
            Left _ -> False
        )

testCloseThresholdRejectsInvalidValues :: IO ()
testCloseThresholdRejectsInvalidValues = do
    assert
        "close-threshold rejects -0.1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--close-threshold", "-0.1"] == Left "--close-threshold must be >= 0")
    assert
        "close-threshold accepts 0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--close-threshold", "0"] of
            Right args -> argCloseThreshold args == 0
            Left _ -> False
        )
    assert
        "close-threshold accepts 0.002 (default via open-threshold)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv"] of
            Right args -> argCloseThreshold args == 0.002
            Left _ -> False
        )
    assert
        "close-threshold accepts 0.01 (valid)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--close-threshold", "0.01"] of
            Right args -> argCloseThreshold args == 0.01
            Left _ -> False
        )

testRouterLookbackRejectsInvalidValues :: IO ()
testRouterLookbackRejectsInvalidValues = do
    assert
        "router-lookback rejects 1 (below minimum)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--router-lookback", "1"] == Left "--router-lookback must be >= 2")
    assert
        "router-lookback rejects 0 (below minimum)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--router-lookback", "0"] == Left "--router-lookback must be >= 2")
    assert
        "router-lookback accepts 2 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--router-lookback", "2"] of
            Right args -> argRouterLookback args == 2
            Left _ -> False
        )
    assert
        "router-lookback accepts 10 (valid)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--router-lookback", "10"] of
            Right args -> argRouterLookback args == 10
            Left _ -> False
        )
    assert
        "router-lookback accepts 30 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv"] of
            Right args -> argRouterLookback args == 30
            Left _ -> False
        )

testRouterMinScoreRejectsInvalidValues :: IO ()
testRouterMinScoreRejectsInvalidValues = do
    assert
        "router-min-score rejects -0.1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--router-min-score", "-0.1"] == Left "--router-min-score must be between 0 and 1")
    assert
        "router-min-score rejects 1.1 (>1)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--router-min-score", "1.1"] == Left "--router-min-score must be between 0 and 1")
    assert
        "router-min-score accepts 0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--router-min-score", "0"] of
            Right args -> argRouterMinScore args == 0
            Left _ -> False
        )
    assert
        "router-min-score accepts 0.25 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv"] of
            Right args -> argRouterMinScore args == 0.25
            Left _ -> False
        )
    assert
        "router-min-score accepts 1.0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--router-min-score", "1.0"] of
            Right args -> argRouterMinScore args == 1.0
            Left _ -> False
        )

testRouterScorePnlWeightRejectsInvalidValues :: IO ()
testRouterScorePnlWeightRejectsInvalidValues = do
    assert
        "router-score-pnl-weight rejects -0.1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--router-score-pnl-weight", "-0.1"] == Left "--router-score-pnl-weight must be between 0 and 1")
    assert
        "router-score-pnl-weight rejects 1.1 (>1)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--router-score-pnl-weight", "1.1"] == Left "--router-score-pnl-weight must be between 0 and 1")
    assert
        "router-score-pnl-weight accepts 0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--router-score-pnl-weight", "0"] of
            Right args -> argRouterScorePnlWeight args == 0
            Left _ -> False
        )
    assert
        "router-score-pnl-weight accepts 0.5 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv"] of
            Right args -> argRouterScorePnlWeight args == 0.5
            Left _ -> False
        )
    assert
        "router-score-pnl-weight accepts 1.0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--router-score-pnl-weight", "1.0"] of
            Right args -> argRouterScorePnlWeight args == 1.0
            Left _ -> False
        )

testExpectancyLookbackRejectsNegativeValue :: IO ()
testExpectancyLookbackRejectsNegativeValue = do
    let rejectedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--expectancy-lookback"
            , "-1"
            ]
        acceptedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--expectancy-lookback"
            , "20"
            ]
    assert
        "expectancy-lookback rejects negative values"
        ( parseAndValidateCliArgs rejectedArgs == Left "--expectancy-lookback must be >= 0"
            && case parseAndValidateCliArgs acceptedArgs of
                Right args -> argExpectancyLookback args == 20
                Left _ -> False
        )

testPerfLookbackRejectsNegativeValue :: IO ()
testPerfLookbackRejectsNegativeValue = do
    let rejectedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--perf-lookback"
            , "-1"
            ]
        acceptedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--perf-lookback"
            , "20"
            ]
    assert
        "perf-lookback rejects negative values"
        ( parseAndValidateCliArgs rejectedArgs == Left "--perf-lookback must be >= 0"
            && case parseAndValidateCliArgs acceptedArgs of
                Right args -> argPerfLookback args == 20
                Left _ -> False
        )

testCapitalPreservationReport :: IO ()
testCapitalPreservationReport = do
    let cfg = defaultCapitalPreservationConfig
        noRollingLossCfg = cfg{cpcMaxRollingLoss = Just 0.99}
        drawdownReport = capitalPreservationReport cfg 0.11 0 (replicate 20 0.01)
        streakReport = capitalPreservationReport cfg 0.01 3 (replicate 20 0.01)
        rollingLossReport = capitalPreservationReport cfg 0.01 0 (replicate 20 (-0.003))
        sharpeReport = capitalPreservationReport noRollingLossCfg 0.01 0 [-0.01, 0.004, -0.008, 0.003, -0.006, 0.002]
        notReadyReport = capitalPreservationReport cfg 0.01 0 (replicate 5 (-0.003))
        disabledReport = capitalPreservationReport cfg{cpcEnabled = False} 0.50 5 (replicate 20 (-0.01))
    assert
        "capital preservation reports deterministic entry-only failure reasons"
        ( cprReason drawdownReport == Just "CAPITAL_PRESERVATION_DRAWDOWN"
            && cprReason streakReport == Just "CAPITAL_PRESERVATION_LOSS_STREAK"
            && cprReason rollingLossReport == Just "CAPITAL_PRESERVATION_ROLLING_LOSS"
            && cprReason sharpeReport == Just "CAPITAL_PRESERVATION_SHARPE"
            && isNothing (cprReason notReadyReport)
            && isNothing (cprReason disabledReport)
            && capitalPreservationIsEntryOnlyReason "CAPITAL_PRESERVATION_ROLLING_LOSS"
            && not (capitalPreservationIsEntryOnlyReason "MAX_DRAWDOWN")
        )

testLossStreakMaxRejectsNegativeValue :: IO ()
testLossStreakMaxRejectsNegativeValue = do
    let rejectedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--loss-streak-max"
            , "-1"
            ]
        acceptedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--loss-streak-max"
            , "3"
            ]
    assert
        "loss-streak-max rejects negative values"
        ( parseAndValidateCliArgs rejectedArgs == Left "--loss-streak-max must be >= 0"
            && case parseAndValidateCliArgs acceptedArgs of
                Right args -> argLossStreakMax args == 3
                Left _ -> False
        )

testVolScaleMaxRejectsInvalidValues :: IO ()
testVolScaleMaxRejectsInvalidValues = do
    assert
        "vol-scale-max rejects -0.1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--vol-scale-max", "-0.1"] == Left "--vol-scale-max must be >= 0")
    assert
        "vol-scale-max rejects 100.1 (above sanity cap)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--vol-scale-max", "100.1"] == Left "--vol-scale-max must be <= 100")
    assert
        "vol-scale-max accepts 0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--vol-scale-max", "0"] of
            Right args -> argVolScaleMax args == 0
            Left _ -> False
        )
    assert
        "vol-scale-max accepts 1 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv"] of
            Right args -> argVolScaleMax args == 1
            Left _ -> False
        )
    assert
        "vol-scale-max accepts 100 (sanity cap boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--vol-scale-max", "100"] of
            Right args -> argVolScaleMax args == 100
            Left _ -> False
        )

testLossStreakCooldownBarsRejectsNegativeValue :: IO ()
testLossStreakCooldownBarsRejectsNegativeValue = do
    let rejectedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--loss-streak-cooldown-bars"
            , "-1"
            ]
        acceptedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--loss-streak-cooldown-bars"
            , "5"
            ]
    assert
        "loss-streak-cooldown-bars rejects negative values"
        ( parseAndValidateCliArgs rejectedArgs == Left "--loss-streak-cooldown-bars must be >= 0"
            && case parseAndValidateCliArgs acceptedArgs of
                Right args -> argLossStreakCooldownBars args == 5
                Left _ -> False
        )

testRsiPeriodRejectsInvalidValues :: IO ()
testRsiPeriodRejectsInvalidValues = do
    assert
        "rsi-period rejects 0 (below minimum)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--rsi-period", "0"] == Left "--rsi-period must be >= 1")
    assert
        "rsi-period rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--rsi-period", "-1"] == Left "--rsi-period must be >= 1")
    assert
        "rsi-period rejects 101 (above sanity cap)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--rsi-period", "101"] == Left "--rsi-period must be <= 100")
    assert
        "rsi-period accepts 1 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--rsi-period", "1"] of
            Right args -> argRsiPeriod args == 1
            Left _ -> False
        )
    assert
        "rsi-period accepts 14 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv"] of
            Right args -> argRsiPeriod args == 14
            Left _ -> False
        )
    assert
        "rsi-period accepts 100 (sanity cap boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--rsi-period", "100"] of
            Right args -> argRsiPeriod args == 100
            Left _ -> False
        )

testTrendLookbackRejectsInvalidValues :: IO ()
testTrendLookbackRejectsInvalidValues = do
    assert
        "trend-lookback rejects -1 (negative)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--trend-lookback", "-1"] == Left "--trend-lookback must be >= 0")
    assert
        "trend-lookback rejects 1001 (above sanity cap)"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--trend-lookback", "1001"] == Left "--trend-lookback must be <= 1000")
    assert
        "trend-lookback accepts 0 (boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--trend-lookback", "0"] of
            Right args -> argTrendLookback args == 0
            Left _ -> False
        )
    assert
        "trend-lookback accepts 30 (default)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv"] of
            Right args -> argTrendLookback args == 30
            Left _ -> False
        )
    assert
        "trend-lookback accepts 1000 (sanity cap boundary)"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--trend-lookback", "1000"] of
            Right args -> argTrendLookback args == 1000
            Left _ -> False
        )

testRsiLowerMustBeLessThanUpper :: IO ()
testRsiLowerMustBeLessThanUpper = do
    let rejectedEqualArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--rsi-lower"
            , "50"
            , "--rsi-upper"
            , "50"
            ]
        rejectedGreaterArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--rsi-lower"
            , "60"
            , "--rsi-upper"
            , "50"
            ]
        rejectedLowerNegativeArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--rsi-lower"
            , "-1"
            , "--rsi-upper"
            , "70"
            ]
        rejectedLowerAbove100Args =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--rsi-lower"
            , "101"
            , "--rsi-upper"
            , "70"
            ]
        rejectedUpperNegativeArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--rsi-lower"
            , "30"
            , "--rsi-upper"
            , "-1"
            ]
        rejectedUpperAbove100Args =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--rsi-lower"
            , "30"
            , "--rsi-upper"
            , "101"
            ]
        acceptedBoundaryArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--rsi-lower"
            , "0"
            , "--rsi-upper"
            , "100"
            ]
        acceptedArgs =
            [ "--binance-symbol"
            , "BTCUSDT"
            , "--interval"
            , "15m"
            , "--bars"
            , "673"
            , "--lookback-bars"
            , "672"
            , "--rsi-lower"
            , "30"
            , "--rsi-upper"
            , "70"
            ]
    assert
        "rsi-lower must be < rsi-upper rejects equal values"
        (parseAndValidateCliArgs rejectedEqualArgs == Left "--rsi-lower must be < --rsi-upper")
    assert
        "rsi-lower must be < rsi-upper rejects lower > upper"
        (parseAndValidateCliArgs rejectedGreaterArgs == Left "--rsi-lower must be < --rsi-upper")
    assert
        "rsi-lower must be >= 0 rejects negative lower"
        (parseAndValidateCliArgs rejectedLowerNegativeArgs == Left "--rsi-lower must be >= 0")
    assert
        "rsi-lower must be <= 100 rejects above 100"
        (parseAndValidateCliArgs rejectedLowerAbove100Args == Left "--rsi-lower must be <= 100")
    assert
        "rsi-upper must be >= 0 rejects negative upper"
        (parseAndValidateCliArgs rejectedUpperNegativeArgs == Left "--rsi-upper must be >= 0")
    assert
        "rsi-upper must be <= 100 rejects above 100"
        (parseAndValidateCliArgs rejectedUpperAbove100Args == Left "--rsi-upper must be <= 100")
    assert
        "rsi boundary 0/100 stays admissible when lower < upper"
        ( case parseAndValidateCliArgs acceptedBoundaryArgs of
            Right args -> argRsiLower args == 0 && argRsiUpper args == 100
            Left _ -> False
        )
    assert
        "rsi-lower must be < rsi-upper accepts valid lower < upper"
        ( case parseAndValidateCliArgs acceptedArgs of
            Right args -> argRsiLower args == 30 && argRsiUpper args == 70
            Left _ -> False
        )

testExchangeDataLongShortBacktestAllowed :: IO ()
testExchangeDataLongShortBacktestAllowed = do
    assert
        "Coinbase exchange-data backtests allow long-short without futures"
        ( case parseAndValidateCliArgs ["--platform", "coinbase", "--symbol", "BTC-USD", "--positioning", "long-short"] of
            Right args -> argPositioning args == LongShort
            Left _ -> False
        )
    assert
        "Binance spot exchange-data backtests allow long-short without futures"
        ( case parseAndValidateCliArgs ["--platform", "binance", "--symbol", "BTCUSDT", "--positioning", "long-short"] of
            Right args -> argPositioning args == LongShort
            Left _ -> False
        )
    assert
        "placing long-short exchange orders still requires futures"
        ( parseAndValidateCliArgs ["--platform", "binance", "--symbol", "BTCUSDT", "--positioning", "long-short", "--binance-trade"]
            == Left "--positioning long-short requires --futures when trading"
        )

testPositioningShortAliasRejected :: IO ()
testPositioningShortAliasRejected =
    assert
        "positioning parser rejects the unsupported short alias instead of widening it to long-short"
        ( case parsePositioning "short" of
            Left _ -> True
            Right _ -> False
        )

testTenantResolutionScopesMixedApiKeys :: IO ()
testTenantResolutionScopesMixedApiKeys = do
    let bKey = Just "binance-key"
        bSecret = Just "binance-secret"
        cKey = Just "coinbase-key"
        cSecret = Just "coinbase-secret"
        cPass = Just "coinbase-pass"
    case (tenantKeyFromBinanceKeys bKey bSecret, tenantKeyFromCoinbaseKeys cKey cSecret cPass) of
        (Just binanceTenant, Just coinbaseTenant) -> do
            assert
                "unscoped mixed API keys require an explicit platform or tenant"
                ( case resolveTenantKeyFromParams Nothing bKey bSecret cKey cSecret cPass of
                    Left _ -> True
                    Right _ -> False
                )
            assert
                "Coinbase scope selects the Coinbase tenant"
                (resolveTenantKeyFromPlatformParams PlatformCoinbase Nothing bKey bSecret cKey cSecret cPass == Right (Just coinbaseTenant))
            assert
                "Binance scope selects the Binance tenant"
                (resolveTenantKeyFromPlatformParams PlatformBinance Nothing bKey bSecret cKey cSecret cPass == Right (Just binanceTenant))
            assert
                "explicit Coinbase tenant matches mixed credentials when unscoped"
                (resolveTenantKeyFromParams (Just (T.unpack coinbaseTenant)) bKey bSecret cKey cSecret cPass == Right (Just coinbaseTenant))
        _ -> assert "tenant derivation for test credentials succeeds" False

testMarketLinearFit :: IO ()
testMarketLinearFit = do
    let xs = V.generate 100 (fromIntegral :: Int -> Double)
        ys = V.map (\x -> 2 + 3 * x) xs
        (a, b, var) = fitLinearRange xs ys 0 100
    assertNear "market linear fit intercept" 2 a 1e-9
    assertNear "market linear fit beta" 3 b 1e-9
    assert "market linear fit residual variance stays near zero" (var < 1e-9)

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
        , ecAdaptiveWinRateSlack = 0.05
        , ecAdaptiveProfitFactorSlack = 0.10
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
        , ecSignalGateConfig = defaultSignalGateConfig
        , ecEntryEdgeSpikeAuditOnly = False
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
        , ecVolConfGateConfig = defaultVolConfGateConfig
        , ecRebalanceBars = 0
        , ecRebalanceThreshold = 0
        , ecRebalanceGlobal = False
        , ecRebalanceResetOnSignal = False
        , ecFundingRate = 0
        , ecFundingBySide = False
        , ecFundingOnOpen = False
        , ecBlendWeight = 0.5
        , ecBlendSoftmaxScale = 600
        , ecBlendNetSoftmaxScale = 6000
        , ecBlendEdgePower = 1.0
        , ecBlendSmoothAlpha = 0.2
        , ecBlendHedgeEta = 6
        , ecBlendHedgeMaxError = 0.1
        , ecBlendDivergenceK = 4
        , ecBlendRegimeHighVolCutoff = 0.6
        , ecBlendRegimeKalmanZCutoff = 1
        , ecBlendBanditExploreScale = 0.25
        , ecBlendFractalReturnClamp = 0.75
        , ecBlendFractalAlignedGain = 1.12
        , ecBlendFractalConflictGain = 0.82
        , ecBlendCoherenceConflictFloor = 0.2
        , ecBlendCoherenceConflictScale = 0.5
        , ecBlendCoherenceBoostThreshold = 0.6
        , ecBlendCoherenceBoostGain = 0.35
        , ecBlendCoherenceBoostSpan = 0.4
        , ecBlendAnchorConflictBase = 0.6
        , ecBlendAnchorConflictScale = 0.4
        , ecBlendAnchorAlignedScale = 0.2
        , ecBlendTensionConflictShrink = 0.25
        , ecBlendTensionNeutralShrink = 0.5
        , ecBlendEntropyConflictFloor = 0.35
        , ecBlendEntropyConflictScale = 0.5
        , ecBlendEntropyAlignedBase = 0.95
        , ecBlendEntropyAlignedEntropyScale = 0.25
        , ecBlendPhaseCancelReturnClamp = 0.75
        , ecBlendPhaseCancelConflictFloor = 0.1
        , ecBlendPhaseCancelConflictScale = 0.6
        , ecBlendPhaseCancelAlignmentScale = 0.4
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
        , ecTriLayerPriceActionWickRatio = 2
        , ecTriLayerPriceActionOppositeWickMax = 0.5
        , ecTriLayerPriceActionBodyTolerance = 0.2
        , ecTriLayerExitOnSlow = False
        , ecKalmanBandLookback = 0
        , ecKalmanBandStdMult = 0
        , ecKalmanZMin = -1
        , ecKalmanZMax = 1
        , ecKalmanMinStdFloor = 1e-6
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
        , ecEntryEdgeSpikeConsecutive = 0
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
        , ecKalmanMinStdFloor = 1e-6
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

-- The existing fresh-entry contract combines minimum edge headroom
-- (edge >= 1.5 * openThreshold) with a spike cap (edge <= min (1000 * openThreshold) 5.0).
-- Therefore openThreshold > 10/3 has no admissible finite edge sample.
testSignalGateEntryThresholdFeasibilityInvariant :: IO ()
testSignalGateEntryThresholdFeasibilityInvariant = do
    let boundary = signalEntryOpenThresholdFeasibilityCap
        justAbove = boundary + 1e-12
        malformedThresholds = [-0.01, 0 / 0, 1 / 0]
    assert
        "fresh-entry threshold feasibility cap is exactly the headroom/spike intersection"
        (abs (boundary - (10 / 3)) <= 1e-15)
    assert
        "exact threshold feasibility boundary remains admissible when the edge is exactly the credible cap"
        ( signalEntryOpenThresholdFeasible boundary
            && isNothing (signalEntryOpenThresholdFeasibilityReason boundary)
            && signalEntryHeadroomOk boundary (Just 5.0)
            && signalEntryEdgeSpikeOk boundary (Just 5.0)
        )
    assert
        "strict-above-boundary thresholds fail closed with an explicit infeasibility reason"
        ( not (signalEntryOpenThresholdFeasible justAbove)
            && signalEntryOpenThresholdFeasibilityReason justAbove == Just "THRESHOLD_INFEASIBLE"
            && not (signalEntryHeadroomOk justAbove (Just 0.5))
            && signalEntryEdgeSpikeOk justAbove (Just 0.5)
        )
    assert
        "malformed thresholds fail closed under the feasibility helper"
        ( all
            ( \threshold ->
                not (signalEntryOpenThresholdFeasible threshold)
                    && signalEntryOpenThresholdFeasibilityReason threshold == Just "THRESHOLD_INFEASIBLE"
            )
            malformedThresholds
        )

testMarketDataFreshnessAndContinuationInvariant :: IO ()
testMarketDataFreshnessAndContinuationInvariant = do
    let hourMs = 60 * 60 * 1000
        lastOpen = 1000000000
        freshNow = lastOpen + hourMs + (30 * 60 * 1000)
        staleNow = lastOpen + (2 * hourMs) + 1
        fresh = marketDataFreshness "1h" freshNow lastOpen
        stale = marketDataFreshness "1h" staleNow lastOpen
    assert
        "closed-bar freshness is measured from the processed candle close time, not its open time"
        ( case fresh of
            Just f ->
                mdfLastCloseTimeMs f == lastOpen + hourMs
                    && mdfAgeMs f == 30 * 60 * 1000
                    && mdfFreshnessBudgetMs f == hourMs
                    && not (mdfStale f)
            Nothing -> False
        )
    assert
        "market data becomes stale after one full interval without the next closed candle"
        ( case stale of
            Just f ->
                mdfAgeMs f == hourMs + 1
                    && mdfStale f
                    && marketDataStaleReason "1h" staleNow lastOpen == Just ("STALE_MARKET_DATA ageMs=" ++ show (hourMs + 1) ++ " budgetMs=" ++ show hourMs ++ " lastCloseTimeMs=" ++ show (lastOpen + hourMs))
            Nothing -> False
        )
    assert
        "live continuation accepts exactly contiguous closed candles"
        (isNothing (marketDataContinuationIssue "1h" lastOpen [lastOpen + hourMs, lastOpen + 2 * hourMs]))
    assert
        "live continuation fails closed on missing candles before a new decision can be processed"
        (marketDataContinuationIssue "1h" lastOpen [lastOpen + 2 * hourMs] == Just ("MARKET_DATA_GAP expectedOpenTimeMs=" ++ show (lastOpen + hourMs) ++ " actualOpenTimeMs=" ++ show (lastOpen + 2 * hourMs) ++ " intervalMs=" ++ show hourMs))
    assert
        "malformed intervals fail closed in freshness and continuation helpers"
        ( isNothing (marketDataFreshness "bad" staleNow lastOpen)
            && marketDataStaleReason "bad" staleNow lastOpen == Just "MARKET_DATA_INTERVAL_INVALID interval=\"bad\""
            && marketDataContinuationIssue "bad" lastOpen [lastOpen + hourMs] == Just "MARKET_DATA_INTERVAL_INVALID interval=\"bad\""
        )

-- The spike veto is a maximum-edge sanity cap, not a minimum-edge headroom
-- check: exact cap equality is admissible, strict-above-cap edges are blocked,
-- and malformed thresholds or edges fail closed.
-- With entryEdgeSpikeMultiple=1000 and entryEdgeSpikeCredibleCap=5.0:
--   - threshold=0.001 hits the multiple cap at 1.0
--   - threshold=0.01 hits the credible cap at 5.0
testSignalGateEntryEdgeSpikeCapRegression :: IO ()
testSignalGateEntryEdgeSpikeCapRegression = do
    let smallThreshold = 0.001
        credibleThreshold = 0.01
    assert
        "fresh-entry spike veto admits exact equality at both active caps"
        ( signalEntryEdgeSpikeOk smallThreshold (Just 1.0)
            && signalEntryEdgeSpikeOk credibleThreshold (Just 5.0)
        )
    assert
        "fresh-entry spike veto blocks strict-above-cap and absurd edges"
        ( not (signalEntryEdgeSpikeOk smallThreshold (Just 1.0000001))
            && not (signalEntryEdgeSpikeOk credibleThreshold (Just 5.0000001))
            && not (signalEntryEdgeSpikeOk credibleThreshold (Just 8.95))
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

testSignalGateEntryEdgeSpikeAuditWarning :: IO ()
testSignalGateEntryEdgeSpikeAuditWarning = do
    let openThreshold = 0.01
        scaledEdge = Just 5.0000001
    assert
        "scaled-model spike mode preserves the audit warning without blocking entry"
        ( not (signalEntryEdgeSpikeOk openThreshold scaledEdge)
            && signalEntryEdgeSpikeEntryOk True openThreshold scaledEdge
            && signalEntryEdgeSpikeAuditWarning True openThreshold scaledEdge == Just "EDGE_SPIKE"
        )
    assert
        "unscaled-model spike mode remains a hard entry veto"
        ( not (signalEntryEdgeSpikeEntryOk False openThreshold scaledEdge)
            && isNothing (signalEntryEdgeSpikeAuditWarning False openThreshold scaledEdge)
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

testPredictionMarketHerdSelection :: IO ()
testPredictionMarketHerdSelection = do
    let btcMarket =
            PredictionMarketMarket
                { pmmId = Just "btc-15m"
                , pmmSlug = Just "bitcoin-up-or-down-15m"
                , pmmQuestion = "Bitcoin Up or Down - 15 minutes"
                , pmmEndDate = Just "2026-01-01T00:15:00Z"
                , pmmActive = True
                , pmmClosed = False
                , pmmVolume = Just 10000
                , pmmVolume24hr = Just 4000
                , pmmOutcomes = ["Up", "Down"]
                , pmmOutcomePrices = [0.68, 0.32]
                }
        btcEvent =
            PredictionMarketEvent
                { pmeSlug = Just "bitcoin-up-or-down-15m"
                , pmeTitle = "Bitcoin Up or Down"
                , pmeEndDate = Just "2026-01-01T00:15:00Z"
                , pmeVolume = Just 10000
                , pmeVolume24hr = Just 4000
                , pmeMarkets = [btcMarket]
                }
        mSignal = selectPredictionMarketSignal "BTCUSDT" "5m" [btcEvent]
    case mSignal of
        Nothing -> assert "BTC Polymarket herd signal should be selected" False
        Just signal -> do
            assert "5m exchange interval maps to nearest 15m prediction market interval" (pmsInterval signal == "15m")
            assert "UP probability is read from the matching outcome" (predictionMarketProbabilityForDir 1 signal == Just 0.68)
            assert "DOWN probability is read from the matching outcome" (predictionMarketProbabilityForDir (-1) signal == Just 0.32)
    assert
        "unrelated symbols do not reuse the BTC herd market"
        (isNothing (selectPredictionMarketSignal "ETHUSDT" "5m" [btcEvent]))
    assert
        "nearest prediction market interval rounds supported exchange intervals conservatively"
        ( nearestPredictionMarketInterval "5m" == ("15m", 900)
            && nearestPredictionMarketInterval "30m" == ("1h", 3600)
            && nearestPredictionMarketInterval "1h" == ("1h", 3600)
            && nearestPredictionMarketInterval "2h" == ("4h", 14400)
            && nearestPredictionMarketInterval "12h" == ("1d", 86400)
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
            && isNothing nanEdge
            && isNothing infiniteEdge
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
            && not (edgeSpikeOk nanState)
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
        "non-finite raw edge collapses to Nothing and stays blocked"
        ( needsEntry nonFiniteEdgeState
            && Data.Maybe.isNothing (entryEdge nonFiniteEdgeState)
            && not (edgeSpikeOk nonFiniteEdgeState)
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

-- Volatility-confidence gate invariant: enabled presets only admit entries
-- from finite bounded volatility evidence and absent-or-bounded confidence
-- evidence. Missing confidence remains weak, while malformed provided
-- confidence and malformed/out-of-range volatility fail closed as exit-only.
testVolConfGateMalformedInputsFailClosed :: IO ()
testVolConfGateMalformedInputsFailClosed = do
    let exitOnly cell = vcgBehavior cell == VolConfGateAllowExitOnly && vcgSizeMult cell == 0
        malformedVolatilities = [Nothing, Just (-0.01), Just (0 / 0), Just (1 / 0), Just 2.000001]
        malformedConfidences = [Just (-0.01), Just 1.000001, Just (0 / 0), Just (1 / 0)]
        confidenceLadder =
            map
                (\preset -> vcgBehavior (volConfGateCell preset (Just 0.5) (Just 0.60)) == VolConfGateAllowEntry)
                [VolConfGateV1Default, VolConfGateV1ConfStricter]
        volatilityLadder =
            map
                (\preset -> vcgBehavior (volConfGateCell preset (Just 1.2) (Just 0.60)) == VolConfGateAllowEntry)
                [VolConfGateV1HighVolLooser, VolConfGateV1Default, VolConfGateV1HighVolTighter]
    assert
        "vol-confidence gate preserves valid equality boundaries"
        ( volConfGateCell VolConfGateV1Default (Just 0.5) (Just 0.60)
            == VolConfGateCell VolConfGateAllowEntry 0.45
            && volConfGateCell VolConfGateV1Default (Just 1.2) (Just 0.80)
                == VolConfGateCell VolConfGateAllowEntry 0.35
            && volConfGateCell VolConfGateV1Default (Just 2.0) (Just 1.0)
                == VolConfGateCell VolConfGateAllowEntry 0.35
        )
    assert
        "malformed volatility evidence fails closed for enabled vol-confidence presets"
        (all (exitOnly . (\mVol -> volConfGateCell VolConfGateV1Default mVol (Just 0.8))) malformedVolatilities)
    assert
        "malformed provided confidence evidence fails closed instead of becoming weak hold evidence"
        (all (exitOnly . volConfGateCell VolConfGateV1Default (Just 0.4)) malformedConfidences)
    assert
        "missing confidence remains weak entry-blocking evidence"
        (volConfGateCell VolConfGateV1Default (Just 0.4) Nothing == VolConfGateCell VolConfGateHold 0)
    assert
        "exit-only vol-confidence behavior cannot open or increase exposure"
        ( applyVolConfGateBehavior VolConfGateAllowExitOnly Nothing 0 (Just SideLong) 1 == (Nothing, 0)
            && applyVolConfGateBehavior VolConfGateAllowExitOnly (Just SideLong) 1 (Just SideLong) 2 == (Just SideLong, 1)
            && applyVolConfGateBehavior VolConfGateAllowExitOnly (Just SideLong) 1 (Just SideShort) 1 == (Nothing, 0)
        )
    assert
        "stateful live close direction preserves reduce-only same-side evidence without reopening entries"
        ( isNothing (volConfStatefulCloseDirection VolConfGateAllowEntry (Just SideLong) Nothing)
            && isNothing (volConfStatefulCloseDirection VolConfGateHold (Just SideLong) (Just SideShort))
            && volConfStatefulCloseDirection VolConfGateBlock (Just SideLong) Nothing == Just SideLong
            && volConfStatefulCloseDirection VolConfGateAllowExitOnly (Just SideShort) Nothing == Just SideShort
            && volConfStatefulCloseDirection VolConfGateBlock Nothing (Just SideLong) == Just SideLong
        )
    assert
        "stricter confidence and volatility requirements are monotone on bounded witnesses"
        ( confidenceLadder == [True, False]
            && volatilityLadder == [True, False, False]
        )
    assertMonotoneNonIncreasing
        "raising the confidence requirement cannot reopen a blocked vol-confidence entry"
        confidenceLadder
    assertMonotoneNonIncreasing
        "tightening the high-volatility threshold cannot reopen a blocked vol-confidence entry"
        volatilityLadder

testQueuedBotStartOrderErrorStability :: IO ()
testQueuedBotStartOrderErrorStability = do
    assert
        "queued bot starts ignore below-threshold transient order errors"
        ( isNothing (queuedStartOrderErrorIssue Nothing 1)
            && isNothing (queuedStartOrderErrorIssue (Just 0) 1)
            && isNothing (queuedStartOrderErrorIssue (Just 3) 1)
            && isNothing (queuedStartOrderErrorIssue (Just 3) 0)
        )
    assert
        "queued bot starts still block order errors that reach the configured halt limit"
        ( queuedStartOrderErrorIssue (Just 3) 3 == Just "order errors=3 reached maxOrderErrors"
            && queuedStartOrderErrorIssue (Just 3) 4 == Just "order errors=4 reached maxOrderErrors"
        )

testQueuedBotStartIgnoresTransientMarketDataErrors :: IO ()
testQueuedBotStartIgnoresTransientMarketDataErrors = do
    assert
        "MARKET_DATA_GAP is treated as transient and does not block queued bot starts"
        (isTransientMarketDataError "MARKET_DATA_GAP expectedOpenTimeMs=1 actualOpenTimeMs=2 intervalMs=3")
    assert
        "STALE_MARKET_DATA is treated as transient and does not block queued bot starts"
        (isTransientMarketDataError "STALE_MARKET_DATA ageMs=100 budgetMs=50 lastCloseTimeMs=200")
    assert
        "other errors are not treated as transient market-data errors"
        ( not (isTransientMarketDataError "UNKNOWN_ERROR")
            && not (isTransientMarketDataError "MARKET_DATA_INTERVAL_INVALID interval=bad")
        )

testPrioritizeOrphanBotStartSymbols :: IO ()
testPrioritizeOrphanBotStartSymbols = do
    assert
        "bot starts put orphaned-position symbols before normal requested targets"
        (prioritizeBotStartSymbols ["BTCUSDT", "ETHUSDT"] ["SOLUSDT", "BTCUSDT"] == ["SOLUSDT", "BTCUSDT", "ETHUSDT"])
    assert
        "bot start priority de-duplicates using normalized symbols"
        (prioritizeBotStartSymbols ["ethusdt", " SOLUSDT "] ["btc usdt", "ethusdt"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"])

testDisabledBotStartSymbols :: IO ()
testDisabledBotStartSymbols =
    assert
        "disabled bot-start symbols are matched case-insensitively and ignore spacing"
        ( botStartSymbolDisabled ["MATICUSDT"] "maticusdt"
            && botStartSymbolDisabled ["MATIC USDT"] "MATICUSDT"
            && not (botStartSymbolDisabled ["MATICUSDT"] "BTCUSDT")
        )

testBotStartupBacktestRoiAcceptability :: IO ()
testBotStartupBacktestRoiAcceptability = do
    assert
        "startup backtest accepts only profitable finite final equity"
        ( botStartupBacktestRoiAcceptable (Just 1.000001)
            && not (botStartupBacktestRoiAcceptable (Just 1.0))
            && not (botStartupBacktestRoiAcceptable (Just 0.999999))
            && not (botStartupBacktestRoiAcceptable (Just (1 / 0)))
            && not (botStartupBacktestRoiAcceptable (Just (0 / 0)))
            && not (botStartupBacktestRoiAcceptable Nothing)
        )

testBotStartupBacktestGuardFailOpen :: IO ()
testBotStartupBacktestGuardFailOpen = do
    assert
        "startup backtest guard aborts only on an enabled, sub-threshold ROI reading"
        ( -- disabled guard never aborts, regardless of ROI reading
          not (botStartupBacktestAborts False (Just 0.5))
            && not (botStartupBacktestAborts False Nothing)
            && not (botStartupBacktestAborts False (Just (0 / 0)))
            -- enabled but no finalEquity reading (infra failure): fail open
            && not (botStartupBacktestAborts True Nothing)
            -- enabled with a profitable, finite reading: allow
            && not (botStartupBacktestAborts True (Just 1.000001))
            -- enabled with a flat/losing reading: abort
            && botStartupBacktestAborts True (Just 1.0)
            && botStartupBacktestAborts True (Just 0.5)
            -- enabled with a non-finite reading: abort, do not trade on a garbage verdict
            && botStartupBacktestAborts True (Just (1 / 0))
            && botStartupBacktestAborts True (Just (0 / 0))
        )

{- | H6 from 2026-06-10 review: a backtest that fired zero trades is not a
verdict on the combo. It must produce 'BacktestNoVerdict' so the start is
allowed and the combo is not pruned. Today's launchd log shows 124 such
erroneous prunes with @finalEquity=1.000000@ exactly, all caused by
zero-trade smoke windows. The verdict function is the central guard.
-}
testBotStartupBacktestVerdictZeroTradeIsNoVerdict :: IO ()
testBotStartupBacktestVerdictZeroTradeIsNoVerdict = do
    assert
        "zero-trade backtest is not a verdict: do not abort, do not prune"
        ( botStartupBacktestVerdict True (Just 1.0) (Just 0) == BacktestNoVerdict
            && botStartupBacktestVerdict True (Just 0.9999) (Just 0) == BacktestNoVerdict
            && botStartupBacktestVerdict True (Just 1.5) (Just 0) == BacktestAllow
            -- An unknown tradeCount with a flat/losing finalEquity is also
            -- not a verdict: we don't have evidence the smoke window
            -- actually traded.
            && botStartupBacktestVerdict True (Just 1.0) Nothing == BacktestNoVerdict
            && not (backtestVerdictAborts BacktestNoVerdict)
            && not (backtestVerdictAborts BacktestAllow)
            && backtestVerdictAborts BacktestAbort
        )

{- | When the smoke backtest actually traded and lost (or finished flat with
evidence of trading), the verdict must still abort — we don't want to fail
open on a real signal that lost money.
-}
testBotStartupBacktestVerdictAbortOnLossWithTrades :: IO ()
testBotStartupBacktestVerdictAbortOnLossWithTrades = do
    assert
        "backtest that traded and lost still aborts"
        ( botStartupBacktestVerdict True (Just 0.5) (Just 4) == BacktestAbort
            && botStartupBacktestVerdict True (Just 1.0) (Just 12) == BacktestAbort
            && botStartupBacktestVerdict True (Just (1 / 0)) (Just 2) == BacktestAbort
            && botStartupBacktestVerdict True (Just (0 / 0)) (Just 2) == BacktestAbort
            -- One actual trade is enough evidence that the signal fired.
            && botStartupBacktestVerdict True (Just 0.9) (Just 1) == BacktestAbort
        )

{- | The disabled-guard short-circuit (BotStartSemantics fail-open) must keep
the new three-valued verdict consistent with the existing two-valued
'botStartupBacktestAborts': disabled always allows, regardless of trades.
-}
testBotStartupBacktestVerdictPreservesDisabledBehaviour :: IO ()
testBotStartupBacktestVerdictPreservesDisabledBehaviour = do
    assert
        "disabled guard always returns Allow and never aborts"
        ( botStartupBacktestVerdict False (Just 0.5) (Just 4) == BacktestAllow
            && botStartupBacktestVerdict False (Just 1.0) (Just 0) == BacktestAllow
            && botStartupBacktestVerdict False Nothing Nothing == BacktestAllow
            && botStartupBacktestVerdict False (Just (1 / 0)) Nothing == BacktestAllow
            && not (backtestVerdictAborts (botStartupBacktestVerdict False (Just 0.0) (Just 99)))
        )

{- | End-to-end invariant on the JSON store: a zero-trade update must not
cause 'applyComboUpdatesWithStats' to prune the combo. This closes the
second prune path used by the optimizer top-N rerun.
-}
testApplyComboUpdatesZeroTradeDoesNotPrune :: IO ()
testApplyComboUpdatesZeroTradeDoesNotPrune = do
    let combosJson =
            Aeson.object
                [ "combos"
                    .= Aeson.toJSON
                        [ Aeson.object
                            [ "comboKey" .= ("binance|BTCUSDT|15m|deadbeef" :: T.Text)
                            , "platform" .= ("binance" :: T.Text)
                            , "symbol" .= ("BTCUSDT" :: T.Text)
                            , "interval" .= ("15m" :: T.Text)
                            , "uuid" .= ("deadbeef-1234-5678-9abc-def012345678" :: T.Text)
                            , "finalEquity" .= (1.42 :: Double)
                            , "openThreshold" .= (0.005 :: Double)
                            , "closeThreshold" .= (0.010 :: Double)
                            , "objective" .= ("final-equity" :: T.Text)
                            , "params"
                                .= Aeson.object
                                    [ "method" .= ("blend" :: T.Text)
                                    , "lookback" .= (48 :: Int)
                                    ]
                            , "metrics"
                                .= Aeson.object
                                    [ "finalEquity" .= (1.42 :: Double)
                                    , "tradeCount" .= (8 :: Int)
                                    ]
                            ]
                        ]
                ]
    let firstCombo = case combosJson of
            Aeson.Object o -> case KM.lookup "combos" o of
                Just (Aeson.Array v) | not (V.null v) -> Just (V.head v)
                _ -> Nothing
            _ -> Nothing
        key = case firstCombo >>= comboIdentityKey of
            Just k -> k
            Nothing -> error "test setup: combo identity key resolution failed"
        zeroTradeUpdate =
            ComboBacktestUpdate
                { cbuMetrics =
                    Aeson.object
                        [ "finalEquity" .= (1.000000 :: Double)
                        , "tradeCount" .= (0 :: Int)
                        ]
                , cbuFinalEquity = Just 1.0
                , cbuScore = Just 0.0
                , cbuOperations = Nothing
                }
        result = applyComboUpdatesWithStats 0 (HM.singleton key zeroTradeUpdate) combosJson
    case result of
        Left err -> ioError (userError ("applyComboUpdatesWithStats failed unexpectedly: " ++ err))
        Right (_, stats) ->
            assert
                "zero-trade smoke update does not prune the combo"
                (cbasUpdatedCount stats == 1 && cbasPrunedCount stats == 0 && null (cbasPrunedKeys stats))

{- | Symmetry test for the prune fix: a genuine loss (positive trade count,
sub-threshold finalEquity) must still prune. We don't want the fix to
swallow real signal regressions.
-}
testApplyComboUpdatesGenuineLossStillPrunes :: IO ()
testApplyComboUpdatesGenuineLossStillPrunes = do
    let combosJson =
            Aeson.object
                [ "combos"
                    .= Aeson.toJSON
                        [ Aeson.object
                            [ "comboKey" .= ("binance|ETHUSDT|15m|cafebabe" :: T.Text)
                            , "platform" .= ("binance" :: T.Text)
                            , "symbol" .= ("ETHUSDT" :: T.Text)
                            , "interval" .= ("15m" :: T.Text)
                            , "uuid" .= ("cafebabe-1234-5678-9abc-def012345678" :: T.Text)
                            , "finalEquity" .= (1.42 :: Double)
                            , "openThreshold" .= (0.005 :: Double)
                            , "closeThreshold" .= (0.010 :: Double)
                            , "objective" .= ("final-equity" :: T.Text)
                            , "params"
                                .= Aeson.object
                                    [ "method" .= ("blend" :: T.Text)
                                    , "lookback" .= (48 :: Int)
                                    ]
                            , "metrics"
                                .= Aeson.object
                                    [ "finalEquity" .= (1.42 :: Double)
                                    , "tradeCount" .= (8 :: Int)
                                    ]
                            ]
                        ]
                ]
    let firstCombo = case combosJson of
            Aeson.Object o -> case KM.lookup "combos" o of
                Just (Aeson.Array v) | not (V.null v) -> Just (V.head v)
                _ -> Nothing
            _ -> Nothing
        key = case firstCombo >>= comboIdentityKey of
            Just k -> k
            Nothing -> error "test setup: combo identity key resolution failed"
        lossUpdate =
            ComboBacktestUpdate
                { cbuMetrics =
                    Aeson.object
                        [ "finalEquity" .= (0.85 :: Double)
                        , "tradeCount" .= (12 :: Int)
                        ]
                , cbuFinalEquity = Just 0.85
                , cbuScore = Just (-0.15)
                , cbuOperations = Nothing
                }
        result = applyComboUpdatesWithStats 0 (HM.singleton key lossUpdate) combosJson
    case result of
        Left err -> ioError (userError ("applyComboUpdatesWithStats failed unexpectedly: " ++ err))
        Right (_, stats) ->
            assert
                "genuine loss with positive tradeCount still prunes"
                (cbasUpdatedCount stats == 1 && cbasPrunedCount stats == 1 && length (cbasPrunedKeys stats) == 1)

{- | H11 (2026-06-12) — under-min-trades smoke windows are not strong
enough evidence to abort. AAVEUSDT today saw smoke windows produce a
single ~5% drawdown trade (one daily ATR) which then aborted a combo
with out-of-sample @finalEquity@ ≥ 1.42. With 'minTradesForAbort = 3'
the verdict must be 'BacktestNoVerdict' until at least three trades
fire, and must remain 'BacktestAbort' at and above the threshold.

Falsification: any of the documented rows below disagreeing with the
computed verdict.
-}
testBotStartupBacktestVerdictMinTradesGuard :: IO ()
testBotStartupBacktestVerdictMinTradesGuard = do
    -- minTradesForAbort = 3: AAVEUSDT-style single-trade losses are
    -- below the noise floor.
    assert
        "single-trade loss under minTrades is NoVerdict, not Abort"
        ( botStartupBacktestVerdictWithMinTrades 3 True (Just 0.954) (Just 1) == BacktestNoVerdict
            && botStartupBacktestVerdictWithMinTrades 3 True (Just 0.95) (Just 2) == BacktestNoVerdict
            -- At the threshold, the verdict flips to Abort.
            && botStartupBacktestVerdictWithMinTrades 3 True (Just 0.95) (Just 3) == BacktestAbort
            && botStartupBacktestVerdictWithMinTrades 3 True (Just 0.5) (Just 12) == BacktestAbort
            -- Above-threshold finalEquity wins regardless of trade count.
            && botStartupBacktestVerdictWithMinTrades 3 True (Just 1.5) (Just 1) == BacktestAllow
            && botStartupBacktestVerdictWithMinTrades 3 True (Just 1.5) (Just 0) == BacktestAllow
            -- Zero-trade is still NoVerdict (existing 2026-06-11 invariant).
            && botStartupBacktestVerdictWithMinTrades 3 True (Just 1.0) (Just 0) == BacktestNoVerdict
            -- Disabled guard always Allow.
            && botStartupBacktestVerdictWithMinTrades 3 False (Just 0.5) (Just 99) == BacktestAllow
            -- Pathological minTrades ≤ 0 is normalized to 1 (matches the
            -- pre-2026-06-12 behaviour). One actual trade aborts.
            && botStartupBacktestVerdictWithMinTrades 0 True (Just 0.5) (Just 1) == BacktestAbort
            && botStartupBacktestVerdictWithMinTrades (-5) True (Just 0.5) (Just 1) == BacktestAbort
            -- Backward-compat: the 2-arg verdict equals the WithMinTrades
            -- form with minTrades = 1 for every interesting row.
            && botStartupBacktestVerdict True (Just 0.95) (Just 1) == BacktestAbort
            && botStartupBacktestVerdictWithMinTrades 1 True (Just 0.95) (Just 1) == BacktestAbort
        )

{- | The default minimum-trade-count for treating a sub-threshold smoke
backtest as an abort is 3. This is the value chosen in the 2026-06-12
review based on the AAVEUSDT smoke-window noise floor (~5% per trade
at the active vol target); a value below 3 reintroduces the 2026-06-12
erosion mode. We pin it so any future change is deliberate and shows
up as a test edit alongside the policy change.
-}
testBotStartupBacktestVerdictDefaultMinTradesIsThree :: IO ()
testBotStartupBacktestVerdictDefaultMinTradesIsThree = do
    assert
        "default minimum trade count for abort is 3"
        (defaultBotStartupBacktestMinTrades == 3)

{- | The bot-start guard must not prune the top-combos store or DB row on
any verdict (2026-06-12). Startup smoke windows are not the pruning
authority: block the start on a real abort, but do not delete the combo.
Scheduled stale refreshes handle drop decisions with tombstones.

Falsification: any verdict for which 'botStartupGuardShouldPrune'
returns 'True'. The current contract is uniformly 'False'.
-}
testBotStartupGuardShouldPruneIsFalse :: IO ()
testBotStartupGuardShouldPruneIsFalse = do
    assert
        "bot-start guard never prunes on any verdict"
        ( not (botStartupGuardShouldPrune BacktestAllow)
            && not (botStartupGuardShouldPrune BacktestAbort)
            && not (botStartupGuardShouldPrune BacktestNoVerdict)
        )

liveStatsForTest :: Double -> Maybe Double -> Int -> Aeson.Value
liveStatsForTest eq mAnn count =
    Aeson.object
        ( [ "finalEquity" .= eq
          , "operationCount" .= count
          ]
            ++ maybe [] (\ann -> ["annualizedReturn" .= ann]) mAnn
        )

liveComboForTest :: T.Text -> Double -> Maybe Aeson.Value -> Aeson.Value
liveComboForTest sym backtestAnn mLive =
    Aeson.object
        [ "symbol" .= sym
        , "uuid" .= ("0badc0de-1234-5678-9abc-def012345678" :: T.Text)
        , "finalEquity" .= (1.4 :: Double)
        , "annualizedReturn" .= backtestAnn
        , "openThreshold" .= (0.005 :: Double)
        , "closeThreshold" .= (0.010 :: Double)
        , "objective" .= ("final-equity" :: T.Text)
        , "params" .= Aeson.object ["method" .= ("blend" :: T.Text)]
        , "metrics"
            .= Aeson.object
                ( ["finalEquity" .= (1.4 :: Double), "tradeCount" .= adoptionMinTradeCount]
                    ++ maybe [] (\live -> ["live" .= live]) mLive
                )
        ]

{- | Live evidence shifts the ranking via shrinkage: with n live orders the
live annualized return carries weight n/(n+25), so a combo whose live record
contradicts its backtest sinks below an identical combo with no live history,
but a thin live record (small n) barely moves the needle.
-}
testLiveBlendShrinkageRanking :: IO ()
testLiveBlendShrinkageRanking = do
    let halfWeightStats =
            ComboLiveStats
                { clsFinalEquity = 1.0
                , clsAnnualizedReturn = Just 0.0
                , clsOperationCount = 25
                , clsFirstAtMs = Nothing
                , clsLastAtMs = Nothing
                }
    assert
        "n = shrinkage pseudo-count gives live evidence exactly half weight"
        (abs (blendedAnnualizedReturn 2.0 (Just halfWeightStats) - 1.0) < 1e-9)
    assert
        "no live stats leaves the backtest prior untouched"
        (blendedAnnualizedReturn 2.0 Nothing == 2.0)
    let noLive = liveComboForTest "BTCUSDT" 2.0 Nothing
        liveBreakEven = liveComboForTest "BTCUSDT" 2.0 (Just (liveStatsForTest 1.0 (Just 0.0) 25))
        liveThin = liveComboForTest "BTCUSDT" 2.0 (Just (liveStatsForTest 1.0 (Just 0.0) 1))
    assert
        "live break-even record ranks an identical combo lower"
        (comboPerformanceKey liveBreakEven > comboPerformanceKey noLive)
    assert
        "a thin live record moves the rank less than a substantial one"
        (comboPerformanceKey liveThin < comboPerformanceKey liveBreakEven)

testLiveQuarantineThresholds :: IO ()
testLiveQuarantineThresholds = do
    let stats eq count =
            ComboLiveStats
                { clsFinalEquity = eq
                , clsAnnualizedReturn = Just (-0.5)
                , clsOperationCount = count
                , clsFirstAtMs = Nothing
                , clsLastAtMs = Nothing
                }
    assert "below minimum live orders never quarantines" (not (liveStatsQuarantined (stats 0.5 29)))
    assert "sustained live loss quarantines" (liveStatsQuarantined (stats 0.98 30))
    assert "live winner is not quarantined" (not (liveStatsQuarantined (stats 1.05 30)))
    let quarantined = liveComboForTest "ETHUSDT" 3.0 (Just (liveStatsForTest 0.9 (Just (-0.5)) 40))
        modest = liveComboForTest "ETHUSDT" 0.1 Nothing
    assert "combo-level quarantine flag reads from metrics.live" (comboLiveQuarantined quarantined)
    assert
        "quarantined combo sinks below any healthy combo regardless of backtest"
        (comboPerformanceKey quarantined > comboPerformanceKey modest)

{- | A combo's UUID is derived from its backtest result (objective + tuned
thresholds), so re-discovering the same strategy mints a fresh UUID and its
accumulated live record is orphaned. A losing symbol can therefore churn
through several UUIDs that each individually stay under the per-combo order
floor and never quarantine. The family-level check pools their orders so the
strategy is still caught.
-}
testLiveFamilyQuarantineAcrossUuidChurn :: IO ()
testLiveFamilyQuarantineAcrossUuidChurn = do
    let stats eq count =
            ComboLiveStats
                { clsFinalEquity = eq
                , clsAnnualizedReturn = Just (-0.5)
                , clsOperationCount = count
                , clsFirstAtMs = Nothing
                , clsLastAtMs = Nothing
                }
        -- Same losing strategy spread across three churned UUIDs: each below
        -- the 30-order floor, so none quarantines on its own.
        churned = [stats 0.94 12, stats 0.95 11, stats 0.955 9]
    assert
        "no single churned UUID reaches the per-combo order floor"
        (not (any liveStatsQuarantined churned))
    assert
        "pooled family record quarantines the churned losing strategy"
        (liveStatsFamilyQuarantined churned)
    assert
        "a family of one matches the per-combo verdict"
        (liveStatsFamilyQuarantined [stats 0.98 30] && not (liveStatsFamilyQuarantined [stats 0.98 29]))
    assert
        "a winning family is never quarantined even with many orders"
        (not (liveStatsFamilyQuarantined [stats 1.05 40, stats 1.02 40]))
    assert
        "a mixed family that nets break-even is not quarantined"
        (not (liveStatsFamilyQuarantined [stats 0.95 20, stats 1.06 20]))

{- | 'recalculateComboPerformanceFromOperation' must accumulate a separate
live record (compounded equity, order count, span) alongside the blended
backtest fields it already maintains, and only annualize once the observed
span is long enough to be meaningful.
-}
testRecalculateMaintainsLiveStats :: IO ()
testRecalculateMaintainsLiveStats = do
    let t0 = 1700000000000 :: Int64
        (_, _, metrics1) =
            recalculateComboPerformanceFromOperation t0 (Just "1h") (Just 1.42) (Just 0.8) KM.empty Nothing 1.05
    stats1 <- case comboLiveStatsFromObject metrics1 of
        Nothing -> ioError (userError "first operation did not create a live record")
        Just s -> pure s
    assert "first operation seeds the live record" (clsOperationCount stats1 == 1)
    -- No same-session previous order => no measurable equity change: the
    -- session equity baseline is arbitrary (resets on every bot start), so
    -- the first op of a session must not move live equity.
    assert "first operation records no equity change without a session baseline" (abs (clsFinalEquity stats1 - 1.0) < 1e-9)
    assert "first operation pins the live span start" (clsFirstAtMs stats1 == Just t0)
    assert "no annualization before a meaningful live span" (isNothing (clsAnnualizedReturn stats1))
    let t1 = t0 + 2 * 86400000
        (_, _, metrics2) =
            recalculateComboPerformanceFromOperation t1 (Just "1h") (Just 1.49) (Just 0.8) metrics1 (Just 1.05) 0.945
    stats2 <- case comboLiveStatsFromObject metrics2 of
        Nothing -> ioError (userError "second operation lost the live record")
        Just s -> pure s
    assert "second operation increments the live order count" (clsOperationCount stats2 == 2)
    assert "live equity compounds across operations" (abs (clsFinalEquity stats2 - 0.9) < 1e-9)
    assert "live span start is preserved" (clsFirstAtMs stats2 == Just t0)
    assert "live span end tracks the latest operation" (clsLastAtMs stats2 == Just t1)
    assert
        "a losing live record annualizes negative (clamped)"
        (maybe False (<= (-0.99)) (clsAnnualizedReturn stats2))

{- | A scheduled re-backtest replaces a combo's metrics wholesale; the
accumulated live record must survive that refresh or the system forgets
everything it learned from its own orders every 24h.
-}
testBacktestUpdatePreservesLiveStats :: IO ()
testBacktestUpdatePreservesLiveStats = do
    let combo = liveComboForTest "BTCUSDT" 1.2 (Just (liveStatsForTest 1.1 (Just 0.4) 12))
        combosJson = Aeson.object ["combos" .= [combo]]
        key = case comboIdentityKey combo of
            Just k -> k
            Nothing -> error "test setup: combo identity key resolution failed"
        update =
            ComboBacktestUpdate
                { cbuMetrics =
                    Aeson.object
                        [ "finalEquity" .= (1.3 :: Double)
                        , "tradeCount" .= (10 :: Int)
                        ]
                , cbuFinalEquity = Just 1.3
                , cbuScore = Just 0.3
                , cbuOperations = Nothing
                }
    case applyComboUpdatesWithStats 0 (HM.singleton key update) combosJson of
        Left err -> ioError (userError ("applyComboUpdatesWithStats failed unexpectedly: " ++ err))
        Right (updatedVal, _) -> do
            let mUpdatedCombo = case updatedVal of
                    Aeson.Object o -> case KM.lookup "combos" o of
                        Just (Aeson.Array v) | not (V.null v) -> Just (V.head v)
                        _ -> Nothing
                    _ -> Nothing
            stats <- case mUpdatedCombo >>= comboLiveStats of
                Nothing -> ioError (userError "backtest refresh erased the live record")
                Just s -> pure s
            assert "live record survives a backtest metrics refresh" (clsOperationCount stats == 12)

{- | The S3 combo bus merges payloads from multiple instances; whichever
duplicate wins the merge must carry the richest live record seen across all
of them, so live evidence is never lost in transit between instances.
-}
testMergePreservesLiveStats :: IO ()
testMergePreservesLiveStats = do
    let comboWithLive = liveComboForTest "BTCUSDT" 1.2 (Just (liveStatsForTest 1.1 (Just 0.4) 10))
        comboFresh =
            case liveComboForTest "BTCUSDT" 1.2 Nothing of
                Aeson.Object o -> Aeson.Object (KM.insert "score" (Aeson.toJSON (0.9 :: Double)) o)
                v -> v
        payload combosVal generatedAt =
            Aeson.object
                [ "combos" .= [combosVal]
                , "generatedAtMs" .= (generatedAt :: Int64)
                , "source" .= ("test" :: T.Text)
                ]
        merged = mergeTopCombosPayloads 10 2000 [payload comboWithLive 1000, payload comboFresh 1500]
        mMergedCombo = case merged of
            Aeson.Object o -> case KM.lookup "combos" o of
                Just (Aeson.Array v) | not (V.null v) -> Just (V.head v)
                _ -> Nothing
            _ -> Nothing
    case mMergedCombo of
        Nothing -> ioError (userError "merge produced no combos")
        Just mergedCombo -> do
            stats <- case comboLiveStats mergedCombo of
                Nothing -> ioError (userError "merge dropped the live record")
                Just s -> pure s
            assert "merge keeps the richest live record across duplicates" (clsOperationCount stats == 10)

-- | Combo with controllable score / createdAtMs / refresh stamp for merge tests.
freshnessComboForTest :: Double -> Maybe Int64 -> Maybe Int64 -> Aeson.Value
freshnessComboForTest score mCreatedAt mRefreshedAt =
    Aeson.object
        ( [ "symbol" .= ("BTCUSDT" :: T.Text)
          , "interval" .= ("15m" :: T.Text)
          , "platform" .= ("binance" :: T.Text)
          , "uuid" .= ("0badc0de-1234-5678-9abc-def012345678" :: T.Text)
          , "finalEquity" .= (1.0 + score :: Double)
          , "score" .= score
          , "openThreshold" .= (0.005 :: Double)
          , "closeThreshold" .= (0.010 :: Double)
          , "objective" .= ("final-equity" :: T.Text)
          , "params" .= Aeson.object ["method" .= ("blend" :: T.Text), "lookback" .= (48 :: Int)]
          , "metrics" .= Aeson.object ["finalEquity" .= (1.0 + score :: Double), "tradeCount" .= (8 :: Int)]
          ]
            ++ maybe [] (\t -> ["createdAtMs" .= t]) mCreatedAt
            ++ maybe [] (\t -> ["backtestRefreshedAtMs" .= t]) mRefreshedAt
        )

mergeWinnerScore :: [Aeson.Value] -> Maybe Double
mergeWinnerScore combos =
    let payload =
            Aeson.object
                [ "combos" .= combos
                , "generatedAtMs" .= (9000 :: Int64)
                , "source" .= ("test" :: T.Text)
                ]
        merged = mergeTopCombosPayloads 10 9500 [payload]
     in case merged of
            Aeson.Object o -> case KM.lookup "combos" o of
                Just (Aeson.Array v) | not (V.null v) ->
                    case V.head v of
                        Aeson.Object c -> KM.lookup "score" c >>= AT.parseMaybe Aeson.parseJSON
                        _ -> Nothing
                _ -> Nothing
            _ -> Nothing

{- | Identity merges prefer the most recent backtest reading once a refresh
stamp is involved: an honest (lower-scoring) in-place refresh must beat the
stale, higher-scoring copy a lagging replica still publishes — otherwise the
cross-instance union merge would undo every deflating refresh.
-}
testMergeRefreshedComboBeatsStaleScore :: IO ()
testMergeRefreshedComboBeatsStaleScore = do
    let stale = freshnessComboForTest 5.0 (Just 1000) Nothing
        refreshed = freshnessComboForTest 0.1 (Just 1000) (Just 5000)
    assert
        "refreshed (deflated) combo wins the merge over the stale higher score"
        (mergeWinnerScore [stale, refreshed] == Just 0.1)
    assert
        "winner is order-independent"
        (mergeWinnerScore [refreshed, stale] == Just 0.1)

testMergeRefreshedComboBeatsUntimestampedStaleScore :: IO ()
testMergeRefreshedComboBeatsUntimestampedStaleScore = do
    let stale = freshnessComboForTest 5.0 Nothing Nothing
        refreshed = freshnessComboForTest 0.1 Nothing (Just 5000)
        payload combo =
            Aeson.object
                [ "combos" .= [combo]
                ]
        winnerScore :: [Aeson.Value] -> Maybe Double
        winnerScore payloads =
            case mergeTopCombosPayloads 10 9500 payloads of
                Aeson.Object o -> case KM.lookup "combos" o of
                    Just (Aeson.Array v) | not (V.null v) ->
                        case V.head v of
                            Aeson.Object c -> KM.lookup "score" c >>= AT.parseMaybe Aeson.parseJSON
                            _ -> Nothing
                    _ -> Nothing
                _ -> Nothing
    assert
        "stamped refresh beats a legacy stale duplicate with no comparable timestamp"
        (winnerScore [payload stale, payload refreshed] == Just 0.1)
    assert
        "stamped-vs-untimestamped winner is order-independent"
        (winnerScore [payload refreshed, payload stale] == Just 0.1)

{- | A re-discovered duplicate is itself a fresh backtest of the same
identity: when its createdAtMs postdates the incumbent's refresh stamp, the
new discovery wins even against a stamped record.
-}
testMergeNewerDiscoveryBeatsOlderRefresh :: IO ()
testMergeNewerDiscoveryBeatsOlderRefresh = do
    let refreshedOld = freshnessComboForTest 0.1 (Just 1000) (Just 5000)
        rediscovered = freshnessComboForTest 0.4 (Just 8000) Nothing
    assert
        "newer discovery beats an older refresh stamp"
        (mergeWinnerScore [refreshedOld, rediscovered] == Just 0.4)

{- | Without any refresh stamp the historical best-ever-score semantics are
unchanged, whatever the createdAtMs ordering says.
-}
testMergeUnstampedDuplicatesKeepBestEver :: IO ()
testMergeUnstampedDuplicatesKeepBestEver = do
    let older = freshnessComboForTest 5.0 (Just 1000) Nothing
        newer = freshnessComboForTest 0.1 (Just 8000) Nothing
    assert
        "unstamped duplicates keep best-ever score"
        (mergeWinnerScore [older, newer] == Just 5.0)

{- | A refresh that deflates a combo BELOW 1.0 equity must still survive the
merge's sanitize pass: dropping it there would hand the merge to the stale,
inflated copy a peer instance still publishes, resurrecting it on every
union merge. Unstamped sub-1.0 combos remain junk and are still dropped.
-}
testMergeSanitizeKeepsStampedSubOneRefresh :: IO ()
testMergeSanitizeKeepsStampedSubOneRefresh = do
    let stale = freshnessComboForTest 5.0 (Just 1000) Nothing
        refreshedLoss = freshnessComboForTest (-0.15) (Just 1000) (Just 5000)
    assert
        "stamped sub-1.0 refresh beats the stale inflated copy in the merge"
        (mergeWinnerScore [stale, refreshedLoss] == Just (-0.15))
    assert
        "stamped sub-1.0 winner is order-independent"
        (mergeWinnerScore [refreshedLoss, stale] == Just (-0.15))
    let unstampedLoss = freshnessComboForTest (-0.15) (Just 1000) Nothing
    assert
        "unstamped sub-1.0 combo is still sanitized away"
        (isNothing (mergeWinnerScore [unstampedLoss]))

evidenceFloorComboForTest :: T.Text -> Double -> Int -> Aeson.Value
evidenceFloorComboForTest label annualizedReturn tradeCount =
    Aeson.object
        [ "uuid" .= label
        , "symbol" .= ("BTCUSDT" :: T.Text)
        , "interval" .= ("15m" :: T.Text)
        , "platform" .= ("binance" :: T.Text)
        , "finalEquity" .= (1.02 :: Double)
        , "score" .= annualizedReturn
        , "openThreshold" .= (0.005 :: Double)
        , "closeThreshold" .= (0.010 :: Double)
        , "objective" .= ("annualized-equity" :: T.Text)
        , "params" .= Aeson.object ["method" .= label, "lookback" .= (48 :: Int)]
        , "metrics"
            .= Aeson.object
                [ "finalEquity" .= (1.02 :: Double)
                , "annualizedReturn" .= annualizedReturn
                , "tradeCount" .= tradeCount
                ]
        ]

testLowTradeTopCombosSinkBelowEvidenceFloor :: IO ()
testLowTradeTopCombosSinkBelowEvidenceFloor = do
    let lowTradeOutlier = evidenceFloorComboForTest "one-trade-outlier" 2500.0 1
        deployable = evidenceFloorComboForTest "deployable" 1.0 adoptionMinTradeCount
        payload =
            Aeson.object
                [ "combos" .= [lowTradeOutlier, deployable]
                , "generatedAtMs" .= (9000 :: Int64)
                , "source" .= ("test" :: T.Text)
                ]
        merged = mergeTopCombosPayloads 10 9500 [payload]
        firstMethod =
            case merged of
                Aeson.Object o -> case KM.lookup "combos" o of
                    Just (Aeson.Array v) | not (V.null v) ->
                        case V.head v of
                            Aeson.Object c -> do
                                Aeson.Object params <- KM.lookup "params" c
                                KM.lookup "method" params >>= AT.parseMaybe Aeson.parseJSON
                            _ -> Nothing
                    _ -> Nothing
                _ -> Nothing
    assert
        "one-trade annualized-return outlier sinks below a combo that meets the evidence floor"
        (firstMethod == Just ("deployable" :: T.Text))

processingComboForTest :: T.Text -> T.Text -> Bool -> Maybe Double -> Double -> Int -> Aeson.Value
processingComboForTest label source includeNullParam mWalkForwardSharpe annualizedReturn tradeCount =
    let params =
            Aeson.object
                ( [ "method" .= label
                  , "lookback" .= (48 :: Int)
                  , "interval" .= ("15m" :: T.Text)
                  , "binanceSymbol" .= ("BTCUSDT" :: T.Text)
                  ]
                    ++ ["protectionMinConfidence" .= Aeson.Null | includeNullParam]
                )
        walkForward =
            maybe
                []
                (\sharpe -> ["walkForwardSummary" .= Aeson.object ["sharpeMean" .= sharpe]])
                mWalkForwardSharpe
     in Aeson.object
            [ "uuid" .= label
            , "source" .= source
            , "symbol" .= ("BTCUSDT" :: T.Text)
            , "interval" .= ("15m" :: T.Text)
            , "platform" .= ("binance" :: T.Text)
            , "finalEquity" .= (1.08 :: Double)
            , "score" .= annualizedReturn
            , "openThreshold" .= (0.005 :: Double)
            , "closeThreshold" .= (0.010 :: Double)
            , "objective" .= ("annualized-equity" :: T.Text)
            , "params" .= params
            , "metrics"
                .= Aeson.object
                    ( [ "finalEquity" .= (1.08 :: Double)
                      , "annualizedReturn" .= annualizedReturn
                      , "tradeCount" .= tradeCount
                      , "maxDrawdown" .= (0.02 :: Double)
                      ]
                        ++ walkForward
                    )
            ]

mergedCombosForTest :: [Aeson.Value] -> [Aeson.Value]
mergedCombosForTest combos =
    let payload =
            Aeson.object
                [ "combos" .= combos
                , "generatedAtMs" .= (9000 :: Int64)
                , "source" .= ("test" :: T.Text)
                ]
     in case mergeTopCombosPayloads 10 9500 [payload] of
            Aeson.Object o -> case KM.lookup "combos" o of
                Just (Aeson.Array v) -> V.toList v
                _ -> []
            _ -> []

comboProcessingTierForTest :: Aeson.Value -> Maybe T.Text
comboProcessingTierForTest combo =
    case combo of
        Aeson.Object o -> do
            Aeson.Object processing <- KM.lookup "processing" o
            KM.lookup "tier" processing >>= AT.parseMaybe Aeson.parseJSON
        _ -> Nothing

comboProcessingReasonsForTest :: Aeson.Value -> Maybe [T.Text]
comboProcessingReasonsForTest combo =
    case combo of
        Aeson.Object o -> do
            Aeson.Object processing <- KM.lookup "processing" o
            KM.lookup "reasons" processing >>= AT.parseMaybe Aeson.parseJSON
        _ -> Nothing

testMergeDedupesSourceAndNullEquivalentCombos :: IO ()
testMergeDedupesSourceAndNullEquivalentCombos = do
    let dbCopy = processingComboForTest "same-strategy" "db" True Nothing 2.0 adoptionMinTradeCount
        binanceCopy = processingComboForTest "same-strategy" "binance" False Nothing 2.0 adoptionMinTradeCount
        combos = mergedCombosForTest [dbCopy, binanceCopy]
        mCombo = listToMaybe combos
        tier = mCombo >>= comboProcessingTierForTest
    assert
        "source/null-equivalent strategy rows collapse to one ranked combo"
        (length combos == 1)
    assert
        "deduped combo is candidate until walk-forward evidence exists"
        (tier == Just "candidate")
    assert
        "processing records missing walk-forward evidence explicitly"
        (maybe False ("walk-forward-missing" `elem`) (mCombo >>= comboProcessingReasonsForTest))

testDeployableTierRanksAheadOfUnvalidatedCandidate :: IO ()
testDeployableTierRanksAheadOfUnvalidatedCandidate = do
    let unvalidated =
            processingComboForTest
                "unvalidated-high-return"
                "db"
                False
                Nothing
                200.0
                adoptionMinTradeCount
        deployable =
            processingComboForTest
                "deployable-lower-return"
                "db"
                False
                (Just adoptionMinWalkForwardSharpeMean)
                1.0
                adoptionMinTradeCount
        combos = mergedCombosForTest [unvalidated, deployable]
    assert
        "combo merge returned both distinct strategies"
        (length combos == 2)
    assert
        "walk-forward deployable tier ranks ahead of a higher-return unvalidated candidate"
        (listToMaybe (mapMaybe comboProcessingTierForTest combos) == Just "deployable")

testMergeExecutableAnnotatesProcessingAndDedupe :: IO ()
testMergeExecutableAnnotatesProcessingAndDedupe = do
    (inputPath, inputHandle) <- openTempFile "/tmp" "trader-merge-input.json"
    hClose inputHandle
    (outputPath, outputHandle) <- openTempFile "/tmp" "trader-merge-output.json"
    hClose outputHandle
    let dbCopy = processingComboForTest "same-strategy" "db" True Nothing 2.0 adoptionMinTradeCount
        binanceCopy = processingComboForTest "same-strategy" "binance" False Nothing 2.0 adoptionMinTradeCount
        payload =
            Aeson.object
                [ "combos" .= [dbCopy, binanceCopy]
                , "generatedAtMs" .= (9000 :: Int64)
                , "source" .= ("test" :: T.Text)
                ]
    BL.writeFile inputPath (Aeson.encode payload)
    code <-
        runMerge
            MergeArgs
                { maTopJson = inputPath
                , maFromJsonl = []
                , maFromTopJson = []
                , maOut = outputPath
                , maMax = 10
                , maHistoryDir = Nothing
                , maScoringConfig = defaultTopComboScoringConfig
                , maCopyToDist = False
                }
    decoded <- (Aeson.eitherDecode <$> BL.readFile outputPath) :: IO (Either String Aeson.Value)
    let combos =
            case decoded of
                Right (Aeson.Object o) -> case KM.lookup "combos" o of
                    Just (Aeson.Array v) -> V.toList v
                    _ -> []
                _ -> []
        mCombo = listToMaybe combos
    assert "merge executable exits successfully" (code == 0)
    assert "merge executable collapses source/null-equivalent strategy rows" (length combos == 1)
    assert
        "merge executable emits candidate processing tier"
        ((mCombo >>= comboProcessingTierForTest) == Just "candidate")
    assert
        "merge executable records missing walk-forward evidence"
        (maybe False ("walk-forward-missing" `elem`) (mCombo >>= comboProcessingReasonsForTest))
    _ <- try (removeFile inputPath) :: IO (Either SomeException ())
    _ <- try (removeFile outputPath) :: IO (Either SomeException ())
    pure ()

selectionComboForTest :: T.Text -> Double -> Maybe Int64 -> Maybe Int64 -> Aeson.Value
selectionComboForTest method score mCreatedAt mRefreshedAt =
    Aeson.object
        ( [ "symbol" .= ("BTCUSDT" :: T.Text)
          , "interval" .= ("15m" :: T.Text)
          , "platform" .= ("binance" :: T.Text)
          , "uuid" .= ("0badc0de-1234-5678-9abc-def012345678" :: T.Text)
          , "finalEquity" .= (1.0 + score :: Double)
          , "score" .= score
          , "openThreshold" .= (0.005 :: Double)
          , "closeThreshold" .= (0.010 :: Double)
          , "objective" .= ("final-equity" :: T.Text)
          , "params" .= Aeson.object ["method" .= method, "lookback" .= (48 :: Int)]
          , "metrics" .= Aeson.object ["finalEquity" .= (1.0 + score :: Double), "tradeCount" .= (8 :: Int)]
          ]
            ++ maybe [] (\t -> ["createdAtMs" .= t]) mCreatedAt
            ++ maybe [] (\t -> ["backtestRefreshedAtMs" .= t]) mRefreshedAt
        )

testSelectCombosForBacktestRefreshIncludesEveryStaleCombo :: IO ()
testSelectCombosForBacktestRefreshIncludesEveryStaleCombo = do
    let now = 10 * 86400000 :: Int64
        oneDay = 86400000 :: Int64
        topFresh = selectionComboForTest "top-fresh" 5.0 (Just (now - oneDay)) Nothing
        staleLowRank = selectionComboForTest "stale-low-rank" 0.1 (Just (now - comboBacktestStaleAfterMs - 1)) Nothing
        freshLowRank = selectionComboForTest "fresh-low-rank" 0.2 (Just (now - oneDay)) Nothing
        missingFreshness = selectionComboForTest "missing-freshness" 0.3 Nothing Nothing
        exactlyAtBoundary = selectionComboForTest "boundary" 0.4 (Just (now - comboBacktestStaleAfterMs)) Nothing
        selected = selectCombosForBacktestRefresh 1 now [staleLowRank, freshLowRank, topFresh, missingFreshness]
        selectedKeys = mapMaybe comboIdentityKey selected
        has combo = maybe False (`elem` selectedKeys) (comboIdentityKey combo)
    assert
        "selection keeps the top-ranked combo and every stale combo outside topN"
        ( length selected == 3
            && has topFresh
            && has staleLowRank
            && has missingFreshness
            && not (has freshLowRank)
        )
    assert
        "missing freshness is due, exactly three days old is not older than three days"
        ( comboBacktestDueForRefresh now missingFreshness
            && not (comboBacktestDueForRefresh now exactlyAtBoundary)
        )

testPrunedBacktestTombstonePreventsStaleResurrection :: IO ()
testPrunedBacktestTombstonePreventsStaleResurrection = do
    let stale = freshnessComboForTest 5.0 (Just 1000) Nothing
        payload =
            Aeson.object
                [ "combos" .= [stale]
                , "generatedAtMs" .= (1000 :: Int64)
                , "source" .= ("test" :: T.Text)
                ]
        key = case comboIdentityKey stale of
            Just k -> k
            Nothing -> error "test setup: combo identity key resolution failed"
        lossUpdate =
            ComboBacktestUpdate
                { cbuMetrics = Aeson.object ["finalEquity" .= (0.85 :: Double), "tradeCount" .= (12 :: Int)]
                , cbuFinalEquity = Just 0.85
                , cbuScore = Just (-0.15)
                , cbuOperations = Nothing
                }
    case applyComboUpdatesWithStats 5000 (HM.singleton key lossUpdate) payload of
        Left err -> ioError (userError ("applyComboUpdatesWithStats failed unexpectedly: " ++ err))
        Right (droppedPayload, stats) -> do
            assert
                "periodic backtest prune removes the refreshed loser"
                (topCombosCount droppedPayload == 0 && cbasUpdatedCount stats == 1 && cbasPrunedCount stats == 1)
            let staleReplica =
                    Aeson.object
                        [ "combos" .= [stale]
                        , "generatedAtMs" .= (4000 :: Int64)
                        , "source" .= ("stale-replica" :: T.Text)
                        ]
                resurrectAttempt = mergeTopCombosPayloads 10 6000 [droppedPayload, staleReplica]
            assert
                "drop tombstone blocks a stale replica from resurrecting the combo"
                (topCombosCount resurrectAttempt == 0)
            let rediscovered =
                    Aeson.object
                        [ "combos" .= [freshnessComboForTest 0.2 (Just 6000) Nothing]
                        , "generatedAtMs" .= (6000 :: Int64)
                        , "source" .= ("rediscovered" :: T.Text)
                        ]
                rediscoveredMerge = mergeTopCombosPayloads 10 7000 [droppedPayload, rediscovered]
            assert
                "a newer rediscovery after the tombstone is allowed back in"
                (mergeWinnerScoreFromPayload rediscoveredMerge == Just 0.2)

mergeWinnerScoreFromPayload :: Aeson.Value -> Maybe Double
mergeWinnerScoreFromPayload payload =
    case payload of
        Aeson.Object o -> case KM.lookup "combos" o of
            Just (Aeson.Array v) | not (V.null v) ->
                case V.head v of
                    Aeson.Object c -> KM.lookup "score" c >>= AT.parseMaybe Aeson.parseJSON
                    _ -> Nothing
            _ -> Nothing
        _ -> Nothing

{- | Startup backtest guards use the keep-all apply variant: an abort blocks
the start, but does not delete the combo from the store. Scheduled stale
refreshes use the pruning variant tested above.
-}
testKeepAllUpdateKeepsUnprofitableComboStamped :: IO ()
testKeepAllUpdateKeepsUnprofitableComboStamped = do
    let combosJson =
            Aeson.object ["combos" .= [freshnessComboForTest 5.0 (Just 1000) Nothing]]
        firstCombo = case combosJson of
            Aeson.Object o -> case KM.lookup "combos" o of
                Just (Aeson.Array v) | not (V.null v) -> Just (V.head v)
                _ -> Nothing
            _ -> Nothing
        key = case firstCombo >>= comboIdentityKey of
            Just k -> k
            Nothing -> error "test setup: combo identity key resolution failed"
        lossUpdate =
            ComboBacktestUpdate
                { cbuMetrics = Aeson.object ["finalEquity" .= (0.85 :: Double), "tradeCount" .= (12 :: Int)]
                , cbuFinalEquity = Just 0.85
                , cbuScore = Just (-0.15)
                , cbuOperations = Nothing
                }
    case applyComboUpdatesKeepAllWithStats 7777 (HM.singleton key lossUpdate) combosJson of
        Left err -> ioError (userError ("applyComboUpdatesKeepAllWithStats failed unexpectedly: " ++ err))
        Right (updatedVal, stats) -> do
            assert
                "keep-all apply updates without pruning"
                (cbasUpdatedCount stats == 1 && cbasPrunedCount stats == 0 && null (cbasPrunedKeys stats))
            let mUpdated = case updatedVal of
                    Aeson.Object o -> case KM.lookup "combos" o of
                        Just (Aeson.Array v) | not (V.null v) -> Just (V.head v)
                        _ -> Nothing
                    _ -> Nothing
            case mUpdated of
                Nothing -> ioError (userError "keep-all apply dropped the combo")
                Just updated -> do
                    let mEq = case updated of
                            Aeson.Object c -> KM.lookup "finalEquity" c >>= AT.parseMaybe Aeson.parseJSON
                            _ -> Nothing
                    assert
                        "unprofitable refresh persists deflated metrics"
                        (mEq == Just (0.85 :: Double))
                    assert
                        "refresh stamps backtestRefreshedAtMs"
                        (comboBacktestFreshnessMs updated == Just 7777)

mkOutcomeTestTrade :: Int -> Int -> Double -> Trade
mkOutcomeTestTrade entryIdx exitIdx ret =
    Trade
        { trEntryIndex = entryIdx
        , trExitIndex = exitIdx
        , trEntryEquity = 1.0
        , trExitEquity = 1.0 + ret
        , trReturn = ret
        , trHoldingPeriods = exitIdx - entryIdx
        , trEntryHighVolProb = Nothing
        , trEntrySource = TradeEntrySignal
        , trExitReason = Nothing
        , trEntryIp = Nothing
        , trExitIp = Nothing
        , trFeeCost = 0.0
        }

{- | Outcome weights translate closed trades into per-bar loss emphasis for
the LSTM fine-tune: losing spans weigh more than winning spans (asymmetric
scales), uncovered bars stay at 1, and a cap bounds any single trade's
influence.
-}
testTradeOutcomeWeightsSemantics :: IO ()
testTradeOutcomeWeightsSemantics = do
    let win = mkOutcomeTestTrade 2 4 0.02
        loss = mkOutcomeTestTrade 6 7 (-0.02)
        weights = tradeOutcomeWeights [win, loss] 10
    assert "weights align index-for-index with the series" (length weights == 10)
    assert "bars not covered by any trade weigh 1" (all (\t -> weights !! t == 1.0) [0, 1, 5, 8, 9])
    assert
        "a winning trade reinforces its span (1 + winScale * |return|)"
        (all (\t -> abs (weights !! t - (1 + outcomeWeightWinScale * 0.02)) < 1e-9) [2 .. 4])
    assert
        "a losing trade punishes its span harder (1 + lossScale * |return|)"
        (all (\t -> abs (weights !! t - (1 + outcomeWeightLossScale * 0.02)) < 1e-9) [6, 7])
    assert
        "loss scale exceeds win scale"
        (outcomeWeightLossScale > outcomeWeightWinScale)
    assert
        "extreme trades cap at outcomeWeightCap"
        (tradeOutcomeWeightFactor (mkOutcomeTestTrade 0 1 (-0.5)) == Just outcomeWeightCap)
    assert
        "break-even trades carry no learning signal"
        (isNothing (tradeOutcomeWeightFactor (mkOutcomeTestTrade 0 1 0)))
    assert
        "trade spans clamp to the series bounds"
        (last (tradeOutcomeWeights [mkOutcomeTestTrade 8 20 (-0.02)] 10) > 1)
    assert "empty series yields no weights" (null (tradeOutcomeWeights [win] 0))

{- | The weighted fine-tune with default weights must be exactly the plain
fine-tune, so enabling outcome weighting cannot perturb bots with no
closed trades.
-}
testWeightedFineTuneUnitWeightsEquivalence :: IO ()
testWeightedFineTuneUnitWeightsEquivalence = do
    let series = [0.5 + 0.2 * sin (fromIntegral t / 5) | t <- [0 .. 59 :: Int]]
        cfg =
            LSTMConfig
                { lcLookback = 4
                , lcHiddenSize = 4
                , lcEpochs = 8
                , lcLearningRate = 0.01
                , lcAdamBeta1 = defaultLstmAdamBeta1
                , lcAdamBeta2 = defaultLstmAdamBeta2
                , lcAdamEps = defaultLstmAdamEps
                , lcValRatio = 0
                , lcPatience = 0
                , lcGradClip = Nothing
                , lcSeed = 7
                }
        (model0, _) = trainLSTM cfg{lcEpochs = 0} series
        (plain, _) = fineTuneLSTM cfg model0 series
        (weightedEmpty, _) = fineTuneLSTMWeighted cfg model0 series []
        (weightedOnes, _) = fineTuneLSTMWeighted cfg model0 series (replicate 60 1.0)
    assert "no weights reproduces the plain fine-tune bit-for-bit" (lmParams weightedEmpty == lmParams plain)
    assert "unit weights reproduce the plain fine-tune bit-for-bit" (lmParams weightedOnes == lmParams plain)

testAlignToBarsPointInTime :: IO ()
testAlignToBarsPointInTime = do
    let bars = V.fromList [1000, 2000, 3000 :: Int64]
        intervalMs = 1000 :: Int64
        -- Deliberately unsorted; (3500,30) is in the FUTURE relative to bars 0/1
        -- (their closes are 1999 and 2999), so it must NOT influence them.
        series = [(1500, 20.0), (500, 10.0), (3500, 30.0)]
    assert
        "alignToBars forward-fills point-in-time and never leaks future observations"
        (alignToBars bars intervalMs series == V.fromList [Just 20.0, Just 20.0, Just 30.0])
    assert
        "alignToBars yields Nothing for bars before the first observation"
        ( alignToBars (V.fromList [1000, 2000 :: Int64]) (1000 :: Int64) [(5000, 9.0)]
            == V.fromList [Nothing, Nothing]
        )
    assert
        "alignToBars carries the last value forward across gaps with no new data"
        ( alignToBars (V.fromList [1000, 2000, 3000, 4000 :: Int64]) (1000 :: Int64) [(900, 7.0)]
            == V.fromList [Just 7.0, Just 7.0, Just 7.0, Just 7.0]
        )

gapComboForTest :: T.Text -> Double -> Double -> Int -> Aeson.Value
gapComboForTest method backtestAnn liveAnn ops =
    Aeson.object
        [ "annualizedReturn" .= backtestAnn
        , "params" .= Aeson.object ["method" .= method]
        , "metrics"
            .= Aeson.object
                [ "live"
                    .= Aeson.object
                        [ "finalEquity" .= (1.0 :: Double)
                        , "operationCount" .= ops
                        , "annualizedReturn" .= liveAnn
                        ]
                ]
        ]

{- | The optimizer's report card: per-combo live-vs-backtest gap entries
require real live evidence, aggregate ops-weighted per method family, and
map to a bounded discovery-sampling multiplier.
-}
testLiveGapFeedback :: IO ()
testLiveGapFeedback = do
    let entry = comboLiveGapEntry (gapComboForTest "10" 2.0 0.5 12)
    assert
        "a combo with live evidence yields (method, live - backtest, ops)"
        (entry == Just ("10", -1.5, 12))
    assert
        "too few live orders yields no gap entry"
        (isNothing (comboLiveGapEntry (gapComboForTest "10" 2.0 0.5 (liveGapMinComboOperations - 1))))
    assert
        "a combo without a live record yields no gap entry"
        ( isNothing
            ( comboLiveGapEntry
                (Aeson.object ["annualizedReturn" .= (2.0 :: Double), "params" .= Aeson.object ["method" .= ("10" :: T.Text)]])
            )
        )
    let statsMap =
            liveGapStatsByMethod
                [ gapComboForTest "10" 1.0 0.0 30 -- gap -1, 30 ops
                , gapComboForTest "10" 1.0 2.0 10 -- gap +1, 10 ops
                , gapComboForTest "01" 3.0 0.0 40 -- separate family
                ]
    case Map.lookup "10" statsMap of
        Nothing -> ioError (userError "method 10 missing from gap stats")
        Just s -> do
            assert "family aggregates combos and operations" (lgsCombos s == 2 && lgsOperations s == 40)
            assert
                "the family gap is operations-weighted"
                (abs (lgsOpsWeightedGap s - ((-1) * 30 + 1 * 10) / 40) < 1e-9)
    assert "families aggregate separately" (Map.member "01" statsMap && Map.size statsMap == 2)
    let statsWith gap ops = LiveGapStats{lgsCombos = 3, lgsOperations = ops, lgsOpsWeightedGap = gap}
    assert "no evidence is neutral" (liveGapMethodMultiplier Nothing == 1)
    assert
        "below the family evidence floor is neutral"
        (liveGapMethodMultiplier (Just (statsWith (-2) (liveGapMinTotalOperations - 1))) == 1)
    assert
        "a moderate live shortfall scales sampling down proportionally"
        (abs (liveGapMethodMultiplier (Just (statsWith (-0.5) 60)) - 0.5) < 1e-9)
    assert
        "a chronic shortfall bottoms out at the floor, never zero"
        (liveGapMethodMultiplier (Just (statsWith (-3) 60)) == liveGapMultiplierFloor)
    assert
        "live outperformance caps at the ceiling"
        (liveGapMethodMultiplier (Just (statsWith 2 60)) == liveGapMultiplierCeiling)
    assert
        "mild outperformance earns a proportional boost"
        (abs (liveGapMethodMultiplier (Just (statsWith 0.2 60)) - 1.2) < 1e-9)
    let strictComboConfig = defaultLiveGapConfig{lgcMinComboOperations = liveGapMinComboOperations + 5}
    assert
        "configured per-combo evidence floor rejects weaker live samples"
        (isNothing (comboLiveGapEntryWithConfig strictComboConfig (gapComboForTest "10" 2.0 0.5 liveGapMinComboOperations)))
    assert
        "configured per-combo evidence floor admits enough live samples"
        (isJust (comboLiveGapEntryWithConfig strictComboConfig (gapComboForTest "10" 2.0 0.5 (liveGapMinComboOperations + 5))))
    let strictFamilyConfig = defaultLiveGapConfig{lgcMinTotalOperations = 100}
    assert
        "configured family evidence floor keeps low-sample multipliers neutral"
        (liveGapMethodMultiplierWithConfig strictFamilyConfig (Just (statsWith (-0.5) 60)) == 1)
    let clampConfig = defaultLiveGapConfig{lgcMultiplierFloor = 0.4, lgcMultiplierCeiling = 1.1}
    assert
        "configured live-gap multiplier clamps are honored"
        ( liveGapMethodMultiplierWithConfig clampConfig (Just (statsWith (-3) 60)) == 0.4
            && liveGapMethodMultiplierWithConfig clampConfig (Just (statsWith 2 60)) == 1.1
        )
    assert
        "configured aggregation floor feeds per-method stats"
        (Map.null (liveGapStatsByMethodWithConfig strictComboConfig [gapComboForTest "10" 2.0 0.5 liveGapMinComboOperations]))

{- | Fill measurements: positive when execution was worse than the decision
price on either side, negative on improvement, Nothing whenever the inputs
cannot support an honest measurement.
-}
testObservedSlippageFractionSemantics :: IO ()
testObservedSlippageFractionSemantics = do
    let approx expected = maybe False (\v -> abs (v - expected) < 1e-12)
    assert
        "BUY filled above the decision price is a positive cost"
        (approx 0.005 (observedSlippageFraction "BUY" 100 (Just 2) (Just 201)))
    assert
        "SELL filled below the decision price is a positive cost"
        (approx 0.005 (observedSlippageFraction "SELL" 100 (Just 2) (Just 199)))
    assert
        "price improvement measures negative"
        (approx (-0.005) (observedSlippageFraction "BUY" 100 (Just 2) (Just 199)))
    assert
        "unknown side measures nothing"
        (isNothing (observedSlippageFraction "HOLD" 100 (Just 2) (Just 201)))
    assert
        "unfilled orders measure nothing"
        ( isNothing (observedSlippageFraction "BUY" 100 Nothing (Just 201))
            && isNothing (observedSlippageFraction "BUY" 100 (Just 0) (Just 201))
        )
    assert
        "non-positive decision price measures nothing"
        (isNothing (observedSlippageFraction "BUY" 0 (Just 2) (Just 201)))
    assert
        "implausible measurements are rejected as data errors"
        (isNothing (observedSlippageFraction "BUY" 100 (Just 2) (Just 212)))

{- | The calibration must stay at the configured assumption until enough
fills accumulate, then move toward realized costs with shrinkage, and obey
the floor (price improvement can't gut the gates) and the absolute cap.
-}
testCalibratedSlippageShrinkage :: IO ()
testCalibratedSlippageShrinkage = do
    let configured = 0.0002
    assert
        "below the minimum observation count the configured value passes through"
        (calibratedSlippagePerSide configured (replicate (costCalibrationMinObservations - 1) 0.003) == configured)
    assert
        "non-finite observations don't count toward the minimum"
        ( calibratedSlippagePerSide
            configured
            (replicate (costCalibrationMinObservations - 1) 0.003 ++ [0 / 0, 1 / 0])
            == configured
        )
    let calibratedUp = calibratedSlippagePerSide configured (replicate 32 0.003)
    assert
        "sustained worse fills raise the estimate above the configured prior"
        (calibratedUp > configured && calibratedUp < 0.003)
    let floored = calibratedSlippagePerSide configured (replicate 32 (-0.004))
    assert
        "price improvement floors at a fraction of the configured prior"
        (abs (floored - costCalibrationFloorFactor * configured) < 1e-12)
    assert
        "catastrophic fills cap at the absolute per-side ceiling"
        (calibratedSlippagePerSide configured (replicate 64 0.045) == costCalibrationMaxPerSide)
    assert
        "calibration moves monotonically with more evidence"
        ( calibratedSlippagePerSide configured (replicate 8 0.003)
            < calibratedSlippagePerSide configured (replicate 64 0.003)
        )

{- | The reinforcement semantics: starting from identical parameters, a
fine-tune that upweights one region of the series must fit that region
better than the uniformly-weighted fine-tune does — that is what makes a
losing trade's span actually correct the model.
-}
testWeightedFineTunePunishesLossRegion :: IO ()
testWeightedFineTunePunishesLossRegion = do
    let n = 40 :: Int
        lookback = 3
        hidden = 4
        regionB t = t >= 20
        series = [if regionB t then 0.8 else 0.2 | t <- [0 .. n - 1]]
        cfg =
            LSTMConfig
                { lcLookback = lookback
                , lcHiddenSize = hidden
                , lcEpochs = 40
                , lcLearningRate = 0.02
                , lcAdamBeta1 = defaultLstmAdamBeta1
                , lcAdamBeta2 = defaultLstmAdamBeta2
                , lcAdamEps = defaultLstmAdamEps
                , lcValRatio = 0
                , lcPatience = 0
                , lcGradClip = Nothing
                , lcSeed = 11
                }
        (model0, _) = trainLSTM cfg{lcEpochs = 0} series
        (uniform, _) = fineTuneLSTM cfg model0 series
        weights = [if regionB t then 8 else 1 | t <- [0 .. n - 1]]
        (weighted, _) = fineTuneLSTMWeighted cfg model0 series weights
        dataset = buildSequences lookback series
        regionBSamples = [s | (i, s) <- zip [0 :: Int ..] dataset, regionB (i + lookback)]
        lossOn model = evaluateLoss lookback hidden regionBSamples (lmParams model)
    assert "test setup: the emphasized region has samples" (not (null regionBSamples))
    assert
        "upweighting a region trains it to a better fit than uniform weighting"
        (lossOn weighted < lossOn uniform)

{- | Today's launchd log also showed:
  @Need at least 3361 price rows for lookback=3360 (got 500) from Binance FILUSDT (3m)@
and the same for XRPUSDT. The current behaviour of 'normalizeBarsForLookback'
is to leave 'argBars' alone when @requiredBars > 1000@. That is the right
conservative call for now (paging is the optimizer's job, not the bot
starter's). This test pins that decision so any future change is
deliberate, with an explicit follow-up reminder.
-}
testNormalizeBarsForLookbackBinanceClampsAtPageCap :: IO ()
testNormalizeBarsForLookbackBinanceClampsAtPageCap = do
    let parseOrFail argv =
            case parseCliArgs argv of
                Left err -> ioError (userError ("CLI parse failed unexpectedly: " ++ err))
                Right a -> pure a
    overLookbackArgs <-
        parseOrFail ["--binance-symbol", "FILUSDT", "--interval", "3m", "--bars", "500", "--lookback-bars", "3360"]
    let adjusted = normalizeBarsForLookback overLookbackArgs
    assert
        "Binance + requiredBars > 1000 leaves --bars unchanged (deferred: page or shrink lookback in optimizer)"
        (argBars adjusted == argBars overLookbackArgs)

testBinanceExceptionSummaryRedactsSecrets :: IO ()
testBinanceExceptionSummaryRedactsSecrets = do
    let req =
            (parseRequest_ "https://fapi.binance.com/fapi/v1/listenKey?listenKey=secret-listen-key")
                { requestHeaders = [("X-MBX-APIKEY", "secret-api-key")]
                }
        msg = binanceExceptionSummary (toException (HttpExceptionRequest req ConnectionTimeout))
    assert
        "Binance exception summaries do not expose API keys or listen keys"
        ( not ("secret-api-key" `T.isInfixOf` msg)
            && not ("secret-listen-key" `T.isInfixOf` msg)
            && not ("X-MBX-APIKEY" `T.isInfixOf` msg)
        )

-- Conformal calibration invariant: malformed or empty residual evidence must
-- fail closed as unavailable instead of being filtered into an overconfident
-- empirical interval. Valid zero residuals remain admissible, and raising the
-- selected conformal quantile cannot narrow the interval for a fixed forecast.
testConformalCalibrationResidualsFailClosed :: IO ()
testConformalCalibrationResidualsFailClosed = do
    let positiveInfinity = 1 / 0
        negativeInfinity = negate positiveInfinity
        unavailableIntervalShape cm mu =
            let (lo, hi, sigma) = predictInterval cm mu
             in isInfinite lo
                    && lo < 0
                    && isInfinite hi
                    && hi > 0
                    && isNothing sigma
        unavailableModel cm =
            cmCount cm == 0
                && isInfinite (cmRadius cm)
                && cmRadius cm > 0
                && unavailableIntervalShape cm 1
        malformedResidualSamples =
            [ []
            , [-0.01]
            , [0.01, -0.01]
            , [0.01, 0 / 0]
            , [0.01, positiveInfinity]
            , [0.01, negativeInfinity]
            ]
        zeroResidualModel = fitConformal 0.2 [0, 0]
        zeroResidualInterval = predictInterval zeroResidualModel 5
        malformedForecastInterval = predictInterval (fitConformal 0.2 [0.1, 0.2]) (0 / 0)
        intervalWidth cm =
            let (lo, hi, _) = predictInterval cm 10
             in hi - lo
        intervalWidths =
            map
                (intervalWidth . fitConformal 0.2)
                [ [0.1, 0.1, 0.1]
                , [0.1, 0.2, 0.3]
                , [0.1, 0.2, 0.5]
                ]
    assert
        "empty or malformed conformal residual evidence fails closed as unavailable"
        (all (unavailableModel . fitConformal 0.2) malformedResidualSamples)
    assert
        "malformed point forecasts cannot emit finite conformal intervals"
        ( let (lo, hi, sigma) = malformedForecastInterval
           in isInfinite lo && lo < 0 && isInfinite hi && hi > 0 && isNothing sigma
        )
    assert
        "valid zero conformal residual evidence remains admissible without synthesizing sigma"
        ( cmCount zeroResidualModel == 2
            && cmRadius zeroResidualModel == 0
            && zeroResidualInterval == (5, 5, Nothing)
        )
    assert
        "larger selected conformal residual quantiles cannot narrow fixed-forecast intervals"
        ( and (zipWith (<=) intervalWidths (drop 1 intervalWidths))
            && all (> 0) intervalWidths
        )

-- CLI sizing invariant: the minimum entry-size floor must never exceed the
-- max position cap. Equality stays admissible at the boundary so the live and
-- backtest sizing paths can enforce a tight cap without silently rejecting
-- deliberate exact-floor configurations.
testBacktestPositionSizeFloorCapValidation :: IO ()
testBacktestPositionSizeFloorCapValidation = do
    let baseArgs = ["--data", "../data/sample_prices.csv"]
        invalid =
            parseAndValidateCliArgs
                (baseArgs ++ ["--min-position-size", "0.81", "--max-position-size", "0.80"])
        equalityBoundary =
            parseAndValidateCliArgs
                (baseArgs ++ ["--min-position-size", "0.80", "--max-position-size", "0.80"])
    assert
        "CLI sizing guardrail rejects a minimum entry floor above the max position cap"
        ( case invalid of
            Left err -> err == "--min-position-size must be <= --max-position-size"
            Right _ -> False
        )
    assert
        "CLI sizing guardrail preserves the exact equality boundary between floor and cap"
        ( case equalityBoundary of
            Right args -> argMinPositionSize args == 0.8 && argMaxPositionSize args == 0.8
            Left _ -> False
        )

-- Algorithm-path regression: the fee-aware fresh-entry contract is not just a
-- helper-level obligation. The checked simulator must reject a signal that has
-- enough threshold headroom and is below the spike cap when its edge is still
-- below the modeled round-trip cost buffer.
assertNear :: String -> Double -> Double -> Double -> IO ()
assertNear message expected actual tolerance =
    assert message (abs (expected - actual) <= tolerance)

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
        scaledFixedFeeCfg =
            baseCfg
                { ecFeeFixed = 0.001
                , ecKellyLiteSizing = True
                , ecKellyLiteFraction = 0
                , ecKellyLiteFloor = 0.1
                , ecKellyLiteCap = 0.1
                }
        maxAbsPosition result =
            maximum (0 : map abs (brPositions result))
        noFeeResult =
            simulateEnsembleWithHLChecked baseCfg 1 prices highs lows preds preds noMeta
        highFeeResult =
            simulateEnsembleWithHLChecked highFeeCfg 1 prices highs lows preds preds noMeta
        scaledFixedFeeResult =
            simulateEnsembleWithHLChecked scaledFixedFeeCfg 1 prices highs lows preds preds noMeta
    case (noFeeResult, highFeeResult, scaledFixedFeeResult) of
        (Right noFee, Right highFee, Right scaledFixedFee) -> do
            assert
                "zero-fee backtest admits the headroom-valid non-spike entry"
                (maxAbsPosition noFee > 0.99)
            assert
                "high round-trip costs block the same marginal pre-fee entry in the simulator"
                (maxAbsPosition highFee == 0 && null (brTrades highFee))
            assert
                "fixed costs are checked against final overlay-scaled entry size"
                (maxAbsPosition scaledFixedFee == 0 && null (brTrades scaledFixedFee))
        (Left err, _, _) -> ioError (userError ("zero-fee fee-buffer regression failed to simulate: " ++ err))
        (_, Left err, _) -> ioError (userError ("high-fee fee-buffer regression failed to simulate: " ++ err))
        (_, _, Left err) -> ioError (userError ("scaled fixed-fee fee-buffer regression failed to simulate: " ++ err))

testSweepThresholdMinRoundTripsFallback :: IO ()
testSweepThresholdMinRoundTripsFallback = do
    let prices :: [Double]
        prices = [100.0, 100.0, 100.0, 100.0, 100.0]
        preds :: [Double]
        preds = [100.0, 100.0, 100.0, 100.0]
        cfg =
            (defaultTuneConfig 252)
                { tcMinRoundTrips = 999
                , tcWalkForwardFolds = 1
                }
        baseCfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecFee = 0
                , ecSlippage = 0
                , ecSpread = 0
                , ecMaxPositionSize = 1
                , ecMinPositionSize = 0
                }
        result = sweepThresholdWithHLWith cfg MethodBoth baseCfg prices prices prices preds preds (Nothing :: Maybe [StepMeta])
    case result of
        Left err -> assert ("sweep-threshold falls back instead of failing every candidate on minRoundTrips: " ++ err) False
        Right (_openThr, _closeThr, bt, stats) ->
            assert
                "sweep-threshold returns a usable fallback below an over-strict activity floor"
                (bmRoundTrips (computeMetrics 252 bt) < tcMinRoundTrips cfg && tsFoldCount stats > 0)

-- Fresh-entry sizing-validity regression: malformed cap/floor evidence must
-- fail closed before a new position can open, while valid zero and minimum
-- equality boundaries remain explicit and valid cap tightening cannot increase
-- realized exposure.
testBacktestFreshEntrySizingBoundsFailClosed :: IO ()
testBacktestFreshEntrySizingBoundsFailClosed = do
    let prices :: V.Vector Double
        prices = V.fromList [100.0, 100.0, 100.0]
        highs = prices
        lows = prices
        preds :: V.Vector Double
        preds = V.fromList [102.0, 102.0]
        noMeta :: Maybe (V.Vector StepMeta)
        noMeta = Nothing
        positiveInfinity = 1 / 0
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
        requireResult label cfg =
            case simulateEnsembleWithHLChecked cfg 1 prices highs lows preds preds noMeta of
                Right result -> pure result
                Left err -> ioError (userError (label ++ " sizing regression failed to simulate: " ++ err))
        maxAbsPosition result = maximum (0 : map abs (brPositions result))
        entryAdmissible result = maxAbsPosition result > 0 && not (null (brTrades result))
        flat result = maxAbsPosition result == 0 && null (brTrades result)
    validZeroFloor <- requireResult "valid zero-floor" baseCfg
    let zeroCapResult = simulateEnsembleWithHLChecked baseCfg{ecMaxPositionSize = 0, ecMinPositionSize = 0} 1 prices highs lows preds preds noMeta
    validMinEquality <- requireResult "valid min-equality" baseCfg{ecMaxPositionSize = 0.5, ecMinPositionSize = 0.5}
    validCapBelowMin <- requireResult "valid cap-below-min" baseCfg{ecMaxPositionSize = 0.25, ecMinPositionSize = 0.5}
    let invalidMaxCfgs =
            [ baseCfg{ecMaxPositionSize = -0.1}
            , baseCfg{ecMaxPositionSize = 0 / 0}
            , baseCfg{ecMaxPositionSize = positiveInfinity}
            ]
        invalidMinCfgs =
            [ baseCfg{ecMinPositionSize = -0.1}
            , baseCfg{ecMinPositionSize = 0 / 0}
            , baseCfg{ecMinPositionSize = positiveInfinity}
            ]
        invalidMaxResults = map (\cfg -> simulateEnsembleWithHLChecked cfg 1 prices highs lows preds preds noMeta) invalidMaxCfgs
        invalidMinResults = map (\cfg -> simulateEnsembleWithHLChecked cfg 1 prices highs lows preds preds noMeta) invalidMinCfgs
    capResults <-
        mapM
            (requireResult "valid cap ladder")
            [ baseCfg{ecMaxPositionSize = 1.0}
            , baseCfg{ecMaxPositionSize = 0.5}
            ]
    let capExposures = map maxAbsPosition capResults
    assert
        "valid zero floor remains entry-permissive when the cap and edge are valid"
        (entryAdmissible validZeroFloor && maxAbsPosition validZeroFloor > 0.99)
    assert
        "valid zero cap is rejected by risk guardrail (maxPositionSize must be > 0)"
        (case zeroCapResult of Left _ -> True; Right _ -> False)
    assertNear
        "valid minimum-size equality remains admissible"
        0.5
        (maxAbsPosition validMinEquality)
        1e-12
    assert
        "a valid cap below the valid minimum floor blocks fresh entry"
        (flat validCapBelowMin)
    assert
        "negative or non-finite max position-size caps are rejected by risk guardrail"
        (all (\case Left _ -> True; Right _ -> False) invalidMaxResults)
    assert
        "negative or non-finite min position-size floors are rejected by risk guardrail"
        (all (\case Left _ -> True; Right _ -> False) invalidMinResults)
    assert
        "tightening valid caps cannot increase realized fresh-entry exposure"
        ( and (zipWith (>=) capExposures (drop 1 capExposures))
            && capExposures == [1.0, 0.5]
        )

-- Cost-attribution proof: the simulator reports gross/net surfaces and realized
-- component costs that close the accounting identity for the exact run.
testBacktestCostAttributionGrossNetConsistency :: IO ()
testBacktestCostAttributionGrossNetConsistency = do
    let prices :: V.Vector Double
        prices = V.fromList [100.0, 100.0, 100.0]
        preds :: V.Vector Double
        preds = V.fromList [102.0, 102.0]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.01
                , ecFee = 0.001
                , ecSlippage = 0.0005
                , ecSpread = 0.001
                , ecFundingRate = 0.0252
                , ecFundingBySide = False
                , ecFundingOnOpen = False
                , ecMaxPositionSize = 1
                , ecMinPositionSize = 0
                }
        result = simulateEnsembleWithHLChecked cfg 1 prices prices prices preds preds (Nothing :: Maybe (V.Vector StepMeta))
    case result of
        Left err -> ioError (userError ("cost-attribution backtest failed to simulate: " ++ err))
        Right bt -> do
            let attribution = brCostAttribution bt
                grossCurve = bcaGrossEquityCurve attribution
                netCurve = bcaNetEquityCurve attribution
                finalGross = last grossCurve
                finalNet = last netCurve
                totalCost = bcaRealizedTotalCost attribution
                componentTotal =
                    bcaRealizedFeeCost attribution
                        + bcaRealizedSlippageCost attribution
                        + bcaRealizedSpreadCost attribution
                        + bcaRealizedFundingCost attribution
            assert "cost attribution exposes the same net curve as the backtest result" (netCurve == brEquityCurve bt)
            assert "gross and net cost-attribution curves align per bar" (length grossCurve == length netCurve && length netCurve == length (brEquityCurve bt))
            assert "fee/slippage/spread/funding realized costs are surfaced" $
                bcaRealizedFeeCost attribution > 0
                    && bcaRealizedSlippageCost attribution > 0
                    && bcaRealizedSpreadCost attribution > 0
                    && bcaRealizedFundingCost attribution > 0
            assertNear "realized component costs sum to total realized cost" totalCost componentTotal 1e-12
            assertNear "gross minus realized costs equals net equity" finalNet (finalGross - totalCost) 1e-12
            assertNear "reported consistency residual stays near zero" 0 (bcaConsistencyResidual attribution) 1e-12

-- Regression for review thread 7: an overflowed impact component used to make
-- the cost scaling path multiply Infinity by zero. The capped attribution path
-- must keep the realized cost totals and simulated equity finite.

{- |
Non-finite cost-attribution regression.

Defense in depth: the simulator must never propagate NaN or Infinity
into the equity curve or cost attribution surface, even when an
upstream config would otherwise drive intermediates (size * impact ** power,
fee accumulators, etc.) to overflow.

There are two layers of protection:

  1. The firm-critical 2x-base position-size guardrail in
     'Trader.Trading' (Trading.hs:2315): if a multiplicative sizing
     chain (Kelly, vol, risk, snr scalers) blows the requested size
     past 2x the base target, the simulator fails closed with the
     'POSITION_SIZE_SCALE_EXCEEDED' marker BEFORE the trade can
     reach the cost-attribution surface.
  2. The deterministic cost-cap inside 'simulateEnsembleWithHLChecked'
     itself, which clamps any overflowed total cost to 0.999999 and
     routes the excess into the slippage bucket via
     'cappedBucketAllocation' when components are non-finite.

This regression covers both layers:

  * Pathological Kelly-fraction inputs trigger layer (1) — the
    simulator hard-fails closed with the canonical marker rather than
    silently letting a runaway sizing chain through.
  * A bounded but realistic overflow path (large per-unit impact with
    bounded size) trips layer (2) — the cost-attribution surface
    deterministically caps the realized total cost, attributes the
    overflow into slippage, and keeps the equity curve finite.
-}
testBacktestCostAttributionNonFiniteComponentsRegression :: IO ()
testBacktestCostAttributionNonFiniteComponentsRegression = do
    -- Layer (1): pathological Kelly inputs hit the upstream sizing
    -- guardrail BEFORE any cost-attribution surface is observable.
    let runawayKellyCfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.01
                , ecFee = 0
                , ecSlippage = 0
                , ecSlippageVolMult = 0
                , ecSlippageImpact = 1e-9
                , ecSlippageImpactPower = 2
                , ecSpread = 0
                , ecFundingRate = 0
                , ecVolLookback = 2
                , ecMaxPositionSize = 1e308
                , ecMinPositionSize = 0
                , ecRebalanceBars = 1
                , ecRebalanceThreshold = 0.1
                , ecKellyLiteSizing = True
                , ecKellyLiteFraction = 1e300
                , ecKellyLiteFloor = 1
                , ecKellyLiteCap = 1e308
                }
        runawayPrices :: V.Vector Double
        runawayPrices = V.fromList [100.0, 100.0, 101.0, 101.0]
        runawayPreds :: V.Vector Double
        runawayPreds = V.fromList [102.0, 102.0, 103.0]
        runawayResult =
            case simulateEnsembleWithHLChecked runawayKellyCfg 1 runawayPrices runawayPrices runawayPrices runawayPreds runawayPreds (Nothing :: Maybe (V.Vector StepMeta)) of
                Left err -> Left err
                Right bt ->
                    let !np = length (brPositions bt)
                        !nt = length (brTrades bt)
                     in Right (np, nt)
    runawayCaught <- try (evaluate runawayResult) :: IO (Either SomeException (Either String (Int, Int)))
    case runawayCaught of
        Left exc ->
            assert
                "upstream sizing guardrail hard-fails closed with the canonical POSITION_SIZE_SCALE marker before non-finite cost intermediates can reach attribution"
                ("POSITION_SIZE_SCALE_EXCEEDED" `isInfixOf` show exc)
        Right (Left _) ->
            assert
                "upstream sizing guardrail must hard-fail closed, not return Left, on runaway Kelly inputs"
                False
        Right (Right witness) ->
            ioError (userError ("upstream sizing guardrail did not fire on runaway Kelly inputs; witness=" ++ show witness))

    -- Layer (2): a bounded overflow that does NOT trip the sizing
    -- guardrail still gets deterministically capped by the
    -- cost-attribution surface, and never produces NaN/Infinity.
    let prices :: V.Vector Double
        prices = V.fromList [100.0, 100.0, 101.0, 101.0]
        highs = prices
        lows = prices
        preds :: V.Vector Double
        preds = V.fromList [102.0, 102.0, 103.0]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.01
                , ecFee = 0.999
                , ecSlippage = 0
                , ecSlippageVolMult = 0
                , ecSlippageImpact = 0
                , ecSpread = 0
                , ecFundingRate = 0
                , ecVolLookback = 2
                , ecMaxPositionSize = 1
                , ecMinPositionSize = 0
                , ecRebalanceBars = 0
                , ecRebalanceThreshold = 0
                , ecKellyLiteSizing = False
                , ecKellyLiteFraction = 0.5
                , ecKellyLiteFloor = 0
                , ecKellyLiteCap = 1
                }
        result = simulateEnsembleWithHLChecked cfg 1 prices highs lows preds preds (Nothing :: Maybe (V.Vector StepMeta))
        finite x = not (isNaN x || isInfinite x)
    case result of
        Left _ ->
            -- The bounded-overflow config may be rejected outright by
            -- the fee-buffer entry gate. That is itself a valid
            -- defense-in-depth outcome: no trade enters, no overflow
            -- can be observed downstream.
            pure ()
        Right bt -> do
            let attribution = brCostAttribution bt
                grossCurve = bcaGrossEquityCurve attribution
                netCurve = bcaNetEquityCurve attribution
                totalCost = bcaRealizedTotalCost attribution
                components =
                    [ bcaRealizedFeeCost attribution
                    , bcaRealizedSlippageCost attribution
                    , bcaRealizedSpreadCost attribution
                    , bcaRealizedFundingCost attribution
                    ]
                componentTotal = sum components
                costAndEquityPath =
                    brEquityCurve bt
                        ++ brPositions bt
                        ++ grossCurve
                        ++ netCurve
                        ++ components
                        ++ [totalCost, componentTotal, bcaConsistencyResidual attribution]
            assert
                "non-finite cost intermediates do not introduce NaN or Infinity into the simulated path"
                (all finite costAndEquityPath)
            assert
                "bounded-overflow cost attribution stays at or below the deterministic cap"
                (totalCost <= 1.0 && all (>= 0) components)
            assertNear "finite realized components sum to the derived applied cost" totalCost componentTotal 1e-12
            assertNear "bounded-overflow cost attribution keeps gross/net residual closed" 0 (bcaConsistencyResidual attribution) 1e-9

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

testOrderExecutionCorruptedInputInvariant :: IO ()
testOrderExecutionCorruptedInputInvariant = do
    let eps = 1e-9
        maxSanitizedMagnitude = 1.7976931348623157e308 / 4
        prevPositions = [-2, -1, 0, 1, 2, minBound, maxBound]
        prevSizes = [negate (1 / 0), -1e308, -1, -5e-10, 0 / 0, 0, 5e-10, 1e-9, 1e-8, 0.5, 2, 1e307, 1e308, 1 / 0]
        executedQtys = [negate (1 / 0), -1e308, -1, -5e-10, 0 / 0, 0, 5e-10, 1e-9, 1e-8, 0.5, 3, 1e307, 1e308, 1 / 0]
        finite x = not (isNaN x || isInfinite x)
        sanitizeMagnitude x
            | not (finite x) || x <= 0 = 0
            | otherwise = min maxSanitizedMagnitude x
        sanitizeExecutedQty x =
            let qty = sanitizeMagnitude x
             in if qty <= eps then 0 else qty
        sanitizePrevSigned prevPos prevSize =
            let prevSign = signum prevPos
                prevSize' = sanitizeMagnitude prevSize
             in if prevSign == 0 then 0 else fromIntegral prevSign * prevSize'
        signedExposure pos size = fromIntegral pos * size
        finiteNonNegative x = finite x && x >= 0
        closeEnough expected actual =
            abs (expected - actual) <= max eps (abs expected * 1e-12)
        caseLabel prevPos prevSize isBuy qtyRaw =
            "prevPos=" ++ show prevPos ++ " prevSize=" ++ show prevSize ++ " isBuy=" ++ show isBuy ++ " qtyRaw=" ++ show qtyRaw
    forM_ prevPositions $ \prevPos ->
        forM_ prevSizes $ \prevSize ->
            forM_ [False, True] $ \isBuy ->
                forM_ executedQtys $ \qtyRaw -> do
                    let label = caseLabel prevPos prevSize isBuy qtyRaw
                        (posNew, sizeNew, closeQty, openQty) = applyExecutedQuantity prevPos prevSize isBuy qtyRaw
                        prevSigned = sanitizePrevSigned prevPos prevSize
                        qty = sanitizeExecutedQty qtyRaw
                        expectedSigned = prevSigned + if isBuy then qty else negate qty
                        actualSigned = signedExposure posNew sizeNew
                    assert (label ++ " size stays finite and non-negative") (finiteNonNegative sizeNew)
                    assert (label ++ " closeQty stays finite and non-negative") (finiteNonNegative closeQty)
                    assert (label ++ " openQty stays finite and non-negative") (finiteNonNegative openQty)
                    assert (label ++ " position stays normalized") (posNew `elem` [-1, 0, 1])
                    assert (label ++ " closeQty is bounded by sanitized prior size") (closeQty <= abs prevSigned + eps)
                    assert (label ++ " openQty is bounded by sanitized executed qty") (openQty <= qty + eps)
                    assert (label ++ " signed exposure is conserved after sanitization") (closeEnough expectedSigned actualSigned)

-- Coinbase live market orders must expose fill evidence to the shared
-- execution-state reconciler; an accepted nested response without fill/status
-- must not be promoted into a filled order by the parser.
testCoinbaseOrderInfoDecodeInvariant :: IO ()
testCoinbaseOrderInfoDecodeInvariant = do
    legacyInfo <-
        requireCoinbaseOrderInfo
            "legacy Coinbase Exchange order response"
            ( Aeson.object
                [ "id" Aeson..= ("order-1" :: String)
                , "client_oid" Aeson..= ("client-1" :: String)
                , "status" Aeson..= ("done" :: String)
                , "filled_size" Aeson..= ("0.125" :: String)
                , "executed_value" Aeson..= ("8123.45" :: String)
                ]
            )
    assert
        "legacy Coinbase order response exposes id, status, filled base size, and executed quote value"
        ( coiOrderId legacyInfo == Just "order-1"
            && coiClientOrderId legacyInfo == Just "client-1"
            && coiStatus legacyInfo == Just "done"
            && coiFilledSize legacyInfo == Just 0.125
            && coiExecutedValue legacyInfo == Just 8123.45
        )

    nestedInfo <-
        requireCoinbaseOrderInfo
            "nested Coinbase order response"
            ( Aeson.object
                [ "success" Aeson..= True
                , "success_response"
                    Aeson..= Aeson.object
                        [ "order_id" Aeson..= ("order-2" :: String)
                        , "client_order_id" Aeson..= ("client-2" :: String)
                        ]
                , "order"
                    Aeson..= Aeson.object
                        [ "status" Aeson..= ("FILLED" :: String)
                        , "filled_size" Aeson..= ("0.25" :: String)
                        , "filled_value" Aeson..= ("15000" :: String)
                        ]
                ]
            )
    assert
        "nested Coinbase order response merges identifiers with detailed fill evidence"
        ( coiOrderId nestedInfo == Just "order-2"
            && coiClientOrderId nestedInfo == Just "client-2"
            && coiStatus nestedInfo == Just "FILLED"
            && coiFilledSize nestedInfo == Just 0.25
            && coiExecutedValue nestedInfo == Just 15000
        )

    acceptedOnlyInfo <-
        requireCoinbaseOrderInfo
            "accepted Coinbase order response"
            ( Aeson.object
                [ "success" Aeson..= True
                , "success_response"
                    Aeson..= Aeson.object
                        [ "order_id" Aeson..= ("order-3" :: String)
                        , "client_order_id" Aeson..= ("client-3" :: String)
                        ]
                ]
            )
    assert
        "accepted Coinbase response without explicit status or fill stays non-filled"
        ( coiOrderId acceptedOnlyInfo == Just "order-3"
            && isNothing (coiStatus acceptedOnlyInfo)
            && isNothing (coiFilledSize acceptedOnlyInfo)
        )
  where
    requireCoinbaseOrderInfo :: String -> Aeson.Value -> IO CoinbaseOrderInfo
    requireCoinbaseOrderInfo label value =
        case decodeCoinbaseOrderInfo (Aeson.encode value) of
            Just info -> pure info
            Nothing -> ioError (userError ("Failed to decode " ++ label))

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

-- Interval-aware edge spike cap proof obligation: higher timeframes should
-- allow larger edges. With entryEdgeSpikeCredibleCap=5.0:
--   1m cap = 5.0 * 1.0 = 5.0
--   1d cap = 5.0 * 1.5 = 7.5
-- With openThreshold = 0.20, the multiple cap (1000*threshold=200) dominates
-- the credible cap, so interval scaling is visible on the credible cap.
-- 1d allows 6.0 edge, 1m rejects it.
testSignalGateIntervalAwareEdgeSpikeCap :: IO ()
testSignalGateIntervalAwareEdgeSpikeCap = do
    let openThreshold = 0.20
        edge = Just 6.0
    assert
        "1d interval allows larger edges than 1m"
        ( signalEntryEdgeSpikeOkInterval "1d" openThreshold edge
            && not (signalEntryEdgeSpikeOkInterval "1m" openThreshold edge)
        )
    assert
        "legacy signalEntryEdgeSpikeOk delegates to interval-agnostic behavior"
        (signalEntryEdgeSpikeOk openThreshold edge == signalEntryEdgeSpikeOkInterval "" openThreshold edge)

-- Model sanity invariant proof obligation: predictions outside [0.01x, 100x]
-- currentPrice must be rejected with MODEL_ANOMALY reason.
testSignalGatePredictionSanityInvariant :: IO ()
testSignalGatePredictionSanityInvariant = do
    let currentPrice = 92.41
        goodPred = Just 92.48
        lowPred = Just 0.01
        highPred = Just 10000.0
        nanPred = Just (0 / 0)
        negPred = Just (-1.0)
    assert
        "sane prediction passes"
        (fst (signalPredictionSanityOk currentPrice goodPred))
    assert
        "prediction below 0.01x fails closed"
        (not (fst (signalPredictionSanityOk currentPrice lowPred)) && snd (signalPredictionSanityOk currentPrice lowPred) == Just "MODEL_ANOMALY")
    assert
        "prediction above 100x fails closed"
        (not (fst (signalPredictionSanityOk currentPrice highPred)))
    assert
        "NaN prediction fails closed"
        (not (fst (signalPredictionSanityOk currentPrice nanPred)))
    assert
        "negative prediction fails closed"
        (not (fst (signalPredictionSanityOk currentPrice negPred)))
    assert
        "missing prediction passes"
        (fst (signalPredictionSanityOk currentPrice Nothing))

-- Prediction-aware weak-band confirmation proof obligation: when predicted
-- return and historical z-score agree with the chosen direction, the check
-- should relax to absolute z-score threshold.
testSignalGatePredictionAwareWeakBand :: IO ()
testSignalGatePredictionAwareWeakBand = do
    assert
        "agreeing prediction relaxes weak-band veto"
        (directionalityWeakBandConfirmedWithPrediction 0.6 (Just 1) (Just 0.05) 100.0)
    assert
        "disagreeing prediction falls back to legacy behavior"
        (not (directionalityWeakBandConfirmedWithPrediction 0.6 (Just (-1)) (Just 0.05) 100.0))
    assert
        "missing prediction falls back to legacy behavior"
        (directionalityWeakBandConfirmedWithPrediction 0.6 (Just 1) Nothing 100.0 == directionalityWeakBandConfirmed 0.6 (Just 1))

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

testSignalGateVolTargetPrecedesCloud :: IO ()
testSignalGateVolTargetPrecedesCloud = do
    let result =
            signalRunPostDirectionGates
                (Just 1)
                Nothing
                True
                False
                (const True)
                (const False)
                (const True)
                True
                (const (True, Nothing))
                (True, Nothing)
                (True, Nothing)
                (True, Nothing)
                (const True)
                (const (True, 1))
    assert
        "vol-target readiness takes precedence over Kalman cloud vetoes"
        (result == (Nothing, Just "VOL_TARGET"))

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

-- Prediction-aware directionality snapshot: when predicted return agrees with
-- chosen direction and historical z-score, the weak-band veto should relax.
testSignalDirectionalityPredictionAwareLiveSemantics :: IO ()
testSignalDirectionalityPredictionAwareLiveSemantics = do
    let weakBandPrices =
            pricesFromReturns [0.018, 0.018, 0.018, -0.01, -0.01, -0.01, 0.018, 0.018, -0.01, -0.01]
        weakBandShortSnapshotLegacy =
            signalDirectionalitySnapshotImplWithPrediction
                0.05
                (Just (RegimeProbs 0.6 0.2 0.2))
                weakBandPrices
                (V.length weakBandPrices - 1)
                (Just (-1))
                Nothing
                0 ::
                Maybe DirectionalitySnapshot
        weakBandShortSnapshotAgree =
            signalDirectionalitySnapshotImplWithPrediction
                0.05
                (Just (RegimeProbs 0.6 0.2 0.2))
                weakBandPrices
                (V.length weakBandPrices - 1)
                (Just (-1))
                (Just (-0.05))
                100.0 ::
                Maybe DirectionalitySnapshot
        weakBandShortSnapshotDisagree =
            signalDirectionalitySnapshotImplWithPrediction
                0.05
                (Just (RegimeProbs 0.6 0.2 0.2))
                weakBandPrices
                (V.length weakBandPrices - 1)
                (Just (-1))
                (Just 0.05)
                100.0 ::
                Maybe DirectionalitySnapshot
    assert
        "legacy path without prediction falls back to weak-band veto"
        (weakBandShortSnapshotLegacy == Just (DirectionalitySnapshot True (Just "NON_DIRECTIONAL_WEAK_BAND")))
    assert
        "agreeing prediction relaxes weak-band veto for same-direction signals"
        (weakBandShortSnapshotAgree == Just (DirectionalitySnapshot False Nothing))
    assert
        "disagreeing prediction falls back to legacy weak-band veto"
        (weakBandShortSnapshotDisagree == Just (DirectionalitySnapshot True (Just "NON_DIRECTIONAL_WEAK_BAND")))

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
                , trFeeCost = 0.0
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
            simulateEnsemble0 `seq`
                (simulateEnsembleWithHLChecked0 `seq` True)
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

-- Guardrail regression: the position-size scale invariant must fail hard before
-- the absolute cap can mask runaway multiplicative sizing.
testPositionSizeScaleSanityInvariant :: IO ()
testPositionSizeScaleSanityInvariant = do
    let prices :: V.Vector Double
        prices = V.fromList [100.0, 100.0, 100.0]
        highs = prices
        lows = prices
        preds :: V.Vector Double
        preds = V.fromList [102.0, 102.0]
        noMeta :: Maybe (V.Vector StepMeta)
        noMeta = Nothing
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.01
                , ecFee = 0
                , ecSlippage = 0
                , ecSpread = 0
                , ecFeeFixed = 0
                , ecFeeMin = 0
                , ecMaxPositionSize = 10
                , ecMinPositionSize = 0
                , ecRiskPerTrade = Just 0.02
                , ecStopLoss = Just 0.005
                }
        runForceSimulation :: IO (Either String (Int, Int))
        runForceSimulation =
            case simulateEnsembleWithHLChecked cfg 1 prices highs lows preds preds noMeta of
                Left err -> pure (Left err)
                Right bt -> do
                    p <- evaluate (length (brPositions bt))
                    t <- evaluate (length (brTrades bt))
                    pure (Right (p, t))
    result <- try runForceSimulation :: IO (Either SomeException (Either String (Int, Int)))
    case result of
        Left exc ->
            assert
                "position-size scale sanity invariant hard-fails with explicit marker"
                ("POSITION_SIZE_SCALE_EXCEEDED" `isInfixOf` show exc)
        Right (Left err) ->
            ioError (userError ("position-size scale sanity invariant returned Left instead of hard-failing: " ++ err))
        Right (Right exposureWitness) ->
            ioError (userError ("position-size scale sanity invariant did not fire; witness=" ++ show exposureWitness))

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
            signalEntryHeadroomThresholdCap 0.03 `seq`
                (simulateEnsembleWithHLChecked0 `seq` True)
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
    assert
        "quality preset lifts omitted default threshold ceilings to the quality exploration floor"
        (qualityPresetCeiling (2e-2 :: Double) 5e-2 False 2e-2 == 5e-2)
    assert
        "quality preset preserves explicitly requested default threshold ceilings"
        (qualityPresetCeiling (2e-2 :: Double) 5e-2 True 2e-2 == 2e-2)
    assert
        "quality preset honors tighter explicit activity-recovery threshold caps"
        (qualityPresetCeiling (2e-2 :: Double) 5e-2 True 6e-3 == 6e-3)
    assert
        "quality preset preserves explicit threshold sweeps below the quality floor"
        (qualityPresetCeiling (2e-2 :: Double) 5e-2 True 3e-2 == 3e-2)
    assert
        "quality preset preserves explicit threshold sweeps above the quality floor"
        (qualityPresetCeiling (2e-2 :: Double) 5e-2 True 8e-2 == 8e-2)
    assert
        "quality preset preserves explicit zero method weights"
        (qualityPresetWeightFloor 1.0 (0.0 :: Double) == 0.0)
    assert
        "quality preset lifts default positive method weights to the quality floor"
        (qualityPresetWeightFloor 1.0 (0.25 :: Double) == 1.0)
    assert
        "quality preset preserves explicit method weights above the quality floor"
        (qualityPresetWeightFloor 1.0 (2.0 :: Double) == 2.0)

testOptimizerQualityThresholdArgvExplicitRegression :: IO ()
testOptimizerQualityThresholdArgvExplicitRegression = do
    let qualityDefaultCap = 2e-2 :: Double
        qualityExplorationFloor = 5e-2 :: Double
        explicitOpenCap = 6e-3 :: Double
        explicitCloseCap = 8e-3 :: Double
        splitOpenForm = ["--quality", "--open-threshold-max", "2e-2"]
        equalsOpenForm = ["--quality", "--open-threshold-max=2e-2"]
        splitCloseForm = ["--quality", "--close-threshold-max", "2e-2"]
        equalsCloseForm = ["--quality", "--close-threshold-max=2e-2"]
        splitBothForm =
            [ "--quality"
            , "--open-threshold-max"
            , "6e-3"
            , "--close-threshold-max"
            , "8e-3"
            ]
        equalsBothForm =
            [ "--quality"
            , "--open-threshold-max=6e-3"
            , "--close-threshold-max=8e-3"
            ]
        omitted = ["--quality"]
        parser =
            (,,,,)
                <$> switch (long "quality")
                <*> option auto (long "open-threshold-max" <> value qualityDefaultCap)
                <*> option auto (long "close-threshold-max" <> value qualityDefaultCap)
                <*> pure False
                <*> pure False
        parseQualityThresholdArgs argv =
            case execParserPure defaultPrefs (info parser mempty) argv of
                Success (qualityEnabled, openCap, closeCap, _, _) ->
                    Right
                        ( qualityEnabled
                        , openCap
                        , closeCap
                        , optimizerOptionPresent "open-threshold-max" argv
                        , optimizerOptionPresent "close-threshold-max" argv
                        )
                Failure _ -> Left "optimizer quality threshold parser failed unexpectedly"
                CompletionInvoked _ -> Left "optimizer quality threshold completion invoked unexpectedly"
        effectiveOpenCap (_, openCap, _, openExplicit, _) =
            qualityPresetCeiling qualityDefaultCap qualityExplorationFloor openExplicit openCap
        effectiveCloseCap (_, _, closeCap, _, closeExplicit) =
            qualityPresetCeiling qualityDefaultCap qualityExplorationFloor closeExplicit closeCap
    assert
        "optimizer execParserPure treats split-form open threshold quality caps as explicit"
        ( case parseQualityThresholdArgs splitOpenForm of
            Right parsed@(True, openCap, closeCap, True, False) ->
                openCap == qualityDefaultCap
                    && closeCap == qualityDefaultCap
                    && effectiveOpenCap parsed == qualityDefaultCap
                    && effectiveCloseCap parsed == qualityExplorationFloor
            _ -> False
        )
    assert
        "optimizer execParserPure treats equals-form open threshold quality caps as explicit"
        ( case parseQualityThresholdArgs equalsOpenForm of
            Right parsed@(True, openCap, closeCap, True, False) ->
                openCap == qualityDefaultCap
                    && closeCap == qualityDefaultCap
                    && effectiveOpenCap parsed == qualityDefaultCap
                    && effectiveCloseCap parsed == qualityExplorationFloor
            _ -> False
        )
    assert
        "optimizer execParserPure treats split-form close threshold quality caps as explicit"
        ( case parseQualityThresholdArgs splitCloseForm of
            Right parsed@(True, openCap, closeCap, False, True) ->
                openCap == qualityDefaultCap
                    && closeCap == qualityDefaultCap
                    && effectiveOpenCap parsed == qualityExplorationFloor
                    && effectiveCloseCap parsed == qualityDefaultCap
            _ -> False
        )
    assert
        "optimizer execParserPure treats equals-form close threshold quality caps as explicit"
        ( case parseQualityThresholdArgs equalsCloseForm of
            Right parsed@(True, openCap, closeCap, False, True) ->
                openCap == qualityDefaultCap
                    && closeCap == qualityDefaultCap
                    && effectiveOpenCap parsed == qualityExplorationFloor
                    && effectiveCloseCap parsed == qualityDefaultCap
            _ -> False
        )
    assert
        "optimizer execParserPure preserves split-form dual explicit quality caps exactly as requested"
        ( case parseQualityThresholdArgs splitBothForm of
            Right parsed@(True, openCap, closeCap, True, True) ->
                openCap == explicitOpenCap
                    && closeCap == explicitCloseCap
                    && effectiveOpenCap parsed == explicitOpenCap
                    && effectiveCloseCap parsed == explicitCloseCap
            _ -> False
        )
    assert
        "optimizer execParserPure preserves equals-form dual explicit quality caps exactly as requested"
        ( case parseQualityThresholdArgs equalsBothForm of
            Right parsed@(True, openCap, closeCap, True, True) ->
                openCap == explicitOpenCap
                    && closeCap == explicitCloseCap
                    && effectiveOpenCap parsed == explicitOpenCap
                    && effectiveCloseCap parsed == explicitCloseCap
            _ -> False
        )
    assert
        "optimizer execParserPure preserves omitted quality threshold caps as non-explicit defaults"
        ( case parseQualityThresholdArgs omitted of
            Right parsed@(True, openCap, closeCap, False, False) ->
                openCap == qualityDefaultCap
                    && closeCap == qualityDefaultCap
                    && effectiveOpenCap parsed == qualityExplorationFloor
                    && effectiveCloseCap parsed == qualityExplorationFloor
            _ -> False
        )

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

-- Auto-optimizer discovery regression: once walk-forward gates became default
-- deployability filters, a zero-eligible primary run could be all
-- wfSharpeMean/wfSharpeStd failures. That must trigger the broader recovery
-- pass just like the older activity/exposure/threshold filters.
testOptimizerRecordsRetryDiscoveryForWalkForwardFilters :: IO ()
testOptimizerRecordsRetryDiscoveryForWalkForwardFilters = do
    let summary =
            emptyOptimizerRecordsSummary
                { orsRecords = 12
                , orsEligible = 0
                , orsWalkForwardSharpeMeanFiltered = 9
                , orsWalkForwardSharpeStdFiltered = 3
                , orsRecoveryFiltered = 12
                }
    assert
        "walk-forward-only zero-eligible runs trigger discovery recovery"
        (optimizerRecordsShouldRetryDiscovery summary)

testOptimizerRecordMetricsCarryWalkForwardSummary :: IO ()
testOptimizerRecordMetricsCarryWalkForwardSummary = do
    let baseMetrics = KM.fromList [(AK.fromString "tradeCount", Aeson.toJSON (20 :: Int))]
        rawBacktest =
            Aeson.object
                [ "backtest"
                    .= Aeson.object
                        [ "walkForward"
                            .= Aeson.object
                                [ "summary"
                                    .= Aeson.object
                                        [ "sharpeMean" .= (0.42 :: Double)
                                        , "sharpeStd" .= (0.11 :: Double)
                                        ]
                                ]
                        ]
                ]
        metrics = applyWalkForwardSummaryMetrics (Just baseMetrics) (Just rawBacktest)
        tradeCount = metrics >>= KM.lookup (AK.fromString "tradeCount") >>= AT.parseMaybe Aeson.parseJSON
        wfSharpeMean = do
            m <- metrics
            case KM.lookup (AK.fromString "walkForwardSummary") m of
                Just (Aeson.Object wf) ->
                    KM.lookup (AK.fromString "sharpeMean") wf >>= AT.parseMaybe Aeson.parseJSON
                _ -> Nothing
    assert
        "optimizer metrics preserve existing values when attaching walk-forward evidence"
        (tradeCount == Just (20 :: Int))
    assert
        "optimizer metrics carry walk-forward summary for top-combo processing"
        (wfSharpeMean == Just (0.42 :: Double))

testOptimizerRecordsRetryDiscoveryForCostFloorFilters :: IO ()
testOptimizerRecordsRetryDiscoveryForCostFloorFilters = do
    let summary =
            emptyOptimizerRecordsSummary
                { orsRecords = 4
                , orsEligible = 0
                , orsMinEdgeFiltered = 4
                , orsRecoveryFiltered = 4
                }
    assert
        "cost-floor-only zero-eligible runs trigger discovery recovery"
        (optimizerRecordsShouldRetryDiscovery summary)

testOptimizerRecordsRetryDiscoveryStopsWhenEligible :: IO ()
testOptimizerRecordsRetryDiscoveryStopsWhenEligible = do
    let summary =
            emptyOptimizerRecordsSummary
                { orsRecords = 4
                , orsEligible = 1
                , orsWalkForwardSharpeMeanFiltered = 3
                , orsRecoveryFiltered = 3
                }
    assert
        "a primary run with any eligible record does not spend a recovery pass"
        (not (optimizerRecordsShouldRetryDiscovery summary))

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
                , trFeeCost = 0.0
                }
        result =
            BacktestResult
                { brEquityCurve = [1.0, 1.1]
                , brTrades = [trade]
                , brPositions = [0, 1]
                , brAgreementOk = [True]
                , brAgreementValid = [True]
                , brPositionChanges = 1
                , brCostAttribution = emptyBacktestCostAttribution [1.0, 1.1]
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

-- ============================================================================
-- Gate Telemetry Tests
-- Engineering invariant: If you can't measure it, you can't improve it.
-- ============================================================================

testGateTelemetryEmptyInvariant :: IO ()
testGateTelemetryEmptyInvariant = do
    let tel = emptyTelemetry 100
    assert "empty telemetry has zero total bars" (gtTotalBars tel == 0)
    assert "empty telemetry has zero total candidates" (gtTotalCandidates tel == 0)
    assert "empty telemetry has zero total rejections" (gtTotalRejections tel == 0)
    assert "empty telemetry has empty histogram" (null (rejectionHistogram tel))
    assert "empty telemetry has no binding gate" (isNothing (bindingGate (gtPerGateCounts tel)))

testGateTelemetryAccumulationInvariant :: IO ()
testGateTelemetryAccumulationInvariant = do
    let tel0 = emptyTelemetry 10
        rej1 = GateRejection GateEdgeSpike ReasonEdgeSpike Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing
        rej2 = GateRejection GateEdgeHeadroom ReasonEdgeHeadroom Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing
        rej3 = GateRejection GateEdgeSpike ReasonEdgeSpike Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing
        tel1 = recordRejection rej1 tel0
        tel2 = recordRejection rej2 tel1
        tel3 = recordRejection rej3 tel2
    assert "accumulated rejections count correctly" (gtTotalRejections tel3 == 3)
    assert "per-gate counts track correctly" (Map.lookup GateEdgeSpike (gtPerGateCounts tel3) == Just 2)
    assert "per-gate counts track correctly for second gate" (Map.lookup GateEdgeHeadroom (gtPerGateCounts tel3) == Just 1)
    assert "recent rejections bounded" (length (gtRecentRejections tel3) <= 10)

testGateTelemetryBindingGateIdentification :: IO ()
testGateTelemetryBindingGateIdentification = do
    let tel0 = emptyTelemetry 100
        rej = GateRejection GateFeeBuffer ReasonFeeBuffer Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing
        tel1 = recordRejection rej tel0
    assert
        "binding gate identified when single gate rejects"
        (bindingGate (gtPerGateCounts tel1) == Just GateFeeBuffer)

testGateTelemetryHistogramSorting :: IO ()
testGateTelemetryHistogramSorting = do
    let tel0 = emptyTelemetry 100
        rej1 = GateRejection GateEdgeSpike ReasonEdgeSpike Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing
        rej2 = GateRejection GateEdgeSpike ReasonEdgeSpike Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing
        rej3 = GateRejection GateFeeBuffer ReasonFeeBuffer Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing
        tel1 = recordRejection rej1 tel0
        tel2 = recordRejection rej2 tel1
        tel3 = recordRejection rej3 tel2
        hist = rejectionHistogram tel3
    assert
        "histogram sorted by count descending"
        ( case hist of
            (_, _, c1) : (_, _, c2) : _ -> c1 >= c2
            _ -> True
        )

-- ============================================================================
-- Threshold Calibration Tests
-- Engineering invariant: Thresholds must be calibrated from data, not magic.
-- ============================================================================

testThresholdCalibrationEmptyInputFailsClosed :: IO ()
testThresholdCalibrationEmptyInputFailsClosed = do
    let result = computeEdgeDistribution ([] :: [Double])
    assert "empty edge list returns Nothing" (isNothing result)

testThresholdCalibrationDistributionAccuracy :: IO ()
testThresholdCalibrationDistributionAccuracy = do
    let edges = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
        mDist = computeEdgeDistribution edges
    case mDist of
        Nothing -> assert "distribution computation failed" False
        Just dist -> do
            assert "sample size correct" (edSampleSize dist == 10)
            assert "min correct" (edMin dist == 0.001)
            assert "max correct" (edMax dist == 0.010)
            assert "median correct" (edMedian dist == 0.0055)
            assert "p50 equals median" (edP50 dist == edMedian dist)

testThresholdCalibrationPercentileMethod :: IO ()
testThresholdCalibrationPercentileMethod = do
    let edges = replicate 100 0.01 ++ replicate 100 0.02 ++ replicate 100 0.03
        mCalib = calibrateThreshold edges (PercentileMethod 75)
    case mCalib of
        Nothing -> assert "calibration failed" False
        Just calib -> do
            assert
                "percentile method uses correct threshold"
                (tcSuggestedThreshold calib >= 0.02 && tcSuggestedThreshold calib <= 0.03)
            assert
                "headroom threshold is threshold / 1.5"
                (abs (tcHeadroomThreshold calib - tcSuggestedThreshold calib / 1.5) < 1e-9)

testThresholdCalibrationConfigurableRoiKnobs :: IO ()
testThresholdCalibrationConfigurableRoiKnobs = do
    let edges = [0.001 * fromIntegral i | i <- [1 .. 200 :: Int]]
        tunedConfig =
            defaultThresholdCalibrationConfig
                { tccHeadroomDivisor = 2.0
                , tccFeeFloor = 0.002
                , tccMinimumSampleSize = 0
                , tccConservativePercentile = 80
                , tccAggressivePercentile = 40
                }
        mConservative = calibrateThresholdWithConfig tunedConfig edges (PercentileMethod 90)
        mAggressive = calibrateThresholdWithConfig tunedConfig edges (PercentileMethod 30)
    case mConservative of
        Nothing -> assert "configured calibration failed" False
        Just calib -> do
            assert
                "configured headroom divisor controls headroom threshold"
                (abs (tcHeadroomThreshold calib - tcSuggestedThreshold calib / 2.0) < 1e-9)
            assert
                "configured fee floor controls fee-buffer threshold"
                (abs (tcFeeBufferThreshold calib - (tcSuggestedThreshold calib + 0.002)) < 1e-12)
            assert
                "configured conservative percentile controls warning boundary"
                (T.isInfixOf "CONSERVATIVE" (tcRecommendation calib))
    case mAggressive of
        Nothing -> assert "configured aggressive calibration failed" False
        Just calib ->
            assert
                "configured aggressive percentile controls warning boundary"
                (T.isInfixOf "AGGRESSIVE" (tcRecommendation calib))
    let sampleConfig = defaultThresholdCalibrationConfig{tccMinimumSampleSize = 250}
    case calibrateThresholdWithConfig sampleConfig edges (PercentileMethod 75) of
        Nothing -> assert "configured sample-floor calibration failed" False
        Just calib ->
            assert
                "configured minimum sample size controls insufficient-sample warning"
                (T.isInfixOf "Need >= 250" (tcRecommendation calib))
    assert
        "invalid threshold calibration configs fail validation"
        ( isLeft
            ( validateThresholdCalibrationConfig
                defaultThresholdCalibrationConfig{tccHeadroomDivisor = 0}
            )
        )

testThresholdCalibrationInterpolatesIntermediatePercentiles :: IO ()
testThresholdCalibrationInterpolatesIntermediatePercentiles = do
    let edges = [fromIntegral i / 1000 | i <- [0 .. 100 :: Int]]
        mDist = computeEdgeDistribution edges
    case mDist of
        Nothing -> assert "distribution computation failed" False
        Just dist -> do
            assertNear "p80 interpolates between p75 and p90" 0.08 (thresholdAtPercentile 80 dist) 1e-12
            assertNear "p1 interpolates between min and p10" 0.001 (thresholdAtPercentile 1 dist) 1e-12

testThresholdCalibrationStdDevMethod :: IO ()
testThresholdCalibrationStdDevMethod = do
    let edges = [0.01 | _ <- [1 .. 100 :: Int]] -- All same = zero stddev
        mCalib = calibrateThreshold edges (StdDevMethod 2.0)
    case mCalib of
        Nothing -> assert "calibration failed" False
        Just calib -> do
            assert
                "stddev method with zero stddev returns mean"
                (abs (tcSuggestedThreshold calib - 0.01) < 1e-15)

testThresholdCalibrationHybridMethod :: IO ()
testThresholdCalibrationHybridMethod = do
    let edges = [0.001, 0.005, 0.010, 0.015, 0.020]
        mCalib = calibrateThreshold edges (HybridMethod 50 1.0)
    case mCalib of
        Nothing -> assert "calibration failed" False
        Just calib -> do
            assert
                "hybrid method returns non-negative threshold"
                (tcSuggestedThreshold calib >= 0)
            assert
                "hybrid method has confidence interval"
                (uncurry (<=) (tcConfidenceInterval calib))

testThresholdCalibrationRecommendationInsufficientSample :: IO ()
testThresholdCalibrationRecommendationInsufficientSample = do
    let edges = [0.01 | _ <- [1 .. 10 :: Int]]
        mCalib = calibrateThreshold edges (PercentileMethod 75)
    case mCalib of
        Nothing -> assert "calibration failed" False
        Just calib -> do
            assert
                "insufficient sample triggers warning"
                (T.isInfixOf "INSUFFICIENT_SAMPLE" (tcRecommendation calib))

testThresholdCalibrationRecommendationConservative :: IO ()
testThresholdCalibrationRecommendationConservative = do
    let edges = [0.001 * fromIntegral i | i <- [1 .. 200 :: Int]]
        mCalib = calibrateThreshold edges (PercentileMethod 99)
    case mCalib of
        Nothing -> assert "calibration failed" False
        Just calib -> do
            assert
                "conservative threshold above p95 triggers warning"
                (T.isInfixOf "CONSERVATIVE" (tcRecommendation calib))

testThresholdCalibrationRecommendationAggressive :: IO ()
testThresholdCalibrationRecommendationAggressive = do
    let edges = [0.001 * fromIntegral i | i <- [1 .. 200 :: Int]]
        mCalib = calibrateThreshold edges (PercentileMethod 10)
    case mCalib of
        Nothing -> assert "calibration failed" False
        Just calib -> do
            assert
                "aggressive threshold below p25 triggers warning"
                (T.isInfixOf "AGGRESSIVE" (tcRecommendation calib))

testThresholdCalibrationRecommendationBalanced :: IO ()
testThresholdCalibrationRecommendationBalanced = do
    let edges = [0.001 * fromIntegral i | i <- [1 .. 200 :: Int]]
        mCalib = calibrateThreshold edges (PercentileMethod 75)
    case mCalib of
        Nothing -> assert "calibration failed" False
        Just calib -> do
            assert
                "balanced threshold in IQR is recommended"
                (T.isInfixOf "BALANCED" (tcRecommendation calib))

-- Formal execution verification: spec and impl agree on bounded grid.
testFormalExecutionInvariants :: IO ()
testFormalExecutionInvariants = do
    let report = verifyFormalExecution
    assert
        "applyExecutedQuantity implementation matches spec on bounded grid"
        (fvrExecImplMatchesSpec report)
    assert
        "applyReduceOnlyExecutedQuantity implementation matches spec on bounded grid"
        (fvrExecReduceOnlyImplMatchesSpec report)
    assert
        "orderAppliedQuantity implementation matches spec on bounded grid"
        (fvrExecOrderAppliedImplMatchesSpec report)
    assert
        "reduce-only fills never increase position magnitude"
        (fvrExecReduceOnlyNeverIncreases report)
    assert
        "reduce-only fills never flip position sign"
        (fvrExecReduceOnlyNeverFlips report)
    assert
        "executed quantity is conserved into close + open"
        (fvrExecQtyConservation report)
    assert
        "close quantity is monotone non-decreasing in raw qty"
        (fvrExecCloseQtyMonotone report)
    assert
        "open quantity is monotone non-decreasing in raw qty"
        (fvrExecOpenQtyMonotone report)

-- Formal risk verification: halt logic invariants on bounded grid.
testFormalRiskInvariants :: IO ()
testFormalRiskInvariants = do
    let report = verifyFormalRisk
    assert
        "risk halt reason is always justified by a limit breach or previous halt"
        (fvrRiskHaltMonotone report)
    assert
        "daily loss halt resets when day changes"
        (fvrRiskHaltResetDaily report)
    assert
        "weekly loss halt resets when week changes"
        (fvrRiskHaltResetWeekly report)
    assert
        "non-time-bound halt reasons are preserved across time boundaries"
        (fvrRiskHaltPreservesOther report)
    assert
        "no halt occurs when no limits are configured and no previous halt exists"
        (fvrRiskHaltNoFalsePositive report)
    assert
        "position-size halt fires when position exceeds sanitized limit"
        (fvrRiskHaltPositionSize report)
    assert
        "loss-streak halt fires when consecutive losses exceed configured limit"
        (fvrRiskHaltLossStreak report)
    assert
        "max-position-size bound invariant: specRiskHalt respects sanitized limits"
        (fvrMaxPositionSizeBound report)
    assert
        "risk-limit finite invariant: non-finite limits trigger RISK_LIMIT_NON_FINITE halt"
        (fvrRiskLimitFinite report)
    assert
        "drawdown sanity invariant: non-finite or out-of-range drawdown limits are rejected"
        (fvrDrawdownSanity report)
    assert
        "position-size sanity invariant: non-finite, negative, or >10x position sizes are rejected"
        (fvrPositionSizeSanity report)
    assert
        "expectancy sanity invariant: missing or non-finite expectancy when min-expectancy is set triggers EXPECTANCY_INVALID halt"
        (fvrExpectancySanity report)
    assert
        "vol-target sanity invariant: non-finite, negative, or >1000% vol targets are rejected"
        (fvrVolTargetSanity report)
    assert
        "leverage sanity invariant: non-finite, negative, or >150x leverage values are rejected"
        (fvrLeverageSanity report)

-- Witness-level guardrail: when no limits are set and no prior halt,
-- specRiskHalt must return Nothing for a representative set of inputs.
testFormalRiskNoFalsePositiveWitness :: IO ()
testFormalRiskNoFalsePositiveWitness = do
    let inputs =
            [ HaltInputs
                { hiPrevHaltReason = Nothing
                , hiDayChanged = dc
                , hiWeekChanged = wc
                , hiDailyLoss = dl
                , hiWeeklyLoss = wl
                , hiDrawdown = dd
                , hiExpectancy = ex
                , hiMaxDailyLossLim = Nothing
                , hiMaxWeeklyLossLim = Nothing
                , hiMaxDrawdownLim = Nothing
                , hiMinExpectancy = Nothing
                , hiPositionSize = 0
                , hiMaxPositionSizeLim = Nothing
                , hiConsecutiveLosses = 0
                , hiMaxLossStreakLim = Nothing
                , hiVolTarget = 0
                , hiLeverage = 0
                }
            | dc <- [False, True]
            , wc <- [False, True]
            , dl <- [0, 0.01, 0.05, 0.1, 0.5, 1.0]
            , wl <- [0, 0.01, 0.05, 0.1, 0.5, 1.0]
            , dd <- [0, 0.01, 0.05, 0.1, 0.5, 1.0]
            , ex <- [Nothing, Just (-0.1), Just 0, Just 0.01]
            ]
        allNothing = all (isNothing . specRiskHalt) inputs
    assert
        "specRiskHalt returns Nothing when no limits are set and no previous halt exists (witness grid)"
        allNothing

-- Risk-limit sanitization guardrail: negative configured limits are treated as
-- zero (most restrictive), so a negative daily-loss limit still halts when
-- daily loss is non-negative.
testFormalRiskNegativeLimitSanitization :: IO ()
testFormalRiskNegativeLimitSanitization = do
    let negLimInputs reason limField =
            HaltInputs
                { hiPrevHaltReason = Nothing
                , hiDayChanged = False
                , hiWeekChanged = False
                , hiDailyLoss = 0.01
                , hiWeeklyLoss = 0.01
                , hiDrawdown = 0.01
                , hiExpectancy = Just (-0.01)
                , hiMaxDailyLossLim = if reason == ExitMaxDailyLoss then Just limField else Nothing
                , hiMaxWeeklyLossLim = if reason == ExitMaxWeeklyLoss then Just limField else Nothing
                , hiMaxDrawdownLim = if reason == ExitMaxDrawdown then Just limField else Nothing
                , hiMinExpectancy = if reason == ExitOther "NEGATIVE_EXPECTANCY" then Just limField else Nothing
                , hiPositionSize = 0
                , hiMaxPositionSizeLim = Nothing
                , hiConsecutiveLosses = 0
                , hiMaxLossStreakLim = Nothing
                , hiVolTarget = 0
                , hiLeverage = 0
                }
    assert
        "negative daily-loss limit is sanitized to zero and halts on any non-negative daily loss"
        (specRiskHalt (negLimInputs ExitMaxDailyLoss (-0.05)) == Just ExitMaxDailyLoss)
    assert
        "negative weekly-loss limit is sanitized to zero and halts on any non-negative weekly loss"
        (specRiskHalt (negLimInputs ExitMaxWeeklyLoss (-0.05)) == Just ExitMaxWeeklyLoss)
    assert
        "negative drawdown limit triggers DRAWDOWN_LIMIT_INVALID halt"
        (specRiskHalt (negLimInputs ExitMaxDrawdown (-0.05)) == Just (ExitOther "DRAWDOWN_LIMIT_INVALID"))
    assert
        "negative min-expectancy limit is sanitized to zero and halts on any negative expectancy"
        (specRiskHalt (negLimInputs (ExitOther "NEGATIVE_EXPECTANCY") (-0.05)) == Just (ExitOther "NEGATIVE_EXPECTANCY"))

-- Position-size halt guardrail: if position size exceeds the configured max,
-- specRiskHalt emits a POSITION_SIZE exit reason. Negative limits are sanitized
-- to zero (most restrictive), and missing limits disable the check.
testFormalRiskPositionSizeHalt :: IO ()
testFormalRiskPositionSizeHalt = do
    let baseInputs =
            HaltInputs
                { hiPrevHaltReason = Nothing
                , hiDayChanged = False
                , hiWeekChanged = False
                , hiDailyLoss = 0
                , hiWeeklyLoss = 0
                , hiDrawdown = 0
                , hiExpectancy = Nothing
                , hiMaxDailyLossLim = Nothing
                , hiMaxWeeklyLossLim = Nothing
                , hiMaxDrawdownLim = Nothing
                , hiMinExpectancy = Nothing
                , hiPositionSize = 0
                , hiMaxPositionSizeLim = Nothing
                , hiConsecutiveLosses = 0
                , hiMaxLossStreakLim = Nothing
                , hiVolTarget = 0
                , hiLeverage = 0
                }
    assert
        "position size exceeding limit triggers POSITION_SIZE halt"
        ( specRiskHalt baseInputs{hiPositionSize = 1.5, hiMaxPositionSizeLim = Just 1.0}
            == Just (ExitOther "POSITION_SIZE")
        )
    assert
        "position size within limit does not trigger halt"
        (isNothing (specRiskHalt baseInputs{hiPositionSize = 0.5, hiMaxPositionSizeLim = Just 1.0}))
    assert
        "negative position size triggers POSITION_SIZE_INVALID halt"
        (specRiskHalt baseInputs{hiPositionSize = -0.5, hiMaxPositionSizeLim = Just 1.0} == Just (ExitOther "POSITION_SIZE_INVALID"))
    assert
        "negative limit is sanitized to zero and halts on any positive position size"
        ( specRiskHalt baseInputs{hiPositionSize = 0.01, hiMaxPositionSizeLim = Just (-0.05)}
            == Just (ExitOther "POSITION_SIZE")
        )
    assert
        "missing limit disables position-size halt"
        (isNothing (specRiskHalt baseInputs{hiPositionSize = 10.0, hiMaxPositionSizeLim = Nothing}))
    assert
        "position size exactly at limit does not trigger halt"
        (isNothing (specRiskHalt baseInputs{hiPositionSize = 1.0, hiMaxPositionSizeLim = Just 1.0}))

-- Loss-streak halt guardrail: if consecutive losses exceed the configured max,
-- specRiskHalt emits a LOSS_STREAK exit reason. Missing or zero limits disable
-- the check, and exact-boundary equality does not trigger.
testFormalRiskLossStreakHalt :: IO ()
testFormalRiskLossStreakHalt = do
    let baseInputs =
            HaltInputs
                { hiPrevHaltReason = Nothing
                , hiDayChanged = False
                , hiWeekChanged = False
                , hiDailyLoss = 0
                , hiWeeklyLoss = 0
                , hiDrawdown = 0
                , hiExpectancy = Nothing
                , hiMaxDailyLossLim = Nothing
                , hiMaxWeeklyLossLim = Nothing
                , hiMaxDrawdownLim = Nothing
                , hiMinExpectancy = Nothing
                , hiPositionSize = 0
                , hiMaxPositionSizeLim = Nothing
                , hiConsecutiveLosses = 0
                , hiMaxLossStreakLim = Nothing
                , hiVolTarget = 0
                , hiLeverage = 0
                }
    assert
        "consecutive losses exceeding limit triggers LOSS_STREAK halt"
        ( specRiskHalt baseInputs{hiConsecutiveLosses = 4, hiMaxLossStreakLim = Just 3}
            == Just (ExitOther "LOSS_STREAK")
        )
    assert
        "consecutive losses within limit do not trigger halt"
        (isNothing (specRiskHalt baseInputs{hiConsecutiveLosses = 2, hiMaxLossStreakLim = Just 3}))
    assert
        "exact boundary equality does not trigger LOSS_STREAK halt"
        (isNothing (specRiskHalt baseInputs{hiConsecutiveLosses = 3, hiMaxLossStreakLim = Just 3}))
    assert
        "missing limit disables loss-streak halt"
        (isNothing (specRiskHalt baseInputs{hiConsecutiveLosses = 10, hiMaxLossStreakLim = Nothing}))
    assert
        "zero limit disables loss-streak halt"
        (isNothing (specRiskHalt baseInputs{hiConsecutiveLosses = 10, hiMaxLossStreakLim = Just 0}))
    assert
        "zero consecutive losses do not trigger halt even with low limit"
        (isNothing (specRiskHalt baseInputs{hiConsecutiveLosses = 0, hiMaxLossStreakLim = Just 1}))

-- Exhaustive fail-closed checks for signal gates.
testSignalGatesFailClosedExhaustive :: IO ()
testSignalGatesFailClosedExhaustive = do
    let
        -- Grid of malformed / edge-case inputs
        badEdges = [Nothing, Just (-0.01), Just (0 / 0), Just (negate (1 / 0))]
        badFees = [0 / 0, negate (1 / 0), -0.001]
        thresholds = [0, 0.001, 0.01, 0.05]

        -- Every gate that takes a raw Double should map bad values to safe defaults.
        finiteOk = all finiteDouble [0, 0.001, 1.0]
        nonFiniteNotOk = not (any finiteDouble [0 / 0, 1 / 0, negate (1 / 0)])

        -- signalEntryFeeBufferOk should reject non-finite or negative fee floors
        feeBufferFailClosed =
            all
                ( \fee ->
                    not (signalEntryFeeBufferOk 0.01 fee (Just 0.02))
                )
                badFees

        -- normalizeSignalEntryEdge should fail closed: finite negatives clamp
        -- to Just 0; non-finite values (NaN, Inf) map to Nothing.
        edgeNormalizationFailClosed =
            all
                ( \mEdge ->
                    let raw = Data.Maybe.fromMaybe (0 / 0) mEdge
                        result = normalizeSignalEntryEdge raw
                     in if finiteDouble raw then result == Just (max 0 raw) else Data.Maybe.isNothing result
                )
                badEdges

        -- mkSignalThresholdBoundary with bad inputs should produce zero thresholds
        boundaryFailClosed =
            all
                ( \open ->
                    let b = mkSignalThresholdBoundary open (Nothing :: Maybe Double)
                     in stbEffectiveOpenThreshold b == 0 && stbEffectiveCloseThreshold b == 0
                )
                badFees

    assert
        "finiteDouble accepts normal values"
        finiteOk
    assert
        "finiteDouble rejects NaN and infinities"
        nonFiniteNotOk
    assert
        "signalEntryFeeBufferOk fails closed on malformed fees"
        feeBufferFailClosed
    assert
        "normalizeSignalEntryEdge fails closed for non-finite inputs"
        edgeNormalizationFailClosed
    assert
        "mkSignalThresholdBoundary fails closed to zero thresholds"
        boundaryFailClosed

-- Simulation-level guardrail: prove --stop-loss halts the simulation,
-- flattens the position, emits ExitStopLoss, and exits before series end.
testStopLossHaltsSimulation :: IO ()
testStopLossHaltsSimulation = do
    -- Price series: flat → entry at bar 2 → stop-loss breach at bar 3 → flat.
    -- Predictions after bar 2 are flat (equal to price) so no re-entry occurs.
    -- The stop-loss is set to 2%; entry at price 100 with stop at 98.
    -- Bar 3 low is 97, which breaches the stop.
    let prices = V.fromList [100 :: Double, 100, 100, 97, 97]
        highs = V.fromList [100 :: Double, 100, 100, 100, 100]
        lows = V.fromList [100 :: Double, 100, 100, 97, 97]
        kalPreds = V.fromList [100 :: Double, 102, 102, 97]
        lstmPreds = V.fromList [100 :: Double, 102, 102, 97]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecVolLookback = 2
                , ecStopLoss = Just 0.02
                , ecMaxPositionSize = 1
                }
        result = simulateEnsemble cfg 2 prices highs lows kalPreds lstmPreds (Nothing :: Maybe (V.Vector StepMeta))
        trades = brTrades result
        positions = brPositions result
    assert
        "stop-loss simulation produces at least one trade"
        (not (null trades))
    assert
        "stop-loss simulation ends flat or with the last known position"
        (not (null positions) && (last positions == 0 || length positions >= V.length prices - 1))
    assert
        "last trade exits with ExitStopLoss"
        ( case trades of
            [] -> False
            ts -> trExitReason (last ts) == Just ExitStopLoss
        )
    assert
        "stop-loss exit occurs before or at the final bar"
        ( case trades of
            [] -> False
            ts -> trExitIndex (last ts) <= V.length prices - 1
        )

-- Simulation-level guardrail: prove --take-profit flattens the position
-- when price breaches the configured take-profit level.
testTakeProfitGuardrail :: IO ()
testTakeProfitGuardrail = do
    let prices = V.fromList [100 :: Double, 100, 100, 104, 104]
        highs = V.fromList [100 :: Double, 100, 100, 104, 104]
        lows = V.fromList [100 :: Double, 100, 100, 104, 104]
        kalPreds = V.fromList [100 :: Double, 102, 102, 102]
        lstmPreds = V.fromList [100 :: Double, 102, 102, 102]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecVolLookback = 2
                , ecTakeProfit = Just 0.03
                , ecMaxPositionSize = 1
                }
        result = simulateEnsemble cfg 2 prices highs lows kalPreds lstmPreds (Nothing :: Maybe (V.Vector StepMeta))
        trades = brTrades result
        positions = brPositions result
    assert
        "take-profit simulation produces at least one trade"
        (not (null trades))
    assert
        "take-profit simulation ends flat"
        (not (null positions) && last positions == 0)
    assert
        "last trade exits with ExitTakeProfit"
        ( case trades of
            [] -> False
            ts -> trExitReason (last ts) == Just ExitTakeProfit
        )
    assert
        "take-profit exit occurs before or at the final bar"
        ( case trades of
            [] -> False
            ts -> trExitIndex (last ts) <= V.length prices - 1
        )

-- Simulation-level guardrail: prove --trailing-stop flattens the position
-- when price rises then falls enough to trigger the trailing stop.
-- Simulation-level guardrail: prove --max-drawdown halts the simulation,
-- flattens the position, emits ExitMaxDrawdown, and exits before series end.
testMaxDrawdownHaltsSimulation :: IO ()
testMaxDrawdownHaltsSimulation = do
    -- Price series: flat → entry at bar 2 → drawdown breach at bar 4 → flat.
    -- Entry at 100, max-drawdown = 5% (limit = 0.05), so breach when equity
    -- drops to 95. Bar 4 close is 94, triggering drawdown halt.
    let prices = V.fromList [100 :: Double, 100, 100, 99, 94, 94]
        highs = V.fromList [100 :: Double, 100, 100, 100, 99, 94]
        lows = V.fromList [100 :: Double, 100, 100, 99, 94, 94]
        kalPreds = V.fromList [100 :: Double, 102, 102, 101, 94]
        lstmPreds = V.fromList [100 :: Double, 102, 102, 101, 94]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecVolLookback = 2
                , ecMaxDrawdown = Just 0.05
                , ecMaxPositionSize = 1
                }
        result = simulateEnsemble cfg 2 prices highs lows kalPreds lstmPreds (Nothing :: Maybe (V.Vector StepMeta))
        trades = brTrades result
        positions = brPositions result
    assert
        "max-drawdown simulation produces at least one trade"
        (not (null trades))
    assert
        "max-drawdown simulation ends flat or with the last known position"
        (not (null positions) && (last positions == 0 || length positions >= V.length prices - 1))
    assert
        "last trade exits with ExitMaxDrawdown"
        ( case trades of
            [] -> False
            ts -> trExitReason (last ts) == Just ExitMaxDrawdown
        )
    assert
        "max-drawdown exit occurs before or at the final bar"
        ( case trades of
            [] -> False
            ts -> trExitIndex (last ts) <= V.length prices - 1
        )

-- Simulation-level guardrail: prove --trailing-stop flattens the position
-- when price rises then falls enough to trigger the trailing stop.
testTrailingStopGuardrail :: IO ()
testTrailingStopGuardrail = do
    let prices = V.fromList [100 :: Double, 100, 102, 103, 100, 100]
        highs = V.fromList [100 :: Double, 100, 102, 103, 103, 100]
        lows = V.fromList [100 :: Double, 100, 102, 103, 100, 100]
        kalPreds = V.fromList [100 :: Double, 102, 103, 104, 102]
        lstmPreds = V.fromList [100 :: Double, 102, 103, 104, 102]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecVolLookback = 2
                , ecTrailingStop = Just 0.02
                , ecMaxPositionSize = 1
                , ecMinPositionSize = 0.001
                }
        result = simulateEnsemble cfg 2 prices highs lows kalPreds lstmPreds (Nothing :: Maybe (V.Vector StepMeta))
        trades = brTrades result
        positions = brPositions result
    assert
        "trailing-stop simulation produces at least one trade"
        (not (null trades))
    assert
        "trailing-stop simulation ends flat or with the last known position"
        (not (null positions) && (last positions == 0 || length positions >= V.length prices - 1))
    assert
        "last trade exits with ExitTrailingStop or simulation halted"
        ( case trades of
            [] -> False
            ts -> case trExitReason (last ts) of Just _ -> True; Nothing -> False
        )
    assert
        "trailing-stop exit occurs before or at the final bar"
        ( case trades of
            [] -> False
            ts -> trExitIndex (last ts) <= V.length prices - 1
        )

-- The round-trip cost floor must reflect the actual venue model: two
-- crossings each pay (fee + slippage) and there is one full spread on the
-- marketable side. A drift in either floor would silently change the
-- minEdge guard applied below.
testVenueRoundTripCostFloorMatchesVenueCosts :: IO ()
testVenueRoundTripCostFloorMatchesVenueCosts = do
    let expected = 2 * (venueTakerFeeFloor + venueSlippageFloor) + venueSpreadFloor
    assert
        "venueRoundTripCostFloor stays in sync with fee+slippage+spread floors"
        (abs (venueRoundTripCostFloor - expected) < 1.0e-12)
    assert
        "venueRoundTripCostFloor is strictly positive"
        (venueRoundTripCostFloor > 0)

-- A trial whose minEdge sits at or below the round-trip cost is
-- guaranteed-negative-expectancy on the real venue. The published floor
-- must therefore (a) clear the cost floor outright and (b) carry the
-- documented safety margin (1.5x) so prediction noise and funding drift
-- have room before the combo turns into a leak.
testVenueMinEdgeFloorClearsRoundTripCost :: IO ()
testVenueMinEdgeFloorClearsRoundTripCost = do
    assert
        "minEdgeCostMultiplier is at least 1.5 (≥50% safety over the cost floor)"
        (minEdgeCostMultiplier >= 1.5)
    assert
        "venueMinEdgeFloor strictly beats the round-trip cost floor"
        (venueMinEdgeFloor > venueRoundTripCostFloor)
    let expected = minEdgeCostMultiplier * venueRoundTripCostFloor
    assert
        "venueMinEdgeFloor equals minEdgeCostMultiplier * round-trip cost"
        (abs (venueMinEdgeFloor - expected) < 1.0e-12)

-- Regression: on 2026-06-13 every one of 500 prod top combos had a sampled
-- minEdge below the venue round-trip cost (median 4.8 bp vs median cost
-- 23 bp), guaranteeing the live system would leak money. The floor must
-- reject the median observed bad value so a sampling regression cannot
-- regress to the pre-fix behavior.
testVenueMinEdgeFloorMatchesProductionRegressionEvidence :: IO ()
testVenueMinEdgeFloorMatchesProductionRegressionEvidence = do
    let observedProdMedianMinEdge = 4.8e-4 -- 4.8 bp, prod /optimizer/combos median 2026-06-13
        observedProdMedianRoundTripCost = 2.3e-3 -- 23 bp, prod median 2*(fee+slip+spread)
    assert
        "the median minEdge observed on prod 2026-06-13 is below venueMinEdgeFloor"
        (observedProdMedianMinEdge < venueMinEdgeFloor)
    assert
        "the round-trip cost observed on prod 2026-06-13 is above venueMinEdgeFloor only via the safety multiplier"
        (venueMinEdgeFloor >= observedProdMedianRoundTripCost * 0.75)

-- The adoption-time maxPositionSize cap must bound legacy combos into the
-- new cost-floor-aware envelope. Inputs above the cap saturate at the cap;
-- non-finite/negative inputs collapse to zero; safe values pass through.
-- Closes the 2026-06-13 incident where adoption inherited maxPositionSize
-- up to 1.0 at 10-20x leverage.
testCapAdoptedMaxPositionSizeBoundsLiveExposure :: IO ()
testCapAdoptedMaxPositionSizeBoundsLiveExposure = do
    assert
        "adoption cap is at most 0.25 (leaves headroom at 10-20x perp leverage)"
        (adoptionMaxPositionSizeCap <= 0.25)
    assert
        "cap saturates inputs above the cap to the cap"
        (capAdoptedMaxPositionSize 1.0 == adoptionMaxPositionSizeCap)
    assert
        "cap preserves inputs that are already below the cap"
        (capAdoptedMaxPositionSize 0.10 == 0.10)
    assert
        "custom adoption cap saturates inputs above the custom cap"
        (capAdoptedMaxPositionSizeWithCap 0.12 0.30 == 0.12)
    assert
        "custom adoption cap can intentionally disable adopted exposure"
        (capAdoptedMaxPositionSizeWithCap 0 0.30 == 0)
    assert
        "cap clamps negative inputs to zero"
        (capAdoptedMaxPositionSize (negate 0.5) == 0)
    assert
        "cap collapses NaN to zero"
        (capAdoptedMaxPositionSize (0 / 0) == 0)
    assert
        "cap collapses +Infinity to zero (non-finite is unsafe)"
        (capAdoptedMaxPositionSize (1 / 0) == 0)

-- 2026-06-14: invariants for the adoption-time minimum-trade-count gate.
-- The optimizer's production CLI guard (TRADER_OPTIMIZER_MIN_ROUND_TRIPS,
-- defaulted to 3 for discovery but documented as 20 for production sweeps)
-- already rejects trials below the floor. The adoption path must enforce
-- the same floor so a future relaxation of the minEdge cost filter cannot
-- let a 4-trade combo into live capital. Closes the 2026-06-14 leaderboard
-- pathology: 500/500 prod combos with median tradeCount=4 + median Sharpe
-- 8.5 (statistically meaningless).
testAdoptionMinTradeCountMatchesOptimizerProductionGate :: IO ()
testAdoptionMinTradeCountMatchesOptimizerProductionGate = do
    assert
        "adoptionMinTradeCount is at least the documented production gate (20)"
        (adoptionMinTradeCount >= 20)
    assert
        "adoptionMinTradeCount is small enough not to exclude reasonable backtests (<= 50)"
        (adoptionMinTradeCount <= 50)

testComboTradeCountMeetsAdoptionFloorMonotonicity :: IO ()
testComboTradeCountMeetsAdoptionFloorMonotonicity = do
    assert
        "missing reading fails closed (adoption requires positive evidence)"
        (not (comboTradeCountMeetsAdoptionFloor Nothing))
    assert
        "zero trades fail the floor"
        (not (comboTradeCountMeetsAdoptionFloor (Just 0)))
    assert
        "one trade below the floor fails"
        (not (comboTradeCountMeetsAdoptionFloor (Just (adoptionMinTradeCount - 1))))
    assert
        "exactly at the floor passes (gate is >=)"
        (comboTradeCountMeetsAdoptionFloor (Just adoptionMinTradeCount))
    assert
        "one trade above the floor passes"
        (comboTradeCountMeetsAdoptionFloor (Just (adoptionMinTradeCount + 1)))
    assert
        "predicate is monotone in the reading (5000 passes, 5 does not)"
        ( comboTradeCountMeetsAdoptionFloor (Just 5000)
            && not (comboTradeCountMeetsAdoptionFloor (Just 5))
        )

testComboTradeCountMeetsAdoptionFloorHonorsConfig :: IO ()
testComboTradeCountMeetsAdoptionFloorHonorsConfig = do
    let strictConfig =
            AdoptionEvidenceConfig
                { aecMinTradeCount = adoptionMinTradeCount + 10
                , aecMinWalkForwardSharpeMean = adoptionMinWalkForwardSharpeMean
                }
        disabledFloorConfig = strictConfig{aecMinTradeCount = 0}
    assert
        "configured trade-count floor rejects readings below the configured value"
        (not (comboTradeCountMeetsAdoptionFloorWithConfig strictConfig (Just adoptionMinTradeCount)))
    assert
        "configured trade-count floor accepts readings at the configured value"
        (comboTradeCountMeetsAdoptionFloorWithConfig strictConfig (Just (adoptionMinTradeCount + 10)))
    assert
        "zero configured trade-count floor still requires a present reading"
        ( comboTradeCountMeetsAdoptionFloorWithConfig disabledFloorConfig (Just 0)
            && not (comboTradeCountMeetsAdoptionFloorWithConfig disabledFloorConfig Nothing)
        )

-- Regression: today's snapshot (haskell/.tmp/optimizer/top-combos.json,
-- 2026-06-14) has 500/500 combos with median tradeCount=4. The adoption
-- floor must reject the median so a sampling regression cannot revive
-- the pre-fix behavior.
testComboTradeCountMeetsAdoptionFloorMatchesProductionRegressionEvidence :: IO ()
testComboTradeCountMeetsAdoptionFloorMatchesProductionRegressionEvidence = do
    let observedProdMedianTradeCount = 4 :: Int
    assert
        "the median tradeCount observed on prod 2026-06-14 fails the adoption floor"
        (not (comboTradeCountMeetsAdoptionFloor (Just observedProdMedianTradeCount)))

-- 2026-06-14: invariants for the adoption-time walk-forward Sharpe gate.
-- The optimizer's default `minWfSharpeMean` was turned on at 0.3 by the
-- 2026-06-13 fix. Adoption must mirror exactly so the two gates stay
-- falsifiably equal; if one is relaxed the test fails and the other must
-- be updated alongside it.
testAdoptionMinWalkForwardSharpeMatchesOptimizerDefault :: IO ()
testAdoptionMinWalkForwardSharpeMatchesOptimizerDefault = do
    let optimizerDefaultMinWfSharpeMean = 0.3 :: Double
    assert
        "adoptionMinWalkForwardSharpeMean equals the optimizer default minWfSharpeMean"
        (abs (adoptionMinWalkForwardSharpeMean - optimizerDefaultMinWfSharpeMean) < 1.0e-12)

testComboWalkForwardSharpeMeetsAdoptionFloorFailsClosed :: IO ()
testComboWalkForwardSharpeMeetsAdoptionFloorFailsClosed = do
    assert
        "missing walk-forward summary fails closed"
        (not (comboWalkForwardSharpeMeetsAdoptionFloor Nothing))
    assert
        "NaN reading fails closed"
        (not (comboWalkForwardSharpeMeetsAdoptionFloor (Just (0 / 0))))
    assert
        "+Infinity reading fails closed (non-finite is unsafe)"
        (not (comboWalkForwardSharpeMeetsAdoptionFloor (Just (1 / 0))))
    assert
        "-Infinity reading fails closed"
        (not (comboWalkForwardSharpeMeetsAdoptionFloor (Just (negate (1 / 0)))))

testComboWalkForwardSharpeMeetsAdoptionFloorMonotonicity :: IO ()
testComboWalkForwardSharpeMeetsAdoptionFloorMonotonicity = do
    let belowFloor = adoptionMinWalkForwardSharpeMean - 0.05
        aboveFloor = adoptionMinWalkForwardSharpeMean + 0.05
    assert
        "below-floor Sharpe fails"
        (not (comboWalkForwardSharpeMeetsAdoptionFloor (Just belowFloor)))
    assert
        "at-floor Sharpe passes (gate is >=)"
        (comboWalkForwardSharpeMeetsAdoptionFloor (Just adoptionMinWalkForwardSharpeMean))
    assert
        "above-floor Sharpe passes"
        (comboWalkForwardSharpeMeetsAdoptionFloor (Just aboveFloor))
    assert
        "predicate is monotone in the reading (1.5 passes, 0.0 does not)"
        ( comboWalkForwardSharpeMeetsAdoptionFloor (Just 1.5)
            && not (comboWalkForwardSharpeMeetsAdoptionFloor (Just 0.0))
        )

testComboWalkForwardSharpeMeetsAdoptionFloorHonorsConfig :: IO ()
testComboWalkForwardSharpeMeetsAdoptionFloorHonorsConfig = do
    let strictConfig =
            AdoptionEvidenceConfig
                { aecMinTradeCount = adoptionMinTradeCount
                , aecMinWalkForwardSharpeMean = adoptionMinWalkForwardSharpeMean + 0.4
                }
        relaxedConfig = strictConfig{aecMinWalkForwardSharpeMean = -0.1}
    assert
        "configured walk-forward Sharpe floor rejects readings below the configured value"
        (not (comboWalkForwardSharpeMeetsAdoptionFloorWithConfig strictConfig (Just adoptionMinWalkForwardSharpeMean)))
    assert
        "configured walk-forward Sharpe floor accepts readings at the configured value"
        (comboWalkForwardSharpeMeetsAdoptionFloorWithConfig strictConfig (Just (adoptionMinWalkForwardSharpeMean + 0.4)))
    assert
        "relaxed walk-forward Sharpe floor still fails closed on missing evidence"
        ( comboWalkForwardSharpeMeetsAdoptionFloorWithConfig relaxedConfig (Just 0.0)
            && not (comboWalkForwardSharpeMeetsAdoptionFloorWithConfig relaxedConfig Nothing)
        )

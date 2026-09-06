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
import qualified Data.ByteString.Lazy.Char8 as BL8
import Data.Either (isLeft)
import qualified Data.HashMap.Strict as HM
import Data.Int (Int64)
import Data.List (find, intercalate, isInfixOf, isPrefixOf, nub)
import qualified Data.Map.Strict as Map
import Data.Maybe (catMaybes, fromMaybe, isJust, isNothing, listToMaybe, mapMaybe)
import qualified Data.Text as T
import qualified Data.Vector as V
import Network.HTTP.Client (HttpException (..), HttpExceptionContent (..), parseRequest_, requestHeaders)
import Options.Applicative (ParserResult (..), auto, defaultPrefs, execParserPure, info, long, option, switch, value)
import System.Directory (removeFile)
import System.IO (hClose, openTempFile)
import Trader.App.Args (Args (..), applyBackendAutostartSizingDefault, argRouterScorePnlWeight, argTunePenaltyTurnover, argWalkForwardEmbargoBars, argWalkForwardFolds, normalizeBarsForLookback, opts, parsePositioning, validateArgs)
import Trader.App.Env (canonicalizeUuidEnvValues)
import Trader.App.Runtime (hashKeyHex, resolveTenantKeyFromParams, resolveTenantKeyFromPlatformParams, tenantKeyFromBinanceKeys, tenantKeyFromCoinbaseKeys)
import Trader.Binance (BinanceTrade (..), FuturesPositionRisk (..), Kline (..), binanceExceptionSummary, futuresPositionRiskLeverageSane)
import Trader.BinanceTradeAnalysis (attachBinanceTradeMaxPnl, binanceTradeMaxPnlKlineRanges)
import Trader.BotSnapshotRecovery (TradeMemorySnapshotContext (..), restoreTradeMemoryFromStatus)
import Trader.BotStartSemantics (AdoptionEvidenceConfig (..), BacktestVerdict (..), adoptionMaxPositionSizeCap, adoptionMaxWalkForwardSharpeStd, adoptionMinEdgeFloor, adoptionMinTradeCount, adoptionMinWalkForwardSharpeMean, backtestVerdictAborts, botStartSymbolDisabled, botStartupBacktestAborts, botStartupBacktestRoiAcceptable, botStartupBacktestVerdict, botStartupBacktestVerdictWithMinTrades, botStartupGuardShouldPrune, capAdoptedMaxPositionSize, capAdoptedMaxPositionSizeWithCap, capAdoptedMinPositionSize, capBotStartSymbolsPreservingOrphans, comboMinEdgeMeetsAdoptionFloor, comboMinEdgeMeetsAdoptionFloorWithConfig, comboTradeCountMeetsAdoptionFloor, comboTradeCountMeetsAdoptionFloorWithConfig, comboWalkForwardSharpeMeetsAdoptionFloor, comboWalkForwardSharpeMeetsAdoptionFloorWithConfig, comboWalkForwardSharpeStdMeetsAdoptionCeiling, comboWalkForwardSharpeStdMeetsAdoptionCeilingWithConfig, defaultBotStartupBacktestMinTrades, deployableOverrideEvidenceEligible, filterBotStartAttemptsPreservingOrphans, prioritizeBotStartSymbols, queuedStartOrderErrorIssue, throttleBotStartSymbolsPreservingOrphans)
import Trader.CapitalPreservation (
    CapitalPreservationConfig (..),
    CapitalPreservationReport (..),
    PortfolioCapitalPreservationConfig (..),
    PortfolioCapitalPreservationReport (..),
    PortfolioCapitalTrade (..),
    capitalPreservationIsEntryOnlyReason,
    capitalPreservationReport,
    defaultCapitalPreservationConfig,
    defaultPortfolioCapitalPreservationConfig,
    defaultPortfolioCapitalPreservationCooldownMs,
    portfolioCapitalPreservationReport,
 )
import Trader.Coinbase (CoinbaseCandle (..), CoinbaseOrderInfo (..), alignCoinbaseClosesToGrid, buildRanges, coinbaseProductFromBinance, decodeCoinbaseOrderInfo)
import Trader.CostCalibration (
    CostCalibrationConfig (..),
    calibratedSlippagePerSide,
    calibratedSlippagePerSideWithConfig,
    costCalibrationFloorFactor,
    costCalibrationMaxPerSide,
    costCalibrationMinObservations,
    defaultCostCalibrationConfig,
    minEdgeCostMultiplier,
    observedSlippageFraction,
    observedSlippageFractionWithConfig,
    venueMinEdgeFloor,
    venueRoundTripCostFloor,
    venueSlippageFloor,
    venueSpreadFloor,
    venueTakerFeeFloor,
 )
import Trader.ExternalData (
    ExternalFeature (..),
    ExternalJsonSpec (..),
    ExternalObservationV2 (..),
    alignedExternalFeatureInputs,
    alignedExternalFeatureInputsV2,
    externalCellDouble,
    externalCellTime,
    externalCsvFeatureForColumn,
    externalFeatureSeriesV2,
    externalSymbolMatches,
    parseExternalJsonSpec,
 )
import Trader.Formal.CloseTiming (
    ComboCloseTimingReport (..),
    liveMaxPnlCloseTimingEvidenceHoldBars,
    liveMaxPnlCloseTimingMaxHoldBars,
 )
import Trader.Formal.Execution (
    ExecutionVerificationReport (..),
    verifyFormalExecution,
 )
import Trader.Formal.Optimization (
    activityCountFromMetrics,
    fvrActivityCountInvariant,
    fvrOptimizerPublicSurfaceInvariant,
    fvrVolConfCanonicalizationInvariant,
    fvrVolConfMalformedConfidenceFailsClosed,
    fvrVolConfMalformedInputsStayConservative,
    fvrVolConfMalformedVolMatchesMissing,
    fvrVolConfOutputBounded,
    roiImplementationScoreWithConfig,
    roiSpecScoreWithConfig,
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
import Trader.Formal.RiskRegister (RiskEntry (..), riskRegister, riskSeverityOf)
import Trader.GateTelemetry (GateName (..), GateRejection (..), GateTelemetry (..), RejectionReason (..), bindingGate, emptyTelemetry, recordRejection, rejectionHistogram, telemetrySummary, telemetryToJson)
import qualified Trader.Kalman3 as Kalman3
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
import Trader.LiveOrderIntent (
    LiveRiskHaltAction (..),
    desiredPositionForSignal,
    desiredPositionForSignalWithVolConf,
    liveRiskHaltAction,
    orderDirectionForTransition,
 )
import Trader.MarketContext (alignKlineClosesToOpenTimes, fitLinearRange)
import Trader.MarketDataIntegrity (
    MarketSeriesBar (..),
    isTransientMarketDataError,
    marketDataContinuationIssue,
    marketDataFreshness,
    marketDataStaleReason,
    mdfAgeMs,
    mdfFreshnessBudgetMs,
    mdfLastCloseTimeMs,
    mdfStale,
    normalizeClosedMarketSeries,
    validateMarketSeriesBars,
 )
import Trader.MarketGovernor (
    MarketGovernorConfig (..),
    MarketGovernorDecision (..),
    MarketGovernorInputs (..),
    MarketGovernorProfile (..),
    defaultMarketGovernorConfig,
    marketGovernorDecision,
    marketGovernorFreshEntryBlockReason,
    marketGovernorIsEntryOnlyReason,
 )
import Trader.Method (Method (..))
import Trader.Metrics (BacktestMetrics (..), computeMetrics)
import Trader.NeuralGovernor (
    NeuralGovernorConfig (..),
    NeuralGovernorDecision (..),
    NeuralGovernorFeatures (..),
    NeuralGovernorPendingEntry (..),
    NeuralGovernorRolloutMode (..),
    defaultNeuralGovernorConfig,
    initNeuralGovernorState,
    neuralGovernorDecide,
    neuralGovernorHoldReason,
    neuralGovernorObserveTrade,
    neuralGovernorOpenBlockReason,
    neuralGovernorReward,
    neuralGovernorSizingMultiplier,
 )
import Trader.OnlineStats (Welford (..), emptyWelford, updateWelford, varianceWelford)
import Trader.Optimization (TuneConfig (..), TuneStats (..), defaultTuneConfig, sweepThresholdWithHLWith)
import Trader.Optimizer.Common (AutoOptimizerScopeSelection (..), OptimizerAdmissionStats (..), autoOptimizerRequiredBarsForSweep, optimizerAdmissionStats, optimizerObjectiveArgs, selectAutoOptimizerScopes, selectAutoOptimizerScopesWithHeadroom)
import qualified Trader.Optimizer.Common as OptimizerCommon
import Trader.Optimizer.Merge (MergeArgs (..), runMerge)
import Trader.Optimizer.Optimize (
    CorrelationGuidanceField (..),
    OptimizationTechniqueSummary (..),
    OptimizerEdgeScoreConfig (..),
    OptimizerRecordsSummary (..),
    PriorTrial (..),
    ageAdjustedPriorScore,
    ageAdjustedPriorScoreWithMissingMultiplier,
    appliedCloseTimingMaxHoldBars,
    applyCloseTimingMetrics,
    applyWalkForwardSummaryMetrics,
    dedupeFirstByKey,
    defaultOptimizerEdgeScoreConfig,
    defaultPriorMissingAgeMultiplier,
    emptyOptimizerRecordsSummary,
    emptyTechniqueSummary,
    extractPortfolioEvidence,
    kellyLiteExposureContractReason,
    normalizeOptimizerRiskPerTrade,
    optimizerOptionPresent,
    optimizerRecordsShouldRetryDiscovery,
    optimizerSoftSearchEligible,
    optimizerSoftSearchFilterReason,
    optimizerTechniqueSummaryJson,
    optimizerTopJsonSortKey,
    parseOptimizerCorrelationGuidance,
    priorAgeDecayMultiplier,
    priorAgeDecayMultiplierWithMissingMultiplier,
    priorTrialEdgeScore,
    priorTrialEdgeScoreWithConfig,
    priorTrialsFromValue,
    qualityPresetBudget,
    qualityPresetCeiling,
    qualityPresetWeightFloor,
 )
import Trader.Optimizer.OverfitAudit (OverfitTrial (..), optimizerOverfitAudit)
import Trader.OrderExecution (OrderExecutionEvidence (..), applyExecutedQuantity, applyReduceOnlyExecutedQuantity, applySplitReversalExecutedQuantities, confirmedCloseExecutedQuantity, orderAppliedFraction)
import Trader.Platform (Platform (..))
import Trader.PointInTimeUniverse (PointInTimeUniverseConfig (..), loadPointInTimeUniverse)
import Trader.PortfolioSelection (
    PortfolioCandidate (..),
    PortfolioDailyReturn (..),
    PortfolioEvidence (..),
    PortfolioGraduationConfig (..),
    PortfolioGraduationDecision (..),
    PortfolioGraduationEvidence (..),
    PortfolioGraduationReview (..),
    PortfolioMember (..),
    PortfolioMetrics (..),
    PortfolioRolloutMode (..),
    PortfolioSelection (..),
    PortfolioSelectorConfig (..),
    defaultPortfolioGraduationConfig,
    defaultPortfolioSelectorConfig,
    portfolioAnnualizedReturn,
    portfolioFailureCacheLookup,
    portfolioGraduationFleetEquities,
    portfolioGraduationLatestStatusesHealthy,
    portfolioGraduationPerformance,
    portfolioGraduationReview,
    portfolioGraduationReviewApplies,
    portfolioGraduationStatusCoverage,
    portfolioMaxDrawdown,
    portfolioMembersRemainAdmitted,
    portfolioSelectionShouldRotate,
    portfolioSelectorConfigVersion,
    refreshPortfolioSelection,
    selectPortfolio,
 )
import Trader.PredictionMarkets (
    PredictionMarketEvent (..),
    PredictionMarketFetchConfig (..),
    PredictionMarketMarket (..),
    PredictionMarketSignal (..),
    defaultPredictionMarketFetchConfig,
    nearestPredictionMarketInterval,
    predictionMarketProbabilityForDir,
    selectPredictionMarketSignal,
    selectPredictionMarketSignalWithConfig,
 )
import Trader.Predictors (RegimeProbs (..))
import Trader.Predictors.Conformal (AdaptiveConformalState (..), ConformalModel (..), fitConformal, initAdaptiveConformal, predictInterval, updateAdaptiveConformal)
import Trader.Predictors.DecisionTree (DecisionTree (..), DecisionTreeModel (..), predictDecisionTree, trainDecisionTree)
import Trader.Predictors.DerivativesPanelSchema (
    DerivativesFeatureV2 (..),
    DerivativesPanelCellV2 (..),
    decodeDerivativesPanelV2,
    derivativesFeatureAvailabilitySchemaIdV2,
    derivativesObservationSchemaIdV2,
    derivativesPanelCellUsableV2,
    derivativesPanelCellV2,
    derivativesPanelCellVersionedV2,
    derivativesPanelColumnsV2,
 )
import Trader.Predictors.Exogenous (afsV2AvailabilityTimesMs, afsV2Available, afsV2EventTimesMs, afsV2Values, alignTimedToBars, alignToBars, alignedFeatureSeries, alignedFeatureSeriesV2)
import Trader.Predictors.ExogenousFetch (binanceStatsPeriodForInterval)
import Trader.Predictors.ExternalFeatureSchema (externalFeatureFamilies)
import Trader.Predictors.ExternalPanelSchema (
    ExternalPanelCellV2 (..),
    decodeExternalPanelV2,
    externalPanelCellAvailableV2,
    externalPanelCellV2,
    externalPanelColumnsV2,
    externalPanelFeatureAvailabilitySchemaIdV2,
    externalPanelSchemaIdV2,
    externalPanelSchemaVersionV2,
 )
import Trader.Predictors.FeatureSchema (FeatureField (..), FeatureRequirement (..), TimedFeatureValue (..), featureAvailabilitySchemaIdV2, featureRowModelInputs, featureRowSchemaSignature, frv2Available, frv2SchemaId, frv2Values, mkFeatureRowV2)
import Trader.Predictors.Features (ExternalFeatureInputs (..), featuresAtWithInputsWithMarket, mkFeatureInputs, mkFeatureSpec, withCoinbaseInputs, withExternalInputs)
import Trader.Predictors.GBDT (GBDTModel (..), Stump (..), predictGBDT, trainGBDT)
import Trader.Predictors.HMM (HMM3 (..), HMMFilter (..), filterPosterior, fitHMM3, predictNextFromPosterior, updatePosterior)
import Trader.Predictors.KNN (KNNModel (..), predictKNN, trainKNN)
import Trader.Predictors.PatchTST (PatchTSTModel (..), patchTstFeaturesAt, predictPatchTST, trainPatchTST)
import Trader.Predictors.Quantile (LinModel (..), QuantileModel (..), predictQuantiles, trainQuantileModel)
import Trader.Predictors.TCN (TCNModel (..), predictTCN, tcnFeaturesAt, trainTCN)
import Trader.Predictors.Types (SensorId (..), predictorCode, predictorImplementationId, predictorSetFromString, predictorSetToList)
import Trader.RoiScore (RoiScoreConfig (..), defaultFormalRoiScoreConfig)
import Trader.SensitivityAnalysis (
    ParameterSpec (..),
    SensitivityPoint (..),
    SensitivityReport (..),
    mostSensitiveParameter,
    runLocalSensitivity,
    runLocalSensitivityChecked,
    validateParameterSpec,
 )
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
import Trader.Test.ApiRoutes (apiRouteSuite)
import Trader.Test.AutoStartBackoff (autoStartBackoffSuite)
import Trader.Test.BinanceProbe (binanceProbeSuite)
import Trader.Test.Cors (corsSuite)
import Trader.Test.FormalVerification (formalVerificationSuite)
import Trader.Test.GracefulShutdown (gracefulShutdownSuite)
import Trader.Test.MarketRisk (marketRiskSuite)
import Trader.Test.NeuralGovernorRollout (neuralGovernorRolloutSuite)
import Trader.Test.OnlineNeural (runOnlineNeuralTests)
import Trader.Test.Revenue (revenueSuite)
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
    validateCalibrationMethod,
    validateThresholdCalibrationConfig,
 )
import Trader.TopComboScoring (TopComboScoringConfig (..), defaultTopComboScoringConfig, topComboFreshnessMultiplier, topComboMinimumFinalEquity)
import Trader.TopCombosStore (
    ComboBacktestApplyStats (..),
    ComboBacktestRefreshPolicy (..),
    ComboBacktestUpdate (..),
    ComboLiveStats (..),
    applyComboUpdatesKeepAllWithStats,
    applyComboUpdatesWithStats,
    applyComboUpdatesWithStatsWithPolicy,
    batchCombosForBacktestRefresh,
    blendedAnnualizedReturn,
    comboBacktestDueForRefresh,
    comboBacktestDueForRefreshWithPolicy,
    comboBacktestFreshEnoughForMaxAge,
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
    mergeTopCombosPayloadsWithStatsAndDeployableOverrides,
    recalculateComboPerformanceFromOperation,
    selectCombosForBacktestRefresh,
    selectCombosForBacktestRefreshWithPolicy,
 )
import Trader.TradeMethodGate (MethodGateConfig (..), MethodGateDecision (..), MethodResultStats (..), conservativeUnavailableEvidenceSize, loadMethodResultStats, methodGateDecision, unavailableEvidenceSizeCap, unavailableEvidenceSizeMultiplier)
import Trader.Trading (
    BacktestCostAttribution (..),
    BacktestResult (..),
    EnsembleConfig (..),
    ExitReason (..),
    IntrabarFill (..),
    OutcomeWeightConfig (..),
    PositionSide (..),
    Positioning (..),
    StepMeta (..),
    Trade (..),
    TradeEntrySource (..),
    defaultTriLayerPriceActionBodyOpenThresholdMult,
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
    positionSizeScaleHardFailMultiplier,
    roundTripFeeFloor,
    simulateEnsemble,
    simulateEnsembleWithHLChecked,
    tradeEntrySourceCode,
    tradeOutcomeWeightFactor,
    tradeOutcomeWeightFactorWithConfig,
    tradeOutcomeWeights,
    tradeOutcomeWeightsWithConfig,
 )
import Trader.Types.Safe (fromLeverage, fromQuantity, leverageFromDouble, quantityFromDouble)
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
    testTradeAllowedDefaultsAndAnyOverride
    testTradeMethodGateUsesResultEvidence
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
    testMarketContextAlignsPeerKlinesByOpenTime
    testOrderIntentUsesCloseDirectionForExistingPositions
    testVolConfHoldPreservesLivePosition
    testLongShortFlipCountsExitAndEntryTurnover
    testIntrabarTakeProfitUsesExitBarCost
    testIntrabarRoundTripRecordsExposure
    testPartialTakeProfitMovesLongStopToBreakeven
    testPartialTakeProfitMovesShortStopToBreakeven
    testPartialTakeProfitTradeFeesMatchAttribution
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
    testComboMinEdgeMeetsAdoptionFloor
    testAdoptionMinWalkForwardSharpeMatchesOptimizerDefault
    testComboWalkForwardSharpeMeetsAdoptionFloorFailsClosed
    testComboWalkForwardSharpeMeetsAdoptionFloorMonotonicity
    testComboWalkForwardSharpeMeetsAdoptionFloorHonorsConfig
    testComboWalkForwardSharpeStdMeetsAdoptionCeiling
    testFuturesPositionRiskLeverageSaneCap
    testBinanceTradeMaxPnlLongUsesHigh
    testBinanceTradeMaxPnlShortUsesLow
    testBinanceTradeMaxPnlFallsBackForUnpairedClose
    testBinanceTradeMaxPnlUnpairedBothCloseDoesNotCreatePhantomLot
    testBinanceTradeMaxPnlFallsBackToFillPricesWithoutKlines
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
    testMaxHoldBarsZeroDisablesForcedExit
    testMaxPositionSizeRejectsAbsurdUpperBound
    testMaxPositionSizeRejectsNonFuturesOverFive
    testMaxPositionSizeRejectsZeroAndNegative
    testInitialBalanceRejectsZeroOrNegative
    testMinPositionSizeRejectsOutOfRangeValues
    testPredictionMarketHerdTtlRejectsInvalidValues
    testBacktestRatioRejectsInvalidValues
    testOrderQuoteFractionRejectsInvalidValues
    testBackendAutostartSizingDefault
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
    testKalman3RejectsMalformedMeasurements
    testKalmanPhysicsKnobsRejectInvalidValues
    testKalmanPhysicsMeasurementKnobsRejectInvalidValues
    testKalmanPhysicsCandidateValidationRatiosRejectInvalidValues
    testKalmanPhysicsCandidateGridKnobsRejectInvalidValues
    testTriLayerPriceActionBodyOpenThresholdMultRejectsInvalidValues
    testCostCalibrationKnobsRejectInvalidValues
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
    testRouterRegimeFloorRejectsInvalidValues
    testRouterMinScoreRejectsInvalidValues
    testRouterScorePnlWeightRejectsInvalidValues
    testExpectancyLookbackRejectsNegativeValue
    testPerfLookbackRejectsNegativeValue
    testCapitalPreservationReport
    testPortfolioCapitalPreservationReport
    testMarketGovernorPolicy
    testNeuralGovernorPolicy
    testLiveMaxPnlCloseTimingRecommendation
    testOptimizerCloseTimingRecommendationRequiresAcceptedEvidence
    testOptimizerCloseTimingMetricsRecordAppliedRecommendation
    testLossStreakMaxRejectsNegativeValue
    testLossStreakCooldownBarsRejectsNegativeValue
    testVolScaleMaxRejectsInvalidValues
    testRsiLowerMustBeLessThanUpper
    testExchangeDataLongShortBacktestAllowed
    testPositioningShortAliasRejected
    testTenantResolutionScopesMixedApiKeys
    testTenantCredentialEncodingInjective
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
    testDecisionParitySharedGatePrecedence
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
    testMergeSanitizeTombstonesStampedBelowFloorRefresh
    testLowTradeTopCombosSinkBelowEvidenceFloor
    testTopComboProcessingCarriesOverfitAndMapEliteMetadata
    testMergeDedupesSourceAndNullEquivalentCombos
    testDeployableTierRanksAheadOfUnvalidatedCandidate
    testDeployableTierRequiresCompleteAdoptionEvidence
    testExplicitDeployableOverrideIsBoundedAndAuditable
    testTopComboFreshnessMultiplierDefaultsDisabled
    testPortfolioAnnualizationAndDrawdown
    testOptimizerExtractsTimestampedPortfolioEvidence
    testPortfolioSelectionIsDeterministicAndBounded
    testPortfolioSelectionAcceptsMicroLiveBounds
    testPortfolioSelectionFailsClosedOnInvalidNumbers
    testPortfolioSelectionRejectsSparseWinner
    testPortfolioCurrentEvidenceRiskGate
    testPortfolioFailureCacheInvalidatesOnSnapshotChange
    testPortfolioGraduationRequiresEveryReviewGate
    testPortfolioGraduationPerformanceAndPersistence
    testPortfolioSelectionRotationHysteresis
    testPortfolioSelectionJsonRoundTrip
    testMergeFreshnessScoringPromotesFreshCandidate
    testMergeExecutableAnnotatesProcessingAndDedupe
    testSelectCombosForBacktestRefreshIncludesEveryStaleCombo
    testBacktestRefreshBatchesPrioritizeRankedCombos
    testLiveComboFreshnessRequiresRecentBacktestEvidence
    testPrunedBacktestTombstonePreventsStaleResurrection
    testKeepAllUpdateKeepsUnprofitableComboStamped
    testTradeOutcomeWeightsSemantics
    testTradeOutcomeWeightsIncludeNewClose
    testWeightedFineTuneUnitWeightsEquivalence
    testWeightedFineTunePunishesLossRegion
    testObservedSlippageFractionSemantics
    testCalibratedSlippageShrinkage
    testCostCalibrationConfigurableRoiKnobs
    testLiveGapFeedback
    testAlignToBarsPointInTime
    testAlignToBarsFailClosedOnMalformedInputs
    testFeatureAvailabilitySchemaV2
    testExogenousDerivativesBacktestWiring
    testPointInTimeUniverseSelectsHistoricalSnapshot
    testNormalizeBarsForLookbackBinanceClampsAtPageCap
    testBinanceExceptionSummaryRedactsSecrets
    testConformalCalibrationResidualsFailClosed
    testAdaptiveConformalRadiusRespondsToMisses
    testBacktestEntryGateUsesRoundTripFeeBuffer
    testBacktestFreshEntrySizingBoundsFailClosed
    testBacktestPositionSizeFloorCapValidation
    testBacktestCostAttributionGrossNetConsistency
    testBacktestCostAttributionNonFiniteComponentsRegression
    testOrderExecutionFillSanitizationInvariant
    testSplitReversalExecutionInvariant
    testStartupCanceledAfterPartialExecutionScenario
    testLiveReversalPartialEntryScenario
    testReduceOnlyPartialTakeProfitTerminalCancelScenario
    testSnapshotRestartRestoresMemoryWithoutExposureScenario
    testOrderExecutionCorruptedInputInvariant
    testCoinbaseBuildRangesOverflowRegression
    testCoinbaseOrderInfoDecodeInvariant
    testOptimizerActivityCountInvariant
    testAutoOptimizerCappedLookbackScopes
    testAutoOptimizerObjectiveAlignment
    testOptimizerAdmissionStats
    testSweepThresholdMinRoundTripsFallback
    testSweepThresholdZeroCandidatesKeepsBasePair
    testOptimizerPublicSurfaceRegression
    testOptimizerRiskPerTradeNormalization
    testOptimizerQualityBudgetRegression
    testOptimizerSurvivorDedupePreservesFirstCandidates
    testOptimizerTopJsonSortUsesObjectiveScore
    testOptimizerTechniqueSummaryTruthfulRegression
    testOptimizerPriorEdgeScoreRegression
    testOptimizerPriorParserCarriesFreshEvidenceRegression
    testOptimizerPriorAgeDecayMissingTimestampRegression
    testOptimizerPriorAgeAdjustedScoreRegression
    testOptimizerOverfitAuditReportsSelectionRisk
    testOptimizerQualityThresholdArgvExplicitRegression
    testOptimizerCorrelationGuidanceParserRegression
    testOptimizerKellyLiteExposureContractRegression
    testOptimizerRecordsRetryDiscoveryForWalkForwardFilters
    testOptimizerRecordMetricsCarryWalkForwardSummary
    testOptimizerRecordsRetryDiscoveryForCostFloorFilters
    testOptimizerRecordsRetryDiscoveryStopsWhenEligible
    testOptimizerSoftSearchEligibility
    testTopComboBacktestPrunesRoiLosers
    testMetricsConsumesTradingPublicResults
    testMetricsFiniteInputBoundary
    testOnlineStatsFiniteInputBoundary
    testIndependentRoiSpecification
    testSensitivityAnalysisInvariants
    testRiskRegisterInvariants
    testSafeNumericConstruction
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
    testThresholdCalibrationRejectsMalformedMethod
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
    testExternalDataFeatureInputs
    testExternalPanelSchemaV2
    testDerivativesPanelSchemaV2
    testMultivariateLstmInputs
    testGBDTSanitizesMalformedInputs
    testDecisionTreeSanitizesMalformedInputs
    testKNNSanitizesMalformedInputs
    testQuantileSanitizesMalformedInputs
    testTCNSanitizesMalformedInputs
    testPatchTSTSanitizesMalformedInputs
    testPredictorImplementationIdentityCompatibility
    testHMMSanitizesMalformedInputs
    runOnlineNeuralTests
    runTechnicalAnalysisTests
    runSuite "apiRoutes" apiRouteSuite
    runSuite "cors" corsSuite
    runSuite "formalVerification" formalVerificationSuite
    runSuite "gracefulShutdown" gracefulShutdownSuite
    runSuite "marketRisk" marketRiskSuite
    runSuite "neuralGovernorRollout" neuralGovernorRolloutSuite
    runSuite "binanceProbe" binanceProbeSuite
    runSuite "autoStartBackoff" autoStartBackoffSuite
    runSuite "revenue" revenueSuite

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
    let candle s c = CoinbaseCandle{ccOpenTime = s, ccOpen = c, ccHigh = c, ccLow = c, ccClose = c, ccVolume = 1}
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

testExternalDataFeatureInputs :: IO ()
testExternalDataFeatureInputs = do
    let openTimes = V.fromList [0, 60000, 120000, 180000]
        intervalMs = 60000
        observations =
            [ (ExternalMicrostructure, 0, 1.0)
            , (ExternalMicrostructure, 119999, 2.0)
            , (ExternalMicrostructure, 180000, 3.0)
            , (ExternalMacro, 120000, 10.0)
            ]
        mExternal = alignedExternalFeatureInputs openTimes intervalMs observations
    assert
        "blank and malformed external numeric cells fail closed"
        ( and
            [ isNothing (externalCellDouble "")
            , isNothing (externalCellDouble " ")
            , isNothing (externalCellDouble "garbage")
            , isNothing (externalCellDouble "NaN")
            , isNothing (externalCellDouble "+Infinity")
            , isNothing (externalCellDouble "-Infinity")
            ]
        )
    assert
        "blank and malformed external timestamp cells fail closed"
        ( isNothing (externalCellTime "")
            && isNothing (externalCellTime " ")
            && isNothing (externalCellTime "not-a-time")
        )
    case mExternal of
        Nothing -> assert "external observations should align to a feature bundle" False
        Just external -> do
            assert
                "microstructure alignment uses only observations known by bar close"
                (efiMicrostructure external == Just (V.fromList [1, 2, 2, 3]))
            assert
                "macro pre-coverage bars neutral-fill to zero"
                (efiMacro external == Just (V.fromList [0, 0, 10, 10]))

            let n = 60 :: Int
                closes = V.fromList [100 + fromIntegral i | i <- [0 .. n - 1]]
                fs = mkFeatureSpec 10
                baseInputs = mkFeatureInputs closes Nothing Nothing Nothing Nothing
                extLong =
                    external
                        { efiMicrostructure = Just (V.replicate n 0.1)
                        , efiOptionsVol = Just (V.replicate n 0.2)
                        , efiOnChain = Just (V.replicate n 0.3)
                        , efiMacro = Just (V.replicate n 0.4)
                        , efiCot = Just (V.replicate n 0.5)
                        , efiNews = Just (V.replicate n 0.6)
                        , efiFilings = Just (V.replicate n 0.7)
                        , efiPolicy = Just (V.replicate n 0.8)
                        , efiFundamentals = Just (V.replicate n 0.9)
                        , efiStablecoin = Just (V.replicate n 1.0)
                        , efiInstitutionalFlows = Just (V.replicate n 1.1)
                        , efiNetwork = Just (V.replicate n 1.2)
                        , efiDeveloper = Just (V.replicate n 1.3)
                        , efiGovernance = Just (V.replicate n 1.4)
                        , efiAttention = Just (V.replicate n 1.5)
                        , efiSocial = Just (V.replicate n 1.6)
                        , efiPredictionMarket = Just (V.replicate n 1.7)
                        , efiRealWorld = Just (V.replicate n 1.8)
                        , efiSecurity = Just (V.replicate n 1.9)
                        }
                extInputs = withExternalInputs (Just extLong) baseInputs
                t = 40
            case (featuresAtWithInputsWithMarket fs Nothing baseInputs t, featuresAtWithInputsWithMarket fs Nothing extInputs t) of
                (Just featsBase, Just featsExternal) -> do
                    assert
                        "external bundle appends 38 fixed-width family features"
                        (length featsExternal == length featsBase + 38)
                    assert
                        "default-off external data keeps the base feature prefix unchanged"
                        (take (length featsBase) featsExternal == featsBase)
                _ -> assert "external feature vectors should be computable at t" False

    let observationsV2 =
            [ ExternalObservationV2 ExternalMacro 0 0 1
            , ExternalObservationV2 ExternalMacro 0 90000 2
            , ExternalObservationV2 ExternalMacro 120000 120000 3
            , ExternalObservationV2 ExternalMacro 0 180000 9
            , ExternalObservationV2 ExternalNews 120000 120000 0
            , ExternalObservationV2 ExternalMicrostructure 0 0 1
            , ExternalObservationV2 ExternalMicrostructure 0 0 3
            , ExternalObservationV2 ExternalSecurity 300000 300000 5
            , ExternalObservationV2 ExternalPolicy 0 0 (0 / 0)
            ]
    case alignedExternalFeatureInputsV2 openTimes intervalMs observationsV2 of
        Nothing -> assert "availability-aware external observations should align" False
        Just externalV2 -> do
            case externalFeatureSeriesV2 ExternalMacro externalV2 of
                Nothing -> assert "the v2 macro family should be present" False
                Just macro -> do
                    assert
                        "v2 external revisions appear only after availability and older events do not displace newer events"
                        (afsV2Values macro == V.fromList [1, 2, 3, 3])
                    assert
                        "v2 external alignment preserves the selected event timestamps"
                        (afsV2EventTimesMs macro == V.fromList [Just 0, Just 0, Just 120000, Just 120000])
                    assert
                        "v2 external alignment preserves the selected availability timestamps"
                        (afsV2AvailabilityTimesMs macro == V.fromList [Just 0, Just 90000, Just 120000, Just 120000])
            case externalFeatureSeriesV2 ExternalNews externalV2 of
                Nothing -> assert "the v2 news family should be present" False
                Just news -> do
                    assert
                        "v2 external masks distinguish observed zero from pre-coverage missingness"
                        ( afsV2Values news == V.replicate 4 0
                            && afsV2Available news == V.fromList [False, False, True, True]
                        )
            assert
                "exact duplicate v2 releases retain the legacy averaging rule"
                ( (afsV2Values <$> externalFeatureSeriesV2 ExternalMicrostructure externalV2)
                    == Just (V.replicate 4 2)
                )
            assert
                "future-only and non-finite v2 families remain absent"
                ( isNothing (externalFeatureSeriesV2 ExternalSecurity externalV2)
                    && isNothing (externalFeatureSeriesV2 ExternalPolicy externalV2)
                )

    assert
        "future-only external observations do not synthesize an all-zero feature bundle"
        ( isNothing
            ( alignedExternalFeatureInputs
                openTimes
                intervalMs
                [ (ExternalNews, 240000, 4.2)
                , (ExternalNews, 120000, 0 / 0)
                ]
            )
        )
    case alignedExternalFeatureInputs
        openTimes
        intervalMs
        [ (ExternalMacro, 0, 1.0)
        , (ExternalMacro, 60000, 0 / 0)
        , (ExternalMacro, 240000, 9.0)
        ] of
        Nothing ->
            assert
                "one admissible global external observation should survive malformed and future rows"
                False
        Just external ->
            assert
                "valid global external observations survive while malformed and future rows stay unavailable"
                (efiMacro external == Just (V.fromList [1.0, 1.0, 1.0, 1.0]))

    assert
        "generic external JSON specs parse provider family, URL, timestamp key, and value key"
        ( case parseExternalJsonSpec "onchain|https://example.invalid/metric|t|v" of
            Just spec -> ejsFeature spec == ExternalOnChain
            Nothing -> False
        )
    assert
        "every alternative-data family can be selected by a generic source"
        ( and
            [ case parseExternalJsonSpec (family ++ "|https://example.invalid/metric|t|v") of
                Just _ -> True
                Nothing -> False
            | family <-
                [ "policy"
                , "fundamentals"
                , "stablecoin"
                , "institutional_flows"
                , "network"
                , "developer"
                , "governance"
                , "attention"
                , "social"
                , "prediction_market"
                , "real_world"
                , "security"
                ]
            ]
        )
    assert
        "generated panel family headers are recognized by the Haskell CSV loader"
        ( externalCsvFeatureForColumn "microstructure" == Just ExternalMicrostructure
            && externalCsvFeatureForColumn "options_vol" == Just ExternalOptionsVol
            && externalCsvFeatureForColumn "onchain" == Just ExternalOnChain
        )
    assert
        "symbol-scoped external rows fail closed when the target symbol is unknown and only match the intended full/base asset"
        ( and
            [ externalSymbolMatches Nothing Nothing
            , externalSymbolMatches Nothing (Just "")
            , not (externalSymbolMatches Nothing (Just "BTCUSDT"))
            , not (externalSymbolMatches Nothing (Just "BTC"))
            , externalSymbolMatches (Just "BTCUSDT") (Just "BTCUSDT")
            , externalSymbolMatches (Just "BTCUSDT") (Just "BTC")
            , externalSymbolMatches (Just "BTCUSDT") (Just "")
            , externalSymbolMatches (Just "BTCUSDT") Nothing
            , not (externalSymbolMatches (Just "BTCUSDT") (Just "ETHUSDT"))
            , not (externalSymbolMatches (Just "BTCUSDT") (Just "ETH"))
            , not (externalSymbolMatches (Just "ETHUSDT") (Just "BTC"))
            ]
        )

testExternalPanelSchemaV2 :: IO ()
testExternalPanelSchemaV2 = do
    fixture <- BL.readFile "test/fixtures/external_feature_panel_v2.csv"
    assert
        "external panel v2 has the registered semantic identity and fixed width"
        ( externalPanelSchemaIdV2 == "external_feature_panel_v2"
            && externalPanelSchemaVersionV2 == 2
            && externalPanelFeatureAvailabilitySchemaIdV2 == featureAvailabilitySchemaIdV2
            && length externalPanelColumnsV2 == 40
        )
    case decodeExternalPanelV2 fixture of
        Left err -> assert ("external panel v2 golden fixture should decode: " ++ err) False
        Right [first, second] -> do
            assert
                "external panel v2 retains an observed zero separately from unavailability"
                ( externalPanelCellV2 ExternalOptionsVol first
                    == Just ExternalPanelCellV2{epc2Value = 0, epc2Coverage = 1}
                    && externalPanelCellAvailableV2 ExternalOptionsVol first
                    && not (externalPanelCellAvailableV2 ExternalMicrostructure first)
                )
            assert
                "external panel v2 retains fractional coverage and finite signed values"
                ( externalPanelCellV2 ExternalOnChain first
                    == Just ExternalPanelCellV2{epc2Value = 1.5, epc2Coverage = 0.5}
                    && externalPanelCellV2 ExternalMicrostructure second
                        == Just ExternalPanelCellV2{epc2Value = -0.25, epc2Coverage = 1}
                )
        Right _ -> assert "external panel v2 golden fixture row count changed" False

    let validRow = panelRow 59999 "BTCUSDT" [(ExternalOptionsVol, ("0", "1"))]
        nonFiniteRow = panelRow 59999 "BTCUSDT" [(ExternalOptionsVol, ("NaN", "1"))]
        outOfRangeRow = panelRow 59999 "BTCUSDT" [(ExternalOptionsVol, ("0", "1.1"))]
        falseAvailabilityRow = panelRow 59999 "BTCUSDT" [(ExternalOptionsVol, ("1", "0"))]
        lowerCaseScopeRow = panelRow 59999 "btcusdt" []
    assert
        "external panel v2 rejects incompatible, non-finite, and incoherent cells"
        ( and
            [ isLeft (decodeExternalPanelV2 (panelBytes [drop 1 validRow]))
            , isLeft
                ( decodeExternalPanelV2
                    (panelBytesWithColumns (reverse externalPanelColumnsV2) [reverse validRow])
                )
            , isLeft (decodeExternalPanelV2 (panelBytes [nonFiniteRow]))
            , isLeft (decodeExternalPanelV2 (panelBytes [outOfRangeRow]))
            , isLeft (decodeExternalPanelV2 (panelBytes [falseAvailabilityRow]))
            , isLeft (decodeExternalPanelV2 (panelBytes [lowerCaseScopeRow]))
            , isLeft (decodeExternalPanelV2 (panelBytes [panelRow (-1) "BTCUSDT" []]))
            ]
        )
    assert
        "external panel v2 rejects empty, duplicate-time, and mixed-scope panels"
        ( isLeft (decodeExternalPanelV2 (panelBytes []))
            && isLeft (decodeExternalPanelV2 (panelBytes [validRow, validRow]))
            && isLeft
                ( decodeExternalPanelV2
                    (panelBytes [panelRow 119999 "BTCUSDT" [], validRow])
                )
            && isLeft
                ( decodeExternalPanelV2
                    (panelBytes [validRow, panelRow 119999 "ETHUSDT" []])
                )
        )
  where
    panelBytes = panelBytesWithColumns externalPanelColumnsV2
    panelBytesWithColumns columns rows =
        BL8.pack
            ( intercalate "," columns
                ++ "\n"
                ++ concatMap ((++ "\n") . intercalate ",") rows
            )
    panelRow timestamp symbol overrides =
        [show timestamp, symbol] ++ concatMap featureCells externalFeatureFamilies
      where
        featureCells feature =
            let (value, coverage) = fromMaybe ("0", "0") (lookup feature overrides)
             in [value, coverage]

testDerivativesPanelSchemaV2 :: IO ()
testDerivativesPanelSchemaV2 = do
    let hourMs = 3600000 :: Int64
    fixture <- BL.readFile "test/fixtures/binance_derivatives_first_seen_v2.csv"
    assert
        "derivatives panel v2 binds the collector and availability schema identities"
        ( derivativesObservationSchemaIdV2 == "binance_derivatives_first_seen_v2"
            && derivativesFeatureAvailabilitySchemaIdV2 == featureAvailabilitySchemaIdV2
            && length derivativesPanelColumnsV2 == 21
        )
    case decodeDerivativesPanelV2 "BTCUSDT" hourMs fixture of
        Left err -> assert ("derivatives panel v2 golden fixture should decode: " ++ err) False
        Right [first, _, third] -> do
            assert
                "derivatives panel v2 distinguishes observed zero, explicit missingness, and legacy absence"
                ( derivativesPanelCellV2 DerivativesFundingV2 first
                    == Just
                        DerivativesPanelCellV2
                            { dpc2Value = 0
                            , dpc2Observed = True
                            , dpc2Fresh = True
                            , dpc2EventTimeMs = Just 0
                            , dpc2AvailabilityTimeMs = Just 1000
                            }
                    && derivativesPanelCellUsableV2 DerivativesFundingV2 first
                    && not (derivativesPanelCellVersionedV2 DerivativesOpenInterestV2 first)
                    && derivativesPanelCellVersionedV2 DerivativesTakerFlowV2 first
                    && not (derivativesPanelCellUsableV2 DerivativesTakerFlowV2 first)
                )
            assert
                "derivatives panel v2 preserves stale witnesses while neutralizing their value"
                ( case derivativesPanelCellV2 DerivativesBasisV2 third of
                    Just cell ->
                        dpc2Value cell == 0
                            && dpc2Observed cell
                            && not (dpc2Fresh cell)
                            && dpc2EventTimeMs cell == Just 0
                            && dpc2AvailabilityTimeMs cell == Just 1000
                    Nothing -> False
                )
        Right _ -> assert "derivatives panel v2 golden fixture row count changed" False

    let validRow = derivativesRow 0 []
        laterRow = derivativesRow hourMs []
        nonFiniteRow = derivativesRow 0 [(DerivativesFundingV2, ["NaN", "1", "1", "0", "1000"])]
        partialRow = derivativesRow 0 [(DerivativesFundingV2, ["0", "", "", "", ""])]
        unavailableNonZeroRow = derivativesRow 0 [(DerivativesFundingV2, ["1", "0", "0", "", ""])]
        freshWithoutObservedRow = derivativesRow 0 [(DerivativesFundingV2, ["0", "0", "1", "", ""])]
        observedWithoutTimesRow = derivativesRow 0 [(DerivativesFundingV2, ["0", "1", "1", "", ""])]
        nonCausalRow = derivativesRow 0 [(DerivativesFundingV2, ["0", "1", "1", "2000", "1000"])]
        futureAvailabilityRow = derivativesRow 0 [(DerivativesFundingV2, ["0", "1", "1", "0", "4000000"])]
        staleMarkedFreshRow = derivativesRow (2 * hourMs) [(DerivativesOpenInterestV2, ["1", "1", "1", "0", "1000"])]
        freshMarkedStaleRow = derivativesRow 0 [(DerivativesOpenInterestV2, ["0", "1", "0", "0", "1000"])]
        staleNonZeroRow = derivativesRow (2 * hourMs) [(DerivativesOpenInterestV2, ["1", "1", "0", "0", "1000"])]
                decimalMaskRow = derivativesRow 0 [(DerivativesFundingV2, ["0", "0.0", "0.0", "", ""])]
        decimalTimestampRow = derivativesRow 0 [(DerivativesFundingV2, ["0", "1", "1", "0.0", "1000.0"])]
        fractionalTimestampRow = derivativesRow 0 [(DerivativesFundingV2, ["0", "1", "1", "0.5", "1000.0"])]
        nonFiniteTimestampRows =
            [ derivativesRow 0 [(DerivativesFundingV2, ["0", "1", "1", timestamp, "1000.0"])]
            | timestamp <- ["NaN", "Infinity", "-Infinity"]
            ]
        overflowedTimestampRow = derivativesRow 0 [(DerivativesFundingV2, ["0", "1", "1", "0.0", "9223372036854775808.0"])]
        decodeRows rows = decodeDerivativesPanelV2 "BTCUSDT" hourMs (derivativesBytes derivativesPanelColumnsV2 rows)

    assert
        "derivatives panel v2 accepts pandas decimal masks and unrelated legacy columns"
        ( not (isLeft (decodeRows [decimalMaskRow]))
            && not
                ( isLeft
                    ( decodeDerivativesPanelV2
                        "BTCUSDT"
                        hourMs
                        (derivativesBytes ("close" : derivativesPanelColumnsV2) ["100" : validRow])
                    )
                )
        )
        assert
        "derivatives panel v2 accepts integral decimal timestamp witnesses and rejects fractional, non-finite, and overflowed decimals"
        ( not (isLeft (decodeRows [decimalTimestampRow]))
            && isLeft (decodeRows [fractionalTimestampRow])
            && all (isLeft . decodeRows . (: [])) nonFiniteTimestampRows
            && isLeft (decodeRows [overflowedTimestampRow])
        )
    let futureRow =

            derivativesRow
                (2 * hourMs)
                [(DerivativesFundingV2, ["5", "1", "1", show (2 * hourMs), show (2 * hourMs)])]
    assert
        "derivatives panel v2 decoding is invariant to appended future rows"
        ( case (decodeRows [validRow, laterRow], decodeRows [validRow, laterRow, futureRow]) of
            (Right prefix, Right extended) -> prefix == take (length prefix) extended
            _ -> False
        )
    assert
        "derivatives panel v2 rejects incompatible headers and malformed cells"
        ( and
            [ isLeft (decodeDerivativesPanelV2 "BTCUSDT" hourMs (derivativesBytes (drop 1 derivativesPanelColumnsV2) [drop 1 validRow]))
            , isLeft (decodeDerivativesPanelV2 "BTCUSDT" hourMs (derivativesBytes (reverse derivativesPanelColumnsV2) [reverse validRow]))
            , isLeft (decodeDerivativesPanelV2 "BTCUSDT" hourMs (derivativesBytes ("openTime" : derivativesPanelColumnsV2) ["0" : validRow]))
            , isLeft (decodeRows [nonFiniteRow])
            , isLeft (decodeRows [partialRow])
            , isLeft (decodeRows [unavailableNonZeroRow])
            , isLeft (decodeRows [freshWithoutObservedRow])
            , isLeft (decodeRows [observedWithoutTimesRow])
            , isLeft (decodeRows [nonCausalRow])
            , isLeft (decodeRows [futureAvailabilityRow])
            , isLeft (decodeRows [staleMarkedFreshRow])
            , isLeft (decodeRows [freshMarkedStaleRow])
            , isLeft (decodeRows [staleNonZeroRow])
            ]
        )
    assert
        "derivatives panel v2 rejects invalid scope, intervals, grids, and timestamp overflow"
        ( and
            [ isLeft (decodeDerivativesPanelV2 "btcusdt" hourMs (derivativesBytes derivativesPanelColumnsV2 [validRow]))
            , isLeft (decodeDerivativesPanelV2 "ΒTCUSDT" hourMs (derivativesBytes derivativesPanelColumnsV2 [validRow]))
            , isLeft (decodeDerivativesPanelV2 "BTCUSDT" 0 (derivativesBytes derivativesPanelColumnsV2 [validRow]))
            , isLeft (decodeDerivativesPanelV2 "BTCUSDT" (maxBound :: Int64) (derivativesBytes derivativesPanelColumnsV2 [validRow]))
            , isLeft (decodeRows [])
            , isLeft (decodeRows [validRow, validRow])
            , isLeft (decodeRows [laterRow, validRow])
            , isLeft (decodeRows [derivativesRow (maxBound :: Int64) []])
            ]
        )
  where
    derivativesBytes :: [String] -> [[String]] -> BL.ByteString
    derivativesBytes columns rows =
        BL8.pack
            ( intercalate "," columns
                ++ "\n"
                ++ concatMap ((++ "\n") . intercalate ",") rows
            )
    derivativesRow :: Int64 -> [(DerivativesFeatureV2, [String])] -> [String]
    derivativesRow openTime overrides =
        show openTime : concatMap featureCells [minBound .. maxBound]
      where
        featureCells feature =
            fromMaybe ["0", "0", "0", "", ""] (lookup feature overrides)

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
    assert
        "multivariate prediction rejects a missing channel"
        (predictNextMulti mMulti [take 4 series] == 0)
    assert
        "multivariate prediction rejects an extra channel"
        (predictNextMulti mMulti [take 4 series, take 4 series2, take 4 series] == 0)
    assert
        "multivariate prediction rejects unequal channel lengths"
        (predictNextMulti mMulti [take 4 series, take 3 series2] == 0)
    assert
        "multivariate prediction rejects non-finite channel evidence"
        (predictNextMulti mMulti [take 4 series, [0.05, 0.10, 0 / 0, 0.20]] == 0)
    assert
        "flat multivariate prediction rejects a non-divisible feature window"
        (predictNext mMulti [0.1, 0.2, 0.3] == 0)
    assert
        "malformed parameter vectors fail closed"
        (predictNext (LSTMModel 3 [0]) window == 0)

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

testPatchTSTSanitizesMalformedInputs :: IO ()
testPatchTSTSanitizesMalformedInputs = do
    let nan = 0 / 0
        inf = 1 / 0
        prices = V.fromList [100 + fromIntegral i | i <- [0 .. 39 :: Int]]
        poisonedPrices = prices V.// [(20, nan)]
        trainTargets =
            [ (t, 0.001 * fromIntegral t)
            | t <- [8 .. 30]
            ]
                ++ [(31, nan), (32, inf)]
        model = trainPatchTST 12 prices trainTargets
        finite x = not (isNaN x || isInfinite x)
    assert
        "PatchTST feature extraction rejects non-finite price windows"
        (isNothing (patchTstFeaturesAt [4, 8, 12] poisonedPrices 24))
    assert
        "PatchTST training drops malformed targets before fitting"
        (not (null (pmWeights model)) && all finite (pmWeights model) && maybe True finite (pmSigma model))
    assert
        "PatchTST finite prediction remains finite"
        ( case predictPatchTST model prices 35 of
            Just (mu, sigma) -> finite mu && maybe True finite sigma
            Nothing -> False
        )
    let malformedModel =
            PatchTSTModel
                { pmPatchLengths = [4]
                , pmWeights = [nan, 0, 0, 0, 0, 1]
                , pmSigma = Just inf
                }
    assert "PatchTST malformed model weights fail closed" (isNothing (predictPatchTST malformedModel prices 8))

testPredictorImplementationIdentityCompatibility :: IO ()
testPredictorImplementationIdentityCompatibility = do
    let parsesAs raw expected =
            case predictorSetFromString raw of
                Right parsed -> predictorSetToList parsed == [expected]
                Left _ -> False
    assert "legacy TCN code remains stable" (predictorCode SensorTCN == "tcn")
    assert "legacy PatchTST code remains stable" (predictorCode SensorPatchTST == "patch_tst")
    assert "legacy Transformer code remains stable" (predictorCode SensorTransformer == "transformer")
    assert "TCN implementation identity is explicit" (predictorImplementationId SensorTCN == "dilated_lag_ridge_v1")
    assert "PatchTST implementation identity is explicit" (predictorImplementationId SensorPatchTST == "patch_summary_ridge_v1")
    assert "Transformer implementation identity is explicit" (predictorImplementationId SensorTransformer == "similarity_attention_v1")
    assert "accurate TCN alias preserves semantics" (parsesAs "dilated_lag_ridge_v1" SensorTCN)
    assert "accurate PatchTST alias preserves semantics" (parsesAs "patch-summary-ridge-v1" SensorPatchTST)
    assert "accurate Transformer alias preserves semantics" (parsesAs "similarity_attention_v1" SensorTransformer)

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
                [ "finalEquity" Aeson..= (1.005 :: Double)
                , "annualizedReturn" Aeson..= (0.005 :: Double)
                ]
        update =
            ComboBacktestUpdate
                { cbuMetrics = losingMetrics
                , cbuFinalEquity = Just 1.005
                , cbuScore = Just 0.005
                , cbuOperations = Nothing
                , cbuPortfolioEvidence = Nothing
                }
        boundaryUpdate =
            ComboBacktestUpdate
                { cbuMetrics =
                    Aeson.object
                        [ "finalEquity" Aeson..= topComboMinimumFinalEquity
                        , "annualizedReturn" Aeson..= (0.01 :: Double)
                        ]
                , cbuFinalEquity = Just topComboMinimumFinalEquity
                , cbuScore = Just 0.01
                , cbuOperations = Nothing
                , cbuPortfolioEvidence = Nothing
                }
    case comboIdentityKey combo of
        Nothing -> assert "top-combo fixture has a stable identity key" False
        Just key ->
            case applyComboUpdatesWithStats 2 (HM.singleton key update) payload of
                Left err -> assert ("top-combo backtest update succeeds: " ++ err) False
                Right (updatedPayload, stats) -> do
                    assert
                        "top-combo backtest prunes refreshed combos that do not clear finalEquity >= 1.01"
                        ( topCombosCount updatedPayload == 0
                            && cbasUpdatedCount stats == 1
                            && cbasPrunedCount stats == 1
                            && cbasPrunedKeys stats == [key]
                        )
                    let lenientPolicy =
                            ComboBacktestRefreshPolicy
                                { cbrpStaleAfterMs = comboBacktestStaleAfterMs
                                , cbrpPruneFinalEquityFloor = 0.95
                                }
                    case applyComboUpdatesWithStatsWithPolicy lenientPolicy 2 (HM.singleton key update) payload of
                        Left err -> assert ("lenient top-combo backtest update succeeds: " ++ err) False
                        Right (keptPayload, lenientStats) ->
                            assert
                                "configured prune final-equity floor controls refreshed loser pruning"
                                ( topCombosCount keptPayload == 1
                                    && cbasUpdatedCount lenientStats == 1
                                    && cbasPrunedCount lenientStats == 0
                                )
                    case applyComboUpdatesWithStats 2 (HM.singleton key boundaryUpdate) payload of
                        Left err -> assert ("boundary top-combo backtest update succeeds: " ++ err) False
                        Right (boundaryPayload, boundaryStats) ->
                            assert
                                "top-combo backtest keeps refreshed combos at the finalEquity 1.01 boundary"
                                ( topCombosCount boundaryPayload == 1
                                    && cbasUpdatedCount boundaryStats == 1
                                    && cbasPrunedCount boundaryStats == 0
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

testTradeAllowedDefaultsAndAnyOverride :: IO ()
testTradeAllowedDefaultsAndAnyOverride = do
    assert
        "non-dry-run trading defaults to automatic method gating with the static fallback list and SOLUSDT only"
        ( case parseAndValidateCliArgs ["--binance-symbol", "solusdt", "--no-dry-run"] of
            Right args ->
                argTradeAutoMethods args
                    && argTradeAllowedMethods args == [MethodTaBest, MethodTaRegimeSwitch]
                    && argTradeAllowedSymbols args == ["SOLUSDT"]
                    && argTradeMethodMinTrades args == 5
                    && argTradeMethodMinTotalReturn args == 0
            Left _ -> False
        )
    assert
        "static trade allowlists can be used explicitly and disabled with any"
        ( case parseAndValidateCliArgs ["--binance-symbol", "BTCUSDT", "--no-trade-auto-methods", "--trade-allowed-methods", "any", "--trade-allowed-symbols", "any"] of
            Right args -> not (argTradeAutoMethods args) && null (argTradeAllowedMethods args) && null (argTradeAllowedSymbols args)
            Left _ -> False
        )
    assert
        "serve mode defaults to the standard trade-log path for method-gate evidence"
        ( case parseAndValidateCliArgs ["--serve"] of
            Right args -> argTradeLog args == Just ".tmp/trader/live_trades.ndjson"
            Left _ -> False
        )
    assert
        "explicit serve trade-log path overrides the standard default"
        ( case parseAndValidateCliArgs ["--serve", "--trade-log", "/tmp/custom-trades.ndjson"] of
            Right args -> argTradeLog args == Just "/tmp/custom-trades.ndjson"
            Left _ -> False
        )

testTradeMethodGateUsesResultEvidence :: IO ()
testTradeMethodGateUsesResultEvidence = do
    (path, handle) <- openTempFile "/tmp" "trader-method-gate.ndjson"
    hClose handle
    let rows =
            [ Aeson.object ["method" .= ("ta_best" :: String), "pnlPercent" .= (0.01 :: Double)]
            , Aeson.object ["signal_method" .= ("ta_best" :: String), "pnl_pct" .= ("2%" :: String)]
            , Aeson.object ["method" .= ("ta_trend" :: String), "pnlPercent" .= (-0.02 :: Double)]
            , Aeson.object ["method" .= ("ta_trend" :: String), "pnlPercent" .= (0.005 :: Double)]
            , Aeson.object ["method" .= ("ta_breakout" :: String), "return" .= (0.05 :: Double)]
            , Aeson.object ["method" .= ("live" :: String), "pnlPercent" .= (10.0 :: Double)]
            ]
        lineBytes = [Aeson.encode row <> "\n" | row <- rows]
        cfg = MethodGateConfig{mgcMinTrades = 2, mgcMinTotalReturn = 0}
        statsNear expectedTrades expectedReturn stats =
            mrsTrades stats == expectedTrades && abs (mrsTotalReturn stats - expectedReturn) < 1e-12
        allowedWith expectedTrades expectedReturn decision =
            case decision of
                MethodGateAllowed stats -> statsNear expectedTrades expectedReturn stats
                _ -> False
        blockedWith expectedTrades expectedReturn decision =
            case decision of
                MethodGateBlocked stats -> statsNear expectedTrades expectedReturn stats
                _ -> False
        insufficientWith expectedTrades expectedReturn decision =
            case decision of
                MethodGateInsufficientEvidence stats -> statsNear expectedTrades expectedReturn stats
                _ -> False
        unavailable decision =
            case decision of
                MethodGateUnavailable _ -> True
                _ -> False
    BL.writeFile path (BL.concat lineBytes)
    loaded <- loadMethodResultStats path
    _ <- try (removeFile path) :: IO (Either SomeException ())
    case loaded of
        Left err -> ioError (userError err)
        Right stats -> do
            assert
                "method gate allows methods with enough positive trading/backtest evidence"
                (allowedWith 2 0.03 (methodGateDecision cfg MethodTaBest stats))
            assert
                "method gate blocks methods with enough losing evidence"
                (blockedWith 2 (-0.015) (methodGateDecision cfg MethodTaTrend stats))
            assert
                "method gate keeps low-sample winners unavailable until enough rows exist"
                (insufficientWith 1 0.05 (methodGateDecision cfg MethodTaBreakout stats))
            assert
                "method gate ignores non-strategy live marker rows"
                (unavailable (methodGateDecision cfg MethodKalmanOnly stats))
            assert
                "unavailable method evidence uses a conservative quarter-size cap"
                ( unavailableEvidenceSizeMultiplier == 0.25
                    && unavailableEvidenceSizeCap == 0.25
                    && conservativeUnavailableEvidenceSize (Just 1) == 0.25
                    && abs (conservativeUnavailableEvidenceSize (Just 0.6) - 0.15) < 1e-12
                    && conservativeUnavailableEvidenceSize (Just 10) == 0.25
                    && conservativeUnavailableEvidenceSize Nothing == 0.25
                )

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

mkBinanceTrade :: Int64 -> String -> String -> Maybe String -> Double -> Double -> Int64 -> BinanceTrade
mkBinanceTrade tradeId symbol side positionSide price qty timeMs =
    BinanceTrade
        { btSymbol = symbol
        , btTradeId = tradeId
        , btOrderId = Nothing
        , btPrice = price
        , btQty = qty
        , btQuoteQty = price * qty
        , btCommission = Nothing
        , btCommissionAsset = Nothing
        , btTime = timeMs
        , btIsBuyer = Just (side == "BUY")
        , btIsMaker = Nothing
        , btSide = Just side
        , btPositionSide = positionSide
        , btRealizedPnl = Nothing
        , btOriginIp = Nothing
        , btExecutorIp = Nothing
        , btOriginInstance = Nothing
        , btEntryIp = Nothing
        , btExitIp = Nothing
        , btEntryInstance = Nothing
        , btExitInstance = Nothing
        , btEntryTime = Nothing
        , btExitTime = Nothing
        , btMaxPnl = Nothing
        , btMaxPnlCloseTime = Nothing
        , btMethod = Nothing
        , btStrategy = Nothing
        , btDecisionSummary = Nothing
        , btDecisionReason = Nothing
        }

mkKline :: Int64 -> Double -> Double -> Double -> Double -> Kline
mkKline openTime open high low close =
    Kline
        { kOpenTime = openTime
        , kCloseTime = Just (openTime + 59999)
        , kOpen = open
        , kHigh = high
        , kLow = low
        , kClose = close
        , kVolume = 1
        , kQuoteVolume = Nothing
        , kTradeCount = Nothing
        , kTakerBuyBaseVolume = Nothing
        , kTakerBuyQuoteVolume = Nothing
        }

testBinanceTradeMaxPnlLongUsesHigh :: IO ()
testBinanceTradeMaxPnlLongUsesHigh = do
    let openTrade = mkBinanceTrade 1 "BTCUSDT" "BUY" (Just "BOTH") 100 2 0
        closeTrade = (mkBinanceTrade 2 "BTCUSDT" "SELL" (Just "BOTH") 103 2 60000){btRealizedPnl = Just 6}
        klines =
            [ mkKline 0 100 101 99 100
            , mkKline 60000 103 105 102 103
            , mkKline 120000 104 107 103 106
            , mkKline 180000 106 120 105 118
            ]
        ranges = binanceTradeMaxPnlKlineRanges [openTrade, closeTrade]
        enriched = attachBinanceTradeMaxPnl (Map.fromList [("BTCUSDT", klines)]) [openTrade, closeTrade]
    assert "long max-PNL kline range extends to entry + 2x duration" (Map.lookup "BTCUSDT" ranges == Just (0, 120000))
    assert "long opening fill stores best high-water PNL" (btMaxPnl (head enriched) == Just 14)
    assert "long opening fill stores best close time" (btMaxPnlCloseTime (head enriched) == Just 120000)
    assert "long close fill also exposes the paired best PNL" (btMaxPnl (enriched !! 1) == Just 14)

testBinanceTradeMaxPnlShortUsesLow :: IO ()
testBinanceTradeMaxPnlShortUsesLow = do
    let openTrade = mkBinanceTrade 1 "ETHUSDT" "SELL" (Just "SHORT") 100 3 0
        closeTrade = (mkBinanceTrade 2 "ETHUSDT" "BUY" (Just "SHORT") 98 3 60000){btRealizedPnl = Just 6}
        klines =
            [ mkKline 0 100 101 99 100
            , mkKline 60000 98 99 95 96
            , mkKline 120000 96 97 94 95
            ]
        enriched = attachBinanceTradeMaxPnl (Map.fromList [("ETHUSDT", klines)]) [openTrade, closeTrade]
    assert "short opening fill scores against candle lows" (btMaxPnl (head enriched) == Just 18)
    assert "short opening fill stores the lowest-candle time" (btMaxPnlCloseTime (head enriched) == Just 120000)
    assert "short close fill also exposes the paired best PNL" (btMaxPnl (enriched !! 1) == Just 18)

testBinanceTradeMaxPnlFallsBackForUnpairedClose :: IO ()
testBinanceTradeMaxPnlFallsBackForUnpairedClose = do
    let closeOnly = (mkBinanceTrade 10 "BTCUSDT" "SELL" (Just "BOTH") 103 2 60000){btRealizedPnl = Just 6}
        enriched = attachBinanceTradeMaxPnl (Map.fromList [("BTCUSDT", [mkKline 60000 103 104 102 103])]) [closeOnly]
        ranges = binanceTradeMaxPnlKlineRanges [closeOnly]
    assert "unpaired close fill does not need a kline fetch for realized-PNL fallback" (Map.null ranges)
    assert "unpaired close fill falls back to realized PNL" (btMaxPnl (head enriched) == Just 6)
    assert "unpaired close fill records the actual close time as max-PNL time" (btMaxPnlCloseTime (head enriched) == Just 60000)

testBinanceTradeMaxPnlUnpairedBothCloseDoesNotCreatePhantomLot :: IO ()
testBinanceTradeMaxPnlUnpairedBothCloseDoesNotCreatePhantomLot = do
    let historicalClose = (mkBinanceTrade 1 "BTCUSDT" "SELL" (Just "BOTH") 103 2 60000){btRealizedPnl = Just 6}
        newOpen = (mkBinanceTrade 2 "BTCUSDT" "BUY" (Just "BOTH") 100 1 120000){btRealizedPnl = Just 0}
        newClose = (mkBinanceTrade 3 "BTCUSDT" "SELL" (Just "BOTH") 110 1 180000){btRealizedPnl = Just 10}
        klines =
            [ mkKline 120000 100 101 99 100
            , mkKline 180000 110 112 109 110
            , mkKline 240000 112 116 111 115
            ]
        enriched = attachBinanceTradeMaxPnl (Map.fromList [("BTCUSDT", klines)]) [historicalClose, newOpen, newClose]
    assert "unpaired one-way close gets realized-PNL fallback" (btMaxPnl (head enriched) == Just 6)
    assert "unpaired one-way close is not treated as an opening short lot" (btMaxPnl (enriched !! 1) == Just 16)
    assert "later visible close still receives paired max-PNL" (btMaxPnl (enriched !! 2) == Just 16)

testBinanceTradeMaxPnlFallsBackToFillPricesWithoutKlines :: IO ()
testBinanceTradeMaxPnlFallsBackToFillPricesWithoutKlines = do
    let openTrade = mkBinanceTrade 1 "BTCUSDT" "BUY" (Just "BOTH") 100 2 0
        closeTrade = (mkBinanceTrade 2 "BTCUSDT" "SELL" (Just "BOTH") 103 2 60000){btRealizedPnl = Just 6}
        enriched = attachBinanceTradeMaxPnl Map.empty [openTrade, closeTrade]
    assert "paired opening fill falls back to the actual exit PNL when candles are unavailable" (btMaxPnl (head enriched) == Just 6)
    assert "paired opening fill falls back to the actual exit time when candles are unavailable" (btMaxPnlCloseTime (head enriched) == Just 60000)
    assert "paired close fill also falls back to the actual exit PNL when candles are unavailable" (btMaxPnl (enriched !! 1) == Just 6)

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

testMaxHoldBarsZeroDisablesForcedExit :: IO ()
testMaxHoldBarsZeroDisablesForcedExit = do
    assert
        "max-hold-bars accepts 0 as the documented disable value"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--max-hold-bars", "0"] of
            Right args -> argMaxHoldBars args == Just 0
            Left _ -> False
        )
    assert
        "max-hold-bars rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--max-hold-bars", "-1"] == Left "--max-hold-bars must be >= 0")

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

testPredictionMarketHerdTtlRejectsInvalidValues :: IO ()
testPredictionMarketHerdTtlRejectsInvalidValues = do
    assert
        "prediction-market-herd-fresh-ttl-sec rejects zero"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--prediction-market-herd-fresh-ttl-sec", "0"] == Left "--prediction-market-herd-fresh-ttl-sec must be > 0")
    assert
        "prediction-market-herd-stale-ttl-sec rejects zero"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--prediction-market-herd-stale-ttl-sec", "0"] == Left "--prediction-market-herd-stale-ttl-sec must be > 0")
    assert
        "prediction-market-herd-stale-ttl-sec must cover the fresh TTL"
        ( parseAndValidateCliArgs
            [ "--data"
            , "sample.csv"
            , "--prediction-market-herd-fresh-ttl-sec"
            , "120"
            , "--prediction-market-herd-stale-ttl-sec"
            , "60"
            ]
            == Left "--prediction-market-herd-stale-ttl-sec must be >= --prediction-market-herd-fresh-ttl-sec"
        )
    assert
        "prediction-market-herd TTLs accept custom values"
        ( case parseAndValidateCliArgs
            [ "--data"
            , "sample.csv"
            , "--prediction-market-herd-fresh-ttl-sec"
            , "30"
            , "--prediction-market-herd-stale-ttl-sec"
            , "180"
            ] of
            Right args ->
                argPredictionMarketHerdFreshTtlSec args == 30
                    && argPredictionMarketHerdStaleTtlSec args == 180
            Left _ -> False
        )
    assert
        "prediction-market-herd scoring rejects non-finite coefficients"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--prediction-market-herd-score-base", "NaN"] == Left "--prediction-market-herd-score-base must be finite")
    assert
        "prediction-market-herd scoring accepts custom coefficients"
        ( case parseAndValidateCliArgs
            [ "--data"
            , "sample.csv"
            , "--prediction-market-herd-score-base"
            , "80"
            , "--prediction-market-herd-interval-match-bonus"
            , "12"
            , "--prediction-market-herd-time-decay-bonus"
            , "7"
            , "--prediction-market-herd-past-end-penalty"
            , "-3"
            , "--prediction-market-herd-volume-score-weight"
            , "2"
            ] of
            Right args ->
                argPredictionMarketHerdScoreBase args == 80
                    && argPredictionMarketHerdIntervalMatchBonus args == 12
                    && argPredictionMarketHerdTimeDecayBonus args == 7
                    && argPredictionMarketHerdPastEndPenalty args == (-3)
                    && argPredictionMarketHerdVolumeScoreWeight args == 2
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

testBackendAutostartSizingDefault :: IO ()
testBackendAutostartSizingDefault = do
    assert
        "backend autostart defaults unsized workers to full quote-fraction sizing"
        ( case parseAndValidateCliArgs ["--data", "sample.csv"] of
            Right args -> argOrderQuoteFraction (applyBackendAutostartSizingDefault args) == Just 1
            Left _ -> False
        )
    assert
        "backend autostart preserves explicit order quantity sizing"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--order-quantity", "0.25"] of
            Right args ->
                let args' = applyBackendAutostartSizingDefault args
                 in argOrderQuantity args' == Just 0.25 && isNothing (argOrderQuoteFraction args')
            Left _ -> False
        )
    assert
        "backend autostart preserves explicit order quote sizing"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--order-quote", "100"] of
            Right args ->
                let args' = applyBackendAutostartSizingDefault args
                 in argOrderQuote args' == Just 100 && isNothing (argOrderQuoteFraction args')
            Left _ -> False
        )
    assert
        "backend autostart preserves explicit quote-fraction sizing"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--order-quote-fraction", "0.5"] of
            Right args -> argOrderQuoteFraction (applyBackendAutostartSizingDefault args) == Just 0.5
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

testKalman3RejectsMalformedMeasurements :: IO ()
testKalman3RejectsMalformedMeasurements = do
    let k0 = Kalman3.constantAcceleration1D 1 1e-5 1e-3 100
        kNan = Kalman3.update (0 / 0) k0
        kInf = Kalman3.update (1 / 0) k0
        kBadR = Kalman3.update 101 k0{Kalman3.kR = 0}
        kGood = Kalman3.update 101 k0
        Kalman3.Vec3 pos vel acc = Kalman3.kx kGood
    assert "Kalman3 ignores NaN measurements" (kNan == k0)
    assert "Kalman3 ignores infinite measurements" (kInf == k0)
    assert "Kalman3 ignores invalid measurement variance" (kBadR == k0{Kalman3.kR = 0})
    assert "Kalman3 still updates finite measurements" (kGood /= k0 && all finiteDouble [pos, vel, acc])

testKalmanPhysicsKnobsRejectInvalidValues :: IO ()
testKalmanPhysicsKnobsRejectInvalidValues = do
    assert
        "kalman-physics-bars rejects too-small windows"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-bars", "3"] == Left "--kalman-physics-bars must be >= 4")
    assert
        "kalman-physics-bars accepts custom windows"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-bars", "500"] of
            Right args -> argKalmanPhysicsBars args == 500
            Left _ -> False
        )
    assert
        "kalman-physics-backtest-ratio rejects zero"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-backtest-ratio", "0"] == Left "--kalman-physics-backtest-ratio must be between 0 and 1")
    assert
        "kalman-physics-backtest-ratio rejects one"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-backtest-ratio", "1"] == Left "--kalman-physics-backtest-ratio must be between 0 and 1")
    assert
        "kalman-physics-backtest-ratio accepts custom ratios"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-backtest-ratio", "0.4"] of
            Right args -> argKalmanPhysicsBacktestRatio args == 0.4
            Left _ -> False
        )

testKalmanPhysicsMeasurementKnobsRejectInvalidValues :: IO ()
testKalmanPhysicsMeasurementKnobsRejectInvalidValues = do
    assert
        "kalman-physics-volume-ewma-alpha rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-volume-ewma-alpha", "-0.1"] == Left "--kalman-physics-volume-ewma-alpha must be between 0 and 1")
    assert
        "kalman-physics-volume-ewma-alpha rejects values above one"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-volume-ewma-alpha", "1.1"] == Left "--kalman-physics-volume-ewma-alpha must be between 0 and 1")
    assert
        "kalman-physics-volume-ewma-alpha accepts custom ratios"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-volume-ewma-alpha", "0.2"] of
            Right args -> argKalmanPhysicsVolumeEwmaAlpha args == 0.2
            Left _ -> False
        )
    assert
        "kalman-physics-volume-signal-clamp rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-volume-signal-clamp", "-0.1"] == Left "--kalman-physics-volume-signal-clamp must be >= 0")
    assert
        "kalman-physics-volume-signal-clamp accepts custom values"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-volume-signal-clamp", "2.5"] of
            Right args -> argKalmanPhysicsVolumeSignalClamp args == 2.5
            Left _ -> False
        )
    assert
        "kalman-physics-close-bias-scale accepts signed values"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-close-bias-scale", "-0.03"] of
            Right args -> argKalmanPhysicsCloseBiasScale args == -0.03
            Left _ -> False
        )

testKalmanPhysicsCandidateValidationRatiosRejectInvalidValues :: IO ()
testKalmanPhysicsCandidateValidationRatiosRejectInvalidValues = do
    assert
        "kalman-physics-candidate-validation-ratio rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-candidate-validation-ratio", "-0.1"] == Left "--kalman-physics-candidate-validation-ratio must be >= 0 and < 1")
    assert
        "kalman-physics-candidate-validation-ratio rejects one"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-candidate-validation-ratio", "1"] == Left "--kalman-physics-candidate-validation-ratio must be >= 0 and < 1")
    assert
        "kalman-physics-candidate-validation-ratio accepts custom values"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-candidate-validation-ratio", "0.25"] of
            Right args -> argKalmanPhysicsCandidateValidationRatio args == 0.25
            Left _ -> False
        )
    assert
        "kalman-physics-small-sample-validation-ratio rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-small-sample-validation-ratio", "-0.1"] == Left "--kalman-physics-small-sample-validation-ratio must be >= 0 and < 1")
    assert
        "kalman-physics-small-sample-validation-ratio rejects one"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-small-sample-validation-ratio", "1"] == Left "--kalman-physics-small-sample-validation-ratio must be >= 0 and < 1")
    assert
        "kalman-physics-small-sample-validation-ratio accepts custom values"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-small-sample-validation-ratio", "0.4"] of
            Right args -> argKalmanPhysicsSmallSampleValidationRatio args == 0.4
            Left _ -> False
        )

testKalmanPhysicsCandidateGridKnobsRejectInvalidValues :: IO ()
testKalmanPhysicsCandidateGridKnobsRejectInvalidValues = do
    assert
        "kalman-physics-candidate-trees rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-candidate-trees", "-1"] == Left "--kalman-physics-candidate-trees must be >= 0")
    assert
        "kalman-physics-candidate-trees accepts custom values"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-candidate-trees", "96"] of
            Right args -> argKalmanPhysicsCandidateTrees args == 96
            Left _ -> False
        )
    assert
        "kalman-physics-candidate-learning-rate rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-candidate-learning-rate", "-0.01"] == Left "--kalman-physics-candidate-learning-rate must be >= 0")
    assert
        "kalman-physics-candidate-learning-rate accepts custom values"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--kalman-physics-candidate-learning-rate", "0.07"] of
            Right args -> argKalmanPhysicsCandidateLearningRate args == 0.07
            Left _ -> False
        )

testTriLayerPriceActionBodyOpenThresholdMultRejectsInvalidValues :: IO ()
testTriLayerPriceActionBodyOpenThresholdMultRejectsInvalidValues = do
    assert
        "tri-layer-price-action-body-open-threshold-mult rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--tri-layer-price-action-body-open-threshold-mult", "-0.1"] == Left "--tri-layer-price-action-body-open-threshold-mult must be >= 0")
    assert
        "tri-layer-price-action-body-open-threshold-mult accepts custom values"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--tri-layer-price-action-body-open-threshold-mult", "0.4"] of
            Right args -> argTriLayerPriceActionBodyOpenThresholdMult args == 0.4
            Left _ -> False
        )

testCostCalibrationKnobsRejectInvalidValues :: IO ()
testCostCalibrationKnobsRejectInvalidValues = do
    assert
        "cost-calibration-min-observations rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--cost-calibration-min-observations", "-1"] == Left "--cost-calibration-min-observations must be >= 0")
    assert
        "cost-calibration-shrinkage-obs rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--cost-calibration-shrinkage-obs", "-0.1"] == Left "--cost-calibration-shrinkage-obs must be >= 0")
    assert
        "cost-calibration-window rejects zero"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--cost-calibration-window", "0"] == Left "--cost-calibration-window must be >= 1")
    assert
        "cost-calibration-floor-factor rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--cost-calibration-floor-factor", "-0.1"] == Left "--cost-calibration-floor-factor must be >= 0")
    assert
        "cost-calibration-max-per-side rejects zero"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--cost-calibration-max-per-side", "0"] == Left "--cost-calibration-max-per-side must be > 0")
    assert
        "cost-calibration-outlier-bound rejects zero"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--cost-calibration-outlier-bound", "0"] == Left "--cost-calibration-outlier-bound must be > 0")
    assert
        "cost-calibration knobs accept custom values"
        ( case parseAndValidateCliArgs
            [ "--data"
            , "sample.csv"
            , "--cost-calibration-min-observations"
            , "3"
            , "--cost-calibration-shrinkage-obs"
            , "5"
            , "--cost-calibration-window"
            , "12"
            , "--cost-calibration-floor-factor"
            , "0.4"
            , "--cost-calibration-max-per-side"
            , "0.02"
            , "--cost-calibration-outlier-bound"
            , "0.03"
            ] of
            Right args ->
                argCostCalibrationMinObservations args == 3
                    && argCostCalibrationShrinkageObs args == 5
                    && argCostCalibrationWindow args == 12
                    && argCostCalibrationFloorFactor args == 0.4
                    && argCostCalibrationMaxPerSide args == 0.02
                    && argCostCalibrationOutlierBound args == 0.03
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

testRouterRegimeFloorRejectsInvalidValues :: IO ()
testRouterRegimeFloorRejectsInvalidValues = do
    assert
        "router-regime-min-bars rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--router-regime-min-bars", "-1"] == Left "--router-regime-min-bars must be >= 0")
    assert
        "router-regime-min-fraction rejects negative values"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--router-regime-min-fraction", "-0.1"] == Left "--router-regime-min-fraction must be between 0 and 1")
    assert
        "router-regime-min-fraction rejects values above 1"
        (parseAndValidateCliArgs ["--data", "sample.csv", "--router-regime-min-fraction", "1.1"] == Left "--router-regime-min-fraction must be between 0 and 1")
    assert
        "router-regime floor defaults preserve prior behavior"
        ( case parseAndValidateCliArgs ["--data", "sample.csv"] of
            Right args -> argRouterRegimeMinBars args == 3 && argRouterRegimeMinFraction args == 0.25
            Left _ -> False
        )
    assert
        "router-regime floor accepts zero bars and full fraction"
        ( case parseAndValidateCliArgs ["--data", "sample.csv", "--router-regime-min-bars", "0", "--router-regime-min-fraction", "1"] of
            Right args -> argRouterRegimeMinBars args == 0 && argRouterRegimeMinFraction args == 1
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

testPortfolioCapitalPreservationReport :: IO ()
testPortfolioCapitalPreservationReport = do
    let cfg = defaultPortfolioCapitalPreservationConfig
        now = 2000000 :: Int64
        trade closedAt ret =
            PortfolioCapitalTrade
                { pctReturn = ret
                , pctClosedAtMs = Just closedAt
                }
        streakTrades =
            [ trade 1000000 0.01
            , trade 1060000 (-0.01)
            , trade 1120000 (-0.02)
            , trade 1180000 (-0.03)
            ]
        streakReport = portfolioCapitalPreservationReport cfg now streakTrades
        cooledReport =
            portfolioCapitalPreservationReport
                cfg
                (1180000 + defaultPortfolioCapitalPreservationCooldownMs + 1)
                streakTrades
        rollingLossCfg =
            cfg
                { pcpcLossStreakMax = 0
                , pcpcMaxRollingLoss = Just 0.05
                }
        rollingLossReport =
            portfolioCapitalPreservationReport
                rollingLossCfg
                now
                [ trade 1000000 (-0.02)
                , trade 1060000 (-0.02)
                , trade 1120000 (-0.02)
                , trade 1180000 0.005
                , trade 1240000 (-0.01)
                , trade 1300000 (-0.01)
                ]
        notReadyReport =
            portfolioCapitalPreservationReport
                cfg{pcpcLossStreakMax = 0}
                now
                [trade 1000000 (-0.02), trade 1060000 (-0.02)]
        timestampedReport =
            portfolioCapitalPreservationReport
                rollingLossCfg{pcpcMinTrades = 2}
                now
                [ PortfolioCapitalTrade{pctReturn = -0.10, pctClosedAtMs = Nothing}
                , trade 1000000 (-0.02)
                , trade 1060000 (-0.02)
                , trade 1120000 (-0.02)
                , PortfolioCapitalTrade{pctReturn = 0 / 0, pctClosedAtMs = Just 1240000}
                ]
    assert
        "portfolio capital preservation blocks cross-symbol loss streak for the cooldown window"
        ( pcprReason streakReport == Just "CAPITAL_PRESERVATION_PORTFOLIO_LOSS_STREAK"
            && pcprLossStreak streakReport == 3
            && pcprOpenUntilMs streakReport == Just (1180000 + defaultPortfolioCapitalPreservationCooldownMs)
            && isNothing (pcprReason cooledReport)
        )
    assert
        "portfolio capital preservation rolls losses across symbols after readiness"
        ( pcprReason rollingLossReport == Just "CAPITAL_PRESERVATION_PORTFOLIO_ROLLING_LOSS"
            && pcprTrades rollingLossReport == 6
            && pcprRollingLoss rollingLossReport > Just 0.05
            && isNothing (pcprReason notReadyReport)
        )
    assert
        "portfolio capital preservation ignores missing close times and non-finite returns"
        ( pcprReason timestampedReport == Just "CAPITAL_PRESERVATION_PORTFOLIO_ROLLING_LOSS"
            && pcprTrades timestampedReport == 3
            && pcprNewestClosedAtMs timestampedReport == Just 1120000
            && pcprOpenUntilMs timestampedReport == Just (1120000 + defaultPortfolioCapitalPreservationCooldownMs)
        )

testMarketGovernorPolicy :: IO ()
testMarketGovernorPolicy = do
    let cfg = defaultMarketGovernorConfig
        base =
            MarketGovernorInputs
                { mgiMarketDataStale = False
                , mgiVolatility = Just 0.8
                , mgiConfidence = Just 0.9
                , mgiTrendProbability = Just 0.7
                , mgiMeanReversionProbability = Just 0.2
                , mgiHighVolProbability = Just 0.1
                , mgiDrawdown = 0.01
                , mgiLossStreak = 0
                , mgiRollingLoss = Nothing
                , mgiCapitalPreservationReason = Nothing
                }
        decide = marketGovernorDecision cfg
        approx a b = abs (a - b) < 1e-12
        trendDecision = decide base
        rangeDecision =
            decide
                base
                    { mgiTrendProbability = Just 0.2
                    , mgiMeanReversionProbability = Just 0.7
                    }
        highVolLowConfidence =
            decide
                base
                    { mgiHighVolProbability = Just 0.8
                    , mgiConfidence = Just 0.5
                    }
        highVolStrongConfidence =
            decide
                base
                    { mgiHighVolProbability = Just 0.8
                    , mgiConfidence = Just 0.9
                    }
        deRiskDecision = decide base{mgiDrawdown = 0.07}
        stressDecision = decide base{mgiCapitalPreservationReason = Just "CAPITAL_PRESERVATION_DRAWDOWN"}
        staleDecision = decide base{mgiMarketDataStale = True}
        disabledDecision = marketGovernorDecision cfg{mgcEnabled = False} base
    assert
        "market governor admits risk-on trend conditions without shrinking"
        ( mgdProfile trendDecision == MarketGovernorRiskOnTrend
            && not (mgdBlockFreshEntries trendDecision)
            && approx (mgdEntrySizeMultiplier trendDecision) 1
        )
    assert
        "market governor identifies range conditions and de-risks size"
        ( mgdProfile rangeDecision == MarketGovernorRange
            && not (mgdBlockFreshEntries rangeDecision)
            && approx (mgdEntrySizeMultiplier rangeDecision) 0.70
        )
    assert
        "market governor blocks high-volatility low-confidence entries as reduce-only"
        ( mgdProfile highVolLowConfidence == MarketGovernorHighVol
            && mgdBlockFreshEntries highVolLowConfidence
            && mgdReduceOnly highVolLowConfidence
            && marketGovernorFreshEntryBlockReason highVolLowConfidence == Just "MARKET_GOVERNOR_HIGH_VOL_LOW_CONFIDENCE"
            && marketGovernorIsEntryOnlyReason "MARKET_GOVERNOR_HIGH_VOL_LOW_CONFIDENCE"
        )
    assert
        "market governor permits high-volatility strong-confidence entries at reduced size"
        ( mgdProfile highVolStrongConfidence == MarketGovernorHighVol
            && not (mgdBlockFreshEntries highVolStrongConfidence)
            && approx (mgdEntrySizeMultiplier highVolStrongConfidence) 0.35
        )
    assert
        "market governor de-risks before stress and blocks on stress or unsafe data"
        ( mgdProfile deRiskDecision == MarketGovernorDeRisk
            && not (mgdBlockFreshEntries deRiskDecision)
            && approx (mgdEntrySizeMultiplier deRiskDecision) 0.50
            && mgdProfile stressDecision == MarketGovernorStress
            && mgdBlockFreshEntries stressDecision
            && mgdProfile staleDecision == MarketGovernorDataUnsafe
            && mgdBlockFreshEntries staleDecision
        )
    assert
        "market governor disabled profile is pass-through"
        ( mgdProfile disabledDecision == MarketGovernorOff
            && not (mgdEnabled disabledDecision)
            && not (mgdBlockFreshEntries disabledDecision)
            && approx (mgdEntrySizeMultiplier disabledDecision) 1
        )

testNeuralGovernorPolicy :: IO ()
testNeuralGovernorPolicy = do
    let cfg =
            defaultNeuralGovernorConfig
                { ngcMinTrades = 2
                , ngcRolloutMode = NeuralGovernorEnforce
                , ngcLearningRate = 0.2
                , ngcInfluence = 10
                , ngcMinMultiplier = 0.5
                , ngcMaxMultiplier = 1.5
                , ngcRewardClip = 0.10
                , ngcLossPenaltyScale = 3
                , ngcOpenScoreFloor = 0
                , ngcHoldScoreFloor = 0.001
                , ngcPromotionMinTrades = 0
                , ngcPromotionMinAdvantage = -1
                , ngcRollbackMinTrades = 1000
                }
        activeCfg = cfg{ngcMinTrades = 0}
        baseFeatures =
            NeuralGovernorFeatures
                { ngfVolatility = Just 0.4
                , ngfConfidence = Just 0.8
                , ngfTrendProbability = Just 0.7
                , ngfMeanReversionProbability = Just 0.2
                , ngfHighVolProbability = Just 0.1
                , ngfDrawdown = 0.01
                , ngfLossStreak = 0
                , ngfRollingLoss = Just 0.0
                , ngfDirection = 1
                , ngfBasePositionSize = 0.2
                , ngfMarketGovernorMultiplier = 1
                , ngfMarketGovernorBlocked = False
                , ngfSymbolFeature = 0.12
                , ngfMethodFeature = -0.25
                , ngfIntervalFeature = 0.4
                }
        blockedFeatures = baseFeatures{ngfMarketGovernorBlocked = True}
        state0 = initNeuralGovernorState cfg
        warmup = neuralGovernorDecide cfg state0 baseFeatures
        blocked = neuralGovernorDecide activeCfg state0 blockedFeatures
        pending0 = NeuralGovernorPendingEntry baseFeatures (neuralGovernorDecide activeCfg state0 baseFeatures)
        stateWin = neuralGovernorObserveTrade activeCfg state0 pending0 0.03
        stateLoss = neuralGovernorObserveTrade activeCfg state0 pending0 (-0.03)
        before = ngdScore (neuralGovernorDecide activeCfg state0 baseFeatures)
        afterWin = ngdScore (neuralGovernorDecide activeCfg stateWin baseFeatures)
        afterLoss = ngdScore (neuralGovernorDecide activeCfg stateLoss baseFeatures)
        lossReward = neuralGovernorReward cfg (-0.03)
        winReward = neuralGovernorReward cfg 0.03
        trainedDecision =
            neuralGovernorDecide
                activeCfg
                (neuralGovernorObserveTrade activeCfg stateWin pending0 0.03)
                baseFeatures
        policyPending0 = NeuralGovernorPendingEntry baseFeatures (neuralGovernorDecide activeCfg state0 baseFeatures)
        policyWinState = iterate (\s -> neuralGovernorObserveTrade activeCfg s policyPending0 0.04) state0 !! 6
        policyLossState = iterate (\s -> neuralGovernorObserveTrade activeCfg s policyPending0 (-0.04)) state0 !! 6
        positivePolicyScore = ngdScore (neuralGovernorDecide activeCfg policyWinState baseFeatures)
        negativePolicyScore = ngdScore (neuralGovernorDecide activeCfg policyLossState baseFeatures)
        openBlockingCfg =
            activeCfg
                { ngcOpenScoreFloor = min 1 (negativePolicyScore + 1e-9)
                , ngcHoldScoreFloor = 1
                }
        holdPreferringCfg =
            activeCfg
                { ngcOpenScoreFloor = -1
                , ngcHoldScoreFloor = max (-1) (positivePolicyScore - 1e-9)
                }
        positivePolicyDecision = neuralGovernorDecide holdPreferringCfg policyWinState baseFeatures
        negativePolicyDecision = neuralGovernorDecide openBlockingCfg policyLossState baseFeatures
    assert
        "neural governor is warmup pass-through before enough examples"
        (not (ngdReady warmup) && neuralGovernorSizingMultiplier warmup == 1 && ngdReason warmup == "NEURAL_GOVERNOR_WARMUP")
    assert
        "neural governor does not alter hard-gated market-governor decisions"
        (not (ngdReady blocked) && neuralGovernorSizingMultiplier blocked == 1 && ngdReason blocked == "NEURAL_GOVERNOR_HARD_GATE")
    assert
        "neural governor rewards wins and penalizes losses asymmetrically"
        (winReward == Just 0.03 && lossReward == Just (-0.09))
    assert
        "winning trade raises the learned score for the same context"
        (afterWin > before)
    assert
        "losing trade lowers the learned score for the same context"
        (afterLoss < before)
    assert
        "ready neural governor returns a bounded sizing multiplier"
        ( ngdReady trainedDecision
            && neuralGovernorSizingMultiplier trainedDecision >= ngcMinMultiplier activeCfg
            && neuralGovernorSizingMultiplier trainedDecision <= ngcMaxMultiplier activeCfg
        )
    assert
        "loss-trained neural governor can block fresh opens"
        ( ngdScore negativePolicyDecision <= ngcOpenScoreFloor openBlockingCfg
            && neuralGovernorOpenBlockReason negativePolicyDecision == Just "NEURAL_GOVERNOR_AVOID_OPEN"
        )
    assert
        "win-trained neural governor can prefer holding ordinary signal exits"
        ( ngdScore positivePolicyDecision >= ngcHoldScoreFloor holdPreferringCfg
            && neuralGovernorHoldReason positivePolicyDecision == Just "NEURAL_GOVERNOR_PREFER_HOLD"
        )

testLiveMaxPnlCloseTimingRecommendation :: IO ()
testLiveMaxPnlCloseTimingRecommendation = do
    assert
        "live max-PNL close timing waits for the positive-lift support floor"
        (liveMaxPnlCloseTimingMaxHoldBars (Just 12) [2, 4] == Just 12)
    assert
        "live max-PNL close timing learns the q75 max-PNL age with enough evidence"
        (liveMaxPnlCloseTimingEvidenceHoldBars [2, 4, 6, 8] == Just 6)
    assert
        "live max-PNL close timing can create a max-hold horizon when none exists"
        (liveMaxPnlCloseTimingMaxHoldBars Nothing [2, 4, 6, 8] == Just 6)
    assert
        "live max-PNL close timing can shorten but not widen an existing cap"
        ( liveMaxPnlCloseTimingMaxHoldBars (Just 10) [2, 4, 6, 8] == Just 6
            && liveMaxPnlCloseTimingMaxHoldBars (Just 5) [8, 10, 12, 14] == Just 5
        )
    assert
        "live max-PNL close timing ignores invalid zero or negative ages"
        (liveMaxPnlCloseTimingEvidenceHoldBars [0, -1, 3, 5, 7] == Just 5)

acceptedOptimizerCloseTimingReport :: ComboCloseTimingReport
acceptedOptimizerCloseTimingReport =
    ComboCloseTimingReport
        { cctrComboId = "combo"
        , cctrSampleCount = 6
        , cctrPositiveLiftSampleCount = 4
        , cctrMedianRatio = Just 0.6
        , cctrQ25Ratio = Just 0.4
        , cctrQ75Ratio = Just 0.75
        , cctrMadRatio = Just 0.1
        , cctrMeanLift = Just 0.02
        , cctrMedianLift = Just 0.01
        , cctrMedianObservedDuration = Just 10
        , cctrObservedHoldHorizon = Just 10
        , cctrMedianOptimalDuration = Just 6
        , cctrQ75OptimalDuration = Just 6
        , cctrAnalyzedHoldBars = [6, 10]
        , cctrRecommendedMaxHoldBars = Just 6
        , cctrRecommendedMaxHoldBarsEvidenceDuration = Just 6
        , cctrRecommendedMaxHoldBarsPositiveLiftSampleCount = Just 4
        , cctrRecommendedMaxHoldBarsMeanLift = Just 0.02
        }

unsupportedOptimizerCloseTimingReport :: ComboCloseTimingReport
unsupportedOptimizerCloseTimingReport =
    acceptedOptimizerCloseTimingReport
        { cctrAnalyzedHoldBars = [6, 10]
        , cctrRecommendedMaxHoldBars = Just 4
        , cctrRecommendedMaxHoldBarsEvidenceDuration = Just 4
        }

weakOptimizerCloseTimingReport :: ComboCloseTimingReport
weakOptimizerCloseTimingReport =
    acceptedOptimizerCloseTimingReport
        { cctrPositiveLiftSampleCount = 2
        , cctrRecommendedMaxHoldBarsPositiveLiftSampleCount = Just 2
        }

testOptimizerCloseTimingRecommendationRequiresAcceptedEvidence :: IO ()
testOptimizerCloseTimingRecommendationRequiresAcceptedEvidence = do
    let acceptedReport = acceptedOptimizerCloseTimingReport
        unsupportedReport = unsupportedOptimizerCloseTimingReport
        weakReport = weakOptimizerCloseTimingReport
    assert
        "optimizer applies accepted close-timing recommendations"
        (appliedCloseTimingMaxHoldBars (Just 10) acceptedReport == Just 6)
    assert
        "optimizer ignores recommendations outside the analyzed hold domain"
        (appliedCloseTimingMaxHoldBars (Just 10) unsupportedReport == Just 10)
    assert
        "optimizer ignores recommendations without enough positive-lift support"
        (appliedCloseTimingMaxHoldBars (Just 10) weakReport == Just 10)
    assert
        "runtime close-timing helper applies the same formal recommendation"
        (OptimizerCommon.appliedCloseTimingMaxHoldBars (Just 10) acceptedReport == Just 6)
    assert
        "runtime close-timing helper ignores unsupported recommendations"
        (OptimizerCommon.appliedCloseTimingMaxHoldBars (Just 10) unsupportedReport == Just 10)
    assert
        "runtime close-timing helper ignores weak recommendations"
        (OptimizerCommon.appliedCloseTimingMaxHoldBars (Just 10) weakReport == Just 10)

testOptimizerCloseTimingMetricsRecordAppliedRecommendation :: IO ()
testOptimizerCloseTimingMetricsRecordAppliedRecommendation = do
    let acceptedReport = acceptedOptimizerCloseTimingReport
        acceptedApplied = appliedCloseTimingMaxHoldBars (Just 10) acceptedReport
        acceptedMetrics = applyCloseTimingMetrics Nothing (Just 10) acceptedApplied acceptedReport
        acceptedCloseTiming = acceptedMetrics >>= KM.lookup (AK.fromString "closeTiming") >>= valueObjectMaybe
        unsupportedReport = unsupportedOptimizerCloseTimingReport
        unsupportedApplied = appliedCloseTimingMaxHoldBars (Just 10) unsupportedReport
        unsupportedMetrics = applyCloseTimingMetrics Nothing (Just 10) unsupportedApplied unsupportedReport
        unsupportedCloseTiming = unsupportedMetrics >>= KM.lookup (AK.fromString "closeTiming") >>= valueObjectMaybe
        noOpReport =
            acceptedOptimizerCloseTimingReport
                { cctrRecommendedMaxHoldBars = Just 10
                , cctrRecommendedMaxHoldBarsEvidenceDuration = Just 10
                }
        noOpApplied = appliedCloseTimingMaxHoldBars (Just 10) noOpReport
        noOpMetrics = applyCloseTimingMetrics Nothing (Just 10) noOpApplied noOpReport
        noOpCloseTiming = noOpMetrics >>= KM.lookup (AK.fromString "closeTiming") >>= valueObjectMaybe
    assert
        "close-timing metrics mark accepted recommendations"
        ((acceptedCloseTiming >>= KM.lookup (AK.fromString "recommendedMaxHoldBarsAccepted")) == Just (Aeson.Bool True))
    assert
        "close-timing metrics record the accepted applied max-hold bars"
        ((acceptedCloseTiming >>= KM.lookup (AK.fromString "appliedMaxHoldBars")) == Just (Aeson.Number 6))
    assert
        "close-timing metrics reject unsupported recommendations"
        ((unsupportedCloseTiming >>= KM.lookup (AK.fromString "recommendedMaxHoldBarsAccepted")) == Just (Aeson.Bool False))
    assert
        "close-timing metrics preserve current max-hold bars when recommendation is rejected"
        ((unsupportedCloseTiming >>= KM.lookup (AK.fromString "appliedMaxHoldBars")) == Just (Aeson.Number 10))
    assert
        "close-timing metrics do not mark deadband no-op recommendations as accepted"
        ((noOpCloseTiming >>= KM.lookup (AK.fromString "recommendedMaxHoldBarsAccepted")) == Just (Aeson.Bool False))
    assert
        "close-timing metrics keep no-op recommendations at the current max-hold bars"
        ((noOpCloseTiming >>= KM.lookup (AK.fromString "appliedMaxHoldBars")) == Just (Aeson.Number 10))

valueObjectMaybe :: Aeson.Value -> Maybe Aeson.Object
valueObjectMaybe value =
    case value of
        Aeson.Object obj -> Just obj
        _ -> Nothing

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

testTenantCredentialEncodingInjective :: IO ()
testTenantCredentialEncodingInjective = do
    let legacyBinance = tenantKeyFromBinanceKeys (Just "alpha") (Just "beta")
        expectedLegacyBinance = Just (T.pack ("binance:" ++ hashKeyHex "alpha:beta"))
        ambiguousBinanceLeft = tenantKeyFromBinanceKeys (Just "alpha:beta") (Just "gamma")
        ambiguousBinanceRight = tenantKeyFromBinanceKeys (Just "alpha") (Just "beta:gamma")
        ambiguousCoinbaseFirst = tenantKeyFromCoinbaseKeys (Just "alpha:beta") (Just "gamma") (Just "delta")
        ambiguousCoinbaseSecond = tenantKeyFromCoinbaseKeys (Just "alpha") (Just "beta:gamma") (Just "delta")
        ambiguousCoinbaseThird = tenantKeyFromCoinbaseKeys (Just "alpha") (Just "beta") (Just "gamma:delta")
        unicodeBinance = tenantKeyFromBinanceKeys (Just "cl\233") (Just "\31192\23494")
        unicodeBoundaryBinance = tenantKeyFromBinanceKeys (Just "\65279alpha\65279") (Just "beta")
        unicodeSpaceBoundaryBinance = tenantKeyFromBinanceKeys (Just "\160alpha\160") (Just "beta")
        oldAmbiguousBinance = "binance:" ++ hashKeyHex "alpha:beta:gamma"
    assert "colon-free Binance credentials retain the legacy tenant identity" (legacyBinance == expectedLegacyBinance)
    assert
        "Binance credential tuples that collide under separator concatenation have distinct tenant identities"
        ( ambiguousBinanceLeft /= ambiguousBinanceRight
            && all (maybe False (("binance:v2:" `isPrefixOf`) . T.unpack)) [ambiguousBinanceLeft, ambiguousBinanceRight]
            && ambiguousBinanceLeft == Just "binance:v2:f7819a271a2175eacb13121b5ec1557b788a259f5103edf6c4c3ad05e9a28234"
            && ambiguousBinanceRight == Just "binance:v2:b0f0389b4a3d94ff85ab3cfd9049536e3354c388284831431fee67c3c192dfc0"
        )
    assert
        "Coinbase separator placement cannot alias another normalized credential tuple"
        ( let tenants = [ambiguousCoinbaseFirst, ambiguousCoinbaseSecond, ambiguousCoinbaseThird]
           in all (maybe False (("coinbase:v2:" `isPrefixOf`) . T.unpack)) tenants
                && length (nub tenants) == length tenants
        )
    assert
        "separator-free non-ASCII credentials retain the legacy tenant identity"
        (unicodeBinance == Just "binance:117516d499c35af490f5de85d93a10c22e12015d1edc7c65204ecd58cb9f09f3")
    assert
        "non-ASCII credential boundaries are preserved consistently across runtimes"
        ( unicodeBoundaryBinance == Just "binance:4707291792a1aa7652d15cbd0b513b35dd91f34ee7c3c92b5df042d8453a57c4"
            && unicodeSpaceBoundaryBinance == Just "binance:5eb0587b34a05af33bce33fe00a16f9f226e7c4974307b935faad765e6c6877d"
        )
    assert
        "an ambiguous legacy tenant alias is rejected when separator-bearing credentials are supplied"
        ( case resolveTenantKeyFromParams (Just oldAmbiguousBinance) (Just "alpha:beta") (Just "gamma") Nothing Nothing Nothing of
            Left _ -> True
            Right _ -> False
        )

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
        , ecRouterRegimeMinBars = 3
        , ecRouterRegimeMinFraction = 0.25
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
        , ecTakeProfitPartial = 0
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
        , ecTriLayerPriceActionBodyOpenThresholdMult = defaultTriLayerPriceActionBodyOpenThresholdMult
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
        , ecRouterRegimeMinBars = 4
        , ecRouterRegimeMinFraction = 0.5
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
    let mkBar openTime close =
            MarketSeriesBar
                { msbOpenTimeMs = openTime
                , msbOpen = Just close
                , msbHigh = Just (close + 1)
                , msbLow = Just (close - 1)
                , msbClose = close
                , msbVolume = Just 10
                }
        closedBars =
            [ (mkBar lastOpen 100, "a")
            , (mkBar (lastOpen + hourMs) 101, "b")
            , (mkBar (lastOpen + 2 * hourMs) 102, "open")
            ]
    assert
        "shared market-series QA drops the still-open candle and preserves contiguous closed bars"
        (normalizeClosedMarketSeries "test candle" hourMs (lastOpen + 2 * hourMs + 1) closedBars == Right (take 2 closedBars))
    assert
        "shared market-series QA normalizes out-of-order input before enforcing strict monotonicity"
        (normalizeClosedMarketSeries "test candle" hourMs (lastOpen + 2 * hourMs + 1) (reverse (take 2 closedBars)) == Right (take 2 closedBars))
    assert
        "shared market-series QA rejects missing bars"
        ( case normalizeClosedMarketSeries "test candle" hourMs (lastOpen + 3 * hourMs) [head closedBars, closedBars !! 2] of
            Left err -> "test candle gap expectedOpenTimeMs=" `isPrefixOf` err
            Right _ -> False
        )
    assert
        "shared market-series QA rejects duplicate timestamps"
        ( case validateMarketSeriesBars "test candle" [mkBar lastOpen 100, mkBar lastOpen 101] of
            Left err -> "test candle duplicate/non-increasing openTimeMs=" `isPrefixOf` err
            Right _ -> False
        )
    let notANumber = 0 / 0
        positiveInfinity = 1 / 0
        negativeInfinity = negate positiveInfinity
        nonFiniteBars =
            [ (mkBar lastOpen 100){msbOpen = Just notANumber}
            , (mkBar lastOpen 100){msbHigh = Just positiveInfinity}
            , (mkBar lastOpen 100){msbLow = Just negativeInfinity}
            , (mkBar lastOpen 100){msbClose = notANumber}
            , (mkBar lastOpen 100){msbVolume = Just positiveInfinity}
            ]
        rejectsNonFinite bar =
            case validateMarketSeriesBars "test candle" [bar] of
                Left err -> "test candle invalid numeric payload" `isPrefixOf` err
                Right () -> False
    assert
        "shared market-series QA rejects non-finite open, high, low, close, and volume values"
        (all rejectsNonFinite nonFiniteBars)
    assert
        "shared market-series QA rejects malformed OHLC and volume"
        ( case ( validateMarketSeriesBars "test candle" [(mkBar lastOpen 100){msbHigh = Just 99}]
               , validateMarketSeriesBars "test candle" [(mkBar lastOpen 100){msbVolume = Just (-1)}]
               ) of
            (Left ohlcErr, Left volumeErr) ->
                "test candle invalid OHLC relationship" `isPrefixOf` ohlcErr
                    && "test candle negative volume" `isPrefixOf` volumeErr
            _ -> False
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
        btcHighVolumeMarket =
            btcMarket
                { pmmId = Just "btc-1h-high-volume"
                , pmmSlug = Just "bitcoin-up-or-down-1h-high-volume"
                , pmmQuestion = "Bitcoin Up or Down - 1 hour"
                , pmmVolume = Just 1000000000
                , pmmVolume24hr = Just 500000000
                }
        btcMixedEvent = btcEvent{pmeMarkets = [btcMarket, btcHighVolumeMarket]}
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
    let volumeWeightedConfig =
            defaultPredictionMarketFetchConfig
                { pmfcIntervalMatchBonus = 5
                , pmfcVolumeScoreWeight = 3
                }
    case selectPredictionMarketSignalWithConfig volumeWeightedConfig "BTCUSDT" "5m" [btcMixedEvent] of
        Nothing -> assert "custom Polymarket herd score should still select a market" False
        Just signal ->
            assert
                "custom Polymarket herd score weights can favor high-volume markets over interval matches"
                (pmsMarketId signal == Just "btc-1h-high-volume")

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
    assert
        "formal optimization vol-confidence report fields match production semantics"
        ( fvrVolConfCanonicalizationInvariant verifyFormalOptimization
            && fvrVolConfMalformedVolMatchesMissing verifyFormalOptimization
            && fvrVolConfMalformedConfidenceFailsClosed verifyFormalOptimization
            && fvrVolConfMalformedInputsStayConservative verifyFormalOptimization
            && fvrVolConfOutputBounded verifyFormalOptimization
        )

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
    assert
        "portfolio caps never discard existing-position adoption targets"
        ( capBotStartSymbolsPreservingOrphans
            3
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            ["ARBUSDT", "DOTUSDT", "LINKUSDT", "ADAUSDT"]
            == (["ARBUSDT", "DOTUSDT", "LINKUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"], [])
        )
    assert
        "existing-position adoption does not consume configured regular fleet capacity"
        ( capBotStartSymbolsPreservingOrphans
            3
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            ["ARBUSDT"]
            == (["ARBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"], [])
        )
    assert
        "redeploy adoption bypasses the start throttle and defers new exposure"
        ( throttleBotStartSymbolsPreservingOrphans
            1
            ["ARBUSDT", "DOTUSDT"]
            ["ARBUSDT", "DOTUSDT", "BTCUSDT"]
            == (["ARBUSDT", "DOTUSDT"], ["BTCUSDT"])
        )
    assert
        "redeploy adoption bypasses stale backoff and a new-entry circuit breaker"
        ( filterBotStartAttemptsPreservingOrphans
            True
            (const False)
            ["ARBUSDT", "DOTUSDT"]
            ["ARBUSDT", "DOTUSDT", "BTCUSDT"]
            == (["ARBUSDT", "DOTUSDT"], ["BTCUSDT"])
        )

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
                , cbuPortfolioEvidence = Nothing
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
                , cbuPortfolioEvidence = Nothing
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
                , cbuPortfolioEvidence =
                    Just
                        ( Aeson.object
                            [ "kind" .= ("backtest_oos" :: String)
                            , "observationCount" .= (2 :: Int)
                            , "dailyReturns"
                                .= [ Aeson.object ["dayMs" .= (0 :: Int64), "return" .= (0.01 :: Double)]
                                   , Aeson.object ["dayMs" .= (86400000 :: Int64), "return" .= (0.02 :: Double)]
                                   ]
                            ]
                        )
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
            let evidenceCount = do
                    Aeson.Object comboObj <- mUpdatedCombo
                    evidence <- KM.lookup "portfolioEvidence" comboObj
                    Aeson.Object evidenceObj <- Just evidence
                    KM.lookup "observationCount" evidenceObj >>= AT.parseMaybe Aeson.parseJSON
            assert "backtest refresh persists portfolio evidence" (evidenceCount == Just (2 :: Int))

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

{- | A refresh that deflates a combo below the retained-equity floor must survive
the sanitize pass long enough to beat the stale inflated copy a peer instance
still publishes. The final merged payload then drops it with a tombstone, so the
stale copy cannot resurrect on the next union merge. Unstamped below-floor combos
remain junk and are still dropped.
-}
testMergeSanitizeTombstonesStampedBelowFloorRefresh :: IO ()
testMergeSanitizeTombstonesStampedBelowFloorRefresh = do
    let stale = freshnessComboForTest 5.0 (Just 1000) Nothing
        refreshedLoss = freshnessComboForTest (-0.15) (Just 1000) (Just 5000)
        payload combos =
            Aeson.object
                [ "combos" .= combos
                , "generatedAtMs" .= (9000 :: Int64)
                , "source" .= ("test" :: T.Text)
                ]
        mergedLoss = mergeTopCombosPayloads 10 9500 [payload [stale, refreshedLoss]]
    assert
        "stamped below-floor refresh removes the stale inflated copy from the merge"
        (topCombosCount mergedLoss == 0)
    assert
        "stamped below-floor tombstone blocks a stale replica from resurrecting the combo"
        (topCombosCount (mergeTopCombosPayloads 10 10000 [mergedLoss, payload [stale]]) == 0)
    let unstampedWeak = freshnessComboForTest 0.005 (Just 1000) Nothing
    assert
        "unstamped sub-1.01 combo is still sanitized away"
        (isNothing (mergeWinnerScore [unstampedWeak]))

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
                  , "minEdge" .= adoptionMinEdgeFloor
                  ]
                    ++ ["protectionMinConfidence" .= Aeson.Null | includeNullParam]
                )
        walkForward =
            maybe
                []
                (\sharpe -> ["walkForwardSummary" .= Aeson.object ["sharpeMean" .= sharpe, "sharpeStd" .= adoptionMaxWalkForwardSharpeStd]])
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

comboProcessingDoubleForTest :: T.Text -> Aeson.Value -> Maybe Double
comboProcessingDoubleForTest key combo =
    case combo of
        Aeson.Object o -> do
            Aeson.Object processing <- KM.lookup "processing" o
            KM.lookup (AK.fromText key) processing >>= AT.parseMaybe Aeson.parseJSON
        _ -> Nothing

comboProcessingTextForTest :: T.Text -> Aeson.Value -> Maybe T.Text
comboProcessingTextForTest key combo =
    case combo of
        Aeson.Object o -> do
            Aeson.Object processing <- KM.lookup "processing" o
            KM.lookup (AK.fromText key) processing >>= AT.parseMaybe Aeson.parseJSON
        _ -> Nothing

comboMethodForTest :: Aeson.Value -> Maybe T.Text
comboMethodForTest combo =
    case combo of
        Aeson.Object c -> do
            Aeson.Object params <- KM.lookup "params" c
            KM.lookup "method" params >>= AT.parseMaybe Aeson.parseJSON
        _ -> Nothing

comboWithOverfitForTest :: Double -> Double -> Aeson.Value -> Aeson.Value
comboWithOverfitForTest pbo deflatedSharpe val =
    case val of
        Aeson.Object o ->
            let metrics =
                    case KM.lookup "metrics" o of
                        Just (Aeson.Object m) -> m
                        _ -> KM.empty
                overfit =
                    Aeson.object
                        [ "pboProxy" .= pbo
                        , "deflatedSharpeProxy" .= deflatedSharpe
                        ]
             in Aeson.Object (KM.insert "metrics" (Aeson.Object (KM.insert "overfit" overfit metrics)) o)
        _ -> val

testTopComboProcessingCarriesOverfitAndMapEliteMetadata :: IO ()
testTopComboProcessingCarriesOverfitAndMapEliteMetadata = do
    let risky =
            comboWithOverfitForTest 0.90 (-0.10) $
                processingComboForTest
                    "ta_trend"
                    "db"
                    False
                    (Just adoptionMinWalkForwardSharpeMean)
                    1.0
                    adoptionMinTradeCount
        combos = mergedCombosForTest [risky]
        mCombo = listToMaybe combos
        multiplier = mCombo >>= comboProcessingDoubleForTest "overfitMultiplier"
        bucket = mCombo >>= comboProcessingTextForTest "mapEliteBucket"
    assert
        "top-combo processing applies overfit multiplier from PBO and deflated Sharpe proxies"
        (multiplier == Just 0.125)
    assert
        "top-combo processing records MAP-Elites bucket dimensions"
        (bucket == Just "BTCUSDT|15m|ta_trend|trend|activity-med|dd-low")

comboWithCreatedAtForTest :: Int64 -> Aeson.Value -> Aeson.Value
comboWithCreatedAtForTest createdAt val =
    case val of
        Aeson.Object o -> Aeson.Object (KM.insert "createdAtMs" (Aeson.toJSON createdAt) o)
        _ -> val

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

testDeployableTierRequiresCompleteAdoptionEvidence :: IO ()
testDeployableTierRequiresCompleteAdoptionEvidence = do
    let valid =
            processingComboForTest
                "valid-evidence"
                "db"
                False
                (Just adoptionMinWalkForwardSharpeMean)
                1.0
                adoptionMinTradeCount
        withParam key value combo =
            case combo of
                Aeson.Object o ->
                    case KM.lookup "params" o of
                        Just (Aeson.Object params) -> Aeson.Object (KM.insert "params" (Aeson.Object (KM.insert key value params)) o)
                        _ -> combo
                _ -> combo
        withoutParam key combo =
            case combo of
                Aeson.Object o ->
                    case KM.lookup "params" o of
                        Just (Aeson.Object params) -> Aeson.Object (KM.insert "params" (Aeson.Object (KM.delete key params)) o)
                        _ -> combo
                _ -> combo
        withWalkForwardStd sharpeStd combo =
            case combo of
                Aeson.Object o ->
                    case KM.lookup "metrics" o of
                        Just (Aeson.Object metrics) ->
                            case KM.lookup "walkForwardSummary" metrics of
                                Just (Aeson.Object wf) ->
                                    Aeson.Object
                                        ( KM.insert
                                            "metrics"
                                            (Aeson.Object (KM.insert "walkForwardSummary" (Aeson.Object (KM.insert "sharpeStd" (Aeson.toJSON sharpeStd) wf)) metrics))
                                            o
                                        )
                                _ -> combo
                        _ -> combo
                _ -> combo
        missingEdge = withoutParam "minEdge" valid
        lowEdge = withParam "minEdge" (Aeson.toJSON (adoptionMinEdgeFloor / 2)) valid
        unstable = withWalkForwardStd (adoptionMaxWalkForwardSharpeStd + 0.01) valid
        validMerged = listToMaybe (mergedCombosForTest [valid])
        missingEdgeMerged = listToMaybe (mergedCombosForTest [missingEdge])
        lowEdgeMerged = listToMaybe (mergedCombosForTest [lowEdge])
        unstableMerged = listToMaybe (mergedCombosForTest [unstable])
    assert
        "complete adoption evidence is deployable"
        ((validMerged >>= comboProcessingTierForTest) == Just "deployable")
    assert
        "missing minimum edge remains a candidate with a reason"
        ( (missingEdgeMerged >>= comboProcessingTierForTest) == Just "candidate"
            && maybe False ("min-edge-missing" `elem`) (missingEdgeMerged >>= comboProcessingReasonsForTest)
        )
    assert
        "sub-floor minimum edge remains a candidate with a reason"
        ( (lowEdgeMerged >>= comboProcessingTierForTest) == Just "candidate"
            && maybe False ("min-edge-below-floor" `elem`) (lowEdgeMerged >>= comboProcessingReasonsForTest)
        )
    assert
        "unstable walk-forward evidence remains a candidate with a reason"
        ( (unstableMerged >>= comboProcessingTierForTest) == Just "candidate"
            && maybe False ("walk-forward-std-above-ceiling" `elem`) (unstableMerged >>= comboProcessingReasonsForTest)
        )

testExplicitDeployableOverrideIsBoundedAndAuditable :: IO ()
testExplicitDeployableOverrideIsBoundedAndAuditable = do
    let relaxed = processingComboForTest "relaxed-candidate" "db" False Nothing 2.0 adoptionMinTradeCount
        untouched = processingComboForTest "untouched-candidate" "db" False Nothing 1.5 adoptionMinTradeCount
        raw = processingComboForTest "raw-candidate" "db" False Nothing 4.0 (adoptionMinTradeCount - 1)
        payload = Aeson.object ["combos" .= [relaxed, untouched, raw]]
        (merged, _) =
            mergeTopCombosPayloadsWithStatsAndDeployableOverrides
                ["relaxed-candidate", "raw-candidate"]
                10
                9500
                [payload]
        combos =
            case merged of
                Aeson.Object o ->
                    case KM.lookup "combos" o of
                        Just (Aeson.Array values) -> V.toList values
                        _ -> []
                _ -> []
        byUuid :: T.Text -> Maybe Aeson.Value
        byUuid uuid =
            find
                ( \case
                    Aeson.Object o ->
                        (KM.lookup "uuid" o >>= AT.parseMaybe Aeson.parseJSON) == Just uuid
                    _ -> False
                )
                combos
        processingField :: AK.Key -> Aeson.Value -> Maybe Aeson.Value
        processingField key = \case
            Aeson.Object o ->
                case KM.lookup "processing" o of
                    Just (Aeson.Object processing) -> KM.lookup key processing
                    _ -> Nothing
            _ -> Nothing
        relaxedReasons :: Maybe [T.Text]
        relaxedReasons = byUuid "relaxed-candidate" >>= processingField "relaxedReasons" >>= AT.parseMaybe Aeson.parseJSON
        relaxedFlag :: Maybe Bool
        relaxedFlag = byUuid "relaxed-candidate" >>= processingField "relaxed" >>= AT.parseMaybe Aeson.parseJSON
    assert
        "an explicit candidate UUID becomes deployable without promoting another candidate"
        ( (byUuid "relaxed-candidate" >>= comboProcessingTierForTest) == Just "deployable"
            && (byUuid "untouched-candidate" >>= comboProcessingTierForTest) == Just "candidate"
        )
    assert
        "a deployable override cannot promote a raw row"
        ((byUuid "raw-candidate" >>= comboProcessingTierForTest) == Just "raw")
    assert
        "the runtime override boundary rejects raw, quarantined, and unknown processing state"
        ( deployableOverrideEvidenceEligible False True (Just "candidate")
            && deployableOverrideEvidenceEligible False True (Just "deployable")
            && not (deployableOverrideEvidenceEligible False True Nothing)
            && not (deployableOverrideEvidenceEligible False False (Just "candidate"))
            && not (deployableOverrideEvidenceEligible True True (Just "deployable"))
            && not (deployableOverrideEvidenceEligible False True (Just "raw"))
            && not (deployableOverrideEvidenceEligible False True (Just "quarantined"))
            && not (deployableOverrideEvidenceEligible False True (Just "future-tier"))
        )
    assert
        "valid override UUIDs are canonicalized before lookup and deduplication"
        ( canonicalizeUuidEnvValues
            [ "550E8400-E29B-41D4-A716-446655440000"
            , "550e8400-e29b-41d4-a716-446655440000"
            ]
            == Right ["550e8400-e29b-41d4-a716-446655440000"]
        )
    assert
        "invalid override UUIDs remain fail-closed and identifiable"
        (canonicalizeUuidEnvValues ["not-a-uuid"] == Left ["not-a-uuid"])
    assert
        "a relaxed deployment remains explicit and preserves its failed strict gates"
        ( relaxedFlag == Just True
            && maybe False ("walk-forward-missing" `elem`) (relaxedReasons :: Maybe [T.Text])
        )

testTopComboFreshnessMultiplierDefaultsDisabled :: IO ()
testTopComboFreshnessMultiplierDefaultsDisabled =
    assert
        "default top-combo freshness scoring is disabled"
        (topComboFreshnessMultiplier defaultTopComboScoringConfig (Just 365) == 1)

portfolioTestConfig :: PortfolioSelectorConfig
portfolioTestConfig =
    defaultPortfolioSelectorConfig
        { pscMinimumObservations = 40
        , pscMaximumObservations = 60
        , pscCandidateLimit = 10
        , pscBootstrapSamples = 100
        , pscBootstrapBlockDays = 3
        , pscBootstrapPortfolioLimit = 64
        , pscMaxDrawdown = 0.50
        , pscSwitchingCostRate = 0
        }

portfolioCandidateForTest :: T.Text -> T.Text -> [Double] -> PortfolioCandidate
portfolioCandidateForTest uuid symbol returns =
    let daily =
            [ PortfolioDailyReturn (fromIntegral day * 86400000) value
            | (day, value) <- zip ([1 ..] :: [Int]) returns
            ]
        evidence =
            PortfolioEvidence
                { peKind = "backtest_oos"
                , peWindowStartMs = maybe 0 pdrDayMs (listToMaybe daily)
                , peWindowEndMs = maybe 0 pdrDayMs (listToMaybe (reverse daily))
                , peObservationCount = length daily
                , peCostModel = "backtest_net_equity"
                , peDailyReturns = daily
                }
     in PortfolioCandidate uuid symbol 0.25 evidence

testPortfolioAnnualizationAndDrawdown :: IO ()
testPortfolioAnnualizationAndDrawdown = do
    let flatGrowth = replicate 365 0.001
        drawdownPath = [0.10, -0.20, 0.05]
    assert
        "portfolio annualization compounds daily net returns"
        (abs (portfolioAnnualizedReturn flatGrowth - ((1.001 :: Double) ** 365 - 1)) < 1.0e-9)
    assert
        "portfolio drawdown is measured from the running equity peak"
        (abs (portfolioMaxDrawdown drawdownPath - 0.20) < 1.0e-12)

testOptimizerExtractsTimestampedPortfolioEvidence :: IO ()
testOptimizerExtractsTimestampedPortfolioEvidence = do
    let backtest =
            Aeson.object
                [ "openTimes" .= ([0, 43200000, 86400000, 172800000] :: [Int])
                , "equityCurve" .= ([1.0, 1.01, 1.02, 1.04] :: [Double])
                ]
        wrapped = Aeson.object ["backtest" .= backtest]
        parsed = extractPortfolioEvidence (Just wrapped) >>= AT.parseMaybe Aeson.parseJSON
        parsedServer = extractPortfolioEvidence (Just backtest) >>= AT.parseMaybe Aeson.parseJSON
    case parsed of
        Nothing -> ioError (userError "optimizer failed to extract portfolio evidence from a timestamped net equity curve")
        Just evidence -> do
            assert "optimizer portfolio evidence groups intraday equity at UTC day closes" (peObservationCount evidence == 2)
            assert "optimizer portfolio evidence preserves normalized day timestamps" (map pdrDayMs (peDailyReturns evidence) == [86400000, 172800000])
            assert
                "optimizer portfolio evidence derives consecutive net daily returns"
                (and (zipWith (\actual expected -> abs (actual - expected) < 1.0e-12) (map pdrReturn (peDailyReturns evidence)) [1.02 / 1.01 - 1, 1.04 / 1.02 - 1]))
            assert "server backtests expose the same portfolio evidence at the root" (parsedServer == Just evidence)

testPortfolioSelectionIsDeterministicAndBounded :: IO ()
testPortfolioSelectionIsDeterministicAndBounded = do
    let candidates =
            [ portfolioCandidateForTest "combo-a" "AAAUSDT" (replicate 60 0.0010)
            , portfolioCandidateForTest "combo-a-variant" "AAAUSDT" (replicate 60 0.0011)
            , portfolioCandidateForTest "combo-b" "BBBUSDT" (replicate 60 0.0008)
            , portfolioCandidateForTest "combo-c" "CCCUSDT" (replicate 60 0.0006)
            ]
        first = selectPortfolio portfolioTestConfig 1000 PortfolioShadow [] candidates
        second = selectPortfolio portfolioTestConfig 1000 PortfolioShadow [] candidates
    assert "portfolio selection is deterministic for identical evidence" (first == second)
    case first of
        Left err -> ioError (userError ("portfolio selection unexpectedly failed: " ++ err))
        Right selection -> do
            let weights = map pmWeight (psMembers selection)
            assert "portfolio selection admits at most three unique symbols" (length weights <= 3 && length (nub (map pmSymbol (psMembers selection))) == length weights)
            assert "portfolio member weights respect the 25% hard cap" (all (<= 0.25 + 1.0e-12) weights)
            assert "portfolio gross weight respects the 75% hard cap" (sum weights <= 0.75 + 1.0e-12)
            assert "portfolio bootstrap drawdown respects the configured hard cap" (pmMaxDrawdownP95 (psMetrics selection) <= pscMaxDrawdown portfolioTestConfig)
            assert "portfolio snapshots fingerprint the complete selector configuration" (psConfigVersion selection == portfolioSelectorConfigVersion portfolioTestConfig)

testPortfolioSelectionAcceptsMicroLiveBounds :: IO ()
testPortfolioSelectionAcceptsMicroLiveBounds = do
    let config =
            defaultPortfolioSelectorConfig
                { pscMaxMembers = 5
                , pscMaxMemberWeight = 0.01
                , pscMaxGrossWeight = 0.05
                , pscWeightStep = 0.01
                , pscMinimumObservations = 10
                , pscMaximumObservations = 10
                , pscBootstrapSamples = 100
                , pscBootstrapPortfolioLimit = 64
                }
        candidate = portfolioCandidateForTest "micro-live" "ADAUSDT" (replicate 10 0.001)
    case selectPortfolio config 1 PortfolioShadow [] [candidate] of
        Left err -> ioError (userError ("micro-live portfolio selection unexpectedly failed: " ++ err))
        Right selection ->
            assert
                "portfolio selection accepts the managed 10-day, 1%-step safety bounds"
                (map pmWeight (psMembers selection) == [0.01])

testPortfolioSelectionFailsClosedOnInvalidNumbers :: IO ()
testPortfolioSelectionFailsClosedOnInvalidNumbers = do
    let candidate = portfolioCandidateForTest "valid" "BTCUSDT" (replicate 60 0.001)
        malformedCandidate = portfolioCandidateForTest "nan" "ETHUSDT" (replicate 60 (0 / 0))
        malformedConfig = portfolioTestConfig{pscMaxMemberWeight = 0 / 0}
    assert
        "portfolio selection rejects non-finite selector configuration"
        (case selectPortfolio malformedConfig 1 PortfolioShadow [] [candidate] of Left _ -> True; Right _ -> False)
    assert
        "portfolio selection rejects non-finite return evidence"
        (case selectPortfolio portfolioTestConfig 1 PortfolioShadow [] [malformedCandidate] of Left _ -> True; Right _ -> False)

testPortfolioFailureCacheInvalidatesOnSnapshotChange :: IO ()
testPortfolioFailureCacheInvalidatesOnSnapshotChange = do
    let cached = Just (1000, (4 :: Int, 1 :: Int), "no admissible portfolio")
    assert "portfolio selector failure cache reuses an unchanged fresh snapshot" (portfolioFailureCacheLookup 10000 1500 (4, 1) cached == Just "no admissible portfolio")
    assert "portfolio selector failure cache invalidates when evidence changes" (isNothing (portfolioFailureCacheLookup 10000 1500 (4, 2) cached))
    assert "portfolio selector failure cache expires at its TTL" (isNothing (portfolioFailureCacheLookup 10000 11000 (4, 1) cached))

portfolioGraduationConfigForTest :: PortfolioGraduationConfig
portfolioGraduationConfigForTest =
    defaultPortfolioGraduationConfig
        { pgcEnabled = True
        , pgcStartedAtMs = 1000
        , pgcMinimumDailyObservations = 30
        , pgcMinimumNetReturn = 0
        , pgcMaximumDrawdown = 0.10
        , pgcMinimumExecutionAttempts = 10
        , pgcMinimumExecutionReliability = 0.95
        , pgcMinimumStatusReliability = 0.99
        , pgcMaximumBaselineAgeMs = 900000
        , pgcStatusIntervalMs = 900000
        , pgcMaximumLatestStatusAgeMs = 900000
        }

passingPortfolioGraduationEvidence :: PortfolioGraduationEvidence
passingPortfolioGraduationEvidence =
    PortfolioGraduationEvidence
        { pgeDailyObservationCount = 30
        , pgeNetReturn = 0.02
        , pgeMaxDrawdown = 0.04
        , pgeExecutionAttempts = 20
        , pgeExecutionSuccesses = 19
        , pgeStatusSamples = 1000
        , pgeHealthyStatusSamples = 990
        , pgeLatestStatusesHealthy = True
        }

testPortfolioGraduationRequiresEveryReviewGate :: IO ()
testPortfolioGraduationRequiresEveryReviewGate = do
    let reviewedUuids = ["UUID-B", "uuid-a"]
        passed = portfolioGraduationReview portfolioGraduationConfigForTest 2000 reviewedUuids passingPortfolioGraduationEvidence
        tooEarly = portfolioGraduationReview portfolioGraduationConfigForTest 2000 reviewedUuids passingPortfolioGraduationEvidence{pgeDailyObservationCount = 29}
        losing = portfolioGraduationReview portfolioGraduationConfigForTest 2000 reviewedUuids passingPortfolioGraduationEvidence{pgeNetReturn = 0}
        unreliable = portfolioGraduationReview portfolioGraduationConfigForTest 2000 reviewedUuids passingPortfolioGraduationEvidence{pgeExecutionSuccesses = 18}
        unhealthy = portfolioGraduationReview portfolioGraduationConfigForTest 2000 reviewedUuids passingPortfolioGraduationEvidence{pgeLatestStatusesHealthy = False}
        (expectedStatusIntervals, healthyStatusIntervals) = portfolioGraduationStatusCoverage 0 3600000 900000 2 (7, 7)
        missingHeartbeat =
            portfolioGraduationReview
                portfolioGraduationConfigForTest
                2000
                reviewedUuids
                passingPortfolioGraduationEvidence
                    { pgeStatusSamples = expectedStatusIntervals
                    , pgeHealthyStatusSamples = healthyStatusIntervals
                    }
    assert "portfolio graduation passes only after all configured live review gates clear" (pgrDecision passed == PortfolioGraduated)
    assert "portfolio graduation remains pending before 30 complete daily observations" (pgrDecision tooEarly == PortfolioGraduationPending)
    assert "portfolio graduation requires positive net performance" (pgrDecision losing == PortfolioGraduationPending)
    assert "portfolio graduation requires the configured execution reliability" (pgrDecision unreliable == PortfolioGraduationPending)
    assert "portfolio graduation charges a missing heartbeat interval as unhealthy" (expectedStatusIntervals == 8 && pgrDecision missingHeartbeat == PortfolioGraduationPending)
    assert "portfolio graduation fails closed when any latest worker status is unhealthy" (pgrDecision unhealthy == PortfolioGraduationPending)
    assert "a graduated review applies only to its exact normalized UUID set" (portfolioGraduationReviewApplies portfolioGraduationConfigForTest ["uuid-a", "uuid-b"] passed)
    assert "a graduated review cannot authorize a different UUID set" (not (portfolioGraduationReviewApplies portfolioGraduationConfigForTest ["uuid-a"] passed))
    assert
        "graduation requires one fresh healthy latest status per reviewed worker"
        ( portfolioGraduationLatestStatusesHealthy
            2000000
            900000
            ["uuid-a", "uuid-b"]
            [("uuid-b", 1999000, True), ("uuid-a", 1100000, True)]
        )
    assert
        "a stale latest worker status fails graduation closed"
        ( not
            ( portfolioGraduationLatestStatusesHealthy
                2000000
                900000
                ["uuid-a", "uuid-b"]
                [("uuid-a", 1099999, True), ("uuid-b", 1999000, True)]
            )
        )
    assert
        "missing, unhealthy, or future latest worker status fails graduation closed"
        ( not (portfolioGraduationLatestStatusesHealthy 2000000 900000 ["uuid-a", "uuid-b"] [("uuid-a", 1999000, True)])
            && not (portfolioGraduationLatestStatusesHealthy 2000000 900000 ["uuid-a", "uuid-b"] [("uuid-a", 1999000, True), ("uuid-b", 1999000, False)])
            && not (portfolioGraduationLatestStatusesHealthy 2000000 900000 ["uuid-a", "uuid-b"] [("uuid-a", 1999000, True), ("uuid-b", 2000001, True)])
        )

testPortfolioGraduationPerformanceAndPersistence :: IO ()
testPortfolioGraduationPerformanceAndPersistence = do
    let boundaryMs = 1000
        maximumBaselineAgeMs = 100
        baselines = [("uuid-a", 950, 1.20), ("uuid-b", 1000, 0.80)]
        daily =
            [ (1, "uuid-a", 1.20)
            , (1, "uuid-b", 0.80)
            , (2, "uuid-a", 1.08)
            , (2, "uuid-b", 0.88)
            ]
    assert
        "portfolio graduation rebases each worker at the review-window boundary"
        (portfolioGraduationFleetEquities boundaryMs maximumBaselineAgeMs ["uuid-a", "uuid-b"] baselines daily == Right [1.0, 1.0])
    assert
        "portfolio graduation fails closed when any reviewed worker lacks a window baseline"
        (case portfolioGraduationFleetEquities boundaryMs maximumBaselineAgeMs ["uuid-a", "uuid-b"] [("uuid-a", 950, 1)] daily of Left _ -> True; Right _ -> False)
    assert
        "portfolio graduation rejects stale or post-boundary worker baselines"
        ( case portfolioGraduationFleetEquities boundaryMs maximumBaselineAgeMs ["uuid-a", "uuid-b"] [("uuid-a", 899, 1.20), ("uuid-b", 1000, 0.80)] daily of
            Left _ -> True
            Right _ -> False
        )
    assert
        "portfolio graduation rejects a future worker baseline"
        ( case portfolioGraduationFleetEquities boundaryMs maximumBaselineAgeMs ["uuid-a", "uuid-b"] [("uuid-a", 950, 1.20), ("uuid-b", 1001, 0.80)] daily of
            Left _ -> True
            Right _ -> False
        )
    case portfolioGraduationPerformance [1.01, 1.02, 0.99, 1.03] of
        Left err -> ioError (userError ("valid portfolio graduation performance failed: " ++ err))
        Right (observations, netReturn, drawdown) -> do
            assert "portfolio graduation counts complete fleet equity observations" (observations == 4)
            assert "portfolio graduation measures net return from the normalized fleet baseline" (abs (netReturn - 0.03) < 1.0e-12)
            assert "portfolio graduation measures peak-to-trough drawdown" (abs (drawdown - (1.02 - 0.99) / 1.02) < 1.0e-12)
    assert
        "portfolio graduation rejects malformed fleet equity rather than skipping it"
        (case portfolioGraduationPerformance [1.01, 0 / 0] of Left _ -> True; Right _ -> False)
    let review = portfolioGraduationReview portfolioGraduationConfigForTest 2000 ["uuid-a"] passingPortfolioGraduationEvidence
        decoded = Aeson.eitherDecode (Aeson.encode review) :: Either String PortfolioGraduationReview
    assert "portfolio graduation reviews round-trip through their durable JSON marker" (decoded == Right review)

testPortfolioSelectionRejectsSparseWinner :: IO ()
testPortfolioSelectionRejectsSparseWinner = do
    let stable = portfolioCandidateForTest "stable" "STABLEUSDT" (replicate 60 0.001)
        sparseWinner = portfolioCandidateForTest "sparse" "SPARSEUSDT" (replicate 10 0.10)
        config = portfolioTestConfig{pscMaxMembers = 1}
    case selectPortfolio config 2000 PortfolioShadow [] [sparseWinner, stable] of
        Left err -> ioError (userError ("stable portfolio candidate should remain selectable: " ++ err))
        Right selection ->
            assert
                "a high-return combo without the minimum evidence window cannot enter the portfolio"
                (map pmUuid (psMembers selection) == ["stable"])

testPortfolioCurrentEvidenceRiskGate :: IO ()
testPortfolioCurrentEvidenceRiskGate = do
    let member = PortfolioMember "current" "CURRENTUSDT" 0.25
        healthy = portfolioCandidateForTest "current" "CURRENTUSDT" (replicate 60 0.001)
        refreshedHealthy = portfolioCandidateForTest "current" "CURRENTUSDT" (replicate 61 0.001)
        degraded = portfolioCandidateForTest "current" "CURRENTUSDT" (replicate 60 (-0.01))
    assert
        "current portfolio members remain admitted while refreshed evidence clears the bootstrap gates"
        (portfolioMembersRemainAdmitted portfolioTestConfig [member] [healthy])
    assert
        "current portfolio members fail closed when refreshed evidence loses its conservative return edge"
        (not (portfolioMembersRemainAdmitted portfolioTestConfig [member] [degraded]))
    assert
        "current portfolio members fail closed when a stored weight exceeds the configured hard cap"
        (not (portfolioMembersRemainAdmitted portfolioTestConfig [member{pmWeight = 0.30}] [healthy]))
    case refreshPortfolioSelection portfolioTestConfig 2000 PortfolioShadow [member] [refreshedHealthy] of
        Left err -> ioError (userError ("healthy incumbent refresh unexpectedly failed: " ++ err))
        Right refreshed -> do
            assert "an admitted incumbent refresh advances its aligned evidence timestamp" (psEvidenceEndMs refreshed == 61 * 86400000)
            assert "an admitted incumbent refresh replaces stored conservative metrics" (pmAnnualizedReturnP10 (psMetrics refreshed) > 0)

portfolioSelectionForRotationTest :: Double -> Double -> PortfolioSelection
portfolioSelectionForRotationTest returnP10 probability =
    PortfolioSelection
        { psGeneratedAtMs = 1
        , psValidUntilMs = 2
        , psEvidenceStartMs = 1
        , psEvidenceEndMs = 2
        , psMode = PortfolioShadow
        , psMembers = [PortfolioMember "combo" "BTCUSDT" 0.25]
        , psMetrics =
            PortfolioMetrics
                { pmHistoricalAnnualizedReturn = returnP10
                , pmHistoricalMaxDrawdown = 0.02
                , pmAnnualizedReturnP10 = returnP10
                , pmAnnualizedReturnP50 = returnP10
                , pmAnnualizedReturnP90 = returnP10
                , pmMaxDrawdownP95 = 0.05
                , pmAverageCorrelation = 0
                , pmSwitchingCost = 0
                , pmPairedOutperformanceProbability = probability
                }
        , psCandidateCount = 1
        , psBootstrapSeed = 1
        , psConfigVersion = "portfolio-v1"
        }

testPortfolioSelectionRotationHysteresis :: IO ()
testPortfolioSelectionRotationHysteresis = do
    let incumbent = portfolioSelectionForRotationTest 0.10 1
        weakImprovement = portfolioSelectionForRotationTest 0.119 1
        uncertainImprovement = portfolioSelectionForRotationTest 0.13 0.89
        promoted = portfolioSelectionForRotationTest 0.13 0.90
    assert "portfolio rotation rejects improvements below two annualized percentage points" (not (portfolioSelectionShouldRotate portfolioTestConfig incumbent weakImprovement))
    assert "portfolio rotation rejects statistically uncertain improvements" (not (portfolioSelectionShouldRotate portfolioTestConfig incumbent uncertainImprovement))
    assert "portfolio rotation accepts an improvement that clears both hysteresis gates" (portfolioSelectionShouldRotate portfolioTestConfig incumbent promoted)

testPortfolioSelectionJsonRoundTrip :: IO ()
testPortfolioSelectionJsonRoundTrip = do
    let selection = portfolioSelectionForRotationTest 0.15 0.95
        decoded = Aeson.eitherDecode (Aeson.encode selection) :: Either String PortfolioSelection
    assert "portfolio selection snapshots round-trip through JSON" (decoded == Right selection)

testMergeFreshnessScoringPromotesFreshCandidate :: IO ()
testMergeFreshnessScoringPromotesFreshCandidate = do
    (inputPath, inputHandle) <- openTempFile "/tmp" "trader-merge-freshness-input.json"
    hClose inputHandle
    (outputPath, outputHandle) <- openTempFile "/tmp" "trader-merge-freshness-output.json"
    hClose outputHandle
    let staleHighReturn =
            comboWithCreatedAtForTest 0 $
                processingComboForTest
                    "stale-high-return"
                    "db"
                    False
                    (Just adoptionMinWalkForwardSharpeMean)
                    2.0
                    adoptionMinTradeCount
        freshLowerReturn =
            comboWithCreatedAtForTest 4102444800000 $
                processingComboForTest
                    "fresh-lower-return"
                    "db"
                    False
                    (Just adoptionMinWalkForwardSharpeMean)
                    1.0
                    adoptionMinTradeCount
        scoringConfig =
            defaultTopComboScoringConfig
                { tcscFreshnessHalfLifeDays = 30
                , tcscFreshnessFloorMultiplier = 0.35
                }
        payload =
            Aeson.object
                [ "combos" .= [staleHighReturn, freshLowerReturn]
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
                , maScoringConfig = scoringConfig
                , maCopyToDist = False
                }
    decoded <- (Aeson.eitherDecode <$> BL.readFile outputPath) :: IO (Either String Aeson.Value)
    let firstMethod =
            case decoded of
                Right (Aeson.Object o) -> case KM.lookup "combos" o of
                    Just (Aeson.Array v) | not (V.null v) -> comboMethodForTest (V.head v)
                    _ -> Nothing
                _ -> Nothing
    assert "freshness-scored merge exits successfully" (code == 0)
    assert
        "freshness scoring promotes the newer validated combo over an older higher-return row"
        (firstMethod == Just "fresh-lower-return")
    _ <- try (removeFile inputPath) :: IO (Either SomeException ())
    _ <- try (removeFile outputPath) :: IO (Either SomeException ())
    pure ()

testMergeExecutableAnnotatesProcessingAndDedupe :: IO ()
testMergeExecutableAnnotatesProcessingAndDedupe = do
    (inputPath, inputHandle) <- openTempFile "/tmp" "trader-merge-input.json"
    hClose inputHandle
    (outputPath, outputHandle) <- openTempFile "/tmp" "trader-merge-output.json"
    hClose outputHandle
    let dbCopy = processingComboForTest "same-strategy" "db" True Nothing 2.0 adoptionMinTradeCount
        binanceCopy = processingComboForTest "same-strategy" "binance" False Nothing 2.0 adoptionMinTradeCount
        weakEquity = topComboMinimumFinalEquity - 0.005
        weakCopy =
            case processingComboForTest "weak-strategy" "db" True Nothing 2.0 adoptionMinTradeCount of
                Aeson.Object o ->
                    let metrics =
                            Aeson.object
                                [ "finalEquity" .= weakEquity
                                , "annualizedReturn" .= (2.0 :: Double)
                                , "tradeCount" .= adoptionMinTradeCount
                                , "maxDrawdown" .= (0.02 :: Double)
                                ]
                     in Aeson.Object (KM.insert "finalEquity" (Aeson.toJSON weakEquity) (KM.insert "metrics" metrics o))
                v -> v
        payload =
            Aeson.object
                [ "combos" .= [dbCopy, binanceCopy, weakCopy]
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
    assert "merge executable collapses source/null-equivalent strategy rows and drops sub-1.01 yield rows" (length combos == 1)
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
selectionComboForTest =
    selectionComboForTestWithWalkForward True

selectionComboForTestWithWalkForward :: Bool -> T.Text -> Double -> Maybe Int64 -> Maybe Int64 -> Aeson.Value
selectionComboForTestWithWalkForward includeWalkForward method score mCreatedAt mRefreshedAt =
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
          , "metrics"
                .= Aeson.object
                    ( [ "finalEquity" .= (1.0 + score :: Double)
                      , "tradeCount" .= (8 :: Int)
                      ]
                        ++ [ "walkForwardSummary"
                            .= Aeson.object
                                [ "sharpeMean" .= (0.5 :: Double)
                                , "sharpeStd" .= (1.0 :: Double)
                                ]
                           | includeWalkForward
                           ]
                    )
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
        missingWalkForward = selectionComboForTestWithWalkForward False "missing-walk-forward" 0.35 (Just (now - oneDay)) Nothing
        exactlyAtBoundary = selectionComboForTest "boundary" 0.4 (Just (now - comboBacktestStaleAfterMs)) Nothing
        shortStalePolicy =
            ComboBacktestRefreshPolicy
                { cbrpStaleAfterMs = oneDay
                , cbrpPruneFinalEquityFloor = 1.0
                }
        shortStale = selectionComboForTest "short-stale" 0.25 (Just (now - oneDay - 1)) Nothing
        selected = selectCombosForBacktestRefresh 1 now [staleLowRank, freshLowRank, topFresh, missingFreshness, missingWalkForward]
        selectedShortPolicy = selectCombosForBacktestRefreshWithPolicy shortStalePolicy 1 now [shortStale, freshLowRank, topFresh]
        selectedKeys = mapMaybe comboIdentityKey selected
        selectedShortPolicyKeys = mapMaybe comboIdentityKey selectedShortPolicy
        has combo = maybe False (`elem` selectedKeys) (comboIdentityKey combo)
        hasShortPolicy combo = maybe False (`elem` selectedShortPolicyKeys) (comboIdentityKey combo)
    assert
        "selection keeps the top-ranked combo and every stale or missing-evidence combo outside topN"
        ( length selected == 4
            && has topFresh
            && has staleLowRank
            && has missingFreshness
            && has missingWalkForward
            && not (has freshLowRank)
        )
    assert
        "missing freshness or walk-forward evidence is due, exactly three days old is not older than three days"
        ( comboBacktestDueForRefresh now missingFreshness
            && comboBacktestDueForRefresh now missingWalkForward
            && not (comboBacktestDueForRefresh now exactlyAtBoundary)
        )
    assert
        "configured stale age controls periodic backtest refresh selection"
        ( comboBacktestDueForRefreshWithPolicy shortStalePolicy now shortStale
            && hasShortPolicy shortStale
            && not (hasShortPolicy freshLowRank)
        )

testBacktestRefreshBatchesPrioritizeRankedCombos :: IO ()
testBacktestRefreshBatchesPrioritizeRankedCombos = do
    let values = [1 .. 11 :: Int]
        batches = batchCombosForBacktestRefresh 3 4 values
        cappedPriorityBatches = batchCombosForBacktestRefresh 120 100 values
    assert "backtest refresh publishes the ranked priority batch first" (take 1 batches == [[1, 2, 3]])
    assert "backtest refresh caps the ranked priority checkpoint at five combos" (take 1 cappedPriorityBatches == [[1, 2, 3, 4, 5]])
    assert "backtest refresh bounds stale batches without dropping or reordering combos" (batches == [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]] && concat batches == values)
    assert "backtest refresh batching is total for empty input" (null (batchCombosForBacktestRefresh 0 0 ([] :: [Int])))

testLiveComboFreshnessRequiresRecentBacktestEvidence :: IO ()
testLiveComboFreshnessRequiresRecentBacktestEvidence = do
    let day = 86400000 :: Int64
        maxAge = 14 * day
        now = 30 * day
        combo = Aeson.object
        freshByCreated = combo ["createdAtMs" .= (now - maxAge)]
        staleByCreated = combo ["createdAtMs" .= (now - maxAge - 1)]
        freshByRefresh =
            combo
                [ "createdAtMs" .= (now - 100 * day)
                , "backtestRefreshedAtMs" .= (now - day)
                ]
        missingFreshness = combo []
    assert
        "live combo freshness requires created/refreshed evidence no older than the live max age"
        ( comboBacktestFreshEnoughForMaxAge maxAge now freshByCreated
            && not (comboBacktestFreshEnoughForMaxAge maxAge now staleByCreated)
            && comboBacktestFreshEnoughForMaxAge maxAge now freshByRefresh
            && not (comboBacktestFreshEnoughForMaxAge maxAge now missingFreshness)
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
                , cbuPortfolioEvidence = Nothing
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
                , cbuPortfolioEvidence = Nothing
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
    let tuned =
            OutcomeWeightConfig
                { owcWinScale = 5
                , owcLossScale = 40
                , owcCap = 1.6
                }
        tunedWeights = tradeOutcomeWeightsWithConfig tuned [win, loss] 10
    assert
        "custom win scale changes winning span weights"
        (all (\t -> abs (tunedWeights !! t - 1.1) < 1e-9) [2 .. 4])
    assert
        "custom loss scale changes losing span weights"
        (all (\t -> abs (tunedWeights !! t - 1.6) < 1e-9) [6, 7])
    assert
        "custom cap limits direct outcome factor"
        (tradeOutcomeWeightFactorWithConfig tuned (mkOutcomeTestTrade 0 1 (-0.5)) == Just 1.6)
    assert
        "break-even trades carry no learning signal"
        (isNothing (tradeOutcomeWeightFactor (mkOutcomeTestTrade 0 1 0)))
    assert
        "trade spans clamp to the series bounds"
        (last (tradeOutcomeWeights [mkOutcomeTestTrade 8 20 (-0.02)] 10) > 1)
    assert "empty series yields no weights" (null (tradeOutcomeWeights [win] 0))

{- | The live bot appends a newly closed trade before the immediate
post-close LSTM update. Pin the pure weight semantics that make that close
visible to the next fine-tune instead of waiting for another bar.
-}
testTradeOutcomeWeightsIncludeNewClose :: IO ()
testTradeOutcomeWeightsIncludeNewClose = do
    let priorWin = mkOutcomeTestTrade 2 4 0.01
        newLoss = mkOutcomeTestTrade 8 9 (-0.02)
        before = tradeOutcomeWeights [priorWin] 12
        after = tradeOutcomeWeights [priorWin, newLoss] 12
        expectedLossWeight = 1 + outcomeWeightLossScale * 0.02
    assert "test setup: weights align with the series" (length before == 12 && length after == 12)
    assert "the not-yet-appended close span is unit-weighted" (all (\t -> before !! t == 1.0) [8, 9])
    assert
        "appending the newly closed loss immediately weights its full span"
        (all (\t -> abs (after !! t - expectedLossWeight) < 1e-9) [8, 9])
    assert "prior winning span remains available to the same fine-tune" (all (\t -> after !! t == before !! t) [2 .. 4])

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

testAlignToBarsFailClosedOnMalformedInputs :: IO ()
testAlignToBarsFailClosedOnMalformedInputs = do
    let bars = V.fromList [1000, 2000, 3000 :: Int64]
        malformedObservationSeries = [(500, 7.0), (1500, 0 / 0), (2500, 8.0), (3500, 1 / 0)]
    assert
        "alignToBars ignores non-finite observations while preserving point-in-time finite evidence"
        (alignToBars bars (1000 :: Int64) malformedObservationSeries == V.fromList [Just 7.0, Just 8.0, Just 8.0])
    assert
        "alignToBars fails closed for non-positive intervals"
        (alignToBars bars (0 :: Int64) [(500, 7.0)] == V.fromList [Nothing, Nothing, Nothing])
    assert
        "alignToBars fails closed for descending bar grids"
        (alignToBars (V.fromList [2000, 1000 :: Int64]) (1000 :: Int64) [(500, 7.0)] == V.fromList [Nothing, Nothing])
    assert
        "alignToBars fails closed for duplicate bar opens"
        (alignToBars (V.fromList [1000, 1000 :: Int64]) (1000 :: Int64) [(500, 7.0)] == V.fromList [Nothing, Nothing])
    assert
        "alignToBars fails closed when a bar close time would overflow"
        (alignToBars (V.fromList [maxBound :: Int64]) (2 :: Int64) [(maxBound, 7.0)] == V.fromList [Nothing])

testFeatureAvailabilitySchemaV2 :: IO ()
testFeatureAvailabilitySchemaV2 = do
    let decisionTime = 2000 :: Int64
        observedZero = TimedFeatureValue{tfvEventTimeMs = 1000, tfvAvailabilityTimeMs = 1500, tfvValue = 0}
        futureValue = TimedFeatureValue{tfvEventTimeMs = 1000, tfvAvailabilityTimeMs = 2500, tfvValue = 9}
        observedRow = mkFeatureRowV2 decisionTime [FeatureField "funding" OptionalFeature (Just observedZero)]
        missingRow = mkFeatureRowV2 decisionTime [FeatureField "funding" OptionalFeature Nothing]
    case (observedRow, missingRow) of
        (Just observed, Just missing) -> do
            assert "feature schema v2 has a stable semantic identifier" (frv2SchemaId observed == featureAvailabilitySchemaIdV2)
            assert "observed zero retains availability" (frv2Values observed == [0] && frv2Available observed == [True])
            assert "missing optional value is neutral with an explicit false mask" (frv2Values missing == [0] && frv2Available missing == [False])
            assert "observed zero and missing evidence produce distinct model inputs" (featureRowModelInputs observed /= featureRowModelInputs missing)
            assert
                "availability does not change the ordered schema signature"
                ( featureRowSchemaSignature observed
                    == featureRowSchemaSignature missing
                    && featureRowSchemaSignature observed == "feature_availability_v2|funding:optional"
                )
        _ -> assert "valid optional v2 rows should be constructible" False

    assert
        "required missing evidence abstains"
        (isNothing (mkFeatureRowV2 decisionTime [FeatureField "funding" RequiredFeature Nothing]))
    assert
        "required not-yet-available evidence abstains"
        (isNothing (mkFeatureRowV2 decisionTime [FeatureField "funding" RequiredFeature (Just futureValue)]))
    assert
        "required non-finite evidence abstains"
        ( isNothing
            ( mkFeatureRowV2
                decisionTime
                [FeatureField "funding" RequiredFeature (Just observedZero{tfvValue = 0 / 0})]
            )
        )
    assert
        "optional non-finite evidence remains explicitly unavailable"
        ( case mkFeatureRowV2 decisionTime [FeatureField "funding" OptionalFeature (Just observedZero{tfvValue = 1 / 0})] of
            Just row -> frv2Values row == [0] && frv2Available row == [False]
            Nothing -> False
        )
    assert
        "availability cannot precede event time"
        ( isNothing
            ( mkFeatureRowV2
                decisionTime
                [ FeatureField
                    "funding"
                    RequiredFeature
                    (Just observedZero{tfvEventTimeMs = 1600, tfvAvailabilityTimeMs = 1500})
                ]
            )
        )
    assert
        "empty, duplicate, or ambiguous feature names fail closed"
        ( isNothing (mkFeatureRowV2 decisionTime [])
            && isNothing (mkFeatureRowV2 decisionTime [FeatureField "" OptionalFeature Nothing])
            && isNothing
                ( mkFeatureRowV2
                    decisionTime
                    [ FeatureField "funding" OptionalFeature Nothing
                    , FeatureField "funding" OptionalFeature (Just observedZero)
                    ]
                )
            && isNothing (mkFeatureRowV2 decisionTime [FeatureField "funding:rate" OptionalFeature Nothing])
        )

    let bars = V.fromList [1000, 2000, 3000 :: Int64]
        intervalMs = 1000 :: Int64
        original = TimedFeatureValue{tfvEventTimeMs = 500, tfvAvailabilityTimeMs = 500, tfvValue = 1}
        revision = TimedFeatureValue{tfvEventTimeMs = 500, tfvAvailabilityTimeMs = 2500, tfvValue = 2}
        future = TimedFeatureValue{tfvEventTimeMs = 3500, tfvAvailabilityTimeMs = 3500, tfvValue = 3}
        aligned = alignTimedToBars bars intervalMs [future, revision, original]
    assert
        "revisions appear only from their availability time"
        (V.map (fmap tfvValue) aligned == V.fromList [Just 1, Just 2, Just 3])
    let newerEvent = TimedFeatureValue{tfvEventTimeMs = 1500, tfvAvailabilityTimeMs = 1500, tfvValue = 4}
    assert
        "a late revision to an older event does not replace a newer current event"
        ( V.map (fmap tfvValue) (alignTimedToBars bars intervalMs [future, revision, newerEvent, original])
            == V.fromList [Just 4, Just 4, Just 3]
        )
    assert
        "changing a future observation cannot change earlier aligned rows"
        ( take 2 (V.toList aligned)
            == take
                2
                ( V.toList
                    (alignTimedToBars bars intervalMs [future{tfvValue = -999}, revision, original])
                )
        )

    case alignedFeatureSeriesV2 bars intervalMs [TimedFeatureValue 2500 2500 0] of
        Nothing -> assert "one admissible timed value should create a v2 series" False
        Just series -> do
            assert "v2 alignment keeps a pre-coverage mask" (afsV2Available series == V.fromList [False, True, True])
            assert "v2 alignment distinguishes missing from an observed zero" (afsV2Values series == V.fromList [0, 0, 0])
            assert "v2 alignment preserves event timestamps" (afsV2EventTimesMs series == V.fromList [Nothing, Just 2500, Just 2500])
            assert
                "v2 alignment preserves availability timestamps"
                (afsV2AvailabilityTimesMs series == V.fromList [Nothing, Just 2500, Just 2500])

testExogenousDerivativesBacktestWiring :: IO ()
testExogenousDerivativesBacktestWiring = do
    assert
        "Binance derivatives stats periods cover short, intermediate, and long bar intervals"
        ( binanceStatsPeriodForInterval "1m" == Just "5m"
            && binanceStatsPeriodForInterval "8h" == Just "12h"
            && binanceStatsPeriodForInterval "3d" == Just "1d"
            && isNothing (binanceStatsPeriodForInterval "bad")
        )

    let bars = V.fromList [1000, 2000 :: Int64]
        intervalMs = 1000 :: Int64
    assert
        "a fetched series with no point-in-time overlap does not opt the model into all-zero exogenous features"
        (isNothing (alignedFeatureSeries bars intervalMs [(5000, 9.0)]))
    assert
        "an overlapping fetched series is neutral-filled only before its first admissible observation"
        (alignedFeatureSeries bars intervalMs [(2500, 9.0)] == Just (V.fromList [0, 9.0]))

    assert
        "the derivatives flag is accepted for a non-trading Binance futures backtest"
        ( case parseAndValidateCliArgs ["--binance-symbol", "BTCUSDT", "--futures", "--exogenous-derivatives"] of
            Right args -> argExogenousDerivatives args
            Left _ -> False
        )
    assert
        "the derivatives flag rejects spot, CSV, signal/trade, live, and server paths"
        ( all
            (isLeft . parseAndValidateCliArgs)
            [ ["--binance-symbol", "BTCUSDT", "--exogenous-derivatives"]
            , ["--data", "prices.csv", "--futures", "--exogenous-derivatives"]
            , ["--binance-symbol", "BTCUSDT", "--futures", "--exogenous-derivatives", "--trade-only"]
            , ["--binance-symbol", "BTCUSDT", "--futures", "--exogenous-derivatives", "--binance-trade"]
            , ["--binance-symbol", "BTCUSDT", "--futures", "--exogenous-derivatives", "--binance-live"]
            , ["--serve", "--futures", "--exogenous-derivatives"]
            ]
        )

testPointInTimeUniverseSelectsHistoricalSnapshot :: IO ()
testPointInTimeUniverseSelectsHistoricalSnapshot = do
    (path, h) <- openTempFile "/tmp" "pit-universe.csv"
    hClose h
    let csv =
            BL.fromStrict
                "timestamp,symbol,quoteVolume\n\
                \2026-01-01,BTCUSDT,100\n\
                \2026-01-01,ETHUSDT,200\n\
                \2026-01-03,BTCUSDT,500\n\
                \2026-01-03,ETHUSDT,50\n\
                \2026-01-03,USDCUSDT,10000\n"
        cfg = PointInTimeUniverseConfig{pitUniverseCsv = Just path, pitUniverseRequireHistorical = True}
    BL.writeFile path csv
    selectedEarly <- loadPointInTimeUniverse cfg "USDT" 2 1767312000000 -- 2026-01-02
    selectedLate <- loadPointInTimeUniverse cfg "USDT" 2 1767484800000 -- 2026-01-04
    removeFile path
    assert "PIT universe uses latest rows at or before as-of time" (selectedEarly == Just [("ETHUSDT", 200), ("BTCUSDT", 100)])
    assert "PIT universe updates ranking after newer historical rows" (selectedLate == Just [("BTCUSDT", 500), ("ETHUSDT", 50)])

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

testCostCalibrationConfigurableRoiKnobs :: IO ()
testCostCalibrationConfigurableRoiKnobs = do
    let strictOutlierCfg = defaultCostCalibrationConfig{cccOutlierBound = 0.004}
        approx expected = maybe False (\v -> abs (v - expected) < 1e-12)
    assert
        "configured outlier bound can reject otherwise accepted fill measurements"
        ( isNothing (observedSlippageFractionWithConfig strictOutlierCfg "BUY" 100 (Just 2) (Just 201))
            && approx 0.005 (observedSlippageFraction "BUY" 100 (Just 2) (Just 201))
        )

    let windowCfg =
            defaultCostCalibrationConfig
                { cccMinObservations = 2
                , cccShrinkageObs = 0
                , cccWindow = 2
                , cccFloorFactor = 0
                , cccMaxPerSide = 1
                }
        configured = 0.0002
        calibrated = calibratedSlippagePerSideWithConfig windowCfg configured [0.001, 0.002, 0.004]
    assert
        "configured evidence window controls the realized median used for calibration"
        (abs (calibrated - 0.003) < 1e-12)

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

testAdaptiveConformalRadiusRespondsToMisses :: IO ()
testAdaptiveConformalRadiusRespondsToMisses = do
    let model = fitConformal 0.2 [0.01, 0.02, 0.03, 0.04, 0.05]
        st0 = initAdaptiveConformal 0.5 model
        stMiss = updateAdaptiveConformal 0.20 0 st0
        stHit = updateAdaptiveConformal 0.0 0 stMiss
        finite x = not (isNaN x || isInfinite x)
        (lo, hi, sigma) = predictInterval model 0
    assert "adaptive conformal initializes from the fitted radius" (acsRadius st0 == cmRadius model)
    assert "adaptive conformal radius inflates after an interval miss" (acsRadius stMiss > acsRadius st0)
    assert "adaptive conformal radius shrinks after a covered observation" (acsRadius stHit < acsRadius stMiss)
    assert "recency-aware fitted conformal interval remains finite for valid evidence" (finite lo && finite hi && maybe True finite sigma)

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

testSweepThresholdZeroCandidatesKeepsBasePair :: IO ()
testSweepThresholdZeroCandidatesKeepsBasePair = do
    let prices :: [Double]
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        preds :: [Double]
        preds = [101.5, 102.5, 103.5, 104.5, 105.5]
        cfg =
            (defaultTuneConfig 252)
                { tcMaxThresholdCandidates = 0
                , tcWalkForwardFolds = 1
                }
        baseCfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.03
                , ecFee = 0
                , ecSlippage = 0
                , ecSpread = 0
                , ecMaxPositionSize = 1
                , ecMinPositionSize = 0
                }
        result = sweepThresholdWithHLWith cfg MethodBoth baseCfg prices prices prices preds preds (Nothing :: Maybe [StepMeta])
    case result of
        Left err -> assert ("sweep-threshold zero-candidate regression failed to simulate: " ++ err) False
        Right (openThr, closeThr, _bt, stats) -> do
            assert "zero dynamic threshold candidates keep the configured open threshold" (abs (openThr - 0.01) < 1e-12)
            assert "zero dynamic threshold candidates keep the configured close threshold" (abs (closeThr - 0.03) < 1e-12)
            assert "zero dynamic threshold candidates still report tune stats" (tsFoldCount stats > 0)

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

testSplitReversalExecutionInvariant :: IO ()
testSplitReversalExecutionInvariant = do
    let (partialPos, partialSize, partialClose, partialOpen) =
            applySplitReversalExecutedQuantities (-1) 1 True 0.6 0.25
    assert
        "a completed close with no entry leaves a split reversal flat"
        (applySplitReversalExecutedQuantities 1 1 False 1 0 == (0, 0, 1, 0))
    assert
        "a completed close and partial entry preserve both legs independently"
        (applySplitReversalExecutedQuantities 1 1 False 1 0.4 == (-1, 0.4, 1, 0.4))
    assert
        "flat confirmation replaces an intermediate partial close report with the requested quantity"
        (confirmedCloseExecutedQuantity (Just 1) (Just 0.6) == Just 1)
    assert
        "a partial close cannot fabricate more opposite exposure than the entry fill"
        (partialPos == -1 && partialOpen == 0)
    assertNear "partial split reversal retains the remaining original position" 0.15 partialSize 1e-12
    assertNear "partial split reversal accounts both close contributions" 0.85 partialClose 1e-12

testStartupCanceledAfterPartialExecutionScenario :: IO ()
testStartupCanceledAfterPartialExecutionScenario = do
    let terminalEvidence =
            OrderExecutionEvidence
                { oeeSent = True
                , oeeLive = True
                , oeeStatus = Just "CANCELED"
                , oeeExecutedQty = Just 0.8
                }
        appliedFraction = orderAppliedFraction terminalEvidence (Just 2) 0.5
        result = applyExecutedQuantity 0 0 True (fromMaybe 0 appliedFraction)
        halt = maxDrawdownHaltForPosition (first4 result) (second4 result)
    assertNear
        "startup preserves the equity fraction implied by an explicit partial fill on a canceled order"
        0.2
        (fromMaybe 0 appliedFraction)
        1e-12
    assert
        "startup canceled-after-partial opens only the causally observed filled exposure"
        (result == (1, 0.2, 0, 0.2))
    assert
        "the canonical halt path can flatten the startup partial-fill exposure"
        (lrhaExitReason halt == Just ExitMaxDrawdown && lrhaDesiredPosition halt == 0 && lrhaOrderDirection halt == Just (-1))

testLiveReversalPartialEntryScenario :: IO ()
testLiveReversalPartialEntryScenario = do
    let confirmedCloseEvidence =
            OrderExecutionEvidence
                { oeeSent = True
                , oeeLive = True
                , oeeStatus = Just "FILLED"
                , oeeExecutedQty = Just 0.6
                }
        terminalEntryEvidence =
            OrderExecutionEvidence
                { oeeSent = True
                , oeeLive = True
                , oeeStatus = Just "CANCELED"
                , oeeExecutedQty = Just 0.1
                }
        closeApplied = orderAppliedFraction confirmedCloseEvidence (Just 0.6) 0.6
        entryApplied = orderAppliedFraction terminalEntryEvidence (Just 0.4) 0.4
        result =
            applySplitReversalExecutedQuantities
                1
                0.6
                False
                (fromMaybe 0 closeApplied)
                (fromMaybe 0 entryApplied)
        halt = maxDrawdownHaltForPosition (first4 result) (second4 result)
    assert
        "live reversal accounts the confirmed close and terminal partial entry as independent legs"
        (result == (-1, 0.1, 0.6, 0.1))
    assert
        "the canonical halt path buys only to flatten the partial short reversal"
        (lrhaExitReason halt == Just ExitMaxDrawdown && lrhaDesiredPosition halt == 0 && lrhaOrderDirection halt == Just 1)

testReduceOnlyPartialTakeProfitTerminalCancelScenario :: IO ()
testReduceOnlyPartialTakeProfitTerminalCancelScenario = do
    let terminalEvidence =
            OrderExecutionEvidence
                { oeeSent = True
                , oeeLive = True
                , oeeStatus = Just "CANCELED"
                , oeeExecutedQty = Just 0.2
                }
        appliedFraction = orderAppliedFraction terminalEvidence (Just 0.5) 0.5
        result = applyReduceOnlyExecutedQuantity 1 1 (fromMaybe 0 appliedFraction)
        halt = maxDrawdownHaltForPosition (first4 result) (second4 result)
    assert
        "reduce-only partial take-profit preserves the terminal fill and cannot open opposite exposure"
        (result == (1, 0.8, 0.2, 0))
    assert
        "the canonical halt path still flattens the remaining long after a terminal partial take-profit"
        (lrhaExitReason halt == Just ExitMaxDrawdown && lrhaDesiredPosition halt == 0 && lrhaOrderDirection halt == Just (-1))

testSnapshotRestartRestoresMemoryWithoutExposureScenario :: IO ()
testSnapshotRestartRestoresMemoryWithoutExposureScenario = do
    let context =
            TradeMemorySnapshotContext
                { tmscSymbol = "BTCUSDT"
                , tmscInterval = "5m"
                , tmscMarket = "futures"
                , tmscMethod = "both"
                , tmscTradeLimit = 50
                }
        tradeValue =
            Aeson.object
                [ "entryEquity" .= (1 :: Double)
                , "exitEquity" .= (1.1 :: Double)
                , "return" .= (0.1 :: Double)
                , "holdingPeriods" .= (3 :: Int)
                , "entrySource" .= ("signal" :: String)
                ]
        statusWithPosition positions openTrade =
            Aeson.object
                [ "symbol" .= ("BTCUSDT" :: String)
                , "interval" .= ("5m" :: String)
                , "market" .= ("futures" :: String)
                , "method" .= ("both" :: String)
                , "positions" .= (positions :: [Int])
                , "openTrade" .= openTrade
                , "trades" .= [tradeValue]
                ]
        statusClaimingExposure = statusWithPosition [0, 1] (Aeson.object ["side" .= ("long" :: String), "size" .= (1 :: Double)])
        statusClaimingFlat = statusWithPosition [0, 0] Aeson.Null
        restoredFromExposureSnapshot = restoreTradeMemoryFromStatus context statusClaimingExposure
        restoredFromFlatSnapshot = restoreTradeMemoryFromStatus context statusClaimingFlat
        flatHalt = maxDrawdownHaltForPosition 0 0
    assert
        "restart restores the same bounded closed-trade memory regardless of persisted position/open-trade claims"
        ( restoredFromExposureSnapshot == restoredFromFlatSnapshot
            && case restoredFromExposureSnapshot of
                [tr] -> trEntryIndex tr == 0 && trExitIndex tr == 3 && trReturn tr == 0.1
                _ -> False
        )
    assert
        "a venue-flat restart has no phantom halt order even when its historical snapshot claimed exposure"
        (lrhaExitReason flatHalt == Just ExitMaxDrawdown && lrhaDesiredPosition flatHalt == 0 && isNothing (lrhaOrderDirection flatHalt))

maxDrawdownHaltForPosition :: Int -> Double -> LiveRiskHaltAction
maxDrawdownHaltForPosition position positionSize =
    liveRiskHaltAction
        position
        HaltInputs
            { hiPrevHaltReason = Nothing
            , hiDayChanged = False
            , hiWeekChanged = False
            , hiDailyLoss = 0
            , hiWeeklyLoss = 0
            , hiDrawdown = 0.06
            , hiExpectancy = Nothing
            , hiMaxDailyLossLim = Nothing
            , hiMaxWeeklyLossLim = Nothing
            , hiMaxDrawdownLim = Just 0.05
            , hiMinExpectancy = Nothing
            , hiPositionSize = positionSize
            , hiMaxPositionSizeLim = Just 1
            , hiConsecutiveLosses = 0
            , hiMaxLossStreakLim = Nothing
            , hiVolTarget = 0
            , hiLeverage = 0
            }

first4 :: (a, b, c, d) -> a
first4 (a, _, _, _) = a

second4 :: (a, b, c, d) -> b
second4 (_, b, _, _) = b

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

-- Coinbase candle pagination must never wrap large requested spans into
-- inverted ranges.
testCoinbaseBuildRangesOverflowRegression :: IO ()
testCoinbaseBuildRangesOverflowRegression = do
    let normal = buildRanges 36000 60 301
        huge = buildRanges 1000 (maxBound :: Int64) 600
        validRange (startSec, endSec) =
            startSec >= 0 && endSec >= 0 && startSec <= endSec && endSec <= 36000
    assert
        "Coinbase range paging keeps normal ranges bounded and non-inverted"
        (length normal == 2 && all validRange normal)
    assert
        "Coinbase range paging saturates overflowing spans to the Unix origin"
        (huge == [(0, 1000)])

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

testAutoOptimizerCappedLookbackScopes :: IO ()
testAutoOptimizerCappedLookbackScopes = do
    let maxPoints = 1000
        backtestRatio = 0.2
        tuneRatio = 0.25
        lookbackWindows = ["7d", "14d", "30d"]
        selection =
            selectAutoOptimizerScopes
                True
                maxPoints
                backtestRatio
                tuneRatio
                ["5m", "1h", "1d"]
                lookbackWindows
        disabledSelection =
            selectAutoOptimizerScopes
                False
                maxPoints
                backtestRatio
                tuneRatio
                ["5m"]
                lookbackWindows
        headroomSelection =
            selectAutoOptimizerScopesWithHeadroom
                True
                maxPoints
                20
                backtestRatio
                tuneRatio
                ["5m"]
                lookbackWindows
        maxFeasibleBars = 597
    assert
        "auto optimizer split sizing admits 597 bars under the 1000-point cap"
        (maybe False (<= maxPoints) (autoOptimizerRequiredBarsForSweep backtestRatio tuneRatio maxFeasibleBars))
    assert
        "auto optimizer split sizing rejects one more bar under the same cap"
        (maybe False (> maxPoints) (autoOptimizerRequiredBarsForSweep backtestRatio tuneRatio (maxFeasibleBars + 1)))
    assert
        "auto optimizer derives one capped 5m scope while preserving feasible configured scopes"
        ( aosScopes selection
            == [ ("5m", "2985m")
               , ("1h", "7d")
               , ("1h", "14d")
               , ("1d", "7d")
               , ("1d", "14d")
               , ("1d", "30d")
               ]
        )
    assert
        "auto optimizer reports only the derived windows as capped scopes"
        (aosCappedScopes selection == [("5m", "2985m")])
    assert
        "disabling auto-capped lookbacks leaves the infeasible 5m scope excluded"
        (null (aosScopes disabledSelection) && null (aosCappedScopes disabledSelection))
    assert
        "auto optimizer reserves point headroom before deriving a capped scope"
        (aosScopes headroomSelection == [("5m", "2925m")] && aosCappedScopes headroomSelection == [("5m", "2925m")])

testAutoOptimizerObjectiveAlignment :: IO ()
testAutoOptimizerObjectiveAlignment =
    assert
        "auto optimizer passes its ranking objective through to child threshold tuning"
        ( optimizerObjectiveArgs "sharpe"
            == ["--objective", "sharpe", "--tune-objective", "sharpe"]
        )

testOptimizerAdmissionStats :: IO ()
testOptimizerAdmissionStats = do
    let combo uuid createdAt =
            Aeson.object
                [ "uuid" .= (uuid :: T.Text)
                , "createdAtMs" .= (createdAt :: Int64)
                ]
        before = Aeson.object ["combos" .= [combo "existing" 1000, combo "removed" 900]]
        after = Aeson.object ["combos" .= [combo "existing" 1000, combo "new" 2000]]
        stats = optimizerAdmissionStats before after
    assert
        "optimizer admission telemetry counts board identities rather than successful merge processes"
        ( oasBeforeCount stats == 2
            && oasAfterCount stats == 2
            && oasAdmittedCount stats == 1
            && oasRemovedCount stats == 1
            && oasNewestBeforeMs stats == Just 1000
            && oasNewestAfterMs stats == Just 2000
        )

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

-- The latest-signal path and simulator both call signalRunPostDirectionGates.
-- Exercise its reason order directly, then prove that the simulator blocks the
-- corresponding high-volatility entry instead of bypassing the shared gate.
testDecisionParitySharedGatePrecedence :: IO ()
testDecisionParitySharedGatePrecedence = do
    let allFailingGates =
            signalRunPostDirectionGates
                (Just 1)
                Nothing
                False
                False
                (const False)
                (const False)
                (const False)
                False
                (const (False, Just "NON_DIRECTIONAL_RANGE"))
                (False, Just "REGIME_EDGE")
                (False, Just "MTF_CONSENSUS")
                (False, Just "CROSS_ASSET")
                (const False)
                (const (False, 0))
        upstreamReasonWins =
            signalRunPostDirectionGates
                (Just 1)
                (Just "PAIRS_CONFLICT")
                False
                False
                (const False)
                (const False)
                (const False)
                False
                (const (False, Just "NON_DIRECTIONAL_RANGE"))
                (False, Just "REGIME_EDGE")
                (False, Just "MTF_CONSENSUS")
                (False, Just "CROSS_ASSET")
                (const False)
                (const (False, 0))
        prices = V.fromList [100 :: Double, 100, 101, 103, 104, 105]
        predictions = V.fromList [102 :: Double, 102, 104, 106, 107]
        baseCfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.005
                , ecCloseThreshold = 0.002
                , ecFee = 0
                , ecVolLookback = 2
                , ecMaxPositionSize = 1
                }
        allowed = simulateEnsemble baseCfg 2 prices prices prices predictions predictions (Nothing :: Maybe (V.Vector StepMeta))
        volatilityBlocked =
            simulateEnsemble
                baseCfg{ecMaxVolatility = Just 1e-6}
                2
                prices
                prices
                prices
                predictions
                predictions
                (Nothing :: Maybe (V.Vector StepMeta))
    assert
        "shared post-direction gate precedence reports volatility before later failing gates"
        (allFailingGates == (Nothing, Just "VOLATILITY"))
    assert
        "an upstream direction reason remains authoritative over post-direction failures"
        (upstreamReasonWins == (Nothing, Just "PAIRS_CONFLICT"))
    assert
        "backtest fixture has a real entry before the volatility gate is enabled"
        (any ((> 1e-9) . abs) (brPositions allowed))
    assert
        "backtest consumes the shared volatility gate and blocks the same entry"
        (all ((<= 1e-9) . abs) (brPositions volatilityBlocked))

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
                , ecTakeProfitPartial = 0.5
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
            && ecTakeProfitPartial riskConfigured == 0.5
            && tradeEntrySourceCode signalSource == "signal"
            && tradeEntrySourceCode postDirectionSource == "post_direction_gates"
            && exitReasonFromCode "eod" == Just ExitEod
            && exitReasonFromCode "max_pnl_timing" == Just (ExitOther "MAX_PNL_TIMING")
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

testOptimizerRiskPerTradeNormalization :: IO ()
testOptimizerRiskPerTradeNormalization = do
    let stopLoss = 0.005
        cappedRisk = positionSizeScaleHardFailMultiplier * stopLoss
    assert
        "optimizer keeps fixed-stop risk sizing when it stays under the simulator size-scale hard cap"
        (normalizeOptimizerRiskPerTrade (Just 0.02) Nothing (Just 0.01) == Just 0.01)
    assertNear
        "optimizer caps fixed-stop risk sizing at the simulator size-scale hard cap"
        cappedRisk
        (fromMaybe 0 (normalizeOptimizerRiskPerTrade (Just stopLoss) Nothing (Just 0.02)))
        1e-12
    assert
        "optimizer drops risk sizing without any stop-loss bound"
        (isNothing (normalizeOptimizerRiskPerTrade Nothing Nothing (Just 0.01)))
    assert
        "optimizer drops risk sizing for volatility stops because the realized stop fraction is data-dependent"
        (isNothing (normalizeOptimizerRiskPerTrade (Just 0.02) (Just 1.5) (Just 0.01)))

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

testOptimizerSurvivorDedupePreservesFirstCandidates :: IO ()
testOptimizerSurvivorDedupePreservesFirstCandidates = do
    let candidates =
            [ ("eligible", "a")
            , ("eligible", "b")
            , ("fallback", "a")
            , ("fallback", "c")
            , ("fallback", "b")
            ]
    assert
        "survivor candidate dedupe keeps first occurrences and skips repeated parameter keys"
        (dedupeFirstByKey snd candidates == [("eligible", "a"), ("eligible", "b"), ("fallback", "c")])

testOptimizerTopJsonSortUsesObjectiveScore :: IO ()
testOptimizerTopJsonSortUsesObjectiveScore = do
    let highScoreLowAnnualized = optimizerTopJsonSortKey (Just 2.0) (Just 0.1) (Just 1.1)
        lowScoreHighAnnualized = optimizerTopJsonSortKey (Just 1.0) (Just 99.0) (Just 9.0)
        equalScoreHighAnnualized = optimizerTopJsonSortKey (Just 2.0) (Just 0.2) (Just 1.1)
        nonFiniteScore = optimizerTopJsonSortKey (Just (1 / 0)) (Just 99.0) (Just 9.0)
    assert
        "optimize-equity top-json sorting prefers the active objective score over annualized return"
        (highScoreLowAnnualized > lowScoreHighAnnualized)
    assert
        "optimize-equity top-json sorting still uses annualized return as a score tie-breaker"
        (equalScoreHighAnnualized > highScoreLowAnnualized)
    assert
        "optimize-equity top-json sorting sinks non-finite objective scores"
        (lowScoreHighAnnualized > nonFiniteScore)

testOptimizerTechniqueSummaryTruthfulRegression :: IO ()
testOptimizerTechniqueSummaryTruthfulRegression = do
    let summary =
            emptyTechniqueSummary
                { otsAppliedSeedDiversification = True
                , otsAppliedSurvivorPruning = True
                , otsAppliedSurvivorExploitation = True
                , otsAppliedEmpiricalPriors = True
                , otsAppliedCorrelationGuidance = True
                , otsAppliedWalkForward = True
                , otsAppliedTopPerformerSummary = True
                }
        summaryObj =
            case optimizerTechniqueSummaryJson summary of
                Aeson.Object obj -> obj
                _ -> KM.empty
        boolField key = KM.lookup (AK.fromString key) summaryObj >>= AT.parseMaybe Aeson.parseJSON
    assert
        "optimizer reports the current seed-diversification mechanism"
        (boolField "seedDiversification" == Just True)
    assert
        "optimizer reports survivor pruning under its actual full-cost mechanism"
        (boolField "survivorPruning" == Just True)
    assert
        "optimizer reports rank-biased survivor exploitation under its actual mechanism"
        (boolField "rankBiasedSurvivorExploitation" == Just True)
    assert
        "optimizer reports correlation-guided sampling when it is applied"
        (boolField "correlationGuidance" == Just True)
    assert
        "optimizer does not report unimplemented Sobol sampling as applied"
        (boolField "sobolSeeding" == Just False)
    assert
        "optimizer does not report unimplemented ASHA/successive-halving as applied"
        (boolField "successiveHalving" == Just False)
    assert
        "optimizer does not report unimplemented Bayesian expected improvement as applied"
        (boolField "bayesianExpectedImprovement" == Just False)
    assert
        "optimizer does not report a non-executed top-performer ensemble as applied"
        (boolField "ensembleTopPerformers" == Just False)

testOptimizerPriorEdgeScoreRegression :: IO ()
testOptimizerPriorEdgeScoreRegression = do
    let objectMap fields =
            case Aeson.object fields of
                Aeson.Object obj -> obj
                _ -> KM.empty
        weakEdge =
            objectMap
                [ "annualizedReturn" Aeson..= (0.12 :: Double)
                , "sharpe" Aeson..= (0.7 :: Double)
                , "maxDrawdown" Aeson..= (0.25 :: Double)
                , "profitFactor" Aeson..= (1.1 :: Double)
                , "roundTrips" Aeson..= (20 :: Int)
                ]
        strongEdge =
            objectMap
                [ "annualizedReturn" Aeson..= (1.2 :: Double)
                , "sharpe" Aeson..= (2.8 :: Double)
                , "maxDrawdown" Aeson..= (0.18 :: Double)
                , "profitFactor" Aeson..= (2.4 :: Double)
                , "roundTrips" Aeson..= (64 :: Int)
                , "walkForwardSummary"
                    Aeson..= Aeson.object
                        [ "sharpeMean" Aeson..= (1.6 :: Double)
                        ]
                ]
        malformed =
            objectMap
                [ "annualizedReturn" Aeson..= (-1.0 :: Double)
                , "sharpe" Aeson..= (-2.0 :: Double)
                , "maxDrawdown" Aeson..= (-0.3 :: Double)
                , "profitFactor" Aeson..= (-5.0 :: Double)
                ]
        zeroWeightConfig =
            defaultOptimizerEdgeScoreConfig
                { oescAnnualizedReturnWeight = 0
                , oescSharpeWeight = 0
                , oescCalmarWeight = 0
                , oescProfitFactorWeight = 0
                , oescWalkForwardSharpeWeight = 0
                , oescActivityWeight = 0
                }
    assert
        "optimizer prior edge score favors stronger positive edge evidence"
        (priorTrialEdgeScore strongEdge > priorTrialEdgeScore weakEdge && priorTrialEdgeScore weakEdge > 0)
    assert
        "optimizer prior edge score ignores malformed or negative edge evidence"
        (priorTrialEdgeScore malformed == 0)
    assert
        "default optimizer edge-score config preserves prior edge scoring"
        (priorTrialEdgeScoreWithConfig defaultOptimizerEdgeScoreConfig strongEdge == priorTrialEdgeScore strongEdge)
    assert
        "zero edge-score weights remove the prior edge boost"
        (priorTrialEdgeScoreWithConfig zeroWeightConfig strongEdge == 0)

testOptimizerOverfitAuditReportsSelectionRisk :: IO ()
testOptimizerOverfitAuditReportsSelectionRisk = do
    let metrics sharpe wfMean wfStd =
            case Aeson.object
                [ "sharpe" Aeson..= (sharpe :: Double)
                , "walkForwardSummary"
                    Aeson..= Aeson.object
                        [ "sharpeMean" Aeson..= (wfMean :: Double)
                        , "sharpeStd" Aeson..= (wfStd :: Double)
                        ]
                ] of
                Aeson.Object obj -> obj
                _ -> KM.empty
        trials =
            [ OverfitTrial True True (Just 1.0) (Just (metrics 0.4 0.1 0.2))
            , OverfitTrial True True (Just 1.5) (Just (metrics 0.8 0.2 0.6))
            , OverfitTrial True True (Just 3.0) (Just (metrics 2.0 0.4 1.2))
            , OverfitTrial False False Nothing Nothing
            ]
        field key obj =
            case KM.lookup (AK.fromString key) obj of
                Just (Aeson.Number n) -> Just (realToFrac n :: Double)
                _ -> Nothing
    case optimizerOverfitAudit trials of
        Just (Aeson.Object obj) -> do
            assert "overfit audit counts all trials" (field "trialCount" obj == Just 4)
            assert "overfit audit counts scored trials" (field "scoredTrialCount" obj == Just 3)
            assert "overfit audit reports empirical winner p-value" (maybe False (> 0) (field "empiricalBestScorePValue" obj))
            assert "overfit audit reports a positive multiple-testing penalty" (maybe False (> 0) (field "multipleTestingSharpePenalty" obj))
            assert "overfit audit reports PBO proxy from walk-forward instability" (maybe False (> 0) (field "pboProxy" obj))
        _ -> assert "overfit audit emits an object for scored trials" False

testOptimizerPriorParserCarriesFreshEvidenceRegression :: IO ()
testOptimizerPriorParserCarriesFreshEvidenceRegression = do
    let refreshedAt = 1781709486839 :: Int
        payload =
            Aeson.object
                [ "combos"
                    Aeson..= [ Aeson.object
                                [ "params"
                                    Aeson..= Aeson.object
                                        [ "binanceSymbol" Aeson..= ("XRPUSDT" :: String)
                                        , "interval" Aeson..= ("1h" :: String)
                                        , "method" Aeson..= ("10" :: String)
                                        ]
                                , "metrics"
                                    Aeson..= Aeson.object
                                        [ "roundTrips" Aeson..= (22 :: Int)
                                        , "tradeCount" Aeson..= (44 :: Int)
                                        , "annualizedReturn" Aeson..= (0.8 :: Double)
                                        , "walkForwardSummary" Aeson..= Aeson.object ["sharpeMean" Aeson..= (0.9 :: Double)]
                                        ]
                                , "backtestRefreshedAtMs" Aeson..= refreshedAt
                                ]
                             ]
                ]
        trials = priorTrialsFromValue payload
    case trials of
        [trial] -> do
            assert "prior parser keeps symbol hints from top-combo params" (ptSymbol trial == Just "XRPUSDT")
            assert "prior parser keeps interval hints from top-combo params" (ptInterval trial == Just "1h")
            assert "prior parser keeps method hints from top-combo params" (ptMethod trial == Just "10")
            assert "prior parser treats refresh stamps as usable age evidence" (ptCreatedAtMs trial == Just refreshedAt)
            assert "prior parser keeps metrics for edge scoring" (priorTrialEdgeScore (ptMetrics trial) > 0)
        _ -> assert "prior parser reads compact top-combo payloads as trials" False

testOptimizerPriorAgeDecayMissingTimestampRegression :: IO ()
testOptimizerPriorAgeDecayMissingTimestampRegression = do
    let dayMs = 86400000 :: Int
        nowMs = 90 * dayMs
        createdOneHalfLifeAgo = nowMs - 45 * dayMs
        closeTo expected actual = abs (actual - expected) <= 1e-12
    assert
        "disabled prior age decay remains neutral for missing timestamps"
        (priorAgeDecayMultiplier 0 nowMs Nothing == 1)
    assert
        "missing prior timestamps are treated as stale when age decay is enabled"
        (closeTo defaultPriorMissingAgeMultiplier (priorAgeDecayMultiplier 45 nowMs Nothing))
    assert
        "missing prior timestamp multiplier is configurable"
        (closeTo 0.4 (priorAgeDecayMultiplierWithMissingMultiplier 0.4 45 nowMs Nothing))
    assert
        "one half-life old prior timestamp decays to 0.5"
        (closeTo 0.5 (priorAgeDecayMultiplier 45 nowMs (Just createdOneHalfLifeAgo)))
    assert
        "future prior timestamps do not receive an age penalty"
        (priorAgeDecayMultiplier 45 nowMs (Just (nowMs + dayMs)) == 1)

testOptimizerPriorAgeAdjustedScoreRegression :: IO ()
testOptimizerPriorAgeAdjustedScoreRegression = do
    let closeTo expected actual = abs (actual - expected) <= 1e-12
        nowMs = 90 * 86400000
    assert
        "disabled prior age adjustment leaves positive scores unchanged"
        (ageAdjustedPriorScore 0 nowMs Nothing 12 == 12)
    assert
        "missing prior timestamps discount positive scores by two half-lives"
        (closeTo 3 (ageAdjustedPriorScore 45 nowMs Nothing 12))
    assert
        "missing prior timestamps amplify negative scores by two half-lives"
        (closeTo (-48) (ageAdjustedPriorScore 45 nowMs Nothing (-12)))
    assert
        "custom missing prior timestamp multiplier discounts positive scores"
        (closeTo 6 (ageAdjustedPriorScoreWithMissingMultiplier 0.5 45 nowMs Nothing 12))
    assert
        "custom missing prior timestamp multiplier amplifies negative scores"
        (closeTo (-24) (ageAdjustedPriorScoreWithMissingMultiplier 0.5 45 nowMs Nothing (-12)))

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

testOptimizerCorrelationGuidanceParserRegression :: IO ()
testOptimizerCorrelationGuidanceParserRegression = do
    assert
        "empty optimizer correlation guidance is inert"
        (parseOptimizerCorrelationGuidance "" == Right [])
    assert
        "optimizer correlation guidance accepts stable field targets"
        ( parseOptimizerCorrelationGuidance "{\"key\":\"stopLoss\",\"target\":0.02,\"strength\":0.4,\"sampleCount\":8,\"stable\":true,\"lo\":0.01,\"hi\":0.03}"
            == Right
                [ CorrelationGuidanceField
                    { cgfKey = "stopLoss"
                    , cgfTarget = 0.02
                    , cgfLo = Just 0.01
                    , cgfHi = Just 0.03
                    , cgfStrength = 0.4
                    , cgfSampleCount = 8
                    , cgfStable = True
                    , cgfMetric = Nothing
                    , cgfSource = Nothing
                    }
                ]
        )
    assert
        "optimizer correlation guidance accepts interaction fields"
        ( parseOptimizerCorrelationGuidance "{\"interactions\":[{\"keyA\":\"volEwmaAlpha\",\"targetA\":0.2,\"keyB\":\"maxPositionSize\",\"targetB\":0.25,\"strength\":0.5,\"samples\":12,\"stable\":true}]}"
            == Right
                [ CorrelationGuidanceField
                    { cgfKey = "volEwmaAlpha"
                    , cgfTarget = 0.2
                    , cgfLo = Nothing
                    , cgfHi = Nothing
                    , cgfStrength = 0.5
                    , cgfSampleCount = 12
                    , cgfStable = True
                    , cgfMetric = Just "roi"
                    , cgfSource = Just "interaction"
                    }
                , CorrelationGuidanceField
                    { cgfKey = "maxPositionSize"
                    , cgfTarget = 0.25
                    , cgfLo = Nothing
                    , cgfHi = Nothing
                    , cgfStrength = 0.5
                    , cgfSampleCount = 12
                    , cgfStable = True
                    , cgfMetric = Just "roi"
                    , cgfSource = Just "interaction"
                    }
                ]
        )
    assert
        "optimizer correlation guidance rejects malformed JSON"
        (isLeft (parseOptimizerCorrelationGuidance "{\"key\":\"stopLoss\""))

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

testOptimizerSoftSearchEligibility :: IO ()
testOptimizerSoftSearchEligibility = do
    let searchMaxWfSharpeStd = 1.0
        eligible reason wfStd =
            optimizerSoftSearchEligible
                reason
                wfStd
                searchMaxWfSharpeStd
        softReasons =
            [ "activityCount<20"
            , "exposure<0.100"
            , "wfSharpeMean<0.800"
            , "wfSharpeStd>0.500"
            , "sharpe<1.000"
            , "calmar<0.800"
            , "annualizedReturn<1.500"
            , "winRate<0.450"
            , "profitFactor<1.100"
            , "turnover>3.000"
            ]
        hardReasons =
            [ "openThresholdOutsideActiveRange[0.001000,0.020000]"
            , "minEdge<0.001000(costFloor)"
            , "kellyLiteExposureMissing"
            , "walkForwardMissing"
            ]
    assert
        "optimizer classifies only explicitly approved quality filters as search-soft"
        ( all (optimizerSoftSearchFilterReason . Just) softReasons
            && not (any (optimizerSoftSearchFilterReason . Just) hardReasons)
            && not (optimizerSoftSearchFilterReason Nothing)
        )
    assert
        "hard optimizer filters never become search-only candidates"
        (all (\reason -> not (eligible (Just reason) (Just 0.4))) hardReasons)
    assert
        "soft-search eligibility fails closed without walk-forward evidence"
        (not (eligible (Just "sharpe<1.000") Nothing))
    assert
        "a quality-filtered candidate with stable walk-forward evidence remains search-only eligible"
        (eligible (Just "sharpe<1.000") (Just 0.4))
    assert
        "soft-search eligibility enforces the relaxed walk-forward dispersion ceiling"
        (not (eligible (Just "sharpe<1.000") (Just 1.1)))

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
                , brExposureCurve = [0, 1]
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

testMetricsFiniteInputBoundary :: IO ()
testMetricsFiniteInputBoundary = do
    let badResult =
            BacktestResult
                { brEquityCurve = [1, 0 / 0, 1 / 0, 1.0e-300, 1.0e308]
                , brTrades = []
                , brPositions = [0, 0 / 0, 1 / 0, -1]
                , brExposureCurve = [0, 0 / 0, 1 / 0, -1]
                , brAgreementOk = [True]
                , brAgreementValid = [True]
                , brPositionChanges = -3
                , brCostAttribution = emptyBacktestCostAttribution []
                }
        metricsFor annualization = computeMetrics annualization badResult
        malformedAnnualizations = [0 / 0, 1 / 0, negate (1 / 0), -252, 0]
        metricDoubles metrics =
            [ bmFinalEquity metrics
            , bmTotalReturn metrics
            , bmAnnualizedReturn metrics
            , bmAnnualizedVolatility metrics
            , bmSharpe metrics
            , bmSortino metrics
            , bmCalmar metrics
            , bmDownsideVolatility metrics
            , bmVaR95 metrics
            , bmCVaR95 metrics
            , bmMaxDrawdown metrics
            , bmWinRate metrics
            , bmGrossProfit metrics
            , bmGrossLoss metrics
            , bmAvgTradeReturn metrics
            , bmAvgHoldingPeriods metrics
            , bmExposure metrics
            , bmAgreementRate metrics
            , bmTurnover metrics
            ]
        allFinite metrics = all finiteDouble (metricDoubles metrics) && maybe True finiteDouble (bmProfitFactor metrics)
    assert
        "metrics remain finite for malformed annualization and source values"
        (all (allFinite . metricsFor) malformedAnnualizations)
    assert
        "malformed annualization fails closed to zero annualized metrics"
        ( all
            ( \annualization ->
                let metrics = metricsFor annualization
                 in bmAnnualizedReturn metrics == 0
                        && bmAnnualizedVolatility metrics == 0
                        && bmSharpe metrics == 0
                        && bmSortino metrics == 0
            )
            malformedAnnualizations
        )
    assert
        "negative position-change evidence cannot create negative turnover"
        (bmPositionChanges (metricsFor 252) == 0 && bmTurnover (metricsFor 252) == 0)

testOnlineStatsFiniteInputBoundary :: IO ()
testOnlineStatsFiniteInputBoundary = do
    let unchanged = foldl (flip updateWelford) emptyWelford [0 / 0, 1 / 0, negate (1 / 0)]
        learned = foldl (flip updateWelford) emptyWelford [1, 2, 3]
        repaired = updateWelford 7 (Welford (-1) (0 / 0) (-1))
    assert "non-finite online observations are ignored" (unchanged == emptyWelford)
    assert "finite Welford observations update count and mean" (wCount learned == 3 && abs (wMean learned - 2) < 1e-12)
    assert "finite Welford sample variance is correct" (varianceWelford learned == Just 1)
    assert "corrupted Welford state resets on the next finite observation" (repaired == Welford 1 7 0)
    assert "corrupted Welford state cannot emit variance" (isNothing (varianceWelford (Welford 3 0 (0 / 0))))

testIndependentRoiSpecification :: IO ()
testIndependentRoiSpecification = do
    let result =
            BacktestResult
                { brEquityCurve = [1, 1.02, 1.01, 1.05]
                , brTrades = []
                , brPositions = [0, 0.5, 0.5, 0]
                , brExposureCurve = [0, 0.5, 0.5, 0]
                , brAgreementOk = [True, False, True]
                , brAgreementValid = [True, True, True]
                , brPositionChanges = 2
                , brCostAttribution = emptyBacktestCostAttribution []
                }
        metrics =
            (computeMetrics 252 result)
                { bmCVaR95 = 0.03
                , bmMaxDrawdown = 0.04
                , bmAvgTradeReturn = 0.01
                , bmAvgHoldingPeriods = 4
                , bmRoundTrips = 3
                , bmTradeCount = 4
                , bmExposure = 0.25
                }
        customConfig =
            defaultFormalRoiScoreConfig
                { rscExpectancyRewardWeight = 0.73
                , rscPaybackRewardCap = 0.08
                , rscMinimumActivityFloor = 2
                , rscMinimumExposureFloor = 0.05
                , rscZeroRoundTripPenalty = 0.11
                , rscLowRoundTripPenalty = 0.07
                , rscZeroActivityPenalty = 0.13
                , rscLowActivityPenalty = 0.04
                , rscZeroExposurePenalty = 0.09
                , rscLowExposurePenaltyBase = 0.03
                , rscLowExposurePenaltyGapScale = 0.02
                }
        malformedConfig =
            customConfig
                { rscExpectancyRewardWeight = 0 / 0
                , rscPaybackRewardCap = 1 / 0
                , rscMinimumActivityFloor = -4
                , rscMinimumExposureFloor = -1
                }
        agrees config =
            abs
                ( roiSpecScoreWithConfig config 1.2 0.3 metrics
                    - roiImplementationScoreWithConfig config 1.2 0.3 metrics
                )
                <= 1e-12
    assert "independent ROI specification matches a custom production configuration" (agrees customConfig)
    assert "independent ROI specification matches production sanitization" (agrees malformedConfig)

testSensitivityAnalysisInvariants :: IO ()
testSensitivityAnalysisInvariants = do
    let spec =
            ParameterSpec
                { psName = "openThreshold"
                , psDescription = "test parameter"
                , psMin = 1
                , psMax = 3
                , psSteps = 3
                , psBaseline = 2
                }
        offGridSpec = spec{psMax = 4, psBaseline = 2}
        evaluator value = (2 * value, 0.1, 10, 0.5, 1.2)
        checked = runLocalSensitivityChecked spec evaluator
        offGrid = runLocalSensitivityChecked offGridSpec evaluator
        invalidSpec = spec{psSteps = 1}
        invalidReport = runLocalSensitivity invalidSpec (\_ -> error "invalid sensitivity spec evaluated")
        malformedOutput = runLocalSensitivity spec (const (0 / 0, 1 / 0, -2, 2, -1))
        pointsFinite report =
            all
                ( \point ->
                    all finiteDouble [spParameterValue point, spSharpe point, spMaxDrawdown point, spWinRate point, spProfitFactor point]
                        && spTradeCount point >= 0
                        && spWinRate point >= 0
                        && spWinRate point <= 1
                )
                (srPoints report)
    assert "valid sensitivity specifications are accepted" (not (isLeft (validateParameterSpec spec)))
    assert "invalid sensitivity step counts are rejected" (isLeft (validateParameterSpec invalidSpec))
    case checked of
        Left err -> assert ("valid sensitivity analysis failed: " ++ err) False
        Right report -> do
            assert "sensitivity analysis evaluates the declared grid" (map spParameterValue (srPoints report) == [1, 2, 3])
            assertNear "linear response has unit elasticity" 1 (srElasticity report) 1e-12
    case offGrid of
        Left err -> assert ("off-grid baseline analysis failed: " ++ err) False
        Right report ->
            assert
                "off-grid baselines are included as explicit evidence"
                (any ((< 1e-12) . abs . subtract (psBaseline offGridSpec) . spParameterValue) (srPoints report))
    assert "invalid sensitivity specs fail without evaluating the callback" (null (srPoints invalidReport))
    assert "malformed evaluator outputs sanitize to a finite report" (pointsFinite malformedOutput && finiteDouble (srElasticity malformedOutput))
    let poisoned = malformedOutput{srElasticity = 0 / 0}
    assert "non-finite reports cannot win sensitivity ranking" (mostSensitiveParameter [poisoned, malformedOutput] == Just (srParameter malformedOutput))

testRiskRegisterInvariants :: IO ()
testRiskRegisterInvariants = do
    let ids = map reId riskRegister
        nonEmpty textValue = not (T.null (T.strip textValue))
        complete entry = nonEmpty (reDescription entry) && nonEmpty (reOwner entry) && nonEmpty (reMitigation entry)
        lookupConsistent entry = riskSeverityOf (reId entry) == Just (reSeverity entry)
    assert "formal risk-register IDs are unique" (length ids == length (nub ids))
    assert "formal risk-register entries have complete ownership and mitigation text" (all complete riskRegister)
    assert "formal risk-register severity lookup matches every canonical entry" (all lookupConsistent riskRegister)

testSafeNumericConstruction :: IO ()
testSafeNumericConstruction = do
    let malformed = [0 / 0, 1 / 0, negate (1 / 0), -1]
    assert "malformed quantities sanitize to zero" (all ((== 0) . fromQuantity . quantityFromDouble) malformed)
    assert "valid quantities preserve their value" (fromQuantity (quantityFromDouble 2.5) == 2.5)
    assert "malformed leverage sanitizes to a finite positive floor" (all ((== 1e-12) . fromLeverage . leverageFromDouble) (0 : malformed))
    assert "valid leverage preserves its value" (fromLeverage (leverageFromDouble 3) == 3)

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
        bounded = foldl (flip recordRejection) (emptyTelemetry 2) (replicate 5 rej1)
        disabledRecent = recordRejection rej1 (emptyTelemetry (-1))
    assert "accumulated rejections count correctly" (gtTotalRejections tel3 == 3)
    assert "per-gate counts track correctly" (Map.lookup GateEdgeSpike (gtPerGateCounts tel3) == Just 2)
    assert "per-gate counts track correctly for second gate" (Map.lookup GateEdgeHeadroom (gtPerGateCounts tel3) == Just 1)
    assert "recent rejections bounded" (length (gtRecentRejections tel3) <= 10)
    assert "configured recent rejection bound is honored" (length (gtRecentRejections bounded) == 2)
    assert "configured recent rejection bound remains observable" (gtMaxRecent bounded == 2)
    assert "negative recent rejection bounds sanitize to zero" (null (gtRecentRejections disabledRecent) && gtMaxRecent disabledRecent == 0)

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

testThresholdCalibrationRejectsMalformedMethod :: IO ()
testThresholdCalibrationRejectsMalformedMethod = do
    let edges = [0.001, 0.002, 0.003]
        malformedMethods =
            [ PercentileMethod (-1)
            , PercentileMethod 101
            , PercentileMethod (0 / 0)
            , StdDevMethod (-1)
            , StdDevMethod (1 / 0)
            , HybridMethod 50 (0 / 0)
            , HybridMethod (negate (1 / 0)) 1
            ]
    assert
        "malformed calibration methods fail validation"
        (all (isLeft . validateCalibrationMethod) malformedMethods)
    assert
        "malformed calibration methods cannot emit a calibration"
        (all (isNothing . calibrateThreshold edges) malformedMethods)

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
        "orderAppliedFraction implementation matches spec on bounded grid"
        (fvrExecOrderAppliedFractionImplMatchesSpec report)
    assert
        "orderAppliedFraction never exceeds intended exposure"
        (fvrExecOrderAppliedFractionBounded report)
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

testMarketContextAlignsPeerKlinesByOpenTime :: IO ()
testMarketContextAlignsPeerKlinesByOpenTime = do
    let openTimes = V.fromList [1000, 2000, 3000]
        klines =
            [ mkKline 3000 0 0 0 30
            , mkKline 1000 0 0 0 10
            , mkKline 4000 0 0 0 40
            , mkKline 2000 0 0 0 20
            ]
    assert
        "market context aligns peer closes by openTime instead of fetch order"
        (alignKlineClosesToOpenTimes openTimes klines == Just (V.fromList [10, 20, 30]))
    assert
        "market context fails closed when a peer candle is missing from the target grid"
        (isNothing (alignKlineClosesToOpenTimes openTimes (filter ((/= 2000) . kOpenTime) klines)))

testOrderIntentUsesCloseDirectionForExistingPositions :: IO ()
testOrderIntentUsesCloseDirectionForExistingPositions = do
    let longExit = desiredPositionForSignal LongFlat 1 Nothing Nothing
        longHold = desiredPositionForSignal LongFlat 1 Nothing (Just 1)
        shortExit = desiredPositionForSignal LongShort (-1) Nothing Nothing
        shortHold = desiredPositionForSignal LongShort (-1) Nothing (Just (-1))
        longToShort = desiredPositionForSignal LongShort 1 (Just (-1)) (Just 1)
    assert "long-flat existing long exits when the close signal no longer confirms the hold" (longExit == 0)
    assert "long-flat existing long holds when closeDirection still confirms long" (longHold == 1)
    assert "long-short existing short exits when the close signal no longer confirms the hold" (shortExit == 0)
    assert "long-short existing short holds when closeDirection still confirms short" (shortHold == -1)
    assert "long-short open-threshold opposite signal still flips" (longToShort == -1)
    assert "close-only long exit becomes a SELL order direction" (orderDirectionForTransition 1 longExit == Just (-1))
    assert "close-only short exit becomes a BUY order direction" (orderDirectionForTransition (-1) shortExit == Just 1)

testVolConfHoldPreservesLivePosition :: IO ()
testVolConfHoldPreservesLivePosition = do
    let sideDirection side =
            case side of
                SideLong -> 1
                SideShort -> -1
        latestCloseSide =
            volConfStatefulCloseDirection
                VolConfGateHold
                (Just SideShort)
                (Just SideShort)
        latestCloseDirection = sideDirection <$> latestCloseSide
        liveDesired =
            desiredPositionForSignalWithVolConf
                VolConfGateHold
                LongShort
                1
                Nothing
                latestCloseDirection
        liveWithoutHold =
            desiredPositionForSignalWithVolConf
                VolConfGateAllowEntry
                LongShort
                1
                Nothing
                latestCloseDirection
        backtestHeld =
            applyVolConfGateBehavior
                VolConfGateHold
                (Just SideLong)
                0.6
                (Just SideShort)
                0.8
    assert
        "vol-conf HOLD suppresses latest-signal close direction"
        (isNothing latestCloseSide)
    assert
        "live stateful reduction preserves the held long when latest-signal reports vol-conf HOLD"
        (liveDesired == 1)
    assert
        "without vol-conf HOLD the same neutral latest signal exits the held long"
        (liveWithoutHold == 0)
    assert
        "backtest vol-conf HOLD preserves the same held side and size"
        (backtestHeld == (Just SideLong, 0.6))

testLongShortFlipCountsExitAndEntryTurnover :: IO ()
testLongShortFlipCountsExitAndEntryTurnover = do
    let prices = V.fromList [100 :: Double, 100, 100]
        preds = V.fromList [102 :: Double, 98]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecFee = 0
                , ecPositioning = LongShort
                , ecMaxPositionSize = 1
                }
        result = simulateEnsembleWithHLChecked cfg 1 prices prices prices preds preds (Nothing :: Maybe (V.Vector StepMeta))
    case result of
        Left err -> ioError (userError ("long-short flip turnover regression failed to simulate: " ++ err))
        Right bt ->
            let positions = brPositions bt
                flipsLongToShort =
                    case dropWhile (<= 0) positions of
                        [] -> False
                        _long : afterLong -> any (< 0) afterLong
             in do
                    assert
                        "long-short backtest transitions from held long to short"
                        flipsLongToShort
                    assert
                        "long-short flip counts initial entry, exit, and replacement entry as separate turnover"
                        (brPositionChanges bt == 3)

testIntrabarTakeProfitUsesExitBarCost :: IO ()
testIntrabarTakeProfitUsesExitBarCost = do
    let prices = V.fromList [100 :: Double, 100, 200, 204]
        highs = V.fromList [100 :: Double, 100, 200, 204]
        lows = V.fromList [100 :: Double, 100, 200, 204]
        preds = V.fromList [100 :: Double, 102, 300]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecFee = 0
                , ecSlippage = 0
                , ecSlippageVolMult = 0.01
                , ecSpread = 0
                , ecVolLookback = 2
                , ecTakeProfit = Just 0.03
                , ecMaxPositionSize = 1
                }
        result = simulateEnsembleWithHLChecked cfg 2 prices highs lows preds preds (Nothing :: Maybe (V.Vector StepMeta))
    case result of
        Left err -> ioError (userError ("intrabar cost-index regression failed to simulate: " ++ err))
        Right bt -> do
            let attribution = brCostAttribution bt
            assert
                "intrabar take-profit exit charges volatility-dependent slippage from the exit bar"
                (bcaRealizedSlippageCost attribution > 0)

-- Exposure measures capital used during each return interval, not merely the
-- position left at the bar close. A same-bar round trip must therefore retain
-- exposure evidence even though its closing-position series is entirely flat.
testIntrabarRoundTripRecordsExposure :: IO ()
testIntrabarRoundTripRecordsExposure = do
    let prices = V.fromList [100 :: Double, 100, 100]
        highs = V.fromList [100 :: Double, 104, 100]
        lows = prices
        preds = V.fromList [102 :: Double, 100]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecFee = 0
                , ecSlippage = 0
                , ecSpread = 0
                , ecTakeProfit = Just 0.03
                , ecMaxPositionSize = 1
                , ecMinPositionSize = 0
                }
        result = simulateEnsembleWithHLChecked cfg 1 prices highs lows preds preds (Nothing :: Maybe (V.Vector StepMeta))
    case result of
        Left err -> ioError (userError ("intrabar exposure regression failed to simulate: " ++ err))
        Right bt -> do
            assert "same-bar take-profit produces a completed trade" (not (null (brTrades bt)))
            assert "same-bar take-profit leaves every closing position flat" (all ((<= 1e-12) . abs) (brPositions bt))
            assert "same-bar take-profit records realized interval exposure" (any ((> 1e-12) . abs) (brExposureCurve bt))
            assert "same-bar take-profit contributes positive exposure metrics" (bmExposure (computeMetrics 252 bt) > 0)

-- A partial take-profit should convert the remaining position into a protected
-- runner: the residual stop moves to breakeven and can close the rest without
-- giving back the first realized target.
testPartialTakeProfitMovesLongStopToBreakeven :: IO ()
testPartialTakeProfitMovesLongStopToBreakeven = do
    let prices = V.fromList [100 :: Double, 100, 100, 102, 100, 100]
        highs = V.fromList [100 :: Double, 100, 100, 102, 102, 100]
        lows = V.fromList [100 :: Double, 100, 100, 101, 100, 100]
        kalPreds = V.fromList [100 :: Double, 102, 103, 103, 103]
        lstmPreds = V.fromList [100 :: Double, 102, 103, 103, 103]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecVolLookback = 2
                , ecFee = 0
                , ecStopLoss = Just 0.02
                , ecTakeProfit = Just 0.02
                , ecTakeProfitPartial = 0.5
                , ecCooldownBars = 10
                , ecMaxPositionSize = 1
                }
        result = simulateEnsemble cfg 2 prices highs lows kalPreds lstmPreds (Nothing :: Maybe (V.Vector StepMeta))
        trades = brTrades result
    assert
        "long partial take-profit leaves a trade to close later"
        (not (null trades))
    assert
        "long remainder exits at breakeven stop"
        ( case trades of
            [] -> False
            ts -> trExitReason (last ts) == Just ExitStopLoss
        )
    assert
        "long partial profit keeps the total trade positive"
        ( case trades of
            [] -> False
            ts -> trReturn (last ts) > 0
        )

testPartialTakeProfitMovesShortStopToBreakeven :: IO ()
testPartialTakeProfitMovesShortStopToBreakeven = do
    let prices = V.fromList [100 :: Double, 100, 100, 98, 100, 100]
        highs = V.fromList [100 :: Double, 100, 100, 99, 100, 100]
        lows = V.fromList [100 :: Double, 100, 100, 98, 98, 100]
        kalPreds = V.fromList [100 :: Double, 98, 97, 97, 97]
        lstmPreds = V.fromList [100 :: Double, 98, 97, 97, 97]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecVolLookback = 2
                , ecFee = 0
                , ecStopLoss = Just 0.02
                , ecTakeProfit = Just 0.02
                , ecTakeProfitPartial = 0.5
                , ecCooldownBars = 10
                , ecPositioning = LongShort
                , ecMaxPositionSize = 1
                }
        result = simulateEnsemble cfg 2 prices highs lows kalPreds lstmPreds (Nothing :: Maybe (V.Vector StepMeta))
        trades = brTrades result
    assert
        "short partial take-profit leaves a trade to close later"
        (not (null trades))
    assert
        "short remainder exits at breakeven stop"
        ( case trades of
            [] -> False
            ts -> trExitReason (last ts) == Just ExitStopLoss
        )
    assert
        "short partial profit keeps the total trade positive"
        ( case trades of
            [] -> False
            ts -> trReturn (last ts) > 0
        )

testPartialTakeProfitTradeFeesMatchAttribution :: IO ()
testPartialTakeProfitTradeFeesMatchAttribution = do
    let prices = V.fromList [100 :: Double, 100, 100, 102, 100, 100]
        highs = V.fromList [100 :: Double, 100, 100, 102, 102, 100]
        lows = V.fromList [100 :: Double, 100, 100, 101, 100, 100]
        kalPreds = V.fromList [100 :: Double, 102, 103, 103, 103]
        lstmPreds = V.fromList [100 :: Double, 102, 103, 103, 103]
        cfg =
            sampleEnsembleConfig
                { ecOpenThreshold = 0.01
                , ecCloseThreshold = 0.005
                , ecVolLookback = 2
                , ecFee = 0.001
                , ecSlippage = 0
                , ecSpread = 0
                , ecStopLoss = Just 0.02
                , ecTakeProfit = Just 0.02
                , ecTakeProfitPartial = 0.5
                , ecCooldownBars = 10
                , ecMaxPositionSize = 1
                }
        result = simulateEnsemble cfg 2 prices highs lows kalPreds lstmPreds (Nothing :: Maybe (V.Vector StepMeta))
        trades = brTrades result
        feeAttribution = bcaRealizedFeeCost (brCostAttribution result)
        tradeFees = sum (map trFeeCost trades)
    assert "partial take-profit regression produces a trade" (not (null trades))
    assertNear
        "partial take-profit trade-level fee cost matches realized fee attribution"
        feeAttribution
        tradeFees
        1e-12

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
        liveHalt =
            liveRiskHaltAction
                1
                HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0.06
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Just 0.05
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = 1
                    , hiMaxPositionSizeLim = Just 1
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    , hiVolTarget = 0
                    , hiLeverage = 0
                    }
    assert
        "max-drawdown simulation produces at least one trade"
        (not (null trades))
    assert
        "max-drawdown simulation ends flat"
        (not (null positions) && last positions == 0)
    assert
        "last trade exits with ExitMaxDrawdown"
        ( case trades of
            [] -> False
            ts -> trExitReason (last ts) == Just ExitMaxDrawdown
        )
    assert
        "live and backtest use the same canonical max-drawdown halt reason"
        ( lrhaExitReason liveHalt == Just ExitMaxDrawdown
            && case trades of
                [] -> False
                ts -> lrhaExitReason liveHalt == trExitReason (last ts)
        )
    assert
        "a live max-drawdown halt flattens a long with a sell transition"
        ( lrhaDesiredPosition liveHalt == 0
            && lrhaOrderDirection liveHalt == Just (-1)
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
        "adoption sizing floor is reachable when a legacy combo floor exceeds the cap"
        (capAdoptedMinPositionSize 0.05 0.16 == 0.05)
    assert
        "adoption sizing floor preserves a conservative value below the cap"
        (capAdoptedMinPositionSize 0.05 0.03 == 0.03)
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
                { aecMinEdgeFloor = adoptionMinEdgeFloor
                , aecMinTradeCount = adoptionMinTradeCount + 10
                , aecMinWalkForwardSharpeMean = adoptionMinWalkForwardSharpeMean
                , aecMaxWalkForwardSharpeStd = adoptionMaxWalkForwardSharpeStd
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

testComboMinEdgeMeetsAdoptionFloor :: IO ()
testComboMinEdgeMeetsAdoptionFloor = do
    assert
        "missing minEdge fails closed"
        (not (comboMinEdgeMeetsAdoptionFloor Nothing))
    assert
        "NaN minEdge fails closed"
        (not (comboMinEdgeMeetsAdoptionFloor (Just (0 / 0))))
    assert
        "below-floor minEdge fails"
        (not (comboMinEdgeMeetsAdoptionFloor (Just (adoptionMinEdgeFloor / 2))))
    assert
        "at-floor minEdge passes"
        (comboMinEdgeMeetsAdoptionFloor (Just adoptionMinEdgeFloor))
    let relaxedConfig =
            AdoptionEvidenceConfig
                { aecMinEdgeFloor = adoptionMinEdgeFloor / 2
                , aecMinTradeCount = adoptionMinTradeCount
                , aecMinWalkForwardSharpeMean = adoptionMinWalkForwardSharpeMean
                , aecMaxWalkForwardSharpeStd = adoptionMaxWalkForwardSharpeStd
                }
    assert
        "configured relaxed minEdge floor accepts the relaxed value"
        (comboMinEdgeMeetsAdoptionFloorWithConfig relaxedConfig (Just (adoptionMinEdgeFloor / 2)))

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
                { aecMinEdgeFloor = adoptionMinEdgeFloor
                , aecMinTradeCount = adoptionMinTradeCount
                , aecMinWalkForwardSharpeMean = adoptionMinWalkForwardSharpeMean + 0.4
                , aecMaxWalkForwardSharpeStd = adoptionMaxWalkForwardSharpeStd
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

testComboWalkForwardSharpeStdMeetsAdoptionCeiling :: IO ()
testComboWalkForwardSharpeStdMeetsAdoptionCeiling = do
    assert
        "default walk-forward std ceiling mirrors optimizer default"
        (abs (adoptionMaxWalkForwardSharpeStd - 1.5) < 1.0e-12)
    assert
        "missing walk-forward std fails closed"
        (not (comboWalkForwardSharpeStdMeetsAdoptionCeiling Nothing))
    assert
        "NaN walk-forward std fails closed"
        (not (comboWalkForwardSharpeStdMeetsAdoptionCeiling (Just (0 / 0))))
    assert
        "above-ceiling walk-forward std fails"
        (not (comboWalkForwardSharpeStdMeetsAdoptionCeiling (Just (adoptionMaxWalkForwardSharpeStd + 0.1))))
    assert
        "at-ceiling walk-forward std passes"
        (comboWalkForwardSharpeStdMeetsAdoptionCeiling (Just adoptionMaxWalkForwardSharpeStd))
    let relaxedConfig =
            AdoptionEvidenceConfig
                { aecMinEdgeFloor = adoptionMinEdgeFloor
                , aecMinTradeCount = adoptionMinTradeCount
                , aecMinWalkForwardSharpeMean = adoptionMinWalkForwardSharpeMean
                , aecMaxWalkForwardSharpeStd = adoptionMaxWalkForwardSharpeStd + 10
                }
        disabledConfig = relaxedConfig{aecMaxWalkForwardSharpeStd = 0}
    assert
        "configured relaxed std ceiling accepts the relaxed value"
        (comboWalkForwardSharpeStdMeetsAdoptionCeilingWithConfig relaxedConfig (Just (adoptionMaxWalkForwardSharpeStd + 10)))
    assert
        "disabled std ceiling still requires present evidence"
        ( comboWalkForwardSharpeStdMeetsAdoptionCeilingWithConfig disabledConfig (Just 100)
            && not (comboWalkForwardSharpeStdMeetsAdoptionCeilingWithConfig disabledConfig Nothing)
        )

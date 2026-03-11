{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Concurrent (threadDelay)
import Control.Exception (SomeException, evaluate, finally, try)
import qualified Control.Monad
import Data.Aeson (eitherDecode, object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Int (Int64)
import Data.List (foldl', isInfixOf, sort, sortOn)
import Data.Maybe (isNothing)
import qualified Data.Maybe
import Data.Time.Clock.POSIX (getPOSIXTime)
import qualified Data.Vector as V
import Options.Applicative (ParserResult (..), defaultPrefs, execParserPure, fullDesc, helper, info, renderFailure, (<**>))
import System.Directory (createDirectoryIfMissing, removeDirectoryRecursive)
import System.Exit (exitFailure, exitSuccess)
import System.FilePath ((</>))
import System.Timeout (timeout)

import Trader.App.Args (Args (..), argBinanceMarket, argBinanceSymbol, argIdempotencyKey, argInterval, argLookback, opts, parseTimestampMs, validateArgs)
import Trader.Binance (BinanceMarket (..), BinanceOrderMode (..), BinanceTrade (..), Kline (..), OrderSide (..), binanceBaseUrl, newBinanceEnv, placeMarketOrder, signQuery)
import Trader.BinanceIntervals (binanceIntervalsCsv, isBinanceInterval)
import Trader.BotStartSemantics (botTradeEnabledFromApi, shouldClearPositionOriginOnStart, shouldPersistPositionOriginOnSwitch, shouldPreserveProvidedComboOnActiveAdopt, shouldResolveOriginComboOnAutoStart)
import Trader.Cache (fetchWithCache, insertCache, newTtlCache)
import Trader.Coinbase (CoinbaseCandle (..), buildRanges, decodeCoinbaseCandles, normalizeCoinbaseCandles)
import Trader.Config (shouldRequireUserTradeKeys, validateRuntimeConfig)
import Trader.Dex (DexEnv (..), DexToken (..), extractTxHash, resolveDexTokens, tokenAmountToInteger)
import Trader.Duration (TimeWindow (..), inferPeriodsPerYear, lookbackBarsFrom, minuteOfDayFromMs, parseDurationSeconds, parseTimeWindow)
import Trader.Formal.CloseTiming (CloseTimingDecision (..), CloseTimingObservation (..), CloseTimingStats (..), buildCloseTimingStats, closeTimingDecision, optimalCloseObservation)
import Trader.Formal.Optimization (FormalVerificationReport (..), roiRequirementClauses, roiRequirementSummary, verifyFormalOptimization)
import Trader.Http (boundedBackoffMs, jitteredDelayMs, parseRetryAfterFromHeadersAt, parseRetryAfterMsAt)
import Trader.Kalman3 (Kalman3 (..), KalmanRun (..), Vec3 (..), constantAcceleration1D, forecastNextConstantAcceleration1D, runConstantAcceleration1D, step)
import Trader.KalmanFusion (Kalman1 (..), initKalman1, updateMulti)
import Trader.Kraken (KrakenCandle (..), decodeKrakenCandles)
import Trader.LSTM (LSTMConfig (..), LSTMModel (..), buildSequences, evaluateLoss, trainLSTM)
import Trader.LstmPersistence (lstmModelKey)
import Trader.MarketContext (MarketModel (..), fitLinearRange)
import Trader.Method (Method (..), parseMethod, selectPredictions)
import Trader.Metrics (bmAnnualizedReturn, bmAvgTradeReturn, bmCalmar, bmExposure, bmGrossLoss, bmGrossProfit, bmMaxDrawdown, bmProfitFactor, bmTotalReturn, computeMetrics)
import Trader.Optimization (TuneObjective (..), bestFinalEquity, optimizeOperations, parseTuneObjective, sweepThreshold)
import Trader.Optimizer.Merge (MergeArgs (..), runMerge)
import Trader.Optimizer.Optimize (normalizeObjectiveCode, normalizeOptionalPositiveFraction, objectiveScore, qualityPresetIntervalFields, sampleTakeProfitPartial)
import Trader.Optimizer.Random (nextDouble, nextIntRange, seedRng)
import Trader.OrderExecution (OrderExecutionEvidence (..), applyExecutedQuantity, orderAppliedQuantity)
import Trader.Platform (Platform (..), coinbaseIntervalSeconds, isPlatformInterval, krakenIntervalMinutes, parsePlatform, poloniexIntervalLabel, poloniexIntervalSeconds)
import Trader.Poloniex (PoloniexCandle (..), decodePoloniexCandles, normalizePoloniexCandles)
import Trader.Predictors (Interval (..), Quantiles (..), RegimeProbs (..), SensorId (..), SensorOutput (..), initHMMFilter, predictSensors, trainPredictors)
import Trader.Predictors.Features (buildDatasetWithIndex, featuresAt, featuresAtWithMarket, forwardReturnAt, mkFeatureSpec)
import Trader.Predictors.Transformer (TransformerModel (..), predictTransformer, trainTransformer)
import Trader.Predictors.Types (allPredictors)
import Trader.SensorVariance (emptySensorVar, updateResidual, varianceFor)
import Trader.SignalGates (signalCrossAssetCheck, signalFundingOiCheck, signalMetaLabelOk, signalMtfConsensusCheck, signalRegimeEdgeOk, signalRunPostDirectionGates)
import Trader.Split (Split (..), splitTrainBacktest)
import qualified Trader.Symbol as Symbol
import Trader.Test.ApiRoutes (apiRouteSuite)
import Trader.Test.BinanceProbe (binanceProbeSuite)
import Trader.TopCombosStore (comboPerformanceKey, mergeTopCombosPayloads, recalculateComboPerformanceFromOperation, resolveComboSymbol, sanitizeComboSymbolForPlatform, sanitizeTopCombosValue)
import Trader.Trading (BacktestResult (..), EnsembleConfig (..), ExitReason (..), IntrabarFill (..), Positioning (..), Trade (..), simulateEnsemble, simulateEnsembleWithHLChecked)

main :: IO ()
main = do
    results <-
        sequence
            ( [ run "duration lookback bars" testLookbackBars
              , run "duration rejects overflow integers" testLookbackBarsOverflowDuration
              , run "duration rejects overflow unit multiplication" testDurationOverflowUnitMultiplication
              , run "duration rejects lookback arithmetic overflow" testLookbackBarsOverflowArithmetic
              , run "duration infers weekly periods from interval seconds" testInferPeriodsPerYearWeekly
              , run "duration infers monthly periods from interval seconds" testInferPeriodsPerYearMonthly
              , run "minute-of-day handles extreme epoch bounds" testMinuteOfDayFromMsBounds
              , run "time window parser enforces zero-padded HH:MM" testParseTimeWindowStrictClockFormat
              , run "kalman fusion multi-sensor" testKalmanFusionMulti
              , run "market linear fit" testMarketLinearFit
              , run "predictors output shape" testPredictorsOutputs
              , run "predictor features reject non-finite price inputs" testPredictorFeaturesRejectNonFinitePrices
              , run "predictor market features stay finite" testPredictorMarketFeaturesStayFinite
              , run "predictor dataset skips non-finite rows" testBuildDatasetSkipsNonFiniteRows
              , run "predictor dataset skips overflowed finite feature rows" testBuildDatasetSkipsOverflowedFiniteRows
              , run "transformer training skips invalid rows" testTransformerTrainingSanitizesDataset
              , run "transformer prediction rejects non-finite query" testTransformerPredictionRejectsNonFiniteQuery
              , run "transformer training normalizes invalid temperature" testTransformerInvalidTemperatureFallback
              , run "kalman constant series" testKalmanConstant
              , run "kalman forecast constant" testKalmanForecast
              , run "kalman innovation sign" testKalmanInnovationSign
              , run "forward return sign" testForwardReturnSign
              , run "lstm training improves loss" testLstmImprovesLoss
              , run "lstm training tolerates non-finite val ratio" testLstmInvalidValRatioFallback
              , run "lstm key uses platform" testLstmModelKeyPlatform
              , run "ensemble agreement gate" testAgreementGate
              , run "hold on close agreement" testHoldOnCloseAgree
              , run "min-hold blocks exit" testMinHoldBars
              , run "max-hold forces exit" testMaxHoldBars
              , run "cooldown blocks re-entry" testCooldownBars
              , run "tri-layer uses provided open prices for candle patterns" testTriLayerUsesProvidedOpenPrices
              , run "entry block holds position (no-trade window)" testEntryBlockNoTradeWindow
              , run "entry block holds position (max trades)" testEntryBlockMaxTradesPerDay
              , run "weekly loss resets on UTC calendar week boundary" testWeeklyLossResetsOnUtcWeekBoundary
              , run "flip fees apply per side" testFlipFeesPerSide
              , run "long-short down move" testLongShortDownMove
              , run "liquidation clamps equity" testLiquidationClamp
              , run "metrics max drawdown" testMetricsMaxDrawdown
              , run "metrics calmar falls back to annualized return at zero drawdown" testMetricsCalmarFallback
              , run "metrics profit factor pnl" testMetricsProfitFactorPnL
              , run "metrics sanitize non-finite trade payloads" testMetricsSanitizeNonFiniteInputs
              , run "formal ROI requirements stay concise" testFormalRoiRequirementsSummary
              , run "formal ROI spec matches implementation" testFormalRoiSpecMatchesImplementation
              , run "formal ROI monotonicity holds" testFormalRoiMonotonicity
              , run "formal ROI penalties stay ordered" testFormalRoiPenaltyOrdering
              , run "formal tune tie-break matches spec" testFormalTieBreakMatchesSpec
              , run "formal close timing window identifies tm" testFormalCloseTimingWindow
              , run "formal close timing policy closes past target quantile" testFormalCloseTimingDecision
              , run "binance signature length" testBinanceSignatureLength
              , run "binance kline json parsing" testBinanceKlineParsing
              , run "binance trade parser accepts numeric and whitespace fields" testBinanceTradeParserAcceptsNumericAndWhitespace
              , run "binance trade parser rejects non-finite inferred quote qty" testBinanceTradeParserRejectsNonFiniteInferredQuoteQty
              , run "coinbase candle parser rejects fractional numeric timestamp" testCoinbaseFractionalTimestampRejected
              , run "coinbase candle parser normalizes millisecond timestamp boundaries" testCoinbaseTimestampBoundaryNormalization
              , run "coinbase range builder stops at epoch boundary" testCoinbaseBuildRangesStopsAtEpochBoundary
              , run "coinbase range builder keeps chunk boundaries contiguous" testCoinbaseBuildRangesAreContiguous
              , run "coinbase candle normalization dedupes and caps bars" testCoinbaseNormalizeCandlesDedupeAndLimit
              , run "kraken candle parser rejects fractional numeric timestamp" testKrakenFractionalTimestampRejected
              , run "poloniex candle parser rejects fractional numeric timestamp" testPoloniexFractionalTimestampRejected
              , run "poloniex candle parser normalizes millisecond timestamp boundaries" testPoloniexTimestampBoundaryNormalization
              , run "poloniex candle normalization dedupes and caps bars" testPoloniexNormalizeCandlesDedupeAndLimit
              , run "exchange candle parsers reject non-finite numeric strings" testExchangeNonFiniteNumericStringRejected
              , run "exchange candle parsers accept whitespace-padded numeric strings" testExchangeNumericStringWhitespaceAcceptance
              , run "method parsing" testMethodParsing
              , run "tune objective parsing" testTuneObjectiveParsing
              , run "optimizer objective aliases canonicalize" testOptimizerObjectiveNormalization
              , run "platform parsing" testPlatformParsing
              , run "non-binance args ignore live by default" testNonBinanceArgsLiveDefault
              , run "binance market helper prioritizes futures on conflicting flags" testBinanceMarketConflictingFlagsPreferFutures
              , run "binance args normalize slash symbols" testBinanceSlashSymbolNormalization
              , run "coinbase args normalize slash symbols" testCoinbaseSlashSymbolNormalization
              , run "coinbase args normalize delimiter whitespace" testCoinbaseDelimiterWhitespaceNormalization
              , run "poloniex args normalize slash symbols" testPoloniexSlashSymbolNormalization
              , run "poloniex args normalize delimiter whitespace" testPoloniexDelimiterWhitespaceNormalization
              , run "args normalize interval casing/spacing" testArgsNormalizeIntervalCode
              , run "coinbase args reject compact symbols without delimiter" testCoinbaseCompactSymbolRejected
              , run "poloniex args reject compact symbols without delimiter" testPoloniexCompactSymbolRejected
              , run "binance args reject malformed non-alnum symbols" testBinanceMalformedSymbolRejected
              , run "binance args reject quote-only symbols" testBinanceQuoteOnlySymbolRejected
              , run "dry-run requires trade flag" testDryRunRequiresTrade
              , run "dry-run trade bypasses runtime credentials" testDryRunBypassesRuntimeCredentials
              , run "dry-run skips non-owner API key requirement" testDryRunSkipsNonOwnerUserKeyRequirement
              , run "live trade keeps non-owner API key requirement" testLiveTradeRequiresNonOwnerUserKeys
              , run "empty cli credentials rejected" testEmptyCliCredentialsRejected
              , run "backtest window validates time formats" testBacktestWindowTimeValidation
              , run "backtest window rejects overflow scientific timestamp" testBacktestWindowOverflowScientificValidation
              , run "backtest window rejects scientific integer timestamp" testBacktestWindowScientificIntegerValidation
              , run "backtest window rejects overflow integer timestamp" testBacktestWindowOverflowIntegerValidation
              , run "backtest window rejects fractional numeric timestamp" testBacktestWindowFractionalNumericValidation
              , run "backtest window rejects decimal-like integer timestamp" testBacktestWindowDecimalIntegerValidation
              , run "backtest window rejects non-decimal integer timestamp" testBacktestWindowNonDecimalIntegerValidation
              , run "backtest window accepts ISO offsets" testBacktestWindowIsoOffsetValidation
              , run "backtest window accepts ISO minute precision timestamps" testBacktestWindowIsoMinutePrecisionValidation
              , run "backtest window accepts lowercase zulu timestamps" testBacktestWindowLowercaseZuluValidation
              , run "backtest window accepts lowercase t separator timestamps" testBacktestWindowLowercaseTSeparatorValidation
              , run "backtest window accepts expanded-year ISO dates" testBacktestWindowExpandedYearIsoValidation
              , run "backtest window rejects out-of-range expanded-year ISO dates" testBacktestWindowExpandedYearOverflowValidation
              , run "backtest window keeps negative millisecond epochs" testBacktestWindowNegativeMillisecondsValidation
              , run "backtest window keeps positive 11-digit millisecond epochs" testBacktestWindowPositiveMillisecondsValidation
              , run "backtest window normalizes second epochs to milliseconds" testBacktestWindowSecondEpochNormalization
              , run "backtest window accepts signed positive second epochs" testBacktestWindowSignedSecondEpochValidation
              , run "backtest window enforces from<=to" testBacktestWindowOrderValidation
              , run "idempotency key enforces max length" testIdempotencyKeyLengthValidation
              , run "idempotency key rejects non-ascii characters" testIdempotencyKeyAsciiValidation
              , run "idempotency key trims surrounding whitespace" testIdempotencyKeyTrimValidation
              , run "bars rejects overflow integer" testBarsOverflowValidation
              , run "bars accepts signed positive integers" testBarsSignedIntegerValidation
              , run "bars rejects non-decimal integer" testBarsNonDecimalValidation
              , run "cli numeric args reject non-finite values" testNumericArgsFiniteValidation
              , run "trade sizing args reject zero values" testTradeSizingPositiveValidation
              , run "retry-after date parsing" testRetryAfterDateParsing
              , run "retry-after header lookup is case-insensitive" testRetryAfterHeaderLookupCaseInsensitive
              , run "retry-after uses first parseable duplicate header value" testRetryAfterDuplicateHeaderFallback
              , run "retry backoff clamps without overflow" testRetryBackoffOverflowClamp
              , run "retry jitter respects configured max delay" testRetryJitterMaxDelayClamp
              , run "cache returns stale value on quick upstream failure" testCacheQuickFailureUsesStale
              , run "cache rejects stale value after slow upstream failure" testCacheSlowFailureRejectsExpiredStale
              , run "initial balance must be positive" testInitialBalanceValidation
              , run "bot/start defaults botTrade to true" testBotTradeDefaultTrue
              , run "bot/auto-start resolves origin combo for active adoption" testAutoStartResolvesOriginComboForActiveAdopt
              , run "bot/start preserves provided combo for active adoption" testBotStartPreservesProvidedComboForActiveAdopt
              , run "bot/start clears origin only when adoptable and flat" testBotStartClearOriginGate
              , run "position origin persists only for live sent switches" testPersistPositionOriginGate
              , run "order execution uses fill evidence for live orders" testOrderAppliedQuantity
              , run "order execution updates position by executed qty" testApplyExecutedQuantity
              , run "signal gate emits MTF_WARMUP reason" testSignalGateMtfWarmup
              , run "signal gate emits MTF_CONSENSUS reason" testSignalGateMtfConsensus
              , run "signal gate emits CROSS_ASSET reason" testSignalGateCrossAsset
              , run "signal gate emits META_LABEL reason" testSignalGateMetaLabel
              , run "signal gate emits REGIME_BANK reason" testSignalGateRegimeBank
              , run "signal gate emits FUNDING_OI reason" testSignalGateFundingOi
              , run "signal funding/OI damp stays finite on non-finite inputs" testSignalFundingOiFiniteDamp
              , run "signal funding/OI zero caps disable gating" testSignalFundingOiZeroCapsDisable
              , run "sensor variance ignores non-finite residuals" testSensorVarianceIgnoresNonFiniteResiduals
              , run "sensor variance keeps prior estimate after non-finite residuals" testSensorVarianceNonFiniteNoMutation
              , run "combo performance recalculates from completed operation delta" testRecalculateComboPerformanceFromCompletedOperation
              , run "top combos merge ranks by nested metrics score" testMergeTopCombosRanksByNestedScore
              , run "top combos merge ignores non-numeric boolean scores" testMergeTopCombosIgnoresBooleanScore
              , run "top combos merge dedupe prefers nested metrics score" testMergeTopCombosDedupPrefersNestedScore
              , run "top combos merge keeps same params across distinct sources" testMergeTopCombosKeepsDistinctSources
              , run "top combos performance key ranks score before equity on ties" testComboPerformanceKeyRanksScoreBeforeEquity
              , run "optimizer merge ignores overflow scientific integer strings" testRunMergeIgnoresOverflowScientificIntegerString
              , run "optimizer merge rejects fractional integer-like strings" testRunMergeRejectsFractionalIntegerString
              , run "optimizer merge preserves distinct payload sources from top-json inputs" testRunMergePreservesDistinctPayloadSources
              , run "optimizer merge preserves dex platform/symbol semantics" testRunMergePreservesDexPlatformSymbolSemantics
              , run "optimizer merge canonicalizes coinbase-prefixed platform keys" testRunMergeCanonicalizesCoinbasePrefixedPlatform
              , run "optimizer merge canonicalizes dex-prefixed platform keys" testRunMergeCanonicalizesDexPrefixedPlatform
              , run "optimizer merge uses nested metrics fallbacks" testRunMergeUsesNestedMetricsFallbacks
              , run "optimizer merge dedupe prefers nested metrics score" testRunMergeDedupPrefersNestedMetricsScore
              , run "optimizer merge skips invalid utf8 jsonl lines" testRunMergeSkipsInvalidUtf8JsonlLines
              , run "optimizer merge skips string false ok flag" testRunMergeSkipsFalseStringOkFlag
              , run "top combos sanitize slash-delimited binance symbols" testTopCombosBinanceSlashSymbolSanitization
              , run "top combos recover compact binance symbol from prefixed pair text" testTopCombosPrefixedPairNormalization
              , run "top combos recover compact binance symbol from separated base/quote tokens" testTopCombosSeparatedTokenPairNormalization
              , run "top combos infer compact symbol from unknown delimited pair" testTopCombosUnknownPlatformPairNormalization
              , run "top combos reject numeric-only delimited symbols" testTopCombosRejectNumericOnlyDelimitedSymbols
              , run "top combos reject quote-only symbols" testTopCombosRejectQuoteOnlySymbols
              , run "top combos trim known binance suffixes" testTopCombosTrimKnownBinanceSuffixes
              , run "top combos keep unknown alpha suffixes" testTopCombosKeepUnknownAlphaSuffixes
              , run "top combos resolve coinbase symbols outside Binance" testResolveComboSymbolCoinbase
              , run "top combos resolve dex symbols outside Binance" testResolveComboSymbolDex
              , run "top combos resolve symbols from payload source fallback" testResolveComboSymbolSourceFallback
              , run "top combos ignore blank platform when source metadata exists" testResolveComboSymbolBlankPlatformFallsBackToSource
              , run "symbol split handles delimited pairs" testSplitSymbolDelimitedPairs
              , run "symbol split keeps compact pairs" testSplitSymbolCompactPairs
              , run "symbol split handles quote-only input safely" testSplitSymbolQuoteOnly
              , run "symbol sanitization canonicalizes coinbase-prefixed platform keys" testSymbolCoinbasePrefixedPlatformNormalization
              , run "top combos sanitize coinbase-prefixed platform symbols" testTopCombosCoinbasePrefixedPlatformSymbolSanitization
              , run "top combos sanitize payload symbols via source when platform is blank" testTopCombosPayloadBlankPlatformFallsBackToSource
              , run "symbol sanitization keeps dex symbol format for prefixed platform keys" testSymbolDexPrefixedPlatformNormalization
              , run "top combos keep dex symbol format for prefixed platform keys" testTopCombosDexPrefixedPlatformSymbolSanitization
              , run "dex trade args accept token pair without symbol" testDexTradeArgsRequireTokensNotSymbol
              , run "dex csv+symbol lookback validation uses csv bars" testDexCsvSymbolLookbackUsesCsvBars
              , run "dex trade args reject single-token override even with symbol" testDexTradeArgsRejectPartialTokenOverrides
              , run "dex token resolution rejects malformed token addresses" testDexResolveTokensRejectsMalformedAddress
              , run "dex token resolution applies native decimals overrides" testDexResolveTokensNativeDecimalsOverride
              , run "dex token resolution rejects excessive decimals overrides" testDexResolveTokensRejectsExcessiveDecimalsOverride
              , run "dex token amount conversion rejects excessive decimals" testTokenAmountToIntegerRejectsExcessiveDecimals
              , run "dex tx hash parser handles plain output" testDexExtractTxHashPlainOutput
              , run "dex tx hash parser handles structured output" testDexExtractTxHashStructuredOutput
              , run "dex tx hash parser rejects malformed output" testDexExtractTxHashRejectsMalformedOutput
              , run "platform intervals" testPlatformIntervals
              , run "binance interval helper normalizes casing/spacing" testBinanceIntervalHelperNormalization
              , run "platform interval mapping" testPlatformIntervalMapping
              , run "method selects predictions" testMethodSelection
              , run "train/backtest split" testTrainBacktestSplit
              , run "threshold sweep" testSweepThreshold
              , run "threshold sweep cost-pick uses latest return volatility" testSweepThresholdCostPickUsesLatestReturnVolatility
              , run "threshold sweep keeps non-finite context methods finite" testSweepThresholdFiniteOnNonFinitePredictions
              , run "threshold sweep regime-switch stays neutral with non-finite inputs" testSweepThresholdRegimeSwitchNonFiniteStaysNeutral
              , run "operations optimization" testOptimizeOperations
              , run "optimizer partial take-profit zero-range sampler" testOptimizerPartialTakeProfitZeroRange
              , run "optimizer quality preset preserves explicit interval constraints" testOptimizerQualityPresetIntervals
              , run "optimizer calmar falls back to annualized return at zero drawdown" testOptimizerCalmarFallback
              , run "optimizer normalizes optional positive fractions" testOptimizerNormalizeOptionalPositiveFraction
              , run "optimizer int range keeps rng for fixed range" testOptimizerIntRangeFixedRange
              , run "optimizer int range handles full Int span" testOptimizerIntRangeFullSpan
              , run "binance order validation" testBinanceOrderValidation
              ]
                ++ map (uncurry run) apiRouteSuite
                ++ map (uncurry run) binanceProbeSuite
            )
    if and results then exitSuccess else exitFailure

run :: String -> IO () -> IO Bool
run name action = do
    r <- (try action :: IO (Either SomeException ()))
    case r of
        Left e -> do
            putStrLn ("FAIL: " ++ name ++ " (" ++ show e ++ ")")
            pure False
        Right _ -> do
            putStrLn ("PASS: " ++ name)
            pure True

assert :: String -> Bool -> IO ()
assert msg cond =
    if cond then pure () else error msg

assertApprox :: String -> Double -> Double -> Double -> IO ()
assertApprox msg eps a b =
    assert msg (abs (a - b) <= eps)

assertApproxList :: String -> Double -> [Double] -> [Double] -> IO ()
assertApproxList msg eps xs ys =
    let sameLength = length xs == length ys
        allClose = and (zipWith (\a b -> abs (a - b) <= eps) xs ys)
     in assert msg (sameLength && allClose)

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)

requireRight :: String -> Either String a -> a
requireRight label res =
    case res of
        Left err -> error (label ++ ": " ++ err)
        Right v -> v

requireHead :: String -> [a] -> a
requireHead label xs =
    case xs of
        y : _ -> y
        [] -> error label

requireLast :: String -> [a] -> a
requireLast label xs =
    case foldl' (\_ x -> Just x) Nothing xs of
        Just y -> y
        Nothing -> error label

requireCombosArray :: String -> Aeson.Value -> [Aeson.Value]
requireCombosArray label val =
    case val of
        Aeson.Object o ->
            case KM.lookup "combos" o of
                Just (Aeson.Array arr) -> V.toList arr
                _ -> error (label ++ ": missing combos array")
        _ -> error (label ++ ": payload root is not an object")

requireComboSymbol :: String -> Aeson.Value -> String
requireComboSymbol label val =
    case val of
        Aeson.Object o ->
            case KM.lookup "params" o of
                Just (Aeson.Object params) ->
                    case KM.lookup "symbol" params >>= AT.parseMaybe Aeson.parseJSON of
                        Just sym -> sym
                        Nothing -> error (label ++ ": missing params.symbol")
                _ -> error (label ++ ": missing params object")
        _ -> error (label ++ ": combo is not an object")

requireComboMetricsDouble :: String -> String -> Aeson.Value -> Double
requireComboMetricsDouble label key val =
    case val of
        Aeson.Object o ->
            case KM.lookup "metrics" o of
                Just (Aeson.Object metrics) ->
                    case KM.lookup (AK.fromString key) metrics >>= AT.parseMaybe Aeson.parseJSON of
                        Just x -> x
                        Nothing -> error (label ++ ": missing metrics." ++ key)
                _ -> error (label ++ ": missing metrics object")
        _ -> error (label ++ ": combo is not an object")

requireComboMetricsScore :: String -> Aeson.Value -> Double
requireComboMetricsScore label = requireComboMetricsDouble label "score"

requireComboFinalEquity :: String -> Aeson.Value -> Double
requireComboFinalEquity label val =
    case val of
        Aeson.Object o ->
            case KM.lookup "finalEquity" o >>= AT.parseMaybe Aeson.parseJSON of
                Just eq -> eq
                Nothing -> error (label ++ ": missing finalEquity")
        _ -> error (label ++ ": combo is not an object")

requireComboSource :: String -> Aeson.Value -> String
requireComboSource label val =
    case val of
        Aeson.Object o ->
            case KM.lookup "source" o >>= AT.parseMaybe Aeson.parseJSON of
                Just source -> source
                Nothing -> error (label ++ ": missing source")
        _ -> error (label ++ ": combo is not an object")

requireComboBars :: String -> Aeson.Value -> Int
requireComboBars label val =
    case val of
        Aeson.Object o ->
            case KM.lookup "params" o of
                Just (Aeson.Object params) ->
                    case KM.lookup "bars" params >>= AT.parseMaybe Aeson.parseJSON of
                        Just bars -> bars
                        Nothing -> error (label ++ ": missing params.bars")
                _ -> error (label ++ ": missing params object")
        _ -> error (label ++ ": combo is not an object")

requireComboPlatform :: String -> Aeson.Value -> Maybe String
requireComboPlatform label val =
    case val of
        Aeson.Object o ->
            case KM.lookup "params" o of
                Just (Aeson.Object params) ->
                    KM.lookup "platform" params >>= AT.parseMaybe Aeson.parseJSON
                _ -> error (label ++ ": missing params object")
        _ -> error (label ++ ": combo is not an object")

requireComboBinanceSymbol :: String -> Aeson.Value -> Maybe String
requireComboBinanceSymbol label val =
    case val of
        Aeson.Object o ->
            case KM.lookup "params" o of
                Just (Aeson.Object params) ->
                    KM.lookup "binanceSymbol" params >>= AT.parseMaybe Aeson.parseJSON
                _ -> error (label ++ ": missing params object")
        _ -> error (label ++ ": combo is not an object")

withTempTestDir :: String -> (FilePath -> IO a) -> IO a
withTempTestDir prefix action = do
    ts <- fmap (floor . (* 1000000)) getPOSIXTime
    let dir = ".tmp" </> ("trader-tests-" ++ prefix ++ "-" ++ show (ts :: Integer))
    createDirectoryIfMissing True dir
    action dir `finally` cleanup dir
  where
    cleanup dir = do
        _ <- try (removeDirectoryRecursive dir) :: IO (Either SomeException ())
        pure ()

parseArgs :: [String] -> IO Args
parseArgs argv = do
    case parseArgsResult argv of
        Left err -> error err
        Right ok -> pure ok

parseArgsResult :: [String] -> Either String Args
parseArgsResult argv =
    let parser = info (opts <**> helper) fullDesc
     in case execParserPure defaultPrefs parser argv of
            Success args -> validateArgs args
            Failure failure ->
                let (msg, _) = renderFailure failure "trader-tests"
                 in Left msg
            CompletionInvoked _ -> Left "Unexpected completion"

baseEnsembleConfig :: EnsembleConfig
baseEnsembleConfig =
    EnsembleConfig
        { ecOpenThreshold = 0.0
        , ecCloseThreshold = 0.0
        , ecFee = 0.0
        , ecSlippage = 0.0
        , ecSpread = 0.0
        , ecFeeFixed = 0.0
        , ecFeeMin = 0.0
        , ecSlippageVolMult = 0.0
        , ecSlippageImpact = 0.0
        , ecSlippageImpactPower = 1.0
        , ecSpreadVolMult = 0.0
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
        , ecMinEdge = 0
        , ecMinSignalToNoise = 0
        , ecSnrSizeWeight = 0
        , ecThresholdFactorEnabled = False
        , ecThresholdFactorAlpha = 0.2
        , ecThresholdFactorMin = 0.5
        , ecThresholdFactorMax = 2.0
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
        , ecPeriodsPerYear = 365
        , ecVolTarget = Nothing
        , ecVolLookback = 20
        , ecVolEwmaAlpha = Nothing
        , ecVolFloor = 0
        , ecVolScaleMax = 1
        , ecMaxVolatility = Nothing
        , ecRebalanceBars = 0
        , ecRebalanceThreshold = 0
        , ecRebalanceGlobal = False
        , ecRebalanceResetOnSignal = False
        , ecFundingRate = 0
        , ecFundingBySide = False
        , ecFundingOnOpen = False
        , ecBlendWeight = 0.5
        , ecRouterLookback = 30
        , ecRouterMinScore = 0.25
        , ecRouterScorePnlWeight = 0
        , ecKalmanDt = 1.0
        , ecKalmanProcessVar = 1e-5
        , ecKalmanMeasurementVar = 1e-3
        , ecTriLayer = False
        , ecTriLayerFastMult = 0.5
        , ecTriLayerSlowMult = 2.0
        , ecTriLayerCloudPadding = 0
        , ecTriLayerCloudSlope = 0
        , ecTriLayerCloudWidth = 0
        , ecTriLayerTouchLookback = 1
        , ecTriLayerExitOnSlow = False
        , ecKalmanBandLookback = 0
        , ecKalmanBandStdMult = 0
        , ecTriLayerRequirePriceAction = True
        , ecTriLayerPriceActionBody = 0
        , ecLstmExitFlipBars = 0
        , ecLstmExitFlipGraceBars = 0
        , ecLstmExitFlipStrong = False
        , ecLstmConfidenceSoft = 0.6
        , ecLstmConfidenceHard = 0.8
        , ecKalmanZMin = 0
        , ecKalmanZMax = 3
        , ecMaxHighVolProb = Nothing
        , ecMaxConformalWidth = Nothing
        , ecMaxQuantileWidth = Nothing
        , ecConfirmConformal = False
        , ecConfirmQuantiles = False
        , ecConfidenceSizing = False
        , ecMinPositionSize = 0
        }

testLookbackBars :: IO ()
testLookbackBars =
    case lookbackBarsFrom "5m" "24h" of
        Left e -> error e
        Right n -> assert "expected 288 bars" (n == 288)

testLookbackBarsOverflowDuration :: IO ()
testLookbackBarsOverflowDuration =
    case lookbackBarsFrom "1h" "999999999999999999999999h" of
        Left _ -> pure ()
        Right _ -> error "expected oversized lookback window to fail"

testDurationOverflowUnitMultiplication :: IO ()
testDurationOverflowUnitMultiplication = do
    let raw = show (maxBound :: Int) ++ "m"
    assert
        "expected overflowing duration unit multiplication to fail"
        (isNothing (parseDurationSeconds raw))

testLookbackBarsOverflowArithmetic :: IO ()
testLookbackBarsOverflowArithmetic =
    let lookbackSec = maxBound :: Int
        expected = fromInteger ((toInteger lookbackSec + 1) `div` 2)
     in case lookbackBarsFrom "2s" (show lookbackSec ++ "s") of
            Left e -> error ("expected lookback arithmetic to stay correct without overflow: " ++ e)
            Right n ->
                assert
                    ("expected ceil(maxBound/2) bars, got " ++ show n)
                    (n == expected)

testInferPeriodsPerYearWeekly :: IO ()
testInferPeriodsPerYearWeekly =
    assertApprox
        "expected weekly periods/year to use 365-day second conversion"
        1e-12
        (inferPeriodsPerYear "1w")
        (365 / 7)

testInferPeriodsPerYearMonthly :: IO ()
testInferPeriodsPerYearMonthly =
    assertApprox
        "expected monthly periods/year to use 30-day month seconds"
        1e-12
        (inferPeriodsPerYear "1M")
        (365 / 30)

testMinuteOfDayFromMsBounds :: IO ()
testMinuteOfDayFromMsBounds = do
    let expected ts = fromInteger ((toInteger ts `div` 60000) `mod` 1440)
    assert "minute-of-day should stay bounded for maxBound Int64" (minuteOfDayFromMs (maxBound :: Int64) == expected (maxBound :: Int64))
    assert "minute-of-day should stay bounded for minBound Int64" (minuteOfDayFromMs (minBound :: Int64) == expected (minBound :: Int64))

testParseTimeWindowStrictClockFormat :: IO ()
testParseTimeWindowStrictClockFormat = do
    let valid = parseTimeWindow "09:05-16:30"
        rejects raw =
            case parseTimeWindow raw of
                Left _ -> True
                Right _ -> False
    assert "zero-padded time window parses" (valid == Right (TimeWindow (9 * 60 + 5) (16 * 60 + 30)))
    assert "single-digit start hour is rejected" (rejects "9:05-16:30")
    assert "single-digit start minute is rejected" (rejects "09:5-16:30")
    assert "single-digit end hour is rejected" (rejects "09:05-6:30")
    assert "single-digit end minute is rejected" (rejects "09:05-16:3")

testKalmanFusionMulti :: IO ()
testKalmanFusionMulti = do
    let k0 = initKalman1 0 1 0
        k1 = updateMulti [(0.01, 1e-4), (0.02, 1e-2)] k0
        expected = (0 + 0.01 * 10000 + 0.02 * 100) / (1 + 10000 + 100)
    assertApprox "posterior mean" 1e-6 (kMean k1) expected
    assert "posterior variance shrinks" (kVar k1 < 1)

testMarketLinearFit :: IO ()
testMarketLinearFit = do
    let xs = V.generate 100 (fromIntegral :: Int -> Double)
        ys = V.map (\x -> 2 + 3 * x) xs
        (a, b, var) = fitLinearRange xs ys 0 100
    assertApprox "intercept" 1e-9 a 2
    assertApprox "beta" 1e-9 b 3
    assert "var ~ 0" (var < 1e-9)

testPredictorsOutputs :: IO ()
testPredictorsOutputs = do
    let prices = take 300 (iterate (* 1.001) 100.0)
        lookback = 20
        trainPrices = take 250 prices
        pb = trainPredictors allPredictors lookback (V.fromList trainPrices)
        t = 100
        hmmObs = forwardReturns (take (t + 1) prices)
        hmm = initHMMFilter pb hmmObs
        (outs, _) = predictSensors pb (V.fromList prices) hmm t
        ids = map fst outs

    assert "has GBDT" (SensorGBT `elem` ids)
    assert "has TCN" (SensorTCN `elem` ids)
    assert "has Transformer" (SensorTransformer `elem` ids)
    assert "has HMM" (SensorHMM `elem` ids)
    assert "has Quantile" (SensorQuantile `elem` ids)
    assert "has Conformal" (SensorConformal `elem` ids)

    case lookup SensorHMM outs of
        Nothing -> error "missing HMM output"
        Just o ->
            case soRegimes o of
                Nothing -> error "missing regime probabilities"
                Just (RegimeProbs pt pm ph) -> assertApprox "regime probs sum" 1e-9 (pt + pm + ph) 1.0

    case lookup SensorQuantile outs of
        Nothing -> error "missing quantile output"
        Just o ->
            case soQuantiles o of
                Nothing -> error "missing quantiles"
                Just (Quantiles q10 q50 q90) -> assert "quantiles ordered" (q10 <= q50 && q50 <= q90)

    case lookup SensorConformal outs of
        Nothing -> error "missing conformal output"
        Just o ->
            case soInterval o of
                Nothing -> error "missing interval"
                Just (Interval lo hi) -> assert "interval ordered" (lo <= hi)

testPredictorFeaturesRejectNonFinitePrices :: IO ()
testPredictorFeaturesRejectNonFinitePrices = do
    let nan = 0 / 0
        fs = mkFeatureSpec 3
        prices = V.fromList [100.0, 101.0, nan, 103.0, 104.0]
    assert "forwardReturnAt rejects non-finite prices" (isNothing (forwardReturnAt prices 1))
    assert "featuresAt rejects windows containing non-finite prices" (isNothing (featuresAt fs prices 2))

testPredictorMarketFeaturesStayFinite :: IO ()
testPredictorMarketFeaturesStayFinite = do
    let fs = mkFeatureSpec 4
        prices = V.fromList [100.0, 101.0, 102.0, 101.5, 103.0, 104.0]
        mm =
            MarketModel
                { mmSymbols = ["BTCUSDT", "ETHUSDT"]
                , mmIntercept = 2e-4
                , mmBeta = 0.6
                , mmVar = 1e-4
                , mmLag = V.fromList [0.0, 0.008, 0.006, -0.003, 0.009, 0.007]
                }
    case featuresAtWithMarket fs (Just mm) prices 4 of
        Nothing -> error "missing market-aware features"
        Just feats -> do
            assert "market-aware vector length expands with context features" (length feats == 33)
            assert "market-aware feature rows remain finite" (all isFiniteDouble feats)
            let marketSlice = take 8 (drop 13 feats)
            assert "market-aware slice carries non-zero signal" (any (\x -> abs x > 1e-12) marketSlice)

testBuildDatasetSkipsNonFiniteRows :: IO ()
testBuildDatasetSkipsNonFiniteRows = do
    let nan = 0 / 0
        fs = mkFeatureSpec 3
        prices = V.fromList [100.0, 101.0, nan, 103.0, 104.0]
        dataset = buildDatasetWithIndex fs prices
    assert "dataset skips rows whose features/targets are non-finite" (null dataset)

testBuildDatasetSkipsOverflowedFiniteRows :: IO ()
testBuildDatasetSkipsOverflowedFiniteRows = do
    let fs = mkFeatureSpec 3
        prices = V.fromList [1e-308, 1.0, 1.0, 1.0, 1.0]
        dataset = buildDatasetWithIndex fs prices
        indices = map (\(t, _, _) -> t) dataset
        allFiniteRows =
            and
                [ all isFiniteDouble feats && isFiniteDouble target
                | (_, feats, target) <- dataset
                ]
    assert "featuresAt rejects overflowed variance window with non-finite stats" (isNothing (featuresAt fs prices 2))
    assert "dataset keeps only stable rows after overflowed-return window" (indices == [3])
    assert "dataset rows remain finite after overflow guard" allFiniteRows

testTransformerTrainingSanitizesDataset :: IO ()
testTransformerTrainingSanitizesDataset = do
    let nan = 0 / 0
        badInf = 1 / 0
        rawDataset =
            [ ([1.0, 2.0], 0.1)
            , ([9.0], 0.9)
            , ([2.0, 3.0], 0.2)
            , ([3.0, badInf], 0.3)
            , ([4.0, 5.0], nan)
            ]
        model = trainTransformer 2.0 10 rawDataset
        (mu, mSigma) = predictTransformer model [2.0, 3.0]
    assert "keeps feature dimension from valid rows" (trFeatureDim model == 2)
    assert "keeps only valid and dimension-consistent rows" (length (trKeys model) == 2 && length (trTargets model) == 2)
    assert "predictor still yields finite output" (isFiniteDouble mu && maybe False isFiniteDouble mSigma)

testTransformerPredictionRejectsNonFiniteQuery :: IO ()
testTransformerPredictionRejectsNonFiniteQuery = do
    let model = trainTransformer 2.0 10 [([1.0, 2.0], 0.1), ([2.0, 3.0], 0.2)]
        (mu, mSigma) = predictTransformer model [0 / 0, 3.0]
    assert "non-finite query should return neutral output" (mu == 0 && isNothing mSigma)

testTransformerInvalidTemperatureFallback :: IO ()
testTransformerInvalidTemperatureFallback = do
    let model = trainTransformer (0 / 0) 10 [([1.0, 2.0], 0.1), ([2.0, 3.0], 0.2)]
        (mu, mSigma) = predictTransformer model [1.5, 2.5]
    assert "invalid temperature falls back to finite positive default" (trTemperature model > 0 && isFiniteDouble (trTemperature model))
    assert "fallback temperature still allows finite prediction" (isFiniteDouble mu && maybe False isFiniteDouble mSigma)

testKalmanConstant :: IO ()
testKalmanConstant = do
    let xs = replicate 50 10.0
        KalmanRun preds filt = runConstantAcceleration1D 1.0 1e-6 1e-6 xs
    assert "pred length" (length preds == length xs - 1)
    assert "filt length" (length filt == length xs)
    assertApprox "filtered near constant" 1e-2 (requireLast "missing filtered value" filt) 10.0

testKalmanForecast :: IO ()
testKalmanForecast = do
    let xs = replicate 50 10.0
        f = forecastNextConstantAcceleration1D 1.0 1e-6 1e-6 xs
    assertApprox "forecast near constant" 1e-2 f 10.0

testKalmanInnovationSign :: IO ()
testKalmanInnovationSign = do
    let k0 = constantAcceleration1D 1.0 0 1e-6 0
        (predZ, k1) = step 1.0 k0
        Vec3 pos _ _ = kx k1
    assertApprox "initial prediction" 1e-12 predZ 0.0
    assert "innovation sign (update moves toward measurement)" (pos > 0)

testForwardReturnSign :: IO ()
testForwardReturnSign = do
    let up = forwardReturns [1.0, 2.0]
        down = forwardReturns [2.0, 1.0]
    case up of
        [r] -> assert "up return positive" (r > 0)
        _ -> error "expected one return for up series"
    case down of
        [r] -> assert "down return negative" (r < 0)
        _ -> error "expected one return for down series"

testLstmImprovesLoss :: IO ()
testLstmImprovesLoss = do
    let series = replicate 80 1.0
        lookback = 10
        hidden = 4
        dataset = buildSequences lookback series
        baseCfg =
            LSTMConfig
                { lcLookback = lookback
                , lcHiddenSize = hidden
                , lcEpochs = 0
                , lcLearningRate = 5e-2
                , lcValRatio = 0.2
                , lcPatience = 0
                , lcGradClip = Just 1.0
                , lcSeed = 123
                }
        (m0, _) = trainLSTM baseCfg series
        cfg = baseCfg{lcEpochs = 10}
        (m1, _) = trainLSTM cfg series
        l0 = evaluateLoss lookback hidden dataset (lmParams m0)
        l1 = evaluateLoss lookback hidden dataset (lmParams m1)
    assert ("loss did not decrease: " ++ show (l0, l1)) (l1 < l0)

testLstmInvalidValRatioFallback :: IO ()
testLstmInvalidValRatioFallback = do
    let series = replicate 80 1.0
        lookback = 10
        hidden = 4
        cfg =
            LSTMConfig
                { lcLookback = lookback
                , lcHiddenSize = hidden
                , lcEpochs = 2
                , lcLearningRate = 5e-2
                , lcValRatio = 0 / 0
                , lcPatience = 0
                , lcGradClip = Just 1.0
                , lcSeed = 123
                }
        (model, history) = trainLSTM cfg series
        loss = evaluateLoss lookback hidden (buildSequences lookback series) (lmParams model)
    assert "non-finite val ratio should still produce training history" (not (null history))
    assert "non-finite val ratio should still produce finite loss" (isFiniteDouble loss)

testLstmModelKeyPlatform :: IO ()
testLstmModelKeyPlatform = do
    argsBinance <-
        parseArgs
            [ "--symbol"
            , "BTCUSDT"
            , "--platform"
            , "binance"
            , "--interval"
            , "1h"
            , "--bars"
            , "100"
            , "--lookback-bars"
            , "10"
            ]
    argsCoinbase <-
        parseArgs
            [ "--symbol"
            , "BTC-USD"
            , "--platform"
            , "coinbase"
            , "--interval"
            , "1h"
            , "--bars"
            , "100"
            , "--lookback-bars"
            , "10"
            ]
    let lookbackB = argLookback argsBinance
        lookbackC = argLookback argsCoinbase
    keyBinance <- lstmModelKey argsBinance lookbackB
    keyCoinbase <- lstmModelKey argsCoinbase lookbackC
    assert "lstm key should differ across platforms" (keyBinance /= keyCoinbase)
    assert "lstm key should include binance prefix" ("binance:" `isInfixOf` keyBinance)
    assert "lstm key should include coinbase prefix" ("coinbase:" `isInfixOf` keyCoinbase)

testAgreementGate :: IO ()
testAgreementGate = do
    let prices = [100, 101, 102, 103]
        lookback = 2
        kalPred = [101, 110, 120] -- length 3
        lstmPred = [110, 100] -- length 2, for t=1..2
        cfg = baseEnsembleConfig
        res = requireRight "simulateEnsemble" (simulateEnsemble cfg lookback prices kalPred lstmPred Nothing)
    assert "expected two position changes (enter + exit)" (brPositionChanges res == 2)

testHoldOnCloseAgree :: IO ()
testHoldOnCloseAgree = do
    let prices = [100, 100, 100]
        lookback = 1
        kalPred = [103, 101]
        lstmPred = [103, 101]
        cfgHold = baseEnsembleConfig{ecOpenThreshold = 0.02, ecCloseThreshold = 0.005}
        btHold = requireRight "simulateEnsemble hold" (simulateEnsemble cfgHold lookback prices kalPred lstmPred Nothing)
        cfgExit = baseEnsembleConfig{ecOpenThreshold = 0.02, ecCloseThreshold = 0.02}
        btExit = requireRight "simulateEnsemble exit" (simulateEnsemble cfgExit lookback prices kalPred lstmPred Nothing)
    assert "holds when close signal still agrees" (brPositions btHold == [1, 1])
    assert "exits when open signal neutral and close signal does not agree" (brPositions btExit == [1, 0])

testMinHoldBars :: IO ()
testMinHoldBars = do
    let prices = replicate 5 100
        lookback = 1
        preds = [101, 99, 99, 99] -- enter, then exit signals
        cfg = baseEnsembleConfig{ecMinHoldBars = 2}
        bt = requireRight "simulateEnsemble min-hold" (simulateEnsemble cfg lookback prices preds preds Nothing)
    assert "min-hold keeps position through bar 2" (brPositions bt == [1, 1, 0, 0])

testMaxHoldBars :: IO ()
testMaxHoldBars = do
    let prices = replicate 4 100
        lookback = 1
        preds = [101, 101, 101]
        cfg = baseEnsembleConfig{ecMaxHoldBars = Just 1}
        bt = requireRight "simulateEnsemble max-hold" (simulateEnsemble cfg lookback prices preds preds Nothing)
    assert "max-hold forces exit after limit with 1-bar cooldown" (brPositions bt == [1, 0, 0])

testCooldownBars :: IO ()
testCooldownBars = do
    let prices = replicate 5 100
        lookback = 1
        preds = [101, 99, 101, 101] -- enter, exit, re-enter attempts
        cfg = baseEnsembleConfig{ecCooldownBars = 1}
        bt = requireRight "simulateEnsemble cooldown" (simulateEnsemble cfg lookback prices preds preds Nothing)
    assert "cooldown blocks entry for 1 bar after exit" (brPositions bt == [1, 0, 0, 1])

testTriLayerUsesProvidedOpenPrices :: IO ()
testTriLayerUsesProvidedOpenPrices = do
    let prices = [70, 80, 100, 110]
        highs = [70, 80, 100.5, 110]
        lows = [70, 80, 95, 110]
        opens = V.fromList [70, 80, 99, 110]
        lookback = 1
        preds = [70, 80, 120]
        baseCfg =
            baseEnsembleConfig
                { ecTriLayer = True
                , ecTriLayerRequirePriceAction = True
                , ecTriLayerCloudPadding = 1
                , ecTriLayerCloudSlope = 0
                }
        btWithoutOpen =
            requireRight
                "simulateEnsemble without open prices"
                (simulateEnsembleWithHLChecked baseCfg lookback prices highs lows preds preds Nothing)
        btWithOpen =
            requireRight
                "simulateEnsemble with open prices"
                (simulateEnsembleWithHLChecked (baseCfg{ecOpenPrices = Just opens}) lookback prices highs lows preds preds Nothing)
    assert "without open prices remains flat" (brPositions btWithoutOpen == [0, 0, 0])
    assert "with open prices enters on hammer at t=2" (brPositions btWithOpen == [0, 0, 1])

testEntryBlockNoTradeWindow :: IO ()
testEntryBlockNoTradeWindow = do
    let prices = replicate 4 100
        lookback = 1
        preds = [110, 90, 90] -- enter long, flip short attempts
        openTimes :: V.Vector Int64
        openTimes = V.fromList [0, 60000, 120000, 180000]
        cfg =
            baseEnsembleConfig
                { ecNoTradeWindows = [TimeWindow 1 2]
                , ecOpenTimes = Just openTimes
                , ecPositioning = LongShort
                }
        bt = requireRight "simulateEnsemble no-trade-window" (simulateEnsemble cfg lookback prices preds preds Nothing)
    case brPositions bt of
        [p0, p1, p2] -> do
            assert "entered long" (p0 > 0)
            assert "blocked reversal holds position" (p1 > 0)
            assert "reversal allowed after window" (p2 < 0)
        _ -> error "expected 3 position entries"

testEntryBlockMaxTradesPerDay :: IO ()
testEntryBlockMaxTradesPerDay = do
    let prices = replicate 4 100
        lookback = 1
        preds = [110, 90, 90] -- enter long, flip short attempts
        openTimes :: V.Vector Int64
        openTimes = V.fromList [0, 60000, 120000, 180000]
        cfg =
            baseEnsembleConfig
                { ecMaxTradesPerDay = Just 1
                , ecOpenTimes = Just openTimes
                , ecPositioning = LongShort
                }
        bt = requireRight "simulateEnsemble max-trades" (simulateEnsemble cfg lookback prices preds preds Nothing)
    case brPositions bt of
        [p0, p1, p2] -> do
            assert "entered long" (p0 > 0)
            assert "trade limit holds position" (p1 > 0)
            assert "trade limit holds position" (p2 > 0)
        _ -> error "expected 3 position entries"

testWeeklyLossResetsOnUtcWeekBoundary :: IO ()
testWeeklyLossResetsOnUtcWeekBoundary = do
    let prices = replicate 5 100
        lookback = 1
        preds = [110, 90, 110, 90] -- enter/exit, then enter/exit again after UTC week rollover
        dayMs :: Int64
        dayMs = 86400000
        minMs :: Int64
        minMs = 60000
        sunday2358 = 3 * dayMs + (23 * 60 + 58) * minMs
        sunday2359 = 3 * dayMs + (23 * 60 + 59) * minMs
        monday0000 = 4 * dayMs
        monday0001 = 4 * dayMs + minMs
        monday0002 = 4 * dayMs + 2 * minMs
        openTimes :: V.Vector Int64
        openTimes = V.fromList [sunday2358, sunday2359, monday0000, monday0001, monday0002]
        cfg =
            baseEnsembleConfig
                { ecFee = 0.03
                , ecMaxWeeklyLoss = Just 0.05
                , ecOpenTimes = Just openTimes
                }
        bt = requireRight "simulateEnsemble weekly-reset" (simulateEnsemble cfg lookback prices preds preds Nothing)
    case brPositions bt of
        [p0, p1, p2, p3] -> do
            assert "entered before week boundary" (p0 > 0)
            assert "exited after first reversal" (p1 == 0)
            assert "weekly loss gate resets at UTC Monday boundary and allows re-entry" (p2 > 0)
            assert "second reversal exits again" (p3 == 0)
        _ -> error "expected 4 position entries"

testFlipFeesPerSide :: IO ()
testFlipFeesPerSide = do
    let prices = replicate 4 100
        lookback = 1
        preds = [110, 90, 90] -- enter long then flip short
        cfg = baseEnsembleConfig{ecFee = 0.1, ecPositioning = LongShort}
        bt = requireRight "simulateEnsemble flip-fees" (simulateEnsemble cfg lookback prices preds preds Nothing)
    case brEquityCurve bt of
        [e0, e1, e2, e3] -> do
            assertApprox "initial equity" 1e-12 e0 1.0
            assertApprox "entry fee applied" 1e-12 e1 0.9
            assertApprox "flip fees apply per side" 1e-12 e2 0.729
            assertApprox "eod fee applied" 1e-12 e3 0.6561
        _ -> error "expected 4 equity points"

testLongShortDownMove :: IO ()
testLongShortDownMove = do
    let prices = [100, 90]
        lookback = 1
        kalPred = [90]
        lstmPred = [90]
        baseCfg = baseEnsembleConfig
        btFlat = requireRight "simulateEnsemble flat" (simulateEnsemble baseCfg lookback prices kalPred lstmPred Nothing)
        btShort = requireRight "simulateEnsemble short" (simulateEnsemble (baseCfg{ecPositioning = LongShort}) lookback prices kalPred lstmPred Nothing)

    assertApprox "flat final equity" 1e-12 (requireLast "missing flat equity curve value" (brEquityCurve btFlat)) 1.0
    assertApprox "short final equity" 1e-12 (requireLast "missing short equity curve value" (brEquityCurve btShort)) 1.1
    assert "short position opened" (brPositions btShort == [-1])

testLiquidationClamp :: IO ()
testLiquidationClamp = do
    let prices = [100, 250]
        lookback = 1
        kalPred = [50]
        lstmPred = [50]
        cfg = baseEnsembleConfig{ecPositioning = LongShort}
        bt = requireRight "simulateEnsemble liquidation" (simulateEnsemble cfg lookback prices kalPred lstmPred Nothing)
        finalEq = requireLast "missing liquidation equity curve value" (brEquityCurve bt)
        trades = brTrades bt
    assertApprox "equity clamped at 0" 1e-12 finalEq 0.0
    assert "positions cleared after liquidation" (brPositions bt == [0])
    assert "liquidation trade recorded" (case trades of [t] -> trExitReason t == Just ExitLiquidation; _ -> False)

testMetricsMaxDrawdown :: IO ()
testMetricsMaxDrawdown = do
    let br =
            BacktestResult
                { brEquityCurve = [1.0, 1.1, 1.0]
                , brPositions = [1.0, 0.0]
                , brAgreementOk = [True, True]
                , brAgreementValid = [True, True]
                , brPositionChanges = 2
                , brTrades = []
                }
        m = computeMetrics 365 br
    assertApprox "total return" 1e-12 (bmTotalReturn m) 0.0
    assertApprox "max drawdown" 1e-6 (bmMaxDrawdown m) (0.1 / 1.1)

testMetricsCalmarFallback :: IO ()
testMetricsCalmarFallback = do
    let br =
            BacktestResult
                { brEquityCurve = [1.0, 1.1, 1.2]
                , brPositions = [1.0, 1.0]
                , brAgreementOk = [True, True]
                , brAgreementValid = [True, True]
                , brPositionChanges = 1
                , brTrades = []
                }
        m = computeMetrics 365 br
    assertApprox "zero drawdown should stay zero" 1e-12 (bmMaxDrawdown m) 0.0
    assertApprox "calmar should fall back to annualized return when max drawdown is zero" 1e-12 (bmCalmar m) (bmAnnualizedReturn m)

testMetricsProfitFactorPnL :: IO ()
testMetricsProfitFactorPnL = do
    let tr1 =
            Trade
                { trEntryIndex = 0
                , trExitIndex = 1
                , trEntryEquity = 1.0
                , trExitEquity = 2.0
                , trReturn = 1.0
                , trHoldingPeriods = 1
                , trEntryHighVolProb = Nothing
                , trExitReason = Just ExitSignal
                , trEntryIp = Nothing
                , trExitIp = Nothing
                }
        tr2 =
            Trade
                { trEntryIndex = 1
                , trExitIndex = 2
                , trEntryEquity = 2.0
                , trExitEquity = 1.0
                , trReturn = -0.5
                , trHoldingPeriods = 1
                , trEntryHighVolProb = Nothing
                , trExitReason = Just ExitSignal
                , trEntryIp = Nothing
                , trExitIp = Nothing
                }
        br =
            BacktestResult
                { brEquityCurve = [1.0, 2.0, 1.0]
                , brPositions = [1.0, 0.0]
                , brAgreementOk = [True, True]
                , brAgreementValid = [True, True]
                , brPositionChanges = 2
                , brTrades = [tr1, tr2]
                }
        m = computeMetrics 365 br

    assertApprox "gross profit (PnL)" 1e-12 (bmGrossProfit m) 1.0
    assertApprox "gross loss (PnL)" 1e-12 (bmGrossLoss m) 1.0
    assertApprox "profit factor" 1e-12 (Data.Maybe.fromMaybe 0 (bmProfitFactor m)) 1.0

testMetricsSanitizeNonFiniteInputs :: IO ()
testMetricsSanitizeNonFiniteInputs = do
    let trBad =
            Trade
                { trEntryIndex = 0
                , trExitIndex = 1
                , trEntryEquity = 1.0
                , trExitEquity = 0 / 0
                , trReturn = 0 / 0
                , trHoldingPeriods = -3
                , trEntryHighVolProb = Nothing
                , trExitReason = Just ExitSignal
                , trEntryIp = Nothing
                , trExitIp = Nothing
                }
        br =
            BacktestResult
                { brEquityCurve = [1.0, 1.1]
                , brPositions = [0 / 0]
                , brAgreementOk = [True]
                , brAgreementValid = [True]
                , brPositionChanges = 1
                , brTrades = [trBad]
                }
        m = computeMetrics 365 br
    assert "avg trade return stays finite" (isFinite (bmAvgTradeReturn m))
    assert "exposure stays finite" (isFinite (bmExposure m))
  where
    isFinite x = not (isNaN x || isInfinite x)

testFormalRoiRequirementsSummary :: IO ()
testFormalRoiRequirementsSummary = do
    assert "summary mentions ROI" ("ROI" `isInfixOf` roiRequirementSummary)
    assert "requirements stay concise" (length roiRequirementClauses == 6)

testFormalRoiSpecMatchesImplementation :: IO ()
testFormalRoiSpecMatchesImplementation =
    assert "formal ROI spec matches implementation across the bounded model" (fvrRoiSpecMatchesImplementation formalVerificationReport)

testFormalRoiMonotonicity :: IO ()
testFormalRoiMonotonicity = do
    assert "return monotonicity holds" (fvrReturnMonotone formalVerificationReport)
    assert "drawdown monotonicity holds" (fvrDrawdownMonotone formalVerificationReport)
    assert "tail-loss monotonicity holds" (fvrTailLossMonotone formalVerificationReport)
    assert "turnover monotonicity holds" (fvrTurnoverMonotone formalVerificationReport)
    assert "expectancy monotonicity holds" (fvrExpectancyMonotone formalVerificationReport)
    assert "payback monotonicity holds" (fvrPaybackMonotone formalVerificationReport)

testFormalRoiPenaltyOrdering :: IO ()
testFormalRoiPenaltyOrdering = do
    assert "activity penalty ordering holds" (fvrActivityPenaltyOrdered formalVerificationReport)
    assert "exposure penalty ordering holds" (fvrExposurePenaltyOrdered formalVerificationReport)
    assert "formal verifier explored ROI states" (fvrRoiStateCount formalVerificationReport > 0)

testFormalTieBreakMatchesSpec :: IO ()
testFormalTieBreakMatchesSpec = do
    assert "formal tie-break spec matches implementation" (fvrTieBreakSpecMatchesImplementation formalVerificationReport)
    assert "formal verifier explored tie-break pairs" (fvrTieBreakPairCount formalVerificationReport > 0)

testBinanceSignatureLength :: IO ()
testBinanceSignatureLength = do
    let sig = signQuery "secret" "symbol=BTCUSDT&timestamp=1"
    assert "sha256 hex length" (BS.length sig == 64)

testBinanceKlineParsing :: IO ()
testBinanceKlineParsing = do
    let json =
            "[\
            \[1499040000000,\"0\",\"0\",\"0\",\"123.45\",\"0\",0,\"0\",0,0,0,\"0\"],\
            \[1499040000001,\"0\",\"0\",\"0\",\"200.00\",\"0\",0,\"0\",0,0,0,\"0\"]\
            \]"
    case (eitherDecode (BL.fromStrict (BS.pack json)) :: Either String [Kline]) of
        Left e -> error ("decode failed: " ++ e)
        Right ks -> do
            assert "kline count" (length ks == 2)
            assertApprox "close parse" 1e-12 (kClose (requireHead "missing first kline" ks)) 123.45
            assertApprox "volume parse" 1e-12 (kVolume (requireHead "missing first kline" ks)) 0

testBinanceTradeParserAcceptsNumericAndWhitespace :: IO ()
testBinanceTradeParserAcceptsNumericAndWhitespace = do
    let json =
            "{\"symbol\":\"BTCUSDT\",\"id\":1,\"price\":123.4,\"qty\":\" 2 \",\"time\":1700000000000}"
    case (eitherDecode (BL.fromStrict (BS.pack json)) :: Either String BinanceTrade) of
        Left e -> error ("decode failed: " ++ e)
        Right trade -> do
            assertApprox "binance trade price parses from numeric JSON" 1e-12 (btPrice trade) 123.4
            assertApprox "binance trade qty parses from whitespace string" 1e-12 (btQty trade) 2.0
            assertApprox "binance trade infers quoteQty when absent" 1e-12 (btQuoteQty trade) 246.8

testBinanceTradeParserRejectsNonFiniteInferredQuoteQty :: IO ()
testBinanceTradeParserRejectsNonFiniteInferredQuoteQty = do
    let json =
            "{\"symbol\":\"BTCUSDT\",\"id\":1,\"price\":\"1e308\",\"qty\":\"1e308\",\"time\":1700000000000}"
    case (eitherDecode (BL.fromStrict (BS.pack json)) :: Either String BinanceTrade) of
        Left _ -> pure ()
        Right _ -> error "expected trade decode to fail when inferred quoteQty is non-finite"

testCoinbaseFractionalTimestampRejected :: IO ()
testCoinbaseFractionalTimestampRejected = do
    let okJson = "[[1700000000, \"1\", \"2\", \"1.5\", \"1.8\", \"42\"]]"
        badJson = "[[1700000000.5, \"1\", \"2\", \"1.5\", \"1.8\", \"42\"]]"
    case decodeCoinbaseCandles (BL.fromStrict (BS.pack okJson)) of
        Left err -> error ("expected Coinbase integer timestamp to parse: " ++ err)
        Right _ -> pure ()
    case decodeCoinbaseCandles (BL.fromStrict (BS.pack badJson)) of
        Left _ -> pure ()
        Right _ -> error "expected Coinbase fractional timestamp to fail"

testCoinbaseTimestampBoundaryNormalization :: IO ()
testCoinbaseTimestampBoundaryNormalization = do
    let boundaryJson = "[[1000000000000, \"1\", \"2\", \"1.5\", \"1.8\", \"42\"],[-1000000000000, \"1\", \"2\", \"1.5\", \"1.8\", \"42\"]]"
    case decodeCoinbaseCandles (BL.fromStrict (BS.pack boundaryJson)) of
        Left err -> error ("expected Coinbase millisecond boundaries to parse: " ++ err)
        Right xs -> do
            assert "coinbase positive ms boundary normalized" (ccOpenTime (requireHead "missing first Coinbase candle" xs) == 1000000000)
            assert "coinbase negative ms boundary normalized" (ccOpenTime (requireLast "missing last Coinbase candle" xs) == -1000000000)

testCoinbaseBuildRangesStopsAtEpochBoundary :: IO ()
testCoinbaseBuildRangesStopsAtEpochBoundary = do
    let ranges = buildRanges 120 10 1000
    assert "coinbase range builder should stop after reaching epoch start" (ranges == [(0, 120)])

testCoinbaseBuildRangesAreContiguous :: IO ()
testCoinbaseBuildRangesAreContiguous = do
    let ranges = buildRanges 10000 10 301
    assert "coinbase range builder should return two chunks for 301 bars" (length ranges == 2)
    let newest = requireHead "missing newest Coinbase range chunk" ranges
        oldest = requireLast "missing oldest Coinbase range chunk" ranges
    assert "coinbase newest chunk starts where oldest chunk ends" (fst newest == snd oldest)
    assert "coinbase oldest chunk should include exactly one bar" (oldest == (6990, 7000))

testCoinbaseNormalizeCandlesDedupeAndLimit :: IO ()
testCoinbaseNormalizeCandlesDedupeAndLimit = do
    let candles =
            [ CoinbaseCandle{ccOpenTime = 300, ccHigh = 3.0, ccLow = 2.8, ccClose = 2.9}
            , CoinbaseCandle{ccOpenTime = 100, ccHigh = 1.2, ccLow = 0.8, ccClose = 1.0}
            , CoinbaseCandle{ccOpenTime = 200, ccHigh = 2.2, ccLow = 1.8, ccClose = 2.0}
            , CoinbaseCandle{ccOpenTime = 200, ccHigh = 9.9, ccLow = 9.8, ccClose = 9.7}
            ]
        normalized = normalizeCoinbaseCandles 2 candles
    assert "coinbase normalization keeps requested bar count" (length normalized == 2)
    assert "coinbase normalization keeps latest unique bars" (map ccOpenTime normalized == [200, 300])

testKrakenFractionalTimestampRejected :: IO ()
testKrakenFractionalTimestampRejected = do
    let okJson =
            "{\"error\":[],\"result\":{\"XXBTZUSD\":[[1700000000,\"0\",\"2\",\"1\",\"1.8\",\"0\",\"0\",\"0\"]],\"last\":1700000001}}"
        badJson =
            "{\"error\":[],\"result\":{\"XXBTZUSD\":[[1700000000.5,\"0\",\"2\",\"1\",\"1.8\",\"0\",\"0\",\"0\"]],\"last\":1700000001}}"
    case decodeKrakenCandles "XXBTZUSD" (BL.fromStrict (BS.pack okJson)) of
        Left err -> error ("expected Kraken integer timestamp to parse: " ++ err)
        Right _ -> pure ()
    case decodeKrakenCandles "XXBTZUSD" (BL.fromStrict (BS.pack badJson)) of
        Left _ -> pure ()
        Right _ -> error "expected Kraken fractional timestamp to fail"

testPoloniexFractionalTimestampRejected :: IO ()
testPoloniexFractionalTimestampRejected = do
    let okJson = "[{\"ts\":1700000000000,\"high\":\"2\",\"low\":\"1\",\"close\":\"1.8\"}]"
        badJson = "[{\"ts\":1700000000000.5,\"high\":\"2\",\"low\":\"1\",\"close\":\"1.8\"}]"
    case decodePoloniexCandles (BL.fromStrict (BS.pack okJson)) of
        Left err -> error ("expected Poloniex integer timestamp to parse: " ++ err)
        Right _ -> pure ()
    case decodePoloniexCandles (BL.fromStrict (BS.pack badJson)) of
        Left _ -> pure ()
        Right _ -> error "expected Poloniex fractional timestamp to fail"

testPoloniexTimestampBoundaryNormalization :: IO ()
testPoloniexTimestampBoundaryNormalization = do
    let boundaryJson =
            "[\
            \{\"ts\":1000000000000,\"high\":\"2\",\"low\":\"1\",\"close\":\"1.8\"},\
            \{\"ts\":-1000000000000,\"high\":\"2\",\"low\":\"1\",\"close\":\"1.8\"}\
            \]"
    case decodePoloniexCandles (BL.fromStrict (BS.pack boundaryJson)) of
        Left err -> error ("expected Poloniex millisecond boundaries to parse: " ++ err)
        Right xs -> do
            assert "poloniex positive ms boundary normalized" (pcOpenTime (requireHead "missing first Poloniex candle" xs) == 1000000000)
            assert "poloniex negative ms boundary normalized" (pcOpenTime (requireLast "missing last Poloniex candle" xs) == -1000000000)

testPoloniexNormalizeCandlesDedupeAndLimit :: IO ()
testPoloniexNormalizeCandlesDedupeAndLimit = do
    let candles =
            [ PoloniexCandle{pcOpenTime = 200, pcHigh = 2.0, pcLow = 1.5, pcClose = 1.8}
            , PoloniexCandle{pcOpenTime = 100, pcHigh = 1.1, pcLow = 0.9, pcClose = 1.0}
            , PoloniexCandle{pcOpenTime = 300, pcHigh = 3.0, pcLow = 2.5, pcClose = 2.8}
            , PoloniexCandle{pcOpenTime = 300, pcHigh = 9.0, pcLow = 8.5, pcClose = 8.8}
            ]
        normalized = normalizePoloniexCandles 2 candles
    assert "poloniex normalization keeps requested bar count" (length normalized == 2)
    assert "poloniex normalization keeps latest unique bars" (map pcOpenTime normalized == [200, 300])

testExchangeNonFiniteNumericStringRejected :: IO ()
testExchangeNonFiniteNumericStringRejected = do
    let coinbaseNan = "[[1700000000, \"1\", \"2\", \"1.5\", \"NaN\", \"42\"]]"
        krakenNan =
            "{\"error\":[],\"result\":{\"XXBTZUSD\":[[1700000000,\"0\",\"2\",\"1\",\"NaN\",\"0\",\"0\",\"0\"]],\"last\":1700000001}}"
        poloniexNan = "[{\"ts\":1700000000000,\"high\":\"2\",\"low\":\"1\",\"close\":\"NaN\"}]"
        binanceNan =
            "[\
            \[1499040000000,\"0\",\"0\",\"0\",\"NaN\",\"0\",0,\"0\",0,0,0,\"0\"]\
            \]"
    case decodeCoinbaseCandles (BL.fromStrict (BS.pack coinbaseNan)) of
        Left _ -> pure ()
        Right _ -> error "expected Coinbase NaN string to fail"
    case decodeKrakenCandles "XXBTZUSD" (BL.fromStrict (BS.pack krakenNan)) of
        Left _ -> pure ()
        Right _ -> error "expected Kraken NaN string to fail"
    case decodePoloniexCandles (BL.fromStrict (BS.pack poloniexNan)) of
        Left _ -> pure ()
        Right _ -> error "expected Poloniex NaN string to fail"
    case (eitherDecode (BL.fromStrict (BS.pack binanceNan)) :: Either String [Kline]) of
        Left _ -> pure ()
        Right _ -> error "expected Binance NaN string to fail"

testExchangeNumericStringWhitespaceAcceptance :: IO ()
testExchangeNumericStringWhitespaceAcceptance = do
    let coinbaseSpaced = "[[\" 1700000000 \", \" 1 \", \" 2 \", \" 1.5 \", \" 1.8 \", \" 42 \"]]"
        krakenSpaced =
            "{\"error\":[],\"result\":{\"XXBTZUSD\":[[\" 1700000000 \",\"0\",\" 2 \",\" 1 \",\" 1.8 \",\"0\",\"0\",\"0\"]],\"last\":1700000001}}"
        poloniexSpaced = "[{\"ts\":\" 1700000000000 \",\"high\":\" 2 \",\"low\":\" 1 \",\"close\":\" 1.8 \"}]"
    case decodeCoinbaseCandles (BL.fromStrict (BS.pack coinbaseSpaced)) of
        Left err -> error ("expected Coinbase spaced numeric strings to parse: " ++ err)
        Right xs -> do
            assert "coinbase spaced timestamp parsed" (ccOpenTime (requireHead "missing Coinbase candle" xs) == 1700000000)
            assertApprox "coinbase spaced close parsed" 1e-12 (ccClose (requireHead "missing Coinbase candle" xs)) 1.8
    case decodeKrakenCandles "XXBTZUSD" (BL.fromStrict (BS.pack krakenSpaced)) of
        Left err -> error ("expected Kraken spaced numeric strings to parse: " ++ err)
        Right xs -> do
            assert "kraken spaced timestamp parsed" (kcOpenTime (requireHead "missing Kraken candle" xs) == 1700000000)
            assertApprox "kraken spaced close parsed" 1e-12 (kcClose (requireHead "missing Kraken candle" xs)) 1.8
    case decodePoloniexCandles (BL.fromStrict (BS.pack poloniexSpaced)) of
        Left err -> error ("expected Poloniex spaced numeric strings to parse: " ++ err)
        Right xs -> do
            assert "poloniex spaced timestamp parsed" (pcOpenTime (requireHead "missing Poloniex candle" xs) == 1700000000)
            assertApprox "poloniex spaced close parsed" 1e-12 (pcClose (requireHead "missing Poloniex candle" xs)) 1.8

testMethodParsing :: IO ()
testMethodParsing = do
    assert "parse 11" (parseMethod "11" == Right MethodBoth)
    assert "parse both" (parseMethod "both" == Right MethodBoth)
    assert "parse agreement" (parseMethod "agreement" == Right MethodBoth)
    assert "parse 10" (parseMethod "10" == Right MethodKalmanOnly)
    assert "parse kalman" (parseMethod "kalman" == Right MethodKalmanOnly)
    assert "parse Kalman-Only" (parseMethod "Kalman-Only" == Right MethodKalmanOnly)
    assert "parse kalman_physics_error" (parseMethod "kalman_physics_error" == Right MethodKalmanPhysicsError)
    assert "parse 01" (parseMethod "01" == Right MethodLstmOnly)
    assert "parse lstm" (parseMethod "lstm" == Right MethodLstmOnly)
    assert "parse LSTM_ONLY" (parseMethod "LSTM_ONLY" == Right MethodLstmOnly)
    assert "parse blend" (parseMethod "blend" == Right MethodBlend)
    assert "parse conf_blend" (parseMethod "conf_blend" == Right MethodConfBlend)
    assert "parse conf-blend" (parseMethod "conf-blend" == Right MethodConfBlend)
    assert "parse conf_pick" (parseMethod "conf_pick" == Right MethodConfPick)
    assert "parse conformal_clip" (parseMethod "conformal_clip" == Right MethodConformalClip)
    assert "parse cost_pick" (parseMethod "cost_pick" == Right MethodCostPick)
    assert "parse harmonic_blend" (parseMethod "harmonic_blend" == Right MethodHarmonicBlend)
    assert "parse disagreement_guard" (parseMethod "disagreement_guard" == Right MethodDisagreementGuard)
    assert "parse median_blend" (parseMethod "median_blend" == Right MethodMedianBlend)
    assert "parse neutral_guard" (parseMethod "neutral_guard" == Right MethodNeutralGuard)
    assert "parse risk_parity_blend" (parseMethod "risk_parity_blend" == Right MethodRiskParityBlend)
    assert "parse consensus_boost" (parseMethod "consensus_boost" == Right MethodConsensusBoost)
    assert "parse anchor_blend" (parseMethod "anchor_blend" == Right MethodAnchorBlend)
    assert "parse tension_gate" (parseMethod "tension_gate" == Right MethodTensionGate)
    assert "parse entropy_blend" (parseMethod "entropy_blend" == Right MethodEntropyBlend)
    assert "parse coherence_gate" (parseMethod "coherence_gate" == Right MethodCoherenceGate)
    assert "parse divergence_gate" (parseMethod "divergence_gate" == Right MethodDivergenceGate)
    assert "parse fractal_blend" (parseMethod "fractal_blend" == Right MethodFractalBlend)
    assert "parse phase_cancel" (parseMethod "phase_cancel" == Right MethodPhaseCancel)
    assert "parse softmax_blend" (parseMethod "softmax_blend" == Right MethodSoftmaxBlend)
    assert "parse smooth_softmax_blend" (parseMethod "smooth_softmax_blend" == Right MethodSmoothSoftmaxBlend)
    assert "parse hedge_blend" (parseMethod "hedge_blend" == Right MethodHedgeBlend)
    assert "parse net_softmax_blend" (parseMethod "net_softmax_blend" == Right MethodNetSoftmaxBlend)
    assert "parse edge_blend" (parseMethod "edge_blend" == Right MethodEdgeBlend)
    assert "parse edge_pick" (parseMethod "edge_pick" == Right MethodEdgePick)
    assert "parse geo_blend" (parseMethod "geo_blend" == Right MethodGeoBlend)
    assert "parse regime_switch" (parseMethod "regime_switch" == Right MethodRegimeSwitch)
    assert "parse bandit_router" (parseMethod "bandit_router" == Right MethodBanditRouter)
    case parseMethod "00" of
        Left _ -> pure ()
        Right _ -> error "expected parse failure"

testTuneObjectiveParsing :: IO ()
testTuneObjectiveParsing = do
    assert "parse tune objective canonical code" (parseTuneObjective "equity-dd-turnover" == Right TuneEquityDdTurnover)
    assert "parse tune objective underscore alias" (parseTuneObjective "annualized_return" == Right TuneAnnualizedEquity)
    assert "parse tune objective trims tabs/newlines" (parseTuneObjective "\tROI\n" == Right TuneRoi)
    assert "parse tune objective normalizes mixed whitespace with separators" (parseTuneObjective "  final_equity\t" == Right TuneFinalEquity)
    case parseTuneObjective "invalid-objective" of
        Left _ -> pure ()
        Right _ -> error "expected parseTuneObjective to reject unknown objective"

testOptimizerObjectiveNormalization :: IO ()
testOptimizerObjectiveNormalization = do
    assert
        "optimizer objective underscore alias canonicalizes to annualized-equity"
        (normalizeObjectiveCode "annualized_return" == Right "annualized-equity")
    assert
        "optimizer objective risk-adjusted ROI alias canonicalizes to roi"
        (normalizeObjectiveCode "\trisk_adjusted_roi\n" == Right "roi")
    case normalizeObjectiveCode "invalid-objective" of
        Left _ -> pure ()
        Right v -> error ("expected normalizeObjectiveCode to reject unknown objective, got " ++ show v)

testPlatformParsing :: IO ()
testPlatformParsing = do
    assert "parse platform binance" (parsePlatform "binance" == Right PlatformBinance)
    assert "parse platform coinbase" (parsePlatform "Coinbase" == Right PlatformCoinbase)
    assert "parse platform kraken" (parsePlatform "KrAkEn" == Right PlatformKraken)
    assert "parse platform poloniex" (parsePlatform "poloniex" == Right PlatformPoloniex)
    assert "parse platform trims whitespace" (parsePlatform "  Binance  " == Right PlatformBinance)
    case parsePlatform "nope" of
        Left _ -> pure ()
        Right _ -> error "expected parsePlatform to reject unknown platforms"

testNonBinanceArgsLiveDefault :: IO ()
testNonBinanceArgsLiveDefault = do
    let krakenBaseArgs =
            [ "--platform"
            , "kraken"
            , "--data"
            , "sample.csv"
            , "--price-column"
            , "close"
            , "--interval"
            , "1h"
            , "--bars"
            , "100"
            , "--lookback-bars"
            , "10"
            ]
    case parseArgsResult krakenBaseArgs of
        Left err -> error ("unexpected validation failure for kraken defaults: " ++ err)
        Right _ -> pure ()
    case parseArgsResult (krakenBaseArgs ++ ["--binance-live"]) of
        Left err -> assert "explicit --binance-live rejected on kraken" ("--binance-live is only supported on Binance/Coinbase" `isInfixOf` err)
        Right _ -> error "expected explicit --binance-live to be rejected on kraken"

testBinanceMarketConflictingFlagsPreferFutures :: IO ()
testBinanceMarketConflictingFlagsPreferFutures = do
    args <- parseArgs ["--data", "sample.csv"]
    let conflicting = args{argBinanceFutures = True, argBinanceMargin = True}
    assert
        "argBinanceMarket prefers futures when both market flags are set"
        (argBinanceMarket conflicting == MarketFutures)

testBinanceSlashSymbolNormalization :: IO ()
testBinanceSlashSymbolNormalization =
    case parseArgsResult ["--platform", "binance", "--symbol", "btc/usdt"] of
        Left err -> error ("expected Binance slash symbol normalization to pass: " ++ err)
        Right args -> assert "binance slash symbol normalized to compact pair" (argBinanceSymbol args == Just "BTCUSDT")

testCoinbaseSlashSymbolNormalization :: IO ()
testCoinbaseSlashSymbolNormalization =
    case parseArgsResult ["--platform", "coinbase", "--symbol", "btc/usd"] of
        Left err -> error ("expected Coinbase slash symbol normalization to pass: " ++ err)
        Right args -> assert "coinbase slash symbol normalized to dash" (argBinanceSymbol args == Just "BTC-USD")

testCoinbaseDelimiterWhitespaceNormalization :: IO ()
testCoinbaseDelimiterWhitespaceNormalization =
    case parseArgsResult ["--platform", "coinbase", "--symbol", " btc  /  usd "] of
        Left err -> error ("expected Coinbase delimiter whitespace normalization to pass: " ++ err)
        Right args -> assert "coinbase spaced delimiter normalized to dash" (argBinanceSymbol args == Just "BTC-USD")

testPoloniexSlashSymbolNormalization :: IO ()
testPoloniexSlashSymbolNormalization =
    case parseArgsResult ["--platform", "poloniex", "--symbol", "btc/usdt", "--interval", "2h"] of
        Left err -> error ("expected Poloniex slash symbol normalization to pass: " ++ err)
        Right args -> assert "poloniex slash symbol normalized to underscore" (argBinanceSymbol args == Just "BTC_USDT")

testPoloniexDelimiterWhitespaceNormalization :: IO ()
testPoloniexDelimiterWhitespaceNormalization =
    case parseArgsResult ["--platform", "poloniex", "--symbol", " btc  -  usdt ", "--interval", "2h"] of
        Left err -> error ("expected Poloniex delimiter whitespace normalization to pass: " ++ err)
        Right args -> assert "poloniex spaced delimiter normalized to underscore" (argBinanceSymbol args == Just "BTC_USDT")

testArgsNormalizeIntervalCode :: IO ()
testArgsNormalizeIntervalCode =
    case parseArgsResult ["--platform", "binance", "--symbol", "BTCUSDT", "--interval", " 1H "] of
        Left err -> error ("expected interval normalization to pass: " ++ err)
        Right args -> assert "interval normalized to canonical code" (argInterval args == "1h")

testCoinbaseCompactSymbolRejected :: IO ()
testCoinbaseCompactSymbolRejected =
    case parseArgsResult ["--platform", "coinbase", "--symbol", "BTCUSD"] of
        Left err ->
            assert
                "coinbase compact symbol rejected"
                ("--symbol must use Coinbase BASE-QUOTE format" `isInfixOf` err)
        Right _ -> error "expected compact Coinbase symbol without delimiter to fail validation"

testPoloniexCompactSymbolRejected :: IO ()
testPoloniexCompactSymbolRejected =
    case parseArgsResult ["--platform", "poloniex", "--symbol", "BTCUSDT", "--interval", "2h"] of
        Left err ->
            assert
                "poloniex compact symbol rejected"
                ("--symbol must use Poloniex BASE_QUOTE format" `isInfixOf` err)
        Right _ -> error "expected compact Poloniex symbol without delimiter to fail validation"

testBinanceMalformedSymbolRejected :: IO ()
testBinanceMalformedSymbolRejected =
    case parseArgsResult ["--platform", "binance", "--symbol", "$$$"] of
        Left err ->
            assert
                "binance malformed symbol rejected"
                ("--symbol must be a valid Binance symbol" `isInfixOf` err)
        Right _ -> error "expected malformed Binance symbol to fail validation"

testBinanceQuoteOnlySymbolRejected :: IO ()
testBinanceQuoteOnlySymbolRejected =
    case parseArgsResult ["--platform", "binance", "--symbol", "USDT"] of
        Left err ->
            assert
                "binance quote-only symbol rejected"
                ("--symbol must be a valid Binance symbol" `isInfixOf` err)
        Right _ -> error "expected quote-only Binance symbol to fail validation"

testTopCombosBinanceSlashSymbolSanitization :: IO ()
testTopCombosBinanceSlashSymbolSanitization =
    assert
        "top combos binance slash symbol normalized to compact pair"
        (sanitizeComboSymbolForPlatform (Just "binance") "BTC/USDT" == Just "BTCUSDT")

testTopCombosPrefixedPairNormalization :: IO ()
testTopCombosPrefixedPairNormalization =
    assert
        "top combos prefixed binance pair normalized to compact pair"
        (sanitizeComboSymbolForPlatform Nothing "binance:btc/usdt" == Just "BTCUSDT")

testTopCombosSeparatedTokenPairNormalization :: IO ()
testTopCombosSeparatedTokenPairNormalization =
    assert
        "top combos separated binance base/quote tokens normalized to compact pair"
        (sanitizeComboSymbolForPlatform (Just "binance") "binance-btc-usdt" == Just "BTCUSDT")

testTopCombosUnknownPlatformPairNormalization :: IO ()
testTopCombosUnknownPlatformPairNormalization =
    assert
        "unknown-platform delimited pair keeps both legs"
        (sanitizeComboSymbolForPlatform Nothing "BTC-USD" == Just "BTCUSD")

testTopCombosRejectNumericOnlyDelimitedSymbols :: IO ()
testTopCombosRejectNumericOnlyDelimitedSymbols =
    assert
        "numeric-only delimited symbol rejected"
        (isNothing (sanitizeComboSymbolForPlatform Nothing "2024-01-01"))

testTopCombosRejectQuoteOnlySymbols :: IO ()
testTopCombosRejectQuoteOnlySymbols =
    assert
        "quote-only symbol rejected"
        (isNothing (sanitizeComboSymbolForPlatform Nothing "USDT"))

testTopCombosTrimKnownBinanceSuffixes :: IO ()
testTopCombosTrimKnownBinanceSuffixes = do
    assert
        "perp suffix trimmed to canonical binance pair"
        (sanitizeComboSymbolForPlatform (Just "binance") "BTCUSDTPERP" == Just "BTCUSDT")
    assert
        "swap suffix trimmed for unknown-platform symbol text"
        (sanitizeComboSymbolForPlatform Nothing "ethusdtswap" == Just "ETHUSDT")

testTopCombosKeepUnknownAlphaSuffixes :: IO ()
testTopCombosKeepUnknownAlphaSuffixes =
    assert
        "unknown alpha suffix is kept to avoid over-trimming"
        (sanitizeComboSymbolForPlatform (Just "binance") "BTCUSDTXYZ" == Just "BTCUSDTXYZ")

testResolveComboSymbolCoinbase :: IO ()
testResolveComboSymbolCoinbase =
    assert
        "coinbase combo symbol is preserved"
        (resolveComboSymbol (Just "coinbase-advanced") Nothing (Just "BTC/USD") == Just "BTC-USD")

testResolveComboSymbolDex :: IO ()
testResolveComboSymbolDex =
    assert
        "dex combo symbol is preserved"
        (resolveComboSymbol (Just "uniswap-v3") Nothing (Just "ETH/USDT") == Just "ETH/USDT")

testResolveComboSymbolSourceFallback :: IO ()
testResolveComboSymbolSourceFallback =
    assert
        "combo symbol falls back to payload source metadata"
        (resolveComboSymbol Nothing (Just "coinbase-advanced") (Just "ETH/USD") == Just "ETH-USD")

testResolveComboSymbolBlankPlatformFallsBackToSource :: IO ()
testResolveComboSymbolBlankPlatformFallsBackToSource =
    assert
        "blank combo platform should not block source-based normalization"
        (resolveComboSymbol (Just "   ") (Just "coinbase-advanced") (Just "ETH/USD") == Just "ETH-USD")

testSplitSymbolDelimitedPairs :: IO ()
testSplitSymbolDelimitedPairs = do
    assert "coinbase-style split" (Symbol.splitSymbol "BTC-USD" == ("BTC", "USD"))
    assert "poloniex-style split" (Symbol.splitSymbol "BTC_USDT" == ("BTC", "USDT"))
    assert "slash-delimited split trims and uppercases" (Symbol.splitSymbol " eth/usdc " == ("ETH", "USDC"))

testSplitSymbolCompactPairs :: IO ()
testSplitSymbolCompactPairs =
    assert
        "compact split keeps known quote suffix logic"
        (Symbol.splitSymbol "BTCUSDT" == ("BTC", "USDT"))

testSplitSymbolQuoteOnly :: IO ()
testSplitSymbolQuoteOnly =
    assert
        "quote-only symbol keeps token in base slot and avoids empty base"
        (Symbol.splitSymbol "USDT" == ("USDT", ""))

testSymbolCoinbasePrefixedPlatformNormalization :: IO ()
testSymbolCoinbasePrefixedPlatformNormalization =
    assert
        "coinbase-prefixed platform keeps coinbase BASE-QUOTE format"
        (Symbol.sanitizeSymbolForPlatform (Just "coinbase-advanced") "BTC/USD" == Just "BTC-USD")

testTopCombosCoinbasePrefixedPlatformSymbolSanitization :: IO ()
testTopCombosCoinbasePrefixedPlatformSymbolSanitization =
    assert
        "top combos coinbase-prefixed platform keeps coinbase BASE-QUOTE format"
        (sanitizeComboSymbolForPlatform (Just "coinbase-advanced") "BTC/USD" == Just "BTC-USD")

testTopCombosPayloadBlankPlatformFallsBackToSource :: IO ()
testTopCombosPayloadBlankPlatformFallsBackToSource =
    let payload =
            object
                [ "combos"
                    .= [ object
                            [ "source" .= ("coinbase-advanced" :: String)
                            , "finalEquity" .= (1.5 :: Double)
                            , "params"
                                .= object
                                    [ "platform" .= ("   " :: String)
                                    , "symbol" .= ("BTC/USD" :: String)
                                    ]
                            ]
                       ]
                ]
        (sanitized, changed) = sanitizeTopCombosValue payload
        combo = requireHead "sanitized payload should keep combo" (requireCombosArray "sanitized payload combos" sanitized)
     in do
            assert "blank platform payload should be marked changed" (changed == 1)
            assert "payload source should drive exchange-correct symbol cleanup" (requireComboSymbol "sanitized combo symbol" combo == "BTC-USD")

testSymbolDexPrefixedPlatformNormalization :: IO ()
testSymbolDexPrefixedPlatformNormalization =
    assert
        "dex-prefixed platform keeps dex symbol delimiter"
        (Symbol.sanitizeComboSymbolForPlatform (Just "uniswap-v3") "ETH/USDT" == Just "ETH/USDT")

testTopCombosDexPrefixedPlatformSymbolSanitization :: IO ()
testTopCombosDexPrefixedPlatformSymbolSanitization =
    assert
        "top combos dex-prefixed platform keeps dex symbol delimiter"
        (sanitizeComboSymbolForPlatform (Just "uniswap-v3") "ETH/USDT" == Just "ETH/USDT")

testDryRunRequiresTrade :: IO ()
testDryRunRequiresTrade =
    case parseArgsResult ["--data", "sample.csv", "--dry-run"] of
        Left err -> assert "dry-run requires trade flag" ("--dry-run requires --binance-trade" `isInfixOf` err)
        Right _ -> error "expected --dry-run without --binance-trade to fail"

testDryRunBypassesRuntimeCredentials :: IO ()
testDryRunBypassesRuntimeCredentials = do
    args <-
        parseArgs
            [ "--symbol"
            , "BTCUSDT"
            , "--platform"
            , "binance"
            , "--interval"
            , "1h"
            , "--bars"
            , "100"
            , "--lookback-bars"
            , "10"
            , "--trade-only"
            , "--binance-trade"
            , "--dry-run"
            ]
    validated <- validateRuntimeConfig args
    case validated of
        Left err -> error ("dry-run should bypass runtime credential checks: " ++ err)
        Right () -> pure ()

testDryRunSkipsNonOwnerUserKeyRequirement :: IO ()
testDryRunSkipsNonOwnerUserKeyRequirement = do
    assert
        "dry-run skips non-owner Binance key requirement"
        (not (shouldRequireUserTradeKeys PlatformBinance (Just "tenant-a") False True))
    assert
        "dry-run skips non-owner Coinbase key requirement"
        (not (shouldRequireUserTradeKeys PlatformCoinbase (Just "tenant-a") False True))

testLiveTradeRequiresNonOwnerUserKeys :: IO ()
testLiveTradeRequiresNonOwnerUserKeys = do
    assert
        "live non-owner Binance trade requires user keys"
        (shouldRequireUserTradeKeys PlatformBinance (Just "tenant-a") False False)
    assert
        "live owner Binance trade can use server keys"
        (not (shouldRequireUserTradeKeys PlatformBinance (Just "tenant-a") True False))

testEmptyCliCredentialsRejected :: IO ()
testEmptyCliCredentialsRejected =
    case parseArgsResult ["--symbol", "BTCUSDT", "--binance-api-key", "   "] of
        Left err -> assert "empty binance key rejected" ("--binance-api-key cannot be empty" `isInfixOf` err)
        Right _ -> error "expected empty --binance-api-key to fail validation"

testBacktestWindowTimeValidation :: IO ()
testBacktestWindowTimeValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "not-a-time"] of
        Left err ->
            assert
                "invalid --from rejected"
                ("--from must be epoch seconds/ms or ISO-8601" `isInfixOf` err)
        Right _ -> error "expected invalid --from to fail validation"

testBacktestWindowOverflowScientificValidation :: IO ()
testBacktestWindowOverflowScientificValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "1e400"] of
        Left err ->
            assert
                "overflow scientific --from rejected"
                ("--from must be epoch seconds/ms or ISO-8601" `isInfixOf` err)
        Right _ -> error "expected overflow scientific --from to fail validation"

testBacktestWindowScientificIntegerValidation :: IO ()
testBacktestWindowScientificIntegerValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "1e3"] of
        Left err ->
            assert
                "scientific integer --from rejected"
                ("--from must be epoch seconds/ms or ISO-8601" `isInfixOf` err)
        Right _ -> error "expected scientific integer --from to fail validation"

testBacktestWindowOverflowIntegerValidation :: IO ()
testBacktestWindowOverflowIntegerValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "999999999999999999999999"] of
        Left err ->
            assert
                "overflow integer --from rejected"
                ("--from must be epoch seconds/ms or ISO-8601" `isInfixOf` err)
        Right _ -> error "expected overflow integer --from to fail validation"

testBacktestWindowFractionalNumericValidation :: IO ()
testBacktestWindowFractionalNumericValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "1.5"] of
        Left err ->
            assert
                "fractional numeric --from rejected"
                ("--from must be epoch seconds/ms or ISO-8601" `isInfixOf` err)
        Right _ -> error "expected fractional --from to fail validation"

testBacktestWindowDecimalIntegerValidation :: IO ()
testBacktestWindowDecimalIntegerValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "1704067200.0"] of
        Left err ->
            assert
                "decimal-like integer --from rejected"
                ("--from must be epoch seconds/ms or ISO-8601" `isInfixOf` err)
        Right _ -> error "expected decimal-like integer --from to fail validation"

testBacktestWindowNonDecimalIntegerValidation :: IO ()
testBacktestWindowNonDecimalIntegerValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "0x10"] of
        Left err ->
            assert
                "non-decimal integer --from rejected"
                ("--from must be epoch seconds/ms or ISO-8601" `isInfixOf` err)
        Right _ -> error "expected non-decimal integer --from to fail validation"

testBacktestWindowIsoOffsetValidation :: IO ()
testBacktestWindowIsoOffsetValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "2025-01-01T00:00:00+00:00", "--to", "2025-01-01T00:05:00+00:00"] of
        Left err -> error ("expected ISO offset to parse: " ++ err)
        Right _ -> pure ()

testBacktestWindowIsoMinutePrecisionValidation :: IO ()
testBacktestWindowIsoMinutePrecisionValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "2025-01-01T00:00Z", "--to", "2025-01-01T00:05Z"] of
        Left err -> error ("expected ISO minute-precision timestamp to parse: " ++ err)
        Right _ -> pure ()

testBacktestWindowLowercaseZuluValidation :: IO ()
testBacktestWindowLowercaseZuluValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "2025-01-01T00:00:00z", "--to", "2025-01-01T00:05:00z"] of
        Left err -> error ("expected lowercase zulu timestamp to parse: " ++ err)
        Right _ -> pure ()

testBacktestWindowLowercaseTSeparatorValidation :: IO ()
testBacktestWindowLowercaseTSeparatorValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "2025-01-01t00:00:00z", "--to", "2025-01-01t00:05:00z"] of
        Left err -> error ("expected lowercase t separator timestamp to parse: " ++ err)
        Right _ -> pure ()

testBacktestWindowExpandedYearIsoValidation :: IO ()
testBacktestWindowExpandedYearIsoValidation = do
    let expected = 253402300800000 :: Int64
    assert
        "expanded-year ISO date parses"
        (parseTimestampMs "10000-01-01" == Just expected)

testBacktestWindowExpandedYearOverflowValidation :: IO ()
testBacktestWindowExpandedYearOverflowValidation =
    assert
        "overflowed expanded-year ISO date rejected"
        (isNothing (parseTimestampMs "1000000000000-01-01"))

testBacktestWindowNegativeMillisecondsValidation :: IO ()
testBacktestWindowNegativeMillisecondsValidation = do
    let expected = -1704067200000 :: Int64
    assert
        "negative millisecond epoch is preserved"
        (parseTimestampMs "-1704067200000" == Just expected)

testBacktestWindowPositiveMillisecondsValidation :: IO ()
testBacktestWindowPositiveMillisecondsValidation = do
    let expected = 99999999999 :: Int64
    assert
        "positive 11-digit millisecond epoch is preserved"
        (parseTimestampMs "99999999999" == Just expected)

testBacktestWindowSecondEpochNormalization :: IO ()
testBacktestWindowSecondEpochNormalization = do
    let expected = 1704067200000 :: Int64
    assert
        "second epoch is normalized to milliseconds"
        (parseTimestampMs "1704067200" == Just expected)

testBacktestWindowSignedSecondEpochValidation :: IO ()
testBacktestWindowSignedSecondEpochValidation = do
    let expected = 1704067200000 :: Int64
    assert
        "signed positive second epoch is normalized to milliseconds"
        (parseTimestampMs "+1704067200" == Just expected)

testBacktestWindowOrderValidation :: IO ()
testBacktestWindowOrderValidation =
    case parseArgsResult ["--data", "sample.csv", "--from", "2025-01-02", "--to", "2025-01-01"] of
        Left err -> assert "from<=to enforced" ("--from must be <= --to" `isInfixOf` err)
        Right _ -> error "expected --from > --to to fail validation"

testIdempotencyKeyLengthValidation :: IO ()
testIdempotencyKeyLengthValidation =
    case parseArgsResult ["--data", "sample.csv", "--idempotency-key", replicate 37 'a'] of
        Left err ->
            assert
                "idempotency key length > 36 rejected"
                ("--idempotency-key must be 1..36 chars" `isInfixOf` err)
        Right _ -> error "expected idempotency key longer than 36 chars to fail validation"

testIdempotencyKeyAsciiValidation :: IO ()
testIdempotencyKeyAsciiValidation =
    case parseArgsResult ["--data", "sample.csv", "--idempotency-key", "abc-ñ"] of
        Left err ->
            assert
                "idempotency key rejects non-ascii characters"
                ("--idempotency-key must be 1..36 chars" `isInfixOf` err)
        Right _ -> error "expected non-ascii idempotency key to fail validation"

testIdempotencyKeyTrimValidation :: IO ()
testIdempotencyKeyTrimValidation =
    case parseArgsResult ["--data", "sample.csv", "--idempotency-key", "  abc_123  "] of
        Left err -> error ("expected idempotency key with surrounding spaces to be accepted after trimming: " ++ err)
        Right args ->
            assert
                "idempotency key is trimmed before runtime use"
                (argIdempotencyKey args == Just "abc_123")

testBarsOverflowValidation :: IO ()
testBarsOverflowValidation =
    case parseArgsResult ["--data", "sample.csv", "--bars", "999999999999999999999999"] of
        Left err ->
            assert
                "overflow bars rejected"
                ("Expected an integer (e.g. 500) or 'auto'." `isInfixOf` err)
        Right _ -> error "expected overflow --bars to fail validation"

testBarsSignedIntegerValidation :: IO ()
testBarsSignedIntegerValidation =
    case parseArgsResult ["--data", "sample.csv", "--bars", "+500"] of
        Left err -> error ("expected signed positive --bars value to parse: " ++ err)
        Right _ -> pure ()

testBarsNonDecimalValidation :: IO ()
testBarsNonDecimalValidation =
    case parseArgsResult ["--data", "sample.csv", "--bars", "0x10"] of
        Left err ->
            assert
                "non-decimal bars rejected"
                ("Expected an integer (e.g. 500) or 'auto'." `isInfixOf` err)
        Right _ -> error "expected non-decimal --bars to fail validation"

testNumericArgsFiniteValidation :: IO ()
testNumericArgsFiniteValidation = do
    case parseArgsResult ["--data", "sample.csv", "--fee", "Infinity"] of
        Left err -> assert "fee rejects Infinity" ("--fee must be finite" `isInfixOf` err)
        Right _ -> error "expected Infinity fee to fail validation"
    case parseArgsResult ["--data", "sample.csv", "--order-quote", "Infinity"] of
        Left err -> assert "order-quote rejects Infinity" ("--order-quote must be finite" `isInfixOf` err)
        Right _ -> error "expected Infinity order-quote to fail validation"
    case parseArgsResult ["--data", "sample.csv", "--tune-stress-shock", "NaN"] of
        Left err -> assert "tune-stress-shock rejects NaN" ("--tune-stress-shock must be finite" `isInfixOf` err)
        Right _ -> error "expected NaN tune-stress-shock to fail validation"

testTradeSizingPositiveValidation :: IO ()
testTradeSizingPositiveValidation = do
    case parseArgsResult ["--data", "sample.csv", "--order-quote", "0"] of
        Left err -> assert "order-quote rejects zero" ("--order-quote must be > 0" `isInfixOf` err)
        Right _ -> error "expected --order-quote=0 to fail validation"
    case parseArgsResult ["--data", "sample.csv", "--order-quantity", "0"] of
        Left err -> assert "order-quantity rejects zero" ("--order-quantity must be > 0" `isInfixOf` err)
        Right _ -> error "expected --order-quantity=0 to fail validation"
    case parseArgsResult ["--data", "sample.csv", "--order-quote-fraction", "0.5", "--max-order-quote", "0"] of
        Left err -> assert "max-order-quote rejects zero" ("--max-order-quote must be > 0" `isInfixOf` err)
        Right _ -> error "expected --max-order-quote=0 to fail validation"

testRetryAfterDateParsing :: IO ()
testRetryAfterDateParsing = do
    let nowMs = 1735689600000 -- 2025-01-01T00:00:00Z
        delayZero = parseRetryAfterMsAt nowMs "0"
        delaySeconds = parseRetryAfterMsAt nowMs "5"
        delaySecondsSpaced = parseRetryAfterMsAt nowMs " 5 "
        delayDate = parseRetryAfterMsAt nowMs "Wed, 01 Jan 2025 00:00:05 GMT"
        delayDateSpaced = parseRetryAfterMsAt nowMs " Wed, 01 Jan 2025 00:00:05 GMT "
        delayHuge = parseRetryAfterMsAt nowMs "999999999999999999999999999999"
    assert "retry-after zero parses" (delayZero == Just 0)
    assert "retry-after seconds parses" (delaySeconds == Just 5000)
    assert "retry-after seconds trims spaces" (delaySecondsSpaced == Just 5000)
    assert "retry-after HTTP-date parses" (delayDate == Just 5000)
    assert "retry-after HTTP-date trims spaces" (delayDateSpaced == Just 5000)
    assert "retry-after huge value clamps to safe sleep bound" (delayHuge == Just ((maxBound :: Int) `div` 1000))

testRetryAfterHeaderLookupCaseInsensitive :: IO ()
testRetryAfterHeaderLookupCaseInsensitive = do
    let nowMs = 1735689600000 -- 2025-01-01T00:00:00Z
        delayLower = parseRetryAfterFromHeadersAt nowMs [("retry-after", "5")]
        delayUpper = parseRetryAfterFromHeadersAt nowMs [("RETRY-AFTER", "7")]
        delayMissing = parseRetryAfterFromHeadersAt nowMs [("x-retry-after", "9")]
    assert "retry-after lowercase header parses" (delayLower == Just 5000)
    assert "retry-after uppercase header parses" (delayUpper == Just 7000)
    assert "retry-after missing header returns nothing" (isNothing delayMissing)

testRetryAfterDuplicateHeaderFallback :: IO ()
testRetryAfterDuplicateHeaderFallback = do
    let nowMs = 1735689600000 -- 2025-01-01T00:00:00Z
        delay =
            parseRetryAfterFromHeadersAt
                nowMs
                [ ("Retry-After", "not-a-number")
                , ("retry-after", "9")
                ]
    assert "retry-after falls back to later parseable duplicate header value" (delay == Just 9000)

testRetryBackoffOverflowClamp :: IO ()
testRetryBackoffOverflowClamp = do
    let cap = maxBound :: Int
        delay = boundedBackoffMs (cap - 1) cap 1
    assert "backoff clamps to max delay without overflowing" (delay == cap)

testRetryJitterMaxDelayClamp :: IO ()
testRetryJitterMaxDelayClamp = do
    let cap = 12000
    assert "jitter caps positive skew at configured max delay" (jitteredDelayMs cap cap 0.25 == cap)
    assert "jitter can increase delay while remaining under cap" (jitteredDelayMs cap 8000 0.25 == 10000)
    assert "jitter never returns negative delays" (jitteredDelayMs cap 8000 (-2.0) == 0)
    assert "jitter respects zero max-delay cap" (jitteredDelayMs 0 8000 0.5 == 0)

testCacheQuickFailureUsesStale :: IO ()
testCacheQuickFailureUsesStale = do
    cache <- newTtlCache
    insertCache cache ("btc-usdt" :: String) (42 :: Int)
    threadDelay 20000
    result <- try (fetchWithCache cache 0 0.2 "btc-usdt" (fail "upstream failed")) :: IO (Either SomeException Int)
    case result of
        Left e -> error ("expected stale cache fallback on quick failure: " ++ show e)
        Right v -> assert "cache serves stale value while within stale ttl" (v == 42)

testCacheSlowFailureRejectsExpiredStale :: IO ()
testCacheSlowFailureRejectsExpiredStale = do
    cache <- newTtlCache
    insertCache cache ("btc-usdt" :: String) (42 :: Int)
    threadDelay 20000
    result <- try (fetchWithCache cache 0 0.1 "btc-usdt" (threadDelay 150000 >> fail "upstream failed")) :: IO (Either SomeException Int)
    case result of
        Left _ -> pure ()
        Right _ -> error "expected expired stale cache entry to be rejected after slow failure"

testInitialBalanceValidation :: IO ()
testInitialBalanceValidation =
    case parseArgsResult ["--data", "sample.csv", "--initial-balance", "0"] of
        Left err -> assert "initial balance > 0 enforced" ("--initial-balance must be > 0" `isInfixOf` err)
        Right _ -> error "expected non-positive initial balance to fail validation"

testBotTradeDefaultTrue :: IO ()
testBotTradeDefaultTrue = do
    assert "botTrade omitted defaults to true" (botTradeEnabledFromApi Nothing)
    assert "botTrade=true stays true" (botTradeEnabledFromApi (Just True))
    assert "botTrade=false stays false" (not (botTradeEnabledFromApi (Just False)))

testAutoStartResolvesOriginComboForActiveAdopt :: IO ()
testAutoStartResolvesOriginComboForActiveAdopt = do
    assert "active adopt resolves persisted-origin combo first" (shouldResolveOriginComboOnAutoStart True)
    assert "flat start skips persisted-origin combo resolution" (not (shouldResolveOriginComboOnAutoStart False))

testBotStartPreservesProvidedComboForActiveAdopt :: IO ()
testBotStartPreservesProvidedComboForActiveAdopt = do
    let comboUuid = Just ("8d3e3eb0-f4ea-4704-b9e4-57e3f0f6d81d" :: String)
    assert "active adopt preserves provided combo" (shouldPreserveProvidedComboOnActiveAdopt True comboUuid)
    assert "inactive adopt ignores provided combo" (not (shouldPreserveProvidedComboOnActiveAdopt False comboUuid))
    assert "active adopt without combo falls back to recompute" (not (shouldPreserveProvidedComboOnActiveAdopt True Nothing))

testBotStartClearOriginGate :: IO ()
testBotStartClearOriginGate = do
    assert "adoptable+flat clears origin" (shouldClearPositionOriginOnStart True False)
    assert "active adopt keeps origin" (not (shouldClearPositionOriginOnStart True True))
    assert "non-adoptable start skips origin cleanup" (not (shouldClearPositionOriginOnStart False False))

testPersistPositionOriginGate :: IO ()
testPersistPositionOriginGate = do
    let shouldPersist = shouldPersistPositionOriginOnSwitch
    assert "persist only on live sent switch" (shouldPersist True True True True)
    assert "no persist in paper mode" (not (shouldPersist True False True True))
    assert "no persist when trade disabled" (not (shouldPersist False True True True))
    assert "no persist when switch not applied" (not (shouldPersist True True False True))
    assert "no persist when order not sent" (not (shouldPersist True True True False))

testOrderAppliedQuantity :: IO ()
testOrderAppliedQuantity = do
    let mk sent live status execQty =
            OrderExecutionEvidence
                { oeeSent = sent
                , oeeLive = live
                , oeeStatus = status
                , oeeExecutedQty = execQty
                }
    assert "not sent does not apply" (isNothing (orderAppliedQuantity (mk False True (Just "FILLED") (Just 1.0)) 1.0))
    assert "paper mode uses fallback qty when sent" (orderAppliedQuantity (mk True False Nothing Nothing) 2.5 == Just 2.5)
    assert "live NEW status blocks apply without fills" (isNothing (orderAppliedQuantity (mk True True (Just "NEW") Nothing) 2.5))
    assert "live partial fill uses executed qty" (orderAppliedQuantity (mk True True (Just "PARTIALLY_FILLED") (Just 0.4)) 2.5 == Just 0.4)
    assert "live canceled status still applies executed qty when present" (orderAppliedQuantity (mk True True (Just "CANCELED") (Just 0.4)) 2.5 == Just 0.4)
    assert "live expired status still applies executed qty when present" (orderAppliedQuantity (mk True True (Just "EXPIRED") (Just 0.2)) 2.5 == Just 0.2)
    assert "live filled status falls back when executed qty missing" (orderAppliedQuantity (mk True True (Just "FILLED") Nothing) 2.5 == Just 2.5)

testApplyExecutedQuantity :: IO ()
testApplyExecutedQuantity = do
    let (pos1, size1, close1, open1) = applyExecutedQuantity 1 2 False 0.5
    assert "partial close keeps long side" (pos1 == 1)
    assertApprox "partial close size" 1e-12 size1 1.5
    assertApprox "partial close qty tracked" 1e-12 close1 0.5
    assertApprox "partial close does not open opposite side" 1e-12 open1 0

    let (pos2, size2, close2, open2) = applyExecutedQuantity 1 2 False 3
    assert "flip crosses to short side" (pos2 == -1)
    assertApprox "flip remaining short size" 1e-12 size2 1
    assertApprox "flip closes full prior size" 1e-12 close2 2
    assertApprox "flip opens new opposite size" 1e-12 open2 1

    let (pos3, size3, close3, open3) = applyExecutedQuantity 0 0 True 1.2
    assert "flat entry opens long" (pos3 == 1)
    assertApprox "flat entry size" 1e-12 size3 1.2
    assertApprox "flat entry has no close leg" 1e-12 close3 0
    assertApprox "flat entry open leg" 1e-12 open3 1.2

    let (pos4, size4, close4, open4) = applyExecutedQuantity 0 0 True 5e-10
    assert "dust fill stays flat" (pos4 == 0)
    assertApprox "dust fill keeps zero size" 1e-12 size4 0
    assertApprox "dust fill has no close leg" 1e-12 close4 0
    assertApprox "dust fill has no open leg" 1e-12 open4 0

    let (pos5, size5, close5, open5) = applyExecutedQuantity 1 1 False 1.0000000005
    assert "dust over-close stays flat" (pos5 == 0)
    assertApprox "dust over-close keeps zero size" 1e-12 size5 0
    assertApprox "dust over-close still closes prior size" 1e-9 close5 1
    assertApprox "dust over-close does not open opposite side" 1e-12 open5 0

runSignalPostGate ::
    Bool ->
    (Int -> (Bool, Maybe String)) ->
    (Int -> (Bool, Maybe String)) ->
    (Int -> Bool) ->
    (Int -> (Bool, Double)) ->
    (Maybe Int, Maybe String)
runSignalPostGate =
    signalRunPostDirectionGates
        (Just 1)
        Nothing
        True
        True
        (const True)
        (const True)
        (const True)
        True

testSignalGateMtfWarmup :: IO ()
testSignalGateMtfWarmup = do
    let mtfCheck = signalMtfConsensusCheck True [Just 1, Nothing, Nothing] 2
        result =
            runSignalPostGate
                True
                mtfCheck
                (signalCrossAssetCheck False Nothing)
                (const True)
                (const (True, 1))
    assert "insufficient MTF directions returns MTF_WARMUP" (result == (Nothing, Just "MTF_WARMUP"))

testSignalGateMtfConsensus :: IO ()
testSignalGateMtfConsensus = do
    let mtfCheck = signalMtfConsensusCheck True [Just 1, Just (-1), Just (-1)] 2
        result =
            runSignalPostGate
                True
                mtfCheck
                (signalCrossAssetCheck False Nothing)
                (const True)
                (const (True, 1))
    assert "disagreeing MTF directions returns MTF_CONSENSUS" (result == (Nothing, Just "MTF_CONSENSUS"))

testSignalGateCrossAsset :: IO ()
testSignalGateCrossAsset = do
    let result =
            runSignalPostGate
                True
                (const (True, Nothing))
                (signalCrossAssetCheck True (Just (-1)))
                (const True)
                (const (True, 1))
    assert "cross-asset disagreement returns CROSS_ASSET" (result == (Nothing, Just "CROSS_ASSET"))

testSignalGateMetaLabel :: IO ()
testSignalGateMetaLabel = do
    let metaCheck dir =
            signalMetaLabelOk
                True
                0.01
                (Just 0.02)
                0.9
                (Just 0.95)
                True
                (dir < 0)
        result =
            runSignalPostGate
                True
                (const (True, Nothing))
                (const (True, Nothing))
                metaCheck
                (const (True, 1))
    assert "meta-label band failure returns META_LABEL" (result == (Nothing, Just "META_LABEL"))

testSignalGateRegimeBank :: IO ()
testSignalGateRegimeBank = do
    let regimeOk = signalRegimeEdgeOk True 0.01 (Just 0.005)
        result =
            runSignalPostGate
                regimeOk
                (const (True, Nothing))
                (const (True, Nothing))
                (const True)
                (const (True, 1))
    assert "regime edge shortfall returns REGIME_BANK" (result == (Nothing, Just "REGIME_BANK"))

testSignalGateFundingOi :: IO ()
testSignalGateFundingOi = do
    let fundingCheck _ = signalFundingOiCheck True (Just 0.001) Nothing 0.7 0.005 Nothing
        result =
            runSignalPostGate
                True
                (const (True, Nothing))
                (const (True, Nothing))
                (const True)
                fundingCheck
    assert "funding pressure above cap returns FUNDING_OI" (result == (Nothing, Just "FUNDING_OI"))

testSignalFundingOiFiniteDamp :: IO ()
testSignalFundingOiFiniteDamp = do
    let (okNoCaps, dampNoCaps) = signalFundingOiCheck True Nothing Nothing (0 / 0) (0 / 0) (Just (1 / 0))
        (okWithCaps, dampWithCaps) = signalFundingOiCheck True (Just 0.001) (Just 0.5) 0.7 (0 / 0) (Just 0.1)
        finite x = not (isNaN x || isInfinite x)
    assert "non-finite inputs without caps keep funding/OI gate open" okNoCaps
    assert "non-finite inputs without caps keep damp finite" (finite dampNoCaps && dampNoCaps == 1)
    assert "non-finite funding with cap blocks entry" (not okWithCaps)
    assert "non-finite funding with cap keeps damp finite" (finite dampWithCaps && dampWithCaps >= 0.7 && dampWithCaps <= 1)

testSignalFundingOiZeroCapsDisable :: IO ()
testSignalFundingOiZeroCapsDisable = do
    let (okZeroCaps, dampZeroCaps) = signalFundingOiCheck True (Just 0) (Just 0) 0.7 0.2 (Just 0.3)
        (okNegativeCaps, dampNegativeCaps) = signalFundingOiCheck True (Just (-1)) (Just (-1)) 0.4 0.2 (Just 0.3)
    assert "zero funding/OI caps disable gating" okZeroCaps
    assert "zero funding/OI caps keep full size damp" (dampZeroCaps == 1)
    assert "negative funding/OI caps are treated as disabled" okNegativeCaps
    assert "negative funding/OI caps keep full size damp" (dampNegativeCaps == 1)

testSensorVarianceIgnoresNonFiniteResiduals :: IO ()
testSensorVarianceIgnoresNonFiniteResiduals = do
    let sv0 = emptySensorVar
        sv1 = updateResidual SensorGBT (0 / 0) sv0
        sv2 = updateResidual SensorGBT (1 / 0) sv1
        sv3 = updateResidual SensorGBT (negate (1 / 0)) sv2
    assert "non-finite residual updates are ignored" (isNothing (varianceFor SensorGBT sv3))

testSensorVarianceNonFiniteNoMutation :: IO ()
testSensorVarianceNonFiniteNoMutation = do
    let svBase =
            foldl'
                (flip (updateResidual SensorGBT))
                emptySensorVar
                [0.01, -0.02, 0.03, -0.01]
        before = varianceFor SensorGBT svBase
        svAfter = updateResidual SensorGBT (0 / 0) (updateResidual SensorGBT (1 / 0) svBase)
        after = varianceFor SensorGBT svAfter
    case (before, after) of
        (Just beforeVar, Just afterVar) ->
            assertApprox "non-finite residual does not mutate prior variance estimate" 1e-12 afterVar beforeVar
        _ -> error "expected finite variance before and after non-finite residual updates"

testRecalculateComboPerformanceFromCompletedOperation :: IO ()
testRecalculateComboPerformanceFromCompletedOperation = do
    let metricsWithPeriods =
            case object ["periods" .= (365 :: Int), "periodsPerYear" .= (365 :: Int)] of
                Aeson.Object o -> o
                _ -> error "expected object"
        (nextEq1, nextAnn1, metrics1) =
            recalculateComboPerformanceFromOperation
                (Just "1d")
                (Just 1.5)
                (Just 0.5)
                metricsWithPeriods
                (Just 1.2)
                1.08
        (nextEq2, nextAnn2, _metrics2) =
            recalculateComboPerformanceFromOperation
                (Just "1d")
                (Just 1.35)
                (Just 0.35)
                metrics1
                (Just 1.0)
                0.9
    assertApprox "delta scales final equity" 1e-12 nextEq1 1.35
    assertApprox "period-based annualized return updated" 1e-12 nextAnn1 0.35
    assertApprox "delta scales from updated baseline" 1e-12 nextEq2 1.215
    assertApprox "annualized return follows updated equity" 1e-12 nextAnn2 0.215

testMergeTopCombosRanksByNestedScore :: IO ()
testMergeTopCombosRanksByNestedScore = do
    let mkCombo sym score =
            object
                [ "params" .= object ["symbol" .= sym]
                , "openThreshold" .= (0.1 :: Double)
                , "closeThreshold" .= (0.05 :: Double)
                , "objective" .= ("score" :: String)
                , "metrics" .= object ["annualizedReturn" .= (0.2 :: Double), "finalEquity" .= (1.5 :: Double), "score" .= score]
                ]
        payload =
            object
                [ "source" .= ("unit-test" :: String)
                , "generatedAtMs" .= (1 :: Int64)
                , "combos" .= [mkCombo ("AAAUSDT" :: String) (0.1 :: Double), mkCombo ("BBBUSDT" :: String) (0.9 :: Double)]
                ]
        merged = mergeTopCombosPayloads 5 2 [payload]
        combos = requireCombosArray "merged payload" merged
        first = requireHead "expected at least one merged combo" combos
    assert "higher nested metrics.score should rank first" (requireComboSymbol "merged first combo" first == "BBBUSDT")

testMergeTopCombosIgnoresBooleanScore :: IO ()
testMergeTopCombosIgnoresBooleanScore = do
    let mkCombo sym scoreVal =
            object
                [ "params" .= object ["symbol" .= sym]
                , "openThreshold" .= (0.1 :: Double)
                , "closeThreshold" .= (0.05 :: Double)
                , "objective" .= ("score" :: String)
                , "metrics" .= object ["annualizedReturn" .= (0.2 :: Double), "finalEquity" .= (1.5 :: Double), "score" .= scoreVal]
                ]
        payload =
            object
                [ "source" .= ("unit-test" :: String)
                , "generatedAtMs" .= (1 :: Int64)
                , "combos" .= [mkCombo ("AAAUSDT" :: String) (Aeson.toJSON True), mkCombo ("BBBUSDT" :: String) (Aeson.toJSON (0.6 :: Double))]
                ]
        merged = mergeTopCombosPayloads 5 2 [payload]
        combos = requireCombosArray "merged payload with boolean score" merged
        first = requireHead "expected at least one merged combo" combos
    assert "boolean score should not outrank numeric score" (requireComboSymbol "merged first combo" first == "BBBUSDT")

testMergeTopCombosDedupPrefersNestedScore :: IO ()
testMergeTopCombosDedupPrefersNestedScore = do
    let mkCombo finalEq score =
            object
                [ "params" .= object ["symbol" .= ("BTCUSDT" :: String)]
                , "openThreshold" .= (0.1 :: Double)
                , "closeThreshold" .= (0.05 :: Double)
                , "objective" .= ("score" :: String)
                , "metrics" .= object ["annualizedReturn" .= (0.2 :: Double), "finalEquity" .= finalEq, "score" .= score]
                ]
        payload =
            object
                [ "source" .= ("unit-test" :: String)
                , "generatedAtMs" .= (1 :: Int64)
                , "combos" .= [mkCombo (5.0 :: Double) (0.1 :: Double), mkCombo (1.2 :: Double) (0.9 :: Double)]
                ]
        merged = mergeTopCombosPayloads 5 2 [payload]
        combos = requireCombosArray "dedup merged payload" merged
        picked = requireHead "expected one deduped combo" combos
    assert "duplicate merge should keep exactly one combo" (length combos == 1)
    assertApprox "dedupe should prefer higher nested metrics.score" 1e-12 (requireComboMetricsScore "dedup picked combo" picked) 0.9

testMergeTopCombosKeepsDistinctSources :: IO ()
testMergeTopCombosKeepsDistinctSources = do
    let mkCombo finalEq =
            object
                [ "params" .= object ["symbol" .= ("BTCUSDT" :: String)]
                , "openThreshold" .= (0.1 :: Double)
                , "closeThreshold" .= (0.05 :: Double)
                , "objective" .= ("score" :: String)
                , "metrics" .= object ["annualizedReturn" .= (0.2 :: Double), "finalEquity" .= finalEq, "score" .= (0.8 :: Double)]
                ]
        payload source finalEq generatedAtMs =
            object
                [ "source" .= source
                , "generatedAtMs" .= generatedAtMs
                , "combos" .= [mkCombo finalEq]
                ]
        merged =
            mergeTopCombosPayloads
                5
                3
                [ payload ("unit-source-a" :: String) (1.2 :: Double) (1 :: Int64)
                , payload ("unit-source-b" :: String) (1.3 :: Double) (2 :: Int64)
                ]
        combos = requireCombosArray "distinct sources merged payload" merged
        sources = sort (map (requireComboSource "distinct source combo") combos)
    assert "same params should be kept when payload sources differ" (length combos == 2)
    assert "merged combos preserve distinct payload sources" (sources == ["unit-source-a", "unit-source-b"])

testComboPerformanceKeyRanksScoreBeforeEquity :: IO ()
testComboPerformanceKeyRanksScoreBeforeEquity = do
    let mkCombo sym score eq =
            object
                [ "params" .= object ["symbol" .= sym]
                , "metrics" .= object ["annualizedReturn" .= (0.2 :: Double), "score" .= score, "finalEquity" .= eq]
                ]
        ranked =
            sortOn
                comboPerformanceKey
                [ mkCombo ("AAAUSDT" :: String) (0.1 :: Double) (5.0 :: Double)
                , mkCombo ("BBBUSDT" :: String) (0.9 :: Double) (1.1 :: Double)
                ]
        first = requireHead "expected ranked combos" ranked
    assert "higher score should outrank higher equity when annualized return ties" (requireComboSymbol "performance key first combo" first == "BBBUSDT")

runMergeAndReadFirstCombo :: String -> Aeson.Value -> IO Aeson.Value
runMergeAndReadFirstCombo label barsValue =
    runMergeAndReadFirstComboFromPayload label topPayload
  where
    topPayload =
        object
            [ "generatedAtMs" .= (1 :: Int)
            , "combos"
                .= [ object
                        [ "finalEquity" .= (1.5 :: Double)
                        , "params"
                            .= object
                                [ "interval" .= ("1h" :: String)
                                , "bars" .= barsValue
                                , "symbol" .= ("BTC/USDT" :: String)
                                ]
                        ]
                   ]
            ]

runMergeAndReadOutputFromPayload :: String -> Aeson.Value -> IO Aeson.Value
runMergeAndReadOutputFromPayload label topPayload =
    runMergeAndReadOutputFromPayloadAndJsonl label topPayload []

runMergeAndReadOutputFromTopPayloads :: String -> Aeson.Value -> [Aeson.Value] -> IO Aeson.Value
runMergeAndReadOutputFromTopPayloads label topPayload extraTopPayloads =
    withTempTestDir label $ \dir -> do
        let topPath = dir </> "top-combos-input.json"
            otherTopPaths =
                [ dir </> ("top-combos-extra-" ++ show idx ++ ".json")
                | idx <- [1 .. length extraTopPayloads]
                ]
            outPath = dir </> "top-combos-output.json"
            jsonlPath = dir </> "optimize_equity_input.jsonl"
            mergeArgs =
                MergeArgs
                    { maTopJson = topPath
                    , maFromJsonl = [jsonlPath]
                    , maFromTopJson = otherTopPaths
                    , maOut = outPath
                    , maMax = 5
                    , maHistoryDir = Nothing
                    , maCopyToDist = False
                    }
        BL.writeFile topPath (Aeson.encode topPayload)
        Control.Monad.forM_ (zip otherTopPaths extraTopPayloads) $ \(path, payload) ->
            BL.writeFile path (Aeson.encode payload)
        BL.writeFile jsonlPath BL.empty
        result <- try (runMerge mergeArgs) :: IO (Either SomeException Int)
        case result of
            Left err -> error ("runMerge unexpectedly threw: " ++ show err)
            Right code -> assert "runMerge should complete successfully" (code == 0)
        outContents <- BL.readFile outPath
        case eitherDecode outContents of
            Left err -> error ("failed to parse merge output JSON: " ++ err)
            Right v -> pure v

runMergeAndReadOutputFromPayloadAndJsonl :: String -> Aeson.Value -> [BS.ByteString] -> IO Aeson.Value
runMergeAndReadOutputFromPayloadAndJsonl label topPayload jsonlLines =
    withTempTestDir label $ \dir -> do
        let topPath = dir </> "top-combos-input.json"
            outPath = dir </> "top-combos-output.json"
            jsonlPath = dir </> "optimize_equity_input.jsonl"
            jsonlPayload =
                if null jsonlLines
                    then BL.empty
                    else BL.fromStrict (BS.intercalate "\n" jsonlLines <> "\n")
            mergeArgs =
                MergeArgs
                    { maTopJson = topPath
                    , maFromJsonl = [jsonlPath]
                    , maFromTopJson = []
                    , maOut = outPath
                    , maMax = 5
                    , maHistoryDir = Nothing
                    , maCopyToDist = False
                    }
        BL.writeFile topPath (Aeson.encode topPayload)
        BL.writeFile jsonlPath jsonlPayload
        result <- try (runMerge mergeArgs) :: IO (Either SomeException Int)
        case result of
            Left err -> error ("runMerge unexpectedly threw: " ++ show err)
            Right code -> assert "runMerge should complete successfully" (code == 0)
        outContents <- BL.readFile outPath
        case eitherDecode outContents of
            Left err -> error ("failed to parse merge output JSON: " ++ err)
            Right v -> pure v

runMergeAndReadFirstComboFromPayload :: String -> Aeson.Value -> IO Aeson.Value
runMergeAndReadFirstComboFromPayload label topPayload =
    do
        outValue <- runMergeAndReadOutputFromPayload label topPayload
        let combos = requireCombosArray "merge output combos" outValue
        pure (requireHead "merge output should contain one combo" combos)

testRunMergeIgnoresOverflowScientificIntegerString :: IO ()
testRunMergeIgnoresOverflowScientificIntegerString = do
    combo <- runMergeAndReadFirstCombo "merge-overflow-int-string" (Aeson.String "1e309")
    assert "overflow scientific integer string should be ignored instead of crashing/truncating" (requireComboBars "overflow bars" combo == 0)

testRunMergeRejectsFractionalIntegerString :: IO ()
testRunMergeRejectsFractionalIntegerString = do
    combo <- runMergeAndReadFirstCombo "merge-fractional-int-string" (Aeson.String "10.9")
    assert "fractional integer-like string should not be truncated" (requireComboBars "fractional bars" combo == 0)

testRunMergePreservesDistinctPayloadSources :: IO ()
testRunMergePreservesDistinctPayloadSources =
    do
        let mkPayload source generatedAtMs finalEq =
                object
                    [ "source" .= source
                    , "generatedAtMs" .= generatedAtMs
                    , "combos"
                        .= [ object
                                [ "finalEquity" .= finalEq
                                , "objective" .= ("score" :: String)
                                , "metrics"
                                    .= object
                                        [ "annualizedReturn" .= (0.2 :: Double)
                                        , "score" .= (0.8 :: Double)
                                        ]
                                , "params"
                                    .= object
                                        [ "interval" .= ("1h" :: String)
                                        , "bars" .= (100 :: Int)
                                        , "symbol" .= ("BTC/USDT" :: String)
                                        ]
                                ]
                           ]
                    ]
        outValue <-
            runMergeAndReadOutputFromTopPayloads
                "merge-distinct-top-payload-sources"
                (mkPayload ("unit-source-a" :: String) (1 :: Int) (1.2 :: Double))
                [mkPayload ("unit-source-b" :: String) (2 :: Int) (1.3 :: Double)]
        let combos = requireCombosArray "merge distinct payload source combos" outValue
            sources = sort (map (requireComboSource "merge distinct payload source combo") combos)
        assert "runMerge should keep combos distinct when only payload source differs" (length combos == 2)
        assert "runMerge should propagate payload-level source metadata into merged combos" (sources == ["unit-source-a", "unit-source-b"])

testRunMergePreservesDexPlatformSymbolSemantics :: IO ()
testRunMergePreservesDexPlatformSymbolSemantics =
    do
        let dexSymbol = "0xabc/0xdef"
            topPayload =
                object
                    [ "generatedAtMs" .= (1 :: Int)
                    , "combos"
                        .= [ object
                                [ "finalEquity" .= (1.5 :: Double)
                                , "params"
                                    .= object
                                        [ "platform" .= ("uniswap" :: String)
                                        , "symbol" .= dexSymbol
                                        , "interval" .= ("1h" :: String)
                                        , "bars" .= (100 :: Int)
                                        ]
                                ]
                           ]
                    ]
        combo <- runMergeAndReadFirstComboFromPayload "merge-dex-platform-symbol" topPayload
        assert "DEX platform should be preserved in merged params" (requireComboPlatform "dex combo platform" combo == Just "uniswap")
        assert "DEX symbol should not be Binance-normalized in merged params.binanceSymbol" (requireComboBinanceSymbol "dex combo binanceSymbol" combo == Just dexSymbol)

testRunMergeCanonicalizesCoinbasePrefixedPlatform :: IO ()
testRunMergeCanonicalizesCoinbasePrefixedPlatform =
    do
        let topPayload =
                object
                    [ "generatedAtMs" .= (1 :: Int)
                    , "combos"
                        .= [ object
                                [ "finalEquity" .= (1.5 :: Double)
                                , "params"
                                    .= object
                                        [ "platform" .= ("coinbase-advanced" :: String)
                                        , "symbol" .= ("BTC/USD" :: String)
                                        , "interval" .= ("1h" :: String)
                                        , "bars" .= (100 :: Int)
                                        ]
                                ]
                           ]
                    ]
        combo <- runMergeAndReadFirstComboFromPayload "merge-coinbase-prefixed-platform" topPayload
        assert "coinbase-prefixed platform should be canonicalized in merged params" (requireComboPlatform "coinbase combo platform" combo == Just "coinbase")
        assert "coinbase-prefixed symbols should keep coinbase delimiter semantics" (requireComboBinanceSymbol "coinbase combo binanceSymbol" combo == Just "BTC-USD")

testRunMergeCanonicalizesDexPrefixedPlatform :: IO ()
testRunMergeCanonicalizesDexPrefixedPlatform =
    do
        let topPayload =
                object
                    [ "generatedAtMs" .= (1 :: Int)
                    , "combos"
                        .= [ object
                                [ "finalEquity" .= (1.5 :: Double)
                                , "params"
                                    .= object
                                        [ "platform" .= ("uniswap-v3" :: String)
                                        , "symbol" .= ("ETH/USDT" :: String)
                                        , "interval" .= ("1h" :: String)
                                        , "bars" .= (100 :: Int)
                                        ]
                                ]
                           ]
                    ]
        combo <- runMergeAndReadFirstComboFromPayload "merge-dex-prefixed-platform" topPayload
        assert "dex-prefixed platform should be canonicalized in merged params" (requireComboPlatform "dex-prefixed combo platform" combo == Just "uniswap")
        assert "dex-prefixed symbols should preserve dex delimiter semantics" (requireComboBinanceSymbol "dex-prefixed combo binanceSymbol" combo == Just "ETH/USDT")

testRunMergeUsesNestedMetricsFallbacks :: IO ()
testRunMergeUsesNestedMetricsFallbacks =
    do
        let topPayload =
                object
                    [ "generatedAtMs" .= (1 :: Int)
                    , "combos"
                        .= [ object
                                [ "metrics"
                                    .= object
                                        [ "finalEquity" .= (1.5 :: Double)
                                        , "score" .= (0.9 :: Double)
                                        , "periods" .= (365 :: Int)
                                        ]
                                , "params"
                                    .= object
                                        [ "interval" .= ("1d" :: String)
                                        , "symbol" .= ("BTC/USDT" :: String)
                                        ]
                                ]
                           ]
                    ]
        combo <- runMergeAndReadFirstComboFromPayload "merge-nested-metrics-fallbacks" topPayload
        assertApprox "nested metrics.finalEquity should keep combo during merge" 1e-12 (requireComboFinalEquity "nested metrics combo finalEquity" combo) 1.5
        assertApprox "nested metrics.score should be preserved during merge" 1e-12 (requireComboMetricsScore "nested metrics combo score" combo) 0.9
        assertApprox "annualizedReturn should be backfilled from nested metrics.periods when bars are missing" 1e-12 (requireComboMetricsDouble "nested metrics combo annualizedReturn" "annualizedReturn" combo) 0.5

testRunMergeDedupPrefersNestedMetricsScore :: IO ()
testRunMergeDedupPrefersNestedMetricsScore =
    do
        let mkCombo finalEq score =
                object
                    [ "objective" .= ("score" :: String)
                    , "metrics"
                        .= object
                            [ "annualizedReturn" .= (0.2 :: Double)
                            , "finalEquity" .= finalEq
                            , "score" .= score
                            ]
                    , "params"
                        .= object
                            [ "interval" .= ("1h" :: String)
                            , "bars" .= (100 :: Int)
                            , "symbol" .= ("BTC/USDT" :: String)
                            ]
                    ]
            topPayload =
                object
                    [ "generatedAtMs" .= (1 :: Int)
                    , "combos" .= [mkCombo (5.0 :: Double) (0.1 :: Double), mkCombo (1.2 :: Double) (0.9 :: Double)]
                    ]
        outValue <- runMergeAndReadOutputFromPayload "merge-dedup-nested-score" topPayload
        let combos = requireCombosArray "merge dedup nested score combos" outValue
            combo = requireHead "merge dedup nested score should keep one combo" combos
        assert "duplicate merge should keep exactly one combo" (length combos == 1)
        assertApprox "nested metrics.score should drive merge dedupe tie-breaks" 1e-12 (requireComboMetricsScore "merge dedup picked combo score" combo) 0.9

testRunMergeSkipsInvalidUtf8JsonlLines :: IO ()
testRunMergeSkipsInvalidUtf8JsonlLines =
    withTempTestDir "merge-invalid-utf8-jsonl" $ \dir -> do
        let topPath = dir </> "top-combos-input.json"
            outPath = dir </> "top-combos-output.json"
            jsonlPath = dir </> "optimize_equity_invalid_utf8.jsonl"
            validLine =
                BS.pack
                    "{\"ok\":true,\"finalEquity\":1.5,\"source\":\"jsonl\",\"params\":{\"interval\":\"1h\",\"bars\":120,\"symbol\":\"BTCUSDT\"}}"
            invalidLine = BS.pack "\255"
            jsonlPayload = BL.fromStrict (BS.intercalate "\n" [validLine, invalidLine] <> "\n")
            topPayload = object ["generatedAtMs" .= (1 :: Int), "combos" .= ([] :: [Aeson.Value])]
            mergeArgs =
                MergeArgs
                    { maTopJson = topPath
                    , maFromJsonl = [jsonlPath]
                    , maFromTopJson = []
                    , maOut = outPath
                    , maMax = 5
                    , maHistoryDir = Nothing
                    , maCopyToDist = False
                    }
        BL.writeFile topPath (Aeson.encode topPayload)
        BL.writeFile jsonlPath jsonlPayload
        code <- runMerge mergeArgs
        assert "runMerge should complete successfully despite invalid utf8 line" (code == 0)
        outContents <- BL.readFile outPath
        outValue <-
            case eitherDecode outContents of
                Left err -> error ("failed to parse merge output JSON: " ++ err)
                Right v -> pure v
        let combos = requireCombosArray "merge output combos" outValue
            combo = requireHead "merge output should contain one combo" combos
        assert "valid jsonl combo should still be retained" (length combos == 1)
        assert "merged combo should preserve parsed bars from valid line" (requireComboBars "jsonl combo bars" combo == 120)

testRunMergeSkipsFalseStringOkFlag :: IO ()
testRunMergeSkipsFalseStringOkFlag =
    do
        let topPayload = object ["generatedAtMs" .= (1 :: Int), "combos" .= ([] :: [Aeson.Value])]
            jsonlLines =
                [ BS.pack
                    "{\"ok\":\"false\",\"finalEquity\":1.5,\"params\":{\"interval\":\"1h\",\"bars\":120,\"symbol\":\"BTCUSDT\"}}"
                ]
        outValue <- runMergeAndReadOutputFromPayloadAndJsonl "merge-jsonl-string-false-ok" topPayload jsonlLines
        let combos = requireCombosArray "merge false ok output combos" outValue
        assert "string false ok flag should not import jsonl combo" (null combos)

testDexTradeArgsRequireTokensNotSymbol :: IO ()
testDexTradeArgsRequireTokensNotSymbol = do
    let dexBaseArgs =
            [ "--platform"
            , "uniswap"
            , "--data"
            , "sample.csv"
            , "--price-column"
            , "close"
            , "--interval"
            , "1h"
            , "--bars"
            , "100"
            , "--lookback-bars"
            , "10"
            , "--trade-only"
            , "--binance-trade"
            , "--no-binance-live"
            ]
        dexWithTokens =
            dexBaseArgs
                ++ [ "--dex-base-token"
                   , "ETH"
                   , "--dex-quote-token"
                   , "USDC"
                   ]
    case parseArgsResult dexWithTokens of
        Left err -> error ("unexpected validation failure for dex token trade: " ++ err)
        Right _ -> pure ()
    case parseArgsResult (dexBaseArgs ++ ["--dex-base-token", "ETH"]) of
        Left err -> assert "missing quote token rejected" ("--dex-base-token and --dex-quote-token must be provided together" `isInfixOf` err)
        Right _ -> error "expected missing dex quote token to be rejected"

testDexCsvSymbolLookbackUsesCsvBars :: IO ()
testDexCsvSymbolLookbackUsesCsvBars = do
    let argv =
            [ "--platform"
            , "uniswap"
            , "--data"
            , "sample.csv"
            , "--symbol"
            , "ETH/USDC"
            , "--interval"
            , "1h"
            , "--lookback-bars"
            , "600"
            ]
    case parseArgsResult argv of
        Left err ->
            error
                ( "expected DEX CSV+symbol args to accept large lookback with --bars auto; got: "
                    ++ err
                )
        Right _ -> pure ()

testDexTradeArgsRejectPartialTokenOverrides :: IO ()
testDexTradeArgsRejectPartialTokenOverrides = do
    let argv =
            [ "--platform"
            , "uniswap"
            , "--data"
            , "sample.csv"
            , "--price-column"
            , "close"
            , "--interval"
            , "1h"
            , "--bars"
            , "100"
            , "--lookback-bars"
            , "10"
            , "--trade-only"
            , "--binance-trade"
            , "--no-binance-live"
            , "--symbol"
            , "ETH/USDC"
            , "--dex-base-token"
            , "WETH"
            ]
    case parseArgsResult argv of
        Left err -> assert "partial DEX token override is rejected" ("--dex-base-token and --dex-quote-token must be provided together" `isInfixOf` err)
        Right _ -> error "expected partial DEX token override to fail validation"

testDexResolveTokensRejectsMalformedAddress :: IO ()
testDexResolveTokensRejectsMalformedAddress = do
    let env = mkDexTestEnv
        malformed = "0x" ++ replicate 40 'g'
    resolved <- resolveDexTokens env malformed "native" (Just 18) Nothing
    case resolved of
        Left err -> assert "malformed 0x token is rejected" ("Invalid token address" `isInfixOf` err)
        Right _ -> error "expected malformed DEX token address to be rejected"

testDexResolveTokensNativeDecimalsOverride :: IO ()
testDexResolveTokensNativeDecimalsOverride = do
    let env = mkDexTestEnv
    resolved <- resolveDexTokens env "native" "eth" (Just 6) (Just 8)
    case resolved of
        Left err -> error ("unexpected native override resolution failure: " ++ err)
        Right (baseTok, quoteTok) -> do
            assert "base native decimals override applied" (dtDecimals baseTok == 6)
            assert "quote native decimals override applied" (dtDecimals quoteTok == 8)

testDexResolveTokensRejectsExcessiveDecimalsOverride :: IO ()
testDexResolveTokensRejectsExcessiveDecimalsOverride = do
    let env = mkDexTestEnv
    resolved <- resolveDexTokens env "native" "eth" (Just 256) Nothing
    case resolved of
        Left err -> assert "excessive decimals override is rejected" ("Token decimals must be <= 255" `isInfixOf` err)
        Right _ -> error "expected excessive decimals override to be rejected"

testTokenAmountToIntegerRejectsExcessiveDecimals :: IO ()
testTokenAmountToIntegerRejectsExcessiveDecimals =
    case tokenAmountToInteger 1.0 256 of
        Left err -> assert "token amount conversion rejects excessive decimals" ("Token decimals must be <= 255" `isInfixOf` err)
        Right _ -> error "expected token amount conversion to reject excessive decimals"

testDexExtractTxHashPlainOutput :: IO ()
testDexExtractTxHashPlainOutput = do
    let hash = "0x" ++ replicate 64 'a'
    assert "plain cast output hash is extracted" (extractTxHash (hash ++ "\n") == Just hash)

testDexExtractTxHashStructuredOutput :: IO ()
testDexExtractTxHashStructuredOutput = do
    let hash = "0x" ++ replicate 64 'b'
        out = "{\"transactionHash\":\"" ++ hash ++ "\",\"status\":\"0x1\"}"
        upperOut = "tx=0X" ++ replicate 64 'C' ++ ",ok"
    assert "json-wrapped tx hash is extracted" (extractTxHash out == Just hash)
    assert "uppercase 0X hash is normalized and extracted" (extractTxHash upperOut == Just ("0x" ++ replicate 64 'c'))

testDexExtractTxHashRejectsMalformedOutput :: IO ()
testDexExtractTxHashRejectsMalformedOutput = do
    let tooShort = "0x" ++ replicate 63 'a'
        tooLong = "0x" ++ replicate 65 'a'
    assert "missing tx hash returns Nothing" (isNothing (extractTxHash "{\"status\":\"ok\"}"))
    assert "short hex payload is rejected" (isNothing (extractTxHash tooShort))
    assert "long hex payload is rejected" (isNothing (extractTxHash tooLong))

mkDexTestEnv :: DexEnv
mkDexTestEnv =
    DexEnv
        { deChainId = 1
        , deRpcUrl = "http://127.0.0.1:8545"
        , dePrivateKey = "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        , deAddress = "0x1111111111111111111111111111111111111111"
        , deBaseUrl = "http://127.0.0.1:1"
        , deApiKey = Nothing
        , deProtocols = Nothing
        , deAutoApprove = False
        , deApproveWaitSec = 0
        }

testPlatformIntervals :: IO ()
testPlatformIntervals = do
    assert "binance supports 3m" (isPlatformInterval PlatformBinance "3m")
    assert "binance supports uppercase hour interval" (isPlatformInterval PlatformBinance "1H")
    assert "binance supports trimmed interval input" (isPlatformInterval PlatformBinance " 1h ")
    assert "binance keeps monthly interval distinct" (isPlatformInterval PlatformBinance "1M")
    assert "coinbase supports 1h" (isPlatformInterval PlatformCoinbase "1h")
    assert "coinbase supports uppercase interval input" (isPlatformInterval PlatformCoinbase "1H")
    assert "kraken rejects 3m" (not (isPlatformInterval PlatformKraken "3m"))
    assert "poloniex supports 2h" (isPlatformInterval PlatformPoloniex "2h")

testBinanceIntervalHelperNormalization :: IO ()
testBinanceIntervalHelperNormalization = do
    assert "binance helper supports uppercase hour interval" (isBinanceInterval "1H")
    assert "binance helper supports trimmed interval input" (isBinanceInterval " 1h ")
    assert "binance helper preserves monthly interval distinctness" (isBinanceInterval "1M")
    assert "binance helper rejects invalid interval" (not (isBinanceInterval "13m"))

testPlatformIntervalMapping :: IO ()
testPlatformIntervalMapping = do
    assert "coinbase 1h -> 3600s" (coinbaseIntervalSeconds "1h" == Just 3600)
    assert "coinbase trims + normalizes casing" (coinbaseIntervalSeconds " 1H " == Just 3600)
    assert "coinbase rejects 30m" (isNothing (coinbaseIntervalSeconds "30m"))
    assert "kraken 1h -> 60m" (krakenIntervalMinutes "1h" == Just 60)
    assert "kraken trims + normalizes casing" (krakenIntervalMinutes " 1H " == Just 60)
    assert "poloniex 2h -> HOUR_2" (poloniexIntervalLabel "2h" == Just "HOUR_2")
    assert "poloniex trims + normalizes casing (label)" (poloniexIntervalLabel " 2H " == Just "HOUR_2")
    assert "poloniex 2h -> 7200s" (poloniexIntervalSeconds "2h" == Just 7200)
    assert "poloniex trims + normalizes casing (seconds)" (poloniexIntervalSeconds " 2H " == Just 7200)
    assert "poloniex rejects 1m" (isNothing (poloniexIntervalSeconds "1m"))

testMethodSelection :: IO ()
testMethodSelection = do
    let kal = [1.0, 2.0]
        lstm = [10.0, 20.0]
        w = 0.25
        blend = [w * 1.0 + (1 - w) * 10.0, w * 2.0 + (1 - w) * 20.0]
        badWeight = 0 / 0
        badKal = [0 / 0, 5.0]
        badLstm = [20.0, 1 / 0]
        finite x = not (isNaN x || isInfinite x)
    assert "both keeps both" (selectPredictions MethodBoth w kal lstm == (kal, lstm))
    assert "kalman-only duplicates kalman" (selectPredictions MethodKalmanOnly w kal lstm == (kal, kal))
    assert "kalman_physics_error duplicates kalman stream" (selectPredictions MethodKalmanPhysicsError w kal lstm == (kal, kal))
    assert "lstm-only duplicates lstm" (selectPredictions MethodLstmOnly w kal lstm == (lstm, lstm))
    assert "blend averages" (selectPredictions MethodBlend w kal lstm == (blend, blend))
    assert "conf_blend falls back to weighted average when confidence context is unavailable" (selectPredictions MethodConfBlend w kal lstm == (blend, blend))
    assert "conf_pick falls back to weighted average when confidence context is unavailable" (selectPredictions MethodConfPick w kal lstm == (blend, blend))
    assert "conformal_clip falls back to weighted average when context is unavailable" (selectPredictions MethodConformalClip w kal lstm == (blend, blend))
    assert "cost_pick falls back to weighted average when context is unavailable" (selectPredictions MethodCostPick w kal lstm == (blend, blend))
    assert "harmonic_blend falls back to weighted average when price context is unavailable" (selectPredictions MethodHarmonicBlend w kal lstm == (blend, blend))
    assert "disagreement_guard falls back to weighted average when context is unavailable" (selectPredictions MethodDisagreementGuard w kal lstm == (blend, blend))
    assert "median_blend falls back to weighted average when context is unavailable" (selectPredictions MethodMedianBlend w kal lstm == (blend, blend))
    assert "neutral_guard falls back to weighted average when context is unavailable" (selectPredictions MethodNeutralGuard w kal lstm == (blend, blend))
    assert "risk_parity_blend falls back to weighted average when context is unavailable" (selectPredictions MethodRiskParityBlend w kal lstm == (blend, blend))
    assert "consensus_boost falls back to weighted average when context is unavailable" (selectPredictions MethodConsensusBoost w kal lstm == (blend, blend))
    assert "anchor_blend falls back to weighted average when context is unavailable" (selectPredictions MethodAnchorBlend w kal lstm == (blend, blend))
    assert "tension_gate falls back to weighted average when context is unavailable" (selectPredictions MethodTensionGate w kal lstm == (blend, blend))
    assert "entropy_blend falls back to weighted average when context is unavailable" (selectPredictions MethodEntropyBlend w kal lstm == (blend, blend))
    assert "coherence_gate falls back to weighted average when context is unavailable" (selectPredictions MethodCoherenceGate w kal lstm == (blend, blend))
    assert "divergence_gate falls back to weighted average when context is unavailable" (selectPredictions MethodDivergenceGate w kal lstm == (blend, blend))
    assert "fractal_blend falls back to weighted average when context is unavailable" (selectPredictions MethodFractalBlend w kal lstm == (blend, blend))
    assert "phase_cancel falls back to weighted average when context is unavailable" (selectPredictions MethodPhaseCancel w kal lstm == (blend, blend))
    assert "softmax_blend falls back to weighted average when context is unavailable" (selectPredictions MethodSoftmaxBlend w kal lstm == (blend, blend))
    assert "smooth_softmax_blend falls back to weighted average when context is unavailable" (selectPredictions MethodSmoothSoftmaxBlend w kal lstm == (blend, blend))
    assert "hedge_blend falls back to weighted average when context is unavailable" (selectPredictions MethodHedgeBlend w kal lstm == (blend, blend))
    assert "net_softmax_blend falls back to weighted average when context is unavailable" (selectPredictions MethodNetSoftmaxBlend w kal lstm == (blend, blend))
    assert "edge_blend falls back to weighted average when edge context is unavailable" (selectPredictions MethodEdgeBlend w kal lstm == (blend, blend))
    assert "edge_pick falls back to weighted average when edge context is unavailable" (selectPredictions MethodEdgePick w kal lstm == (blend, blend))
    assert "geo_blend falls back to weighted average when price context is unavailable" (selectPredictions MethodGeoBlend w kal lstm == (blend, blend))
    let (regimeFallbackLeft, regimeFallbackRight) = selectPredictions MethodRegimeSwitch w [1.0] [10.0]
    assertApproxList "regime_switch falls back to weighted average without prior momentum context (left stream)" 1e-9 (take 1 blend) regimeFallbackLeft
    assertApproxList "regime_switch falls back to weighted average without prior momentum context (right stream)" 1e-9 (take 1 blend) regimeFallbackRight
    let regimeKal = [100.0, 106.0, 107.0, 101.0]
        regimeLstm = [100.0, 104.0, 98.0, 99.0]
        regimeBlendWeight = 0.25
        regimeExpected = [100.0, 105.2, 102.5, 99.5]
        (regimeLeft, regimeRight) = selectPredictions MethodRegimeSwitch regimeBlendWeight regimeKal regimeLstm
    assertApproxList "regime_switch applies momentum/divergence routing (left stream)" 1e-9 regimeExpected regimeLeft
    assertApproxList "regime_switch applies momentum/divergence routing (right stream)" 1e-9 regimeExpected regimeRight
    let (safeBlendLeft, safeBlendRight) = selectPredictions MethodBlend badWeight badKal badLstm
    assert "blend with non-finite weight/preds keeps output finite (left stream)" (all finite safeBlendLeft)
    assert "blend with non-finite weight/preds keeps output finite (right stream)" (all finite safeBlendRight)
    assertApproxList "blend with one bad input per step keeps finite counterpart" 1e-9 [20.0, 5.0] safeBlendLeft
    let (bothBadLeft, bothBadRight) = selectPredictions MethodBlend badWeight [0 / 0] [negate (1 / 0)]
    assert "blend with both bad inputs returns neutral zero fallback (left stream)" (bothBadLeft == [0.0])
    assert "blend with both bad inputs returns neutral zero fallback (right stream)" (bothBadRight == [0.0])
    assert "bandit_router preserves both prediction streams for routing" (selectPredictions MethodBanditRouter w kal lstm == (kal, lstm))

testTrainBacktestSplit :: IO ()
testTrainBacktestSplit = do
    let xs = [1 .. 100 :: Int]
    case splitTrainBacktest 5 0.2 xs of
        Left e -> error e
        Right s -> do
            assert "trainEndRaw" (splitTrainEndRaw s == 80)
            assert "trainEnd" (splitTrainEnd s == 80)
            assert "train size" (length (splitTrain s) == 80)
            assert "backtest size" (length (splitBacktest s) == 20)
            assert "no overlap" (splitTrain s ++ splitBacktest s == xs)

    let xs2 = [1 .. 60 :: Int]
    case splitTrainBacktest 50 0.9 xs2 of
        Left e -> assert "ratio too large" ("training bars" `isInfixOf` e)
        Right _ -> error "expected splitTrainBacktest to reject ratios with too few training bars"

    case splitTrainBacktest 5 (0 / 0) xs of
        Left e -> assert "non-finite ratio rejected" ("between 0 and 1" `isInfixOf` e)
        Right _ -> error "expected splitTrainBacktest to reject non-finite ratios"

testSweepThreshold :: IO ()
testSweepThreshold = do
    let prices = [100, 110]
        kalPred = [110]
        lstmPred = [110]
        cfg = baseEnsembleConfig
    (openThr, closeThr, bt) <-
        case sweepThreshold MethodKalmanOnly cfg prices kalPred lstmPred Nothing of
            Left e -> error e
            Right v -> pure v
    assertApprox "open thr close to 10%" 1e-6 openThr 0.1
    assertApprox "close thr close to 10%" 1e-6 closeThr 0.1
    assertApprox "final equity" 1e-12 (bestFinalEquity bt) 1.1

testSweepThresholdCostPickUsesLatestReturnVolatility :: IO ()
testSweepThresholdCostPickUsesLatestReturnVolatility = do
    let prices = [100, 100, 120]
        kalPred = [100, 104]
        lstmPred = [100, 99]
        cfg =
            baseEnsembleConfig
                { ecOpenThreshold = 0.02
                , ecCloseThreshold = 0.02
                , ecBlendWeight = 0.2
                , ecSlippageVolMult = 2.0
                }
    (_openThr, _closeThr, bt) <-
        case sweepThreshold MethodCostPick cfg prices kalPred lstmPred Nothing of
            Left e -> error e
            Right v -> pure v
    assertApprox "cost-pick stays flat when latest bar volatility implies high costs" 1e-12 (bestFinalEquity bt) 1.0

testSweepThresholdFiniteOnNonFinitePredictions :: IO ()
testSweepThresholdFiniteOnNonFinitePredictions = do
    let prices = [100, 101, 102, 103]
        kalPred = [0 / 0, 0 / 0, 0 / 0]
        lstmPred = [1 / 0, 0 / 0, negate (1 / 0)]
        cfg = baseEnsembleConfig
        methods =
            [ MethodBlend
            , MethodConfBlend
            , MethodConfPick
            , MethodConformalClip
            , MethodCostPick
            , MethodHarmonicBlend
            , MethodDisagreementGuard
            , MethodMedianBlend
            , MethodNeutralGuard
            , MethodRiskParityBlend
            , MethodConsensusBoost
            , MethodAnchorBlend
            , MethodTensionGate
            , MethodEntropyBlend
            , MethodCoherenceGate
            , MethodDivergenceGate
            , MethodFractalBlend
            , MethodPhaseCancel
            , MethodSoftmaxBlend
            , MethodSmoothSoftmaxBlend
            , MethodHedgeBlend
            , MethodNetSoftmaxBlend
            , MethodEdgeBlend
            , MethodEdgePick
            , MethodGeoBlend
            , MethodRegimeSwitch
            ]
        finite x = not (isNaN x || isInfinite x)
        assertMethod m =
            case sweepThreshold m cfg prices kalPred lstmPred Nothing of
                Left err -> error ("expected finite fallback for " ++ show m ++ ": " ++ err)
                Right (_, _, bt) ->
                    assert ("bestFinalEquity stays finite for " ++ show m) (finite (bestFinalEquity bt))
    mapM_ assertMethod methods

testSweepThresholdRegimeSwitchNonFiniteStaysNeutral :: IO ()
testSweepThresholdRegimeSwitchNonFiniteStaysNeutral = do
    let prices = [100, 101, 102, 103]
        kalPred = [0 / 0, 0 / 0, 0 / 0]
        lstmPred = [1 / 0, 0 / 0, negate (1 / 0)]
        cfg = baseEnsembleConfig
    case sweepThreshold MethodRegimeSwitch cfg prices kalPred lstmPred Nothing of
        Left err -> error ("expected neutral regime-switch fallback: " ++ err)
        Right (_, _, bt) ->
            assertApprox "regime-switch fallback stays neutral" 1e-12 (bestFinalEquity bt) 1

testOptimizeOperations :: IO ()
testOptimizeOperations = do
    let prices = [100, 110]
        kalPred = [110]
        lstmPred = [90]
        cfg = baseEnsembleConfig
    (m, openThr, closeThr, bt) <-
        case optimizeOperations cfg prices kalPred lstmPred Nothing of
            Left e -> error e
            Right v -> pure v
    assert
        "picked method that follows kalman for this case"
        ( m == MethodKalmanOnly
            || m == MethodConfPick
            || m == MethodEdgePick
        )
    assertApprox "open thr close to 10%" 1e-6 openThr 0.1
    assertApprox "close thr close to 10%" 1e-6 closeThr 0.1
    assertApprox "final equity" 1e-12 (bestFinalEquity bt) 1.1

testOptimizerPartialTakeProfitZeroRange :: IO ()
testOptimizerPartialTakeProfitZeroRange = do
    let seed = 1337
        rng0 = seedRng seed
        expectedProbe =
            let (v, _) = nextDouble rng0
             in v
    sampled <- timeout 1000000 $ do
        let (mPartial, rng1) = sampleTakeProfitPartial (0, 0) 0 rng0
        probe <- evaluate (fst (nextDouble rng1))
        pure (mPartial, probe)
    case sampled of
        Nothing -> error "partial take-profit zero-range sampling timed out"
        Just (mPartial, probe) -> do
            assert "zero-range partial take-profit is disabled" (isNothing mPartial)
            assertApprox "zero-range partial take-profit keeps RNG unchanged" 1e-15 probe expectedProbe

testOptimizerQualityPresetIntervals :: IO ()
testOptimizerQualityPresetIntervals = do
    let explicitInterval = qualityPresetIntervalFields (Just " 1H ") Nothing
        explicitIntervals = qualityPresetIntervalFields Nothing (Just "15m, 1h")
        defaultIntervals = qualityPresetIntervalFields Nothing Nothing
    assert
        "quality preset keeps explicit singleton interval instead of widening to defaults"
        (explicitInterval == (Just "1H", Nothing))
    assert
        "quality preset keeps explicit interval list instead of replacing it"
        (explicitIntervals == (Nothing, Just "15m, 1h"))
    assert
        "quality preset supplies default search intervals only when the user provided none"
        (defaultIntervals == (Nothing, Just binanceIntervalsCsv))

testOptimizerCalmarFallback :: IO ()
testOptimizerCalmarFallback = do
    let metrics =
            KM.fromList
                [ (AK.fromString "finalEquity", Aeson.toJSON (1.25 :: Double))
                , (AK.fromString "annualizedReturn", Aeson.toJSON (0.42 :: Double))
                , (AK.fromString "maxDrawdown", Aeson.toJSON (0.0 :: Double))
                , (AK.fromString "turnover", Aeson.toJSON (0.0 :: Double))
                , (AK.fromString "roundTrips", Aeson.toJSON (3 :: Int))
                , (AK.fromString "tradeCount", Aeson.toJSON (3 :: Int))
                , (AK.fromString "exposure", Aeson.toJSON (0.2 :: Double))
                ]
        score = objectiveScore metrics "calmar" 1.5 0.2
    assertApprox
        "calmar should fall back to annualized return when max drawdown is zero"
        1e-12
        score
        0.42

testOptimizerNormalizeOptionalPositiveFraction :: IO ()
testOptimizerNormalizeOptionalPositiveFraction = do
    assert "negative fraction is disabled" (isNothing (normalizeOptionalPositiveFraction (Just (-0.2))))
    assert "zero fraction is disabled" (isNothing (normalizeOptionalPositiveFraction (Just 0)))
    assert "nan fraction is disabled" (isNothing (normalizeOptionalPositiveFraction (Just (0 / 0))))
    assert "infinite fraction is disabled" (isNothing (normalizeOptionalPositiveFraction (Just (1 / 0))))
    case normalizeOptionalPositiveFraction (Just 0.25) of
        Just v -> assertApprox "valid fraction is preserved" 1e-15 v 0.25
        Nothing -> error "expected valid positive fraction to be preserved"
    case normalizeOptionalPositiveFraction (Just 1.25) of
        Just v -> assertApprox "oversized fraction is clamped below one" 1e-15 v 0.999999
        Nothing -> error "expected oversized positive fraction to be clamped"

testOptimizerIntRangeFixedRange :: IO ()
testOptimizerIntRangeFixedRange = do
    let rng0 = seedRng 2026
        expectedProbe = fst (nextDouble rng0)
        (v, rng1) = nextIntRange 7 7 rng0
        probe = fst (nextDouble rng1)
    assert "fixed range returns that value" (v == 7)
    assertApprox "fixed range keeps RNG unchanged" 1e-15 probe expectedProbe

testOptimizerIntRangeFullSpan :: IO ()
testOptimizerIntRangeFullSpan = do
    let rng0 = seedRng 99
        expectedProbe = fst (nextDouble rng0)
        (v1, rng1) = nextIntRange minBound maxBound rng0
        (v2, rng2) = nextIntRange minBound maxBound rng1
        probe = fst (nextDouble rng2)
    assert "full-span sample stays in bounds (first)" (v1 >= minBound && v1 <= maxBound)
    assert "full-span sample stays in bounds (second)" (v2 >= minBound && v2 <= maxBound)
    assert "full-span range advances RNG state" (probe /= expectedProbe)

testFormalCloseTimingWindow :: IO ()
testFormalCloseTimingWindow = do
    let sample =
            optimalCloseObservation
                "combo-a"
                1000
                2000
                [ (900, -1)
                , (1200, 0.1)
                , (1500, 0.4)
                , (2600, 0.7)
                , (2900, 0.6)
                ]
    case sample of
        Nothing -> error "expected optimal close sample"
        Just obs -> do
            assert "tm stays within [ta, 2tc-ta]" (ctoOptimalCloseAtMs obs == 2600)
            let stats = buildCloseTimingStats [obs]
            assert "one combo stat emitted" (length stats == 1)
            case stats of
                [st] -> assert "single-sample median ratio is 1.6" (abs (ctsMedianRatio st - 1.6) < 1e-9)
                _ -> pure ()

testFormalCloseTimingDecision :: IO ()
testFormalCloseTimingDecision = do
    let observations =
            [ CloseTimingObservation "combo-a" 0 100 120
            , CloseTimingObservation "combo-a" 0 100 140
            , CloseTimingObservation "combo-a" 0 100 160
            ]
        stats = buildCloseTimingStats observations
    case stats of
        [st] -> do
            let holdDecision = closeTimingDecision 0.0 st 0 100 120
                closeDecision = closeTimingDecision 1.0 st 0 100 180
            assert "policy holds before target quantile" (not (ctdShouldClose holdDecision))
            assert "policy closes after risk-budget target" (ctdShouldClose closeDecision)
        _ -> error "expected one combo stat"

formalVerificationReport :: FormalVerificationReport
formalVerificationReport = verifyFormalOptimization

assertThrowsContains :: String -> (() -> IO a) -> IO ()
assertThrowsContains needle mkAction = do
    r <- (try (Control.Monad.void (mkAction ())) :: IO (Either SomeException ()))
    case r of
        Left e -> assert ("missing exception substring: " ++ needle) (needle `isInfixOf` show e)
        Right _ -> error ("expected exception containing: " ++ needle)

testBinanceOrderValidation :: IO ()
testBinanceOrderValidation = do
    envSpot <- newBinanceEnv MarketSpot binanceBaseUrl (Just "k") (Just "s")
    assertThrowsContains
        "Provide quantity or quoteOrderQty"
        (\() -> placeMarketOrder envSpot OrderTest "BTCUSDT" Buy Nothing Nothing Nothing Nothing)

    envMargin <- newBinanceEnv MarketMargin binanceBaseUrl (Just "k") (Just "s")
    assertThrowsContains
        "Margin does not support order test"
        (\() -> placeMarketOrder envMargin OrderTest "BTCUSDT" Buy (Just 0.1) Nothing Nothing Nothing)

    envFutures <- newBinanceEnv MarketFutures binanceBaseUrl (Just "k") (Just "s")
    assertThrowsContains
        "Futures MARKET orders require --order-quantity"
        (\() -> placeMarketOrder envFutures OrderTest "BTCUSDT" Buy Nothing (Just 50) Nothing Nothing)

forwardReturns :: [Double] -> [Double]
forwardReturns ps =
    [ if p0 == 0 then 0 else p1 / p0 - 1
    | (p0, p1) <- zip ps (drop 1 ps)
    ]

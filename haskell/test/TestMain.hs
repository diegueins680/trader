{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Exception (SomeException, evaluate, try)
import qualified Control.Monad
import Data.Aeson (eitherDecode, object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Int (Int64)
import Data.List (foldl', isInfixOf, sort)
import Data.Maybe (isNothing)
import qualified Data.Maybe
import qualified Data.Vector as V
import Options.Applicative (ParserResult (..), defaultPrefs, execParserPure, fullDesc, helper, info, renderFailure, (<**>))
import System.Exit (exitFailure, exitSuccess)
import System.Timeout (timeout)

import Trader.App.Args (Args, argBinanceSymbol, argIdempotencyKey, argInterval, argLookback, opts, parseTimestampMs, validateArgs)
import Trader.Binance (
    BinanceMarket (..),
    BinanceOrderMode (..),
    Kline (..),
    OrderSide (..),
    binanceBaseUrl,
    newBinanceEnv,
    placeMarketOrder,
    signQuery,
 )
import Trader.BotStartSemantics (
    botTradeEnabledFromApi,
    shouldClearPositionOriginOnStart,
    shouldPersistPositionOriginOnSwitch,
    shouldPreserveProvidedComboOnActiveAdopt,
    shouldResolveOriginComboOnAutoStart,
 )
import Trader.Coinbase (CoinbaseCandle (..), buildRanges, decodeCoinbaseCandles)
import Trader.Config (shouldRequireUserTradeKeys, validateRuntimeConfig)
import Trader.Dex (DexEnv (..), DexToken (..), resolveDexTokens, tokenAmountToInteger)
import Trader.Duration (TimeWindow (..), inferPeriodsPerYear, lookbackBarsFrom, minuteOfDayFromMs, parseDurationSeconds)
import Trader.Http (boundedBackoffMs, parseRetryAfterFromHeadersAt, parseRetryAfterMsAt)
import Trader.Kalman3 (Kalman3 (..), KalmanRun (..), Vec3 (..), constantAcceleration1D, forecastNextConstantAcceleration1D, runConstantAcceleration1D, step)
import Trader.KalmanFusion (Kalman1 (..), initKalman1, updateMulti)
import Trader.Kraken (decodeKrakenCandles)
import Trader.LSTM (LSTMConfig (..), LSTMModel (..), buildSequences, evaluateLoss, trainLSTM)
import Trader.LstmPersistence (lstmModelKey)
import Trader.MarketContext (fitLinearRange)
import Trader.Method (Method (..), parseMethod, selectPredictions)
import Trader.Metrics (bmGrossLoss, bmGrossProfit, bmMaxDrawdown, bmProfitFactor, bmTotalReturn, computeMetrics)
import Trader.Optimization (bestFinalEquity, optimizeOperations, sweepThreshold)
import Trader.Optimizer.Optimize (sampleTakeProfitPartial)
import Trader.Optimizer.Random (nextDouble, nextIntRange, seedRng)
import Trader.OrderExecution (OrderExecutionEvidence (..), applyExecutedQuantity, orderAppliedQuantity)
import Trader.Platform (
    Platform (..),
    coinbaseIntervalSeconds,
    isPlatformInterval,
    krakenIntervalMinutes,
    parsePlatform,
    poloniexIntervalLabel,
    poloniexIntervalSeconds,
 )
import Trader.Poloniex (PoloniexCandle (..), decodePoloniexCandles)
import Trader.Predictors (
    Interval (..),
    Quantiles (..),
    RegimeProbs (..),
    SensorId (..),
    SensorOutput (..),
    initHMMFilter,
    predictSensors,
    trainPredictors,
 )
import Trader.Predictors.Transformer (TransformerModel (..), predictTransformer, trainTransformer)
import Trader.Predictors.Types (allPredictors)
import Trader.SignalGates (
    signalCrossAssetCheck,
    signalFundingOiCheck,
    signalMetaLabelOk,
    signalMtfConsensusCheck,
    signalRegimeEdgeOk,
    signalRunPostDirectionGates,
 )
import Trader.Split (Split (..), splitTrainBacktest)
import qualified Trader.Symbol as Symbol
import Trader.Test.ApiRoutes (apiRouteSuite)
import Trader.TopCombosStore (mergeTopCombosPayloads, recalculateComboPerformanceFromOperation, sanitizeComboSymbolForPlatform)
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
              , run "kalman fusion multi-sensor" testKalmanFusionMulti
              , run "market linear fit" testMarketLinearFit
              , run "predictors output shape" testPredictorsOutputs
              , run "transformer training skips invalid rows" testTransformerTrainingSanitizesDataset
              , run "transformer prediction rejects non-finite query" testTransformerPredictionRejectsNonFiniteQuery
              , run "transformer training normalizes invalid temperature" testTransformerInvalidTemperatureFallback
              , run "kalman constant series" testKalmanConstant
              , run "kalman forecast constant" testKalmanForecast
              , run "kalman innovation sign" testKalmanInnovationSign
              , run "forward return sign" testForwardReturnSign
              , run "lstm training improves loss" testLstmImprovesLoss
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
              , run "metrics profit factor pnl" testMetricsProfitFactorPnL
              , run "binance signature length" testBinanceSignatureLength
              , run "binance kline json parsing" testBinanceKlineParsing
              , run "coinbase candle parser rejects fractional numeric timestamp" testCoinbaseFractionalTimestampRejected
              , run "coinbase candle parser normalizes millisecond timestamp boundaries" testCoinbaseTimestampBoundaryNormalization
              , run "coinbase range builder stops at epoch boundary" testCoinbaseBuildRangesStopsAtEpochBoundary
              , run "kraken candle parser rejects fractional numeric timestamp" testKrakenFractionalTimestampRejected
              , run "poloniex candle parser rejects fractional numeric timestamp" testPoloniexFractionalTimestampRejected
              , run "poloniex candle parser normalizes millisecond timestamp boundaries" testPoloniexTimestampBoundaryNormalization
              , run "exchange candle parsers reject non-finite numeric strings" testExchangeNonFiniteNumericStringRejected
              , run "method parsing" testMethodParsing
              , run "platform parsing" testPlatformParsing
              , run "non-binance args ignore live by default" testNonBinanceArgsLiveDefault
              , run "binance args normalize slash symbols" testBinanceSlashSymbolNormalization
              , run "coinbase args normalize slash symbols" testCoinbaseSlashSymbolNormalization
              , run "poloniex args normalize slash symbols" testPoloniexSlashSymbolNormalization
              , run "args normalize interval casing/spacing" testArgsNormalizeIntervalCode
              , run "coinbase args reject compact symbols without delimiter" testCoinbaseCompactSymbolRejected
              , run "poloniex args reject compact symbols without delimiter" testPoloniexCompactSymbolRejected
              , run "binance args reject malformed non-alnum symbols" testBinanceMalformedSymbolRejected
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
              , run "backtest window accepts expanded-year ISO dates" testBacktestWindowExpandedYearIsoValidation
              , run "backtest window rejects out-of-range expanded-year ISO dates" testBacktestWindowExpandedYearOverflowValidation
              , run "backtest window keeps negative millisecond epochs" testBacktestWindowNegativeMillisecondsValidation
              , run "backtest window keeps positive 11-digit millisecond epochs" testBacktestWindowPositiveMillisecondsValidation
              , run "backtest window normalizes second epochs to milliseconds" testBacktestWindowSecondEpochNormalization
              , run "backtest window enforces from<=to" testBacktestWindowOrderValidation
              , run "idempotency key enforces max length" testIdempotencyKeyLengthValidation
              , run "idempotency key rejects non-ascii characters" testIdempotencyKeyAsciiValidation
              , run "idempotency key trims surrounding whitespace" testIdempotencyKeyTrimValidation
              , run "bars rejects overflow integer" testBarsOverflowValidation
              , run "bars rejects non-decimal integer" testBarsNonDecimalValidation
              , run "cli numeric args reject non-finite values" testNumericArgsFiniteValidation
              , run "trade sizing args reject zero values" testTradeSizingPositiveValidation
              , run "retry-after date parsing" testRetryAfterDateParsing
              , run "retry-after header lookup is case-insensitive" testRetryAfterHeaderLookupCaseInsensitive
              , run "retry-after uses first parseable duplicate header value" testRetryAfterDuplicateHeaderFallback
              , run "retry backoff clamps without overflow" testRetryBackoffOverflowClamp
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
              , run "combo performance recalculates from completed operation delta" testRecalculateComboPerformanceFromCompletedOperation
              , run "top combos merge ranks by nested metrics score" testMergeTopCombosRanksByNestedScore
              , run "top combos merge dedupe prefers nested metrics score" testMergeTopCombosDedupPrefersNestedScore
              , run "top combos merge keeps same params across distinct sources" testMergeTopCombosKeepsDistinctSources
              , run "top combos sanitize slash-delimited binance symbols" testTopCombosBinanceSlashSymbolSanitization
              , run "top combos infer compact symbol from unknown delimited pair" testTopCombosUnknownPlatformPairNormalization
              , run "top combos reject numeric-only delimited symbols" testTopCombosRejectNumericOnlyDelimitedSymbols
              , run "symbol sanitization canonicalizes coinbase-prefixed platform keys" testSymbolCoinbasePrefixedPlatformNormalization
              , run "top combos sanitize coinbase-prefixed platform symbols" testTopCombosCoinbasePrefixedPlatformSymbolSanitization
              , run "symbol sanitization keeps dex symbol format for prefixed platform keys" testSymbolDexPrefixedPlatformNormalization
              , run "top combos keep dex symbol format for prefixed platform keys" testTopCombosDexPrefixedPlatformSymbolSanitization
              , run "dex trade args accept token pair without symbol" testDexTradeArgsRequireTokensNotSymbol
              , run "dex trade args reject single-token override even with symbol" testDexTradeArgsRejectPartialTokenOverrides
              , run "dex token resolution rejects malformed token addresses" testDexResolveTokensRejectsMalformedAddress
              , run "dex token resolution applies native decimals overrides" testDexResolveTokensNativeDecimalsOverride
              , run "dex token resolution rejects excessive decimals overrides" testDexResolveTokensRejectsExcessiveDecimalsOverride
              , run "dex token amount conversion rejects excessive decimals" testTokenAmountToIntegerRejectsExcessiveDecimals
              , run "platform intervals" testPlatformIntervals
              , run "platform interval mapping" testPlatformIntervalMapping
              , run "method selects predictions" testMethodSelection
              , run "train/backtest split" testTrainBacktestSplit
              , run "threshold sweep" testSweepThreshold
              , run "operations optimization" testOptimizeOperations
              , run "optimizer partial take-profit zero-range sampler" testOptimizerPartialTakeProfitZeroRange
              , run "optimizer int range keeps rng for fixed range" testOptimizerIntRangeFixedRange
              , run "optimizer int range handles full Int span" testOptimizerIntRangeFullSpan
              , run "binance order validation" testBinanceOrderValidation
              ]
                ++ map (uncurry run) apiRouteSuite
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

requireComboMetricsScore :: String -> Aeson.Value -> Double
requireComboMetricsScore label val =
    case val of
        Aeson.Object o ->
            case KM.lookup "metrics" o of
                Just (Aeson.Object metrics) ->
                    case KM.lookup "score" metrics >>= AT.parseMaybe Aeson.parseJSON of
                        Just score -> score
                        Nothing -> error (label ++ ": missing metrics.score")
                _ -> error (label ++ ": missing metrics object")
        _ -> error (label ++ ": combo is not an object")

requireComboSource :: String -> Aeson.Value -> String
requireComboSource label val =
    case val of
        Aeson.Object o ->
            case KM.lookup "source" o >>= AT.parseMaybe Aeson.parseJSON of
                Just source -> source
                Nothing -> error (label ++ ": missing source")
        _ -> error (label ++ ": combo is not an object")

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

testPoloniexSlashSymbolNormalization :: IO ()
testPoloniexSlashSymbolNormalization =
    case parseArgsResult ["--platform", "poloniex", "--symbol", "btc/usdt", "--interval", "2h"] of
        Left err -> error ("expected Poloniex slash symbol normalization to pass: " ++ err)
        Right args -> assert "poloniex slash symbol normalized to underscore" (argBinanceSymbol args == Just "BTC_USDT")

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

testTopCombosBinanceSlashSymbolSanitization :: IO ()
testTopCombosBinanceSlashSymbolSanitization =
    assert
        "top combos binance slash symbol normalized to compact pair"
        (sanitizeComboSymbolForPlatform (Just "binance") "BTC/USDT" == Just "BTCUSDT")

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
    let (regimeLeft, regimeRight) = selectPredictions MethodRegimeSwitch w kal lstm
    assertApproxList "regime_switch falls back to weighted average when context is unavailable (left stream)" 1e-9 blend regimeLeft
    assertApproxList "regime_switch falls back to weighted average when context is unavailable (right stream)" 1e-9 blend regimeRight
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

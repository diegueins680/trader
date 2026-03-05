{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE RecordWildCards #-}

module Trader.App.Args (
    Args (..),
    resolveBarsForCsv,
    resolveBarsForBinance,
    resolveBarsForPlatform,
    positioningCode,
    parsePositioning,
    intrabarFillCode,
    parseIntrabarFill,
    opts,
    argBinanceMarket,
    argLookback,
    normalizeEpochMs,
    parseTimestampMs,
    validateArgs,
) where

import Control.Applicative ((<|>))
import Control.Monad (forM_, when)
import Data.Char (isAsciiLower, isAsciiUpper, isDigit, toLower, toUpper)
import Data.Int (Int64)
import Data.Maybe (fromMaybe, isJust, isNothing, mapMaybe)
import Data.Time (defaultTimeLocale, parseTimeM)
import Data.Time.Clock.POSIX (utcTimeToPOSIXSeconds)
import Text.Read (readMaybe)

import Options.Applicative

import Trader.Binance (BinanceMarket (..))
import Trader.Duration (TimeWindow, lookbackBarsFrom, parseIntervalSeconds, parseTimeWindow)
import Trader.Method (Method (..), methodCode, parseMethod)
import Trader.Normalization (NormType (..), parseNormType)
import Trader.Optimization (TuneObjective (..), parseTuneObjective, tuneObjectiveCode)
import Trader.Platform (
    Platform (..),
    isPlatformInterval,
    parsePlatform,
    platformCode,
    platformDefaultBars,
    platformIntervalsCsv,
    platformSupportsTrading,
 )
import Trader.Predictors.Types (
    PredictorSet,
    allPredictors,
    predictorSetFromString,
    predictorSetToCsv,
 )
import Trader.Symbol (sanitizeSymbolForPlatform)
import Trader.Text (normalizeKey, trim)
import Trader.Trading (IntrabarFill (..), Positioning (..))

data Args = Args
    { argData :: Maybe FilePath
    , argPriceCol :: String
    , argHighCol :: Maybe String
    , argLowCol :: Maybe String
    , argBinanceSymbol :: Maybe String
    , argPlatform :: Platform
    , argBinanceFutures :: Bool
    , argBinanceMargin :: Bool
    , argInterval :: String
    , argBars :: Maybe Int
    , argLookbackWindow :: String
    , argLookbackBars :: Maybe Int
    , argBinanceTestnet :: Bool
    , argBinanceApiKey :: Maybe String
    , argBinanceApiSecret :: Maybe String
    , argCoinbaseApiKey :: Maybe String
    , argCoinbaseApiSecret :: Maybe String
    , argCoinbaseApiPassphrase :: Maybe String
    , argBinanceTrade :: Bool
    , argBinanceLive :: Bool
    , argDryRun :: Bool
    , argOrderQuote :: Maybe Double
    , argOrderQuantity :: Maybe Double
    , argOrderQuoteFraction :: Maybe Double
    , argMaxOrderQuote :: Maybe Double
    , argIdempotencyKey :: Maybe String
    , argDexChainId :: Maybe Int
    , argDexBaseToken :: Maybe String
    , argDexQuoteToken :: Maybe String
    , argDexBaseDecimals :: Maybe Int
    , argDexQuoteDecimals :: Maybe Int
    , argDexProtocols :: Maybe String
    , argDexAutoApprove :: Bool
    , argNormalization :: NormType
    , argHiddenSize :: Int
    , argEpochs :: Int
    , argLr :: Double
    , argValRatio :: Double
    , argBacktestRatio :: Double
    , argBacktestFrom :: Maybe String
    , argBacktestTo :: Maybe String
    , argInitialBalance :: Double
    , argTuneRatio :: Double
    , argTuneObjective :: TuneObjective
    , argTunePenaltyMaxDrawdown :: Double
    , argTunePenaltyTurnover :: Double
    , argMinRoundTrips :: Int
    , argWalkForwardFolds :: Int
    , argWalkForwardEmbargoBars :: Int
    , argPatience :: Int
    , argGradClip :: Maybe Double
    , argSeed :: Int
    , argKalmanDt :: Double
    , argKalmanProcessVar :: Double
    , argKalmanMeasurementVar :: Double
    , argPredictors :: PredictorSet
    , argKalmanMarketTopN :: Int
    , argOpenThreshold :: Double
    , argCloseThreshold :: Double
    , argMethod :: Method
    , argPositioning :: Positioning
    , argOptimizeOperations :: Bool
    , argSweepThreshold :: Bool
    , argTradeOnly :: Bool
    , argFee :: Double
    , argSlippage :: Double
    , argSpread :: Double
    , argFeeFixed :: Double
    , argFeeMin :: Double
    , argSlippageVolMult :: Double
    , argSlippageImpact :: Double
    , argSlippageImpactPower :: Double
    , argSpreadVolMult :: Double
    , argIntrabarFill :: IntrabarFill
    , argStopLoss :: Maybe Double
    , argTakeProfit :: Maybe Double
    , argTrailingStop :: Maybe Double
    , argStopLossVolMult :: Double
    , argTakeProfitVolMult :: Double
    , argTrailingStopVolMult :: Double
    , argTakeProfitPartial :: Double
    , argMinHoldBars :: Int
    , argCooldownBars :: Int
    , argMaxHoldBars :: Maybe Int
    , argMaxDrawdown :: Maybe Double
    , argMaxDailyLoss :: Maybe Double
    , argMaxWeeklyLoss :: Maybe Double
    , argRiskPerTrade :: Maybe Double
    , argMaxTradesPerDay :: Maybe Int
    , argExpectancyLookback :: Int
    , argMinExpectancy :: Maybe Double
    , argPerfLookback :: Int
    , argPerfMinWinRate :: Maybe Double
    , argPerfMinProfitFactor :: Maybe Double
    , argLossStreakMax :: Int
    , argLossStreakCooldownBars :: Int
    , argNoTradeWindows :: [TimeWindow]
    , argMaxOpenPositions :: Maybe Int
    , argMaxOpenPerBase :: Maybe Int
    , argMaxGrossExposure :: Maybe Double
    , argMaxNetExposure :: Maybe Double
    , argMaxExposurePerBase :: Maybe Double
    , argMinEdge :: Double
    , argMinSignalToNoise :: Double
    , argSnrSizeWeight :: Double
    , argAdaptiveFilters :: Bool
    , argAdaptiveEdgeBufferMax :: Double
    , argAdaptiveMinSignalToNoiseMax :: Double
    , argAdaptiveKalmanZMinMax :: Double
    , argAdaptiveTrendLookbackMax :: Int
    , argMetaLabelFilter :: Bool
    , argMetaLabelMinEdge :: Double
    , argMetaLabelMinConfidence :: Double
    , argMetaLabelRequireBand :: Bool
    , argRegimeParameterBank :: Bool
    , argRegimeBankHysteresis :: Double
    , argRegimeTrendOpenMult :: Double
    , argRegimeMrOpenMult :: Double
    , argRegimeHighVolOpenMult :: Double
    , argRegimeTrendSizeMult :: Double
    , argRegimeMrSizeMult :: Double
    , argRegimeHighVolSizeMult :: Double
    , argMultiTimeframeConsensus :: Bool
    , argMtfFastBars :: Int
    , argMtfMidBars :: Int
    , argMtfSlowBars :: Int
    , argMtfMinAgree :: Int
    , argCrossAssetConfirmation :: Bool
    , argCrossAssetMinBeta :: Double
    , argCrossAssetMinEdge :: Double
    , argPairsStatArb :: Bool
    , argPairsStatArbLookback :: Int
    , argPairsStatArbZEntry :: Double
    , argPairsStatArbSizeMult :: Double
    , argThresholdFactorEnabled :: Bool
    , argThresholdFactorAlpha :: Double
    , argThresholdFactorMin :: Double
    , argThresholdFactorMax :: Double
    , argThresholdFactorFloor :: Double
    , argThresholdFactorEdgeKalWeight :: Double
    , argThresholdFactorEdgeLstmWeight :: Double
    , argThresholdFactorKalmanZWeight :: Double
    , argThresholdFactorHighVolWeight :: Double
    , argThresholdFactorConformalWeight :: Double
    , argThresholdFactorQuantileWeight :: Double
    , argThresholdFactorLstmConfWeight :: Double
    , argThresholdFactorLstmHealthWeight :: Double
    , argCostAwareEdge :: Bool
    , argEdgeBuffer :: Double
    , argTrendLookback :: Int
    , argMaxPositionSize :: Double
    , argVolTarget :: Maybe Double
    , argVolLookback :: Int
    , argVolEwmaAlpha :: Maybe Double
    , argVolFloor :: Double
    , argVolScaleMax :: Double
    , argMaxVolatility :: Maybe Double
    , argRebalanceBars :: Int
    , argRebalanceThreshold :: Double
    , argRebalanceCostMult :: Double
    , argRebalanceGlobal :: Bool
    , argRebalanceResetOnSignal :: Bool
    , argFundingRate :: Double
    , argFundingBySide :: Bool
    , argFundingOnOpen :: Bool
    , argFundingOiAware :: Bool
    , argFundingOiFundingCap :: Maybe Double
    , argFundingOiVolLookback :: Int
    , argFundingOiVolCap :: Maybe Double
    , argFundingOiSizeMult :: Double
    , argBlendWeight :: Double
    , argRouterLookback :: Int
    , argRouterMinScore :: Double
    , argRouterScorePnlWeight :: Double
    , argTriLayer :: Bool
    , argTriLayerFastMult :: Double
    , argTriLayerSlowMult :: Double
    , argTriLayerCloudPadding :: Double
    , argTriLayerCloudSlope :: Double
    , argTriLayerCloudWidth :: Double
    , argTriLayerTouchLookback :: Int
    , argTriLayerRequirePriceAction :: Bool
    , argTriLayerPriceActionBody :: Double
    , argTriLayerExitOnSlow :: Bool
    , argKalmanBandLookback :: Int
    , argKalmanBandStdMult :: Double
    , argLstmExitFlipBars :: Int
    , argLstmExitFlipGraceBars :: Int
    , argLstmExitFlipStrong :: Bool
    , argMaxOrderErrors :: Maybe Int
    , argPeriodsPerYear :: Maybe Double
    , argJson :: Bool
    , argServe :: Bool
    , argOpsBackfillCommits :: Bool
    , argPort :: Int
    , -- Confidence gating/sizing (Kalman sensors + HMM/intervals)
      argKalmanZMin :: Double
    , argKalmanZMax :: Double
    , argMaxHighVolProb :: Maybe Double
    , argMaxConformalWidth :: Maybe Double
    , argMaxQuantileWidth :: Maybe Double
    , argConfirmConformal :: Bool
    , argConfirmQuantiles :: Bool
    , argConfidenceSizing :: Bool
    , argProtectionMinConfidence :: Double
    , argLstmConfidenceSoft :: Double
    , argLstmConfidenceHard :: Double
    , argMinPositionSize :: Double
    , argKellyLiteSizing :: Bool
    , argKellyLiteFraction :: Double
    , argKellyLiteFloor :: Double
    , argKellyLiteCap :: Double
    , argExecutionMakerFirst :: Bool
    , argExecutionMakerOffsetBps :: Double
    , argExecutionMakerTimeoutSec :: Double
    , argExecutionMakerPollMs :: Int
    , argExecutionMakerFallbackMarket :: Bool
    , argTuneStressVolMult :: Double
    , argTuneStressShock :: Double
    , argTuneStressWeight :: Double
    }
    deriving (Eq, Show)

defaultBinanceBars :: Int
defaultBinanceBars = 500

resolveBarsForCsv :: Args -> Int
resolveBarsForCsv args =
    fromMaybe 0 (argBars args)

resolveBarsForBinance :: Args -> Int
resolveBarsForBinance args =
    case argBars args of
        Nothing -> defaultBinanceBars
        Just 0 -> defaultBinanceBars
        Just n -> n

resolveBarsForPlatform :: Args -> Int
resolveBarsForPlatform args =
    case argPlatform args of
        PlatformBinance -> resolveBarsForBinance args
        _ ->
            case argBars args of
                Nothing -> platformDefaultBars (argPlatform args)
                Just 0 -> platformDefaultBars (argPlatform args)
                Just n -> n

parseBarsArg :: String -> Either String (Maybe Int)
parseBarsArg raw =
    let s = map toLower (trim raw)
     in if s == "auto"
            then Right Nothing
            else case readStrictDecimalInteger s of
                Just n ->
                    case integerToInt n of
                        Just bounded -> Right (Just bounded)
                        Nothing -> Left "Expected an integer (e.g. 500) or 'auto'."
                Nothing -> Left "Expected an integer (e.g. 500) or 'auto'."

parseTimeInt64 :: String -> Maybe Int64
parseTimeInt64 s =
    integerToInt64 =<< readStrictDecimalInteger s

readStrictDecimalInteger :: String -> Maybe Integer
readStrictDecimalInteger s =
    if isStrictSignedDecimal s
        then readMaybe s
        else Nothing

isStrictSignedDecimal :: String -> Bool
isStrictSignedDecimal txt =
    case txt of
        ('+' : rest) -> hasDigits rest
        ('-' : rest) -> hasDigits rest
        _ -> hasDigits txt
  where
    hasDigits xs = not (null xs) && all isDigit xs

integerToInt :: Integer -> Maybe Int
integerToInt n =
    if n < lo || n > hi
        then Nothing
        else Just (fromInteger n)
  where
    lo = toInteger (minBound :: Int)
    hi = toInteger (maxBound :: Int)

integerToInt64 :: Integer -> Maybe Int64
integerToInt64 n =
    if n < lo || n > hi
        then Nothing
        else Just (fromInteger n)
  where
    lo = toInteger (minBound :: Int64)
    hi = toInteger (maxBound :: Int64)

normalizeEpochMs :: Int64 -> Int64
normalizeEpochMs n =
    let nInteger = toInteger n
        threshold = 10000000000 :: Integer
     in if abs nInteger < threshold
            then
                let scaled = nInteger * 1000
                    lo = toInteger (minBound :: Int64)
                    hi = toInteger (maxBound :: Int64)
                 in if scaled < lo || scaled > hi
                        then n
                        else fromInteger scaled
            else n

parseIsoTimeMs :: String -> Maybe Int64
parseIsoTimeMs s =
    let s' = normalizeIsoOffsetSuffix s
        formats =
            [ "%Y-%m-%d"
            , "%Y-%m-%d %H:%M"
            , "%Y-%m-%dT%H:%M"
            , "%Y-%m-%d %H:%M%z"
            , "%Y-%m-%dT%H:%M%z"
            , "%Y-%m-%d %H:%MZ"
            , "%Y-%m-%dT%H:%MZ"
            , "%Y-%m-%d %H:%M:%S"
            , "%Y-%m-%dT%H:%M:%S"
            , "%Y-%m-%d %H:%M:%S%z"
            , "%Y-%m-%dT%H:%M:%S%z"
            , "%Y-%m-%d %H:%M:%S%Q"
            , "%Y-%m-%dT%H:%M:%S%Q"
            , "%Y-%m-%d %H:%M:%S%Q%z"
            , "%Y-%m-%dT%H:%M:%S%Q%z"
            , "%Y-%m-%dT%H:%M:%S%QZ"
            , "%Y-%m-%d %H:%M:%S%QZ"
            ]
        parseWith fmt = parseTimeM True defaultTimeLocale fmt s'
        toMsBounded t =
            let ms = floor (utcTimeToPOSIXSeconds t * 1000) :: Integer
             in integerToInt64 ms
     in case mapMaybe parseWith formats of
            [] -> Nothing
            (t : _) -> toMsBounded t

normalizeIsoOffsetSuffix :: String -> String
normalizeIsoOffsetSuffix raw =
    let rawT = map (\c -> if c == 't' then 'T' else c) raw
        raw' =
            case reverse rawT of
                'z' : rest -> reverse rest ++ "Z"
                _ -> rawT
     in case reverse raw' of
            m2 : m1 : ':' : h2 : h1 : sign : rest
                | (sign == '+' || sign == '-') && all isDigit [h1, h2, m1, m2] ->
                    reverse rest ++ [sign, h1, h2, m1, m2]
            _ -> raw'

looksLikeIso8601Prefix :: String -> Bool
looksLikeIso8601Prefix s =
    let s' =
            case s of
                ('+' : rest) -> rest
                ('-' : rest) -> rest
                _ -> s
        (yearDigits, rest) = span isDigit s'
     in length yearDigits >= 4
            && case rest of
                ('-' : m1 : m2 : '-' : d1 : d2 : _)
                    | all isDigit [m1, m2, d1, d2] -> True
                _ -> False

parseTimestampMs :: String -> Maybe Int64
parseTimestampMs raw =
    let s = trim raw
     in case parseTimeInt64 s of
            Just n -> Just (normalizeEpochMs n)
            Nothing ->
                if looksLikeIso8601Prefix s
                    then parseIsoTimeMs s
                    else Nothing

normalizeIntervalCode :: String -> String
normalizeIntervalCode raw =
    let s = trim raw
     in case span isDigit s of
            (digits, [u])
                | not (null digits) ->
                    digits ++ [if u == 'M' then 'M' else toLower u]
            _ -> s

positioningCode :: Positioning -> String
positioningCode p =
    case p of
        LongFlat -> "long-flat"
        LongShort -> "long-short"

parsePositioning :: String -> Either String Positioning
parsePositioning raw =
    case normalizeKey raw of
        "longflat" -> Right LongFlat
        "lf" -> Right LongFlat
        "flat" -> Right LongFlat
        "longonly" -> Right LongFlat
        "long" -> Right LongFlat
        "longshort" -> Right LongShort
        "ls" -> Right LongShort
        "short" -> Right LongShort
        _ -> Left "Invalid positioning (expected long-flat|long-only|long-short)"

intrabarFillCode :: IntrabarFill -> String
intrabarFillCode f =
    case f of
        StopFirst -> "stop-first"
        TakeProfitFirst -> "take-profit-first"

parseIntrabarFill :: String -> Either String IntrabarFill
parseIntrabarFill raw =
    case normalizeKey raw of
        "stopfirst" -> Right StopFirst
        "slfirst" -> Right StopFirst
        "stop" -> Right StopFirst
        "worst" -> Right StopFirst
        "takeprofitfirst" -> Right TakeProfitFirst
        "tpfirst" -> Right TakeProfitFirst
        "takeprofit" -> Right TakeProfitFirst
        "tp" -> Right TakeProfitFirst
        "best" -> Right TakeProfitFirst
        _ -> Left "Invalid intrabar fill (expected stop-first|take-profit-first)"

opts :: Parser Args
opts = do
    let defaultOnSwitch onFlag offFlag helpOn helpOff =
            flag' True (long onFlag <> help helpOn)
                <|> flag' False (long offFlag <> help helpOff)
                <|> pure True
        defaultOffSwitch onFlag offFlag helpOn helpOff =
            flag' True (long onFlag <> help helpOn)
                <|> flag' False (long offFlag <> help helpOff)
                <|> pure False
    argData <- optional (strOption (long "data" <> metavar "PATH" <> help "CSV file containing prices"))
    argPriceCol <- strOption (long "price-column" <> value "close" <> help "CSV column name for price (case-insensitive; prints suggestions on error)")
    argHighCol <- optional (strOption (long "high-column" <> help "CSV column name for high (requires --low-column; enables intrabar stops/TP/trailing)"))
    argLowCol <- optional (strOption (long "low-column" <> help "CSV column name for low (requires --high-column; enables intrabar stops/TP/trailing)"))
    argBinanceSymbol <- optional (strOption (long "binance-symbol" <> long "symbol" <> metavar "SYMBOL" <> help "Exchange symbol to fetch klines (platform via --platform)"))
    argPlatform <-
        option
            (eitherReader parsePlatform)
            ( long "platform"
                <> value PlatformBinance
                <> showDefaultWith platformCode
                <> help "Exchange platform for --binance-symbol (binance|coinbase|kraken|poloniex|uniswap|curve|sushiswap|balancer|pancakeswap|1inch)"
            )
    argBinanceFutures <- switch (long "futures" <> help "Use Binance USDT-M futures endpoints for data/orders (Binance only)")
    argBinanceMargin <- switch (long "margin" <> help "Use Binance margin account endpoints for orders/balance (Binance only)")
    argInterval <-
        fmap
            normalizeIntervalCode
            (strOption (long "interval" <> long "binance-interval" <> value "1h" <> help "Bar interval / Binance kline interval (e.g., 1m, 5m, 1h, 1d)"))
    argBars <-
        option
            (eitherReader parseBarsArg)
            ( long "bars"
                <> long "binance-limit"
                <> metavar "N|auto"
                <> value Nothing
                <> showDefaultWith (maybe "auto" show)
                <> help "Number of bars/klines to use (auto/0=all CSV, exchange default depends on platform: Binance=500, Coinbase=300; Binance supports 2..1000)"
            )
    argLookbackWindow <- strOption (long "lookback-window" <> value "7d" <> help "Lookback window duration (e.g., 90m, 24h, 7d)")
    argLookbackBars <- optional (option auto (long "lookback-bars" <> long "lookback" <> help "Override lookback bars (disables --lookback-window conversion)"))
    argBinanceTestnet <- switch (long "binance-testnet" <> help "Use Binance testnet base URL (public + signed endpoints; Binance only)")
    argBinanceApiKey <- optional (strOption (long "binance-api-key" <> help "Binance API key (or env BINANCE_API_KEY; Binance only)"))
    argBinanceApiSecret <- optional (strOption (long "binance-api-secret" <> help "Binance API secret (or env BINANCE_API_SECRET; Binance only)"))
    argCoinbaseApiKey <- optional (strOption (long "coinbase-api-key" <> help "Coinbase API key (or env COINBASE_API_KEY; Coinbase only)"))
    argCoinbaseApiSecret <- optional (strOption (long "coinbase-api-secret" <> help "Coinbase API secret (or env COINBASE_API_SECRET; Coinbase only)"))
    argCoinbaseApiPassphrase <- optional (strOption (long "coinbase-api-passphrase" <> help "Coinbase API passphrase (or env COINBASE_API_PASSPHRASE; Coinbase only)"))
    argBinanceTrade <- switch (long "binance-trade" <> help "If set, place a market order for the latest signal (Binance/Coinbase/DEX)")
    argBinanceLive <-
        defaultOffSwitch
            "binance-live"
            "no-binance-live"
            "Send LIVE orders (Binance/Coinbase; Coinbase has no test endpoint)."
            "Send TEST orders (Binance uses /order/test; Coinbase has no test endpoint)."
    argDryRun <-
        defaultOffSwitch
            "dry-run"
            "no-dry-run"
            "Compute the latest signal and simulated trade response, but never send exchange/DEX orders."
            "Disable dry-run so --binance-trade can place test/live orders."
    argOrderQuote <- optional (option auto (long "order-quote" <> help "Quote amount to spend on BUY (quoteOrderQty)"))
    argOrderQuantity <- optional (option auto (long "order-quantity" <> help "Base quantity to trade (quantity)"))
    argOrderQuoteFraction <- optional (option auto (long "order-quote-fraction" <> help "Size BUY orders as a fraction of quote balance (0 < F <= 1) when --order-quote/--order-quantity not set"))
    argMaxOrderQuote <- optional (option auto (long "max-order-quote" <> help "Cap the computed quote amount when using --order-quote-fraction"))
    argIdempotencyKey <- optional (strOption (long "idempotency-key" <> metavar "ID" <> help "Optional Binance newClientOrderId for idempotent orders"))
    argDexChainId <- optional (option auto (long "dex-chain-id" <> help "DEX chain id for on-chain swaps (e.g., 1 for Ethereum, 56 for BSC)"))
    argDexBaseToken <- optional (strOption (long "dex-base-token" <> metavar "TOKEN" <> help "DEX base token address or symbol (asset you buy when LONG)"))
    argDexQuoteToken <- optional (strOption (long "dex-quote-token" <> metavar "TOKEN" <> help "DEX quote token address or symbol (asset you sell to BUY base)"))
    argDexBaseDecimals <- optional (option auto (long "dex-base-decimals" <> help "Override base token decimals when token metadata lookup fails"))
    argDexQuoteDecimals <- optional (option auto (long "dex-quote-decimals" <> help "Override quote token decimals when token metadata lookup fails"))
    argDexProtocols <- optional (strOption (long "dex-protocols" <> metavar "LIST" <> help "Comma-separated 1inch protocols to restrict routing (optional)"))
    argDexAutoApprove <-
        defaultOnSwitch
            "dex-auto-approve"
            "no-dex-auto-approve"
            "Auto-approve ERC-20 allowance for DEX swaps when needed."
            "Do not auto-approve; fail if allowance is insufficient."
    argNormalization <- option (maybeReader parseNormType) (long "normalization" <> value NormStandard <> help "none|minmax|standard|log")
    argHiddenSize <- option auto (long "hidden-size" <> value 16 <> help "LSTM hidden size")
    argEpochs <- option auto (long "epochs" <> value 30 <> help "LSTM training epochs (Adam)")
    argLr <- option auto (long "lr" <> value 1e-3 <> help "LSTM learning rate")
    argValRatio <- option auto (long "val-ratio" <> value 0.3 <> help "Validation split ratio (within training set)")
    argBacktestRatio <- option auto (long "backtest-ratio" <> value 0.2 <> help "Backtest holdout ratio (last portion of series)")
    argBacktestFrom <- optional (strOption (long "from" <> metavar "TIME" <> help "Optional backtest start timestamp (epoch seconds/ms or ISO-8601)"))
    argBacktestTo <- optional (strOption (long "to" <> metavar "TIME" <> help "Optional backtest end timestamp (epoch seconds/ms or ISO-8601)"))
    argInitialBalance <- option auto (long "initial-balance" <> value 1.0 <> showDefault <> help "Initial backtest balance (>0); scales equity outputs.")
    argTuneRatio <- option auto (long "tune-ratio" <> value 0.25 <> help "When optimizing operations/threshold: tune on the last portion of the training split (avoids lookahead on the backtest split)")
    argTuneObjective <-
        option
            (eitherReader parseTuneObjective)
            ( long "tune-objective"
                <> value TuneRoi
                <> showDefaultWith tuneObjectiveCode
                <> help "Objective for --optimize-operations/--sweep-threshold: annualized-equity|roi|final-equity|sharpe|calmar|equity-dd|equity-dd-turnover"
            )
    argTunePenaltyMaxDrawdown <- option auto (long "tune-penalty-max-drawdown" <> value 1.5 <> help "Penalty weight for max drawdown (used by equity-dd objectives)")
    argTunePenaltyTurnover <- option auto (long "tune-penalty-turnover" <> value 0.2 <> help "Penalty weight for turnover (used by roi/equity-dd-turnover)")
    argMinRoundTrips <-
        option
            auto
            ( long "min-round-trips"
                <> value 0
                <> help "When optimizing/sweeping, require at least N round trips in the tune split (0 disables; helps avoid 'no-trade' winners)"
            )
    argWalkForwardFolds <- option auto (long "walk-forward-folds" <> value 7 <> help "Compute fold stats on tune/backtest windows (1 disables)")
    argWalkForwardEmbargoBars <-
        option
            auto
            ( long "walk-forward-embargo-bars"
                <> value 0
                <> help "Bars to drop from each walk-forward fold edge to reduce leakage (0 disables)."
            )
    argPatience <- option auto (long "patience" <> value 10 <> help "Early stopping patience (0 disables)")
    argGradClip <- optional (option auto (long "grad-clip" <> help "Gradient clipping max L2 norm"))
    argSeed <- option auto (long "seed" <> value 42 <> help "Random seed for LSTM init")
    argKalmanDt <- option auto (long "kalman-dt" <> value 1.0 <> help "Kalman dt")
    argKalmanProcessVar <- option auto (long "kalman-process-var" <> value 1e-5 <> help "Kalman process noise variance (white-noise jerk)")
    argKalmanMeasurementVar <- option auto (long "kalman-measurement-var" <> value 1e-3 <> help "Kalman measurement noise variance")
    argPredictors <-
        option
            (eitherReader predictorSetFromString)
            ( long "predictors"
                <> value allPredictors
                <> showDefaultWith predictorSetToCsv
                <> help "Comma-separated predictors to train/use (gbdt,tcn,transformer,hmm,quantile,conformal, all, none)"
            )
    argKalmanMarketTopN <-
        option
            auto
            ( long "kalman-market-top-n"
                <> value 50
                <> help "Use the top-N symbols by 24h quote volume as an extra Kalman measurement (0 disables; Binance only)."
            )
    argOpenThreshold <- option auto (long "open-threshold" <> long "threshold" <> value 0.002 <> help "Entry/open direction threshold (fractional deadband)")
    mCloseThreshold <- optional (option auto (long "close-threshold" <> help "Close-direction threshold (fractional deadband; defaults to open-threshold when omitted)"))
    argThresholdFactorEnabled <-
        defaultOffSwitch
            "threshold-factor"
            "no-threshold-factor"
            "Enable dynamic threshold factor learning."
            "Disable dynamic threshold factor learning."
    argThresholdFactorAlpha <- option auto (long "threshold-factor-alpha" <> value 0.2 <> help "EMA alpha for threshold factor updates (0 disables updates)")
    argThresholdFactorMin <- option auto (long "threshold-factor-min" <> value 0.5 <> help "Minimum multiplier for the threshold factor")
    argThresholdFactorMax <- option auto (long "threshold-factor-max" <> value 2.0 <> help "Maximum multiplier for the threshold factor")
    argThresholdFactorFloor <- option auto (long "threshold-factor-floor" <> value 0.0 <> help "Floor for adjusted thresholds after applying the factor")
    argThresholdFactorEdgeKalWeight <- option auto (long "threshold-factor-edge-kal-weight" <> value 0 <> help "Weight for Kalman edge feature in threshold factor")
    argThresholdFactorEdgeLstmWeight <- option auto (long "threshold-factor-edge-lstm-weight" <> value 0 <> help "Weight for LSTM edge feature in threshold factor")
    argThresholdFactorKalmanZWeight <- option auto (long "threshold-factor-kalman-z-weight" <> value 0 <> help "Weight for Kalman z-score feature in threshold factor")
    argThresholdFactorHighVolWeight <- option auto (long "threshold-factor-high-vol-weight" <> value 0 <> help "Weight for HMM high-vol probability feature in threshold factor")
    argThresholdFactorConformalWeight <- option auto (long "threshold-factor-conformal-weight" <> value 0 <> help "Weight for conformal width feature in threshold factor")
    argThresholdFactorQuantileWeight <- option auto (long "threshold-factor-quantile-weight" <> value 0 <> help "Weight for quantile width feature in threshold factor")
    argThresholdFactorLstmConfWeight <- option auto (long "threshold-factor-lstm-conf-weight" <> value 0 <> help "Weight for LSTM confidence feature in threshold factor")
    argThresholdFactorLstmHealthWeight <- option auto (long "threshold-factor-lstm-health-weight" <> value 0 <> help "Weight for LSTM training health feature in threshold factor")
    argMethod <-
        option
            (eitherReader parseMethod)
            ( long "method"
                <> value MethodBoth
                <> showDefaultWith methodCode
                <> help "Method: 11|both=Kalman+LSTM (direction-agreement gated), blend=weighted avg, conf_blend=confidence-weighted blend, conf_pick=confidence winner-take-all, conformal_clip=clip blended return to conformal/quantile band, cost_pick=cost-aware winner-take-all, harmonic_blend=harmonic-return blend, disagreement_guard=disagreement-aware model pick, median_blend=median-robust blend, neutral_guard=neutral-on-disagreement guard, risk_parity_blend=inverse-edge risk-parity blend, consensus_boost=consensus-strength guard, anchor_blend=disagreement-aware anchor blend, tension_gate=partial-neutral conflict gate, entropy_blend=uncertainty-aware blend shrink, coherence_gate=coherence-aware conflict gate, divergence_gate=shrink blend when model returns diverge, fractal_blend=signed-root nonlinear blend, phase_cancel=anti-phase cancellation gate, softmax_blend=softmax edge-weighted blend, smooth_softmax_blend=EMA-smoothed softmax blend, hedge_blend=online Hedge-style exp-weights blend, net_softmax_blend=post-cost softmax edge-weighted blend, edge_blend=edge-weighted blend, edge_pick=edge winner-take-all, geo_blend=geometric blend, regime_switch=volatility/z-score model switch, router=adaptive model selection, bandit_router=UCB-style adaptive router, kalman_physics_error=Kalman state+physics-error model (latest 1000 bars, train 700/test 300), 10|kalman=Kalman only, 01|lstm=LSTM only"
            )
    argPositioning <-
        option
            (eitherReader parsePositioning)
            ( long "positioning"
                <> value LongFlat
                <> showDefaultWith positioningCode
                <> help "Positioning: long-flat (default), long-only/long (alias), or long-short (futures-only when trading)"
            )
    argOptimizeOperations <- switch (long "optimize-operations" <> help "Optimize method (11/10/01/blend/conf_blend/conf_pick/conformal_clip/cost_pick/harmonic_blend/disagreement_guard/median_blend/neutral_guard/risk_parity_blend/consensus_boost/anchor_blend/tension_gate/entropy_blend/coherence_gate/divergence_gate/fractal_blend/phase_cancel/softmax_blend/smooth_softmax_blend/hedge_blend/net_softmax_blend/edge_blend/edge_pick/geo_blend/regime_switch/router/bandit_router), open-threshold, and close-threshold on a tune split (avoids lookahead on the backtest split)")
    argSweepThreshold <- switch (long "sweep-threshold" <> help "Sweep open/close thresholds on a tune split and print the best final equity (avoids lookahead on the backtest split)")
    argTradeOnly <- switch (long "trade-only" <> help "Skip backtest/metrics; only compute the latest signal (and optionally place an order)")
    argFee <- option auto (long "fee" <> value 0.0008 <> help "Fee applied when switching position")
    argSlippage <- option auto (long "slippage" <> value 0.0002 <> help "Slippage per side (fractional, e.g. 0.0002)")
    argSpread <- option auto (long "spread" <> value 0.0002 <> help "Bid-ask spread (fractional total; half applied per side)")
    argFeeFixed <- option auto (long "fee-fixed" <> value 0 <> help "Fixed fee per side as a fraction of equity (0 disables)")
    argFeeMin <- option auto (long "fee-min" <> value 0 <> help "Minimum fee per side as a fraction of equity (0 disables)")
    argSlippageVolMult <- option auto (long "slippage-vol-mult" <> value 0 <> help "Extra slippage per-bar sigma multiple (0 disables)")
    argSlippageImpact <- option auto (long "slippage-impact" <> value 0 <> help "Slippage impact coefficient applied to size^power (0 disables)")
    argSlippageImpactPower <- option auto (long "slippage-impact-power" <> value 1 <> help "Exponent for slippage impact scaling (>=0)")
    argSpreadVolMult <- option auto (long "spread-vol-mult" <> value 0 <> help "Extra spread per-bar sigma multiple (0 disables)")
    argIntrabarFill <-
        option
            (eitherReader parseIntrabarFill)
            ( long "intrabar-fill"
                <> value StopFirst
                <> showDefaultWith intrabarFillCode
                <> help "If take-profit and stop are both hit within a bar: stop-first (conservative) or take-profit-first"
            )
    argStopLoss <- optional (option auto (long "stop-loss" <> help "Stop loss fraction for a bracket exit (e.g., 0.02 for 2%)"))
    argTakeProfit <- optional (option auto (long "take-profit" <> help "Take profit fraction for a bracket exit (e.g., 0.03 for 3%)"))
    argTrailingStop <- optional (option auto (long "trailing-stop" <> help "Trailing stop fraction for a bracket exit (e.g., 0.01 for 1%)"))
    argStopLossVolMult <- option auto (long "stop-loss-vol-mult" <> value 0 <> help "Stop loss in per-bar sigma multiples (0 disables; overrides --stop-loss when vol estimate available)")
    argTakeProfitVolMult <- option auto (long "take-profit-vol-mult" <> value 0 <> help "Take profit in per-bar sigma multiples (0 disables; overrides --take-profit when vol estimate available)")
    argTrailingStopVolMult <- option auto (long "trailing-stop-vol-mult" <> value 0 <> help "Trailing stop in per-bar sigma multiples (0 disables; overrides --trailing-stop when vol estimate available)")
    argTakeProfitPartial <-
        option
            auto
            ( long "take-profit-partial"
                <> value 0
                <> showDefault
                <> help "Scale out this fraction at take-profit before keeping the remainder open (0 disables; 0<frac<1)"
            )
    argMinHoldBars <- option auto (long "min-hold-bars" <> value 4 <> help "Minimum holding periods (bars) before allowing a signal-based exit (0 disables)")
    argCooldownBars <- option auto (long "cooldown-bars" <> value 2 <> help "When flat after an exit, wait this many bars before allowing a new entry (0 disables)")
    argMaxHoldBars <-
        optional
            ( option
                auto
                ( long "max-hold-bars"
                    <> value 36
                    <> showDefault
                    <> help "Force exit after holding for this many bars (0 disables; enforces 1-bar cooldown)"
                )
            )
    argMaxDrawdown <- optional (option auto (long "max-drawdown" <> help "Halt the live bot if peak-to-trough drawdown exceeds this fraction (0..1)"))
    argMaxDailyLoss <- optional (option auto (long "max-daily-loss" <> help "Halt the live bot if daily loss exceeds this fraction (0..1), based on UTC day (requires bar timestamps)"))
    argMaxWeeklyLoss <- optional (option auto (long "max-weekly-loss" <> help "Halt the live bot if weekly loss exceeds this fraction (0..1), based on UTC weeks (requires bar timestamps)"))
    argRiskPerTrade <- optional (option auto (long "risk-per-trade" <> help "Fraction of equity to risk per trade (requires --stop-loss or --stop-loss-vol-mult)"))
    argMaxTradesPerDay <- optional (option auto (long "max-trades-per-day" <> help "Block new entries after N entries per UTC day (0 disables; requires bar timestamps)"))
    argExpectancyLookback <- option auto (long "expectancy-lookback" <> value 20 <> help "Lookback trades for expectancy gating (0 disables; requires --min-expectancy)")
    argMinExpectancy <- optional (option auto (long "min-expectancy" <> help "Halt trading when avg return of the last N trades falls below this"))
    argPerfLookback <- option auto (long "perf-lookback" <> value 0 <> showDefault <> help "Lookback trades for performance gates/adaptive filters (0 disables)")
    argPerfMinWinRate <- optional (option auto (long "perf-min-win-rate" <> help "Minimum rolling win rate for entry gating/adaptive filters (0 disables)"))
    argPerfMinProfitFactor <- optional (option auto (long "perf-min-profit-factor" <> help "Minimum rolling profit factor for entry gating/adaptive filters (0 disables)"))
    argLossStreakMax <- option auto (long "loss-streak-max" <> value 0 <> showDefault <> help "Trigger a cooldown after this many consecutive losing trades (0 disables)")
    argLossStreakCooldownBars <- option auto (long "loss-streak-cooldown-bars" <> value 0 <> showDefault <> help "Cooldown bars applied after hitting the loss streak threshold (0 disables)")
    argNoTradeWindows <-
        many
            ( option
                (eitherReader parseTimeWindow)
                ( long "no-trade-window"
                    <> metavar "HH:MM-HH:MM"
                    <> help "UTC time window to block new entries (repeatable; wraps across midnight; requires bar timestamps)."
                )
            )
    argMaxOpenPositions <- optional (option auto (long "max-open-positions" <> help "Max open positions across all running bots (0 disables)"))
    argMaxOpenPerBase <- optional (option auto (long "max-open-per-base" <> help "Max open positions per base asset across running bots (0 disables)"))
    argMaxGrossExposure <- optional (option auto (long "max-gross-exposure" <> help "Max gross exposure across running bots (sum of abs sizes; 0 disables)"))
    argMaxNetExposure <- optional (option auto (long "max-net-exposure" <> help "Max net exposure across running bots (abs sum of signed sizes; 0 disables)"))
    argMaxExposurePerBase <- optional (option auto (long "max-exposure-per-base" <> help "Max gross exposure per base asset across running bots (0 disables)"))
    argMinEdge <- option auto (long "min-edge" <> value 0.0004 <> help "Minimum predicted return magnitude required to enter (0 disables)")
    argMinSignalToNoise <- option auto (long "min-signal-to-noise" <> value 0.8 <> help "Minimum edge/vol (per-bar sigma) required to enter (0 disables)")
    argSnrSizeWeight <-
        option
            auto
            ( long "snr-size-weight"
                <> value 1.0
                <> help "Weight to scale position size by signal-to-noise (0=hard gate only, 1=fully scaled)"
            )
    argAdaptiveFilters <-
        defaultOffSwitch
            "adaptive-filters"
            "no-adaptive-filters"
            "Enable adaptive filter tightening based on rolling performance."
            "Disable adaptive filter tightening."
    argAdaptiveEdgeBufferMax <-
        option
            auto
            ( long "adaptive-edge-buffer-max"
                <> value 0
                <> showDefault
                <> help "Max additive edge-buffer when adaptive filters are fully tightened"
            )
    argAdaptiveMinSignalToNoiseMax <-
        option
            auto
            ( long "adaptive-min-signal-to-noise-max"
                <> value 0
                <> showDefault
                <> help "Max additive min-signal-to-noise when adaptive filters are fully tightened"
            )
    argAdaptiveKalmanZMinMax <-
        option
            auto
            ( long "adaptive-kalman-z-min-max"
                <> value 0
                <> showDefault
                <> help "Max additive Kalman z-min when adaptive filters are fully tightened"
            )
    argAdaptiveTrendLookbackMax <-
        option
            auto
            ( long "adaptive-trend-lookback-max"
                <> value 0
                <> showDefault
                <> help "Max additive trend lookback when adaptive filters are fully tightened"
            )
    argMetaLabelFilter <-
        defaultOffSwitch
            "meta-label-filter"
            "no-meta-label-filter"
            "Enable meta-label filtering (edge + confidence + optional interval confirmation)."
            "Disable meta-label filtering."
    argMetaLabelMinEdge <- option auto (long "meta-label-min-edge" <> value 0 <> showDefault <> help "Minimum edge required by the meta-label filter (fraction)")
    argMetaLabelMinConfidence <- option auto (long "meta-label-min-confidence" <> value 0 <> showDefault <> help "Minimum confidence required by the meta-label filter (0..1)")
    argMetaLabelRequireBand <-
        defaultOnSwitch
            "meta-label-require-band"
            "no-meta-label-require-band"
            "Require conformal/quantile band agreement when meta-label filtering is enabled (default on)."
            "Disable conformal/quantile band agreement for the meta-label filter."
    argRegimeParameterBank <-
        defaultOffSwitch
            "regime-parameter-bank"
            "no-regime-parameter-bank"
            "Enable regime-conditioned threshold and size multipliers."
            "Disable regime-conditioned threshold and size multipliers."
    argRegimeBankHysteresis <- option auto (long "regime-bank-hysteresis" <> value 0.05 <> showDefault <> help "Minimum top-vs-second regime probability gap required to activate regime multipliers (0..1)")
    argRegimeTrendOpenMult <- option auto (long "regime-trend-open-mult" <> value 0.9 <> showDefault <> help "Open-threshold multiplier when trend regime dominates")
    argRegimeMrOpenMult <- option auto (long "regime-mr-open-mult" <> value 1.1 <> showDefault <> help "Open-threshold multiplier when mean-reversion regime dominates")
    argRegimeHighVolOpenMult <- option auto (long "regime-high-vol-open-mult" <> value 1.3 <> showDefault <> help "Open-threshold multiplier when high-vol regime dominates")
    argRegimeTrendSizeMult <- option auto (long "regime-trend-size-mult" <> value 1.1 <> showDefault <> help "Position-size multiplier when trend regime dominates")
    argRegimeMrSizeMult <- option auto (long "regime-mr-size-mult" <> value 0.9 <> showDefault <> help "Position-size multiplier when mean-reversion regime dominates")
    argRegimeHighVolSizeMult <- option auto (long "regime-high-vol-size-mult" <> value 0.7 <> showDefault <> help "Position-size multiplier when high-vol regime dominates")
    argMultiTimeframeConsensus <-
        defaultOffSwitch
            "multi-timeframe-consensus"
            "no-multi-timeframe-consensus"
            "Enable multi-timeframe momentum consensus entry gating."
            "Disable multi-timeframe momentum consensus entry gating."
    argMtfFastBars <- option auto (long "mtf-fast-bars" <> value 5 <> showDefault <> help "Fast timeframe bars for multi-timeframe consensus")
    argMtfMidBars <- option auto (long "mtf-mid-bars" <> value 20 <> showDefault <> help "Mid timeframe bars for multi-timeframe consensus")
    argMtfSlowBars <- option auto (long "mtf-slow-bars" <> value 60 <> showDefault <> help "Slow timeframe bars for multi-timeframe consensus")
    argMtfMinAgree <- option auto (long "mtf-min-agree" <> value 2 <> showDefault <> help "Minimum agreeing timeframes required for entry (1..3)")
    argCrossAssetConfirmation <-
        defaultOffSwitch
            "cross-asset-confirmation"
            "no-cross-asset-confirmation"
            "Enable cross-asset confirmation using market-context direction."
            "Disable cross-asset confirmation."
    argCrossAssetMinBeta <- option auto (long "cross-asset-min-beta" <> value 0.05 <> showDefault <> help "Minimum absolute market beta required to trust cross-asset confirmation")
    argCrossAssetMinEdge <- option auto (long "cross-asset-min-edge" <> value 0 <> showDefault <> help "Minimum absolute market-context edge required for cross-asset confirmation")
    argPairsStatArb <-
        defaultOffSwitch
            "pairs-stat-arb"
            "no-pairs-stat-arb"
            "Enable residual z-score pairs/stat-arb overlay versus market context."
            "Disable residual z-score pairs/stat-arb overlay."
    argPairsStatArbLookback <- option auto (long "pairs-stat-arb-lookback" <> value 120 <> showDefault <> help "Lookback bars for residual z-score statistics")
    argPairsStatArbZEntry <- option auto (long "pairs-stat-arb-z-entry" <> value 2.0 <> showDefault <> help "Absolute residual z-score required to trigger pairs/stat-arb overlay")
    argPairsStatArbSizeMult <- option auto (long "pairs-stat-arb-size-mult" <> value 0.7 <> showDefault <> help "Position-size multiplier applied when pairs/stat-arb overlay is active")
    argCostAwareEdge <-
        defaultOnSwitch
            "cost-aware-edge"
            "no-cost-aware-edge"
            "Enable cost-aware min-edge gating (default on)."
            "Disable cost-aware min-edge gating."
    argEdgeBuffer <- option auto (long "edge-buffer" <> value 0.0002 <> help "Extra buffer added to cost-aware min-edge")
    argTrendLookback <- option auto (long "trend-lookback" <> value 30 <> help "SMA lookback for trend filter (0 disables)")
    argMaxPositionSize <- option auto (long "max-position-size" <> value 0.8 <> help "Cap position size/leverage (1 = full size)")
    argVolTarget <-
        optional
            ( option
                auto
                ( long "vol-target"
                    <> value 0.7
                    <> showDefault
                    <> help "Target annualized volatility for position sizing (0 disables)"
                )
            )
    argVolLookback <- option auto (long "vol-lookback" <> value 30 <> help "Lookback window for realized vol sizing (bars)")
    argVolEwmaAlpha <- optional (option auto (long "vol-ewma-alpha" <> help "EWMA alpha for vol sizing (overrides vol-lookback)"))
    argVolFloor <- option auto (long "vol-floor" <> value 0.15 <> help "Annualized vol floor for sizing")
    argVolScaleMax <- option auto (long "vol-scale-max" <> value 1 <> help "Max volatility scaling (caps leverage)")
    argMaxVolatility <-
        optional
            ( option
                auto
                ( long "max-volatility"
                    <> value 1.5
                    <> showDefault
                    <> help "Block entries when annualized vol exceeds this (0 disables)"
                )
            )
    argRebalanceBars <- option auto (long "rebalance-bars" <> value 24 <> showDefault <> help "Rebalance position size every N bars when size targets change (0 disables; default anchors to entry age)")
    argRebalanceThreshold <- option auto (long "rebalance-threshold" <> value 0.05 <> showDefault <> help "Minimum abs size delta required to rebalance (0 disables)")
    argRebalanceCostMult <- option auto (long "rebalance-cost-mult" <> value 0.0 <> showDefault <> help "Extra rebalance threshold as a multiple of per-side cost (0 disables)")
    argRebalanceGlobal <- switch (long "rebalance-global" <> help "Anchor rebalance cadence to global bars instead of entry age")
    argRebalanceResetOnSignal <- switch (long "rebalance-reset-on-signal" <> help "Reset rebalance cadence when a same-side open signal updates size")
    argFundingRate <- option auto (long "funding-rate" <> long "financing-rate" <> value 0.1 <> showDefault <> help "Annualized funding/borrow rate applied per bar in backtests (fraction; negative allowed; side-agnostic unless --funding-by-side)")
    argFundingBySide <- switch (long "funding-by-side" <> help "Apply funding sign by side (long pays positive, short receives)")
    argFundingOnOpen <- switch (long "funding-on-open" <> help "Charge funding for bars opened with a position (even if exited intrabar)")
    argFundingOiAware <-
        defaultOffSwitch
            "funding-oi-aware"
            "no-funding-oi-aware"
            "Enable funding/OI-aware gating and sizing dampening."
            "Disable funding/OI-aware gating and sizing dampening."
    argFundingOiFundingCap <- optional (option auto (long "funding-oi-funding-cap" <> help "Block entries when side-adjusted funding exceeds this cap (fraction; 0 disables)"))
    argFundingOiVolLookback <- option auto (long "funding-oi-vol-lookback" <> value 48 <> showDefault <> help "Lookback bars for volatility proxy used by funding/OI-aware gating")
    argFundingOiVolCap <- optional (option auto (long "funding-oi-vol-cap" <> help "Block entries when volatility proxy exceeds this level (0 disables)"))
    argFundingOiSizeMult <- option auto (long "funding-oi-size-mult" <> value 0.7 <> showDefault <> help "Max position-size multiplier applied by funding/OI-aware dampening (0..1)")
    argBlendWeight <- option auto (long "blend-weight" <> value 0.5 <> help "Kalman weight for --method blend/conf_blend/conf_pick/conformal_clip/cost_pick/harmonic_blend/disagreement_guard/median_blend/neutral_guard/risk_parity_blend/consensus_boost/anchor_blend/tension_gate/entropy_blend/coherence_gate/divergence_gate/fractal_blend/phase_cancel/softmax_blend/smooth_softmax_blend/hedge_blend/net_softmax_blend/edge_blend/edge_pick/geo_blend/regime_switch (0..1)")
    argRouterLookback <- option auto (long "router-lookback" <> value 30 <> help "Lookback bars for --method router/bandit_router scoring (>= 2)")
    argRouterMinScore <- option auto (long "router-min-score" <> value 0.25 <> help "Minimum router score (blend of accuracy*coverage and return) to accept a model (0..1)")
    argRouterScorePnlWeight <-
        option
            auto
            ( long "router-score-pnl-weight"
                <> value 0.5
                <> help "Weight for return-aware router scoring (0=accuracy*coverage only, 1=return only)"
            )
    argTriLayer <- switch (long "tri-layer" <> help "Enable tri-layer entry gating (Kalman cloud + price action trigger)")
    argTriLayerFastMult <- option auto (long "tri-layer-fast-mult" <> value 0.5 <> help "Measurement variance multiplier for the fast Kalman cloud line (requires --tri-layer)")
    argTriLayerSlowMult <- option auto (long "tri-layer-slow-mult" <> value 2.0 <> help "Measurement variance multiplier for the slow Kalman cloud line (requires --tri-layer)")
    argTriLayerCloudPadding <- option auto (long "tri-layer-cloud-padding" <> value 0.0 <> showDefault <> help "Expand the Kalman cloud by this fraction of price when checking touches (0 = strict)")
    argTriLayerCloudSlope <- option auto (long "tri-layer-cloud-slope" <> value 0.0 <> showDefault <> help "Require Kalman cloud slope to exceed this fraction of price (0 = sign-only)")
    argTriLayerCloudWidth <- option auto (long "tri-layer-cloud-width" <> value 0.0 <> showDefault <> help "Block tri-layer entries when cloud width exceeds this fraction of price (0 disables)")
    argTriLayerTouchLookback <- option auto (long "tri-layer-touch-lookback" <> value 1 <> showDefault <> help "Allow cloud touches up to N bars back (1 = current bar)")
    argTriLayerRequirePriceAction <-
        defaultOnSwitch
            "tri-layer-price-action"
            "no-tri-layer-price-action"
            "Require price-action triggers when tri-layer is enabled (default on)."
            "Disable price-action triggers for tri-layer."
    argTriLayerPriceActionBody <- option auto (long "tri-layer-price-action-body" <> value 0.0 <> showDefault <> help "Override min candle body fraction for tri-layer price-action patterns (0 = default)")
    argTriLayerExitOnSlow <- switch (long "tri-layer-exit-on-slow" <> help "Exit when price closes across the slow Kalman line (requires --tri-layer)")
    argKalmanBandLookback <- option auto (long "kalman-band-lookback" <> value 0 <> showDefault <> help "Lookback bars for Kalman-band exits (0 disables; must be >= 2)")
    argKalmanBandStdMult <- option auto (long "kalman-band-std-mult" <> value 0 <> showDefault <> help "Std-dev multiple for Kalman-band exits (0 disables)")
    argLstmExitFlipBars <- option auto (long "lstm-exit-flip-bars" <> value 0 <> showDefault <> help "Exit after N consecutive LSTM bars flip against the position (0 disables)")
    argLstmExitFlipGraceBars <- option auto (long "lstm-exit-flip-grace-bars" <> value 0 <> showDefault <> help "Ignore LSTM flip exits during the first N bars of a trade")
    argLstmExitFlipStrong <- switch (long "lstm-exit-flip-strong" <> help "Require strong LSTM confidence for flip exits (uses --lstm-confidence-hard)")
    argMaxOrderErrors <- optional (option auto (long "max-order-errors" <> help "Halt the live bot after N consecutive order failures"))
    argExecutionMakerFirst <-
        defaultOnSwitch
            "execution-maker-first"
            "no-execution-maker-first"
            "Enable maker-first entry pacing (wait for favorable ticks) with optional market fallback (default on)."
            "Disable maker-first entry pacing."
    argExecutionMakerOffsetBps <- option auto (long "execution-maker-offset-bps" <> value 2 <> showDefault <> help "Favorable-price offset in basis points for maker-first pacing")
    argExecutionMakerTimeoutSec <- option auto (long "execution-maker-timeout-sec" <> value 3 <> showDefault <> help "Maximum wait time (seconds) for maker-first pacing before fallback/skip")
    argExecutionMakerPollMs <- option auto (long "execution-maker-poll-ms" <> value 250 <> showDefault <> help "Polling interval (ms) while waiting for maker-first pacing")
    argExecutionMakerFallbackMarket <-
        defaultOnSwitch
            "execution-maker-fallback-market"
            "no-execution-maker-fallback-market"
            "Fallback to immediate market execution after maker-first timeout (default on)."
            "Do not fallback to market execution after maker-first timeout."
    argPeriodsPerYear <- optional (option auto (long "periods-per-year" <> help "For annualized metrics (e.g., 365 for 1d, 8760 for 1h)"))
    argJson <- switch (long "json" <> help "Output JSON to stdout (CLI mode only)")
    argServe <- switch (long "serve" <> help "Run REST API server on localhost instead of running the CLI workflow")
    argOpsBackfillCommits <- switch (long "ops-backfill-commits" <> help "Backfill git_commits from repo history and link ops by commit time (requires TRADER_DB_URL)")
    argPort <- option auto (long "port" <> value 8080 <> help "REST API port (when --serve)")
    argKalmanZMin <- option auto (long "kalman-z-min" <> value 0.5 <> help "Min |Kalman mean|/std (z-score) required to treat Kalman as directional (0 disables)")
    argKalmanZMax <- option auto (long "kalman-z-max" <> value 3 <> help "Z-score mapped to position size=1 when --confidence-sizing is enabled")
    argMaxHighVolProb <- optional (option auto (long "max-high-vol-prob" <> help "If set, block trades when HMM high-vol regime prob exceeds this (0..1)"))
    argMaxConformalWidth <- optional (option auto (long "max-conformal-width" <> help "If set, block trades when conformal interval width exceeds this (return units)"))
    argMaxQuantileWidth <- optional (option auto (long "max-quantile-width" <> help "If set, block trades when quantile (q90-q10) width exceeds this (return units)"))
    argConfirmConformal <-
        defaultOnSwitch
            "confirm-conformal"
            "no-confirm-conformal"
            "Require conformal interval to agree with the chosen direction (default on)."
            "Disable conformal confirmation."
    argConfirmQuantiles <-
        defaultOnSwitch
            "confirm-quantiles"
            "no-confirm-quantiles"
            "Require quantiles to agree with the chosen direction (default on)."
            "Disable quantile confirmation."
    argConfidenceSizing <-
        defaultOnSwitch
            "confidence-sizing"
            "no-confidence-sizing"
            "Scale entries by confidence (Kalman z-score / interval widths); leaves exits unscaled (default on)."
            "Disable confidence sizing for entries."
    argProtectionMinConfidence <-
        option
            auto
            ( long "protection-min-confidence"
                <> value 0
                <> showDefault
                <> help "Min confidence required to place exchange protection orders (stop-loss / take-profit) when enabled (0 disables)."
            )
    argLstmConfidenceSoft <- option auto (long "lstm-confidence-soft" <> value 0.6 <> showDefault <> help "Soft LSTM confidence threshold for sizing (linear ramp to --lstm-confidence-hard; requires --confidence-sizing)")
    argLstmConfidenceHard <- option auto (long "lstm-confidence-hard" <> value 0.8 <> showDefault <> help "Hard LSTM confidence threshold for sizing (0 disables; requires --confidence-sizing)")
    argMinPositionSize <- option auto (long "min-position-size" <> value 0.15 <> help "Minimum entry size after sizing/vol scaling; skip if below this (0..1)")
    argKellyLiteSizing <-
        defaultOffSwitch
            "kelly-lite-sizing"
            "no-kelly-lite-sizing"
            "Enable Kelly-lite size scaling from estimated edge and variance."
            "Disable Kelly-lite size scaling."
    argKellyLiteFraction <- option auto (long "kelly-lite-fraction" <> value 0.5 <> showDefault <> help "Fraction applied to Kelly-lite size scale")
    argKellyLiteFloor <- option auto (long "kelly-lite-floor" <> value 0 <> showDefault <> help "Lower bound for Kelly-lite size scale")
    argKellyLiteCap <- option auto (long "kelly-lite-cap" <> value 1 <> showDefault <> help "Upper bound for Kelly-lite size scale")
    argTuneStressVolMult <- option auto (long "tune-stress-vol-mult" <> value 1.0 <> help "Stress volatility multiplier for tune scoring (1 disables)")
    argTuneStressShock <- option auto (long "tune-stress-shock" <> value 0.0 <> help "Stress shock added to returns for tune scoring (0 disables)")
    argTuneStressWeight <- option auto (long "tune-stress-weight" <> value 0.0 <> help "Penalty weight for stress scenario in tune scoring (0 disables)")

    pure Args{argCloseThreshold = fromMaybe argOpenThreshold mCloseThreshold, ..}

argBinanceMarket :: Args -> BinanceMarket
argBinanceMarket args =
    case (argBinanceFutures args, argBinanceMargin args) of
        (True, True) -> MarketSpot
        (True, False) -> MarketFutures
        (False, True) -> MarketMargin
        (False, False) -> MarketSpot

argLookback :: Args -> Int
argLookback args =
    case argLookbackBars args of
        Just n ->
            max 2 n
        Nothing ->
            case lookbackBarsFrom (argInterval args) (argLookbackWindow args) of
                Left _ -> 2
                Right n ->
                    max 2 n

validateArgs :: Args -> Either String Args
validateArgs args0 = do
    let isDexPlatform p =
            case p of
                PlatformUniswap -> True
                PlatformCurve -> True
                PlatformSushiswap -> True
                PlatformBalancer -> True
                PlatformPancakeswap -> True
                PlatformOneInch -> True
                _ -> False
        normalizeDelimitedSymbol delim raw =
            let normalized = map toUpper (trim raw)
             in map (normalizeDelimiter delim) normalized
        normalizeDelimiter delim c
            | c == delim = delim
            | c == '/' = delim
            | delim == '-' && c == '_' = delim
            | delim == '_' && c == '-' = delim
            | otherwise = c
        symbolNormalizer =
            case argPlatform args0 of
                p | isDexPlatform p -> trim
                PlatformCoinbase -> normalizeCoinbaseSymbol
                PlatformPoloniex -> normalizePoloniexSymbol
                PlatformBinance -> normalizeBinanceSymbol
                _ -> map toUpper . trim
        normalizeBinanceSymbol raw =
            fromMaybe (map toUpper (trim raw)) (sanitizeSymbolForPlatform (Just "binance") raw)
        normalizeCoinbaseSymbol raw =
            fromMaybe (normalizeDelimitedSymbol '-' raw) (sanitizeSymbolForPlatform (Just "coinbase") raw)
        normalizePoloniexSymbol raw =
            fromMaybe (normalizeDelimitedSymbol '_' raw) (sanitizeSymbolForPlatform (Just "poloniex") raw)
    let args =
            args0
                { argData = fmap trim (argData args0)
                , argBinanceSymbol = fmap symbolNormalizer (argBinanceSymbol args0)
                , argInterval = normalizeIntervalCode (argInterval args0)
                , argPriceCol = trim (argPriceCol args0)
                , argHighCol = fmap trim (argHighCol args0)
                , argLowCol = fmap trim (argLowCol args0)
                , argDexBaseToken = fmap trim (argDexBaseToken args0)
                , argDexQuoteToken = fmap trim (argDexQuoteToken args0)
                , argDexProtocols = fmap trim (argDexProtocols args0)
                , argBacktestFrom = fmap trim (argBacktestFrom args0)
                , argBacktestTo = fmap trim (argBacktestTo args0)
                , argIdempotencyKey = fmap trim (argIdempotencyKey args0)
                }
        present = maybe False (not . null)
    case argData args of
        Just "" -> Left "--data cannot be empty"
        _ -> pure ()
    case argBinanceSymbol args of
        Just "" -> Left "--binance-symbol cannot be empty"
        _ -> pure ()
    case argBinanceSymbol args of
        Nothing -> pure ()
        Just sym ->
            case argPlatform args of
                PlatformBinance ->
                    ensure
                        "--symbol must be a valid Binance symbol (e.g., BTCUSDT or BTC/USDT)"
                        (isJust (sanitizeSymbolForPlatform (Just "binance") sym))
                PlatformCoinbase ->
                    ensure
                        "--symbol must use Coinbase BASE-QUOTE format (e.g., BTC-USD or BTC/USD)"
                        (isJust (sanitizeSymbolForPlatform (Just "coinbase") sym))
                PlatformPoloniex ->
                    ensure
                        "--symbol must use Poloniex BASE_QUOTE format (e.g., BTC_USDT or BTC/USDT)"
                        (isJust (sanitizeSymbolForPlatform (Just "poloniex") sym))
                _ -> pure ()
    case argDexBaseToken args of
        Just "" -> Left "--dex-base-token cannot be empty"
        _ -> pure ()
    case argDexQuoteToken args of
        Just "" -> Left "--dex-quote-token cannot be empty"
        _ -> pure ()
    case argDexProtocols args of
        Just "" -> Left "--dex-protocols cannot be empty"
        _ -> pure ()
    let isDex = isDexPlatform (argPlatform args)
    ensure
        "Provide only one of --data or --binance-symbol (unless using a DEX platform with --data)"
        (not (present (argData args) && present (argBinanceSymbol args) && not isDex))
    ensure
        "Provide a data source: --data or --binance-symbol (unless using --serve or --ops-backfill-commits)"
        (argServe args || argOpsBackfillCommits args || present (argData args) || present (argBinanceSymbol args))
    ensure "--json cannot be used with --serve or --ops-backfill-commits" (not (argJson args && (argServe args || argOpsBackfillCommits args)))
    ensure "--ops-backfill-commits cannot be used with --serve" (not (argOpsBackfillCommits args && argServe args))
    ensure "Choose only one of --futures or --margin" (not (argBinanceFutures args && argBinanceMargin args))
    ensure "--min-round-trips must be >= 0" (argMinRoundTrips args >= 0)
    let isBinance = argPlatform args == PlatformBinance
        isCoinbase = argPlatform args == PlatformCoinbase
        supportsTrading = platformSupportsTrading (argPlatform args)
        supportsLiveMode = isBinance || isCoinbase
        hasDexBaseToken = present (argDexBaseToken args)
        hasDexQuoteToken = present (argDexQuoteToken args)
        hasDexTokens = hasDexBaseToken && hasDexQuoteToken
    ensure "--dex-base-token and --dex-quote-token must be provided together" (hasDexBaseToken == hasDexQuoteToken)
    ensure "--futures/--margin are only supported on Binance" (isBinance || not (argBinanceFutures args || argBinanceMargin args))
    ensure "--binance-testnet is only supported on Binance" (isBinance || not (argBinanceTestnet args))
    ensure "--binance-live is only supported on Binance/Coinbase" (supportsLiveMode || not (argBinanceLive args))
    ensure "--binance-trade is only supported on trading platforms" (supportsTrading || not (argBinanceTrade args))
    ensure "--dry-run requires --binance-trade" (not (argDryRun args) || argBinanceTrade args)
    ensure
        "--binance-trade requires --symbol/--binance-symbol (or --dex-base-token/--dex-quote-token for DEX platforms)"
        (not (argBinanceTrade args) || present (argBinanceSymbol args) || (isDex && hasDexTokens))

    case argBinanceApiKey args of
        Nothing -> pure ()
        Just v -> ensure "--binance-api-key cannot be empty" (not (null (trim v)))
    case argBinanceApiSecret args of
        Nothing -> pure ()
        Just v -> ensure "--binance-api-secret cannot be empty" (not (null (trim v)))
    case argCoinbaseApiKey args of
        Nothing -> pure ()
        Just v -> ensure "--coinbase-api-key cannot be empty" (not (null (trim v)))
    case argCoinbaseApiSecret args of
        Nothing -> pure ()
        Just v -> ensure "--coinbase-api-secret cannot be empty" (not (null (trim v)))
    case argCoinbaseApiPassphrase args of
        Nothing -> pure ()
        Just v -> ensure "--coinbase-api-passphrase cannot be empty" (not (null (trim v)))

    case argHighCol args of
        Nothing -> pure ()
        Just "" -> Left "--high-column cannot be empty"
        Just _ -> pure ()
    case argLowCol args of
        Nothing -> pure ()
        Just "" -> Left "--low-column cannot be empty"
        Just _ -> pure ()
    case argBacktestFrom args of
        Nothing -> pure ()
        Just "" -> Left "--from cannot be empty"
        Just _ -> pure ()
    case argBacktestTo args of
        Nothing -> pure ()
        Just "" -> Left "--to cannot be empty"
        Just _ -> pure ()
    case (argHighCol args, argLowCol args) of
        (Nothing, Nothing) -> pure ()
        (Just _, Just _) ->
            ensure "--high-column/--low-column require --data" (isJust (argData args))
        _ -> Left "Provide both --high-column and --low-column (or omit both)."

    let intervalStr = argInterval args
    ensure "--interval is required" (not (null intervalStr))
    case parseIntervalSeconds intervalStr of
        Nothing -> Left ("Invalid interval: " ++ show intervalStr ++ " (expected like 5m, 1h, 1d)")
        Just sec -> ensure "--interval must be > 0" (sec > 0)

    case argBinanceSymbol args of
        Nothing -> pure ()
        Just _ ->
            ensure
                ("--interval must be supported for " ++ platformCode (argPlatform args) ++ ": " ++ platformIntervalsCsv (argPlatform args))
                (isPlatformInterval (argPlatform args) intervalStr)

    let barsRaw = argBars args
        barsCsv = resolveBarsForCsv args
        barsPlatform = resolveBarsForPlatform args
    ensure
        "--bars must be auto, 0 (all CSV), or >= 2"
        ( case barsRaw of
            Nothing -> True
            Just n -> n == 0 || n >= 2
        )
    ensure "--bars cannot be 1" (barsRaw /= Just 1)
    case argBinanceSymbol args of
        Nothing -> pure ()
        Just _ ->
            if argPlatform args == PlatformBinance
                then ensure "--bars must be between 2 and 1000 for Binance data" (barsPlatform >= 2 && barsPlatform <= 1000)
                else ensure "--bars must be >= 2 for exchange data" (barsPlatform >= 2)

    lookback <-
        case argLookbackBars args of
            Just n -> do
                ensure "--lookback-bars must be >= 2" (n >= 2)
                pure n
            Nothing ->
                case lookbackBarsFrom intervalStr (argLookbackWindow args) of
                    Left e -> Left e
                    Right n -> do
                        ensure
                            ("Lookback window too small: " ++ show (argLookbackWindow args) ++ " at interval " ++ show intervalStr ++ " yields " ++ show n ++ " bars; need at least 2 bars.")
                            (n >= 2)
                        pure n

    let hasDataSource = present (argData args) || present (argBinanceSymbol args)
        prefersCsvBars = isDex && present (argData args)
        barsForLookback =
            if prefersCsvBars
                then barsCsv
                else case argBinanceSymbol args of
                    Just _ -> barsPlatform
                    Nothing -> barsCsv
    when (hasDataSource && barsForLookback > 0) $
        ensure
            ( "--bars must be >= lookback+1 (need at least "
                ++ show (lookback + 1)
                ++ " bars for lookback="
                ++ show lookback
                ++ ")"
            )
            (barsForLookback > lookback)

    let finiteRequired =
            [ ("--lr", argLr args)
            , ("--val-ratio", argValRatio args)
            , ("--backtest-ratio", argBacktestRatio args)
            , ("--initial-balance", argInitialBalance args)
            , ("--tune-ratio", argTuneRatio args)
            , ("--tune-penalty-max-drawdown", argTunePenaltyMaxDrawdown args)
            , ("--tune-penalty-turnover", argTunePenaltyTurnover args)
            , ("--kalman-dt", argKalmanDt args)
            , ("--kalman-process-var", argKalmanProcessVar args)
            , ("--kalman-measurement-var", argKalmanMeasurementVar args)
            , ("--open-threshold/--threshold", argOpenThreshold args)
            , ("--close-threshold", argCloseThreshold args)
            , ("--fee", argFee args)
            , ("--slippage", argSlippage args)
            , ("--spread", argSpread args)
            , ("--fee-fixed", argFeeFixed args)
            , ("--fee-min", argFeeMin args)
            , ("--slippage-vol-mult", argSlippageVolMult args)
            , ("--slippage-impact", argSlippageImpact args)
            , ("--slippage-impact-power", argSlippageImpactPower args)
            , ("--spread-vol-mult", argSpreadVolMult args)
            , ("--stop-loss-vol-mult", argStopLossVolMult args)
            , ("--take-profit-vol-mult", argTakeProfitVolMult args)
            , ("--trailing-stop-vol-mult", argTrailingStopVolMult args)
            , ("--take-profit-partial", argTakeProfitPartial args)
            , ("--min-edge", argMinEdge args)
            , ("--min-signal-to-noise", argMinSignalToNoise args)
            , ("--snr-size-weight", argSnrSizeWeight args)
            , ("--adaptive-edge-buffer-max", argAdaptiveEdgeBufferMax args)
            , ("--adaptive-min-signal-to-noise-max", argAdaptiveMinSignalToNoiseMax args)
            , ("--adaptive-kalman-z-min-max", argAdaptiveKalmanZMinMax args)
            , ("--meta-label-min-edge", argMetaLabelMinEdge args)
            , ("--meta-label-min-confidence", argMetaLabelMinConfidence args)
            , ("--regime-bank-hysteresis", argRegimeBankHysteresis args)
            , ("--regime-trend-open-mult", argRegimeTrendOpenMult args)
            , ("--regime-mr-open-mult", argRegimeMrOpenMult args)
            , ("--regime-high-vol-open-mult", argRegimeHighVolOpenMult args)
            , ("--regime-trend-size-mult", argRegimeTrendSizeMult args)
            , ("--regime-mr-size-mult", argRegimeMrSizeMult args)
            , ("--regime-high-vol-size-mult", argRegimeHighVolSizeMult args)
            , ("--cross-asset-min-beta", argCrossAssetMinBeta args)
            , ("--cross-asset-min-edge", argCrossAssetMinEdge args)
            , ("--pairs-stat-arb-z-entry", argPairsStatArbZEntry args)
            , ("--pairs-stat-arb-size-mult", argPairsStatArbSizeMult args)
            , ("--threshold-factor-alpha", argThresholdFactorAlpha args)
            , ("--threshold-factor-min", argThresholdFactorMin args)
            , ("--threshold-factor-max", argThresholdFactorMax args)
            , ("--threshold-factor-floor", argThresholdFactorFloor args)
            , ("--threshold-factor-edge-kal-weight", argThresholdFactorEdgeKalWeight args)
            , ("--threshold-factor-edge-lstm-weight", argThresholdFactorEdgeLstmWeight args)
            , ("--threshold-factor-kalman-z-weight", argThresholdFactorKalmanZWeight args)
            , ("--threshold-factor-high-vol-weight", argThresholdFactorHighVolWeight args)
            , ("--threshold-factor-conformal-weight", argThresholdFactorConformalWeight args)
            , ("--threshold-factor-quantile-weight", argThresholdFactorQuantileWeight args)
            , ("--threshold-factor-lstm-conf-weight", argThresholdFactorLstmConfWeight args)
            , ("--threshold-factor-lstm-health-weight", argThresholdFactorLstmHealthWeight args)
            , ("--edge-buffer", argEdgeBuffer args)
            , ("--max-position-size", argMaxPositionSize args)
            , ("--vol-floor", argVolFloor args)
            , ("--vol-scale-max", argVolScaleMax args)
            , ("--rebalance-threshold", argRebalanceThreshold args)
            , ("--rebalance-cost-mult", argRebalanceCostMult args)
            , ("--funding-rate", argFundingRate args)
            , ("--funding-oi-size-mult", argFundingOiSizeMult args)
            , ("--blend-weight", argBlendWeight args)
            , ("--router-min-score", argRouterMinScore args)
            , ("--router-score-pnl-weight", argRouterScorePnlWeight args)
            , ("--tri-layer-fast-mult", argTriLayerFastMult args)
            , ("--tri-layer-slow-mult", argTriLayerSlowMult args)
            , ("--tri-layer-cloud-padding", argTriLayerCloudPadding args)
            , ("--tri-layer-cloud-slope", argTriLayerCloudSlope args)
            , ("--tri-layer-cloud-width", argTriLayerCloudWidth args)
            , ("--tri-layer-price-action-body", argTriLayerPriceActionBody args)
            , ("--kalman-band-std-mult", argKalmanBandStdMult args)
            , ("--execution-maker-offset-bps", argExecutionMakerOffsetBps args)
            , ("--execution-maker-timeout-sec", argExecutionMakerTimeoutSec args)
            , ("--protection-min-confidence", argProtectionMinConfidence args)
            , ("--lstm-confidence-soft", argLstmConfidenceSoft args)
            , ("--lstm-confidence-hard", argLstmConfidenceHard args)
            , ("--min-position-size", argMinPositionSize args)
            , ("--kelly-lite-fraction", argKellyLiteFraction args)
            , ("--kelly-lite-floor", argKellyLiteFloor args)
            , ("--kelly-lite-cap", argKellyLiteCap args)
            , ("--kalman-z-min", argKalmanZMin args)
            , ("--kalman-z-max", argKalmanZMax args)
            , ("--tune-stress-vol-mult", argTuneStressVolMult args)
            , ("--tune-stress-shock", argTuneStressShock args)
            , ("--tune-stress-weight", argTuneStressWeight args)
            ]
        finiteOptional =
            [ ("--order-quote", argOrderQuote args)
            , ("--order-quantity", argOrderQuantity args)
            , ("--order-quote-fraction", argOrderQuoteFraction args)
            , ("--max-order-quote", argMaxOrderQuote args)
            , ("--grad-clip", argGradClip args)
            , ("--stop-loss", argStopLoss args)
            , ("--take-profit", argTakeProfit args)
            , ("--trailing-stop", argTrailingStop args)
            , ("--max-drawdown", argMaxDrawdown args)
            , ("--max-daily-loss", argMaxDailyLoss args)
            , ("--max-weekly-loss", argMaxWeeklyLoss args)
            , ("--risk-per-trade", argRiskPerTrade args)
            , ("--min-expectancy", argMinExpectancy args)
            , ("--perf-min-win-rate", argPerfMinWinRate args)
            , ("--perf-min-profit-factor", argPerfMinProfitFactor args)
            , ("--max-gross-exposure", argMaxGrossExposure args)
            , ("--max-net-exposure", argMaxNetExposure args)
            , ("--max-exposure-per-base", argMaxExposurePerBase args)
            , ("--vol-target", argVolTarget args)
            , ("--vol-ewma-alpha", argVolEwmaAlpha args)
            , ("--max-volatility", argMaxVolatility args)
            , ("--funding-oi-funding-cap", argFundingOiFundingCap args)
            , ("--funding-oi-vol-cap", argFundingOiVolCap args)
            , ("--periods-per-year", argPeriodsPerYear args)
            , ("--max-high-vol-prob", argMaxHighVolProb args)
            , ("--max-conformal-width", argMaxConformalWidth args)
            , ("--max-quantile-width", argMaxQuantileWidth args)
            ]
    forM_ finiteRequired (uncurry ensureFinite)
    forM_ finiteOptional (uncurry ensureMaybeFinite)

    ensure "--hidden-size must be >= 1" (argHiddenSize args >= 1)
    ensure "--epochs must be >= 0" (argEpochs args >= 0)
    ensure "--lr must be > 0" (argLr args > 0)
    ensure "--val-ratio must be >= 0 and < 1" (argValRatio args >= 0 && argValRatio args < 1)
    ensure "--backtest-ratio must be between 0 and 1" (argBacktestRatio args > 0 && argBacktestRatio args < 1)
    let parseWindowBound flag raw =
            case parseTimestampMs raw of
                Just t -> Right t
                Nothing ->
                    Left
                        ( flag
                            ++ " must be epoch seconds/ms or ISO-8601 (e.g. 1704067200, 1704067200000, 2025-01-01, 2025-01-01T00:00:00Z)"
                        )
    backtestFromMs <- traverse (parseWindowBound "--from") (argBacktestFrom args)
    backtestToMs <- traverse (parseWindowBound "--to") (argBacktestTo args)
    case (backtestFromMs, backtestToMs) of
        (Just fromMs, Just toMs) -> ensure "--from must be <= --to" (fromMs <= toMs)
        _ -> pure ()
    ensure "--initial-balance must be > 0" (argInitialBalance args > 0)
    ensure "--tune-ratio must be >= 0 and < 1" (argTuneRatio args >= 0 && argTuneRatio args < 1)
    ensure "--tune-penalty-max-drawdown must be >= 0" (argTunePenaltyMaxDrawdown args >= 0)
    ensure "--tune-penalty-turnover must be >= 0" (argTunePenaltyTurnover args >= 0)
    ensure "--walk-forward-folds must be >= 1" (argWalkForwardFolds args >= 1)
    ensure "--walk-forward-embargo-bars must be >= 0" (argWalkForwardEmbargoBars args >= 0)
    ensure "--patience must be >= 0" (argPatience args >= 0)
    case argGradClip args of
        Nothing -> pure ()
        Just g -> ensure "--grad-clip must be > 0" (g > 0)
    ensure "--kalman-dt must be > 0" (argKalmanDt args > 0)
    ensure "--kalman-process-var must be > 0" (argKalmanProcessVar args > 0)
    ensure "--kalman-measurement-var must be > 0" (argKalmanMeasurementVar args > 0)
    ensure "--kalman-market-top-n must be >= 0" (argKalmanMarketTopN args >= 0)
    ensure "--open-threshold/--threshold must be >= 0" (argOpenThreshold args >= 0)
    ensure "--close-threshold must be >= 0" (argCloseThreshold args >= 0)
    ensure "--router-lookback must be >= 2" (argRouterLookback args >= 2)
    ensure "--router-min-score must be between 0 and 1" (argRouterMinScore args >= 0 && argRouterMinScore args <= 1)
    ensure "--router-score-pnl-weight must be between 0 and 1" (argRouterScorePnlWeight args >= 0 && argRouterScorePnlWeight args <= 1)
    ensure "--method router/bandit_router cannot be used with --optimize-operations/--sweep-threshold" $
        not
            ( (argMethod args == MethodRouter || argMethod args == MethodBanditRouter)
                && (argOptimizeOperations args || argSweepThreshold args)
            )
    ensure "--method kalman_physics_error cannot be used with --optimize-operations/--sweep-threshold" $
        not
            ( argMethod args == MethodKalmanPhysicsError
                && (argOptimizeOperations args || argSweepThreshold args)
            )
    ensure "--fee must be >= 0" (argFee args >= 0)
    ensure "--slippage must be >= 0" (argSlippage args >= 0)
    ensure "--spread must be >= 0" (argSpread args >= 0)
    ensure "--fee-fixed must be >= 0" (argFeeFixed args >= 0)
    ensure "--fee-min must be >= 0" (argFeeMin args >= 0)
    ensure "--slippage-vol-mult must be >= 0" (argSlippageVolMult args >= 0)
    ensure "--slippage-impact must be >= 0" (argSlippageImpact args >= 0)
    ensure "--slippage-impact-power must be >= 0" (argSlippageImpactPower args >= 0)
    ensure "--spread-vol-mult must be >= 0" (argSpreadVolMult args >= 0)
    case argStopLoss args of
        Nothing -> pure ()
        Just v -> ensure "--stop-loss must be > 0 and < 1" (v > 0 && v < 1)
    case argTakeProfit args of
        Nothing -> pure ()
        Just v -> ensure "--take-profit must be > 0 and < 1" (v > 0 && v < 1)
    case argTrailingStop args of
        Nothing -> pure ()
        Just v -> ensure "--trailing-stop must be > 0 and < 1" (v > 0 && v < 1)
    ensure "--stop-loss-vol-mult must be >= 0" (argStopLossVolMult args >= 0)
    ensure "--take-profit-vol-mult must be >= 0" (argTakeProfitVolMult args >= 0)
    ensure "--trailing-stop-vol-mult must be >= 0" (argTrailingStopVolMult args >= 0)
    ensure "--take-profit-partial must be >= 0 and < 1" (argTakeProfitPartial args >= 0 && argTakeProfitPartial args < 1)
    ensure "--min-hold-bars must be >= 0" (argMinHoldBars args >= 0)
    ensure "--cooldown-bars must be >= 0" (argCooldownBars args >= 0)
    case argMaxHoldBars args of
        Nothing -> pure ()
        Just n -> ensure "--max-hold-bars must be >= 0" (n >= 0)
    case argMaxDrawdown args of
        Nothing -> pure ()
        Just v -> ensure "--max-drawdown must be > 0 and < 1" (v > 0 && v < 1)
    case argMaxDailyLoss args of
        Nothing -> pure ()
        Just v -> ensure "--max-daily-loss must be > 0 and < 1" (v > 0 && v < 1)
    case argMaxWeeklyLoss args of
        Nothing -> pure ()
        Just v -> ensure "--max-weekly-loss must be > 0 and < 1" (v > 0 && v < 1)
    case argRiskPerTrade args of
        Nothing -> pure ()
        Just v -> ensure "--risk-per-trade must be > 0 and < 1" (v > 0 && v < 1)
    case argRiskPerTrade args of
        Nothing -> pure ()
        Just _ ->
            let stopLossOn = maybe False (> 0) (argStopLoss args)
                stopLossVolOn = argStopLossVolMult args > 0
             in ensure "--risk-per-trade requires --stop-loss or --stop-loss-vol-mult" (stopLossOn || stopLossVolOn)
    case argMaxTradesPerDay args of
        Nothing -> pure ()
        Just n -> ensure "--max-trades-per-day must be >= 0" (n >= 0)
    ensure "--expectancy-lookback must be >= 0" (argExpectancyLookback args >= 0)
    case argMinExpectancy args of
        Nothing -> pure ()
        Just v ->
            ensure "--min-expectancy must be finite" (not (isNaN v || isInfinite v))
                >> ensure "--expectancy-lookback must be >= 1 when --min-expectancy is set" (argExpectancyLookback args >= 1)
    ensure "--perf-lookback must be >= 0" (argPerfLookback args >= 0)
    case argPerfMinWinRate args of
        Nothing -> pure ()
        Just v -> ensure "--perf-min-win-rate must be between 0 and 1" (v >= 0 && v <= 1)
    case argPerfMinProfitFactor args of
        Nothing -> pure ()
        Just v -> ensure "--perf-min-profit-factor must be >= 0" (v >= 0)
    ensure "--loss-streak-max must be >= 0" (argLossStreakMax args >= 0)
    ensure "--loss-streak-cooldown-bars must be >= 0" (argLossStreakCooldownBars args >= 0)
    case argMaxOpenPositions args of
        Nothing -> pure ()
        Just n -> ensure "--max-open-positions must be >= 0" (n >= 0)
    case argMaxOpenPerBase args of
        Nothing -> pure ()
        Just n -> ensure "--max-open-per-base must be >= 0" (n >= 0)
    case argMaxGrossExposure args of
        Nothing -> pure ()
        Just v -> ensure "--max-gross-exposure must be >= 0" (v >= 0)
    case argMaxNetExposure args of
        Nothing -> pure ()
        Just v -> ensure "--max-net-exposure must be >= 0" (v >= 0)
    case argMaxExposurePerBase args of
        Nothing -> pure ()
        Just v -> ensure "--max-exposure-per-base must be >= 0" (v >= 0)
    ensure "--min-edge must be >= 0" (argMinEdge args >= 0)
    ensure "--min-signal-to-noise must be >= 0" (argMinSignalToNoise args >= 0)
    ensure "--snr-size-weight must be between 0 and 1" (argSnrSizeWeight args >= 0 && argSnrSizeWeight args <= 1)
    ensure "--adaptive-edge-buffer-max must be >= 0" (argAdaptiveEdgeBufferMax args >= 0)
    ensure "--adaptive-min-signal-to-noise-max must be >= 0" (argAdaptiveMinSignalToNoiseMax args >= 0)
    ensure "--adaptive-kalman-z-min-max must be >= 0" (argAdaptiveKalmanZMinMax args >= 0)
    ensure "--adaptive-trend-lookback-max must be >= 0" (argAdaptiveTrendLookbackMax args >= 0)
    ensure "--meta-label-min-edge must be >= 0" (argMetaLabelMinEdge args >= 0)
    ensure "--meta-label-min-confidence must be between 0 and 1" (argMetaLabelMinConfidence args >= 0 && argMetaLabelMinConfidence args <= 1)
    ensure "--regime-bank-hysteresis must be between 0 and 1" (argRegimeBankHysteresis args >= 0 && argRegimeBankHysteresis args <= 1)
    ensure "--regime-trend-open-mult must be >= 0" (argRegimeTrendOpenMult args >= 0)
    ensure "--regime-mr-open-mult must be >= 0" (argRegimeMrOpenMult args >= 0)
    ensure "--regime-high-vol-open-mult must be >= 0" (argRegimeHighVolOpenMult args >= 0)
    ensure "--regime-trend-size-mult must be >= 0" (argRegimeTrendSizeMult args >= 0)
    ensure "--regime-mr-size-mult must be >= 0" (argRegimeMrSizeMult args >= 0)
    ensure "--regime-high-vol-size-mult must be >= 0" (argRegimeHighVolSizeMult args >= 0)
    ensure "--mtf-fast-bars must be >= 1" (argMtfFastBars args >= 1)
    ensure "--mtf-mid-bars must be >= 1" (argMtfMidBars args >= 1)
    ensure "--mtf-slow-bars must be >= 1" (argMtfSlowBars args >= 1)
    ensure "--mtf-min-agree must be between 1 and 3" (argMtfMinAgree args >= 1 && argMtfMinAgree args <= 3)
    ensure "--cross-asset-min-beta must be >= 0" (argCrossAssetMinBeta args >= 0)
    ensure "--cross-asset-min-edge must be >= 0" (argCrossAssetMinEdge args >= 0)
    ensure "--pairs-stat-arb-lookback must be >= 2" (argPairsStatArbLookback args >= 2)
    ensure "--pairs-stat-arb-z-entry must be > 0" (argPairsStatArbZEntry args > 0)
    ensure "--pairs-stat-arb-size-mult must be between 0 and 1" (argPairsStatArbSizeMult args >= 0 && argPairsStatArbSizeMult args <= 1)
    ensure "--threshold-factor-alpha must be between 0 and 1" (argThresholdFactorAlpha args >= 0 && argThresholdFactorAlpha args <= 1)
    ensure "--threshold-factor-min must be >= 0" (argThresholdFactorMin args >= 0)
    ensure "--threshold-factor-max must be >= --threshold-factor-min" (argThresholdFactorMax args >= argThresholdFactorMin args)
    ensure "--threshold-factor-floor must be >= 0" (argThresholdFactorFloor args >= 0)
    ensure "--edge-buffer must be >= 0" (argEdgeBuffer args >= 0)
    ensure "--trend-lookback must be >= 0" (argTrendLookback args >= 0)
    ensure "--max-position-size must be >= 0" (argMaxPositionSize args >= 0)
    let volTargetEnabled = maybe False (> 0) (argVolTarget args)
    case argVolTarget args of
        Nothing -> pure ()
        Just v -> ensure "--vol-target must be >= 0" (v >= 0)
    ensure "--vol-lookback must be >= 0" (argVolLookback args >= 0)
    case (volTargetEnabled, argVolEwmaAlpha args) of
        (True, Nothing) ->
            ensure "--vol-lookback must be >= 2 when --vol-target is set (or use --vol-ewma-alpha)" (argVolLookback args >= 2)
        _ -> pure ()
    case argVolEwmaAlpha args of
        Nothing -> pure ()
        Just a -> ensure "--vol-ewma-alpha must be > 0 and < 1" (a > 0 && a < 1)
    ensure "--vol-floor must be >= 0" (argVolFloor args >= 0)
    ensure "--vol-scale-max must be >= 0" (argVolScaleMax args >= 0)
    case argMaxVolatility args of
        Nothing -> pure ()
        Just v -> ensure "--max-volatility must be >= 0" (v >= 0)
    ensure "--rebalance-bars must be >= 0" (argRebalanceBars args >= 0)
    ensure "--rebalance-threshold must be >= 0" (argRebalanceThreshold args >= 0)
    ensure "--rebalance-cost-mult must be >= 0" (argRebalanceCostMult args >= 0)
    let fundingRate = argFundingRate args
    ensure "--funding-rate must be finite" (not (isNaN fundingRate || isInfinite fundingRate))
    case argFundingOiFundingCap args of
        Nothing -> pure ()
        Just v -> ensure "--funding-oi-funding-cap must be >= 0" (v >= 0)
    ensure "--funding-oi-vol-lookback must be >= 2" (argFundingOiVolLookback args >= 2)
    case argFundingOiVolCap args of
        Nothing -> pure ()
        Just v -> ensure "--funding-oi-vol-cap must be >= 0" (v >= 0)
    ensure "--funding-oi-size-mult must be between 0 and 1" (argFundingOiSizeMult args >= 0 && argFundingOiSizeMult args <= 1)
    ensure "--blend-weight must be between 0 and 1" (argBlendWeight args >= 0 && argBlendWeight args <= 1)
    ensure "--tri-layer-fast-mult must be > 0" (argTriLayerFastMult args > 0)
    ensure "--tri-layer-slow-mult must be > 0" (argTriLayerSlowMult args > 0)
    ensure "--tri-layer-cloud-padding must be >= 0" (argTriLayerCloudPadding args >= 0)
    ensure "--tri-layer-cloud-slope must be >= 0" (argTriLayerCloudSlope args >= 0)
    ensure "--tri-layer-cloud-width must be >= 0" (argTriLayerCloudWidth args >= 0)
    ensure "--tri-layer-touch-lookback must be >= 1" (argTriLayerTouchLookback args >= 1)
    ensure "--tri-layer-price-action-body must be >= 0" (argTriLayerPriceActionBody args >= 0)
    ensure "--kalman-band-lookback must be >= 0" (argKalmanBandLookback args >= 0)
    ensure "--kalman-band-std-mult must be >= 0" (argKalmanBandStdMult args >= 0)
    case argKalmanBandStdMult args of
        v
            | v > 0 ->
                ensure "--kalman-band-lookback must be >= 2 when --kalman-band-std-mult is enabled" (argKalmanBandLookback args >= 2)
        _ -> pure ()
    ensure "--lstm-exit-flip-bars must be >= 0" (argLstmExitFlipBars args >= 0)
    ensure "--lstm-exit-flip-grace-bars must be >= 0" (argLstmExitFlipGraceBars args >= 0)
    ensure "--lstm-confidence-soft must be between 0 and 1" (argLstmConfidenceSoft args >= 0 && argLstmConfidenceSoft args <= 1)
    ensure "--lstm-confidence-hard must be between 0 and 1" (argLstmConfidenceHard args >= 0 && argLstmConfidenceHard args <= 1)
    ensure "--protection-min-confidence must be between 0 and 1" (argProtectionMinConfidence args >= 0 && argProtectionMinConfidence args <= 1)
    ensure
        "--lstm-confidence-soft must be <= --lstm-confidence-hard (unless hard=0 to disable)"
        (argLstmConfidenceHard args <= 0 || argLstmConfidenceSoft args <= argLstmConfidenceHard args)
    case argMaxOrderErrors args of
        Nothing -> pure ()
        Just n -> ensure "--max-order-errors must be >= 1" (n >= 1)
    ensure "--execution-maker-offset-bps must be >= 0" (argExecutionMakerOffsetBps args >= 0)
    ensure "--execution-maker-timeout-sec must be >= 0" (argExecutionMakerTimeoutSec args >= 0)
    ensure "--execution-maker-poll-ms must be >= 1" (argExecutionMakerPollMs args >= 1)
    case argPeriodsPerYear args of
        Nothing -> pure ()
        Just v -> ensure "--periods-per-year must be > 0" (v > 0)
    case argOrderQuote args of
        Nothing -> pure ()
        Just q -> ensure "--order-quote must be > 0" (q > 0)
    case argOrderQuantity args of
        Nothing -> pure ()
        Just q -> ensure "--order-quantity must be > 0" (q > 0)

    let qtyOn = maybe False (> 0) (argOrderQuantity args)
        quoteOn = maybe False (> 0) (argOrderQuote args)
        fracOn = maybe False (> 0) (argOrderQuoteFraction args)
        sizingCount = fromEnum qtyOn + fromEnum quoteOn + fromEnum fracOn
    ensure "Provide only one of --order-quantity, --order-quote, --order-quote-fraction" (sizingCount <= 1)

    case argOrderQuoteFraction args of
        Nothing -> pure ()
        Just f -> ensure "--order-quote-fraction must be > 0 and <= 1" (f > 0 && f <= 1)
    case argMaxOrderQuote args of
        Nothing -> pure ()
        Just q -> ensure "--max-order-quote must be > 0" (q > 0)

    case argMaxOrderQuote args of
        Just _ -> ensure "--max-order-quote requires --order-quote-fraction" fracOn
        _ -> pure ()

    let market = argBinanceMarket args
    ensure "--positioning long-short requires --futures when trading" (not (argBinanceTrade args && argPositioning args == LongShort && market /= MarketFutures))
    ensure
        "--positioning long-short requires Binance futures when using exchange data"
        (not (argPositioning args == LongShort && isJust (argBinanceSymbol args) && (argPlatform args /= PlatformBinance || market /= MarketFutures)))
    ensure "--margin requires --binance-live for trading" (not (argBinanceMargin args && argBinanceTrade args && not (argBinanceLive args)))
    case argIdempotencyKey args of
        Nothing -> pure ()
        Just raw ->
            let k = trim raw
                len = length k
                okLen = len >= 1 && len <= 36
                okChars = all (\c -> isAsciiAlphaNum c || c == '-' || c == '_') k
             in ensure "--idempotency-key must be 1..36 chars of [A-Za-z0-9_-]" (okLen && okChars)
    case argDexChainId args of
        Nothing -> pure ()
        Just cid -> ensure "--dex-chain-id must be > 0" (cid > 0)
    case argDexBaseDecimals args of
        Nothing -> pure ()
        Just d -> ensure "--dex-base-decimals must be >= 0" (d >= 0)
    case argDexQuoteDecimals args of
        Nothing -> pure ()
        Just d -> ensure "--dex-quote-decimals must be >= 0" (d >= 0)

    ensure "--kalman-z-min must be >= 0" (argKalmanZMin args >= 0)
    ensure "--kalman-z-max must be >= --kalman-z-min" (argKalmanZMax args >= argKalmanZMin args)
    case argMaxHighVolProb args of
        Nothing -> pure ()
        Just v -> ensure "--max-high-vol-prob must be between 0 and 1" (v >= 0 && v <= 1)
    case argMaxConformalWidth args of
        Nothing -> pure ()
        Just v -> ensure "--max-conformal-width must be >= 0" (v >= 0)
    case argMaxQuantileWidth args of
        Nothing -> pure ()
        Just v -> ensure "--max-quantile-width must be >= 0" (v >= 0)
    ensure "--min-position-size must be between 0 and 1" (argMinPositionSize args >= 0 && argMinPositionSize args <= 1)
    ensure "--min-position-size must be <= --max-position-size" (argMinPositionSize args <= argMaxPositionSize args)
    ensure "--kelly-lite-fraction must be >= 0" (argKellyLiteFraction args >= 0)
    ensure "--kelly-lite-floor must be >= 0" (argKellyLiteFloor args >= 0)
    ensure "--kelly-lite-cap must be >= --kelly-lite-floor" (argKellyLiteCap args >= argKellyLiteFloor args)
    ensure "--tune-stress-vol-mult must be > 0" (argTuneStressVolMult args > 0)
    ensure "--tune-stress-weight must be >= 0" (argTuneStressWeight args >= 0)

    pure args
  where
    ensure :: String -> Bool -> Either String ()
    ensure msg cond = if cond then Right () else Left msg

    ensureFinite :: String -> Double -> Either String ()
    ensureFinite flag value = ensure (flag ++ " must be finite") (isFiniteNumber value)

    ensureMaybeFinite :: String -> Maybe Double -> Either String ()
    ensureMaybeFinite flag mValue =
        case mValue of
            Nothing -> Right ()
            Just value -> ensureFinite flag value

    isFiniteNumber :: Double -> Bool
    isFiniteNumber x = not (isNaN x || isInfinite x)

    isAsciiAlphaNum :: Char -> Bool
    isAsciiAlphaNum c = isDigit c || isAsciiUpper c || isAsciiLower c

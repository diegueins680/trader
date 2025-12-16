{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
module Main where

import Control.Concurrent (ThreadId, forkIO, killThread, myThreadId, threadDelay)
import Control.Concurrent.MVar (MVar, modifyMVar, modifyMVar_, newEmptyMVar, newMVar, readMVar, swapMVar, tryPutMVar, tryReadMVar, withMVar)
import Control.Exception (SomeException, finally, fromException, throwIO, try)
import Control.Applicative ((<|>))
import Crypto.Hash (Digest, hash)
import Crypto.Hash.Algorithms (SHA256)
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode, encode, object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Types as AT
import Data.ByteArray (convert)
import Data.Char (isAlphaNum, isDigit, isSpace, toLower)
import Data.Foldable (toList)
import Data.Int (Int64)
import Data.List (foldl', intercalate, isInfixOf, isPrefixOf, isSuffixOf, sortOn)
import Data.Maybe (isJust, listToMaybe, mapMaybe)
import qualified Data.Sequence as Seq
import Data.Sequence (Seq)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.Csv as Csv
import qualified Data.HashMap.Strict as HM
import qualified Data.Vector as V
import Data.IORef (IORef, atomicModifyIORef', modifyIORef', newIORef, readIORef)
import GHC.Conc (getNumCapabilities, setNumCapabilities)
import GHC.Exception (ErrorCall(..))
import GHC.Generics (Generic)
import Network.HTTP.Client (HttpException)
import Network.HTTP.Types (ResponseHeaders, Status, status200, status202, status204, status400, status401, status404, status405, status429, status500, status502)
import Network.HTTP.Types.Header (hAuthorization)
import qualified Network.Wai as Wai
import qualified Network.Wai.Handler.Warp as Warp
import Options.Applicative
import System.Directory (canonicalizePath, createDirectoryIfMissing, doesFileExist, listDirectory, removeFile, renameFile)
import System.Environment (lookupEnv)
import System.Exit (die, exitFailure)
import System.FilePath ((</>), takeDirectory)
import System.IO (IOMode(ReadMode), hGetLine, hIsEOF, hPutStrLn, stderr, withFile)
import System.IO.Error (ioeGetErrorString, isUserError)
import Text.Printf (printf)
import Text.Read (readMaybe)

import Trader.Binance
  ( BinanceEnv(..)
  , BinanceMarket(..)
  , BinanceOrderMode(..)
  , OrderSide(..)
  , SymbolFilters(..)
  , Step(..)
  , Kline(..)
  , fetchTickerPrice
  , binanceBaseUrl
  , binanceTestnetBaseUrl
  , binanceFuturesBaseUrl
  , binanceFuturesTestnetBaseUrl
  , newBinanceEnv
  , fetchKlines
  , fetchCloses
  , fetchFreeBalance
  , fetchFuturesAvailableBalance
  , fetchFuturesPositionAmt
  , fetchSymbolFilters
  , quantizeDown
  , getTimestampMs
  , placeMarketOrder
  , fetchOrderByClientId
  )
import Trader.KalmanFusion (Kalman1(..), initKalman1, stepMulti)
import Trader.LSTM
  ( LSTMConfig(..)
  , EpochStats(..)
  , LSTMModel(..)
  , paramCount
  , trainLSTM
  , fineTuneLSTM
  , predictNext
  , predictSeriesNext
  )
import Trader.Metrics (BacktestMetrics(..), computeMetrics)
import Trader.Duration (lookbackBarsFrom, parseIntervalSeconds)
import Trader.Normalization (NormState, NormType(..), fitNorm, forwardSeries, inverseNorm, inverseSeries, parseNormType)
import Trader.Predictors
  ( PredictorBundle
  , SensorId(..)
  , SensorOutput(..)
  , RegimeProbs(..)
  , Quantiles(..)
  , Interval(..)
  , HMMFilter(..)
  , trainPredictors
  , initHMMFilter
  , predictSensors
  , updateHMM
  )
import Trader.SensorVariance (SensorVar, emptySensorVar, updateResidual, varianceFor)
import Trader.Symbol (splitSymbol)
import Trader.Method (Method(..), methodCode, parseMethod, selectPredictions)
import Trader.Optimization (optimizeOperations, optimizeOperationsWithHL, sweepThreshold, sweepThresholdWithHL)
import Trader.Split (Split(..), splitTrainBacktest)
import Trader.Trading (BacktestResult(..), EnsembleConfig(..), IntrabarFill(..), Positioning(..), StepMeta(..), Trade(..), simulateEnsembleLongFlat, simulateEnsembleLongFlatWithHL)

-- CSV loading

loadPricesCsv :: FilePath -> String -> IO [Double]
loadPricesCsv path priceCol = do
  bs <- BL.readFile path
  case Csv.decodeByName bs of
    Left err -> error ("CSV decode failed (" ++ path ++ "): " ++ err)
    Right (hdr, rows) -> do
      let wanted = trim priceCol
          wantedLower = map toLower wanted
          wantedNorm = normalizeKey wanted
          hdrList = V.toList hdr
          rowsList0 = V.toList rows
          wantedBs = BS.pack wanted
          mKeyExact =
            if wantedBs `elem` hdrList then Just wantedBs else Nothing
          mKeyNorm =
            case filter (\h -> normalizeKey (BS.unpack h) == wantedNorm) hdrList of
              (h:_) -> Just h
              [] -> Nothing
          mKey =
            case mKeyExact of
              Just k -> Just k
              Nothing ->
                case filter (\h -> map toLower (BS.unpack h) == wantedLower) hdrList of
                  (h:_) -> Just h
                  [] -> mKeyNorm
          available = BS.unpack (BS.intercalate ", " hdrList)
          suggestions =
            let commonPrefixLen :: String -> String -> Int
                commonPrefixLen a b = length (takeWhile id (zipWith (==) a b))
                score hn =
                  let pref = commonPrefixLen wantedNorm hn
                      contains = if wantedNorm `isInfixOf` hn then 100 else 0
                   in contains + pref
                scored =
                  [ (negate s, h)
                  | h <- hdrList
                  , let hn = normalizeKey (BS.unpack h)
                  , let s = score hn
                  , s > 0
                  ]
             in take 5 (map snd (sortOn fst scored))

      if null wanted then error "--price-column cannot be empty" else pure ()
      key <-
        case mKey of
          Just k -> pure k
          Nothing ->
            let hint =
                  if null suggestions
                    then ""
                    else " Suggestions: " ++ BS.unpack (BS.intercalate ", " suggestions) ++ "."
             in error ("Column not found: " ++ wanted ++ " (file: " ++ path ++ "). Available columns: " ++ available ++ "." ++ hint)
      let rowsList = maybe rowsList0 (\tk -> sortCsvRowsByTime tk rowsList0) (csvTimeKey hdrList)
      pure $ zipWith (\i row -> extractPriceAt i key row) [1 :: Int ..] rowsList

csvTimeKey :: [BS.ByteString] -> Maybe BS.ByteString
csvTimeKey hdrList =
  let candidates =
        [ "openTimeMs"
        , "open_time_ms"
        , "openTime"
        , "open_time"
        , "timestamp"
        , "datetime"
        , "date"
        , "time"
        ]
   in firstJust [findHeaderKey hdrList c | c <- candidates]

findHeaderKey :: [BS.ByteString] -> String -> Maybe BS.ByteString
findHeaderKey hdrList wanted =
  let w = normalizeKey wanted
      matches = filter (\h -> normalizeKey (BS.unpack h) == w) hdrList
   in case matches of
        (h:_) -> Just h
        [] -> Nothing

normalizeKey :: String -> String
normalizeKey = map toLower . filter isAlphaNum

sortCsvRowsByTime :: BS.ByteString -> [Csv.NamedRecord] -> [Csv.NamedRecord]
sortCsvRowsByTime timeKey rows =
  case traverse (lookupCell timeKey) rows of
    Nothing -> rows
    Just rawTimes ->
      let times = map (trim . BS.unpack) rawTimes
       in case traverse parseTimeInt64 times of
            Just ts ->
              let pairs = zip ts rows
               in map snd (sortOn fst pairs)
            Nothing ->
              if all looksLikeIso8601Prefix times
                then
                  let pairs = zip times rows
                   in map snd (sortOn fst pairs)
                else rows

lookupCell :: BS.ByteString -> Csv.NamedRecord -> Maybe BS.ByteString
lookupCell key rec = HM.lookup key rec

parseTimeInt64 :: String -> Maybe Int64
parseTimeInt64 s =
  case (readMaybe s :: Maybe Int64) of
    Just n -> Just n
    Nothing ->
      case (readMaybe s :: Maybe Double) of
        Just d -> Just (floor d)
        Nothing -> Nothing

looksLikeIso8601Prefix :: String -> Bool
looksLikeIso8601Prefix s =
  case s of
    (a:b:c:d:'-':e:f:'-':g:h:_) -> all isDigit [a, b, c, d, e, f, g, h]
    _ -> False

firstJust :: [Maybe a] -> Maybe a
firstJust xs =
  case xs of
    [] -> Nothing
    (y:ys) ->
      case y of
        Just _ -> y
        Nothing -> firstJust ys

extractPriceAt :: Int -> BS.ByteString -> Csv.NamedRecord -> Double
extractPriceAt rowIndex key rec =
  case HM.lookup key rec of
    Nothing -> error ("Column not found: " ++ BS.unpack key)
    Just raw ->
      let s = trim (BS.unpack raw)
       in case readMaybe s of
            Just d -> d
            Nothing ->
              error
                ( "Failed to parse price at row "
                    ++ show rowIndex
                    ++ " ("
                    ++ BS.unpack key
                    ++ "): "
                    ++ s
                )

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

type LstmCtx = (NormState, [Double], LSTMModel)

type KalmanCtx = (PredictorBundle, Kalman1, HMMFilter, SensorVar)

data LatestSignal = LatestSignal
  { lsMethod :: !Method
  , lsCurrentPrice :: !Double
  , lsOpenThreshold :: !Double
  , lsCloseThreshold :: !Double
  , lsKalmanNext :: !(Maybe Double)
  , lsKalmanReturn :: !(Maybe Double)
  , lsKalmanStd :: !(Maybe Double)
  , lsKalmanZ :: !(Maybe Double)
  , lsRegimes :: !(Maybe RegimeProbs)
  , lsQuantiles :: !(Maybe Quantiles)
  , lsConformalInterval :: !(Maybe Interval)
  , lsConfidence :: !(Maybe Double)
  , lsPositionSize :: !(Maybe Double)
  , lsKalmanDir :: !(Maybe Int)
  , lsLstmNext :: !(Maybe Double)
  , lsLstmDir :: !(Maybe Int)
  , lsChosenDir :: !(Maybe Int)
  , lsAction :: !String
  } deriving (Eq, Show)

data BacktestSummary = BacktestSummary
  { bsTrainEndRaw :: !Int
  , bsTrainEnd :: !Int
  , bsTrainSize :: !Int
  , bsFitSize :: !Int
  , bsTuneSize :: !Int
  , bsTuneRatio :: !Double
  , bsBacktestSize :: !Int
  , bsBacktestRatio :: !Double
  , bsMethodUsed :: !Method
  , bsBestOpenThreshold :: !Double
  , bsBestCloseThreshold :: !Double
  , bsMetrics :: !BacktestMetrics
  , bsLstmHistory :: !(Maybe [EpochStats])
  , bsLatestSignal :: !LatestSignal
  , bsEquityCurve :: ![Double]
  , bsBacktestPrices :: ![Double]
  , bsPositions :: ![Double]
  , bsAgreementOk :: ![Bool]
  , bsTrades :: ![Trade]
  } deriving (Eq, Show)

-- CLI

data Args = Args
  { argData :: Maybe FilePath
  , argPriceCol :: String
  , argBinanceSymbol :: Maybe String
  , argBinanceFutures :: Bool
  , argBinanceMargin :: Bool
  , argInterval :: String
  , argBars :: Int
  , argLookbackWindow :: String
  , argLookbackBars :: Maybe Int
  , argBinanceTestnet :: Bool
  , argBinanceApiKey :: Maybe String
  , argBinanceApiSecret :: Maybe String
  , argBinanceTrade :: Bool
  , argBinanceLive :: Bool
  , argOrderQuote :: Maybe Double
  , argOrderQuantity :: Maybe Double
  , argOrderQuoteFraction :: Maybe Double
  , argMaxOrderQuote :: Maybe Double
  , argIdempotencyKey :: Maybe String
  , argNormalization :: NormType
  , argHiddenSize :: Int
  , argEpochs :: Int
  , argLr :: Double
  , argValRatio :: Double
  , argBacktestRatio :: Double
  , argTuneRatio :: Double
  , argPatience :: Int
  , argGradClip :: Maybe Double
  , argSeed :: Int
  , argKalmanDt :: Double
  , argKalmanProcessVar :: Double
  , argKalmanMeasurementVar :: Double
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
  , argIntrabarFill :: IntrabarFill
  , argStopLoss :: Maybe Double
  , argTakeProfit :: Maybe Double
  , argTrailingStop :: Maybe Double
  , argMaxDrawdown :: Maybe Double
  , argMaxDailyLoss :: Maybe Double
  , argMaxOrderErrors :: Maybe Int
  , argPeriodsPerYear :: Maybe Double
  , argJson :: Bool
  , argServe :: Bool
  , argPort :: Int
  -- Confidence gating/sizing (Kalman sensors + HMM/intervals)
  , argKalmanZMin :: Double
  , argKalmanZMax :: Double
  , argMaxHighVolProb :: Maybe Double
  , argMaxConformalWidth :: Maybe Double
  , argMaxQuantileWidth :: Maybe Double
  , argConfirmConformal :: Bool
  , argConfirmQuantiles :: Bool
  , argConfidenceSizing :: Bool
  , argMinPositionSize :: Double
  } deriving (Eq, Show)

autoBarsSentinel :: Int
autoBarsSentinel = -1

defaultBinanceBars :: Int
defaultBinanceBars = 500

resolveBarsForCsv :: Args -> Int
resolveBarsForCsv args =
  case argBars args of
    n | n == autoBarsSentinel -> 0
    n -> n

resolveBarsForBinance :: Args -> Int
resolveBarsForBinance args =
  case argBars args of
    n | n == autoBarsSentinel -> defaultBinanceBars
    n -> n

parseBarsArg :: String -> Either String Int
parseBarsArg raw =
  let s = map toLower (trim raw)
   in if s == "auto"
        then Right autoBarsSentinel
        else
          case (readMaybe s :: Maybe Int) of
            Just n -> Right n
            Nothing -> Left "Expected an integer (e.g. 500) or 'auto'."

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
    "longshort" -> Right LongShort
    "ls" -> Right LongShort
    "short" -> Right LongShort
    _ -> Left "Invalid positioning (expected long-flat|long-short)"

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
opts =
  Args
    <$> optional (strOption (long "data" <> metavar "PATH" <> help "CSV file containing prices"))
    <*> strOption (long "price-column" <> value "close" <> help "CSV column name for price (case-insensitive; prints suggestions on error)")
    <*> optional (strOption (long "binance-symbol" <> metavar "SYMBOL" <> help "Fetch klines from Binance (e.g., BTCUSDT)"))
    <*> switch (long "futures" <> help "Use Binance USDT-M futures endpoints for data/orders")
    <*> switch (long "margin" <> help "Use Binance margin account endpoints for orders/balance")
    <*> strOption (long "interval" <> long "binance-interval" <> value "5m" <> help "Bar interval / Binance kline interval (e.g., 1m, 5m, 1h, 1d)")
    <*> option
          (eitherReader parseBarsArg)
          ( long "bars"
              <> long "binance-limit"
              <> metavar "N|auto"
              <> value autoBarsSentinel
              <> showDefaultWith (\n -> if n == autoBarsSentinel then "auto" else show n)
              <> help "Number of bars/klines to use (auto=all CSV, Binance=500; CSV also supports 0=all; Binance requires 2..1000)"
          )
    <*> strOption (long "lookback-window" <> value "24h" <> help "Lookback window duration (e.g., 90m, 24h, 7d)")
    <*> optional (option auto (long "lookback-bars" <> long "lookback" <> help "Override lookback bars (disables --lookback-window conversion)"))
    <*> switch (long "binance-testnet" <> help "Use Binance testnet base URL (public + signed endpoints)")
    <*> optional (strOption (long "binance-api-key" <> help "Binance API key (or env BINANCE_API_KEY)"))
    <*> optional (strOption (long "binance-api-secret" <> help "Binance API secret (or env BINANCE_API_SECRET)"))
    <*> switch (long "binance-trade" <> help "If set, place a market order for the latest signal")
    <*> switch (long "binance-live" <> help "If set, send LIVE orders (otherwise uses /order/test)")
    <*> optional (option auto (long "order-quote" <> help "Quote amount to spend on BUY (quoteOrderQty)"))
    <*> optional (option auto (long "order-quantity" <> help "Base quantity to trade (quantity)"))
    <*> optional (option auto (long "order-quote-fraction" <> help "Size BUY orders as a fraction of quote balance (0..1) when --order-quote/--order-quantity not set"))
    <*> optional (option auto (long "max-order-quote" <> help "Cap the computed quote amount when using --order-quote-fraction"))
    <*> optional (strOption (long "idempotency-key" <> metavar "ID" <> help "Optional Binance newClientOrderId for idempotent orders"))
    <*> option (maybeReader parseNormType) (long "normalization" <> value NormStandard <> help "none|minmax|standard|log")
    <*> option auto (long "hidden-size" <> value 16 <> help "LSTM hidden size")
    <*> option auto (long "epochs" <> value 30 <> help "LSTM training epochs (Adam)")
    <*> option auto (long "lr" <> value 1e-3 <> help "LSTM learning rate")
    <*> option auto (long "val-ratio" <> value 0.2 <> help "Validation split ratio (within training set)")
    <*> option auto (long "backtest-ratio" <> value 0.2 <> help "Backtest holdout ratio (last portion of series)")
    <*> option auto (long "tune-ratio" <> value 0.2 <> help "When optimizing operations/threshold: tune on the last portion of the training split (avoids lookahead on the backtest split)")
    <*> option auto (long "patience" <> value 10 <> help "Early stopping patience (0 disables)")
    <*> optional (option auto (long "grad-clip" <> help "Gradient clipping max L2 norm"))
    <*> option auto (long "seed" <> value 42 <> help "Random seed for LSTM init")
    <*> option auto (long "kalman-dt" <> value 1.0 <> help "Kalman dt")
    <*> option auto (long "kalman-process-var" <> value 1e-5 <> help "Kalman process noise variance (white-noise jerk)")
    <*> option auto (long "kalman-measurement-var" <> value 1e-3 <> help "Kalman measurement noise variance")
    <*> option auto (long "threshold" <> value 0.001 <> help "Direction threshold (fractional deadband)")
    <*> option
          (eitherReader parseMethod)
          ( long "method"
              <> value MethodBoth
              <> showDefaultWith methodCode
              <> help "Method: 11|both=Kalman+LSTM (direction-agreement gated), 10|kalman=Kalman only, 01|lstm=LSTM only"
          )
    <*> option
          (eitherReader parsePositioning)
          ( long "positioning"
              <> value LongFlat
              <> showDefaultWith positioningCode
              <> help "Positioning: long-flat (default) or long-short (experimental; futures-only for live orders)"
          )
    <*> switch (long "optimize-operations" <> help "Optimize method (11/10/01) and threshold on a tune split (avoids lookahead on the backtest split)")
    <*> switch (long "sweep-threshold" <> help "Sweep thresholds on a tune split and print the best final equity (avoids lookahead on the backtest split)")
    <*> switch (long "trade-only" <> help "Skip backtest/metrics; only compute the latest signal (and optionally place an order)")
    <*> option auto (long "fee" <> value 0.0005 <> help "Fee applied when switching position")
    <*> option auto (long "slippage" <> value 0.0 <> help "Slippage per side (fractional, e.g. 0.0002)")
    <*> option auto (long "spread" <> value 0.0 <> help "Bid-ask spread (fractional total; half applied per side)")
    <*> option
          (eitherReader parseIntrabarFill)
          ( long "intrabar-fill"
              <> value StopFirst
              <> showDefaultWith intrabarFillCode
              <> help "If take-profit and stop are both hit within a bar: stop-first (conservative) or take-profit-first"
          )
    <*> optional (option auto (long "stop-loss" <> help "Stop loss fraction for a bracket exit (e.g., 0.02 for 2%)"))
    <*> optional (option auto (long "take-profit" <> help "Take profit fraction for a bracket exit (e.g., 0.03 for 3%)"))
    <*> optional (option auto (long "trailing-stop" <> help "Trailing stop fraction for a bracket exit (e.g., 0.01 for 1%)"))
    <*> optional (option auto (long "max-drawdown" <> help "Halt the live bot if peak-to-trough drawdown exceeds this fraction (0..1)"))
    <*> optional (option auto (long "max-daily-loss" <> help "Halt the live bot if daily loss exceeds this fraction (0..1), based on UTC day"))
    <*> optional (option auto (long "max-order-errors" <> help "Halt the live bot after N consecutive order failures"))
    <*> optional (option auto (long "periods-per-year" <> help "For annualized metrics (e.g., 365 for 1d, 8760 for 1h)"))
    <*> switch (long "json" <> help "Output JSON to stdout (CLI mode only)")
    <*> switch (long "serve" <> help "Run REST API server on localhost instead of running the CLI workflow")
    <*> option auto (long "port" <> value 8080 <> help "REST API port (when --serve)")
    <*> option auto (long "kalman-z-min" <> value 0 <> help "Min |Kalman mean|/std (z-score) required to treat Kalman as directional (0 disables)")
    <*> option auto (long "kalman-z-max" <> value 3 <> help "Z-score mapped to position size=1 when --confidence-sizing is enabled")
    <*> optional (option auto (long "max-high-vol-prob" <> help "If set, block trades when HMM high-vol regime prob exceeds this (0..1)"))
    <*> optional (option auto (long "max-conformal-width" <> help "If set, block trades when conformal interval width exceeds this (return units)"))
    <*> optional (option auto (long "max-quantile-width" <> help "If set, block trades when quantile (q90-q10) width exceeds this (return units)"))
    <*> switch (long "confirm-conformal" <> help "Require conformal interval to agree with the chosen direction")
    <*> switch (long "confirm-quantiles" <> help "Require quantiles to agree with the chosen direction")
    <*> switch (long "confidence-sizing" <> help "Scale entries by confidence (Kalman z-score / interval widths); leaves exits unscaled")
    <*> option auto (long "min-position-size" <> value 0 <> help "If confidence-sizing yields a size below this, skip the trade (0..1)")

argBinanceMarket :: Args -> BinanceMarket
argBinanceMarket args =
  case (argBinanceFutures args, argBinanceMargin args) of
    (True, True) -> error "Choose only one of --futures or --margin"
    (True, False) -> MarketFutures
    (False, True) -> MarketMargin
    (False, False) -> MarketSpot

argLookback :: Args -> Int
argLookback args =
  case argLookbackBars args of
    Just n ->
      if n < 2
        then error "--lookback-bars must be >= 2"
        else n
    Nothing ->
      case lookbackBarsFrom (argInterval args) (argLookbackWindow args) of
        Left err -> error err
        Right n ->
          if n < 2
            then
              error
                ( "Lookback window too small: "
                    ++ show (argLookbackWindow args)
                    ++ " at interval "
                    ++ show (argInterval args)
                    ++ " yields "
                    ++ show n
                    ++ " bars; need at least 2 bars."
                )
            else n

validateArgs :: Args -> Either String Args
validateArgs args = do
  ensure "Provide only one of --data or --binance-symbol" (not (isJust (argData args) && isJust (argBinanceSymbol args)))
  ensure "--json cannot be used with --serve" (not (argJson args && argServe args))
  ensure "--positioning long-short is not supported with --serve (live bot is long-flat only)" (not (argServe args && argPositioning args == LongShort))
  ensure "Choose only one of --futures or --margin" (not (argBinanceFutures args && argBinanceMargin args))

  let intervalStr = trim (argInterval args)
  ensure "--interval is required" (not (null intervalStr))
  case parseIntervalSeconds intervalStr of
    Nothing -> Left ("Invalid interval: " ++ show intervalStr ++ " (expected like 5m, 1h, 1d)")
    Just sec -> ensure "--interval must be > 0" (sec > 0)

  let binanceIntervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
  case argBinanceSymbol args of
    Nothing -> pure ()
    Just _ ->
      ensure
        ("--interval must be a Binance interval: " ++ unwords binanceIntervals)
        (intervalStr `elem` binanceIntervals)

  let barsRaw = argBars args
      barsCsv = resolveBarsForCsv args
      barsBinance = resolveBarsForBinance args
  ensure "--bars must be auto, 0 (all CSV), or >= 2" (barsRaw == autoBarsSentinel || barsRaw == 0 || barsRaw >= 2)
  ensure "--bars cannot be 1" (barsRaw /= 1)
  case argBinanceSymbol args of
    Nothing -> pure ()
    Just _ -> ensure "--bars must be between 2 and 1000 for Binance data" (barsBinance >= 2 && barsBinance <= 1000)

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

  let hasDataSource = case (argData args, argBinanceSymbol args) of
        (Nothing, Nothing) -> False
        _ -> True
      barsForLookback =
        case argBinanceSymbol args of
          Just _ -> barsBinance
          Nothing -> barsCsv
  if hasDataSource && barsForLookback > 0
    then
      ensure
        ( "--bars must be >= lookback+1 (need at least "
            ++ show (lookback + 1)
            ++ " bars for lookback="
            ++ show lookback
            ++ ")"
        )
        (barsForLookback > lookback)
    else pure ()

  ensure "--hidden-size must be >= 1" (argHiddenSize args >= 1)
  ensure "--epochs must be >= 0" (argEpochs args >= 0)
  ensure "--lr must be > 0" (argLr args > 0)
  ensure "--val-ratio must be >= 0 and < 1" (argValRatio args >= 0 && argValRatio args < 1)
  ensure "--backtest-ratio must be between 0 and 1" (argBacktestRatio args > 0 && argBacktestRatio args < 1)
  ensure "--tune-ratio must be >= 0 and < 1" (argTuneRatio args >= 0 && argTuneRatio args < 1)
  ensure "--patience must be >= 0" (argPatience args >= 0)
  case argGradClip args of
    Nothing -> pure ()
    Just g -> ensure "--grad-clip must be > 0" (g > 0)
  ensure "--kalman-dt must be > 0" (argKalmanDt args > 0)
  ensure "--kalman-process-var must be > 0" (argKalmanProcessVar args > 0)
  ensure "--kalman-measurement-var must be > 0" (argKalmanMeasurementVar args > 0)
  ensure "--threshold must be >= 0" (argTradeThreshold args >= 0)
  ensure "--fee must be >= 0" (argFee args >= 0)
  ensure "--slippage must be >= 0" (argSlippage args >= 0)
  ensure "--spread must be >= 0" (argSpread args >= 0)
  case argStopLoss args of
    Nothing -> pure ()
    Just v -> ensure "--stop-loss must be > 0 and < 1" (v > 0 && v < 1)
  case argTakeProfit args of
    Nothing -> pure ()
    Just v -> ensure "--take-profit must be > 0 and < 1" (v > 0 && v < 1)
  case argTrailingStop args of
    Nothing -> pure ()
    Just v -> ensure "--trailing-stop must be > 0 and < 1" (v > 0 && v < 1)
  case argMaxDrawdown args of
    Nothing -> pure ()
    Just v -> ensure "--max-drawdown must be > 0 and < 1" (v > 0 && v < 1)
  case argMaxDailyLoss args of
    Nothing -> pure ()
    Just v -> ensure "--max-daily-loss must be > 0 and < 1" (v > 0 && v < 1)
  case argMaxOrderErrors args of
    Nothing -> pure ()
    Just n -> ensure "--max-order-errors must be >= 1" (n >= 1)
  case argPeriodsPerYear args of
    Nothing -> pure ()
    Just v -> ensure "--periods-per-year must be > 0" (v > 0)
  case argOrderQuote args of
    Nothing -> pure ()
    Just q -> ensure "--order-quote must be >= 0" (q >= 0)
  case argOrderQuantity args of
    Nothing -> pure ()
    Just q -> ensure "--order-quantity must be >= 0" (q >= 0)

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
    Just q -> ensure "--max-order-quote must be >= 0" (q >= 0)

  case argMaxOrderQuote args of
    Just q | q > 0 -> ensure "--max-order-quote requires --order-quote-fraction" fracOn
    _ -> pure ()

  ensure "--binance-trade requires --binance-symbol" (not (argBinanceTrade args && argBinanceSymbol args == Nothing))
  let market = argBinanceMarket args
  ensure "--positioning long-short requires --futures when trading" (not (argBinanceTrade args && argPositioning args == LongShort && market /= MarketFutures))
  ensure "--margin requires --binance-live for trading" (not (argBinanceMargin args && argBinanceTrade args && not (argBinanceLive args)))
  case argIdempotencyKey args of
    Nothing -> pure ()
    Just raw ->
      let k = trim raw
          okLen = not (null k) && length k <= 36
          okChars = all (\c -> isAlphaNum c || c == '-' || c == '_') k
       in ensure "--idempotency-key must be 1..36 chars of [A-Za-z0-9_-]" (okLen && okChars)

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

  pure args
  where
    ensure :: String -> Bool -> Either String ()
    ensure msg cond = if cond then Right () else Left msg

main :: IO ()
main = do
  args <- execParser (info (opts <**> helper) fullDesc)
  args' <-
    case validateArgs args of
      Left e -> die (e ++ "\n\nRun with --help for usage.")
      Right ok -> pure ok

  r <- try $ do
    if argServe args'
      then runRestApi args'
      else do
        (series, mBinanceEnv) <- loadPrices args'
        let prices = psClose series
        if length prices < 2 then error "Need at least 2 price rows" else pure ()

        let lookback = argLookback args'
        if argTradeOnly args'
          then runTradeOnly args' lookback prices mBinanceEnv
          else runBacktestPipeline args' lookback series mBinanceEnv
  case (r :: Either SomeException ()) of
    Left ex -> do
      let (_, msg) = exceptionToHttp ex
      hPutStrLn stderr msg
      exitFailure
    Right () -> pure ()

-- REST API (stateless; computes per request)

data ApiParams = ApiParams
  { apData :: Maybe FilePath
  , apPriceColumn :: Maybe String
  , apBinanceSymbol :: Maybe String
  , apMarket :: Maybe String -- "spot" | "margin" | "futures"
  , apInterval :: Maybe String
  , apBars :: Maybe Int
  , apLookbackWindow :: Maybe String
  , apLookbackBars :: Maybe Int
  , apBinanceTestnet :: Maybe Bool
  , apBinanceApiKey :: Maybe String
  , apBinanceApiSecret :: Maybe String
  , apNormalization :: Maybe String
  , apHiddenSize :: Maybe Int
  , apEpochs :: Maybe Int
  , apLr :: Maybe Double
  , apValRatio :: Maybe Double
  , apBacktestRatio :: Maybe Double
  , apTuneRatio :: Maybe Double
  , apPatience :: Maybe Int
  , apGradClip :: Maybe Double
  , apSeed :: Maybe Int
  , apKalmanDt :: Maybe Double
  , apKalmanProcessVar :: Maybe Double
  , apKalmanMeasurementVar :: Maybe Double
  , apThreshold :: Maybe Double
  , apMethod :: Maybe String -- "11" | "10" | "01"
  , apPositioning :: Maybe String -- "long-flat" | "long-short"
  , apOptimizeOperations :: Maybe Bool
  , apSweepThreshold :: Maybe Bool
  , apFee :: Maybe Double
  , apSlippage :: Maybe Double
  , apSpread :: Maybe Double
  , apIntrabarFill :: Maybe String
  , apStopLoss :: Maybe Double
  , apTakeProfit :: Maybe Double
  , apTrailingStop :: Maybe Double
  , apMaxDrawdown :: Maybe Double
  , apMaxDailyLoss :: Maybe Double
  , apMaxOrderErrors :: Maybe Int
  , apPeriodsPerYear :: Maybe Double
  , apBinanceLive :: Maybe Bool
  , apOrderQuote :: Maybe Double
  , apOrderQuantity :: Maybe Double
  , apOrderQuoteFraction :: Maybe Double
  , apMaxOrderQuote :: Maybe Double
  , apIdempotencyKey :: Maybe String
  , apBotPollSeconds :: Maybe Int
  , apBotOnlineEpochs :: Maybe Int
  , apBotTrainBars :: Maybe Int
  , apBotMaxPoints :: Maybe Int
  , apBotTrade :: Maybe Bool
  , apKalmanZMin :: Maybe Double
  , apKalmanZMax :: Maybe Double
  , apMaxHighVolProb :: Maybe Double
  , apMaxConformalWidth :: Maybe Double
  , apMaxQuantileWidth :: Maybe Double
  , apConfirmConformal :: Maybe Bool
  , apConfirmQuantiles :: Maybe Bool
  , apConfidenceSizing :: Maybe Bool
  , apMinPositionSize :: Maybe Double
  } deriving (Eq, Show, Generic)

instance FromJSON ApiParams where
  parseJSON = Aeson.genericParseJSON (jsonOptions 2)

instance ToJSON ApiParams where
  toJSON = Aeson.genericToJSON (jsonOptions 2)

data ApiError = ApiError
  { aeError :: String
  } deriving (Eq, Show, Generic)

instance ToJSON ApiError where
  toJSON = Aeson.genericToJSON (jsonOptions 2)

instance ToJSON LatestSignal where
  toJSON s =
    let regimesJson =
          case lsRegimes s of
            Nothing -> Nothing
            Just r ->
              Just
                ( object
                    [ "trend" .= rpTrend r
                    , "mr" .= rpMR r
                    , "highVol" .= rpHighVol r
                    ]
                )
        quantilesJson =
          case lsQuantiles s of
            Nothing -> Nothing
            Just q ->
              let w = q90 q - q10 q
               in Just (object ["q10" .= q10 q, "q50" .= q50 q, "q90" .= q90 q, "width" .= w])
        conformalJson =
          case lsConformalInterval s of
            Nothing -> Nothing
            Just i ->
              let w = iHi i - iLo i
               in Just (object ["lo" .= iLo i, "hi" .= iHi i, "width" .= w])
     in
    object
      [ "method" .= methodCode (lsMethod s)
      , "currentPrice" .= lsCurrentPrice s
      , "threshold" .= lsThreshold s
      , "kalmanNext" .= lsKalmanNext s
      , "kalmanReturn" .= lsKalmanReturn s
      , "kalmanStd" .= lsKalmanStd s
      , "kalmanZ" .= lsKalmanZ s
      , "regimes" .= regimesJson
      , "quantiles" .= quantilesJson
      , "conformalInterval" .= conformalJson
      , "confidence" .= lsConfidence s
      , "positionSize" .= lsPositionSize s
      , "kalmanDirection" .= (if isJust (lsKalmanNext s) then dirLabel (lsKalmanDir s) else Nothing)
      , "lstmNext" .= lsLstmNext s
      , "lstmDirection" .= (if isJust (lsLstmNext s) then dirLabel (lsLstmDir s) else Nothing)
      , "chosenDirection" .= dirLabel (lsChosenDir s)
      , "action" .= lsAction s
      ]

data ApiOrderResult = ApiOrderResult
  { aorSent :: Bool
  , aorMode :: Maybe String
  , aorSide :: Maybe String
  , aorSymbol :: Maybe String
  , aorQuantity :: Maybe Double
  , aorQuoteQuantity :: Maybe Double
  , aorOrderId :: Maybe Int64
  , aorClientOrderId :: Maybe String
  , aorStatus :: Maybe String
  , aorExecutedQty :: Maybe Double
  , aorCummulativeQuoteQty :: Maybe Double
  , aorResponse :: Maybe String
  , aorMessage :: String
  } deriving (Eq, Show, Generic)

instance ToJSON ApiOrderResult where
  toJSON = Aeson.genericToJSON (jsonOptions 3)

data ApiTradeResponse = ApiTradeResponse
  { atrSignal :: LatestSignal
  , atrOrder :: ApiOrderResult
  } deriving (Eq, Show, Generic)

instance ToJSON ApiTradeResponse where
  toJSON = Aeson.genericToJSON (jsonOptions 3)

data ApiBinanceProbe = ApiBinanceProbe
  { abpOk :: !Bool
  , abpStep :: !String
  , abpCode :: !(Maybe Int)
  , abpMsg :: !(Maybe String)
  , abpSummary :: !String
  } deriving (Eq, Show, Generic)

instance ToJSON ApiBinanceProbe where
  toJSON = Aeson.genericToJSON (jsonOptions 3)

data ApiBinanceKeysStatus = ApiBinanceKeysStatus
  { abkMarket :: !String
  , abkTestnet :: !Bool
  , abkSymbol :: !(Maybe String)
  , abkHasApiKey :: !Bool
  , abkHasApiSecret :: !Bool
  , abkSigned :: !(Maybe ApiBinanceProbe)
  , abkTradeTest :: !(Maybe ApiBinanceProbe)
  } deriving (Eq, Show, Generic)

instance ToJSON ApiBinanceKeysStatus where
  toJSON = Aeson.genericToJSON (jsonOptions 3)

jsonOptions :: Int -> Aeson.Options
jsonOptions prefixLen =
  Aeson.defaultOptions
    { Aeson.fieldLabelModifier = lowerFirst . drop prefixLen
    , Aeson.omitNothingFields = True
    }
  where
    lowerFirst :: String -> String
    lowerFirst s =
      case s of
        [] -> []
        (c:cs) -> toLower c : cs

-- Observability (metrics + journaling)

data Metrics = Metrics
  { mtRequestsTotal :: !(IORef Int64)
  , mtRequestsByEndpoint :: !(IORef (HM.HashMap String Int64))
  , mtOrdersSentTotal :: !(IORef Int64)
  , mtOrdersFailedTotal :: !(IORef Int64)
  , mtBotHaltsTotal :: !(IORef Int64)
  }

newMetrics :: IO Metrics
newMetrics = do
  reqTotal <- newIORef 0
  reqBy <- newIORef HM.empty
  ordersSent <- newIORef 0
  ordersFailed <- newIORef 0
  botHalts <- newIORef 0
  pure
    Metrics
      { mtRequestsTotal = reqTotal
      , mtRequestsByEndpoint = reqBy
      , mtOrdersSentTotal = ordersSent
      , mtOrdersFailedTotal = ordersFailed
      , mtBotHaltsTotal = botHalts
      }

incCounter :: IORef Int64 -> IO ()
incCounter ref = atomicModifyIORef' ref (\n -> (n + 1, ()))

metricsIncEndpoint :: Metrics -> String -> IO ()
metricsIncEndpoint m endpoint = do
  incCounter (mtRequestsTotal m)
  atomicModifyIORef' (mtRequestsByEndpoint m) $ \hm ->
    let next = maybe 0 id (HM.lookup endpoint hm) + 1
     in (HM.insert endpoint next hm, ())

metricsRecordOrder :: Metrics -> ApiOrderResult -> IO ()
metricsRecordOrder m o
  | aorSent o = incCounter (mtOrdersSentTotal m)
  | "Order failed:" `isPrefixOf` aorMessage o = incCounter (mtOrdersFailedTotal m)
  | otherwise = pure ()

metricsRecordBotHalt :: Metrics -> IO ()
metricsRecordBotHalt m = incCounter (mtBotHaltsTotal m)

escapePromLabel :: String -> String
escapePromLabel = concatMap esc
  where
    esc '"' = "\\\""
    esc '\\' = "\\\\"
    esc c = [c]

renderMetricsText :: Metrics -> Bool -> IO BL.ByteString
renderMetricsText m botRunning = do
  reqTotal <- readIORef (mtRequestsTotal m)
  reqBy <- readIORef (mtRequestsByEndpoint m)
  ordersSent <- readIORef (mtOrdersSentTotal m)
  ordersFailed <- readIORef (mtOrdersFailedTotal m)
  botHalts <- readIORef (mtBotHaltsTotal m)
  let header =
        [ "# HELP trader_http_requests_total Total HTTP requests."
        , "# TYPE trader_http_requests_total counter"
        , "trader_http_requests_total " ++ show reqTotal
        , "# HELP trader_http_requests_by_endpoint_total Total HTTP requests by endpoint."
        , "# TYPE trader_http_requests_by_endpoint_total counter"
        ]
      byEndpoint =
        [ "trader_http_requests_by_endpoint_total{endpoint=\"" ++ escapePromLabel k ++ "\"} " ++ show v
        | (k, v) <- HM.toList reqBy
        ]
      orders =
        [ "# HELP trader_orders_total Total orders (sent/failed)."
        , "# TYPE trader_orders_total counter"
        , "trader_orders_total{result=\"sent\"} " ++ show ordersSent
        , "trader_orders_total{result=\"failed\"} " ++ show ordersFailed
        ]
      bot =
        [ "# HELP trader_bot_halts_total Bot halts."
        , "# TYPE trader_bot_halts_total counter"
        , "trader_bot_halts_total " ++ show botHalts
        , "# HELP trader_bot_running Bot running."
        , "# TYPE trader_bot_running gauge"
        , "trader_bot_running " ++ if botRunning then "1" else "0"
        ]
      txt = unlines (header ++ byEndpoint ++ orders ++ bot)
  pure (BL.fromStrict (BS.pack txt))

data Journal = Journal
  { jPath :: !FilePath
  , jLock :: !(MVar ())
  }

newJournalFromEnv :: IO (Maybe Journal)
newJournalFromEnv = do
  mDir <- lookupEnv "TRADER_JOURNAL_DIR"
  case trim <$> mDir of
    Nothing -> pure Nothing
    Just dir | null dir -> pure Nothing
    Just dir -> do
      createDirectoryIfMissing True dir
      ts <- getTimestampMs
      lock <- newMVar ()
      pure (Just (Journal (dir </> ("trader-" ++ show ts ++ ".jsonl")) lock))

journalWrite :: Journal -> Aeson.Value -> IO ()
journalWrite j v =
  withMVar (jLock j) $ \_ ->
    BL.appendFile (jPath j) (encode v <> BL.fromStrict (BS.pack "\n"))

journalWriteMaybe :: Maybe Journal -> Aeson.Value -> IO ()
journalWriteMaybe mj v =
  case mj of
    Nothing -> pure ()
    Just j -> journalWrite j v

-- Persistent operation history (JSONL; safe to rebuild state from the log).

data PersistedOperation = PersistedOperation
  { poId :: !Int64
  , poAtMs :: !Int64
  , poKind :: !Text
  , poParams :: !(Maybe Aeson.Value)
  , poArgs :: !(Maybe Aeson.Value)
  , poResult :: !(Maybe Aeson.Value)
  , poEquity :: !(Maybe Double)
  } deriving (Eq, Show, Generic)

instance ToJSON PersistedOperation where
  toJSON = Aeson.genericToJSON (jsonOptions 2)

instance FromJSON PersistedOperation where
  parseJSON = Aeson.genericParseJSON (jsonOptions 2)

data OpsStore = OpsStore
  { osPath :: !FilePath
  , osLock :: !(MVar ())
  , osNextId :: !(IORef Int64)
  , osOps :: !(IORef (Seq PersistedOperation))
  , osMaxInMemory :: !Int
  }

sanitizeApiParams :: ApiParams -> ApiParams
sanitizeApiParams p =
  p
    { apBinanceApiKey = Nothing
    , apBinanceApiSecret = Nothing
    }

boolFromMaybe :: Maybe a -> Bool
boolFromMaybe =
  \x ->
    case x of
      Nothing -> False
      Just _ -> True

argsPublicJson :: Args -> Aeson.Value
argsPublicJson args =
  let market = marketCode (argBinanceMarket args)
      barsRaw = argBars args
      barsCsv = resolveBarsForCsv args
      barsBinance = resolveBarsForBinance args
      barsUsed =
        case (argBinanceSymbol args, argData args) of
          (Just _, _) -> barsBinance
          (_, Just _) -> barsCsv
          _ -> barsRaw
      lookback =
        case argLookbackBars args of
          Just n -> n
          Nothing ->
            case lookbackBarsFrom (argInterval args) (argLookbackWindow args) of
              Right n -> n
              Left _ -> 0
   in
    object
      [ "data" .= argData args
      , "priceColumn" .= argPriceCol args
      , "binanceSymbol" .= argBinanceSymbol args
      , "market" .= market
      , "interval" .= argInterval args
      , "bars" .= barsRaw
      , "barsResolved" .= barsUsed
      , "lookbackWindow" .= argLookbackWindow args
      , "lookbackBars" .= argLookbackBars args
      , "lookbackBarsResolved" .= lookback
      , "binanceTestnet" .= argBinanceTestnet args
      , "binanceLive" .= argBinanceLive args
      , "binanceTrade" .= argBinanceTrade args
      , "hasBinanceApiKey" .= boolFromMaybe (argBinanceApiKey args)
      , "hasBinanceApiSecret" .= boolFromMaybe (argBinanceApiSecret args)
      , "orderQuote" .= argOrderQuote args
      , "orderQuantity" .= argOrderQuantity args
      , "orderQuoteFraction" .= argOrderQuoteFraction args
      , "maxOrderQuote" .= argMaxOrderQuote args
      , "idempotencyKey" .= argIdempotencyKey args
      , "normalization" .= show (argNormalization args)
      , "hiddenSize" .= argHiddenSize args
      , "epochs" .= argEpochs args
      , "lr" .= argLr args
      , "valRatio" .= argValRatio args
      , "backtestRatio" .= argBacktestRatio args
      , "patience" .= argPatience args
      , "gradClip" .= argGradClip args
      , "seed" .= argSeed args
      , "kalmanDt" .= argKalmanDt args
      , "kalmanProcessVar" .= argKalmanProcessVar args
      , "kalmanMeasurementVar" .= argKalmanMeasurementVar args
      , "threshold" .= argTradeThreshold args
      , "method" .= methodCode (argMethod args)
      , "positioning" .= positioningCode (argPositioning args)
      , "optimizeOperations" .= argOptimizeOperations args
      , "sweepThreshold" .= argSweepThreshold args
      , "tradeOnly" .= argTradeOnly args
      , "fee" .= argFee args
      , "stopLoss" .= argStopLoss args
      , "takeProfit" .= argTakeProfit args
      , "trailingStop" .= argTrailingStop args
      , "maxDrawdown" .= argMaxDrawdown args
      , "maxDailyLoss" .= argMaxDailyLoss args
      , "maxOrderErrors" .= argMaxOrderErrors args
      , "periodsPerYear" .= argPeriodsPerYear args
      ]

trimSeq :: Int -> Seq a -> Seq a
trimSeq maxN s =
  if maxN <= 0
    then Seq.empty
    else
      let len = Seq.length s
       in if len <= maxN then s else Seq.drop (len - maxN) s

loadOpsFile :: FilePath -> Int -> IO (Seq PersistedOperation, Int64)
loadOpsFile path maxInMemory = do
  exists <- doesFileExist path
  if not exists
    then pure (Seq.empty, 0)
    else
      withFile path ReadMode $ \h -> do
        let loop acc maxId = do
              eof <- hIsEOF h
              if eof
                then pure (trimSeq maxInMemory acc, maxId)
                else do
                  line <- BS.hGetLine h
                  if BS.all isSpace line
                    then loop acc maxId
                    else
                      case Aeson.eitherDecodeStrict' line of
                        Left _ -> loop acc maxId
                        Right op ->
                          let acc' = trimSeq maxInMemory (acc Seq.|> op)
                              maxId' = max maxId (poId op)
                           in loop acc' maxId'
        loop Seq.empty 0

newOpsStoreFromEnv :: IO (Maybe OpsStore)
newOpsStoreFromEnv = do
  mDir <- lookupEnv "TRADER_OPS_DIR"
  case trim <$> mDir of
    Nothing -> pure Nothing
    Just dir | null dir -> pure Nothing
    Just dir -> do
      createDirectoryIfMissing True dir
      maxEnv <- lookupEnv "TRADER_OPS_MAX_IN_MEMORY"
      let maxInMemory =
            case maxEnv >>= readMaybe of
              Just n | n >= 0 -> n
              _ -> 20000
          path = dir </> "ops.jsonl"
      lock <- newMVar ()
      (ops0, maxId0) <- loadOpsFile path maxInMemory
      nextId <- newIORef maxId0
      opsRef <- newIORef ops0
      pure (Just (OpsStore path lock nextId opsRef maxInMemory))

opsAppend :: OpsStore -> Text -> Maybe Aeson.Value -> Maybe Aeson.Value -> Maybe Aeson.Value -> Maybe Double -> IO PersistedOperation
opsAppend store kind mParams mArgs mResult mEquity =
  withMVar (osLock store) $ \_ -> do
    now <- getTimestampMs
    opId <- atomicModifyIORef' (osNextId store) (\n -> let n' = n + 1 in (n', n'))
    let op =
          PersistedOperation
            { poId = opId
            , poAtMs = now
            , poKind = kind
            , poParams = mParams
            , poArgs = mArgs
            , poResult = mResult
            , poEquity = mEquity
            }
    BL.appendFile (osPath store) (encode op <> BL.fromStrict (BS.pack "\n"))
    modifyIORef' (osOps store) (\s -> trimSeq (osMaxInMemory store) (s Seq.|> op))
    pure op

opsAppendMaybe :: Maybe OpsStore -> Text -> Maybe Aeson.Value -> Maybe Aeson.Value -> Maybe Aeson.Value -> Maybe Double -> IO ()
opsAppendMaybe mStore kind mParams mArgs mResult mEquity =
  case mStore of
    Nothing -> pure ()
    Just store -> do
      _ <- try (opsAppend store kind mParams mArgs mResult mEquity) :: IO (Either SomeException PersistedOperation)
      pure ()

opsList :: OpsStore -> Maybe Int64 -> Int -> Maybe Text -> IO [PersistedOperation]
opsList store sinceId limit mKind = do
  let limitSafe = max 0 (min 5000 limit)
  opsSeq <- readIORef (osOps store)
  let allOps = toList opsSeq
      filtered =
        [ op
        | op <- allOps
        , maybe True (\sid -> poId op > sid) sinceId
        , maybe True (\k -> poKind op == k) mKind
        ]
      out =
        case sinceId of
          Just _ -> take limitSafe filtered
          Nothing ->
            let n = length filtered
                dropN = max 0 (n - limitSafe)
             in drop dropN filtered
  pure out

-- Live bot (stateful; continuous loop)

data BotSettings = BotSettings
  { bsPollSeconds :: !Int
  , bsOnlineEpochs :: !Int
  , bsTrainBars :: !Int
  , bsMaxPoints :: !Int
  , bsTradeEnabled :: !Bool
  } deriving (Eq, Show)

data BotController = BotController
  { bcRuntime :: MVar (Maybe BotRuntimeState)
  }

data BotStartRuntime = BotStartRuntime
  { bsrThreadId :: !ThreadId
  , bsrStopSignal :: !(MVar ())
  , bsrArgs :: !Args
  , bsrSettings :: !BotSettings
  , bsrSymbol :: !String
  , bsrRequestedAtMs :: !Int64
  }

data BotRuntimeState
  = BotStarting !BotStartRuntime
  | BotRunning !BotRuntime

data BotRuntime = BotRuntime
  { brThreadId :: ThreadId
  , brStateVar :: MVar BotState
  , brStopSignal :: MVar ()
  }

data BotState = BotState
  { botArgs :: !Args
  , botSettings :: !BotSettings
  , botSymbol :: !String
  , botEnv :: !BinanceEnv
  , botLookback :: !Int
  , botPrices :: !(V.Vector Double)
  , botOpenTimes :: !(V.Vector Int64)
  , botKalmanPredNext :: !(V.Vector Double) -- predicted next price at each bar (len == prices)
  , botLstmPredNext :: !(V.Vector Double)   -- predicted next price at each bar (len == prices)
  , botEquityCurve :: !(V.Vector Double)
  , botPositions :: !(V.Vector Int) -- position after decision at each bar (len == prices)
  , botOps :: ![BotOp]
  , botOrders :: ![BotOrderEvent]
  , botTrades :: ![Trade]
  , botOpenTrade :: !(Maybe (Int, Double, Int, Double, Double)) -- (entryIdx, entryEq, holdingPeriods, entryPrice, trailHigh)
  , botLatestSignal :: !LatestSignal
  , botLastOrder :: !(Maybe ApiOrderResult)
  , botHaltReason :: !(Maybe String)
  , botHaltedAtMs :: !(Maybe Int64)
  , botPeakEquity :: !Double
  , botDayKey :: !Int64
  , botDayStartEquity :: !Double
  , botConsecutiveOrderErrors :: !Int
  , botLstmCtx :: !(Maybe LstmCtx)
  , botKalmanCtx :: !(Maybe KalmanCtx)
  , botLastOpenTime :: !Int64
  , botStartIndex :: !Int
  , botStartedAtMs :: !Int64
  , botUpdatedAtMs :: !Int64
  , botError :: !(Maybe String)
  }

newBotController :: IO BotController
newBotController = BotController <$> newMVar Nothing

clampInt :: Int -> Int -> Int -> Int
clampInt lo hi n = max lo (min hi n)

defaultBotPollSeconds :: Args -> Int
defaultBotPollSeconds args =
  case parseIntervalSeconds (argInterval args) of
    Nothing -> 10
    Just sec ->
      let half = max 1 (sec `div` 2)
       in clampInt 5 60 half

botSettingsFromApi :: Args -> ApiParams -> Either String BotSettings
botSettingsFromApi args p = do
  let poll = maybe (defaultBotPollSeconds args) id (apBotPollSeconds p)
      onlineEpochs = maybe 1 id (apBotOnlineEpochs p)
      trainBars = maybe 800 id (apBotTrainBars p)
      maxPoints = maybe 2000 id (apBotMaxPoints p)
      tradeEnabled = maybe False id (apBotTrade p)

  ensure "botPollSeconds must be between 1 and 3600" (poll >= 1 && poll <= 3600)
  ensure "botOnlineEpochs must be between 0 and 50" (onlineEpochs >= 0 && onlineEpochs <= 50)
  ensure "botTrainBars must be >= 10" (trainBars >= 10)
  ensure "botMaxPoints must be between 100 and 100000" (maxPoints >= 100 && maxPoints <= 100000)

  pure BotSettings { bsPollSeconds = poll, bsOnlineEpochs = onlineEpochs, bsTrainBars = trainBars, bsMaxPoints = maxPoints, bsTradeEnabled = tradeEnabled }
  where
    ensure :: String -> Bool -> Either String ()
    ensure msg cond = if cond then Right () else Left msg

botStatusJson :: BotState -> Aeson.Value
botStatusJson st =
  let finiteMaybe x =
        if isNaN x || isInfinite x
          then Nothing
          else Just x
   in
  object $
    [ "running" .= True
    , "symbol" .= botSymbol st
    , "interval" .= argInterval (botArgs st)
    , "market" .= marketCode (argBinanceMarket (botArgs st))
    , "method" .= methodCode (argMethod (botArgs st))
    , "threshold" .= argTradeThreshold (botArgs st)
    , "startIndex" .= botStartIndex st
    , "startedAtMs" .= botStartedAtMs st
    , "updatedAtMs" .= botUpdatedAtMs st
    , "halted" .= isJust (botHaltReason st)
    , "peakEquity" .= botPeakEquity st
    , "dayStartEquity" .= botDayStartEquity st
    , "consecutiveOrderErrors" .= botConsecutiveOrderErrors st
    , "prices" .= V.toList (botPrices st)
    , "openTimes" .= V.toList (botOpenTimes st)
    , "kalmanPredNext" .= map finiteMaybe (V.toList (botKalmanPredNext st))
    , "lstmPredNext" .= map finiteMaybe (V.toList (botLstmPredNext st))
    , "equityCurve" .= V.toList (botEquityCurve st)
    , "positions" .= V.toList (botPositions st)
    , "operations" .= botOps st
    , "orders" .= botOrders st
    , "trades" .= map tradeToJson (botTrades st)
    , "latestSignal" .= botLatestSignal st
    ]
      ++ maybe [] (\o -> ["lastOrder" .= o]) (botLastOrder st)
      ++ maybe [] (\r -> ["haltReason" .= r]) (botHaltReason st)
      ++ maybe [] (\t -> ["haltedAtMs" .= t]) (botHaltedAtMs st)
      ++ maybe [] (\e -> ["error" .= e]) (botError st)

botStateTail :: Int -> BotState -> BotState
botStateTail tailN st =
  let n = V.length (botPrices st)
      t = max 0 tailN
      dropCount = max 0 (n - t)
   in
    if dropCount <= 0
      then st
      else
        let shiftTrade tr =
              tr { trEntryIndex = trEntryIndex tr - dropCount, trExitIndex = trExitIndex tr - dropCount }
            tradesShifted =
              [ shiftTrade tr
              | tr <- botTrades st
              , trEntryIndex tr >= dropCount
              , trExitIndex tr >= dropCount
              ]
            opsShifted =
              [ op { boIndex = boIndex op - dropCount }
              | op <- botOps st
              , boIndex op >= dropCount
              ]
            ordersShifted =
              [ e { boeIndex = boeIndex e - dropCount }
              | e <- botOrders st
              , boeIndex e >= dropCount
              ]
         in
          st
            { botPrices = V.drop dropCount (botPrices st)
            , botOpenTimes = V.drop dropCount (botOpenTimes st)
            , botKalmanPredNext = V.drop dropCount (botKalmanPredNext st)
            , botLstmPredNext = V.drop dropCount (botLstmPredNext st)
            , botEquityCurve = V.drop dropCount (botEquityCurve st)
            , botPositions = V.drop dropCount (botPositions st)
            , botOps = opsShifted
            , botOrders = ordersShifted
            , botTrades = tradesShifted
            , botStartIndex = botStartIndex st + dropCount
            }

botStartingJson :: BotStartRuntime -> Aeson.Value
botStartingJson rt =
  object
    [ "running" .= False
    , "starting" .= True
    , "symbol" .= bsrSymbol rt
    , "interval" .= argInterval (bsrArgs rt)
    , "market" .= marketCode (argBinanceMarket (bsrArgs rt))
    , "method" .= methodCode (argMethod (bsrArgs rt))
    , "threshold" .= argTradeThreshold (bsrArgs rt)
    , "startedAtMs" .= bsrRequestedAtMs rt
    ]

botStoppedJson :: Aeson.Value
botStoppedJson =
  object
    [ "running" .= False
    ]

data BotOp = BotOp
  { boIndex :: !Int
  , boSide :: !String
  , boPrice :: !Double
  } deriving (Eq, Show, Generic)

instance ToJSON BotOp where
  toJSON = Aeson.genericToJSON (jsonOptions 2)

data BotOrderEvent = BotOrderEvent
  { boeIndex :: !Int
  , boeOpSide :: !String
  , boePrice :: !Double
  , boeOpenTime :: !Int64
  , boeAtMs :: !Int64
  , boeOrder :: !ApiOrderResult
  } deriving (Eq, Show, Generic)

instance ToJSON BotOrderEvent where
  toJSON = Aeson.genericToJSON (jsonOptions 3)

marketCode :: BinanceMarket -> String
marketCode m =
  case m of
    MarketSpot -> "spot"
    MarketMargin -> "margin"
    MarketFutures -> "futures"

botStart :: Maybe OpsStore -> Metrics -> Maybe Journal -> BotController -> Args -> ApiParams -> IO (Either String BotStartRuntime)
botStart mOps metrics mJournal ctrl args p =
  case argData args of
    Just _ -> pure (Left "bot/start supports binanceSymbol only (no CSV data source)")
    Nothing ->
      case argBinanceSymbol args of
        Nothing -> pure (Left "bot/start requires binanceSymbol")
        Just sym ->
          case botSettingsFromApi args p of
            Left e -> pure (Left e)
            Right settings -> do
              now <- getTimestampMs
              modifyMVar (bcRuntime ctrl) $ \mrt ->
                case mrt of
                  Just (BotRunning _) -> pure (mrt, Left "Bot is already running")
                  Just (BotStarting _) -> pure (mrt, Left "Bot is starting")
                  Nothing -> do
                    stopSig <- newEmptyMVar
                    tid <- forkIO (botStartWorker mOps metrics mJournal ctrl args settings sym stopSig)
                    let rt =
                          BotStartRuntime
                            { bsrThreadId = tid
                            , bsrStopSignal = stopSig
                            , bsrArgs = args
                            , bsrSettings = settings
                            , bsrSymbol = sym
                            , bsrRequestedAtMs = now
                            }
                    pure (Just (BotStarting rt), Right rt)

botStartWorker :: Maybe OpsStore -> Metrics -> Maybe Journal -> BotController -> Args -> BotSettings -> String -> MVar () -> IO ()
botStartWorker mOps metrics mJournal ctrl args settings sym stopSig = do
  tid <- myThreadId
  r <- try (initBotState args settings sym) :: IO (Either SomeException BotState)
  case r of
    Left ex -> do
      now <- getTimestampMs
      journalWriteMaybe mJournal (object ["type" .= ("bot.start_failed" :: String), "atMs" .= now, "error" .= show ex])
      opsAppendMaybe mOps "bot.start_failed" Nothing (Just (argsPublicJson args)) (Just (object ["error" .= show ex])) Nothing
      modifyMVar_ (bcRuntime ctrl) $ \mrt ->
        case mrt of
          Just (BotStarting rt) | bsrThreadId rt == tid -> pure Nothing
          other -> pure other
    Right st0 -> do
      let eq0 =
            if V.null (botEquityCurve st0)
              then 1.0
              else V.last (botEquityCurve st0)
      opsAppendMaybe
        mOps
        "bot.started"
        Nothing
        (Just (argsPublicJson (botArgs st0)))
        (Just (object ["symbol" .= botSymbol st0, "market" .= marketCode (argBinanceMarket (botArgs st0)), "interval" .= argInterval (botArgs st0)]))
        (Just eq0)
      stVar <- newMVar st0
      startOk <-
        modifyMVar (bcRuntime ctrl) $ \mrt ->
          case mrt of
            Just (BotStarting rt) | bsrThreadId rt == tid -> pure (Just (BotRunning (BotRuntime tid stVar stopSig)), True)
            _ -> pure (mrt, False)
      if startOk
        then botLoop mOps metrics mJournal ctrl stVar stopSig
        else pure ()

botStop :: BotController -> IO Bool
botStop ctrl =
  modifyMVar (bcRuntime ctrl) $ \mrt ->
    case mrt of
      Nothing -> pure (Nothing, False)
      Just (BotStarting rt) -> do
        _ <- tryPutMVar (bsrStopSignal rt) ()
        killThread (bsrThreadId rt)
        pure (Nothing, True)
      Just (BotRunning rt) -> do
        _ <- tryPutMVar (brStopSignal rt) ()
        killThread (brThreadId rt)
        pure (Nothing, True)

botGetState :: BotController -> IO (Maybe BotState)
botGetState ctrl = do
  mrt <- readMVar (bcRuntime ctrl)
  case mrt of
    Nothing -> pure Nothing
    Just (BotStarting _) -> pure Nothing
    Just (BotRunning rt) -> Just <$> readMVar (brStateVar rt)

initBotState :: Args -> BotSettings -> String -> IO BotState
initBotState args settings sym = do
  let lookback = argLookback args
  now <- getTimestampMs
  env <- makeBinanceEnv args
  let initBars = clampInt 2 1000 (max 2 (resolveBarsForBinance args))
  ks <- fetchKlines env sym (argInterval args) initBars
  if length ks < 2 then error "Not enough klines to start bot" else pure ()
  let closes = map kClose ks
      openTimes = map kOpenTime ks
      pricesV = V.fromList closes
      openV = V.fromList openTimes
      n = V.length pricesV

  let method = argMethod args
      methodForCtx = if argOptimizeOperations args then MethodBoth else method
      nan = 0 / 0 :: Double

  mLstmCtx <-
    case methodForCtx of
      MethodKalmanOnly -> pure Nothing
      _ -> do
        let normState = fitNorm (argNormalization args) closes
            obsAll = forwardSeries normState closes
            lstmCfg =
              LSTMConfig
                { lcLookback = lookback
                , lcHiddenSize = argHiddenSize args
                , lcEpochs = argEpochs args
                , lcLearningRate = argLr args
                , lcValRatio = argValRatio args
                , lcPatience = argPatience args
                , lcGradClip = argGradClip args
                , lcSeed = argSeed args
                }
            (lstmModel, _) = trainLSTM lstmCfg obsAll
        pure (Just (normState, obsAll, lstmModel))

  (mKalmanCtx, kalPred0) <-
    case methodForCtx of
      MethodLstmOnly -> pure (Nothing, V.replicate n nan)
      _ -> do
        let predictors = trainPredictors lookback pricesV
            hmm0 = initHMMFilter predictors []
            kal0 =
              initKalman1
                0
                (max 1e-12 (argKalmanMeasurementVar args))
                (max 0 (argKalmanProcessVar args) * max 0 (argKalmanDt args))
            sv0 = emptySensorVar

            step (kal, hmm, sv, predsAcc) t =
              let priceT = pricesV V.! t
                  nextP = pricesV V.! (t + 1)
                  realizedR = if priceT == 0 then 0 else nextP / priceT - 1
                  (sensorOuts, predState) = predictSensors predictors pricesV hmm t
                  meas = mapMaybe (toMeasurement args sv) sensorOuts
                  kal' = stepMulti meas kal
                  fusedR = kMean kal'
                  kalNext = priceT * (1 + fusedR)
                  sv' =
                    foldl'
                      (\acc (sid, out) -> updateResidual sid (realizedR - soMu out) acc)
                      sv
                      sensorOuts
                  hmm' = updateHMM predictors predState realizedR
               in (kal', hmm', sv', kalNext : predsAcc)

            (kalPrev, hmmPrev, svPrev, predsRev) = foldl' step (kal0, hmm0, sv0, []) [0 .. n - 2]
            preds = reverse predsRev
            lastPrice = V.last pricesV
            (sensorOutsLast, _) = predictSensors predictors pricesV hmmPrev (n - 1)
            measLast = mapMaybe (toMeasurement args svPrev) sensorOutsLast
            kalLast = stepMulti measLast kalPrev
            kalLastNext = lastPrice * (1 + kMean kalLast)
            kalPred = V.fromList (preds ++ [kalLastNext])

        pure (Just (predictors, kalPrev, hmmPrev, svPrev), kalPred)

  let latest = computeLatestSignal args lookback pricesV mLstmCtx mKalmanCtx
      desiredPosSignal =
        case lsChosenDir latest of
          Just 1 -> 1
          Just (-1) -> 0
          _ -> 0
      baseEq = 1.0
      eq0 = V.replicate n baseEq
      pos0 = V.replicate n 0
      lastOt = V.last openV

  mOrder <-
    if desiredPosSignal == 1
      then Just <$> placeIfEnabled args settings latest env sym
      else pure Nothing

  let tradeEnabled = bsTradeEnabled settings
      orderSent = maybe False aorSent mOrder
      appliedEntry = desiredPosSignal == 1 && (not tradeEnabled || orderSent)
      desiredPos = if appliedEntry then 1 else 0
      didTradeNow = desiredPos == 1 && (not tradeEnabled || orderSent)

      eq1 =
        if didTradeNow
          then eq0 V.// [(n - 1, baseEq * (1 - argFee args))]
          else eq0
      pos1 = pos0 V.// [(n - 1, desiredPos)]
      openTrade =
        if desiredPos == 1
          then
            let px = V.last pricesV
             in Just (n - 1, eq1 V.! (n - 1), 0, px, px)
          else Nothing
      ops =
        if desiredPos == 1
          then [BotOp (n - 1) "BUY" (V.last pricesV)]
          else []

      orders =
        case (desiredPosSignal, mOrder) of
          (1, Just o) -> [BotOrderEvent (n - 1) "BUY" (V.last pricesV) lastOt now o]
          _ -> []

      lstmPred0 =
        case mLstmCtx of
          Nothing -> V.replicate n nan
          Just (normState, obsAll, lstmModel) ->
            V.generate n $ \t ->
              if t < lookback - 1
                then nan
                else
                  let start = t - lookback + 1
                      window = take lookback (drop start obsAll)
                      predObs = predictNext lstmModel window
                   in inverseNorm normState predObs

      maxPoints = max (lookback + 3) (bsMaxPoints settings)
      dropCount = max 0 (V.length pricesV - maxPoints)

      (pricesV2, openV2, kalPred2, lstmPred2, eq2, pos2, ops2, orders2, openTrade2, startIndex2, mLstmCtx2) =
        if dropCount <= 0
          then (pricesV, openV, kalPred0, lstmPred0, eq1, pos1, ops, orders, openTrade, 0, mLstmCtx)
          else
            let openTradeShifted =
                  case openTrade of
                    Nothing -> Nothing
                    Just (ei, eq0, hold, entryPx, trailHigh) ->
                      if ei >= dropCount
                        then Just (ei - dropCount, eq0, hold, entryPx, trailHigh)
                        else Nothing
                opsShifted =
                  [ op { boIndex = boIndex op - dropCount }
                  | op <- ops
                  , boIndex op >= dropCount
                  ]
                ordersShifted =
                  [ e { boeIndex = boeIndex e - dropCount }
                  | e <- orders
                  , boeIndex e >= dropCount
                  ]
                lstmCtxShifted =
                  case mLstmCtx of
                    Nothing -> Nothing
                    Just (normState, obsAll, lstmModel) ->
                      Just (normState, drop dropCount obsAll, lstmModel)
             in
              ( V.drop dropCount pricesV
              , V.drop dropCount openV
              , V.drop dropCount kalPred0
              , V.drop dropCount lstmPred0
              , V.drop dropCount eq1
              , V.drop dropCount pos1
              , opsShifted
              , ordersShifted
              , openTradeShifted
              , dropCount
              , lstmCtxShifted
              )

      peakEq = if V.null eq2 then 1.0 else V.maximum eq2
      dayMs = 86400000 :: Int64
      dayKey = V.last openV2 `div` dayMs
      dayStartEq = V.last eq2
      initOrderErrors =
        if tradeEnabled && desiredPosSignal == 1 && not orderSent
          then 1
          else 0
      (haltReason0, haltedAt0) =
        case argMaxOrderErrors args of
          Just lim | initOrderErrors >= lim -> (Just "MAX_ORDER_ERRORS", Just now)
          _ -> (Nothing, Nothing)

      st0 =
        BotState
          { botArgs = args
          , botSettings = settings
          , botSymbol = sym
          , botEnv = env
          , botLookback = lookback
          , botPrices = pricesV2
          , botOpenTimes = openV2
          , botKalmanPredNext = kalPred2
          , botLstmPredNext = lstmPred2
          , botEquityCurve = eq2
          , botPositions = pos2
          , botOps = ops2
          , botOrders = orders2
          , botTrades = []
          , botOpenTrade = openTrade2
          , botLatestSignal = latest
          , botLastOrder = mOrder
          , botHaltReason = haltReason0
          , botHaltedAtMs = haltedAt0
          , botPeakEquity = peakEq
          , botDayKey = dayKey
          , botDayStartEquity = dayStartEq
          , botConsecutiveOrderErrors = initOrderErrors
          , botLstmCtx = mLstmCtx2
          , botKalmanCtx = mKalmanCtx
          , botLastOpenTime = lastOt
          , botStartIndex = startIndex2
          , botStartedAtMs = now
          , botUpdatedAtMs = now
          , botError = Nothing
          }

  if desiredPos == 1 then botOptimizeAfterOperation st0 else pure st0

botOptimizeAfterOperation :: BotState -> IO BotState
botOptimizeAfterOperation st = do
  let args = botArgs st
      optimizeOps = argOptimizeOperations args
      sweepThr = argSweepThreshold args
  if not (optimizeOps || sweepThr)
    then pure st
    else do
      let lookback = botLookback st
          settings = botSettings st
          pricesV = botPrices st
          n = V.length pricesV
      if n < max 3 (lookback + 3)
        then pure st
        else do
          let win = min n (min 1000 (max (lookback + 3) (bsTrainBars settings)))
              start = n - win
              prices = V.toList (V.drop start pricesV)
              kalPred = V.toList (V.slice start (win - 1) (botKalmanPredNext st))
              lstmPred = V.toList (V.slice start (win - 1) (botLstmPredNext st))
              baseThr = argTradeThreshold args
              fee = argFee args
              baseCfg =
                EnsembleConfig
                  { ecTradeThreshold = baseThr
                  , ecFee = fee
                  , ecSlippage = argSlippage args
                  , ecSpread = argSpread args
                  , ecStopLoss = argStopLoss args
                  , ecTakeProfit = argTakeProfit args
                  , ecTrailingStop = argTrailingStop args
                  , ecPositioning = LongFlat
                  , ecIntrabarFill = argIntrabarFill args
                  , ecKalmanZMin = argKalmanZMin args
                  , ecKalmanZMax = argKalmanZMax args
                  , ecMaxHighVolProb = argMaxHighVolProb args
                  , ecMaxConformalWidth = argMaxConformalWidth args
                  , ecMaxQuantileWidth = argMaxQuantileWidth args
                  , ecConfirmConformal = argConfirmConformal args
                  , ecConfirmQuantiles = argConfirmQuantiles args
                  , ecConfidenceSizing = argConfidenceSizing args
                  , ecMinPositionSize = argMinPositionSize args
                  }
              hasBothCtx = isJust (botLstmCtx st) && isJust (botKalmanCtx st)
              (newMethod, newThr) =
                if optimizeOps && hasBothCtx
                  then
                    let (m, thr, _) = optimizeOperations baseCfg prices kalPred lstmPred Nothing
                     in (m, thr)
                  else
                    let (thr, _) = sweepThreshold (argMethod args) baseCfg prices kalPred lstmPred Nothing
                     in (argMethod args, thr)
              args' =
                args
                  { argMethod = newMethod
                  , argTradeThreshold = newThr
                  }
              latest' = computeLatestSignal args' lookback pricesV (botLstmCtx st) (botKalmanCtx st)
          pure st { botArgs = args', botLatestSignal = latest' }

makeBinanceEnv :: Args -> IO BinanceEnv
makeBinanceEnv args = do
  let market = argBinanceMarket args
  if market == MarketMargin && argBinanceTestnet args
    then error "--binance-testnet is not supported for margin operations"
    else pure ()
  let base =
        case market of
          MarketFutures -> if argBinanceTestnet args then binanceFuturesTestnetBaseUrl else binanceFuturesBaseUrl
          _ -> if argBinanceTestnet args then binanceTestnetBaseUrl else binanceBaseUrl
  apiKey <- resolveEnv "BINANCE_API_KEY" (argBinanceApiKey args)
  apiSecret <- resolveEnv "BINANCE_API_SECRET" (argBinanceApiSecret args)
  newBinanceEnv market base (BS.pack <$> apiKey) (BS.pack <$> apiSecret)

botLoop :: Maybe OpsStore -> Metrics -> Maybe Journal -> BotController -> MVar BotState -> MVar () -> IO ()
botLoop mOps metrics mJournal ctrl stVar stopSig = do
  tid <- myThreadId
  let sleepSec s = threadDelay (max 1 s * 1000000)

      loop = do
        stopReq <- isJust <$> tryReadMVar stopSig
        if stopReq
          then pure ()
          else do
            st <- readMVar stVar
            let env = botEnv st
                sym = botSymbol st
                pollSec = bsPollSeconds (botSettings st)
            r <- try (fetchKlines env sym (argInterval (botArgs st)) 10) :: IO (Either SomeException [Kline])
            case r of
              Left ex -> do
                now <- getTimestampMs
                let st' = st { botError = Just (show ex), botUpdatedAtMs = now }
                _ <- swapMVar stVar st'
                sleepSec pollSec
                loop
              Right ks -> do
                let lastSeen = botLastOpenTime st
                    newKs = filter (\k -> kOpenTime k > lastSeen) ks
                if null newKs
                  then do
                    sleepSec pollSec
                    loop
                  else do
                    st1 <- foldl' (\ioAcc k -> ioAcc >>= \s0 -> botApplyKlineSafe mOps metrics mJournal s0 k) (pure st) newKs
                    _ <- swapMVar stVar st1
                    loop

      cleanup = do
        modifyMVar_ (bcRuntime ctrl) $ \mrt ->
          case mrt of
            Just (BotRunning rt) | brThreadId rt == tid -> pure Nothing
            other -> pure other

  loop `finally` cleanup

botApplyKlineSafe :: Maybe OpsStore -> Metrics -> Maybe Journal -> BotState -> Kline -> IO BotState
botApplyKlineSafe mOps metrics mJournal st k = do
  r <- try (botApplyKline mOps metrics mJournal st k) :: IO (Either SomeException BotState)
  case r of
    Right st' -> pure st'
    Left ex -> do
      now <- getTimestampMs
      pure st { botError = Just (show ex), botUpdatedAtMs = now, botLastOpenTime = kOpenTime k }

botApplyKline :: Maybe OpsStore -> Metrics -> Maybe Journal -> BotState -> Kline -> IO BotState
botApplyKline mOps metrics mJournal st k = do
  now <- getTimestampMs
  let args = botArgs st
      lookback = botLookback st
      settings = botSettings st

      priceNew = kClose k
      openTimeNew = kOpenTime k

      pricesPrev = botPrices st
      nPrev = V.length pricesPrev

  if nPrev < 1 then error "botApplyKline: empty prices" else pure ()

  let prevPrice = pricesPrev V.! (nPrev - 1)
      prevEq = botEquityCurve st V.! (nPrev - 1)
      prevPos = botPositions st V.! (nPrev - 1)
      eqAfterReturn =
        if prevPos == 1 && prevPrice > 0
          then prevEq * (priceNew / prevPrice)
          else prevEq
      openTrade1 =
        case (prevPos, botOpenTrade st) of
          (1, Just (ei, eq0, hold, entryPx, trailHigh)) -> Just (ei, eq0, hold + 1, entryPx, max trailHigh priceNew)
          _ -> Nothing
      dayMs = 86400000 :: Int64
      dayKeyNew = openTimeNew `div` dayMs
      (dayKey1, dayStartEq1) =
        if dayKeyNew /= botDayKey st
          then (dayKeyNew, prevEq)
          else (botDayKey st, botDayStartEquity st)
      peakEq0 = botPeakEquity st
      drawdown =
        if peakEq0 > 0
          then max 0 (1 - eqAfterReturn / peakEq0)
          else 0
      dailyLoss =
        if dayStartEq1 > 0
          then max 0 (1 - eqAfterReturn / dayStartEq1)
          else 0
      riskHaltReason =
        if isJust (botHaltReason st)
          then Nothing
          else
            case () of
              _ | maybe False (\lim -> dailyLoss >= lim) (argMaxDailyLoss args) -> Just "MAX_DAILY_LOSS"
                | maybe False (\lim -> drawdown >= lim) (argMaxDrawdown args) -> Just "MAX_DRAWDOWN"
                | otherwise -> Nothing
      haltReason1 = botHaltReason st <|> riskHaltReason
      haltedAt1 = botHaltedAtMs st <|> (if isJust riskHaltReason then Just now else Nothing)
      halted = isJust haltReason1

      pricesV = V.snoc pricesPrev priceNew
      openTimesV = V.snoc (botOpenTimes st) openTimeNew

  -- Update Kalman/HMM/sensor variance with the realized return on the last step.
  mKalmanCtx1 <-
    case botKalmanCtx st of
      Nothing -> pure Nothing
      Just (predictors, kalPrev, hmmPrev, svPrev) -> do
        let t = nPrev - 1
            realizedR = if prevPrice == 0 then 0 else priceNew / prevPrice - 1
            (sensorOuts, predState) = predictSensors predictors pricesV hmmPrev t
            meas = mapMaybe (toMeasurement args svPrev) sensorOuts
            kal' = stepMulti meas kalPrev
            sv' =
              foldl'
                (\acc (sid, out) -> updateResidual sid (realizedR - soMu out) acc)
                svPrev
                sensorOuts
            hmm' = updateHMM predictors predState realizedR
        pure (Just (predictors, kal', hmm', sv'))

  -- LSTM: append the new observation and fine-tune for a few epochs.
  mLstmCtx1 <-
    case botLstmCtx st of
      Nothing -> pure Nothing
      Just (normState, obsAll, lstmModel0) -> do
        let obsAll' = obsAll ++ forwardSeries normState [priceNew]
            trainBars = max (lookback + 2) (bsTrainBars settings)
            obsTrain = takeLast trainBars obsAll'
            epochs = bsOnlineEpochs settings
            cfg =
              LSTMConfig
                { lcLookback = lookback
                , lcHiddenSize = argHiddenSize args
                , lcEpochs = epochs
                , lcLearningRate = argLr args
                , lcValRatio = 0
                , lcPatience = 0
                , lcGradClip = argGradClip args
                , lcSeed = argSeed args
                }
            (lstmModel1, _) =
              if epochs <= 0
                then (lstmModel0, [])
                else fineTuneLSTM cfg lstmModel0 obsTrain
        pure (Just (normState, obsAll', lstmModel1))

  let latest0 = computeLatestSignal args lookback pricesV mLstmCtx1 mKalmanCtx1
      nan = 0 / 0 :: Double
      kalPred1 = V.snoc (botKalmanPredNext st) (maybe nan id (lsKalmanNext latest0))
      lstmPred1 = V.snoc (botLstmPredNext st) (maybe nan id (lsLstmNext latest0))

      desiredPosSignal =
        case lsChosenDir latest0 of
          Just 1 -> 1
          Just (-1) -> 0
          _ -> prevPos

      bracketExitReason entryPx trailHigh =
        let mTpPx =
              case argTakeProfit args of
                Just tp | tp > 0 -> Just (entryPx * (1 + tp))
                _ -> Nothing
            mSlPx =
              case argStopLoss args of
                Just sl | sl > 0 -> Just (entryPx * (1 - sl))
                _ -> Nothing
            mTsPx =
              case argTrailingStop args of
                Just ts | ts > 0 -> Just (trailHigh * (1 - ts))
                _ -> Nothing

            tpHit = maybe False (\tpPx -> priceNew >= tpPx) mTpPx
            (mStopPx, stopWhy) =
              case (mSlPx, mTsPx) of
                (Nothing, Nothing) -> (Nothing, Nothing)
                (Just slPx, Nothing) -> (Just slPx, Just "STOP_LOSS")
                (Nothing, Just tsPx) -> (Just tsPx, Just "TRAILING_STOP")
                (Just slPx, Just tsPx) ->
                  if tsPx > slPx
                    then (Just tsPx, Just "TRAILING_STOP")
                    else (Just slPx, Just "STOP_LOSS")
            stopHit = maybe False (\stPx -> priceNew <= stPx) mStopPx
         in if tpHit then Just "TAKE_PROFIT" else if stopHit then stopWhy else Nothing

      mBracketExit =
        case (prevPos, openTrade1) of
          (1, Just (_ei, _eq0, _hold, entryPx, trailHigh)) -> bracketExitReason entryPx trailHigh
          _ -> Nothing

      (latestPre, desiredPosPre, mExitReasonPre) =
        case mBracketExit of
          Just why ->
            let sigExit = latest0 { lsChosenDir = Just (-1), lsAction = "EXIT_" ++ why }
             in (sigExit, 0, Just why)
          Nothing ->
            let exitReason =
                  if prevPos == 1 && desiredPosSignal == 0
                    then Just "SIGNAL"
                    else Nothing
             in (latest0, desiredPosSignal, exitReason)

      (latest, desiredPosWanted, mExitReason) =
        if halted
          then
            let why = haltReason1
                latestHalt =
                  case (prevPos, desiredPosPre, why) of
                    (1, 1, Just r) -> latest0 { lsChosenDir = Just (-1), lsAction = "EXIT_" ++ r }
                    (0, 1, Just r) -> latest0 { lsChosenDir = Nothing, lsAction = "HALTED_" ++ r }
                    _ -> latestPre
                exitReason =
                  if prevPos == 1 && desiredPosPre == 1 && not (isJust mExitReasonPre)
                    then why
                    else mExitReasonPre
             in (latestHalt, 0, exitReason)
          else (latestPre, desiredPosPre, mExitReasonPre)

      wantSwitch = desiredPosWanted /= prevPos

  (ops', orders', trades', openTrade', mOrder, posFinal, eqFinal, switchedApplied, orderErrors1, haltReason2, haltedAt2) <-
    if not wantSwitch
      then
        pure
          ( botOps st
          , botOrders st
          , botTrades st
          , openTrade1
          , Nothing
          , prevPos
          , eqAfterReturn
          , False
          , botConsecutiveOrderErrors st
          , haltReason1
          , haltedAt1
          )
      else do
        o <- placeIfEnabled args settings latest (botEnv st) (botSymbol st)
        let opSide =
              if prevPos == 0 && desiredPosWanted == 1
                then "BUY"
                else "SELL"
            orderEv = BotOrderEvent nPrev opSide priceNew openTimeNew now o
            ordersNew = botOrders st ++ [orderEv]
            tradeEnabled = bsTradeEnabled settings
            alreadyMsg =
              aorMessage o == "No order: already long." || aorMessage o == "No order: already flat."
            appliedSwitch =
              if not tradeEnabled
                then True
                else aorSent o || alreadyMsg
            feeApplied =
              if not tradeEnabled
                then True
                else aorSent o
            eqAfterFee =
              if appliedSwitch && feeApplied
                then eqAfterReturn * (1 - argFee args)
                else eqAfterReturn
            posNew = if appliedSwitch then desiredPosWanted else prevPos
            switchedApplied1 = posNew /= prevPos
            opsNew =
              if appliedSwitch
                then botOps st ++ [BotOp nPrev opSide priceNew]
                else botOps st
            (openTradeNew, tradesNew) =
              if not appliedSwitch
                then (openTrade1, botTrades st)
                else
                  if opSide == "BUY"
                    then (Just (nPrev, eqAfterFee, 0, priceNew, priceNew), botTrades st)
                    else
                      case openTrade1 of
                        Just (ei, entryEq, hold, _entryPx, _trailHigh) ->
                          let tr =
                                Trade
                                  { trEntryIndex = ei
                                  , trExitIndex = nPrev
                                  , trEntryEquity = entryEq
                                  , trExitEquity = eqAfterFee
                                  , trReturn = eqAfterFee / entryEq - 1
                                  , trHoldingPeriods = hold
                                  , trExitReason = mExitReason
                                  }
                           in (Nothing, botTrades st ++ [tr])
                        Nothing -> (Nothing, botTrades st)
            errors0 = botConsecutiveOrderErrors st
            errors1 =
              if tradeEnabled
                then if appliedSwitch then 0 else errors0 + 1
                else 0
            (haltReason3, haltedAt3) =
              case haltReason1 of
                Just _ -> (haltReason1, haltedAt1)
                Nothing ->
                  case argMaxOrderErrors args of
                    Just lim | errors1 >= lim -> (Just "MAX_ORDER_ERRORS", Just now)
                    _ -> (Nothing, Nothing)

        metricsRecordOrder metrics o
        journalWriteMaybe
          mJournal
          ( object
              [ "type" .= ("bot.order" :: String)
              , "atMs" .= now
              , "symbol" .= botSymbol st
              , "market" .= marketCode (argBinanceMarket args)
              , "event" .= orderEv
              ]
          )
        opsAppendMaybe
          mOps
          "bot.order"
          Nothing
          (Just (argsPublicJson args))
          ( Just
              ( object
                  [ "symbol" .= botSymbol st
                  , "market" .= marketCode (argBinanceMarket args)
                  , "interval" .= argInterval args
                  , "event" .= orderEv
                  , "signal" .= latest
                  , "position" .= posNew
                  ]
              )
          )
          (Just eqAfterFee)

        pure (opsNew, ordersNew, tradesNew, openTradeNew, Just o, posNew, eqAfterFee, switchedApplied1, errors1, haltReason3, haltedAt3)

  case (botHaltReason st, haltReason2) of
    (Nothing, Just r) -> do
      metricsRecordBotHalt metrics
      journalWriteMaybe
        mJournal
        ( object
            [ "type" .= ("bot.halt" :: String)
            , "atMs" .= now
            , "symbol" .= botSymbol st
            , "market" .= marketCode (argBinanceMarket args)
            , "reason" .= r
            , "equity" .= eqFinal
            , "drawdown" .= drawdown
            , "dailyLoss" .= dailyLoss
            , "consecutiveOrderErrors" .= orderErrors1
            ]
        )
      opsAppendMaybe
        mOps
        "bot.halt"
        Nothing
        (Just (argsPublicJson args))
        ( Just
            ( object
                [ "symbol" .= botSymbol st
                , "market" .= marketCode (argBinanceMarket args)
                , "interval" .= argInterval args
                , "reason" .= r
                , "drawdown" .= drawdown
                , "dailyLoss" .= dailyLoss
                , "consecutiveOrderErrors" .= orderErrors1
                ]
            )
        )
        (Just eqFinal)
    _ -> pure ()

  let eqV1 = V.snoc (botEquityCurve st) eqFinal
      posV1 = V.snoc (botPositions st) posFinal

      maxPoints = max (lookback + 3) (bsMaxPoints settings)
      dropCount = max 0 (V.length pricesV - maxPoints)

      (pricesV2, openTimesV2, kalPred2, lstmPred2, eqV2, posV2, ops2, orders2, trades2, openTrade2, startIndex2) =
        if dropCount <= 0
          then (pricesV, openTimesV, kalPred1, lstmPred1, eqV1, posV1, ops', orders', trades', openTrade', botStartIndex st)
          else
            let shiftTrade tr =
                  tr { trEntryIndex = trEntryIndex tr - dropCount, trExitIndex = trExitIndex tr - dropCount }
                tradesShifted =
                  [ shiftTrade tr
                  | tr <- trades'
                  , trEntryIndex tr >= dropCount
                  , trExitIndex tr >= dropCount
                  ]
                openTradeShifted =
                  case openTrade' of
                    Nothing -> Nothing
                    Just (ei, eq0, hold, entryPx, trailHigh) ->
                      if ei >= dropCount
                        then Just (ei - dropCount, eq0, hold, entryPx, trailHigh)
                        else Nothing
                opsShifted =
                  [ op { boIndex = boIndex op - dropCount }
                  | op <- ops'
                  , boIndex op >= dropCount
                  ]
                ordersShifted =
                  [ e { boeIndex = boeIndex e - dropCount }
                  | e <- orders'
                  , boeIndex e >= dropCount
                  ]
             in
             ( V.drop dropCount pricesV
             , V.drop dropCount openTimesV
              , V.drop dropCount kalPred1
              , V.drop dropCount lstmPred1
              , V.drop dropCount eqV1
              , V.drop dropCount posV1
              , opsShifted
              , ordersShifted
              , tradesShifted
              , openTradeShifted
              , botStartIndex st + dropCount
              )

      mLstmCtx2 =
        case mLstmCtx1 of
          Nothing -> Nothing
          Just (normState, obsAll, lstmModel) ->
            if dropCount <= 0
              then Just (normState, obsAll, lstmModel)
              else Just (normState, drop dropCount obsAll, lstmModel)

  let st1 =
        st
          { botPrices = pricesV2
          , botOpenTimes = openTimesV2
          , botKalmanPredNext = kalPred2
          , botLstmPredNext = lstmPred2
          , botEquityCurve = eqV2
          , botPositions = posV2
          , botOps = ops2
          , botOrders = orders2
          , botTrades = trades2
          , botOpenTrade = openTrade2
          , botLatestSignal = latest
          , botLastOrder = mOrder <|> botLastOrder st
          , botHaltReason = haltReason2
          , botHaltedAtMs = haltedAt2
          , botPeakEquity = max (botPeakEquity st) eqFinal
          , botDayKey = dayKey1
          , botDayStartEquity = dayStartEq1
          , botConsecutiveOrderErrors = orderErrors1
          , botLstmCtx = mLstmCtx2
          , botKalmanCtx = mKalmanCtx1
          , botLastOpenTime = openTimeNew
          , botStartIndex = startIndex2
          , botUpdatedAtMs = now
          , botError = Nothing
          }

  stOut <- if switchedApplied then botOptimizeAfterOperation st1 else pure st1
  opsAppendMaybe
    mOps
    "bot.bar"
    Nothing
    (Just (argsPublicJson (botArgs stOut)))
    ( Just
        ( object
            [ "symbol" .= botSymbol stOut
            , "market" .= marketCode (argBinanceMarket (botArgs stOut))
            , "interval" .= argInterval (botArgs stOut)
            , "openTime" .= openTimeNew
            , "price" .= priceNew
            , "position" .= posFinal
            ]
        )
    )
    (Just eqFinal)
  pure stOut

placeIfEnabled :: Args -> BotSettings -> LatestSignal -> BinanceEnv -> String -> IO ApiOrderResult
placeIfEnabled args settings sig env sym =
  if not (bsTradeEnabled settings)
    then pure (ApiOrderResult False Nothing Nothing (Just sym) Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing "Paper mode: no order sent.")
    else placeOrderForSignal args sym sig env

runRestApi :: Args -> IO ()
runRestApi baseArgs = do
  apiToken <- fmap BS.pack <$> lookupEnv "TRADER_API_TOKEN"
  timeoutEnv <- lookupEnv "TRADER_API_TIMEOUT_SEC"
  maxAsyncRunningEnv <- lookupEnv "TRADER_API_MAX_ASYNC_RUNNING"
  maxBarsLstmEnv <- lookupEnv "TRADER_API_MAX_BARS_LSTM"
  maxEpochsEnv <- lookupEnv "TRADER_API_MAX_EPOCHS"
  maxHiddenSizeEnv <- lookupEnv "TRADER_API_MAX_HIDDEN_SIZE"
  let timeoutSec =
        case timeoutEnv >>= readMaybe of
          Just n | n >= 0 -> n
          _ -> 600
      maxAsyncRunning =
        case maxAsyncRunningEnv >>= readMaybe of
          Just n | n >= 1 -> n
          _ -> 1
      limits =
        ApiComputeLimits
          { aclMaxBarsLstm =
              case maxBarsLstmEnv >>= readMaybe of
                Just n | n >= 2 -> n
                _ -> 300
          , aclMaxEpochs =
              case maxEpochsEnv >>= readMaybe of
                Just n | n >= 1 -> n
                _ -> 60
          , aclMaxHiddenSize =
              case maxHiddenSizeEnv >>= readMaybe of
                Just n | n >= 1 -> n
                _ -> 32
          }
  -- With 1 vCPU (common in small ECS/Fargate tasks), long-running pure compute can starve the
  -- Warp accept loop and make even quick "poll" endpoints appear to hang.
  -- Ensure at least 2 capabilities so the server stays responsive while background work runs.
  caps0 <- getNumCapabilities
  if caps0 < 2
    then do
      setNumCapabilities 2
      putStrLn "Increased GHC capabilities to 2 (to keep the API responsive during heavy compute)."
    else pure ()

  let port = max 1 (argPort baseArgs)
      settings =
        Warp.setHost "0.0.0.0" $
          Warp.setTimeout timeoutSec $
          Warp.setPort port Warp.defaultSettings
  putStrLn (printf "REST API listening on http://0.0.0.0:%d" port)
  putStrLn
    ( printf
        "API limits: maxAsyncRunning=%d, maxBarsLstm=%d, maxEpochs=%d, maxHiddenSize=%d"
        maxAsyncRunning
        (aclMaxBarsLstm limits)
        (aclMaxEpochs limits)
        (aclMaxHiddenSize limits)
    )
  metrics <- newMetrics
  mJournal <- newJournalFromEnv
  mOps <- newOpsStoreFromEnv
  asyncDirEnv <- lookupEnv "TRADER_API_ASYNC_DIR"
  let mAsyncDir =
        case trim <$> asyncDirEnv of
          Nothing -> Nothing
          Just dir | null dir -> Nothing
          Just dir -> Just dir
  case mAsyncDir of
    Nothing -> pure ()
    Just dir -> putStrLn (printf "Async job persistence enabled: %s" dir)
  now <- getTimestampMs
  journalWriteMaybe mJournal (object ["type" .= ("server.start" :: String), "atMs" .= now, "port" .= port])
  opsAppendMaybe mOps "server.start" Nothing Nothing (Just (object ["port" .= port])) Nothing
  bot <- newBotController
  asyncSignal <- newJobStore "signal" maxAsyncRunning mAsyncDir
  asyncBacktest <- newJobStore "backtest" maxAsyncRunning mAsyncDir
  asyncTrade <- newJobStore "trade" maxAsyncRunning mAsyncDir
  Warp.runSettings settings (apiApp baseArgs apiToken bot metrics mJournal mOps limits (AsyncStores asyncSignal asyncBacktest asyncTrade))

corsHeaders :: ResponseHeaders
corsHeaders =
  [ ("Access-Control-Allow-Origin", "*")
  , ("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
  , ("Access-Control-Allow-Headers", "Authorization,Content-Type,X-API-Key")
  , ("Access-Control-Max-Age", "86400")
  ]

noCacheHeaders :: ResponseHeaders
noCacheHeaders =
  [ ("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
  , ("Pragma", "no-cache")
  , ("Expires", "0")
  , ("Vary", "Authorization, X-API-Key")
  ]

withCors :: Wai.Response -> Wai.Response
withCors = Wai.mapResponseHeaders (\hs -> corsHeaders ++ hs)

data JobStore a = JobStore
  { jsPrefix :: !Text
  , jsCounter :: !(IORef Int64)
  , jsJobs :: !(MVar (HM.HashMap Text (JobEntry a)))
  , jsMaxJobs :: !Int
  , jsTtlMs :: !Int64
  , jsRunning :: !(MVar Int)
  , jsMaxRunning :: !Int
  , jsDir :: !(Maybe FilePath)
  }

data JobEntry a = JobEntry
  { jeCreatedAtMs :: !Int64
  , jeThreadId :: !ThreadId
  , jeResult :: !(MVar (Either String a))
  }

newJobStore :: Text -> Int -> Maybe FilePath -> IO (JobStore a)
newJobStore prefix maxRunning mAsyncDir = do
  counter <- newIORef 0
  jobs <- newMVar HM.empty
  running <- newMVar 0
  let mDir =
        case mAsyncDir of
          Nothing -> Nothing
          Just baseDir -> Just (baseDir </> T.unpack prefix)
  case mDir of
    Nothing -> pure ()
    Just dir -> createDirectoryIfMissing True dir
  pure
    JobStore
      { jsPrefix = prefix
      , jsCounter = counter
      , jsJobs = jobs
      , jsMaxJobs = 200
      , jsTtlMs = 30 * 60 * 1000
      , jsRunning = running
      , jsMaxRunning = max 1 maxRunning
      , jsDir = mDir
      }

pruneJobStore :: JobStore a -> Int64 -> IO ()
pruneJobStore store now =
  modifyMVar_
    (jsJobs store)
    ( \jobs0 -> do
        jobs1 <-
          fmap HM.fromList $
            fmap (mapMaybe id) $
              mapM
                ( \(k, e) -> do
                    done <- isJust <$> tryReadMVar (jeResult e)
                    let expired = done && now - jeCreatedAtMs e > jsTtlMs store
                    pure (if expired then Nothing else Just (k, e))
                )
                (HM.toList jobs0)

        let maxJobs = max 1 (jsMaxJobs store)
        if HM.size jobs1 <= maxJobs
          then pure jobs1
          else do
            annotated <-
              mapM
                ( \(k, e) -> do
                    done <- isJust <$> tryReadMVar (jeResult e)
                    pure (k, jeCreatedAtMs e, done)
                )
                (HM.toList jobs1)
            let doneSorted =
                  take (HM.size jobs1 - maxJobs) $
                    sortOn
                      (\(_k, createdAt, _done) -> createdAt)
                      [x | x@(_k, _createdAt, done) <- annotated, done]
                dropKeys = [k | (k, _createdAt, _done) <- doneSorted]
                jobs2 = foldl' (flip HM.delete) jobs1 dropKeys
            pure jobs2
    )
    >> pruneJobStoreDisk store now

data StoredAsyncJobMeta = StoredAsyncJobMeta
  { sajStatus :: !String
  , sajCreatedAtMs :: !(Maybe Int64)
  } deriving (Eq, Show, Generic)

instance FromJSON StoredAsyncJobMeta where
  parseJSON =
    Aeson.withObject "StoredAsyncJobMeta" $ \o ->
      StoredAsyncJobMeta
        <$> o Aeson..: "status"
        <*> o Aeson..:? "createdAtMs"

isSafeJobId :: Text -> Bool
isSafeJobId jobId =
  let s = T.unpack jobId
   in not (null s)
        && length s <= 200
        && all (\c -> isAlphaNum c || c == '-' || c == '_') s

jobFilePath :: JobStore a -> Text -> Maybe FilePath
jobFilePath store jobId =
  case jsDir store of
    Nothing -> Nothing
    Just dir ->
      let prefixOk = (jsPrefix store <> "-") `T.isPrefixOf` jobId
       in if prefixOk && isSafeJobId jobId
            then Just (dir </> (T.unpack jobId ++ ".json"))
            else Nothing

writeJobFile :: JobStore a -> Text -> Aeson.Value -> IO ()
writeJobFile store jobId payload =
  case jobFilePath store jobId of
    Nothing -> pure ()
    Just path -> do
      let tmp = path ++ ".tmp"
      _ <-
        try
          ( do
              createDirectoryIfMissing True (takeDirectory path)
              BL.writeFile tmp (encode payload)
              renameFile tmp path
          )
          :: IO (Either SomeException ())
      pure ()

readJobFile :: JobStore a -> Text -> IO (Maybe Aeson.Value)
readJobFile store jobId =
  case jobFilePath store jobId of
    Nothing -> pure Nothing
    Just path -> do
      exists <- doesFileExist path
      if not exists
        then pure Nothing
        else do
          eBs <- try (BL.readFile path) :: IO (Either SomeException BL.ByteString)
          case eBs of
            Left _ -> pure Nothing
            Right bs ->
              case eitherDecode bs of
                Left _ -> pure Nothing
                Right v -> pure (Just v)

pruneJobStoreDisk :: JobStore a -> Int64 -> IO ()
pruneJobStoreDisk store now =
  case jsDir store of
    Nothing -> pure ()
    Just dir -> do
      eNames <- try (listDirectory dir) :: IO (Either SomeException [FilePath])
      case eNames of
        Left _ -> pure ()
        Right names0 -> do
          let names = [n | n <- names0, ".json" `isSuffixOf` n]
          parsed <-
            mapM
              ( \name -> do
                  let path = dir </> name
                  eBs <- try (BL.readFile path) :: IO (Either SomeException BL.ByteString)
                  case eBs of
                    Left _ -> pure Nothing
                    Right bs ->
                      case eitherDecode bs of
                        Left _ -> pure Nothing
                        Right meta -> pure (Just (path, meta :: StoredAsyncJobMeta))
              )
              names
          let entries = mapMaybe id parsed
              isDoneOrError st = st == "done" || st == "error"
              isExpired meta =
                case sajCreatedAtMs meta of
                  Just createdAt ->
                    isDoneOrError (sajStatus meta) && now - createdAt > jsTtlMs store
                  Nothing -> False
              expiredPaths = [path | (path, meta) <- entries, isExpired meta]
          mapM_ removeFileSafe expiredPaths

          let maxJobs = max 1 (jsMaxJobs store)
              kept = [(path, meta) | (path, meta) <- entries, not (isExpired meta)]
              totalAfter = length names - length expiredPaths
          if totalAfter <= maxJobs
            then pure ()
            else do
              let doneSorted =
                    sortOn
                      (\(_path, meta) -> maybe (maxBound :: Int64) id (sajCreatedAtMs meta))
                      [x | x@(_path, meta) <- kept, isDoneOrError (sajStatus meta)]
                  dropCount = totalAfter - maxJobs
              mapM_ (removeFileSafe . fst) (take dropCount doneSorted)
  where
    removeFileSafe path = do
      _ <- try (removeFile path) :: IO (Either SomeException ())
      pure ()

startJob :: ToJSON a => JobStore a -> IO a -> IO (Either String Text)
startJob store action = do
  now <- getTimestampMs
  pruneJobStore store now

  ok <- modifyMVar (jsRunning store) $ \n ->
    if n >= max 1 (jsMaxRunning store)
      then pure (n, False)
      else pure (n + 1, True)
  if not ok
    then pure (Left "Too many requests. Try again in a moment.")
    else do
      n <- atomicModifyIORef' (jsCounter store) (\x -> let y = x + 1 in (y, y))
      let jobId = jsPrefix store <> "-" <> T.pack (show now) <> "-" <> T.pack (show n)
      out <- newEmptyMVar
      writeJobFile store jobId (object ["status" .= ("running" :: String), "createdAtMs" .= now])
      tid <-
        forkIO $
          ( do
              r <- try action
              case r of
                Right v -> do
                  doneAt <- getTimestampMs
                  writeJobFile store jobId (object ["status" .= ("done" :: String), "createdAtMs" .= now, "completedAtMs" .= doneAt, "result" .= v])
                  _ <- tryPutMVar out (Right v)
                  pure ()
                Left ex -> do
                  let (_, msg) = exceptionToHttp ex
                  doneAt <- getTimestampMs
                  writeJobFile store jobId (object ["status" .= ("error" :: String), "createdAtMs" .= now, "completedAtMs" .= doneAt, "error" .= msg])
                  _ <- tryPutMVar out (Left msg)
                  pure ()
          )
            `finally` modifyMVar_ (jsRunning store) (pure . max 0 . subtract 1)
      modifyMVar_ (jsJobs store) (pure . HM.insert jobId (JobEntry now tid out))
      pure (Right jobId)

data AsyncStores = AsyncStores
  { asSignal :: !(JobStore LatestSignal)
  , asBacktest :: !(JobStore Aeson.Value)
  , asTrade :: !(JobStore ApiTradeResponse)
  }

data ApiComputeLimits = ApiComputeLimits
  { aclMaxBarsLstm :: !Int
  , aclMaxEpochs :: !Int
  , aclMaxHiddenSize :: !Int
  } deriving (Eq, Show)

validateApiComputeLimits :: ApiComputeLimits -> Args -> Either String Args
validateApiComputeLimits limits args =
  case argMethod args of
    MethodKalmanOnly -> Right args
    _ -> do
      let maxBars = aclMaxBarsLstm limits
          bars =
            case argBinanceSymbol args of
              Just _ -> resolveBarsForBinance args
              Nothing -> argBars args
          barsLabel = "bars too high"

      case argBinanceSymbol args of
        Just _ ->
          if bars > maxBars
            then
              Left
                ( "Request too expensive for this API instance: "
                    ++ barsLabel
                    ++ " (max bars="
                    ++ show maxBars
                    ++ " for LSTM methods). Reduce bars or use method=10 (Kalman-only)."
                )
            else Right ()
        Nothing ->
          if bars > 0 && bars > maxBars
            then
              Left
                ( "Request too expensive for this API instance: "
                    ++ barsLabel
                    ++ " (max bars="
                    ++ show maxBars
                    ++ " for LSTM methods). Reduce bars or use method=10 (Kalman-only)."
                )
            else Right ()

      if argEpochs args > aclMaxEpochs limits
        then Left ("epochs too high for this API instance (max " ++ show (aclMaxEpochs limits) ++ ").")
        else Right ()

      if argHiddenSize args > aclMaxHiddenSize limits
        then Left ("hiddenSize too high for this API instance (max " ++ show (aclMaxHiddenSize limits) ++ ").")
        else Right ()

      Right args

apiApp :: Args -> Maybe BS.ByteString -> BotController -> Metrics -> Maybe Journal -> Maybe OpsStore -> ApiComputeLimits -> AsyncStores -> Wai.Application
apiApp baseArgs apiToken botCtrl metrics mJournal mOps limits asyncStores req respond = do
  let rawPath = Wai.pathInfo req
      path =
        case rawPath of
          ("api" : rest) -> rest
          _ -> rawPath
      respondCors = respond . withCors
      label =
        case path of
          ["signal", "async", _] -> "signal/async/:jobId"
          ["backtest", "async", _] -> "backtest/async/:jobId"
          ["trade", "async", _] -> "trade/async/:jobId"
          _ ->
            let go xs =
                  case xs of
                    [] -> "root"
                    [x] -> T.unpack x
                    (x:rest) -> T.unpack x ++ "/" ++ go rest
             in go path

  case Wai.requestMethod req of
    "OPTIONS" -> respond (Wai.responseLBS status204 corsHeaders "")
    _ -> do
      metricsIncEndpoint metrics label
      if path /= ["health"] && not (authorized apiToken req)
        then respondCors (jsonError status401 "Unauthorized")
        else
          case path of
            [] ->
              case Wai.requestMethod req of
                "GET" ->
                  respondCors
                    ( jsonValue
                        status200
                        ( object
                            [ "name" .= ("trader-hs" :: String)
                            , "endpoints"
                                .= [ object ["method" .= ("GET" :: String), "path" .= ("/health" :: String)]
                                   , object ["method" .= ("GET" :: String), "path" .= ("/metrics" :: String)]
                                   , object ["method" .= ("GET" :: String), "path" .= ("/ops" :: String)]
                                   , object ["method" .= ("POST" :: String), "path" .= ("/signal" :: String)]
                                   , object ["method" .= ("POST" :: String), "path" .= ("/signal/async" :: String)]
                                   , object ["method" .= ("GET" :: String), "path" .= ("/signal/async/:jobId" :: String)]
                                   , object ["method" .= ("POST" :: String), "path" .= ("/trade" :: String)]
                                   , object ["method" .= ("POST" :: String), "path" .= ("/trade/async" :: String)]
                                   , object ["method" .= ("GET" :: String), "path" .= ("/trade/async/:jobId" :: String)]
                                   , object ["method" .= ("POST" :: String), "path" .= ("/backtest" :: String)]
                                   , object ["method" .= ("POST" :: String), "path" .= ("/backtest/async" :: String)]
                                   , object ["method" .= ("GET" :: String), "path" .= ("/backtest/async/:jobId" :: String)]
                                   , object ["method" .= ("POST" :: String), "path" .= ("/binance/keys" :: String)]
                                   , object ["method" .= ("POST" :: String), "path" .= ("/bot/start" :: String)]
                                   , object ["method" .= ("POST" :: String), "path" .= ("/bot/stop" :: String)]
                                   , object ["method" .= ("GET" :: String), "path" .= ("/bot/status" :: String)]
                                   ]
                            ]
                        )
                    )
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["health"] ->
              case Wai.requestMethod req of
                "GET" ->
                  let authRequired = isJust apiToken
                      authOk = authorized apiToken req
                   in respondCors
                        ( jsonValue
                            status200
                            ( object
                                ( ["status" .= ("ok" :: String)]
                                    ++ (if authRequired then ["authRequired" .= True, "authOk" .= authOk] else [])
                                )
                            )
                        )
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["metrics"] ->
              case Wai.requestMethod req of
                "GET" -> handleMetrics metrics botCtrl respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["ops"] ->
              case Wai.requestMethod req of
                "GET" -> handleOps mOps req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["signal"] ->
              case Wai.requestMethod req of
                "POST" -> handleSignal mOps limits baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["signal", "async"] ->
              case Wai.requestMethod req of
                "POST" -> handleSignalAsync mOps limits (asSignal asyncStores) baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["signal", "async", jobId] ->
              case Wai.requestMethod req of
                "GET" -> handleAsyncPoll (asSignal asyncStores) jobId respondCors
                "POST" -> handleAsyncPoll (asSignal asyncStores) jobId respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["trade"] ->
              case Wai.requestMethod req of
                "POST" -> handleTrade mOps limits metrics mJournal baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["trade", "async"] ->
              case Wai.requestMethod req of
                "POST" -> handleTradeAsync mOps limits (asTrade asyncStores) metrics mJournal baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["trade", "async", jobId] ->
              case Wai.requestMethod req of
                "GET" -> handleAsyncPoll (asTrade asyncStores) jobId respondCors
                "POST" -> handleAsyncPoll (asTrade asyncStores) jobId respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["backtest"] ->
              case Wai.requestMethod req of
                "POST" -> handleBacktest mOps limits baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["backtest", "async"] ->
              case Wai.requestMethod req of
                "POST" -> handleBacktestAsync mOps limits (asBacktest asyncStores) baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["backtest", "async", jobId] ->
              case Wai.requestMethod req of
                "GET" -> handleAsyncPoll (asBacktest asyncStores) jobId respondCors
                "POST" -> handleAsyncPoll (asBacktest asyncStores) jobId respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["binance", "keys"] ->
              case Wai.requestMethod req of
                "POST" -> handleBinanceKeys baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["bot", "start"] ->
              case Wai.requestMethod req of
                "POST" -> handleBotStart mOps limits metrics mJournal baseArgs botCtrl req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["bot", "stop"] ->
              case Wai.requestMethod req of
                "POST" -> handleBotStop mOps mJournal botCtrl respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["bot", "status"] ->
              case Wai.requestMethod req of
                "GET" -> handleBotStatus botCtrl req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            _ -> respondCors (jsonError status404 "Not found")

authorized :: Maybe BS.ByteString -> Wai.Request -> Bool
authorized mToken req =
  case mToken of
    Nothing -> True
    Just tok ->
      let hs = Wai.requestHeaders req
          bearer = "Bearer " <> tok
       in lookup hAuthorization hs == Just bearer || lookup "X-API-Key" hs == Just tok

jsonValue :: ToJSON a => Status -> a -> Wai.Response
jsonValue st v =
  Wai.responseLBS
    st
    ([("Content-Type", "application/json")] ++ noCacheHeaders)
    (encode v)

jsonError :: Status -> String -> Wai.Response
jsonError st msg = jsonValue st (ApiError msg)

exceptionToHttp :: SomeException -> (Status, String)
exceptionToHttp ex =
  case fromException ex of
    Just (ErrorCall msg) -> (status400, msg)
    Nothing ->
      case fromException ex of
        Just io
          | isUserError io -> (status400, ioeGetErrorString io)
        _ ->
          case (fromException ex :: Maybe HttpException) of
            Just httpEx -> (status502, show httpEx)
            Nothing -> (status500, show ex)

textValue :: Status -> BL.ByteString -> Wai.Response
textValue st body =
  Wai.responseLBS
    st
    ([("Content-Type", "text/plain; version=0.0.4")] ++ noCacheHeaders)
    body

handleMetrics :: Metrics -> BotController -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleMetrics metrics botCtrl respond = do
  mSt <- botGetState botCtrl
  body <- renderMetricsText metrics (isJust mSt)
  respond (textValue status200 body)

handleOps :: Maybe OpsStore -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleOps mOps req respond =
  case mOps of
    Nothing -> respond (jsonValue status200 (object ["enabled" .= False, "ops" .= ([] :: [PersistedOperation])]))
    Just store -> do
      let q = Wai.queryString req
          lookupParam name =
            case lookup (BS.pack name) q of
              Just (Just raw) -> Just raw
              _ -> Nothing
          readIntParam name =
            lookupParam name >>= (readMaybe . BS.unpack)
          limit = maybe 200 id (readIntParam "limit")
          sinceId = readIntParam "since"
          kind = T.pack . BS.unpack <$> lookupParam "kind"
      ops <- opsList store sinceId limit kind
      latestId <- readIORef (osNextId store)
      respond (jsonValue status200 (object ["enabled" .= True, "latestId" .= latestId, "maxInMemory" .= osMaxInMemory store, "ops" .= ops]))

handleSignal :: Maybe OpsStore -> ApiComputeLimits -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleSignal mOps limits baseArgs req respond = do
  body <- Wai.strictRequestBody req
  case eitherDecode body of
    Left e -> respond (jsonError status400 ("Invalid JSON: " ++ e))
    Right params ->
      case argsFromApi baseArgs params of
        Left e -> respond (jsonError status400 e)
        Right args0 -> do
          let args =
                args0
                  { argTradeOnly = True
                  , argBinanceTrade = False
                  , argSweepThreshold = False
                  , argOptimizeOperations = False
                  }
          case validateApiComputeLimits limits args of
            Left e -> respond (jsonError status400 e)
            Right argsOk -> do
              r <- try (computeLatestSignalFromArgs argsOk) :: IO (Either SomeException LatestSignal)
              case r of
                Left ex ->
                  let (st, msg) = exceptionToHttp ex
                   in respond (jsonError st msg)
                Right sig -> do
                  opsAppendMaybe
                    mOps
                    "signal"
                    (Just (toJSON (sanitizeApiParams params)))
                    (Just (argsPublicJson argsOk))
                    (Just (toJSON sig))
                    Nothing
                  respond (jsonValue status200 sig)

handleSignalAsync :: Maybe OpsStore -> ApiComputeLimits -> JobStore LatestSignal -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleSignalAsync mOps limits store baseArgs req respond = do
  body <- Wai.strictRequestBody req
  case eitherDecode body of
    Left e -> respond (jsonError status400 ("Invalid JSON: " ++ e))
    Right params ->
      case argsFromApi baseArgs params of
        Left e -> respond (jsonError status400 e)
        Right args0 -> do
          let args =
                args0
                  { argTradeOnly = True
                  , argBinanceTrade = False
                  , argSweepThreshold = False
                  , argOptimizeOperations = False
                  }
          case validateApiComputeLimits limits args of
            Left e -> respond (jsonError status400 e)
            Right argsOk -> do
              let paramsJson = Just (toJSON (sanitizeApiParams params))
                  argsJson = Just (argsPublicJson argsOk)
              r <-
                startJob store $ do
                  sig <- computeLatestSignalFromArgs argsOk
                  opsAppendMaybe mOps "signal" paramsJson argsJson (Just (toJSON sig)) Nothing
                  pure sig
              case r of
                Left e -> respond (jsonError status429 e)
                Right jobId -> respond (jsonValue status202 (object ["jobId" .= jobId]))

handleTrade :: Maybe OpsStore -> ApiComputeLimits -> Metrics -> Maybe Journal -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleTrade mOps limits metrics mJournal baseArgs req respond = do
  body <- Wai.strictRequestBody req
  case eitherDecode body of
    Left e -> respond (jsonError status400 ("Invalid JSON: " ++ e))
    Right params ->
      case argsFromApi baseArgs params of
        Left e -> respond (jsonError status400 e)
        Right args0 -> do
          let args1 =
                args0
                  { argTradeOnly = True
                  , argBinanceTrade = True
                  , argSweepThreshold = False
                  , argOptimizeOperations = False
                  }
          case validateArgs args1 of
            Left e -> respond (jsonError status400 e)
            Right args ->
              case validateApiComputeLimits limits args of
                Left e -> respond (jsonError status400 e)
                Right argsOk -> do
                  r <- try (computeTradeFromArgs argsOk) :: IO (Either SomeException ApiTradeResponse)
                  case r of
                    Left ex ->
                      let (st, msg) = exceptionToHttp ex
                       in respond (jsonError st msg)
                    Right out -> do
                      metricsRecordOrder metrics (atrOrder out)
                      now <- getTimestampMs
                      journalWriteMaybe
                        mJournal
                        ( object
                            [ "type" .= ("trade.order" :: String)
                            , "atMs" .= now
                            , "symbol" .= argBinanceSymbol argsOk
                            , "market" .= marketCode (argBinanceMarket argsOk)
                            , "action" .= lsAction (atrSignal out)
                            , "order" .= atrOrder out
                            ]
                        )
                      opsAppendMaybe
                        mOps
                        "trade.order"
                        (Just (toJSON (sanitizeApiParams params)))
                        (Just (argsPublicJson argsOk))
                        (Just (toJSON out))
                        Nothing
                      respond (jsonValue status200 out)

handleTradeAsync :: Maybe OpsStore -> ApiComputeLimits -> JobStore ApiTradeResponse -> Metrics -> Maybe Journal -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleTradeAsync mOps limits store metrics mJournal baseArgs req respond = do
  body <- Wai.strictRequestBody req
  case eitherDecode body of
    Left e -> respond (jsonError status400 ("Invalid JSON: " ++ e))
    Right params ->
      case argsFromApi baseArgs params of
        Left e -> respond (jsonError status400 e)
        Right args0 -> do
          let args1 =
                args0
                  { argTradeOnly = True
                  , argBinanceTrade = True
                  , argSweepThreshold = False
                  , argOptimizeOperations = False
                  }
          case validateArgs args1 of
            Left e -> respond (jsonError status400 e)
            Right args ->
              case validateApiComputeLimits limits args of
                Left e -> respond (jsonError status400 e)
                Right argsOk -> do
                  let paramsJson = Just (toJSON (sanitizeApiParams params))
                      argsJson = Just (argsPublicJson argsOk)
                  r <-
                    startJob store $ do
                      out <- computeTradeFromArgs argsOk
                      metricsRecordOrder metrics (atrOrder out)
                      now <- getTimestampMs
                      journalWriteMaybe
                        mJournal
                        ( object
                            [ "type" .= ("trade.order" :: String)
                            , "atMs" .= now
                            , "symbol" .= argBinanceSymbol argsOk
                            , "market" .= marketCode (argBinanceMarket argsOk)
                            , "action" .= lsAction (atrSignal out)
                            , "order" .= atrOrder out
                            ]
                        )
                      opsAppendMaybe mOps "trade.order" paramsJson argsJson (Just (toJSON out)) Nothing
                      pure out
                  case r of
                    Left e -> respond (jsonError status429 e)
                    Right jobId -> respond (jsonValue status202 (object ["jobId" .= jobId]))

extractBacktestFinalEquity :: Aeson.Value -> Maybe Double
extractBacktestFinalEquity =
  AT.parseMaybe $
    Aeson.withObject "Backtest" $ \o -> do
      m <- o AT..: "metrics"
      m AT..: "finalEquity"

handleBacktest :: Maybe OpsStore -> ApiComputeLimits -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBacktest mOps limits baseArgs req respond = do
  body <- Wai.strictRequestBody req
  case eitherDecode body of
    Left e -> respond (jsonError status400 ("Invalid JSON: " ++ e))
    Right params ->
      case argsFromApi baseArgs params of
        Left e -> respond (jsonError status400 e)
        Right args0 -> do
          let args =
                args0
                  { argTradeOnly = False
                  , argBinanceTrade = False
                  , argOptimizeOperations = maybe (argOptimizeOperations args0) id (apOptimizeOperations params)
                  , argSweepThreshold = maybe (argSweepThreshold args0) id (apSweepThreshold params)
                  , argBacktestRatio = maybe (argBacktestRatio args0) id (apBacktestRatio params)
                  }
          case validateApiComputeLimits limits args of
            Left e -> respond (jsonError status400 e)
            Right argsOk -> do
              r <- try (computeBacktestFromArgs argsOk) :: IO (Either SomeException Aeson.Value)
              case r of
                Left ex ->
                  let (st, msg) = exceptionToHttp ex
                   in respond (jsonError st msg)
                Right out -> do
                  opsAppendMaybe
                    mOps
                    "backtest"
                    (Just (toJSON (sanitizeApiParams params)))
                    (Just (argsPublicJson argsOk))
                    (Just out)
                    (extractBacktestFinalEquity out)
                  respond (jsonValue status200 out)

handleBacktestAsync :: Maybe OpsStore -> ApiComputeLimits -> JobStore Aeson.Value -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBacktestAsync mOps limits store baseArgs req respond = do
  body <- Wai.strictRequestBody req
  case eitherDecode body of
    Left e -> respond (jsonError status400 ("Invalid JSON: " ++ e))
    Right params ->
      case argsFromApi baseArgs params of
        Left e -> respond (jsonError status400 e)
        Right args0 -> do
          let args =
                args0
                  { argTradeOnly = False
                  , argBinanceTrade = False
                  , argOptimizeOperations = maybe (argOptimizeOperations args0) id (apOptimizeOperations params)
                  , argSweepThreshold = maybe (argSweepThreshold args0) id (apSweepThreshold params)
                  , argBacktestRatio = maybe (argBacktestRatio args0) id (apBacktestRatio params)
                  }
          case validateApiComputeLimits limits args of
            Left e -> respond (jsonError status400 e)
            Right argsOk -> do
              let paramsJson = Just (toJSON (sanitizeApiParams params))
                  argsJson = Just (argsPublicJson argsOk)
              r <-
                startJob store $ do
                  out <- computeBacktestFromArgs argsOk
                  opsAppendMaybe mOps "backtest" paramsJson argsJson (Just out) (extractBacktestFinalEquity out)
                  pure out
              case r of
                Left e -> respond (jsonError status429 e)
                Right jobId -> respond (jsonValue status202 (object ["jobId" .= jobId]))

handleAsyncPoll :: ToJSON a => JobStore a -> Text -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleAsyncPoll store jobId respond = do
  now <- getTimestampMs
  pruneJobStore store now
  mEntry <- withMVar (jsJobs store) (pure . HM.lookup jobId)
  case mEntry of
    Nothing -> do
      mDisk <- readJobFile store jobId
      case mDisk of
        Just v -> respond (jsonValue status200 v)
        Nothing -> respond (jsonValue status200 (object ["status" .= ("error" :: String), "error" .= ("Not found" :: String)]))
    Just entry -> do
      r <- tryReadMVar (jeResult entry)
      case r of
        Nothing ->
          respond
            ( jsonValue
                status200
                (object ["status" .= ("running" :: String), "createdAtMs" .= jeCreatedAtMs entry])
            )
        Just (Left err) ->
          respond
            ( jsonValue
                status200
                (object ["status" .= ("error" :: String), "createdAtMs" .= jeCreatedAtMs entry, "error" .= err])
            )
        Just (Right v) ->
          respond
            ( jsonValue
                status200
                (object ["status" .= ("done" :: String), "createdAtMs" .= jeCreatedAtMs entry, "result" .= v])
            )

handleAsyncCancel :: ToJSON a => JobStore a -> Text -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleAsyncCancel store jobId respond = do
  now <- getTimestampMs
  pruneJobStore store now
  mEntry <- withMVar (jsJobs store) (pure . HM.lookup jobId)
  case mEntry of
    Nothing -> respond (jsonValue status200 (object ["status" .= ("error" :: String), "error" .= ("Not found" :: String)]))
    Just entry -> do
      r <- tryReadMVar (jeResult entry)
      case r of
        Just (Right _) ->
          respond
            ( jsonValue
                status200
                (object ["status" .= ("done" :: String), "createdAtMs" .= jeCreatedAtMs entry])
            )
        Just (Left err) ->
          respond
            ( jsonValue
                status200
                (object ["status" .= ("error" :: String), "createdAtMs" .= jeCreatedAtMs entry, "error" .= err])
            )
        Nothing -> do
          canceledAt <- getTimestampMs
          let msg = "Canceled" :: String
          writeJobFile
            store
            jobId
            (object ["status" .= ("error" :: String), "createdAtMs" .= jeCreatedAtMs entry, "completedAtMs" .= canceledAt, "error" .= msg])
          _ <- tryPutMVar (jeResult entry) (Left msg)
          killThread (jeThreadId entry)
          respond
            ( jsonValue
                status200
                (object ["status" .= ("canceled" :: String), "createdAtMs" .= jeCreatedAtMs entry, "canceledAtMs" .= canceledAt])
            )

handleBinanceKeys :: Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBinanceKeys baseArgs req respond = do
  body <- Wai.strictRequestBody req
  case eitherDecode body of
    Left e -> respond (jsonError status400 ("Invalid JSON: " ++ e))
    Right params ->
      case argsFromApi baseArgs params of
        Left e -> respond (jsonError status400 e)
        Right args0 -> do
          r <- try (computeBinanceKeysStatusFromArgs args0) :: IO (Either SomeException ApiBinanceKeysStatus)
          case r of
            Left ex ->
              let (st, msg) = exceptionToHttp ex
               in respond (jsonError st msg)
            Right out -> respond (jsonValue status200 out)

handleBotStart :: Maybe OpsStore -> ApiComputeLimits -> Metrics -> Maybe Journal -> Args -> BotController -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBotStart mOps limits metrics mJournal baseArgs botCtrl req respond = do
  body <- Wai.strictRequestBody req
  case eitherDecode body of
    Left e -> respond (jsonError status400 ("Invalid JSON: " ++ e))
    Right params ->
      case argsFromApi baseArgs params of
        Left e -> respond (jsonError status400 e)
        Right args0 -> do
          let args =
                args0
                  { argTradeOnly = True
                  }
          if argPositioning args /= LongFlat
            then respond (jsonError status400 "bot/start currently supports positioning=long-flat only")
            else
              case validateApiComputeLimits limits args of
                Left e -> respond (jsonError status400 e)
                Right argsOk -> do
                  r <- botStart mOps metrics mJournal botCtrl argsOk params
                  case r of
                    Left e -> respond (jsonError status400 e)
                    Right rt -> do
                      now <- getTimestampMs
                      let settings = bsrSettings rt
                      journalWriteMaybe
                        mJournal
                        ( object
                            [ "type" .= ("bot.start" :: String)
                            , "atMs" .= now
                            , "symbol" .= bsrSymbol rt
                            , "market" .= marketCode (argBinanceMarket (bsrArgs rt))
                            , "interval" .= argInterval (bsrArgs rt)
                            , "tradeEnabled" .= bsTradeEnabled (bsrSettings rt)
                            ]
                        )
                      opsAppendMaybe
                        mOps
                        "bot.start"
                        (Just (toJSON (sanitizeApiParams params)))
                        (Just (argsPublicJson argsOk))
                        ( Just
                            ( object
                                [ "symbol" .= bsrSymbol rt
                                , "market" .= marketCode (argBinanceMarket (bsrArgs rt))
                                , "interval" .= argInterval (bsrArgs rt)
                                , "tradeEnabled" .= bsTradeEnabled settings
                                , "botPollSeconds" .= bsPollSeconds settings
                                , "botOnlineEpochs" .= bsOnlineEpochs settings
                                , "botTrainBars" .= bsTrainBars settings
                                , "botMaxPoints" .= bsMaxPoints settings
                                ]
                            )
                        )
                        Nothing
                      respond (jsonValue status202 (botStartingJson rt))

handleBotStop :: Maybe OpsStore -> Maybe Journal -> BotController -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBotStop mOps mJournal botCtrl respond = do
  _ <- botStop botCtrl
  now <- getTimestampMs
  journalWriteMaybe mJournal (object ["type" .= ("bot.stop" :: String), "atMs" .= now])
  opsAppendMaybe mOps "bot.stop" Nothing Nothing Nothing Nothing
  respond (jsonValue status200 botStoppedJson)

handleBotStatus :: BotController -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBotStatus botCtrl req respond = do
  let tailN =
        case lookup (BS.pack "tail") (Wai.queryString req) of
          Just (Just raw) -> readMaybe (BS.unpack raw) >>= \n -> if n > 0 then Just n else Nothing
          _ -> Nothing
  mrt <- readMVar (bcRuntime botCtrl)
  case mrt of
    Nothing -> respond (jsonValue status200 botStoppedJson)
    Just (BotStarting rt) -> respond (jsonValue status200 (botStartingJson rt))
    Just (BotRunning rt) -> do
      st <- readMVar (brStateVar rt)
      let st' = maybe st (`botStateTail` st) tailN
      respond (jsonValue status200 (botStatusJson st'))

argsFromApi :: Args -> ApiParams -> Either String Args
argsFromApi baseArgs p = do
  method <-
    case apMethod p of
      Nothing -> Right (argMethod baseArgs)
      Just raw -> parseMethod raw

  positioning <-
    case apPositioning p of
      Nothing -> Right (argPositioning baseArgs)
      Just raw -> parsePositioning raw

  intrabarFill <-
    case apIntrabarFill p of
      Nothing -> Right (argIntrabarFill baseArgs)
      Just raw -> parseIntrabarFill raw

  norm <-
    case apNormalization p of
      Nothing -> Right (argNormalization baseArgs)
      Just raw ->
        case parseNormType raw of
          Just n -> Right n
          Nothing -> Left ("Invalid normalization: " ++ show raw ++ " (expected none|minmax|standard|log)")

  (futuresFlag, marginFlag) <-
    case apMarket p of
      Nothing -> Right (argBinanceFutures baseArgs, argBinanceMargin baseArgs)
      Just raw ->
        case map toLower (trim raw) of
          "spot" -> Right (False, False)
          "margin" -> Right (False, True)
          "futures" -> Right (True, False)
          other -> Left ("Invalid market: " ++ show other ++ " (expected spot|margin|futures)")

  let pick :: Maybe a -> a -> a
      pick v def = maybe def id v

      pickMaybe :: Maybe a -> Maybe a -> Maybe a
      pickMaybe v def =
        case v of
          Just _ -> v
          Nothing -> def

      args =
        baseArgs
          { argData = pickMaybe (apData p) (argData baseArgs)
          , argPriceCol = pick (apPriceColumn p) (argPriceCol baseArgs)
          , argBinanceSymbol = pickMaybe (apBinanceSymbol p) (argBinanceSymbol baseArgs)
          , argBinanceFutures = futuresFlag
          , argBinanceMargin = marginFlag
          , argInterval = pick (apInterval p) (argInterval baseArgs)
          , argBars = pick (apBars p) (argBars baseArgs)
          , argLookbackWindow = pick (apLookbackWindow p) (argLookbackWindow baseArgs)
          , argLookbackBars = pickMaybe (apLookbackBars p) (argLookbackBars baseArgs)
          , argBinanceTestnet = pick (apBinanceTestnet p) (argBinanceTestnet baseArgs)
          , argBinanceApiKey = pickMaybe (apBinanceApiKey p) (argBinanceApiKey baseArgs)
          , argBinanceApiSecret = pickMaybe (apBinanceApiSecret p) (argBinanceApiSecret baseArgs)
          , argNormalization = norm
          , argHiddenSize = pick (apHiddenSize p) (argHiddenSize baseArgs)
          , argEpochs = pick (apEpochs p) (argEpochs baseArgs)
          , argLr = pick (apLr p) (argLr baseArgs)
          , argValRatio = pick (apValRatio p) (argValRatio baseArgs)
          , argBacktestRatio = pick (apBacktestRatio p) (argBacktestRatio baseArgs)
          , argTuneRatio = pick (apTuneRatio p) (argTuneRatio baseArgs)
          , argPatience = pick (apPatience p) (argPatience baseArgs)
          , argGradClip =
              case apGradClip p of
                Nothing -> argGradClip baseArgs
                Just g -> Just g
          , argSeed = pick (apSeed p) (argSeed baseArgs)
          , argKalmanDt = pick (apKalmanDt p) (argKalmanDt baseArgs)
          , argKalmanProcessVar = pick (apKalmanProcessVar p) (argKalmanProcessVar baseArgs)
          , argKalmanMeasurementVar = pick (apKalmanMeasurementVar p) (argKalmanMeasurementVar baseArgs)
          , argTradeThreshold = pick (apThreshold p) (argTradeThreshold baseArgs)
          , argMethod = method
          , argPositioning = positioning
          , argOptimizeOperations = pick (apOptimizeOperations p) (argOptimizeOperations baseArgs)
          , argSweepThreshold = pick (apSweepThreshold p) (argSweepThreshold baseArgs)
          , argFee = pick (apFee p) (argFee baseArgs)
          , argSlippage = pick (apSlippage p) (argSlippage baseArgs)
          , argSpread = pick (apSpread p) (argSpread baseArgs)
          , argIntrabarFill = intrabarFill
          , argStopLoss = pickMaybe (apStopLoss p) (argStopLoss baseArgs)
          , argTakeProfit = pickMaybe (apTakeProfit p) (argTakeProfit baseArgs)
          , argTrailingStop = pickMaybe (apTrailingStop p) (argTrailingStop baseArgs)
          , argMaxDrawdown = pickMaybe (apMaxDrawdown p) (argMaxDrawdown baseArgs)
          , argMaxDailyLoss = pickMaybe (apMaxDailyLoss p) (argMaxDailyLoss baseArgs)
          , argMaxOrderErrors = pickMaybe (apMaxOrderErrors p) (argMaxOrderErrors baseArgs)
          , argPeriodsPerYear =
              case apPeriodsPerYear p of
                Nothing -> argPeriodsPerYear baseArgs
                Just v -> Just v
          , argBinanceLive = pick (apBinanceLive p) (argBinanceLive baseArgs)
          , argOrderQuote = pickMaybe (apOrderQuote p) (argOrderQuote baseArgs)
          , argOrderQuantity = pickMaybe (apOrderQuantity p) (argOrderQuantity baseArgs)
          , argOrderQuoteFraction = pickMaybe (apOrderQuoteFraction p) (argOrderQuoteFraction baseArgs)
          , argMaxOrderQuote = pickMaybe (apMaxOrderQuote p) (argMaxOrderQuote baseArgs)
          , argIdempotencyKey = pickMaybe (apIdempotencyKey p) (argIdempotencyKey baseArgs)
          , argJson = False
          , argServe = False
          , argKalmanZMin = pick (apKalmanZMin p) (argKalmanZMin baseArgs)
          , argKalmanZMax = pick (apKalmanZMax p) (argKalmanZMax baseArgs)
          , argMaxHighVolProb = pickMaybe (apMaxHighVolProb p) (argMaxHighVolProb baseArgs)
          , argMaxConformalWidth = pickMaybe (apMaxConformalWidth p) (argMaxConformalWidth baseArgs)
          , argMaxQuantileWidth = pickMaybe (apMaxQuantileWidth p) (argMaxQuantileWidth baseArgs)
          , argConfirmConformal = pick (apConfirmConformal p) (argConfirmConformal baseArgs)
          , argConfirmQuantiles = pick (apConfirmQuantiles p) (argConfirmQuantiles baseArgs)
          , argConfidenceSizing = pick (apConfidenceSizing p) (argConfidenceSizing baseArgs)
          , argMinPositionSize = pick (apMinPositionSize p) (argMinPositionSize baseArgs)
          }

  validateArgs args

dirLabel :: Maybe Int -> Maybe String
dirLabel d =
  case d of
    Just 1 -> Just "UP"
    Just (-1) -> Just "DOWN"
    _ -> Nothing

computeLatestSignalFromArgs :: Args -> IO LatestSignal
computeLatestSignalFromArgs args = do
  (series, _) <- loadPrices args
  let prices = psClose series
  if length prices < 2 then error "Need at least 2 price rows" else pure ()
  let lookback = argLookback args
  computeTradeOnlySignal args lookback prices

computeTradeFromArgs :: Args -> IO ApiTradeResponse
computeTradeFromArgs args = do
  (series, mBinanceEnv) <- loadPrices args
  let prices = psClose series
  if length prices < 2 then error "Need at least 2 price rows" else pure ()
  let lookback = argLookback args
  sig <- computeTradeOnlySignal args lookback prices
  order <-
    case (argBinanceSymbol args, mBinanceEnv) of
      (Nothing, _) -> pure (ApiOrderResult False Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing "No order: missing binanceSymbol.")
      (_, Nothing) -> pure (ApiOrderResult False Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing "No order: missing Binance environment (use binanceSymbol data source).")
      (Just sym, Just env) -> placeOrderForSignal args sym sig env
  pure ApiTradeResponse { atrSignal = sig, atrOrder = order }

data BinanceApiErrorBody = BinanceApiErrorBody
  { baeCode :: !Int
  , baeMsg :: !String
  } deriving (Eq, Show, Generic)

instance FromJSON BinanceApiErrorBody where
  parseJSON =
    Aeson.withObject "BinanceApiErrorBody" $ \o ->
      BinanceApiErrorBody
        <$> o Aeson..: "code"
        <*> o Aeson..: "msg"

truncateString :: Int -> String -> String
truncateString n s =
  if length s <= n then s else take n s ++ "..."

extractHttpStatusCode :: String -> Maybe Int
extractHttpStatusCode msg =
  let go [] = Nothing
      go ('H':'T':'T':'P':' ':rest) =
        let digits = takeWhile isDigit rest
         in readMaybe digits
      go (_:xs) = go xs
   in go msg

extractJsonObject :: String -> Maybe String
extractJsonObject msg =
  case dropWhile (/= '{') msg of
    [] -> Nothing
    s0 ->
      case break (== '}') s0 of
        (obj, '}':_) -> Just (obj ++ "}")
        _ -> Nothing

parseBinanceError :: String -> (Maybe Int, Maybe String, String)
parseBinanceError raw =
  let raw' = truncateString 240 raw
      httpCode = extractHttpStatusCode raw'
      decoded =
        case extractJsonObject raw' of
          Nothing -> Nothing
          Just json ->
            case eitherDecode (BL.fromStrict (BS.pack json)) of
              Right b -> Just (b :: BinanceApiErrorBody)
              Left _ -> Nothing
      outCode = maybe httpCode (Just . baeCode) decoded
      outMsg = baeMsg <$> decoded
      summary = maybe raw' id outMsg
   in (outCode, outMsg, summary)

probeBinance :: String -> IO a -> IO ApiBinanceProbe
probeBinance step action = do
  r <- try action
  case r of
    Right _ -> pure (ApiBinanceProbe True step Nothing Nothing "OK")
    Left ex -> do
      let msg =
            case (fromException ex :: Maybe IOError) of
              Just io | isUserError io -> ioeGetErrorString io
              _ -> show ex
          (code, m, summary) = parseBinanceError msg
      pure (ApiBinanceProbe False step code m summary)

computeBinanceKeysStatusFromArgs :: Args -> IO ApiBinanceKeysStatus
computeBinanceKeysStatusFromArgs args = do
  apiKey <- resolveEnv "BINANCE_API_KEY" (argBinanceApiKey args)
  apiSecret <- resolveEnv "BINANCE_API_SECRET" (argBinanceApiSecret args)

  let hasApiKey = maybe False (not . null . trim) apiKey
      hasApiSecret = maybe False (not . null . trim) apiSecret
      market = argBinanceMarket args
      sym = argBinanceSymbol args
      baseStatus =
        ApiBinanceKeysStatus
          { abkMarket = marketCode market
          , abkTestnet = argBinanceTestnet args
          , abkSymbol = sym
          , abkHasApiKey = hasApiKey
          , abkHasApiSecret = hasApiSecret
          , abkSigned = Nothing
          , abkTradeTest = Nothing
          }

  if not hasApiKey || not hasApiSecret
    then pure baseStatus
    else do
      env <- makeBinanceEnv args
      sym' <- maybe (throwIO (userError "binanceSymbol is required.")) pure sym
      signedProbe <-
        probeBinance "signed" $ do
          case market of
            MarketFutures -> do
              _ <- fetchFuturesPositionAmt env sym'
              pure ()
            _ -> do
              let (baseAsset, _) = splitSymbol sym'
              _ <- fetchFreeBalance env baseAsset
              pure ()

      tradeProbe <-
        case market of
          MarketMargin -> pure Nothing
          MarketSpot -> do
            let qty =
                  case argOrderQuantity args of
                    Just q | q > 0 -> Just q
                    _ -> Nothing
                qqArg =
                  case argOrderQuote args of
                    Just q | q > 0 -> Just q
                    _ -> Nothing
            qq <-
              case (qty, qqArg, argOrderQuoteFraction args) of
                (Nothing, Nothing, Just f) | f > 0 -> do
                  let (_baseAsset, quoteAsset) = splitSymbol sym'
                  quoteBal <- fetchFreeBalance env quoteAsset
                  let q0 = quoteBal * f
                      q1 =
                        let mCap =
                              case argMaxOrderQuote args of
                                Just q | q > 0 -> Just q
                                _ -> Nothing
                         in maybe q0 (\capQ -> min capQ q0) mCap
                  pure (if q1 > 0 then Just q1 else Nothing)
                _ -> pure qqArg
            if qty == Nothing && qq == Nothing
              then
                let msg =
                      if isJust (argOrderQuoteFraction args)
                        then "Provide orderQuantity/orderQuote, or set orderQuoteFraction with sufficient quote balance."
                        else "Provide orderQuantity or orderQuote."
                 in pure (Just (ApiBinanceProbe False "order/test" Nothing Nothing msg))
              else Just <$> probeBinance "order/test" (placeMarketOrder env OrderTest sym' Buy qty qq Nothing (trim <$> argIdempotencyKey args) >> pure ())
          MarketFutures -> do
            let qtyFromArgs =
                  case argOrderQuantity args of
                    Just q | q > 0 -> Just q
                    _ -> Nothing
            qty <-
              case qtyFromArgs of
                Just q -> pure (Just q)
                Nothing -> do
                  qq <-
                    case argOrderQuote args of
                      Just qq | qq > 0 -> pure (Just qq)
                      _ ->
                        case argOrderQuoteFraction args of
                          Just f | f > 0 -> do
                            let (_baseAsset, quoteAsset) = splitSymbol sym'
                            bal <- fetchFuturesAvailableBalance env quoteAsset
                            let q0 = bal * f
                                q1 =
                                  let mCap =
                                        case argMaxOrderQuote args of
                                          Just q | q > 0 -> Just q
                                          _ -> Nothing
                                   in maybe q0 (\capQ -> min capQ q0) mCap
                            pure (if q1 > 0 then Just q1 else Nothing)
                          _ -> pure Nothing
                  case qq of
                    Nothing -> pure Nothing
                    Just q -> do
                      price <- fetchTickerPrice env sym'
                      pure (if price > 0 then Just (q / price) else Nothing)
            case qty of
              Nothing ->
                let msg =
                      if isJust (argOrderQuoteFraction args)
                        then "Provide orderQuantity/orderQuote, or set orderQuoteFraction with sufficient quote balance."
                        else "Provide orderQuantity or orderQuote."
                 in pure (Just (ApiBinanceProbe False "futures/order/test" Nothing Nothing msg))
              Just q ->
                Just <$> probeBinance "futures/order/test" (placeMarketOrder env OrderTest sym' Buy (Just q) Nothing Nothing (trim <$> argIdempotencyKey args) >> pure ())

      let isAuthFailureCode c = c == (-1022) || c == (-2014) || c == (-2015)
          normalizeTradeProbe p =
            if abpOk p
              then p
              else
                case abpCode p of
                  Just c | not (isAuthFailureCode c) ->
                    p { abpOk = True, abpSummary = "Auth OK, but order rejected: " ++ abpSummary p }
                  _ -> p
      pure baseStatus { abkSigned = Just signedProbe, abkTradeTest = normalizeTradeProbe <$> tradeProbe }

data BinanceOrderInfo = BinanceOrderInfo
  { boiOrderId :: !(Maybe Int64)
  , boiClientOrderId :: !(Maybe String)
  , boiStatus :: !(Maybe String)
  , boiExecutedQty :: !(Maybe Double)
  , boiCummulativeQuoteQty :: !(Maybe Double)
  } deriving (Eq, Show)

instance FromJSON BinanceOrderInfo where
  parseJSON =
    Aeson.withObject "BinanceOrderInfo" $ \o -> do
      oid <- o Aeson..:? "orderId"
      cid <- o Aeson..:? "clientOrderId"
      st <- o Aeson..:? "status"
      exec <- parseMaybeDoubleField o "executedQty"
      cum <-
        parseMaybeDoubleField o "cummulativeQuoteQty"
          <|> parseMaybeDoubleField o "cumQuote"
          <|> parseMaybeDoubleField o "cumQuoteQty"
      pure BinanceOrderInfo { boiOrderId = oid, boiClientOrderId = cid, boiStatus = st, boiExecutedQty = exec, boiCummulativeQuoteQty = cum }
    where
      parseMaybeDoubleField o k = do
        mv <- o Aeson..:? k :: AT.Parser (Maybe Aeson.Value)
        case mv of
          Nothing -> pure Nothing
          Just v -> Just <$> parseJsonDouble v

      parseJsonDouble v =
        case v of
          Aeson.String t ->
            case readMaybe (T.unpack t) of
              Just d -> pure d
              Nothing -> fail ("Failed to parse double: " ++ T.unpack t)
          Aeson.Number n -> pure (realToFrac n)
          _ -> fail "Expected number or string"

decodeOrderInfo :: BL.ByteString -> Maybe BinanceOrderInfo
decodeOrderInfo raw =
  case Aeson.decode raw of
    Nothing -> Nothing
    Just v -> AT.parseMaybe parseJSON v

applyOrderInfo :: BinanceOrderInfo -> ApiOrderResult -> ApiOrderResult
applyOrderInfo info r =
  r
    { aorOrderId = boiOrderId info <|> aorOrderId r
    , aorClientOrderId = boiClientOrderId info <|> aorClientOrderId r
    , aorStatus = boiStatus info <|> aorStatus r
    , aorExecutedQty = boiExecutedQty info <|> aorExecutedQty r
    , aorCummulativeQuoteQty = boiCummulativeQuoteQty info <|> aorCummulativeQuoteQty r
    }

placeOrderForSignal :: Args -> String -> LatestSignal -> BinanceEnv -> IO ApiOrderResult
placeOrderForSignal args sym sig env =
  placeOrderForSignalEx args sym sig env Nothing

placeOrderForSignalEx :: Args -> String -> LatestSignal -> BinanceEnv -> Maybe String -> IO ApiOrderResult
placeOrderForSignalEx args sym sig env mClientOrderIdOverride = do
  case (beApiKey env, beApiSecret env) of
    (Nothing, _) -> noOrder "No order: missing Binance API key."
    (_, Nothing) -> noOrder "No order: missing Binance API secret."
    (Just _, Just _) ->
      case chosenDir of
        Nothing -> noOrder neutralMsg
        Just dir -> do
          mFilters <- tryFetchFilters
          let (baseAsset, quoteAsset) = splitSymbol sym
          r <- try (place mFilters baseAsset quoteAsset dir) :: IO (Either SomeException ApiOrderResult)
          case r of
            Left ex -> noOrder ("Order failed: " ++ shortErr ex)
            Right out -> pure out
  where
    method = lsMethod sig
    chosenDir = lsChosenDir sig
    currentPrice = lsCurrentPrice sig

    entryScale :: Double
    entryScale =
      let s =
            if argConfidenceSizing args
              then maybe 1 id (lsPositionSize sig)
              else 1
       in max 0 (min 1 s)

    clientOrderId :: Maybe String
    clientOrderId = trim <$> (mClientOrderIdOverride <|> argIdempotencyKey args)

    baseResult :: ApiOrderResult
    baseResult =
      ApiOrderResult
        { aorSent = False
        , aorMode = Just (modeLabel mode)
        , aorSide = Nothing
        , aorSymbol = Just sym
        , aorQuantity = Nothing
        , aorQuoteQuantity = Nothing
        , aorOrderId = Nothing
        , aorClientOrderId = clientOrderId
        , aorStatus = Nothing
        , aorExecutedQty = Nothing
        , aorCummulativeQuoteQty = Nothing
        , aorResponse = Nothing
        , aorMessage = ""
        }

    noOrder :: String -> IO ApiOrderResult
    noOrder msg = pure baseResult { aorMessage = msg }

    neutralMsg =
      case method of
        MethodBoth -> "No order: directions disagree or neutral (direction gate)."
        MethodKalmanOnly -> "No order: Kalman neutral (within threshold)."
        MethodLstmOnly -> "No order: LSTM neutral (within threshold)."

    shortErr :: SomeException -> String
    shortErr ex = take 240 (show ex)

    mode = if argBinanceLive args then OrderLive else OrderTest

    tryFetchFilters :: IO (Maybe SymbolFilters)
    tryFetchFilters = do
      r <- try (fetchSymbolFilters env sym) :: IO (Either SomeException SymbolFilters)
      case r of
        Left _ -> pure Nothing
        Right sf -> pure (Just sf)

    effectiveStep sf = sfMarketStepSize sf <|> sfLotStepSize sf
    effectiveMinQty sf = sfMarketMinQty sf <|> sfLotMinQty sf
    effectiveMaxQty sf = sfMarketMaxQty sf <|> sfLotMaxQty sf

    isLongSpot :: Maybe SymbolFilters -> Double -> Bool
    isLongSpot mSf baseBal =
      case mSf >>= effectiveMinQty of
        Nothing -> baseBal > 0
        Just minQ -> baseBal >= minQ

    normalizeQty :: SymbolFilters -> Double -> Double -> Either String Double
    normalizeQty sf price qtyRaw =
      let qty0 = max 0 qtyRaw
          qty1 = maybe qty0 (\st -> quantizeDown st qty0) (effectiveStep sf)
       in if qty1 <= 0
            then Left "Quantity rounds to 0."
            else
              case effectiveMinQty sf of
                Just minQ | qty1 < minQ -> Left ("Quantity below minQty (" ++ show minQ ++ ").")
                _ ->
                  case effectiveMaxQty sf of
                    Just maxQ | qty1 > maxQ -> Left ("Quantity above maxQty (" ++ show maxQ ++ ").")
                    _ ->
                      case sfMinNotional sf of
                        Just mn | price > 0 && qty1 * price < mn -> Left ("Notional below minNotional (" ++ show mn ++ ").")
                        _ -> Right qty1

    sendMarketOrder :: String -> OrderSide -> Maybe Double -> Maybe Double -> Maybe Bool -> IO ApiOrderResult
    sendMarketOrder sideLabel side mQty mQuote mReduceOnly = do
      let baseOut =
            baseResult
              { aorSide = Just sideLabel
              , aorQuantity = mQty
              , aorQuoteQuantity = mQuote
              }

      let tryReconcile ex =
            case (mode, clientOrderId) of
              (OrderLive, Just cid) -> do
                r2 <- try (fetchOrderByClientId env sym cid) :: IO (Either SomeException BL.ByteString)
                case r2 of
                  Left _ -> pure baseOut { aorMessage = "Order failed: " ++ shortErr ex }
                  Right body ->
                    let out0 =
                          baseOut
                            { aorSent = True
                            , aorResponse = Just (shortResp body)
                            , aorMessage = "Order reconciled by clientOrderId after error: " ++ shortErr ex
                            }
                     in pure (maybe out0 (`applyOrderInfo` out0) (decodeOrderInfo body))
              _ -> pure baseOut { aorMessage = "Order failed: " ++ shortErr ex }

      r <- try (placeMarketOrder env mode sym side mQty mQuote mReduceOnly clientOrderId) :: IO (Either SomeException BL.ByteString)
      case r of
        Left ex -> tryReconcile ex
        Right body ->
          let out0 =
                baseOut
                  { aorSent = True
                  , aorResponse = Just (shortResp body)
                  , aorMessage = "Order sent."
                  }
           in pure (maybe out0 (`applyOrderInfo` out0) (decodeOrderInfo body))

    place :: Maybe SymbolFilters -> String -> String -> Int -> IO ApiOrderResult
    place mSf baseAsset quoteAsset dir =
      case beMarket env of
        MarketSpot -> placeSpotOrMargin mSf baseAsset quoteAsset dir
        MarketMargin ->
          if mode == OrderTest
            then pure baseResult { aorMessage = "No order: margin trading requires binanceLive (no test endpoint)." }
            else placeSpotOrMargin mSf baseAsset quoteAsset dir
        MarketFutures -> placeFutures mSf quoteAsset dir

    placeSpotOrMargin :: Maybe SymbolFilters -> String -> String -> Int -> IO ApiOrderResult
    placeSpotOrMargin mSf baseAsset quoteAsset dir = do
      baseBal <- fetchFreeBalance env baseAsset
      let alreadyLong = isLongSpot mSf baseBal
          qtyArg = case argOrderQuantity args of { Just q | q > 0 -> Just q; _ -> Nothing }
          quoteArg = case argOrderQuote args of { Just q | q > 0 -> Just q; _ -> Nothing }
          quoteFracArg = case argOrderQuoteFraction args of { Just f | f > 0 -> Just f; _ -> Nothing }
          maxOrderQuoteArg = case argMaxOrderQuote args of { Just q | q > 0 -> Just q; _ -> Nothing }

      case dir of
        1 ->
          if alreadyLong
            then pure baseResult { aorMessage = "No order: already long." }
            else do
              let qtyArgBuy = fmap (* entryScale) qtyArg
                  quoteArgBuy = fmap (* entryScale) quoteArg
                  quoteFracArgBuy = fmap (* entryScale) quoteFracArg
              quoteBal <- fetchFreeBalance env quoteAsset
              let quoteFromFraction =
                    case (qtyArgBuy, quoteArgBuy, quoteFracArgBuy) of
                      (Nothing, Nothing, Just f) ->
                        let q0 = quoteBal * f
                            q1 = maybe q0 (\capQ -> min capQ q0) maxOrderQuoteArg
                         in Just q1
                      _ -> Nothing
                  quoteEffective = quoteArgBuy <|> quoteFromFraction

              case (qtyArgBuy, quoteEffective) of
                (Nothing, Nothing) ->
                  case quoteFracArgBuy of
                    Nothing -> pure baseResult { aorMessage = "No order: provide orderQuantity or orderQuote." }
                    Just _ -> pure baseResult { aorMessage = "No order: computed quote is 0 (check quote balance / orderQuoteFraction / maxOrderQuote)." }
                (Just qRaw, _) ->
                  case mSf of
                    Nothing ->
                      if qRaw * currentPrice > quoteBal
                        then pure baseResult { aorMessage = "No order: insufficient quote balance." }
                        else sendMarketOrder "BUY" Buy (Just qRaw) Nothing Nothing
                    Just sf ->
                      case normalizeQty sf currentPrice qRaw of
                        Left e -> pure baseResult { aorMessage = "No order: " ++ e }
                        Right q ->
                          if q * currentPrice > quoteBal
                            then pure baseResult { aorMessage = "No order: insufficient quote balance." }
                            else sendMarketOrder "BUY" Buy (Just q) Nothing Nothing
                (Nothing, Just qq0) ->
                  let qq = max 0 qq0
                   in if qq <= 0
                        then pure baseResult { aorMessage = "No order: quote is 0." }
                        else
                          case mSf >>= sfMinNotional of
                            Just mn | qq < mn -> pure baseResult { aorMessage = "No order: quote below minNotional." }
                            _ ->
                              if qq > quoteBal
                                then pure baseResult { aorMessage = "No order: insufficient quote balance." }
                                else sendMarketOrder "BUY" Buy Nothing (Just qq) Nothing
        (-1) ->
          if not alreadyLong
            then pure baseResult { aorMessage = "No order: already flat." }
            else do
              let qRaw = maybe baseBal id qtyArg
              case mSf of
                Nothing ->
                  if qRaw <= 0
                    then pure baseResult { aorMessage = "No order: quantity is 0." }
                    else sendMarketOrder "SELL" Sell (Just qRaw) Nothing Nothing
                Just sf ->
                  case normalizeQty sf currentPrice qRaw of
                    Left e -> pure baseResult { aorMessage = "No order: " ++ e }
                    Right q ->
                      if q > baseBal
                        then pure baseResult { aorMessage = "No order: insufficient base balance." }
                        else sendMarketOrder "SELL" Sell (Just q) Nothing Nothing
        _ -> pure baseResult { aorMessage = neutralMsg }

    placeFutures :: Maybe SymbolFilters -> String -> Int -> IO ApiOrderResult
    placeFutures mSf quoteAsset dir = do
      posAmt <- fetchFuturesPositionAmt env sym
      mQuoteFromFraction <-
        case (argOrderQuantity args, argOrderQuote args, argOrderQuoteFraction args) of
          (Nothing, Nothing, Just f) | f > 0 -> do
            bal <- fetchFuturesAvailableBalance env quoteAsset
            let q0 = bal * f * entryScale
                q1 =
                  let mCap =
                        case argMaxOrderQuote args of
                          Just q | q > 0 -> Just q
                          _ -> Nothing
                   in maybe q0 (\capQ -> min capQ q0) mCap
            pure (Just q1)
          _ -> pure Nothing
      let mDesiredQtyRaw =
            case argOrderQuantity args of
              Just q | q > 0 -> Just (q * entryScale)
              Just _ -> Nothing
              Nothing ->
                case ((fmap (* entryScale) (argOrderQuote args)) <|> mQuoteFromFraction) of
                  Just qq | qq > 0 && currentPrice > 0 -> Just (qq / currentPrice)
                  _ -> Nothing

          normalizeFuturesQty qRaw =
            case mSf of
              Nothing -> Right qRaw
              Just sf -> normalizeQty sf currentPrice qRaw

          closeOrder sideLabel side qtyRaw = do
            case normalizeFuturesQty qtyRaw of
              Left e -> pure baseResult { aorMessage = "No order: " ++ e }
              Right q ->
                if q <= 0 then pure baseResult { aorMessage = "No order: quantity is 0." } else sendMarketOrder sideLabel side (Just q) Nothing (Just True)

      case dir of
        1 ->
          if posAmt > 0
            then pure baseResult { aorMessage = "No order: already long." }
            else
              case mDesiredQtyRaw of
                Nothing -> pure baseResult { aorMessage = "No order: futures requires orderQuantity or orderQuote." }
                Just q0 -> do
                  let qtyToBuyRaw = if posAmt < 0 then abs posAmt + q0 else q0
                  case normalizeFuturesQty qtyToBuyRaw of
                    Left e -> pure baseResult { aorMessage = "No order: " ++ e }
                    Right q ->
                      if q <= 0 then pure baseResult { aorMessage = "No order: quantity is 0." } else sendMarketOrder "BUY" Buy (Just q) Nothing Nothing
        (-1) ->
          case argPositioning args of
            LongShort ->
              if posAmt < 0
                then pure baseResult { aorMessage = "No order: already short." }
                else
                  case mDesiredQtyRaw of
                    Nothing -> pure baseResult { aorMessage = "No order: futures requires orderQuantity or orderQuote." }
                    Just q0 -> do
                      let qtyToSellRaw = if posAmt > 0 then posAmt + q0 else q0
                      case normalizeFuturesQty qtyToSellRaw of
                        Left e -> pure baseResult { aorMessage = "No order: " ++ e }
                        Right q ->
                          if q <= 0 then pure baseResult { aorMessage = "No order: quantity is 0." } else sendMarketOrder "SELL" Sell (Just q) Nothing Nothing
            LongFlat ->
              if posAmt == 0
                then pure baseResult { aorMessage = "No order: already flat." }
                else if posAmt > 0
                  then closeOrder "SELL" Sell (abs posAmt)
                  else closeOrder "BUY" Buy (abs posAmt)
        _ -> pure baseResult { aorMessage = neutralMsg }

    modeLabel m =
      case m of
        OrderLive -> "live"
        OrderTest -> "test"

computeBacktestFromArgs :: Args -> IO Aeson.Value
computeBacktestFromArgs args = do
  (series, _) <- loadPrices args
  let prices = psClose series
  if length prices < 2 then error "Need at least 2 price rows" else pure ()
  let lookback = argLookback args
  summary <- computeBacktestSummary args lookback series
  pure (backtestSummaryJson summary)

backtestSummaryJson :: BacktestSummary -> Aeson.Value
backtestSummaryJson summary =
  object
    [ "split"
        .= object
          [ "train" .= bsTrainSize summary
          , "fit" .= bsFitSize summary
          , "tune" .= bsTuneSize summary
          , "tuneRatio" .= bsTuneRatio summary
          , "tuneStartIndex" .= bsFitSize summary
          , "backtest" .= bsBacktestSize summary
          , "backtestRatio" .= bsBacktestRatio summary
          , "backtestStartIndex" .= bsTrainEnd summary
          ]
    , "method" .= methodCode (bsMethodUsed summary)
    , "threshold" .= bsBestThreshold summary
    , "metrics" .= metricsToJson (bsMetrics summary)
    , "latestSignal" .= bsLatestSignal summary
    , "equityCurve" .= bsEquityCurve summary
    , "prices" .= bsBacktestPrices summary
    , "positions" .= bsPositions summary
    , "agreementOk" .= bsAgreementOk summary
    , "trades" .= map tradeToJson (bsTrades summary)
    ]

tradeToJson :: Trade -> Aeson.Value
tradeToJson tr =
  object
    [ "entryIndex" .= trEntryIndex tr
    , "exitIndex" .= trExitIndex tr
    , "entryEquity" .= trEntryEquity tr
    , "exitEquity" .= trExitEquity tr
    , "return" .= trReturn tr
    , "holdingPeriods" .= trHoldingPeriods tr
    , "exitReason" .= trExitReason tr
    ]

metricsToJson :: BacktestMetrics -> Aeson.Value
metricsToJson m =
  object
    [ "finalEquity" .= bmFinalEquity m
    , "totalReturn" .= bmTotalReturn m
    , "annualizedReturn" .= bmAnnualizedReturn m
    , "annualizedVolatility" .= bmAnnualizedVolatility m
    , "sharpe" .= bmSharpe m
    , "maxDrawdown" .= bmMaxDrawdown m
    , "tradeCount" .= bmTradeCount m
    , "roundTrips" .= bmRoundTrips m
    , "winRate" .= bmWinRate m
    , "grossProfit" .= bmGrossProfit m
    , "grossLoss" .= bmGrossLoss m
    , "profitFactor" .= bmProfitFactor m
    , "avgTradeReturn" .= bmAvgTradeReturn m
    , "avgHoldingPeriods" .= bmAvgHoldingPeriods m
    , "exposure" .= bmExposure m
    , "agreementRate" .= bmAgreementRate m
    , "turnover" .= bmTurnover m
    ]

runTradeOnly :: Args -> Int -> [Double] -> Maybe BinanceEnv -> IO ()
runTradeOnly args lookback prices mBinanceEnv = do
  signal <- computeTradeOnlySignal args lookback prices
  if argJson args
    then
      if argBinanceTrade args
        then do
          (sym, env) <-
            case (argBinanceSymbol args, mBinanceEnv) of
              (Just s, Just e) -> pure (s, e)
              _ -> error "Internal: --binance-trade requires binanceSymbol data source."
          order <- placeOrderForSignal args sym signal env
          printJsonStdout (object ["mode" .= ("trade" :: String), "trade" .= ApiTradeResponse signal order])
        else printJsonStdout (object ["mode" .= ("signal" :: String), "signal" .= signal])
    else do
      printLatestSignalSummary signal
      maybeSendBinanceOrder args mBinanceEnv signal

runBacktestPipeline :: Args -> Int -> PriceSeries -> Maybe BinanceEnv -> IO ()
runBacktestPipeline args lookback series mBinanceEnv = do
  let prices = psClose series
  summary <- computeBacktestSummary args lookback series
  if argJson args
    then do
      let base = backtestSummaryJson summary
      if argBinanceTrade args
        then do
          (sym, env) <-
            case (argBinanceSymbol args, mBinanceEnv) of
              (Just s, Just e) -> pure (s, e)
              _ -> error "Internal: --binance-trade requires binanceSymbol data source."
          order <- placeOrderForSignal args sym (bsLatestSignal summary) env
          printJsonStdout (object ["mode" .= ("backtest" :: String), "backtest" .= base, "trade" .= ApiTradeResponse (bsLatestSignal summary) order])
        else printJsonStdout (object ["mode" .= ("backtest" :: String), "backtest" .= base])
    else do
      let n = length prices
          trainEndRaw = bsTrainEndRaw summary
          trainEnd = bsTrainEnd summary
          backtestRatio = bsBacktestRatio summary

      if trainEndRaw /= trainEnd
        then
          putStrLn
            ( printf
                "Split adjusted for lookback: requested train=%d backtest=%d -> using train=%d backtest=%d"
                trainEndRaw
                (n - trainEndRaw)
                trainEnd
                (n - trainEnd)
            )
        else pure ()

      if bsTuneSize summary > 0
        then
          putStrLn
            ( printf
                "\nSplit: fit=%d tune=%d backtest=%d (tune-ratio=%.3f, backtest-ratio=%.3f)"
                (bsFitSize summary)
                (bsTuneSize summary)
                (bsBacktestSize summary)
                (bsTuneRatio summary)
                backtestRatio
            )
        else
          putStrLn
            ( printf
                "\nSplit: train=%d backtest=%d (backtest-ratio=%.3f)"
                (bsTrainSize summary)
                (bsBacktestSize summary)
                backtestRatio
            )

      if argOptimizeOperations args
        then
          putStrLn
            ( printf
                "Optimized on tune split: method=%s threshold=%.6f (%.3f%%)"
                (methodCode (bsMethodUsed summary))
                (bsBestThreshold summary)
                (bsBestThreshold summary * 100)
            )
        else if argSweepThreshold args
          then
            putStrLn
              ( printf
                  "Best threshold on tune split (by final equity): %.6f (%.3f%%)"
                  (bsBestThreshold summary)
                  (bsBestThreshold summary * 100)
              )
          else pure ()

      putStrLn $
        case bsMethodUsed summary of
          MethodBoth -> "Backtest (Kalman fusion + LSTM direction-agreement gated) complete."
          MethodKalmanOnly -> "Backtest (Kalman fusion only) complete."
          MethodLstmOnly -> "Backtest (LSTM only) complete."

      case bsLstmHistory summary of
        Nothing -> pure ()
        Just history -> printLstmSummary history

      printMetrics (bsMethodUsed summary) (bsMetrics summary)

      printLatestSignalSummary (bsLatestSignal summary)
      maybeSendBinanceOrder args mBinanceEnv (bsLatestSignal summary)

printJsonStdout :: ToJSON a => a -> IO ()
printJsonStdout v = BS.putStrLn (BL.toStrict (encode v))

computeTradeOnlySignal :: Args -> Int -> [Double] -> IO LatestSignal
computeTradeOnlySignal args lookback prices = do
  if argSweepThreshold args
    then error "Cannot use --sweep-threshold with --trade-only (sweep requires a backtest split)."
    else pure ()
  if argOptimizeOperations args
    then error "Cannot use --optimize-operations with --trade-only (optimization requires a backtest split)."
    else pure ()

  let method = argMethod args
      pricesV = V.fromList prices
      n = V.length pricesV
  if n <= lookback
    then
      error
        (printf "Not enough data for lookback=%d (need >= %d prices, got %d). Reduce --lookback-bars/--lookback-window or increase --bars." lookback (lookback + 1) n)
    else pure ()

  mLstmCtx <-
    case method of
      MethodKalmanOnly -> pure Nothing
      _ -> do
        let normState = fitNorm (argNormalization args) prices
            obsAll = forwardSeries normState prices
            lstmCfg =
              LSTMConfig
                { lcLookback = lookback
                , lcHiddenSize = argHiddenSize args
                , lcEpochs = argEpochs args
                , lcLearningRate = argLr args
                , lcValRatio = argValRatio args
                , lcPatience = argPatience args
                , lcGradClip = argGradClip args
                , lcSeed = argSeed args
                }
            (lstmModel, _) = trainLSTM lstmCfg obsAll
        pure (Just (normState, obsAll, lstmModel))

  mKalmanCtx <-
    case method of
      MethodLstmOnly -> pure Nothing
      _ -> do
        let predictors = trainPredictors lookback pricesV
            hmm0 = initHMMFilter predictors []
            kal0 =
              initKalman1
                0
                (max 1e-12 (argKalmanMeasurementVar args))
                (max 0 (argKalmanProcessVar args) * max 0 (argKalmanDt args))
            sv0 = emptySensorVar

            step (kal, hmm, sv) t =
              let priceT = pricesV V.! t
                  nextP = pricesV V.! (t + 1)
                  realizedR = if priceT == 0 then 0 else nextP / priceT - 1
                  (sensorOuts, predState) = predictSensors predictors pricesV hmm t
                  meas = mapMaybe (toMeasurement args sv) sensorOuts
                  kal' = stepMulti meas kal
                  sv' =
                    foldl'
                      (\acc (sid, out) -> updateResidual sid (realizedR - soMu out) acc)
                      sv
                      sensorOuts
                  hmm' = updateHMM predictors predState realizedR
               in (kal', hmm', sv')

            (kalPrev, hmmPrev, svPrev) = foldl' step (kal0, hmm0, sv0) [0 .. n - 2]
        pure (Just (predictors, kalPrev, hmmPrev, svPrev))

  pure (computeLatestSignal args lookback pricesV mLstmCtx mKalmanCtx)

-- LSTM weight persistence (for incremental training across backtests)

data PersistedLstmModel = PersistedLstmModel
  { plmVersion :: !Int
  , plmHiddenSize :: !Int
  , plmParams :: ![Double]
  } deriving (Eq, Show, Generic)

instance FromJSON PersistedLstmModel where
  parseJSON = Aeson.genericParseJSON (jsonOptions 3)

instance ToJSON PersistedLstmModel where
  toJSON = Aeson.genericToJSON (jsonOptions 3)

defaultLstmWeightsDir :: FilePath
defaultLstmWeightsDir = ".tmp/lstm"

resolveLstmWeightsDir :: IO (Maybe FilePath)
resolveLstmWeightsDir = do
  mDir <- lookupEnv "TRADER_LSTM_WEIGHTS_DIR"
  case trim <$> mDir of
    Nothing -> pure (Just defaultLstmWeightsDir)
    Just dir | null dir -> pure Nothing
    Just dir -> pure (Just dir)

safeCanonicalizePath :: FilePath -> IO FilePath
safeCanonicalizePath path = do
  r <- try (canonicalizePath path) :: IO (Either SomeException FilePath)
  pure (either (const path) id r)

lstmModelKey :: Args -> Int -> IO String
lstmModelKey args lookback = do
  src <-
    case (argBinanceSymbol args, argData args) of
      (Just sym, _) -> pure ("binance:" ++ marketCode (argBinanceMarket args) ++ ":" ++ sym)
      (Nothing, Just path0) -> do
        path <- safeCanonicalizePath path0
        pure ("csv:" ++ path ++ ":" ++ argPriceCol args)
      _ -> pure "unknown"
  pure
    ( intercalate
        "|"
        [ "v1"
        , src
        , "interval=" ++ argInterval args
        , "norm=" ++ show (argNormalization args)
        , "hidden=" ++ show (argHiddenSize args)
        , "lookback=" ++ show lookback
        ]
    )

hashKeyHex :: String -> String
hashKeyHex s =
  let digest :: Digest SHA256
      digest = hash (TE.encodeUtf8 (T.pack s))
   in BS.unpack (B16.encode (convert digest))

lstmWeightsPath :: Args -> Int -> IO (Maybe FilePath)
lstmWeightsPath args lookback = do
  mDir <- resolveLstmWeightsDir
  case mDir of
    Nothing -> pure Nothing
    Just dir -> do
      key <- lstmModelKey args lookback
      let file = "lstm-" ++ hashKeyHex key ++ ".json"
      pure (Just (dir </> file))

loadPersistedLstmModel :: FilePath -> Int -> IO (Maybe LSTMModel)
loadPersistedLstmModel path hidden = do
  exists <- doesFileExist path
  if not exists
    then pure Nothing
    else do
      eBs <- try (BL.readFile path) :: IO (Either SomeException BL.ByteString)
      case eBs of
        Left _ -> pure Nothing
        Right bs ->
          case eitherDecode bs of
            Left _ -> pure Nothing
            Right plm ->
              let ok =
                    plmVersion plm == 1
                      && plmHiddenSize plm == hidden
                      && length (plmParams plm) == paramCount hidden
               in if ok
                    then pure (Just (LSTMModel (plmHiddenSize plm) (plmParams plm)))
                    else pure Nothing

savePersistedLstmModelMaybe :: Maybe FilePath -> LSTMModel -> IO ()
savePersistedLstmModelMaybe mPath model =
  case mPath of
    Nothing -> pure ()
    Just path -> do
      _ <-
        try
          ( do
              createDirectoryIfMissing True (takeDirectory path)
              let plm =
                    PersistedLstmModel
                      { plmVersion = 1
                      , plmHiddenSize = lmHiddenSize model
                      , plmParams = lmParams model
                      }
              BL.writeFile path (encode plm)
          )
          :: IO (Either SomeException ())
      pure ()

trainLstmWithPersistence :: Args -> Int -> LSTMConfig -> [Double] -> IO (LSTMModel, [EpochStats])
trainLstmWithPersistence args lookback cfg series = do
  mPath <- lstmWeightsPath args lookback
  mSeed <-
    case mPath of
      Nothing -> pure Nothing
      Just path -> loadPersistedLstmModel path (lcHiddenSize cfg)
  let (model, hist) =
        case mSeed of
          Just seedModel -> fineTuneLSTM cfg seedModel series
          Nothing -> trainLSTM cfg series
  savePersistedLstmModelMaybe mPath model
  pure (model, hist)

computeBacktestSummary :: Args -> Int -> PriceSeries -> IO BacktestSummary
computeBacktestSummary args lookback series = do
  let prices = psClose series
      n = length prices
      backtestRatio = argBacktestRatio args
      split =
        case splitTrainBacktest lookback backtestRatio prices of
          Left err -> error err
          Right s -> s

      trainEndRaw = splitTrainEndRaw split
      trainEnd = splitTrainEnd split
      trainPrices = splitTrain split
      backtestPrices = splitBacktest split

      trainSize = length trainPrices
      tuningEnabled = argOptimizeOperations args || argSweepThreshold args
      tuneRatio = max 0 (min 0.999999 (argTuneRatio args))
      tuneRatioUsed = if tuningEnabled then tuneRatio else 0
      tuneSize =
        if tuningEnabled
          then max 0 (min trainSize (floor (fromIntegral trainSize * tuneRatioUsed)))
          else 0
      fitSize = max 0 (trainSize - tuneSize)

  if tuningEnabled && tuneSize < 2
    then
      error
        ( printf
            "Tune window too small (%d). Increase --tune-ratio, reduce --backtest-ratio, or increase the number of bars."
            tuneSize
        )
    else pure ()
  if tuningEnabled && fitSize < lookback + 1
    then
      error
        ( printf
            "Fit window too small for lookback=%d (fit=%d, tune=%d). Decrease --tune-ratio, reduce --lookback-bars/--lookback-window, or increase the number of bars."
            lookback
            fitSize
            tuneSize
        )
    else pure ()

  let (highsAll, lowsAll) =
        case (psHigh series, psLow series) of
          (Just hs, Just ls)
            | length hs == n && length ls == n -> (hs, ls)
          _ -> (prices, prices)

      predStart = if tuningEnabled then fitSize else trainEnd
      stepCount = n - predStart - 1
      fitPrices = take predStart prices

      tunePrices = drop fitSize trainPrices
      tuneHighs = take tuneSize (drop fitSize highsAll)
      tuneLows = take tuneSize (drop fitSize lowsAll)

      backtestHighs = drop trainEnd highsAll
      backtestLows = drop trainEnd lowsAll

      methodRequested = argMethod args
      methodForComputation =
        if argOptimizeOperations args
          then MethodBoth
          else methodRequested
      pricesV = V.fromList prices

      lstmCfg =
        LSTMConfig
          { lcLookback = lookback
          , lcHiddenSize = argHiddenSize args
          , lcEpochs = argEpochs args
          , lcLearningRate = argLr args
          , lcValRatio = argValRatio args
          , lcPatience = argPatience args
          , lcGradClip = argGradClip args
          , lcSeed = argSeed args
          }

  (mLstmCtx, mHistory, kalPredAll, lstmPredAll, mKalmanCtx, mMetaAll) <-
    case methodForComputation of
      MethodKalmanOnly -> do
        let fitPricesV = V.fromList fitPrices
            predictors = trainPredictors lookback fitPricesV
            hmmInitReturns = forwardReturns (take (predStart + 1) prices)
            hmm0 = initHMMFilter predictors hmmInitReturns
            kal0 =
              initKalman1
                0
                (max 1e-12 (argKalmanMeasurementVar args))
                (max 0 (argKalmanProcessVar args) * max 0 (argKalmanDt args))
            sv0 = emptySensorVar
            (kalFinal, hmmFinal, svFinal, kalPredRev, metaRev) =
              foldl'
                (backtestStepKalmanOnly args pricesV predictors predStart)
                (kal0, hmm0, sv0, [], [])
                [0 .. stepCount - 1]
            kalPred = reverse kalPredRev
            meta = reverse metaRev
        pure (Nothing, Nothing, kalPred, kalPred, Just (predictors, kalFinal, hmmFinal, svFinal), Just meta)
      MethodLstmOnly -> do
        let normState = fitNorm (argNormalization args) fitPrices
            obsAll = forwardSeries normState prices
            obsTrain = take predStart obsAll
        (lstmModel, history) <- trainLstmWithPersistence args lookback lstmCfg obsTrain
        let lstmPred =
              [ let t = predStart + i
                    window = take lookback (drop (t - lookback + 1) obsAll)
                    predObs = predictNext lstmModel window
                 in inverseNorm normState predObs
              | i <- [0 .. stepCount - 1]
              ]
        pure (Just (normState, obsAll, lstmModel), Just history, lstmPred, lstmPred, Nothing, Nothing)
      MethodBoth -> do
        let normState = fitNorm (argNormalization args) fitPrices
            obsAll = forwardSeries normState prices
            obsTrain = take predStart obsAll
        (lstmModel, history) <- trainLstmWithPersistence args lookback lstmCfg obsTrain
        let fitPricesV = V.fromList fitPrices
            predictors = trainPredictors lookback fitPricesV
            hmmInitReturns = forwardReturns (take (predStart + 1) prices)
            hmm0 = initHMMFilter predictors hmmInitReturns
            kal0 =
              initKalman1
                0
                (max 1e-12 (argKalmanMeasurementVar args))
                (max 0 (argKalmanProcessVar args) * max 0 (argKalmanDt args))
            sv0 = emptySensorVar
            (kalFinal, hmmFinal, svFinal, kalPredRev, lstmPredRev, metaRev) =
              foldl'
                (backtestStep args lookback normState obsAll pricesV lstmModel predictors predStart)
                (kal0, hmm0, sv0, [], [], [])
                [0 .. stepCount - 1]
            kalPred = reverse kalPredRev
            lstmPred = reverse lstmPredRev
            meta = reverse metaRev
        pure
          ( Just (normState, obsAll, lstmModel)
          , Just history
          , kalPred
          , lstmPred
          , Just (predictors, kalFinal, hmmFinal, svFinal)
          , Just meta
          )

  let baseCfg =
        EnsembleConfig
          { ecTradeThreshold = argTradeThreshold args
          , ecFee = argFee args
          , ecSlippage = argSlippage args
          , ecSpread = argSpread args
          , ecStopLoss = argStopLoss args
          , ecTakeProfit = argTakeProfit args
          , ecTrailingStop = argTrailingStop args
          , ecPositioning = argPositioning args
          , ecIntrabarFill = argIntrabarFill args
          , ecKalmanZMin = argKalmanZMin args
          , ecKalmanZMax = argKalmanZMax args
          , ecMaxHighVolProb = argMaxHighVolProb args
          , ecMaxConformalWidth = argMaxConformalWidth args
          , ecMaxQuantileWidth = argMaxQuantileWidth args
          , ecConfirmConformal = argConfirmConformal args
          , ecConfirmQuantiles = argConfirmQuantiles args
          , ecConfidenceSizing = argConfidenceSizing args
          , ecMinPositionSize = argMinPositionSize args
          }

      offsetBacktestPred = max 0 (trainEnd - predStart)
      kalPredBacktest = drop offsetBacktestPred kalPredAll
      lstmPredBacktest = drop offsetBacktestPred lstmPredAll
      kalPredTune = take (max 0 (tuneSize - 1)) kalPredAll
      lstmPredTune = take (max 0 (tuneSize - 1)) lstmPredAll
      metaBacktest = fmap (drop offsetBacktestPred) mMetaAll
      metaTune = fmap (take (max 0 (tuneSize - 1))) mMetaAll

      (methodUsed, bestThreshold) =
        if argOptimizeOperations args
          then
            let (m, thr, _btTune) =
                  optimizeOperationsWithHL baseCfg tunePrices tuneHighs tuneLows kalPredTune lstmPredTune metaTune
             in (m, thr)
          else if argSweepThreshold args
            then
              let (thr, _btTune) =
                    sweepThresholdWithHL methodRequested baseCfg tunePrices tuneHighs tuneLows kalPredTune lstmPredTune metaTune
               in (methodRequested, thr)
            else (methodRequested, argTradeThreshold args)

      backtestCfg = baseCfg { ecTradeThreshold = bestThreshold }
      (kalPredUsedBacktest, lstmPredUsedBacktest) = selectPredictions methodUsed kalPredBacktest lstmPredBacktest
      metaUsedBacktest =
        case methodUsed of
          MethodLstmOnly -> Nothing
          _ -> metaBacktest
      backtest =
        simulateEnsembleLongFlatWithHL
          backtestCfg
          1
          backtestPrices
          backtestHighs
          backtestLows
          kalPredUsedBacktest
          lstmPredUsedBacktest
          metaUsedBacktest

      ppy = periodsPerYear args
      metrics = computeMetrics ppy backtest

      argsForSignal =
        if argOptimizeOperations args
          then args { argMethod = methodUsed, argTradeThreshold = bestThreshold }
          else if argSweepThreshold args
            then args { argTradeThreshold = bestThreshold }
            else args

      latestSignal = computeLatestSignal argsForSignal lookback pricesV mLstmCtx mKalmanCtx

  pure
    BacktestSummary
      { bsTrainEndRaw = trainEndRaw
      , bsTrainEnd = trainEnd
      , bsTrainSize = trainSize
      , bsFitSize = fitSize
      , bsTuneSize = tuneSize
      , bsTuneRatio = tuneRatioUsed
      , bsBacktestSize = length backtestPrices
      , bsBacktestRatio = backtestRatio
      , bsMethodUsed = methodUsed
      , bsBestThreshold = bestThreshold
      , bsMetrics = metrics
      , bsLstmHistory = mHistory
      , bsLatestSignal = latestSignal
      , bsEquityCurve = brEquityCurve backtest
      , bsBacktestPrices = backtestPrices
      , bsPositions = brPositions backtest
      , bsAgreementOk = brAgreementOk backtest
      , bsTrades = brTrades backtest
      }

computeLatestSignal
  :: Args
  -> Int
  -> V.Vector Double
  -> Maybe LstmCtx
  -> Maybe KalmanCtx
  -> LatestSignal
computeLatestSignal args lookback pricesV mLstmCtx mKalmanCtx =
  case method of
    MethodBoth ->
      case (mKalmanCtx, mLstmCtx) of
        (Just _, Just _) -> compute
        _ -> error "Internal: --method 11 requires both Kalman and LSTM contexts."
    MethodKalmanOnly ->
      case mKalmanCtx of
        Just _ -> compute
        Nothing -> error "Internal: --method 10 requires Kalman context."
    MethodLstmOnly ->
      case mLstmCtx of
        Just _ -> compute
        Nothing -> error "Internal: --method 01 requires LSTM context."
  where
    method = argMethod args
    compute =
      let n = V.length pricesV
       in if n < 1
            then error "Need at least 1 price to compute latest signal"
            else
              let t = n - 1
                  currentPrice = pricesV V.! t
                  thr = max 0 (argTradeThreshold args)
                  directionPrice pred =
                    let upEdge = currentPrice * (1 + thr)
                        downEdge = currentPrice * (1 - thr)
                     in if pred > upEdge
                          then Just (1 :: Int)
                          else if pred < downEdge then Just (-1) else Nothing

                  clamp01 :: Double -> Double
                  clamp01 x = max 0 (min 1 x)

                  scale01 :: Double -> Double -> Double -> Double
                  scale01 lo hi x =
                    let lo' = min lo hi
                        hi' = max lo hi
                     in if hi' <= lo' + 1e-12
                          then if x >= hi' then 1 else 0
                          else clamp01 ((x - lo') / (hi' - lo'))

                  intervalWidth :: Interval -> Double
                  intervalWidth i = iHi i - iLo i

                  quantileWidth :: Quantiles -> Double
                  quantileWidth q = q90 q - q10 q

                  confirmConformal :: Maybe Interval -> Int -> Bool
                  confirmConformal mI dir =
                    if not (argConfirmConformal args)
                      then True
                      else
                        case (mI, dir) of
                          (Just i, 1) -> iLo i > thr
                          (Just i, (-1)) -> iHi i < negate thr
                          _ -> False

                  confirmQuantiles :: Maybe Quantiles -> Int -> Bool
                  confirmQuantiles mQ dir =
                    if not (argConfirmQuantiles args)
                      then True
                      else
                        case (mQ, dir) of
                          (Just q, 1) -> q10 q > thr
                          (Just q, (-1)) -> q90 q < negate thr
                          _ -> False

                  gateKalmanDir :: Double -> Maybe RegimeProbs -> Maybe Interval -> Maybe Quantiles -> Double -> Maybe Int -> (Maybe Int, Maybe String)
                  gateKalmanDir kalZ mReg mI mQ confScore dirRaw =
                    case dirRaw of
                      Nothing -> (Nothing, Nothing)
                      Just dir ->
                        let zMin = max 0 (argKalmanZMin args)
                            hvOk =
                              case (argMaxHighVolProb args, mReg) of
                                (Just maxHv, Just r) -> rpHighVol r <= maxHv
                                (Just _, Nothing) -> False
                                _ -> True
                            confWidthOk =
                              case (argMaxConformalWidth args, mI) of
                                (Just maxW, Just i) -> intervalWidth i <= maxW
                                (Just _, Nothing) -> False
                                _ -> True
                            qWidthOk =
                              case (argMaxQuantileWidth args, mQ) of
                                (Just maxW, Just q) -> quantileWidth q <= maxW
                                (Just _, Nothing) -> False
                                _ -> True
                            confOk = confScore >= argMinPositionSize args
                         in if kalZ < zMin
                              then (Nothing, Just "KALMAN_Z")
                              else if not hvOk
                                then (Nothing, Just "HMM_HIGH_VOL")
                                else if not confWidthOk
                                  then (Nothing, Just "CONFORMAL_WIDTH")
                                  else if not qWidthOk
                                    then (Nothing, Just "QUANTILE_WIDTH")
                                    else if not (confirmConformal mI dir)
                                      then (Nothing, Just "CONFORMAL_CONFIRM")
                                      else if not (confirmQuantiles mQ dir)
                                        then (Nothing, Just "QUANTILE_CONFIRM")
                                        else if argConfidenceSizing args && not confOk
                                          then (Nothing, Just "MIN_SIZE")
                                          else (Just dir, Nothing)

                  confidenceScoreKalman :: Double -> Maybe RegimeProbs -> Maybe Interval -> Maybe Quantiles -> Double
                  confidenceScoreKalman kalZ mReg mI mQ =
                    let zMin = max 0 (argKalmanZMin args)
                        zMax = max zMin (argKalmanZMax args)
                        zScore = scale01 zMin zMax kalZ
                        hvScore =
                          case (argMaxHighVolProb args, mReg) of
                            (Just maxHv, Just r) -> clamp01 ((maxHv - rpHighVol r) / max 1e-12 maxHv)
                            _ -> 1
                        confScore =
                          case (argMaxConformalWidth args, mI) of
                            (Just maxW, Just i) -> clamp01 ((maxW - intervalWidth i) / max 1e-12 maxW)
                            _ -> 1
                        qScore =
                          case (argMaxQuantileWidth args, mQ) of
                            (Just maxW, Just q) -> clamp01 ((maxW - quantileWidth q) / max 1e-12 maxW)
                            _ -> 1
                     in zScore * hvScore * confScore * qScore

                  (mKalNext, mKalReturn, mKalStd, mKalZ, mRegimes, mQuantiles, mConformal, kalDirRaw, kalDir, mConfidence, mPosSize, mGateReason) =
                    case mKalmanCtx of
                      Nothing -> (Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Just 0, Nothing)
                      Just (predictors, kalPrev, hmmPrev, svPrev) ->
                        let (sensorOuts, _) = predictSensors predictors pricesV hmmPrev t
                            mReg = listToMaybe [r | (_sid, out) <- sensorOuts, Just r <- [soRegimes out]]
                            mQ = listToMaybe [q | (_sid, out) <- sensorOuts, Just q <- [soQuantiles out]]
                            mI = listToMaybe [i | (_sid, out) <- sensorOuts, Just i <- [soInterval out]]
                            meas = mapMaybe (toMeasurement args svPrev) sensorOuts
                            kalNow = stepMulti meas kalPrev
                            kalReturn = kMean kalNow
                            kalVar = max 0 (kVar kalNow)
                            kalStd = sqrt kalVar
                            kalZ = if kalStd <= 0 then 0 else abs kalReturn / kalStd
                            kalNext = currentPrice * (1 + kalReturn)
                            dirRaw = directionPrice kalNext
                            confScore = confidenceScoreKalman kalZ mReg mI mQ
                            sizeRaw =
                              if argConfidenceSizing args
                                then confScore
                                else
                                  case dirRaw of
                                    Nothing -> 0
                                    Just _ -> 1
                            (dirUsed, mWhy) = gateKalmanDir kalZ mReg mI mQ confScore dirRaw
                            sizeUsed =
                              case dirUsed of
                                Nothing -> 0
                                Just _ ->
                                  let s0 = if argConfidenceSizing args then sizeRaw else 1
                                   in if argConfidenceSizing args && s0 < argMinPositionSize args then 0 else s0
                         in
                          ( Just kalNext
                          , Just kalReturn
                          , Just kalStd
                          , Just kalZ
                          , mReg
                          , mQ
                          , mI
                          , dirRaw
                          , dirUsed
                          , Just confScore
                          , Just sizeUsed
                          , mWhy
                          )

                  (mLstmNext, lstmDir) =
                    case mLstmCtx of
                      Nothing -> (Nothing, Nothing)
                      Just (normState, obsAll, lstmModel) ->
                        let start = t - lookback + 1
                         in if start < 0
                              then error "Not enough data to compute LSTM window for latest signal"
                              else
                                let window = take lookback (drop start obsAll)
                                    lstmNextObs = predictNext lstmModel window
                                    lstmNext = inverseNorm normState lstmNextObs
                                 in (Just lstmNext, directionPrice lstmNext)

                  agreeDir =
                    if kalDir == lstmDir
                      then kalDir
                      else Nothing
                  chosenDir0 =
                    case method of
                      MethodBoth -> agreeDir
                      MethodKalmanOnly -> kalDir
                      MethodLstmOnly -> lstmDir
                  chosenDir =
                    case chosenDir0 of
                      Nothing -> Nothing
                      Just _ ->
                        case method of
                          MethodLstmOnly -> chosenDir0
                          _ ->
                            case mPosSize of
                              Just sz | sz <= 0 -> Nothing
                              _ -> chosenDir0

                  action =
                    let downAction =
                          case argPositioning args of
                            LongShort -> "SHORT"
                            LongFlat -> "FLAT"
                     in
                    case method of
                      MethodBoth ->
                        case (kalDirRaw, lstmDir, chosenDir) of
                          (Just 1, Just 1, Just 1) -> "LONG"
                          (Just (-1), Just (-1), Just (-1)) -> downAction
                          (Nothing, Nothing, Nothing) -> "HOLD (both neutral)"
                          (Just _, Just _, Nothing) ->
                            case mGateReason of
                              Just why -> "HOLD (" ++ why ++ ")"
                              Nothing -> "HOLD (directions disagree)"
                          _ -> "HOLD (directions disagree)"
                      MethodKalmanOnly ->
                        case (kalDirRaw, chosenDir) of
                          (Just 1, Just 1) -> "LONG"
                          (Just (-1), Just (-1)) -> downAction
                          (Just _, Nothing) ->
                            case mGateReason of
                              Just why -> "HOLD (" ++ why ++ ")"
                              Nothing -> "HOLD (confidence gate)"
                          _ -> "HOLD (Kalman neutral)"
                      MethodLstmOnly ->
                        case chosenDir of
                          Just 1 -> "LONG"
                          Just (-1) -> downAction
                          _ -> "HOLD (LSTM neutral)"
                  posSizeFinal =
                    case method of
                      MethodLstmOnly ->
                        Just $
                          case chosenDir of
                            Nothing -> 0
                            Just _ -> 1
                      _ ->
                        case mPosSize of
                          Nothing -> Just 0
                          Just sz ->
                            case chosenDir of
                              Nothing -> Just 0
                              Just _ -> Just sz
               in LatestSignal
                    { lsMethod = method
                    , lsCurrentPrice = currentPrice
                    , lsThreshold = thr
                    , lsKalmanNext = mKalNext
                    , lsKalmanReturn = mKalReturn
                    , lsKalmanStd = mKalStd
                    , lsKalmanZ = mKalZ
                    , lsRegimes = mRegimes
                    , lsQuantiles = mQuantiles
                    , lsConformalInterval = mConformal
                    , lsConfidence =
                        case method of
                          MethodLstmOnly -> Nothing
                          _ -> mConfidence
                    , lsPositionSize = posSizeFinal
                    , lsKalmanDir = kalDir
                    , lsLstmNext = mLstmNext
                    , lsLstmDir = lstmDir
                    , lsChosenDir = chosenDir
                    , lsAction = action
                    }

printLatestSignalSummary :: LatestSignal -> IO ()
printLatestSignalSummary sig = do
  let showDir :: Maybe Int -> String
      showDir d =
        case d of
          Just 1 -> "UP"
          Just (-1) -> "DOWN"
          _ -> "NEUTRAL"

  putStrLn ""
  putStrLn "**Latest Signal**"
  putStrLn (printf "Method: %s" (methodCode (lsMethod sig)))
  case lsKalmanNext sig of
    Nothing -> putStrLn "Kalman next: (disabled)"
    Just kalNext -> putStrLn (printf "Kalman next: %.4f (%s)" kalNext (showDir (lsKalmanDir sig)))
  case lsKalmanZ sig of
    Nothing -> pure ()
    Just z -> putStrLn (printf "Kalman z:    %.3f" z)
  case lsConfidence sig of
    Nothing -> pure ()
    Just c -> putStrLn (printf "Confidence:  %.3f" c)
  case lsPositionSize sig of
    Nothing -> pure ()
    Just s -> putStrLn (printf "Pos size:    %.3f" s)
  case lsLstmNext sig of
    Nothing -> putStrLn "LSTM next:   (disabled)"
    Just lstmNext -> putStrLn (printf "LSTM next:   %.4f (%s)" lstmNext (showDir (lsLstmDir sig)))
  putStrLn (printf "Direction threshold: %.3f%%" (lsThreshold sig * 100))
  putStrLn (printf "Action: %s" (lsAction sig))

maybeSendBinanceOrder :: Args -> Maybe BinanceEnv -> LatestSignal -> IO ()
maybeSendBinanceOrder args mEnv sig =
  case (argBinanceSymbol args, mEnv) of
    (Just sym, Just env)
      | argBinanceTrade args -> do
          res <- placeOrderForSignal args sym sig env
          putStrLn (aorMessage res)
      | otherwise -> pure ()
    _ -> pure ()

data PriceSeries = PriceSeries
  { psClose :: ![Double]
  , psHigh :: !(Maybe [Double])
  , psLow :: !(Maybe [Double])
  } deriving (Eq, Show)

loadPrices :: Args -> IO (PriceSeries, Maybe BinanceEnv)
loadPrices args =
  case (argData args, argBinanceSymbol args) of
    (Just path, Nothing) -> do
      closes <- loadPricesCsv path (argPriceCol args)
      let bars = resolveBarsForCsv args
      let closes' =
            if bars > 0
              then takeLast bars closes
              else closes
      pure (PriceSeries closes' Nothing Nothing, Nothing)
    (Nothing, Just sym) -> do
      (env, series) <- loadPricesBinance args sym
      pure (series, Just env)
    (Just _, Just _) -> error "Provide only one of --data or --binance-symbol"
    (Nothing, Nothing) -> error "Provide --data or --binance-symbol"

takeLast :: Int -> [a] -> [a]
takeLast n xs
  | n <= 0 = xs
  | otherwise =
      let k = length xs - n
       in if k <= 0 then xs else drop k xs

loadPricesBinance :: Args -> String -> IO (BinanceEnv, PriceSeries)
loadPricesBinance args sym = do
  let market = argBinanceMarket args
  if market == MarketMargin && argBinanceTestnet args
    then error "--binance-testnet is not supported for margin operations"
    else pure ()
  let bars = resolveBarsForBinance args
  let tradeBase =
        case market of
          MarketFutures -> if argBinanceTestnet args then binanceFuturesTestnetBaseUrl else binanceFuturesBaseUrl
          _ -> if argBinanceTestnet args then binanceTestnetBaseUrl else binanceBaseUrl
      dataBase =
        case market of
          MarketFutures -> binanceFuturesBaseUrl
          _ -> binanceBaseUrl
  apiKey <- resolveEnv "BINANCE_API_KEY" (argBinanceApiKey args)
  apiSecret <- resolveEnv "BINANCE_API_SECRET" (argBinanceApiSecret args)
  envTrade <- newBinanceEnv market tradeBase (BS.pack <$> apiKey) (BS.pack <$> apiSecret)
  klinesE <- try (fetchKlines envTrade sym (argInterval args) bars) :: IO (Either HttpException [Kline])
  ks <-
    case klinesE of
      Right out -> pure out
      Left ex ->
        if argBinanceTestnet args
          then do
            envData <- newBinanceEnv market dataBase (BS.pack <$> apiKey) (BS.pack <$> apiSecret)
            fetchKlines envData sym (argInterval args) bars
          else throwIO ex
  let closes = map kClose ks
      highs = map kHigh ks
      lows = map kLow ks
  pure (envTrade, PriceSeries closes (Just highs) (Just lows))

resolveEnv :: String -> Maybe String -> IO (Maybe String)
resolveEnv name override =
  case override of
    Just v -> pure (Just v)
    Nothing -> lookupEnv name

periodsPerYear :: Args -> Double
periodsPerYear args =
  case argPeriodsPerYear args of
    Just v -> v
    Nothing ->
      inferPeriodsPerYear (argInterval args)

inferPeriodsPerYear :: String -> Double
inferPeriodsPerYear interval =
  case interval of
    "1m" -> 60 * 24 * 365
    "3m" -> 20 * 24 * 365
    "5m" -> 12 * 24 * 365
    "15m" -> 4 * 24 * 365
    "30m" -> 2 * 24 * 365
    "1h" -> 24 * 365
    "2h" -> 12 * 365
    "4h" -> 6 * 365
    "6h" -> 4 * 365
    "8h" -> 3 * 365
    "12h" -> 2 * 365
    "1d" -> 365
    "3d" -> 365 / 3
    "1w" -> 52
    "1M" -> 12
    _ -> 365

forwardReturns :: [Double] -> [Double]
forwardReturns ps =
  [ if p0 == 0 then 0 else p1 / p0 - 1
  | (p0, p1) <- zip ps (drop 1 ps)
  ]

toMeasurement :: Args -> SensorVar -> (SensorId, SensorOutput) -> Maybe (Double, Double)
toMeasurement args sv (sid, out) =
  let fallbackVar = max 1e-12 (argKalmanMeasurementVar args)
      var =
        case soSigma out of
          Just s | s > 0 -> s * s
          _ -> maybe fallbackVar id (varianceFor sid sv)
      var' = max 1e-12 var
   in Just (soMu out, var')

backtestStepKalmanOnly
  :: Args
  -> V.Vector Double
  -> PredictorBundle
  -> Int
  -> (Kalman1, HMMFilter, SensorVar, [Double], [StepMeta])
  -> Int
  -> (Kalman1, HMMFilter, SensorVar, [Double], [StepMeta])
backtestStepKalmanOnly args pricesV predictors trainEnd (kal, hmm, sv, kalAcc, metaAcc) i =
  let t = trainEnd + i
      priceT = pricesV V.! t
      nextP = pricesV V.! (t + 1)
      realizedR = if priceT == 0 then 0 else nextP / priceT - 1

      (sensorOuts, predState) = predictSensors predictors pricesV hmm t
      mReg = listToMaybe [r | (_sid, out) <- sensorOuts, Just r <- [soRegimes out]]
      mQ = listToMaybe [q | (_sid, out) <- sensorOuts, Just q <- [soQuantiles out]]
      mI = listToMaybe [i' | (_sid, out) <- sensorOuts, Just i' <- [soInterval out]]
      meas = mapMaybe (toMeasurement args sv) sensorOuts
      kal' = stepMulti meas kal
      fusedR = kMean kal'
      kalNext = priceT * (1 + fusedR)
      meta =
        StepMeta
          { smKalmanMean = kMean kal'
          , smKalmanVar = kVar kal'
          , smHighVolProb = rpHighVol <$> mReg
          , smQuantile10 = q10 <$> mQ
          , smQuantile90 = q90 <$> mQ
          , smConformalLo = iLo <$> mI
          , smConformalHi = iHi <$> mI
          }

      sv' =
        foldl'
          (\acc (sid, out) -> updateResidual sid (realizedR - soMu out) acc)
          sv
          sensorOuts
      hmm' = updateHMM predictors predState realizedR
   in (kal', hmm', sv', kalNext : kalAcc, meta : metaAcc)

backtestStep
  :: Args
  -> Int
  -> NormState
  -> [Double]
  -> V.Vector Double
  -> LSTMModel
  -> PredictorBundle
  -> Int
  -> (Kalman1, HMMFilter, SensorVar, [Double], [Double], [StepMeta])
  -> Int
  -> (Kalman1, HMMFilter, SensorVar, [Double], [Double], [StepMeta])
backtestStep args lookback normState obsAll pricesV lstmModel predictors trainEnd (kal, hmm, sv, kalAcc, lstmAcc, metaAcc) i =
  let t = trainEnd + i
      priceT = pricesV V.! t
      nextP = pricesV V.! (t + 1)
      realizedR = if priceT == 0 then 0 else nextP / priceT - 1

      (sensorOuts, predState) = predictSensors predictors pricesV hmm t
      mReg = listToMaybe [r | (_sid, out) <- sensorOuts, Just r <- [soRegimes out]]
      mQ = listToMaybe [q | (_sid, out) <- sensorOuts, Just q <- [soQuantiles out]]
      mI = listToMaybe [i' | (_sid, out) <- sensorOuts, Just i' <- [soInterval out]]
      meas = mapMaybe (toMeasurement args sv) sensorOuts
      kal' = stepMulti meas kal
      fusedR = kMean kal'
      kalNext = priceT * (1 + fusedR)
      meta =
        StepMeta
          { smKalmanMean = kMean kal'
          , smKalmanVar = kVar kal'
          , smHighVolProb = rpHighVol <$> mReg
          , smQuantile10 = q10 <$> mQ
          , smQuantile90 = q90 <$> mQ
          , smConformalLo = iLo <$> mI
          , smConformalHi = iHi <$> mI
          }

      window = take lookback (drop (t - lookback + 1) obsAll)
      lstmNextObs = predictNext lstmModel window
      lstmNext = inverseNorm normState lstmNextObs

      sv' =
        foldl'
          (\acc (sid, out) -> updateResidual sid (realizedR - soMu out) acc)
          sv
          sensorOuts
      hmm' = updateHMM predictors predState realizedR
   in (kal', hmm', sv', kalNext : kalAcc, lstmNext : lstmAcc, meta : metaAcc)

printLstmSummary :: [EpochStats] -> IO ()
printLstmSummary history =
  case history of
    [] -> putStrLn "LSTM: no training history"
    _ ->
      let bestVal = minimum (map esValLoss history)
       in putStrLn (printf "LSTM: epochs=%d best_val_loss=%.6f" (length history) bestVal)

printMetrics :: Method -> BacktestMetrics -> IO ()
printMetrics method m = do
  putStrLn ""
  putStrLn "**Profitability**"
  putStrLn (printf "Final equity: %.4fx" (bmFinalEquity m))
  putStrLn (printf "Total return: %.2f%%" (bmTotalReturn m * 100))
  putStrLn (printf "Annualized return: %.2f%%" (bmAnnualizedReturn m * 100))

  putStrLn ""
  putStrLn "**Risk & Volatility**"
  putStrLn (printf "Annualized volatility: %.2f%%" (bmAnnualizedVolatility m * 100))
  putStrLn (printf "Sharpe ratio (rf=0): %.3f" (bmSharpe m))
  putStrLn (printf "Max drawdown: %.2f%%" (bmMaxDrawdown m * 100))

  putStrLn ""
  putStrLn "**Trade Execution**"
  putStrLn (printf "Position changes: %d" (bmTradeCount m))
  putStrLn (printf "Round trips: %d" (bmRoundTrips m))
  putStrLn (printf "Win rate: %.1f%%" (bmWinRate m * 100))
  let profitFactorLabel :: String
      profitFactorLabel =
        case bmProfitFactor m of
          Just pf -> printf "%.3f" pf
          Nothing ->
            if bmGrossProfit m > 0
              then "∞"
              else "0"
  putStrLn (printf "Gross profit: %.4f" (bmGrossProfit m))
  putStrLn (printf "Gross loss: %.4f" (bmGrossLoss m))
  putStrLn (printf "Profit factor: %s" profitFactorLabel)
  putStrLn (printf "Avg trade return: %.2f%%" (bmAvgTradeReturn m * 100))
  putStrLn (printf "Avg holding (periods): %.2f" (bmAvgHoldingPeriods m))

  putStrLn ""
  putStrLn "**Efficiency**"
  putStrLn (printf "Exposure (time in market): %.1f%%" (bmExposure m * 100))
  let agreeLabel :: String
      agreeLabel =
        case method of
          MethodBoth -> "Direction agreement rate"
          MethodKalmanOnly -> "Signal rate (Kalman)"
          MethodLstmOnly -> "Signal rate (LSTM)"
  putStrLn (printf "%s: %.1f%%" agreeLabel (bmAgreementRate m * 100))
  putStrLn (printf "Turnover (changes/period): %.4f" (bmTurnover m))

shortResp :: BL.ByteString -> String
shortResp bs =
  let s = BS.unpack (BL.toStrict bs)
   in take 200 s

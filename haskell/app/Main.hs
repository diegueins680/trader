{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
module Main where

import Control.Concurrent (ThreadId, forkIO, killThread, myThreadId, threadDelay)
import Control.Concurrent.MVar (MVar, modifyMVar, modifyMVar_, newEmptyMVar, newMVar, readMVar, swapMVar, tryPutMVar, tryReadMVar, withMVar)
import Control.Exception (IOException, SomeException, finally, fromException, throwIO, try)
import Control.Applicative ((<|>))
import Control.Monad (forM, forM_, when)
import Crypto.Hash (Digest, hash)
import Crypto.Hash.Algorithms (SHA256)
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode, encode, object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.Types as AT
import Data.ByteArray (convert)
import Data.Char (isAlphaNum, isDigit, isSpace, toLower, toUpper)
import Data.Foldable (toList)
import Data.Int (Int64)
import Data.List (foldl', intercalate, isInfixOf, isPrefixOf, isSuffixOf, sortOn)
import Data.Maybe (catMaybes, fromMaybe, isJust, isNothing, listToMaybe, mapMaybe, maybeToList)
import qualified Data.Sequence as Seq
import Data.Sequence (Seq)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Version (showVersion)
import Data.Word (Word64)
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.Csv as Csv
import qualified Data.HashMap.Strict as HM
import qualified Data.Vector as V
import Data.IORef (IORef, atomicModifyIORef', modifyIORef', newIORef, readIORef, writeIORef)
import GHC.Conc (getNumCapabilities, setNumCapabilities)
import GHC.Exception (ErrorCall(..))
import GHC.Generics (Generic)
import Network.HTTP.Client (HttpException, Manager, Request, RequestBody(..), Response, httpLbs, newManager, parseRequest, requestBody, requestHeaders, method, responseTimeout, responseTimeoutMicro)
import Network.HTTP.Client.TLS (tlsManagerSettings)
import Network.HTTP.Types (ResponseHeaders, Status, status200, status202, status204, status400, status401, status404, status405, status413, status429, status500, status502, status504, statusCode)
import Network.HTTP.Types.Header (hAuthorization, hCacheControl, hPragma)
import qualified Network.Wai as Wai
import qualified Network.Wai.Handler.Warp as Warp
import Options.Applicative
import System.Directory (canonicalizePath, createDirectoryIfMissing, doesDirectoryExist, doesFileExist, findExecutable, getCurrentDirectory, listDirectory, removeFile, renameFile)
import System.Environment (getExecutablePath, lookupEnv)
import System.Exit (ExitCode(..), die, exitFailure)
import System.FilePath ((</>), takeDirectory)
import System.IO (Handle, IOMode(ReadMode), hClose, hFlush, hGetLine, hIsEOF, hPutStrLn, openTempFile, stderr, stdout, withFile)
import System.IO.Error (ioeGetErrorString, isUserError)
import System.Process (CreateProcess(..), proc, readCreateProcessWithExitCode)
import System.Random (randomIO)
import System.Timeout (timeout)
import Data.Time.Clock.POSIX (getPOSIXTime, utcTimeToPOSIXSeconds)
import Data.Time.Format (defaultTimeLocale, parseTimeM)
import Text.Printf (printf)
import Text.Read (readMaybe)
import qualified Paths_trader as Paths

import Trader.Binance
  ( BinanceEnv(..)
  , BinanceLog(..)
  , BinanceMarket(..)
  , BinanceOrderMode(..)
  , BinanceTrade(..)
  , FuturesPositionRisk(..)
  , OrderSide(..)
  , BinanceOpenOrder(..)
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
  , fetchFuturesPositionRisks
  , fetchOpenOrders
  , cancelFuturesOpenOrdersByClientPrefix
  , fetchSymbolFilters
  , quantizeDown
  , getTimestampMs
  , placeMarketOrder
  , placeFuturesTriggerMarketOrder
  , fetchOrderByClientId
  , fetchAccountTrades
  , createListenKey
  , keepAliveListenKey
  , closeListenKey
  )
import Trader.Kalman3 (KalmanRun(..), runConstantAcceleration1D)
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
import Trader.LstmPersistence (lstmModelKey)
import Trader.Metrics (BacktestMetrics(..), computeMetrics)
import Trader.MarketContext (MarketModel, buildMarketModel, marketMeasurementAt)
import Trader.BinanceIntervals (binanceIntervals)
import Trader.Duration (lookbackBarsFrom, parseIntervalSeconds)
import Trader.Normalization (NormState, NormType(..), fitNorm, forwardSeries, inverseNorm, inverseSeries, parseNormType)
import Trader.Optimizer.Json (encodePretty)
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
import Trader.Optimization
  ( TuneConfig(..)
  , TuneObjective(..)
  , TuneStats(..)
  , tuneObjectiveCode
  , parseTuneObjective
  , optimizeOperationsWith
  , optimizeOperationsWithHLWith
  , sweepThresholdWith
  , sweepThresholdWithHLWith
  )
import Trader.Split (Split(..), splitTrainBacktest)
import Trader.Text (dedupeStable, normalizeKey, trim)
import Trader.App.Args
  ( Args(..)
  , argBinanceMarket
  , argLookback
  , intrabarFillCode
  , opts
  , parseIntrabarFill
  , parsePositioning
  , positioningCode
  , resolveBarsForBinance
  , resolveBarsForCsv
  , resolveBarsForPlatform
  , validateArgs
  )
import Trader.Trading
  ( BacktestResult(..)
  , EnsembleConfig(..)
  , IntrabarFill(..)
  , Positioning(..)
  , PositionSide(..)
  , StepMeta(..)
  , Trade(..)
  , exitReasonFromCode
  , simulateEnsemble
  , simulateEnsembleWithHL
  )
import Trader.S3
  ( S3State(..)
  , resolveS3State
  , s3GetObject
  , s3KeyFor
  , s3PutObject
  )
import Trader.Platform
  ( Platform(..)
  , isPlatformInterval
  , parsePlatform
  , platformCode
  , platformLabel
  , platformIntervalsCsv
  , platformSupportsMarketContext
  , platformSupportsLiveBot
  , coinbaseIntervalSeconds
  , krakenIntervalMinutes
  , poloniexIntervalLabel
  , poloniexIntervalSeconds
  )
import Trader.Coinbase
  ( CoinbaseCandle(..)
  , CoinbaseEnv(..)
  , fetchCoinbaseAccounts
  , fetchCoinbaseAvailableBalance
  , fetchCoinbaseCandles
  , newCoinbaseEnv
  , placeCoinbaseMarketOrder
  )
import Trader.Kraken (KrakenCandle(..), fetchKrakenCandles)
import Trader.Poloniex (PoloniexCandle(..), fetchPoloniexCandles)

-- CSV loading

resolveCsvColumnKey :: FilePath -> String -> [BS.ByteString] -> String -> BS.ByteString
resolveCsvColumnKey path flagName hdrList raw =
  let wanted = trim raw
      wantedLower = map toLower wanted
      wantedNorm = normalizeKey wanted
      wantedBs = BS.pack wanted
      mKeyExact =
        if wantedBs `elem` hdrList then Just wantedBs else Nothing
      mKeyNorm =
        case filter (\h -> normalizeKey (BS.unpack h) == wantedNorm) hdrList of
          (h : _) -> Just h
          [] -> Nothing
      mKey =
        case mKeyExact of
          Just k -> Just k
          Nothing ->
            case filter (\h -> map toLower (BS.unpack h) == wantedLower) hdrList of
              (h : _) -> Just h
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
   in
    if null wanted
      then error (flagName ++ " cannot be empty")
      else
        case mKey of
          Just k -> k
          Nothing ->
            let hint =
                  if null suggestions
                    then ""
                    else " Suggestions: " ++ BS.unpack (BS.intercalate ", " suggestions) ++ "."
             in
              error
                ( "Column not found for "
                    ++ flagName
                    ++ ": "
                    ++ wanted
                    ++ " (file: "
                    ++ path
                    ++ "). Available columns: "
                    ++ available
                    ++ "."
                    ++ hint
                )

extractCellDoubleAt :: Int -> BS.ByteString -> Csv.NamedRecord -> Double
extractCellDoubleAt rowIndex key rec =
  case HM.lookup key rec of
    Nothing -> error ("Column not found: " ++ BS.unpack key)
    Just raw ->
      let s = trim (BS.unpack raw)
       in case readMaybe s of
            Just d -> d
            Nothing ->
              error
                ( "Failed to parse value at row "
                    ++ show rowIndex
                    ++ " ("
                    ++ BS.unpack key
                    ++ "): "
                    ++ s
                )

loadCsvPriceSeries :: FilePath -> String -> Maybe String -> Maybe String -> IO ([Double], Maybe [Double], Maybe [Double], Maybe [Int64])
loadCsvPriceSeries path closeCol mHighCol mLowCol = do
  exists <- doesFileExist path
  if not exists
    then do
      cwd <- getCurrentDirectory
      error ("CSV path not found: " ++ path ++ " (cwd: " ++ cwd ++ ")")
    else pure ()
  bs <- BL.readFile path
  case Csv.decodeByName bs of
    Left err -> error ("CSV decode failed (" ++ path ++ "): " ++ err)
    Right (hdr, rows) -> do
      let hdrList = V.toList hdr
          rowsList0 = V.toList rows
          mTimeKey = csvTimeKey hdrList
          rowsList = maybe rowsList0 (\tk -> sortCsvRowsByTime tk rowsList0) mTimeKey
          closeKey = resolveCsvColumnKey path "--price-column" hdrList closeCol
          mHighKey = fmap (resolveCsvColumnKey path "--high-column" hdrList) mHighCol
          mLowKey = fmap (resolveCsvColumnKey path "--low-column" hdrList) mLowCol
          closeSeries = zipWith (\i row -> extractCellDoubleAt i closeKey row) [1 :: Int ..] rowsList
          highSeries = fmap (\k -> zipWith (\i row -> extractCellDoubleAt i k row) [1 :: Int ..] rowsList) mHighKey
          lowSeries = fmap (\k -> zipWith (\i row -> extractCellDoubleAt i k row) [1 :: Int ..] rowsList) mLowKey
          openTimes = mTimeKey >>= \tk -> parseCsvTimes tk rowsList
      pure (closeSeries, highSeries, lowSeries, openTimes)

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

parseCsvTimes :: BS.ByteString -> [Csv.NamedRecord] -> Maybe [Int64]
parseCsvTimes timeKey rows =
  case traverse (lookupCell timeKey) rows of
    Nothing -> Nothing
    Just rawTimes ->
      let times = map (trim . BS.unpack) rawTimes
       in traverse parseTimeMs times

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

normalizeEpochMs :: Int64 -> Int64
normalizeEpochMs n =
  if n < 100000000000
    then n * 1000
    else n

parseTimeMs :: String -> Maybe Int64
parseTimeMs s =
  case parseTimeInt64 s of
    Just n -> Just (normalizeEpochMs n)
    Nothing ->
      if looksLikeIso8601Prefix s
        then parseIsoTimeMs s
        else Nothing

parseIsoTimeMs :: String -> Maybe Int64
parseIsoTimeMs s =
  let formats =
        [ "%Y-%m-%d"
        , "%Y-%m-%d %H:%M:%S"
        , "%Y-%m-%dT%H:%M:%S"
        , "%Y-%m-%d %H:%M:%S%Q"
        , "%Y-%m-%dT%H:%M:%S%Q"
        , "%Y-%m-%dT%H:%M:%S%QZ"
        , "%Y-%m-%d %H:%M:%S%QZ"
        ]
      parseWith fmt = parseTimeM True defaultTimeLocale fmt s
   in case mapMaybe parseWith formats of
        [] -> Nothing
        (t : _) -> Just (floor (utcTimeToPOSIXSeconds t * 1000))

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
  , lsVolatility :: !(Maybe Double)
  , lsRegimes :: !(Maybe RegimeProbs)
  , lsQuantiles :: !(Maybe Quantiles)
  , lsConformalInterval :: !(Maybe Interval)
  , lsConfidence :: !(Maybe Double)
  , lsPositionSize :: !(Maybe Double)
  , lsKalmanDir :: !(Maybe Int)
  , lsLstmNext :: !(Maybe Double)
  , lsLstmDir :: !(Maybe Int)
  , lsChosenDir :: !(Maybe Int)
  , lsCloseDir :: !(Maybe Int)
  , lsAction :: !String
  } deriving (Eq, Show)

data BacktestSummary = BacktestSummary
  { bsTrainEndRaw :: !Int
  , bsTrainEnd :: !Int
  , bsTrainSize :: !Int
  , bsFitSize :: !Int
  , bsTuneSize :: !Int
  , bsTuneRatio :: !Double
  , bsTuneObjective :: !TuneObjective
  , bsTunePenaltyMaxDrawdown :: !Double
  , bsTunePenaltyTurnover :: !Double
  , bsTuneStressVolMult :: !Double
  , bsTuneStressShock :: !Double
  , bsTuneStressWeight :: !Double
  , bsMinRoundTrips :: !Int
  , bsWalkForwardFolds :: !Int
  , bsTuneStats :: !(Maybe TuneStats)
  , bsTuneMetrics :: !(Maybe BacktestMetrics)
  , bsBacktestSize :: !Int
  , bsBacktestRatio :: !Double
  , bsMethodUsed :: !Method
  , bsBestOpenThreshold :: !Double
  , bsBestCloseThreshold :: !Double
  , bsMinHoldBars :: !Int
  , bsCooldownBars :: !Int
  , bsMaxHoldBars :: !(Maybe Int)
  , bsStopLossVolMult :: !Double
  , bsTakeProfitVolMult :: !Double
  , bsTrailingStopVolMult :: !Double
  , bsMaxPositionSize :: !Double
  , bsMinEdge :: !Double
  , bsMinSignalToNoise :: !Double
  , bsCostAwareEdge :: !Bool
  , bsEdgeBuffer :: !Double
  , bsTrendLookback :: !Int
  , bsVolTarget :: !(Maybe Double)
  , bsVolLookback :: !Int
  , bsVolEwmaAlpha :: !(Maybe Double)
  , bsVolFloor :: !Double
  , bsVolScaleMax :: !Double
  , bsMaxVolatility :: !(Maybe Double)
  , bsRebalanceBars :: !Int
  , bsRebalanceThreshold :: !Double
  , bsFundingRate :: !Double
  , bsRebalanceGlobal :: !Bool
  , bsFundingBySide :: !Bool
  , bsRebalanceResetOnSignal :: !Bool
  , bsFundingOnOpen :: !Bool
  , bsBlendWeight :: !Double
  , bsTriLayer :: !Bool
  , bsTriLayerFastMult :: !Double
  , bsTriLayerSlowMult :: !Double
  , bsTriLayerCloudPadding :: !Double
  , bsTriLayerPriceAction :: !Bool
  , bsLstmExitFlipBars :: !Int
  , bsFee :: !Double
  , bsSlippage :: !Double
  , bsSpread :: !Double
  , bsEstimatedPerSideCost :: !Double
  , bsEstimatedRoundTripCost :: !Double
  , bsMetrics :: !BacktestMetrics
  , bsBaselines :: ![Baseline]
  , bsWalkForward :: !(Maybe WalkForwardReport)
  , bsLstmHistory :: !(Maybe [EpochStats])
  , bsLatestSignal :: !LatestSignal
  , bsEquityCurve :: ![Double]
  , bsBacktestPrices :: ![Double]
  , bsKalmanPredNext :: ![Maybe Double]
  , bsLstmPredNext :: ![Maybe Double]
  , bsPositions :: ![Double]
  , bsAgreementOk :: ![Bool]
  , bsTrades :: ![Trade]
  } deriving (Eq, Show)

data Baseline = Baseline
  { blName :: !String
  , blMetrics :: !BacktestMetrics
  } deriving (Eq, Show)

data WalkForwardFold = WalkForwardFold
  { wffStartIndex :: !Int
  , wffEndIndex :: !Int
  , wffMetrics :: !BacktestMetrics
  } deriving (Eq, Show)

data WalkForwardSummary = WalkForwardSummary
  { wfsFinalEquityMean :: !Double
  , wfsFinalEquityStd :: !Double
  , wfsAnnualizedReturnMean :: !Double
  , wfsAnnualizedReturnStd :: !Double
  , wfsSharpeMean :: !Double
  , wfsSharpeStd :: !Double
  , wfsMaxDrawdownMean :: !Double
  , wfsMaxDrawdownStd :: !Double
  , wfsTurnoverMean :: !Double
  , wfsTurnoverStd :: !Double
  } deriving (Eq, Show)

data WalkForwardReport = WalkForwardReport
  { wfrFoldCount :: !Int
  , wfrFolds :: ![WalkForwardFold]
  , wfrSummary :: !WalkForwardSummary
  } deriving (Eq, Show)

-- CLI

-- (moved to Trader.App.Args)

traderVersion :: String
traderVersion = showVersion Paths.version

data BuildInfo = BuildInfo
  { biVersion :: !String
  , biCommit :: !(Maybe String)
  } deriving (Eq, Show)

getBuildCommit :: IO (Maybe String)
getBuildCommit = do
  let keys = ["TRADER_GIT_COMMIT", "TRADER_COMMIT", "GIT_COMMIT", "COMMIT_SHA"]
  m <- firstJust <$> mapM lookupEnv keys
  case fmap trim m of
    Just s | not (null s) -> pure (Just s)
    _ -> pure Nothing

main :: IO ()
main = do
  let versionOption =
        infoOption
          ("trader-hs " ++ traderVersion)
          (long "version" <> short 'V' <> help "Show version")
  args <- execParser (info (opts <**> helper <**> versionOption) fullDesc)
  args' <-
    case validateArgs args of
      Left e -> die (e ++ "\n\nRun with --help for usage.")
      Right ok -> pure ok

  mWebhook <- newWebhookFromEnv
  r <- try $ do
    if argServe args'
      then runRestApi args' mWebhook
      else do
        (series, mBinanceEnv) <- loadPrices Nothing args'
        let prices = psClose series
        ensureMinPriceRows args' 2 prices

        let lookback = argLookback args'
        if argTradeOnly args'
          then runTradeOnly mWebhook args' lookback prices mBinanceEnv
          else runBacktestPipeline mWebhook args' lookback series mBinanceEnv
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
  , apHighColumn :: Maybe String
  , apLowColumn :: Maybe String
  , apBinanceSymbol :: Maybe String
  , apBotSymbols :: Maybe [String]
  , apPlatform :: Maybe String
  , apMarket :: Maybe String -- "spot" | "margin" | "futures"
  , apInterval :: Maybe String
  , apBars :: Maybe Int
  , apLookbackWindow :: Maybe String
  , apLookbackBars :: Maybe Int
  , apBinanceTestnet :: Maybe Bool
  , apBinanceApiKey :: Maybe String
  , apBinanceApiSecret :: Maybe String
  , apCoinbaseApiKey :: Maybe String
  , apCoinbaseApiSecret :: Maybe String
  , apCoinbaseApiPassphrase :: Maybe String
  , apNormalization :: Maybe String
  , apHiddenSize :: Maybe Int
  , apEpochs :: Maybe Int
  , apLr :: Maybe Double
  , apValRatio :: Maybe Double
  , apBacktestRatio :: Maybe Double
  , apTuneRatio :: Maybe Double
  , apTuneObjective :: Maybe String
  , apTunePenaltyMaxDrawdown :: Maybe Double
  , apTunePenaltyTurnover :: Maybe Double
  , apMinRoundTrips :: Maybe Int
  , apWalkForwardFolds :: Maybe Int
  , apPatience :: Maybe Int
  , apGradClip :: Maybe Double
  , apSeed :: Maybe Int
  , apKalmanDt :: Maybe Double
  , apKalmanProcessVar :: Maybe Double
  , apKalmanMeasurementVar :: Maybe Double
  , apKalmanMarketTopN :: Maybe Int
  , apThreshold :: Maybe Double
  , apOpenThreshold :: Maybe Double
  , apCloseThreshold :: Maybe Double
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
  , apStopLossVolMult :: Maybe Double
  , apTakeProfitVolMult :: Maybe Double
  , apTrailingStopVolMult :: Maybe Double
  , apMinHoldBars :: Maybe Int
  , apCooldownBars :: Maybe Int
  , apMaxHoldBars :: Maybe Int
  , apMaxDrawdown :: Maybe Double
  , apMaxDailyLoss :: Maybe Double
  , apMinEdge :: Maybe Double
  , apMinSignalToNoise :: Maybe Double
  , apCostAwareEdge :: Maybe Bool
  , apEdgeBuffer :: Maybe Double
  , apTrendLookback :: Maybe Int
  , apMaxPositionSize :: Maybe Double
  , apVolTarget :: Maybe Double
  , apVolLookback :: Maybe Int
  , apVolEwmaAlpha :: Maybe Double
  , apVolFloor :: Maybe Double
  , apVolScaleMax :: Maybe Double
  , apMaxVolatility :: Maybe Double
  , apRebalanceBars :: Maybe Int
  , apRebalanceThreshold :: Maybe Double
  , apRebalanceGlobal :: Maybe Bool
  , apFundingRate :: Maybe Double
  , apFundingBySide :: Maybe Bool
  , apRebalanceResetOnSignal :: Maybe Bool
  , apFundingOnOpen :: Maybe Bool
  , apBlendWeight :: Maybe Double
  , apTriLayer :: Maybe Bool
  , apTriLayerFastMult :: Maybe Double
  , apTriLayerSlowMult :: Maybe Double
  , apTriLayerCloudPadding :: Maybe Double
  , apTriLayerPriceAction :: Maybe Bool
  , apLstmExitFlipBars :: Maybe Int
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
  , apBotAdoptExistingPosition :: Maybe Bool
  , apKalmanZMin :: Maybe Double
  , apKalmanZMax :: Maybe Double
  , apMaxHighVolProb :: Maybe Double
  , apMaxConformalWidth :: Maybe Double
  , apMaxQuantileWidth :: Maybe Double
  , apConfirmConformal :: Maybe Bool
  , apConfirmQuantiles :: Maybe Bool
  , apConfidenceSizing :: Maybe Bool
  , apMinPositionSize :: Maybe Double
  , apTuneStressVolMult :: Maybe Double
  , apTuneStressShock :: Maybe Double
  , apTuneStressWeight :: Maybe Double
  } deriving (Eq, Show, Generic)

data OptimizerSource
  = OptimizerSourceBinance
  | OptimizerSourceCoinbase
  | OptimizerSourceKraken
  | OptimizerSourcePoloniex
  | OptimizerSourceCsv
  deriving (Eq, Show)

instance FromJSON OptimizerSource where
  parseJSON =
    AT.withText "OptimizerSource" $ \txt ->
      case T.toLower txt of
        "csv" -> pure OptimizerSourceCsv
        "binance" -> pure OptimizerSourceBinance
        "coinbase" -> pure OptimizerSourceCoinbase
        "kraken" -> pure OptimizerSourceKraken
        "poloniex" -> pure OptimizerSourcePoloniex
        _ -> fail "Invalid optimizer source (expected binance|coinbase|kraken|poloniex|csv)"

optimizerSourcePlatform :: OptimizerSource -> Maybe Platform
optimizerSourcePlatform source =
  case source of
    OptimizerSourceBinance -> Just PlatformBinance
    OptimizerSourceCoinbase -> Just PlatformCoinbase
    OptimizerSourceKraken -> Just PlatformKraken
    OptimizerSourcePoloniex -> Just PlatformPoloniex
    OptimizerSourceCsv -> Nothing

data ApiOptimizerRunRequest = ApiOptimizerRunRequest
  { arrSource :: !(Maybe OptimizerSource)
  , arrBinanceSymbol :: !(Maybe String)
  , arrData :: !(Maybe FilePath)
  , arrPriceColumn :: !(Maybe String)
  , arrHighColumn :: !(Maybe String)
  , arrLowColumn :: !(Maybe String)
  , arrIntervals :: !(Maybe String)
  , arrPlatforms :: !(Maybe String)
  , arrLookbackWindow :: !(Maybe String)
  , arrBarsMin :: !(Maybe Int)
  , arrBarsMax :: !(Maybe Int)
  , arrBarsAutoProb :: !(Maybe Double)
  , arrBarsDistribution :: !(Maybe String)
  , arrEpochsMin :: !(Maybe Int)
  , arrEpochsMax :: !(Maybe Int)
  , arrHiddenSizeMin :: !(Maybe Int)
  , arrHiddenSizeMax :: !(Maybe Int)
  , arrLrMin :: !(Maybe Double)
  , arrLrMax :: !(Maybe Double)
  , arrPatienceMax :: !(Maybe Int)
  , arrGradClipMin :: !(Maybe Double)
  , arrGradClipMax :: !(Maybe Double)
  , arrPDisableGradClip :: !(Maybe Double)
  , arrTrials :: !(Maybe Int)
  , arrTimeoutSec :: !(Maybe Double)
  , arrSeed :: !(Maybe Int)
  , arrSlippageMax :: !(Maybe Double)
  , arrSpreadMax :: !(Maybe Double)
  , arrNormalizations :: !(Maybe String)
  , arrBacktestRatio :: !(Maybe Double)
  , arrTuneRatio :: !(Maybe Double)
  , arrObjective :: !(Maybe String)
  , arrPenaltyMaxDrawdown :: !(Maybe Double)
  , arrPenaltyTurnover :: !(Maybe Double)
  , arrMinRoundTrips :: !(Maybe Int)
  , arrMinWinRate :: !(Maybe Double)
  , arrMinProfitFactor :: !(Maybe Double)
  , arrMinExposure :: !(Maybe Double)
  , arrMinSharpe :: !(Maybe Double)
  , arrMinWalkForwardSharpeMean :: !(Maybe Double)
  , arrMaxWalkForwardSharpeStd :: !(Maybe Double)
  , arrTuneObjective :: !(Maybe String)
  , arrTunePenaltyMaxDrawdown :: !(Maybe Double)
  , arrTunePenaltyTurnover :: !(Maybe Double)
  , arrTuneStressVolMult :: !(Maybe Double)
  , arrTuneStressShock :: !(Maybe Double)
  , arrTuneStressWeight :: !(Maybe Double)
  , arrTuneStressVolMultMin :: !(Maybe Double)
  , arrTuneStressVolMultMax :: !(Maybe Double)
  , arrTuneStressShockMin :: !(Maybe Double)
  , arrTuneStressShockMax :: !(Maybe Double)
  , arrTuneStressWeightMin :: !(Maybe Double)
  , arrTuneStressWeightMax :: !(Maybe Double)
  , arrWalkForwardFoldsMin :: !(Maybe Int)
  , arrWalkForwardFoldsMax :: !(Maybe Int)
  , arrMinHoldBarsMin :: !(Maybe Int)
  , arrMinHoldBarsMax :: !(Maybe Int)
  , arrCooldownBarsMin :: !(Maybe Int)
  , arrCooldownBarsMax :: !(Maybe Int)
  , arrMaxHoldBarsMin :: !(Maybe Int)
  , arrMaxHoldBarsMax :: !(Maybe Int)
  , arrMinEdgeMin :: !(Maybe Double)
  , arrMinEdgeMax :: !(Maybe Double)
  , arrMinSignalToNoiseMin :: !(Maybe Double)
  , arrMinSignalToNoiseMax :: !(Maybe Double)
  , arrEdgeBufferMin :: !(Maybe Double)
  , arrEdgeBufferMax :: !(Maybe Double)
  , arrPCostAwareEdge :: !(Maybe Double)
  , arrTrendLookbackMin :: !(Maybe Int)
  , arrTrendLookbackMax :: !(Maybe Int)
  , arrPIntrabarTakeProfitFirst :: !(Maybe Double)
  , arrPTriLayer :: !(Maybe Double)
  , arrTriLayerFastMultMin :: !(Maybe Double)
  , arrTriLayerFastMultMax :: !(Maybe Double)
  , arrTriLayerSlowMultMin :: !(Maybe Double)
  , arrTriLayerSlowMultMax :: !(Maybe Double)
  , arrPTriLayerPriceAction :: !(Maybe Double)
  , arrTriLayerCloudPaddingMin :: !(Maybe Double)
  , arrTriLayerCloudPaddingMax :: !(Maybe Double)
  , arrLstmExitFlipBarsMin :: !(Maybe Int)
  , arrLstmExitFlipBarsMax :: !(Maybe Int)
  , arrStopMin :: !(Maybe Double)
  , arrStopMax :: !(Maybe Double)
  , arrTpMin :: !(Maybe Double)
  , arrTpMax :: !(Maybe Double)
  , arrTrailMin :: !(Maybe Double)
  , arrTrailMax :: !(Maybe Double)
  , arrPDisableStop :: !(Maybe Double)
  , arrPDisableTp :: !(Maybe Double)
  , arrPDisableTrail :: !(Maybe Double)
  , arrStopVolMultMin :: !(Maybe Double)
  , arrStopVolMultMax :: !(Maybe Double)
  , arrTpVolMultMin :: !(Maybe Double)
  , arrTpVolMultMax :: !(Maybe Double)
  , arrTrailVolMultMin :: !(Maybe Double)
  , arrTrailVolMultMax :: !(Maybe Double)
  , arrPDisableStopVolMult :: !(Maybe Double)
  , arrPDisableTpVolMult :: !(Maybe Double)
  , arrPDisableTrailVolMult :: !(Maybe Double)
  , arrMinPositionSizeMin :: !(Maybe Double)
  , arrMinPositionSizeMax :: !(Maybe Double)
  , arrMaxPositionSizeMin :: !(Maybe Double)
  , arrMaxPositionSizeMax :: !(Maybe Double)
  , arrVolTargetMin :: !(Maybe Double)
  , arrVolTargetMax :: !(Maybe Double)
  , arrPDisableVolTarget :: !(Maybe Double)
  , arrVolLookbackMin :: !(Maybe Int)
  , arrVolLookbackMax :: !(Maybe Int)
  , arrVolEwmaAlphaMin :: !(Maybe Double)
  , arrVolEwmaAlphaMax :: !(Maybe Double)
  , arrPDisableVolEwmaAlpha :: !(Maybe Double)
  , arrVolFloorMin :: !(Maybe Double)
  , arrVolFloorMax :: !(Maybe Double)
  , arrVolScaleMaxMin :: !(Maybe Double)
  , arrVolScaleMaxMax :: !(Maybe Double)
  , arrMaxVolatilityMin :: !(Maybe Double)
  , arrMaxVolatilityMax :: !(Maybe Double)
  , arrPDisableMaxVolatility :: !(Maybe Double)
  , arrPeriodsPerYearMin :: !(Maybe Double)
  , arrPeriodsPerYearMax :: !(Maybe Double)
  , arrKalmanZMinMin :: !(Maybe Double)
  , arrKalmanZMinMax :: !(Maybe Double)
  , arrKalmanZMaxMin :: !(Maybe Double)
  , arrKalmanZMaxMax :: !(Maybe Double)
  , arrPDisableMaxHighVolProb :: !(Maybe Double)
  , arrMaxHighVolProbMin :: !(Maybe Double)
  , arrMaxHighVolProbMax :: !(Maybe Double)
  , arrPDisableMaxConformalWidth :: !(Maybe Double)
  , arrMaxConformalWidthMin :: !(Maybe Double)
  , arrMaxConformalWidthMax :: !(Maybe Double)
  , arrPDisableMaxQuantileWidth :: !(Maybe Double)
  , arrMaxQuantileWidthMin :: !(Maybe Double)
  , arrMaxQuantileWidthMax :: !(Maybe Double)
  , arrPConfirmConformal :: !(Maybe Double)
  , arrPConfirmQuantiles :: !(Maybe Double)
  , arrPConfidenceSizing :: !(Maybe Double)
  , arrKalmanMarketTopNMin :: !(Maybe Int)
  , arrKalmanMarketTopNMax :: !(Maybe Int)
  , arrMethodWeightBlend :: !(Maybe Double)
  , arrBlendWeightMin :: !(Maybe Double)
  , arrBlendWeightMax :: !(Maybe Double)
  , arrDisableLstmPersistence :: !(Maybe Bool)
  , arrNoSweepThreshold :: !(Maybe Bool)
  } deriving (Eq, Show, Generic)

instance FromJSON ApiOptimizerRunRequest where
  parseJSON = Aeson.genericParseJSON (jsonOptions 3)

data ApiOptimizerRunResponse = ApiOptimizerRunResponse
  { arrLastRecord :: !Aeson.Value
  , arrStdout :: !String
  , arrStderr :: !String
  } deriving (Eq, Show, Generic)

instance ToJSON ApiOptimizerRunResponse where
  toJSON = Aeson.genericToJSON (jsonOptions 3)

instance FromJSON ApiParams where
  parseJSON = Aeson.genericParseJSON (jsonOptions 2)

instance ToJSON ApiParams where
  toJSON = Aeson.genericToJSON (jsonOptions 2)

data ApiError = ApiError
  { aeError :: String
  , aeHint :: !(Maybe String)
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
      , "threshold" .= lsOpenThreshold s
      , "openThreshold" .= lsOpenThreshold s
      , "closeThreshold" .= lsCloseThreshold s
      , "kalmanNext" .= lsKalmanNext s
      , "kalmanReturn" .= lsKalmanReturn s
      , "kalmanStd" .= lsKalmanStd s
      , "kalmanZ" .= lsKalmanZ s
      , "volatility" .= lsVolatility s
      , "regimes" .= regimesJson
      , "quantiles" .= quantilesJson
      , "conformalInterval" .= conformalJson
      , "confidence" .= lsConfidence s
      , "positionSize" .= lsPositionSize s
      , "kalmanDirection" .= (if isJust (lsKalmanNext s) then dirLabel (lsKalmanDir s) else Nothing)
      , "lstmNext" .= lsLstmNext s
      , "lstmDirection" .= (if isJust (lsLstmNext s) then dirLabel (lsLstmDir s) else Nothing)
      , "chosenDirection" .= dirLabel (lsChosenDir s)
      , "closeDirection" .= dirLabel (lsCloseDir s)
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
  , abpSkipped :: !Bool
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

data ApiCoinbaseKeysStatus = ApiCoinbaseKeysStatus
  { ackHasApiKey :: !Bool
  , ackHasApiSecret :: !Bool
  , ackHasApiPassphrase :: !Bool
  , ackSigned :: !(Maybe ApiBinanceProbe)
  } deriving (Eq, Show, Generic)

instance ToJSON ApiCoinbaseKeysStatus where
  toJSON = Aeson.genericToJSON (jsonOptions 3)

data ApiListenKeyStartParams = ApiListenKeyStartParams
  { alsMarket :: !(Maybe String)
  , alsBinanceTestnet :: !(Maybe Bool)
  , alsBinanceApiKey :: !(Maybe String)
  , alsBinanceApiSecret :: !(Maybe String)
  } deriving (Eq, Show, Generic)

instance FromJSON ApiListenKeyStartParams where
  parseJSON = Aeson.genericParseJSON (jsonOptions 3)

data ApiListenKeyActionParams = ApiListenKeyActionParams
  { alaMarket :: !(Maybe String)
  , alaBinanceTestnet :: !(Maybe Bool)
  , alaBinanceApiKey :: !(Maybe String)
  , alaBinanceApiSecret :: !(Maybe String)
  , alaListenKey :: !String
  } deriving (Eq, Show, Generic)

instance FromJSON ApiListenKeyActionParams where
  parseJSON = Aeson.genericParseJSON (jsonOptions 3)

data ApiListenKeyResponse = ApiListenKeyResponse
  { alrListenKey :: !String
  , alrMarket :: !String
  , alrTestnet :: !Bool
  , alrWsUrl :: !String
  , alrKeepAliveMs :: !Int
  } deriving (Eq, Show, Generic)

instance ToJSON ApiListenKeyResponse where
  toJSON = Aeson.genericToJSON (jsonOptions 3)

data ApiBinanceTradesRequest = ApiBinanceTradesRequest
  { abrMarket :: !(Maybe String)
  , abrBinanceTestnet :: !(Maybe Bool)
  , abrBinanceApiKey :: !(Maybe String)
  , abrBinanceApiSecret :: !(Maybe String)
  , abrSymbol :: !(Maybe String)
  , abrSymbols :: !(Maybe [String])
  , abrLimit :: !(Maybe Int)
  , abrStartTimeMs :: !(Maybe Int64)
  , abrEndTimeMs :: !(Maybe Int64)
  , abrFromId :: !(Maybe Int64)
  } deriving (Eq, Show, Generic)

instance FromJSON ApiBinanceTradesRequest where
  parseJSON = Aeson.genericParseJSON (jsonOptions 3)

data ApiBinanceTradesResponse = ApiBinanceTradesResponse
  { abtrMarket :: !String
  , abtrTestnet :: !Bool
  , abtrSymbols :: ![String]
  , abtrAllSymbols :: !Bool
  , abtrTrades :: ![BinanceTrade]
  , abtrFetchedAtMs :: !Int64
  } deriving (Eq, Show, Generic)

instance ToJSON ApiBinanceTradesResponse where
  toJSON = Aeson.genericToJSON (jsonOptions 4)

data ApiBinancePositionsRequest = ApiBinancePositionsRequest
  { abpMarket :: !(Maybe String)
  , abpBinanceTestnet :: !(Maybe Bool)
  , abpBinanceApiKey :: !(Maybe String)
  , abpBinanceApiSecret :: !(Maybe String)
  , abpInterval :: !(Maybe String)
  , abpLimit :: !(Maybe Int)
  } deriving (Eq, Show, Generic)

instance FromJSON ApiBinancePositionsRequest where
  parseJSON = Aeson.genericParseJSON (jsonOptions 3)

data ApiBinancePosition = ApiBinancePosition
  { abpSymbol :: !String
  , abpPositionAmt :: !Double
  , abpEntryPrice :: !Double
  , abpMarkPrice :: !Double
  , abpUnrealizedPnl :: !Double
  , abpLiquidationPrice :: !(Maybe Double)
  , abpBreakEvenPrice :: !(Maybe Double)
  , abpLeverage :: !Double
  , abpMarginType :: !(Maybe String)
  , abpPositionSide :: !(Maybe String)
  } deriving (Eq, Show, Generic)

instance ToJSON ApiBinancePosition where
  toJSON = Aeson.genericToJSON (jsonOptions 3)

data ApiBinancePositionChart = ApiBinancePositionChart
  { abpcSymbol :: !String
  , abpcOpenTimes :: ![Int64]
  , abpcPrices :: ![Double]
  } deriving (Eq, Show, Generic)

instance ToJSON ApiBinancePositionChart where
  toJSON = Aeson.genericToJSON (jsonOptions 4)

data ApiBinancePositionsResponse = ApiBinancePositionsResponse
  { abprMarket :: !String
  , abprTestnet :: !Bool
  , abprInterval :: !String
  , abprLimit :: !Int
  , abprPositions :: ![ApiBinancePosition]
  , abprCharts :: ![ApiBinancePositionChart]
  , abprFetchedAtMs :: !Int64
  } deriving (Eq, Show, Generic)

instance ToJSON ApiBinancePositionsResponse where
  toJSON = Aeson.genericToJSON (jsonOptions 4)

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

stateDirFromEnv :: IO (Maybe FilePath)
stateDirFromEnv = do
  mDir <- lookupEnv "TRADER_STATE_DIR"
  case trim <$> mDir of
    Just dir | not (null dir) -> pure (Just dir)
    _ -> pure Nothing

stateSubdirFromEnv :: String -> IO (Maybe FilePath)
stateSubdirFromEnv subdir = do
  mState <- stateDirFromEnv
  pure (fmap (</> subdir) mState)

newJournalFromEnv :: IO (Maybe Journal)
newJournalFromEnv = do
  mDir <- lookupEnv "TRADER_JOURNAL_DIR"
  mStateDir <- stateSubdirFromEnv "journal"
  let resolvedDir =
        case trim <$> mDir of
          Just dir | null dir -> Nothing
          Just dir -> Just dir
          Nothing -> mStateDir
  case resolvedDir of
    Nothing -> pure Nothing
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

data Webhook = Webhook
  { whRequest :: !Request
  , whManager :: !Manager
  , whEvents :: !(Maybe [String])
  }

data WebhookEvent = WebhookEvent
  { weType :: !String
  , weContent :: !String
  } deriving (Eq, Show)

webhookMaxMessageLen :: Int
webhookMaxMessageLen = 300

normalizeWebhookEvent :: String -> String
normalizeWebhookEvent = map toLower . trim

parseWebhookEvents :: Maybe String -> Maybe [String]
parseWebhookEvents raw =
  case raw of
    Nothing -> Nothing
    Just s ->
      let events = filter (not . null) (map normalizeWebhookEvent (splitEnvList s))
       in Just events

newWebhookFromEnv :: IO (Maybe Webhook)
newWebhookFromEnv = do
  mUrl <- lookupEnv "TRADER_WEBHOOK_URL"
  case trim <$> mUrl of
    Just url | not (null url) -> do
      eventsEnv <- lookupEnv "TRADER_WEBHOOK_EVENTS"
      req0 <- parseRequest url
      manager <- newManager tlsManagerSettings
      let req =
            req0
              { method = "POST"
              , requestHeaders = ("Content-Type", "application/json") : requestHeaders req0
              , responseTimeout = responseTimeoutMicro (5 * 1000000)
              }
      pure (Just (Webhook req manager (parseWebhookEvents eventsEnv)))
    _ -> pure Nothing

webhookAllows :: Webhook -> String -> Bool
webhookAllows wh ev =
  case whEvents wh of
    Nothing -> True
    Just allowed -> normalizeWebhookEvent ev `elem` allowed

webhookNotifyMaybe :: Maybe Webhook -> WebhookEvent -> IO ()
webhookNotifyMaybe mWh ev =
  case mWh of
    Nothing -> pure ()
    Just wh ->
      when (webhookAllows wh (weType ev)) $ do
        _ <- forkIO (webhookSend wh ev)
        pure ()

webhookNotifyMaybeSync :: Maybe Webhook -> WebhookEvent -> IO ()
webhookNotifyMaybeSync mWh ev =
  case mWh of
    Nothing -> pure ()
    Just wh ->
      when (webhookAllows wh (weType ev)) $ webhookSend wh ev

webhookSend :: Webhook -> WebhookEvent -> IO ()
webhookSend wh ev = do
  let payload = object ["content" .= weContent ev]
      req = (whRequest wh) { requestBody = RequestBodyLBS (encode payload) }
  _ <- try (httpLbs req (whManager wh)) :: IO (Either SomeException (Response BL.ByteString))
  pure ()

sanitizeWebhookText :: Int -> String -> String
sanitizeWebhookText maxLen raw =
  let cleaned = map (\c -> if c == '\n' || c == '\r' then ' ' else c) raw
      trimmed = trim cleaned
      limit = max 0 maxLen
   in if length trimmed <= limit || limit < 4
        then trimmed
        else take (limit - 3) trimmed ++ "..."

webhookEventBotStarted :: Args -> String -> WebhookEvent
webhookEventBotStarted args sym =
  let market = marketCode (argBinanceMarket args)
      summary = printf "bot.started %s (%s) interval=%s" sym market (argInterval args)
   in WebhookEvent "bot.started" summary

webhookEventBotStartFailed :: Args -> String -> SomeException -> WebhookEvent
webhookEventBotStartFailed args sym ex =
  let market = marketCode (argBinanceMarket args)
      err = sanitizeWebhookText webhookMaxMessageLen (show ex)
      summary = printf "bot.start_failed %s (%s) error=%s" sym market err
   in WebhookEvent "bot.start_failed" summary

webhookEventBotStop :: [String] -> WebhookEvent
webhookEventBotStop syms =
  let label =
        if null syms
          then "none"
          else intercalate "," syms
   in WebhookEvent "bot.stop" ("bot.stop symbols=" ++ label)

webhookEventBotOrder :: Args -> String -> String -> Double -> ApiOrderResult -> WebhookEvent
webhookEventBotOrder args sym side price order =
  let market = marketCode (argBinanceMarket args)
      status :: String
      status = if aorSent order then "sent" else "not_sent"
      msg = sanitizeWebhookText 200 (aorMessage order)
      summary = printf "bot.order %s (%s) %s @ %.6f %s msg=%s" sym market side price status msg
   in WebhookEvent "bot.order" summary

webhookEventBotHalt :: Args -> String -> String -> Double -> Double -> Int -> WebhookEvent
webhookEventBotHalt args sym reason drawdown dailyLoss orderErrors =
  let market = marketCode (argBinanceMarket args)
      reason' = sanitizeWebhookText webhookMaxMessageLen reason
      summary =
        printf
          "bot.halt %s (%s) reason=%s drawdown=%.3f dailyLoss=%.3f orderErrors=%d"
          sym
          market
          reason'
          drawdown
          dailyLoss
          orderErrors
   in WebhookEvent "bot.halt" summary

webhookEventTradeOrder :: Args -> LatestSignal -> ApiOrderResult -> WebhookEvent
webhookEventTradeOrder args sig order =
  let sym = fromMaybe "unknown" (argBinanceSymbol args)
      platform = platformCode (argPlatform args)
      market = marketCode (argBinanceMarket args)
      action = sanitizeWebhookText 40 (lsAction sig)
      status :: String
      status = if aorSent order then "sent" else "not_sent"
      msg = sanitizeWebhookText 200 (aorMessage order)
      summary = printf "trade.order %s (%s/%s) %s %s msg=%s" sym platform market action status msg
   in WebhookEvent "trade.order" summary

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
    , apCoinbaseApiKey = Nothing
    , apCoinbaseApiSecret = Nothing
    , apCoinbaseApiPassphrase = Nothing
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
      barsPlatform = resolveBarsForPlatform args
      barsUsed =
        case (argBinanceSymbol args, argData args) of
          (Just _, _) -> barsPlatform
          (_, Just _) -> barsCsv
          _ -> fromMaybe 0 barsRaw
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
      , "highColumn" .= argHighCol args
      , "lowColumn" .= argLowCol args
      , "binanceSymbol" .= argBinanceSymbol args
      , "platform" .= platformCode (argPlatform args)
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
      , "hasCoinbaseApiKey" .= boolFromMaybe (argCoinbaseApiKey args)
      , "hasCoinbaseApiSecret" .= boolFromMaybe (argCoinbaseApiSecret args)
      , "hasCoinbaseApiPassphrase" .= boolFromMaybe (argCoinbaseApiPassphrase args)
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
      , "threshold" .= argOpenThreshold args
      , "openThreshold" .= argOpenThreshold args
      , "closeThreshold" .= argCloseThreshold args
      , "method" .= methodCode (argMethod args)
      , "positioning" .= positioningCode (argPositioning args)
      , "optimizeOperations" .= argOptimizeOperations args
      , "sweepThreshold" .= argSweepThreshold args
      , "tradeOnly" .= argTradeOnly args
      , "fee" .= argFee args
      , "stopLoss" .= argStopLoss args
      , "takeProfit" .= argTakeProfit args
      , "trailingStop" .= argTrailingStop args
      , "stopLossVolMult" .= argStopLossVolMult args
      , "takeProfitVolMult" .= argTakeProfitVolMult args
      , "trailingStopVolMult" .= argTrailingStopVolMult args
      , "minHoldBars" .= argMinHoldBars args
      , "cooldownBars" .= argCooldownBars args
      , "maxHoldBars" .= argMaxHoldBars args
      , "maxDrawdown" .= argMaxDrawdown args
      , "maxDailyLoss" .= argMaxDailyLoss args
      , "minEdge" .= argMinEdge args
      , "minSignalToNoise" .= argMinSignalToNoise args
      , "costAwareEdge" .= argCostAwareEdge args
      , "edgeBuffer" .= argEdgeBuffer args
      , "trendLookback" .= argTrendLookback args
      , "maxPositionSize" .= argMaxPositionSize args
      , "volTarget" .= argVolTarget args
      , "volLookback" .= argVolLookback args
      , "volEwmaAlpha" .= argVolEwmaAlpha args
      , "volFloor" .= argVolFloor args
      , "volScaleMax" .= argVolScaleMax args
      , "maxVolatility" .= argMaxVolatility args
      , "rebalanceBars" .= argRebalanceBars args
      , "rebalanceThreshold" .= argRebalanceThreshold args
      , "rebalanceGlobal" .= argRebalanceGlobal args
      , "fundingRate" .= argFundingRate args
      , "fundingBySide" .= argFundingBySide args
      , "rebalanceResetOnSignal" .= argRebalanceResetOnSignal args
      , "fundingOnOpen" .= argFundingOnOpen args
      , "blendWeight" .= argBlendWeight args
      , "triLayer" .= argTriLayer args
      , "triLayerFastMult" .= argTriLayerFastMult args
      , "triLayerSlowMult" .= argTriLayerSlowMult args
      , "triLayerCloudPadding" .= argTriLayerCloudPadding args
      , "triLayerPriceAction" .= argTriLayerRequirePriceAction args
      , "lstmExitFlipBars" .= argLstmExitFlipBars args
      , "tuneStressVolMult" .= argTuneStressVolMult args
      , "tuneStressShock" .= argTuneStressShock args
      , "tuneStressWeight" .= argTuneStressWeight args
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
  mStateDir <- stateSubdirFromEnv "ops"
  let resolvedDir =
        case trim <$> mDir of
          Just dir | null dir -> Nothing
          Just dir -> Just dir
          Nothing -> mStateDir
  case resolvedDir of
    Nothing -> pure Nothing
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

binanceOpsLogger :: OpsStore -> BinanceLog -> IO ()
binanceOpsLogger store entry = do
  let params =
        object
          [ "label" .= blLabel entry
          , "method" .= blMethod entry
          , "path" .= blPath entry
          , "market" .= marketCode (blMarket entry)
          , "params" .= [object ["key" .= k, "value" .= v] | (k, v) <- blParams entry]
          ]
      result =
        object
          [ "ok" .= blOk entry
          , "status" .= blStatus entry
          , "latencyMs" .= blLatencyMs entry
          , "error" .= blError entry
          ]
  opsAppendMaybe (Just store) "binance.request" (Just params) Nothing (Just result) Nothing

attachBinanceLogger :: Maybe OpsStore -> BinanceEnv -> BinanceEnv
attachBinanceLogger mOps env =
  case mOps of
    Nothing -> env
    Just store -> env { beLogger = Just (binanceOpsLogger store) }

newBinanceEnvWithOps :: Maybe OpsStore -> BinanceMarket -> String -> Maybe BS.ByteString -> Maybe BS.ByteString -> IO BinanceEnv
newBinanceEnvWithOps mOps market baseUrl apiKey apiSecret =
  attachBinanceLogger mOps <$> newBinanceEnv market baseUrl apiKey apiSecret

botStatusLogIntervalMs :: Int64
botStatusLogIntervalMs = 60000

botStatusLogMaybe :: Maybe OpsStore -> Bool -> BotState -> IO ()
botStatusLogMaybe mOps running st =
  let args = botArgs st
      settings = botSettings st
      eq =
        if V.null (botEquityCurve st)
          then Nothing
          else Just (V.last (botEquityCurve st))
      result =
        object
          [ "symbol" .= botSymbol st
          , "market" .= marketCode (argBinanceMarket args)
          , "interval" .= argInterval args
          , "running" .= running
          , "live" .= argBinanceLive args
          , "tradeEnabled" .= bsTradeEnabled settings
          , "halted" .= isJust (botHaltReason st)
          , "haltReason" .= botHaltReason st
          , "error" .= botError st
          , "updatedAtMs" .= botUpdatedAtMs st
          , "polledAtMs" .= botPolledAtMs st
          ]
   in opsAppendMaybe mOps "bot.status" Nothing (Just (argsPublicJson args)) (Just result) eq

-- Live bot (stateful; continuous loop)

data BotSettings = BotSettings
  { bsPollSeconds :: !Int
  , bsOnlineEpochs :: !Int
  , bsTrainBars :: !Int
  , bsMaxPoints :: !Int
  , bsTradeEnabled :: !Bool
  , bsAdoptExistingPosition :: !Bool
  } deriving (Eq, Show)

type BotRuntimeMap = HM.HashMap String BotRuntimeState

data BotController = BotController
  { bcRuntime :: MVar BotRuntimeMap
  }

data BotStartRuntime = BotStartRuntime
  { bsrThreadId :: !ThreadId
  , bsrStopSignal :: !(MVar ())
  , bsrArgs :: !Args
  , bsrSettings :: !BotSettings
  , bsrSymbol :: !String
  , bsrRequestedAtMs :: !Int64
  , bsrStartReason :: !String
  }

data BotStartOutcome = BotStartOutcome
  { bsoSymbol :: !String
  , bsoState :: !BotRuntimeState
  , bsoStarted :: !Bool
  }

data BotRuntimeState
  = BotStarting !BotStartRuntime
  | BotRunning !BotRuntime

data BotRuntime = BotRuntime
  { brThreadId :: ThreadId
  , brStateVar :: MVar BotState
  , brStopSignal :: MVar ()
  , brOptimizer :: Maybe BotOptimizerRuntime
  }

data BotOptimizerRuntime = BotOptimizerRuntime
  { borThreadId :: !ThreadId
  , borStopSignal :: !(MVar ())
  , borPendingUpdate :: !(MVar (Maybe BotOptimizerUpdate))
  }

data BotOptimizerUpdate = BotOptimizerUpdate
  { bouArgs :: !Args
  , bouLookback :: !Int
  , bouLstmCtx :: !(Maybe LstmCtx)
  , bouKalmanCtx :: !(Maybe KalmanCtx)
  , bouFinalEquity :: !(Maybe Double)
  , bouObjective :: !(Maybe String)
  , bouScore :: !(Maybe Double)
  }

data TopCombosExport = TopCombosExport
  { tceCombos :: ![TopCombo]
  }

data TopCombo = TopCombo
  { tcRank :: !(Maybe Int)
  , tcFinalEquity :: !(Maybe Double)
  , tcObjectiveLabel :: !(Maybe String)
  , tcScore :: !(Maybe Double)
  , tcOpenThreshold :: !(Maybe Double)
  , tcCloseThreshold :: !(Maybe Double)
  , tcParams :: !Aeson.Object
  }

instance FromJSON TopCombosExport where
  parseJSON = AT.withObject "TopCombosExport" $ \o -> TopCombosExport <$> o Aeson..: "combos"

instance FromJSON TopCombo where
  parseJSON = AT.withObject "TopCombo" $ \o -> do
    params <- o Aeson..: "params"
    TopCombo
      <$> o Aeson..:? "rank"
      <*> o Aeson..:? "finalEquity"
      <*> o Aeson..:? "objective"
      <*> o Aeson..:? "score"
      <*> o Aeson..:? "openThreshold"
      <*> o Aeson..:? "closeThreshold"
      <*> pure params

data BotOpenTrade = BotOpenTrade
  { botOpenEntryIndex :: !Int
  , botOpenEntryEquity :: !Double
  , botOpenHoldingPeriods :: !Int
  , botOpenEntryPrice :: !Double
  , botOpenTrail :: !Double
  , botOpenSide :: !PositionSide
  } deriving (Eq, Show)

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
  , botOpenTrade :: !(Maybe BotOpenTrade)
  , botCooldownLeft :: !Int
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
  , botPolledAtMs :: !Int64
  , botPollLatencyMs :: !Int
  , botFetchedKlines :: !Int
  , botFetchedLastKline :: !(Maybe Kline)
  , botLastBatchAtMs :: !Int64
  , botLastBatchSize :: !Int
  , botLastBatchMs :: !Int
  , botError :: !(Maybe String)
  }

newBotController :: IO BotController
newBotController = BotController <$> newMVar HM.empty

clampInt :: Int -> Int -> Int -> Int
clampInt lo hi n = max lo (min hi n)

defaultBotPollSeconds :: Args -> Int
defaultBotPollSeconds args =
  case parseIntervalSeconds (argInterval args) of
    Nothing -> 10
    Just sec ->
      let half = max 1 (sec `div` 2)
       in clampInt 5 60 half

defaultBotSettings :: Args -> BotSettings
defaultBotSettings args =
  BotSettings
    { bsPollSeconds = defaultBotPollSeconds args
    , bsOnlineEpochs = 1
    , bsTrainBars = 800
    , bsMaxPoints = 2000
    , bsTradeEnabled = True
    , bsAdoptExistingPosition = True
    }

botSettingsFromApi :: Args -> ApiParams -> Either String BotSettings
botSettingsFromApi args p = do
  let poll = maybe (defaultBotPollSeconds args) id (apBotPollSeconds p)
      onlineEpochs = maybe 1 id (apBotOnlineEpochs p)
      trainBars = maybe 800 id (apBotTrainBars p)
      maxPoints = maybe 2000 id (apBotMaxPoints p)
      tradeEnabled = maybe True id (apBotTrade p)
      adoptExistingPosition = True

  ensure "botPollSeconds must be between 1 and 3600" (poll >= 1 && poll <= 3600)
  ensure "botOnlineEpochs must be between 0 and 50" (onlineEpochs >= 0 && onlineEpochs <= 50)
  ensure "botTrainBars must be >= 10" (trainBars >= 10)
  ensure "botMaxPoints must be between 100 and 100000" (maxPoints >= 100 && maxPoints <= 100000)

  pure
    BotSettings
      { bsPollSeconds = poll
      , bsOnlineEpochs = onlineEpochs
      , bsTrainBars = trainBars
      , bsMaxPoints = maxPoints
      , bsTradeEnabled = tradeEnabled
      , bsAdoptExistingPosition = adoptExistingPosition
      }
  where
    ensure :: String -> Bool -> Either String ()
    ensure msg cond = if cond then Right () else Left msg

botStatusJson :: BotState -> Aeson.Value
botStatusJson st =
  let finiteMaybe x =
        if isNaN x || isInfinite x
          then Nothing
          else Just x

      klineJson k =
        object
          [ "openTime" .= kOpenTime k
          , "open" .= kOpen k
          , "high" .= kHigh k
          , "low" .= kLow k
          , "close" .= kClose k
          ]
    in
  object $
    [ "running" .= True
    , "symbol" .= botSymbol st
    , "interval" .= argInterval (botArgs st)
    , "market" .= marketCode (argBinanceMarket (botArgs st))
    , "method" .= methodCode (argMethod (botArgs st))
    , "threshold" .= argOpenThreshold (botArgs st)
    , "openThreshold" .= argOpenThreshold (botArgs st)
    , "closeThreshold" .= argCloseThreshold (botArgs st)
    , "settings"
        .= object
          [ "pollSeconds" .= bsPollSeconds (botSettings st)
          , "onlineEpochs" .= bsOnlineEpochs (botSettings st)
          , "trainBars" .= bsTrainBars (botSettings st)
          , "maxPoints" .= bsMaxPoints (botSettings st)
          , "tradeEnabled" .= bsTradeEnabled (botSettings st)
          , "adoptExistingPosition" .= bsAdoptExistingPosition (botSettings st)
          ]
    , "startIndex" .= botStartIndex st
    , "startedAtMs" .= botStartedAtMs st
    , "updatedAtMs" .= botUpdatedAtMs st
    , "polledAtMs" .= botPolledAtMs st
    , "pollLatencyMs" .= botPollLatencyMs st
    , "fetchedKlines" .= botFetchedKlines st
    , "lastBatchAtMs" .= botLastBatchAtMs st
    , "lastBatchSize" .= botLastBatchSize st
    , "lastBatchMs" .= botLastBatchMs st
    , "halted" .= isJust (botHaltReason st)
    , "peakEquity" .= botPeakEquity st
    , "dayStartEquity" .= botDayStartEquity st
    , "consecutiveOrderErrors" .= botConsecutiveOrderErrors st
    , "cooldownLeft" .= botCooldownLeft st
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
      ++ maybe [] (\k -> ["fetchedLastKline" .= klineJson k]) (botFetchedLastKline st)
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
            openTradeShifted =
              case botOpenTrade st of
                Nothing -> Nothing
                Just ot -> Just (shiftOpenTradeIndex dropCount ot)
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
            , botOpenTrade = openTradeShifted
            , botStartIndex = botStartIndex st + dropCount
            }

shiftOpenTradeIndex :: Int -> BotOpenTrade -> BotOpenTrade
shiftOpenTradeIndex dropCount ot =
  ot { botOpenEntryIndex = max 0 (botOpenEntryIndex ot - dropCount) }

botStartingJson :: BotStartRuntime -> Aeson.Value
botStartingJson rt =
  object
    [ "running" .= False
    , "starting" .= True
    , "startingReason" .= bsrStartReason rt
    , "symbol" .= bsrSymbol rt
    , "interval" .= argInterval (bsrArgs rt)
    , "market" .= marketCode (argBinanceMarket (bsrArgs rt))
    , "method" .= methodCode (argMethod (bsrArgs rt))
    , "threshold" .= argOpenThreshold (bsrArgs rt)
    , "openThreshold" .= argOpenThreshold (bsrArgs rt)
    , "closeThreshold" .= argCloseThreshold (bsrArgs rt)
    , "startedAtMs" .= bsrRequestedAtMs rt
    ]

botStartReasonFromAdopt :: AdoptRequirement -> String
botStartReasonFromAdopt req =
  if arActive req
    then "Waiting for a compatible top combo to adopt existing positions or orders."
    else "Initializing model."

botStoppedJson :: Aeson.Value
botStoppedJson =
  object
    [ "running" .= False
    ]

botStatusJsonMulti :: [Aeson.Value] -> Aeson.Value
botStatusJsonMulti bots =
  let running = any isRunning bots
      starting = any isStarting bots
   in
    object
      [ "running" .= running
      , "starting" .= starting
      , "multi" .= True
      , "bots" .= bots
      ]
  where
    isRunning v =
      case v of
        Aeson.Object o -> KM.lookup "running" o == Just (Aeson.Bool True)
        _ -> False
    isStarting v =
      case v of
        Aeson.Object o -> KM.lookup "starting" o == Just (Aeson.Bool True)
        _ -> False

data BotStatusSnapshot = BotStatusSnapshot
  { bssSavedAtMs :: !Int64
  , bssStatus :: !Aeson.Value
  } deriving (Eq, Show, Generic)

instance ToJSON BotStatusSnapshot where
  toJSON = Aeson.genericToJSON (jsonOptions 3)

instance FromJSON BotStatusSnapshot where
  parseJSON = Aeson.genericParseJSON (jsonOptions 3)

botStoppedSnapshotJson :: BotStatusSnapshot -> Aeson.Value
botStoppedSnapshotJson snap =
  object
    [ "running" .= False
    , "snapshotAtMs" .= bssSavedAtMs snap
    , "snapshot" .= bssStatus snap
    ]

botStoppedMultiJson :: [BotStatusSnapshot] -> Aeson.Value
botStoppedMultiJson snaps =
  let bots = map botStoppedSnapshotJson snaps
      snapshotAt =
        case snaps of
          [] -> Nothing
          _ -> Just (maximum (map bssSavedAtMs snaps))
   in
    object
      ( [ "running" .= False
        , "multi" .= True
        , "bots" .= bots
        ]
          ++ maybe [] (\t -> ["snapshotAtMs" .= t]) snapshotAt
      )

botSnapshotSymbol :: BotStatusSnapshot -> Maybe String
botSnapshotSymbol snap =
  case bssStatus snap of
    Aeson.Object o -> KM.lookup "symbol" o >>= AT.parseMaybe parseJSON
    _ -> Nothing

botStateFileName :: FilePath
botStateFileName = "bot-state.json"

botStateFilePrefix :: FilePath
botStateFilePrefix = "bot-state-"

defaultBotStateDir :: FilePath
defaultBotStateDir = ".tmp/bot"

resolveBotStateDir :: IO (Maybe FilePath)
resolveBotStateDir = do
  mDir <- lookupEnv "TRADER_BOT_STATE_DIR"
  case trim <$> mDir of
    Just dir | null dir -> pure Nothing
    Just dir -> pure (Just dir)
    Nothing -> do
      mStateDir <- stateSubdirFromEnv "bot"
      let baseDir = fromMaybe defaultBotStateDir mStateDir
      pure (Just baseDir)

botStatePathFor :: FilePath -> String -> FilePath
botStatePathFor dir sym =
  dir </> (botStateFilePrefix ++ sanitizeFileComponent sym ++ ".json")

s3BotSnapshotKey :: S3State -> String -> String
s3BotSnapshotKey st sym =
  s3KeyFor st ["bot", botStateFilePrefix ++ sanitizeFileComponent sym ++ ".json"]

s3BotSnapshotIndexKey :: S3State -> String
s3BotSnapshotIndexKey st =
  s3KeyFor st ["bot", "bot-state-index.json"]

writeBotStatusSnapshot :: FilePath -> BotStatusSnapshot -> IO ()
writeBotStatusSnapshot path snap = do
  createDirectoryIfMissing True (takeDirectory path)
  randId <- randomIO :: IO Word64
  let tmpPath = path ++ ".tmp-" ++ show randId
  BL.writeFile tmpPath (encode snap)
  -- Atomic on POSIX when within the same filesystem; on Windows, fall back to replace.
  r1 <- try (renameFile tmpPath path) :: IO (Either SomeException ())
  case r1 of
    Right _ -> pure ()
    Left _ -> do
      _ <- try (removeFile path) :: IO (Either SomeException ())
      _ <- try (renameFile tmpPath path) :: IO (Either SomeException ())
      pure ()

writeBotStatusSnapshotMaybe :: Maybe FilePath -> String -> BotStatusSnapshot -> IO ()
writeBotStatusSnapshotMaybe mDir sym snap = do
  case mDir of
    Nothing -> pure ()
    Just dir -> do
      let path = botStatePathFor dir sym
      _ <- try (writeBotStatusSnapshot path snap) :: IO (Either SomeException ())
      pure ()
  mS3 <- resolveS3State
  case mS3 of
    Nothing -> pure ()
    Just st -> do
      let key = s3BotSnapshotKey st sym
      _ <- try (s3PutObject st key (encode snap)) :: IO (Either SomeException (Either String ()))
      _ <- try (updateBotSnapshotIndex st sym) :: IO (Either SomeException ())
      pure ()

persistBotStatusMaybe :: Maybe FilePath -> BotState -> IO ()
persistBotStatusMaybe mDir st =
  case mDir of
    Nothing -> pure ()
    Just _ -> do
      now <- getTimestampMs
      let snap =
            BotStatusSnapshot
              { bssSavedAtMs = now
              , bssStatus = botStatusJson st
              }
      writeBotStatusSnapshotMaybe mDir (botSymbol st) snap

readBotStatusSnapshotAtPath :: FilePath -> IO (Maybe BotStatusSnapshot)
readBotStatusSnapshotAtPath path = do
  exists <- doesFileExist path
  if not exists
    then pure Nothing
    else do
      contentsOrErr <- (try (BL.readFile path) :: IO (Either SomeException BL.ByteString))
      case contentsOrErr of
        Left _ -> pure Nothing
        Right contents ->
          case Aeson.eitherDecode' contents of
            Left _ -> pure Nothing
            Right snap -> pure (Just snap)

readBotStatusSnapshotMaybe :: Maybe FilePath -> String -> IO (Maybe BotStatusSnapshot)
readBotStatusSnapshotMaybe mDir sym = do
  localSnap <-
    case mDir of
      Nothing -> pure Nothing
      Just dir -> do
        let path = botStatePathFor dir sym
        readBotStatusSnapshotAtPath path
  case localSnap of
    Just snap -> pure (Just snap)
    Nothing -> do
      mS3 <- resolveS3State
      s3Snap <-
        case mS3 of
          Nothing -> pure Nothing
          Just st -> readBotStatusSnapshotS3 st sym
      case (s3Snap, mDir) of
        (Just snap, Just dir) -> do
          let path = botStatePathFor dir sym
          _ <- try (writeBotStatusSnapshot path snap) :: IO (Either SomeException ())
          pure (Just snap)
        (Just snap, _) -> pure (Just snap)
        (Nothing, _) -> pure Nothing

readBotStatusSnapshotsMaybe :: Maybe FilePath -> IO [BotStatusSnapshot]
readBotStatusSnapshotsMaybe mDir = do
  localSnaps <-
    case mDir of
      Nothing -> pure []
      Just dir -> do
        exists <- doesFileExist (dir </> botStateFileName)
        legacySnap <-
          if exists
            then readBotStatusSnapshotAtPath (dir </> botStateFileName)
            else pure Nothing
        dirExists <- doesDirectoryExist dir
        if not dirExists
          then pure (maybeToList legacySnap)
          else do
            entries <- listDirectory dir
            let targets =
                  [ dir </> name
                  | name <- entries
                  , botStateFilePrefix `isPrefixOf` name
                  , ".json" `isSuffixOf` name
                  ]
            snaps <- mapM readBotStatusSnapshotAtPath targets
            pure (maybeToList legacySnap ++ mapMaybe id snaps)
  if not (null localSnaps)
    then pure localSnaps
    else do
      mS3 <- resolveS3State
      s3Snaps <-
        case mS3 of
          Nothing -> pure []
          Just st -> do
            symbols <- readBotSnapshotIndex st
            snaps <- mapM (readBotStatusSnapshotS3 st) symbols
            pure (mapMaybe id snaps)
      case mDir of
        Nothing -> pure ()
        Just dir ->
          forM_ s3Snaps $ \snap ->
            case botSnapshotSymbol snap of
              Nothing -> pure ()
              Just sym -> do
                let path = botStatePathFor dir sym
                _ <- try (writeBotStatusSnapshot path snap) :: IO (Either SomeException ())
                pure ()
      pure (mergeBotSnapshots s3Snaps)

readBotSnapshotIndex :: S3State -> IO [String]
readBotSnapshotIndex st = do
  let key = s3BotSnapshotIndexKey st
  r <- s3GetObject st key
  case r of
    Left _ -> pure []
    Right Nothing -> pure []
    Right (Just bytes) ->
      case Aeson.eitherDecode' bytes of
        Left _ -> pure []
        Right symbols -> pure (filter (not . null) (map normalizeSymbol symbols))

updateBotSnapshotIndex :: S3State -> String -> IO ()
updateBotSnapshotIndex st sym = do
  symbols <- readBotSnapshotIndex st
  let symNorm = normalizeSymbol sym
      next = if symNorm `elem` symbols then symbols else symbols ++ [symNorm]
  _ <- s3PutObject st (s3BotSnapshotIndexKey st) (encode next)
  pure ()

readBotStatusSnapshotS3 :: S3State -> String -> IO (Maybe BotStatusSnapshot)
readBotStatusSnapshotS3 st sym = do
  let key = s3BotSnapshotKey st sym
  r <- s3GetObject st key
  case r of
    Left _ -> pure Nothing
    Right Nothing -> pure Nothing
    Right (Just bytes) ->
      case Aeson.eitherDecode' bytes of
        Left _ -> pure Nothing
        Right snap -> pure (Just snap)

mergeBotSnapshots :: [BotStatusSnapshot] -> [BotStatusSnapshot]
mergeBotSnapshots snaps =
  HM.elems (foldl' insertSnap HM.empty snaps)
  where
    insertSnap acc snap =
      case snapshotSymbol snap of
        Nothing -> acc
        Just sym ->
          case HM.lookup sym acc of
            Nothing -> HM.insert sym snap acc
            Just existing ->
              if bssSavedAtMs snap >= bssSavedAtMs existing
                then HM.insert sym snap acc
                else acc

    snapshotSymbol :: BotStatusSnapshot -> Maybe String
    snapshotSymbol snap =
      case bssStatus snap of
        Aeson.Object o -> KM.lookup "symbol" o >>= AT.parseMaybe parseJSON
        _ -> Nothing

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

ensureBinanceKeysPresent :: BinanceEnv -> IO ()
ensureBinanceKeysPresent env =
  let missingKey = maybe True BS.null (beApiKey env)
      missingSecret = maybe True BS.null (beApiSecret env)
   in
    if missingKey || missingSecret
      then throwIO (userError "botTrade=true requires BINANCE_API_KEY and BINANCE_API_SECRET (or pass binanceApiKey/binanceApiSecret in request)")
      else pure ()

fetchLongFlatAccountPos :: Args -> BinanceEnv -> String -> IO Int
fetchLongFlatAccountPos args env sym =
  case argBinanceMarket args of
    MarketFutures -> do
      summary <- fetchFuturesPositionSummary env sym
      let netAmt = fpsNetAmt summary
          hasLong = fpsHasLong summary
          hasShort = fpsHasShort summary
          label =
            if hasLong && hasShort
              then "both long and short futures positions"
              else "a short futures position"
      if hasShort
        then
          throwIO
            ( userError
                ( "bot/start positioning=long-flat cannot start with "
                    ++ label
                    ++ " for "
                    ++ sym
                    ++ " (net positionAmt="
                    ++ show netAmt
                    ++ ")"
                )
            )
        else pure (if hasLong then 1 else 0)
    _ -> do
      let (baseAsset, _quoteAsset) = splitSymbol sym
      mSf <- tryFetchFilters
      baseBal <- fetchFreeBalance env baseAsset
      pure (if isLongSpotBalance mSf baseBal then 1 else 0)
  where
    tryFetchFilters :: IO (Maybe SymbolFilters)
    tryFetchFilters = do
      r <- try (fetchSymbolFilters env sym) :: IO (Either SomeException SymbolFilters)
      pure $
        case r of
          Left _ -> Nothing
          Right sf -> Just sf

    effectiveMinQty sf = sfMarketMinQty sf <|> sfLotMinQty sf

    isLongSpotBalance :: Maybe SymbolFilters -> Double -> Bool
    isLongSpotBalance mSf baseBal =
      case mSf >>= effectiveMinQty of
        Nothing -> baseBal > 0
        Just minQ -> baseBal >= minQ

accountPosSign :: Double -> Int
accountPosSign posAmt
  | posAmt > 1e-12 = 1
  | posAmt < -1e-12 = -1
  | otherwise = 0

fetchLongShortAccountPos :: Args -> BinanceEnv -> String -> IO Int
fetchLongShortAccountPos args env sym =
  case argBinanceMarket args of
    MarketFutures -> do
      summary <- fetchFuturesPositionSummary env sym
      let hasLong = fpsHasLong summary
          hasShort = fpsHasShort summary
          netAmt = fpsNetAmt summary
      if hasLong && hasShort
        then
          throwIO
            ( userError
                ( "bot/start positioning=long-short cannot start with both long and short futures positions for "
                    ++ sym
                    ++ "."
                )
            )
        else pure (accountPosSign netAmt)
    _ -> throwIO (userError "bot/start positioning=long-short requires market=futures")

fetchBotAccountPos :: Args -> BinanceEnv -> String -> IO Int
fetchBotAccountPos args env sym =
  case argPositioning args of
    LongFlat -> fetchLongFlatAccountPos args env sym
    LongShort -> fetchLongShortAccountPos args env sym

fetchAdoptableAccountPos :: Args -> BinanceEnv -> String -> IO Int
fetchAdoptableAccountPos args env sym =
  case argBinanceMarket args of
    MarketFutures -> do
      summary <- fetchFuturesPositionSummary env sym
      let hasLong = fpsHasLong summary
          hasShort = fpsHasShort summary
          netAmt = fpsNetAmt summary
      if hasLong && hasShort
        then
          throwIO
            ( userError
                ( "bot/start cannot adopt both long and short futures positions for "
                    ++ sym
                    ++ ". Flatten one side first."
                )
            )
        else pure (accountPosSign netAmt)
    _ -> fetchLongFlatAccountPos args env sym

preflightBotStart :: Maybe OpsStore -> Args -> BotSettings -> String -> IO (Either String ())
preflightBotStart mOps args settings sym =
  if not (bsTradeEnabled settings)
    then pure (Right ())
    else do
      r <- try $ do
        env <- makeBinanceEnv mOps args
        ensureBinanceKeysPresent env
        pos <- fetchBotAccountPos args env sym
        if pos /= 0 && not (bsAdoptExistingPosition settings)
          then
            let sideLabel = if pos > 0 then "long" else "short"
             in
              pure
                ( Left
                    ( "Refusing to start: existing "
                        ++ sideLabel
                        ++ " position detected for "
                        ++ sym
                        ++ " ("
                        ++ marketCode (argBinanceMarket args)
                        ++ "). Flatten first, or set botAdoptExistingPosition=true."
                    )
                )
          else pure (Right ())
      case r of
        Left ex -> pure (Left (snd (exceptionToHttp ex)))
        Right out -> pure out

resolveBotSymbols :: Args -> ApiParams -> IO (Either String [String])
resolveBotSymbols args p = do
  envRaw <- lookupEnv "TRADER_BOT_SYMBOLS"
  let symbolsReq = maybe [] (map trim) (apBotSymbols p)
      symbolsEnv = maybe [] splitEnvList envRaw
      symbolsFallback = maybeToList (argBinanceSymbol args)
      raw =
        case symbolsReq of
          [] ->
            case symbolsEnv of
              [] -> symbolsFallback
              xs -> xs
          xs -> xs
      normalized = filter (not . null) (dedupeStable (map normalizeSymbol raw))
  if null normalized
    then pure (Left "bot/start requires binanceSymbol or botSymbols")
    else pure (Right normalized)

resolveBotSymbolsAuto :: Args -> IO (Either String [String])
resolveBotSymbolsAuto args = do
  envRaw <- lookupEnv "TRADER_BOT_SYMBOLS"
  let symbolsEnv = maybe [] splitEnvList envRaw
      symbolsFallback = maybeToList (argBinanceSymbol args)
      raw =
        case symbolsEnv of
          [] -> symbolsFallback
          xs -> xs
      normalized = filter (not . null) (dedupeStable (map normalizeSymbol raw))
  if null normalized
    then pure (Left "bot autostart requires binanceSymbol or TRADER_BOT_SYMBOLS")
    else pure (Right normalized)

comboPollSecondsFromEnv :: IO Int
comboPollSecondsFromEnv = do
  pollEnv <- lookupEnv "TRADER_BOT_COMBOS_POLL_SEC"
  pure $
    case pollEnv >>= readMaybe of
      Just n | n >= 5 -> n
      _ -> 30

data AdoptRequirement = AdoptRequirement
  { arActive :: !Bool
  , arRequiresLongShort :: !Bool
  } deriving (Eq, Show)

openOrdersRequireShort :: BinanceMarket -> [BinanceOpenOrder] -> Bool
openOrdersRequireShort market orders =
  case market of
    MarketFutures -> any requiresShort orders
    _ -> False
  where
    requiresShort order =
      case fmap normalizeKey (booPositionSide order) of
        Just "short" -> True
        Just "long" -> False
        _ ->
          case (booReduceOnly order, booClosePosition order) of
            (Just True, _) -> False
            (_, Just True) -> False
            _ ->
              case booSide order of
                Just Sell -> True
                _ -> False

positionRequiresLongShort :: FuturesPositionRisk -> Bool
positionRequiresLongShort pos =
  case fmap normalizeKey (fprPositionSide pos) of
    Just "short" -> True
    Just "long" -> False
    _ -> accountPosSign (fprPositionAmt pos) < 0

data FuturesPositionSummary = FuturesPositionSummary
  { fpsNetAmt :: !Double
  , fpsHasLong :: !Bool
  , fpsHasShort :: !Bool
  } deriving (Eq, Show)

positionAmtSigned :: FuturesPositionRisk -> Double
positionAmtSigned pos =
  case fmap normalizeKey (fprPositionSide pos) of
    Just "short" -> negate (abs (fprPositionAmt pos))
    Just "long" -> abs (fprPositionAmt pos)
    _ -> fprPositionAmt pos

futuresPositionSummary :: String -> [FuturesPositionRisk] -> FuturesPositionSummary
futuresPositionSummary sym positions =
  let symNorm = normalizeSymbol sym
      matches = filter (\p -> normalizeSymbol (fprSymbol p) == symNorm) positions
      signed = map positionAmtSigned matches
      eps = 1e-12
      hasLong = any (> eps) signed
      hasShort = any (< -eps) signed
      netAmt = sum signed
   in FuturesPositionSummary { fpsNetAmt = netAmt, fpsHasLong = hasLong, fpsHasShort = hasShort }

fetchFuturesPositionSummary :: BinanceEnv -> String -> IO FuturesPositionSummary
fetchFuturesPositionSummary env sym = do
  positions <- fetchFuturesPositionRisks env
  pure (futuresPositionSummary sym positions)

argsCompatibleWithAdoption :: Args -> AdoptRequirement -> Bool
argsCompatibleWithAdoption args req
  | not (arActive req) = True
  | otherwise =
      case argBinanceMarket args of
        MarketFutures ->
          if arRequiresLongShort req
            then argPositioning args == LongShort
            else True
        _ ->
          not (arRequiresLongShort req) && argPositioning args /= LongShort

selectCompatibleTopComboArgs :: ApiComputeLimits -> String -> Args -> AdoptRequirement -> TopCombosExport -> Maybe Args
selectCompatibleTopComboArgs limits sym args req export =
  let combos = filter (topComboMatchesSymbol sym Nothing) (tceCombos export)
      sortedCombos = sortOn topComboRankKey combos
      pick [] = Nothing
      pick (combo : rest) =
        case applyTopComboForStart args combo of
          Left _ -> pick rest
          Right args' ->
            if argsCompatibleWithAdoption args' req
              then
                case validateApiComputeLimits limits args' of
                  Left _ -> pick rest
                  Right args'' -> Just args''
              else pick rest
   in pick sortedCombos

applyLatestTopCombo :: FilePath -> ApiComputeLimits -> String -> Args -> AdoptRequirement -> IO Args
applyLatestTopCombo optimizerTmp limits sym args req = do
  topJsonPath <- resolveOptimizerCombosPath optimizerTmp
  if arActive req
    then do
      pollSec <- comboPollSecondsFromEnv
      let sleepSec s = threadDelay (max 1 s * 1000000)
          loop = do
            combosOrErr <- readTopCombosExport topJsonPath
            case combosOrErr of
              Left _ -> sleepSec pollSec >> loop
              Right export ->
                case selectCompatibleTopComboArgs limits sym args req export of
                  Nothing -> sleepSec pollSec >> loop
                  Just args' -> pure args'
      loop
    else do
      combosOrErr <- readTopCombosExport topJsonPath
      case combosOrErr of
        Left _ -> pure args
        Right export ->
          case bestTopComboForSymbol sym Nothing export of
            Nothing -> pure args
            Just combo ->
              case applyTopComboForStart args combo of
                Left _ -> pure args
                Right args' -> pure args'

resolveOrphanOpenPositionSymbols :: Maybe OpsStore -> ApiComputeLimits -> FilePath -> Args -> [String] -> IO [String]
resolveOrphanOpenPositionSymbols mOps limits optimizerTmp args requested =
  if not (platformSupportsLiveBot (argPlatform args)) || argBinanceMarket args /= MarketFutures
    then pure []
    else do
      topJsonPath <- resolveOptimizerCombosPath optimizerTmp
      combosOrErr <- readTopCombosExport topJsonPath
      case combosOrErr of
        Left _ -> pure []
        Right export -> do
          positionsOrErr <-
            ( try $ do
                env <- makeBinanceEnv mOps args
                ensureBinanceKeysPresent env
                fetchFuturesPositionRisks env
            ) ::
              IO (Either SomeException [FuturesPositionRisk])
          case positionsOrErr of
            Left _ -> pure []
            Right positions -> do
              let openPositions = filter (\p -> accountPosSign (fprPositionAmt p) /= 0) positions
                  requestedNorm = map normalizeSymbol requested
                  isRequested sym = normalizeSymbol sym `elem` requestedNorm
                  openBySymbol =
                    foldl'
                      ( \acc pos ->
                          let sym = normalizeSymbol (fprSymbol pos)
                              requiresShort = positionRequiresLongShort pos
                           in HM.insertWith (||) sym requiresShort acc
                      )
                      HM.empty
                      openPositions
                  hasCombo sym requiresShort =
                    let adoptReq = AdoptRequirement True requiresShort
                        argsSym = args { argBinanceSymbol = Just sym }
                     in isJust (selectCompatibleTopComboArgs limits sym argsSym adoptReq export)
                  orphans =
                    [ sym
                    | (sym, requiresShort) <- sortOn fst (HM.toList openBySymbol)
                    , not (isRequested sym)
                    , let summary = futuresPositionSummary sym openPositions
                    , not (fpsHasLong summary && fpsHasShort summary)
                    , hasCombo sym requiresShort
                    ]
              pure (dedupeStable orphans)

resolveAdoptionRequirement :: Maybe OpsStore -> Args -> String -> IO (Either String AdoptRequirement)
resolveAdoptionRequirement mOps args sym = do
  r <- try $ do
    env <- makeBinanceEnv mOps args
    resolveAdoptionRequirementWithEnv args env sym
  pure $
    case r of
      Left ex -> Left (snd (exceptionToHttp ex))
      Right reqOrErr -> reqOrErr

resolveAdoptionRequirementWithEnv :: Args -> BinanceEnv -> String -> IO (Either String AdoptRequirement)
resolveAdoptionRequirementWithEnv args env sym = do
  r <- try $ do
    ensureBinanceKeysPresent env
    pos <- fetchAdoptableAccountPos args env sym
    orders <- fetchOpenOrders env sym
    let requiresShort = pos < 0 || openOrdersRequireShort (argBinanceMarket args) orders
        active = pos /= 0 || not (null orders)
    pure AdoptRequirement { arActive = active, arRequiresLongShort = requiresShort }
  pure $
    case r of
      Left ex -> Left (snd (exceptionToHttp ex))
      Right req -> Right req

adoptRequirementLongShortOverride :: Args -> AdoptRequirement -> Maybe Double
adoptRequirementLongShortOverride args req
  | not (arActive req) = Nothing
  | argBinanceMarket args == MarketFutures =
      if arRequiresLongShort req
        then Just 1.0
        else Nothing
  | otherwise = Just 0.0

botStartSymbol ::
  Bool ->
  Maybe OpsStore ->
  Metrics ->
  Maybe Journal ->
  Maybe Webhook ->
  Maybe FilePath ->
  FilePath ->
  ApiComputeLimits ->
  AdoptRequirement ->
  BotController ->
  Args ->
  ApiParams ->
  String ->
  IO (Either String BotStartOutcome)
botStartSymbol allowExisting mOps metrics mJournal mWebhook mBotStateDir optimizerTmp limits adoptReq ctrl args p symRaw =
  case botSettingsFromApi args p of
    Left e -> pure (Left e)
    Right settings ->
      botStartSymbolWithSettings
        allowExisting
        mOps
        metrics
        mJournal
        mWebhook
        mBotStateDir
        optimizerTmp
        limits
        adoptReq
        ctrl
        args
        settings
        symRaw

botStartSymbolWithSettings ::
  Bool ->
  Maybe OpsStore ->
  Metrics ->
  Maybe Journal ->
  Maybe Webhook ->
  Maybe FilePath ->
  FilePath ->
  ApiComputeLimits ->
  AdoptRequirement ->
  BotController ->
  Args ->
  BotSettings ->
  String ->
  IO (Either String BotStartOutcome)
botStartSymbolWithSettings allowExisting mOps metrics mJournal mWebhook mBotStateDir optimizerTmp limits adoptReq ctrl args settings symRaw =
  if not (platformSupportsLiveBot (argPlatform args))
    then pure (Left ("bot/start supports Binance only (platform=" ++ platformCode (argPlatform args) ++ ")"))
    else
      case argData args of
        Just _ -> pure (Left "bot/start supports binanceSymbol only (no CSV data source)")
        Nothing ->
          case trim symRaw of
            "" -> pure (Left "bot/start requires binanceSymbol or botSymbols")
            _ -> do
              let sym = normalizeSymbol symRaw
                  argsSym = args { argBinanceSymbol = Just sym }
                  startReason = botStartReasonFromAdopt adoptReq
              now <- getTimestampMs
              modifyMVar (bcRuntime ctrl) $ \mrt ->
                case HM.lookup sym mrt of
                  Just st ->
                    if allowExisting
                      then pure (mrt, Right (BotStartOutcome sym st False))
                      else
                        case st of
                          BotRunning _ -> pure (mrt, Left "Bot is already running")
                          BotStarting _ -> pure (mrt, Left "Bot is starting")
                  Nothing -> do
                    preflight <- preflightBotStart mOps argsSym settings sym
                    case preflight of
                      Left e | not (arActive adoptReq) -> pure (mrt, Left e)
                      Left _ -> do
                        stopSig <- newEmptyMVar
                        tid <- forkIO (botStartWorker mOps metrics mJournal mWebhook mBotStateDir optimizerTmp limits adoptReq ctrl argsSym settings sym stopSig)
                        let rt =
                              BotStartRuntime
                                { bsrThreadId = tid
                                , bsrStopSignal = stopSig
                                , bsrArgs = argsSym
                                , bsrSettings = settings
                                , bsrSymbol = sym
                                , bsrRequestedAtMs = now
                                , bsrStartReason = startReason
                                }
                            st = BotStarting rt
                        pure (HM.insert sym st mrt, Right (BotStartOutcome sym st True))
                      Right () -> do
                        stopSig <- newEmptyMVar
                        tid <- forkIO (botStartWorker mOps metrics mJournal mWebhook mBotStateDir optimizerTmp limits adoptReq ctrl argsSym settings sym stopSig)
                        let rt =
                              BotStartRuntime
                                { bsrThreadId = tid
                                , bsrStopSignal = stopSig
                                , bsrArgs = argsSym
                                , bsrSettings = settings
                                , bsrSymbol = sym
                                , bsrRequestedAtMs = now
                                , bsrStartReason = startReason
                                }
                            st = BotStarting rt
                        pure (HM.insert sym st mrt, Right (BotStartOutcome sym st True))

botStartWorker ::
  Maybe OpsStore ->
  Metrics ->
  Maybe Journal ->
  Maybe Webhook ->
  Maybe FilePath ->
  FilePath ->
  ApiComputeLimits ->
  AdoptRequirement ->
  BotController ->
  Args ->
  BotSettings ->
  String ->
  MVar () ->
  IO ()
botStartWorker mOps metrics mJournal mWebhook mBotStateDir optimizerTmp limits adoptReq ctrl args settings sym stopSig = do
  tid <- myThreadId
  let doStart baseArgs = do
        argsFinal <-
          if arActive adoptReq
            then applyLatestTopCombo optimizerTmp limits sym baseArgs adoptReq
            else pure baseArgs
        preflight <- preflightBotStart mOps argsFinal settings sym
        case preflight of
          Left e -> throwIO (userError e)
          Right () -> initBotState mOps argsFinal settings sym
  r <- try (doStart args) :: IO (Either SomeException BotState)
  case r of
    Left ex -> do
      now <- getTimestampMs
      journalWriteMaybe mJournal (object ["type" .= ("bot.start_failed" :: String), "atMs" .= now, "error" .= show ex])
      opsAppendMaybe mOps "bot.start_failed" Nothing (Just (argsPublicJson args)) (Just (object ["error" .= show ex])) Nothing
      webhookNotifyMaybe mWebhook (webhookEventBotStartFailed args sym ex)
      modifyMVar_ (bcRuntime ctrl) $ \mrt ->
        case HM.lookup sym mrt of
          Just (BotStarting rt) | bsrThreadId rt == tid -> pure (HM.delete sym mrt)
          _ -> pure mrt
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
      webhookNotifyMaybe mWebhook (webhookEventBotStarted (botArgs st0) (botSymbol st0))
      stVar <- newMVar st0
      persistBotStatusMaybe mBotStateDir st0
      optimizerStopSig <- newEmptyMVar
      optimizerPending <- newMVar Nothing
      optimizerTid <- forkIO (botOptimizerLoop mOps metrics mJournal stVar optimizerStopSig optimizerPending)
      let optimizerRt =
            BotOptimizerRuntime
              { borThreadId = optimizerTid
              , borStopSignal = optimizerStopSig
              , borPendingUpdate = optimizerPending
              }
      startOk <-
        modifyMVar (bcRuntime ctrl) $ \mrt ->
          case HM.lookup sym mrt of
            Just (BotStarting rt) | bsrThreadId rt == tid ->
              pure (HM.insert sym (BotRunning (BotRuntime tid stVar stopSig (Just optimizerRt))) mrt, True)
            _ -> pure (mrt, False)
      if startOk
        then botLoop mOps metrics mJournal mWebhook mBotStateDir ctrl stVar stopSig (Just optimizerPending)
        else do
          _ <- tryPutMVar optimizerStopSig ()
          killThread optimizerTid

botAutoStartLoop ::
  Maybe OpsStore ->
  Metrics ->
  Maybe Journal ->
  Maybe Webhook ->
  Maybe FilePath ->
  FilePath ->
  ApiComputeLimits ->
  Args ->
  BotController ->
  IO ()
botAutoStartLoop mOps metrics mJournal mWebhook mBotStateDir optimizerTmp limits baseArgs botCtrl = do
  symbolsOrErr <- resolveBotSymbolsAuto baseArgs
  case symbolsOrErr of
    Left err -> putStrLn ("Live bot auto-start disabled: " ++ err)
    Right symbols ->
      if not (platformSupportsLiveBot (argPlatform baseArgs))
        then
          putStrLn
            ( "Live bot auto-start disabled: platform="
                ++ platformCode (argPlatform baseArgs)
                ++ " does not support live bots."
            )
        else
          case argData baseArgs of
            Just _ -> putStrLn "Live bot auto-start disabled: CSV data source is not supported."
            Nothing -> do
              let argsBase = baseArgs { argTradeOnly = True }
                  settings = defaultBotSettings argsBase
              pollSec <- comboPollSecondsFromEnv
              errRef <- newIORef HM.empty
              putStrLn ("Live bot auto-start enabled for symbols: " ++ intercalate ", " symbols)

              let sleepSec s = threadDelay (max 1 s * 1000000)
                  recordError sym msg = do
                    prev <- readIORef errRef
                    if HM.lookup sym prev == Just msg
                      then pure ()
                      else do
                        writeIORef errRef (HM.insert sym msg prev)
                        putStrLn ("Live bot auto-start failed for " ++ sym ++ ": " ++ msg)
                  clearError sym = modifyIORef' errRef (HM.delete sym)
                  startSymbol sym = do
                    let argsSym = argsBase { argBinanceSymbol = Just sym }
                        adoptable = bsTradeEnabled settings && platformSupportsLiveBot (argPlatform argsSym)
                    adoptReqOrErr <-
                      if adoptable
                        then resolveAdoptionRequirement mOps argsSym sym
                        else pure (Right (AdoptRequirement False False))
                    case adoptReqOrErr of
                      Left err -> recordError sym err
                      Right adoptReq -> do
                        argsCombo <-
                          if arActive adoptReq
                            then pure argsSym
                            else applyLatestTopCombo optimizerTmp limits sym argsSym adoptReq
                        case validateApiComputeLimits limits argsCombo of
                          Left err -> recordError sym err
                          Right argsOk -> do
                            r <-
                              botStartSymbolWithSettings
                                True
                                mOps
                                metrics
                                mJournal
                                mWebhook
                                mBotStateDir
                                optimizerTmp
                                limits
                                adoptReq
                                botCtrl
                                argsOk
                                settings
                                sym
                            case r of
                              Left err -> recordError sym err
                              Right _ -> clearError sym
                  loop = do
                    mrt <- readMVar (bcRuntime botCtrl)
                    let missing = filter (not . (`HM.member` mrt)) symbols
                    mapM_ startSymbol missing
                    sleepSec pollSec
                    loop
              loop

botStop :: BotController -> Maybe String -> IO [BotState]
botStop ctrl mSymbol =
  modifyMVar (bcRuntime ctrl) $ \mrt -> do
    let (targets, remaining) =
          case mSymbol of
            Nothing -> (HM.toList mrt, HM.empty)
            Just symRaw ->
              let sym = normalizeSymbol symRaw
               in case HM.lookup sym mrt of
                    Nothing -> ([], mrt)
                    Just st -> ([(sym, st)], HM.delete sym mrt)
    stopped <- mapM (stopRuntime . snd) targets
    pure (remaining, catMaybes stopped)
  where
    stopRuntime st =
      case st of
        BotStarting rt -> do
          _ <- tryPutMVar (bsrStopSignal rt) ()
          killThread (bsrThreadId rt)
          pure Nothing
        BotRunning rt -> do
          mSt <- try (readMVar (brStateVar rt)) :: IO (Either SomeException BotState)
          case brOptimizer rt of
            Nothing -> pure ()
            Just optRt -> do
              _ <- tryPutMVar (borStopSignal optRt) ()
              killThread (borThreadId optRt)
          _ <- tryPutMVar (brStopSignal rt) ()
          killThread (brThreadId rt)
          pure (either (const Nothing) Just mSt)

runtimeStateToState :: BotRuntimeState -> IO (Maybe BotState)
runtimeStateToState st =
  case st of
    BotRunning rt -> Just <$> readMVar (brStateVar rt)
    _ -> pure Nothing

botGetStates :: BotController -> IO [BotState]
botGetStates ctrl = do
  mrt <- readMVar (bcRuntime ctrl)
  states <- mapM runtimeStateToState (HM.elems mrt)
  pure (catMaybes states)

botGetStateFor :: BotController -> String -> IO (Maybe BotState)
botGetStateFor ctrl symRaw = do
  mrt <- readMVar (bcRuntime ctrl)
  case HM.lookup (normalizeSymbol symRaw) mrt of
    Just (BotRunning rt) -> Just <$> readMVar (brStateVar rt)
    _ -> pure Nothing

initBotState :: Maybe OpsStore -> Args -> BotSettings -> String -> IO BotState
initBotState mOps args settings sym = do
  let lookback = argLookback args
  now <- getTimestampMs
  env <- makeBinanceEnv mOps args
  let tradeEnabled = bsTradeEnabled settings
  startPos0 <-
    if tradeEnabled
      then do
        ensureBinanceKeysPresent env
        pos <- fetchBotAccountPos args env sym
        if pos /= 0 && not (bsAdoptExistingPosition settings)
          then
            let sideLabel = if pos > 0 then "long" else "short"
             in
              throwIO
                ( userError
                    ( "Refusing to start: existing "
                        ++ sideLabel
                        ++ " position detected for "
                        ++ sym
                        ++ " ("
                        ++ marketCode (argBinanceMarket args)
                        ++ "). Flatten first, or set botAdoptExistingPosition=true."
                    )
                )
          else pure pos
      else pure 0
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
        (lstmModel, _) <- trainLstmWithPersistence args lookback lstmCfg obsAll
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

  let latest0Raw = computeLatestSignal args lookback pricesV mLstmCtx mKalmanCtx Nothing

      -- Startup decision:
      -- - Adopted positions use closeThreshold (hysteresis) via the gated closeDirection.
      -- - Otherwise, entry uses openThreshold via lsChosenDir.
      closeDir = lsCloseDir latest0Raw

      wantLongClose = closeDir == Just 1
      wantShortClose = closeDir == Just (-1)
      allowShort = argPositioning args == LongShort

      desiredPosSignal =
        case startPos0 of
          1 ->
            if wantLongClose
              then 1
              else
                case (allowShort, lsChosenDir latest0Raw) of
                  (True, Just (-1)) -> -1
                  _ -> 0
          (-1) ->
            if wantShortClose
              then -1
              else
                case (allowShort, lsChosenDir latest0Raw) of
                  (True, Just 1) -> 1
                  _ -> 0
          _ ->
            case lsChosenDir latest0Raw of
              Just 1 -> 1
              Just (-1) | allowShort -> -1
              _ -> 0

      latest =
        case (startPos0, desiredPosSignal) of
          (1, 0) ->
            if lsChosenDir latest0Raw /= Just (-1)
              then latest0Raw { lsChosenDir = Just (-1), lsAction = "FLAT (close)" }
              else latest0Raw
          (-1, 0) ->
            if lsChosenDir latest0Raw /= Just 1
              then latest0Raw { lsChosenDir = Just 1, lsAction = "FLAT (close)" }
              else latest0Raw
          _ -> latest0Raw

      baseEq = 1.0
      eq0 = V.replicate n baseEq
      pos0 = V.replicate n 0
      lastOt = V.last openV
      wantSwitch = desiredPosSignal /= startPos0
      opSide =
        if desiredPosSignal > startPos0
          then "BUY"
          else "SELL"

  mOrder <-
    if wantSwitch
      then
        if argPositioning args == LongShort && desiredPosSignal == 0 && startPos0 /= 0
          then Just <$> placeBotCloseIfEnabled args settings latest env sym
          else Just <$> placeIfEnabled args settings latest env sym
      else pure Nothing

  let orderSent = maybe False aorSent mOrder
      alreadyMsg =
        case mOrder of
          Nothing -> False
          Just o ->
            let msg = aorMessage o
             in "already long" `isInfixOf` msg || "already short" `isInfixOf` msg || "already flat" `isInfixOf` msg
      appliedSwitch =
        if not tradeEnabled
          then True
          else orderSent || alreadyMsg
      feeApplied =
        if not tradeEnabled
          then True
          else orderSent
      desiredPos =
        if not wantSwitch
          then startPos0
          else if appliedSwitch then desiredPosSignal else startPos0
      didTradeNow = wantSwitch && appliedSwitch && feeApplied

      eq1 =
        if didTradeNow
          then eq0 V.// [(n - 1, baseEq * (1 - argFee args))]
          else eq0
      pos1 = pos0 V.// [(n - 1, desiredPos)]
      openTrade =
        case desiredPos of
          1 ->
            let px = V.last pricesV
             in Just (BotOpenTrade (n - 1) (eq1 V.! (n - 1)) 0 px px SideLong)
          (-1) ->
            let px = V.last pricesV
             in Just (BotOpenTrade (n - 1) (eq1 V.! (n - 1)) 0 px px SideShort)
          _ -> Nothing
      ops =
        if wantSwitch && appliedSwitch
          then [BotOp (n - 1) opSide (V.last pricesV)]
          else []

      orders =
        case (wantSwitch, mOrder) of
          (True, Just o) -> [BotOrderEvent (n - 1) opSide (V.last pricesV) lastOt now o]
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
                    Just ot -> Just (shiftOpenTradeIndex dropCount ot)
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
        if tradeEnabled && wantSwitch && not appliedSwitch
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
          , botCooldownLeft = 0
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
          , botPolledAtMs = now
          , botPollLatencyMs = 0
          , botFetchedKlines = length ks
          , botFetchedLastKline =
              if null ks
                then Nothing
                else Just (last ks)
          , botLastBatchAtMs = now
          , botLastBatchSize = length ks
          , botLastBatchMs = 0
          , botError = Nothing
          }

  if wantSwitch && appliedSwitch && desiredPos /= 0
    then botOptimizeAfterOperation st0
    else pure st0

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
              baseOpenThr = argOpenThreshold args
              baseCloseThr = argCloseThreshold args
              fee = argFee args
              perSideCost = estimatedPerSideCost fee (argSlippage args) (argSpread args)
              minEdgeBase = max 0 (argMinEdge args)
              minEdge =
                if argCostAwareEdge args
                  then max minEdgeBase (breakEvenThresholdFromPerSideCost perSideCost + max 0 (argEdgeBuffer args))
                  else minEdgeBase
              baseCfg =
                EnsembleConfig
                  { ecOpenThreshold = baseOpenThr
                  , ecCloseThreshold = baseCloseThr
                  , ecFee = fee
                  , ecSlippage = argSlippage args
                  , ecSpread = argSpread args
                  , ecStopLoss = argStopLoss args
                  , ecTakeProfit = argTakeProfit args
                  , ecTrailingStop = argTrailingStop args
                  , ecStopLossVolMult = argStopLossVolMult args
                  , ecTakeProfitVolMult = argTakeProfitVolMult args
                  , ecTrailingStopVolMult = argTrailingStopVolMult args
                  , ecMinHoldBars = argMinHoldBars args
                  , ecCooldownBars = argCooldownBars args
                  , ecMaxHoldBars = argMaxHoldBars args
                  , ecMaxDrawdown = argMaxDrawdown args
                  , ecMaxDailyLoss = argMaxDailyLoss args
                  , ecIntervalSeconds = parseIntervalSeconds (argInterval args)
                  , ecOpenTimes = Just (V.drop start (botOpenTimes st))
                  , ecPositioning = argPositioning args
                  , ecIntrabarFill = argIntrabarFill args
                  , ecMaxPositionSize = argMaxPositionSize args
                  , ecMinEdge = minEdge
                  , ecMinSignalToNoise = argMinSignalToNoise args
                  , ecTrendLookback = argTrendLookback args
                  , ecPeriodsPerYear = periodsPerYear args
                  , ecVolTarget = argVolTarget args
                  , ecVolLookback = argVolLookback args
                  , ecVolEwmaAlpha = argVolEwmaAlpha args
                  , ecVolFloor = argVolFloor args
                  , ecVolScaleMax = argVolScaleMax args
                  , ecMaxVolatility = argMaxVolatility args
                  , ecRebalanceBars = argRebalanceBars args
                  , ecRebalanceThreshold = argRebalanceThreshold args
                  , ecRebalanceGlobal = argRebalanceGlobal args
                  , ecRebalanceResetOnSignal = argRebalanceResetOnSignal args
                  , ecFundingRate = argFundingRate args
                  , ecFundingBySide = argFundingBySide args
                  , ecFundingOnOpen = argFundingOnOpen args
                  , ecBlendWeight = argBlendWeight args
                  , ecKalmanDt = argKalmanDt args
                  , ecKalmanProcessVar = argKalmanProcessVar args
                  , ecKalmanMeasurementVar = argKalmanMeasurementVar args
                  , ecTriLayer = argTriLayer args
                  , ecTriLayerFastMult = argTriLayerFastMult args
                  , ecTriLayerSlowMult = argTriLayerSlowMult args
                  , ecTriLayerCloudPadding = argTriLayerCloudPadding args
                  , ecTriLayerRequirePriceAction = argTriLayerRequirePriceAction args
                  , ecLstmExitFlipBars = argLstmExitFlipBars args
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
              ppy = periodsPerYear args
              tuneCfg =
                TuneConfig
                  { tcObjective = argTuneObjective args
                  , tcPenaltyMaxDrawdown = argTunePenaltyMaxDrawdown args
                  , tcPenaltyTurnover = argTunePenaltyTurnover args
                  , tcPeriodsPerYear = ppy
                  , tcWalkForwardFolds = argWalkForwardFolds args
                  , tcMinRoundTrips = argMinRoundTrips args
                  , tcStressVolMultiplier = argTuneStressVolMult args
                  , tcStressShock = argTuneStressShock args
                  , tcStressWeight = argTuneStressWeight args
                  }
              thresholdResult =
                if optimizeOps && hasBothCtx
                  then
                    fmap
                      (\(m, openThr, closeThr, _bt, _stats) -> (m, openThr, closeThr))
                      (optimizeOperationsWith tuneCfg baseCfg prices kalPred lstmPred Nothing)
                  else
                    fmap
                      (\(openThr, closeThr, _bt, _stats) -> (argMethod args, openThr, closeThr))
                      (sweepThresholdWith tuneCfg (argMethod args) baseCfg prices kalPred lstmPred Nothing)
          (newMethod, newOpenThr, newCloseThr) <-
            case thresholdResult of
              Right res -> pure res
              Left err -> do
                hPutStrLn stderr ("Warning: Threshold tuning failed and was skipped: " ++ err)
                pure (argMethod args, baseOpenThr, baseCloseThr)
          let args' =
                args
                  { argMethod = newMethod
                  , argOpenThreshold = newOpenThr
                  , argCloseThreshold = newCloseThr
                  }
              latest' = computeLatestSignal args' lookback pricesV (botLstmCtx st) (botKalmanCtx st) Nothing
          pure st { botArgs = args', botLatestSignal = latest' }

argLookbackEither :: Args -> Either String Int
argLookbackEither args =
  case argLookbackBars args of
    Just n ->
      if n < 2
        then Left "--lookback-bars must be >= 2"
        else Right n
    Nothing ->
      case lookbackBarsFrom (argInterval args) (argLookbackWindow args) of
        Left err -> Left err
        Right n ->
          if n < 2
            then
              Left
                ( "Lookback window too small: "
                    ++ show (argLookbackWindow args)
                    ++ " at interval "
                    ++ show (argInterval args)
                    ++ " yields "
                    ++ show n
                    ++ " bars; need at least 2 bars."
                )
            else Right n

botApplyOptimizerUpdate :: BotState -> BotOptimizerUpdate -> IO BotState
botApplyOptimizerUpdate st upd = do
  now <- getTimestampMs
  let args' =
        (bouArgs upd)
          { argOptimizeOperations = False
          , argSweepThreshold = False
          }
      lookback' = max 2 (bouLookback upd)
      pricesV = botPrices st
      n = V.length pricesV
      mLstmCtx' = bouLstmCtx upd <|> botLstmCtx st
      mKalmanCtx' = bouKalmanCtx upd <|> botKalmanCtx st
      method = argMethod args'
      ctxOk =
        case method of
          MethodBoth -> isJust mLstmCtx' && isJust mKalmanCtx'
          MethodKalmanOnly -> isJust mKalmanCtx'
          MethodLstmOnly -> isJust mLstmCtx'
          MethodBlend -> isJust mLstmCtx' && isJust mKalmanCtx'
      hasLstmWindow = method /= MethodKalmanOnly && n >= lookback'

  if not ctxOk
    then pure st { botError = Just "Optimizer update skipped: missing model context.", botUpdatedAtMs = now }
    else
      if not hasLstmWindow
        then pure st { botError = Just "Optimizer update skipped: not enough data for lookback.", botUpdatedAtMs = now }
        else do
          let latest = computeLatestSignal args' lookback' pricesV mLstmCtx' mKalmanCtx' Nothing
          pure
            st
              { botArgs = args'
              , botLookback = lookback'
              , botLstmCtx = mLstmCtx'
              , botKalmanCtx = mKalmanCtx'
              , botLatestSignal = latest
              , botUpdatedAtMs = now
              , botError = Nothing
              }

rebuildLstmCtx :: Args -> Int -> V.Vector Double -> IO (Either String LstmCtx)
rebuildLstmCtx args lookback pricesV =
  let closes = V.toList pricesV
      n = length closes
   in
    if n <= lookback
      then pure (Left "Not enough bars to rebuild LSTM context.")
      else do
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
        (lstmModel, _) <- trainLstmWithPersistence args lookback lstmCfg obsAll
        pure (Right (normState, obsAll, lstmModel))

rebuildKalmanCtx :: Args -> Int -> V.Vector Double -> Either String KalmanCtx
rebuildKalmanCtx args lookback pricesV =
  let n = V.length pricesV
   in if n < 2
        then Left "Not enough bars to rebuild Kalman context."
        else
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
           in Right (predictors, kalPrev, hmmPrev, svPrev)

parseTopComboToArgs :: Args -> TopCombo -> Either String Args
parseTopComboToArgs base combo = do
  let params = tcParams combo
      getStr k = KM.lookup k params >>= AT.parseMaybe parseJSON
      getInt k = KM.lookup k params >>= AT.parseMaybe parseJSON
      getDbl k = KM.lookup k params >>= AT.parseMaybe parseJSON
      getBool k = KM.lookup k params >>= AT.parseMaybe parseJSON
      getMaybeDbl k =
        case KM.lookup k params of
          Nothing -> Nothing
          Just Aeson.Null -> Just Nothing
          Just v ->
            case AT.parseMaybe parseJSON v of
              Nothing -> Nothing
              Just d -> Just (Just d)

      methodRaw = getStr "method"
      normRaw = getStr "normalization"
      positioningRaw = getStr "positioning"
      intrabarFillRaw = getStr "intrabarFill"

  method <-
    case methodRaw of
      Nothing -> Right (argMethod base)
      Just v -> parseMethod v
  let normalization =
        case normRaw >>= parseNormType of
          Just n -> n
          Nothing -> argNormalization base
  _positioning <-
    case positioningRaw of
      Nothing -> Right (argPositioning base)
      Just v -> parsePositioning v
  intrabarFill <-
    case intrabarFillRaw of
      Nothing -> Right (argIntrabarFill base)
      Just v -> parseIntrabarFill v

  let openThr = max 0 (fromMaybe (argOpenThreshold base) (tcOpenThreshold combo))
      closeThr = max 0 (fromMaybe (argCloseThreshold base) (tcCloseThreshold combo <|> tcOpenThreshold combo))

      clamp01 x = max 0 (min 1 x)

      readMaybeInt = fmap (max 0) . getInt
      readMaybeDouble = fmap (\x -> if isNaN x || isInfinite x then 0 else x) . getDbl

      pickMaybeMaybeDbl k current =
        case getMaybeDbl k of
          Nothing -> current
          Just v -> v

      pickMaybeMaybeInt k current =
        case KM.lookup k params of
          Nothing -> current
          Just Aeson.Null -> Nothing
          Just v ->
            case AT.parseMaybe parseJSON v of
              Nothing -> current
              Just n -> Just (max 0 n)

      pickBool k current =
        case getBool k of
          Nothing -> current
          Just b -> b

      pickD k current =
        case readMaybeDouble k of
          Nothing -> current
          Just v -> v

      pickI k current =
        case readMaybeInt k of
          Nothing -> current
          Just v -> v

      gradClip =
        case getMaybeDbl "gradClip" of
          Nothing -> argGradClip base
          Just Nothing -> Nothing
          Just (Just v) -> Just (max 0 v)

      fee = max 0 (pickD "fee" (argFee base))
      slippage = max 0 (pickD "slippage" (argSlippage base))
      spread = max 0 (pickD "spread" (argSpread base))

      kalZMin = max 0 (pickD "kalmanZMin" (argKalmanZMin base))
      kalZMaxRaw = max 0 (pickD "kalmanZMax" (argKalmanZMax base))
      kalZMax = max kalZMin kalZMaxRaw

      maxHighVolProb = pickMaybeMaybeDbl "maxHighVolProb" (argMaxHighVolProb base)
      maxConformalWidth = pickMaybeMaybeDbl "maxConformalWidth" (argMaxConformalWidth base)
      maxQuantileWidth = pickMaybeMaybeDbl "maxQuantileWidth" (argMaxQuantileWidth base)

      maxDD = pickMaybeMaybeDbl "maxDrawdown" (argMaxDrawdown base)
      maxDL = pickMaybeMaybeDbl "maxDailyLoss" (argMaxDailyLoss base)
      maxOE = pickMaybeMaybeInt "maxOrderErrors" (argMaxOrderErrors base)

      stopLoss = pickMaybeMaybeDbl "stopLoss" (argStopLoss base)
      takeProfit = pickMaybeMaybeDbl "takeProfit" (argTakeProfit base)
      trailingStop = pickMaybeMaybeDbl "trailingStop" (argTrailingStop base)

      confidenceSizing = pickBool "confidenceSizing" (argConfidenceSizing base)
      confirmConformal = pickBool "confirmConformal" (argConfirmConformal base)
      confirmQuantiles = pickBool "confirmQuantiles" (argConfirmQuantiles base)

      minPositionSize =
        case readMaybeDouble "minPositionSize" of
          Nothing -> argMinPositionSize base
          Just v -> clamp01 v

      minHoldBars = max 0 (pickI "minHoldBars" (argMinHoldBars base))
      cooldownBars = max 0 (pickI "cooldownBars" (argCooldownBars base))
      maxHoldBars =
        case pickMaybeMaybeInt "maxHoldBars" (argMaxHoldBars base) of
          Nothing -> Nothing
          Just n -> if n <= 0 then Nothing else Just n

      minEdge = max 0 (pickD "minEdge" (argMinEdge base))
      edgeBuffer = max 0 (pickD "edgeBuffer" (argEdgeBuffer base))
      costAwareEdge = pickBool "costAwareEdge" (argCostAwareEdge base)
      trendLookback = max 0 (pickI "trendLookback" (argTrendLookback base))
      maxPositionSize = max 0 (pickD "maxPositionSize" (argMaxPositionSize base))

      volTarget =
        case pickMaybeMaybeDbl "volTarget" (argVolTarget base) of
          Nothing -> Nothing
          Just v -> if v <= 0 then Nothing else Just v
      volLookback = max 0 (pickI "volLookback" (argVolLookback base))
      volEwmaAlpha =
        case pickMaybeMaybeDbl "volEwmaAlpha" (argVolEwmaAlpha base) of
          Nothing -> Nothing
          Just v ->
            if v > 0 && v < 1
              then Just v
              else Nothing
      volFloor = max 0 (pickD "volFloor" (argVolFloor base))
      volScaleMax = max 0 (pickD "volScaleMax" (argVolScaleMax base))
      maxVolatility =
        case pickMaybeMaybeDbl "maxVolatility" (argMaxVolatility base) of
          Nothing -> Nothing
          Just v -> if v <= 0 then Nothing else Just v

      rebalanceBars = max 0 (pickI "rebalanceBars" (argRebalanceBars base))
      rebalanceThreshold = max 0 (pickD "rebalanceThreshold" (argRebalanceThreshold base))
      rebalanceGlobal = pickBool "rebalanceGlobal" (argRebalanceGlobal base)
      rebalanceResetOnSignal = pickBool "rebalanceResetOnSignal" (argRebalanceResetOnSignal base)
      fundingRate = pickD "fundingRate" (argFundingRate base)
      fundingBySide = pickBool "fundingBySide" (argFundingBySide base)
      fundingOnOpen = pickBool "fundingOnOpen" (argFundingOnOpen base)

      blendWeight = clamp01 (pickD "blendWeight" (argBlendWeight base))
      triLayer = pickBool "triLayer" (argTriLayer base)
      triLayerFastMult = max 1e-6 (pickD "triLayerFastMult" (argTriLayerFastMult base))
      triLayerSlowMult = max 1e-6 (pickD "triLayerSlowMult" (argTriLayerSlowMult base))
      periodsPerYear =
        case pickMaybeMaybeDbl "periodsPerYear" (argPeriodsPerYear base) of
          Nothing -> Nothing
          Just v -> if v <= 0 then Nothing else Just v
      kalmanMarketTopN = max 0 (pickI "kalmanMarketTopN" (argKalmanMarketTopN base))

      walkForwardFolds = max 1 (pickI "walkForwardFolds" (argWalkForwardFolds base))
      tuneStressVolMult = max 1e-12 (pickD "tuneStressVolMult" (argTuneStressVolMult base))
      tuneStressShock = pickD "tuneStressShock" (argTuneStressShock base)
      tuneStressWeight = max 0 (pickD "tuneStressWeight" (argTuneStressWeight base))

      out =
        base
          { argMethod = method
          , argPositioning = _positioning
          , argNormalization = normalization
          , argOpenThreshold = openThr
          , argCloseThreshold = closeThr
          , argFee = fee
          , argEpochs = max 1 (pickI "epochs" (argEpochs base))
          , argHiddenSize = max 1 (pickI "hiddenSize" (argHiddenSize base))
          , argLr = max 1e-12 (pickD "learningRate" (argLr base))
          , argValRatio = clamp01 (pickD "valRatio" (argValRatio base))
          , argPatience = max 0 (pickI "patience" (argPatience base))
          , argGradClip = gradClip
          , argSlippage = slippage
          , argSpread = spread
          , argIntrabarFill = intrabarFill
          , argStopLoss = stopLoss
          , argTakeProfit = takeProfit
          , argTrailingStop = trailingStop
          , argMinHoldBars = minHoldBars
          , argCooldownBars = cooldownBars
          , argMaxHoldBars = maxHoldBars
          , argMaxDrawdown = maxDD
          , argMaxDailyLoss = maxDL
          , argMaxOrderErrors = maxOE
          , argMinEdge = minEdge
          , argCostAwareEdge = costAwareEdge
          , argEdgeBuffer = edgeBuffer
          , argTrendLookback = trendLookback
          , argMaxPositionSize = maxPositionSize
          , argVolTarget = volTarget
          , argVolLookback = volLookback
          , argVolEwmaAlpha = volEwmaAlpha
          , argVolFloor = volFloor
          , argVolScaleMax = volScaleMax
          , argMaxVolatility = maxVolatility
          , argRebalanceBars = rebalanceBars
          , argRebalanceThreshold = rebalanceThreshold
          , argRebalanceGlobal = rebalanceGlobal
          , argRebalanceResetOnSignal = rebalanceResetOnSignal
          , argFundingRate = fundingRate
          , argFundingBySide = fundingBySide
          , argFundingOnOpen = fundingOnOpen
          , argBlendWeight = blendWeight
          , argTriLayer = triLayer
          , argTriLayerFastMult = triLayerFastMult
          , argTriLayerSlowMult = triLayerSlowMult
          , argKalmanDt = max 1e-12 (pickD "kalmanDt" (argKalmanDt base))
          , argKalmanProcessVar = max 1e-12 (pickD "kalmanProcessVar" (argKalmanProcessVar base))
          , argKalmanMeasurementVar = max 1e-12 (pickD "kalmanMeasurementVar" (argKalmanMeasurementVar base))
          , argKalmanMarketTopN = kalmanMarketTopN
          , argKalmanZMin = kalZMin
          , argKalmanZMax = kalZMax
          , argMaxHighVolProb = maxHighVolProb
          , argMaxConformalWidth = maxConformalWidth
          , argMaxQuantileWidth = maxQuantileWidth
          , argConfirmConformal = confirmConformal
          , argConfirmQuantiles = confirmQuantiles
          , argConfidenceSizing = confidenceSizing
          , argMinPositionSize = minPositionSize
          , argPeriodsPerYear = periodsPerYear
          , argWalkForwardFolds = walkForwardFolds
          , argTuneStressVolMult = tuneStressVolMult
          , argTuneStressShock = tuneStressShock
          , argTuneStressWeight = tuneStressWeight
          , argOptimizeOperations = False
          , argSweepThreshold = False
          }

  pure out

buildOptimizerUpdate :: BotState -> TopCombo -> IO (Either String BotOptimizerUpdate)
buildOptimizerUpdate st combo = do
  let curArgs = botArgs st
  case parseTopComboToArgs curArgs combo of
    Left e -> pure (Left e)
    Right args0 ->
      case argLookbackEither args0 of
        Left e -> pure (Left e)
        Right lookback -> do
          let pricesV = botPrices st
              n = V.length pricesV
              hasLstmCtx = isJust (botLstmCtx st)
              hasKalmanCtx = isJust (botKalmanCtx st)
              needsLstm' =
                argMethod args0 /= MethodKalmanOnly
                  && ( not hasLstmCtx
                        || argHiddenSize args0 /= argHiddenSize curArgs
                        || argNormalization args0 /= argNormalization curArgs
                        || lookback /= botLookback st
                     )
              needsKalman' =
                argMethod args0 /= MethodLstmOnly
                  && ( not hasKalmanCtx
                        || argKalmanDt args0 /= argKalmanDt curArgs
                        || argKalmanProcessVar args0 /= argKalmanProcessVar curArgs
                        || argKalmanMeasurementVar args0 /= argKalmanMeasurementVar curArgs
                        || lookback /= botLookback st
                     )

          if n < max 3 (lookback + 1)
            then pure (Left "Not enough bars to apply optimizer update.")
            else do
              mLstmE <- if needsLstm' then Just <$> rebuildLstmCtx args0 lookback pricesV else pure Nothing
              let mKalE = if needsKalman' then Just (rebuildKalmanCtx args0 lookback pricesV) else Nothing
                  collect =
                    case (mLstmE, mKalE) of
                      (Just (Left e), _) -> Left e
                      (_, Just (Left e)) -> Left e
                      (Just (Right l), Just (Right k)) -> Right (Just l, Just k)
                      (Just (Right l), Nothing) -> Right (Just l, Nothing)
                      (Nothing, Just (Right k)) -> Right (Nothing, Just k)
                      (Nothing, Nothing) -> Right (Nothing, Nothing)
              pure $
                case collect of
                  Left e -> Left e
                  Right (mLstm, mKal) ->
                    Right
                      BotOptimizerUpdate
                        { bouArgs = args0
                        , bouLookback = lookback
                        , bouLstmCtx = mLstm
                        , bouKalmanCtx = mKal
                        , bouFinalEquity = tcFinalEquity combo
                        , bouObjective = tcObjectiveLabel combo
                        , bouScore = tcScore combo
                        }

sanitizeFileComponent :: String -> String
sanitizeFileComponent raw =
  let go c = if isAlphaNum c || c == '-' || c == '_' then c else '-'
   in map go raw

botOptimizerLoop :: Maybe OpsStore -> Metrics -> Maybe Journal -> MVar BotState -> MVar () -> MVar (Maybe BotOptimizerUpdate) -> IO ()
botOptimizerLoop mOps _metrics mJournal stVar stopSig pending = do
  projectRoot <- getCurrentDirectory
  mStateDir <- stateDirFromEnv
  let tmpRoot = projectRoot </> ".tmp"
      optimizerTmp = fromMaybe (tmpRoot </> "optimizer") (fmap (</> "optimizer") mStateDir)
  createDirectoryIfMissing True optimizerTmp
  topJsonPath <- resolveOptimizerCombosPath optimizerTmp
  pollEnv <- lookupEnv "TRADER_BOT_COMBOS_POLL_SEC"
  let pollSec =
        case pollEnv >>= readMaybe of
          Just n | n >= 5 -> n
          _ -> 30
  lastErrRef <- newIORef (Nothing :: Maybe (String, String))

  let sleepSec s = threadDelay (max 1 s * 1000000)

      recordError kind msg sym interval = do
        prev <- readIORef lastErrRef
        let cur = Just (kind, msg)
        if prev == cur
          then pure ()
          else do
            writeIORef lastErrRef cur
            now <- getTimestampMs
            journalWriteMaybe
              mJournal
              ( object
                  [ "type" .= kind
                  , "atMs" .= now
                  , "error" .= msg
                  , "symbol" .= sym
                  , "interval" .= interval
                  , "path" .= topJsonPath
                  ]
              )
            opsAppendMaybe
              mOps
              (T.pack kind)
              Nothing
              Nothing
              (Just (object ["error" .= msg, "symbol" .= sym, "interval" .= interval, "path" .= topJsonPath]))
              Nothing

      clearError = writeIORef lastErrRef Nothing

      loop = do
        stopReq <- isJust <$> tryReadMVar stopSig
        if stopReq
          then pure ()
          else do
            st <- readMVar stVar
            let sym = botSymbol st
                interval = argInterval (botArgs st)
            combosOrErr <- readTopCombosExport topJsonPath
            case combosOrErr of
              Left e -> recordError "bot.combo.sync_failed" e sym interval
              Right export ->
                case bestTopComboForSymbol sym (Just interval) export of
                  Nothing -> recordError "bot.combo.sync_failed" "No top combo for symbol+interval." sym interval
                  Just bestCombo -> do
                    clearError
                    updOrErr <- buildOptimizerUpdate st bestCombo
                    case updOrErr of
                      Left e -> recordError "bot.combo.apply_failed" e sym interval
                      Right upd -> do
                        let argsChanged = bouArgs upd /= botArgs st || bouLookback upd /= botLookback st
                            ctxChanged = isJust (bouLstmCtx upd) || isJust (bouKalmanCtx upd)
                            shouldApply = argsChanged || ctxChanged
                        if shouldApply
                          then do
                            _ <- swapMVar pending (Just upd)
                            now <- getTimestampMs
                            opsAppendMaybe
                              mOps
                              "bot.optimizer.best"
                              Nothing
                              (Just (argsPublicJson (botArgs st)))
                              ( Just
                                  ( object
                                      [ "symbol" .= sym
                                      , "interval" .= interval
                                      , "finalEquity" .= bouFinalEquity upd
                                      , "objective" .= bouObjective upd
                                      , "score" .= bouScore upd
                                      , "appliedAtMs" .= now
                                      ]
                                  )
                              )
                              (bouFinalEquity upd)
                          else pure ()
            sleepSec pollSec
            loop

  loop

autoOptimizerLoop :: Args -> Maybe OpsStore -> Maybe Journal -> FilePath -> IO ()
autoOptimizerLoop baseArgs mOps mJournal optimizerTmp = do
  enabledEnv <- lookupEnv "TRADER_OPTIMIZER_ENABLED"
  let enabled = readEnvBool enabledEnv True
  if not enabled
    then pure ()
    else
      if argPlatform baseArgs /= PlatformBinance
        then do
          now <- getTimestampMs
          let msg :: String
              msg = "Auto optimizer supports Binance only."
          journalWriteMaybe
            mJournal
            ( object
                [ "type" .= ("optimizer.auto.platform_unsupported" :: String)
                , "atMs" .= now
                , "platform" .= platformCode (argPlatform baseArgs)
                , "error" .= msg
                ]
            )
          opsAppendMaybe
            mOps
            "optimizer.auto.platform_unsupported"
            Nothing
            (Just (argsPublicJson baseArgs))
            (Just (object ["error" .= msg]))
            Nothing
        else do
          projectRoot <- getCurrentDirectory
          topJsonPath <- resolveOptimizerCombosPath optimizerTmp
          optimizerExe <- resolveOptimizerExecutable projectRoot "optimize-equity"
          mergeExe <- resolveOptimizerExecutable projectRoot "merge-top-combos"
          let missing = [(name, err) | (name, Left err) <- [("optimize-equity", optimizerExe), ("merge-top-combos", mergeExe)]]
          if not (null missing)
            then do
              now <- getTimestampMs
              let missingNames = intercalate ", " (map fst missing)
                  missingErrs = intercalate "; " (map snd missing)
              journalWriteMaybe
                mJournal
                ( object
                    [ "type" .= ("optimizer.auto.missing_script" :: String)
                    , "atMs" .= now
                    , "script" .= missingNames
                    , "error" .= missingErrs
                    ]
                )
              opsAppendMaybe
                mOps
                "optimizer.auto.missing_script"
                Nothing
                Nothing
                (Just (object ["script" .= missingNames, "error" .= missingErrs]))
                Nothing
            else do
              envOrErr <- try (makeBinanceEnv mOps baseArgs) :: IO (Either SomeException BinanceEnv)
              case envOrErr of
                Left ex -> do
                  now <- getTimestampMs
                  journalWriteMaybe mJournal (object ["type" .= ("optimizer.auto.env_failed" :: String), "atMs" .= now, "error" .= show ex])
                Right env -> do
                  everySecEnv <- lookupEnv "TRADER_OPTIMIZER_EVERY_SEC"
                  trialsEnv <- lookupEnv "TRADER_OPTIMIZER_TRIALS"
                  timeoutEnv <- lookupEnv "TRADER_OPTIMIZER_TIMEOUT_SEC"
                  objectiveEnv <- lookupEnv "TRADER_OPTIMIZER_OBJECTIVE"
                  lookbackEnv <- lookupEnv "TRADER_OPTIMIZER_LOOKBACK_WINDOW"
                  backtestEnv <- lookupEnv "TRADER_OPTIMIZER_BACKTEST_RATIO"
                  tuneEnv <- lookupEnv "TRADER_OPTIMIZER_TUNE_RATIO"
                  maxPointsEnv <- lookupEnv "TRADER_OPTIMIZER_MAX_POINTS"
                  symbolsEnv <- lookupEnv "TRADER_OPTIMIZER_SYMBOLS"
                  intervalsEnv <- lookupEnv "TRADER_OPTIMIZER_INTERVALS"
                  maxCombos <- optimizerMaxCombosFromEnv
                  exePath <- getExecutablePath

                  let everySec :: Int
                      everySec =
                        case everySecEnv >>= readMaybe of
                          Just n | n >= 5 -> n
                          _ -> 300
                      trials :: Int
                      trials =
                        case trialsEnv >>= readMaybe of
                          Just n | n >= 1 -> n
                          _ -> 20
                      timeoutSec :: Int
                      timeoutSec =
                        case timeoutEnv >>= readMaybe of
                          Just n | n >= 1 -> n
                          _ -> 45
                      maxPoints :: Int
                      maxPoints =
                        case maxPointsEnv >>= readMaybe of
                          Just n | n >= 2 -> clampInt 2 1000 n
                          _ -> 1000
                      lookbackWindow = pickDefaultString "7d" lookbackEnv
                      backtestRatio =
                        case backtestEnv >>= readMaybe of
                          Just n -> clamp01 n
                          _ -> 0.2
                      tuneRatio =
                        case tuneEnv >>= readMaybe of
                          Just n -> clamp01 n
                          _ -> 0.25
                      objectiveAllowed = ["final-equity", "sharpe", "calmar", "equity-dd", "equity-dd-turnover"]
                      objectiveRaw = fmap (map toLower . trim) objectiveEnv
                      objective =
                        case objectiveRaw of
                          Just v | v `elem` objectiveAllowed -> v
                          _ -> "equity-dd-turnover"
                      symbols =
                        case symbolsEnv of
                          Just raw ->
                            let parsed = map normalizeSymbol (splitEnvList raw)
                             in if null parsed then defaultOptimizerSymbols else parsed
                          Nothing -> defaultOptimizerSymbols
                      intervalsRaw =
                        case intervalsEnv of
                          Just raw -> filter (isPlatformInterval PlatformBinance) (splitEnvList raw)
                          Nothing -> ["1h", "2h", "4h", "6h", "12h", "1d"]
                      intervals =
                        filter
                          (\v -> case lookbackBarsFrom v lookbackWindow of
                            Left _ -> False
                            Right lb -> lb >= 2 && lb + 3 <= maxPoints)
                          intervalsRaw
                  if null symbols || null intervals
                    then do
                      now <- getTimestampMs
                      journalWriteMaybe
                        mJournal
                        (object ["type" .= ("optimizer.auto.config_invalid" :: String), "atMs" .= now, "symbols" .= symbols, "intervals" .= intervals])
                    else do
                      putStrLn
                        ( printf
                            "Auto optimizer enabled: symbols=%d intervals=%d everySec=%d trials=%d"
                            (length symbols)
                            (length intervals)
                            everySec
                            trials
                        )
                      let sleepSec s = threadDelay (max 1 s * 1000000)

                          loop = do
                            sym <- pickRandom symbols
                            interval <- pickRandom intervals
                            let csvPath = optimizerTmp </> ("auto-" ++ sanitizeFileComponent sym ++ "-" ++ sanitizeFileComponent interval ++ ".csv")
                                argsSym = baseArgs { argBinanceSymbol = Just sym }

                            ksOrErr <- try (fetchKlines env sym interval maxPoints) :: IO (Either SomeException [Kline])
                            case ksOrErr of
                              Left ex -> do
                                now <- getTimestampMs
                                journalWriteMaybe mJournal (object ["type" .= ("optimizer.auto.fetch_failed" :: String), "atMs" .= now, "error" .= show ex])
                                sleepSec everySec
                                loop
                              Right ks -> do
                                if length ks < 2
                                  then do
                                    sleepSec everySec
                                    loop
                                  else do
                                    writeKlinesCsv csvPath ks
                                    ts <- fmap (floor . (* 1000)) getPOSIXTime
                                    randId <- randomIO :: IO Word64
                                    seedId <- randomIO :: IO Word64
                                    adoptOverride <-
                                      fmap
                                        (either (const Nothing) (adoptRequirementLongShortOverride argsSym))
                                        (resolveAdoptionRequirementWithEnv argsSym env sym)
                                    let recordsPath = optimizerTmp </> printf "optimizer-auto-%d-%016x.jsonl" (ts :: Integer) randId
                                        seed = fromIntegral (seedId `mod` 2000000000)
                                        cliArgs =
                                          [ "--data"
                                          , csvPath
                                          , "--price-column"
                                          , "close"
                                          , "--high-column"
                                          , "high"
                                          , "--low-column"
                                          , "low"
                                          , "--interval"
                                          , interval
                                          , "--lookback-window"
                                          , lookbackWindow
                                          , "--backtest-ratio"
                                          , show backtestRatio
                                          , "--tune-ratio"
                                          , show tuneRatio
                                          , "--trials"
                                          , show trials
                                          , "--timeout-sec"
                                          , show (timeoutSec :: Int)
                                          , "--seed"
                                          , show (seed :: Int)
                                          , "--output"
                                          , recordsPath
                                          , "--symbol-label"
                                          , sym
                                          , "--source-label"
                                          , "binance"
                                          , "--objective"
                                          , objective
                                          , "--binary"
                                          , exePath
                                          , "--disable-lstm-persistence"
                                          ]
                                        cliArgs' =
                                          case adoptOverride of
                                            Nothing -> cliArgs
                                            Just pLongShort -> cliArgs ++ ["--p-long-short", show pLongShort]

                                    runResult <- runOptimizerProcess projectRoot recordsPath cliArgs'
                                    case runResult of
                                      Left (msg, out, err) -> do
                                        now <- getTimestampMs
                                        journalWriteMaybe
                                          mJournal
                                          ( object
                                              [ "type" .= ("optimizer.auto.run_failed" :: String)
                                              , "atMs" .= now
                                              , "error" .= msg
                                              , "stdout" .= out
                                              , "stderr" .= err
                                              ]
                                          )
                                      Right _ -> do
                                        mergeResult <- runMergeTopCombos projectRoot topJsonPath recordsPath maxCombos
                                        case mergeResult of
                                          Left (msg, out, err) -> do
                                            now <- getTimestampMs
                                            journalWriteMaybe
                                              mJournal
                                              ( object
                                                  [ "type" .= ("optimizer.auto.merge_failed" :: String)
                                                  , "atMs" .= now
                                                  , "error" .= msg
                                                  , "stdout" .= out
                                                  , "stderr" .= err
                                                  ]
                                              )
                                          Right _ -> do
                                            now <- getTimestampMs
                                            opsAppendMaybe
                                              mOps
                                              "optimizer.auto.updated"
                                              Nothing
                                              (Just (object ["symbol" .= sym, "interval" .= interval]))
                                              Nothing
                                              Nothing
                                        _ <- try (removeFile recordsPath) :: IO (Either SomeException ())
                                        pure ()
                                    sleepSec everySec
                                    loop

                      loop

autoTopCombosBacktestLoop :: Args -> ApiComputeLimits -> BacktestGate -> Maybe OpsStore -> Maybe Journal -> FilePath -> IO ()
autoTopCombosBacktestLoop baseArgs limits backtestGate mOps mJournal optimizerTmp = do
  enabledEnv <- lookupEnv "TRADER_TOP_COMBOS_BACKTEST_ENABLED"
  let enabled = readEnvBool enabledEnv True
  if not enabled
    then pure ()
    else do
      topNEnv <- lookupEnv "TRADER_TOP_COMBOS_BACKTEST_TOP_N"
      everySecEnv <- lookupEnv "TRADER_TOP_COMBOS_BACKTEST_EVERY_SEC"
      let topN =
            case topNEnv >>= readMaybe of
              Just n | n >= 1 -> n
              _ -> 10
          everySec =
            case everySecEnv >>= readMaybe of
              Just n | n >= 60 -> n
              _ -> 86400
      topJsonPath <- resolveOptimizerCombosPath optimizerTmp
      putStrLn (printf "Top combos backtest enabled: topN=%d everySec=%d path=%s" topN everySec topJsonPath)
      let sleepSec s = threadDelay (max 1 s * 1000000)

          recordEvent :: String -> [(AK.Key, Aeson.Value)] -> IO ()
          recordEvent kind details = do
            now <- getTimestampMs
            journalWriteMaybe mJournal (object ("type" .= kind : "atMs" .= now : details))
            opsAppendMaybe mOps (T.pack kind) Nothing Nothing (Just (object details)) Nothing

          recordError :: String -> String -> Maybe String -> Maybe String -> IO ()
          recordError kind err sym interval = do
            let details =
                  [ "error" .= err
                  , "symbol" .= sym
                  , "interval" .= interval
                  , "path" .= topJsonPath
                  ]
            recordEvent kind details

          backtestCombo comboVal = do
            let mKey = comboIdentityKey comboVal
            case mKey of
              Nothing -> do
                recordError "optimizer.combos.backtest_failed" "Missing combo identity." Nothing Nothing
                pure Nothing
              Just key ->
                case Aeson.fromJSON comboVal :: Aeson.Result TopCombo of
                  Aeson.Error err -> do
                    recordError "optimizer.combos.backtest_failed" err Nothing Nothing
                    pure Nothing
                  Aeson.Success combo -> do
                    let mSymbol =
                          topComboParamString "binanceSymbol" combo
                            <|> topComboParamString "symbol" combo
                        mInterval = topComboParamString "interval" combo
                        mPlatformRaw = topComboParamString "platform" combo
                        platformOrErr =
                          case mPlatformRaw of
                            Nothing -> Right (argPlatform baseArgs)
                            Just raw ->
                              if normalizeKey raw == "csv"
                                then Left "CSV combos require --data."
                                else parsePlatform raw
                    case (mSymbol, platformOrErr) of
                      (Nothing, _) -> do
                        recordError "optimizer.combos.backtest_failed" "Missing combo symbol." Nothing mInterval
                        pure Nothing
                      (_, Left e) -> do
                        recordError "optimizer.combos.backtest_failed" e mSymbol mInterval
                        pure Nothing
                      (Just sym, Right platform) -> do
                        args0 <-
                          case applyTopComboForStart baseArgs combo of
                            Left e -> do
                              recordError "optimizer.combos.backtest_failed" e (Just sym) mInterval
                              pure Nothing
                            Right out -> pure (Just out)
                        case args0 of
                          Nothing -> pure Nothing
                          Just argsBase -> do
                            let args1 =
                                  argsBase
                                    { argBinanceSymbol = Just sym
                                    , argPlatform = platform
                                    , argData = Nothing
                                    , argTradeOnly = False
                                    , argBinanceTrade = False
                                    , argOptimizeOperations = False
                                    , argSweepThreshold = False
                                    }
                            case validateApiComputeLimits limits args1 of
                              Left e -> do
                                recordError "optimizer.combos.backtest_failed" e (Just sym) mInterval
                                pure Nothing
                              Right argsOk -> do
                                backtestResult <- runBacktestWithGate backtestGate (computeBacktestFromArgs mOps argsOk)
                                case backtestResult of
                                  Left failure ->
                                    case failure of
                                      BacktestBusy -> do
                                        recordEvent
                                          "optimizer.combos.backtest_skipped"
                                          [ "reason" .= ("busy" :: String)
                                          , "symbol" .= sym
                                          , "interval" .= mInterval
                                          , "path" .= topJsonPath
                                          ]
                                        pure Nothing
                                      _ -> do
                                        recordError "optimizer.combos.backtest_failed" (backtestFailureMessage backtestGate failure) (Just sym) mInterval
                                        pure Nothing
                                  Right out ->
                                    case extractBacktestMetrics out of
                                      Nothing -> do
                                        recordError "optimizer.combos.backtest_failed" "Backtest missing metrics." (Just sym) mInterval
                                        pure Nothing
                                      Just metricsVal -> do
                                        let mFinalEq = comboMetricDouble "finalEquity" metricsVal
                                            objective = fromMaybe "final-equity" (tcObjectiveLabel combo)
                                            mScore = objectiveScoreFromMetrics argsOk objective metricsVal
                                            mOps = extractBacktestOperations out
                                            update =
                                              ComboBacktestUpdate
                                                { cbuMetrics = metricsVal
                                                , cbuFinalEquity = mFinalEq
                                                , cbuScore = mScore
                                                , cbuOperations = mOps
                                                }
                                        recordEvent
                                          "optimizer.combos.backtest"
                                          ( [ "symbol" .= sym
                                            , "interval" .= mInterval
                                            , "finalEquity" .= mFinalEq
                                            , "score" .= mScore
                                            ]
                                              ++ maybe [] (\v -> ["objective" .= v]) (tcObjectiveLabel combo)
                                          )
                                        pure (Just (key, update))

          runOnce = do
            baseValOrErr <- readTopCombosValue topJsonPath
            case baseValOrErr of
              Left err -> recordError "optimizer.combos.backtest_failed" err Nothing Nothing
              Right baseVal ->
                case baseVal of
                  Aeson.Object o ->
                    case KM.lookup (AK.fromString "combos") o of
                      Just (Aeson.Array combos) -> do
                        let combosList = V.toList combos
                            ranked = take topN (sortOn (\(_, v) -> comboPerformanceKey v) (zip [0 ..] combosList))
                        updates <- fmap catMaybes (forM ranked (\(_, v) -> backtestCombo v))
                        if null updates
                          then recordEvent "optimizer.combos.backtest_skipped" ["reason" .= ("no updates" :: String)]
                          else do
                            latestValOrErr <- readTopCombosValue topJsonPath
                            now <- getTimestampMs
                            let latestVal = either (const baseVal) id latestValOrErr
                                updateMap = HM.fromList updates
                            case applyComboUpdates now updateMap latestVal of
                              Left err -> recordError "optimizer.combos.backtest_failed" err Nothing Nothing
                              Right (updatedVal, updatedCount) ->
                                if updatedCount <= 0
                                  then recordEvent "optimizer.combos.backtest_skipped" ["reason" .= ("no matching combos" :: String)]
                                  else do
                                    writeResult <- writeTopCombosValue topJsonPath updatedVal
                                    case writeResult of
                                      Left err -> recordError "optimizer.combos.backtest_failed" err Nothing Nothing
                                      Right _ -> do
                                        persistTopCombosMaybe topJsonPath
                                        recordEvent
                                          "optimizer.combos.backtest_updated"
                                          [ "updated" .= updatedCount
                                          , "topN" .= topN
                                          , "path" .= topJsonPath
                                          ]
                      _ -> recordError "optimizer.combos.backtest_failed" "Top combos JSON missing combos array." Nothing Nothing
                  _ -> recordError "optimizer.combos.backtest_failed" "Top combos JSON root must be an object." Nothing Nothing

          loop = do
            runOnce
            sleepSec everySec
            loop

      loop

makeBinanceEnv :: Maybe OpsStore -> Args -> IO BinanceEnv
makeBinanceEnv mOps args = do
  if argPlatform args /= PlatformBinance
    then error ("Binance environment requires platform=binance (got " ++ platformCode (argPlatform args) ++ ").")
    else pure ()
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
  newBinanceEnvWithOps mOps market base (BS.pack <$> apiKey) (BS.pack <$> apiSecret)

makeCoinbaseEnv :: Args -> IO CoinbaseEnv
makeCoinbaseEnv args = do
  if argPlatform args /= PlatformCoinbase
    then error ("Coinbase environment requires platform=coinbase (got " ++ platformCode (argPlatform args) ++ ").")
    else pure ()
  apiKey <- resolveEnv "COINBASE_API_KEY" (argCoinbaseApiKey args)
  apiSecret <- resolveEnv "COINBASE_API_SECRET" (argCoinbaseApiSecret args)
  apiPassphrase <- resolveEnv "COINBASE_API_PASSPHRASE" (argCoinbaseApiPassphrase args)
  newCoinbaseEnv (BS.pack <$> apiKey) (BS.pack <$> apiSecret) (BS.pack <$> apiPassphrase)

botLoop :: Maybe OpsStore -> Metrics -> Maybe Journal -> Maybe Webhook -> Maybe FilePath -> BotController -> MVar BotState -> MVar () -> Maybe (MVar (Maybe BotOptimizerUpdate)) -> IO ()
botLoop mOps metrics mJournal mWebhook mBotStateDir ctrl stVar stopSig mOptimizerPending = do
  tid <- myThreadId
  sym <- botSymbol <$> readMVar stVar
  lastStatusLogRef <- newIORef 0
  st0 <- readMVar stVar
  now0 <- getTimestampMs
  botStatusLogMaybe mOps True st0
  writeIORef lastStatusLogRef now0
  let sleepSec s = threadDelay (max 1 s * 1000000)
      logStatusIfDue st = do
        now <- getTimestampMs
        lastAt <- readIORef lastStatusLogRef
        if now - lastAt >= botStatusLogIntervalMs
          then do
            writeIORef lastStatusLogRef now
            botStatusLogMaybe mOps True st
          else pure ()

      loop = do
        stopReq <- isJust <$> tryReadMVar stopSig
        if stopReq
          then pure ()
          else do
            case mOptimizerPending of
              Nothing -> pure ()
              Just pending -> do
                mUpd <- modifyMVar pending $ \v -> pure (Nothing, v)
                case mUpd of
                  Nothing -> pure ()
                  Just upd -> do
                    st0 <- readMVar stVar
                    st1 <- botApplyOptimizerUpdate st0 upd
                    _ <- swapMVar stVar st1
                    persistBotStatusMaybe mBotStateDir st1
                    pure ()
            st <- readMVar stVar
            let env = botEnv st
                sym = botSymbol st
                pollSec = bsPollSeconds (botSettings st)
            t0 <- getTimestampMs
            r <- try (fetchKlines env sym (argInterval (botArgs st)) 10) :: IO (Either SomeException [Kline])
            t1 <- getTimestampMs
            let latMs = max 0 (fromIntegral (t1 - t0) :: Int)
            case r of
              Left ex -> do
                let st' =
                      st
                        { botError = Just (show ex)
                        , botUpdatedAtMs = t1
                        , botPolledAtMs = t1
                        , botPollLatencyMs = latMs
                        }
                _ <- swapMVar stVar st'
                persistBotStatusMaybe mBotStateDir st'
                logStatusIfDue st'
                sleepSec pollSec
                loop
              Right ks -> do
                let fetchedLast =
                      if null ks
                        then Nothing
                        else Just (last ks)
                    stPolled =
                      st
                        { botError = Nothing
                        , botPolledAtMs = t1
                        , botPollLatencyMs = latMs
                        , botFetchedKlines = length ks
                        , botFetchedLastKline = fetchedLast
                        }
                    lastSeen = botLastOpenTime stPolled
                    newKs = filter (\k -> kOpenTime k > lastSeen) ks
                if null newKs
                  then do
                    _ <- swapMVar stVar stPolled
                    persistBotStatusMaybe mBotStateDir stPolled
                    logStatusIfDue stPolled
                    sleepSec pollSec
                    loop
                  else do
                    tProc0 <- getTimestampMs
                    st1 <- foldl' (\ioAcc k -> ioAcc >>= \s0 -> botApplyKlineSafe mOps metrics mJournal mWebhook s0 k) (pure stPolled) newKs
                    tProc1 <- getTimestampMs
                    let batchMs = max 0 (fromIntegral (tProc1 - tProc0) :: Int)
                        batchSize = length newKs
                        st1' =
                          st1
                            { botLastBatchAtMs = tProc1
                            , botLastBatchSize = batchSize
                            , botLastBatchMs = batchMs
                            }
                    _ <- swapMVar stVar st1'
                    persistBotStatusMaybe mBotStateDir st1'
                    logStatusIfDue st1'
                    loop

      cleanup = do
        modifyMVar_ (bcRuntime ctrl) $ \mrt ->
          case HM.lookup sym mrt of
            Just (BotRunning rt) | brThreadId rt == tid -> do
              case brOptimizer rt of
                Nothing -> pure ()
                Just optRt -> do
                  _ <- tryPutMVar (borStopSignal optRt) ()
                  killThread (borThreadId optRt)
              pure (HM.delete sym mrt)
            _ -> pure mrt

  loop `finally` cleanup

botApplyKlineSafe :: Maybe OpsStore -> Metrics -> Maybe Journal -> Maybe Webhook -> BotState -> Kline -> IO BotState
botApplyKlineSafe mOps metrics mJournal mWebhook st k = do
  r <- try (botApplyKline mOps metrics mJournal mWebhook st k) :: IO (Either SomeException BotState)
  case r of
    Right st' -> pure st'
    Left ex -> do
      now <- getTimestampMs
      pure st { botError = Just (show ex), botUpdatedAtMs = now, botLastOpenTime = kOpenTime k }

botApplyKline :: Maybe OpsStore -> Metrics -> Maybe Journal -> Maybe Webhook -> BotState -> Kline -> IO BotState
botApplyKline mOps metrics mJournal mWebhook st k = do
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
      markToMarket side basePx px =
        if basePx == 0
          then 1
          else
            case side of
              SideLong -> px / basePx
              SideShort -> 2 - px / basePx
      eqAfterReturn =
        case prevPos of
          1 | prevPrice > 0 -> prevEq * markToMarket SideLong prevPrice priceNew
          (-1) | prevPrice > 0 -> prevEq * markToMarket SideShort prevPrice priceNew
          _ -> prevEq
      openTrade1 =
        case botOpenTrade st of
          Nothing -> Nothing
          Just ot ->
            let side = botOpenSide ot
                expectedPos =
                  case side of
                    SideLong -> 1
                    SideShort -> -1
             in
              if prevPos /= expectedPos
                then Nothing
                else
                  let trail1 =
                        case side of
                          SideLong -> max (botOpenTrail ot) priceNew
                          SideShort -> min (botOpenTrail ot) priceNew
                   in
                    Just
                      ot
                        { botOpenHoldingPeriods = botOpenHoldingPeriods ot + 1
                        , botOpenTrail = trail1
                        }
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
        mPath <- lstmWeightsPath args lookback
        savePersistedLstmModelMaybe mPath (length obsTrain) lstmModel1
        pure (Just (normState, obsAll', lstmModel1))

  let latest0Raw = computeLatestSignal args lookback pricesV mLstmCtx1 mKalmanCtx1 Nothing
      nan = 0 / 0 :: Double
      kalPred1 = V.snoc (botKalmanPredNext st) (maybe nan id (lsKalmanNext latest0Raw))
      lstmPred1 = V.snoc (botLstmPredNext st) (maybe nan id (lsLstmNext latest0Raw))

      -- Stateful decision:
      -- - Entry uses openThreshold (direction-agreement gated).
      -- - Exit/hold uses closeThreshold (hysteresis), matching backtest logic.
      closeDir = lsCloseDir latest0Raw
      wantLongClose = closeDir == Just 1
      wantShortClose = closeDir == Just (-1)
      allowShort = argPositioning args == LongShort

      desiredPosSignal =
        case prevPos of
          1 ->
            if wantLongClose
              then 1
              else
                case (allowShort, lsChosenDir latest0Raw) of
                  (True, Just (-1)) -> -1
                  _ -> 0
          (-1) ->
            if wantShortClose
              then -1
              else
                case (allowShort, lsChosenDir latest0Raw) of
                  (True, Just 1) -> 1
                  _ -> 0
          _ ->
            case lsChosenDir latest0Raw of
              Just 1 -> 1
              Just (-1) | allowShort -> -1
              _ -> 0

      -- If we decide to exit based on closeThreshold (even if openThreshold signal is neutral),
      -- force chosenDir to the exit side so a closing order can be placed when trading is enabled.
      latest0 =
        case (prevPos, desiredPosSignal) of
          (1, 0) ->
            if lsChosenDir latest0Raw /= Just (-1)
              then latest0Raw { lsChosenDir = Just (-1), lsAction = "FLAT (close)" }
              else latest0Raw
          (-1, 0) ->
            if lsChosenDir latest0Raw /= Just 1
              then latest0Raw { lsChosenDir = Just 1, lsAction = "FLAT (close)" }
              else latest0Raw
          _ -> latest0Raw

      volPerBar =
        case lsVolatility latest0Raw of
          Just vol ->
            let perBar = vol / sqrt (max 1e-12 (periodsPerYear args))
             in if isNaN perBar || isInfinite perBar || perBar <= 0 then Nothing else Just perBar
          _ -> Nothing

      clampFrac x = min 0.999999 (max 0 x)

      stopFromVol mult =
        if mult <= 0
          then Nothing
          else
            case volPerBar of
              Just v ->
                let frac = clampFrac (mult * v)
                 in if isNaN frac || isInfinite frac || frac <= 0 then Nothing else Just frac
              _ -> Nothing

      stopLoss0 =
        case stopFromVol (argStopLossVolMult args) of
          Just v -> Just v
          Nothing ->
            case argStopLoss args of
              Just v | v > 0 -> Just (clampFrac v)
              _ -> Nothing

      takeProfit0 =
        case stopFromVol (argTakeProfitVolMult args) of
          Just v -> Just v
          Nothing ->
            case argTakeProfit args of
              Just v | v > 0 -> Just (clampFrac v)
              _ -> Nothing

      trailingStop0 =
        case stopFromVol (argTrailingStopVolMult args) of
          Just v -> Just v
          Nothing ->
            case argTrailingStop args of
              Just v | v > 0 -> Just (clampFrac v)
              _ -> Nothing

      bracketExitReason side entryPx trail =
        let (tpHit, stopHit, stopWhy) =
              case side of
                SideLong ->
                  let mTp =
                        case takeProfit0 of
                          Just tp -> Just (entryPx * (1 + tp))
                          Nothing -> Nothing
                      mSl =
                        case stopLoss0 of
                          Just sl -> Just (entryPx * (1 - sl))
                          Nothing -> Nothing
                      mTs =
                        case trailingStop0 of
                          Just ts -> Just (trail * (1 - ts))
                          Nothing -> Nothing
                      (mStop, why) =
                        case (mSl, mTs) of
                          (Nothing, Nothing) -> (Nothing, Nothing)
                          (Just slPx, Nothing) -> (Just slPx, Just "STOP_LOSS")
                          (Nothing, Just tsPx) -> (Just tsPx, Just "TRAILING_STOP")
                          (Just slPx, Just tsPx) ->
                            if tsPx > slPx
                              then (Just tsPx, Just "TRAILING_STOP")
                              else (Just slPx, Just "STOP_LOSS")
                      tpOk = maybe False (\tpPx -> priceNew >= tpPx) mTp
                      stopOk = maybe False (\stPx -> priceNew <= stPx) mStop
                   in (tpOk, stopOk, why)
                SideShort ->
                  let mTp =
                        case takeProfit0 of
                          Just tp -> Just (entryPx * (1 - tp))
                          Nothing -> Nothing
                      mSl =
                        case stopLoss0 of
                          Just sl -> Just (entryPx * (1 + sl))
                          Nothing -> Nothing
                      mTs =
                        case trailingStop0 of
                          Just ts -> Just (trail * (1 + ts))
                          Nothing -> Nothing
                      (mStop, why) =
                        case (mSl, mTs) of
                          (Nothing, Nothing) -> (Nothing, Nothing)
                          (Just slPx, Nothing) -> (Just slPx, Just "STOP_LOSS")
                          (Nothing, Just tsPx) -> (Just tsPx, Just "TRAILING_STOP")
                          (Just slPx, Just tsPx) ->
                            if tsPx < slPx
                              then (Just tsPx, Just "TRAILING_STOP")
                              else (Just slPx, Just "STOP_LOSS")
                      tpOk = maybe False (\tpPx -> priceNew <= tpPx) mTp
                      stopOk = maybe False (\stPx -> priceNew >= stPx) mStop
                   in (tpOk, stopOk, why)
         in if tpHit then Just "TAKE_PROFIT" else if stopHit then stopWhy else Nothing

      mBracketExit =
        case openTrade1 of
          Just ot ->
            let side = botOpenSide ot
                expectedPos =
                  case side of
                    SideLong -> 1
                    SideShort -> -1
             in
              if prevPos == expectedPos
                then bracketExitReason side (botOpenEntryPrice ot) (botOpenTrail ot)
                else Nothing
          _ -> Nothing

      (latestPre, desiredPosPre, mExitReasonPre) =
        case mBracketExit of
          Just why ->
            let closeDir = if prevPos > 0 then Just (-1) else Just 1
                sigExit = latest0 { lsChosenDir = closeDir, lsAction = "EXIT_" ++ why }
             in (sigExit, 0, Just why)
          Nothing ->
            let exitReason =
                  if prevPos /= 0 && desiredPosSignal /= prevPos
                    then Just "SIGNAL"
                    else Nothing
             in (latest0, desiredPosSignal, exitReason)

      (latest0b, desiredPosWanted0b, mExitReason0b) =
        if halted
          then
            let why = haltReason1
                latestHalt =
                  case (prevPos, why) of
                    (1, Just r) -> latestPre { lsChosenDir = Just (-1), lsAction = "EXIT_" ++ r }
                    (-1, Just r) -> latestPre { lsChosenDir = Just 1, lsAction = "EXIT_" ++ r }
                    (0, Just r) -> latestPre { lsChosenDir = Nothing, lsAction = "HALTED_" ++ r }
                    _ -> latestPre
                exitReason =
                  case (prevPos, why) of
                    (1, Just r) -> Just r
                    (-1, Just r) -> Just r
                    _ -> mExitReasonPre
             in (latestHalt, 0, exitReason)
          else (latestPre, desiredPosPre, mExitReasonPre)

      holdBars =
        case openTrade1 of
          Nothing -> 0
          Just ot -> botOpenHoldingPeriods ot

      minHoldBars = max 0 (argMinHoldBars args)
      maxHoldBars =
        case argMaxHoldBars args of
          Just v | v > 0 -> Just v
          _ -> Nothing
      cooldownBars = max 0 (argCooldownBars args)
      cooldownLeft0 = max 0 (botCooldownLeft st)
      cooldownBlocked = prevPos == 0 && cooldownLeft0 > 0

      (latest1, desiredPosWanted1, mExitReason1) =
        if prevPos /= 0 && desiredPosWanted0b == 0 && mExitReason0b == Just "SIGNAL" && holdBars < minHoldBars
          then (latest0b { lsAction = "HOLD_MIN_HOLD" }, prevPos, Nothing)
          else (latest0b, desiredPosWanted0b, mExitReason0b)

      holdTooLong =
        case maxHoldBars of
          Nothing -> False
          Just lim -> prevPos /= 0 && holdBars >= lim && desiredPosWanted1 == prevPos

      (latest2, desiredPosWanted2, mExitReason2) =
        if holdTooLong
          then
            let closeDir = if prevPos > 0 then Just (-1) else Just 1
             in (latest1 { lsChosenDir = closeDir, lsAction = "EXIT_MAX_HOLD" }, 0, Just "MAX_HOLD")
          else (latest1, desiredPosWanted1, mExitReason1)

      (latest, desiredPosWanted, mExitReason) =
        if prevPos == 0 && desiredPosWanted2 /= 0 && cooldownBlocked
          then (latest2 { lsAction = "HOLD_COOLDOWN" }, 0, Nothing)
          else (latest2, desiredPosWanted2, mExitReason2)

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
        o <-
          if argPositioning args == LongShort && desiredPosWanted == 0 && prevPos /= 0
            then placeBotCloseIfEnabled args settings latest (botEnv st) (botSymbol st)
            else placeIfEnabled args settings latest (botEnv st) (botSymbol st)
        let opSide =
              if desiredPosWanted > prevPos
                then "BUY"
                else "SELL"
            orderEv = BotOrderEvent nPrev opSide priceNew openTimeNew now o
            ordersNew = botOrders st ++ [orderEv]
            tradeEnabled = bsTradeEnabled settings
            alreadyMsg =
              let msg = aorMessage o
               in "already long" `isInfixOf` msg || "already short" `isInfixOf` msg || "already flat" `isInfixOf` msg
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
            openTradeFor side eqEntry =
              BotOpenTrade
                { botOpenEntryIndex = nPrev
                , botOpenEntryEquity = eqEntry
                , botOpenHoldingPeriods = 0
                , botOpenEntryPrice = priceNew
                , botOpenTrail = priceNew
                , botOpenSide = side
                }
            closeTradeAt exitEq ot =
              Trade
                { trEntryIndex = botOpenEntryIndex ot
                , trExitIndex = nPrev
                , trEntryEquity = botOpenEntryEquity ot
                , trExitEquity = exitEq
                , trReturn = exitEq / botOpenEntryEquity ot - 1
                , trHoldingPeriods = botOpenHoldingPeriods ot
                , trExitReason = exitReasonFromCode <$> mExitReason
                }
            closeTrades =
              case openTrade1 of
                Nothing -> botTrades st
                Just ot -> botTrades st ++ [closeTradeAt eqAfterFee ot]
            (openTradeNew, tradesNew) =
              if not appliedSwitch
                then (openTrade1, botTrades st)
                else
                  case (prevPos, posNew) of
                    (0, 1) -> (Just (openTradeFor SideLong eqAfterFee), botTrades st)
                    (0, -1) -> (Just (openTradeFor SideShort eqAfterFee), botTrades st)
                    (1, 0) -> (Nothing, closeTrades)
                    (-1, 0) -> (Nothing, closeTrades)
                    (1, -1) -> (Just (openTradeFor SideShort eqAfterFee), closeTrades)
                    (-1, 1) -> (Just (openTradeFor SideLong eqAfterFee), closeTrades)
                    _ -> (openTrade1, botTrades st)
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
        webhookNotifyMaybe mWebhook (webhookEventBotOrder args (botSymbol st) opSide priceNew o)

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
      webhookNotifyMaybe mWebhook (webhookEventBotHalt args (botSymbol st) r drawdown dailyLoss orderErrors1)
    _ -> pure ()

  let cooldownDec =
        if prevPos == 0
          then max 0 (cooldownLeft0 - 1)
          else 0
      cooldownLeftNext =
        if posFinal == 0
          then if prevPos /= 0 then cooldownBars else cooldownDec
          else 0
      eqV1 = V.snoc (botEquityCurve st) eqFinal
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
                    Just ot -> Just (shiftOpenTradeIndex dropCount ot)
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
          , botCooldownLeft = cooldownLeftNext
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
    else placeOrderForSignalBot args sym sig env

placeBotCloseIfEnabled :: Args -> BotSettings -> LatestSignal -> BinanceEnv -> String -> IO ApiOrderResult
placeBotCloseIfEnabled args settings sig env sym =
  if not (bsTradeEnabled settings)
    then pure (ApiOrderResult False Nothing Nothing (Just sym) Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing "Paper mode: no order sent.")
    else placeBotCloseOrder args sym sig env

placeBotCloseOrder :: Args -> String -> LatestSignal -> BinanceEnv -> IO ApiOrderResult
placeBotCloseOrder args sym sig env =
  let args' = args { argPositioning = LongFlat }
      sig' = sig { lsChosenDir = Just (-1) }
   in placeOrderForSignalEx args' sym sig' env Nothing False

runRestApi :: Args -> Maybe Webhook -> IO ()
runRestApi baseArgs mWebhook = do
  mCommit <- getBuildCommit
  let buildInfo = BuildInfo traderVersion mCommit
  apiToken <- fmap BS.pack <$> lookupEnv "TRADER_API_TOKEN"
  timeoutEnv <- lookupEnv "TRADER_API_TIMEOUT_SEC"
  maxAsyncRunningEnv <- lookupEnv "TRADER_API_MAX_ASYNC_RUNNING"
  maxBacktestRunningEnv <- lookupEnv "TRADER_API_MAX_BACKTEST_RUNNING"
  backtestTimeoutEnv <- lookupEnv "TRADER_API_BACKTEST_TIMEOUT_SEC"
  cacheTtlEnv <- lookupEnv "TRADER_API_CACHE_TTL_MS"
  cacheMaxEnv <- lookupEnv "TRADER_API_CACHE_MAX_ENTRIES"
  maxBarsLstmEnv <- lookupEnv "TRADER_API_MAX_BARS_LSTM"
  maxEpochsEnv <- lookupEnv "TRADER_API_MAX_EPOCHS"
  maxHiddenSizeEnv <- lookupEnv "TRADER_API_MAX_HIDDEN_SIZE"
  maxBodyBytesEnv <- lookupEnv "TRADER_API_MAX_BODY_BYTES"
  maxOptimizerOutputBytesEnv <- lookupEnv "TRADER_API_MAX_OPTIMIZER_OUTPUT_BYTES"
  let timeoutSec =
        case timeoutEnv >>= readMaybe of
          Just n | n >= 0 -> n
          _ -> 1800
      maxAsyncRunning =
        case maxAsyncRunningEnv >>= readMaybe of
          Just n | n >= 1 -> n
          _ -> 1
      maxBacktestRunning =
        case maxBacktestRunningEnv >>= readMaybe of
          Just n | n >= 1 -> n
          _ -> 1
      backtestTimeoutSec =
        case backtestTimeoutEnv >>= readMaybe of
          Just n | n >= 30 -> n
          _ -> 900
      limits =
        ApiComputeLimits
          { aclMaxBarsLstm =
              case maxBarsLstmEnv >>= readMaybe of
                Just n | n >= 2 -> n
                _ -> 1000
          , aclMaxEpochs =
              case maxEpochsEnv >>= readMaybe of
                Just n | n >= 1 -> n
                _ -> 100
          , aclMaxHiddenSize =
              case maxHiddenSizeEnv >>= readMaybe of
                Just n | n >= 1 -> n
                _ -> 32
          }
      cacheTtlMs =
        case cacheTtlEnv >>= readMaybe of
          Just n | n >= 0 -> n
          _ -> 30000 :: Int64
      cacheMaxEntries =
        case cacheMaxEnv >>= readMaybe of
          Just n | n >= 0 -> n
          _ -> 64 :: Int
      maxBodyBytes =
        case (maxBodyBytesEnv >>= readMaybe :: Maybe Int64) of
          Just n | n > 0 -> n
          _ -> defaultApiMaxBodyBytes
      maxOptimizerOutputBytes =
        case (maxOptimizerOutputBytesEnv >>= readMaybe :: Maybe Int) of
          Just n | n > 0 -> n
          _ -> defaultApiMaxOptimizerOutputBytes
      reqLimits =
        ApiRequestLimits
          { arlMaxBodyBytes = maxBodyBytes
          , arlMaxOptimizerOutputBytes = maxOptimizerOutputBytes
          , arlMaxBotStatusTail = defaultApiMaxBotStatusTail
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

  let bindHostStr = "0.0.0.0" :: String
      bindHost = ("0.0.0.0" :: Warp.HostPreference)
      displayHost = "127.0.0.1" :: String
      port = max 1 (argPort baseArgs)
      settings =
        Warp.setHost bindHost $
          Warp.setTimeout timeoutSec $
          Warp.setPort port Warp.defaultSettings
  putStrLn (printf "Build: %s%s" (biVersion buildInfo) (maybe "" (\c -> " (" ++ take 12 c ++ ")") (biCommit buildInfo)))
  putStrLn (printf "Starting REST API on http://%s:%d (bind: %s:%d)" displayHost port bindHostStr port)
  putStrLn
    ( printf
        "API limits: maxAsyncRunning=%d, maxBacktestRunning=%d, maxBarsLstm=%d, maxEpochs=%d, maxHiddenSize=%d"
        maxAsyncRunning
        maxBacktestRunning
        (aclMaxBarsLstm limits)
        (aclMaxEpochs limits)
        (aclMaxHiddenSize limits)
    )
  putStrLn
    ( printf
        "API backtest gate: maxRunning=%d timeoutSec=%d"
        maxBacktestRunning
        backtestTimeoutSec
    )
  putStrLn
    ( printf
        "API request limits: maxBodyBytes=%d, maxOptimizerOutputBytes=%d, botStatusTailMax=%d"
        (arlMaxBodyBytes reqLimits)
        (arlMaxOptimizerOutputBytes reqLimits)
        (arlMaxBotStatusTail reqLimits)
    )
  apiCache <- newApiCache cacheMaxEntries cacheTtlMs
  putStrLn
    ( printf
        "API cache: ttlMs=%d maxEntries=%d"
        cacheTtlMs
        cacheMaxEntries
    )
  projectRoot <- getCurrentDirectory
  mStateDir <- stateDirFromEnv
  let tmpRoot = projectRoot </> ".tmp"
  createDirectoryIfMissing True tmpRoot
  let optimizerTmp = fromMaybe (tmpRoot </> "optimizer") (fmap (</> "optimizer") mStateDir)
  createDirectoryIfMissing True optimizerTmp
  metrics <- newMetrics
  mJournal <- newJournalFromEnv
  mOps <- newOpsStoreFromEnv
  mBotStateDir <- resolveBotStateDir
  backtestGate <- newBacktestGate maxBacktestRunning backtestTimeoutSec
  let defaultAsyncDir = fromMaybe (tmpRoot </> "async") (fmap (</> "async") mStateDir)
  asyncDirEnv <- lookupEnv "TRADER_API_ASYNC_DIR"
  let asyncDirTrimmed = trim <$> asyncDirEnv
      mAsyncDir =
        case asyncDirTrimmed of
          Nothing -> Just defaultAsyncDir
          Just dir | null dir -> Nothing
          Just dir -> Just dir
      asyncDirFromEnv =
        case asyncDirTrimmed of
          Nothing -> False
          Just dir -> not (null dir)
      asyncDirFromState = isNothing asyncDirTrimmed && isJust mStateDir
  case mAsyncDir of
    Nothing -> pure ()
    Just dir -> do
      let suffix :: String
          suffix
            | asyncDirFromEnv = ""
            | asyncDirFromState = " (from TRADER_STATE_DIR; set TRADER_API_ASYNC_DIR to override)"
            | otherwise = " (default; set TRADER_API_ASYNC_DIR to override)"
      putStrLn (printf "Async job persistence enabled: %s%s" dir suffix)
  now <- getTimestampMs
  journalWriteMaybe mJournal (object ["type" .= ("server.start" :: String), "atMs" .= now, "port" .= port])
  opsAppendMaybe mOps "server.start" Nothing Nothing (Just (object ["port" .= port])) Nothing
  bot <- newBotController
  _ <- forkIO (botAutoStartLoop mOps metrics mJournal mWebhook mBotStateDir optimizerTmp limits baseArgs bot)
  _ <- forkIO (autoOptimizerLoop baseArgs mOps mJournal optimizerTmp)
  _ <- forkIO (autoTopCombosBacktestLoop baseArgs limits backtestGate mOps mJournal optimizerTmp)
  asyncSignal <- newJobStore "signal" maxAsyncRunning mAsyncDir
  asyncBacktest <- newJobStore "backtest" maxAsyncRunning mAsyncDir
  asyncTrade <- newJobStore "trade" maxAsyncRunning mAsyncDir
  hFlush stdout
  res <-
    ( try
        (Warp.runSettings settings (apiApp buildInfo baseArgs apiToken bot metrics mJournal mWebhook mOps mBotStateDir limits reqLimits apiCache backtestGate (AsyncStores asyncSignal asyncBacktest asyncTrade) projectRoot optimizerTmp)) ::
        IO (Either IOException ())
    )
  case res of
    Right () -> pure ()
    Left e ->
      ioError
        ( userError
            ( printf
                "Failed to start REST API on %s:%d: %s\nTry a different --port (or check permissions / sandbox restrictions)."
                bindHostStr
                port
                (show e)
            )
        )

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

data CacheEntry a = CacheEntry
  { ceCreatedAtMs :: !Int64
  , ceLastAccessAtMs :: !Int64
  , ceValue :: !a
  }

data ApiCache = ApiCache
  { acTtlMs :: !Int64
  , acMaxEntries :: !Int
  , acSignals :: !(MVar (HM.HashMap Text (CacheEntry LatestSignal)))
  , acBacktests :: !(MVar (HM.HashMap Text (CacheEntry Aeson.Value)))
  , acSignalHits :: !(IORef Int64)
  , acSignalMisses :: !(IORef Int64)
  , acBacktestHits :: !(IORef Int64)
  , acBacktestMisses :: !(IORef Int64)
  }

newApiCache :: Int -> Int64 -> IO ApiCache
newApiCache maxEntries ttlMs = do
  sig <- newMVar HM.empty
  bt <- newMVar HM.empty
  sigHits <- newIORef 0
  sigMiss <- newIORef 0
  btHits <- newIORef 0
  btMiss <- newIORef 0
  pure
    ApiCache
      { acTtlMs = ttlMs
      , acMaxEntries = maxEntries
      , acSignals = sig
      , acBacktests = bt
      , acSignalHits = sigHits
      , acSignalMisses = sigMiss
      , acBacktestHits = btHits
      , acBacktestMisses = btMiss
      }

cacheEnabled :: ApiCache -> Bool
cacheEnabled c = acTtlMs c > 0 && acMaxEntries c > 0

requestWantsNoCache :: Wai.Request -> Bool
requestWantsNoCache req =
  let qs = Wai.queryString req
      nocacheQs =
        case lookup "nocache" qs of
          Nothing -> False
          Just Nothing -> True
          Just (Just raw) ->
            let s = map toLower (BS.unpack raw)
             in s == "1" || s == "true" || s == "yes" || s == "on"

      hasNoCacheDirective raw =
        let v = BS.map toLower raw
         in "no-cache" `BS.isInfixOf` v || "no-store" `BS.isInfixOf` v

      hdrs = Wai.requestHeaders req
      cc = lookup hCacheControl hdrs
      pragma = lookup hPragma hdrs
      nocacheHdr =
        maybe False hasNoCacheDirective cc
          || maybe False hasNoCacheDirective pragma
   in nocacheQs || nocacheHdr

barsResolvedForCache :: Args -> Int
barsResolvedForCache args =
  case (argBinanceSymbol args, argData args) of
    (Just _, _) -> resolveBarsForPlatform args
    (_, Just _) -> resolveBarsForCsv args
    _ -> fromMaybe 0 (argBars args)

argsCacheJsonSignal :: Args -> Aeson.Value
argsCacheJsonSignal args =
  let market = marketCode (argBinanceMarket args)
      barsResolved = barsResolvedForCache args
      lookbackResolved = argLookback args
   in
    object
      [ "data" .= argData args
      , "priceColumn" .= argPriceCol args
      , "highColumn" .= argHighCol args
      , "lowColumn" .= argLowCol args
      , "binanceSymbol" .= argBinanceSymbol args
      , "platform" .= platformCode (argPlatform args)
      , "market" .= market
      , "interval" .= argInterval args
      , "bars" .= barsResolved
      , "lookbackBars" .= lookbackResolved
      , "binanceTestnet" .= argBinanceTestnet args
      , "normalization" .= show (argNormalization args)
      , "hiddenSize" .= argHiddenSize args
      , "epochs" .= argEpochs args
      , "lr" .= argLr args
      , "valRatio" .= argValRatio args
      , "patience" .= argPatience args
      , "gradClip" .= argGradClip args
      , "seed" .= argSeed args
      , "kalmanDt" .= argKalmanDt args
      , "kalmanProcessVar" .= argKalmanProcessVar args
      , "kalmanMeasurementVar" .= argKalmanMeasurementVar args
      , "kalmanMarketTopN" .= argKalmanMarketTopN args
      , "openThreshold" .= argOpenThreshold args
      , "closeThreshold" .= argCloseThreshold args
      , "method" .= methodCode (argMethod args)
      , "positioning" .= positioningCode (argPositioning args)
      , "maxHoldBars" .= argMaxHoldBars args
      , "minEdge" .= argMinEdge args
      , "minSignalToNoise" .= argMinSignalToNoise args
      , "costAwareEdge" .= argCostAwareEdge args
      , "edgeBuffer" .= argEdgeBuffer args
      , "trendLookback" .= argTrendLookback args
      , "maxPositionSize" .= argMaxPositionSize args
      , "volTarget" .= argVolTarget args
      , "volLookback" .= argVolLookback args
      , "volEwmaAlpha" .= argVolEwmaAlpha args
      , "volFloor" .= argVolFloor args
      , "volScaleMax" .= argVolScaleMax args
      , "maxVolatility" .= argMaxVolatility args
      , "blendWeight" .= argBlendWeight args
      , "triLayer" .= argTriLayer args
      , "triLayerFastMult" .= argTriLayerFastMult args
      , "triLayerSlowMult" .= argTriLayerSlowMult args
      , "kalmanZMin" .= argKalmanZMin args
      , "kalmanZMax" .= argKalmanZMax args
      , "maxHighVolProb" .= argMaxHighVolProb args
      , "maxConformalWidth" .= argMaxConformalWidth args
      , "maxQuantileWidth" .= argMaxQuantileWidth args
      , "confirmConformal" .= argConfirmConformal args
      , "confirmQuantiles" .= argConfirmQuantiles args
      , "confidenceSizing" .= argConfidenceSizing args
      , "minPositionSize" .= argMinPositionSize args
      ]

argsCacheJsonBacktest :: Args -> Aeson.Value
argsCacheJsonBacktest args =
  let market = marketCode (argBinanceMarket args)
      barsResolved = barsResolvedForCache args
      lookbackResolved = argLookback args
   in
    object
      [ "data" .= argData args
      , "priceColumn" .= argPriceCol args
      , "highColumn" .= argHighCol args
      , "lowColumn" .= argLowCol args
      , "binanceSymbol" .= argBinanceSymbol args
      , "platform" .= platformCode (argPlatform args)
      , "market" .= market
      , "interval" .= argInterval args
      , "bars" .= barsResolved
      , "lookbackBars" .= lookbackResolved
      , "binanceTestnet" .= argBinanceTestnet args
      , "normalization" .= show (argNormalization args)
      , "hiddenSize" .= argHiddenSize args
      , "epochs" .= argEpochs args
      , "lr" .= argLr args
      , "valRatio" .= argValRatio args
      , "patience" .= argPatience args
      , "gradClip" .= argGradClip args
      , "seed" .= argSeed args
      , "kalmanDt" .= argKalmanDt args
      , "kalmanProcessVar" .= argKalmanProcessVar args
      , "kalmanMeasurementVar" .= argKalmanMeasurementVar args
      , "kalmanMarketTopN" .= argKalmanMarketTopN args
      , "openThreshold" .= argOpenThreshold args
      , "closeThreshold" .= argCloseThreshold args
      , "method" .= methodCode (argMethod args)
      , "positioning" .= positioningCode (argPositioning args)
      , "backtestRatio" .= argBacktestRatio args
      , "tuneRatio" .= argTuneRatio args
      , "tuneObjective" .= tuneObjectiveCode (argTuneObjective args)
      , "tunePenaltyMaxDrawdown" .= argTunePenaltyMaxDrawdown args
      , "tunePenaltyTurnover" .= argTunePenaltyTurnover args
      , "tuneStressVolMult" .= argTuneStressVolMult args
      , "tuneStressShock" .= argTuneStressShock args
      , "tuneStressWeight" .= argTuneStressWeight args
      , "minRoundTrips" .= argMinRoundTrips args
      , "walkForwardFolds" .= argWalkForwardFolds args
      , "optimizeOperations" .= argOptimizeOperations args
      , "sweepThreshold" .= argSweepThreshold args
      , "fee" .= argFee args
      , "slippage" .= argSlippage args
      , "spread" .= argSpread args
      , "intrabarFill" .= intrabarFillCode (argIntrabarFill args)
      , "stopLoss" .= argStopLoss args
      , "takeProfit" .= argTakeProfit args
      , "trailingStop" .= argTrailingStop args
      , "stopLossVolMult" .= argStopLossVolMult args
      , "takeProfitVolMult" .= argTakeProfitVolMult args
      , "trailingStopVolMult" .= argTrailingStopVolMult args
      , "minHoldBars" .= argMinHoldBars args
      , "cooldownBars" .= argCooldownBars args
      , "maxHoldBars" .= argMaxHoldBars args
      , "maxDrawdown" .= argMaxDrawdown args
      , "maxDailyLoss" .= argMaxDailyLoss args
      , "minEdge" .= argMinEdge args
      , "minSignalToNoise" .= argMinSignalToNoise args
      , "costAwareEdge" .= argCostAwareEdge args
      , "edgeBuffer" .= argEdgeBuffer args
      , "trendLookback" .= argTrendLookback args
      , "maxPositionSize" .= argMaxPositionSize args
      , "volTarget" .= argVolTarget args
      , "volLookback" .= argVolLookback args
      , "volEwmaAlpha" .= argVolEwmaAlpha args
      , "volFloor" .= argVolFloor args
      , "volScaleMax" .= argVolScaleMax args
      , "maxVolatility" .= argMaxVolatility args
      , "rebalanceBars" .= argRebalanceBars args
      , "rebalanceThreshold" .= argRebalanceThreshold args
      , "rebalanceGlobal" .= argRebalanceGlobal args
      , "fundingRate" .= argFundingRate args
      , "fundingBySide" .= argFundingBySide args
      , "rebalanceResetOnSignal" .= argRebalanceResetOnSignal args
      , "fundingOnOpen" .= argFundingOnOpen args
      , "blendWeight" .= argBlendWeight args
      , "triLayer" .= argTriLayer args
      , "triLayerFastMult" .= argTriLayerFastMult args
      , "triLayerSlowMult" .= argTriLayerSlowMult args
      , "maxOrderErrors" .= argMaxOrderErrors args
      , "periodsPerYear" .= argPeriodsPerYear args
      , "kalmanZMin" .= argKalmanZMin args
      , "kalmanZMax" .= argKalmanZMax args
      , "maxHighVolProb" .= argMaxHighVolProb args
      , "maxConformalWidth" .= argMaxConformalWidth args
      , "maxQuantileWidth" .= argMaxQuantileWidth args
      , "confirmConformal" .= argConfirmConformal args
      , "confirmQuantiles" .= argConfirmQuantiles args
      , "confidenceSizing" .= argConfidenceSizing args
      , "minPositionSize" .= argMinPositionSize args
      ]

cacheKeyForArgs :: Text -> (Args -> Aeson.Value) -> Args -> Text
cacheKeyForArgs prefix toVal args =
  let payload = encode (toVal args)
      hex = hashBytesHex payload
   in prefix <> ":" <> T.pack hex

evictLRU :: Int -> HM.HashMap Text (CacheEntry a) -> HM.HashMap Text (CacheEntry a)
evictLRU maxN hm =
  let n = HM.size hm
   in if maxN <= 0 || n <= maxN
        then hm
        else
          let pickOldest :: [(Text, CacheEntry a)] -> Maybe Text
              pickOldest xs =
                case xs of
                  [] -> Nothing
                  (k, e) : rest ->
                    let (_, bestK) =
                          foldl'
                            (\(bestT, bestKey) (k2, e2) -> if ceLastAccessAtMs e2 < bestT then (ceLastAccessAtMs e2, k2) else (bestT, bestKey))
                            (ceLastAccessAtMs e, k)
                            rest
                     in Just bestK
              hm1 =
                case pickOldest (HM.toList hm) of
                  Nothing -> hm
                  Just k -> HM.delete k hm
           in evictLRU maxN hm1

cacheLookup :: ApiCache -> MVar (HM.HashMap Text (CacheEntry a)) -> Text -> IO (Maybe a)
cacheLookup cache store key =
  if not (cacheEnabled cache)
    then pure Nothing
    else
      modifyMVar store $ \hm -> do
        now <- getTimestampMs
        let ttl = acTtlMs cache
            fresh e = now - ceCreatedAtMs e <= ttl
            hmFresh = HM.filter fresh hm
        case HM.lookup key hmFresh of
          Nothing -> pure (hmFresh, Nothing)
          Just e ->
            let e' = e { ceLastAccessAtMs = now }
             in pure (HM.insert key e' hmFresh, Just (ceValue e))

cacheInsert :: ApiCache -> MVar (HM.HashMap Text (CacheEntry a)) -> Text -> a -> IO ()
cacheInsert cache store key val =
  if not (cacheEnabled cache)
    then pure ()
    else
      modifyMVar_ store $ \hm -> do
        now <- getTimestampMs
        let ttl = acTtlMs cache
            fresh e = now - ceCreatedAtMs e <= ttl
            hmFresh = HM.filter fresh hm
            e = CacheEntry { ceCreatedAtMs = now, ceLastAccessAtMs = now, ceValue = val }
            hm' = HM.insert key e hmFresh
        pure (evictLRU (acMaxEntries cache) hm')

cachePruneSize :: ApiCache -> MVar (HM.HashMap Text (CacheEntry a)) -> IO Int
cachePruneSize cache store =
  if not (cacheEnabled cache)
    then pure 0
    else
      modifyMVar store $ \hm -> do
        now <- getTimestampMs
        let ttl = acTtlMs cache
            fresh e = now - ceCreatedAtMs e <= ttl
            hmFresh = HM.filter fresh hm
        pure (hmFresh, HM.size hmFresh)

apiCacheStatsJson :: ApiCache -> IO Aeson.Value
apiCacheStatsJson cache = do
  now <- getTimestampMs
  sigEntries <- cachePruneSize cache (acSignals cache)
  btEntries <- cachePruneSize cache (acBacktests cache)
  sigHits <- readIORef (acSignalHits cache)
  sigMiss <- readIORef (acSignalMisses cache)
  btHits <- readIORef (acBacktestHits cache)
  btMiss <- readIORef (acBacktestMisses cache)
  pure
    ( object
        [ "enabled" .= cacheEnabled cache
        , "ttlMs" .= acTtlMs cache
        , "maxEntries" .= acMaxEntries cache
        , "signals"
            .= object
              [ "entries" .= sigEntries
              , "hits" .= sigHits
              , "misses" .= sigMiss
              ]
        , "backtests"
            .= object
              [ "entries" .= btEntries
              , "hits" .= btHits
              , "misses" .= btMiss
              ]
        , "atMs" .= now
        ]
    )

apiCacheClear :: ApiCache -> IO ()
apiCacheClear cache = do
  modifyMVar_ (acSignals cache) (const (pure HM.empty))
  modifyMVar_ (acBacktests cache) (const (pure HM.empty))
  writeIORef (acSignalHits cache) 0
  writeIORef (acSignalMisses cache) 0
  writeIORef (acBacktestHits cache) 0
  writeIORef (acBacktestMisses cache) 0

computeLatestSignalFromArgsCached :: ApiCache -> Maybe OpsStore -> Args -> IO LatestSignal
computeLatestSignalFromArgsCached cache mOps args = do
  if not (cacheEnabled cache)
    then computeLatestSignalFromArgs mOps args
    else do
      let key = cacheKeyForArgs "signal" argsCacheJsonSignal args
      mHit <- cacheLookup cache (acSignals cache) key
      case mHit of
        Just v -> do
          incCounter (acSignalHits cache)
          pure v
        Nothing -> do
          incCounter (acSignalMisses cache)
          v <- computeLatestSignalFromArgs mOps args
          cacheInsert cache (acSignals cache) key v
          pure v

computeBacktestFromArgsCached :: ApiCache -> Maybe OpsStore -> Args -> IO Aeson.Value
computeBacktestFromArgsCached cache mOps args = do
  if not (cacheEnabled cache)
    then computeBacktestFromArgs mOps args
    else do
      let key = cacheKeyForArgs "backtest" argsCacheJsonBacktest args
      mHit <- cacheLookup cache (acBacktests cache) key
      case mHit of
        Just v -> do
          incCounter (acBacktestHits cache)
          pure v
        Nothing -> do
          incCounter (acBacktestMisses cache)
          v <- computeBacktestFromArgs mOps args
          cacheInsert cache (acBacktests cache) key v
          pure v

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
    Just dir -> do
      createDirectoryIfMissing True dir
      validateWritableDir dir
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

validateWritableDir :: FilePath -> IO ()
validateWritableDir dir = do
  now <- getTimestampMs
  r <- (randomIO :: IO Word64)
  let probe = dir </> (".probe-" ++ show now ++ "-" ++ printf "%016x" r)
  e <- try (BS.writeFile probe (BS.pack "ok")) :: IO (Either SomeException ())
  case e of
    Left ex ->
      die
        ( "TRADER_API_ASYNC_DIR is set but not writable ("
            ++ dir
            ++ "): "
            ++ show ex
        )
    Right _ -> do
      _ <- try (removeFile probe) :: IO (Either SomeException ())
      pure ()

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
      e <-
        try
          ( do
              createDirectoryIfMissing True (takeDirectory path)
              BL.writeFile tmp (encode payload)
              renameFile tmp path
          )
          :: IO (Either SomeException ())
      case e of
        Left ex -> hPutStrLn stderr (printf "WARN: failed to persist async job %s (%s): %s" (T.unpack jobId) path (show ex))
        Right _ -> pure ()

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
  pruneJobStoreDisk store now

  let maxRunning = max 1 (jsMaxRunning store)
  (running, ok) <-
    modifyMVar (jsRunning store) $ \n ->
      if n >= maxRunning
        then pure (n, (n, False))
        else pure (n + 1, (n, True))
  if not ok
    then
      pure
        ( Left
            ( printf
                "Async %s queue is full (%d/%d). Wait for the current job to finish/cancel, or increase TRADER_API_MAX_ASYNC_RUNNING."
                (T.unpack (jsPrefix store))
                running
                maxRunning
            )
        )
    else do
      n <- atomicModifyIORef' (jsCounter store) (\x -> let y = x + 1 in (y, y))
      r <- (randomIO :: IO Word64)
      let jobId =
            jsPrefix store
              <> "-"
              <> T.pack (show now)
              <> "-"
              <> T.pack (show n)
              <> "-"
              <> T.pack (printf "%016x" r)
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

data ApiRequestLimits = ApiRequestLimits
  { arlMaxBodyBytes :: !Int64
  , arlMaxOptimizerOutputBytes :: !Int
  , arlMaxBotStatusTail :: !Int
  } deriving (Eq, Show)

data BacktestGate = BacktestGate
  { btRunning :: !(MVar Int)
  , btMaxRunning :: !Int
  , btTimeoutSec :: !Int
  }

data BacktestFailure
  = BacktestBusy
  | BacktestTimedOut
  | BacktestException SomeException
  deriving (Show)

newBacktestGate :: Int -> Int -> IO BacktestGate
newBacktestGate maxRunning timeoutSec = do
  running <- newMVar 0
  pure BacktestGate { btRunning = running, btMaxRunning = max 1 maxRunning, btTimeoutSec = max 1 timeoutSec }

runBacktestWithGate :: BacktestGate -> IO a -> IO (Either BacktestFailure a)
runBacktestWithGate gate action = do
  let maxRunning = max 1 (btMaxRunning gate)
  acquired <-
    modifyMVar (btRunning gate) $ \n ->
      if n >= maxRunning
        then pure (n, False)
        else pure (n + 1, True)
  if not acquired
    then pure (Left BacktestBusy)
    else do
      let release = modifyMVar_ (btRunning gate) (pure . max 0 . subtract 1)
      result <- timeout (btTimeoutSec gate * 1000000) (try action) `finally` release
      case result of
        Nothing -> pure (Left BacktestTimedOut)
        Just (Left ex) -> pure (Left (BacktestException ex))
        Just (Right v) -> pure (Right v)

runBacktestWithGateWait :: BacktestGate -> IO a -> IO (Either BacktestFailure a)
runBacktestWithGateWait gate action = do
  let maxRunning = max 1 (btMaxRunning gate)
      tryAcquire =
        modifyMVar (btRunning gate) $ \n ->
          if n >= maxRunning
            then pure (n, False)
            else pure (n + 1, True)
      waitLoop = do
        acquired <- tryAcquire
        if acquired
          then pure ()
          else do
            threadDelay 1000000
            waitLoop
  waitLoop
  let release = modifyMVar_ (btRunning gate) (pure . max 0 . subtract 1)
  result <- timeout (btTimeoutSec gate * 1000000) (try action) `finally` release
  case result of
    Nothing -> pure (Left BacktestTimedOut)
    Just (Left ex) -> pure (Left (BacktestException ex))
    Just (Right v) -> pure (Right v)

backtestFailureMessage :: BacktestGate -> BacktestFailure -> String
backtestFailureMessage gate failure =
  case failure of
    BacktestBusy ->
      "Backtest queue is busy. Wait for the current job to finish, or increase TRADER_API_MAX_BACKTEST_RUNNING."
    BacktestTimedOut ->
      printf
        "Backtest timed out after %ds. Increase TRADER_API_BACKTEST_TIMEOUT_SEC if needed."
        (btTimeoutSec gate)
    BacktestException ex -> snd (exceptionToHttp ex)

backtestFailureToHttp :: BacktestGate -> BacktestFailure -> (Status, String)
backtestFailureToHttp gate failure =
  case failure of
    BacktestBusy -> (status429, backtestFailureMessage gate failure)
    BacktestTimedOut -> (status504, backtestFailureMessage gate failure)
    BacktestException ex -> exceptionToHttp ex

defaultApiMaxBodyBytes :: Int64
defaultApiMaxBodyBytes = 1024 * 1024

defaultApiMaxOptimizerOutputBytes :: Int
defaultApiMaxOptimizerOutputBytes = 20000

defaultApiMaxBotStatusTail :: Int
defaultApiMaxBotStatusTail = 5000

validateApiComputeLimits :: ApiComputeLimits -> Args -> Either String Args
validateApiComputeLimits limits args =
  case argMethod args of
    MethodKalmanOnly -> Right args
    _ -> do
      let maxBars = aclMaxBarsLstm limits
          bars =
            case (argBinanceSymbol args, argData args) of
              (Just _, _) -> resolveBarsForPlatform args
              (_, Just _) -> resolveBarsForCsv args
              _ -> fromMaybe 0 (argBars args)
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

apiApp ::
  BuildInfo ->
  Args ->
  Maybe BS.ByteString ->
  BotController ->
  Metrics ->
  Maybe Journal ->
  Maybe Webhook ->
  Maybe OpsStore ->
  Maybe FilePath ->
  ApiComputeLimits ->
  ApiRequestLimits ->
  ApiCache ->
  BacktestGate ->
  AsyncStores ->
  FilePath ->
  FilePath ->
  Wai.Application
apiApp buildInfo baseArgs apiToken botCtrl metrics mJournal mWebhook mOps mBotStateDir limits reqLimits apiCache backtestGate asyncStores projectRoot optimizerTmp req respond = do
  startMs <- getTimestampMs
  let rawPath = Wai.pathInfo req
      path =
        case rawPath of
          ("api" : rest) -> rest
          _ -> rawPath
      method = BS.unpack (Wai.requestMethod req)
      pathLabel =
        case path of
          [] -> "/"
          _ -> "/" ++ intercalate "/" (map T.unpack path)
      pathIsHealth = path == ["health"]
      respondLogged resp = do
        endMs <- getTimestampMs
        let code = statusCode (Wai.responseStatus resp)
            durMs = max 0 (endMs - startMs)
            shouldLog = method /= "OPTIONS" && not (pathIsHealth && code < 400)
        when shouldLog $
          putStrLn (printf "Request %s %s -> %d (%dms)" method pathLabel code durMs)
        respond resp
      respondCors = respondLogged . withCors
      label =
        case path of
          ["signal", "async", _, "cancel"] -> "signal/async/:jobId/cancel"
          ["backtest", "async", _, "cancel"] -> "backtest/async/:jobId/cancel"
          ["trade", "async", _, "cancel"] -> "trade/async/:jobId/cancel"
          ["signal", "async", _] -> "signal/async/:jobId"
          ["backtest", "async", _] -> "backtest/async/:jobId"
          ["trade", "async", _] -> "trade/async/:jobId"

          ["optimizer", "run"] -> "optimizer/run"
          ["optimizer", "combos"] -> "optimizer/combos"
          _ ->
            let go xs =
                  case xs of
                    [] -> "root"
                    [x] -> T.unpack x
                    (x:rest) -> T.unpack x ++ "/" ++ go rest
             in go path

  case Wai.requestMethod req of
    "OPTIONS" -> respondLogged (Wai.responseLBS status204 corsHeaders "")
    _ -> do
      metricsIncEndpoint metrics label
      if path /= ["health"] && not (authorized apiToken req)
        then respondCors (jsonError status401 "Unauthorized (send Authorization: Bearer <token> or X-API-Key)")
        else
          case path of
            [] ->
              case Wai.requestMethod req of
                "GET" ->
                  respondCors $
                    jsonValue
                      status200
                      ( object
                          ( [ "name" .= ("trader-hs" :: String)
                            , "version" .= biVersion buildInfo
                            ]
                              ++ maybe [] (\c -> ["commit" .= c]) (biCommit buildInfo)
                              ++ [ "endpoints"
                                    .= [ object ["method" .= ("GET" :: String), "path" .= ("/health" :: String)]
                                       , object ["method" .= ("GET" :: String), "path" .= ("/metrics" :: String)]
                                       , object ["method" .= ("GET" :: String), "path" .= ("/ops" :: String)]
                                       , object ["method" .= ("GET" :: String), "path" .= ("/cache" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/cache/clear" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/signal" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/signal/async" :: String)]
                                       , object ["method" .= ("GET" :: String), "path" .= ("/signal/async/:jobId" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/signal/async/:jobId/cancel" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/trade" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/trade/async" :: String)]
                                       , object ["method" .= ("GET" :: String), "path" .= ("/trade/async/:jobId" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/trade/async/:jobId/cancel" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/backtest" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/backtest/async" :: String)]
                                       , object ["method" .= ("GET" :: String), "path" .= ("/backtest/async/:jobId" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/backtest/async/:jobId/cancel" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/binance/keys" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/binance/trades" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/binance/positions" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/coinbase/keys" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/binance/listenKey" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/binance/listenKey/keepAlive" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/binance/listenKey/close" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/bot/start" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/bot/stop" :: String)]
                                       , object ["method" .= ("GET" :: String), "path" .= ("/bot/status" :: String)]
                                       , object ["method" .= ("POST" :: String), "path" .= ("/optimizer/run" :: String)]
                                       , object ["method" .= ("GET" :: String), "path" .= ("/optimizer/combos" :: String)]
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
                      asyncCfg = asBacktest asyncStores
                      pairs =
                        ["status" .= ("ok" :: String), "version" .= biVersion buildInfo, "authRequired" .= authRequired, "authOk" .= authOk]
                          ++ maybe [] (\c -> ["commit" .= c]) (biCommit buildInfo)
                          ++ [ "computeLimits"
                                .= object
                                  [ "maxBarsLstm" .= aclMaxBarsLstm limits
                                  , "maxEpochs" .= aclMaxEpochs limits
                                  , "maxHiddenSize" .= aclMaxHiddenSize limits
                                  ]
                             , "asyncJobs"
                                .= object
                                  [ "maxRunning" .= jsMaxRunning asyncCfg
                                  , "ttlMs" .= jsTtlMs asyncCfg
                                  , "persistence" .= isJust (jsDir asyncCfg)
                                  ]
                             , "cache"
                                .= object
                                  [ "enabled" .= cacheEnabled apiCache
                                  , "ttlMs" .= acTtlMs apiCache
                                  , "maxEntries" .= acMaxEntries apiCache
                                  ]
                             ]
                   in respondCors (jsonValue status200 (object pairs))
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["metrics"] ->
              case Wai.requestMethod req of
                "GET" -> handleMetrics metrics botCtrl respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["ops"] ->
              case Wai.requestMethod req of
                "GET" -> handleOps mOps req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["cache"] ->
              case Wai.requestMethod req of
                "GET" -> do
                  v <- apiCacheStatsJson apiCache
                  respondCors (jsonValue status200 v)
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["cache", "clear"] ->
              case Wai.requestMethod req of
                "POST" -> do
                  apiCacheClear apiCache
                  now <- getTimestampMs
                  respondCors (jsonValue status200 (object ["ok" .= True, "atMs" .= now]))
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["signal"] ->
              case Wai.requestMethod req of
                "POST" -> handleSignal reqLimits apiCache mOps limits baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["signal", "async"] ->
              case Wai.requestMethod req of
                "POST" -> handleSignalAsync reqLimits apiCache mOps limits (asSignal asyncStores) baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["signal", "async", jobId] ->
              case Wai.requestMethod req of
                "GET" -> handleAsyncPoll (asSignal asyncStores) jobId respondCors
                "POST" -> handleAsyncPoll (asSignal asyncStores) jobId respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["signal", "async", jobId, "cancel"] ->
              case Wai.requestMethod req of
                "POST" -> handleAsyncCancel (asSignal asyncStores) jobId respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["trade"] ->
              case Wai.requestMethod req of
                "POST" -> handleTrade reqLimits mOps limits metrics mJournal mWebhook baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["trade", "async"] ->
              case Wai.requestMethod req of
                "POST" -> handleTradeAsync reqLimits mOps limits (asTrade asyncStores) metrics mJournal mWebhook baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["trade", "async", jobId] ->
              case Wai.requestMethod req of
                "GET" -> handleAsyncPoll (asTrade asyncStores) jobId respondCors
                "POST" -> handleAsyncPoll (asTrade asyncStores) jobId respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["trade", "async", jobId, "cancel"] ->
              case Wai.requestMethod req of
                "POST" -> handleAsyncCancel (asTrade asyncStores) jobId respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["backtest"] ->
              case Wai.requestMethod req of
                "POST" -> handleBacktest reqLimits apiCache mOps limits backtestGate baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["backtest", "async"] ->
              case Wai.requestMethod req of
                "POST" -> handleBacktestAsync reqLimits apiCache mOps limits backtestGate (asBacktest asyncStores) baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["backtest", "async", jobId] ->
              case Wai.requestMethod req of
                "GET" -> handleAsyncPoll (asBacktest asyncStores) jobId respondCors
                "POST" -> handleAsyncPoll (asBacktest asyncStores) jobId respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["backtest", "async", jobId, "cancel"] ->
              case Wai.requestMethod req of
                "POST" -> handleAsyncCancel (asBacktest asyncStores) jobId respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["binance", "keys"] ->
              case Wai.requestMethod req of
                "POST" -> handleBinanceKeys reqLimits mOps baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["binance", "positions"] ->
              case Wai.requestMethod req of
                "POST" -> handleBinancePositions reqLimits mOps baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["binance", "trades"] ->
              case Wai.requestMethod req of
                "POST" -> handleBinanceTrades reqLimits mOps baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["coinbase", "keys"] ->
              case Wai.requestMethod req of
                "POST" -> handleCoinbaseKeys reqLimits baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["binance", "listenKey"] ->
              case Wai.requestMethod req of
                "POST" -> handleBinanceListenKey reqLimits mOps baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["binance", "listenKey", "keepAlive"] ->
              case Wai.requestMethod req of
                "POST" -> handleBinanceListenKeyKeepAlive reqLimits mOps baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["binance", "listenKey", "close"] ->
              case Wai.requestMethod req of
                "POST" -> handleBinanceListenKeyClose reqLimits mOps baseArgs req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["bot", "start"] ->
              case Wai.requestMethod req of
                "POST" -> handleBotStart reqLimits mOps limits metrics mJournal mWebhook mBotStateDir optimizerTmp baseArgs botCtrl req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["bot", "stop"] ->
              case Wai.requestMethod req of
                "POST" -> handleBotStop mOps mJournal mWebhook mBotStateDir botCtrl req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["bot", "status"] ->
              case Wai.requestMethod req of
                "GET" -> handleBotStatus reqLimits botCtrl mBotStateDir req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["optimizer", "run"] ->
              case Wai.requestMethod req of
                "POST" -> handleOptimizerRun reqLimits projectRoot optimizerTmp req respondCors
                _ -> respondCors (jsonError status405 "Method not allowed")
            ["optimizer", "combos"] ->
              case Wai.requestMethod req of
                "GET" -> handleOptimizerCombos projectRoot optimizerTmp respondCors
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

longShortFuturesTradeError :: String
longShortFuturesTradeError = "--positioning long-short requires --futures when trading"

longShortFuturesDataError :: String
longShortFuturesDataError = "--positioning long-short requires --futures for binanceSymbol data"

apiErrorFromMsg :: String -> ApiError
apiErrorFromMsg msg =
  let hint =
        if msg == longShortFuturesTradeError || msg == longShortFuturesDataError
          then Just "Set market=futures or switch positioning to long-flat for spot/margin."
          else Nothing
   in ApiError msg hint

jsonError :: Status -> String -> Wai.Response
jsonError st msg = jsonValue st (apiErrorFromMsg msg)

exceptionToHttp :: SomeException -> (Status, String)
exceptionToHttp ex =
  case fromException ex of
    Just (ErrorCall msg) ->
      if isInternalErrorCall msg
        then
          let msg' = dropInternalPrefix msg
              suffix = if null msg' then "" else ": " ++ msg'
           in (status500, "Internal server error" ++ suffix)
        else (status400, msg)
    Nothing ->
      case fromException ex of
        Just io
          | isUserError io -> (status400, ioeGetErrorString io)
        _ ->
          case (fromException ex :: Maybe HttpException) of
            Just httpEx -> (status502, show httpEx)
            Nothing -> (status500, show ex)
  where
    internalIndicators = ["Singular matrix", "missing output bias", "argMaxAbs: empty"]
    isInternalErrorCall msg = "Internal:" `isPrefixOf` msg || any (`isInfixOf` msg) internalIndicators
    dropInternalPrefix msg =
      if "Internal:" `isPrefixOf` msg
        then dropWhile isSpace (drop (length ("Internal:" :: String)) msg)
        else msg

requestBodyTooLargeMessage :: Int64 -> String
requestBodyTooLargeMessage maxBytes =
  "Request body too large (max "
    ++ show maxBytes
    ++ " bytes; set TRADER_API_MAX_BODY_BYTES to override)."

readRequestBodyLimited :: Int64 -> Wai.Request -> IO (Either String BL.ByteString)
readRequestBodyLimited maxBytes req = do
  let maxBytes' = max 0 maxBytes
      maxBytesWord = fromIntegral maxBytes' :: Word64
      tooLarge = requestBodyTooLargeMessage maxBytes'
      go acc chunks = do
        chunk <- Wai.getRequestBodyChunk req
        if BS.null chunk
          then pure (Right (BL.fromChunks (reverse chunks)))
          else do
            let acc' = acc + fromIntegral (BS.length chunk)
            if maxBytes' > 0 && acc' > maxBytes'
              then pure (Left (requestBodyTooLargeMessage maxBytes'))
              else go acc' (chunk : chunks)
  case Wai.requestBodyLength req of
    Wai.KnownLength len | maxBytes' > 0 && len > maxBytesWord -> pure (Left tooLarge)
    _ -> go 0 []

decodeRequestBodyLimited :: FromJSON a => ApiRequestLimits -> Wai.Request -> String -> IO (Either Wai.Response a)
decodeRequestBodyLimited limits req errPrefix = do
  bodyOrErr <- readRequestBodyLimited (arlMaxBodyBytes limits) req
  case bodyOrErr of
    Left msg -> pure (Left (jsonError status413 msg))
    Right body ->
      case eitherDecode body of
        Left err -> pure (Left (jsonError status400 (errPrefix ++ err)))
        Right payload -> pure (Right payload)

textValue :: Status -> BL.ByteString -> Wai.Response
textValue st body =
  Wai.responseLBS
    st
    ([("Content-Type", "text/plain; version=0.0.4")] ++ noCacheHeaders)
    body

defaultOptimizerIntervals :: String
defaultOptimizerIntervals = intercalate "," ["1h", "2h", "4h", "6h", "12h", "1d"]

defaultOptimizerNormalizations :: String
defaultOptimizerNormalizations = "none,minmax,standard,log"

defaultOptimizerSymbols :: [String]
defaultOptimizerSymbols =
  [ "BTCUSDT"
  , "ETHUSDT"
  , "BNBUSDT"
  , "SOLUSDT"
  , "XRPUSDT"
  , "ADAUSDT"
  , "DOGEUSDT"
  , "MATICUSDT"
  , "AVAXUSDT"
  , "LINKUSDT"
  , "DOTUSDT"
  , "LTCUSDT"
  , "BCHUSDT"
  , "TRXUSDT"
  , "ATOMUSDT"
  , "ETCUSDT"
  , "UNIUSDT"
  , "AAVEUSDT"
  , "FILUSDT"
  , "NEARUSDT"
  , "OPUSDT"
  , "ARBUSDT"
  , "SUIUSDT"
  ]

splitEnvList :: String -> [String]
splitEnvList raw =
  let cleaned = map (\c -> if c == ',' then ' ' else c) raw
   in filter (not . null) (map trim (words cleaned))

readEnvBool :: Maybe String -> Bool -> Bool
readEnvBool raw def =
  case fmap (map toLower . trim) raw of
    Just "1" -> True
    Just "true" -> True
    Just "yes" -> True
    Just "on" -> True
    Just "0" -> False
    Just "false" -> False
    Just "no" -> False
    Just "off" -> False
    _ -> def

normalizeSymbol :: String -> String
normalizeSymbol = map toUpper . trim

parsePlatformsArg :: String -> Either String [Platform]
parsePlatformsArg raw =
  let parsed = splitEnvList raw
   in if null parsed
        then Left "platforms must include at least one entry"
        else traverse parsePlatform parsed

pickRandom :: [a] -> IO a
pickRandom xs =
  case xs of
    [] -> error "pickRandom: empty list"
    _ -> do
      r <- randomIO :: IO Word64
      let len = length xs
          idx = fromIntegral (r `mod` fromIntegral len)
      pure (xs !! idx)

pickDefaultString :: String -> Maybe String -> String
pickDefaultString def mb =
  case fmap trim mb of
    Just v | not (null v) -> v
    _ -> def

clamp01 :: Double -> Double
clamp01 x = max 0 (min 1 x)

maybeIntArg :: String -> Maybe Int -> [String]
maybeIntArg _ Nothing = []
maybeIntArg flag (Just n) = [flag, show (max 0 n)]

maybeDoubleArg :: String -> Maybe Double -> [String]
maybeDoubleArg _ Nothing = []
maybeDoubleArg flag (Just n) = [flag, show n]

prepareOptimizerArgs :: FilePath -> ApiOptimizerRunRequest -> IO (Either String [String])
prepareOptimizerArgs outputPath req = do
  exePath <- getExecutablePath
  let source = fromMaybe OptimizerSourceBinance (arrSource req)
      sourcePlatform = optimizerSourcePlatform source
      platformsRaw = fmap trim (arrPlatforms req)
      platformsResult =
        case source of
          OptimizerSourceCsv ->
            case platformsRaw of
              Just raw | not (null raw) ->
                Left "platforms is only valid for exchange sources"
              _ -> Right []
          _ ->
            case platformsRaw of
              Just raw | not (null raw) -> parsePlatformsArg raw
              _ -> Right (fromMaybe [PlatformBinance] (fmap pure sourcePlatform))
      platformArgsResult =
        case platformsResult of
          Left e -> Left e
          Right [] -> Right []
          Right ps -> Right ["--platforms", intercalate "," (map platformCode ps)]
  sourceArgsResult <-
    case source of
      OptimizerSourceCsv ->
        case fmap trim (arrData req) of
          Just raw | not (null raw) -> do
            resolved <- (try (canonicalizePath raw) :: IO (Either SomeException FilePath))
            case resolved of
              Left e -> pure (Left ("Failed to resolve CSV path: " ++ show e))
              Right path -> do
                exists <- doesFileExist path
                if exists
                  then pure (Right ["--data", path])
                  else pure (Left ("CSV path not found: " ++ path))
          _ -> pure (Left "CSV path is required when using csv source")
      _ ->
        case fmap (trim . map toUpper) (arrBinanceSymbol req) of
          Just sym | not (null sym) -> pure (Right ["--binance-symbol", sym])
          _ -> pure (Left "binanceSymbol is required when using exchange data")
  case sourceArgsResult of
    Left err -> pure (Left err)
    Right sourceArgs -> do
      let priceColumnArgs =
            case source of
              OptimizerSourceCsv -> ["--price-column", pickDefaultString "close" (arrPriceColumn req)]
              _ -> []
          highCol = case fmap trim (arrHighColumn req) of
            Just v | not (null v) -> Just v
            _ -> Nothing
          lowCol = case fmap trim (arrLowColumn req) of
            Just v | not (null v) -> Just v
            _ -> Nothing
          ohlcArgsResult =
            case source of
              OptimizerSourceCsv ->
                case (highCol, lowCol) of
                  (Nothing, Nothing) -> Right []
                  (Just h, Just l) -> Right ["--high-column", h, "--low-column", l]
                  _ -> Left "Provide both highColumn and lowColumn (or omit both)."
              _ ->
                if isJust highCol || isJust lowCol
                  then Left "highColumn/lowColumn are only supported for csv source"
                  else Right []
          objectiveAllowed = ["final-equity", "sharpe", "calmar", "equity-dd", "equity-dd-turnover"]
          objectiveRaw = fmap (map toLower . trim) (arrObjective req)
          objectiveArgsResult =
            case objectiveRaw of
              Nothing -> Right []
              Just v | null v -> Right []
              Just v | v `elem` objectiveAllowed -> Right ["--objective", v]
              Just v ->
                Left
                  ( "Invalid objective: "
                      ++ show v
                      ++ " (expected one of: "
                      ++ intercalate ", " objectiveAllowed
                      ++ ")"
                  )
          tuneObjectiveRaw = fmap (map toLower . trim) (arrTuneObjective req)
          tuneObjectiveArgsResult =
            case tuneObjectiveRaw of
              Nothing -> Right []
              Just v | null v -> Right []
              Just v | v `elem` objectiveAllowed -> Right ["--tune-objective", v]
              Just v ->
                Left
                  ( "Invalid tuneObjective: "
                      ++ show v
                      ++ " (expected one of: "
                      ++ intercalate ", " objectiveAllowed
                      ++ ")"
                  )
          barsDistributionAllowed = ["uniform", "log"]
          barsDistributionRaw = fmap (map toLower . trim) (arrBarsDistribution req)
          barsDistributionArgsResult =
            case barsDistributionRaw of
              Nothing -> Right []
              Just v | null v -> Right []
              Just v | v `elem` barsDistributionAllowed -> Right ["--bars-distribution", v]
              Just v ->
                Left
                  ( "Invalid barsDistribution: "
                      ++ show v
                      ++ " (expected one of: "
                      ++ intercalate ", " barsDistributionAllowed
                      ++ ")"
                  )
          penaltyMaxDdArgs =
            case arrPenaltyMaxDrawdown req of
              Nothing -> []
              Just w -> ["--penalty-max-drawdown", show (max 0 w)]
          penaltyTurnoverArgs =
            case arrPenaltyTurnover req of
              Nothing -> []
              Just w -> ["--penalty-turnover", show (max 0 w)]
          minRoundTripsArgs =
            case arrMinRoundTrips req of
              Nothing -> []
              Just n -> ["--min-round-trips", show (max 0 n)]
          minWinRateArgs =
            maybeDoubleArg "--min-win-rate" (fmap clamp01 (arrMinWinRate req))
          minProfitFactorArgs =
            maybeDoubleArg "--min-profit-factor" (fmap (max 0) (arrMinProfitFactor req))
          minExposureArgs =
            maybeDoubleArg "--min-exposure" (fmap clamp01 (arrMinExposure req))
          minSharpeArgs =
            maybeDoubleArg "--min-sharpe" (fmap (max 0) (arrMinSharpe req))
          minWalkForwardSharpeMeanArgs =
            maybeDoubleArg "--min-wf-sharpe-mean" (fmap (max 0) (arrMinWalkForwardSharpeMean req))
          maxWalkForwardSharpeStdArgs =
            maybeDoubleArg "--max-wf-sharpe-std" (fmap (max 0) (arrMaxWalkForwardSharpeStd req))
          tunePenaltyMaxDdArgs =
            maybeDoubleArg "--tune-penalty-max-drawdown" (fmap (max 0) (arrTunePenaltyMaxDrawdown req))
          tunePenaltyTurnoverArgs =
            maybeDoubleArg "--tune-penalty-turnover" (fmap (max 0) (arrTunePenaltyTurnover req))
          tuneStressVolMultArgs =
            maybeDoubleArg "--tune-stress-vol-mult" (fmap (max 1e-12) (arrTuneStressVolMult req))
          tuneStressShockArgs = maybeDoubleArg "--tune-stress-shock" (arrTuneStressShock req)
          tuneStressWeightArgs =
            maybeDoubleArg "--tune-stress-weight" (fmap (max 0) (arrTuneStressWeight req))
          tuneStressVolMultRangeArgs =
            maybeDoubleArg "--tune-stress-vol-mult-min" (fmap (max 1e-12) (arrTuneStressVolMultMin req))
              ++ maybeDoubleArg "--tune-stress-vol-mult-max" (fmap (max 1e-12) (arrTuneStressVolMultMax req))
          tuneStressShockRangeArgs =
            maybeDoubleArg "--tune-stress-shock-min" (arrTuneStressShockMin req)
              ++ maybeDoubleArg "--tune-stress-shock-max" (arrTuneStressShockMax req)
          tuneStressWeightRangeArgs =
            maybeDoubleArg "--tune-stress-weight-min" (fmap (max 0) (arrTuneStressWeightMin req))
              ++ maybeDoubleArg "--tune-stress-weight-max" (fmap (max 0) (arrTuneStressWeightMax req))
          walkForwardFoldsArgs =
            maybeIntArg "--walk-forward-folds-min" (fmap (max 1) (arrWalkForwardFoldsMin req))
              ++ maybeIntArg "--walk-forward-folds-max" (fmap (max 1) (arrWalkForwardFoldsMax req))
          intervalsVal = pickDefaultString defaultOptimizerIntervals (arrIntervals req)
          lookbackVal = pickDefaultString "7d" (arrLookbackWindow req)
          backtestRatioVal = clamp01 (fromMaybe 0.2 (arrBacktestRatio req))
          tuneRatioVal = clamp01 (fromMaybe 0.25 (arrTuneRatio req))
          trialsVal = max 1 (fromMaybe 50 (arrTrials req))
          timeoutVal = max 1 (fromMaybe 60 (arrTimeoutSec req))
          seedVal = max 0 (fromMaybe 42 (arrSeed req))
          slippageVal = max 0 (fromMaybe 0.0005 (arrSlippageMax req))
          spreadVal = max 0 (fromMaybe 0.0005 (arrSpreadMax req))
          epochsMinRaw = fmap (max 0) (arrEpochsMin req)
          epochsMaxRaw = fmap (max 0) (arrEpochsMax req)
          epochsMaxNorm =
            case (epochsMinRaw, epochsMaxRaw) of
              (Just mn, Just mx) -> Just (max mn mx)
              (_, Just mx) -> Just mx
              _ -> Nothing
          hiddenMinRaw = fmap (max 1) (arrHiddenSizeMin req)
          hiddenMaxRaw =
            case (hiddenMinRaw, arrHiddenSizeMax req) of
              (Just mn, Just mx) -> Just (max mn (max 1 mx))
              (_, Just mx) -> Just (max 1 mx)
              _ -> Nothing
          barsMinRaw = fmap (max 0) (arrBarsMin req)
          barsMaxRaw =
            case (barsMinRaw, arrBarsMax req) of
              (Just mn, Just mx) -> Just (max mn (max 0 mx))
              (_, Just mx) -> Just (max 0 mx)
              _ -> Nothing
          barsAutoProbArgs =
            maybeDoubleArg "--bars-auto-prob" (fmap clamp01 (arrBarsAutoProb req))
          minHoldBarsArgs =
            maybeIntArg "--min-hold-bars-min" (fmap (max 0) (arrMinHoldBarsMin req))
              ++ maybeIntArg "--min-hold-bars-max" (fmap (max 0) (arrMinHoldBarsMax req))
          cooldownBarsArgs =
            maybeIntArg "--cooldown-bars-min" (fmap (max 0) (arrCooldownBarsMin req))
              ++ maybeIntArg "--cooldown-bars-max" (fmap (max 0) (arrCooldownBarsMax req))
          maxHoldBarsArgs =
            maybeIntArg "--max-hold-bars-min" (fmap (max 0) (arrMaxHoldBarsMin req))
              ++ maybeIntArg "--max-hold-bars-max" (fmap (max 0) (arrMaxHoldBarsMax req))
          minEdgeArgs =
            maybeDoubleArg "--min-edge-min" (fmap (max 0) (arrMinEdgeMin req))
              ++ maybeDoubleArg "--min-edge-max" (fmap (max 0) (arrMinEdgeMax req))
          minSignalToNoiseArgs =
            maybeDoubleArg "--min-signal-to-noise-min" (fmap (max 0) (arrMinSignalToNoiseMin req))
              ++ maybeDoubleArg "--min-signal-to-noise-max" (fmap (max 0) (arrMinSignalToNoiseMax req))
          edgeBufferArgs =
            maybeDoubleArg "--edge-buffer-min" (fmap (max 0) (arrEdgeBufferMin req))
              ++ maybeDoubleArg "--edge-buffer-max" (fmap (max 0) (arrEdgeBufferMax req))
          pCostAwareEdgeArgs =
            maybeDoubleArg "--p-cost-aware-edge" (arrPCostAwareEdge req)
          trendLookbackArgs =
            maybeIntArg "--trend-lookback-min" (fmap (max 0) (arrTrendLookbackMin req))
              ++ maybeIntArg "--trend-lookback-max" (fmap (max 0) (arrTrendLookbackMax req))
          intrabarArgs =
            maybeDoubleArg "--p-intrabar-take-profit-first" (fmap clamp01 (arrPIntrabarTakeProfitFirst req))
          triLayerArgs =
            maybeDoubleArg "--p-tri-layer" (fmap clamp01 (arrPTriLayer req))
              ++ maybeDoubleArg "--tri-layer-fast-mult-min" (fmap (max 1e-6) (arrTriLayerFastMultMin req))
              ++ maybeDoubleArg "--tri-layer-fast-mult-max" (fmap (max 1e-6) (arrTriLayerFastMultMax req))
              ++ maybeDoubleArg "--tri-layer-slow-mult-min" (fmap (max 1e-6) (arrTriLayerSlowMultMin req))
              ++ maybeDoubleArg "--tri-layer-slow-mult-max" (fmap (max 1e-6) (arrTriLayerSlowMultMax req))
              ++ maybeDoubleArg "--p-tri-layer-price-action" (fmap clamp01 (arrPTriLayerPriceAction req))
              ++ maybeDoubleArg "--tri-layer-cloud-padding-min" (fmap (max 0) (arrTriLayerCloudPaddingMin req))
              ++ maybeDoubleArg "--tri-layer-cloud-padding-max" (fmap (max 0) (arrTriLayerCloudPaddingMax req))
              ++ maybeIntArg "--lstm-exit-flip-bars-min" (fmap (max 0) (arrLstmExitFlipBarsMin req))
              ++ maybeIntArg "--lstm-exit-flip-bars-max" (fmap (max 0) (arrLstmExitFlipBarsMax req))
          stopRangeArgs =
            maybeDoubleArg "--stop-min" (fmap (max 0) (arrStopMin req))
              ++ maybeDoubleArg "--stop-max" (fmap (max 0) (arrStopMax req))
              ++ maybeDoubleArg "--tp-min" (fmap (max 0) (arrTpMin req))
              ++ maybeDoubleArg "--tp-max" (fmap (max 0) (arrTpMax req))
              ++ maybeDoubleArg "--trail-min" (fmap (max 0) (arrTrailMin req))
              ++ maybeDoubleArg "--trail-max" (fmap (max 0) (arrTrailMax req))
              ++ maybeDoubleArg "--p-disable-stop" (fmap clamp01 (arrPDisableStop req))
              ++ maybeDoubleArg "--p-disable-tp" (fmap clamp01 (arrPDisableTp req))
              ++ maybeDoubleArg "--p-disable-trail" (fmap clamp01 (arrPDisableTrail req))
          stopVolMultArgs =
            maybeDoubleArg "--stop-vol-mult-min" (fmap (max 0) (arrStopVolMultMin req))
              ++ maybeDoubleArg "--stop-vol-mult-max" (fmap (max 0) (arrStopVolMultMax req))
              ++ maybeDoubleArg "--tp-vol-mult-min" (fmap (max 0) (arrTpVolMultMin req))
              ++ maybeDoubleArg "--tp-vol-mult-max" (fmap (max 0) (arrTpVolMultMax req))
              ++ maybeDoubleArg "--trail-vol-mult-min" (fmap (max 0) (arrTrailVolMultMin req))
              ++ maybeDoubleArg "--trail-vol-mult-max" (fmap (max 0) (arrTrailVolMultMax req))
              ++ maybeDoubleArg "--p-disable-stop-vol-mult" (fmap clamp01 (arrPDisableStopVolMult req))
              ++ maybeDoubleArg "--p-disable-tp-vol-mult" (fmap clamp01 (arrPDisableTpVolMult req))
              ++ maybeDoubleArg "--p-disable-trail-vol-mult" (fmap clamp01 (arrPDisableTrailVolMult req))
          minPositionSizeArgs =
            maybeDoubleArg "--min-position-size-min" (fmap clamp01 (arrMinPositionSizeMin req))
              ++ maybeDoubleArg "--min-position-size-max" (fmap clamp01 (arrMinPositionSizeMax req))
          maxPositionSizeArgs =
            maybeDoubleArg "--max-position-size-min" (fmap (max 0) (arrMaxPositionSizeMin req))
              ++ maybeDoubleArg "--max-position-size-max" (fmap (max 0) (arrMaxPositionSizeMax req))
          volTargetArgs =
            maybeDoubleArg "--vol-target-min" (fmap (max 0) (arrVolTargetMin req))
              ++ maybeDoubleArg "--vol-target-max" (fmap (max 0) (arrVolTargetMax req))
          pDisableVolTargetArgs =
            maybeDoubleArg "--p-disable-vol-target" (fmap clamp01 (arrPDisableVolTarget req))
          volLookbackArgs =
            maybeIntArg "--vol-lookback-min" (fmap (max 0) (arrVolLookbackMin req))
              ++ maybeIntArg "--vol-lookback-max" (fmap (max 0) (arrVolLookbackMax req))
          volEwmaAlphaArgs =
            maybeDoubleArg "--vol-ewma-alpha-min" (fmap clamp01 (arrVolEwmaAlphaMin req))
              ++ maybeDoubleArg "--vol-ewma-alpha-max" (fmap clamp01 (arrVolEwmaAlphaMax req))
          pDisableVolEwmaAlphaArgs =
            maybeDoubleArg "--p-disable-vol-ewma-alpha" (fmap clamp01 (arrPDisableVolEwmaAlpha req))
          volFloorArgs =
            maybeDoubleArg "--vol-floor-min" (fmap (max 0) (arrVolFloorMin req))
              ++ maybeDoubleArg "--vol-floor-max" (fmap (max 0) (arrVolFloorMax req))
          volScaleMaxArgs =
            maybeDoubleArg "--vol-scale-max-min" (fmap (max 0) (arrVolScaleMaxMin req))
              ++ maybeDoubleArg "--vol-scale-max-max" (fmap (max 0) (arrVolScaleMaxMax req))
          maxVolatilityArgs =
            maybeDoubleArg "--max-volatility-min" (fmap (max 0) (arrMaxVolatilityMin req))
              ++ maybeDoubleArg "--max-volatility-max" (fmap (max 0) (arrMaxVolatilityMax req))
          pDisableMaxVolatilityArgs =
            maybeDoubleArg "--p-disable-max-volatility" (fmap clamp01 (arrPDisableMaxVolatility req))
          periodsPerYearArgs =
            maybeDoubleArg "--periods-per-year-min" (fmap (max 0) (arrPeriodsPerYearMin req))
              ++ maybeDoubleArg "--periods-per-year-max" (fmap (max 0) (arrPeriodsPerYearMax req))
          kalmanZArgs =
            maybeDoubleArg "--kalman-z-min-min" (fmap (max 0) (arrKalmanZMinMin req))
              ++ maybeDoubleArg "--kalman-z-min-max" (fmap (max 0) (arrKalmanZMinMax req))
              ++ maybeDoubleArg "--kalman-z-max-min" (fmap (max 0) (arrKalmanZMaxMin req))
              ++ maybeDoubleArg "--kalman-z-max-max" (fmap (max 0) (arrKalmanZMaxMax req))
          maxHighVolProbArgs =
            maybeDoubleArg "--p-disable-max-high-vol-prob" (fmap clamp01 (arrPDisableMaxHighVolProb req))
              ++ maybeDoubleArg "--max-high-vol-prob-min" (fmap clamp01 (arrMaxHighVolProbMin req))
              ++ maybeDoubleArg "--max-high-vol-prob-max" (fmap clamp01 (arrMaxHighVolProbMax req))
          maxConformalWidthArgs =
            maybeDoubleArg "--p-disable-max-conformal-width" (fmap clamp01 (arrPDisableMaxConformalWidth req))
              ++ maybeDoubleArg "--max-conformal-width-min" (fmap (max 0) (arrMaxConformalWidthMin req))
              ++ maybeDoubleArg "--max-conformal-width-max" (fmap (max 0) (arrMaxConformalWidthMax req))
          maxQuantileWidthArgs =
            maybeDoubleArg "--p-disable-max-quantile-width" (fmap clamp01 (arrPDisableMaxQuantileWidth req))
              ++ maybeDoubleArg "--max-quantile-width-min" (fmap (max 0) (arrMaxQuantileWidthMin req))
              ++ maybeDoubleArg "--max-quantile-width-max" (fmap (max 0) (arrMaxQuantileWidthMax req))
          confirmArgs =
            maybeDoubleArg "--p-confirm-conformal" (fmap clamp01 (arrPConfirmConformal req))
              ++ maybeDoubleArg "--p-confirm-quantiles" (fmap clamp01 (arrPConfirmQuantiles req))
          confidenceSizingArgs =
            maybeDoubleArg "--p-confidence-sizing" (fmap clamp01 (arrPConfidenceSizing req))
          kalmanMarketTopNArgs =
            maybeIntArg "--kalman-market-top-n-min" (fmap (max 0) (arrKalmanMarketTopNMin req))
              ++ maybeIntArg "--kalman-market-top-n-max" (fmap (max 0) (arrKalmanMarketTopNMax req))
          methodWeightBlendArgs =
            maybeDoubleArg "--method-weight-blend" (fmap (max 0) (arrMethodWeightBlend req))
          blendWeightArgs =
            maybeDoubleArg "--blend-weight-min" (fmap clamp01 (arrBlendWeightMin req))
              ++ maybeDoubleArg "--blend-weight-max" (fmap clamp01 (arrBlendWeightMax req))
          normalizationsVal = pickDefaultString defaultOptimizerNormalizations (arrNormalizations req)
          boolArg flag val = if val then [flag] else []
          disableLstm = fromMaybe False (arrDisableLstmPersistence req)
          noSweep = fromMaybe False (arrNoSweepThreshold req)
          epochArgs =
            maybeIntArg "--epochs-min" epochsMinRaw
              ++ maybeIntArg "--epochs-max" epochsMaxNorm
          hiddenArgs =
            maybeIntArg "--hidden-size-min" hiddenMinRaw
              ++ maybeIntArg "--hidden-size-max" hiddenMaxRaw
          lrArgs =
            maybeDoubleArg "--lr-min" (fmap (max 1e-12) (arrLrMin req))
              ++ maybeDoubleArg "--lr-max" (fmap (max 1e-12) (arrLrMax req))
          patienceArgs =
            maybeIntArg "--patience-max" (fmap (max 0) (arrPatienceMax req))
          gradClipArgs =
            maybeDoubleArg "--grad-clip-min" (fmap (max 1e-12) (arrGradClipMin req))
              ++ maybeDoubleArg "--grad-clip-max" (fmap (max 1e-12) (arrGradClipMax req))
              ++ maybeDoubleArg "--p-disable-grad-clip" (fmap clamp01 (arrPDisableGradClip req))
          barsArgs =
            maybeIntArg "--bars-min" barsMinRaw
              ++ maybeIntArg "--bars-max" barsMaxRaw
          tuneArgsBase =
            [ "--intervals", intervalsVal
            , "--lookback-window", lookbackVal
            , "--backtest-ratio", show backtestRatioVal
            , "--tune-ratio", show tuneRatioVal
            , "--trials", show trialsVal
            , "--timeout-sec", show timeoutVal
            , "--seed", show seedVal
            , "--slippage-max", show slippageVal
            , "--spread-max", show spreadVal
            , "--normalizations", normalizationsVal
            ]
              ++ barsAutoProbArgs
          tuneArgsSuffix =
            penaltyMaxDdArgs
              ++ penaltyTurnoverArgs
              ++ minRoundTripsArgs
              ++ minWinRateArgs
              ++ minProfitFactorArgs
              ++ minExposureArgs
              ++ minSharpeArgs
              ++ minWalkForwardSharpeMeanArgs
              ++ maxWalkForwardSharpeStdArgs
              ++ tunePenaltyMaxDdArgs
              ++ tunePenaltyTurnoverArgs
              ++ tuneStressVolMultArgs
              ++ tuneStressShockArgs
              ++ tuneStressWeightArgs
              ++ tuneStressVolMultRangeArgs
              ++ tuneStressShockRangeArgs
              ++ tuneStressWeightRangeArgs
              ++ walkForwardFoldsArgs
              ++ lrArgs
              ++ patienceArgs
              ++ gradClipArgs
              ++ minHoldBarsArgs
              ++ cooldownBarsArgs
              ++ maxHoldBarsArgs
              ++ minEdgeArgs
              ++ minSignalToNoiseArgs
              ++ edgeBufferArgs
              ++ pCostAwareEdgeArgs
              ++ trendLookbackArgs
              ++ intrabarArgs
              ++ triLayerArgs
              ++ stopRangeArgs
              ++ stopVolMultArgs
              ++ minPositionSizeArgs
              ++ maxPositionSizeArgs
              ++ volTargetArgs
              ++ pDisableVolTargetArgs
              ++ volLookbackArgs
              ++ volEwmaAlphaArgs
              ++ pDisableVolEwmaAlphaArgs
              ++ volFloorArgs
              ++ volScaleMaxArgs
              ++ maxVolatilityArgs
              ++ pDisableMaxVolatilityArgs
              ++ periodsPerYearArgs
              ++ kalmanZArgs
              ++ maxHighVolProbArgs
              ++ maxConformalWidthArgs
              ++ maxQuantileWidthArgs
              ++ confirmArgs
              ++ confidenceSizingArgs
              ++ methodWeightBlendArgs
              ++ blendWeightArgs
              ++ kalmanMarketTopNArgs
              ++ ["--output", outputPath]
      pure $
        case (platformArgsResult, ohlcArgsResult, objectiveArgsResult, tuneObjectiveArgsResult, barsDistributionArgsResult) of
          (Left e, _, _, _, _) -> Left e
          (_, Left e, _, _, _) -> Left e
          (_, _, Left e, _, _) -> Left e
          (_, _, _, Left e, _) -> Left e
          (_, _, _, _, Left e) -> Left e
          (Right platformArgs, Right ohlcArgs, Right objectiveArgs, Right tuneObjectiveArgs, Right barsDistributionArgs) ->
            let tuneArgs = tuneArgsBase ++ objectiveArgs ++ tuneObjectiveArgs ++ barsDistributionArgs ++ tuneArgsSuffix
             in Right
                  ( sourceArgs
                      ++ platformArgs
                      ++ priceColumnArgs
                      ++ ohlcArgs
                      ++ tuneArgs
                      ++ barsArgs
                      ++ epochArgs
                      ++ hiddenArgs
                      ++ ["--binary", exePath]
                      ++ boolArg "--disable-lstm-persistence" disableLstm
                      ++ boolArg "--no-sweep-threshold" noSweep
                  )

optimizerCombosFileName :: FilePath
optimizerCombosFileName = "top-combos.json"

s3TopCombosKey :: S3State -> String
s3TopCombosKey st =
  s3KeyFor st ["optimizer", optimizerCombosFileName]

s3TopCombosHistoryKey :: S3State -> Int64 -> String
s3TopCombosHistoryKey st ts =
  s3KeyFor st ["optimizer", "history", printf "top-combos-%d.json" ts]

resolveOptimizerCombosPath :: FilePath -> IO FilePath
resolveOptimizerCombosPath optimizerTmp = do
  mDir <- lookupEnv "TRADER_OPTIMIZER_COMBOS_DIR"
  case trim <$> mDir of
    Just dir | not (null dir) -> pure (dir </> optimizerCombosFileName)
    _ -> do
      mStateDir <- stateSubdirFromEnv "optimizer"
      pure ((fromMaybe optimizerTmp mStateDir) </> optimizerCombosFileName)

resolveOptimizerCombosHistoryDir :: FilePath -> IO (Maybe FilePath)
resolveOptimizerCombosHistoryDir topJsonPath = do
  mDir <- lookupEnv "TRADER_OPTIMIZER_COMBOS_HISTORY_DIR"
  case trim <$> mDir of
    Nothing -> pure (Just (takeDirectory topJsonPath </> "top-combos-history"))
    Just dir ->
      let lowered = map toLower dir
       in
        if null dir || lowered `elem` ["0", "false", "off", "no"]
          then pure Nothing
          else pure (Just dir)

defaultOptimizerMaxCombos :: Int
defaultOptimizerMaxCombos = 200

optimizerMaxCombosFromEnv :: IO Int
optimizerMaxCombosFromEnv = do
  maxCombosEnv <- lookupEnv "TRADER_OPTIMIZER_MAX_COMBOS"
  pure $
    case maxCombosEnv >>= readMaybe of
      Just n | n >= 1 -> n
      _ -> defaultOptimizerMaxCombos

truncateProcessOutput :: Int -> String -> String
truncateProcessOutput maxBytes raw =
  let maxBytes' = max 0 maxBytes
      bs = BS.pack raw
      bsLen = BS.length bs
   in if maxBytes' <= 0
        then ""
        else if bsLen <= maxBytes'
          then raw
          else
            let kept = BS.drop (bsLen - maxBytes') bs
                prefix = printf "... (truncated, showing last %d bytes)\n" maxBytes'
             in prefix ++ BS.unpack kept

resolveOptimizerExecutable :: FilePath -> String -> IO (Either String FilePath)
resolveOptimizerExecutable projectRoot name = do
  exePath <- getExecutablePath
  let localPath = takeDirectory exePath </> name
  localExists <- doesFileExist localPath
  if localExists
    then pure (Right localPath)
    else do
      mPath <- findExecutable name
      case mPath of
        Just p -> pure (Right p)
        Nothing -> do
          cabalDir <- findCabalDir
          case cabalDir of
            Nothing -> pure (Left ("Optimizer executable not found: " ++ name))
            Just dir -> do
              let proc' = (proc "cabal" ["list-bin", name]) {cwd = Just dir}
              r <- try (readCreateProcessWithExitCode proc' "") :: IO (Either SomeException (ExitCode, String, String))
              case r of
                Left e -> pure (Left ("Failed to discover " ++ name ++ " via cabal: " ++ show e))
                Right (ExitFailure _, out, err) ->
                  pure (Left ("Failed to discover " ++ name ++ " via cabal: " ++ trim (if null err then out else err)))
                Right (ExitSuccess, out, _) ->
                  let path = trim out
                   in if null path
                        then pure (Left ("cabal returned empty binary path for " ++ name))
                        else do
                          exists <- doesFileExist path
                          if exists
                            then pure (Right path)
                            else pure (Left ("cabal returned non-existent binary path: " ++ path))
  where
    findCabalDir = do
      let candidates = [projectRoot </> "haskell", projectRoot]
      firstExisting candidates
    firstExisting [] = pure Nothing
    firstExisting (dir : rest) = do
      exists <- doesFileExist (dir </> "trader.cabal")
      if exists then pure (Just dir) else firstExisting rest

runOptimizerProcess ::
  FilePath ->
  FilePath ->
  [String] ->
  IO (Either (String, String, String) ApiOptimizerRunResponse)
runOptimizerProcess projectRoot outputPath cliArgs = do
  exeResult <- resolveOptimizerExecutable projectRoot "optimize-equity"
  case exeResult of
    Left err -> pure (Left (err, "", ""))
    Right exePath -> do
      let proc' = (proc exePath cliArgs) {cwd = Just projectRoot}
      (exitCode, out, err) <- readCreateProcessWithExitCode proc' ""
      case exitCode of
        ExitSuccess -> do
          recordOrErr <- readLastOptimizerRecord outputPath
          case recordOrErr of
            Left msg -> pure (Left (msg, out, err))
            Right val -> pure (Right (ApiOptimizerRunResponse val out err))
        ExitFailure code -> pure (Left (printf "Optimizer executable failed (exit %d)" code, out, err))

runMergeTopCombos ::
  FilePath ->
  FilePath ->
  FilePath ->
  Int ->
  IO (Either (String, String, String) ())
runMergeTopCombos projectRoot topJsonPath recordsPath maxItems = do
  dirResult <- try (createDirectoryIfMissing True (takeDirectory topJsonPath)) :: IO (Either SomeException ())
  case dirResult of
    Left e -> pure (Left ("Failed to create top combos directory: " ++ show e, "", ""))
    Right _ -> do
      s3TempPath <- writeS3TopCombosTemp topJsonPath
      historyDir <- resolveOptimizerCombosHistoryDir topJsonPath
      exeResult <- resolveOptimizerExecutable projectRoot "merge-top-combos"
      case exeResult of
        Left err -> pure (Left (err, "", ""))
        Right exePath -> do
          let baseArgs = ["--top-json", topJsonPath, "--from-jsonl", recordsPath, "--max", show (max 1 maxItems)]
              s3Args = maybe [] (\p -> ["--from-top-json", p]) s3TempPath
              historyArgs = maybe [] (\dir -> ["--history-dir", dir]) historyDir
              args = baseArgs ++ s3Args ++ historyArgs
              proc' = (proc exePath args) {cwd = Just projectRoot}
              cleanup = maybe (pure ()) removeTempTopCombo s3TempPath
          (exitCode, out, err) <- readCreateProcessWithExitCode proc' "" `finally` cleanup
          case exitCode of
            ExitSuccess -> do
              persistTopCombosMaybe topJsonPath
              pure (Right ())
            ExitFailure code -> pure (Left (printf "Merge executable failed (exit %d)" code, out, err))

writeS3TopCombosTemp :: FilePath -> IO (Maybe FilePath)
writeS3TopCombosTemp topJsonPath = do
  mS3 <- resolveS3State
  case mS3 of
    Nothing -> pure Nothing
    Just st -> do
      let key = s3TopCombosKey st
      r <- s3GetObject st key
      case r of
        Left _ -> pure Nothing
        Right Nothing -> pure Nothing
        Right (Just contents) -> do
          let dir = takeDirectory topJsonPath
          tempResult <- try (openTempFile dir "top-combos-s3.json") :: IO (Either SomeException (FilePath, Handle))
          case tempResult of
            Left _ -> pure Nothing
            Right (path, handle) -> do
              _ <- try (BL.hPut handle contents) :: IO (Either SomeException ())
              hClose handle
              pure (Just path)

removeTempTopCombo :: FilePath -> IO ()
removeTempTopCombo path = do
  _ <- try (removeFile path) :: IO (Either SomeException ())
  pure ()

readLastOptimizerRecord :: FilePath -> IO (Either String Aeson.Value)
readLastOptimizerRecord path = do
  exists <- doesFileExist path
  if not exists
    then pure (Left "Optimizer did not emit any records.")
    else do
      contentsOrErr <- (try (BL.readFile path) :: IO (Either SomeException BL.ByteString))
      case contentsOrErr of
        Left e -> pure (Left ("Failed to read optimizer records: " ++ show e))
        Right contents ->
          let nonEmpty = filter (not . BS.null) (BS.lines (BL.toStrict contents))
           in case listToMaybe (reverse nonEmpty) of
                Nothing -> pure (Left "Optimizer records file was empty.")
                Just lastLine ->
                  case Aeson.eitherDecodeStrict' lastLine of
                    Left err -> pure (Left ("Failed to parse optimizer record: " ++ err))
                    Right val -> pure (Right val)

writeKlinesCsv :: FilePath -> [Kline] -> IO ()
writeKlinesCsv path ks = do
  let header = "openTime,close,high,low\n"
      row k =
        intercalate
          ","
          [ show (kOpenTime k)
          , show (kClose k)
          , show (kHigh k)
          , show (kLow k)
          ]
          ++ "\n"
      body = header ++ concatMap row ks
  writeFile path body

readTopCombosExport :: FilePath -> IO (Either String TopCombosExport)
readTopCombosExport path = do
  valOrErr <- readTopCombosValue path
  case valOrErr of
    Left err -> pure (Left err)
    Right val ->
      case Aeson.fromJSON val of
        Aeson.Error err -> pure (Left ("Failed to parse top combos JSON: " ++ err))
        Aeson.Success out -> pure (Right out)

persistTopCombosMaybe :: FilePath -> IO ()
persistTopCombosMaybe path = do
  mS3 <- resolveS3State
  case mS3 of
    Nothing -> pure ()
    Just st -> do
      contentsOrErr <- (try (BL.readFile path) :: IO (Either SomeException BL.ByteString))
      case contentsOrErr of
        Left _ -> pure ()
        Right contents -> do
          _ <- s3PutObject st (s3TopCombosKey st) contents
          persistTopCombosHistoryMaybe path st contents
          pure ()

persistTopCombosHistoryMaybe :: FilePath -> S3State -> BL.ByteString -> IO ()
persistTopCombosHistoryMaybe topJsonPath st contents = do
  historyDir <- resolveOptimizerCombosHistoryDir topJsonPath
  case historyDir of
    Nothing -> pure ()
    Just _ -> do
      ts <- getTimestampMs
      let key = s3TopCombosHistoryKey st ts
      _ <- s3PutObject st key contents
      pure ()

data ComboBacktestUpdate = ComboBacktestUpdate
  { cbuMetrics :: !Aeson.Value
  , cbuFinalEquity :: !(Maybe Double)
  , cbuScore :: !(Maybe Double)
  , cbuOperations :: !(Maybe Aeson.Value)
  }

readTopCombosValueLocal :: FilePath -> IO (Either String Aeson.Value)
readTopCombosValueLocal path = do
  exists <- doesFileExist path
  if not exists
    then pure (Left "Top combos JSON not found.")
    else do
      contentsOrErr <- (try (BL.readFile path) :: IO (Either SomeException BL.ByteString))
      case contentsOrErr of
        Left e -> pure (Left ("Failed to read top combos JSON: " ++ show e))
        Right contents ->
          case Aeson.eitherDecode' contents of
            Left err -> pure (Left ("Failed to parse top combos JSON: " ++ err))
            Right val -> pure (Right val)

readTopCombosValue :: FilePath -> IO (Either String Aeson.Value)
readTopCombosValue path = do
  localResult <- readTopCombosValueLocal path
  case localResult of
    Right val -> pure (Right val)
    Left localErr -> do
      mS3 <- resolveS3State
      case mS3 of
        Nothing -> pure (Left localErr)
        Just st -> do
          let key = s3TopCombosKey st
          r <- s3GetObject st key
          case r of
            Left e -> pure (Left ("Failed to read top combos from S3: " ++ e))
            Right Nothing -> pure (Left localErr)
            Right (Just contents) ->
              case Aeson.eitherDecode' contents of
                Left err -> pure (Left ("Failed to parse top combos JSON from S3: " ++ err))
                Right val -> do
                  _ <- writeTopCombosValue path val
                  pure (Right val)

writeTopCombosValue :: FilePath -> Aeson.Value -> IO (Either String ())
writeTopCombosValue path val = do
  let dir = takeDirectory path
  dirResult <- try (createDirectoryIfMissing True dir) :: IO (Either SomeException ())
  case dirResult of
    Left e -> pure (Left ("Failed to create top combos directory: " ++ show e))
    Right _ -> do
      tempResult <- try (openTempFile dir "top-combos-backtest.json") :: IO (Either SomeException (FilePath, Handle))
      case tempResult of
        Left e -> pure (Left ("Failed to create temp top combos file: " ++ show e))
        Right (tmpPath, handle) -> do
          _ <- try (BL.hPut handle (encodePretty val <> "\n")) :: IO (Either SomeException ())
          hClose handle
          renameResult <- try (renameFile tmpPath path) :: IO (Either SomeException ())
          case renameResult of
            Left e -> pure (Left ("Failed to write top combos JSON: " ++ show e))
            Right _ -> pure (Right ())

comboMetricValue :: String -> Aeson.Value -> Maybe Aeson.Value
comboMetricValue key val =
  case val of
    Aeson.Object o -> KM.lookup (AK.fromString key) o
    _ -> Nothing

comboMetricDouble :: String -> Aeson.Value -> Maybe Double
comboMetricDouble key val =
  comboMetricValue key val >>= AT.parseMaybe parseJSON

comboMetricsDouble :: String -> Aeson.Value -> Maybe Double
comboMetricsDouble key val = do
  metrics <- comboMetricValue "metrics" val
  comboMetricDouble key metrics

comboIdentityKey :: Aeson.Value -> Maybe BS.ByteString
comboIdentityKey val = do
  params <- comboMetricValue "params" val
  let openThr = comboMetricValue "openThreshold" val
      closeThr = comboMetricValue "closeThreshold" val
      objective = comboMetricValue "objective" val
      identity =
        object
          [ "params" .= params
          , "openThreshold" .= openThr
          , "closeThreshold" .= closeThr
          , "objective" .= objective
          ]
  pure (BL.toStrict (encodePretty identity))

comboPerformanceKey :: Aeson.Value -> (Double, Double, Int)
comboPerformanceKey val =
  let eq = fromMaybe 0 (comboMetricDouble "finalEquity" val <|> comboMetricsDouble "finalEquity" val)
      score = fromMaybe (negate (1 / 0)) (comboMetricDouble "score" val)
      rank =
        case val of
          Aeson.Object o -> fromMaybe maxBound (KM.lookup (AK.fromString "rank") o >>= AT.parseMaybe parseJSON)
          _ -> maxBound
      eq' = if isNaN eq || isInfinite eq then 0 else eq
      score' = if isNaN score || isInfinite score then negate (1 / 0) else score
   in (negate eq', negate score', rank)

extractBacktestMetrics :: Aeson.Value -> Maybe Aeson.Value
extractBacktestMetrics val =
  case val of
    Aeson.Object o -> KM.lookup (AK.fromString "metrics") o
    _ -> Nothing

extractBacktestOperations :: Aeson.Value -> Maybe Aeson.Value
extractBacktestOperations val =
  case val of
    Aeson.Object o ->
      case KM.lookup (AK.fromString "trades") o of
        Just (Aeson.Array trades) ->
          let ops = mapMaybe tradeToOp (V.toList trades)
           in if null ops then Nothing else Just (Aeson.Array (V.fromList ops))
        _ -> Nothing
    _ -> Nothing
  where
    tradeToOp :: Aeson.Value -> Maybe Aeson.Value
    tradeToOp tradeVal =
      case tradeVal of
        Aeson.Object trade -> do
          entryIdx <- (KM.lookup (AK.fromString "entryIndex") trade >>= AT.parseMaybe parseJSON :: Maybe Int)
          exitIdx <- (KM.lookup (AK.fromString "exitIndex") trade >>= AT.parseMaybe parseJSON :: Maybe Int)
          let entryEq = KM.lookup (AK.fromString "entryEquity") trade >>= AT.parseMaybe parseJSON :: Maybe Double
              exitEq = KM.lookup (AK.fromString "exitEquity") trade >>= AT.parseMaybe parseJSON :: Maybe Double
              retVal = KM.lookup (AK.fromString "return") trade >>= AT.parseMaybe parseJSON :: Maybe Double
              holding = KM.lookup (AK.fromString "holdingPeriods") trade >>= AT.parseMaybe parseJSON :: Maybe Int
              exitReason = KM.lookup (AK.fromString "exitReason") trade >>= AT.parseMaybe parseJSON :: Maybe String
              base =
                [ "entryIndex" .= entryIdx
                , "exitIndex" .= exitIdx
                ]
              extras =
                catMaybes
                  [ fmap (\v -> "entryEquity" .= v) entryEq
                  , fmap (\v -> "exitEquity" .= v) exitEq
                  , fmap (\v -> "return" .= v) retVal
                  , fmap (\v -> "holdingPeriods" .= v) holding
                  , fmap (\v -> "exitReason" .= v) exitReason
                  ]
          pure (object (base ++ extras))
        _ -> Nothing

objectiveScoreFromMetrics :: Args -> String -> Aeson.Value -> Maybe Double
objectiveScoreFromMetrics args objective metricsVal =
  let metric k =
        case metricsVal of
          Aeson.Object o -> KM.lookup (AK.fromString k) o >>= AT.parseMaybe parseJSON
          _ -> Nothing
      finalEq = fromMaybe 0 (metric "finalEquity")
      maxDd = fromMaybe 0 (metric "maxDrawdown")
      sharpe = fromMaybe 0 (metric "sharpe")
      annRet = fromMaybe 0 (metric "annualizedReturn")
      turnover = fromMaybe 0 (metric "turnover")
      obj = map toLower (trim objective)
      penaltyMaxDd = max 0 (argTunePenaltyMaxDrawdown args)
      penaltyTurnover = max 0 (argTunePenaltyTurnover args)
      rawScore
        | obj `elem` ["final-equity", "final_equity", "finalequity"] = finalEq
        | obj == "sharpe" = sharpe
        | obj == "calmar" = annRet / max 1e-12 maxDd
        | obj `elem` ["equity-dd", "equity_maxdd", "equity-dd-only"] =
            finalEq - penaltyMaxDd * maxDd
        | obj `elem` ["equity-dd-turnover", "equity-dd-ops", "equity-dd-turn"] =
            finalEq - penaltyMaxDd * maxDd - penaltyTurnover * turnover
        | otherwise = finalEq
      score =
        if isNaN rawScore || isInfinite rawScore
          then Nothing
          else Just rawScore
   in score

updateComboWithBacktest :: ComboBacktestUpdate -> Aeson.Value -> Aeson.Value
updateComboWithBacktest update comboVal =
  case comboVal of
    Aeson.Object o ->
      let o1 = KM.insert (AK.fromString "metrics") (cbuMetrics update) o
          o2 =
            case cbuFinalEquity update of
              Nothing -> o1
              Just eq -> KM.insert (AK.fromString "finalEquity") (toJSON eq) o1
          o3 =
            case cbuScore update of
              Nothing -> o2
              Just score -> KM.insert (AK.fromString "score") (toJSON score) o2
          o4 =
            case cbuOperations update of
              Nothing -> o3
              Just ops -> KM.insert (AK.fromString "operations") ops o3
       in Aeson.Object o4
    _ -> comboVal

applyComboUpdates :: Int64 -> HM.HashMap BS.ByteString ComboBacktestUpdate -> Aeson.Value -> Either String (Aeson.Value, Int)
applyComboUpdates now updates val =
  case val of
    Aeson.Object o ->
      case KM.lookup (AK.fromString "combos") o of
        Just (Aeson.Array combos) -> do
          let combosList = V.toList combos
              (updatedCombos, updatedCount) = foldl' applyOne ([], 0 :: Int) combosList
              applyOne (acc, count) comboVal =
                case comboIdentityKey comboVal >>= (`HM.lookup` updates) of
                  Nothing -> (comboVal : acc, count)
                  Just upd -> (updateComboWithBacktest upd comboVal : acc, count + 1)
              combosOut = Aeson.Array (V.fromList (reverse updatedCombos))
              o' = KM.insert (AK.fromString "combos") combosOut (KM.insert (AK.fromString "generatedAtMs") (toJSON now) o)
          Right (Aeson.Object o', updatedCount)
        _ -> Left "Top combos JSON missing combos array."
    _ -> Left "Top combos JSON root must be an object."

topComboParamString :: String -> TopCombo -> Maybe String
topComboParamString key combo =
  KM.lookup (AK.fromString key) (tcParams combo) >>= AT.parseMaybe parseJSON

topComboParamInt :: String -> TopCombo -> Maybe Int
topComboParamInt key combo =
  KM.lookup (AK.fromString key) (tcParams combo) >>= AT.parseMaybe parseJSON

topComboRankKey :: TopCombo -> (Int, Double, Double)
topComboRankKey combo =
  let rank = fromMaybe (maxBound :: Int) (tcRank combo)
      score = fromMaybe (negate (1 / 0)) (tcScore combo)
      eq = fromMaybe 0 (tcFinalEquity combo)
   in (rank, negate score, negate eq)

topComboMatchesSymbol :: String -> Maybe String -> TopCombo -> Bool
topComboMatchesSymbol symRaw mInterval combo =
  let sym = normalizeSymbol symRaw
      comboSym = topComboParamString "binanceSymbol" combo <|> topComboParamString "symbol" combo
      symOk = maybe False (\s -> normalizeSymbol s == sym) comboSym
      intervalOk =
        case mInterval of
          Nothing -> True
          Just interval ->
            case topComboParamString "interval" combo of
              Nothing -> True
              Just v -> v == interval
      platformOk =
        case topComboParamString "platform" combo of
          Nothing -> True
          Just p -> normalizeKey p == "binance"
   in symOk && intervalOk && platformOk

bestTopComboFromList :: [TopCombo] -> Maybe TopCombo
bestTopComboFromList combos =
  case sortOn topComboRankKey combos of
    [] -> Nothing
    (c : _) -> Just c

bestTopCombo :: TopCombosExport -> Maybe TopCombo
bestTopCombo export = bestTopComboFromList (tceCombos export)

bestTopComboForSymbol :: String -> Maybe String -> TopCombosExport -> Maybe TopCombo
bestTopComboForSymbol symRaw mInterval export =
  bestTopComboFromList (filter (topComboMatchesSymbol symRaw mInterval) (tceCombos export))

applyTopComboForStart :: Args -> TopCombo -> Either String Args
applyTopComboForStart base combo = do
  args0 <- parseTopComboToArgs base combo
  let mInterval = topComboParamString "interval" combo
      mBars = topComboParamInt "bars" combo
      intervalOk =
        case mInterval of
          Nothing -> Nothing
          Just v | isPlatformInterval (argPlatform args0) v -> Just v
          _ -> Nothing
      barsOk =
        case mBars of
          Nothing -> Nothing
          Just n -> Just (max 0 n)
      args1 =
        case intervalOk of
          Nothing -> args0
          Just v -> args0 { argInterval = v }
      args2 =
        case barsOk of
          Nothing -> args1
          Just n -> args1 { argBars = Just n }
  pure args2

handleOptimizerRun ::
  ApiRequestLimits ->
  FilePath ->
  FilePath ->
  Wai.Request ->
  (Wai.Response -> IO Wai.ResponseReceived) ->
  IO Wai.ResponseReceived
handleOptimizerRun reqLimits projectRoot optimizerTmp req respond = do
  payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid optimizer payload: "
  case payloadOrErr of
    Left resp -> respond resp
    Right payload -> do
      ts <- fmap (floor . (* 1000)) getPOSIXTime
      randId <- randomIO :: IO Word64
      topJsonPath <- resolveOptimizerCombosPath optimizerTmp
      let recordsPath = optimizerTmp </> printf "optimizer-%d-%016x.jsonl" (ts :: Integer) randId
          maxOutputBytes = arlMaxOptimizerOutputBytes reqLimits
          truncateOut = truncateProcessOutput maxOutputBytes
      argsOrErr <- prepareOptimizerArgs recordsPath payload
      case argsOrErr of
        Left msg -> respond (jsonError status400 msg)
        Right args -> do
          runResult <- runOptimizerProcess projectRoot recordsPath args
          case runResult of
            Left (msg, out, err) ->
              respond
                ( jsonValue
                    status500
                    (object ["error" .= msg, "stdout" .= truncateOut out, "stderr" .= truncateOut err])
                )
            Right resp -> do
              let resp' =
                    resp
                      { arrStdout = truncateOut (arrStdout resp)
                      , arrStderr = truncateOut (arrStderr resp)
                      }
              maxCombos <- optimizerMaxCombosFromEnv
              mergeResult <- runMergeTopCombos projectRoot topJsonPath recordsPath maxCombos
              case mergeResult of
                Left (msg, out, err) ->
                  respond
                    ( jsonValue
                        status500
                        (object ["error" .= msg, "stdout" .= truncateOut out, "stderr" .= truncateOut err])
                    )
                Right _ -> respond (jsonValue status200 resp')

handleOptimizerCombos ::
  FilePath ->
  FilePath ->
  (Wai.Response -> IO Wai.ResponseReceived) ->
  IO Wai.ResponseReceived
handleOptimizerCombos projectRoot optimizerTmp respond = do
  topJsonPath <- resolveOptimizerCombosPath optimizerTmp
  let tmpPath = optimizerTmp </> optimizerCombosFileName
      fallbackPath = projectRoot </> "web" </> "public" </> optimizerCombosFileName
  topVal <- readTopCombosValue topJsonPath
  tmpVal <-
    if tmpPath /= topJsonPath
      then readTopCombosValueLocal tmpPath
      else pure (Left "missing")
  fallbackVal <-
    if fallbackPath /= topJsonPath && fallbackPath /= tmpPath
      then readTopCombosValueLocal fallbackPath
      else pure (Left "missing")
  let allVals = [topVal, tmpVal, fallbackVal]
  now <- getTimestampMs

  let combosBySource =
        map
          (either (const ([], Nothing)) (\val -> (extractCombos val, extractPayloadSource val)))
          allVals
      combos = concatMap fst combosBySource
      payloadSources =
        concatMap
          (\(cs, src) -> if null cs then [] else maybeToList src)
          combosBySource
      payloadSource = listToMaybe payloadSources
  if null combos
    then respond (jsonError status404 "Optimizer combos not available yet.")
    else do
      let combosSorted = sortOn comboKey combos
          combosRanked = zipWith addRank [1 ..] combosSorted
          out =
            object
              [ "generatedAtMs" .= now
              , "source" .= ("optimizer/combos" :: String)
              , "payloadSource" .= payloadSource
              , "payloadSources" .= payloadSources
              , "combos" .= combosRanked
              ]
      respond (jsonValue status200 out)
  where
    extractCombos :: Aeson.Value -> [Aeson.Value]
    extractCombos val =
      case val of
        Aeson.Object o ->
          case KM.lookup "combos" o of
            Just (Aeson.Array arr) -> V.toList arr
            _ -> []
        _ -> []

    extractPayloadSource :: Aeson.Value -> Maybe String
    extractPayloadSource val =
      case val of
        Aeson.Object o -> KM.lookup "source" o >>= AT.parseMaybe parseJSON >>= cleanPayloadSource
        _ -> Nothing

    cleanPayloadSource :: String -> Maybe String
    cleanPayloadSource raw =
      let s = trim raw
       in if null s then Nothing else Just s

    comboMetric key val =
      case val of
        Aeson.Object o -> KM.lookup key o >>= AT.parseMaybe parseJSON
        _ -> Nothing

    comboKey :: Aeson.Value -> (Double, Double, Int)
    comboKey v =
      let eq = fromMaybe 0 (comboMetric "finalEquity" v)
          score = fromMaybe (negate (1 / 0)) (comboMetric "score" v)
          rank =
            case v of
              Aeson.Object o -> fromMaybe maxBound (KM.lookup "rank" o >>= AT.parseMaybe parseJSON)
              _ -> maxBound
       in (negate eq, negate score, rank)

    addRank :: Int -> Aeson.Value -> Aeson.Value
    addRank rank val =
      case val of
        Aeson.Object o -> Aeson.Object (KM.insert "rank" (toJSON rank) o)
        other -> other

handleMetrics :: Metrics -> BotController -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleMetrics metrics botCtrl respond = do
  states <- botGetStates botCtrl
  body <- renderMetricsText metrics (not (null states))
  respond (textValue status200 body)

handleOps :: Maybe OpsStore -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleOps mOps req respond =
  case mOps of
    Nothing ->
      respond
        ( jsonValue
            status200
            ( object
                [ "enabled" .= False
                , "hint" .= ("Set TRADER_OPS_DIR or TRADER_STATE_DIR to enable ops persistence." :: String)
                , "ops" .= ([] :: [PersistedOperation])
                ]
            )
        )
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

handleSignal :: ApiRequestLimits -> ApiCache -> Maybe OpsStore -> ApiComputeLimits -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleSignal reqLimits apiCache mOps limits baseArgs req respond = do
  payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
  case payloadOrErr of
    Left resp -> respond resp
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
          case validateArgs args of
            Left e -> respond (jsonError status400 e)
            Right args1 -> do
              case validateApiComputeLimits limits args1 of
                Left e -> respond (jsonError status400 e)
                Right argsOk -> do
                  let noCache = requestWantsNoCache req
                  r <-
                    try
                      ( if noCache
                          then computeLatestSignalFromArgs mOps argsOk
                          else computeLatestSignalFromArgsCached apiCache mOps argsOk
                      )
                      :: IO (Either SomeException LatestSignal)
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

handleSignalAsync :: ApiRequestLimits -> ApiCache -> Maybe OpsStore -> ApiComputeLimits -> JobStore LatestSignal -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleSignalAsync reqLimits apiCache mOps limits store baseArgs req respond = do
  payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
  case payloadOrErr of
    Left resp -> respond resp
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
          case validateArgs args of
            Left e -> respond (jsonError status400 e)
            Right args1 -> do
              case validateApiComputeLimits limits args1 of
                Left e -> respond (jsonError status400 e)
                Right argsOk -> do
                  let paramsJson = Just (toJSON (sanitizeApiParams params))
                      argsJson = Just (argsPublicJson argsOk)
                      noCache = requestWantsNoCache req
                  r <-
                    startJob store $ do
                      sig <-
                        if noCache
                          then computeLatestSignalFromArgs mOps argsOk
                          else computeLatestSignalFromArgsCached apiCache mOps argsOk
                      opsAppendMaybe mOps "signal" paramsJson argsJson (Just (toJSON sig)) Nothing
                      pure sig
                  case r of
                    Left e -> respond (jsonError status429 e)
                    Right jobId -> respond (jsonValue status202 (object ["jobId" .= jobId]))

handleTrade :: ApiRequestLimits -> Maybe OpsStore -> ApiComputeLimits -> Metrics -> Maybe Journal -> Maybe Webhook -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleTrade reqLimits mOps limits metrics mJournal mWebhook baseArgs req respond = do
  payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
  case payloadOrErr of
    Left resp -> respond resp
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
                  r <- try (computeTradeFromArgs mOps argsOk) :: IO (Either SomeException ApiTradeResponse)
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
                      webhookNotifyMaybe mWebhook (webhookEventTradeOrder argsOk (atrSignal out) (atrOrder out))
                      respond (jsonValue status200 out)

handleTradeAsync :: ApiRequestLimits -> Maybe OpsStore -> ApiComputeLimits -> JobStore ApiTradeResponse -> Metrics -> Maybe Journal -> Maybe Webhook -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleTradeAsync reqLimits mOps limits store metrics mJournal mWebhook baseArgs req respond = do
  payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
  case payloadOrErr of
    Left resp -> respond resp
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
                      out <- computeTradeFromArgs mOps argsOk
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
                      webhookNotifyMaybe mWebhook (webhookEventTradeOrder argsOk (atrSignal out) (atrOrder out))
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

handleBacktest :: ApiRequestLimits -> ApiCache -> Maybe OpsStore -> ApiComputeLimits -> BacktestGate -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBacktest reqLimits apiCache mOps limits backtestGate baseArgs req respond = do
  payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
  case payloadOrErr of
    Left resp -> respond resp
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
              let noCache = requestWantsNoCache req
                  computeAction =
                    if noCache
                      then computeBacktestFromArgs mOps argsOk
                      else computeBacktestFromArgsCached apiCache mOps argsOk
              r <- runBacktestWithGateWait backtestGate computeAction
              case r of
                Left failure ->
                  let (st, msg) = backtestFailureToHttp backtestGate failure
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

handleBacktestAsync :: ApiRequestLimits -> ApiCache -> Maybe OpsStore -> ApiComputeLimits -> BacktestGate -> JobStore Aeson.Value -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBacktestAsync reqLimits apiCache mOps limits backtestGate store baseArgs req respond = do
  payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
  case payloadOrErr of
    Left resp -> respond resp
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
                  noCache = requestWantsNoCache req
              r <-
                startJob store $ do
                  let computeAction =
                        if noCache
                          then computeBacktestFromArgs mOps argsOk
                          else computeBacktestFromArgsCached apiCache mOps argsOk
                  gateResult <- runBacktestWithGateWait backtestGate computeAction
                  case gateResult of
                    Left failure -> throwIO (userError (backtestFailureMessage backtestGate failure))
                    Right out -> do
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

handleBinanceKeys :: ApiRequestLimits -> Maybe OpsStore -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBinanceKeys reqLimits mOps baseArgs req respond = do
  payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
  case payloadOrErr of
    Left resp -> respond resp
    Right params ->
      case argsFromApi baseArgs params of
        Left e -> respond (jsonError status400 e)
        Right args0 -> do
          if argPlatform args0 /= PlatformBinance
            then respond (jsonError status400 ("Binance keys require platform=binance (got " ++ platformCode (argPlatform args0) ++ ")."))
            else do
              r <- try (computeBinanceKeysStatusFromArgs mOps args0) :: IO (Either SomeException ApiBinanceKeysStatus)
              case r of
                Left ex ->
                  let (st, msg) = exceptionToHttp ex
                   in respond (jsonError st msg)
                Right out -> respond (jsonValue status200 out)

handleCoinbaseKeys :: ApiRequestLimits -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleCoinbaseKeys reqLimits baseArgs req respond = do
  payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
  case payloadOrErr of
    Left resp -> respond resp
    Right params ->
      case argsFromApi baseArgs params of
        Left e -> respond (jsonError status400 e)
        Right args0 -> do
          if argPlatform args0 /= PlatformCoinbase
            then respond (jsonError status400 ("Coinbase keys require platform=coinbase (got " ++ platformCode (argPlatform args0) ++ ")."))
            else do
              r <- try (computeCoinbaseKeysStatusFromArgs args0) :: IO (Either SomeException ApiCoinbaseKeysStatus)
              case r of
                Left ex ->
                  let (st, msg) = exceptionToHttp ex
                   in respond (jsonError st msg)
                Right out -> respond (jsonValue status200 out)

parseMarketForListenKey :: Args -> Maybe String -> Either String BinanceMarket
parseMarketForListenKey baseArgs raw =
  case raw of
    Nothing -> Right (argBinanceMarket baseArgs)
    Just r ->
      case normalizeKey r of
        "spot" -> Right MarketSpot
        "margin" -> Right MarketMargin
        "futures" -> Right MarketFutures
        other -> Left ("Invalid market: " ++ show other ++ " (expected spot|margin|futures)")

resolveTestnetForListenKey :: Args -> Maybe Bool -> Bool
resolveTestnetForListenKey baseArgs raw = maybe (argBinanceTestnet baseArgs) id raw

binanceUserStreamWsBase :: BinanceMarket -> Bool -> String
binanceUserStreamWsBase market testnet =
  case market of
    MarketFutures ->
      if testnet
        then "wss://stream.binancefuture.com/ws"
        else "wss://fstream.binance.com/ws"
    _ ->
      if testnet
        then "wss://testnet.binance.vision/ws"
        else "wss://stream.binance.com:9443/ws"

binanceUserStreamWsUrl :: BinanceMarket -> Bool -> String -> String
binanceUserStreamWsUrl market testnet listenKey =
  binanceUserStreamWsBase market testnet ++ "/" ++ listenKey

handleBinanceListenKey :: ApiRequestLimits -> Maybe OpsStore -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBinanceListenKey reqLimits mOps baseArgs req respond = do
  if argPlatform baseArgs /= PlatformBinance
    then respond (jsonError status400 ("Binance listenKey requires platform=binance (got " ++ platformCode (argPlatform baseArgs) ++ ")."))
    else do
      payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
      case payloadOrErr of
        Left resp -> respond resp
        Right params -> do
          let testnet = resolveTestnetForListenKey baseArgs (alsBinanceTestnet params)
          case parseMarketForListenKey baseArgs (alsMarket params) of
            Left e -> respond (jsonError status400 e)
            Right market -> do
              if market == MarketMargin && testnet
                then respond (jsonError status400 "binanceTestnet is not supported for margin operations")
                else do
                  apiKey <- resolveEnv "BINANCE_API_KEY" (alsBinanceApiKey params <|> argBinanceApiKey baseArgs)
                  apiSecret <- resolveEnv "BINANCE_API_SECRET" (alsBinanceApiSecret params <|> argBinanceApiSecret baseArgs)
                  let baseUrl =
                        case market of
                          MarketFutures -> if testnet then binanceFuturesTestnetBaseUrl else binanceFuturesBaseUrl
                          _ -> if testnet then binanceTestnetBaseUrl else binanceBaseUrl
                  env <- newBinanceEnvWithOps mOps market baseUrl (BS.pack <$> apiKey) (BS.pack <$> apiSecret)
                  r <- try (createListenKey env) :: IO (Either SomeException String)
                  case r of
                    Left ex ->
                      let (st, msg) = exceptionToHttp ex
                       in respond (jsonError st msg)
                    Right lk -> do
                      let resp =
                            ApiListenKeyResponse
                              { alrListenKey = lk
                              , alrMarket = marketCode market
                              , alrTestnet = testnet
                              , alrWsUrl = binanceUserStreamWsUrl market testnet lk
                              , alrKeepAliveMs = 25 * 60 * 1000
                              }
                      respond (jsonValue status200 resp)

handleBinanceListenKeyKeepAlive :: ApiRequestLimits -> Maybe OpsStore -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBinanceListenKeyKeepAlive reqLimits mOps baseArgs req respond = do
  if argPlatform baseArgs /= PlatformBinance
    then respond (jsonError status400 ("Binance listenKey keepAlive requires platform=binance (got " ++ platformCode (argPlatform baseArgs) ++ ")."))
    else do
      payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
      case payloadOrErr of
        Left resp -> respond resp
        Right params -> do
          let testnet = resolveTestnetForListenKey baseArgs (alaBinanceTestnet params)
          case parseMarketForListenKey baseArgs (alaMarket params) of
            Left e -> respond (jsonError status400 e)
            Right market -> do
              if market == MarketMargin && testnet
                then respond (jsonError status400 "binanceTestnet is not supported for margin operations")
                else do
                  apiKey <- resolveEnv "BINANCE_API_KEY" (alaBinanceApiKey params <|> argBinanceApiKey baseArgs)
                  apiSecret <- resolveEnv "BINANCE_API_SECRET" (alaBinanceApiSecret params <|> argBinanceApiSecret baseArgs)
                  let baseUrl =
                        case market of
                          MarketFutures -> if testnet then binanceFuturesTestnetBaseUrl else binanceFuturesBaseUrl
                          _ -> if testnet then binanceTestnetBaseUrl else binanceBaseUrl
                  env <- newBinanceEnvWithOps mOps market baseUrl (BS.pack <$> apiKey) (BS.pack <$> apiSecret)
                  r <- try (keepAliveListenKey env (alaListenKey params)) :: IO (Either SomeException ())
                  case r of
                    Left ex ->
                      let (st, msg) = exceptionToHttp ex
                       in respond (jsonError st msg)
                    Right _ -> do
                      now <- getTimestampMs
                      respond (jsonValue status200 (object ["ok" .= True, "atMs" .= now]))

handleBinanceListenKeyClose :: ApiRequestLimits -> Maybe OpsStore -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBinanceListenKeyClose reqLimits mOps baseArgs req respond = do
  if argPlatform baseArgs /= PlatformBinance
    then respond (jsonError status400 ("Binance listenKey close requires platform=binance (got " ++ platformCode (argPlatform baseArgs) ++ ")."))
    else do
      payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
      case payloadOrErr of
        Left resp -> respond resp
        Right params -> do
          let testnet = resolveTestnetForListenKey baseArgs (alaBinanceTestnet params)
          case parseMarketForListenKey baseArgs (alaMarket params) of
            Left e -> respond (jsonError status400 e)
            Right market -> do
              if market == MarketMargin && testnet
                then respond (jsonError status400 "binanceTestnet is not supported for margin operations")
                else do
                  apiKey <- resolveEnv "BINANCE_API_KEY" (alaBinanceApiKey params <|> argBinanceApiKey baseArgs)
                  apiSecret <- resolveEnv "BINANCE_API_SECRET" (alaBinanceApiSecret params <|> argBinanceApiSecret baseArgs)
                  let baseUrl =
                        case market of
                          MarketFutures -> if testnet then binanceFuturesTestnetBaseUrl else binanceFuturesBaseUrl
                          _ -> if testnet then binanceTestnetBaseUrl else binanceBaseUrl
                  env <- newBinanceEnvWithOps mOps market baseUrl (BS.pack <$> apiKey) (BS.pack <$> apiSecret)
                  r <- try (closeListenKey env (alaListenKey params)) :: IO (Either SomeException ())
                  case r of
                    Left ex ->
                      let (st, msg) = exceptionToHttp ex
                       in respond (jsonError st msg)
                    Right _ -> do
                      now <- getTimestampMs
                      respond (jsonValue status200 (object ["ok" .= True, "atMs" .= now]))

handleBinanceTrades :: ApiRequestLimits -> Maybe OpsStore -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBinanceTrades reqLimits mOps baseArgs req respond = do
  if argPlatform baseArgs /= PlatformBinance
    then respond (jsonError status400 ("Binance trades require platform=binance (got " ++ platformCode (argPlatform baseArgs) ++ ")."))
    else do
      payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
      case payloadOrErr of
        Left resp -> respond resp
        Right params -> do
          let testnet = resolveTestnetForListenKey baseArgs (abrBinanceTestnet params)
          case parseMarketForListenKey baseArgs (abrMarket params) of
            Left e -> respond (jsonError status400 e)
            Right market -> do
              if market == MarketMargin && testnet
                then respond (jsonError status400 "binanceTestnet is not supported for margin operations")
                else do
                  apiKey <- resolveEnv "BINANCE_API_KEY" (abrBinanceApiKey params <|> argBinanceApiKey baseArgs)
                  apiSecret <- resolveEnv "BINANCE_API_SECRET" (abrBinanceApiSecret params <|> argBinanceApiSecret baseArgs)
                  let baseUrl =
                        case market of
                          MarketFutures -> if testnet then binanceFuturesTestnetBaseUrl else binanceFuturesBaseUrl
                          _ -> if testnet then binanceTestnetBaseUrl else binanceBaseUrl
                  env <- newBinanceEnvWithOps mOps market baseUrl (BS.pack <$> apiKey) (BS.pack <$> apiSecret)
                  let symbolsRaw =
                        case abrSymbols params of
                          Just xs -> map trim xs
                          Nothing ->
                            case fmap trim (abrSymbol params) of
                              Just s | not (null s) -> [s]
                              _ -> []
                      symbols = filter (not . null) (dedupeStable (map normalizeSymbol symbolsRaw))
                      allSymbols = null symbols
                      limit = abrLimit params
                      startTime = abrStartTimeMs params
                      endTime = abrEndTimeMs params
                      fromId = abrFromId params
                  if allSymbols && market /= MarketFutures
                    then respond (jsonError status400 "binance trades require symbol for spot/margin markets")
                    else do
                      trades <- do
                        if allSymbols
                          then fetchAccountTrades env Nothing limit startTime endTime fromId
                          else fmap concat $
                            mapM
                              (\sym -> fetchAccountTrades env (Just sym) limit startTime endTime fromId)
                              symbols
                      let tradesSorted = sortOn (negate . btTime) trades
                      now <- getTimestampMs
                      respond $
                        jsonValue
                          status200
                          ApiBinanceTradesResponse
                            { abtrMarket = marketCode market
                            , abtrTestnet = testnet
                            , abtrSymbols = symbols
                            , abtrAllSymbols = allSymbols
                            , abtrTrades = tradesSorted
                            , abtrFetchedAtMs = now
                            }

handleBinancePositions :: ApiRequestLimits -> Maybe OpsStore -> Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBinancePositions reqLimits mOps baseArgs req respond = do
  if argPlatform baseArgs /= PlatformBinance
    then respond (jsonError status400 ("Binance positions require platform=binance (got " ++ platformCode (argPlatform baseArgs) ++ ")."))
    else do
      payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
      case payloadOrErr of
        Left resp -> respond resp
        Right params -> do
          let testnet = resolveTestnetForListenKey baseArgs (abpBinanceTestnet params)
          case parseMarketForListenKey baseArgs (abpMarket params) of
            Left e -> respond (jsonError status400 e)
            Right market -> do
              if market /= MarketFutures
                then respond (jsonError status400 "binance positions require market=futures")
                else do
                  apiKey <- resolveEnv "BINANCE_API_KEY" (abpBinanceApiKey params <|> argBinanceApiKey baseArgs)
                  apiSecret <- resolveEnv "BINANCE_API_SECRET" (abpBinanceApiSecret params <|> argBinanceApiSecret baseArgs)
                  let baseUrl = if testnet then binanceFuturesTestnetBaseUrl else binanceFuturesBaseUrl
                  env <- newBinanceEnvWithOps mOps market baseUrl (BS.pack <$> apiKey) (BS.pack <$> apiSecret)
                  r <- try (fetchFuturesPositionRisks env) :: IO (Either SomeException [FuturesPositionRisk])
                  case r of
                    Left ex ->
                      let (st, msg) = exceptionToHttp ex
                       in respond (jsonError st msg)
                    Right positions -> do
                      let interval = fromMaybe (argInterval baseArgs) (abpInterval params)
                          limitRaw = fromMaybe 120 (abpLimit params)
                          limitSafe = max 10 (min 1000 limitRaw)
                          openPositions = filter (\p -> abs (fprPositionAmt p) > 1e-12) positions
                          toApiPosition p =
                            ApiBinancePosition
                              { abpSymbol = fprSymbol p
                              , abpPositionAmt = fprPositionAmt p
                              , abpEntryPrice = fprEntryPrice p
                              , abpMarkPrice = fprMarkPrice p
                              , abpUnrealizedPnl = fprUnrealizedProfit p
                              , abpLiquidationPrice = fprLiquidationPrice p
                              , abpBreakEvenPrice = fprBreakEvenPrice p
                              , abpLeverage = fprLeverage p
                              , abpMarginType = fprMarginType p
                              , abpPositionSide = fprPositionSide p
                              }
                      chartsRaw <-
                        forM openPositions $ \pos -> do
                          let sym = fprSymbol pos
                          kr <- try (fetchKlines env sym interval limitSafe) :: IO (Either SomeException [Kline])
                          pure $
                            case kr of
                              Left _ -> Nothing
                              Right ks ->
                                Just
                                  ApiBinancePositionChart
                                    { abpcSymbol = sym
                                    , abpcOpenTimes = map kOpenTime ks
                                    , abpcPrices = map kClose ks
                                    }
                      now <- getTimestampMs
                      respond $
                        jsonValue
                          status200
                          ApiBinancePositionsResponse
                            { abprMarket = marketCode market
                            , abprTestnet = testnet
                            , abprInterval = interval
                            , abprLimit = limitSafe
                            , abprPositions = map toApiPosition openPositions
                            , abprCharts = catMaybes chartsRaw
                            , abprFetchedAtMs = now
                            }

handleBotStart :: ApiRequestLimits -> Maybe OpsStore -> ApiComputeLimits -> Metrics -> Maybe Journal -> Maybe Webhook -> Maybe FilePath -> FilePath -> Args -> BotController -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBotStart reqLimits mOps limits metrics mJournal mWebhook mBotStateDir optimizerTmp baseArgs botCtrl req respond = do
  payloadOrErr <- decodeRequestBodyLimited reqLimits req "Invalid JSON: "
  case payloadOrErr of
    Left resp -> respond resp
    Right params ->
      case argsFromApi baseArgs params of
        Left e -> respond (jsonError status400 e)
        Right args0 -> do
          let argsBase = args0 { argTradeOnly = True }
              tradeEnabled = maybe False id (apBotTrade params)
          symbolsOrErr <- resolveBotSymbols argsBase params
          let requestedSymbols =
                case symbolsOrErr of
                  Left _ -> []
                  Right syms -> syms
          orphanSymbols <-
            if tradeEnabled && platformSupportsLiveBot (argPlatform argsBase)
              then resolveOrphanOpenPositionSymbols mOps limits optimizerTmp argsBase requestedSymbols
              else pure []
          let symbols = dedupeStable (requestedSymbols ++ orphanSymbols)
              errorMsg =
                case symbolsOrErr of
                  Left e -> e
                  Right _ -> "bot/start requires binanceSymbol or botSymbols"
          if null symbols
            then respond (jsonError status400 errorMsg)
            else do
              let allowExistingBase = length requestedSymbols > 1
                  orphanNorm = map normalizeSymbol orphanSymbols
                  allowExistingFor sym = allowExistingBase || normalizeSymbol sym `elem` orphanNorm
                  noAdoptRequirement = AdoptRequirement False False
              results <- forM symbols $ \sym -> do
                let argsSym = argsBase { argBinanceSymbol = Just sym }
                    allowExisting = allowExistingFor sym
                adoptReqOrErr <-
                  if tradeEnabled && platformSupportsLiveBot (argPlatform argsSym)
                    then resolveAdoptionRequirement mOps argsSym sym
                    else pure (Right noAdoptRequirement)
                case adoptReqOrErr of
                  Left err -> pure (sym, Left err)
                  Right adoptReq -> do
                    argsCombo <-
                      if arActive adoptReq
                        then pure argsSym
                        else applyLatestTopCombo optimizerTmp limits sym argsSym adoptReq
                    case validateApiComputeLimits limits argsCombo of
                      Left err -> pure (sym, Left err)
                      Right argsOk -> do
                        r <- botStartSymbol allowExisting mOps metrics mJournal mWebhook mBotStateDir optimizerTmp limits adoptReq botCtrl argsOk params sym
                        pure (sym, r)

              let errors =
                    [ object ["symbol" .= sym, "error" .= err]
                    | (sym, Left err) <- results
                    ]
              statuses <- forM [outcome | (_, Right outcome) <- results] $ \outcome ->
                case bsoState outcome of
                  BotStarting rt -> pure (botStartingJson rt)
                  BotRunning rt -> do
                    st <- readMVar (brStateVar rt)
                    pure (botStatusJson st)

              now <- getTimestampMs
              forM_ [outcome | (_, Right outcome) <- results, bsoStarted outcome] $ \outcome ->
                case bsoState outcome of
                  BotStarting rt -> do
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
                      (Just (argsPublicJson (bsrArgs rt)))
                      ( Just
                          ( object
                              [ "symbol" .= bsrSymbol rt
                              , "market" .= marketCode (argBinanceMarket (bsrArgs rt))
                              , "interval" .= argInterval (bsrArgs rt)
                              , "tradeEnabled" .= bsTradeEnabled settings
                              , "botAdoptExistingPosition" .= bsAdoptExistingPosition settings
                              , "botPollSeconds" .= bsPollSeconds settings
                              , "botOnlineEpochs" .= bsOnlineEpochs settings
                              , "botTrainBars" .= bsTrainBars settings
                              , "botMaxPoints" .= bsMaxPoints settings
                              ]
                          )
                      )
                      Nothing
                  _ -> pure ()

              case (symbols, errors, statuses) of
                ([_], [], [status]) -> respond (jsonValue status202 status)
                _ ->
                  if null statuses
                    then
                      case errors of
                        [] -> respond (jsonError status400 "Failed to start bot.")
                        errs -> do
                          let errorMsgFromValue err =
                                case err of
                                  Aeson.Object o ->
                                    case KM.lookup "error" o of
                                      Just (Aeson.String t) -> Just (T.unpack t)
                                      _ -> Nothing
                                  _ -> Nothing
                              msg =
                                case errs of
                                  [err] -> fromMaybe "Failed to start bot." (errorMsgFromValue err)
                                  _ -> "Failed to start bot."
                              payload = object ["error" .= msg, "errors" .= errs]
                          respond (jsonValue status400 payload)
                    else do
                      let base = botStatusJsonMulti statuses
                          payload =
                            case base of
                              Aeson.Object o ->
                                let o' =
                                      if null errors
                                        then o
                                        else KM.insert "errors" (toJSON errors) o
                                 in Aeson.Object o'
                              _ -> base
                      respond (jsonValue status202 payload)

handleBotStop :: Maybe OpsStore -> Maybe Journal -> Maybe Webhook -> Maybe FilePath -> BotController -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBotStop mOps mJournal mWebhook mBotStateDir botCtrl req respond = do
  let mSymbol =
        case lookup (BS.pack "symbol") (Wai.queryString req) of
          Just (Just raw) -> Just (BS.unpack raw)
          _ -> Nothing
  stopped <- botStop botCtrl mSymbol
  now <- getTimestampMs
  let stoppedSymbols = map botSymbol stopped
  journalWriteMaybe mJournal (object ["type" .= ("bot.stop" :: String), "atMs" .= now, "symbols" .= stoppedSymbols])
  opsAppendMaybe mOps "bot.stop" Nothing Nothing (Just (object ["symbols" .= stoppedSymbols])) Nothing
  forM_ stopped (botStatusLogMaybe mOps False)
  webhookNotifyMaybe mWebhook (webhookEventBotStop stoppedSymbols)

  let snaps =
        [ BotStatusSnapshot
            { bssSavedAtMs = now
            , bssStatus = botStatusJson st
            }
        | st <- stopped
        ]
  forM_ stopped $ \st -> do
    let snap =
          BotStatusSnapshot
            { bssSavedAtMs = now
            , bssStatus = botStatusJson st
            }
    writeBotStatusSnapshotMaybe mBotStateDir (botSymbol st) snap

  case snaps of
    [] -> do
      case mSymbol of
        Nothing -> respond (jsonValue status200 botStoppedJson)
        Just sym -> do
          mSnap <- readBotStatusSnapshotMaybe mBotStateDir sym
          case mSnap of
            Nothing -> respond (jsonValue status200 botStoppedJson)
            Just snap -> respond (jsonValue status200 (botStoppedSnapshotJson snap))
    [snap] -> respond (jsonValue status200 (botStoppedSnapshotJson snap))
    _ -> respond (jsonValue status200 (botStoppedMultiJson snaps))

handleBotStatus :: ApiRequestLimits -> BotController -> Maybe FilePath -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBotStatus reqLimits botCtrl mBotStateDir req respond = do
  let q = Wai.queryString req
      maxTail = arlMaxBotStatusTail reqLimits
      tailN =
        case lookup (BS.pack "tail") q of
          Just (Just raw) -> readMaybe (BS.unpack raw) >>= \n -> if n > 0 then Just (min n maxTail) else Nothing
          _ -> Nothing
      mSymbol =
        case lookup (BS.pack "symbol") q of
          Just (Just raw) -> Just (BS.unpack raw)
          _ -> Nothing

  case mSymbol of
    Just sym -> do
      mSt <- botGetStateFor botCtrl sym
      case mSt of
        Just st -> do
          let st' = maybe st (`botStateTail` st) tailN
          respond (jsonValue status200 (botStatusJson st'))
        Nothing -> do
          mSnap <- readBotStatusSnapshotMaybe mBotStateDir sym
          case mSnap of
            Nothing -> respond (jsonValue status200 botStoppedJson)
            Just snap -> respond (jsonValue status200 (botStoppedSnapshotJson snap))
    Nothing -> do
      mrt <- readMVar (bcRuntime botCtrl)
      if HM.null mrt
        then do
          snaps <- readBotStatusSnapshotsMaybe mBotStateDir
          case snaps of
            [] -> respond (jsonValue status200 botStoppedJson)
            [snap] -> respond (jsonValue status200 (botStoppedSnapshotJson snap))
            _ -> respond (jsonValue status200 (botStoppedMultiJson snaps))
        else do
          let entries = sortOn fst (HM.toList mrt)
          case entries of
            [(_sym, state)] ->
              case state of
                BotStarting rt -> respond (jsonValue status200 (botStartingJson rt))
                BotRunning rt -> do
                  st <- readMVar (brStateVar rt)
                  let st' = maybe st (`botStateTail` st) tailN
                  respond (jsonValue status200 (botStatusJson st'))
            _ -> do
              statuses <-
                forM entries $ \(_sym, state) ->
                  case state of
                    BotStarting rt -> pure (botStartingJson rt)
                    BotRunning rt -> do
                      st <- readMVar (brStateVar rt)
                      let st' = maybe st (`botStateTail` st) tailN
                      pure (botStatusJson st')
              respond (jsonValue status200 (botStatusJsonMulti statuses))

argsFromApi :: Args -> ApiParams -> Either String Args
argsFromApi baseArgs p = do
  platform <-
    case apPlatform p of
      Nothing -> Right (argPlatform baseArgs)
      Just raw -> parsePlatform raw
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

  tuneObjective <-
    case apTuneObjective p of
      Nothing -> Right (argTuneObjective baseArgs)
      Just raw -> parseTuneObjective raw

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

      openThr =
        pick
          (apOpenThreshold p <|> apThreshold p)
          (argOpenThreshold baseArgs)
      closeThr =
        case (apCloseThreshold p <|> apThreshold p) of
          Just v -> v
          Nothing ->
            case apOpenThreshold p of
              Just _ -> openThr
              Nothing -> argCloseThreshold baseArgs

      args =
        baseArgs
          { argData = pickMaybe (apData p) (argData baseArgs)
          , argPriceCol = pick (apPriceColumn p) (argPriceCol baseArgs)
          , argHighCol = pickMaybe (apHighColumn p) (argHighCol baseArgs)
          , argLowCol = pickMaybe (apLowColumn p) (argLowCol baseArgs)
          , argBinanceSymbol = pickMaybe (apBinanceSymbol p) (argBinanceSymbol baseArgs)
          , argPlatform = platform
          , argBinanceFutures = futuresFlag
          , argBinanceMargin = marginFlag
          , argInterval = pick (apInterval p) (argInterval baseArgs)
          , argBars = pickMaybe (apBars p) (argBars baseArgs)
          , argLookbackWindow = pick (apLookbackWindow p) (argLookbackWindow baseArgs)
          , argLookbackBars = pickMaybe (apLookbackBars p) (argLookbackBars baseArgs)
          , argBinanceTestnet = pick (apBinanceTestnet p) (argBinanceTestnet baseArgs)
          , argBinanceApiKey = pickMaybe (apBinanceApiKey p) (argBinanceApiKey baseArgs)
          , argBinanceApiSecret = pickMaybe (apBinanceApiSecret p) (argBinanceApiSecret baseArgs)
          , argCoinbaseApiKey = pickMaybe (apCoinbaseApiKey p) (argCoinbaseApiKey baseArgs)
          , argCoinbaseApiSecret = pickMaybe (apCoinbaseApiSecret p) (argCoinbaseApiSecret baseArgs)
          , argCoinbaseApiPassphrase = pickMaybe (apCoinbaseApiPassphrase p) (argCoinbaseApiPassphrase baseArgs)
          , argNormalization = norm
          , argHiddenSize = pick (apHiddenSize p) (argHiddenSize baseArgs)
          , argEpochs = pick (apEpochs p) (argEpochs baseArgs)
          , argLr = pick (apLr p) (argLr baseArgs)
          , argValRatio = pick (apValRatio p) (argValRatio baseArgs)
          , argBacktestRatio = pick (apBacktestRatio p) (argBacktestRatio baseArgs)
          , argTuneRatio = pick (apTuneRatio p) (argTuneRatio baseArgs)
          , argTuneObjective = tuneObjective
          , argTunePenaltyMaxDrawdown = pick (apTunePenaltyMaxDrawdown p) (argTunePenaltyMaxDrawdown baseArgs)
          , argTunePenaltyTurnover = pick (apTunePenaltyTurnover p) (argTunePenaltyTurnover baseArgs)
          , argMinRoundTrips = pick (apMinRoundTrips p) (argMinRoundTrips baseArgs)
          , argWalkForwardFolds = pick (apWalkForwardFolds p) (argWalkForwardFolds baseArgs)
          , argPatience = pick (apPatience p) (argPatience baseArgs)
          , argGradClip =
              case apGradClip p of
                Nothing -> argGradClip baseArgs
                Just g -> Just g
          , argSeed = pick (apSeed p) (argSeed baseArgs)
          , argKalmanDt = pick (apKalmanDt p) (argKalmanDt baseArgs)
          , argKalmanProcessVar = pick (apKalmanProcessVar p) (argKalmanProcessVar baseArgs)
          , argKalmanMeasurementVar = pick (apKalmanMeasurementVar p) (argKalmanMeasurementVar baseArgs)
          , argKalmanMarketTopN = pick (apKalmanMarketTopN p) (argKalmanMarketTopN baseArgs)
          , argOpenThreshold = openThr
          , argCloseThreshold = closeThr
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
          , argStopLossVolMult = pick (apStopLossVolMult p) (argStopLossVolMult baseArgs)
          , argTakeProfitVolMult = pick (apTakeProfitVolMult p) (argTakeProfitVolMult baseArgs)
          , argTrailingStopVolMult = pick (apTrailingStopVolMult p) (argTrailingStopVolMult baseArgs)
          , argMinHoldBars = pick (apMinHoldBars p) (argMinHoldBars baseArgs)
          , argCooldownBars = pick (apCooldownBars p) (argCooldownBars baseArgs)
          , argMaxHoldBars = pickMaybe (apMaxHoldBars p) (argMaxHoldBars baseArgs)
          , argMaxDrawdown = pickMaybe (apMaxDrawdown p) (argMaxDrawdown baseArgs)
          , argMaxDailyLoss = pickMaybe (apMaxDailyLoss p) (argMaxDailyLoss baseArgs)
          , argMinEdge = pick (apMinEdge p) (argMinEdge baseArgs)
          , argMinSignalToNoise = pick (apMinSignalToNoise p) (argMinSignalToNoise baseArgs)
          , argCostAwareEdge = pick (apCostAwareEdge p) (argCostAwareEdge baseArgs)
          , argEdgeBuffer = pick (apEdgeBuffer p) (argEdgeBuffer baseArgs)
          , argTrendLookback = pick (apTrendLookback p) (argTrendLookback baseArgs)
          , argMaxPositionSize = pick (apMaxPositionSize p) (argMaxPositionSize baseArgs)
          , argVolTarget = pickMaybe (apVolTarget p) (argVolTarget baseArgs)
          , argVolLookback = pick (apVolLookback p) (argVolLookback baseArgs)
          , argVolEwmaAlpha = pickMaybe (apVolEwmaAlpha p) (argVolEwmaAlpha baseArgs)
          , argVolFloor = pick (apVolFloor p) (argVolFloor baseArgs)
          , argVolScaleMax = pick (apVolScaleMax p) (argVolScaleMax baseArgs)
          , argMaxVolatility = pickMaybe (apMaxVolatility p) (argMaxVolatility baseArgs)
          , argRebalanceBars = pick (apRebalanceBars p) (argRebalanceBars baseArgs)
          , argRebalanceThreshold = pick (apRebalanceThreshold p) (argRebalanceThreshold baseArgs)
          , argRebalanceGlobal = pick (apRebalanceGlobal p) (argRebalanceGlobal baseArgs)
          , argRebalanceResetOnSignal = pick (apRebalanceResetOnSignal p) (argRebalanceResetOnSignal baseArgs)
          , argFundingRate = pick (apFundingRate p) (argFundingRate baseArgs)
          , argFundingBySide = pick (apFundingBySide p) (argFundingBySide baseArgs)
          , argFundingOnOpen = pick (apFundingOnOpen p) (argFundingOnOpen baseArgs)
          , argBlendWeight = pick (apBlendWeight p) (argBlendWeight baseArgs)
          , argTriLayer = pick (apTriLayer p) (argTriLayer baseArgs)
          , argTriLayerFastMult = pick (apTriLayerFastMult p) (argTriLayerFastMult baseArgs)
          , argTriLayerSlowMult = pick (apTriLayerSlowMult p) (argTriLayerSlowMult baseArgs)
          , argTriLayerCloudPadding = pick (apTriLayerCloudPadding p) (argTriLayerCloudPadding baseArgs)
          , argTriLayerRequirePriceAction = pick (apTriLayerPriceAction p) (argTriLayerRequirePriceAction baseArgs)
          , argLstmExitFlipBars = pick (apLstmExitFlipBars p) (argLstmExitFlipBars baseArgs)
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
          , argTuneStressVolMult = pick (apTuneStressVolMult p) (argTuneStressVolMult baseArgs)
          , argTuneStressShock = pick (apTuneStressShock p) (argTuneStressShock baseArgs)
          , argTuneStressWeight = pick (apTuneStressWeight p) (argTuneStressWeight baseArgs)
          }

  validateArgs args

dirLabel :: Maybe Int -> Maybe String
dirLabel d =
  case d of
    Just 1 -> Just "UP"
    Just (-1) -> Just "DOWN"
    _ -> Nothing

computeLatestSignalFromArgs :: Maybe OpsStore -> Args -> IO LatestSignal
computeLatestSignalFromArgs mOps args = do
  (series, mBinanceEnv) <- loadPrices mOps args
  let prices = psClose series
  ensureMinPriceRows args 2 prices
  let lookback = argLookback args
  computeTradeOnlySignal args lookback prices mBinanceEnv

computeTradeFromArgs :: Maybe OpsStore -> Args -> IO ApiTradeResponse
computeTradeFromArgs mOps args = do
  (series, mBinanceEnv) <- loadPrices mOps args
  let prices = psClose series
  ensureMinPriceRows args 2 prices
  let lookback = argLookback args
  sig <- computeTradeOnlySignal args lookback prices mBinanceEnv
  order <-
    case argPlatform args of
      PlatformBinance ->
        case (argBinanceSymbol args, mBinanceEnv) of
          (Nothing, _) -> noOrder "No order: missing binanceSymbol."
          (_, Nothing) -> noOrder "No order: missing Binance environment (use binanceSymbol data source)."
          (Just sym, Just env) -> placeOrderForSignal args sym sig env
      PlatformCoinbase ->
        case argBinanceSymbol args of
          Nothing -> noOrder "No order: missing symbol."
          Just sym -> do
            envOrErr <- try (makeCoinbaseEnv args) :: IO (Either SomeException CoinbaseEnv)
            case envOrErr of
              Left ex -> noOrder ("No order: " ++ take 240 (show ex))
              Right env -> placeCoinbaseOrderForSignal args sym sig env
      _ -> noOrder "No order: trading is supported on Binance and Coinbase only."
  pure ApiTradeResponse { atrSignal = sig, atrOrder = order }
  where
    noOrder msg =
      pure (ApiOrderResult False Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing msg)

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

parseCoinbaseError :: String -> (Maybe Int, Maybe String, String)
parseCoinbaseError raw =
  let raw' = truncateString 240 raw
      httpCode = extractHttpStatusCode raw'
      decoded =
        case extractJsonObject raw' of
          Nothing -> Nothing
          Just json ->
            case eitherDecode (BL.fromStrict (BS.pack json)) :: Either String Aeson.Value of
              Right (Aeson.Object o) ->
                case KM.lookup "message" o of
                  Just (Aeson.String t) -> Just (T.unpack t)
                  _ -> Nothing
              _ -> Nothing
      summary = maybe raw' id decoded
   in (httpCode, decoded, summary)

probeBinance :: String -> IO a -> IO ApiBinanceProbe
probeBinance step action = do
  r <- try action
  case r of
    Right _ -> pure (ApiBinanceProbe True False step Nothing Nothing "OK")
    Left ex -> do
      let msg =
            case (fromException ex :: Maybe IOError) of
              Just io | isUserError io -> ioeGetErrorString io
              _ -> show ex
          (code, m, summary) = parseBinanceError msg
          isMinNotional = code == Just (-4164) && "order/test" `isInfixOf` step
      pure $
        if isMinNotional
          then ApiBinanceProbe False True step Nothing Nothing "Trade test skipped: order notional below minNotional (increase order size)."
          else ApiBinanceProbe False False step code m summary

probeCoinbase :: String -> IO a -> IO ApiBinanceProbe
probeCoinbase step action = do
  r <- try action
  case r of
    Right _ -> pure (ApiBinanceProbe True False step Nothing Nothing "OK")
    Left ex -> do
      let msg =
            case (fromException ex :: Maybe IOError) of
              Just io | isUserError io -> ioeGetErrorString io
              _ -> show ex
          (code, m, summary) = parseCoinbaseError msg
      pure (ApiBinanceProbe False False step code m summary)

computeBinanceKeysStatusFromArgs :: Maybe OpsStore -> Args -> IO ApiBinanceKeysStatus
computeBinanceKeysStatusFromArgs mOps args = do
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
      env <- makeBinanceEnv mOps args
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
              then pure (Just (mkSkippedProbe "order/test" missingSizingMsg))
              else do
                mFilters <- fetchFilters env sym'
                case qty of
                  Nothing ->
                    case qq of
                      Nothing -> pure (Just (mkSkippedProbe "order/test" missingSizingMsg))
                      Just qq0 ->
                        case validateProbeQuote mFilters qq0 of
                          Left e -> pure (Just (mkSkippedProbe "order/test" e))
                          Right () ->
                            Just <$> probeBinance "order/test" (placeMarketOrder env OrderTest sym' Buy qty (Just qq0) Nothing (trim <$> argIdempotencyKey args) >> pure ())
                  Just qRaw -> do
                    mPrice <- fetchPriceIfNeeded env sym' mFilters
                    case normalizeProbeQty mFilters mPrice qRaw of
                      Left e -> pure (Just (mkSkippedProbe "order/test" e))
                      Right q -> Just <$> probeBinance "order/test" (placeMarketOrder env OrderTest sym' Buy (Just q) qq Nothing (trim <$> argIdempotencyKey args) >> pure ())
          MarketFutures -> do
            let qtyFromArgs =
                  case argOrderQuantity args of
                    Just q | q > 0 -> Just q
                    _ -> Nothing
            case qtyFromArgs of
              Just qRaw -> do
                mFilters <- fetchFilters env sym'
                mPrice <- fetchPriceIfNeeded env sym' mFilters
                case normalizeProbeQty mFilters mPrice qRaw of
                  Left e -> pure (Just (mkSkippedProbe "futures/order/test" e))
                  Right q -> Just <$> probeBinance "futures/order/test" (placeMarketOrder env OrderTest sym' Buy (Just q) Nothing Nothing (trim <$> argIdempotencyKey args) >> pure ())
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
                  Nothing -> pure (Just (mkSkippedProbe "futures/order/test" missingSizingMsg))
                  Just qq0 -> do
                    mFilters <- fetchFilters env sym'
                    case validateProbeQuote mFilters qq0 of
                      Left e -> pure (Just (mkSkippedProbe "futures/order/test" e))
                      Right () -> do
                        mPrice <- fetchPriceMaybe env sym'
                        case mPrice of
                          Nothing -> pure (Just (mkSkippedProbe "futures/order/test" "Price unavailable for quote sizing."))
                          Just price -> do
                            let qRaw = qq0 / price
                            case normalizeProbeQty mFilters (Just price) qRaw of
                              Left e -> pure (Just (mkSkippedProbe "futures/order/test" e))
                              Right q -> Just <$> probeBinance "futures/order/test" (placeMarketOrder env OrderTest sym' Buy (Just q) Nothing Nothing (trim <$> argIdempotencyKey args) >> pure ())

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
  where
    missingSizingMsg =
      if isJust (argOrderQuoteFraction args)
        then "Provide orderQuantity/orderQuote, or set orderQuoteFraction with sufficient quote balance."
        else "Provide orderQuantity or orderQuote."

    mkSkippedProbe step reason =
      ApiBinanceProbe False True step Nothing Nothing ("Trade test skipped: " ++ reason)

    fetchFilters env sym = do
      r <- try (fetchSymbolFilters env sym) :: IO (Either SomeException SymbolFilters)
      pure (either (const Nothing) Just r)

    fetchPriceMaybe env sym = do
      r <- try (fetchTickerPrice env sym) :: IO (Either SomeException Double)
      pure $
        case r of
          Right price | price > 0 -> Just price
          _ -> Nothing

    fetchPriceIfNeeded env sym mSf =
      case mSf >>= sfMinNotional of
        Nothing -> pure Nothing
        Just _ -> fetchPriceMaybe env sym

    effectiveStep sf = sfMarketStepSize sf <|> sfLotStepSize sf
    effectiveMinQty sf = sfMarketMinQty sf <|> sfLotMinQty sf
    effectiveMaxQty sf = sfMarketMaxQty sf <|> sfLotMaxQty sf

    validateProbeQuote mSf qq =
      if qq <= 0
        then Left "Quote is 0."
        else
          case mSf >>= sfMinNotional of
            Just mn | qq < mn -> Left ("Quote below minNotional (" ++ show mn ++ ").")
            _ -> Right ()

    normalizeProbeQty mSf mPrice qtyRaw =
      case mSf of
        Nothing ->
          if qtyRaw > 0
            then Right qtyRaw
            else Left "Quantity is 0."
        Just sf ->
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
                          case (mPrice, sfMinNotional sf) of
                            (Just price, Just mn) | price > 0 && qty1 * price < mn -> Left ("Notional below minNotional (" ++ show mn ++ ").")
                            _ -> Right qty1

computeCoinbaseKeysStatusFromArgs :: Args -> IO ApiCoinbaseKeysStatus
computeCoinbaseKeysStatusFromArgs args = do
  apiKeyRaw <- resolveEnv "COINBASE_API_KEY" (argCoinbaseApiKey args)
  apiSecretRaw <- resolveEnv "COINBASE_API_SECRET" (argCoinbaseApiSecret args)
  apiPassphraseRaw <- resolveEnv "COINBASE_API_PASSPHRASE" (argCoinbaseApiPassphrase args)

  let cleanKey v =
        case fmap trim v of
          Just "" -> Nothing
          other -> other
      apiKey = cleanKey apiKeyRaw
      apiSecret = cleanKey apiSecretRaw
      apiPassphrase = cleanKey apiPassphraseRaw
      hasApiKey = isJust apiKey
      hasApiSecret = isJust apiSecret
      hasApiPassphrase = isJust apiPassphrase
      baseStatus =
        ApiCoinbaseKeysStatus
          { ackHasApiKey = hasApiKey
          , ackHasApiSecret = hasApiSecret
          , ackHasApiPassphrase = hasApiPassphrase
          , ackSigned = Nothing
          }

  if not (hasApiKey && hasApiSecret && hasApiPassphrase)
    then pure baseStatus
    else do
      env <- newCoinbaseEnv (BS.pack <$> apiKey) (BS.pack <$> apiSecret) (BS.pack <$> apiPassphrase)
      signedProbe <- probeCoinbase "signed" (fetchCoinbaseAccounts env >> pure ())
      pure baseStatus { ackSigned = Just signedProbe }

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

lstmConfidenceScore :: LatestSignal -> Maybe Double
lstmConfidenceScore sig = do
  next <- lsLstmNext sig
  let cur = lsCurrentPrice sig
      thr = max 1e-12 (lsOpenThreshold sig)
      bad x = isNaN x || isInfinite x
  if cur <= 0 || bad cur || bad next
    then Nothing
    else
      let edge = abs (next / cur - 1)
          raw = edge / (2 * thr)
       in if bad edge || bad raw
            then Nothing
            else Just (max 0 (min 1 raw))

lstmConfidenceSizing :: Args -> LatestSignal -> (Double, Maybe String)
lstmConfidenceSizing args sig =
  if not (argConfidenceSizing args)
    then (1, Nothing)
    else
      case lstmConfidenceScore sig of
        Nothing -> (1, Nothing)
        Just score
          | score > 0.8 -> (1, Nothing)
          | score >= 0.6 -> (0.5, Nothing)
          | otherwise ->
              ( 0
              , Just (printf "No order: LSTM confidence %.1f%% (<60%%)." (score * 100))
              )

placeOrderForSignal :: Args -> String -> LatestSignal -> BinanceEnv -> IO ApiOrderResult
placeOrderForSignal args sym sig env =
  placeOrderForSignalEx args sym sig env Nothing True

placeOrderForSignalBot :: Args -> String -> LatestSignal -> BinanceEnv -> IO ApiOrderResult
placeOrderForSignalBot args sym sig env =
  placeOrderForSignalEx args sym sig env Nothing False

placeOrderForSignalEx :: Args -> String -> LatestSignal -> BinanceEnv -> Maybe String -> Bool -> IO ApiOrderResult
placeOrderForSignalEx args sym sig env mClientOrderIdOverride enableProtectionOrders = do
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
      let s0 = maybe 1 id (lsPositionSize sig)
          s1 = max 0 s0
          s2 = min s1 (max 0 (argMaxPositionSize args))
          (lstmScale, _) = lstmConfidenceSizing args sig
          s3 =
            case beMarket env of
              MarketFutures -> s2
              _ -> min 1 s2
       in max 0 (s3 * lstmScale)

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
        MethodBlend -> "No order: Blend neutral (within threshold)."

    shortErr :: SomeException -> String
    shortErr ex = take 240 (show ex)

    mode = if argBinanceLive args then OrderLive else OrderTest

    lstmBlockMsg :: Maybe String
    lstmBlockMsg = snd (lstmConfidenceSizing args sig)

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
            else
              case lstmBlockMsg of
                Just msg -> pure baseResult { aorMessage = msg }
                Nothing -> do
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
              let qRaw =
                    case qtyArg of
                      Just q -> min q baseBal
                      Nothing -> baseBal
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
      summary <- fetchFuturesPositionSummary env sym
      if fpsHasLong summary && fpsHasShort summary
        then
          pure $
            baseResult
              { aorMessage = "No order: both long and short futures positions are open; flatten one side first."
              }
        else do
          let posAmt = fpsNetAmt summary
              protectPrefix = "trader_prot_"
              volPerBar =
                case lsVolatility sig of
                  Just vol ->
                    let perBar = vol / sqrt (max 1e-12 (periodsPerYear args))
                     in if isNaN perBar || isInfinite perBar || perBar <= 0 then Nothing else Just perBar
                  _ -> Nothing
              stopFromVol mult =
                if mult <= 0
                  then Nothing
                  else
                    case volPerBar of
                      Just v ->
                        let frac = mult * v
                            fracClamped = min 0.999999 (max 0 frac)
                         in if isNaN fracClamped || isInfinite fracClamped || fracClamped <= 0
                              then Nothing
                              else Just fracClamped
                      _ -> Nothing
              stopLoss0 =
                case stopFromVol (argStopLossVolMult args) of
                  Just v -> Just v
                  Nothing ->
                    case argStopLoss args of
                      Just v | v > 0 -> Just v
                      _ -> Nothing
              takeProfit0 =
                case stopFromVol (argTakeProfitVolMult args) of
                  Just v -> Just v
                  Nothing ->
                    case argTakeProfit args of
                      Just v | v > 0 -> Just v
                      _ -> Nothing
              protectionManaged = enableProtectionOrders && mode == OrderLive
              protectionEnabled = protectionManaged && (isJust stopLoss0 || isJust takeProfit0)
              normalizeStopPrice px =
                case mSf >>= sfTickSize of
                  Nothing -> px
                  Just st -> quantizeDown st (max 0 px)

              cancelProtectionOrders :: IO ()
              cancelProtectionOrders =
                if not protectionManaged
                  then pure ()
                  else do
                    _ <- try (cancelFuturesOpenOrdersByClientPrefix env sym protectPrefix) :: IO (Either SomeException Int)
                    pure ()

              placeProtectionOrders :: Int -> Double -> IO (Either String ())
              placeProtectionOrders protectDir entryPrice
                | not protectionEnabled = pure (Right ())
                | otherwise = do
                    ts <- getTimestampMs
                    let mkCid suffix =
                          let raw = protectPrefix ++ show ts ++ "_" ++ suffix
                           in if length raw <= 36 then raw else take 36 raw

                        place1 :: OrderSide -> String -> Double -> String -> IO (Either String ())
                        place1 side orderType px suffix = do
                          let cid = mkCid suffix
                              px1 = normalizeStopPrice px
                          r <- try (placeFuturesTriggerMarketOrder env mode sym side orderType px1 (Just cid)) :: IO (Either SomeException BL.ByteString)
                          pure $
                            case r of
                              Left ex -> Left (take 240 (show ex))
                              Right _ -> Right ()

                        (slSide, mSlPx, tpSide, mTpPx) =
                          if protectDir < 0
                            then
                              ( Buy
                              , (\sl -> entryPrice * (1 + sl)) <$> stopLoss0
                              , Buy
                              , (\tp -> entryPrice * (1 - tp)) <$> takeProfit0
                              )
                            else
                              ( Sell
                              , (\sl -> entryPrice * (1 - sl)) <$> stopLoss0
                              , Sell
                              , (\tp -> entryPrice * (1 + tp)) <$> takeProfit0
                              )

                    rSl <-
                      case mSlPx of
                        Nothing -> pure (Right ())
                        Just px -> place1 slSide "STOP_MARKET" px "sl"
                    rTp <-
                      case mTpPx of
                        Nothing -> pure (Right ())
                        Just px -> place1 tpSide "TAKE_PROFIT_MARKET" px "tp"
                    pure $
                      case (rSl, rTp) of
                        (Left e, _) -> Left ("Protection order failed: " ++ e)
                        (_, Left e) -> Left ("Protection order failed: " ++ e)
                        _ -> Right ()

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
                    case (fmap (* entryScale) (argOrderQuote args) <|> mQuoteFromFraction) of
                      Just qq | qq > 0 && currentPrice > 0 -> Just (qq / currentPrice)
                      _ -> Nothing
    
              normalizeFuturesQty qRaw =
                case mSf of
                  Nothing -> Right qRaw
                  Just sf -> normalizeQty sf currentPrice qRaw
    
              closeOrder sideLabel side qtyRaw =
                case normalizeFuturesQty qtyRaw of
                  Left e -> pure baseResult { aorMessage = "No order: " ++ e }
                  Right q ->
                    if q <= 0
                      then pure baseResult { aorMessage = "No order: quantity is 0." }
                      else sendMarketOrder sideLabel side (Just q) Nothing (Just True)
    
              noFuturesSizingMsg =
                if maybe False (> 0) (argOrderQuoteFraction args)
                  then "No order: computed quote is 0 (check futures balance / orderQuoteFraction / maxOrderQuote)."
                  else "No order: futures requires orderQuantity or orderQuote."
    
          case dir of
                1 ->
                  if posAmt > 0
                    then
                      if protectionEnabled
                        then do
                          cancelProtectionOrders
                          r <- placeProtectionOrders 1 currentPrice
                          pure $
                            case r of
                              Left e -> baseResult { aorMessage = "No market order: already long. " ++ e }
                              Right () -> baseResult { aorMessage = "No market order: already long. Protection orders refreshed." }
                        else pure baseResult { aorMessage = "No order: already long." }
                    else
                      case lstmBlockMsg of
                        Just msg | posAmt == 0 -> pure baseResult { aorMessage = msg }
                        _ ->
                          case mDesiredQtyRaw of
                            Nothing -> pure baseResult { aorMessage = noFuturesSizingMsg }
                            Just q0 -> do
                              cancelProtectionOrders
                              let qtyToBuyRaw = if posAmt < 0 then abs posAmt + q0 else q0
                              case normalizeFuturesQty qtyToBuyRaw of
                                Left e -> pure baseResult { aorMessage = "No order: " ++ e }
                                Right q ->
                                  if q <= 0
                                    then pure baseResult { aorMessage = "No order: quantity is 0." }
                                    else do
                                      out <- sendMarketOrder "BUY" Buy (Just q) Nothing Nothing
                                      if aorSent out && protectionEnabled
                                        then do
                                          let fillPx =
                                                case (aorExecutedQty out, aorCummulativeQuoteQty out) of
                                                  (Just eq, Just qq) | eq > 0 && qq > 0 -> qq / eq
                                                  _ -> currentPrice
                                          r <- placeProtectionOrders 1 fillPx
                                          pure $
                                            case r of
                                              Left e -> out { aorMessage = aorMessage out ++ " " ++ e }
                                              Right () -> out { aorMessage = aorMessage out ++ " Protection orders placed." }
                                        else pure out
                (-1) ->
                  case argPositioning args of
                    LongShort ->
                      if posAmt < 0
                        then
                          if protectionEnabled
                            then do
                              cancelProtectionOrders
                              r <- placeProtectionOrders (-1) currentPrice
                              pure $
                                case r of
                                  Left e -> baseResult { aorMessage = "No market order: already short. " ++ e }
                                  Right () -> baseResult { aorMessage = "No market order: already short. Protection orders refreshed." }
                            else pure baseResult { aorMessage = "No order: already short." }
                        else
                          case lstmBlockMsg of
                            Just msg | posAmt == 0 -> pure baseResult { aorMessage = msg }
                            _ ->
                              case mDesiredQtyRaw of
                                Nothing -> pure baseResult { aorMessage = noFuturesSizingMsg }
                                Just q0 -> do
                                  cancelProtectionOrders
                                  let qtyToSellRaw = if posAmt > 0 then posAmt + q0 else q0
                                  case normalizeFuturesQty qtyToSellRaw of
                                    Left e -> pure baseResult { aorMessage = "No order: " ++ e }
                                    Right q ->
                                      if q <= 0
                                        then pure baseResult { aorMessage = "No order: quantity is 0." }
                                        else do
                                          out <- sendMarketOrder "SELL" Sell (Just q) Nothing Nothing
                                          if aorSent out && protectionEnabled
                                            then do
                                              let fillPx =
                                                    case (aorExecutedQty out, aorCummulativeQuoteQty out) of
                                                      (Just eq, Just qq) | eq > 0 && qq > 0 -> qq / eq
                                                      _ -> currentPrice
                                              r <- placeProtectionOrders (-1) fillPx
                                              pure $
                                                case r of
                                                  Left e -> out { aorMessage = aorMessage out ++ " " ++ e }
                                                  Right () -> out { aorMessage = aorMessage out ++ " Protection orders placed." }
                                            else pure out
                    LongFlat ->
                      if posAmt == 0
                        then do
                          cancelProtectionOrders
                          pure baseResult { aorMessage = "No order: already flat." }
                        else if posAmt > 0
                          then do
                            cancelProtectionOrders
                            closeOrder "SELL" Sell (abs posAmt)
                          else do
                            cancelProtectionOrders
                            closeOrder "BUY" Buy (abs posAmt)
                _ -> pure baseResult { aorMessage = neutralMsg }

    modeLabel m =
      case m of
        OrderLive -> "live"
        OrderTest -> "test"

placeCoinbaseOrderForSignal :: Args -> String -> LatestSignal -> CoinbaseEnv -> IO ApiOrderResult
placeCoinbaseOrderForSignal args symRaw sig env = do
  case (ceApiKey env, ceApiSecret env, ceApiPassphrase env) of
    (Nothing, _, _) -> noOrder "No order: missing Coinbase API key."
    (_, Nothing, _) -> noOrder "No order: missing Coinbase API secret."
    (_, _, Nothing) -> noOrder "No order: missing Coinbase API passphrase."
    _ ->
      if not (argBinanceLive args)
        then noOrder "No order: Coinbase does not support test orders; enable Live orders to send trades."
        else
          case splitCoinbaseSymbol sym of
            Left err -> noOrder ("No order: " ++ err)
            Right (baseAsset, quoteAsset) ->
              if argPositioning args == LongShort
                then noOrder "No order: Coinbase supports spot only (positioning=long-flat)."
                else do
                  baseBal <- fetchCoinbaseAvailableBalance env baseAsset
                  quoteBal <- fetchCoinbaseAvailableBalance env quoteAsset
                  case chosenDir of
                    Nothing -> noOrder neutralMsg
                    Just dir ->
                      case dir of
                        1 -> placeBuy baseBal quoteBal baseAsset quoteAsset
                        (-1) -> placeSell baseBal baseAsset
                        _ -> noOrder neutralMsg
  where
    sym = normalizeSymbol symRaw
    method = lsMethod sig
    chosenDir = lsChosenDir sig
    currentPrice = lsCurrentPrice sig

    entryScale :: Double
    entryScale =
      let s0 = maybe 1 id (lsPositionSize sig)
          s1 = max 0 s0
          s2 = min s1 (max 0 (argMaxPositionSize args))
          (lstmScale, _) = lstmConfidenceSizing args sig
          s3 = min 1 s2
       in max 0 (s3 * lstmScale)

    clientOrderId :: Maybe String
    clientOrderId = trim <$> argIdempotencyKey args

    baseResult :: ApiOrderResult
    baseResult =
      ApiOrderResult
        { aorSent = False
        , aorMode = Just (if argBinanceLive args then "live" else "paper")
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
        MethodBlend -> "No order: Blend neutral (within threshold)."

    lstmBlockMsg :: Maybe String
    lstmBlockMsg = snd (lstmConfidenceSizing args sig)

    splitCoinbaseSymbol s =
      case break (== '-') s of
        (base, '-':quote) | not (null base) && not (null quote) ->
          Right (map toUpper base, map toUpper quote)
        _ -> Left "Coinbase symbols must be BASE-QUOTE (e.g., BTC-USD)."

    sendOrder :: String -> Maybe Double -> Maybe Double -> IO ApiOrderResult
    sendOrder sideLabel mQty mFunds = do
      let baseOut =
            baseResult
              { aorSide = Just sideLabel
              , aorQuantity = mQty
              , aorQuoteQuantity = mFunds
              }
      r <- try (placeCoinbaseMarketOrder env sym sideLabel mQty mFunds clientOrderId) :: IO (Either SomeException BL.ByteString)
      case r of
        Left ex -> pure baseOut { aorMessage = "Order failed: " ++ take 240 (show ex) }
        Right body ->
          pure
            baseOut
              { aorSent = True
              , aorResponse = Just (shortResp body)
              , aorMessage = "Order sent."
              }

    placeBuy baseBal quoteBal _baseAsset quoteAsset =
      if baseBal > 0
        then noOrder "No order: already long."
        else
          case lstmBlockMsg of
            Just msg -> noOrder msg
            Nothing -> do
              let qtyArg =
                    case argOrderQuantity args of
                      Just q | q > 0 -> Just (q * entryScale)
                      _ -> Nothing
                  quoteArg =
                    case argOrderQuote args of
                      Just q | q > 0 -> Just (q * entryScale)
                      _ -> Nothing
                  quoteFracArg =
                    case argOrderQuoteFraction args of
                      Just f | f > 0 -> Just (f * entryScale)
                      _ -> Nothing
                  maxOrderQuoteArg =
                    case argMaxOrderQuote args of
                      Just q | q > 0 -> Just q
                      _ -> Nothing
                  quoteFromFraction =
                    case (qtyArg, quoteArg, quoteFracArg) of
                      (Nothing, Nothing, Just f) ->
                        let q0 = quoteBal * f
                            q1 = maybe q0 (\capQ -> min capQ q0) maxOrderQuoteArg
                         in Just q1
                      _ -> Nothing
                  quoteEffective = quoteArg <|> quoteFromFraction

              case (qtyArg, quoteEffective) of
                (Nothing, Nothing) ->
                  case quoteFracArg of
                    Nothing -> noOrder "No order: provide orderQuantity or orderQuote."
                    Just _ -> noOrder "No order: computed quote is 0 (check quote balance / orderQuoteFraction / maxOrderQuote)."
                (Just qRaw, _) ->
                  if currentPrice > 0 && qRaw * currentPrice > quoteBal
                    then noOrder ("No order: insufficient " ++ quoteAsset ++ " balance.")
                    else sendOrder "BUY" (Just qRaw) Nothing
                (Nothing, Just qq0) ->
                  let qq = max 0 qq0
                   in if qq <= 0
                        then noOrder "No order: quote is 0."
                        else
                          if qq > quoteBal
                            then noOrder ("No order: insufficient " ++ quoteAsset ++ " balance.")
                            else sendOrder "BUY" Nothing (Just qq)

    placeSell baseBal _baseAsset =
      if baseBal <= 0
        then noOrder "No order: already flat."
        else do
          let qtyArg =
                case argOrderQuantity args of
                  Just q | q > 0 -> Just q
                  _ -> Nothing
              qRaw =
                case qtyArg of
                  Just q -> min q baseBal
                  Nothing -> baseBal
          if qRaw <= 0
            then noOrder "No order: quantity is 0."
            else sendOrder "SELL" (Just qRaw) Nothing

computeBacktestFromArgs :: Maybe OpsStore -> Args -> IO Aeson.Value
computeBacktestFromArgs mOps args = do
  (series, mBinanceEnv) <- loadPrices mOps args
  let prices = psClose series
  ensureMinPriceRows args 2 prices
  let lookback = argLookback args
  summary <- computeBacktestSummary args lookback series mBinanceEnv
  pure (backtestSummaryJson summary)

backtestSummaryJson :: BacktestSummary -> Aeson.Value
backtestSummaryJson summary =
  let tuneStatsJson =
        case bsTuneStats summary of
          Nothing -> Nothing
          Just st ->
            Just
              ( object
                  [ "folds" .= tsFoldCount st
                  , "scores" .= tsFoldScores st
                  , "meanScore" .= tsMeanScore st
                  , "stdScore" .= tsStdScore st
                  ]
              )
      tuneMetricsJson =
        case bsTuneMetrics summary of
          Nothing -> Nothing
          Just m -> Just (metricsToJson m)
      tuningJson =
        object
          [ "objective" .= tuneObjectiveCode (bsTuneObjective summary)
          , "penaltyMaxDrawdown" .= bsTunePenaltyMaxDrawdown summary
          , "penaltyTurnover" .= bsTunePenaltyTurnover summary
          , "stressVolMult" .= bsTuneStressVolMult summary
          , "stressShock" .= bsTuneStressShock summary
          , "stressWeight" .= bsTuneStressWeight summary
          , "minRoundTrips" .= bsMinRoundTrips summary
          , "walkForwardFolds" .= bsWalkForwardFolds summary
          , "tuneStats" .= tuneStatsJson
          , "tuneMetrics" .= tuneMetricsJson
          ]
      costsJson =
        object
          [ "fee" .= bsFee summary
          , "slippage" .= bsSlippage summary
          , "spread" .= bsSpread summary
          , "perSideCost" .= bsEstimatedPerSideCost summary
          , "roundTripCost" .= bsEstimatedRoundTripCost summary
          , "breakEvenThreshold" .= breakEvenThresholdFromPerSideCost (bsEstimatedPerSideCost summary)
          ]
      walkForwardJson =
        case bsWalkForward summary of
          Nothing -> Nothing
          Just wf ->
            let foldJson f =
                  object
                    [ "startIndex" .= wffStartIndex f
                    , "endIndex" .= wffEndIndex f
                    , "metrics" .= metricsToJson (wffMetrics f)
                    ]
                s = wfrSummary wf
                summaryJson =
                  object
                    [ "finalEquityMean" .= wfsFinalEquityMean s
                    , "finalEquityStd" .= wfsFinalEquityStd s
                    , "annualizedReturnMean" .= wfsAnnualizedReturnMean s
                    , "annualizedReturnStd" .= wfsAnnualizedReturnStd s
                    , "sharpeMean" .= wfsSharpeMean s
                    , "sharpeStd" .= wfsSharpeStd s
                    , "maxDrawdownMean" .= wfsMaxDrawdownMean s
                    , "maxDrawdownStd" .= wfsMaxDrawdownStd s
                    , "turnoverMean" .= wfsTurnoverMean s
                    , "turnoverStd" .= wfsTurnoverStd s
                    ]
             in Just (object ["foldCount" .= wfrFoldCount wf, "folds" .= map foldJson (wfrFolds wf), "summary" .= summaryJson])
   in
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
    , "threshold" .= bsBestOpenThreshold summary
    , "openThreshold" .= bsBestOpenThreshold summary
    , "closeThreshold" .= bsBestCloseThreshold summary
    , "minHoldBars" .= bsMinHoldBars summary
    , "cooldownBars" .= bsCooldownBars summary
    , "maxHoldBars" .= bsMaxHoldBars summary
    , "stopLossVolMult" .= bsStopLossVolMult summary
    , "takeProfitVolMult" .= bsTakeProfitVolMult summary
    , "trailingStopVolMult" .= bsTrailingStopVolMult summary
    , "maxPositionSize" .= bsMaxPositionSize summary
    , "minEdge" .= bsMinEdge summary
    , "minSignalToNoise" .= bsMinSignalToNoise summary
    , "costAwareEdge" .= bsCostAwareEdge summary
    , "edgeBuffer" .= bsEdgeBuffer summary
    , "trendLookback" .= bsTrendLookback summary
    , "volTarget" .= bsVolTarget summary
    , "volLookback" .= bsVolLookback summary
    , "volEwmaAlpha" .= bsVolEwmaAlpha summary
    , "volFloor" .= bsVolFloor summary
    , "volScaleMax" .= bsVolScaleMax summary
    , "maxVolatility" .= bsMaxVolatility summary
    , "rebalanceBars" .= bsRebalanceBars summary
    , "rebalanceThreshold" .= bsRebalanceThreshold summary
    , "rebalanceGlobal" .= bsRebalanceGlobal summary
    , "fundingRate" .= bsFundingRate summary
    , "fundingBySide" .= bsFundingBySide summary
    , "rebalanceResetOnSignal" .= bsRebalanceResetOnSignal summary
    , "fundingOnOpen" .= bsFundingOnOpen summary
    , "blendWeight" .= bsBlendWeight summary
    , "triLayer" .= bsTriLayer summary
    , "triLayerFastMult" .= bsTriLayerFastMult summary
    , "triLayerSlowMult" .= bsTriLayerSlowMult summary
    , "triLayerCloudPadding" .= bsTriLayerCloudPadding summary
    , "triLayerPriceAction" .= bsTriLayerPriceAction summary
    , "lstmExitFlipBars" .= bsLstmExitFlipBars summary
    , "tuning" .= tuningJson
    , "costs" .= costsJson
    , "walkForward" .= walkForwardJson
    , "metrics" .= metricsToJson (bsMetrics summary)
    , "baselines" .= map baselineToJson (bsBaselines summary)
    , "latestSignal" .= bsLatestSignal summary
    , "equityCurve" .= bsEquityCurve summary
    , "prices" .= bsBacktestPrices summary
    , "kalmanPredNext" .= bsKalmanPredNext summary
    , "lstmPredNext" .= bsLstmPredNext summary
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

baselineToJson :: Baseline -> Aeson.Value
baselineToJson b =
  object
    [ "name" .= blName b
    , "metrics" .= metricsToJson (blMetrics b)
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
    , "positionChanges" .= bmPositionChanges m
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

runTradeOnly :: Maybe Webhook -> Args -> Int -> [Double] -> Maybe BinanceEnv -> IO ()
runTradeOnly mWebhook args lookback prices mBinanceEnv = do
  signal <- computeTradeOnlySignal args lookback prices mBinanceEnv
  if argJson args
    then
      if argBinanceTrade args
        then do
          (sym, env) <-
            case (argBinanceSymbol args, mBinanceEnv) of
              (Just s, Just e) -> pure (s, e)
              _ -> error "Internal: --binance-trade requires binanceSymbol data source."
          order <- placeOrderForSignal args sym signal env
          webhookNotifyMaybeSync mWebhook (webhookEventTradeOrder args signal order)
          printJsonStdout (object ["mode" .= ("trade" :: String), "trade" .= ApiTradeResponse signal order])
        else printJsonStdout (object ["mode" .= ("signal" :: String), "signal" .= signal])
    else do
      printLatestSignalSummary signal
      let perSide = estimatedPerSideCost (argFee args) (argSlippage args) (argSpread args)
          roundTrip = estimatedRoundTripCost (argFee args) (argSlippage args) (argSpread args)
          minEdgeBase = max 0 (argMinEdge args)
          minEdge =
            if argCostAwareEdge args
              then max minEdgeBase (breakEvenThresholdFromPerSideCost perSide + max 0 (argEdgeBuffer args))
              else minEdgeBase
          openThr = max (argOpenThreshold args) minEdge
      printCostGuidance openThr (argCloseThreshold args) perSide roundTrip
      maybeSendOrder mWebhook args mBinanceEnv signal

runBacktestPipeline :: Maybe Webhook -> Args -> Int -> PriceSeries -> Maybe BinanceEnv -> IO ()
runBacktestPipeline mWebhook args lookback series mBinanceEnv = do
  let prices = psClose series
  summary <- computeBacktestSummary args lookback series mBinanceEnv
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
          webhookNotifyMaybeSync mWebhook (webhookEventTradeOrder args (bsLatestSignal summary) order)
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
                "Optimized on tune split (objective=%s): method=%s open=%.6f (%.3f%%) close=%.6f (%.3f%%)"
                (tuneObjectiveCode (bsTuneObjective summary))
                (methodCode (bsMethodUsed summary))
                (bsBestOpenThreshold summary)
                (bsBestOpenThreshold summary * 100)
                (bsBestCloseThreshold summary)
                (bsBestCloseThreshold summary * 100)
            )
        else if argSweepThreshold args
          then
            putStrLn
              ( printf
                  "Best thresholds on tune split (objective=%s): open=%.6f (%.3f%%) close=%.6f (%.3f%%)"
                  (tuneObjectiveCode (bsTuneObjective summary))
                  (bsBestOpenThreshold summary)
                  (bsBestOpenThreshold summary * 100)
                  (bsBestCloseThreshold summary)
                  (bsBestCloseThreshold summary * 100)
              )
          else pure ()

      case bsTuneStats summary of
        Nothing -> pure ()
        Just st ->
          if bsTuneSize summary > 0
            then
              putStrLn
                ( printf
                    "Tune objective stats: folds=%d mean=%.6f std=%.6f (ddPenalty=%.3f, turnoverPenalty=%.3f)"
                    (tsFoldCount st)
                    (tsMeanScore st)
                    (tsStdScore st)
                    (bsTunePenaltyMaxDrawdown summary)
                    (bsTunePenaltyTurnover summary)
                )
            else pure ()

      printCostGuidance
        (max (bsBestOpenThreshold summary) (bsMinEdge summary))
        (bsBestCloseThreshold summary)
        (bsEstimatedPerSideCost summary)
        (bsEstimatedRoundTripCost summary)
      printFundingGuidance (argFundingRate args) (argFundingBySide args)

      putStrLn $
        case bsMethodUsed summary of
          MethodBoth -> "Backtest (Kalman fusion + LSTM direction-agreement gated) complete."
          MethodKalmanOnly -> "Backtest (Kalman fusion only) complete."
          MethodLstmOnly -> "Backtest (LSTM only) complete."
          MethodBlend -> "Backtest (Kalman + LSTM blend) complete."

      case bsLstmHistory summary of
        Nothing -> pure ()
        Just history -> printLstmSummary history

      printMetrics (bsMethodUsed summary) (bsMetrics summary)

      case bsWalkForward summary of
        Nothing -> pure ()
        Just wf -> do
          let s = wfrSummary wf
          putStrLn ""
          putStrLn "**Walk-forward backtest**"
          putStrLn
            ( printf
                "Folds=%d finalEq=%.4fx±%.4fx sharpe=%.3f±%.3f maxDD=%.2f%%±%.2f%% turnover=%.4f±%.4f"
                (wfrFoldCount wf)
                (wfsFinalEquityMean s)
                (wfsFinalEquityStd s)
                (wfsSharpeMean s)
                (wfsSharpeStd s)
                (wfsMaxDrawdownMean s * 100)
                (wfsMaxDrawdownStd s * 100)
                (wfsTurnoverMean s)
                (wfsTurnoverStd s)
            )

      printLatestSignalSummary (bsLatestSignal summary)
      maybeSendOrder mWebhook args mBinanceEnv (bsLatestSignal summary)

printJsonStdout :: ToJSON a => a -> IO ()
printJsonStdout v = BS.putStrLn (BL.toStrict (encode v))

computeTradeOnlySignal :: Args -> Int -> [Double] -> Maybe BinanceEnv -> IO LatestSignal
computeTradeOnlySignal args lookback prices mBinanceEnv = do
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

  mMarketModel <-
    case (method, mBinanceEnv, argBinanceSymbol args) of
      (MethodLstmOnly, _, _) -> pure Nothing
      (_, Just env, Just sym)
        | platformSupportsMarketContext (argPlatform args) -> do
            r <- try (buildMarketModel args env sym n pricesV) :: IO (Either SomeException (Maybe MarketModel))
            pure (either (const Nothing) id r)
      _ -> pure Nothing

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
        (lstmModel, _) <- trainLstmWithPersistence args lookback lstmCfg obsAll
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
                  meas = mapMaybe (toMeasurement args sv) sensorOuts ++ maybeToList (mMarketModel >>= (`marketMeasurementAt` t))
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

  pure (computeLatestSignal args lookback pricesV mLstmCtx mKalmanCtx mMarketModel)

-- LSTM weight persistence (for incremental training across backtests)

data PersistedLstmModel = PersistedLstmModel
  { plmVersion :: !Int
  , plmHiddenSize :: !Int
  , plmTrainBars :: !Int
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
    Just dir | null dir -> pure Nothing
    Just dir -> pure (Just dir)
    Nothing -> do
      mStateDir <- stateSubdirFromEnv "lstm"
      pure (Just (fromMaybe defaultLstmWeightsDir mStateDir))

hashKeyHex :: String -> String
hashKeyHex s =
  let digest :: Digest SHA256
      digest = hash (TE.encodeUtf8 (T.pack s))
   in BS.unpack (B16.encode (convert digest))

hashBytesHex :: BL.ByteString -> String
hashBytesHex bs =
  let digest :: Digest SHA256
      digest = hash (BL.toStrict bs)
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

loadPersistedLstmModel :: FilePath -> Int -> Int -> IO (Maybe LSTMModel)
loadPersistedLstmModel path hidden trainBars = do
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
                    plmVersion plm == 2
                      && plmHiddenSize plm == hidden
                      && plmTrainBars plm <= max 0 trainBars
                      && length (plmParams plm) == paramCount hidden
               in if ok
                    then pure (Just (LSTMModel (plmHiddenSize plm) (plmParams plm)))
                    else pure Nothing

savePersistedLstmModelMaybe :: Maybe FilePath -> Int -> LSTMModel -> IO ()
savePersistedLstmModelMaybe mPath trainBars model =
  case mPath of
    Nothing -> pure ()
    Just path -> do
      _ <-
        try
          ( do
              createDirectoryIfMissing True (takeDirectory path)
              let plm =
                    PersistedLstmModel
                      { plmVersion = 2
                      , plmHiddenSize = lmHiddenSize model
                      , plmTrainBars = max 0 trainBars
                      , plmParams = lmParams model
                      }
              randId <- randomIO :: IO Word64
              let tmpPath = path ++ ".tmp-" ++ show randId
              BL.writeFile tmpPath (encode plm)
              -- Atomic on POSIX when within the same filesystem; on Windows, fall back to replace.
              r1 <- try (renameFile tmpPath path) :: IO (Either SomeException ())
              case r1 of
                Right _ -> pure ()
                Left _ -> do
                  _ <- try (removeFile path) :: IO (Either SomeException ())
                  _ <- try (renameFile tmpPath path) :: IO (Either SomeException ())
                  pure ()
          )
          :: IO (Either SomeException ())
      pure ()

trainLstmWithPersistence :: Args -> Int -> LSTMConfig -> [Double] -> IO (LSTMModel, [EpochStats])
trainLstmWithPersistence args lookback cfg series = do
  let trainBars = length series
  mPath <- lstmWeightsPath args lookback
  mSeed <-
    case mPath of
      Nothing -> pure Nothing
      Just path -> loadPersistedLstmModel path (lcHiddenSize cfg) trainBars
  let (model, hist) =
        case mSeed of
          Just seedModel -> fineTuneLSTM cfg seedModel series
          Nothing -> trainLSTM cfg series
  savePersistedLstmModelMaybe mPath trainBars model
  pure (model, hist)

computeBacktestSummary :: Args -> Int -> PriceSeries -> Maybe BinanceEnv -> IO BacktestSummary
computeBacktestSummary args lookback series mBinanceEnv = do
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
      openTimesAll =
        case psOpenTimes series of
          Just ts | length ts == n -> Just ts
          _ -> Nothing

      predStart = if tuningEnabled then fitSize else trainEnd
      stepCount = n - predStart - 1
      fitPrices = take predStart prices

      tunePrices = drop fitSize trainPrices
      tuneHighs = take tuneSize (drop fitSize highsAll)
      tuneLows = take tuneSize (drop fitSize lowsAll)
      tuneOpenTimes = fmap (take tuneSize . drop fitSize) openTimesAll

      backtestHighs = drop trainEnd highsAll
      backtestLows = drop trainEnd lowsAll
      backtestOpenTimes = fmap (drop trainEnd) openTimesAll

      methodRequested = argMethod args
      methodForComputation =
        if argOptimizeOperations args
          then MethodBoth
          else
            case methodRequested of
              MethodBlend -> MethodBoth
              _ -> methodRequested
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

  mMarketModel <-
    case (methodForComputation, mBinanceEnv, argBinanceSymbol args) of
      (MethodLstmOnly, _, _) -> pure Nothing
      (_, Just env, Just sym) -> do
        r <- try (buildMarketModel args env sym predStart pricesV) :: IO (Either SomeException (Maybe MarketModel))
        pure (either (const Nothing) id r)
      _ -> pure Nothing

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
                (backtestStepKalmanOnly args pricesV predictors predStart mMarketModel)
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
                (backtestStep args lookback normState obsAll pricesV lstmModel predictors predStart mMarketModel)
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
      MethodBlend -> do
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
                (backtestStep args lookback normState obsAll pricesV lstmModel predictors predStart mMarketModel)
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

  let feeUsed = max 0 (argFee args)
      slippageUsed = max 0 (argSlippage args)
      spreadUsed = max 0 (argSpread args)
      perSideCost = estimatedPerSideCost feeUsed slippageUsed spreadUsed
      roundTripCost = estimatedRoundTripCost feeUsed slippageUsed spreadUsed
      minEdgeBase = max 0 (argMinEdge args)
      minEdge =
        if argCostAwareEdge args
          then max minEdgeBase (breakEvenThresholdFromPerSideCost perSideCost + max 0 (argEdgeBuffer args))
          else minEdgeBase

      baseCfg =
        EnsembleConfig
          { ecOpenThreshold = argOpenThreshold args
          , ecCloseThreshold = argCloseThreshold args
          , ecFee = feeUsed
          , ecSlippage = slippageUsed
          , ecSpread = spreadUsed
          , ecStopLoss = argStopLoss args
          , ecTakeProfit = argTakeProfit args
          , ecTrailingStop = argTrailingStop args
          , ecStopLossVolMult = argStopLossVolMult args
          , ecTakeProfitVolMult = argTakeProfitVolMult args
          , ecTrailingStopVolMult = argTrailingStopVolMult args
          , ecMinHoldBars = argMinHoldBars args
          , ecCooldownBars = argCooldownBars args
          , ecMaxHoldBars = argMaxHoldBars args
          , ecMaxDrawdown = argMaxDrawdown args
          , ecMaxDailyLoss = argMaxDailyLoss args
          , ecIntervalSeconds = parseIntervalSeconds (argInterval args)
          , ecOpenTimes = Nothing
          , ecPositioning = argPositioning args
          , ecIntrabarFill = argIntrabarFill args
          , ecMaxPositionSize = argMaxPositionSize args
          , ecMinEdge = minEdge
          , ecMinSignalToNoise = argMinSignalToNoise args
          , ecTrendLookback = argTrendLookback args
          , ecPeriodsPerYear = periodsPerYear args
          , ecVolTarget = argVolTarget args
          , ecVolLookback = argVolLookback args
          , ecVolEwmaAlpha = argVolEwmaAlpha args
          , ecVolFloor = argVolFloor args
          , ecVolScaleMax = argVolScaleMax args
          , ecMaxVolatility = argMaxVolatility args
          , ecRebalanceBars = argRebalanceBars args
          , ecRebalanceThreshold = argRebalanceThreshold args
          , ecRebalanceGlobal = argRebalanceGlobal args
          , ecRebalanceResetOnSignal = argRebalanceResetOnSignal args
          , ecFundingRate = argFundingRate args
          , ecFundingBySide = argFundingBySide args
          , ecFundingOnOpen = argFundingOnOpen args
          , ecBlendWeight = argBlendWeight args
          , ecKalmanDt = argKalmanDt args
          , ecKalmanProcessVar = argKalmanProcessVar args
          , ecKalmanMeasurementVar = argKalmanMeasurementVar args
          , ecTriLayer = argTriLayer args
          , ecTriLayerFastMult = argTriLayerFastMult args
          , ecTriLayerSlowMult = argTriLayerSlowMult args
          , ecTriLayerCloudPadding = argTriLayerCloudPadding args
          , ecTriLayerRequirePriceAction = argTriLayerRequirePriceAction args
          , ecLstmExitFlipBars = argLstmExitFlipBars args
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

      baseCfgTune = baseCfg { ecOpenTimes = V.fromList <$> tuneOpenTimes }
      baseCfgBacktest = baseCfg { ecOpenTimes = V.fromList <$> backtestOpenTimes }

      offsetBacktestPred = max 0 (trainEnd - predStart)
      kalPredBacktest = drop offsetBacktestPred kalPredAll
      lstmPredBacktest = drop offsetBacktestPred lstmPredAll
      kalPredTune = take (max 0 (tuneSize - 1)) kalPredAll
      lstmPredTune = take (max 0 (tuneSize - 1)) lstmPredAll
      metaBacktest = fmap (drop offsetBacktestPred) mMetaAll
      metaTune = fmap (take (max 0 (tuneSize - 1))) mMetaAll

      ppy = periodsPerYear args
      tuneCfg =
        TuneConfig
          { tcObjective = argTuneObjective args
          , tcPenaltyMaxDrawdown = argTunePenaltyMaxDrawdown args
          , tcPenaltyTurnover = argTunePenaltyTurnover args
          , tcPeriodsPerYear = ppy
          , tcWalkForwardFolds = argWalkForwardFolds args
          , tcMinRoundTrips = argMinRoundTrips args
          , tcStressVolMultiplier = argTuneStressVolMult args
          , tcStressShock = argTuneStressShock args
          , tcStressWeight = argTuneStressWeight args
          }

      (methodUsed, bestOpenThr, bestCloseThr, mTuneStats, mTuneMetrics) =
        if argOptimizeOperations args
          then
            case optimizeOperationsWithHLWith tuneCfg baseCfgTune tunePrices tuneHighs tuneLows kalPredTune lstmPredTune metaTune of
              Left e -> error ("optimizeOperations: " ++ e)
              Right (m, openThr, closeThr, btTune, stats) -> (m, openThr, closeThr, Just stats, Just (computeMetrics ppy btTune))
          else if argSweepThreshold args
            then
              case sweepThresholdWithHLWith tuneCfg methodRequested baseCfgTune tunePrices tuneHighs tuneLows kalPredTune lstmPredTune metaTune of
                Left e -> error ("sweepThreshold: " ++ e)
                Right (openThr, closeThr, btTune, stats) -> (methodRequested, openThr, closeThr, Just stats, Just (computeMetrics ppy btTune))
            else (methodRequested, argOpenThreshold args, argCloseThreshold args, Nothing, Nothing)

      backtestCfg = baseCfgBacktest { ecOpenThreshold = bestOpenThr, ecCloseThreshold = bestCloseThr }
      (kalPredUsedBacktest, lstmPredUsedBacktest) =
        selectPredictions methodUsed (argBlendWeight args) kalPredBacktest lstmPredBacktest
      metaUsedBacktest =
        case methodUsed of
          MethodLstmOnly -> Nothing
          _ -> metaBacktest

  let backtest =
        simulateEnsembleWithHL
          backtestCfg
          1
          backtestPrices
          backtestHighs
          backtestLows
          kalPredUsedBacktest
          lstmPredUsedBacktest
          metaUsedBacktest

  let metrics = computeMetrics ppy backtest
      baselines = computeBaselines ppy perSideCost backtestPrices

      walkForward =
        let wfReq = max 1 (argWalkForwardFolds args)
            stepsAll = max 0 (length backtestPrices - 1)

            foldRangesSteps :: Int -> Int -> [(Int, Int)]
            foldRangesSteps steps0 k0 =
              let steps = max 0 steps0
                  k = max 1 (min steps (max 1 k0))
                  base = if k <= 0 then 0 else steps `div` k
                  extra = if k <= 0 then 0 else steps `mod` k
                  go i start =
                    if i >= k
                      then []
                      else
                        let len = base + if i < extra then 1 else 0
                            end = start + len - 1
                         in if len <= 0 then [] else (start, end) : go (i + 1) (end + 1)
               in go 0 0

            bad x = isNaN x || isInfinite x
            meanD xs =
              let ys = filter (not . bad) xs
               in if null ys then 0 else sum ys / fromIntegral (length ys)
            stdD xs =
              let ys = filter (not . bad) xs
               in case ys of
                    [] -> 0
                    [_] -> 0
                    _ ->
                      let m = meanD ys
                          var = sum (map (\x -> (x - m) ** 2) ys) / fromIntegral (length ys - 1)
                       in sqrt var

            mkFold (t0, t1) =
              let steps = t1 - t0 + 1
                  pricesF = take (steps + 1) (drop t0 backtestPrices)
                  highsF = take (steps + 1) (drop t0 backtestHighs)
                  lowsF = take (steps + 1) (drop t0 backtestLows)
                  kalF = take steps (drop t0 kalPredUsedBacktest)
                  lstmF = take steps (drop t0 lstmPredUsedBacktest)
                  metaF = fmap (take steps . drop t0) metaUsedBacktest
                  openTimesF = fmap (\ot -> V.slice t0 (steps + 1) ot) (ecOpenTimes backtestCfg)
                  backtestCfgFold = backtestCfg { ecOpenTimes = openTimesF }
                  btFold = simulateEnsembleWithHL backtestCfgFold 1 pricesF highsF lowsF kalF lstmF metaF
                  mFold = computeMetrics ppy btFold
               in WalkForwardFold { wffStartIndex = trainEnd + t0, wffEndIndex = trainEnd + t1 + 1, wffMetrics = mFold }

            folds =
              [ mkFold r
              | r@(t0, t1) <- foldRangesSteps stepsAll wfReq
              , t1 >= t0
              ]
         in
          if wfReq <= 1 || stepsAll <= 1
            then Nothing
            else
              let ms = map wffMetrics folds
                  finalEqs = map bmFinalEquity ms
                  annRets = map bmAnnualizedReturn ms
                  sharpes = map bmSharpe ms
                  maxDds = map bmMaxDrawdown ms
                  turns = map bmTurnover ms
                  summary =
                    WalkForwardSummary
                      { wfsFinalEquityMean = meanD finalEqs
                      , wfsFinalEquityStd = stdD finalEqs
                      , wfsAnnualizedReturnMean = meanD annRets
                      , wfsAnnualizedReturnStd = stdD annRets
                      , wfsSharpeMean = meanD sharpes
                      , wfsSharpeStd = stdD sharpes
                      , wfsMaxDrawdownMean = meanD maxDds
                      , wfsMaxDrawdownStd = stdD maxDds
                      , wfsTurnoverMean = meanD turns
                      , wfsTurnoverStd = stdD turns
                      }
               in Just (WalkForwardReport { wfrFoldCount = length folds, wfrFolds = folds, wfrSummary = summary })

      argsForSignal =
        if argOptimizeOperations args
          then args { argMethod = methodUsed, argOpenThreshold = bestOpenThr, argCloseThreshold = bestCloseThr }
          else if argSweepThreshold args
            then args { argOpenThreshold = bestOpenThr, argCloseThreshold = bestCloseThr }
            else args

      latestSignal = computeLatestSignal argsForSignal lookback pricesV mLstmCtx mKalmanCtx mMarketModel
      finiteMaybe x =
        if isNaN x || isInfinite x
          then Nothing
          else Just x
      alignPred mCtx preds =
        let n0 = length backtestPrices
            -- `preds` are next-bar predictions for each bar `t` (0..n-2).
            -- Align to the bar index so `predNext[t]` lines up with decisions/positions for t→t+1.
            aligned = map finiteMaybe preds ++ [Nothing]
         in if n0 <= 0
              then []
              else
                if isJust mCtx
                  then take n0 (aligned ++ repeat Nothing)
                  else replicate n0 Nothing
      kalmanPredNext = alignPred mKalmanCtx kalPredBacktest
      lstmPredNext = alignPred mLstmCtx lstmPredBacktest

  pure
    BacktestSummary
      { bsTrainEndRaw = trainEndRaw
      , bsTrainEnd = trainEnd
      , bsTrainSize = trainSize
      , bsFitSize = fitSize
      , bsTuneSize = tuneSize
      , bsTuneRatio = tuneRatioUsed
      , bsTuneObjective = argTuneObjective args
      , bsTunePenaltyMaxDrawdown = argTunePenaltyMaxDrawdown args
      , bsTunePenaltyTurnover = argTunePenaltyTurnover args
      , bsTuneStressVolMult = argTuneStressVolMult args
      , bsTuneStressShock = argTuneStressShock args
      , bsTuneStressWeight = argTuneStressWeight args
      , bsMinRoundTrips = argMinRoundTrips args
      , bsWalkForwardFolds = argWalkForwardFolds args
      , bsTuneStats = mTuneStats
      , bsTuneMetrics = mTuneMetrics
      , bsBacktestSize = length backtestPrices
      , bsBacktestRatio = backtestRatio
      , bsMethodUsed = methodUsed
      , bsBestOpenThreshold = bestOpenThr
      , bsBestCloseThreshold = bestCloseThr
      , bsMinHoldBars = argMinHoldBars args
      , bsCooldownBars = argCooldownBars args
      , bsMaxHoldBars = argMaxHoldBars args
      , bsStopLossVolMult = argStopLossVolMult args
      , bsTakeProfitVolMult = argTakeProfitVolMult args
      , bsTrailingStopVolMult = argTrailingStopVolMult args
      , bsMaxPositionSize = argMaxPositionSize args
      , bsMinEdge = minEdge
      , bsMinSignalToNoise = argMinSignalToNoise args
      , bsCostAwareEdge = argCostAwareEdge args
      , bsEdgeBuffer = argEdgeBuffer args
      , bsTrendLookback = argTrendLookback args
      , bsVolTarget = argVolTarget args
      , bsVolLookback = argVolLookback args
      , bsVolEwmaAlpha = argVolEwmaAlpha args
      , bsVolFloor = argVolFloor args
      , bsVolScaleMax = argVolScaleMax args
      , bsMaxVolatility = argMaxVolatility args
      , bsRebalanceBars = argRebalanceBars args
      , bsRebalanceThreshold = argRebalanceThreshold args
      , bsRebalanceGlobal = argRebalanceGlobal args
      , bsFundingRate = argFundingRate args
      , bsFundingBySide = argFundingBySide args
      , bsRebalanceResetOnSignal = argRebalanceResetOnSignal args
      , bsFundingOnOpen = argFundingOnOpen args
      , bsBlendWeight = argBlendWeight args
      , bsTriLayer = argTriLayer args
      , bsTriLayerFastMult = argTriLayerFastMult args
      , bsTriLayerSlowMult = argTriLayerSlowMult args
      , bsTriLayerCloudPadding = argTriLayerCloudPadding args
      , bsTriLayerPriceAction = argTriLayerRequirePriceAction args
      , bsLstmExitFlipBars = argLstmExitFlipBars args
      , bsFee = feeUsed
      , bsSlippage = slippageUsed
      , bsSpread = spreadUsed
      , bsEstimatedPerSideCost = perSideCost
      , bsEstimatedRoundTripCost = roundTripCost
      , bsMetrics = metrics
      , bsBaselines = baselines
      , bsWalkForward = walkForward
      , bsLstmHistory = mHistory
      , bsLatestSignal = latestSignal
      , bsEquityCurve = brEquityCurve backtest
      , bsBacktestPrices = backtestPrices
      , bsKalmanPredNext = kalmanPredNext
      , bsLstmPredNext = lstmPredNext
      , bsPositions = brPositions backtest
      , bsAgreementOk = brAgreementOk backtest
      , bsTrades = brTrades backtest
      }

computeBaselines :: Double -> Double -> [Double] -> [Baseline]
computeBaselines periodsPerYear perSideCost prices =
  let ppy = max 1e-12 periodsPerYear
      perSide = min 0.999999 (max 0 perSideCost)
      n = length prices
      steps = max 0 (n - 1)
      buyHoldBt = baselineSimLongFlat perSide prices (replicate steps True)
      buyHold = Baseline { blName = "buy-hold", blMetrics = computeMetrics ppy buyHoldBt }

      pricesV = V.fromList prices
      prefix = V.scanl (+) 0 pricesV
      smaAt w i =
        if w <= 0 || i < 0
          then Nothing
          else
            let w' = max 1 w
                i1 = i + 1
             in if i1 < w'
                  then Nothing
                  else
                    let s = (prefix V.! i1) - (prefix V.! (i1 - w'))
                     in Just (s / fromIntegral w')

      longW0 = max 10 (min 60 (n `div` 5))
      longW = min (max 2 longW0) (max 2 (n - 1))
      shortW = max 2 (min (longW - 1) (max 2 (longW `div` 2)))

      smaCrossBaseline =
        if n < 3 || shortW >= longW || longW >= n
          then Nothing
          else
            let wantLong =
                  [ case (smaAt shortW t, smaAt longW t) of
                      (Just s, Just l) -> s > l
                      _ -> False
                  | t <- [0 .. steps - 1]
                  ]
                bt = baselineSimLongFlat perSide prices wantLong
                name = printf "sma-cross(%d/%d)" shortW longW
             in Just (Baseline { blName = name, blMetrics = computeMetrics ppy bt })
   in buyHold : maybe [] (:[]) smaCrossBaseline

baselineSimLongFlat :: Double -> [Double] -> [Bool] -> BacktestResult
baselineSimLongFlat perSideCost prices wantLong =
  let perSide = min 0.999999 (max 0 perSideCost)
      pricesV = V.fromList prices
      n = V.length pricesV
      stepCount = max 0 (n - 1)
      wantV = V.fromList (take stepCount wantLong <> replicate (max 0 (stepCount - length wantLong)) False)

      applyCost eq size =
        let s = min 1 (max 0 (abs size))
         in eq * (1 - perSide * s)

      isBad x = isNaN x || isInfinite x

      closeTrade exitIndex eqExit (entryIndex, entryEquity, holdingPeriods) =
        Trade
          { trEntryIndex = entryIndex
          , trExitIndex = exitIndex
          , trEntryEquity = entryEquity
          , trExitEquity = eqExit
          , trReturn = if entryEquity == 0 then 0 else eqExit / entryEquity - 1
          , trHoldingPeriods = holdingPeriods
          , trExitReason = Nothing
          }

      stepFn (posSize, equity, eqAcc, posAcc, changes, mOpen, tradesAcc) t =
        let prev = pricesV V.! t
            next = pricesV V.! (t + 1)
            desired = if wantV V.! t then 1 else 0

            (pos1, eq1, changes1, mOpen1, tradesAcc1) =
              if desired == posSize
                then (posSize, equity, changes, mOpen, tradesAcc)
                else
                  if desired == 0 && posSize > 0
                    then
                      let eqExit = applyCost equity posSize
                          tradesAcc' =
                            case mOpen of
                              Nothing -> tradesAcc
                              Just ot -> closeTrade t eqExit ot : tradesAcc
                       in (0, eqExit, changes + 1, Nothing, tradesAcc')
                    else
                      if desired > 0 && posSize == 0
                        then
                          let eqEntry = applyCost equity desired
                           in (desired, eqEntry, changes + 1, Just (t, eqEntry, 0 :: Int), tradesAcc)
                        else (posSize, equity, changes, mOpen, tradesAcc)

            factor = if prev == 0 then 1 else next / prev
            eqClose0 = eq1 * (1 + pos1 * (factor - 1))
            eqClose = if isBad eqClose0 then eq1 else eqClose0

            mOpen2 =
              case mOpen1 of
                Nothing -> Nothing
                Just (entryIndex, entryEq, holding) -> Just (entryIndex, entryEq, holding + 1)
         in (pos1, eqClose, eqClose : eqAcc, pos1 : posAcc, changes1, mOpen2, tradesAcc1)

      (posFinal, eqFinal, eqRev, posRev, changesFinal, mOpenFinal, tradesRev) =
        foldl'
          stepFn
          (0 :: Double, 1.0 :: Double, [1.0], [], 0 :: Int, Nothing :: Maybe (Int, Double, Int), [])
          [0 .. stepCount - 1]

      (eqRev', changesFinal', tradesRev') =
        case (posFinal, mOpenFinal) of
          (p, Just ot) | p > 0 ->
            let eqExit = applyCost eqFinal p
                eqRev1 =
                  case eqRev of
                    [] -> [eqExit]
                    (_ : rest) -> eqExit : rest
                tr = closeTrade stepCount eqExit ot
             in (eqRev1, changesFinal + 1, tr : tradesRev)
          _ -> (eqRev, changesFinal, tradesRev)
   in BacktestResult
        { brEquityCurve = reverse eqRev'
        , brPositions = reverse posRev
        , brAgreementOk = replicate stepCount True
        , brAgreementValid = replicate stepCount True
        , brPositionChanges = changesFinal'
        , brTrades = reverse tradesRev'
        }

computeLatestSignal
  :: Args
  -> Int
  -> V.Vector Double
  -> Maybe LstmCtx
  -> Maybe KalmanCtx
  -> Maybe MarketModel
  -> LatestSignal
computeLatestSignal args lookback pricesV mLstmCtx mKalmanCtx mMarketModel =
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
    MethodBlend ->
      case (mKalmanCtx, mLstmCtx) of
        (Just _, Just _) -> compute
        _ -> error "Internal: --method blend requires both Kalman and LSTM contexts."
  where
    method = argMethod args
    compute =
      let n = V.length pricesV
       in if n < 1
            then error ("Need at least 1 price to compute latest signal (got " ++ show n ++ ")")
            else
              let t = n - 1
                  currentPrice = pricesV V.! t
                  perSideCost = estimatedPerSideCost (argFee args) (argSlippage args) (argSpread args)
                  minEdgeBase = max 0 (argMinEdge args)
                  minEdge =
                    if argCostAwareEdge args
                      then max minEdgeBase (breakEvenThresholdFromPerSideCost perSideCost + max 0 (argEdgeBuffer args))
                      else minEdgeBase
                  minSignalToNoise = max 0 (argMinSignalToNoise args)
                  openThr = max (max 0 (argOpenThreshold args)) minEdge
                  closeThr = max 0 (argCloseThreshold args)
                  directionPrice thr pred =
                    let upEdge = currentPrice * (1 + thr)
                        downEdge = currentPrice * (1 - thr)
                     in if pred > upEdge
                          then Just (1 :: Int)
                          else if pred < downEdge then Just (-1) else Nothing

                  trendLookback = max 0 (argTrendLookback args)
                  maxPositionSize = max 0 (argMaxPositionSize args)
                  ppy = max 1e-12 (periodsPerYear args)
                  volTarget =
                    case argVolTarget args of
                      Just v | v > 0 && not (isNaN v || isInfinite v) -> Just v
                      _ -> Nothing
                  volLookback = max 0 (argVolLookback args)
                  volFloor = max 0 (argVolFloor args)
                  volScaleMax = max 0 (argVolScaleMax args)
                  volAlpha =
                    case argVolEwmaAlpha args of
                      Just a | a > 0 && not (isNaN a || isInfinite a) -> Just (max 0 (min 1 a))
                      _ -> Nothing
                  maxVolatility =
                    case argMaxVolatility args of
                      Just v | v > 0 && not (isNaN v || isInfinite v) -> Just v
                      _ -> Nothing

                  bad x = isNaN x || isInfinite x

                  returnsFromPrices =
                    case n of
                      _ | n < 2 -> []
                      _ ->
                        let pricesList = V.toList pricesV
                         in
                          [ let r = if p0 == 0 || bad p0 || bad p1 then 0 else p1 / p0 - 1
                             in if bad r then 0 else r
                          | (p0, p1) <- zip pricesList (tail pricesList)
                          ]

                  meanList xs =
                    if null xs then 0 else sum xs / fromIntegral (length xs)

                  stddevList xs =
                    case xs of
                      [] -> 0
                      [_] -> 0
                      _ ->
                        let m = meanList xs
                            var = sum (map (\x -> (x - m) ** 2) xs) / fromIntegral (length xs - 1)
                         in sqrt var

                  volEstimate =
                    let total = length returnsFromPrices
                     in if total < 2
                          then Nothing
                          else
                            case volAlpha of
                              Just a ->
                                let var = foldl' (\v r -> a * v + (1 - a) * (r * r)) 0 returnsFromPrices
                                    vol = sqrt (max 0 var) * sqrt ppy
                                 in if bad vol then Nothing else Just vol
                              Nothing ->
                                let lb = if volLookback <= 0 then total else min total volLookback
                                    window = drop (total - lb) returnsFromPrices
                                    vol = stddevList window * sqrt ppy
                                 in if bad vol then Nothing else Just vol

                  volScale =
                    case volTarget of
                      Nothing -> 1
                      Just target ->
                        case volEstimate of
                          Nothing -> 1
                          Just vol ->
                            let volAdj = max volFloor vol
                                scale0 = if volAdj <= 0 then 1 else target / volAdj
                                scale1 = min volScaleMax (max 0 scale0)
                             in if bad scale1 then 1 else scale1

                  volOk =
                    case maxVolatility of
                      Nothing -> True
                      Just maxVol ->
                        case volEstimate of
                          Just vol -> vol <= maxVol
                          Nothing -> False

                  volPerBar =
                    case volEstimate of
                      Just vol ->
                        let perBar = vol / sqrt ppy
                         in if bad perBar || perBar <= 0 then Nothing else Just perBar
                      _ -> Nothing

                  volTargetReady =
                    case volTarget of
                      Nothing -> True
                      Just _ ->
                        case volEstimate of
                          Just _ -> True
                          Nothing -> False

                  trendOk dir =
                    if trendLookback <= 1 || n < trendLookback
                      then True
                      else
                        let start = n - trendLookback
                            v = V.slice start trendLookback pricesV
                            sma = V.sum v / fromIntegral trendLookback
                         in if bad sma || bad currentPrice
                              then True
                              else
                                case dir of
                                  1 -> currentPrice >= sma
                                  (-1) -> currentPrice <= sma
                                  _ -> True

                  triLayerEnabled = argTriLayer args
                  cloudReady = triLayerEnabled && n >= 2
                  bodyMinFrac = max 1e-6 (0.25 * openThr)
                  kalDt = max 1e-12 (argKalmanDt args)
                  kalProcessVar = max 0 (argKalmanProcessVar args)
                  kalMeasVar = max 1e-12 (argKalmanMeasurementVar args)
                  fastMult = max 1e-6 (argTriLayerFastMult args)
                  slowMult = max 1e-6 (argTriLayerSlowMult args)
                  (cloudFastV, cloudSlowV) =
                    if cloudReady
                      then
                        let run mult =
                              let mv = max 1e-12 (kalMeasVar * mult)
                                  KalmanRun { krFiltered = filts } =
                                    runConstantAcceleration1D kalDt kalProcessVar mv (V.toList pricesV)
                               in V.fromList filts
                         in (run fastMult, run slowMult)
                      else (pricesV, pricesV)

                  candleAt i =
                    let c = pricesV V.! i
                        o = if i <= 0 then c else pricesV V.! (i - 1)
                        h = max o c
                        l = min o c
                     in (o, h, l, c)

                  candleOpen (o, _, _, _) = o
                  candleClose (_, _, _, c) = c
                  candleBody (o, _, _, c) = abs (c - o)
                  candleBull (o, _, _, c) = c > o
                  candleBear (o, _, _, c) = c < o
                  candleUpperWick (o, h, _, c) = max 0 (h - max o c)
                  candleLowerWick (o, _, l, c) = max 0 (min o c - l)

                  bodyOk closePx body =
                    let denom = max 1e-12 (abs closePx)
                     in body / denom >= bodyMinFrac

                  hammer candle =
                    let body = candleBody candle
                        upper = candleUpperWick candle
                        lower = candleLowerWick candle
                        closePx = candleClose candle
                     in candleBull candle
                          && body > 0
                          && bodyOk closePx body
                          && lower >= 2 * body
                          && upper <= 0.5 * body

                  shootingStar candle =
                    let body = candleBody candle
                        upper = candleUpperWick candle
                        lower = candleLowerWick candle
                        closePx = candleClose candle
                     in candleBear candle
                          && body > 0
                          && bodyOk closePx body
                          && upper >= 2 * body
                          && lower <= 0.5 * body

                  bullishEngulf cur prev =
                    let bodyCur = candleBody cur
                        bodyPrev = candleBody prev
                        closePx = candleClose cur
                     in candleBull cur
                          && candleBear prev
                          && bodyCur >= bodyPrev
                          && bodyOk closePx bodyCur
                          && candleClose cur >= candleOpen prev
                          && candleOpen cur <= candleClose prev

                  bearishEngulf cur prev =
                    let bodyCur = candleBody cur
                        bodyPrev = candleBody prev
                        closePx = candleClose cur
                     in candleBear cur
                          && candleBull prev
                          && bodyCur >= bodyPrev
                          && bodyOk closePx bodyCur
                          && candleClose cur <= candleOpen prev
                          && candleOpen cur >= candleClose prev

                  railroadTracksLong cur prev =
                    let bodyCur = candleBody cur
                        bodyPrev = candleBody prev
                        closePx = candleClose cur
                        tol = 0.2
                     in candleBull cur
                          && candleBear prev
                          && bodyPrev > 0
                          && bodyOk closePx bodyCur
                          && abs (bodyCur - bodyPrev) / bodyPrev <= tol

                  darkCloudCover cur prev =
                    let midPrev = (candleOpen prev + candleClose prev) / 2
                     in candleBear cur && candleBull prev && candleClose cur < midPrev

                  cloudOk dir =
                    if not cloudReady || t <= 0
                      then True
                      else
                        let fast = cloudFastV V.! t
                            slow = cloudSlowV V.! t
                            slope = slow - cloudSlowV V.! (t - 1)
                            cloudTop = max fast slow
                            cloudBot = min fast slow
                            (_, h, l, _) = candleAt t
                            touchCloud = l <= cloudTop && h >= cloudBot
                            trendOkCloud =
                              case dir of
                                1 -> fast > slow && slope >= 0
                                (-1) -> fast < slow && slope <= 0
                                _ -> True
                         in if bad fast || bad slow || bad slope
                              then True
                              else touchCloud && trendOkCloud

                  priceActionOk dir =
                    if not triLayerEnabled || t < 2
                      then True
                      else
                        let cur = candleAt t
                            prev = candleAt (t - 1)
                            bullish = hammer cur || bullishEngulf cur prev || railroadTracksLong cur prev
                            bearish = shootingStar cur || bearishEngulf cur prev || darkCloudCover cur prev
                         in case dir of
                              1 -> bullish
                              (-1) -> bearish
                              _ -> True

                  clamp01 :: Double -> Double
                  clamp01 x = max 0 (min 1 x)

                  blendWeight = clamp01 (argBlendWeight args)

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

                  confirmConformal :: Double -> Maybe Interval -> Int -> Bool
                  confirmConformal thr mI dir =
                    if not (argConfirmConformal args)
                      then True
                      else
                        case (mI, dir) of
                          (Just i, 1) -> iLo i > thr
                          (Just i, (-1)) -> iHi i < negate thr
                          _ -> False

                  confirmQuantiles :: Double -> Maybe Quantiles -> Int -> Bool
                  confirmQuantiles thr mQ dir =
                    if not (argConfirmQuantiles args)
                      then True
                      else
                        case (mQ, dir) of
                          (Just q, 1) -> q10 q > thr
                          (Just q, (-1)) -> q90 q < negate thr
                          _ -> False

                  gateKalmanDir :: Bool -> Double -> Double -> Maybe RegimeProbs -> Maybe Interval -> Maybe Quantiles -> Double -> Maybe Int -> (Maybe Int, Maybe String)
                  gateKalmanDir useSizing thr kalZ mReg mI mQ confScore dirRaw =
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
                         in if kalZ < zMin
                              then (Nothing, Just "KALMAN_Z")
                              else if not hvOk
                                then (Nothing, Just "HMM_HIGH_VOL")
                                else if not confWidthOk
                                  then (Nothing, Just "CONFORMAL_WIDTH")
                                  else if not qWidthOk
                                    then (Nothing, Just "QUANTILE_WIDTH")
                                    else if not (confirmConformal thr mI dir)
                                      then (Nothing, Just "CONFORMAL_CONFIRM")
                                      else if not (confirmQuantiles thr mQ dir)
                                        then (Nothing, Just "QUANTILE_CONFIRM")
                                        else if useSizing && confScore < argMinPositionSize args
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

                  (mKalNext, mKalReturn, mKalStd, mKalZ, mRegimes, mQuantiles, mConformal, kalDirRaw, kalDir, kalCloseDir, mConfidence, mPosSize, mGateReason) =
                    case mKalmanCtx of
                      Nothing -> (Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Just 0, Nothing)
                      Just (predictors, kalPrev, hmmPrev, svPrev) ->
                        let (sensorOuts, _) = predictSensors predictors pricesV hmmPrev t
                            mReg = listToMaybe [r | (_sid, out) <- sensorOuts, Just r <- [soRegimes out]]
                            mQ = listToMaybe [q | (_sid, out) <- sensorOuts, Just q <- [soQuantiles out]]
                            mI = listToMaybe [i | (_sid, out) <- sensorOuts, Just i <- [soInterval out]]
                            meas = mapMaybe (toMeasurement args svPrev) sensorOuts ++ maybeToList (mMarketModel >>= (`marketMeasurementAt` t))
                            kalNow = stepMulti meas kalPrev
                            kalReturn = kMean kalNow
                            kalVar = max 0 (kVar kalNow)
                            kalStd = sqrt kalVar
                            kalZ = if kalStd <= 0 then 0 else abs kalReturn / kalStd
                            kalNext = currentPrice * (1 + kalReturn)
                            dirRaw = directionPrice openThr kalNext
                            closeDirRaw = directionPrice closeThr kalNext
                            confScore = confidenceScoreKalman kalZ mReg mI mQ
                            sizeRaw =
                              if argConfidenceSizing args
                                then confScore
                                else
                                  case dirRaw of
                                    Nothing -> 0
                                    Just _ -> 1
                            (dirUsed, mWhy) = gateKalmanDir (argConfidenceSizing args) openThr kalZ mReg mI mQ confScore dirRaw
                            (closeDirUsed, _) = gateKalmanDir False closeThr kalZ mReg mI mQ confScore closeDirRaw
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
                          , closeDirUsed
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
                                 in (Just lstmNext, directionPrice openThr lstmNext)

                  blendNext =
                    case (mKalNext, mLstmNext) of
                      (Just k, Just l) -> Just (blendWeight * k + (1 - blendWeight) * l)
                      _ -> Nothing
                  edgeFromPred pred =
                    if bad pred || bad currentPrice || currentPrice == 0
                      then Nothing
                      else
                        let edge = abs (pred / currentPrice - 1)
                         in if bad edge then Nothing else Just edge
                  edgeKal = mKalNext >>= edgeFromPred
                  edgeLstm = mLstmNext >>= edgeFromPred
                  edgeBlend = blendNext >>= edgeFromPred
                  edgeForMethod =
                    case method of
                      MethodBoth ->
                        case (edgeKal, edgeLstm) of
                          (Just a, Just b) -> Just (min a b)
                          _ -> Nothing
                      MethodKalmanOnly -> edgeKal
                      MethodLstmOnly -> edgeLstm
                      MethodBlend -> edgeBlend
                  signalToNoiseOk =
                    if minSignalToNoise <= 0
                      then True
                      else
                        case (edgeForMethod, volPerBar) of
                          (Just edge, Just vol) | vol > 0 -> edge / vol >= minSignalToNoise
                          _ -> False
                  blendDir = blendNext >>= directionPrice openThr
                  lstmCloseDir = mLstmNext >>= directionPrice closeThr
                  blendCloseDir = blendNext >>= directionPrice closeThr
                  (blendDirGated, blendCloseDirGated, blendPosSize, blendGateReason) =
                    case (method, mKalZ, mConfidence) of
                      (MethodBlend, Just kalZ, Just confScore) ->
                        let sizeRaw =
                              if argConfidenceSizing args
                                then confScore
                                else if blendDir == Nothing then 0 else 1
                            (dirUsed, mWhy) =
                              gateKalmanDir (argConfidenceSizing args) openThr kalZ mRegimes mConformal mQuantiles confScore blendDir
                            (closeDirUsed, _) =
                              gateKalmanDir False closeThr kalZ mRegimes mConformal mQuantiles confScore blendCloseDir
                            sizeUsed =
                              case dirUsed of
                                Nothing -> 0
                                Just _ ->
                                  let s0 = if argConfidenceSizing args then sizeRaw else 1
                                   in if argConfidenceSizing args && s0 < argMinPositionSize args then 0 else s0
                         in (dirUsed, closeDirUsed, Just sizeUsed, mWhy)
                      _ -> (Nothing, Nothing, Nothing, Nothing)
                  closeDir =
                    case method of
                      MethodBoth ->
                        if kalCloseDir == lstmCloseDir
                          then kalCloseDir
                          else Nothing
                      MethodKalmanOnly -> kalCloseDir
                      MethodLstmOnly -> lstmCloseDir
                      MethodBlend -> blendCloseDirGated

                  agreeDir =
                    if kalDir == lstmDir
                      then kalDir
                      else Nothing
                  chosenDir0 =
                    case method of
                      MethodBoth -> agreeDir
                      MethodKalmanOnly -> kalDir
                      MethodLstmOnly -> lstmDir
                      MethodBlend -> blendDir
                  (chosenDir1, mPostGateReason) =
                    case chosenDir0 of
                      Nothing -> (Nothing, Nothing)
                      Just dir ->
                        if not volOk
                          then (Nothing, Just "MAX_VOLATILITY")
                          else if not volTargetReady
                            then (Nothing, Just "VOL_TARGET_WARMUP")
                            else if not (trendOk dir)
                            then (Nothing, Just "TREND_FILTER")
                            else if not (cloudOk dir)
                              then (Nothing, Just "KALMAN_CLOUD")
                            else if not (priceActionOk dir)
                              then (Nothing, Just "PRICE_ACTION")
                            else if not signalToNoiseOk
                              then (Nothing, Just "SIGNAL_TO_NOISE")
                              else (Just dir, Nothing)

                  chosenDir2 =
                    case chosenDir1 of
                      Nothing -> Nothing
                      Just _ ->
                        case method of
                          MethodLstmOnly -> chosenDir1
                          _ ->
                            case tradePosSize of
                              Just sz | sz <= 0 -> Nothing
                              _ -> chosenDir1

                  baseSize =
                    case method of
                      MethodLstmOnly ->
                        case chosenDir2 of
                          Nothing -> 0
                          Just _ -> 1
                      _ ->
                        case (chosenDir2, tradePosSize) of
                          (Just _, Just sz) -> sz
                          _ -> 0

                  sizeScaled = baseSize * volScale
                  sizeCapped = min maxPositionSize (max 0 sizeScaled)
                  sizeFinal0 =
                    if argConfidenceSizing args && sizeCapped < argMinPositionSize args
                      then 0
                      else sizeCapped

                  (chosenDir, mSizeGateReason) =
                    case chosenDir2 of
                      Nothing -> (Nothing, Nothing)
                      Just _ ->
                        if sizeFinal0 <= 0
                          then (Nothing, Just "MIN_SIZE")
                          else (chosenDir2, Nothing)

                  gateReasonFinal = mPostGateReason <|> mSizeGateReason <|> gateReasonForMethod

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
                            case agreeDir of
                              Just _ ->
                                case gateReasonFinal of
                                  Just why -> "HOLD (" ++ why ++ ")"
                                  Nothing -> "HOLD (confidence gate)"
                              Nothing -> "HOLD (directions disagree)"
                          _ -> "HOLD (directions disagree)"
                      MethodKalmanOnly ->
                        case (kalDirRaw, chosenDir) of
                          (Just 1, Just 1) -> "LONG"
                          (Just (-1), Just (-1)) -> downAction
                          (Just _, Nothing) ->
                            case gateReasonFinal of
                              Just why -> "HOLD (" ++ why ++ ")"
                              Nothing -> "HOLD (confidence gate)"
                          _ -> "HOLD (Kalman neutral)"
                      MethodLstmOnly ->
                        case chosenDir of
                          Just 1 -> "LONG"
                          Just (-1) -> downAction
                          _ ->
                            case gateReasonFinal of
                              Just why -> "HOLD (" ++ why ++ ")"
                              Nothing -> "HOLD (LSTM neutral)"
                      MethodBlend ->
                        case chosenDir of
                          Just 1 -> "LONG"
                          Just (-1) -> downAction
                          _ ->
                            case gateReasonFinal of
                              Just why -> "HOLD (" ++ why ++ ")"
                              Nothing -> "HOLD (blend neutral)"
                  posSizeFinal = Just sizeFinal0
                  tradePosSize =
                    case method of
                      MethodBlend -> blendPosSize
                      _ -> mPosSize
                  gateReasonForMethod =
                    case method of
                      MethodBlend -> blendGateReason
                      _ -> mGateReason
               in LatestSignal
                    { lsMethod = method
                    , lsCurrentPrice = currentPrice
                    , lsOpenThreshold = openThr
                    , lsCloseThreshold = closeThr
                    , lsKalmanNext = mKalNext
                    , lsKalmanReturn = mKalReturn
                    , lsKalmanStd = mKalStd
                    , lsKalmanZ = mKalZ
                    , lsVolatility = volEstimate
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
                    , lsCloseDir = closeDir
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
  putStrLn (printf "Open threshold:  %.3f%%" (lsOpenThreshold sig * 100))
  putStrLn (printf "Close threshold: %.3f%%" (lsCloseThreshold sig * 100))
  putStrLn (printf "Close dir:       %s" (showDir (lsCloseDir sig)))
  putStrLn (printf "Action: %s" (lsAction sig))

estimatedPerSideCost :: Double -> Double -> Double -> Double
estimatedPerSideCost fee slippage spread =
  let fee' = max 0 fee
      slip' = max 0 slippage
      spr' = max 0 spread
      c = fee' + slip' + spr' / 2
   in min 0.999999 (max 0 c)

estimatedRoundTripCost :: Double -> Double -> Double -> Double
estimatedRoundTripCost fee slippage spread =
  let perSide = estimatedPerSideCost fee slippage spread
   in min 0.999999 (2 * perSide)

breakEvenThresholdFromPerSideCost :: Double -> Double
breakEvenThresholdFromPerSideCost perSide0 =
  let perSide = min 0.999999 (max 0 perSide0)
      denom = max 1e-12 ((1 - perSide) * (1 - perSide))
      be = 1 / denom - 1
   in max 0 be

estimatedBreakEvenThreshold :: Double -> Double -> Double -> Double
estimatedBreakEvenThreshold fee slippage spread =
  let perSide = estimatedPerSideCost fee slippage spread
   in breakEvenThresholdFromPerSideCost perSide

printCostGuidance :: Double -> Double -> Double -> Double -> IO ()
printCostGuidance openThr closeThr perSide roundTrip = do
  let breakEven = breakEvenThresholdFromPerSideCost perSide
  putStrLn ""
  putStrLn "**Estimated Costs**"
  putStrLn (printf "Per side:  %.3f%%" (perSide * 100))
  putStrLn (printf "Round trip (approx): %.3f%%" (roundTrip * 100))
  putStrLn (printf "Break-even threshold: %.3f%%" (breakEven * 100))
  if openThr < breakEven
    then
      putStrLn
        ( printf
            "Warning: openThreshold %.3f%% is below break-even threshold %.3f%% (may churn after costs)."
            (openThr * 100)
            (breakEven * 100)
        )
    else pure ()
  if closeThr < breakEven
    then
      putStrLn
        ( printf
            "Note: closeThreshold %.3f%% is below break-even threshold %.3f%%."
            (closeThr * 100)
            (breakEven * 100)
        )
    else pure ()

printFundingGuidance :: Double -> Bool -> IO ()
printFundingGuidance fundingRate fundingBySide =
  when (fundingRate < 0 && not fundingBySide) $
    putStrLn
      ( printf
          "Warning: fundingRate %.3f%% is negative with side-agnostic funding; enable --funding-by-side for long/short sign."
          (fundingRate * 100)
      )

maybeSendOrder :: Maybe Webhook -> Args -> Maybe BinanceEnv -> LatestSignal -> IO ()
maybeSendOrder mWebhook args mBinanceEnv sig =
  case argPlatform args of
    PlatformBinance ->
      case (argBinanceSymbol args, mBinanceEnv) of
        (Just sym, Just env)
          | argBinanceTrade args -> do
              res <- placeOrderForSignal args sym sig env
              putStrLn (aorMessage res)
              webhookNotifyMaybeSync mWebhook (webhookEventTradeOrder args sig res)
          | otherwise -> pure ()
        _ -> pure ()
    PlatformCoinbase ->
      case argBinanceSymbol args of
        Just sym
          | argBinanceTrade args -> do
              envOrErr <- try (makeCoinbaseEnv args) :: IO (Either SomeException CoinbaseEnv)
              case envOrErr of
                Left _ -> pure ()
                Right env -> do
                  res <- placeCoinbaseOrderForSignal args sym sig env
                  putStrLn (aorMessage res)
                  webhookNotifyMaybeSync mWebhook (webhookEventTradeOrder args sig res)
          | otherwise -> pure ()
        _ -> pure ()
    _ -> pure ()

data PriceSeries = PriceSeries
  { psClose :: ![Double]
  , psHigh :: !(Maybe [Double])
  , psLow :: !(Maybe [Double])
  , psOpenTimes :: !(Maybe [Int64])
  } deriving (Eq, Show)

priceSourceLabel :: Args -> String
priceSourceLabel args =
  case (argBinanceSymbol args, argData args) of
    (Just sym, _) ->
      platformLabel (argPlatform args) ++ " " ++ sym ++ " (" ++ argInterval args ++ ")"
    (_, Just path) ->
      "CSV " ++ path ++ " (column " ++ show (argPriceCol args) ++ ")"
    _ ->
      "data source"

ensureMinPriceRows :: Args -> Int -> [Double] -> IO ()
ensureMinPriceRows args minRows prices =
  let n = length prices
      hint =
        case (argBinanceSymbol args, argData args) of
          (Just _, _) ->
            " Check symbol/interval and increase --bars (requested " ++ show (resolveBarsForPlatform args) ++ ")."
          (_, Just _) ->
            " Check the CSV has at least " ++ show minRows ++ " data rows (not counting the header)."
          _ -> ""
   in if n < minRows
        then error ("Need at least " ++ show minRows ++ " price rows (got " ++ show n ++ ") from " ++ priceSourceLabel args ++ "." ++ hint)
        else pure ()

loadPrices :: Maybe OpsStore -> Args -> IO (PriceSeries, Maybe BinanceEnv)
loadPrices mOps args =
  case (argData args, argBinanceSymbol args) of
    (Just path, Nothing) -> do
      (closes, mHighs, mLows, mOpenTimes) <- loadCsvPriceSeries path (argPriceCol args) (argHighCol args) (argLowCol args)
      let bars = resolveBarsForCsv args
      let closes' =
            if bars > 0
              then takeLast bars closes
              else closes
          highs' =
            if bars > 0
              then fmap (takeLast bars) mHighs
              else mHighs
          lows' =
            if bars > 0
              then fmap (takeLast bars) mLows
              else mLows
          openTimes' =
            if bars > 0
              then fmap (takeLast bars) mOpenTimes
              else mOpenTimes
      pure (PriceSeries closes' highs' lows' openTimes', Nothing)
    (Nothing, Just sym) ->
      case argPlatform args of
        PlatformBinance -> do
          (env, series) <- loadPricesBinance mOps args sym
          pure (series, Just env)
        PlatformCoinbase -> do
          series <- loadPricesCoinbase args sym
          pure (series, Nothing)
        PlatformKraken -> do
          series <- loadPricesKraken args sym
          pure (series, Nothing)
        PlatformPoloniex -> do
          series <- loadPricesPoloniex args sym
          pure (series, Nothing)
    (Just _, Just _) -> error "Provide only one of --data or --binance-symbol"
    (Nothing, Nothing) -> error "Provide --data or --binance-symbol"

takeLast :: Int -> [a] -> [a]
takeLast n xs
  | n <= 0 = []
  | otherwise =
      let k = length xs - n
       in if k <= 0 then xs else drop k xs

loadPricesBinance :: Maybe OpsStore -> Args -> String -> IO (BinanceEnv, PriceSeries)
loadPricesBinance mOps args sym = do
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
  envTrade <- newBinanceEnvWithOps mOps market tradeBase (BS.pack <$> apiKey) (BS.pack <$> apiSecret)
  klinesE <- try (fetchKlines envTrade sym (argInterval args) bars) :: IO (Either HttpException [Kline])
  ks <-
    case klinesE of
      Right out -> pure out
      Left ex ->
        if argBinanceTestnet args
          then do
            envData <- newBinanceEnvWithOps mOps market dataBase (BS.pack <$> apiKey) (BS.pack <$> apiSecret)
            fetchKlines envData sym (argInterval args) bars
          else throwIO ex
  let closes = map kClose ks
      highs = map kHigh ks
      lows = map kLow ks
      openTimes = map (normalizeEpochMs . kOpenTime) ks
  pure (envTrade, PriceSeries closes (Just highs) (Just lows) (Just openTimes))

loadPricesCoinbase :: Args -> String -> IO PriceSeries
loadPricesCoinbase args sym = do
  let interval = argInterval args
      bars = resolveBarsForPlatform args
  granularity <-
    case coinbaseIntervalSeconds interval of
      Just v -> pure v
      Nothing -> error ("Interval not supported by Coinbase: " ++ interval)
  candles <- fetchCoinbaseCandles sym granularity (max 1 bars)
  let closes = map ccClose candles
      highs = map ccHigh candles
      lows = map ccLow candles
      openTimes = map (normalizeEpochMs . ccOpenTime) candles
      closes' = if bars > 0 then takeLast bars closes else closes
      highs' = if bars > 0 then takeLast bars highs else highs
      lows' = if bars > 0 then takeLast bars lows else lows
      openTimes' = if bars > 0 then takeLast bars openTimes else openTimes
  pure (PriceSeries closes' (Just highs') (Just lows') (Just openTimes'))

loadPricesKraken :: Args -> String -> IO PriceSeries
loadPricesKraken args sym = do
  let interval = argInterval args
      bars = resolveBarsForPlatform args
  minutes <-
    case krakenIntervalMinutes interval of
      Just v -> pure v
      Nothing -> error ("Interval not supported by Kraken: " ++ interval)
  candles <- fetchKrakenCandles sym minutes
  let closes = map kcClose candles
      highs = map kcHigh candles
      lows = map kcLow candles
      openTimes = map (normalizeEpochMs . kcOpenTime) candles
      closes' = if bars > 0 then takeLast bars closes else closes
      highs' = if bars > 0 then takeLast bars highs else highs
      lows' = if bars > 0 then takeLast bars lows else lows
      openTimes' = if bars > 0 then takeLast bars openTimes else openTimes
  pure (PriceSeries closes' (Just highs') (Just lows') (Just openTimes'))

loadPricesPoloniex :: Args -> String -> IO PriceSeries
loadPricesPoloniex args sym = do
  let interval = argInterval args
      bars = resolveBarsForPlatform args
  (label, period) <-
    case (poloniexIntervalLabel interval, poloniexIntervalSeconds interval) of
      (Just l, Just p) -> pure (l, p)
      _ -> error ("Interval not supported by Poloniex: " ++ interval)
  candles <- fetchPoloniexCandles sym label period (max 1 bars)
  let closes = map pcClose candles
      highs = map pcHigh candles
      lows = map pcLow candles
      openTimes = map (normalizeEpochMs . pcOpenTime) candles
      closes' = if bars > 0 then takeLast bars closes else closes
      highs' = if bars > 0 then takeLast bars highs else highs
      lows' = if bars > 0 then takeLast bars lows else lows
      openTimes' = if bars > 0 then takeLast bars openTimes else openTimes
  pure (PriceSeries closes' (Just highs') (Just lows') (Just openTimes'))

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
  -> Maybe MarketModel
  -> (Kalman1, HMMFilter, SensorVar, [Double], [StepMeta])
  -> Int
  -> (Kalman1, HMMFilter, SensorVar, [Double], [StepMeta])
backtestStepKalmanOnly args pricesV predictors trainEnd mMarketModel (kal, hmm, sv, kalAcc, metaAcc) i =
  let t = trainEnd + i
      priceT = pricesV V.! t
      nextP = pricesV V.! (t + 1)
      realizedR = if priceT == 0 then 0 else nextP / priceT - 1

      (sensorOuts, predState) = predictSensors predictors pricesV hmm t
      mReg = listToMaybe [r | (_sid, out) <- sensorOuts, Just r <- [soRegimes out]]
      mQ = listToMaybe [q | (_sid, out) <- sensorOuts, Just q <- [soQuantiles out]]
      mI = listToMaybe [i' | (_sid, out) <- sensorOuts, Just i' <- [soInterval out]]
      meas = mapMaybe (toMeasurement args sv) sensorOuts ++ maybeToList (mMarketModel >>= (`marketMeasurementAt` t))
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
  -> Maybe MarketModel
  -> (Kalman1, HMMFilter, SensorVar, [Double], [Double], [StepMeta])
  -> Int
  -> (Kalman1, HMMFilter, SensorVar, [Double], [Double], [StepMeta])
backtestStep args lookback normState obsAll pricesV lstmModel predictors trainEnd mMarketModel (kal, hmm, sv, kalAcc, lstmAcc, metaAcc) i =
  let t = trainEnd + i
      priceT = pricesV V.! t
      nextP = pricesV V.! (t + 1)
      realizedR = if priceT == 0 then 0 else nextP / priceT - 1

      (sensorOuts, predState) = predictSensors predictors pricesV hmm t
      mReg = listToMaybe [r | (_sid, out) <- sensorOuts, Just r <- [soRegimes out]]
      mQ = listToMaybe [q | (_sid, out) <- sensorOuts, Just q <- [soQuantiles out]]
      mI = listToMaybe [i' | (_sid, out) <- sensorOuts, Just i' <- [soInterval out]]
      meas = mapMaybe (toMeasurement args sv) sensorOuts ++ maybeToList (mMarketModel >>= (`marketMeasurementAt` t))
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
  putStrLn (printf "Position changes: %d" (bmPositionChanges m))
  putStrLn (printf "Trades: %d" (bmTradeCount m))
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
          MethodBlend -> "Signal rate (Blend)"
  putStrLn (printf "%s: %.1f%%" agreeLabel (bmAgreementRate m * 100))
  putStrLn (printf "Turnover (changes/period): %.4f" (bmTurnover m))

shortResp :: BL.ByteString -> String
shortResp bs =
  let s = BS.unpack (BL.toStrict bs)
   in take 200 s

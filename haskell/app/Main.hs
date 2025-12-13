{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
module Main where

import Control.Exception (SomeException, fromException, try)
import Control.Applicative ((<|>))
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode, encode, object, (.=))
import qualified Data.Aeson as Aeson
import Data.Char (isSpace, toLower)
import Data.List (foldl')
import Data.Maybe (isJust, mapMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.Csv as Csv
import qualified Data.HashMap.Strict as HM
import qualified Data.Vector as V
import GHC.Exception (ErrorCall(..))
import GHC.Generics (Generic)
import Network.HTTP.Types (Status, status200, status400, status401, status404, status405, status500)
import Network.HTTP.Types.Header (hAuthorization)
import qualified Network.Wai as Wai
import qualified Network.Wai.Handler.Warp as Warp
import Options.Applicative
import System.Environment (lookupEnv)
import System.IO.Error (ioeGetErrorString, isUserError)
import Text.Printf (printf)
import Text.Read (readMaybe)

import Trader.Binance
  ( BinanceEnv(..)
  , BinanceMarket(..)
  , BinanceOrderMode(..)
  , OrderSide(..)
  , binanceBaseUrl
  , binanceTestnetBaseUrl
  , binanceFuturesBaseUrl
  , binanceFuturesTestnetBaseUrl
  , newBinanceEnv
  , fetchCloses
  , fetchFreeBalance
  , fetchFuturesPositionAmt
  , placeMarketOrder
  )
import Trader.KalmanFusion (Kalman1(..), initKalman1, stepMulti)
import Trader.LSTM
  ( LSTMConfig(..)
  , EpochStats(..)
  , LSTMModel
  , trainLSTM
  , predictNext
  , predictSeriesNext
  )
import Trader.Metrics (BacktestMetrics(..), computeMetrics)
import Trader.Duration (lookbackBarsFrom)
import Trader.Normalization (NormState, NormType(..), fitNorm, forwardSeries, inverseNorm, inverseSeries, parseNormType)
import Trader.Predictors
  ( PredictorBundle
  , SensorId(..)
  , SensorOutput(..)
  , HMMFilter(..)
  , trainPredictors
  , initHMMFilter
  , predictSensors
  , updateHMM
  )
import Trader.SensorVariance (SensorVar, emptySensorVar, updateResidual, varianceFor)
import Trader.Symbol (splitSymbol)
import Trader.Method (Method(..), methodCode, parseMethod, selectPredictions)
import Trader.Optimization (optimizeOperations, sweepThreshold)
import Trader.Split (Split(..), splitTrainBacktest)
import Trader.Trading (BacktestResult(..), EnsembleConfig(..), Trade(..), simulateEnsembleLongFlat)

-- CSV loading

loadPricesCsv :: FilePath -> String -> IO [Double]
loadPricesCsv path priceCol = do
  bs <- BL.readFile path
  case Csv.decodeByName bs of
    Left err -> error ("CSV decode failed: " ++ err)
    Right (_, rows) -> do
      let key = BS.pack priceCol
      pure $ map (extractPrice key) (V.toList rows)

extractPrice :: BS.ByteString -> Csv.NamedRecord -> Double
extractPrice key rec =
  case HM.lookup key rec of
    Nothing -> error ("Column not found: " ++ BS.unpack key)
    Just raw ->
      let s = trim (BS.unpack raw)
       in case readMaybe s of
            Just d -> d
            Nothing -> error ("Failed to parse price: " ++ s)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

type LstmCtx = (NormState, [Double], LSTMModel)

type KalmanCtx = (PredictorBundle, Kalman1, HMMFilter, SensorVar)

data LatestSignal = LatestSignal
  { lsMethod :: !Method
  , lsCurrentPrice :: !Double
  , lsThreshold :: !Double
  , lsKalmanNext :: !(Maybe Double)
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
  , bsBacktestSize :: !Int
  , bsBacktestRatio :: !Double
  , bsMethodUsed :: !Method
  , bsBestThreshold :: !Double
  , bsMetrics :: !BacktestMetrics
  , bsLstmHistory :: !(Maybe [EpochStats])
  , bsLatestSignal :: !LatestSignal
  , bsEquityCurve :: ![Double]
  , bsBacktestPrices :: ![Double]
  , bsPositions :: ![Int]
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
  , argNormalization :: NormType
  , argHiddenSize :: Int
  , argEpochs :: Int
  , argLr :: Double
  , argValRatio :: Double
  , argBacktestRatio :: Double
  , argPatience :: Int
  , argGradClip :: Maybe Double
  , argSeed :: Int
  , argKalmanDt :: Double
  , argKalmanProcessVar :: Double
  , argKalmanMeasurementVar :: Double
  , argTradeThreshold :: Double
  , argMethod :: Method
  , argOptimizeOperations :: Bool
  , argSweepThreshold :: Bool
  , argTradeOnly :: Bool
  , argFee :: Double
  , argPeriodsPerYear :: Maybe Double
  , argServe :: Bool
  , argPort :: Int
  } deriving (Eq, Show)

opts :: Parser Args
opts =
  Args
    <$> optional (strOption (long "data" <> metavar "PATH" <> help "CSV file containing prices"))
    <*> strOption (long "price-column" <> value "close" <> help "CSV column name for price")
    <*> optional (strOption (long "binance-symbol" <> metavar "SYMBOL" <> help "Fetch klines from Binance (e.g., BTCUSDT)"))
    <*> switch (long "futures" <> help "Use Binance USDT-M futures endpoints for data/orders")
    <*> switch (long "margin" <> help "Use Binance margin account endpoints for orders/balance")
    <*> strOption (long "interval" <> long "binance-interval" <> value "5m" <> help "Bar interval / Binance kline interval (e.g., 1m, 5m, 1h, 1d)")
    <*> option auto (long "bars" <> long "binance-limit" <> value 500 <> help "Number of bars/klines to use (Binance max 1000)")
    <*> strOption (long "lookback-window" <> value "24h" <> help "Lookback window duration (e.g., 90m, 24h, 7d)")
    <*> optional (option auto (long "lookback-bars" <> long "lookback" <> help "Override lookback bars (disables --lookback-window conversion)"))
    <*> switch (long "binance-testnet" <> help "Use Binance testnet base URL (public + signed endpoints)")
    <*> optional (strOption (long "binance-api-key" <> help "Binance API key (or env BINANCE_API_KEY)"))
    <*> optional (strOption (long "binance-api-secret" <> help "Binance API secret (or env BINANCE_API_SECRET)"))
    <*> switch (long "binance-trade" <> help "If set, place a market order for the latest signal")
    <*> switch (long "binance-live" <> help "If set, send LIVE orders (otherwise uses /order/test)")
    <*> optional (option auto (long "order-quote" <> help "Quote amount to spend on BUY (quoteOrderQty)"))
    <*> optional (option auto (long "order-quantity" <> help "Base quantity to trade (quantity)"))
    <*> option (maybeReader parseNormType) (long "normalization" <> value NormStandard <> help "none|minmax|standard|log")
    <*> option auto (long "hidden-size" <> value 16 <> help "LSTM hidden size")
    <*> option auto (long "epochs" <> value 30 <> help "LSTM training epochs (Adam)")
    <*> option auto (long "lr" <> value 1e-3 <> help "LSTM learning rate")
    <*> option auto (long "val-ratio" <> value 0.2 <> help "Validation split ratio (within training set)")
    <*> option auto (long "backtest-ratio" <> value 0.2 <> help "Backtest holdout ratio (last portion of series)")
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
              <> help "Method: 11=Kalman+LSTM (direction-agreement gated), 10=Kalman only, 01=LSTM only"
          )
    <*> switch (long "optimize-operations" <> help "Optimize method (11/10/01) and threshold on the backtest split")
    <*> switch (long "sweep-threshold" <> help "Sweep trade thresholds on the backtest split and print the best final equity")
    <*> switch (long "trade-only" <> help "Skip backtest/metrics; only compute the latest signal (and optionally place an order)")
    <*> option auto (long "fee" <> value 0.0005 <> help "Fee applied when switching position")
    <*> optional (option auto (long "periods-per-year" <> help "For annualized metrics (e.g., 365 for 1d, 8760 for 1h)"))
    <*> switch (long "serve" <> help "Run REST API server on localhost instead of running the CLI workflow")
    <*> option auto (long "port" <> value 8080 <> help "REST API port (when --serve)")

argBinanceMarket :: Args -> BinanceMarket
argBinanceMarket args =
  case (argBinanceFutures args, argBinanceMargin args) of
    (True, True) -> error "Choose only one of --futures or --margin"
    (True, False) -> MarketFutures
    (False, True) -> MarketMargin
    (False, False) -> MarketSpot

argLookback :: Args -> Int
argLookback args =
  let positive n =
        if n < 2
          then error "--lookback-bars must be >= 2"
          else n
   in case argLookbackBars args of
        Just n -> positive n
        Nothing ->
          case lookbackBarsFrom (argInterval args) (argLookbackWindow args) of
            Left err -> error err
            Right n -> positive n

main :: IO ()
main = do
  args <- execParser (info (opts <**> helper) fullDesc)

  if argServe args
    then runRestApi args
    else do
      (prices, mBinanceEnv) <- loadPrices args
      if length prices < 2 then error "Need at least 2 price rows" else pure ()

      let lookback = argLookback args
      if argTradeOnly args
        then runTradeOnly args lookback prices mBinanceEnv
        else runBacktestPipeline args lookback prices mBinanceEnv

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
  , apNormalization :: Maybe String
  , apHiddenSize :: Maybe Int
  , apEpochs :: Maybe Int
  , apLr :: Maybe Double
  , apValRatio :: Maybe Double
  , apBacktestRatio :: Maybe Double
  , apPatience :: Maybe Int
  , apGradClip :: Maybe Double
  , apSeed :: Maybe Int
  , apKalmanDt :: Maybe Double
  , apKalmanProcessVar :: Maybe Double
  , apKalmanMeasurementVar :: Maybe Double
  , apThreshold :: Maybe Double
  , apMethod :: Maybe String -- "11" | "10" | "01"
  , apOptimizeOperations :: Maybe Bool
  , apSweepThreshold :: Maybe Bool
  , apFee :: Maybe Double
  , apPeriodsPerYear :: Maybe Double
  , apBinanceLive :: Maybe Bool
  , apOrderQuote :: Maybe Double
  , apOrderQuantity :: Maybe Double
  } deriving (Eq, Show, Generic)

instance FromJSON ApiParams where
  parseJSON = Aeson.genericParseJSON (jsonOptions 2)

data ApiError = ApiError
  { aeError :: String
  } deriving (Eq, Show, Generic)

instance ToJSON ApiError where
  toJSON = Aeson.genericToJSON (jsonOptions 2)

instance ToJSON LatestSignal where
  toJSON s =
    object
      [ "method" .= methodCode (lsMethod s)
      , "currentPrice" .= lsCurrentPrice s
      , "threshold" .= lsThreshold s
      , "kalmanNext" .= lsKalmanNext s
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

runRestApi :: Args -> IO ()
runRestApi baseArgs = do
  apiToken <- fmap BS.pack <$> lookupEnv "TRADER_API_TOKEN"
  let port = max 1 (argPort baseArgs)
      settings =
        Warp.setHost "0.0.0.0" $
          Warp.setPort port Warp.defaultSettings
  putStrLn (printf "REST API listening on http://0.0.0.0:%d" port)
  Warp.runSettings settings (apiApp baseArgs apiToken)

apiApp :: Args -> Maybe BS.ByteString -> Wai.Application
apiApp baseArgs apiToken req respond =
  let path = Wai.pathInfo req
   in if path /= ["health"] && not (authorized apiToken req)
        then respond (jsonError status401 "Unauthorized")
        else
          case path of
            ["health"] ->
              case Wai.requestMethod req of
                "GET" -> respond (jsonValue status200 (object ["status" .= ("ok" :: String)]))
                _ -> respond (jsonError status405 "Method not allowed")
            ["signal"] ->
              case Wai.requestMethod req of
                "POST" -> handleSignal baseArgs req respond
                _ -> respond (jsonError status405 "Method not allowed")
            ["trade"] ->
              case Wai.requestMethod req of
                "POST" -> handleTrade baseArgs req respond
                _ -> respond (jsonError status405 "Method not allowed")
            ["backtest"] ->
              case Wai.requestMethod req of
                "POST" -> handleBacktest baseArgs req respond
                _ -> respond (jsonError status405 "Method not allowed")
            _ -> respond (jsonError status404 "Not found")

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
    [("Content-Type", "application/json")]
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
        _ -> (status500, show ex)

handleSignal :: Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleSignal baseArgs req respond = do
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
          r <- try (computeLatestSignalFromArgs args) :: IO (Either SomeException LatestSignal)
          case r of
            Left ex ->
              let (st, msg) = exceptionToHttp ex
               in respond (jsonError st msg)
            Right sig -> respond (jsonValue status200 sig)

handleTrade :: Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleTrade baseArgs req respond = do
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
                  , argBinanceTrade = True
                  , argBinanceLive = maybe (argBinanceLive args0) id (apBinanceLive params)
                  , argOrderQuote = apOrderQuote params <|> argOrderQuote args0
                  , argOrderQuantity = apOrderQuantity params <|> argOrderQuantity args0
                  , argSweepThreshold = False
                  , argOptimizeOperations = False
                  }
          r <- try (computeTradeFromArgs args) :: IO (Either SomeException ApiTradeResponse)
          case r of
            Left ex ->
              let (st, msg) = exceptionToHttp ex
               in respond (jsonError st msg)
            Right out -> respond (jsonValue status200 out)

handleBacktest :: Args -> Wai.Request -> (Wai.Response -> IO Wai.ResponseReceived) -> IO Wai.ResponseReceived
handleBacktest baseArgs req respond = do
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
          r <- try (computeBacktestFromArgs args) :: IO (Either SomeException Aeson.Value)
          case r of
            Left ex ->
              let (st, msg) = exceptionToHttp ex
               in respond (jsonError st msg)
            Right out -> respond (jsonValue status200 out)

argsFromApi :: Args -> ApiParams -> Either String Args
argsFromApi baseArgs p = do
  method <-
    case apMethod p of
      Nothing -> Right (argMethod baseArgs)
      Just raw -> parseMethod raw

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
          , argNormalization = norm
          , argHiddenSize = pick (apHiddenSize p) (argHiddenSize baseArgs)
          , argEpochs = pick (apEpochs p) (argEpochs baseArgs)
          , argLr = pick (apLr p) (argLr baseArgs)
          , argValRatio = pick (apValRatio p) (argValRatio baseArgs)
          , argBacktestRatio = pick (apBacktestRatio p) (argBacktestRatio baseArgs)
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
          , argOptimizeOperations = pick (apOptimizeOperations p) (argOptimizeOperations baseArgs)
          , argSweepThreshold = pick (apSweepThreshold p) (argSweepThreshold baseArgs)
          , argFee = pick (apFee p) (argFee baseArgs)
          , argPeriodsPerYear =
              case apPeriodsPerYear p of
                Nothing -> argPeriodsPerYear baseArgs
                Just v -> Just v
          , argBinanceLive = pick (apBinanceLive p) (argBinanceLive baseArgs)
          , argOrderQuote = pickMaybe (apOrderQuote p) (argOrderQuote baseArgs)
          , argOrderQuantity = pickMaybe (apOrderQuantity p) (argOrderQuantity baseArgs)
          }

  pure args

dirLabel :: Maybe Int -> Maybe String
dirLabel d =
  case d of
    Just 1 -> Just "UP"
    Just (-1) -> Just "DOWN"
    _ -> Nothing

computeLatestSignalFromArgs :: Args -> IO LatestSignal
computeLatestSignalFromArgs args = do
  (prices, _) <- loadPrices args
  if length prices < 2 then error "Need at least 2 price rows" else pure ()
  let lookback = argLookback args
  computeTradeOnlySignal args lookback prices

computeTradeFromArgs :: Args -> IO ApiTradeResponse
computeTradeFromArgs args = do
  (prices, mBinanceEnv) <- loadPrices args
  if length prices < 2 then error "Need at least 2 price rows" else pure ()
  let lookback = argLookback args
  sig <- computeTradeOnlySignal args lookback prices
  order <-
    case (argBinanceSymbol args, mBinanceEnv) of
      (Nothing, _) -> pure (ApiOrderResult False Nothing Nothing Nothing Nothing Nothing Nothing "No order: missing binanceSymbol.")
      (_, Nothing) -> pure (ApiOrderResult False Nothing Nothing Nothing Nothing Nothing Nothing "No order: missing Binance environment (use binanceSymbol data source).")
      (Just sym, Just env) -> placeOrderForSignal args sym sig env
  pure ApiTradeResponse { atrSignal = sig, atrOrder = order }

placeOrderForSignal :: Args -> String -> LatestSignal -> BinanceEnv -> IO ApiOrderResult
placeOrderForSignal args sym sig env = do
  let mode = if argBinanceLive args then OrderLive else OrderTest
      modeLabelStr = case mode of { OrderLive -> "live"; OrderTest -> "test" }
      base = ApiOrderResult False (Just modeLabelStr) Nothing (Just sym) Nothing Nothing Nothing

  let noOrder msg = pure (base msg)

  case (beApiKey env, beApiSecret env) of
    (Nothing, _) -> noOrder "No order: missing Binance API key."
    (_, Nothing) -> noOrder "No order: missing Binance API secret."
    (Just _, Just _) -> do
      let (baseAsset, _) = splitSymbol sym
      r <- try (place baseAsset) :: IO (Either SomeException ApiOrderResult)
      case r of
        Left ex -> noOrder ("Order failed: " ++ show ex)
        Right out -> pure out
  where
    method = lsMethod sig
    chosenDir = lsChosenDir sig
    currentPrice = lsCurrentPrice sig

    place baseAsset =
      case beMarket env of
        MarketSpot -> placeSpotOrMargin baseAsset
        MarketMargin ->
          if mode == OrderTest
            then pure (ApiOrderResult False (Just "test") Nothing (Just sym) Nothing Nothing Nothing "No order: margin trading requires binanceLive (no test endpoint).")
            else placeSpotOrMargin baseAsset
        MarketFutures -> placeFutures

    mode = if argBinanceLive args then OrderLive else OrderTest

    placeSpotOrMargin baseAsset = do
      baseBal <- fetchFreeBalance env baseAsset
      case chosenDir of
        Just 1 ->
          if baseBal <= 0
            then do
              let qty = argOrderQuantity args
                  qq = argOrderQuote args
              case (qty, qq) of
                (Nothing, Nothing) ->
                  pure (ApiOrderResult False (Just (modeLabel mode)) Nothing (Just sym) Nothing Nothing Nothing "No order: provide orderQuantity or orderQuote.")
                _ -> do
                  resp <- placeMarketOrder env mode sym Buy qty qq Nothing
                  pure (ApiOrderResult True (Just (modeLabel mode)) (Just "BUY") (Just sym) qty qq (Just (shortResp resp)) "Order sent.")
            else pure (ApiOrderResult False (Just (modeLabel mode)) Nothing (Just sym) Nothing Nothing Nothing "No order: already long.")
        Just (-1) ->
          if baseBal > 0
            then do
              let qty = Just (maybe baseBal id (argOrderQuantity args))
              resp <- placeMarketOrder env mode sym Sell qty Nothing Nothing
              pure (ApiOrderResult True (Just (modeLabel mode)) (Just "SELL") (Just sym) qty Nothing (Just (shortResp resp)) "Order sent.")
            else pure (ApiOrderResult False (Just (modeLabel mode)) Nothing (Just sym) Nothing Nothing Nothing "No order: already flat.")
        _ ->
          let msg =
                case method of
                  MethodBoth -> "No order: directions disagree or neutral (direction gate)."
                  MethodKalmanOnly -> "No order: Kalman neutral (within threshold)."
                  MethodLstmOnly -> "No order: LSTM neutral (within threshold)."
           in pure (ApiOrderResult False (Just (modeLabel mode)) Nothing (Just sym) Nothing Nothing Nothing msg)

    placeFutures = do
      posAmt <- fetchFuturesPositionAmt env sym
      let mDesiredQty =
            case argOrderQuantity args of
              Just q | q > 0 -> Just q
              Just _ -> Nothing
              Nothing ->
                case argOrderQuote args of
                  Just qq | qq > 0 && currentPrice > 0 -> Just (qq / currentPrice)
                  _ -> Nothing
          closeOrder side qty = do
            resp <- placeMarketOrder env mode sym side (Just qty) Nothing (Just True)
            pure (ApiOrderResult True (Just (modeLabel mode)) (Just (show side)) (Just sym) (Just qty) Nothing (Just (shortResp resp)) "Order sent.")
      case chosenDir of
        Just 1 ->
          if posAmt > 0
            then pure (ApiOrderResult False (Just (modeLabel mode)) Nothing (Just sym) Nothing Nothing Nothing "No order: already long.")
            else
              case mDesiredQty of
                Nothing -> pure (ApiOrderResult False (Just (modeLabel mode)) Nothing (Just sym) Nothing Nothing Nothing "No order: futures requires orderQuantity or orderQuote.")
                Just q -> do
                  let qtyToBuy = if posAmt < 0 then abs posAmt + q else q
                  resp <- placeMarketOrder env mode sym Buy (Just qtyToBuy) Nothing Nothing
                  pure (ApiOrderResult True (Just (modeLabel mode)) (Just "BUY") (Just sym) (Just qtyToBuy) Nothing (Just (shortResp resp)) "Order sent.")
        Just (-1) ->
          if posAmt == 0
            then pure (ApiOrderResult False (Just (modeLabel mode)) Nothing (Just sym) Nothing Nothing Nothing "No order: already flat.")
            else if posAmt > 0
              then closeOrder Sell (abs posAmt)
              else closeOrder Buy (abs posAmt)
        _ ->
          let msg =
                case method of
                  MethodBoth -> "No order: directions disagree or neutral (direction gate)."
                  MethodKalmanOnly -> "No order: Kalman neutral (within threshold)."
                  MethodLstmOnly -> "No order: LSTM neutral (within threshold)."
           in pure (ApiOrderResult False (Just (modeLabel mode)) Nothing (Just sym) Nothing Nothing Nothing msg)

    modeLabel m =
      case m of
        OrderLive -> "live"
        OrderTest -> "test"

computeBacktestFromArgs :: Args -> IO Aeson.Value
computeBacktestFromArgs args = do
  (prices, _) <- loadPrices args
  if length prices < 2 then error "Need at least 2 price rows" else pure ()
  let lookback = argLookback args
  summary <- computeBacktestSummary args lookback prices
  pure $
    object
      [ "split"
          .= object
            [ "train" .= bsTrainSize summary
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
  printLatestSignalSummary signal
  maybeSendBinanceOrder args mBinanceEnv signal

runBacktestPipeline :: Args -> Int -> [Double] -> Maybe BinanceEnv -> IO ()
runBacktestPipeline args lookback prices mBinanceEnv = do
  summary <- computeBacktestSummary args lookback prices
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
            "Optimized operations: method=%s threshold=%.6f (%.3f%%)"
            (methodCode (bsMethodUsed summary))
            (bsBestThreshold summary)
            (bsBestThreshold summary * 100)
        )
    else if argSweepThreshold args
      then
        putStrLn
          ( printf
              "Best threshold (by final equity): %.6f (%.3f%%)"
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

computeBacktestSummary :: Args -> Int -> [Double] -> IO BacktestSummary
computeBacktestSummary args lookback prices = do
  let backtestRatio = argBacktestRatio args
      split =
        case splitTrainBacktest lookback backtestRatio prices of
          Left err -> error err
          Right s -> s

      trainEndRaw = splitTrainEndRaw split
      trainEnd = splitTrainEnd split
      trainPrices = splitTrain split
      backtestPrices = splitBacktest split
      methodRequested = argMethod args
      methodForComputation =
        if argOptimizeOperations args
          then MethodBoth
          else methodRequested
      pricesV = V.fromList prices
      stepCount = length backtestPrices - 1

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

      (mLstmCtx, mHistory, kalPredPrice, lstmPredPrice, mKalmanCtx) =
        case methodForComputation of
          MethodKalmanOnly ->
            let trainPricesV = V.fromList trainPrices
                predictors = trainPredictors lookback trainPricesV
                hmmInitReturns = forwardReturns (take (trainEnd + 1) prices)
                hmm0 = initHMMFilter predictors hmmInitReturns
                kal0 =
                  initKalman1
                    0
                    (max 1e-12 (argKalmanMeasurementVar args))
                    (max 0 (argKalmanProcessVar args) * max 0 (argKalmanDt args))
                sv0 = emptySensorVar
                (kalFinal, hmmFinal, svFinal, kalPredRev) =
                  foldl'
                    (backtestStepKalmanOnly args pricesV predictors trainEnd)
                    (kal0, hmm0, sv0, [])
                    [0 .. stepCount - 1]
                kalPred = reverse kalPredRev
             in (Nothing, Nothing, kalPred, kalPred, Just (predictors, kalFinal, hmmFinal, svFinal))
          MethodLstmOnly ->
            let normState = fitNorm (argNormalization args) trainPrices
                obsAll = forwardSeries normState prices
                obsTrain = take trainEnd obsAll
                (lstmModel, history) = trainLSTM lstmCfg obsTrain
                lstmPred =
                  [ let t = trainEnd + i
                        window = take lookback (drop (t - lookback + 1) obsAll)
                        predObs = predictNext lstmModel window
                     in inverseNorm normState predObs
                  | i <- [0 .. stepCount - 1]
                  ]
             in (Just (normState, obsAll, lstmModel), Just history, lstmPred, lstmPred, Nothing)
          MethodBoth ->
            let normState = fitNorm (argNormalization args) trainPrices
                obsAll = forwardSeries normState prices
                obsTrain = take trainEnd obsAll
                (lstmModel, history) = trainLSTM lstmCfg obsTrain
                trainPricesV = V.fromList trainPrices
                predictors = trainPredictors lookback trainPricesV
                hmmInitReturns = forwardReturns (take (trainEnd + 1) prices)
                hmm0 = initHMMFilter predictors hmmInitReturns
                kal0 =
                  initKalman1
                    0
                    (max 1e-12 (argKalmanMeasurementVar args))
                    (max 0 (argKalmanProcessVar args) * max 0 (argKalmanDt args))
                sv0 = emptySensorVar
                (kalFinal, hmmFinal, svFinal, kalPredRev, lstmPredRev) =
                  foldl'
                    (backtestStep args lookback normState obsAll pricesV lstmModel predictors trainEnd)
                    (kal0, hmm0, sv0, [], [])
                    [0 .. stepCount - 1]
                kalPred = reverse kalPredRev
                lstmPred = reverse lstmPredRev
             in
              ( Just (normState, obsAll, lstmModel)
              , Just history
              , kalPred
              , lstmPred
              , Just (predictors, kalFinal, hmmFinal, svFinal)
              )

      (methodUsed, bestThreshold, backtest) =
        if argOptimizeOperations args
          then optimizeOperations (argTradeThreshold args) (argFee args) backtestPrices kalPredPrice lstmPredPrice
          else if argSweepThreshold args
            then
              let (thr, bt) = sweepThreshold methodRequested (argTradeThreshold args) (argFee args) backtestPrices kalPredPrice lstmPredPrice
               in (methodRequested, thr, bt)
            else
              let (kalPredUsed, lstmPredUsed) = selectPredictions methodRequested kalPredPrice lstmPredPrice
                  cfg =
                    EnsembleConfig
                      { ecTradeThreshold = argTradeThreshold args
                      , ecFee = argFee args
                      }
               in (methodRequested, argTradeThreshold args, simulateEnsembleLongFlat cfg 1 backtestPrices kalPredUsed lstmPredUsed)

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
      , bsTrainSize = length trainPrices
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
                  direction pred =
                    let upEdge = currentPrice * (1 + thr)
                        downEdge = currentPrice * (1 - thr)
                     in if pred > upEdge
                          then Just (1 :: Int)
                          else if pred < downEdge then Just (-1) else Nothing

                  (mKalNext, kalDir) =
                    case mKalmanCtx of
                      Nothing -> (Nothing, Nothing)
                      Just (predictors, kalPrev, hmmPrev, svPrev) ->
                        let (sensorOuts, _) = predictSensors predictors pricesV hmmPrev t
                            meas = mapMaybe (toMeasurement args svPrev) sensorOuts
                            kalNow = stepMulti meas kalPrev
                            kalReturn = kMean kalNow
                            kalNext = currentPrice * (1 + kalReturn)
                         in (Just kalNext, direction kalNext)

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
                                 in (Just lstmNext, direction lstmNext)

                  agreeDir =
                    if kalDir == lstmDir
                      then kalDir
                      else Nothing
                  chosenDir =
                    case method of
                      MethodBoth -> agreeDir
                      MethodKalmanOnly -> kalDir
                      MethodLstmOnly -> lstmDir

                  action =
                    case method of
                      MethodBoth ->
                        case (kalDir, lstmDir) of
                          (Just 1, Just 1) -> "LONG"
                          (Just (-1), Just (-1)) -> "FLAT"
                          (Nothing, Nothing) -> "HOLD (both neutral)"
                          _ -> "HOLD (directions disagree)"
                      MethodKalmanOnly ->
                        case kalDir of
                          Just 1 -> "LONG"
                          Just (-1) -> "FLAT"
                          _ -> "HOLD (Kalman neutral)"
                      MethodLstmOnly ->
                        case lstmDir of
                          Just 1 -> "LONG"
                          Just (-1) -> "FLAT"
                          _ -> "HOLD (LSTM neutral)"
               in LatestSignal
                    { lsMethod = method
                    , lsCurrentPrice = currentPrice
                    , lsThreshold = thr
                    , lsKalmanNext = mKalNext
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

loadPrices :: Args -> IO ([Double], Maybe BinanceEnv)
loadPrices args =
  case (argData args, argBinanceSymbol args) of
    (Just path, Nothing) -> do
      ps <- loadPricesCsv path (argPriceCol args)
      let ps' =
            if argBars args > 0
              then takeLast (argBars args) ps
              else ps
      pure (ps', Nothing)
    (Nothing, Just sym) -> do
      (env, ps) <- loadPricesBinance args sym
      pure (ps, Just env)
    (Just _, Just _) -> error "Provide only one of --data or --binance-symbol"
    (Nothing, Nothing) -> error "Provide --data or --binance-symbol"

takeLast :: Int -> [a] -> [a]
takeLast n xs
  | n <= 0 = xs
  | otherwise =
      let k = length xs - n
       in if k <= 0 then xs else drop k xs

loadPricesBinance :: Args -> String -> IO (BinanceEnv, [Double])
loadPricesBinance args sym = do
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
  env <- newBinanceEnv market base (BS.pack <$> apiKey) (BS.pack <$> apiSecret)
  closes <- fetchCloses env sym (argInterval args) (argBars args)
  pure (env, closes)

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
    "1w" -> 52
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
  -> (Kalman1, HMMFilter, SensorVar, [Double])
  -> Int
  -> (Kalman1, HMMFilter, SensorVar, [Double])
backtestStepKalmanOnly args pricesV predictors trainEnd (kal, hmm, sv, kalAcc) i =
  let t = trainEnd + i
      priceT = pricesV V.! t
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
   in (kal', hmm', sv', kalNext : kalAcc)

backtestStep
  :: Args
  -> Int
  -> NormState
  -> [Double]
  -> V.Vector Double
  -> LSTMModel
  -> PredictorBundle
  -> Int
  -> (Kalman1, HMMFilter, SensorVar, [Double], [Double])
  -> Int
  -> (Kalman1, HMMFilter, SensorVar, [Double], [Double])
backtestStep args lookback normState obsAll pricesV lstmModel predictors trainEnd (kal, hmm, sv, kalAcc, lstmAcc) i =
  let t = trainEnd + i
      priceT = pricesV V.! t
      nextP = pricesV V.! (t + 1)
      realizedR = if priceT == 0 then 0 else nextP / priceT - 1

      (sensorOuts, predState) = predictSensors predictors pricesV hmm t
      meas = mapMaybe (toMeasurement args sv) sensorOuts
      kal' = stepMulti meas kal
      fusedR = kMean kal'
      kalNext = priceT * (1 + fusedR)

      window = take lookback (drop (t - lookback + 1) obsAll)
      lstmNextObs = predictNext lstmModel window
      lstmNext = inverseNorm normState lstmNextObs

      sv' =
        foldl'
          (\acc (sid, out) -> updateResidual sid (realizedR - soMu out) acc)
          sv
          sensorOuts
      hmm' = updateHMM predictors predState realizedR
   in (kal', hmm', sv', kalNext : kalAcc, lstmNext : lstmAcc)

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
  putStrLn (printf "Profit factor: %.3f" (bmProfitFactor m))
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

{-# LANGUAGE OverloadedStrings #-}
module Main where

import Data.Char (isSpace)
import Data.List (foldl')
import Data.Maybe (mapMaybe)
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.Csv as Csv
import qualified Data.HashMap.Strict as HM
import qualified Data.Vector as V
import Options.Applicative
import System.Environment (lookupEnv)
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
import Trader.Trading (BacktestResult(..), EnsembleConfig(..), simulateEnsembleLongFlat)

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

data Method
  = MethodBoth
  | MethodKalmanOnly
  | MethodLstmOnly
  deriving (Eq, Show)

methodCode :: Method -> String
methodCode m =
  case m of
    MethodBoth -> "11"
    MethodKalmanOnly -> "10"
    MethodLstmOnly -> "01"

parseMethod :: String -> Either String Method
parseMethod raw =
  case trim raw of
    "11" -> Right MethodBoth
    "10" -> Right MethodKalmanOnly
    "01" -> Right MethodLstmOnly
    other ->
      Left
        ( "Invalid --method: "
            ++ show other
            ++ " (expected 11 for both, 10 for Kalman only, 01 for LSTM only)"
        )

selectPredictions :: Method -> [Double] -> [Double] -> ([Double], [Double])
selectPredictions m kalPred lstmPred =
  case m of
    MethodBoth -> (kalPred, lstmPred)
    MethodKalmanOnly -> (kalPred, kalPred)
    MethodLstmOnly -> (lstmPred, lstmPred)

type LstmCtx = (NormState, [Double], LSTMModel)

type KalmanCtx = (PredictorBundle, Kalman1, HMMFilter, SensorVar)

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

  (prices, mBinanceEnv) <- loadPrices args
  if length prices < 2 then error "Need at least 2 price rows" else pure ()

  let lookback = argLookback args
  if argTradeOnly args
    then runTradeOnly args lookback prices mBinanceEnv
    else runBacktestPipeline args lookback prices mBinanceEnv

runTradeOnly :: Args -> Int -> [Double] -> Maybe BinanceEnv -> IO ()
runTradeOnly args lookback prices mBinanceEnv = do
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

  printLatestSignal args lookback pricesV mLstmCtx mKalmanCtx mBinanceEnv

runBacktestPipeline :: Args -> Int -> [Double] -> Maybe BinanceEnv -> IO ()
runBacktestPipeline args lookback prices mBinanceEnv = do
  let n = length prices
      backtestRatio = argBacktestRatio args
      trainEndRaw = floor (fromIntegral n * (1 - backtestRatio))
      minTrainEnd = lookback + 1
      maxTrainEnd = n - 2
      trainEnd = min maxTrainEnd (max minTrainEnd trainEndRaw)

  if backtestRatio <= 0 || backtestRatio >= 1
    then error "--backtest-ratio must be between 0 and 1"
    else pure ()
  if n < lookback + 3
    then error (printf "Not enough data for train/backtest split with lookback=%d (need >= %d prices, got %d). Reduce --lookback-bars/--lookback-window or increase --bars." lookback (lookback + 3) n)
    else pure ()
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

  let trainPrices = take trainEnd prices
      backtestPrices = drop trainEnd prices
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
          then optimizeOperations args backtestPrices kalPredPrice lstmPredPrice
          else if argSweepThreshold args
            then
              let (thr, bt) = sweepThreshold args backtestPrices kalPredPrice lstmPredPrice
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

  putStrLn (printf "\nSplit: train=%d backtest=%d (backtest-ratio=%.3f)" (length trainPrices) (length backtestPrices) backtestRatio)
  if argOptimizeOperations args
    then
      putStrLn
        ( printf
            "Optimized operations: method=%s threshold=%.6f (%.3f%%)"
            (methodCode methodUsed)
            bestThreshold
            (bestThreshold * 100)
        )
    else if argSweepThreshold args
      then putStrLn (printf "Best threshold (by final equity): %.6f (%.3f%%)" bestThreshold (bestThreshold * 100))
      else pure ()
  putStrLn $
    case methodUsed of
      MethodBoth -> "Backtest (Kalman fusion + LSTM direction-agreement gated) complete."
      MethodKalmanOnly -> "Backtest (Kalman fusion only) complete."
      MethodLstmOnly -> "Backtest (LSTM only) complete."
  case mHistory of
    Nothing -> pure ()
    Just history -> printLstmSummary history
  printMetrics methodUsed metrics

  printLatestSignal argsForSignal lookback pricesV mLstmCtx mKalmanCtx mBinanceEnv

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

bestFinalEquity :: BacktestResult -> Double
bestFinalEquity br =
  case reverse (brEquityCurve br) of
    (x:_) -> x
    [] -> 1.0

optimizeOperations :: Args -> [Double] -> [Double] -> [Double] -> (Method, Double, BacktestResult)
optimizeOperations args prices kalPred lstmPred =
  let eps = 1e-12
      methodRank m =
        case m of
          MethodBoth -> 2 :: Int
          MethodKalmanOnly -> 1
          MethodLstmOnly -> 0
      eval m =
        let (thr, bt) = sweepThreshold (args { argMethod = m }) prices kalPred lstmPred
            eq = bestFinalEquity bt
         in (eq, m, thr, bt)
      candidates = map eval [MethodBoth, MethodKalmanOnly, MethodLstmOnly]
      pick (bestEq, bestM, bestThr, bestBt) (eq, m, thr, bt) =
        if eq > bestEq + eps
          then (eq, m, thr, bt)
          else if abs (eq - bestEq) <= eps
            then
              let r = methodRank m
                  bestR = methodRank bestM
               in if r > bestR || (r == bestR && thr > bestThr)
                    then (eq, m, thr, bt)
                    else (bestEq, bestM, bestThr, bestBt)
            else (bestEq, bestM, bestThr, bestBt)
   in case candidates of
        [] -> (argMethod args, argTradeThreshold args, simulateEnsembleLongFlat (EnsembleConfig (argTradeThreshold args) (argFee args)) 1 prices kalPred lstmPred)
        (c:cs) ->
          let (_, bestM, bestThr, bestBt) = foldl' pick c cs
           in (bestM, bestThr, bestBt)

sweepThreshold :: Args -> [Double] -> [Double] -> [Double] -> (Double, BacktestResult)
sweepThreshold args prices kalPred lstmPred =
  let n = length prices
      stepCount = n - 1
      eps = 1e-12
      method = argMethod args
      (kalUsed, lstmUsed) = selectPredictions method kalPred lstmPred
      predSources =
        case method of
          MethodBoth -> [kalPred, lstmPred]
          MethodKalmanOnly -> [kalPred]
          MethodLstmOnly -> [lstmPred]

      mags =
        [ v
        | t <- [0 .. stepCount - 1]
        , let prev = prices !! t
        , prev /= 0
        , preds <- predSources
        , let pred = preds !! t
        , let v = abs (pred / prev - 1)
        , not (isNaN v)
        , not (isInfinite v)
        ]

      candidates = 0 : map (\v -> max 0 (v - eps)) mags

      eval thr =
        let cfg =
              EnsembleConfig
                { ecTradeThreshold = thr
                , ecFee = argFee args
                }
            bt = simulateEnsembleLongFlat cfg 1 prices kalUsed lstmUsed
         in (bestFinalEquity bt, thr, bt)

      (baseEq, baseThr, baseBt) = eval (max 0 (argTradeThreshold args))
      eqEps = 1e-12
      pick (bestEq, bestThr, bestBt) thr =
        let (eq, thr', bt) = eval thr
         in if eq > bestEq + eqEps || (abs (eq - bestEq) <= eqEps && thr' > bestThr)
              then (eq, thr', bt)
              else (bestEq, bestThr, bestBt)

      (_, bestThr, bestBt) = foldl' pick (baseEq, baseThr, baseBt) candidates
   in (bestThr, bestBt)

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

printLatestSignal
  :: Args
  -> Int
  -> V.Vector Double
  -> Maybe LstmCtx
  -> Maybe KalmanCtx
  -> Maybe BinanceEnv
  -> IO ()
printLatestSignal args lookback pricesV mLstmCtx mKalmanCtx mEnv = do
  let method = argMethod args
  case method of
    MethodBoth ->
      case (mKalmanCtx, mLstmCtx) of
        (Just _, Just _) -> pure ()
        _ -> error "Internal: --method 11 requires both Kalman and LSTM contexts."
    MethodKalmanOnly ->
      case mKalmanCtx of
        Just _ -> pure ()
        Nothing -> error "Internal: --method 10 requires Kalman context."
    MethodLstmOnly ->
      case mLstmCtx of
        Just _ -> pure ()
        Nothing -> error "Internal: --method 01 requires LSTM context."

  let n = V.length pricesV
      t = n - 1
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
            let window = take lookback (drop (t - lookback + 1) obsAll)
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
      action :: String
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

      showDir :: Maybe Int -> String
      showDir d =
        case d of
          Just 1 -> "UP"
          Just (-1) -> "DOWN"
          _ -> "NEUTRAL"

  putStrLn ""
  putStrLn "**Latest Signal**"
  putStrLn (printf "Method: %s" (methodCode method))
  case mKalNext of
    Nothing -> putStrLn "Kalman next: (disabled)"
    Just kalNext -> putStrLn (printf "Kalman next: %.4f (%s)" kalNext (showDir kalDir))
  case mLstmNext of
    Nothing -> putStrLn "LSTM next:   (disabled)"
    Just lstmNext -> putStrLn (printf "LSTM next:   %.4f (%s)" lstmNext (showDir lstmDir))
  putStrLn (printf "Direction threshold: %.3f%%" (thr * 100))
  putStrLn (printf "Action: %s" action)

  case (argBinanceSymbol args, mEnv) of
    (Just sym, Just env)
      | argBinanceTrade args -> do
          case (beApiKey env, beApiSecret env) of
            (Nothing, _) -> putStrLn "No order: missing Binance API key."
            (_, Nothing) -> putStrLn "No order: missing Binance API secret."
            (Just _, Just _) -> do
              let (baseAsset, _) = splitSymbol sym
              let mode = if argBinanceLive args then OrderLive else OrderTest
              case beMarket env of
                MarketSpot -> do
                  baseBal <- fetchFreeBalance env baseAsset
                  case chosenDir of
                    Just 1 ->
                      if baseBal <= 0
                        then do
                          let qty = argOrderQuantity args
                              qq = argOrderQuote args
                          case (qty, qq) of
                            (Nothing, Nothing) -> putStrLn "No order: provide --order-quantity or --order-quote."
                            _ -> do
                              resp <- placeMarketOrder env mode sym Buy qty qq Nothing
                              putStrLn ("Order sent (" ++ show mode ++ "): BUY " ++ sym ++ " response=" ++ shortResp resp)
                        else putStrLn "No order: already long."
                    Just (-1) ->
                      if baseBal > 0
                        then do
                          let qty = Just (maybe baseBal id (argOrderQuantity args))
                          resp <- placeMarketOrder env mode sym Sell qty Nothing Nothing
                          putStrLn ("Order sent (" ++ show mode ++ "): SELL " ++ sym ++ " response=" ++ shortResp resp)
                        else putStrLn "No order: already flat."
                    _ ->
                      case method of
                        MethodBoth -> putStrLn "No order: directions disagree or neutral (direction gate)."
                        MethodKalmanOnly -> putStrLn "No order: Kalman neutral (within threshold)."
                        MethodLstmOnly -> putStrLn "No order: LSTM neutral (within threshold)."
                MarketMargin -> do
                  if mode == OrderTest
                    then putStrLn "No order: margin trading requires --binance-live (no test endpoint)."
                    else do
                      baseBal <- fetchFreeBalance env baseAsset
                      case chosenDir of
                        Just 1 ->
                          if baseBal <= 0
                            then do
                              let qty = argOrderQuantity args
                                  qq = argOrderQuote args
                              case (qty, qq) of
                                (Nothing, Nothing) -> putStrLn "No order: provide --order-quantity or --order-quote."
                                _ -> do
                                  resp <- placeMarketOrder env mode sym Buy qty qq Nothing
                                  putStrLn ("Order sent (" ++ show mode ++ "): BUY " ++ sym ++ " response=" ++ shortResp resp)
                            else putStrLn "No order: already long."
                        Just (-1) ->
                          if baseBal > 0
                            then do
                              let qty = Just (maybe baseBal id (argOrderQuantity args))
                              resp <- placeMarketOrder env mode sym Sell qty Nothing Nothing
                              putStrLn ("Order sent (" ++ show mode ++ "): SELL " ++ sym ++ " response=" ++ shortResp resp)
                            else putStrLn "No order: already flat."
                        _ ->
                          case method of
                            MethodBoth -> putStrLn "No order: directions disagree or neutral (direction gate)."
                            MethodKalmanOnly -> putStrLn "No order: Kalman neutral (within threshold)."
                            MethodLstmOnly -> putStrLn "No order: LSTM neutral (within threshold)."
                MarketFutures -> do
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
                        putStrLn ("Order sent (" ++ show mode ++ "): " ++ show side ++ " " ++ sym ++ " qty=" ++ show qty ++ " response=" ++ shortResp resp)
                  case chosenDir of
                    Just 1 ->
                      if posAmt > 0
                        then putStrLn "No order: already long."
                        else
                          case mDesiredQty of
                            Nothing -> putStrLn "No order: futures requires --order-quantity or --order-quote."
                            Just q -> do
                              let qtyToBuy = if posAmt < 0 then abs posAmt + q else q
                              resp <- placeMarketOrder env mode sym Buy (Just qtyToBuy) Nothing Nothing
                              putStrLn ("Order sent (" ++ show mode ++ "): BUY " ++ sym ++ " qty=" ++ show qtyToBuy ++ " response=" ++ shortResp resp)
                    Just (-1) ->
                      if posAmt == 0
                        then putStrLn "No order: already flat."
                        else if posAmt > 0
                          then closeOrder Sell (abs posAmt)
                          else closeOrder Buy (abs posAmt)
                    _ ->
                      case method of
                        MethodBoth -> putStrLn "No order: directions disagree or neutral (direction gate)."
                        MethodKalmanOnly -> putStrLn "No order: Kalman neutral (within threshold)."
                        MethodLstmOnly -> putStrLn "No order: LSTM neutral (within threshold)."
      | otherwise -> pure ()
    _ -> pure ()

shortResp :: BL.ByteString -> String
shortResp bs =
  let s = BS.unpack (BL.toStrict bs)
   in take 200 s

{-# LANGUAGE OverloadedStrings #-}
module Main where

import Control.Exception (SomeException, evaluate, try)
import Data.Aeson (eitherDecode)
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.List (isInfixOf)
import qualified Data.Vector as V
import System.Exit (exitFailure, exitSuccess)

import Trader.Binance
  ( BinanceMarket(..)
  , BinanceOrderMode(..)
  , OrderSide(..)
  , Kline(..)
  , binanceBaseUrl
  , newBinanceEnv
  , placeMarketOrder
  , signQuery
  )
import Trader.Duration (lookbackBarsFrom)
import Trader.KalmanFusion (Kalman1(..), initKalman1, updateMulti)
import Trader.Kalman3 (Vec3(..), Kalman3(..), constantAcceleration1D, step, forecastNextConstantAcceleration1D, runConstantAcceleration1D, KalmanRun(..))
import Trader.LSTM (LSTMConfig(..), LSTMModel(..), trainLSTM, buildSequences, evaluateLoss)
import Trader.Method (Method(..), parseMethod, selectPredictions)
import Trader.Metrics (computeMetrics, bmMaxDrawdown, bmTotalReturn)
import Trader.Optimization (bestFinalEquity, optimizeOperations, sweepThreshold)
import Trader.Predictors
  ( SensorId(..)
  , SensorOutput(..)
  , RegimeProbs(..)
  , Quantiles(..)
  , Interval(..)
  , trainPredictors
  , initHMMFilter
  , predictSensors
  )
import Trader.Trading (BacktestResult(..), EnsembleConfig(..), IntrabarFill(..), Positioning(..), Trade(..), simulateEnsembleLongFlat)
import Trader.Split (Split(..), splitTrainBacktest)

main :: IO ()
main = do
  results <- sequence
    [ run "duration lookback bars" testLookbackBars
    , run "kalman fusion multi-sensor" testKalmanFusionMulti
    , run "predictors output shape" testPredictorsOutputs
    , run "kalman constant series" testKalmanConstant
    , run "kalman forecast constant" testKalmanForecast
    , run "kalman innovation sign" testKalmanInnovationSign
    , run "forward return sign" testForwardReturnSign
    , run "lstm training improves loss" testLstmImprovesLoss
    , run "ensemble agreement gate" testAgreementGate
    , run "long-short down move" testLongShortDownMove
    , run "liquidation clamps equity" testLiquidationClamp
    , run "metrics max drawdown" testMetricsMaxDrawdown
    , run "binance signature length" testBinanceSignatureLength
    , run "binance kline json parsing" testBinanceKlineParsing
    , run "method parsing" testMethodParsing
    , run "method selects predictions" testMethodSelection
    , run "train/backtest split" testTrainBacktestSplit
    , run "threshold sweep" testSweepThreshold
    , run "operations optimization" testOptimizeOperations
    , run "binance order validation" testBinanceOrderValidation
    ]
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

testLookbackBars :: IO ()
testLookbackBars =
  case lookbackBarsFrom "5m" "24h" of
    Left e -> error e
    Right n -> assert "expected 288 bars" (n == 288)

testKalmanFusionMulti :: IO ()
testKalmanFusionMulti = do
  let k0 = initKalman1 0 1 0
      k1 = updateMulti [(0.01, 1e-4), (0.02, 1e-2)] k0
      expected = (0 * 1 + 0.01 * 10000 + 0.02 * 100) / (1 + 10000 + 100)
  assertApprox "posterior mean" 1e-6 (kMean k1) expected
  assert "posterior variance shrinks" (kVar k1 < 1)

testPredictorsOutputs :: IO ()
testPredictorsOutputs = do
  let prices = take 300 (iterate (* 1.001) 100.0)
      lookback = 20
      trainPrices = take 250 prices
      pb = trainPredictors lookback (V.fromList trainPrices)
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

testKalmanConstant :: IO ()
testKalmanConstant = do
  let xs = replicate 50 10.0
      KalmanRun preds filt = runConstantAcceleration1D 1.0 1e-6 1e-6 xs
  assert "pred length" (length preds == length xs - 1)
  assert "filt length" (length filt == length xs)
  assertApprox "filtered near constant" 1e-2 (last filt) 10.0

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
      cfg = baseCfg { lcEpochs = 10 }
      (m1, _) = trainLSTM cfg series
      l0 = evaluateLoss lookback hidden dataset (lmParams m0)
      l1 = evaluateLoss lookback hidden dataset (lmParams m1)
  assert ("loss did not decrease: " ++ show (l0, l1)) (l1 < l0)

testAgreementGate :: IO ()
testAgreementGate = do
  let prices = [100, 101, 102, 103]
      lookback = 2
      kalPred = [101, 110, 120]  -- length 3
      lstmPred = [110, 100]      -- length 2, for t=1..2
      cfg =
        EnsembleConfig
          { ecOpenThreshold = 0.0
          , ecCloseThreshold = 0.0
          , ecFee = 0.0
          , ecSlippage = 0.0
          , ecSpread = 0.0
          , ecStopLoss = Nothing
          , ecTakeProfit = Nothing
          , ecTrailingStop = Nothing
          , ecPositioning = LongFlat
          , ecIntrabarFill = StopFirst
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
      res = simulateEnsembleLongFlat cfg lookback prices kalPred lstmPred Nothing
  assert "expected two position changes (enter + exit)" (brPositionChanges res == 2)

testLongShortDownMove :: IO ()
testLongShortDownMove = do
  let prices = [100, 90]
      lookback = 1
      kalPred = [90]
      lstmPred = [90]
      baseCfg =
        EnsembleConfig
          { ecOpenThreshold = 0.0
          , ecCloseThreshold = 0.0
          , ecFee = 0.0
          , ecSlippage = 0.0
          , ecSpread = 0.0
          , ecStopLoss = Nothing
          , ecTakeProfit = Nothing
          , ecTrailingStop = Nothing
          , ecPositioning = LongFlat
          , ecIntrabarFill = StopFirst
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
      btFlat = simulateEnsembleLongFlat baseCfg lookback prices kalPred lstmPred Nothing
      btShort = simulateEnsembleLongFlat (baseCfg { ecPositioning = LongShort }) lookback prices kalPred lstmPred Nothing

  assertApprox "flat final equity" 1e-12 (last (brEquityCurve btFlat)) 1.0
  assertApprox "short final equity" 1e-12 (last (brEquityCurve btShort)) 1.1
  assert "short position opened" (brPositions btShort == [-1])

testLiquidationClamp :: IO ()
testLiquidationClamp = do
  let prices = [100, 250]
      lookback = 1
      kalPred = [50]
      lstmPred = [50]
      cfg =
        EnsembleConfig
          { ecOpenThreshold = 0.0
          , ecCloseThreshold = 0.0
          , ecFee = 0.0
          , ecSlippage = 0.0
          , ecSpread = 0.0
          , ecStopLoss = Nothing
          , ecTakeProfit = Nothing
          , ecTrailingStop = Nothing
          , ecPositioning = LongShort
          , ecIntrabarFill = StopFirst
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
      bt = simulateEnsembleLongFlat cfg lookback prices kalPred lstmPred Nothing
      finalEq = last (brEquityCurve bt)
      trades = brTrades bt
  assertApprox "equity clamped at 0" 1e-12 finalEq 0.0
  assert "positions cleared after liquidation" (brPositions bt == [-1])
  assert "liquidation trade recorded" (case trades of { [t] -> trExitReason t == Just "LIQUIDATION"; _ -> False })

testMetricsMaxDrawdown :: IO ()
testMetricsMaxDrawdown = do
  let br =
        BacktestResult
          { brEquityCurve = [1.0, 1.1, 1.0]
          , brPositions = [1.0, 0.0]
          , brAgreementOk = [True, True]
          , brPositionChanges = 2
          , brTrades = []
          }
      m = computeMetrics 365 br
  assertApprox "total return" 1e-12 (bmTotalReturn m) 0.0
  assertApprox "max drawdown" 1e-6 (bmMaxDrawdown m) (0.1 / 1.1)

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
      assertApprox "close parse" 1e-12 (kClose (head ks)) 123.45

testMethodParsing :: IO ()
testMethodParsing = do
  assert "parse 11" (parseMethod "11" == Right MethodBoth)
  assert "parse both" (parseMethod "both" == Right MethodBoth)
  assert "parse agreement" (parseMethod "agreement" == Right MethodBoth)
  assert "parse 10" (parseMethod "10" == Right MethodKalmanOnly)
  assert "parse kalman" (parseMethod "kalman" == Right MethodKalmanOnly)
  assert "parse Kalman-Only" (parseMethod "Kalman-Only" == Right MethodKalmanOnly)
  assert "parse 01" (parseMethod "01" == Right MethodLstmOnly)
  assert "parse lstm" (parseMethod "lstm" == Right MethodLstmOnly)
  assert "parse LSTM_ONLY" (parseMethod "LSTM_ONLY" == Right MethodLstmOnly)
  case parseMethod "00" of
    Left _ -> pure ()
    Right _ -> error "expected parse failure"

testMethodSelection :: IO ()
testMethodSelection = do
  let kal = [1.0, 2.0]
      lstm = [10.0, 20.0]
  assert "both keeps both" (selectPredictions MethodBoth kal lstm == (kal, lstm))
  assert "kalman-only duplicates kalman" (selectPredictions MethodKalmanOnly kal lstm == (kal, kal))
  assert "lstm-only duplicates lstm" (selectPredictions MethodLstmOnly kal lstm == (lstm, lstm))

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
    Left e -> error e
    Right s -> do
      assert "adjusted for lookback" (splitTrainEndRaw s == 6 && splitTrainEnd s == 51)
      assert "train size2" (length (splitTrain s) == 51)
      assert "backtest size2" (length (splitBacktest s) == 9)

testSweepThreshold :: IO ()
testSweepThreshold = do
  let prices = [100, 110]
      kalPred = [110]
      lstmPred = [110]
      cfg =
        EnsembleConfig
          { ecOpenThreshold = 0.0
          , ecCloseThreshold = 0.0
          , ecFee = 0.0
          , ecSlippage = 0.0
          , ecSpread = 0.0
          , ecStopLoss = Nothing
          , ecTakeProfit = Nothing
          , ecTrailingStop = Nothing
          , ecPositioning = LongFlat
          , ecIntrabarFill = StopFirst
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
      (openThr, closeThr, bt) = sweepThreshold MethodKalmanOnly cfg prices kalPred lstmPred Nothing
  assert "open thr close to 10%" (openThr > 0.099999 && openThr < 0.1)
  assert "close thr close to 10%" (closeThr > 0.099999 && closeThr < 0.1)
  assertApprox "final equity" 1e-12 (bestFinalEquity bt) 1.1

testOptimizeOperations :: IO ()
testOptimizeOperations = do
  let prices = [100, 110]
      kalPred = [110]
      lstmPred = [90]
      cfg =
        EnsembleConfig
          { ecOpenThreshold = 0.0
          , ecCloseThreshold = 0.0
          , ecFee = 0.0
          , ecSlippage = 0.0
          , ecSpread = 0.0
          , ecStopLoss = Nothing
          , ecTakeProfit = Nothing
          , ecTrailingStop = Nothing
          , ecPositioning = LongFlat
          , ecIntrabarFill = StopFirst
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
      (m, openThr, closeThr, bt) = optimizeOperations cfg prices kalPred lstmPred Nothing
  assert "picked kalman-only" (m == MethodKalmanOnly)
  assert "open thr close to 10%" (openThr > 0.099999 && openThr < 0.1)
  assert "close thr close to 10%" (closeThr > 0.099999 && closeThr < 0.1)
  assertApprox "final equity" 1e-12 (bestFinalEquity bt) 1.1

assertThrowsContains :: String -> (() -> IO a) -> IO ()
assertThrowsContains needle mkAction = do
  r <- (try (mkAction () >> pure ()) :: IO (Either SomeException ()))
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

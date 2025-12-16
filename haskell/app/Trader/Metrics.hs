module Trader.Metrics
  ( BacktestMetrics(..)
  , computeMetrics
  ) where

import Data.List (foldl')

import Trader.Trading (BacktestResult(..), Trade(..))

data BacktestMetrics = BacktestMetrics
  { bmPeriods :: !Int
  , bmFinalEquity :: !Double
  , bmTotalReturn :: !Double
  , bmAnnualizedReturn :: !Double
  , bmAnnualizedVolatility :: !Double
  , bmSharpe :: !Double
  , bmMaxDrawdown :: !Double
  , bmTradeCount :: !Int
  , bmRoundTrips :: !Int
  , bmWinRate :: !Double
  , bmGrossProfit :: !Double
  , bmGrossLoss :: !Double
  , bmProfitFactor :: !(Maybe Double)
  , bmAvgTradeReturn :: !Double
  , bmAvgHoldingPeriods :: !Double
  , bmExposure :: !Double
  , bmAgreementRate :: !Double
  , bmTurnover :: !Double
  } deriving (Eq, Show)

computeMetrics :: Double -> BacktestResult -> BacktestMetrics
computeMetrics periodsPerYear br =
  let eq = brEquityCurve br
      periods = max 0 (length eq - 1)
      finalEq =
        case reverse eq of
          (x:_) -> x
          [] -> 1.0
      totalRet = finalEq - 1
      rets = returnsFromEquity eq
      meanR = mean rets
      stdR = stddev rets
      annVol = stdR * sqrt periodsPerYear
      sharpe =
        if stdR <= 0
          then 0
          else (meanR / stdR) * sqrt periodsPerYear
      annRet =
        if periods <= 0
          then 0
          else finalEq ** (periodsPerYear / fromIntegral periods) - 1
      maxDd = abs (min 0 (minDrawdown eq))

      trades = brTrades br
      tradeReturns = map trReturn trades
      wins = length (filter (> 0) tradeReturns)
      winRate = if null tradeReturns then 0 else fromIntegral wins / fromIntegral (length tradeReturns)
      grossProfits = sum (filter (> 0) tradeReturns)
      grossLosses = abs (sum (filter (< 0) tradeReturns))
      profitFactor =
        if grossLosses > 0
          then Just (grossProfits / grossLosses)
          else if grossProfits > 0
            then Nothing
            else Just 0
      avgTrade = if null tradeReturns then 0 else mean tradeReturns
      holding = map trHoldingPeriods trades
      avgHold = if null holding then 0 else fromIntegral (sum holding) / fromIntegral (length holding)

      exposure =
        let pos = brPositions br
         in if null pos then 0 else sum (map abs pos) / fromIntegral (length pos)

      agree =
        let flags = brAgreementOk br
            total = length flags
         in if total == 0 then 0 else fromIntegral (length (filter id flags)) / fromIntegral total

      turnover = if periods == 0 then 0 else fromIntegral (brPositionChanges br) / fromIntegral periods
      roundTrips = length (filter (\t -> trEntryIndex t < trExitIndex t) trades)
   in BacktestMetrics
        { bmPeriods = periods
        , bmFinalEquity = finalEq
        , bmTotalReturn = totalRet
        , bmAnnualizedReturn = annRet
        , bmAnnualizedVolatility = annVol
        , bmSharpe = sharpe
        , bmMaxDrawdown = maxDd
        , bmTradeCount = brPositionChanges br
        , bmRoundTrips = roundTrips
        , bmWinRate = winRate
        , bmGrossProfit = grossProfits
        , bmGrossLoss = grossLosses
        , bmProfitFactor = profitFactor
        , bmAvgTradeReturn = avgTrade
        , bmAvgHoldingPeriods = avgHold
        , bmExposure = exposure
        , bmAgreementRate = agree
        , bmTurnover = turnover
        }

returnsFromEquity :: [Double] -> [Double]
returnsFromEquity eq =
  case eq of
    [] -> []
    [_] -> []
    _ -> zipWith (\a b -> b / a - 1) eq (tail eq)

mean :: [Double] -> Double
mean xs =
  if null xs
    then 0
    else sum xs / fromIntegral (length xs)

stddev :: [Double] -> Double
stddev xs =
  case xs of
    [] -> 0
    [_] -> 0
    _ ->
      let m = mean xs
          var = sum (map (\x -> (x - m) ** 2) xs) / fromIntegral (length xs - 1)
       in sqrt var

minDrawdown :: [Double] -> Double
minDrawdown eq =
  let step (peak, minDd) e =
        let peak' = max peak e
            dd = if peak' <= 0 then 0 else (e - peak') / peak'
         in (peak', min minDd dd)
      (_, ddMin) = foldl' step (0, 0) eq
   in ddMin

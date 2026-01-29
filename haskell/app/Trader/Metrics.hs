module Trader.Metrics (
    BacktestMetrics (..),
    computeMetrics,
) where

import Data.List (foldl')

import Trader.Trading (BacktestResult (..), ExitReason (..), Trade (..))

data BacktestMetrics = BacktestMetrics
    { bmPeriods :: !Int
    , bmFinalEquity :: !Double
    , bmTotalReturn :: !Double
    , bmAnnualizedReturn :: !Double
    , bmAnnualizedVolatility :: !Double
    , bmSharpe :: !Double
    , bmMaxDrawdown :: !Double
    , bmPositionChanges :: !Int
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
    }
    deriving (Eq, Show)

computeMetrics :: Double -> BacktestResult -> BacktestMetrics
computeMetrics periodsPerYear br =
    let eq = brEquityCurve br
        periods = max 0 (length eq - 1)
        finalEq =
            case eq of
                [] -> 1.0
                xs ->
                    let v = last xs
                     in if isNaN v || isInfinite v || v < 0 then 0 else v
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
        tradeCount = length trades
        isRoundTrip t =
            case trExitReason t of
                Just ExitEod -> False
                _ -> True
        (wins, sumReturns, grossProfits, grossLossSum, totalHold, roundTrips) =
            foldl'
                ( \(w, rSum, gp, gl, hold, rt) t ->
                    let r = trReturn t
                        pnl = trExitEquity t - trEntryEquity t
                        w' = if r > 0 then w + 1 else w
                        gp' = if pnl > 0 then gp + pnl else gp
                        gl' = if pnl < 0 then gl + pnl else gl
                        hold' = hold + trHoldingPeriods t
                        rt' = if isRoundTrip t then rt + 1 else rt
                     in (w', rSum + r, gp', gl', hold', rt')
                )
                (0, 0, 0, 0, 0, 0)
                trades
        winRate = if tradeCount == 0 then 0 else fromIntegral wins / fromIntegral tradeCount
        grossLosses = abs grossLossSum
        profitFactor =
            if grossLosses > 0
                then Just (grossProfits / grossLosses)
                else
                    if grossProfits > 0
                        then Nothing
                        else Just 0
        avgTrade = if tradeCount == 0 then 0 else sumReturns / fromIntegral tradeCount
        avgHold = if tradeCount == 0 then 0 else fromIntegral totalHold / fromIntegral tradeCount

        exposure =
            let pos = brPositions br
             in if null pos then 0 else foldl' (\acc v -> acc + abs v) 0 pos / fromIntegral (length pos)

        agree =
            let flags = brAgreementOk br
                valids = brAgreementValid br
                step (accOk, accTotal) (ok, valid) =
                    if valid
                        then (accOk + if ok then 1 else 0, accTotal + 1)
                        else (accOk, accTotal)
                (agrees, total) = foldl' step (0, 0) (zip flags valids)
             in if total == 0 then 0 else fromIntegral agrees / fromIntegral total

        positionChanges = brPositionChanges br
        turnover = if periods == 0 then 0 else fromIntegral positionChanges / fromIntegral periods
     in BacktestMetrics
            { bmPeriods = periods
            , bmFinalEquity = finalEq
            , bmTotalReturn = totalRet
            , bmAnnualizedReturn = annRet
            , bmAnnualizedVolatility = annVol
            , bmSharpe = sharpe
            , bmMaxDrawdown = maxDd
            , bmPositionChanges = positionChanges
            , bmTradeCount = tradeCount
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
    case eq' of
        [] -> []
        [_] -> []
        _ -> zipWith ret eq' (tail eq')
  where
    bad x = isNaN x || isInfinite x
    clamp x =
        if bad x || x < 0
            then 0
            else x
    eq' = map clamp eq
    ret a b =
        if a <= 0
            then 0
            else b / a - 1

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

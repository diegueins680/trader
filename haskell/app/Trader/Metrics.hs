module Trader.Metrics (
    BacktestMetrics (..),
    computeMetrics,
) where

import Data.List (foldl', sort)

import Trader.Trading (BacktestResult (..), ExitReason (..), Trade (..))

data BacktestMetrics = BacktestMetrics
    { bmPeriods :: !Int
    , bmFinalEquity :: !Double
    , bmTotalReturn :: !Double
    , bmAnnualizedReturn :: !Double
    , bmAnnualizedVolatility :: !Double
    , bmSharpe :: !Double
    , bmSortino :: !Double
    , bmCalmar :: !Double
    , bmDownsideVolatility :: !Double
    , bmVaR95 :: !Double
    , bmCVaR95 :: !Double
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
        finalEq = sanitizeEquity (lastOr 1.0 eq)
        totalRet = finalEq - 1
        rets = returnsFromEquity eq
        meanR = mean rets
        stdR = stddev rets
        annVol = stdR * sqrt periodsPerYear
        sharpe =
            if stdR <= 0
                then 0
                else (meanR / stdR) * sqrt periodsPerYear
        downsideDev = downsideDeviation rets
        downsideVol = downsideDev * sqrt periodsPerYear
        sortino =
            if downsideDev <= 0
                then 0
                else (meanR / downsideDev) * sqrt periodsPerYear
        annRet =
            if periods <= 0
                then 0
                else sanitizeFinite 0 (finalEq ** (periodsPerYear / fromIntegral periods) - 1)
        maxDd = abs (min 0 (minDrawdown eq))
        calmar =
            if maxDd <= 0
                then annRet
                else annRet / maxDd
        (var95, cvar95) = varCvar 0.95 rets

        trades = brTrades br
        tradeCount = length trades
        isRoundTrip t =
            case trExitReason t of
                Just ExitEod -> False
                _ -> True
        (wins, sumReturns, grossProfits, grossLossSum, totalHold, roundTrips) =
            foldl'
                ( \(w, rSum, gp, gl, hold, rt) t ->
                    let r = sanitizeFinite 0 (trReturn t)
                        pnl = sanitizeFinite 0 (trExitEquity t - trEntryEquity t)
                        holdPeriods = max 0 (trHoldingPeriods t)
                        w' = if r > 0 then w + 1 else w
                        gp' = if pnl > 0 then gp + pnl else gp
                        gl' = if pnl < 0 then gl + pnl else gl
                        hold' = hold + holdPeriods
                        rt' = if isRoundTrip t then rt + 1 else rt
                     in (w', rSum + r, gp', gl', hold', rt')
                )
                (0, 0, 0, 0, 0, 0)
                trades
        winRate = if tradeCount == 0 then 0 else fromIntegral wins / fromIntegral tradeCount
        grossLosses = abs grossLossSum
        profitFactor
            | grossLosses > 0 = Just (grossProfits / grossLosses)
            | grossProfits > 0 = Nothing
            | otherwise = Just 0
        avgTrade = if tradeCount == 0 then 0 else sumReturns / fromIntegral tradeCount
        avgHold = if tradeCount == 0 then 0 else fromIntegral totalHold / fromIntegral tradeCount

        exposure =
            let pos = brPositions br
                (sumAbs, count) =
                    foldl'
                        ( \(acc, n) v ->
                            let pos = sanitizeFinite 0 v
                             in (acc + abs pos, n + 1 :: Int)
                        )
                        (0, 0)
                        pos
             in if count == 0 then 0 else sumAbs / fromIntegral count

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
            , bmSortino = sortino
            , bmCalmar = calmar
            , bmDownsideVolatility = downsideVol
            , bmVaR95 = var95
            , bmCVaR95 = cvar95
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
    zipWith ret eq' (drop1 eq')
  where
    bad x = isNaN x || isInfinite x
    clamp x =
        if bad x || x < 0
            then 0
            else x
    eq' = map clamp eq
    drop1 xs =
        case xs of
            [] -> []
            _ : ys -> ys
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
             in sqrt (max 0 var)

downsideDeviation :: [Double] -> Double
downsideDeviation xs =
    let downs = map (min 0) xs
        n = length downs
     in if n <= 0
            then 0
            else sqrt (max 0 (sum (map (\r -> r * r) downs) / fromIntegral n))

varCvar :: Double -> [Double] -> (Double, Double)
varCvar level xs =
    let xs' = sort xs
        n = length xs'
        alpha = max 0 (min 1 (1 - level))
     in if n <= 0
            then (0, 0)
            else
                let idx = floor (alpha * fromIntegral (n - 1))
                    safeIdx = max 0 (min (n - 1) idx)
                    (left, right) = splitAt safeIdx xs'
                    q =
                        case right of
                            y : _ -> y
                            [] -> lastOr 0 left
                    tailXs = take (safeIdx + 1) xs'
                    varLoss = max 0 (-q)
                    cvarLoss =
                        if null tailXs
                            then 0
                            else max 0 (negate (mean tailXs))
                 in (varLoss, cvarLoss)

minDrawdown :: [Double] -> Double
minDrawdown eq =
    let step (peak, minDd) e =
            let peak' = max peak e
                dd = if peak' <= 0 then 0 else (e - peak') / peak'
             in (peak', min minDd dd)
        (_, ddMin) = foldl' step (0, 0) eq
     in ddMin

lastOr :: a -> [a] -> a
lastOr = foldl' (\_ x -> x)

sanitizeEquity :: Double -> Double
sanitizeEquity x =
    if isNaN x || isInfinite x || x < 0
        then 0
        else x

sanitizeFinite :: Double -> Double -> Double
sanitizeFinite fallback x =
    if isNaN x || isInfinite x
        then fallback
        else x

module Trader.Optimization
  ( bestFinalEquity
  , optimizeOperations
  , sweepThreshold
  ) where

import Data.List (foldl', group, sort)
import qualified Data.Vector as V

import Trader.Method (Method(..))
import Trader.Trading (BacktestResult(..), EnsembleConfig(..), simulateEnsembleLongFlatV)

bestFinalEquity :: BacktestResult -> Double
bestFinalEquity br =
  case brEquityCurve br of
    [] -> 1.0
    xs -> last xs

optimizeOperations :: EnsembleConfig -> [Double] -> [Double] -> [Double] -> (Method, Double, BacktestResult)
optimizeOperations baseCfg prices kalPred lstmPred =
  let eps = 1e-12
      methodRank m =
        case m of
          MethodBoth -> 2 :: Int
          MethodKalmanOnly -> 1
          MethodLstmOnly -> 0
      eval m =
        let (thr, bt) = sweepThreshold m baseCfg prices kalPred lstmPred
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
      (c : cs) = candidates
      (_, bestM, bestThr, bestBt) = foldl' pick c cs
   in (bestM, bestThr, bestBt)

sweepThreshold :: Method -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> (Double, BacktestResult)
sweepThreshold method baseCfg prices kalPred lstmPred =
  let pricesV = V.fromList prices
      stepCount = V.length pricesV - 1
      eps = 1e-12
      baseThreshold = ecTradeThreshold baseCfg

      kalV = V.fromList kalPred
      lstmV = V.fromList lstmPred

      (kalUsedV, lstmUsedV) =
        case method of
          MethodBoth -> (kalV, lstmV)
          MethodKalmanOnly -> (kalV, kalV)
          MethodLstmOnly -> (lstmV, lstmV)

      predSources =
        case method of
          MethodBoth -> [kalV, lstmV]
          MethodKalmanOnly -> [kalV]
          MethodLstmOnly -> [lstmV]

      () =
        case method of
          MethodBoth ->
            if V.length kalV < stepCount
              then error "kalPred too short for sweepThreshold"
              else if V.length lstmV < stepCount
                then error "lstmPred too short for sweepThreshold"
                else ()
          MethodKalmanOnly ->
            if V.length kalV < stepCount
              then error "kalPred too short for sweepThreshold"
              else ()
          MethodLstmOnly ->
            if V.length lstmV < stepCount
              then error "lstmPred too short for sweepThreshold"
              else ()

      mags =
        [ v
        | t <- [0 .. stepCount - 1]
        , let prev = pricesV V.! t
        , prev /= 0
        , predsV <- predSources
        , let pred = predsV V.! t
        , let v = abs (pred / prev - 1)
        , not (isNaN v)
        , not (isInfinite v)
        ]

      uniqueSorted = map head . group . sort
      candidates = uniqueSorted (0 : map (\v -> max 0 (v - eps)) mags)

      eval thr =
        let cfg = baseCfg { ecTradeThreshold = thr }
            bt = simulateEnsembleLongFlatV cfg 1 pricesV kalUsedV lstmUsedV
         in (bestFinalEquity bt, thr, bt)

      (baseEq, baseThr, baseBt) = eval (max 0 baseThreshold)
      eqEps = 1e-12
      pick (bestEq, bestThr, bestBt) thr =
        let (eq, thr', bt) = eval thr
         in if eq > bestEq + eqEps || (abs (eq - bestEq) <= eqEps && thr' > bestThr)
              then (eq, thr', bt)
              else (bestEq, bestThr, bestBt)

      (_, bestThr, bestBt) = foldl' pick (baseEq, baseThr, baseBt) candidates
   in (bestThr, bestBt)

module Trader.Optimization
  ( bestFinalEquity
  , optimizeOperations
  , sweepThreshold
  ) where

import Data.List (foldl')

import Trader.Method (Method(..), selectPredictions)
import Trader.Trading (BacktestResult(..), EnsembleConfig(..), simulateEnsembleLongFlat)

bestFinalEquity :: BacktestResult -> Double
bestFinalEquity br =
  case reverse (brEquityCurve br) of
    (x : _) -> x
    [] -> 1.0

optimizeOperations :: Double -> Double -> [Double] -> [Double] -> [Double] -> (Method, Double, BacktestResult)
optimizeOperations baseThreshold fee prices kalPred lstmPred =
  let eps = 1e-12
      methodRank m =
        case m of
          MethodBoth -> 2 :: Int
          MethodKalmanOnly -> 1
          MethodLstmOnly -> 0
      eval m =
        let (thr, bt) = sweepThreshold m baseThreshold fee prices kalPred lstmPred
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
        [] ->
          ( MethodBoth
          , max 0 baseThreshold
          , simulateEnsembleLongFlat (EnsembleConfig (max 0 baseThreshold) fee) 1 prices kalPred lstmPred
          )
        (c : cs) ->
          let (_, bestM, bestThr, bestBt) = foldl' pick c cs
           in (bestM, bestThr, bestBt)

sweepThreshold :: Method -> Double -> Double -> [Double] -> [Double] -> [Double] -> (Double, BacktestResult)
sweepThreshold method baseThreshold fee prices kalPred lstmPred =
  let n = length prices
      stepCount = n - 1
      eps = 1e-12
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
                , ecFee = fee
                }
            bt = simulateEnsembleLongFlat cfg 1 prices kalUsed lstmUsed
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

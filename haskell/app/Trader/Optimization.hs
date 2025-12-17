module Trader.Optimization
  ( bestFinalEquity
  , optimizeOperations
  , optimizeOperationsWithHL
  , sweepThreshold
  , sweepThresholdWithHL
  ) where

import Data.List (foldl', group, intercalate, sort)
import qualified Data.Vector as V

import Trader.Method (Method(..))
import Trader.Trading (BacktestResult(..), EnsembleConfig(..), StepMeta(..), simulateEnsembleLongFlatVWithHL)

bestFinalEquity :: BacktestResult -> Double
bestFinalEquity br =
  case brEquityCurve br of
    [] -> 1.0
    xs -> last xs

optimizeOperations :: EnsembleConfig -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Method, Double, Double, BacktestResult)
optimizeOperations baseCfg prices kalPred lstmPred mMeta =
  optimizeOperationsWithHL baseCfg prices prices prices kalPred lstmPred mMeta

optimizeOperationsWithHL :: EnsembleConfig -> [Double] -> [Double] -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Method, Double, Double, BacktestResult)
optimizeOperationsWithHL baseCfg closes highs lows kalPred lstmPred mMeta =
  let eps = 1e-12
      methodRank m =
        case m of
          MethodBoth -> 2 :: Int
          MethodKalmanOnly -> 1
          MethodLstmOnly -> 0
      eval m =
        case sweepThresholdWithHL m baseCfg closes highs lows kalPred lstmPred mMeta of
          Left e -> Left e
          Right (openThr, closeThr, bt) ->
            let eq = bestFinalEquity bt
             in Right (eq, m, openThr, closeThr, bt)
      candidates = [MethodBoth, MethodKalmanOnly, MethodLstmOnly]
      results = map eval candidates
      evaluated = [v | Right v <- results]
      errors = [e | Left e <- results]
      pick (bestEq, bestM, bestOpenThr, bestCloseThr, bestBt) (eq, m, openThr, closeThr, bt) =
        if eq > bestEq + eps
          then (eq, m, openThr, closeThr, bt)
          else if abs (eq - bestEq) <= eps
            then
              let r = methodRank m
                  bestR = methodRank bestM
               in if r > bestR || (r == bestR && (openThr, closeThr) > (bestOpenThr, bestCloseThr))
                    then (eq, m, openThr, closeThr, bt)
                    else (bestEq, bestM, bestOpenThr, bestCloseThr, bestBt)
            else (bestEq, bestM, bestOpenThr, bestCloseThr, bestBt)
   in case evaluated of
        [] ->
          Left
            ( "optimizeOperations: no eligible candidates"
                ++ if null errors
                  then ""
                  else " (" ++ intercalate "; " errors ++ ")"
            )
        c : cs ->
          let (_, bestM, bestOpenThr, bestCloseThr, bestBt) = foldl' pick c cs
           in Right (bestM, bestOpenThr, bestCloseThr, bestBt)

sweepThreshold :: Method -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Double, Double, BacktestResult)
sweepThreshold method baseCfg prices kalPred lstmPred mMeta =
  sweepThresholdWithHL method baseCfg prices prices prices kalPred lstmPred mMeta

sweepThresholdWithHL :: Method -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Double, Double, BacktestResult)
sweepThresholdWithHL method baseCfg closes highs lows kalPred lstmPred mMeta =
  let pricesV = V.fromList closes
      highsV = V.fromList highs
      lowsV = V.fromList lows
      n = V.length pricesV
      stepCount = n - 1
      eps = 1e-12
      baseOpenThreshold = max 0 (ecOpenThreshold baseCfg)
      baseCloseThreshold = max 0 (ecCloseThreshold baseCfg)
      maxCandidates = 60 :: Int

      downsample :: Int -> [Double] -> [Double]
      downsample k xs
        | k <= 0 = []
        | otherwise =
            let v = V.fromList xs
                n = V.length v
             in if n <= k
                  then xs
                  else
                    let denom = max 1 (k - 1)
                        pick i = (i * (n - 1)) `div` denom
                     in [v V.! pick i | i <- [0 .. k - 1]]

      kalV = V.fromList kalPred
      lstmV = V.fromList lstmPred

      metaV = V.fromList <$> mMeta
      metaUsed =
        case method of
          MethodLstmOnly -> Nothing
          _ -> metaV

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
      candidates0 = uniqueSorted (0 : map (\v -> max 0 (v - eps)) mags)
      candidates =
        uniqueSorted
          ( baseOpenThreshold
              : baseCloseThreshold
              : downsample maxCandidates candidates0
          )

      eval openThr closeThr =
        let cfg = baseCfg { ecOpenThreshold = openThr, ecCloseThreshold = closeThr }
            bt = simulateEnsembleLongFlatVWithHL cfg 1 pricesV highsV lowsV kalUsedV lstmUsedV metaUsed
         in (bestFinalEquity bt, openThr, closeThr, bt)

      (baseEq, baseOpenThr, baseCloseThr, baseBt) = eval baseOpenThreshold baseCloseThreshold
      eqEps = 1e-12
      pick (bestEq, bestOpenThr, bestCloseThr, bestBt) (openThr, closeThr) =
        let (eq, openThr', closeThr', bt) = eval openThr closeThr
         in
          if eq > bestEq + eqEps || (abs (eq - bestEq) <= eqEps && (openThr', closeThr') > (bestOpenThr, bestCloseThr))
            then (eq, openThr', closeThr', bt)
            else (bestEq, bestOpenThr, bestCloseThr, bestBt)

      foldClose acc openThr = foldl' (\acc0 closeThr -> pick acc0 (openThr, closeThr)) acc candidates
      (_, bestOpenThr, bestCloseThr, bestBt) = foldl' foldClose (baseEq, baseOpenThr, baseCloseThr, baseBt) candidates

      result = (bestOpenThr, bestCloseThr, bestBt)
   in
    if n < 2
      then Left "sweepThreshold: need at least 2 prices"
      else if V.length highsV /= n || V.length lowsV /= n
        then Left "sweepThreshold: high/low series must match closes length"
        else if maybe False (\mv -> V.length mv < stepCount) metaUsed
          then Left "sweepThreshold: meta vector too short"
          else
            case method of
              MethodBoth
                | V.length kalV < stepCount -> Left "sweepThreshold: kalPred too short"
                | V.length lstmV < stepCount -> Left "sweepThreshold: lstmPred too short"
                | otherwise -> Right result
              MethodKalmanOnly
                | V.length kalV < stepCount -> Left "sweepThreshold: kalPred too short"
                | otherwise -> Right result
              MethodLstmOnly
                | V.length lstmV < stepCount -> Left "sweepThreshold: lstmPred too short"
                | otherwise -> Right result

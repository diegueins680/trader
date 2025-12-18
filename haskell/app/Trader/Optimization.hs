module Trader.Optimization
  ( bestFinalEquity
  , optimizeOperations
  , optimizeOperationsWithHL
  , sweepThreshold
  , sweepThresholdWithHL
  ) where

import Data.List (foldl', sort)
import Data.List.NonEmpty (NonEmpty((:|)))
import qualified Data.List.NonEmpty as NE
import qualified Data.Vector as V

import Trader.Method (Method(..))
import Trader.Trading (BacktestResult(..), EnsembleConfig(..), StepMeta(..), simulateEnsembleLongFlatV, simulateEnsembleLongFlatVWithHL)

bestFinalEquity :: BacktestResult -> Double
bestFinalEquity br =
  case brEquityCurve br of
    [] -> 1.0
    xs -> last xs

optimizeOperations :: EnsembleConfig -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Method, Double, Double, BacktestResult)
optimizeOperations baseCfg prices kalPred lstmPred mMeta =
  optimizeOperationsWithHL baseCfg prices prices prices kalPred lstmPred mMeta

optimizeOperationsWithHL :: EnsembleConfig -> [Double] -> [Double] -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Method, Double, Double, BacktestResult)
optimizeOperationsWithHL baseCfg closes highs lows kalPred lstmPred mMeta = do
  let eps = 1e-12
      methodRank m =
        case m of
          MethodBoth -> 2 :: Int
          MethodKalmanOnly -> 1
          MethodLstmOnly -> 0
      eval m = do
        (openThr, closeThr, bt) <- sweepThresholdWithHL m baseCfg closes highs lows kalPred lstmPred mMeta
        let eq = bestFinalEquity bt
        pure (eq, m, openThr, closeThr, bt)

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

  candidates <- NE.fromList <$> traverse eval [MethodBoth, MethodKalmanOnly, MethodLstmOnly]
  let (c :| cs) = candidates
      (_, bestM, bestOpenThr, bestCloseThr, bestBt) = foldl' pick c cs
   in Right (bestM, bestOpenThr, bestCloseThr, bestBt)

sweepThreshold :: Method -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Double, Double, BacktestResult)
sweepThreshold method baseCfg prices kalPred lstmPred mMeta =
  sweepThresholdWithHL method baseCfg prices prices prices kalPred lstmPred mMeta

sweepThresholdWithHL :: Method -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Double, Double, BacktestResult)
sweepThresholdWithHL method baseCfg closes highs lows kalPred lstmPred mMeta = do
  let pricesV = V.fromList closes
      highsV = V.fromList highs
      lowsV = V.fromList lows
      stepCount = V.length pricesV - 1
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
                     in [ v V.! pick i | i <- [0 .. k - 1] ]

      kalV = V.fromList kalPred
      lstmV = V.fromList lstmPred

      metaV = V.fromList <$> mMeta

      buildContext =
        case method of
          MethodBoth
            | V.length kalV < stepCount -> Left ("kalPred has length " ++ show (V.length kalV) ++ " but needs at least " ++ show stepCount ++ " for sweepThreshold")
            | V.length lstmV < stepCount -> Left ("lstmPred has length " ++ show (V.length lstmV) ++ " but needs at least " ++ show stepCount ++ " for sweepThreshold")
            | otherwise ->
                Right
                  ( [kalV, lstmV]
                  , (kalV, lstmV)
                  , metaV
                  )
          MethodKalmanOnly
            | V.length kalV < stepCount -> Left ("kalPred has length " ++ show (V.length kalV) ++ " but needs at least " ++ show stepCount ++ " for sweepThreshold")
            | otherwise -> Right ([kalV], (kalV, kalV), metaV)
          MethodLstmOnly
            | V.length lstmV < stepCount -> Left ("lstmPred has length " ++ show (V.length lstmV) ++ " but needs at least " ++ show stepCount ++ " for sweepThreshold")
            | otherwise -> Right ([lstmV], (lstmV, lstmV), Nothing)

  (predSources, (kalUsedV, lstmUsedV), metaUsed) <- buildContext

  let mags =
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

      uniqueSorted xs =
        case sort xs of
          [] -> []
          y : ys -> map NE.head (NE.group (y :| ys))
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

  pure (bestOpenThr, bestCloseThr, bestBt)

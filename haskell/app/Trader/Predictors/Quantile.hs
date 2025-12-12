module Trader.Predictors.Quantile
  ( LinModel(..)
  , QuantileModel(..)
  , trainQuantileModel
  , predictQuantiles
  , sigmaFromQ1090
  ) where

import Data.List (foldl')

data LinModel = LinModel
  { lmW :: [Double]
  , lmB :: !Double
  } deriving (Eq, Show)

data QuantileModel = QuantileModel
  { qm10 :: LinModel
  , qm50 :: LinModel
  , qm90 :: LinModel
  } deriving (Eq, Show)

trainQuantileModel :: Int -> Double -> Double -> [([Double], Double)] -> QuantileModel
trainQuantileModel epochs lr l2 dataset
  | epochs < 0 = error "epochs must be >= 0"
  | lr <= 0 = error "lr must be > 0"
  | l2 < 0 = error "l2 must be >= 0"
  | null dataset = error "quantile dataset is empty"
  | otherwise =
      let d = length (fst (head dataset))
          initM = LinModel { lmW = replicate d 0, lmB = 0 }
          qs0 = QuantileModel initM initM initM
       in iterate (epochStep lr l2 dataset) qs0 !! epochs

predictQuantiles :: QuantileModel -> [Double] -> (Double, Double, Double, Double, Maybe Double)
predictQuantiles qm x =
  let q10' = predictLin (qm10 qm) x
      q50' = predictLin (qm50 qm) x
      q90' = predictLin (qm90 qm) x
      (lo, hi) = if q10' <= q90' then (q10', q90') else (q90', q10')
      sigma = sigmaFromQ1090 lo hi
   in (lo, q50', hi, q50', Just sigma)

predictLin :: LinModel -> [Double] -> Double
predictLin m x = dot (lmW m) x + lmB m

epochStep :: Double -> Double -> [([Double], Double)] -> QuantileModel -> QuantileModel
epochStep lr l2 dataset qm0 =
  foldl' (\qm (x, y) -> sgdOne lr l2 x y qm) qm0 dataset

sgdOne :: Double -> Double -> [Double] -> Double -> QuantileModel -> QuantileModel
sgdOne lr l2 x y qm =
  QuantileModel
    { qm10 = stepTau 0.1 (qm10 qm)
    , qm50 = stepTau 0.5 (qm50 qm)
    , qm90 = stepTau 0.9 (qm90 qm)
    }
  where
    stepTau :: Double -> LinModel -> LinModel
    stepTau tau m =
      let pred = predictLin m x
          -- Pinball loss gradient wrt prediction:
          -- y > pred  => dL/dpred = -tau
          -- y < pred  => dL/dpred = 1 - tau
          g = if y > pred then (-tau) else (1 - tau)
          w' = zipWith (\wi xi -> wi - lr * (g * xi + l2 * wi)) (lmW m) x
          b' = lmB m - lr * g
       in LinModel { lmW = w', lmB = b' }

sigmaFromQ1090 :: Double -> Double -> Double
sigmaFromQ1090 q10 q90 =
  let z = 1.281551565545 -- Phi^{-1}(0.9)
   in max 1e-12 ((q90 - q10) / (2 * z))

dot :: [Double] -> [Double] -> Double
dot a b = sum (zipWith (*) a b)


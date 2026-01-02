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

featureDimFromDataset :: [([Double], Double)] -> Int
featureDimFromDataset dataset =
  case map (length . fst) dataset of
    [] -> 0
    d : ds
      | d <= 0 -> error "quantile dataset has empty feature vectors"
      | any (/= d) ds -> error "quantile dataset has inconsistent feature dimensions"
      | otherwise -> d

trainQuantileModel :: Int -> Double -> Double -> [([Double], Double)] -> QuantileModel
trainQuantileModel epochs lr l2 dataset
  | epochs < 0 = error "epochs must be >= 0"
  | lr <= 0 = error "lr must be > 0"
  | l2 < 0 = error "l2 must be >= 0"
  | null dataset = error "quantile dataset is empty"
  | otherwise =
      let d = featureDimFromDataset dataset
          initM = LinModel { lmW = replicate d 0, lmB = 0 }
          qs0 = QuantileModel initM initM initM
       in applyN epochs (epochStep lr l2 dataset) qs0

predictQuantiles :: QuantileModel -> [Double] -> Maybe (Double, Double, Double, Double, Maybe Double)
predictQuantiles qm x =
  let dims =
        [ length (lmW (qm10 qm))
        , length (lmW (qm50 qm))
        , length (lmW (qm90 qm))
        ]
      expected =
        case dims of
          [] -> 0
          (d:ds) -> if d > 0 && all (== d) ds then d else 0
      actual = length x
   in if expected <= 0 || actual /= expected
        then Nothing
        else
          let q10Raw = predictLin (qm10 qm) x
              q50Raw = predictLin (qm50 qm) x
              q90Raw = predictLin (qm90 qm) x
              (lo, hi) = if q10Raw <= q90Raw then (q10Raw, q90Raw) else (q90Raw, q10Raw)
              q50Clamped = min hi (max lo q50Raw)
              sigma = sigmaFromQ1090 lo hi
           in Just (lo, q50Clamped, hi, q50Raw, sigma)

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

sigmaFromQ1090 :: Double -> Double -> Maybe Double
sigmaFromQ1090 q10 q90 =
  let z = 1.281551565545 -- Phi^{-1}(0.9)
      width = q90 - q10
   in
    if not (isFinite width) || width <= 0
      then Nothing
      else Just (width / (2 * z))

dot :: [Double] -> [Double] -> Double
dot a b =
  let la = length a
      lb = length b
   in if la /= lb
        then error ("Quantile model feature dimension mismatch: expected " ++ show la ++ ", got " ++ show lb)
        else sum (zipWith (*) a b)

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)

applyN :: Int -> (a -> a) -> a -> a
applyN n f x0 = go n x0
  where
    go k x
      | k <= 0 = x
      | otherwise =
          let x' = f x
           in x' `seq` go (k - 1) x'

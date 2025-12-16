module Trader.Predictors.Transformer
  ( TransformerModel(..)
  , trainTransformer
  , predictTransformer
  ) where

import Data.List (foldl')

data TransformerModel = TransformerModel
  { trKeys :: [[Double]]
  , trTargets :: [Double]
  , trTemperature :: !Double
  } deriving (Eq, Show)

trainTransformer :: Double -> Int -> [([Double], Double)] -> TransformerModel
trainTransformer temperature maxExamples dataset =
  let ds = take (max 1 maxExamples) dataset
   in TransformerModel
        { trKeys = map fst ds
        , trTargets = map snd ds
        , trTemperature = temperature
        }

predictTransformer :: TransformerModel -> [Double] -> (Double, Maybe Double)
predictTransformer m query =
  let keys = trKeys m
      ys = trTargets m
   in if null keys
        then (0, Nothing)
        else
          let d = fromIntegral (length query)
              sim k = dot query k / sqrt (max 1 d)
              scoresRaw = map (\k -> trTemperature m * sim k) keys
              mx = maximum scoresRaw
              ws = map (exp . (\s -> s - mx)) scoresRaw
              z = sum ws
              wNorm = map (/ z) ws
              mu = sum (zipWith (*) wNorm ys)
              var = sum (zipWith (\w y -> w * (y - mu) * (y - mu)) wNorm ys)
           in (mu, Just (sqrt (var + 1e-12)))

dot :: [Double] -> [Double] -> Double
dot a b = sum (zipWith (*) a b)


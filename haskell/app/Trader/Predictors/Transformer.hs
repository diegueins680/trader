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
  , trFeatureDim :: !Int
  } deriving (Eq, Show)

featureDimFromDataset :: [([Double], Double)] -> Maybe Int
featureDimFromDataset dataset =
  case map (length . fst) dataset of
    [] -> Nothing
    d : ds
      | d <= 0 -> error "transformer dataset has empty feature vectors"
      | any (/= d) ds -> error "transformer dataset has inconsistent feature dimensions"
      | otherwise -> Just d

trainTransformer :: Double -> Int -> [([Double], Double)] -> TransformerModel
trainTransformer temperature maxExamples dataset =
  let ds = take (max 1 maxExamples) dataset
      featureDim = maybe 0 id (featureDimFromDataset ds)
   in TransformerModel
        { trKeys = map fst ds
        , trTargets = map snd ds
        , trTemperature = temperature
        , trFeatureDim = featureDim
        }

predictTransformer :: TransformerModel -> [Double] -> (Double, Maybe Double)
predictTransformer m query =
  let keys = trKeys m
      ys = trTargets m
   in if null keys
        then (0, Nothing)
        else
          let expected = trFeatureDim m
              actual = length query
              d = fromIntegral actual
           in
            if expected > 0 && actual /= expected
              then (0, Nothing)
              else
                let sim k = dot query k / sqrt (max 1 d)
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

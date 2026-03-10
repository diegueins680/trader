module Trader.Predictors.Transformer (
    TransformerModel (..),
    trainTransformer,
    predictTransformer,
) where

import Data.Maybe (listToMaybe)

data TransformerModel = TransformerModel
    { trKeys :: [[Double]]
    , trTargets :: [Double]
    , trTemperature :: !Double
    , trFeatureDim :: !Int
    }
    deriving (Eq, Show)

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)

sanitizeTemperature :: Double -> Double
sanitizeTemperature t
    | isFiniteDouble t && t > 0 = t
    | otherwise = 1.0

sanitizeDataset :: Int -> [([Double], Double)] -> [([Double], Double)]
sanitizeDataset maxExamples dataset
    | maxExamples <= 0 = []
    | otherwise =
        let finiteRows =
                [ (x, y)
                | (x, y) <- dataset
                , not (null x)
                , all isFiniteDouble x
                , isFiniteDouble y
                ]
         in case finiteRows of
                [] -> []
                ((x0, _) : _) ->
                    let featureDim = length x0
                        consistent = filter ((== featureDim) . length . fst) finiteRows
                     in take maxExamples consistent

trainTransformer :: Double -> Int -> [([Double], Double)] -> TransformerModel
trainTransformer temperature maxExamples dataset =
    let ds = sanitizeDataset maxExamples dataset
        featureDim = maybe 0 (length . fst) (listToMaybe ds)
     in TransformerModel
            { trKeys = map fst ds
            , trTargets = map snd ds
            , trTemperature = sanitizeTemperature temperature
            , trFeatureDim = featureDim
            }

predictTransformer :: TransformerModel -> [Double] -> (Double, Maybe Double)
predictTransformer m query =
    let keys = trKeys m
        ys = trTargets m
        expected = trFeatureDim m
        d = fromIntegral (max 1 expected)
        modelValid =
            expected > 0
                && not (null keys)
                && length keys == length ys
                && all ((== expected) . length) keys
                && all (all isFiniteDouble) keys
                && all isFiniteDouble ys
        queryValid = length query == expected && all isFiniteDouble query
        temp = sanitizeTemperature (trTemperature m)
     in if not modelValid || not queryValid
            then (0, Nothing)
            else
                let sim k = dot query k / sqrt d
                    scoresRaw = map (\k -> temp * sim k) keys
                 in if not (all isFiniteDouble scoresRaw)
                        then (0, Nothing)
                        else
                            let mx = maximum scoresRaw
                                ws = map (exp . (\s -> s - mx)) scoresRaw
                                z = sum ws
                             in if not (isFiniteDouble z) || z <= 0 || not (all isFiniteDouble ws)
                                    then (0, Nothing)
                                    else
                                        let wNorm = map (/ z) ws
                                            mu = sum (zipWith (*) wNorm ys)
                                            var = sum (zipWith (\w y -> w * (y - mu) * (y - mu)) wNorm ys)
                                            sigma = sqrt (max 0 var + 1e-12)
                                         in if isFiniteDouble mu && isFiniteDouble sigma
                                                then (mu, Just sigma)
                                                else (0, Nothing)

dot :: [Double] -> [Double] -> Double
dot a b = sum (zipWith (*) a b)

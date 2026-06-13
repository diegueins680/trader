module Trader.Predictors.KNN (
    KNNModel (..),
    trainKNN,
    predictKNN,
) where

import Data.List (sortOn)

data KNNModel = KNNModel
    { kmK :: !Int
    , kmFeatureDim :: !Int
    , kmMeans :: ![Double]
    , kmScales :: ![Double]
    , kmExamples :: [([Double], Double)]
    , kmSigmaBase :: !(Maybe Double)
    }
    deriving (Eq, Show)

trainKNN :: Int -> Int -> [([Double], Double)] -> KNNModel
trainKNN maxExamples requestedK dataset
    | maxExamples <= 0 = emptyKNNModel
    | requestedK <= 0 = emptyKNNModel
    | null dataset = emptyKNNModel
    | otherwise =
        let ds0 = sanitizeDataset dataset
            ds = evenlySample maxExamples ds0
            featureDim = featureDimFromDataset ds
         in if featureDim <= 0 || null ds
                then emptyKNNModel
                else
                    let means = featureMeans featureDim ds
                        scales = featureScales featureDim means ds
                        examples =
                            [ (normalizeVector means scales x, y)
                            | (x, y) <- ds
                            ]
                        sigmaBase = sigmaFromTargets (map snd ds)
                        k = min (length examples) (max 1 requestedK)
                     in KNNModel
                            { kmK = k
                            , kmFeatureDim = featureDim
                            , kmMeans = means
                            , kmScales = scales
                            , kmExamples = examples
                            , kmSigmaBase = sigmaBase
                            }

predictKNN :: KNNModel -> [Double] -> (Double, Maybe Double)
predictKNN model query =
    let expected = kmFeatureDim model
        safeSigmaBase = kmSigmaBase model >>= finiteMaybe
        queryOk = expected > 0 && length query == expected && all isFiniteDouble query
        modelOk =
            expected > 0
                && kmK model > 0
                && not (null (kmExamples model))
                && length (kmMeans model) == expected
                && length (kmScales model) == expected
                && all isFiniteDouble (kmMeans model)
                && all isFiniteDouble (kmScales model)
                && all (\(x, y) -> length x == expected && all isFiniteDouble x && isFiniteDouble y) (kmExamples model)
     in if not modelOk || not queryOk
            then (0, safeSigmaBase)
            else
                let qNorm = normalizeVector (kmMeans model) (kmScales model) query
                    neighbors =
                        take
                            (kmK model)
                            (sortOn fst [(squaredDistance qNorm xNorm, y) | (xNorm, y) <- kmExamples model])
                    weights =
                        [ let d = max 0 dist
                           in 1 / (1e-6 + d)
                        | (dist, _y) <- neighbors
                        ]
                    totalW = sum weights
                 in if totalW <= 0 || not (isFiniteDouble totalW)
                        then (0, safeSigmaBase)
                        else
                            let weightedYs = zip weights (map snd neighbors)
                                mu = sum [w * y | (w, y) <- weightedYs] / totalW
                                var =
                                    sum [w * (y - mu) * (y - mu) | (w, y) <- weightedYs]
                                        / totalW
                                sigma =
                                    let s = sqrt (max 0 var + 1e-12)
                                     in if isFiniteDouble s then Just s else safeSigmaBase
                             in if isFiniteDouble mu
                                    then (mu, sigma <|> safeSigmaBase)
                                    else (0, safeSigmaBase)
  where
    (<|>) left right =
        case left of
            Just _ -> left
            Nothing -> right

emptyKNNModel :: KNNModel
emptyKNNModel =
    KNNModel
        { kmK = 0
        , kmFeatureDim = 0
        , kmMeans = []
        , kmScales = []
        , kmExamples = []
        , kmSigmaBase = Nothing
        }

sanitizeDataset :: [([Double], Double)] -> [([Double], Double)]
sanitizeDataset dataset =
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
                 in filter ((== featureDim) . length . fst) finiteRows

featureDimFromDataset :: [([Double], Double)] -> Int
featureDimFromDataset dataset =
    case map (length . fst) dataset of
        [] -> 0
        d : ds
            | d <= 0 -> 0
            | any (/= d) ds -> 0
            | otherwise -> d

evenlySample :: Int -> [a] -> [a]
evenlySample maxExamples xs
    | maxExamples <= 0 = []
    | length xs <= maxExamples = xs
    | maxExamples == 1 =
        case xs of
            y : _ -> [y]
            [] -> []
    | otherwise =
        let n = length xs
            atIndex i =
                let ix =
                        floor
                            ( fromIntegral i
                                * fromIntegral (n - 1)
                                / fromIntegral (maxExamples - 1)
                            )
                 in xs !! ix
         in [atIndex i | i <- [0 .. maxExamples - 1]]

featureMeans :: Int -> [([Double], Double)] -> [Double]
featureMeans featureDim dataset =
    [ mean [x !! j | (x, _y) <- dataset]
    | j <- [0 .. featureDim - 1]
    ]

featureScales :: Int -> [Double] -> [([Double], Double)] -> [Double]
featureScales featureDim means dataset =
    [ let vals = [x !! j | (x, _y) <- dataset]
          mu = means !! j
          n = length vals
          var =
            if n < 2
                then 0
                else
                    sum (map (\v -> (v - mu) * (v - mu)) vals)
                        / fromIntegral (n - 1)
          s = sqrt (max 0 var + 1e-12)
       in if isFiniteDouble s && s > 1e-6 then s else 1
    | j <- [0 .. featureDim - 1]
    ]

normalizeVector :: [Double] -> [Double] -> [Double] -> [Double]
normalizeVector =
    zipWith3
        (\mu s x -> if s <= 1e-12 then x - mu else (x - mu) / s)

squaredDistance :: [Double] -> [Double] -> Double
squaredDistance xs ys =
    sum (zipWith (\x y -> let d = x - y in d * d) xs ys)

sigmaFromTargets :: [Double] -> Maybe Double
sigmaFromTargets ys =
    case ys of
        [] -> Nothing
        [_] -> Just 1e-6
        _ ->
            let mu = mean ys
                var = sum (map (\y -> (y - mu) * (y - mu)) ys) / fromIntegral (length ys - 1)
                sigma = sqrt (max 0 var + 1e-12)
             in if isFiniteDouble sigma then Just sigma else Nothing

mean :: [Double] -> Double
mean xs =
    case xs of
        [] -> 0
        _ -> sum xs / fromIntegral (length xs)

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)

finiteMaybe :: Double -> Maybe Double
finiteMaybe x
    | isFiniteDouble x = Just x
    | otherwise = Nothing

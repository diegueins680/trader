module Trader.Predictors.GBDT (
    Stump (..),
    GBDTModel (..),
    trainGBDT,
    predictGBDT,
) where

import Data.List (minimumBy, sort)
import Data.Ord (comparing)
import qualified Data.Vector as V

import Trader.Text (dedupeStable)

data Stump = Stump
    { stFeature :: !Int
    , stThreshold :: !Double
    , stLeftValue :: !Double
    , stRightValue :: !Double
    }
    deriving (Eq, Show)

data GBDTModel = GBDTModel
    { gmBase :: !Double
    , gmLearningRate :: !Double
    , gmFeatureDim :: !Int
    , gmStumps :: [Stump]
    , gmSigma :: !(Maybe Double)
    }
    deriving (Eq, Show)

predictGBDT :: GBDTModel -> [Double] -> (Double, Maybe Double)
predictGBDT m feats =
    let expected = gmFeatureDim m
        actual = length feats
        base = gmBase m
        lr = gmLearningRate m
     in -- Fall back to the base prediction when feature vectors are inconsistent.
        if actual /= expected || expected <= 0
            then (base, gmSigma m)
            else
                let featsV = V.fromList feats
                    applySt Stump{stFeature = j, stThreshold = thr, stLeftValue = l, stRightValue = r} =
                        let x = featsV V.! j
                         in if x <= thr then l else r
                    y = base + lr * sum (map applySt (gmStumps m))
                 in (y, gmSigma m)

featureDimFromDataset :: [([Double], Double)] -> Int
featureDimFromDataset dataset =
    case map (length . fst) dataset of
        [] -> 0
        d : ds
            | d <= 0 -> error "GBDT dataset has empty feature vectors"
            | any (/= d) ds -> error "GBDT dataset has inconsistent feature dimensions"
            | otherwise -> d

trainGBDT :: Int -> Double -> [([Double], Double)] -> GBDTModel
trainGBDT nTrees learningRate dataset
    | nTrees <= 0 = error "nTrees must be > 0"
    | learningRate <= 0 = error "learningRate must be > 0"
    | null dataset = error "GBDT dataset is empty"
    | otherwise =
        let featureDim = featureDimFromDataset dataset
            featsAll = V.fromList (map (V.fromList . fst) dataset)
            ys = V.fromList (map snd dataset)
            base = meanV ys
            preds0 = V.replicate (V.length ys) base
            go 0 acc preds = (acc, preds)
            go k acc preds =
                let residuals = V.zipWith (-) ys preds
                    stump = fitStump featsAll residuals
                    stumpOut = V.map (stumpPredict stump) featsAll
                    preds' = V.zipWith (+) preds (V.map (* learningRate) stumpOut)
                 in go (k - 1) (stump : acc) preds'
            (stumps, predsFinal) = go nTrees [] preds0
            residuals = V.zipWith (-) ys predsFinal
            sigma = sqrt (meanV (V.map (\e -> e * e) residuals) + 1e-12)
         in GBDTModel
                { gmBase = base
                , gmLearningRate = learningRate
                , gmFeatureDim = featureDim
                , gmStumps = reverse stumps
                , gmSigma = Just sigma
                }

stumpPredict :: Stump -> V.Vector Double -> Double
stumpPredict Stump{stFeature = j, stThreshold = thr, stLeftValue = l, stRightValue = r} feats =
    if feats V.! j <= thr then l else r

fitStump :: V.Vector (V.Vector Double) -> V.Vector Double -> Stump
fitStump feats residuals =
    let d = V.length (V.head feats)
        candidates = [bestForFeature j | j <- [0 .. d - 1]]
     in minimumBy (comparing snd) candidates |> fst
  where
    (|>) = flip ($)

    bestForFeature :: Int -> (Stump, Double)
    bestForFeature j =
        let xs = V.map (V.! j) feats
            thrs = candidateThresholds (V.toList xs)
            options =
                [ let (lRes, rRes) = splitOnThreshold thr xs residuals
                      lMean = if null lRes then 0 else mean lRes
                      rMean = if null rRes then 0 else mean rRes
                      sse = sumSqErr lMean lRes + sumSqErr rMean rRes
                      stump = Stump{stFeature = j, stThreshold = thr, stLeftValue = lMean, stRightValue = rMean}
                   in (stump, sse)
                | thr <- thrs
                , let (lRes, rRes) = splitOnThreshold thr xs residuals
                , not (null lRes) && not (null rRes)
                ]
         in case options of
                [] ->
                    let mu = meanV residuals
                     in (Stump{stFeature = j, stThreshold = 0, stLeftValue = mu, stRightValue = mu}, sumSqErr mu (V.toList residuals))
                _ -> minimumBy (comparing snd) options

candidateThresholds :: [Double] -> [Double]
candidateThresholds xs =
    let s = sort xs
        n = length s
        buckets = 16 :: Int
     in if n < 3
            then []
            else
                let ix q = max 1 (min (n - 2) (floor (fromIntegral q * fromIntegral n / fromIntegral buckets)))
                    thrs = [s !! ix q | q <- [1 .. buckets - 2]]
                 in dedupeStable thrs

splitOnThreshold :: Double -> V.Vector Double -> V.Vector Double -> ([Double], [Double])
splitOnThreshold thr xs residuals =
    V.foldl'
        (\(l, rr) (x, r) -> if x <= thr then (r : l, rr) else (l, r : rr))
        ([], [])
        (V.zip xs residuals)

mean :: [Double] -> Double
mean xs =
    case xs of
        [] -> 0
        _ -> sum xs / fromIntegral (length xs)

meanV :: V.Vector Double -> Double
meanV xs =
    if V.null xs
        then 0
        else V.sum xs / fromIntegral (V.length xs)

sumSqErr :: Double -> [Double] -> Double
sumSqErr mu xs = sum (map (\x -> let e = x - mu in e * e) xs)

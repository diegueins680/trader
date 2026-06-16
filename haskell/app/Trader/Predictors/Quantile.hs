module Trader.Predictors.Quantile (
    LinModel (..),
    QuantileModel (..),
    defaultQuantileEpochs,
    defaultQuantileLearningRate,
    defaultQuantileL2,
    trainQuantileModel,
    predictQuantiles,
    sigmaFromQ1090,
) where

import Data.List (foldl')

data LinModel = LinModel
    { lmW :: [Double]
    , lmB :: !Double
    }
    deriving (Eq, Show)

data QuantileModel = QuantileModel
    { qm10 :: LinModel
    , qm50 :: LinModel
    , qm90 :: LinModel
    }
    deriving (Eq, Show)

defaultQuantileEpochs :: Int
defaultQuantileEpochs = 20

defaultQuantileLearningRate :: Double
defaultQuantileLearningRate = 5e-2

defaultQuantileL2 :: Double
defaultQuantileL2 = 1e-3

featureDimFromDataset :: [([Double], Double)] -> Int
featureDimFromDataset dataset =
    case map (length . fst) dataset of
        [] -> 0
        d : ds
            | d <= 0 -> 0
            | any (/= d) ds -> 0
            | otherwise -> d

trainQuantileModel :: Int -> Double -> Double -> [([Double], Double)] -> QuantileModel
trainQuantileModel epochs lr l2 dataset
    | epochs < 0 = emptyQuantileModel
    | not (isFinite lr) || lr <= 0 = emptyQuantileModel
    | not (isFinite l2) || l2 < 0 = emptyQuantileModel
    | otherwise =
        let ds = sanitizeDataset dataset
            d = featureDimFromDataset ds
         in if d <= 0 || null ds
                then emptyQuantileModel
                else
                    let initM = LinModel{lmW = replicate d 0, lmB = 0}
                        qs0 = QuantileModel initM initM initM
                     in applyN epochs (epochStep lr l2 ds) qs0

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
                (d : ds) -> if d > 0 && all (== d) ds then d else 0
        actual = length x
        modelValid =
            expected > 0
                && all (linModelValid expected) [qm10 qm, qm50 qm, qm90 qm]
        queryValid = actual == expected && all isFinite x
     in if not modelValid || not queryValid
            then Nothing
            else
                let q10Raw = predictLin (qm10 qm) x
                    q50Raw = predictLin (qm50 qm) x
                    q90Raw = predictLin (qm90 qm) x
                 in admissibleQuantilePrediction q10Raw q50Raw q90Raw

sanitizeDataset :: [([Double], Double)] -> [([Double], Double)]
sanitizeDataset dataset =
    let finiteRows =
            [ (x, y)
            | (x, y) <- dataset
            , not (null x)
            , all isFinite x
            , isFinite y
            ]
     in case finiteRows of
            [] -> []
            ((x0, _) : _) ->
                let featureDim = length x0
                 in filter ((== featureDim) . length . fst) finiteRows

linModelValid :: Int -> LinModel -> Bool
linModelValid expected LinModel{lmW = w, lmB = b} =
    length w == expected
        && all isFinite w
        && isFinite b

admissibleQuantilePrediction ::
    Double ->
    Double ->
    Double ->
    Maybe (Double, Double, Double, Double, Maybe Double)
admissibleQuantilePrediction q10Raw q50Raw q90Raw
    | not (all isFinite [q10Raw, q50Raw, q90Raw]) = Nothing
    | q10Raw > q90Raw = Nothing
    | otherwise =
        let q50Clamped = min q90Raw (max q10Raw q50Raw)
            sigma = sigmaFromQ1090 q10Raw q90Raw
         in Just (q10Raw, q50Clamped, q90Raw, q50Raw, sigma)

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
            g
                | y > pred = (-tau)
                | y < pred = (1 - tau)
                | otherwise = 0
            w' = zipWith (\wi xi -> wi - lr * (g * xi + l2 * wi)) (lmW m) x
            b' = lmB m - lr * g
         in LinModel{lmW = w', lmB = b'}

sigmaFromQ1090 :: Double -> Double -> Maybe Double
sigmaFromQ1090 q10 q90 =
    let z = 1.281551565545 -- Phi^{-1}(0.9)
        width = q90 - q10
     in if not (isFinite width) || width <= 0
            then Nothing
            else Just (width / (2 * z))

dot :: [Double] -> [Double] -> Double
dot a b = sum (zipWith (*) a b)

emptyQuantileModel :: QuantileModel
emptyQuantileModel =
    let empty = LinModel{lmW = [], lmB = 0}
     in QuantileModel empty empty empty

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)

applyN :: Int -> (a -> a) -> a -> a
applyN n f = go n
  where
    go k x
        | k <= 0 = x
        | otherwise =
            let x' = f x
             in x' `seq` go (k - 1) x'

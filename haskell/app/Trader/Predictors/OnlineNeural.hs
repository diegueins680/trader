module Trader.Predictors.OnlineNeural (
    OnlineNeuralConfig (..),
    OnlineNeuralSignal (..),
    defaultOnlineNeuralConfig,
    onlineNeuralPredictionAt,
    onlineNeuralPredictionsRange,
) where

import Data.List (foldl', transpose)
import Data.Maybe (fromMaybe, listToMaybe)
import qualified Data.Vector as V
import System.Random (mkStdGen, randomRs)

import Trader.MarketContext (MarketModel)
import Trader.Predictors.Features (
    FeatureInputs (..),
    buildDatasetWithIndexWithInputsWithMarket,
    featuresAtWithInputsWithMarket,
    mkFeatureSpec,
 )

data OnlineNeuralConfig = OnlineNeuralConfig
    { oncLookbackBars :: !Int
    , oncHidden1 :: !Int
    , oncHidden2 :: !Int
    , oncInitialEpochs :: !Int
    , oncLearningRate :: !Double
    , oncL2 :: !Double
    , oncTargetClamp :: !Double
    , oncGradientClip :: !Double
    , oncErrorEwmaAlpha :: !Double
    , oncSeed :: !Int
    }
    deriving (Eq, Show)

data OnlineNeuralSignal = OnlineNeuralSignal
    { onsPrediction :: !Double
    , onsReturn :: !Double
    , onsConfidence :: !Double
    , onsExamples :: !Int
    }
    deriving (Eq, Show)

data OnlineNeuralModel = OnlineNeuralModel
    { onmInputDim :: !Int
    , onmHidden1 :: !Int
    , onmHidden2 :: !Int
    , onmW1 :: [[Double]]
    , onmB1 :: [Double]
    , onmW2 :: [[Double]]
    , onmB2 :: [Double]
    , onmW3 :: [Double]
    , onmB3 :: !Double
    }
    deriving (Eq, Show)

data OnlineState = OnlineState
    { osModel :: !OnlineNeuralModel
    , osExamples :: !Int
    , osErrEwma :: !(Maybe Double)
    }
    deriving (Eq, Show)

defaultOnlineNeuralConfig :: Int -> Int -> Double -> Int -> Int -> OnlineNeuralConfig
defaultOnlineNeuralConfig lookback hidden lr epochs seed =
    OnlineNeuralConfig
        { oncLookbackBars = max 2 lookback
        , oncHidden1 = max 2 hidden
        , oncHidden2 = max 2 ((max 2 hidden + 1) `div` 2)
        , oncInitialEpochs = max 1 epochs
        , oncLearningRate = finitePositive 1e-3 lr
        , oncL2 = 1e-5
        , oncTargetClamp = 0.20
        , oncGradientClip = 5.0
        , oncErrorEwmaAlpha = 0.08
        , oncSeed = seed
        }

onlineNeuralPredictionAt ::
    OnlineNeuralConfig ->
    Maybe MarketModel ->
    FeatureInputs ->
    Int ->
    Maybe OnlineNeuralSignal
onlineNeuralPredictionAt cfg0 mMarket inputs t = do
    let cfg = sanitizeConfig cfg0
        fs = mkFeatureSpec (oncLookbackBars cfg)
        prices = fiClose inputs
    current <- vectorValue prices t
    feats <- featuresAtWithInputsWithMarket fs mMarket inputs t
    let dataset = validSamples cfg (buildDatasetWithIndexWithInputsWithMarket fs mMarket inputs)
        trainSamples = takeWhile (\(ix, _, _) -> ix < t) dataset
        model0 = initModel cfg (length feats)
        state = trainState cfg model0 trainSamples
        predR = predictReturn cfg (osModel state) feats
        pred = current * (1 + predR)
    if isFiniteDouble pred
        then
            Just
                OnlineNeuralSignal
                    { onsPrediction = pred
                    , onsReturn = predR
                    , onsConfidence = confidenceFromState cfg predR (osErrEwma state)
                    , onsExamples = osExamples state
                    }
        else Nothing

onlineNeuralPredictionsRange ::
    OnlineNeuralConfig ->
    Maybe MarketModel ->
    FeatureInputs ->
    Int ->
    Int ->
    V.Vector Double
onlineNeuralPredictionsRange cfg0 mMarket inputs start count
    | count <= 0 = V.empty
    | otherwise =
        case firstFeature of
            Nothing -> V.replicate count (neutralAt start)
            Just feats0 -> V.fromList (go initialState start count)
              where
                model0 = initModel cfg (length feats0)
                initialState = trainState cfg model0 pretrainSamples
  where
    cfg = sanitizeConfig cfg0
    fs = mkFeatureSpec (oncLookbackBars cfg)
    prices = fiClose inputs
    dataset = validSamples cfg (buildDatasetWithIndexWithInputsWithMarket fs mMarket inputs)
    pretrainSamples = takeWhile (\(ix, _, _) -> ix < start) dataset
    firstFeature =
        featuresAtWithInputsWithMarket fs mMarket inputs start
            `orElse` fmap (\(_, feats, _) -> feats) (listToMaybe dataset)

    go _ _ remaining | remaining <= 0 = []
    go state t remaining =
        let current = neutralAt t
            pred =
                case featuresAtWithInputsWithMarket fs mMarket inputs t of
                    Just feats
                        | length feats == onmInputDim (osModel state) ->
                            let predR = predictReturn cfg (osModel state) feats
                                next = current * (1 + predR)
                             in if isFiniteDouble next then next else current
                    _ -> current
            state' =
                case sampleAt t dataset of
                    Just (_, feats, target)
                        | length feats == onmInputDim (osModel state) -> trainOneState cfg state feats target
                    _ -> state
         in pred : go state' (t + 1) (remaining - 1)

    neutralAt ix =
        fromMaybe 0 (vectorValue prices ix)

sanitizeConfig :: OnlineNeuralConfig -> OnlineNeuralConfig
sanitizeConfig cfg =
    cfg
        { oncLookbackBars = max 2 (oncLookbackBars cfg)
        , oncHidden1 = max 2 (oncHidden1 cfg)
        , oncHidden2 = max 2 (oncHidden2 cfg)
        , oncInitialEpochs = max 1 (oncInitialEpochs cfg)
        , oncLearningRate = finitePositive 1e-3 (oncLearningRate cfg)
        , oncL2 = finiteNonNegative 0 (oncL2 cfg)
        , oncTargetClamp = finitePositive 0.20 (oncTargetClamp cfg)
        , oncGradientClip = finitePositive 5 (oncGradientClip cfg)
        , oncErrorEwmaAlpha = clamp 0 1 (finiteNonNegative 0.08 (oncErrorEwmaAlpha cfg))
        }

validSamples :: OnlineNeuralConfig -> [(Int, [Double], Double)] -> [(Int, [Double], Double)]
validSamples cfg =
    filter
        ( \(_, feats, target) ->
            not (null feats)
                && all isFiniteDouble feats
                && isFiniteDouble target
                && abs target <= oncTargetClamp cfg * 20
        )

trainState :: OnlineNeuralConfig -> OnlineNeuralModel -> [(Int, [Double], Double)] -> OnlineState
trainState cfg model samples =
    foldl' trainEpoch OnlineState{osModel = model, osExamples = 0, osErrEwma = Nothing} [1 .. oncInitialEpochs cfg]
  where
    trainEpoch state _ =
        foldl'
            ( \acc (_, feats, target) ->
                if length feats == onmInputDim (osModel acc)
                    then trainOneState cfg acc feats target
                    else acc
            )
            state
            samples

trainOneState :: OnlineNeuralConfig -> OnlineState -> [Double] -> Double -> OnlineState
trainOneState cfg state feats target =
    let model = osModel state
        pred = predictReturn cfg model feats
        errAbs = abs (pred - clampTarget cfg target)
        errEwma =
            case osErrEwma state of
                Nothing -> Just errAbs
                Just prev ->
                    let a = oncErrorEwmaAlpha cfg
                     in Just ((1 - a) * prev + a * errAbs)
     in state
            { osModel = trainOne cfg model feats target
            , osExamples = osExamples state + 1
            , osErrEwma = errEwma
            }

trainOne :: OnlineNeuralConfig -> OnlineNeuralModel -> [Double] -> Double -> OnlineNeuralModel
trainOne cfg model feats target =
    let lr = oncLearningRate cfg
        l2 = oncL2 cfg
        x = sanitizeFeatures (onmInputDim model) feats
        (a1, a2, yRaw) = forwardRaw model x
        y = clampTarget cfg target
        err = clipGrad cfg (yRaw - y)
        delta3 = err
        delta2 =
            [ clipGrad cfg (delta3 * w3 * tanhDeriv a)
            | (w3, a) <- zip (onmW3 model) a2
            ]
        w2Cols = transpose (onmW2 model)
        delta1 =
            [ clipGrad cfg (sum (zipWith (*) delta2 col) * tanhDeriv a)
            | (col, a) <- zip w2Cols a1
            ]
        w3' =
            [ finiteOr old (old - lr * clipGrad cfg (delta3 * a + l2 * old))
            | (old, a) <- zip (onmW3 model) a2
            ]
        b3' = finiteOr (onmB3 model) (onmB3 model - lr * clipGrad cfg delta3)
        w2' =
            [ [ finiteOr old (old - lr * clipGrad cfg (d2 * a + l2 * old))
              | (old, a) <- zip row a1
              ]
            | (row, d2) <- zip (onmW2 model) delta2
            ]
        b2' =
            [ finiteOr old (old - lr * clipGrad cfg d2)
            | (old, d2) <- zip (onmB2 model) delta2
            ]
        w1' =
            [ [ finiteOr old (old - lr * clipGrad cfg (d1 * xi + l2 * old))
              | (old, xi) <- zip row x
              ]
            | (row, d1) <- zip (onmW1 model) delta1
            ]
        b1' =
            [ finiteOr old (old - lr * clipGrad cfg d1)
            | (old, d1) <- zip (onmB1 model) delta1
            ]
     in model
            { onmW1 = w1'
            , onmB1 = b1'
            , onmW2 = w2'
            , onmB2 = b2'
            , onmW3 = w3'
            , onmB3 = b3'
            }

predictReturn :: OnlineNeuralConfig -> OnlineNeuralModel -> [Double] -> Double
predictReturn cfg model feats =
    let x = sanitizeFeatures (onmInputDim model) feats
        (_, _, yRaw) = forwardRaw model x
     in clampTarget cfg yRaw

forwardRaw :: OnlineNeuralModel -> [Double] -> ([Double], [Double], Double)
forwardRaw model x =
    let a1 = zipWith (\row b -> tanh (dot row x + b)) (onmW1 model) (onmB1 model)
        a2 = zipWith (\row b -> tanh (dot row a1 + b)) (onmW2 model) (onmB2 model)
        yRaw = dot (onmW3 model) a2 + onmB3 model
     in (a1, a2, finiteOr 0 yRaw)

initModel :: OnlineNeuralConfig -> Int -> OnlineNeuralModel
initModel cfg inputDim0 =
    let inputDim = max 1 inputDim0
        h1 = oncHidden1 cfg
        h2 = oncHidden2 cfg
        gen = mkStdGen (oncSeed cfg + inputDim * 131 + h1 * 17 + h2)
        vals = randomRs (-1.0, 1.0) gen
        scale fanIn fanOut = sqrt (6 / fromIntegral (max 1 (fanIn + fanOut)))
        (w1Vals, rest1) = splitAt (h1 * inputDim) vals
        (w2Vals, rest2) = splitAt (h2 * h1) rest1
        (w3Vals, _) = splitAt h2 rest2
     in OnlineNeuralModel
            { onmInputDim = inputDim
            , onmHidden1 = h1
            , onmHidden2 = h2
            , onmW1 = scaleMatrix (scale inputDim h1) (chunksOf inputDim w1Vals)
            , onmB1 = replicate h1 0
            , onmW2 = scaleMatrix (scale h1 h2) (chunksOf h1 w2Vals)
            , onmB2 = replicate h2 0
            , onmW3 = map (* scale h2 1) w3Vals
            , onmB3 = 0
            }

confidenceFromState :: OnlineNeuralConfig -> Double -> Maybe Double -> Double
confidenceFromState cfg predR mErr =
    let denom = max 1e-6 (fromMaybe (oncTargetClamp cfg / 10) mErr)
     in clamp 0 1 (abs predR / (2 * denom))

sampleAt :: Int -> [(Int, [Double], Double)] -> Maybe (Int, [Double], Double)
sampleAt t =
    listToMaybe . dropWhile (\(ix, _, _) -> ix < t) . takeWhile (\(ix, _, _) -> ix <= t)

sanitizeFeatures :: Int -> [Double] -> [Double]
sanitizeFeatures inputDim feats =
    take inputDim (map sanitizeFeature feats ++ repeat 0)

sanitizeFeature :: Double -> Double
sanitizeFeature x
    | not (isFiniteDouble x) = 0
    | otherwise = clamp (-8) 8 x

clampTarget :: OnlineNeuralConfig -> Double -> Double
clampTarget cfg = clamp (negate (oncTargetClamp cfg)) (oncTargetClamp cfg)

clipGrad :: OnlineNeuralConfig -> Double -> Double
clipGrad cfg = clamp (negate (oncGradientClip cfg)) (oncGradientClip cfg)

tanhDeriv :: Double -> Double
tanhDeriv a = max 0 (1 - a * a)

dot :: [Double] -> [Double] -> Double
dot xs ys = sum (zipWith (*) xs ys)

chunksOf :: Int -> [a] -> [[a]]
chunksOf n xs
    | n <= 0 = []
    | otherwise =
        case splitAt n xs of
            ([], _) -> []
            (chunk, rest) -> chunk : chunksOf n rest

scaleMatrix :: Double -> [[Double]] -> [[Double]]
scaleMatrix scaleV = map (map (* scaleV))

vectorValue :: V.Vector Double -> Int -> Maybe Double
vectorValue values ix =
    if ix < 0 || ix >= V.length values
        then Nothing
        else
            let x = values V.! ix
             in if isFiniteDouble x then Just x else Nothing

orElse :: Maybe a -> Maybe a -> Maybe a
orElse (Just x) _ = Just x
orElse Nothing y = y

finitePositive :: Double -> Double -> Double
finitePositive fallback x
    | not (isFiniteDouble x) || x <= 0 = fallback
    | otherwise = x

finiteNonNegative :: Double -> Double -> Double
finiteNonNegative fallback x
    | not (isFiniteDouble x) || x < 0 = fallback
    | otherwise = x

finiteOr :: Double -> Double -> Double
finiteOr fallback x =
    if isFiniteDouble x then x else fallback

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)

clamp :: Double -> Double -> Double -> Double
clamp lo hi x
    | not (isFiniteDouble x) = lo
    | x < lo = lo
    | x > hi = hi
    | otherwise = x

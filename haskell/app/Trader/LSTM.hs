module Trader.LSTM (
    LSTMConfig (..),
    EpochStats (..),
    LSTMModel (..),
    paramCount,
    paramCountD,
    inputDimFromModel,
    buildSequences,
    buildSequencesMultiV,
    evaluateLoss,
    trainLSTM,
    trainLSTMMulti,
    fineTuneLSTM,
    predictNext,
    predictNextMulti,
    predictSeriesNext,
) where

import qualified Data.Bifunctor
import Data.List (foldl')
import qualified Data.Vector as V
import Numeric.AD (grad)
import System.Random (mkStdGen, randomRs)

data LSTMConfig = LSTMConfig
    { lcLookback :: !Int
    , lcHiddenSize :: !Int
    , lcEpochs :: !Int
    , lcLearningRate :: !Double
    , lcValRatio :: !Double
    , lcPatience :: !Int
    , lcGradClip :: !(Maybe Double)
    , lcSeed :: !Int
    }
    deriving (Eq, Show)

data EpochStats = EpochStats
    { esEpoch :: !Int
    , esTrainLoss :: !Double
    , esValLoss :: !Double
    }
    deriving (Eq, Show)

data LSTMModel = LSTMModel
    { lmHiddenSize :: !Int
    , lmParams :: [Double]
    }
    deriving (Eq, Show)

paramCount :: Int -> Int
paramCount h = paramCountD h 1

{- | Parameter count for a hidden size @h@ and input dimension @d@ (number of
channels per timestep). The univariate model is @d == 1@. Each gate matrix is
@h x (h + d)@ (recurrent state plus the @d@ inputs) with an @h@ bias.
-}
paramCountD :: Int -> Int -> Int
paramCountD h d =
    let gate = h * (h + d) + h -- W (h x (h+d)) plus b (h)
     in 4 * gate + h + 1 -- output head

{- | Recover the input dimension @d@ a model was trained with from its flat
parameter length and hidden size — so multivariate models need no extra stored
field and the existing persisted (univariate) format is unchanged.
Inverts 'paramCountD': @len = 4*(h*(h+d)+h) + h + 1@.
-}
inputDimFromModel :: LSTMModel -> Int
inputDimFromModel model =
    let h = lmHiddenSize model
        len = length (lmParams model)
     in if h <= 0 then 1 else max 1 ((len - (5 * h + 1)) `div` (4 * h) - h)

buildSequences :: Int -> [Double] -> [([Double], Double)]
buildSequences lookback xs
    | lookback <= 0 = []
    | otherwise =
        let xsV = V.fromList xs
         in map (Data.Bifunctor.first V.toList) (buildSequencesV lookback xsV)

buildSequencesV :: Int -> V.Vector Double -> [(V.Vector Double, Double)]
buildSequencesV lookback xsV
    | lookback <= 0 = []
    | V.length xsV <= lookback = []
    | otherwise =
        let count = V.length xsV - lookback
         in [ (V.slice i lookback xsV, xsV V.! (i + lookback))
            | i <- [0 .. count - 1]
            ]

{- | Multivariate supervised windows from @d@ parallel channels (channel 0 is
the primary/target series). Each window is flattened timestep-major
(@[t0c0,t0c1,...,t1c0,...]@) to match 'forwardWindowDV'; the target is the
primary channel's next value. With a single channel this is identical to
'buildSequencesV'.
-}
buildSequencesMultiV :: Int -> [V.Vector Double] -> [(V.Vector Double, Double)]
buildSequencesMultiV lookback channels
    | lookback <= 0 = []
    | null channels = []
    | otherwise =
        let d = length channels
            c0 = head channels
            n = minimum (map V.length channels)
         in if n <= lookback
                then []
                else
                    [ ( V.fromList [(channels !! ci) V.! (i + t) | t <- [0 .. lookback - 1], ci <- [0 .. d - 1]]
                      , c0 V.! (i + lookback)
                      )
                    | i <- [0 .. n - lookback - 1]
                    ]

evaluateLoss :: Int -> Int -> [([Double], Double)] -> [Double] -> Double
evaluateLoss lookback hidden dataset flat =
    let datasetV = map (Data.Bifunctor.first V.fromList) dataset
     in realToFrac (lossFromFlatV lookback hidden datasetV flat)

evaluateLossV :: Int -> Int -> [(V.Vector Double, Double)] -> [Double] -> Double
evaluateLossV lookback hidden dataset flat =
    realToFrac (lossFromFlatV lookback hidden dataset flat)

trainLSTM :: LSTMConfig -> [Double] -> (LSTMModel, [EpochStats])
trainLSTM cfg series =
    let lookback = lcLookback cfg
        hidden = max 0 (lcHiddenSize cfg)
        dataset = buildSequencesV lookback (V.fromList series)
     in if lookback <= 0 || hidden <= 0 || null dataset
            then (LSTMModel{lmHiddenSize = hidden, lmParams = replicate (paramCount hidden) 0}, [])
            else
                let (trainSet, valSet) = splitTrainVal (lcValRatio cfg) dataset
                    initFlat = initParams (paramCount hidden) (lcSeed cfg)
                    (bestFlat, history) = trainLoop cfg 1 hidden trainSet valSet initFlat
                 in (LSTMModel{lmHiddenSize = hidden, lmParams = bestFlat}, history)

{- | Train a multivariate LSTM over @d@ parallel channels (channel 0 is the
target series). With a single channel this is identical to 'trainLSTM'. The
resulting model's input dimension is recoverable via 'inputDimFromModel', so no
extra metadata or persistence-format change is needed.
-}
trainLSTMMulti :: LSTMConfig -> [[Double]] -> (LSTMModel, [EpochStats])
trainLSTMMulti cfg channels =
    let lookback = lcLookback cfg
        hidden = max 0 (lcHiddenSize cfg)
        d = length channels
        chans = map V.fromList channels
        dataset = buildSequencesMultiV lookback chans
     in if lookback <= 0 || hidden <= 0 || d <= 0 || null dataset
            then (LSTMModel{lmHiddenSize = hidden, lmParams = replicate (paramCountD hidden (max 1 d)) 0}, [])
            else
                let (trainSet, valSet) = splitTrainVal (lcValRatio cfg) dataset
                    initFlat = initParams (paramCountD hidden d) (lcSeed cfg)
                    (bestFlat, history) = trainLoop cfg d hidden trainSet valSet initFlat
                 in (LSTMModel{lmHiddenSize = hidden, lmParams = bestFlat}, history)

fineTuneLSTM :: LSTMConfig -> LSTMModel -> [Double] -> (LSTMModel, [EpochStats])
fineTuneLSTM cfg model series =
    let lookback = lcLookback cfg
        hidden = lcHiddenSize cfg
        dataset = buildSequencesV lookback (V.fromList series)
     in if hidden /= lmHiddenSize model || hidden <= 0 || lookback <= 0 || null dataset || lcEpochs cfg <= 0
            then (model, [])
            else
                let (trainSet, valSet) = splitTrainVal (lcValRatio cfg) dataset
                    initFlat = lmParams model
                    (bestFlat, history) = trainLoop cfg 1 hidden trainSet valSet initFlat
                 in (LSTMModel{lmHiddenSize = hidden, lmParams = bestFlat}, history)

predictNext :: LSTMModel -> [Double] -> Double
predictNext model window =
    predictNextV model (V.fromList window)

predictNextV :: LSTMModel -> V.Vector Double -> Double
predictNextV model window =
    let d = inputDimFromModel model
        params = unflattenParamsD (lmHiddenSize model) d (map realToFrac (lmParams model))
        win = V.map realToFrac window
        y = forwardWindowDV d params win
     in realToFrac y

predictSeriesNext :: LSTMModel -> Int -> [Double] -> [Double]
predictSeriesNext model lookback xs =
    let seqs = buildSequencesV lookback (V.fromList xs)
     in map (\(w, _) -> predictNextV model w) seqs

{- | Multivariate next-step prediction. @channels@ holds @d@ parallel channels,
each the last @lookback@ values ending at the current bar (channel 0 is the
primary/target channel). Channels are interleaved timestep-major into the flat
window the model expects. Falls back to the primary channel alone if shapes are
inconsistent.
-}
predictNextMulti :: LSTMModel -> [[Double]] -> Double
predictNextMulti model channels =
    case channels of
        [] -> 0
        (c0 : _) ->
            let lens = map length channels
                ok = not (null channels) && all (== head lens) lens && head lens > 0
                flat =
                    if ok
                        then [v | t <- [0 .. head lens - 1], ch <- channels, let v = ch !! t]
                        else c0
             in predictNextV model (V.fromList flat)

-- Internal helpers

initParams :: Int -> Int -> [Double]
initParams n seed =
    take n (randomRs (-0.08, 0.08) (mkStdGen seed))

splitTrainVal :: Double -> [a] -> ([a], [a])
splitTrainVal valRatio xs
    | not (isFiniteDouble valRatio) || valRatio <= 0 || valRatio >= 1 = (xs, [])
    | otherwise =
        let n = length xs
            splitAtN = max 1 (floor (fromIntegral n * (1 - valRatio)))
         in splitAt splitAtN xs

-- | The third argument is the input dimension @d@ (1 for the univariate model).
trainLoop :: LSTMConfig -> Int -> Int -> [(V.Vector Double, Double)] -> [(V.Vector Double, Double)] -> [Double] -> ([Double], [EpochStats])
trainLoop cfg d hidden trainSet valSet initFlat =
    let beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = lcLearningRate cfg
        clip = lcGradClip cfg
        maxEpochs = lcEpochs cfg
        patience = max 0 (lcPatience cfg)

        zeroVec = replicate (length initFlat) 0.0

        go epoch flat m v bestFlat bestValLoss wait history =
            if epoch > maxEpochs
                then (bestFlat, reverse history)
                else
                    let trainLoss = realToFrac (lossFromFlatDV d hidden trainSet flat)
                        valLoss = if null valSet then trainLoss else realToFrac (lossFromFlatDV d hidden valSet flat)

                        history' = EpochStats epoch trainLoss valLoss : history

                        (bestFlat', bestValLoss', wait') =
                            if valLoss < bestValLoss
                                then (flat, valLoss, 0)
                                else (bestFlat, bestValLoss, wait + 1)

                        shouldStop = patience > 0 && wait' >= patience
                     in if shouldStop
                            then (bestFlat', reverse history')
                            else
                                let g = grad (lossFromFlatDV d hidden trainSet) flat
                                    g' =
                                        case clip of
                                            Nothing -> g
                                            Just c -> clipByL2 c g
                                    t = fromIntegral epoch
                                    m' = zipWith (\mi gi -> beta1 * mi + (1 - beta1) * gi) m g'
                                    v' = zipWith (\vi gi -> beta2 * vi + (1 - beta2) * gi * gi) v g'
                                    mHat = map (\mi -> mi / (1 - beta1 ** t)) m'
                                    vHat = map (\vi -> vi / (1 - beta2 ** t)) v'
                                    stepVec = zipWith (\mh vh -> lr * mh / (sqrt vh + eps)) mHat vHat
                                    flat' = zipWith (-) flat stepVec
                                 in go (epoch + 1) flat' m' v' bestFlat' bestValLoss' wait' history'
     in go 1 initFlat zeroVec zeroVec initFlat (1 / 0) 0 []

clipByL2 :: Double -> [Double] -> [Double]
clipByL2 maxNorm g =
    let norm = sqrt (sum (map (\x -> x * x) g))
     in if norm > maxNorm && norm > 0
            then map (\x -> x * (maxNorm / norm)) g
            else g

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)

data LSTMParams a = LSTMParams
    { pWi :: [[a]]
    , pBi :: [a]
    , pWf :: [[a]]
    , pBf :: [a]
    , pWo :: [[a]]
    , pBo :: [a]
    , pWg :: [[a]]
    , pBg :: [a]
    , pWy :: [a]
    , pBy :: a
    }

unflattenParams :: (Num a) => Int -> [a] -> LSTMParams a
unflattenParams h = unflattenParamsD h 1

{- | Unflatten parameters for hidden size @h@ and input dimension @d@. Each gate
weight matrix is @h x (h+d)@. @unflattenParamsD h 1@ matches the original
univariate layout exactly.
-}
unflattenParamsD :: (Num a) => Int -> Int -> [a] -> LSTMParams a
unflattenParamsD h d xs =
    let cols = h + d
        wSize = h * cols
        splitN n ys = let (a, b) = splitAt n ys in (a, b)
        chunk n = takeWhile (not . null) . map (take n) . iterate (drop n)

        (wiFlat, r1) = splitN wSize xs
        (bi, r2) = splitN h r1
        (wfFlat, r3) = splitN wSize r2
        (bf, r4) = splitN h r3
        (woFlat, r5) = splitN wSize r4
        (bo, r6) = splitN h r5
        (wgFlat, r7) = splitN wSize r6
        (bg, r8) = splitN h r7
        (wy, r9) = splitN h r8
        by =
            case r9 of
                (v : _) -> v
                [] -> 0
     in LSTMParams
            { pWi = chunk cols wiFlat
            , pBi = bi
            , pWf = chunk cols wfFlat
            , pBf = bf
            , pWo = chunk cols woFlat
            , pBo = bo
            , pWg = chunk cols wgFlat
            , pBg = bg
            , pWy = wy
            , pBy = by
            }

lossFromFlatV :: (Floating a) => Int -> Int -> [(V.Vector Double, Double)] -> [a] -> a
lossFromFlatV _lookback hidden = lossFromFlatDV 1 hidden

{- | Mean-squared error of the model over a dataset of (flat window, target),
where each window holds @lookback@ timesteps of @d@ interleaved channels
(timestep-major: @[t0c0,t0c1,...,t1c0,...]@). @d == 1@ is the univariate case.
-}
lossFromFlatDV :: (Floating a) => Int -> Int -> [(V.Vector Double, Double)] -> [a] -> a
lossFromFlatDV d hidden dataset flat =
    let p = unflattenParamsD hidden d flat
        nInt = length dataset
        n = fromIntegral nInt
        err (w, yTrue) =
            let wA = V.map realToFrac w
                yA = realToFrac yTrue
                yPred = forwardWindowDV d p wA
                e = yPred - yA
             in e * e
     in if nInt <= 0
            then 0
            else foldl' (\acc x -> acc + err x) 0 dataset / n

forwardWindowV :: (Floating a) => LSTMParams a -> V.Vector a -> a
forwardWindowV = forwardWindowDV 1

{- | Forward pass over a flat window of @d@ interleaved channels. With @d == 1@
each step feeds a single scalar, identical to the original univariate fold.
-}
forwardWindowDV :: (Floating a) => Int -> LSTMParams a -> V.Vector a -> a
forwardWindowDV d p xsV =
    let h = length (pBi p)
        h0 = replicate h 0
        c0 = replicate h 0
        dim = max 1 d
        nSteps = V.length xsV `div` dim
        steps = [V.toList (V.slice (k * dim) dim xsV) | k <- [0 .. nSteps - 1]]
        (hT, _) = foldl' (lstmStep p) (h0, c0) steps
     in dot (pWy p) hT + pBy p

forwardWindow :: (Floating a) => LSTMParams a -> [a] -> a
forwardWindow p xs =
    forwardWindowV p (V.fromList xs)

{- | One recurrent step consuming the @d@-dimensional input vector for this
timestep. For @d == 1@, @u == x : hPrev@, matching the original.
-}
lstmStep :: (Floating a) => LSTMParams a -> ([a], [a]) -> [a] -> ([a], [a])
lstmStep p (hPrev, cPrev) xVec =
    let u = xVec ++ hPrev
        i = map sigmoid (zipWith (+) (matVec (pWi p) u) (pBi p))
        f = map sigmoid (zipWith (+) (matVec (pWf p) u) (pBf p))
        o = map sigmoid (zipWith (+) (matVec (pWo p) u) (pBo p))
        g = map tanh (zipWith (+) (matVec (pWg p) u) (pBg p))
        cNew = zipWith (+) (zipWith (*) f cPrev) (zipWith (*) i g)
        hNew = zipWith (*) o (map tanh cNew)
     in (hNew, cNew)

sigmoid :: (Floating a) => a -> a
sigmoid x = 1 / (1 + exp (-x))

matVec :: (Num a) => [[a]] -> [a] -> [a]
matVec m v = map (dot v) m

dot :: (Num a) => [a] -> [a] -> a
dot a b = sum (zipWith (*) a b)

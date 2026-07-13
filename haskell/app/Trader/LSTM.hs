module Trader.LSTM (
    LSTMConfig (..),
    EpochStats (..),
    LSTMModel (..),
    defaultLstmAdamBeta1,
    defaultLstmAdamBeta2,
    defaultLstmAdamEps,
    paramCount,
    paramCountD,
    inputDimFromModel,
    buildSequences,
    buildSequencesMultiV,
    evaluateLoss,
    trainLSTM,
    trainLSTMMulti,
    fineTuneLSTM,
    fineTuneLSTMWeighted,
    predictNext,
    predictNextMulti,
    predictSeriesNext,
) where

import qualified Data.Bifunctor
import Data.List (foldl')
import Data.Maybe (fromMaybe)
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VU
import Numeric.AD (grad)
import System.Random (mkStdGen, randomRs)

import Trader.LstmDefaults (defaultLstmAdamBeta1, defaultLstmAdamBeta2, defaultLstmAdamEps)

data LSTMConfig = LSTMConfig
    { lcLookback :: !Int
    , lcHiddenSize :: !Int
    , lcEpochs :: !Int
    , lcLearningRate :: !Double
    , lcAdamBeta1 :: !Double
    , lcAdamBeta2 :: !Double
    , lcAdamEps :: !Double
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
    fromMaybe 1 (validInputDimFromModel model)

validInputDimFromModel :: LSTMModel -> Maybe Int
validInputDimFromModel model =
    let h = lmHiddenSize model
        len = length (lmParams model)
        fixed = 5 * h + 1
        divisor = 4 * h
        variable = len - fixed
        d = if divisor <= 0 then 0 else variable `div` divisor - h
     in if h > 0 && variable >= 0 && variable `mod` divisor == 0 && d > 0 && paramCountD h d == len
            then Just d
            else Nothing

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

{- | Supervised windows with a per-sample loss weight. The sample whose target
is @xsV ! t@ gets weight @wsV ! t@; indices beyond the weight vector and
non-finite or negative weights default to 1.
-}
buildSequencesWeightedV :: Int -> V.Vector Double -> V.Vector Double -> [(V.Vector Double, Double, Double)]
buildSequencesWeightedV lookback xsV wsV
    | lookback <= 0 = []
    | V.length xsV <= lookback = []
    | otherwise =
        let count = V.length xsV - lookback
            weightAt t =
                if t < V.length wsV
                    then sanitizeWeight (wsV V.! t)
                    else 1
            sanitizeWeight w
                | isNaN w || isInfinite w || w < 0 = 1
                | otherwise = w
         in [ (V.slice i lookback xsV, xsV V.! (i + lookback), weightAt (i + lookback))
            | i <- [0 .. count - 1]
            ]

unitWeights :: [(V.Vector Double, Double)] -> [(V.Vector Double, Double, Double)]
unitWeights = map (\(w, y) -> (w, y, 1))

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
                let (trainSet, valSet) = splitTrainVal (lcValRatio cfg) (unitWeights dataset)
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
                let (trainSet, valSet) = splitTrainVal (lcValRatio cfg) (unitWeights dataset)
                    initFlat = initParams (paramCountD hidden d) (lcSeed cfg)
                    (bestFlat, history) = trainLoop cfg d hidden trainSet valSet initFlat
                 in (LSTMModel{lmHiddenSize = hidden, lmParams = bestFlat}, history)

fineTuneLSTM :: LSTMConfig -> LSTMModel -> [Double] -> (LSTMModel, [EpochStats])
fineTuneLSTM cfg model series =
    fineTuneLSTMWeighted cfg model series []

{- | Fine-tune with a per-observation loss weight. @weights@ aligns index-for-
index with @series@; the supervised sample whose target is @series !! t@ gets
weight @weights !! t@. Missing or non-finite entries default to 1, so the
empty list reproduces plain 'fineTuneLSTM' exactly.

This is the hook for outcome-driven learning: bars covered by a closed trade
carry extra weight, so the model is pulled hardest toward the true prices
exactly where its predictions were acted on — correcting itself where a trade
lost (punishment) and consolidating the patterns where a trade won
(reinforcement) — while the objective stays supervised toward observed
prices, so the feedback cannot inflate the model's own predictions.
-}
fineTuneLSTMWeighted :: LSTMConfig -> LSTMModel -> [Double] -> [Double] -> (LSTMModel, [EpochStats])
fineTuneLSTMWeighted cfg model series weights =
    let lookback = lcLookback cfg
        hidden = lcHiddenSize cfg
        dataset = buildSequencesWeightedV lookback (V.fromList series) (V.fromList weights)
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
    case validInputDimFromModel model of
        Nothing -> 0
        Just d
            | V.null window || V.length window `mod` d /= 0 -> 0
            | not (all isFiniteDouble (lmParams model)) || V.any (not . isFiniteDouble) window -> 0
            | otherwise ->
                let params = unflattenParamsD (lmHiddenSize model) d (map realToFrac (lmParams model))
                    win = V.map realToFrac window
                    y = realToFrac (forwardWindowDV d params win)
                 in if isFiniteDouble y then y else 0

predictSeriesNext :: LSTMModel -> Int -> [Double] -> [Double]
predictSeriesNext model lookback xs =
    let seqs = buildSequencesV lookback (V.fromList xs)
     in map (\(w, _) -> predictNextV model w) seqs

{- | Multivariate next-step prediction. @channels@ holds @d@ parallel channels,
each the last @lookback@ values ending at the current bar (channel 0 is the
primary/target channel). Channels are interleaved timestep-major into the flat
window the model expects. A channel-count or shape mismatch fails closed to 0;
it must not reinterpret a multivariate model as a differently shaped model.
-}
predictNextMulti :: LSTMModel -> [[Double]] -> Double
predictNextMulti model channels =
    case validInputDimFromModel model of
        Nothing -> 0
        Just d ->
            let lens = map length channels
                shapeOk =
                    length channels == d
                        && not (null lens)
                        && head lens > 0
                        && all (== head lens) lens
                valuesOk = all (all isFiniteDouble) channels
                flat = [v | t <- [0 .. head lens - 1], ch <- channels, let v = ch !! t]
             in if shapeOk && valuesOk then predictNextV model (V.fromList flat) else 0

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

{- | The third argument is the input dimension @d@ (1 for the univariate
model). Datasets carry a per-sample loss weight; weight 1 everywhere is
exactly the unweighted MSE.
-}
trainLoop :: LSTMConfig -> Int -> Int -> [(V.Vector Double, Double, Double)] -> [(V.Vector Double, Double, Double)] -> [Double] -> ([Double], [EpochStats])
trainLoop cfg d hidden trainSet valSet initFlat =
    let beta1 = adamBetaOrDefault defaultLstmAdamBeta1 (lcAdamBeta1 cfg)
        beta2 = adamBetaOrDefault defaultLstmAdamBeta2 (lcAdamBeta2 cfg)
        eps = adamEpsOrDefault defaultLstmAdamEps (lcAdamEps cfg)
        lr = lcLearningRate cfg
        clip = lcGradClip cfg
        maxEpochs = lcEpochs cfg
        patience = max 0 (lcPatience cfg)

        zeroVec = replicate (length initFlat) 0.0

        go epoch flat m v bestFlat bestValLoss wait history =
            if epoch > maxEpochs
                then (bestFlat, reverse history)
                else
                    let trainLoss = realToFrac (lossFromFlatWeightedDV d hidden trainSet flat)
                        valLoss = if null valSet then trainLoss else realToFrac (lossFromFlatWeightedDV d hidden valSet flat)

                        history' = EpochStats epoch trainLoss valLoss : history

                        (bestFlat', bestValLoss', wait') =
                            if valLoss < bestValLoss
                                then (flat, valLoss, 0)
                                else (bestFlat, bestValLoss, wait + 1)

                        shouldStop = patience > 0 && wait' >= patience
                     in if shouldStop
                            then (bestFlat', reverse history')
                            else
                                let g = gradWeightedPerSample d hidden trainSet flat
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

adamBetaOrDefault :: Double -> Double -> Double
adamBetaOrDefault fallback value
    | isFiniteDouble value && value >= 0 && value < 1 = value
    | otherwise = fallback

adamEpsOrDefault :: Double -> Double -> Double
adamEpsOrDefault fallback value
    | isFiniteDouble value && value > 0 = value
    | otherwise = fallback

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
lossFromFlatV _lookback = lossFromFlatDV 1

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

{- | Gradient of 'lossFromFlatWeightedDV' computed sample-by-sample.

Reverse-mode AD materializes a tape node for every arithmetic operation under
the differentiated function. Differentiating the whole-dataset loss in one
call (the old @grad (lossFromFlatWeightedDV ...)@) therefore held
O(samples x lookback x hidden^2) boxed tape nodes live at once — gigabytes at
research shapes (lookback 720, hidden 64), which is what OOM-killed LSTM
optimizer trials. Differentiating per sample keeps the tape at one window and
lets each one be collected before the next, with the identical result:
@d/dp [ sum_i (w_i e_i^2) / sum w ] = sum_i d/dp (w_i e_i^2) / sum w@.
The per-sample gradients are accumulated in the same left-to-right sample
order the dataset fold used, into a strict unboxed vector so no thunk chains
rebuild the leak.
-}
gradWeightedPerSample :: Int -> Int -> [(V.Vector Double, Double, Double)] -> [Double] -> [Double]
gradWeightedPerSample d hidden dataset flat =
    let nParams = length flat
        sumW = foldl' (\acc (_, _, w) -> acc + w) 0 dataset :: Double
        sampleLoss s fl =
            let p = unflattenParamsD hidden d fl
                (w, yTrue, wt) = s
                yPred = forwardWindowDV d p (V.map realToFrac w)
                e = yPred - realToFrac yTrue
             in realToFrac wt * e * e
        accum acc s = VU.zipWith (+) acc (VU.fromListN nParams (grad (sampleLoss s) flat))
        total = foldl' accum (VU.replicate nParams 0) dataset
     in if null dataset || sumW <= 0 || isNaN sumW || isInfinite sumW
            then replicate nParams 0
            else VU.toList (VU.map (/ sumW) total)

{- | Weighted mean-squared error: @sum(w_i * e_i^2) / sum(w_i)@. With unit
weights this is numerically identical to 'lossFromFlatDV' (multiplying by the
constant 1.0 and summing unit weights are exact float operations).
-}
lossFromFlatWeightedDV :: (Floating a) => Int -> Int -> [(V.Vector Double, Double, Double)] -> [a] -> a
lossFromFlatWeightedDV d hidden dataset flat =
    let p = unflattenParamsD hidden d flat
        sumW = foldl' (\acc (_, _, w) -> acc + w) 0 dataset :: Double
        err (w, yTrue, wt) =
            let wA = V.map realToFrac w
                yA = realToFrac yTrue
                yPred = forwardWindowDV d p wA
                e = yPred - yA
             in realToFrac wt * e * e
     in if null dataset || sumW <= 0 || isNaN sumW || isInfinite sumW
            then 0
            else foldl' (\acc x -> acc + err x) 0 dataset / realToFrac sumW

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

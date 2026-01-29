module Trader.LSTM (
    LSTMConfig (..),
    EpochStats (..),
    LSTMModel (..),
    paramCount,
    buildSequences,
    evaluateLoss,
    trainLSTM,
    fineTuneLSTM,
    predictNext,
    predictSeriesNext,
) where

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
paramCount h =
    let gate = h * (h + 1) + h -- W (h x (h+1)) plus b (h)
     in 4 * gate + h + 1 -- output head

buildSequences :: Int -> [Double] -> [([Double], Double)]
buildSequences lookback xs
    | lookback <= 0 = error "lookback must be positive"
    | length xs <= lookback = error "not enough data for lookback"
    | otherwise =
        let xsV = V.fromList xs
         in map (\(w, y) -> (V.toList w, y)) (buildSequencesV lookback xsV)

buildSequencesV :: Int -> V.Vector Double -> [(V.Vector Double, Double)]
buildSequencesV lookback xsV
    | lookback <= 0 = error "lookback must be positive"
    | V.length xsV <= lookback = error "not enough data for lookback"
    | otherwise =
        let count = V.length xsV - lookback
         in [ (V.slice i lookback xsV, xsV V.! (i + lookback))
            | i <- [0 .. count - 1]
            ]

evaluateLoss :: Int -> Int -> [([Double], Double)] -> [Double] -> Double
evaluateLoss lookback hidden dataset flat =
    let datasetV = map (\(w, y) -> (V.fromList w, y)) dataset
     in realToFrac (lossFromFlatV lookback hidden datasetV flat)

evaluateLossV :: Int -> Int -> [(V.Vector Double, Double)] -> [Double] -> Double
evaluateLossV lookback hidden dataset flat =
    realToFrac (lossFromFlatV lookback hidden dataset flat)

trainLSTM :: LSTMConfig -> [Double] -> (LSTMModel, [EpochStats])
trainLSTM cfg series =
    let lookback = lcLookback cfg
        hidden = lcHiddenSize cfg
        dataset = buildSequencesV lookback (V.fromList series)
        (trainSet, valSet) = splitTrainVal (lcValRatio cfg) dataset

        initFlat = initParams (paramCount hidden) (lcSeed cfg)

        (bestFlat, history) = trainLoop cfg lookback hidden trainSet valSet initFlat
     in (LSTMModel{lmHiddenSize = hidden, lmParams = bestFlat}, history)

fineTuneLSTM :: LSTMConfig -> LSTMModel -> [Double] -> (LSTMModel, [EpochStats])
fineTuneLSTM cfg model series =
    let lookback = lcLookback cfg
        hidden = lcHiddenSize cfg
     in if hidden /= lmHiddenSize model
            then error "fineTuneLSTM: hidden size mismatch"
            else
                let dataset = buildSequencesV lookback (V.fromList series)
                    (trainSet, valSet) = splitTrainVal (lcValRatio cfg) dataset
                    initFlat = lmParams model
                    (bestFlat, history) = trainLoop cfg lookback hidden trainSet valSet initFlat
                 in (LSTMModel{lmHiddenSize = hidden, lmParams = bestFlat}, history)

predictNext :: LSTMModel -> [Double] -> Double
predictNext model window =
    predictNextV model (V.fromList window)

predictNextV :: LSTMModel -> V.Vector Double -> Double
predictNextV model window =
    let params = unflattenParams (lmHiddenSize model) (map realToFrac (lmParams model))
        win = V.map realToFrac window
        y = forwardWindowV params win
     in realToFrac y

predictSeriesNext :: LSTMModel -> Int -> [Double] -> [Double]
predictSeriesNext model lookback xs =
    let seqs = buildSequencesV lookback (V.fromList xs)
     in map (\(w, _) -> predictNextV model w) seqs

-- Internal helpers

initParams :: Int -> Int -> [Double]
initParams n seed =
    take n (randomRs (-0.08, 0.08) (mkStdGen seed))

splitTrainVal :: Double -> [a] -> ([a], [a])
splitTrainVal valRatio xs
    | valRatio <= 0 || valRatio >= 1 = (xs, [])
    | otherwise =
        let n = length xs
            splitAtN = max 1 (floor (fromIntegral n * (1 - valRatio)))
         in splitAt splitAtN xs

trainLoop :: LSTMConfig -> Int -> Int -> [(V.Vector Double, Double)] -> [(V.Vector Double, Double)] -> [Double] -> ([Double], [EpochStats])
trainLoop cfg lookback hidden trainSet valSet initFlat =
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
                    let trainLoss = evaluateLossV lookback hidden trainSet flat
                        valLoss = if null valSet then trainLoss else evaluateLossV lookback hidden valSet flat

                        history' = EpochStats epoch trainLoss valLoss : history

                        (bestFlat', bestValLoss', wait') =
                            if valLoss < bestValLoss
                                then (flat, valLoss, 0)
                                else (bestFlat, bestValLoss, wait + 1)

                        shouldStop = patience > 0 && wait' >= patience
                     in if shouldStop
                            then (bestFlat', reverse history')
                            else
                                let g = grad (lossFromFlatV lookback hidden trainSet) flat
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

unflattenParams :: Int -> [a] -> LSTMParams a
unflattenParams h xs =
    let wSize = h * (h + 1)
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
                [] -> error "missing output bias"
     in LSTMParams
            { pWi = chunk (h + 1) wiFlat
            , pBi = bi
            , pWf = chunk (h + 1) wfFlat
            , pBf = bf
            , pWo = chunk (h + 1) woFlat
            , pBo = bo
            , pWg = chunk (h + 1) wgFlat
            , pBg = bg
            , pWy = wy
            , pBy = by
            }

lossFromFlatV :: (Floating a) => Int -> Int -> [(V.Vector Double, Double)] -> [a] -> a
lossFromFlatV lookback hidden dataset flat =
    let p = unflattenParams hidden flat
        nInt = length dataset
        n = fromIntegral nInt
        err (w, yTrue) =
            let wA = V.map realToFrac w
                yA = realToFrac yTrue
                yPred = forwardWindowV p wA
                e = yPred - yA
             in e * e
     in if nInt <= 0
            then 0
            else foldl' (\acc x -> acc + err x) 0 dataset / n

forwardWindowV :: (Floating a) => LSTMParams a -> V.Vector a -> a
forwardWindowV p xsV =
    let h = length (pBi p)
        h0 = replicate h 0
        c0 = replicate h 0
        (hT, _) = V.foldl' (lstmStep p) (h0, c0) xsV
     in dot (pWy p) hT + pBy p

forwardWindow :: (Floating a) => LSTMParams a -> [a] -> a
forwardWindow p xs =
    forwardWindowV p (V.fromList xs)

lstmStep :: (Floating a) => LSTMParams a -> ([a], [a]) -> a -> ([a], [a])
lstmStep p (hPrev, cPrev) x =
    let u = x : hPrev
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

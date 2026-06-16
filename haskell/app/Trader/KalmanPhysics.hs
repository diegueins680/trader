module Trader.KalmanPhysics (
    KalmanPhysicsConfig (..),
    OhlcvBar (..),
    defaultKalmanPhysicsConfig,
    predictKalmanPhysicsError,
    predictKalmanPhysicsErrorWithConfig,
) where

import Data.List (foldl', minimumBy)
import Data.Ord (comparing)
import qualified Data.Vector as V

import Trader.Kalman3 (
    Kalman3,
    Vec3 (..),
    constantAcceleration1D,
    kx,
    predict,
    update,
 )
import Trader.Predictors.GBDT (GBDTModel (..), predictGBDT, trainGBDT)

data OhlcvBar = OhlcvBar
    { obOpen :: !Double
    , obHigh :: !Double
    , obLow :: !Double
    , obClose :: !Double
    , obVolume :: !Double
    }
    deriving (Eq, Show)

data FeatureMode
    = FeatureDeltaAccel
    | FeatureJerk
    deriving (Eq, Show)

data Candidate = Candidate
    { cdMode :: !FeatureMode
    , cdTrees :: !Int
    , cdLearningRate :: !Double
    }
    deriving (Eq, Show)

data KalmanPhysicsConfig = KalmanPhysicsConfig
    { kpcVolumeEwmaAlpha :: !Double
    , kpcVolumeSignalClamp :: !Double
    , kpcCloseBiasScale :: !Double
    }
    deriving (Eq, Show)

defaultKalmanPhysicsConfig :: KalmanPhysicsConfig
defaultKalmanPhysicsConfig =
    KalmanPhysicsConfig
        { kpcVolumeEwmaAlpha = 0.1
        , kpcVolumeSignalClamp = 3
        , kpcCloseBiasScale = 0.05
        }

data PhysicsRow = PhysicsRow
    { prVelocity :: !Double
    , prAcceleration :: !Double
    , prDeltaAccel :: !Double
    , prJerk :: !Double
    , prCurrentClose :: !Double
    , prNextClose :: !Double
    , prPhysicsPred :: !Double
    , prDiscriminantTrain :: !Double
    , prTargetError :: !Double
    }
    deriving (Eq, Show)

predictKalmanPhysicsError ::
    Double ->
    Double ->
    Double ->
    Int ->
    V.Vector OhlcvBar ->
    Either String [Double]
predictKalmanPhysicsError =
    predictKalmanPhysicsErrorWithConfig defaultKalmanPhysicsConfig

predictKalmanPhysicsErrorWithConfig ::
    KalmanPhysicsConfig ->
    Double ->
    Double ->
    Double ->
    Int ->
    V.Vector OhlcvBar ->
    Either String [Double]
predictKalmanPhysicsErrorWithConfig cfg dt processVar measurementVar trainEnd barsV
    | n < 3 = Left "Need at least 3 bars for Kalman physics error prediction."
    | trainEnd <= 1 = Left "Kalman physics train split is too small (need at least 2 training bars)."
    | trainEnd >= n - 1 = Left "Kalman physics train split leaves no test rows."
    | otherwise =
        let statesV = runKalmanStates cfg dt processVar measurementVar barsV
            rows = buildPhysicsRows dt barsV statesV
            trainRows = take (trainEnd - 1) rows
            testRows = drop trainEnd rows
         in if null testRows
                then Right []
                else
                    if null trainRows
                        then Right (map prPhysicsPred testRows)
                        else
                            let best = chooseBestCandidate trainRows
                                model = fitCandidate best trainRows
                                preds = map (predictRow model (cdMode best)) testRows
                             in Right preds
  where
    n = V.length barsV

runKalmanStates :: KalmanPhysicsConfig -> Double -> Double -> Double -> V.Vector OhlcvBar -> V.Vector Vec3
runKalmanStates cfg dt processVar measurementVar barsV =
    let n = V.length barsV
     in if n <= 0
            then V.empty
            else
                let firstBar = barsV V.! 0
                    k0 = constantAcceleration1D dt processVar measurementVar (obClose firstBar)
                    volEwma0 = max 1e-9 (abs (obVolume firstBar))
                    (k0', volEwma0') = assimilateBar cfg k0 volEwma0 firstBar
                    step (kPrev, volEwma, acc) bar =
                        let kPred = predict kPrev
                            (kUpd, volEwma') = assimilateBar cfg kPred volEwma bar
                         in (kUpd, volEwma', kx kUpd : acc)
                    tailBars =
                        if n <= 1
                            then V.empty
                            else V.slice 1 (n - 1) barsV
                    (_, _, statesRev) =
                        V.foldl'
                            step
                            (k0', volEwma0', [kx k0'])
                            tailBars
                 in V.fromList (reverse statesRev)

assimilateBar ::
    KalmanPhysicsConfig ->
    Kalman3 ->
    Double ->
    OhlcvBar ->
    (Kalman3, Double)
assimilateBar cfg k volEwma bar =
    let alpha = clamp 0 1 (kpcVolumeEwmaAlpha cfg)
        signalClamp = max 0 (kpcVolumeSignalClamp cfg)
        closeBiasScale = kpcCloseBiasScale cfg
        vRaw = max 0 (obVolume bar)
        volBase = if volEwma > 0 then volEwma else max 1e-9 vRaw
        volEwma' = (1 - alpha) * volBase + alpha * max 1e-9 vRaw
        volNorm = (vRaw - volEwma') / max 1e-9 volEwma'
        volSignal = clamp (-signalClamp) signalClamp volNorm
        priceImbalance = obClose bar - obOpen bar
        closeBias = obClose bar + closeBiasScale * priceImbalance * volSignal
        measurements = [obOpen bar, obHigh bar, obLow bar, obClose bar, closeBias]
        k' = foldl' (flip update) k measurements
     in (k', volEwma')

buildPhysicsRows :: Double -> V.Vector OhlcvBar -> V.Vector Vec3 -> [PhysicsRow]
buildPhysicsRows dt barsV statesV =
    let n = min (V.length barsV) (V.length statesV)
        dtSafe = max 1e-6 dt
        rowAt t =
            let Vec3 _ v a = statesV V.! t
                prevA =
                    if t <= 0
                        then a
                        else
                            let Vec3 _ _ aPrev = statesV V.! (t - 1)
                             in aPrev
                da = a - prevA
                jerk = da / dtSafe
                c = obClose (barsV V.! t)
                cNext = obClose (barsV V.! (t + 1))
                physicsPred = c + v + 0.5 * a
                discrim = v * v
                targetErr = cNext - physicsPred
             in PhysicsRow
                    { prVelocity = v
                    , prAcceleration = a
                    , prDeltaAccel = da
                    , prJerk = jerk
                    , prCurrentClose = c
                    , prNextClose = cNext
                    , prPhysicsPred = physicsPred
                    , prDiscriminantTrain = discrim
                    , prTargetError = targetErr
                    }
     in if n <= 1 then [] else map rowAt [0 .. n - 2]

chooseBestCandidate :: [PhysicsRow] -> Candidate
chooseBestCandidate trainRows =
    let candidates =
            [ Candidate FeatureDeltaAccel 40 0.05
            , Candidate FeatureDeltaAccel 80 0.08
            , Candidate FeatureDeltaAccel 120 0.05
            , Candidate FeatureJerk 40 0.05
            , Candidate FeatureJerk 80 0.08
            , Candidate FeatureJerk 120 0.05
            ]
        scored = map (\c -> (scoreCandidate c trainRows, c)) candidates
        bad x = isNaN x || isInfinite x
        scoreOrInf x = if bad x then 1 / 0 else x
     in snd (minimumBy (comparing (scoreOrInf . fst)) scored)

scoreCandidate :: Candidate -> [PhysicsRow] -> Double
scoreCandidate cand rows =
    let total = length rows
        valCount =
            if total < 5
                then max 1 (total `div` 3)
                else max 1 (total `div` 5)
        trainCount = max 1 (total - valCount)
        trainRows = take trainCount rows
        valRows = drop trainCount rows
        model = fitCandidate cand trainRows
        evalRows = if null valRows then trainRows else valRows
     in if null evalRows
            then 1 / 0
            else rmse [predictRow model (cdMode cand) r - prNextClose r | r <- evalRows]

fitCandidate :: Candidate -> [PhysicsRow] -> GBDTModel
fitCandidate cand rows =
    let dataset = [(trainingFeatures (cdMode cand) r, prTargetError r) | r <- rows]
     in trainGBDT (cdTrees cand) (cdLearningRate cand) dataset

predictRow :: GBDTModel -> FeatureMode -> PhysicsRow -> Double
predictRow model mode row =
    let base = prPhysicsPred row
     in if gmFeatureDim model <= 0
            then base
            else
                let (errPred, _sig) = predictGBDT model (inferenceFeatures mode row)
                 in base + errPred

trainingFeatures :: FeatureMode -> PhysicsRow -> [Double]
trainingFeatures mode row =
    [ prVelocity row
    , prAcceleration row
    , selectTerm mode row
    , prDiscriminantTrain row
    ]

inferenceFeatures :: FeatureMode -> PhysicsRow -> [Double]
inferenceFeatures mode row =
    let v = prVelocity row
     in [v, prAcceleration row, selectTerm mode row, v * v]

selectTerm :: FeatureMode -> PhysicsRow -> Double
selectTerm mode row =
    case mode of
        FeatureDeltaAccel -> prDeltaAccel row
        FeatureJerk -> prJerk row

rmse :: [Double] -> Double
rmse xs =
    case xs of
        [] -> 0
        _ ->
            let sq x = x * x
                m = sum (map sq xs) / fromIntegral (length xs)
             in sqrt (max 0 m)

clamp :: Double -> Double -> Double -> Double
clamp lo hi x = max lo (min hi x)

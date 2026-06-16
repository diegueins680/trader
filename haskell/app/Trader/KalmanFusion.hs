module Trader.KalmanFusion (
    Kalman1 (..),
    KalmanFusionConfig (..),
    defaultKalmanFusionConfig,
    initKalman1,
    predict,
    updateMulti,
    measurementVarianceWithResidualFloor,
    inflateCorrelatedMeasurements,
    innovationInflationFactor,
    stepMultiWithConfig,
    stepMulti,
) where

{- | Scalar-state Kalman filter with a multi-sensor observation vector.
State: x_t ∈ R (latent expected forward return).
Observations: y_t ∈ R^m where each sensor measures x_t with independent noise.
-}
data Kalman1 = Kalman1
    { kMean :: !Double
    , kVar :: !Double
    , kProcessVar :: !Double
    }
    deriving (Eq, Show)

data KalmanFusionConfig = KalmanFusionConfig
    { kfcSensorCorrelationInflation :: !Double
    , kfcInnovationInflationThreshold :: !Double
    , kfcInnovationInflationMax :: !Double
    }
    deriving (Eq, Show)

defaultKalmanFusionConfig :: KalmanFusionConfig
defaultKalmanFusionConfig =
    KalmanFusionConfig
        { kfcSensorCorrelationInflation = 0
        , kfcInnovationInflationThreshold = 0
        , kfcInnovationInflationMax = 10
        }

initKalman1 :: Double -> Double -> Double -> Kalman1
initKalman1 mean0 var0 processVar =
    let eps = 1e-12
        var0' = max eps var0
        processVar' = max 0 processVar
     in Kalman1{kMean = mean0, kVar = var0', kProcessVar = processVar'}

-- | Predict step for a random-walk state model: x_t = x_{t-1} + w, w~N(0,Q).
predict :: Kalman1 -> Kalman1
predict k = k{kVar = kVar k + kProcessVar k}

{- | Multi-sensor measurement update.
Each measurement is (y_i, r_i) where r_i is the measurement variance.
Assumes H is a column of 1s and R is diagonal (independent sensors).
Malformed observations are ignored; if none survive validation, the prior is returned unchanged.
-}
updateMulti :: [(Double, Double)] -> Kalman1 -> Kalman1
updateMulti meas k =
    case foldr accumulateMeasurement (False, 0, 0) meas of
        (False, _, _) -> k
        (True, precSum, meanPrecSum) ->
            let p0 = max eps (kVar k)
                m0 = kMean k
                priorPrec = 1 / p0
                postPrec = priorPrec + precSum
                postVar = 1 / postPrec
                postMean = postVar * (m0 * priorPrec + meanPrecSum)
             in k{kMean = postMean, kVar = postVar}
  where
    eps = 1e-12

    accumulateMeasurement :: (Double, Double) -> (Bool, Double, Double) -> (Bool, Double, Double)
    accumulateMeasurement (y, rRaw) (hasValid, precSum, meanPrecSum)
        | not (isFinite y) = (hasValid, precSum, meanPrecSum)
        | not (isFinite rRaw) = (hasValid, precSum, meanPrecSum)
        | rRaw <= 0 = (hasValid, precSum, meanPrecSum)
        | otherwise =
            let r = max eps rRaw
                pr = 1 / r
             in (True, precSum + pr, meanPrecSum + y * pr)

    isFinite :: Double -> Bool
    isFinite x = not (isNaN x || isInfinite x)

measurementVarianceWithResidualFloor :: Double -> Maybe Double -> Maybe Double -> Double
measurementVarianceWithResidualFloor fallbackVar mSigma mLearnedVar =
    max eps (maximum (fallback : modelVar ++ learnedVar))
  where
    eps = 1e-12
    fallback = max eps fallbackVar
    modelVar =
        case mSigma of
            Just s | isFinite s && s > 0 -> [s * s]
            _ -> []
    learnedVar =
        case mLearnedVar of
            Just v | isFinite v && v > 0 -> [v]
            _ -> []

inflateCorrelatedMeasurements :: KalmanFusionConfig -> [(Double, Double)] -> [(Double, Double)]
inflateCorrelatedMeasurements cfg meas =
    let validCount = length (filter validMeasurement meas)
        rho = clamp 0 1 (kfcSensorCorrelationInflation cfg)
        mult =
            if validCount <= 1
                then 1
                else max 1 (1 + rho * fromIntegral (validCount - 1))
     in map (inflateOne mult) meas
  where
    inflateOne mult (y, r)
        | validMeasurement (y, r) = (y, max eps r * mult)
        | otherwise = (y, r)

innovationInflationFactor :: KalmanFusionConfig -> Kalman1 -> [(Double, Double)] -> Double
innovationInflationFactor cfg prior meas
    | threshold <= 0 = 1
    | maxMult <= 1 = 1
    | otherwise =
        case foldr accumulateMeasurement (False, 0, 0) meas of
            (False, _, _) -> 1
            (True, precSum, meanPrecSum) ->
                let measMean = meanPrecSum / precSum
                    measVar = 1 / precSum
                    innovation = measMean - kMean prior
                    expectedVar = max eps (kVar prior + measVar)
                    nis = innovation * innovation / expectedVar
                 in clamp 1 maxMult (nis / threshold)
  where
    threshold = kfcInnovationInflationThreshold cfg
    maxMult = max 1 (kfcInnovationInflationMax cfg)

stepMultiWithConfig :: KalmanFusionConfig -> [(Double, Double)] -> Kalman1 -> Kalman1
stepMultiWithConfig cfg meas kal =
    let prior = predict kal
        correlatedMeas = inflateCorrelatedMeasurements cfg meas
        inflation = innovationInflationFactor cfg prior correlatedMeas
        prior' =
            if inflation > 1
                then prior{kVar = max eps (kVar prior * sqrt inflation)}
                else prior
        meas' =
            if inflation > 1
                then map (inflateOne (inflation * inflation)) correlatedMeas
                else correlatedMeas
     in updateMulti meas' prior'
  where
    inflateOne mult (y, r)
        | validMeasurement (y, r) = (y, max eps r * mult)
        | otherwise = (y, r)

stepMulti :: [(Double, Double)] -> Kalman1 -> Kalman1
stepMulti = stepMultiWithConfig defaultKalmanFusionConfig

validMeasurement :: (Double, Double) -> Bool
validMeasurement (y, r) = isFinite y && isFinite r && r > 0

accumulateMeasurement :: (Double, Double) -> (Bool, Double, Double) -> (Bool, Double, Double)
accumulateMeasurement (y, rRaw) (hasValid, precSum, meanPrecSum)
    | not (isFinite y) = (hasValid, precSum, meanPrecSum)
    | not (isFinite rRaw) = (hasValid, precSum, meanPrecSum)
    | rRaw <= 0 = (hasValid, precSum, meanPrecSum)
    | otherwise =
        let r = max eps rRaw
            pr = 1 / r
         in (True, precSum + pr, meanPrecSum + y * pr)

clamp :: Double -> Double -> Double -> Double
clamp lo hi x = max lo (min hi x)

eps :: Double
eps = 1e-12

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)

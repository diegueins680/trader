module Trader.KalmanFusion (
    Kalman1 (..),
    initKalman1,
    predict,
    updateMulti,
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

stepMulti :: [(Double, Double)] -> Kalman1 -> Kalman1
stepMulti meas = updateMulti meas . predict

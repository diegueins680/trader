module Trader.Kalman3
  ( Vec3(..)
  , Mat3(..)
  , identity3
  , Kalman3(..)
  , predict
  , predictMeasurement
  , update
  , step
  , constantAcceleration1D
  , KalmanRun(..)
  , KalmanRunV(..)
  , runConstantAcceleration1D
  , runConstantAcceleration1DVec
  , forecastNextConstantAcceleration1D
  ) where

import Control.Monad.ST (runST)
import Data.List (foldl')
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV

data Vec3 = Vec3 !Double !Double !Double deriving (Eq, Show)

data Mat3 = Mat3 !Vec3 !Vec3 !Vec3 deriving (Eq, Show) -- rows

identity3 :: Mat3
identity3 = mat3 1 0 0 0 1 0 0 0 1

mat3 :: Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Mat3
mat3 a11 a12 a13 a21 a22 a23 a31 a32 a33 =
  Mat3 (Vec3 a11 a12 a13) (Vec3 a21 a22 a23) (Vec3 a31 a32 a33)

vecAdd :: Vec3 -> Vec3 -> Vec3
vecAdd (Vec3 a b c) (Vec3 d e f) = Vec3 (a + d) (b + e) (c + f)

vecScale :: Double -> Vec3 -> Vec3
vecScale s (Vec3 a b c) = Vec3 (s * a) (s * b) (s * c)

vecDot :: Vec3 -> Vec3 -> Double
vecDot (Vec3 a b c) (Vec3 d e f) = a * d + b * e + c * f

matAdd :: Mat3 -> Mat3 -> Mat3
matAdd (Mat3 r1 r2 r3) (Mat3 s1 s2 s3) = Mat3 (vecAdd r1 s1) (vecAdd r2 s2) (vecAdd r3 s3)

matSub :: Mat3 -> Mat3 -> Mat3
matSub (Mat3 r1 r2 r3) (Mat3 s1 s2 s3) = Mat3 (vecAdd r1 (vecScale (-1) s1)) (vecAdd r2 (vecScale (-1) s2)) (vecAdd r3 (vecScale (-1) s3))

matScale :: Double -> Mat3 -> Mat3
matScale s (Mat3 r1 r2 r3) = Mat3 (vecScale s r1) (vecScale s r2) (vecScale s r3)

matTranspose :: Mat3 -> Mat3
matTranspose (Mat3 (Vec3 a11 a12 a13) (Vec3 a21 a22 a23) (Vec3 a31 a32 a33)) =
  mat3 a11 a21 a31 a12 a22 a32 a13 a23 a33

matVec :: Mat3 -> Vec3 -> Vec3
matVec (Mat3 r1 r2 r3) v = Vec3 (vecDot r1 v) (vecDot r2 v) (vecDot r3 v)

matMul :: Mat3 -> Mat3 -> Mat3
matMul a b =
  let Mat3 c1 c2 c3 = matTranspose b
      row r = Vec3 (vecDot r c1) (vecDot r c2) (vecDot r c3)
  in case a of
       Mat3 r1 r2 r3 -> Mat3 (row r1) (row r2) (row r3)

outer :: Vec3 -> Vec3 -> Mat3
outer (Vec3 u1 u2 u3) v = Mat3 (vecScale u1 v) (vecScale u2 v) (vecScale u3 v)

-- | 3D-state Kalman filter with scalar measurement.
data Kalman3 = Kalman3
  { kF :: Mat3
  , kH :: Vec3 -- 1x3 row
  , kQ :: Mat3
  , kR :: Double
  , kx :: Vec3
  , kP :: Mat3
  } deriving (Eq, Show)

predict :: Kalman3 -> Kalman3
predict k =
  let x' = matVec (kF k) (kx k)
      p' = matAdd (matMul (matMul (kF k) (kP k)) (matTranspose (kF k))) (kQ k)
  in k { kx = x', kP = p' }

predictMeasurement :: Kalman3 -> Double
predictMeasurement k = vecDot (kH k) (kx k)

update :: Double -> Kalman3 -> Kalman3
update z k =
  let h = kH k
      p = kP k
      x = kx k
      y = z - vecDot h x
      phT = matVec p h
      s = vecDot h phT + kR k
      kGain = vecScale (1 / s) phT
      x' = vecAdd x (vecScale y kGain)
      a = matSub identity3 (outer kGain h)
      p' = matAdd (matMul (matMul a p) (matTranspose a)) (matScale (kR k) (outer kGain kGain))
  in k { kx = x', kP = p' }

step :: Double -> Kalman3 -> (Double, Kalman3)
step z k =
  let kPred = predict k
      predZ = predictMeasurement kPred
      kUpd = update z kPred
  in (predZ, kUpd)

-- | Constant-acceleration model for a 1D signal.
-- State: [position, velocity, acceleration]
constantAcceleration1D :: Double -> Double -> Double -> Double -> Kalman3
constantAcceleration1D dt processVar measurementVar initialPosition
  | dt <= 0 = error "dt must be > 0"
  | processVar < 0 = error "processVar must be >= 0"
  | measurementVar <= 0 = error "measurementVar must be > 0"
  | otherwise =
      let dt2 = dt * dt
          dt3 = dt2 * dt
          dt4 = dt2 * dt2
          dt5 = dt4 * dt
          f = mat3 1 dt (0.5 * dt2)
                  0 1 dt
                  0 0 1
          q = matScale processVar $
                mat3 (dt5 / 20) (dt4 / 8)  (dt3 / 6)
                     (dt4 / 8)  (dt3 / 3)  (dt2 / 2)
                     (dt3 / 6)  (dt2 / 2)  dt
          h = Vec3 1 0 0
          x0 = Vec3 initialPosition 0 0
          p0 = identity3
      in Kalman3 { kF = f, kH = h, kQ = q, kR = measurementVar, kx = x0, kP = p0 }

data KalmanRun = KalmanRun
  { krPredicted :: [Double] -- length n-1
  , krFiltered  :: [Double] -- length n
  } deriving (Eq, Show)

data KalmanRunV = KalmanRunV
  { krPredictedV :: V.Vector Double -- length n-1
  , krFilteredV  :: V.Vector Double -- length n
  } deriving (Eq, Show)

runConstantAcceleration1D :: Double -> Double -> Double -> [Double] -> KalmanRun
runConstantAcceleration1D dt processVar measurementVar values =
  case values of
    [] -> error "Need at least 2 values"
    [_] -> error "Need at least 2 values"
    (x0:xs) ->
      let k0 = constantAcceleration1D dt processVar measurementVar x0
          stepFn (k, preds, filts) z =
            let (pred, k') = step z k
                Vec3 pos _ _ = kx k'
            in (k', pred : preds, pos : filts)
          (_, predsRev, filtsRev) = foldl' stepFn (k0, [], [x0]) xs
      in KalmanRun { krPredicted = reverse predsRev, krFiltered = reverse filtsRev }

runConstantAcceleration1DVec :: Double -> Double -> Double -> V.Vector Double -> KalmanRunV
runConstantAcceleration1DVec dt processVar measurementVar valuesV =
  let n = V.length valuesV
  in if n < 2
       then error "Need at least 2 values"
       else
         runST $ do
           preds <- MV.new (n - 1)
           filts <- MV.new n
           let x0 = valuesV V.! 0
               k0 = constantAcceleration1D dt processVar measurementVar x0
           MV.write filts 0 x0
           let go i k =
                 if i >= n
                   then pure ()
                   else do
                     let z = valuesV V.! i
                         (pred, k') = step z k
                         Vec3 pos _ _ = kx k'
                     MV.write preds (i - 1) pred
                     MV.write filts i pos
                     go (i + 1) k'
           go 1 k0
           predsV <- V.unsafeFreeze preds
           filtsV <- V.unsafeFreeze filts
           pure KalmanRunV { krPredictedV = predsV, krFilteredV = filtsV }

forecastNextConstantAcceleration1D :: Double -> Double -> Double -> [Double] -> Double
forecastNextConstantAcceleration1D dt processVar measurementVar values =
  case values of
    [] -> error "Need at least 2 values"
    [_] -> error "Need at least 2 values"
    (x0:xs) ->
      let k0 = constantAcceleration1D dt processVar measurementVar x0
          kFinal = foldl' (\k z -> snd (step z k)) k0 xs
          kPred = predict kFinal
      in predictMeasurement kPred

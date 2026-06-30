module Trader.Predictors.Conformal (
    AdaptiveConformalState (..),
    ConformalModel (..),
    adaptivePredictInterval,
    fitConformal,
    initAdaptiveConformal,
    predictInterval,
    sigmaFromInterval,
    updateAdaptiveConformal,
) where

import Data.List (sort)
import Data.Maybe (fromMaybe, listToMaybe)

data ConformalModel = ConformalModel
    { cmAlpha :: !Double
    , cmRadius :: !Double -- quantile of |residual|, or Infinity when unavailable
    , cmCount :: !Int
    , cmRecentRadius :: !Double
    , cmRecentCount :: !Int
    }
    deriving (Eq, Show)

data AdaptiveConformalState = AdaptiveConformalState
    { acsAlpha :: !Double
    , acsEta :: !Double
    , acsRadius :: !Double
    , acsMissEwma :: !Double
    , acsCount :: !Int
    }
    deriving (Eq, Show)

fitConformal :: Double -> [Double] -> ConformalModel
fitConformal alpha absResiduals =
    let alpha' = clampAlpha alpha
     in case admissibleResiduals absResiduals of
            Nothing -> unavailableConformal alpha'
            Just cleaned ->
                let q = conformalRadius alpha' cleaned
                    recent = recentResiduals cleaned
                    qRecent = conformalRadius alpha' recent
                 in ConformalModel
                        { cmAlpha = alpha'
                        , cmRadius = max q qRecent
                        , cmCount = length cleaned
                        , cmRecentRadius = qRecent
                        , cmRecentCount = length recent
                        }

clampAlpha :: Double -> Double
clampAlpha a = min 0.999999 (max 1e-6 a)

predictInterval :: ConformalModel -> Double -> (Double, Double, Maybe Double)
predictInterval cm mu =
    let radius = cmRadius cm
     in if cmCount cm <= 0 || not (isFinite mu) || not (isAdmissibleResidual radius)
            then unavailableInterval
            else
                let lo = mu - radius
                    hi = mu + radius
                 in (lo, hi, sigmaFromInterval (cmAlpha cm) lo hi)

initAdaptiveConformal :: Double -> ConformalModel -> AdaptiveConformalState
initAdaptiveConformal eta cm =
    AdaptiveConformalState
        { acsAlpha = cmAlpha cm
        , acsEta = sanitizeEta eta
        , acsRadius =
            if isAdmissibleResidual (cmRadius cm)
                then cmRadius cm
                else 1 / 0
        , acsMissEwma = cmAlpha cm
        , acsCount = cmCount cm
        }

adaptivePredictInterval :: AdaptiveConformalState -> Double -> (Double, Double, Maybe Double)
adaptivePredictInterval st mu =
    let cm =
            ConformalModel
                { cmAlpha = acsAlpha st
                , cmRadius = acsRadius st
                , cmCount = acsCount st
                , cmRecentRadius = acsRadius st
                , cmRecentCount = acsCount st
                }
     in predictInterval cm mu

updateAdaptiveConformal :: Double -> Double -> AdaptiveConformalState -> AdaptiveConformalState
updateAdaptiveConformal realized mu st =
    if not (isFinite realized && isFinite mu && isAdmissibleResidual (acsRadius st))
        then st
        else
            let resid = abs (realized - mu)
                miss = (if resid > acsRadius st then 1 else 0) :: Double
                eta = sanitizeEta (acsEta st)
                targetMiss = acsAlpha st
                missEwma' = (1 - eta) * acsMissEwma st + eta * miss
                baseRadius =
                    if acsRadius st <= 0 && miss > 0
                        then max 1e-12 resid
                        else acsRadius st
                radius' = max 0 (baseRadius * exp (eta * (miss - targetMiss)))
             in st
                    { acsRadius = radius'
                    , acsMissEwma = missEwma'
                    , acsCount = acsCount st + 1
                    }

{- | Approximate sigma from a symmetric interval [lo, hi] assuming it corresponds
to a Normal (mu, sigma) with central coverage (1 - alpha).
-}
sigmaFromInterval :: Double -> Double -> Double -> Maybe Double
sigmaFromInterval alpha lo hi =
    let width = hi - lo
        p = 1 - alpha / 2
        z = normalInv p
     in if not (isFinite width) || width <= 0 || not (isFinite z) || z <= 0
            then Nothing
            else Just (width / (2 * z))

admissibleResiduals :: [Double] -> Maybe [Double]
admissibleResiduals xs
    | null xs = Nothing
    | all isAdmissibleResidual xs = Just xs
    | otherwise = Nothing

isAdmissibleResidual :: Double -> Bool
isAdmissibleResidual v = isFinite v && v >= 0

unavailableConformal :: Double -> ConformalModel
unavailableConformal alpha =
    ConformalModel
        { cmAlpha = alpha
        , cmRadius = 1 / 0
        , cmCount = 0
        , cmRecentRadius = 1 / 0
        , cmRecentCount = 0
        }

unavailableInterval :: (Double, Double, Maybe Double)
unavailableInterval = (negate (1 / 0), 1 / 0, Nothing)

recentResiduals :: [Double] -> [Double]
recentResiduals xs =
    let n = length xs
        k = max 1 (min n (ceiling (sqrt (fromIntegral n :: Double))))
     in drop (n - k) xs

conformalRadius :: Double -> [Double] -> Double
conformalRadius alpha xs =
    let s = sort xs
        n = length s
        k = ceiling ((1 - alpha) * fromIntegral (n + 1))
        idx = max 0 (min (n - 1) (k - 1))
     in fromMaybe 0 (listToMaybe (drop idx s))

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)

sanitizeEta :: Double -> Double
sanitizeEta eta
    | not (isFinite eta) = 0.02
    | otherwise = min 1 (max 1e-6 eta)

-- Approximation of the standard normal inverse CDF.
normalInv :: Double -> Double
normalInv p
    | p <= 0 = -(1 / 0)
    | p >= 1 = 1 / 0
    | p < plow =
        let q = sqrt (-(2 * log p))
         in (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    | p > phigh =
        let q = sqrt (-(2 * log (1 - p)))
         in -( (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
             )
    | otherwise =
        let q = p - 0.5
            r = q * q
         in (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
                * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
  where
    plow = 0.02425
    phigh = 1 - plow
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

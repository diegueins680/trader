module Trader.Predictors.Conformal (
    ConformalModel (..),
    fitConformal,
    predictInterval,
    sigmaFromInterval,
) where

import Data.List (sort)

data ConformalModel = ConformalModel
    { cmAlpha :: !Double
    , cmRadius :: !Double -- quantile of |residual|
    , cmCount :: !Int
    }
    deriving (Eq, Show)

fitConformal :: Double -> [Double] -> ConformalModel
fitConformal alpha absResiduals
    | alpha <= 0 || alpha >= 1 = error "alpha must be in (0,1)"
    | otherwise =
        let cleaned = filter (\v -> isFinite v && v >= 0) absResiduals
            count = length cleaned
         in if null cleaned
                then ConformalModel{cmAlpha = alpha, cmRadius = 0, cmCount = 0}
                else
                    let q = conformalRadius alpha cleaned
                     in ConformalModel{cmAlpha = alpha, cmRadius = q, cmCount = count}

predictInterval :: ConformalModel -> Double -> (Double, Double, Maybe Double)
predictInterval cm mu =
    let lo = mu - cmRadius cm
        hi = mu + cmRadius cm
        sigma =
            if cmCount cm <= 0
                then Nothing
                else sigmaFromInterval (cmAlpha cm) lo hi
     in (lo, hi, sigma)

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

conformalRadius :: Double -> [Double] -> Double
conformalRadius alpha xs =
    let s = sort xs
        n = length s
        k = ceiling ((1 - alpha) * fromIntegral (n + 1))
        idx = max 0 (min (n - 1) (k - 1))
     in s !! idx

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)

-- Approximation of the standard normal inverse CDF.
normalInv :: Double -> Double
normalInv p
    | p <= 0 = -1 / 0
    | p >= 1 = 1 / 0
    | p < plow =
        let q = sqrt (-2 * log p)
         in (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    | p > phigh =
        let q = sqrt (-2 * log (1 - p))
         in -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
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

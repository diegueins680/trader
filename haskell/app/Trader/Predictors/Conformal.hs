module Trader.Predictors.Conformal
  ( ConformalModel(..)
  , fitConformal
  , predictInterval
  , sigmaFromInterval
  ) where

import Data.List (sort)

data ConformalModel = ConformalModel
  { cmAlpha :: !Double
  , cmRadius :: !Double -- quantile of |residual|
  } deriving (Eq, Show)

fitConformal :: Double -> [Double] -> ConformalModel
fitConformal alpha absResiduals
  | alpha <= 0 || alpha >= 1 = error "alpha must be in (0,1)"
  | null absResiduals = ConformalModel { cmAlpha = alpha, cmRadius = 0 }
  | otherwise =
      let q = quantile (1 - alpha) absResiduals
       in ConformalModel { cmAlpha = alpha, cmRadius = q }

predictInterval :: ConformalModel -> Double -> (Double, Double, Double)
predictInterval cm mu =
  let lo = mu - cmRadius cm
      hi = mu + cmRadius cm
      sigma = sigmaFromInterval lo hi
   in (lo, hi, sigma)

-- | Approximate sigma from a symmetric interval [lo, hi] assuming it corresponds
-- to a Normal (mu, sigma) 80% central interval (q10/q90).
sigmaFromInterval :: Double -> Double -> Double
sigmaFromInterval lo hi =
  let z = 1.281551565545 -- Phi^{-1}(0.9)
   in max 1e-12 ((hi - lo) / (2 * z))

quantile :: Double -> [Double] -> Double
quantile q xs
  | q <= 0 = minimum xs
  | q >= 1 = maximum xs
  | otherwise =
      let s = sort xs
          n = length s
          idx = floor (q * fromIntegral (n - 1))
       in s !! idx


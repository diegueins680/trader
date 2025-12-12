module Trader.OnlineStats
  ( Welford(..)
  , emptyWelford
  , updateWelford
  , varianceWelford
  ) where

-- | Numerically-stable online mean/variance estimator (Welford).
data Welford = Welford
  { wCount :: !Int
  , wMean :: !Double
  , wM2 :: !Double
  } deriving (Eq, Show)

emptyWelford :: Welford
emptyWelford = Welford { wCount = 0, wMean = 0, wM2 = 0 }

updateWelford :: Double -> Welford -> Welford
updateWelford x w =
  let n1 = wCount w + 1
      delta = x - wMean w
      mean' = wMean w + delta / fromIntegral n1
      delta2 = x - mean'
      m2' = wM2 w + delta * delta2
   in Welford { wCount = n1, wMean = mean', wM2 = m2' }

varianceWelford :: Welford -> Maybe Double
varianceWelford w =
  if wCount w < 2
    then Nothing
    else Just (wM2 w / fromIntegral (wCount w - 1))


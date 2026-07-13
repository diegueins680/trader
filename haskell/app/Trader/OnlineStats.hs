module Trader.OnlineStats (
    Welford (..),
    emptyWelford,
    updateWelford,
    varianceWelford,
) where

-- | Numerically-stable online mean/variance estimator (Welford).
data Welford = Welford
    { wCount :: !Int
    , wMean :: !Double
    , wM2 :: !Double
    }
    deriving (Eq, Show)

emptyWelford :: Welford
emptyWelford = Welford{wCount = 0, wMean = 0, wM2 = 0}

updateWelford :: Double -> Welford -> Welford
updateWelford x w
    | not (finite x) = w
    | not (validWelfordState w) = Welford{wCount = 1, wMean = x, wM2 = 0}
    | otherwise =
        let n1 = wCount w + 1
            delta = x - wMean w
            mean' = wMean w + delta / fromIntegral n1
            delta2 = x - mean'
            m2' = wM2 w + delta * delta2
         in if finite mean' && finite m2' && m2' >= 0
                then Welford{wCount = n1, wMean = mean', wM2 = m2'}
                else w

varianceWelford :: Welford -> Maybe Double
varianceWelford w =
    if wCount w < 2 || not (validWelfordState w)
        then Nothing
        else
            let variance = wM2 w / fromIntegral (wCount w - 1)
             in if finite variance && variance >= 0 then Just variance else Nothing

validWelfordState :: Welford -> Bool
validWelfordState w =
    wCount w >= 0
        && finite (wMean w)
        && finite (wM2 w)
        && wM2 w >= 0

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

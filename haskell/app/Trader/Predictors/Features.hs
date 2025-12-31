module Trader.Predictors.Features
  ( FeatureSpec(..)
  , mkFeatureSpec
  , featuresAt
  , forwardReturnAt
  , buildDataset
  ) where

import qualified Data.Vector as V

data FeatureSpec = FeatureSpec
  { fsLookbackBars :: !Int
  , fsShortBars :: !Int
  , fsMidBars :: !Int
  } deriving (Eq, Show)

mkFeatureSpec :: Int -> FeatureSpec
mkFeatureSpec lookbackBars
  | lookbackBars <= 1 = error "lookbackBars must be >= 2"
  | otherwise =
      let lb = lookbackBars
          shortB = max 2 (min 12 (lb - 1))
          midB = max 2 (min 48 (lb - 1))
       in FeatureSpec { fsLookbackBars = lb, fsShortBars = shortB, fsMidBars = midB }

-- | Forward return r_t = p_{t+1}/p_t - 1.
forwardReturnAt :: V.Vector Double -> Int -> Maybe Double
forwardReturnAt prices t =
  if t < 0 || t + 1 >= V.length prices
    then Nothing
    else
      let p0 = prices V.! t
          p1 = prices V.! (t + 1)
       in if p0 == 0 then Nothing else Just (p1 / p0 - 1)

-- | Feature vector at bar t using only prices up to t.
-- Requires at least fsLookbackBars history (prices window ending at t).
featuresAt :: FeatureSpec -> V.Vector Double -> Int -> Maybe [Double]
featuresAt fs prices t = do
  let lb = fsLookbackBars fs
      shortB = fsShortBars fs
      midB = fsMidBars fs
  if t < lb - 1 || t >= V.length prices
    then Nothing
    else do
      ret1 <- retOver prices t 1
      ret3 <- retOver prices t 3
      retShort <- retOver prices t shortB
      retMid <- retOver prices t midB
      retLb <- retOver prices t (lb - 1)
      rsShort <- returnsEndingAt prices t shortB
      rsMid <- returnsEndingAt prices t midB
      let (muS, sigS) = meanStd rsShort
          (muM, sigM) = meanStd rsMid
      pure
        [ ret1
        , ret3
        , retShort
        , retMid
        , retLb
        , muS
        , sigS
        , muM
        , sigM
        ]

-- | Build a supervised dataset (features at t, target forward return at t).
-- Uses t in [lookbackBars-1 .. n-2].
buildDataset :: FeatureSpec -> V.Vector Double -> [( [Double], Double )]
buildDataset fs prices =
  let n = V.length prices
      startT = fsLookbackBars fs - 1
      endT = n - 2
      retLen = max 0 (n - 1)
      returns =
        V.generate retLen $ \i ->
          let p0 = prices V.! i
              p1 = prices V.! (i + 1)
           in if p0 == 0 then Nothing else Just (p1 / p0 - 1)
      retVals = V.map (maybe 0 id) returns
      retSqVals = V.map (maybe 0 (\r -> r * r)) returns
      retInvalid = V.map (\m -> case m of Nothing -> 1; Just _ -> 0) returns
      prefixSum = V.scanl' (+) 0 retVals
      prefixSumSq = V.scanl' (+) 0 retSqVals
      prefixInvalid = V.scanl' (+) 0 retInvalid

      windowStats t k =
        if k <= 0 || t - k < 0 || t - 1 >= retLen
          then Nothing
          else
            let i0 = t - k
                i1 = t - 1
                invalid = prefixInvalid V.! (i1 + 1) - prefixInvalid V.! i0
             in if invalid > 0
                  then Nothing
                  else
                    let s = prefixSum V.! (i1 + 1) - prefixSum V.! i0
                        ss = prefixSumSq V.! (i1 + 1) - prefixSumSq V.! i0
                        k' = fromIntegral k
                        mu = s / k'
                        var =
                          if k < 2
                            then 0
                            else (ss - k' * mu * mu) / fromIntegral (k - 1)
                     in Just (mu, sqrt (var + 1e-12))

      retOverFast t bars =
        if bars <= 0 || t - bars < 0
          then Nothing
          else
            let p0 = prices V.! (t - bars)
                p1 = prices V.! t
             in if p0 == 0 then Nothing else Just (p1 / p0 - 1)

      featuresAtFast t = do
        if t < fsLookbackBars fs - 1 || t >= n
          then Nothing
          else do
            ret1 <- retOverFast t 1
            ret3 <- retOverFast t 3
            retShort <- retOverFast t (fsShortBars fs)
            retMid <- retOverFast t (fsMidBars fs)
            retLb <- retOverFast t (fsLookbackBars fs - 1)
            (muS, sigS) <- windowStats t (fsShortBars fs)
            (muM, sigM) <- windowStats t (fsMidBars fs)
            pure
              [ ret1
              , ret3
              , retShort
              , retMid
              , retLb
              , muS
              , sigS
              , muM
              , sigM
              ]

      forwardReturnFast t =
        if t < 0 || t >= retLen
          then Nothing
          else returns V.! t
   in
    if startT > endT
      then []
      else
        [ (f, y)
        | t <- [startT .. endT]
        , Just f <- [featuresAtFast t]
        , Just y <- [forwardReturnFast t]
        ]

retOver :: V.Vector Double -> Int -> Int -> Maybe Double
retOver prices t bars =
  if bars <= 0 || t - bars < 0
    then Nothing
    else
      let p0 = prices V.! (t - bars)
          p1 = prices V.! t
       in if p0 == 0 then Nothing else Just (p1 / p0 - 1)

returnsEndingAt :: V.Vector Double -> Int -> Int -> Maybe [Double]
returnsEndingAt prices t k =
  if k <= 0 || t - k < 0
    then Nothing
    else
      let rs =
            [ let p0 = prices V.! i
                  p1 = prices V.! (i + 1)
               in if p0 == 0 then Nothing else Just (p1 / p0 - 1)
            | i <- [t - k .. t - 1]
            ]
       in sequence rs

meanStd :: [Double] -> (Double, Double)
meanStd xs =
  case xs of
    [] -> (0, 0)
    _ ->
      let n = length xs
          mu = sum xs / fromIntegral n
          var =
            if n < 2
              then 0
              else
                let denom = fromIntegral (n - 1)
                 in sum (map (\v -> (v - mu) * (v - mu)) xs) / denom
       in (mu, sqrt (var + 1e-12))

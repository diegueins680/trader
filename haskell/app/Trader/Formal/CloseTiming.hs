module Trader.Formal.CloseTiming (
    CloseTimingDecision (..),
    CloseTimingObservation (..),
    CloseTimingStats (..),
    buildCloseTimingStats,
    closeTimingDecision,
    optimalCloseObservation,
) where

import Data.Function (on)
import Data.List (groupBy, sortOn)
import Data.Maybe (mapMaybe)

-- | Historical position sample grouped by combo.
data CloseTimingObservation = CloseTimingObservation
    { ctoCombo :: !String
    , ctoOpenAtMs :: !Int
    , ctoCloseAtMs :: !Int
    , ctoOptimalCloseAtMs :: !Int
    }
    deriving (Eq, Show)

{- | Robust summary over tm in normalized units:
  r = (tm - ta) / (tc - ta), with support [0, 2].
-}
data CloseTimingStats = CloseTimingStats
    { ctsCombo :: !String
    , ctsSamples :: !Int
    , ctsMedianRatio :: !Double
    , ctsMadRatio :: !Double
    , ctsQ25Ratio :: !Double
    , ctsQ75Ratio :: !Double
    }
    deriving (Eq, Show)

-- | Close policy: recommend hold if age ratio is below the risk-budget quantile.
data CloseTimingDecision = CloseTimingDecision
    { ctdShouldClose :: !Bool
    , ctdAgeRatio :: !Double
    , ctdTargetRatio :: !Double
    }
    deriving (Eq, Show)

optimalCloseObservation :: String -> Int -> Int -> [(Int, Double)] -> Maybe CloseTimingObservation
optimalCloseObservation combo ta tc pnlPath
    | tc <= ta = Nothing
    | null candidates = Nothing
    | otherwise =
        let (tm, _) = foldl1 chooseBetterClose candidates
         in Just
                CloseTimingObservation
                    { ctoCombo = combo
                    , ctoOpenAtMs = ta
                    , ctoCloseAtMs = tc
                    , ctoOptimalCloseAtMs = tm
                    }
  where
    candidates = filter (isCloseTimingCandidate ta tc) pnlPath

isCloseTimingCandidate :: Int -> Int -> (Int, Double) -> Bool
isCloseTimingCandidate ta tc (t, pnl) =
    inCloseTimingWindow ta tc t && isFinite pnl

inCloseTimingWindow :: Int -> Int -> Int -> Bool
inCloseTimingWindow ta tc t =
    let taInteger = toInteger ta
        tInteger = toInteger t
     in tInteger >= taInteger && tInteger <= closeTimingWindowUpper ta tc

closeTimingWindowUpper :: Int -> Int -> Integer
closeTimingWindowUpper ta tc =
    let taInteger = toInteger ta
        tcInteger = toInteger tc
     in taInteger + 2 * (tcInteger - taInteger)

-- | Higher PnL wins; equal PnL picks the earliest timestamp so ties stay order-invariant.
chooseBetterClose :: (Int, Double) -> (Int, Double) -> (Int, Double)
chooseBetterClose best cur
    | snd cur > snd best = cur
    | snd cur < snd best = best
    | fst cur < fst best = cur
    | otherwise = best

buildCloseTimingStats :: [CloseTimingObservation] -> [CloseTimingStats]
buildCloseTimingStats obs =
    map statsFor grouped
  where
    grouped =
        groupBy ((==) `on` ctoCombo) . sortOn ctoCombo $ filter hasValidRatio obs

    hasValidRatio = maybe False (const True) . observationRatio

    statsFor xs =
        let combo = ctoCombo (head xs)
            ratios = sortOn id (mapMaybe observationRatio xs)
            med = percentile 0.5 ratios
            mad = percentile 0.5 (sortOn id (map (abs . subtract med) ratios))
         in CloseTimingStats
                { ctsCombo = combo
                , ctsSamples = length ratios
                , ctsMedianRatio = med
                , ctsMadRatio = mad
                , ctsQ25Ratio = percentile 0.25 ratios
                , ctsQ75Ratio = percentile 0.75 ratios
                }

closeTimingDecision :: Double -> CloseTimingStats -> Int -> Int -> Int -> CloseTimingDecision
closeTimingDecision riskBudget stats ta expectedDurationMs now =
    let denom = positiveDurationInteger expectedDurationMs
        age = nonNegativeIntegerDelta ta now
        ageRatio = fromInteger age / fromInteger denom
        target = mix (clamp 0 1 riskBudget) (ctsMedianRatio stats) (ctsQ75Ratio stats)
     in CloseTimingDecision
            { ctdShouldClose = ageRatio >= target
            , ctdAgeRatio = ageRatio
            , ctdTargetRatio = target
            }

observationRatio :: CloseTimingObservation -> Maybe Double
observationRatio x = do
    (num, denom) <- observationSpan x
    pure (clamp 0 2 (fromInteger num / fromInteger denom))

observationSpan :: CloseTimingObservation -> Maybe (Integer, Integer)
observationSpan x =
    let denom = integerDelta (ctoOpenAtMs x) (ctoCloseAtMs x)
        num = integerDelta (ctoOpenAtMs x) (ctoOptimalCloseAtMs x)
     in if denom > 0 && num >= 0 && num <= 2 * denom
            then Just (num, denom)
            else Nothing

positiveDurationInteger :: Int -> Integer
positiveDurationInteger = max 1 . toInteger

integerDelta :: Int -> Int -> Integer
integerDelta start end = toInteger end - toInteger start

nonNegativeIntegerDelta :: Int -> Int -> Integer
nonNegativeIntegerDelta start end = max 0 (integerDelta start end)

mix :: Double -> Double -> Double -> Double
mix w a b = (1 - w) * a + w * b

percentile :: Double -> [Double] -> Double
percentile _ [] = 1
percentile p xs =
    let ys = sortOn id xs
        n = length ys
        idx = floor (clamp 0 1 p * fromIntegral (n - 1))
     in ys !! idx

clamp :: Double -> Double -> Double -> Double
clamp lo hi = max lo . min hi

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)
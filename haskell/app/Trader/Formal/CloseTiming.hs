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

-- | Historical position sample grouped by combo.
data CloseTimingObservation = CloseTimingObservation
    { ctoCombo :: !String
    , ctoOpenAtMs :: !Int
    , ctoCloseAtMs :: !Int
    , ctoOptimalCloseAtMs :: !Int
    }
    deriving (Eq, Show)

-- | Robust summary over tm in normalized units:
--   r = (tm - ta) / (tc - ta), with support [0, 2].
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
        let (tm, _) =
                foldl1
                    (\best cur -> if snd cur > snd best then cur else best)
                    candidates
         in Just
                CloseTimingObservation
                    { ctoCombo = combo
                    , ctoOpenAtMs = ta
                    , ctoCloseAtMs = tc
                    , ctoOptimalCloseAtMs = tm
                    }
  where
    upper = ta + 2 * (tc - ta)
    candidates = filter (\(t, pnl) -> t >= ta && t <= upper && isFinite pnl) pnlPath

buildCloseTimingStats :: [CloseTimingObservation] -> [CloseTimingStats]
buildCloseTimingStats obs =
    map statsFor grouped
  where
    grouped =
        groupBy ((==) `on` ctoCombo) . sortOn ctoCombo $ filter valid obs

    valid x =
        let denom = ctoCloseAtMs x - ctoOpenAtMs x
            num = ctoOptimalCloseAtMs x - ctoOpenAtMs x
         in denom > 0 && num >= 0 && num <= 2 * denom

    statsFor xs =
        let combo = ctoCombo (head xs)
            ratios = sortOn id (map ratio xs)
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

    ratio x =
        let denom = fromIntegral (ctoCloseAtMs x - ctoOpenAtMs x)
            num = fromIntegral (ctoOptimalCloseAtMs x - ctoOpenAtMs x)
         in clamp 0 2 (num / denom)

closeTimingDecision :: Double -> CloseTimingStats -> Int -> Int -> Int -> CloseTimingDecision
closeTimingDecision riskBudget stats ta expectedDurationMs now =
    let denom = max 1 expectedDurationMs
        age = max 0 (now - ta)
        ageRatio = fromIntegral age / fromIntegral denom
        target = mix (clamp 0 1 riskBudget) (ctsMedianRatio stats) (ctsQ75Ratio stats)
     in CloseTimingDecision
            { ctdShouldClose = ageRatio >= target
            , ctdAgeRatio = ageRatio
            , ctdTargetRatio = target
            }

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

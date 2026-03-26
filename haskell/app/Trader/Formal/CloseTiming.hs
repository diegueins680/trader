{-# LANGUAGE DeriveGeneric #-}

module Trader.Formal.CloseTiming (
    ComboTrade (..),
    ComboTimingStats (..),
    TradeTimingSample (..),
    timingSamples,
    summarizeComboTiming,
    summarizeAllCombos
) where

import Data.Function (on)
import Data.List (foldl', groupBy, sort, sortOn)
import GHC.Generics (Generic)

-- | Trade tagged with combo identity and bar indices.
data ComboTrade = ComboTrade
    { ctComboId :: !String
    , ctEntryIndex :: !Int
    , ctExitIndex :: !Int
    , ctEntryPrice :: !Double
    , ctSide :: !Double -- +1 long, -1 short
    }
    deriving (Eq, Show, Generic)

-- | Per-trade sample with the optimal close bar tm in [ta, ta + 2*(tc-ta)].
data TradeTimingSample = TradeTimingSample
    { ttsComboId :: !String
    , ttsEntryIndex :: !Int
    , ttsExitIndex :: !Int
    , ttsOptimalIndex :: !Int
    , ttsObservedDuration :: !Int
    , ttsOptimalDuration :: !Int
    , ttsObservedReturn :: !Double
    , ttsOptimalReturn :: !Double
    , ttsReturnLift :: !Double
    }
    deriving (Eq, Show, Generic)

-- | Robust summary for tm distribution per combo.
data ComboTimingStats = ComboTimingStats
    { ctsComboId :: !String
    , ctsSampleCount :: !Int
    , ctsMedianRatio :: !Double
    , ctsQ25Ratio :: !Double
    , ctsQ75Ratio :: !Double
    , ctsMadRatio :: !Double
    , ctsMeanLift :: !Double
    , ctsMedianLift :: !Double
    }
    deriving (Eq, Show, Generic)

safeAt :: [Double] -> Int -> Maybe Double
safeAt xs i
    | i < 0 = Nothing
    | otherwise = go xs i
  where
    go [] _ = Nothing
    go (y : ys) k
        | k == 0 = Just y
        | otherwise = go ys (k - 1)

returnAt :: [Double] -> ComboTrade -> Int -> Maybe Double
returnAt prices tr i = do
    px <- safeAt prices i
    pure (ctSide tr * (px / ctEntryPrice tr - 1))

optimalIndexInWindow :: [Double] -> ComboTrade -> Int
optimalIndexInWindow prices tr =
    let ta = ctEntryIndex tr
        tc = ctExitIndex tr
        obsDuration = max 1 (tc - ta)
        maxI = min (length prices - 1) (ta + (2 * obsDuration))
        candidates = [ta .. maxI]
        scored = [(i, returnAt prices tr i) | i <- candidates]
        valid = [(i, r) | (i, Just r) <- scored]
     in case valid of
            [] -> tc
            _ -> fst (maximumBySnd valid)

maximumBySnd :: [(Int, Double)] -> (Int, Double)
maximumBySnd = foldl1 pick
  where
    pick a@(_, ra) b@(_, rb)
        | rb > ra = b
        | otherwise = a

sampleForTrade :: [Double] -> ComboTrade -> Maybe TradeTimingSample
sampleForTrade prices tr = do
    let ta = ctEntryIndex tr
        tc = ctExitIndex tr
        tm = optimalIndexInWindow prices tr
    observed <- returnAt prices tr tc
    optimal <- returnAt prices tr tm
    let durObs = max 0 (tc - ta)
        durOpt = max 0 (tm - ta)
    pure
        TradeTimingSample
            { ttsComboId = ctComboId tr
            , ttsEntryIndex = ta
            , ttsExitIndex = tc
            , ttsOptimalIndex = tm
            , ttsObservedDuration = durObs
            , ttsOptimalDuration = durOpt
            , ttsObservedReturn = observed
            , ttsOptimalReturn = optimal
            , ttsReturnLift = optimal - observed
            }

timingSamples :: [Double] -> [ComboTrade] -> [TradeTimingSample]
timingSamples prices = foldr step []
  where
    step tr acc = case sampleForTrade prices tr of
        Just s -> s : acc
        Nothing -> acc

quantile :: Double -> [Double] -> Double
quantile q xs
    | null xs = 0
    | otherwise =
        let ys = sort xs
            n = length ys
            idx = floor (q * fromIntegral (n - 1))
         in ys !! max 0 (min (n - 1) idx)

median :: [Double] -> Double
median = quantile 0.5

mean :: [Double] -> Double
mean xs =
    let n = length xs
     in if n == 0 then 0 else sum xs / fromIntegral n

mad :: [Double] -> Double
mad xs =
    let m = median xs
        deviations = map (abs . subtract m) xs
     in median deviations

summarizeComboTiming :: [TradeTimingSample] -> Maybe ComboTimingStats
summarizeComboTiming [] = Nothing
summarizeComboTiming ss@(s0 : _) =
    let ratios =
            [ if ttsObservedDuration s <= 0
                then 0
                else fromIntegral (ttsOptimalDuration s) / fromIntegral (ttsObservedDuration s)
            | s <- ss
            ]
        lifts = map ttsReturnLift ss
     in Just
            ComboTimingStats
                { ctsComboId = ttsComboId s0
                , ctsSampleCount = length ss
                , ctsMedianRatio = median ratios
                , ctsQ25Ratio = quantile 0.25 ratios
                , ctsQ75Ratio = quantile 0.75 ratios
                , ctsMadRatio = mad ratios
                , ctsMeanLift = mean lifts
                , ctsMedianLift = median lifts
                }

summarizeAllCombos :: [TradeTimingSample] -> [ComboTimingStats]
summarizeAllCombos ss =
    let grouped = groupBy ((==) `on` ttsComboId) (sortOn ttsComboId ss)
     in foldl' step [] grouped
  where
    step acc grp = case summarizeComboTiming grp of
        Just st -> st : acc
        Nothing -> acc

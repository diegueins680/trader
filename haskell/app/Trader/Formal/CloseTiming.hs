{-# LANGUAGE DuplicateRecordFields #-}

module Trader.Formal.CloseTiming (
    ComboCloseTimingReport (..),
    CloseTimingDecision (..),
    CloseTimingObservation (..),
    CloseTimingStats (..),
    ComboTimingStats (..),
    ComboTrade (..),
    TradeTimingSample (..),
    analyzeComboCloseTiming,
    buildCloseTimingStats,
    closeTimingDecision,
    optimalCloseObservation,
    summarizeAllCombos,
    summarizeComboTiming,
    timingSamples,
) where

import Data.Function (on)
import Data.List (foldl', groupBy, sort, sortOn)
import Data.Maybe (mapMaybe)
import qualified Data.Vector as V

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

-- | Close policy: mark close-ready once age ratio meets or exceeds the risk-budget quantile.
data CloseTimingDecision = CloseTimingDecision
    { ctdShouldClose :: !Bool
    , ctdAgeRatio :: !Double
    , ctdTargetRatio :: !Double
    }
    deriving (Eq, Show)

-- | Trade tagged with combo identity and price-series indices.
data ComboTrade = ComboTrade
    { ctComboId :: !String
    , ctEntryIndex :: !Int
    , ctExitIndex :: !Int
    , ctEntryPrice :: !Double
    , ctSide :: !Double
    }
    deriving (Eq, Show)

-- | Per-trade sample with the optimal close index tm in [ta, ta + 2 * (tc - ta)].
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
    deriving (Eq, Show)

-- | Robust tm-distribution summary per combo.
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
    deriving (Eq, Show)

-- | Combo-level report used to retune timing-based exits from historical trades.
data ComboCloseTimingReport = ComboCloseTimingReport
    { cctrComboId :: !String
    , cctrSampleCount :: !Int
    , cctrMedianRatio :: !(Maybe Double)
    , cctrQ25Ratio :: !(Maybe Double)
    , cctrQ75Ratio :: !(Maybe Double)
    , cctrMadRatio :: !(Maybe Double)
    , cctrMeanLift :: !(Maybe Double)
    , cctrMedianLift :: !(Maybe Double)
    , cctrMedianObservedDuration :: !(Maybe Int)
    , cctrMedianOptimalDuration :: !(Maybe Int)
    , cctrQ75OptimalDuration :: !(Maybe Int)
    , cctrRecommendedMaxHoldBars :: !(Maybe Int)
    }
    deriving (Eq, Show)

-- | Modeled quality envelope for a candidate maxHoldBars retune.
data MaxHoldBarsQuality = MaxHoldBarsQuality
    { mhqFinalEquity :: !Double
    , mhqExpectancy :: !Double
    , mhqMeanDuration :: !Double
    }
    deriving (Eq, Show)

-- | Exhaustively modeled candidate within the bounded maxHoldBars search space.
data MaxHoldBarsCandidate = MaxHoldBarsCandidate
    { mhbcHoldBars :: !Int
    , mhbcQuality :: !MaxHoldBarsQuality
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
    mapMaybe statsFor grouped
  where
    grouped =
        groupBy ((==) `on` ctoCombo) . sortOn ctoCombo $ filter validObservation obs

    -- groupBy emits non-empty groups, but we still pattern-match so the helper
    -- stays total without relying on partial list functions.
    statsFor [] = Nothing
    statsFor xs@(x : _) =
        let combo = ctoCombo x
            ratios = sortOn id (map observationRatio xs)
            q25 = boundedPercentile 0.25 ratios
            q50 = boundedPercentile 0.5 ratios
            q75 = boundedPercentile 0.75 ratios
            (q25Bound, q50Bound, q75Bound) = orderQuartiles q25 q50 q75
            mad = boundedPercentile 0.5 (sortOn id (map (abs . subtract q50Bound) ratios))
         in Just
                CloseTimingStats
                    { ctsCombo = combo
                    , ctsSamples = length ratios
                    , ctsMedianRatio = q50Bound
                    , ctsMadRatio = clampRatio mad
                    , ctsQ25Ratio = q25Bound
                    , ctsQ75Ratio = q75Bound
                    }

closeTimingDecision :: Double -> CloseTimingStats -> Int -> Int -> Int -> CloseTimingDecision
closeTimingDecision riskBudget stats ta expectedDurationMs now =
    let denom = positiveDurationInteger expectedDurationMs
        age = nonNegativeIntegerDelta ta now
        ageRatio = fromInteger age / fromInteger denom
        budget = normalizeRiskBudget riskBudget
        (medianRatio, q75Ratio) = decisionTargetBand stats
        target = clampRatio (mix budget medianRatio q75Ratio)
     in CloseTimingDecision
            { ctdShouldClose = ageRatio >= target
            , ctdAgeRatio = ageRatio
            , ctdTargetRatio = target
            }

timingSamples :: [Double] -> [ComboTrade] -> [TradeTimingSample]
timingSamples prices =
    let priceVec = V.fromList prices
     in mapMaybe (sampleForTrade priceVec)

sampleForTrade :: V.Vector Double -> ComboTrade -> Maybe TradeTimingSample
sampleForTrade prices tr = do
    observed <- returnAt prices tr (ctExitIndex tr)
    let ta = ctEntryIndex tr
        tc = ctExitIndex tr
        tm = optimalIndexInWindow prices tr
    optimal <- returnAt prices tr tm
    pure
        TradeTimingSample
            { ttsComboId = ctComboId tr
            , ttsEntryIndex = ta
            , ttsExitIndex = tc
            , ttsOptimalIndex = tm
            , ttsObservedDuration = max 0 (tc - ta)
            , ttsOptimalDuration = max 0 (tm - ta)
            , ttsObservedReturn = observed
            , ttsOptimalReturn = optimal
            , ttsReturnLift = optimal - observed
            }

returnAt :: V.Vector Double -> ComboTrade -> Int -> Maybe Double
returnAt prices tr i
    | not (validComboTrade tr) = Nothing
    | otherwise = do
        px <- safeAt prices i
        if isFinite px
            then pure (ctSide tr * (px / ctEntryPrice tr - 1))
            else Nothing

validComboTrade :: ComboTrade -> Bool
validComboTrade tr =
    ctEntryPrice tr > 0
        && isFinite (ctEntryPrice tr)
        && isFinite (ctSide tr)
        && ctSide tr /= 0

optimalIndexInWindow :: V.Vector Double -> ComboTrade -> Int
optimalIndexInWindow prices tr =
    let ta = ctEntryIndex tr
        tc = ctExitIndex tr
     in if tc <= ta
            then ta
            else
                let observedDuration = tc - ta
                    maxIndex = min (V.length prices - 1) (ta + 2 * observedDuration)
                    candidates = [ta .. maxIndex]
                    scored = mapMaybe scoreAt candidates
                 in case scored of
                        [] -> tc
                        xs -> fst (foldl1 chooseBetterClose xs)
  where
    scoreAt i = do
        ret <- returnAt prices tr i
        pure (i, ret)

summarizeComboTiming :: [TradeTimingSample] -> Maybe ComboTimingStats
summarizeComboTiming [] = Nothing
summarizeComboTiming samples@(sample0 : _) =
    let ratios = map timingRatio samples
        lifts = map ttsReturnLift samples
     in Just
            ComboTimingStats
                { ctsComboId = ttsComboId sample0
                , ctsSampleCount = length samples
                , ctsMedianRatio = boundedPercentile 0.5 ratios
                , ctsQ25Ratio = boundedPercentile 0.25 ratios
                , ctsQ75Ratio = boundedPercentile 0.75 ratios
                , ctsMadRatio = mad ratios
                , ctsMeanLift = mean lifts
                , ctsMedianLift = percentile 0.5 lifts
                }

summarizeAllCombos :: [TradeTimingSample] -> [ComboTimingStats]
summarizeAllCombos samples =
    mapMaybe summarizeComboTiming grouped
  where
    grouped =
        groupBy ((==) `on` ttsComboId) (sortOn ttsComboId samples)

analyzeComboCloseTiming :: String -> [Double] -> [ComboTrade] -> ComboCloseTimingReport
analyzeComboCloseTiming comboId prices trades =
    let normalizedTrades = map normalizeTrade trades
        priceVec = V.fromList prices
        sampledTrades = mapMaybe (sampledTrade priceVec) normalizedTrades
        samples = map snd sampledTrades
        stats = summarizeComboTiming samples
        medianObservedDuration = percentileInt 0.5 (map ttsObservedDuration samples)
        medianOptimalDuration = percentileInt 0.5 (map ttsOptimalDuration samples)
        q75OptimalDuration = percentileInt 0.75 (map ttsOptimalDuration samples)
        recommendedMaxHoldBars = recommendedCloseTimingHold priceVec sampledTrades
     in ComboCloseTimingReport
            { cctrComboId = comboId
            , cctrSampleCount = length samples
            , cctrMedianRatio = fmap (\ComboTimingStats{ctsMedianRatio = v} -> v) stats
            , cctrQ25Ratio = fmap (\ComboTimingStats{ctsQ25Ratio = v} -> v) stats
            , cctrQ75Ratio = fmap (\ComboTimingStats{ctsQ75Ratio = v} -> v) stats
            , cctrMadRatio = fmap (\ComboTimingStats{ctsMadRatio = v} -> v) stats
            , cctrMeanLift = fmap (\ComboTimingStats{ctsMeanLift = v} -> v) stats
            , cctrMedianLift = fmap (\ComboTimingStats{ctsMedianLift = v} -> v) stats
            , cctrMedianObservedDuration = medianObservedDuration
            , cctrMedianOptimalDuration = medianOptimalDuration
            , cctrQ75OptimalDuration = q75OptimalDuration
            , cctrRecommendedMaxHoldBars = recommendedMaxHoldBars
            }
  where
    normalizeTrade trade
        | ctComboId trade == "unknown" || null (ctComboId trade) = trade{ctComboId = comboId}
        | otherwise = trade

sampledTrade :: V.Vector Double -> ComboTrade -> Maybe (ComboTrade, TradeTimingSample)
sampledTrade prices trade = do
    sample <- sampleForTrade prices trade
    pure (trade, sample)

{- | Proof sketch for the maxHoldBars retune contract:

1. The modeled candidate space is the bounded exhaustive set [1 .. max observed duration].
   Any larger hold window is observationally equivalent to the baseline because modeled exits use
   min(observedExit, entry + holdBars).
2. `preservesBaselineQuality` only accepts candidates whose modeled final equity and expectancy
   are non-decreasing versus the observed baseline.
3. Candidates are scanned in ascending hold-bar order; modeled durations are monotone in holdBars,
   so the first accepted candidate is the smallest hold window and fastest-payback member of any
   equal-quality preserving set.
-}
recommendedCloseTimingHold :: V.Vector Double -> [(ComboTrade, TradeTimingSample)] -> Maybe Int
recommendedCloseTimingHold _ [] = Nothing
recommendedCloseTimingHold prices sampledTrades =
    case positiveDurationPairs sampledTrades of
        [] -> Nothing
        retunePairs ->
            let baselineMaxHoldBars = maximum (map (ttsObservedDuration . snd) retunePairs)
                baselineQuality = observedBaselineQuality (map snd retunePairs)
                candidates = modeledMaxHoldBarsCandidates prices retunePairs baselineMaxHoldBars
             in case earliestPreservingCandidate baselineQuality candidates of
                    Just MaxHoldBarsCandidate{mhbcHoldBars = holdBars}
                        | holdBars < baselineMaxHoldBars -> Just holdBars
                    _ -> Nothing

positiveDurationPairs :: [(ComboTrade, TradeTimingSample)] -> [(ComboTrade, TradeTimingSample)]
positiveDurationPairs = filter (\(_, sample) -> ttsObservedDuration sample > 0)

modeledMaxHoldBarsCandidates :: V.Vector Double -> [(ComboTrade, TradeTimingSample)] -> Int -> [MaxHoldBarsCandidate]
modeledMaxHoldBarsCandidates prices sampledTrades baselineMaxHoldBars =
    mapMaybe (modeledMaxHoldBarsCandidate prices sampledTrades) [1 .. baselineMaxHoldBars]

modeledMaxHoldBarsCandidate :: V.Vector Double -> [(ComboTrade, TradeTimingSample)] -> Int -> Maybe MaxHoldBarsCandidate
modeledMaxHoldBarsCandidate prices sampledTrades holdBars
    | holdBars <= 0 = Nothing
    | otherwise = do
        modeledOutcomes <- mapM (modeledTradeOutcome prices holdBars) sampledTrades
        let returns = map fst modeledOutcomes
            durations = map snd modeledOutcomes
        pure
            MaxHoldBarsCandidate
                { mhbcHoldBars = holdBars
                , mhbcQuality = qualityFromReturns returns durations
                }

modeledTradeOutcome :: V.Vector Double -> Int -> (ComboTrade, TradeTimingSample) -> Maybe (Double, Int)
modeledTradeOutcome prices holdBars (trade, _) = do
    let entryIndex = ctEntryIndex trade
        observedExitIndex = ctExitIndex trade
        modeledExitIndex = min observedExitIndex (entryIndex + holdBars)
    modeledReturn <- returnAt prices trade modeledExitIndex
    pure (modeledReturn, max 0 (modeledExitIndex - entryIndex))

observedBaselineQuality :: [TradeTimingSample] -> MaxHoldBarsQuality
observedBaselineQuality samples =
    qualityFromReturns
        (map ttsObservedReturn samples)
        (map ttsObservedDuration samples)

qualityFromReturns :: [Double] -> [Int] -> MaxHoldBarsQuality
qualityFromReturns returns durations =
    MaxHoldBarsQuality
        { mhqFinalEquity = foldl' (*) 1 (map equityFactor returns)
        , mhqExpectancy = mean returns
        , mhqMeanDuration = meanDurations durations
        }

meanDurations :: [Int] -> Double
meanDurations [] = 0
meanDurations xs = fromIntegral (sum xs) / fromIntegral (length xs)

equityFactor :: Double -> Double
equityFactor ret
    | isFinite ret = max 0 (1 + ret)
    | otherwise = 0

preservesBaselineQuality :: MaxHoldBarsQuality -> MaxHoldBarsCandidate -> Bool
preservesBaselineQuality baseline MaxHoldBarsCandidate{mhbcQuality = quality} =
    mhqFinalEquity quality + qualityTolerance >= mhqFinalEquity baseline
        && mhqExpectancy quality + qualityTolerance >= mhqExpectancy baseline

earliestPreservingCandidate :: MaxHoldBarsQuality -> [MaxHoldBarsCandidate] -> Maybe MaxHoldBarsCandidate
earliestPreservingCandidate _ [] = Nothing
earliestPreservingCandidate baseline (candidate : rest)
    | preservesBaselineQuality baseline candidate = Just candidate
    | otherwise = earliestPreservingCandidate baseline rest

qualityTolerance :: Double
qualityTolerance = 1e-12

timingRatio :: TradeTimingSample -> Double
timingRatio sample
    | ttsObservedDuration sample <= 0 = 0
    | otherwise =
        clampRatio
            ( fromIntegral (ttsOptimalDuration sample)
                / fromIntegral (ttsObservedDuration sample)
            )

mean :: [Double] -> Double
mean [] = 0
mean xs = sum xs / fromIntegral (length xs)

mad :: [Double] -> Double
mad [] = 0
mad xs =
    let med = boundedPercentile 0.5 xs
        deviations = map (abs . subtract med) xs
     in boundedPercentile 0.5 deviations

validObservation :: CloseTimingObservation -> Bool
validObservation x =
    case observationRatioParts x of
        Just (num, denom) -> num >= 0 && num <= 2 * denom
        Nothing -> False

observationRatio :: CloseTimingObservation -> Double
observationRatio x =
    case observationRatioParts x of
        Just (num, denom) -> clampRatio (fromInteger num / fromInteger denom)
        Nothing -> 0

observationRatioParts :: CloseTimingObservation -> Maybe (Integer, Integer)
observationRatioParts x =
    let denom = integerDelta (ctoOpenAtMs x) (ctoCloseAtMs x)
        num = integerDelta (ctoOpenAtMs x) (ctoOptimalCloseAtMs x)
     in if denom > 0 then Just (num, denom) else Nothing

positiveDurationInteger :: Int -> Integer
positiveDurationInteger = max 1 . toInteger

integerDelta :: Int -> Int -> Integer
integerDelta start end = toInteger end - toInteger start

nonNegativeIntegerDelta :: Int -> Int -> Integer
nonNegativeIntegerDelta start end = max 0 (integerDelta start end)

decisionTargetBand :: CloseTimingStats -> (Double, Double)
decisionTargetBand CloseTimingStats{ctsMedianRatio = medianRatio0, ctsQ75Ratio = q75Ratio0} =
    let medianRatio = clampRatio medianRatio0
        q75Ratio = max medianRatio (clampRatio q75Ratio0)
     in (medianRatio, q75Ratio)

boundedPercentile :: Double -> [Double] -> Double
boundedPercentile p = clampRatio . percentile p

orderQuartiles :: Double -> Double -> Double -> (Double, Double, Double)
orderQuartiles q25 q50 q75 =
    case sort (map clampRatio [q25, q50, q75]) of
        [a, b, c] -> (a, b, c)
        _ -> (0, 0, 0)

-- Non-finite budgets collapse to beta=0, preserving the median-target policy.
normalizeRiskBudget :: Double -> Double
normalizeRiskBudget x
    | isFinite x = clamp 0 1 x
    | otherwise = 0

mix :: Double -> Double -> Double -> Double
mix w a b = (1 - w) * a + w * b

percentile :: Double -> [Double] -> Double
percentile _ [] = 1
percentile p xs =
    let ys = sort xs
        n = length ys
        idx = floor (clamp 0 1 p * fromIntegral (n - 1))
     in ys !! idx

percentileInt :: Double -> [Int] -> Maybe Int
percentileInt _ [] = Nothing
percentileInt p xs =
    let ys = sort xs
        n = length ys
        idx = floor (clamp 0 1 p * fromIntegral (n - 1))
     in Just (ys !! idx)

clampRatio :: Double -> Double
clampRatio = clamp 0 2

clamp :: Double -> Double -> Double -> Double
clamp lo hi = max lo . min hi

safeAt :: V.Vector a -> Int -> Maybe a
safeAt xs i
    | i < 0 = Nothing
    | otherwise = xs V.!? i

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)
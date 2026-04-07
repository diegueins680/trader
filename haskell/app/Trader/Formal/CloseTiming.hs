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
    minimumCloseTimingSamples,
    minimumPositiveLiftSupportSamples,
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

{- | Combo-level report used to retune timing-based exits from historical trades.
Retunes fail closed unless the positive-lift q75 candidate is supported by
enough analyzed trades, is conservatively capped to the last hold bucket with
enough profitable-support evidence, the exact recommended duration bucket itself
has enough profitable-support samples and positive estimated lift, matches an
analyzed hold bucket, and stays within the analyzed hold window.
-}
data ComboCloseTimingReport = ComboCloseTimingReport
    { cctrComboId :: !String
    , cctrSampleCount :: !Int
    , cctrPositiveLiftSampleCount :: !Int
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
    , cctrRecommendedMaxHoldBarsEvidenceDuration :: !(Maybe Int)
    , cctrRecommendedMaxHoldBarsPositiveLiftSampleCount :: !(Maybe Int)
    , cctrRecommendedMaxHoldBarsMeanLift :: !(Maybe Double)
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
        samples = timingSamples prices normalizedTrades
        stats = summarizeComboTiming samples
        positiveLiftSampleCount = positiveLiftSupportCount samples
        medianObservedDuration = percentileInt 0.5 (map ttsObservedDuration samples)
        medianOptimalDuration = percentileInt 0.5 (map ttsOptimalDuration samples)
        q75OptimalDuration = supportedPositiveLiftDuration samples
        supportedHorizon = supportedPositiveLiftHorizon samples
        analyzedHoldBars = analyzedHoldBarDomain samples
        analyzedUpperBound = analyzedHoldWindowUpperBound samples
        recommendedMaxHoldBars =
            recommendedCloseTimingHold
                stats
                positiveLiftSampleCount
                q75OptimalDuration
                supportedHorizon
                analyzedUpperBound
                analyzedHoldBars
                samples
        (recommendedEvidenceDuration, recommendedPositiveLiftSampleCount, recommendedMeanLift) =
            recommendedCloseTimingEvidence recommendedMaxHoldBars samples
     in ComboCloseTimingReport
            { cctrComboId = comboId
            , cctrSampleCount = length samples
            , cctrPositiveLiftSampleCount = positiveLiftSampleCount
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
            , cctrRecommendedMaxHoldBarsEvidenceDuration = recommendedEvidenceDuration
            , cctrRecommendedMaxHoldBarsPositiveLiftSampleCount = recommendedPositiveLiftSampleCount
            , cctrRecommendedMaxHoldBarsMeanLift = recommendedMeanLift
            }
  where
    normalizeTrade trade
        | ctComboId trade == "unknown" || null (ctComboId trade) = trade{ctComboId = comboId}
        | otherwise = trade

minimumCloseTimingSamples :: Int
minimumCloseTimingSamples = 5

minimumPositiveLiftSupportSamples :: Int
minimumPositiveLiftSupportSamples = 3

positiveLiftSamples :: [TradeTimingSample] -> [TradeTimingSample]
positiveLiftSamples =
    filter (\sample -> ttsReturnLift sample > 0 && ttsOptimalDuration sample > 0)

positiveLiftSupportDurations :: [TradeTimingSample] -> [Int]
positiveLiftSupportDurations =
    map ttsOptimalDuration . positiveLiftSamples

positiveLiftSupportCount :: [TradeTimingSample] -> Int
positiveLiftSupportCount = length . positiveLiftSupportDurations

hasMinimumPositiveLiftSupport :: [TradeTimingSample] -> Bool
hasMinimumPositiveLiftSupport samples =
    positiveLiftSupportCount samples >= minimumPositiveLiftSupportSamples

supportedPositiveLiftDuration :: [TradeTimingSample] -> Maybe Int
supportedPositiveLiftDuration samples
    | not (hasMinimumPositiveLiftSupport samples) = Nothing
    | otherwise = percentileInt 0.75 (positiveLiftSupportDurations samples)

positiveLiftSupportHoldBarDomain :: [TradeTimingSample] -> [Int]
positiveLiftSupportHoldBarDomain =
    analyzedHoldBarDomain . positiveLiftSamples

supportedPositiveLiftHorizon :: [TradeTimingSample] -> Maybe Int
supportedPositiveLiftHorizon samples =
    case filter hasMinimumSupport (positiveLiftSupportHoldBarDomain samples) of
        [] -> Nothing
        supportedHoldBars -> Just (maximum supportedHoldBars)
  where
    hasMinimumSupport holdBars =
        closeTimingRecommendationHasMinimumBucketSupport holdBars samples

holdDurationPositiveLiftSupportSamples :: Int -> [TradeTimingSample] -> [TradeTimingSample]
holdDurationPositiveLiftSupportSamples holdBars =
    filter (\sample -> ttsOptimalDuration sample == holdBars) . positiveLiftSamples

holdDurationPositiveLiftSupportCount :: Int -> [TradeTimingSample] -> Int
holdDurationPositiveLiftSupportCount holdBars =
    length . holdDurationPositiveLiftSupportSamples holdBars

closeTimingRecommendationHasMinimumBucketSupport :: Int -> [TradeTimingSample] -> Bool
closeTimingRecommendationHasMinimumBucketSupport holdBars samples =
    holdDurationPositiveLiftSupportCount holdBars samples >= minimumPositiveLiftSupportSamples

legacyCloseTimingRecommendationEligible :: ComboTimingStats -> Int -> Bool
legacyCloseTimingRecommendationEligible stats q75Bars =
    ctsMedianLift stats > 0 && q75Bars > 0

positiveLiftSupportEvidenceOrdered :: ComboTimingStats -> Int -> Bool
positiveLiftSupportEvidenceOrdered stats positiveLiftCount =
    positiveLiftCount >= minimumPositiveLiftSupportSamples
        && positiveLiftCount <= ctsSampleCount stats

sampleHoldWindowUpperBound :: TradeTimingSample -> Int
sampleHoldWindowUpperBound sample =
    max 0 (2 * ttsObservedDuration sample)

analyzedHoldWindowUpperBound :: [TradeTimingSample] -> Maybe Int
analyzedHoldWindowUpperBound [] = Nothing
analyzedHoldWindowUpperBound samples =
    Just (maximum (map sampleHoldWindowUpperBound samples))

analyzedHoldBarDomain :: [TradeTimingSample] -> [Int]
analyzedHoldBarDomain =
    mapMaybe groupHead
        . groupBy (==)
        . sort
        . filter (> 0)
        . map ttsOptimalDuration
  where
    groupHead [] = Nothing
    groupHead (holdBars : _) = Just holdBars

closeTimingRecommendationWithinAnalyzedBounds :: Int -> Int -> Bool
closeTimingRecommendationWithinAnalyzedBounds q75Bars analyzedUpperBound =
    q75Bars > 0 && q75Bars <= analyzedUpperBound

closeTimingRecommendationWithinAnalyzedDomain :: Int -> [Int] -> Bool
closeTimingRecommendationWithinAnalyzedDomain q75Bars analyzedHoldBars =
    q75Bars > 0 && q75Bars `elem` analyzedHoldBars

holdDurationLiftSamples :: Int -> [TradeTimingSample] -> [Double]
holdDurationLiftSamples holdBars =
    map ttsReturnLift . filter (\sample -> ttsOptimalDuration sample == holdBars)

holdDurationEstimatedLift :: Int -> [TradeTimingSample] -> Maybe Double
holdDurationEstimatedLift holdBars samples =
    case holdDurationLiftSamples holdBars samples of
        [] -> Nothing
        lifts -> Just (mean lifts)

closeTimingRecommendationHasPositiveEstimatedLift :: Int -> [TradeTimingSample] -> Bool
closeTimingRecommendationHasPositiveEstimatedLift holdBars samples =
    case holdDurationEstimatedLift holdBars samples of
        Just liftEstimate -> liftEstimate > 0
        Nothing -> False

closeTimingRecommendationEligible :: ComboTimingStats -> Int -> Int -> Int -> [Int] -> [TradeTimingSample] -> Bool
closeTimingRecommendationEligible stats positiveLiftCount q75Bars analyzedUpperBound analyzedHoldBars samples =
    ctsSampleCount stats >= minimumCloseTimingSamples
        && positiveLiftSupportEvidenceOrdered stats positiveLiftCount
        && closeTimingRecommendationWithinAnalyzedBounds q75Bars analyzedUpperBound
        && closeTimingRecommendationWithinAnalyzedDomain q75Bars analyzedHoldBars
        && closeTimingRecommendationHasMinimumBucketSupport q75Bars samples
        && closeTimingRecommendationHasPositiveEstimatedLift q75Bars samples
        && legacyCloseTimingRecommendationEligible stats q75Bars

{- | Formal invariant / proof sketch for upward hold retunes:

1. Sparse profitable support fails closed: if the profitable-support sample is empty or has
   fewer than `minimumPositiveLiftSupportSamples` members, then
   `supportedPositiveLiftDuration = Nothing`, `supportedPositiveLiftHorizon = Nothing`,
   and `recommendedMaxHoldBars = Nothing`.
2. Even when positive-lift support exists, low total sample count still fails closed:
   `ctsSampleCount < minimumCloseTimingSamples` keeps `recommendedMaxHoldBars = Nothing`.
3. Evidence counters stay ordered: `positiveLiftCount <= ctsSampleCount` is required before
   a recommendation can exist, so malformed or inflated support counts fail closed.
4. Sparse-tail upward retunes are capped before validation: if
   `supportedPositiveLiftDuration = Just q75Bars` and
   `supportedPositiveLiftHorizon = Just supportedHorizon`, then the candidate
   recommendation is `capRecommendationToSupportedPositiveLiftHorizon q75Bars supportedHorizon`,
   so thin positive-lift tails can never extend a retune past the last hold bucket with
   enough profitable-support samples.
5. Therefore any emitted recommendation satisfies
   `recommendedMaxHoldBars <= supportedPositiveLiftHorizon`; weakening or inconsistent
   profitable-support evidence can only shrink or remove `supportedPositiveLiftHorizon`,
   so the recommendation can stay unchanged or decrease, never increase.
6. Any emitted recommendation must match an analyzed hold bucket: `recommendedHoldBars > 0`
   and `recommendedHoldBars `elem` analyzedHoldBarDomain samples` are both required before
   emitting a retune.
7. The exact recommended hold bucket must itself carry enough profitable support and
   positive estimated lift:
   `holdDurationPositiveLiftSupportCount recommendedHoldBars samples >= minimumPositiveLiftSupportSamples`
   and `mean [ttsReturnLift s | s <- samples, ttsOptimalDuration s == recommendedHoldBars] > 0`
   are both required before emitting a retune.
8. Any emitted recommendation stays inside the analyzed window:
   `recommendedHoldBars <= analyzedHoldWindowUpperBound` is required before emitting
   `Just recommendedHoldBars`.
9. Non-positive combo-level lift fails closed: descriptive ratios may still exist, but
   `ctsMedianLift <= 0` keeps `recommendedMaxHoldBars = Nothing`.
10. Any emitted recommendation serializes exact-bucket evidence:
    `cctrRecommendedMaxHoldBarsEvidenceDuration == Just recommendedHoldBars`,
    `cctrRecommendedMaxHoldBarsPositiveLiftSampleCount == Just (holdDurationPositiveLiftSupportCount recommendedHoldBars samples)`,
    and `cctrRecommendedMaxHoldBarsMeanLift == holdDurationEstimatedLift recommendedHoldBars samples`.
-}
capRecommendationToSupportedPositiveLiftHorizon :: Int -> Int -> Int
capRecommendationToSupportedPositiveLiftHorizon q75Bars supportedHorizon =
    min q75Bars supportedHorizon

recommendedCloseTimingHold :: Maybe ComboTimingStats -> Int -> Maybe Int -> Maybe Int -> Maybe Int -> [Int] -> [TradeTimingSample] -> Maybe Int
recommendedCloseTimingHold (Just stats) positiveLiftCount (Just q75Bars) (Just supportedHorizon) (Just analyzedUpperBound) analyzedHoldBars samples
    | closeTimingRecommendationEligible stats positiveLiftCount recommendedHoldBars analyzedUpperBound analyzedHoldBars samples =
        Just recommendedHoldBars
  where
    recommendedHoldBars =
        capRecommendationToSupportedPositiveLiftHorizon q75Bars supportedHorizon
recommendedCloseTimingHold _ _ _ _ _ _ _ = Nothing

recommendedCloseTimingEvidence :: Maybe Int -> [TradeTimingSample] -> (Maybe Int, Maybe Int, Maybe Double)
recommendedCloseTimingEvidence Nothing _ = (Nothing, Nothing, Nothing)
recommendedCloseTimingEvidence (Just holdBars) samples =
    ( Just holdBars
    , Just (holdDurationPositiveLiftSupportCount holdBars samples)
    , holdDurationEstimatedLift holdBars samples
    )

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
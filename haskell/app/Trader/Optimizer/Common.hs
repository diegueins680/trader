{-# LANGUAGE OverloadedStrings #-}

module Trader.Optimizer.Common (
    AutoOptimizerScopeSelection (..),
    appliedCloseTimingMaxHoldBars,
    applyCloseTimingMetrics,
    autoOptimizerRequiredBarsForSweep,
    closeTimingReportFromBacktest,
    normalizeObjectiveCode,
    objectiveScore,
    objectiveScoreWithConfig,
    selectAutoOptimizerScopes,
) where

import Control.Applicative ((<|>))
import Data.Aeson (Value (..), object, (.=))
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.Char (isSpace, toLower)
import Data.List (dropWhileEnd)
import Data.Maybe (fromMaybe, mapMaybe)
import Data.Scientific (Scientific, toBoundedInteger, toRealFloat)
import qualified Data.Text as T
import qualified Data.Vector as V
import Text.Read (readMaybe)

import Trader.Duration (lookbackBarsFrom, parseIntervalSeconds, positiveFiniteDuration)
import Trader.Formal.CloseTiming (
    ComboCloseTimingReport (..),
    ComboTrade (..),
    acceptedCloseTimingMaxHoldBars,
    analyzeComboCloseTiming,
    closeTimingRecommendationAccepted,
    minimumCloseTimingSamples,
    minimumPositiveLiftSupportSamples,
 )
import Trader.Optimization (TuneObjective (..), parseTuneObjective, tuneObjectiveCode)
import Trader.RoiScore (
    RoiScoreConfig (..),
    defaultRoiScoreConfig,
    paybackBonusForWithConfig,
    roiEvidencePenaltyWithConfig,
    sanitizeRoiScoreConfig,
 )

-- Keep baseline backtest helpers separate from the optimizer search engine so
-- trader-hs does not need to compile optimizer internals on simple execution paths.

data AutoOptimizerScopeSelection = AutoOptimizerScopeSelection
    { aosScopes :: ![(String, String)]
    , aosCappedScopes :: ![(String, String)]
    }
    deriving (Eq, Show)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

metricFloat :: Maybe (KM.KeyMap Value) -> String -> Double -> Double
metricFloat m key def =
    case m of
        Nothing -> def
        Just metrics ->
            case KM.lookup (Key.fromString key) metrics of
                Just (Number n) -> fromMaybe def (scientificToDouble n)
                _ -> def

metricPositiveDurationMaybe :: Maybe (KM.KeyMap Value) -> String -> Maybe Double
metricPositiveDurationMaybe m key =
    case m of
        Nothing -> Nothing
        Just metrics ->
            case KM.lookup (Key.fromString key) metrics of
                Just (Number n) -> scientificToDouble n >>= positiveFiniteDuration
                Just (String s) ->
                    let trimmed = trim (T.unpack s)
                     in if null trimmed
                            then Nothing
                            else readMaybe trimmed >>= positiveFiniteDuration
                _ -> Nothing

metricInt :: Maybe (KM.KeyMap Value) -> String -> Int -> Int
metricInt m key def =
    case m of
        Nothing -> def
        Just metrics ->
            case KM.lookup (Key.fromString key) metrics of
                Just (Bool v) -> if v then 1 else 0
                Just (Number n) -> fromMaybe def (scientificToBoundedInt n)
                _ -> def

scientificToDouble :: Scientific -> Maybe Double
scientificToDouble n =
    let d = toRealFloat n
     in if isInfinite d || isNaN d then Nothing else Just d

scientificToBoundedInt :: Scientific -> Maybe Int
scientificToBoundedInt = toBoundedInteger

valueObjectAt :: Value -> String -> Maybe (KM.KeyMap Value)
valueObjectAt val key =
    case val of
        Object obj ->
            case KM.lookup (Key.fromString key) obj of
                Just (Object v) -> Just v
                _ -> Nothing
        _ -> Nothing

coerceFloatValue :: Value -> Maybe Double
coerceFloatValue value =
    case value of
        Null -> Nothing
        Bool v -> Just (if v then 1 else 0)
        Number n -> scientificToDouble n
        String s ->
            let trimmed = trim (T.unpack s)
             in if null trimmed
                    then Nothing
                    else case reads trimmed of
                        [(v, "")] -> if isInfinite v || isNaN v then Nothing else Just v
                        _ -> Nothing
        _ -> Nothing

coerceIntValue :: Value -> Maybe Int
coerceIntValue value =
    case value of
        Null -> Nothing
        Bool v -> Just (if v then 1 else 0)
        Number n -> scientificToBoundedInt n
        String s ->
            let trimmed = trim (T.unpack s)
             in if null trimmed
                    then Nothing
                    else case readMaybe trimmed :: Maybe Scientific of
                        Just n -> scientificToBoundedInt n
                        _ -> Nothing
        _ -> Nothing

objectiveScore :: KM.KeyMap Value -> String -> Double -> Double -> Double
objectiveScore = objectiveScoreWithConfig defaultRoiScoreConfig

objectiveScoreWithConfig :: RoiScoreConfig -> KM.KeyMap Value -> String -> Double -> Double -> Double
objectiveScoreWithConfig roiCfg0 metrics objective penaltyMaxDd penaltyTurnover =
    let finalEq = metricFloat (Just metrics) "finalEquity" 0
        maxDd = metricFloat (Just metrics) "maxDrawdown" 0
        cvar95 = metricFloat (Just metrics) "cvar95" 0
        sharpe = metricFloat (Just metrics) "sharpe" 0
        annRet = metricFloat (Just metrics) "annualizedReturn" 0
        turnover = metricFloat (Just metrics) "turnover" 0
        maxDdN = max 0 maxDd
        cvar95N = max 0 cvar95
        turnoverN = max 0 turnover
        pDd = max 0 penaltyMaxDd
        pTurn = max 0 penaltyTurnover
        avgTradeReturn = metricFloat (Just metrics) "avgTradeReturn" 0
        paybackDuration = metricPositiveDurationMaybe (Just metrics) "avgHoldingPeriods"
        exposure = metricFloat (Just metrics) "exposure" 0
        roundTrips = metricInt (Just metrics) "roundTrips" 0
        tradeCount = metricInt (Just metrics) "tradeCount" 0
        activityCount = max roundTrips tradeCount
        roiCfg = sanitizeRoiScoreConfig roiCfg0
        sparseEvidencePenalty = roiEvidencePenaltyWithConfig roiCfg roundTrips activityCount exposure
        paybackBonus
            | avgTradeReturn <= 0 = 0
            | activityCount < rscMinimumActivityFloor roiCfg = 0
            | exposure < rscMinimumExposureFloor roiCfg = 0
            | otherwise = maybe 0 (paybackBonusForWithConfig roiCfg) paybackDuration
        baseScore =
            case parseTuneObjective objective of
                Right TuneFinalEquity -> finalEq
                Right TuneAnnualizedEquity -> annRet
                Right TuneRoi ->
                    annRet - pDd * (maxDdN + cvar95N) - pTurn * turnoverN + rscExpectancyRewardWeight roiCfg * avgTradeReturn + paybackBonus
                Right TuneSharpe -> sharpe
                Right TuneCalmar ->
                    if maxDdN <= 0
                        then annRet
                        else annRet / max 1e-12 maxDdN
                Right TuneEquityDd -> finalEq - pDd * maxDdN
                Right TuneEquityDdTurnover -> finalEq - pDd * maxDdN - pTurn * turnoverN
                Left _ -> finalEq
     in case parseTuneObjective objective of
            Right TuneRoi -> baseScore
            _ -> baseScore - sparseEvidencePenalty

normalizeObjectiveCode :: String -> Either String String
normalizeObjectiveCode raw = tuneObjectiveCode <$> parseTuneObjective raw

autoOptimizerRequiredBarsForSweep :: Double -> Double -> Int -> Maybe Int
autoOptimizerRequiredBarsForSweep backtestRatio tuneRatio lookbackBars
    | backtestRatio <= 0 || backtestRatio >= 1 = Nothing
    | tuneRatio <= 0 || tuneRatio >= 1 = Nothing
    | otherwise =
        let minRequired0 = lookbackBars + 3
            denom = max 1e-12 ((1 - backtestRatio) * (1 - tuneRatio))
            minRequired1 = max minRequired0 (ceiling ((fromIntegral lookbackBars + 1) / denom) + 2)
            minTrain = ceiling (2 / tuneRatio)
            minRequired2 = max minRequired1 (ceiling (fromIntegral minTrain / max 1e-12 (1 - backtestRatio)) + 2)
         in Just minRequired2

selectAutoOptimizerScopes ::
    Bool ->
    Int ->
    Double ->
    Double ->
    [String] ->
    [String] ->
    AutoOptimizerScopeSelection
selectAutoOptimizerScopes autoCappedLookbacksEnabled maxPoints backtestRatio tuneRatio intervals lookbackWindows =
    AutoOptimizerScopeSelection
        { aosScopes = optimizerScopes
        , aosCappedScopes =
            [ (interval, lookbackWindow)
            | (interval, lookbackWindow) <- optimizerScopes
            , lookbackWindow `notElem` lookbackWindows
            ]
        }
  where
    scopeFeasible interval lookbackWindow =
        case lookbackBarsFrom interval lookbackWindow of
            Left _ -> False
            Right lookbackBars ->
                lookbackBars >= 2
                    && maybe False (<= maxPoints) (autoOptimizerRequiredBarsForSweep backtestRatio tuneRatio lookbackBars)

    maxFeasibleLookbackBars =
        let go lo hi best
                | lo > hi = best
                | otherwise =
                    let mid = (lo + hi) `div` 2
                     in if maybe False (<= maxPoints) (autoOptimizerRequiredBarsForSweep backtestRatio tuneRatio mid)
                            then go (mid + 1) hi (Just mid)
                            else go lo (mid - 1) best
         in go 2 maxPoints Nothing

    renderLookbackSeconds totalSeconds
        | totalSeconds `mod` 86400 == 0 = show (max 1 (totalSeconds `div` 86400)) ++ "d"
        | totalSeconds `mod` 3600 == 0 = show (max 1 (totalSeconds `div` 3600)) ++ "h"
        | totalSeconds `mod` 60 == 0 = show (max 1 (totalSeconds `div` 60)) ++ "m"
        | otherwise = show (max 1 totalSeconds) ++ "s"

    cappedLookbackWindowFor interval = do
        if autoCappedLookbacksEnabled then Just () else Nothing
        intervalSec <- parseIntervalSeconds interval
        lookbackBars <- maxFeasibleLookbackBars
        let lookbackWindow = renderLookbackSeconds (lookbackBars * intervalSec)
        if scopeFeasible interval lookbackWindow then Just lookbackWindow else Nothing

    scopesForInterval interval =
        let configured =
                [ (interval, lookbackWindow)
                | lookbackWindow <- lookbackWindows
                , scopeFeasible interval lookbackWindow
                ]
         in case configured of
                [] -> maybe [] (\lookbackWindow -> [(interval, lookbackWindow)]) (cappedLookbackWindowFor interval)
                _ -> configured

    optimizerScopes = concatMap scopesForInterval intervals

coerceFloatArray :: Value -> Maybe [Double]
coerceFloatArray value =
    case value of
        Array xs -> traverse coerceFloatValue (V.toList xs)
        _ -> Nothing

extractCloseTimingInputs :: String -> Maybe Value -> Maybe ([Double], [ComboTrade])
extractCloseTimingInputs comboId raw = do
    v <- raw
    bt <- valueObjectAt v "backtest"
    pricesValue <- KM.lookup (Key.fromString "prices") bt
    tradesValue <- KM.lookup (Key.fromString "trades") bt
    prices <- coerceFloatArray pricesValue
    let positions =
            case KM.lookup (Key.fromString "positions") bt of
                Just value -> fromMaybe [] (coerceFloatArray value)
                Nothing -> []
    trades <- coerceCloseTimingTrades comboId prices positions tradesValue
    pure (prices, trades)

coerceCloseTimingTrades :: String -> [Double] -> [Double] -> Value -> Maybe [ComboTrade]
coerceCloseTimingTrades comboId prices positions value =
    case value of
        Array trades -> Just (mapMaybe (coerceCloseTimingTrade comboId prices positions) (V.toList trades))
        _ -> Nothing

coerceCloseTimingTrade :: String -> [Double] -> [Double] -> Value -> Maybe ComboTrade
coerceCloseTimingTrade comboId prices positions value =
    case value of
        Object trade -> do
            entryIdx <- KM.lookup (Key.fromString "entryIndex") trade >>= coerceIntValue
            exitIdx <- KM.lookup (Key.fromString "exitIndex") trade >>= coerceIntValue
            entryPrice <- safeAtList prices entryIdx
            let side = resolveCloseTimingTradeSide trade entryIdx exitIdx prices positions
            if entryPrice > 0 && side /= 0
                then
                    Just
                        ComboTrade
                            { ctComboId = comboId
                            , ctEntryIndex = entryIdx
                            , ctExitIndex = exitIdx
                            , ctEntryPrice = entryPrice
                            , ctSide = side
                            }
                else Nothing
        _ -> Nothing

resolveCloseTimingTradeSide :: KM.KeyMap Value -> Int -> Int -> [Double] -> [Double] -> Double
resolveCloseTimingTradeSide trade entryIdx exitIdx prices positions =
    fromMaybe 1 (sideFromPositions entryIdx positions <|> sideFromReturn trade entryIdx exitIdx prices)

sideFromPositions :: Int -> [Double] -> Maybe Double
sideFromPositions entryIdx positions = do
    pos <- safeAtList positions entryIdx
    let side = signum pos
    if isNaN side || isInfinite side || side == 0
        then Nothing
        else Just side

sideFromReturn :: KM.KeyMap Value -> Int -> Int -> [Double] -> Maybe Double
sideFromReturn trade entryIdx exitIdx prices = do
    retVal <- KM.lookup (Key.fromString "return") trade >>= coerceFloatValue
    entryPrice <- safeAtList prices entryIdx
    exitPrice <- safeAtList prices exitIdx
    inferTradeSideFromReturn entryPrice exitPrice retVal

inferTradeSideFromReturn :: Double -> Double -> Double -> Maybe Double
inferTradeSideFromReturn entryPrice exitPrice retVal
    | entryPrice <= 0 || exitPrice <= 0 = Nothing
    | isNaN retVal || isInfinite retVal = Nothing
    | otherwise =
        let priceMove = signum (exitPrice - entryPrice)
            retMove = signum retVal
            side = retMove * priceMove
         in if priceMove == 0 || retMove == 0 || side == 0 then Nothing else Just side

safeAtList :: [a] -> Int -> Maybe a
safeAtList xs idx
    | idx < 0 = Nothing
    | otherwise = go xs idx
  where
    go [] _ = Nothing
    go (y : ys) n
        | n == 0 = Just y
        | otherwise = go ys (n - 1)

closeTimingReportFromBacktest :: String -> Maybe Value -> ComboCloseTimingReport
closeTimingReportFromBacktest comboId raw =
    case extractCloseTimingInputs comboId raw of
        Just (prices, trades) -> analyzeComboCloseTiming comboId prices trades
        Nothing -> analyzeComboCloseTiming comboId [] []

appliedCloseTimingMaxHoldBars :: Maybe Int -> ComboCloseTimingReport -> Maybe Int
appliedCloseTimingMaxHoldBars = acceptedCloseTimingMaxHoldBars

applyCloseTimingMetrics ::
    Maybe (KM.KeyMap Value) ->
    Maybe Int ->
    Maybe Int ->
    ComboCloseTimingReport ->
    Maybe (KM.KeyMap Value)
applyCloseTimingMetrics metrics currentMaxHoldBars appliedMaxHoldBars report =
    let base = fromMaybe KM.empty metrics
        reportValue = closeTimingReportToValue currentMaxHoldBars appliedMaxHoldBars report
     in Just (KM.insert (Key.fromString "closeTiming") reportValue base)

closeTimingReportToValue :: Maybe Int -> Maybe Int -> ComboCloseTimingReport -> Value
closeTimingReportToValue currentMaxHoldBars appliedMaxHoldBars report =
    let recommendationAccepted =
            case cctrRecommendedMaxHoldBars report of
                Just recommended -> closeTimingRecommendationAccepted currentMaxHoldBars recommended report
                Nothing -> False
     in object
            [ "comboId" .= cctrComboId report
            , "sampleCount" .= cctrSampleCount report
            , "minimumSampleCount" .= minimumCloseTimingSamples
            , "positiveLiftSampleCount" .= cctrPositiveLiftSampleCount report
            , "minimumPositiveLiftSampleCount" .= minimumPositiveLiftSupportSamples
            , "medianRatio" .= cctrMedianRatio report
            , "q25Ratio" .= cctrQ25Ratio report
            , "q75Ratio" .= cctrQ75Ratio report
            , "madRatio" .= cctrMadRatio report
            , "meanLift" .= cctrMeanLift report
            , "medianLift" .= cctrMedianLift report
            , "medianObservedDuration" .= cctrMedianObservedDuration report
            , "medianOptimalDuration" .= cctrMedianOptimalDuration report
            , "q75OptimalDuration" .= cctrQ75OptimalDuration report
            , "recommendedMaxHoldBars" .= cctrRecommendedMaxHoldBars report
            , "recommendedMaxHoldBarsAccepted" .= recommendationAccepted
            , "recommendedMaxHoldBarsEvidenceDuration" .= cctrRecommendedMaxHoldBarsEvidenceDuration report
            , "recommendedMaxHoldBarsPositiveLiftSampleCount" .= cctrRecommendedMaxHoldBarsPositiveLiftSampleCount report
            , "recommendedMaxHoldBarsMeanLift" .= cctrRecommendedMaxHoldBarsMeanLift report
            , "originalMaxHoldBars" .= currentMaxHoldBars
            , "appliedMaxHoldBars" .= appliedMaxHoldBars
            , "positiveLift" .= maybe False (> 0) (cctrMedianLift report)
            ]

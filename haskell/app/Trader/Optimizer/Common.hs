{-# LANGUAGE OverloadedStrings #-}

module Trader.Optimizer.Common (
    appliedCloseTimingMaxHoldBars,
    applyCloseTimingMetrics,
    closeTimingReportFromBacktest,
    normalizeObjectiveCode,
    objectiveScore,
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

import Trader.Formal.CloseTiming (
    ComboCloseTimingReport (..),
    ComboTrade (..),
    analyzeComboCloseTiming,
    minimumCloseTimingSamples,
    minimumPositiveLiftSupportSamples,
 )
import Trader.Optimization (TuneObjective (..), parseTuneObjective, tuneObjectiveCode)

-- Keep baseline backtest helpers separate from the optimizer search engine so
-- trader-hs does not need to compile optimizer internals on simple execution paths.

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
objectiveScore metrics objective penaltyMaxDd penaltyTurnover =
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
        avgHoldingPeriods = metricFloat (Just metrics) "avgHoldingPeriods" 0
        exposure = metricFloat (Just metrics) "exposure" 0
        roundTrips = metricInt (Just metrics) "roundTrips" 0
        tradeCount = metricInt (Just metrics) "tradeCount" 0
        activityCount = max roundTrips tradeCount
        activityPenalty
            | activityCount <= 0 = 0.25
            | activityCount < 3 = fromIntegral (3 - activityCount) * 0.03
            | otherwise = 0
        exposurePenalty
            | exposure <= 0 = 0.05
            | exposure < 0.01 = 0.02
            | otherwise = 0
        paybackBonus
            | avgHoldingPeriods <= 0 = 0
            | otherwise = min 0.05 (1 / (1 + avgHoldingPeriods))
        baseScore =
            case parseTuneObjective objective of
                Right TuneFinalEquity -> finalEq
                Right TuneAnnualizedEquity -> annRet
                Right TuneRoi ->
                    annRet - pDd * (maxDdN + cvar95N) - pTurn * turnoverN + 0.5 * avgTradeReturn + paybackBonus
                Right TuneSharpe -> sharpe
                Right TuneCalmar ->
                    if maxDdN <= 0
                        then annRet
                        else annRet / max 1e-12 maxDdN
                Right TuneEquityDd -> finalEq - pDd * maxDdN
                Right TuneEquityDdTurnover -> finalEq - pDd * maxDdN - pTurn * turnoverN
                Left _ -> finalEq
     in baseScore - activityPenalty - exposurePenalty

normalizeObjectiveCode :: String -> Either String String
normalizeObjectiveCode raw = tuneObjectiveCode <$> parseTuneObjective raw

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
appliedCloseTimingMaxHoldBars currentMaxHoldBars report =
    case cctrRecommendedMaxHoldBars report of
        Just recommended
            | closeTimingRecommendationAccepted currentMaxHoldBars recommended report -> Just recommended
        _ -> currentMaxHoldBars

closeTimingRecommendationAccepted :: Maybe Int -> Int -> ComboCloseTimingReport -> Bool
closeTimingRecommendationAccepted currentMaxHoldBars recommended report
    | not (isMorePermissiveMaxHoldBars currentMaxHoldBars recommended) = True
    | otherwise =
        case
            ( cctrRecommendedMaxHoldBarsEvidenceDuration report
            , cctrRecommendedMaxHoldBarsPositiveLiftSampleCount report
            , cctrRecommendedMaxHoldBarsMeanLift report
            ) of
            (Just evidenceDuration, Just supportCount, Just meanLift) ->
                evidenceDuration == recommended
                    && closeTimingRecommendationHasContiguousPositiveLiftSupport recommended report
                    && supportCount >= minimumPositiveLiftSupportSamples
                    && meanLift > 0
            _ -> False

closeTimingRecommendationHasContiguousPositiveLiftSupport :: Int -> ComboCloseTimingReport -> Bool
closeTimingRecommendationHasContiguousPositiveLiftSupport recommended report =
    case cctrRecommendedMaxHoldBarsContiguousPositiveLiftHorizon report of
        Just contiguousHorizon -> recommended > 0 && recommended <= contiguousHorizon
        Nothing -> False

isMorePermissiveMaxHoldBars :: Maybe Int -> Int -> Bool
isMorePermissiveMaxHoldBars Nothing _ = False
isMorePermissiveMaxHoldBars (Just currentMaxHoldBars) recommendedMaxHoldBars =
    recommendedMaxHoldBars > currentMaxHoldBars

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
                Just recommended ->
                    closeTimingRecommendationAccepted currentMaxHoldBars recommended report
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
            , "recommendedMaxHoldBarsContiguousPositiveLiftHorizon" .= cctrRecommendedMaxHoldBarsContiguousPositiveLiftHorizon report
            , "recommendedMaxHoldBarsAccepted" .= recommendationAccepted
            , "recommendedMaxHoldBarsEvidenceDuration" .= cctrRecommendedMaxHoldBarsEvidenceDuration report
            , "recommendedMaxHoldBarsPositiveLiftSampleCount" .= cctrRecommendedMaxHoldBarsPositiveLiftSampleCount report
            , "recommendedMaxHoldBarsMeanLift" .= cctrRecommendedMaxHoldBarsMeanLift report
            , "originalMaxHoldBars" .= currentMaxHoldBars
            , "appliedMaxHoldBars" .= appliedMaxHoldBars
            , "positiveLift" .= maybe False (> 0) (cctrMedianLift report)
            ]
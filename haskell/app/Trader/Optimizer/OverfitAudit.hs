{-# LANGUAGE OverloadedStrings #-}

module Trader.Optimizer.OverfitAudit (
    OverfitTrial (..),
    optimizerOverfitAudit,
) where

import Data.Aeson (Value (..), object, (.=))
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.Scientific (toRealFloat)
import qualified Data.Text as T

data OverfitTrial = OverfitTrial
    { oatEligible :: !Bool
    , oatSearchEligible :: !Bool
    , oatScore :: !(Maybe Double)
    , oatMetrics :: !(Maybe (KM.KeyMap Value))
    }
    deriving (Eq, Show)

optimizerOverfitAudit :: [OverfitTrial] -> Maybe Value
optimizerOverfitAudit trials =
    let scored =
            [ (score, trial)
            | trial <- trials
            , Just score <- [finiteMaybe =<< oatScore trial]
            ]
        scoredCount = length scored
        eligibleCount = length (filter oatEligible trials)
        searchEligibleCount = length (filter oatSearchEligible trials)
        sharpeSamples =
            [ sharpe
            | trial <- trials
            , metrics <- maybeToList (oatMetrics trial)
            , Just sharpe <- [metricDouble metrics "sharpe"]
            ]
     in case scored of
            [] -> Nothing
            _ ->
                let (bestScore, bestTrial) = maximumByFst scored
                    scoreRankCount = length [s | (s, _) <- scored, s >= bestScore]
                    scorePValue :: Double
                    scorePValue = fromIntegral (1 + scoreRankCount) / fromIntegral (1 + scoredCount)
                    mBestMetrics = oatMetrics bestTrial
                    selectedSharpe = mBestMetrics >>= (`metricDouble` "sharpe")
                    selectedWfSummary = mBestMetrics >>= valueObjectAt "walkForwardSummary"
                    wfSharpeMean = selectedWfSummary >>= (`metricDouble` "sharpeMean")
                    wfSharpeStd = selectedWfSummary >>= (`metricDouble` "sharpeStd")
                    sharpeMean = meanMaybe sharpeSamples
                    sharpeStd = stddevMaybe sharpeSamples
                    penalty =
                        case sharpeStd of
                            Nothing -> Nothing
                            Just sd ->
                                let n = max 1 scoredCount
                                    p = 1 - (1 / fromIntegral (n + 1))
                                    z = normalInv p
                                 in finiteMaybe (max 0 sd * max 0 z)
                    deflatedSharpe =
                        case (selectedSharpe, penalty) of
                            (Just sr, Just pen) -> finiteMaybe (sr - pen)
                            _ -> selectedSharpe
                    instability =
                        case (wfSharpeMean, wfSharpeStd) of
                            (Just m, Just s)
                                | s >= 0 ->
                                    let denom = max 1e-12 (abs m)
                                     in finiteMaybe (s / denom)
                            _ -> Nothing
                    pboProxy =
                        case (wfSharpeMean, instability) of
                            (Just m, Just i)
                                | m <= 0 -> Just 1
                                | otherwise -> Just (clamp01 (i / (1 + i)))
                            _ -> Nothing
                 in Just $
                        object
                            [ "trialCount" .= length trials
                            , "scoredTrialCount" .= scoredCount
                            , "eligibleTrialCount" .= eligibleCount
                            , "searchEligibleTrialCount" .= searchEligibleCount
                            , "selectedScore" .= bestScore
                            , "empiricalBestScorePValue" .= scorePValue
                            , "selectedSharpe" .= selectedSharpe
                            , "sharpeMeanAcrossTrials" .= sharpeMean
                            , "sharpeStdAcrossTrials" .= sharpeStd
                            , "multipleTestingSharpePenalty" .= penalty
                            , "deflatedSharpeProxy" .= deflatedSharpe
                            , "walkForwardSharpeMean" .= wfSharpeMean
                            , "walkForwardSharpeStd" .= wfSharpeStd
                            , "pboProxy" .= pboProxy
                            ]

metricDouble :: KM.KeyMap Value -> String -> Maybe Double
metricDouble metrics key =
    case KM.lookup (Key.fromString key) metrics of
        Just (Number n) -> finiteMaybe (toRealFloat n)
        Just (String s) ->
            case reads (T.unpack s) of
                [(v, "")] -> finiteMaybe v
                _ -> Nothing
        _ -> Nothing

valueObjectAt :: String -> KM.KeyMap Value -> Maybe (KM.KeyMap Value)
valueObjectAt key metrics =
    case KM.lookup (Key.fromString key) metrics of
        Just (Object obj) -> Just obj
        _ -> Nothing

finiteMaybe :: Double -> Maybe Double
finiteMaybe x
    | isNaN x || isInfinite x = Nothing
    | otherwise = Just x

meanMaybe :: [Double] -> Maybe Double
meanMaybe xs =
    case filterFinite xs of
        [] -> Nothing
        clean -> Just (sum clean / fromIntegral (length clean))

stddevMaybe :: [Double] -> Maybe Double
stddevMaybe xs =
    case filterFinite xs of
        clean
            | length clean < 2 -> Nothing
            | otherwise ->
                let m = sum clean / fromIntegral (length clean)
                    var = sum (map (\x -> (x - m) * (x - m)) clean) / fromIntegral (length clean - 1)
                 in finiteMaybe (sqrt (max 0 var))

filterFinite :: [Double] -> [Double]
filterFinite = filter (\x -> not (isNaN x || isInfinite x))

maximumByFst :: [(Double, a)] -> (Double, a)
maximumByFst xs =
    case xs of
        [] -> error "maximumByFst: empty list"
        y : ys -> foldl pick y ys
  where
    pick best@(b, _) cand@(c, _)
        | c > b = cand
        | otherwise = best

maybeToList :: Maybe a -> [a]
maybeToList Nothing = []
maybeToList (Just x) = [x]

clamp01 :: Double -> Double
clamp01 x
    | isNaN x || isInfinite x = 0
    | otherwise = max 0 (min 1 x)

normalInv :: Double -> Double
normalInv p
    | p <= 0 = -(1 / 0)
    | p >= 1 = 1 / 0
    | p < plow =
        let q = sqrt (-(2 * log p))
         in (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    | p > phigh =
        let q = sqrt (-(2 * log (1 - p)))
         in -( (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
             )
    | otherwise =
        let q = p - 0.5
            r = q * q
         in (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
                * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
  where
    plow = 0.02425
    phigh = 1 - plow
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

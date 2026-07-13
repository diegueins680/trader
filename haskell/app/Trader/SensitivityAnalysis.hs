{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Trader.SensitivityAnalysis
Description : Local sensitivity analysis for trading parameters
Copyright   : (c) Trader Engineering Team, 2026
License     : MIT
Maintainer  : engineering@trader.system
Stability   : experimental
Portability : POSIX

This module implements local sensitivity analysis for trading parameters.
Instead of changing 8 parameters simultaneously (the anti-pattern observed
on 2026-05-11), this module varies ONE parameter at a time while holding
all others constant, measuring the isolated impact on trading performance.

Engineering principle: "Vary one thing, measure the effect, attribute the change."
-}
module Trader.SensitivityAnalysis (
    -- * Types
    SensitivityPoint (..),
    SensitivityReport (..),
    ParameterSpec (..),

    -- * Analysis
    runLocalSensitivity,
    runLocalSensitivityChecked,
    validateParameterSpec,
    parameterElasticity,
    mostSensitiveParameter,

    -- * JSON
    sensitivityReportToJson,
) where

import Data.Aeson (ToJSON (..), (.=))
import qualified Data.Aeson as Aeson
import Data.List (sortOn)
import Data.Ord (Down (..))
import Data.Text (Text)
import qualified Data.Text as T
import GHC.Generics (Generic)

-- | A single parameter value and its measured outcome.
data SensitivityPoint = SensitivityPoint
    { spParameterValue :: !Double
    , spSharpe :: !Double
    , spMaxDrawdown :: !Double
    , spTradeCount :: !Int
    , spWinRate :: !Double
    , spProfitFactor :: !Double
    }
    deriving (Eq, Show, Generic)

instance ToJSON SensitivityPoint where
    toJSON SensitivityPoint{..} =
        Aeson.object
            [ "parameterValue" .= spParameterValue
            , "sharpe" .= spSharpe
            , "maxDrawdown" .= spMaxDrawdown
            , "tradeCount" .= spTradeCount
            , "winRate" .= spWinRate
            , "profitFactor" .= spProfitFactor
            ]

-- | Specification for a parameter to be analyzed.
data ParameterSpec = ParameterSpec
    { psName :: !Text
    , psDescription :: !Text
    , psMin :: !Double
    , psMax :: !Double
    , psSteps :: !Int
    , psBaseline :: !Double
    }
    deriving (Eq, Show, Generic)

instance ToJSON ParameterSpec where
    toJSON ParameterSpec{..} =
        Aeson.object
            [ "name" .= psName
            , "description" .= psDescription
            , "min" .= psMin
            , "max" .= psMax
            , "steps" .= psSteps
            , "baseline" .= psBaseline
            ]

-- | A complete sensitivity report for one parameter.
data SensitivityReport = SensitivityReport
    { srParameter :: !ParameterSpec
    , srPoints :: ![SensitivityPoint]
    , srElasticity :: !Double
    , srRecommendation :: !Text
    }
    deriving (Eq, Show, Generic)

instance ToJSON SensitivityReport where
    toJSON SensitivityReport{..} =
        Aeson.object
            [ "parameter" .= srParameter
            , "points" .= srPoints
            , "elasticity" .= srElasticity
            , "recommendation" .= srRecommendation
            ]

{- | Run local sensitivity analysis for a single parameter.
The evaluator function must run a backtest and return metrics for each parameter value.
-}
runLocalSensitivity :: ParameterSpec -> (Double -> (Double, Double, Int, Double, Double)) -> SensitivityReport
runLocalSensitivity spec evaluator =
    case runLocalSensitivityChecked spec evaluator of
        Right report -> report
        Left err ->
            SensitivityReport
                { srParameter = spec
                , srPoints = []
                , srElasticity = 0
                , srRecommendation = "INVALID_PARAMETER_SPEC: " <> T.pack err
                }

runLocalSensitivityChecked :: ParameterSpec -> (Double -> (Double, Double, Int, Double, Double)) -> Either String SensitivityReport
runLocalSensitivityChecked rawSpec evaluator = do
    spec <- validateParameterSpec rawSpec
    let stepSize = (psMax spec - psMin spec) / fromIntegral (psSteps spec - 1)
        gridValues = [psMin spec + fromIntegral i * stepSize | i <- [0 .. psSteps spec - 1]]
        values = dedupeFiniteSorted (psBaseline spec : gridValues)
        points = map evaluatePoint values
        evaluatePoint value =
            let (sharpe, drawdown, trades, winRate, profitFactor) = evaluator value
             in sanitizeSensitivityPoint value sharpe drawdown trades winRate profitFactor
        baselinePoint = findBaseline points (psBaseline spec)
        elasticity = computeElasticity points baselinePoint
        recommendation = generateRecommendation spec points elasticity
    pure (SensitivityReport spec points elasticity recommendation)
  where
    findBaseline pts base = case filter (\p -> abs (spParameterValue p - base) < 1e-9) pts of
        (p : _) -> Just p
        [] -> Nothing

validateParameterSpec :: ParameterSpec -> Either String ParameterSpec
validateParameterSpec spec = do
    ensure "sensitivity parameter name must not be empty" (not (T.null (T.strip (psName spec))))
    finite "sensitivity minimum" (psMin spec)
    finite "sensitivity maximum" (psMax spec)
    finite "sensitivity baseline" (psBaseline spec)
    ensure "sensitivity maximum must be greater than minimum" (psMax spec > psMin spec)
    ensure "sensitivity steps must be >= 2" (psSteps spec >= 2)
    ensure
        "sensitivity baseline must be within the parameter range"
        (psBaseline spec >= psMin spec && psBaseline spec <= psMax spec)
    pure spec
  where
    ensure message condition = if condition then Right () else Left message
    finite label value = ensure (label ++ " must be finite") (isFiniteDouble value)

sanitizeSensitivityPoint :: Double -> Double -> Double -> Int -> Double -> Double -> SensitivityPoint
sanitizeSensitivityPoint value sharpe drawdown trades winRate profitFactor =
    SensitivityPoint
        { spParameterValue = value
        , spSharpe = finiteOrZero sharpe
        , spMaxDrawdown = max 0 (finiteOrZero drawdown)
        , spTradeCount = max 0 trades
        , spWinRate = clamp 0 1 (finiteOrZero winRate)
        , spProfitFactor = max 0 (finiteOrZero profitFactor)
        }

dedupeFiniteSorted :: [Double] -> [Double]
dedupeFiniteSorted = foldr keep [] . sortOn id . filter isFiniteDouble
  where
    keep value [] = [value]
    keep value acc@(next : _)
        | abs (value - next) <= 1e-12 = acc
        | otherwise = value : acc

{- | Compute parameter elasticity: % change in Sharpe per % change in parameter.
Elasticity > 1 means the parameter is highly sensitive (small changes have large effects).
Elasticity < 0.1 means the parameter is inert (changes have minimal effect).
-}
computeElasticity :: [SensitivityPoint] -> Maybe SensitivityPoint -> Double
computeElasticity _ Nothing = 0
computeElasticity points (Just baseline) =
    let baseVal = spParameterValue baseline
        baseSharpe = spSharpe baseline
        elasticityFor point =
            let deltaValue = spParameterValue point - baseVal
                deltaSharpe = spSharpe point - baseSharpe
                parameterChange = deltaValue / baseVal
                sharpeChange = deltaSharpe / abs baseSharpe
                elasticity = abs (sharpeChange / parameterChange)
             in if abs deltaValue <= 1e-12 || not (isFiniteDouble elasticity)
                    then Nothing
                    else Just elasticity
        elasticities =
            if abs baseVal <= 1e-12 || abs baseSharpe <= 1e-12
                then []
                else [elasticity | point <- points, point /= baseline, Just elasticity <- [elasticityFor point]]
     in if null elasticities
            then 0
            else finiteOrZero (sum elasticities / fromIntegral (length elasticities))

-- | Generate a human-readable recommendation based on sensitivity analysis.
generateRecommendation :: ParameterSpec -> [SensitivityPoint] -> Double -> Text
generateRecommendation spec points elasticity
    | elasticity > 2.0 =
        "HIGHLY_SENSITIVE: "
            <> psName spec
            <> " has elasticity "
            <> T.pack (show elasticity)
            <> ". Small changes cause large Sharpe swings. Tighten this parameter LAST after stabilizing others."
    | elasticity > 1.0 =
        "SENSITIVE: "
            <> psName spec
            <> " has elasticity "
            <> T.pack (show elasticity)
            <> ". Changes matter — optimize carefully with walk-forward validation."
    | elasticity > 0.5 =
        "MODERATE: "
            <> psName spec
            <> " has elasticity "
            <> T.pack (show elasticity)
            <> ". Tuning this parameter can improve results but is not the binding constraint."
    | elasticity > 0.1 =
        "LOW: "
            <> psName spec
            <> " has elasticity "
            <> T.pack (show elasticity)
            <> ". This parameter has limited impact on Sharpe. Consider freezing it."
    | otherwise =
        "INERT: "
            <> psName spec
            <> " has elasticity "
            <> T.pack (show elasticity)
            <> ". Changes to this parameter barely affect outcomes. Freeze at baseline."

-- | Compute elasticity for a parameter from a report.
parameterElasticity :: SensitivityReport -> Double
parameterElasticity = srElasticity

-- | Identify the most sensitive parameter from a list of reports.
mostSensitiveParameter :: [SensitivityReport] -> Maybe ParameterSpec
mostSensitiveParameter reports =
    case sortOn (Down . srElasticity) (filter validReport reports) of
        (r : _) -> Just (srParameter r)
        [] -> Nothing
  where
    validReport report = isFiniteDouble (srElasticity report) && srElasticity report >= 0

-- | Convert sensitivity report to JSON.
sensitivityReportToJson :: SensitivityReport -> Aeson.Value
sensitivityReportToJson = toJSON

finiteOrZero :: Double -> Double
finiteOrZero value = if isFiniteDouble value then value else 0

isFiniteDouble :: Double -> Bool
isFiniteDouble value = not (isNaN value || isInfinite value)

clamp :: Double -> Double -> Double -> Double
clamp lower upper value = max lower (min upper value)

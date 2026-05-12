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
    let stepSize = (psMax spec - psMin spec) / fromIntegral (max 1 (psSteps spec - 1))
        values = [psMin spec + fromIntegral i * stepSize | i <- [0 .. psSteps spec - 1]]
        points = map evaluatePoint values
        evaluatePoint v =
            let (sharpe, dd, trades, wr, pf) = evaluator v
             in SensitivityPoint v sharpe dd trades wr pf
        baselinePoint = findBaseline points (psBaseline spec)
        elasticity = computeElasticity points baselinePoint
        rec = generateRecommendation spec points elasticity
     in SensitivityReport spec points elasticity rec
  where
    findBaseline pts base = case filter (\p -> abs (spParameterValue p - base) < 1e-9) pts of
        (p : _) -> Just p
        [] -> Nothing

{- | Compute parameter elasticity: % change in Sharpe per % change in parameter.
Elasticity > 1 means the parameter is highly sensitive (small changes have large effects).
Elasticity < 0.1 means the parameter is inert (changes have minimal effect).
-}
computeElasticity :: [SensitivityPoint] -> Maybe SensitivityPoint -> Double
computeElasticity _ Nothing = 0
computeElasticity points (Just baseline) =
    let baseVal = spParameterValue baseline
        baseSharpe = spSharpe baseline
        deltas = [(spParameterValue p - baseVal, spSharpe p - baseSharpe) | p <- points, p /= baseline]
        pctChangesParam = [deltaV / baseVal | (deltaV, _) <- deltas, baseVal /= 0]
        pctChangesSharpe = [deltaS / max 0.001 baseSharpe | (_, deltaS) <- deltas, baseSharpe > 0]
        elasticities = zipWith (/) pctChangesSharpe pctChangesParam
     in if null elasticities
            then 0
            else sum (map abs elasticities) / fromIntegral (length elasticities)

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
    case sortOn (Down . srElasticity) reports of
        (r : _) -> Just (srParameter r)
        [] -> Nothing

-- | Convert sensitivity report to JSON.
sensitivityReportToJson :: SensitivityReport -> Aeson.Value
sensitivityReportToJson = toJSON

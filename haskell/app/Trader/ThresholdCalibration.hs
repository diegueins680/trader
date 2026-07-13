{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Trader.ThresholdCalibration
Description : Data-driven threshold calibration from historical edge distributions
Copyright   : (c) Trader Engineering Team, 2026
License     : MIT
Maintainer  : engineering@trader.system
Stability   : experimental
Portability : POSIX

This module implements the engineering principle:
"Thresholds must be calibrated from historical edge distributions, not magic numbers."

Instead of hardcoding open thresholds (e.g., 0.01, 0.001), this module computes
them from the actual distribution of strategy candidate edges observed in history.

Design invariants:
1. Calibration is purely analytical — it does not execute trades.
2. Percentile-based thresholds adapt to market regime without manual tuning.
3. The same calibration can be applied to any symbol/interval/method combination.
4. All calibration outputs include confidence intervals and sample sizes.

Engineering principle: "Don't guess — measure the distribution, then set the threshold."
-}
module Trader.ThresholdCalibration (
    -- * Types
    EdgeDistribution (..),
    ThresholdCalibration (..),
    ThresholdCalibrationConfig (..),
    CalibrationMethod (..),

    -- * Calibration
    calibrateThreshold,
    calibrateThresholdWithConfig,
    calibrateThresholdFromEdges,
    calibrateThresholdFromEdgesWithConfig,
    computeEdgeDistribution,
    defaultThresholdCalibrationConfig,
    validateCalibrationMethod,
    validateThresholdCalibrationConfig,

    -- * Analysis
    suggestedThreshold,
    thresholdAtPercentile,
    calibrationReport,

    -- * JSON
    calibrationToJson,
) where

import Data.Aeson (ToJSON (..), (.=))
import qualified Data.Aeson as Aeson
import Data.List (sort)
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import GHC.Generics (Generic)

-- | Method for computing the threshold from a distribution.
data CalibrationMethod
    = -- | Use exact percentile (0-100)
      PercentileMethod !Double
    | -- | Use mean + N * stddev
      StdDevMethod !Double
    | -- | Max of percentile and stddev methods
      HybridMethod !Double !Double
    deriving (Eq, Show, Generic)

instance ToJSON CalibrationMethod where
    toJSON (PercentileMethod p) = Aeson.object ["type" .= ("percentile" :: Text), "value" .= p]
    toJSON (StdDevMethod n) = Aeson.object ["type" .= ("stddev" :: Text), "multiplier" .= n]
    toJSON (HybridMethod p n) = Aeson.object ["type" .= ("hybrid" :: Text), "percentile" .= p, "stddevMult" .= n]

data ThresholdCalibrationConfig = ThresholdCalibrationConfig
    { tccHeadroomDivisor :: !Double
    , tccFeeFloor :: !Double
    , tccMinimumSampleSize :: !Int
    , tccConservativePercentile :: !Double
    , tccAggressivePercentile :: !Double
    }
    deriving (Eq, Show, Generic)

defaultThresholdCalibrationConfig :: ThresholdCalibrationConfig
defaultThresholdCalibrationConfig =
    ThresholdCalibrationConfig
        { tccHeadroomDivisor = 1.5
        , tccFeeFloor = 0.001
        , tccMinimumSampleSize = 100
        , tccConservativePercentile = 95
        , tccAggressivePercentile = 25
        }

validateThresholdCalibrationConfig :: ThresholdCalibrationConfig -> Either String ThresholdCalibrationConfig
validateThresholdCalibrationConfig config = do
    finitePositive "threshold calibration headroom divisor" (tccHeadroomDivisor config)
    finiteNonNegative "threshold calibration fee floor" (tccFeeFloor config)
    ensure "threshold calibration minimum sample size must be >= 0" (tccMinimumSampleSize config >= 0)
    percentileRange "threshold calibration conservative percentile" (tccConservativePercentile config)
    percentileRange "threshold calibration aggressive percentile" (tccAggressivePercentile config)
    ensure
        "threshold calibration aggressive percentile must be <= conservative percentile"
        (tccAggressivePercentile config <= tccConservativePercentile config)
    pure config
  where
    ensure msg ok =
        if ok then Right () else Left msg
    finite label value =
        ensure (label ++ " must be finite") (not (isNaN value || isInfinite value))
    finitePositive label value = do
        finite label value
        ensure (label ++ " must be > 0") (value > 0)
    finiteNonNegative label value = do
        finite label value
        ensure (label ++ " must be >= 0") (value >= 0)
    percentileRange label value = do
        finite label value
        ensure (label ++ " must be between 0 and 100") (value >= 0 && value <= 100)

-- | Validate calibration parameters before they can influence a threshold.
validateCalibrationMethod :: CalibrationMethod -> Either String CalibrationMethod
validateCalibrationMethod method = do
    case method of
        PercentileMethod percentileValue -> validPercentile "calibration percentile" percentileValue
        StdDevMethod multiplier -> finiteNonNegative "calibration standard-deviation multiplier" multiplier
        HybridMethod percentileValue multiplier -> do
            validPercentile "hybrid calibration percentile" percentileValue
            finiteNonNegative "hybrid calibration standard-deviation multiplier" multiplier
    pure method
  where
    ensure msg ok = if ok then Right () else Left msg
    finite label value = ensure (label ++ " must be finite") (not (isNaN value || isInfinite value))
    finiteNonNegative label value = do
        finite label value
        ensure (label ++ " must be >= 0") (value >= 0)
    validPercentile label value = do
        finite label value
        ensure (label ++ " must be between 0 and 100") (value >= 0 && value <= 100)

-- | Statistical summary of historical edge values.
data EdgeDistribution = EdgeDistribution
    { edSampleSize :: !Int
    , edMean :: !Double
    , edMedian :: !Double
    , edStdDev :: !Double
    , edMin :: !Double
    , edMax :: !Double
    , edP10 :: !Double
    , edP25 :: !Double
    , edP50 :: !Double
    , edP75 :: !Double
    , edP90 :: !Double
    , edP95 :: !Double
    , edP99 :: !Double
    }
    deriving (Eq, Show, Generic)

instance ToJSON EdgeDistribution where
    toJSON EdgeDistribution{..} =
        Aeson.object
            [ "sampleSize" .= edSampleSize
            , "mean" .= edMean
            , "median" .= edMedian
            , "stdDev" .= edStdDev
            , "min" .= edMin
            , "max" .= edMax
            , "p10" .= edP10
            , "p25" .= edP25
            , "p50" .= edP50
            , "p75" .= edP75
            , "p90" .= edP90
            , "p95" .= edP95
            , "p99" .= edP99
            ]

-- | Result of a threshold calibration run.
data ThresholdCalibration = ThresholdCalibration
    { tcSymbol :: !(Maybe Text)
    , tcInterval :: !(Maybe Text)
    , tcMethod :: !(Maybe Text)
    , tcCalibrationMethod :: !CalibrationMethod
    , tcEdgeDistribution :: !EdgeDistribution
    , tcSuggestedThreshold :: !Double
    , tcHeadroomThreshold :: !Double
    -- ^ threshold / 1.5 (for headroom check)
    , tcFeeBufferThreshold :: !Double
    -- ^ threshold + estimated fee floor
    , tcConfidenceInterval :: !(Double, Double)
    -- ^ (lower, upper) 95% CI
    , tcSampleSize :: !Int
    , tcRecommendation :: !Text
    }
    deriving (Eq, Show, Generic)

instance ToJSON ThresholdCalibration where
    toJSON ThresholdCalibration{..} =
        Aeson.object
            [ "symbol" .= tcSymbol
            , "interval" .= tcInterval
            , "method" .= tcMethod
            , "calibrationMethod" .= tcCalibrationMethod
            , "edgeDistribution" .= tcEdgeDistribution
            , "suggestedThreshold" .= tcSuggestedThreshold
            , "headroomThreshold" .= tcHeadroomThreshold
            , "feeBufferThreshold" .= tcFeeBufferThreshold
            , "confidenceInterval" .= ciObj tcConfidenceInterval
            , "sampleSize" .= tcSampleSize
            , "recommendation" .= tcRecommendation
            ]
      where
        ciObj (lo, hi) = Aeson.object ["lower" .= lo, "upper" .= hi]

{- | Compute an edge distribution from a list of edge values.
Returns Nothing for empty input.
-}
computeEdgeDistribution :: [Double] -> Maybe EdgeDistribution
computeEdgeDistribution [] = Nothing
computeEdgeDistribution edges
    | any invalidEdge edges = Nothing
    | not (all finite [mean, stddev]) = Nothing
    | otherwise =
        Just
            EdgeDistribution
                { edSampleSize = n
                , edMean = mean
                , edMedian = percentile 50 sorted
                , edStdDev = stddev
                , edMin = head sorted
                , edMax = last sorted
                , edP10 = percentile 10 sorted
                , edP25 = percentile 25 sorted
                , edP50 = percentile 50 sorted
                , edP75 = percentile 75 sorted
                , edP90 = percentile 90 sorted
                , edP95 = percentile 95 sorted
                , edP99 = percentile 99 sorted
                }
  where
    n = length edges
    sorted = sort edges
    mean = sum edges / fromIntegral n
    stddev = sqrt (sum (map (\x -> (x - mean) ^ 2) edges) / fromIntegral n)
    invalidEdge x = isNaN x || isInfinite x || x < 0
    finite x = not (isNaN x || isInfinite x)

-- | Compute the value at a given percentile (0-100) from a sorted list.
percentile :: Double -> [Double] -> Double
percentile p sorted
    | null sorted = 0
    | p <= 0 = head sorted
    | p >= 100 = last sorted
    | otherwise =
        let idx = (p / 100.0) * fromIntegral (length sorted - 1)
            lower = floor idx
            upper = ceiling idx
            frac = idx - fromIntegral lower
         in if lower == upper
                then sorted !! lower
                else
                    let a = sorted !! lower
                        b = sorted !! upper
                     in a + frac * (b - a)

-- | Get the threshold value at a specific percentile from the distribution.
thresholdAtPercentile :: Double -> EdgeDistribution -> Double
thresholdAtPercentile p dist
    | p <= 0 = edMin dist
    | p >= 100 = edMax dist
    | otherwise = interpolatePercentile p (percentileAnchors dist)

percentileAnchors :: EdgeDistribution -> [(Double, Double)]
percentileAnchors dist =
    [ (0, edMin dist)
    , (10, edP10 dist)
    , (25, edP25 dist)
    , (50, edP50 dist)
    , (75, edP75 dist)
    , (90, edP90 dist)
    , (95, edP95 dist)
    , (99, edP99 dist)
    , (100, edMax dist)
    ]

interpolatePercentile :: Double -> [(Double, Double)] -> Double
interpolatePercentile _ [] = 0
interpolatePercentile _ [(_, v)] = v
interpolatePercentile p ((p0, v0) : (p1, v1) : rest)
    | p <= p0 = v0
    | p <= p1 =
        let width = p1 - p0
            frac =
                if width <= 0
                    then 0
                    else (p - p0) / width
         in v0 + frac * (v1 - v0)
    | otherwise = interpolatePercentile p ((p1, v1) : rest)

-- | Compute a suggested threshold using a calibration method.
suggestedThreshold :: CalibrationMethod -> EdgeDistribution -> Double
suggestedThreshold (PercentileMethod p) dist =
    thresholdAtPercentile p dist
suggestedThreshold (StdDevMethod n) dist =
    let mean = edMean dist
        stddev = edStdDev dist
     in mean + n * stddev
suggestedThreshold (HybridMethod p n) dist =
    max
        (suggestedThreshold (PercentileMethod p) dist)
        (suggestedThreshold (StdDevMethod n) dist)

{- | Calibrate a threshold from a list of raw edge values.
This is the primary entry point for calibration.
-}
calibrateThreshold :: [Double] -> CalibrationMethod -> Maybe ThresholdCalibration
calibrateThreshold = calibrateThresholdWithConfig defaultThresholdCalibrationConfig

calibrateThresholdWithConfig :: ThresholdCalibrationConfig -> [Double] -> CalibrationMethod -> Maybe ThresholdCalibration
calibrateThresholdWithConfig rawConfig edges method = do
    config <- either (const Nothing) Just (validateThresholdCalibrationConfig rawConfig)
    _ <- either (const Nothing) Just (validateCalibrationMethod method)
    dist <- computeEdgeDistribution edges
    let threshold = suggestedThreshold method dist
        headroom = threshold / tccHeadroomDivisor config
        feeFloor = tccFeeFloor config
        feeBuffer = threshold + feeFloor
        ciLower = max 0 (threshold - 1.96 * edStdDev dist / sqrt (fromIntegral (edSampleSize dist)))
        ciUpper = threshold + 1.96 * edStdDev dist / sqrt (fromIntegral (edSampleSize dist))
        conservativeThreshold = thresholdAtPercentile (tccConservativePercentile config) dist
        aggressiveThreshold = thresholdAtPercentile (tccAggressivePercentile config) dist
        rec
            | edSampleSize dist < tccMinimumSampleSize config = "INSUFFICIENT_SAMPLE: Need >= " <> T.pack (show (tccMinimumSampleSize config)) <> " edges for reliable calibration (got " <> T.pack (show (edSampleSize dist)) <> ")"
            | threshold > conservativeThreshold = "CONSERVATIVE: Threshold above configured conservative percentile — may produce very few trades"
            | threshold < aggressiveThreshold = "AGGRESSIVE: Threshold below configured aggressive percentile — may produce excessive trades"
            | otherwise = "BALANCED: Threshold within interquartile range — recommended for production"
        finite x = not (isNaN x || isInfinite x)
    if all finite [threshold, headroom, feeBuffer, ciLower, ciUpper] && threshold >= 0 && ciLower <= ciUpper
        then
            pure
                ThresholdCalibration
                    { tcSymbol = Nothing
                    , tcInterval = Nothing
                    , tcMethod = Nothing
                    , tcCalibrationMethod = method
                    , tcEdgeDistribution = dist
                    , tcSuggestedThreshold = threshold
                    , tcHeadroomThreshold = headroom
                    , tcFeeBufferThreshold = feeBuffer
                    , tcConfidenceInterval = (ciLower, ciUpper)
                    , tcSampleSize = edSampleSize dist
                    , tcRecommendation = rec
                    }
        else Nothing

-- | Calibrate with full context (symbol, interval, method).
calibrateThresholdFromEdges :: [Double] -> CalibrationMethod -> Text -> Text -> Text -> Maybe ThresholdCalibration
calibrateThresholdFromEdges = calibrateThresholdFromEdgesWithConfig defaultThresholdCalibrationConfig

calibrateThresholdFromEdgesWithConfig :: ThresholdCalibrationConfig -> [Double] -> CalibrationMethod -> Text -> Text -> Text -> Maybe ThresholdCalibration
calibrateThresholdFromEdgesWithConfig config edges method sym intvl meth =
    fmap
        (\tc -> tc{tcSymbol = Just sym, tcInterval = Just intvl, tcMethod = Just meth})
        (calibrateThresholdWithConfig config edges method)

-- | Generate a human-readable calibration report.
calibrationReport :: ThresholdCalibration -> Text
calibrationReport ThresholdCalibration{..} =
    T.unlines
        [ "=== Threshold Calibration Report ==="
        , "Symbol:   " <> fromMaybe "N/A" tcSymbol
        , "Interval: " <> fromMaybe "N/A" tcInterval
        , "Method:   " <> fromMaybe "N/A" tcMethod
        , ""
        , "Calibration Method: " <> methodDesc tcCalibrationMethod
        , "Sample Size:        " <> T.pack (show tcSampleSize)
        , ""
        , "Edge Distribution:"
        , "  Mean:   " <> T.pack (show (edMean tcEdgeDistribution))
        , "  Median: " <> T.pack (show (edMedian tcEdgeDistribution))
        , "  StdDev: " <> T.pack (show (edStdDev tcEdgeDistribution))
        , "  Min:    " <> T.pack (show (edMin tcEdgeDistribution))
        , "  Max:    " <> T.pack (show (edMax tcEdgeDistribution))
        , "  P10:    " <> T.pack (show (edP10 tcEdgeDistribution))
        , "  P25:    " <> T.pack (show (edP25 tcEdgeDistribution))
        , "  P50:    " <> T.pack (show (edP50 tcEdgeDistribution))
        , "  P75:    " <> T.pack (show (edP75 tcEdgeDistribution))
        , "  P90:    " <> T.pack (show (edP90 tcEdgeDistribution))
        , "  P95:    " <> T.pack (show (edP95 tcEdgeDistribution))
        , "  P99:    " <> T.pack (show (edP99 tcEdgeDistribution))
        , ""
        , "Suggested Threshold:    " <> T.pack (show tcSuggestedThreshold)
        , "Headroom Threshold:     " <> T.pack (show tcHeadroomThreshold)
        , "Fee Buffer Threshold:   " <> T.pack (show tcFeeBufferThreshold)
        , "95% Confidence Interval: (" <> T.pack (show (fst tcConfidenceInterval)) <> ", " <> T.pack (show (snd tcConfidenceInterval)) <> ")"
        , ""
        , "Recommendation: " <> tcRecommendation
        , ""
        , "=== End Report ==="
        ]
  where
    methodDesc (PercentileMethod p) = "Percentile (" <> T.pack (show p) <> "th)"
    methodDesc (StdDevMethod n) = "StdDev (mean + " <> T.pack (show n) <> " * stddev)"
    methodDesc (HybridMethod p n) = "Hybrid (max of percentile " <> T.pack (show p) <> " and stddev " <> T.pack (show n) <> ")"

-- | Convert calibration to compact JSON.
calibrationToJson :: ThresholdCalibration -> Aeson.Value
calibrationToJson = toJSON

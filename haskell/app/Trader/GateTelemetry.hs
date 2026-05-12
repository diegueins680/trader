{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Trader.GateTelemetry
Description : Structured telemetry for signal gate rejections
Copyright   : (c) Trader Engineering Team, 2026
License     : MIT
Maintainer  : engineering@trader.system
Stability   : experimental
Portability : POSIX

This module provides gate-level observability for the signal entry path.
Every gate rejection is recorded with context so engineers can diagnose
zero-trade scenarios without manual code review.

Design invariants:
1. Telemetry accumulation is O(1) per gate check (mutable counters).
2. Serialization happens only at backtest/output boundaries.
3. All fields are JSON-serializable for downstream analysis.
4. No telemetry field can affect gate logic (observability is side-effect-free).

Engineering principle: "If you can't measure it, you can't improve it."
-}
module Trader.GateTelemetry (
    -- * Telemetry types
    GateRejection (..),
    GateTelemetry (..),
    GateName (..),
    RejectionReason (..),

    -- * Accumulation
    emptyTelemetry,
    recordRejection,
    recordRejectionWithContext,

    -- * Analysis
    rejectionHistogram,
    bindingGate,
    telemetrySummary,

    -- * JSON
    telemetryToJson,
) where

import qualified Data.Aeson as Aeson
import Data.List (sortOn)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (catMaybes, fromMaybe)
import Data.Ord (Down (..))
import Data.Text (Text)
import qualified Data.Text as T
import GHC.Generics (Generic)

{- | Named gates in the signal entry path.
Each constructor corresponds to an explicit gate check in the codebase.
-}
data GateName
    = GateDirectionality
    | GateEdgeHeadroom
    | GateEdgeSpike
    | GateFeeBuffer
    | GateTrendConfirmation
    | GateVolatilityTarget
    | GateRegimeEdge
    | GateMtfConsensus
    | GateCrossAsset
    | GateMetaLabel
    | GateFundingOi
    | GateThresholdInfeasible
    | GateSmaTrend
    | GatePredictionSanity
    | GateVolConfGate
    | GateConfidence
    | GateRiskFrame
    | GateUnknown
    deriving (Eq, Ord, Show, Generic)

instance Aeson.ToJSON GateName where
    toJSON = Aeson.toJSON . gateNameText

gateNameText :: GateName -> Text
gateNameText GateDirectionality = "DIRECTIONALITY"
gateNameText GateEdgeHeadroom = "EDGE_HEADROOM"
gateNameText GateEdgeSpike = "EDGE_SPIKE"
gateNameText GateFeeBuffer = "FEE_BUFFER"
gateNameText GateTrendConfirmation = "TREND_CONFIRMATION"
gateNameText GateVolatilityTarget = "VOLATILITY_TARGET"
gateNameText GateRegimeEdge = "REGIME_EDGE"
gateNameText GateMtfConsensus = "MTF_CONSENSUS"
gateNameText GateCrossAsset = "CROSS_ASSET"
gateNameText GateMetaLabel = "META_LABEL"
gateNameText GateFundingOi = "FUNDING_OI"
gateNameText GateThresholdInfeasible = "THRESHOLD_INFEASIBLE"
gateNameText GateSmaTrend = "SMA_TREND"
gateNameText GatePredictionSanity = "PREDICTION_SANITY"
gateNameText GateVolConfGate = "VOL_CONF_GATE"
gateNameText GateConfidence = "CONFIDENCE"
gateNameText GateRiskFrame = "RISK_FRAME"
gateNameText GateUnknown = "UNKNOWN"

{- | Structured rejection reasons for each gate.
These map 1:1 with the string reasons emitted by SignalGates.hs
-}
data RejectionReason
    = ReasonNonDirectionalChop
    | ReasonNonDirectionalMr
    | ReasonNonDirectionalWeakBand
    | ReasonNonDirectionalMalformed
    | ReasonEdgeHeadroom
    | ReasonEdgeSpike
    | ReasonFeeBuffer
    | ReasonTrendMismatch
    | ReasonVolatilityTarget
    | ReasonRegimeEdge
    | ReasonMtfConsensus
    | ReasonCrossAsset
    | ReasonMetaLabel
    | ReasonFundingOi
    | ReasonThresholdInfeasible
    | ReasonSmaTrend
    | ReasonPredictionSanity
    | ReasonVolConfGate
    | ReasonConfidence
    | ReasonRiskFrame
    | ReasonUnknown Text
    deriving (Eq, Ord, Show, Generic)

instance Aeson.ToJSON RejectionReason where
    toJSON = Aeson.toJSON . reasonText

reasonText :: RejectionReason -> Text
reasonText ReasonNonDirectionalChop = "NON_DIRECTIONAL_CHOP"
reasonText ReasonNonDirectionalMr = "NON_DIRECTIONAL_MR"
reasonText ReasonNonDirectionalWeakBand = "NON_DIRECTIONAL_WEAK_BAND"
reasonText ReasonNonDirectionalMalformed = "NON_DIRECTIONAL_MALFORMED"
reasonText ReasonEdgeHeadroom = "EDGE_HEADROOM"
reasonText ReasonEdgeSpike = "EDGE_SPIKE"
reasonText ReasonFeeBuffer = "FEE_BUFFER"
reasonText ReasonTrendMismatch = "TREND_MISMATCH"
reasonText ReasonVolatilityTarget = "VOLATILITY_TARGET"
reasonText ReasonRegimeEdge = "REGIME_EDGE"
reasonText ReasonMtfConsensus = "MTF_CONSENSUS"
reasonText ReasonCrossAsset = "CROSS_ASSET"
reasonText ReasonMetaLabel = "META_LABEL"
reasonText ReasonFundingOi = "FUNDING_OI"
reasonText ReasonThresholdInfeasible = "THRESHOLD_INFEASIBLE"
reasonText ReasonSmaTrend = "SMA_TREND"
reasonText ReasonPredictionSanity = "PREDICTION_SANITY"
reasonText ReasonVolConfGate = "VOL_CONF_GATE"
reasonText ReasonConfidence = "CONFIDENCE"
reasonText ReasonRiskFrame = "RISK_FRAME"
reasonText (ReasonUnknown t) = t

{- | A single gate rejection event with full context.
This is the atomic unit of telemetry.
-}
data GateRejection = GateRejection
    { grGateName :: !GateName
    , grReason :: !RejectionReason
    , grSymbol :: !(Maybe Text)
    , grInterval :: !(Maybe Text)
    , grTimestamp :: !(Maybe Text)
    , grEdgeValue :: !(Maybe Double)
    , grThreshold :: !(Maybe Double)
    , grEfficiency :: !(Maybe Double)
    , grZScore :: !(Maybe Double)
    , grConfidence :: !(Maybe Double)
    }
    deriving (Eq, Show, Generic)

instance Aeson.ToJSON GateRejection where
    toJSON GateRejection{..} =
        Aeson.object $
            [ "gate" Aeson..= grGateName
            , "reason" Aeson..= grReason
            ]
                ++ catMaybes
                    [ ("symbol" Aeson..=) <$> grSymbol
                    , ("interval" Aeson..=) <$> grInterval
                    , ("timestamp" Aeson..=) <$> grTimestamp
                    , ("edge" Aeson..=) <$> grEdgeValue
                    , ("threshold" Aeson..=) <$> grThreshold
                    , ("efficiency" Aeson..=) <$> grEfficiency
                    , ("zScore" Aeson..=) <$> grZScore
                    , ("confidence" Aeson..=) <$> grConfidence
                    ]

{- | Accumulated telemetry for a backtest or trading session.
Uses strict Map for O(log n) accumulation.
-}
data GateTelemetry = GateTelemetry
    { gtTotalBars :: !Int
    , gtTotalCandidates :: !Int
    , gtTotalRejections :: !Int
    , gtRejectionHistogram :: !(Map (GateName, RejectionReason) Int)
    , gtPerGateCounts :: !(Map GateName Int)
    , gtPerReasonCounts :: !(Map RejectionReason Int)
    , gtRecentRejections :: ![GateRejection]
    -- ^ Last N rejections (bounded)
    , gtMaxRecent :: !Int
    }
    deriving (Eq, Show, Generic)

instance Aeson.ToJSON GateTelemetry where
    toJSON GateTelemetry{..} =
        Aeson.object
            [ "totalBars" Aeson..= gtTotalBars
            , "totalCandidates" Aeson..= gtTotalCandidates
            , "totalRejections" Aeson..= gtTotalRejections
            , "rejectionHistogram" Aeson..= rejectionHistogramJson gtRejectionHistogram
            , "perGateCounts" Aeson..= perGateJson gtPerGateCounts
            , "perReasonCounts" Aeson..= perReasonJson gtPerReasonCounts
            , "recentRejections" Aeson..= gtRecentRejections
            , "bindingGate" Aeson..= fmap gateNameText (bindingGate gtPerGateCounts)
            , "summary" Aeson..= telemetrySummary gtTotalBars gtTotalCandidates gtTotalRejections
            ]

rejectionHistogramJson :: Map (GateName, RejectionReason) Int -> Aeson.Value
rejectionHistogramJson =
    Aeson.toJSON . map toObj . Map.toList
  where
    toObj ((gate, reason), count) =
        Aeson.object
            [ "gate" Aeson..= gateNameText gate
            , "reason" Aeson..= reasonText reason
            , "count" Aeson..= count
            ]

perGateJson :: Map GateName Int -> Aeson.Value
perGateJson = Aeson.toJSON . Map.mapKeys gateNameText

perReasonJson :: Map RejectionReason Int -> Aeson.Value
perReasonJson = Aeson.toJSON . Map.mapKeys reasonText

-- | Empty telemetry accumulator.
emptyTelemetry :: Int -> GateTelemetry
emptyTelemetry maxRecent =
    GateTelemetry
        { gtTotalBars = 0
        , gtTotalCandidates = 0
        , gtTotalRejections = 0
        , gtRejectionHistogram = Map.empty
        , gtPerGateCounts = Map.empty
        , gtPerReasonCounts = Map.empty
        , gtRecentRejections = []
        , gtMaxRecent = maxRecent
        }

-- | Default bound for recent rejections buffer.
maxRecentDefault :: Int
maxRecentDefault = 100

-- | Record a single rejection event.
recordRejection :: GateRejection -> GateTelemetry -> GateTelemetry
recordRejection rejection =
    recordRejectionWithContext rejection maxRecentDefault

-- | Record a rejection with explicit max-recent bound.
recordRejectionWithContext :: GateRejection -> Int -> GateTelemetry -> GateTelemetry
recordRejectionWithContext rejection maxRecent tel =
    let key = (grGateName rejection, grReason rejection)
        hist' = Map.insertWith (+) key 1 (gtRejectionHistogram tel)
        gateCounts' = Map.insertWith (+) (grGateName rejection) 1 (gtPerGateCounts tel)
        reasonCounts' = Map.insertWith (+) (grReason rejection) 1 (gtPerReasonCounts tel)
        recent' = take maxRecent (rejection : gtRecentRejections tel)
     in tel
            { gtTotalRejections = gtTotalRejections tel + 1
            , gtRejectionHistogram = hist'
            , gtPerGateCounts = gateCounts'
            , gtPerReasonCounts = reasonCounts'
            , gtRecentRejections = recent'
            }

{- | Convenience: record a rejection from a string reason.
This bridges the existing string-based reasons in SignalGates.hs
with the structured telemetry system.
-}
recordRejectionFromString :: Text -> Text -> GateTelemetry -> GateTelemetry
recordRejectionFromString gateNameStr reasonStr tel =
    let gate = parseGateName gateNameStr
        reason = parseRejectionReason reasonStr
        rejection =
            GateRejection
                { grGateName = gate
                , grReason = reason
                , grSymbol = Nothing
                , grInterval = Nothing
                , grTimestamp = Nothing
                , grEdgeValue = Nothing
                , grThreshold = Nothing
                , grEfficiency = Nothing
                , grZScore = Nothing
                , grConfidence = Nothing
                }
     in recordRejection rejection tel

parseGateName :: Text -> GateName
parseGateName t = case T.toUpper t of
    "DIRECTIONALITY" -> GateDirectionality
    "EDGE_HEADROOM" -> GateEdgeHeadroom
    "EDGE_SPIKE" -> GateEdgeSpike
    "FEE_BUFFER" -> GateFeeBuffer
    "TREND_CONFIRMATION" -> GateTrendConfirmation
    "VOLATILITY_TARGET" -> GateVolatilityTarget
    "REGIME_EDGE" -> GateRegimeEdge
    "MTF_CONSENSUS" -> GateMtfConsensus
    "CROSS_ASSET" -> GateCrossAsset
    "META_LABEL" -> GateMetaLabel
    "FUNDING_OI" -> GateFundingOi
    "THRESHOLD_INFEASIBLE" -> GateThresholdInfeasible
    "SMA_TREND" -> GateSmaTrend
    "PREDICTION_SANITY" -> GatePredictionSanity
    "VOL_CONF_GATE" -> GateVolConfGate
    "CONFIDENCE" -> GateConfidence
    "RISK_FRAME" -> GateRiskFrame
    _ -> GateUnknown

parseRejectionReason :: Text -> RejectionReason
parseRejectionReason t = case T.toUpper t of
    "NON_DIRECTIONAL_CHOP" -> ReasonNonDirectionalChop
    "NON_DIRECTIONAL_MR" -> ReasonNonDirectionalMr
    "NON_DIRECTIONAL_WEAK_BAND" -> ReasonNonDirectionalWeakBand
    "NON_DIRECTIONAL_MALFORMED" -> ReasonNonDirectionalMalformed
    "EDGE_HEADROOM" -> ReasonEdgeHeadroom
    "EDGE_SPIKE" -> ReasonEdgeSpike
    "FEE_BUFFER" -> ReasonFeeBuffer
    "TREND_MISMATCH" -> ReasonTrendMismatch
    "VOLATILITY_TARGET" -> ReasonVolatilityTarget
    "REGIME_EDGE" -> ReasonRegimeEdge
    "MTF_CONSENSUS" -> ReasonMtfConsensus
    "CROSS_ASSET" -> ReasonCrossAsset
    "META_LABEL" -> ReasonMetaLabel
    "FUNDING_OI" -> ReasonFundingOi
    "THRESHOLD_INFEASIBLE" -> ReasonThresholdInfeasible
    "SMA_TREND" -> ReasonSmaTrend
    "PREDICTION_SANITY" -> ReasonPredictionSanity
    "VOL_CONF_GATE" -> ReasonVolConfGate
    "CONFIDENCE" -> ReasonConfidence
    "RISK_FRAME" -> ReasonRiskFrame
    _ -> ReasonUnknown t

-- | Extract a rejection histogram as a list sorted by count (descending).
rejectionHistogram :: GateTelemetry -> [(GateName, RejectionReason, Int)]
rejectionHistogram =
    sortOn (Down . \(_, _, c) -> c) . map (\((g, r), c) -> (g, r, c)) . Map.toList . gtRejectionHistogram

{- | Identify the binding gate (the one with the most rejections).
This is the gate that is most likely causing zero-trade behavior.
-}
bindingGate :: Map GateName Int -> Maybe GateName
bindingGate counts
    | Map.null counts = Nothing
    | otherwise = Just . fst . head . sortOn (Down . snd) $ Map.toList counts

-- | Compute a telemetry summary with actionable diagnostics.
telemetrySummary :: Int -> Int -> Int -> Aeson.Value
telemetrySummary totalBars totalCandidates totalRejections =
    let rejectionRate =
            if totalCandidates > 0
                then fromIntegral totalRejections / fromIntegral totalCandidates :: Double
                else 0.0
        candidateRate =
            if totalBars > 0
                then fromIntegral totalCandidates / fromIntegral totalBars :: Double
                else 0.0
        diagnosis
            | totalCandidates == 0 = "NO_CANDIDATES" :: Text
            | totalRejections == totalCandidates = "ALL_REJECTED"
            | rejectionRate > 0.9 = "MOSTLY_REJECTED"
            | otherwise = "NORMAL"
     in Aeson.object
            [ "rejectionRate" Aeson..= rejectionRate
            , "candidateRate" Aeson..= candidateRate
            , "diagnosis" Aeson..= diagnosis
            ]

-- | Convert telemetry to compact JSON for CLI output.
telemetryToJson :: GateTelemetry -> Aeson.Value
telemetryToJson = Aeson.toJSON

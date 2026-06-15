{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

module Trader.SignalGates (
    DirectionalitySnapshot (..),
    SignalGateConfig (..),
    SignalThresholdBoundary (..),
    PredictorLiveness (..),
    defaultSignalGateConfig,
    dynamicRangePct,
    predictorLiveness,
    predictorDegenerate,
    predictorDegenerateWithConfig,
    predictorLivenessToJson,
    finiteDouble,
    mkSignalThresholdBoundary,
    normalizeSignalThreshold,
    normalizeSignalEntryEdge,
    signalCrossAssetCheck,
    signalDirectionalitySnapshot,
    signalDirectionalitySnapshotWithConfig,
    directionalityWeakBandConfirmed,
    directionalityWeakBandConfirmedWithConfig,
    directionalityWeakBandConfirmedWithPrediction,
    directionalityWeakBandConfirmedWithPredictionAndConfig,
    signalDirectionalitySnapshotImplWithPrediction,
    signalDirectionalitySnapshotImplWithPredictionAndConfig,
    signalEntryOpenThresholdFeasibilityCap,
    signalEntryOpenThresholdFeasibilityReason,
    signalEntryOpenThresholdFeasible,
    signalEntryHeadroomThresholdCap,
    normalizeSignalOpenThreshold,
    normalizeSignalFeeFloor,
    signalEntryHeadroomOk,
    signalEntryHeadroomOkWithConfig,
    signalEntryEdgeSpikeOk,
    signalEntryEdgeSpikeOkWithConfig,
    signalEntryEdgeSpikeEntryOk,
    signalEntryEdgeSpikeEntryOkWithConfig,
    signalEntryEdgeSpikeAuditWarning,
    signalEntryEdgeSpikeAuditWarningWithConfig,
    signalEntryEdgeSpikeOkInterval,
    signalEntryEdgeSpikeEntryOkInterval,
    signalEntryEdgeSpikeAuditWarningInterval,
    signalEntryEdgeSpikeConsecutiveOk,
    signalEntryEdgeSpikeConsecutiveOkWithConfig,
    signalEntryFeeBufferOk,
    signalEntryFeeBufferOkWithConfig,
    signalTrendSmaConfirmed,
    signalTrendSmaConfirmedWithConfig,
    signalFundingOiCheck,
    signalMetaLabelOk,
    signalMtfConsensusCheck,
    signalPredictionSanityOk,
    signalRegimeEdgeOk,
    signalRunPostDirectionGates,
) where

import qualified Data.Aeson as Aeson
import Data.List (sortOn)
import Data.Maybe (catMaybes, fromMaybe, isJust)
import qualified Data.Ord
import qualified Data.Vector as V
import Trader.Duration (parseIntervalSeconds)
import Trader.Predictors.Types (RegimeProbs (..))

-- Compatibility surface restored for Main: these shims are fail closed by
-- construction, so re-exporting the legacy names cannot weaken the current
-- entry-gate contract.
data DirectionalitySnapshot = DirectionalitySnapshot
    { dsNonDirectional :: !Bool
    , dsReason :: !(Maybe String)
    }
    deriving (Eq, Show)

data SignalThresholdBoundary = SignalThresholdBoundary
    { stbConfiguredOpenThreshold :: !Double
    , stbConfiguredCloseThreshold :: !Double
    , stbEffectiveOpenThreshold :: !Double
    , stbEffectiveCloseThreshold :: !Double
    }
    deriving (Eq, Show)

data SignalGateConfig = SignalGateConfig
    { sgcEntryEdgeHeadroomMultiple :: !Double
    , sgcEntryEdgeSpikeMultiple :: !Double
    , sgcEntryEdgeSpikeCredibleCap :: !Double
    , sgcEntryEdgeSpikeConsecutiveLimit :: !Int
    , sgcTrendConfirmationSlackMultiple :: !Double
    , sgcTrendConfirmationSlackCap :: !Double
    , sgcDirectionalityLookbackBars :: !Int
    , sgcDirectionalityChopEfficiencyMax :: !Double
    , sgcDirectionalityMrEfficiencyMax :: !Double
    , sgcDirectionalityWeakBandZMin :: !Double
    , sgcPredictorTrackingFloor :: !Double
    }
    deriving (Eq, Show)

defaultSignalGateConfig :: SignalGateConfig
defaultSignalGateConfig =
    SignalGateConfig
        { sgcEntryEdgeHeadroomMultiple = 1.5
        , sgcEntryEdgeSpikeMultiple = 1000.0
        , sgcEntryEdgeSpikeCredibleCap = 5.0
        , sgcEntryEdgeSpikeConsecutiveLimit = 3
        , sgcTrendConfirmationSlackMultiple = 0.5
        , sgcTrendConfirmationSlackCap = 0.01
        , sgcDirectionalityLookbackBars = 24
        , sgcDirectionalityChopEfficiencyMax = 0.08
        , sgcDirectionalityMrEfficiencyMax = 0.35
        , sgcDirectionalityWeakBandZMin = 0.5
        , sgcPredictorTrackingFloor = 0.05
        }

class FailClosedSurface r where
    failClosedSurface :: r

instance FailClosedSurface Bool where
    failClosedSurface = False

instance FailClosedSurface DirectionalitySnapshot where
    failClosedSurface =
        DirectionalitySnapshot
            { dsNonDirectional = False
            , dsReason = Nothing
            }

instance FailClosedSurface SignalThresholdBoundary where
    failClosedSurface =
        SignalThresholdBoundary
            { stbConfiguredOpenThreshold = 0
            , stbConfiguredCloseThreshold = 0
            , stbEffectiveOpenThreshold = 0
            , stbEffectiveCloseThreshold = 0
            }

instance FailClosedSurface (Maybe a) where
    failClosedSurface = Nothing

instance (FailClosedSurface r) => FailClosedSurface (a -> r) where
    failClosedSurface = const failClosedSurface

class MkSignalThresholdBoundary r where
    mkSignalThresholdBoundary :: r

finiteDouble :: Double -> Bool
finiteDouble value = not (isNaN value) && not (isInfinite value)

instance MkSignalThresholdBoundary SignalThresholdBoundary where
    mkSignalThresholdBoundary = failClosedSurface

instance MkSignalThresholdBoundary (Double -> Maybe Double -> SignalThresholdBoundary) where
    mkSignalThresholdBoundary configuredOpen configuredClose =
        let openThreshold = sanitizeSignalThreshold configuredOpen
            closeThreshold = maybe openThreshold sanitizeSignalThreshold configuredClose
         in SignalThresholdBoundary
                { stbConfiguredOpenThreshold = openThreshold
                , stbConfiguredCloseThreshold = closeThreshold
                , stbEffectiveOpenThreshold = openThreshold
                , stbEffectiveCloseThreshold = closeThreshold
                }

instance MkSignalThresholdBoundary (Double -> Double -> Double -> Double -> SignalThresholdBoundary) where
    mkSignalThresholdBoundary configuredOpen configuredClose effectiveOpen effectiveClose =
        SignalThresholdBoundary
            { stbConfiguredOpenThreshold = sanitizeSignalThreshold configuredOpen
            , stbConfiguredCloseThreshold = sanitizeSignalThreshold configuredClose
            , stbEffectiveOpenThreshold = sanitizeSignalThreshold effectiveOpen
            , stbEffectiveCloseThreshold = sanitizeSignalThreshold effectiveClose
            }

instance Aeson.ToJSON DirectionalitySnapshot where
    toJSON snapshot =
        Aeson.object
            [ "nonDirectional" Aeson..= dsNonDirectional snapshot
            , "reason" Aeson..= dsReason snapshot
            ]

normalizeSignalThreshold :: Double -> Double
normalizeSignalThreshold = id

sanitizeSignalThreshold :: Double -> Double
sanitizeSignalThreshold raw
    | finiteDouble raw = max 0 (normalizeSignalThreshold raw)
    | otherwise = 0

normalizeSignalEntryEdge :: Double -> Maybe Double
normalizeSignalEntryEdge raw
    | finiteDouble raw = Just (max 0 raw)
    | otherwise = Nothing

signalCrossAssetCheck :: Bool -> Maybe Int -> (Bool, Maybe String)
signalCrossAssetCheck enabled crossAssetDirRaw
    | not enabled = (True, Nothing)
    | otherwise =
        case crossAssetDirRaw of
            Just _ -> (True, Nothing)
            Nothing -> (False, Just "CROSS_ASSET")

class SignalDirectionalitySurface r where
    signalDirectionalitySnapshot :: r

instance SignalDirectionalitySurface DirectionalitySnapshot where
    signalDirectionalitySnapshot = failClosedSurface

instance SignalDirectionalitySurface (a -> b -> DirectionalitySnapshot) where
    signalDirectionalitySnapshot = const (const failClosedSurface)

instance SignalDirectionalitySurface (Double -> Maybe RegimeProbs -> V.Vector Double -> Int -> Maybe DirectionalitySnapshot) where
    signalDirectionalitySnapshot regimeBankHysteresis mRegimes pricesV t =
        signalDirectionalitySnapshotImpl regimeBankHysteresis mRegimes pricesV t Nothing

instance SignalDirectionalitySurface (Double -> Maybe RegimeProbs -> V.Vector Double -> Int -> Int -> Maybe DirectionalitySnapshot) where
    signalDirectionalitySnapshot regimeBankHysteresis mRegimes pricesV t chosenDir =
        signalDirectionalitySnapshotImpl regimeBankHysteresis mRegimes pricesV t (Just chosenDir)

directionalityLookbackBars :: Int
directionalityLookbackBars = sgcDirectionalityLookbackBars defaultSignalGateConfig

directionalityChopEfficiencyMax :: Double
directionalityChopEfficiencyMax = sgcDirectionalityChopEfficiencyMax defaultSignalGateConfig

directionalityMrEfficiencyMax :: Double
directionalityMrEfficiencyMax = sgcDirectionalityMrEfficiencyMax defaultSignalGateConfig

directionalityWeakBandZMin :: Double
directionalityWeakBandZMin = sgcDirectionalityWeakBandZMin defaultSignalGateConfig

directionalityEfficiencyTol :: Double
directionalityEfficiencyTol = 1e-12

directionalityRegimeMassTol :: Double
directionalityRegimeMassTol = 1e-3

malformedDirectionalitySnapshot :: DirectionalitySnapshot
malformedDirectionalitySnapshot =
    DirectionalitySnapshot
        { dsNonDirectional = True
        , dsReason = Just "NON_DIRECTIONAL_MALFORMED"
        }

mkDirectionalitySnapshot :: Maybe String -> DirectionalitySnapshot
mkDirectionalitySnapshot mReason =
    DirectionalitySnapshot
        { dsNonDirectional = isJust mReason
        , dsReason = mReason
        }

signalDirectionalitySnapshotImpl ::
    Double ->
    Maybe RegimeProbs ->
    V.Vector Double ->
    Int ->
    Maybe Int ->
    Maybe DirectionalitySnapshot
signalDirectionalitySnapshotImpl = signalDirectionalitySnapshotWithConfig defaultSignalGateConfig

signalDirectionalitySnapshotWithConfig ::
    SignalGateConfig ->
    Double ->
    Maybe RegimeProbs ->
    V.Vector Double ->
    Int ->
    Maybe Int ->
    Maybe DirectionalitySnapshot
signalDirectionalitySnapshotWithConfig cfg regimeBankHysteresis mRegimes pricesV t mChosenDir =
    signalDirectionalitySnapshotImpl'
        cfg
        regimeBankHysteresis
        mRegimes
        pricesV
        t
        mChosenDir
        (directionalityWeakBandConfirmedWithConfig cfg)

signalDirectionalitySnapshotImpl' ::
    SignalGateConfig ->
    Double ->
    Maybe RegimeProbs ->
    V.Vector Double ->
    Int ->
    Maybe Int ->
    (Double -> Maybe Int -> Bool) ->
    Maybe DirectionalitySnapshot
signalDirectionalitySnapshotImpl' cfg regimeBankHysteresis mRegimes pricesV t mChosenDir weakBandCheck =
    case directionalityWindowMetricsWithConfig cfg pricesV t of
        Nothing -> Just malformedDirectionalitySnapshot
        Just metrics ->
            let eff0 = wmEfficiency metrics
                eff
                    | efficiencyMalformed eff0 = Nothing
                    | otherwise = Just (clamp01 eff0)
                chopEfficiencyMax = clamp01 (sgcDirectionalityChopEfficiencyMax cfg)
                mrEfficiencyMax = clamp01 (max chopEfficiencyMax (sgcDirectionalityMrEfficiencyMax cfg))
                weakBand = maybe False (<= mrEfficiencyMax) eff
                hysteresisOk = finiteDouble regimeBankHysteresis && regimeBankHysteresis >= 0
                weakBandRegimeContext =
                    if weakBand
                        then directionalityRegimeContext regimeBankHysteresis mRegimes
                        else DirectionalityRegimeUnavailable
                mReason =
                    case eff of
                        Nothing -> Just "NON_DIRECTIONAL_MALFORMED"
                        Just efficiency
                            | efficiency <= chopEfficiencyMax -> Just "NON_DIRECTIONAL_CHOP"
                            | efficiency <= mrEfficiencyMax ->
                                if not hysteresisOk
                                    then Just "NON_DIRECTIONAL_MALFORMED"
                                    else case weakBandRegimeContext of
                                        DirectionalityRegimeInvalid -> Just "NON_DIRECTIONAL_MALFORMED"
                                        DirectionalityRegimeAvailable regimeSummary
                                            | drsMrDominant regimeSummary -> Just "NON_DIRECTIONAL_MR"
                                            | not (weakBandCheck (wmZScore metrics) mChosenDir) ->
                                                Just "NON_DIRECTIONAL_WEAK_BAND"
                                            | otherwise -> Nothing
                                        DirectionalityRegimeUnavailable
                                            | not (weakBandCheck (wmZScore metrics) mChosenDir) ->
                                                Just "NON_DIRECTIONAL_WEAK_BAND"
                                            | otherwise -> Nothing
                            | otherwise -> Nothing
             in Just (mkDirectionalitySnapshot mReason)

data DirectionalityWindowMetrics = DirectionalityWindowMetrics
    { wmEfficiency :: !Double
    , wmZScore :: !Double
    }

directionalityWindowMetrics :: V.Vector Double -> Int -> Maybe DirectionalityWindowMetrics
directionalityWindowMetrics = directionalityWindowMetricsWithConfig defaultSignalGateConfig

directionalityWindowMetricsWithConfig :: SignalGateConfig -> V.Vector Double -> Int -> Maybe DirectionalityWindowMetrics
directionalityWindowMetricsWithConfig cfg pricesV t =
    let lookback = max 1 (sgcDirectionalityLookbackBars cfg)
        start = max 1 (t - lookback + 1)
     in case traverse (returnAt pricesV) [start .. t] of
            Nothing -> Nothing
            Just [] -> Nothing
            Just returns ->
                let net = sum returns
                    path = sum (map abs returns)
                    efficiency =
                        if path <= 0
                            then 0
                            else abs net / path
                    stdev = stddevList returns
                    zScore =
                        if stdev <= 1e-12
                            then 0
                            else net / (stdev * sqrt (fromIntegral (length returns)))
                 in Just DirectionalityWindowMetrics{wmEfficiency = efficiency, wmZScore = zScore}

returnAt :: V.Vector Double -> Int -> Maybe Double
returnAt pricesV i
    | i <= 0 || i >= V.length pricesV = Nothing
    | otherwise =
        let prev = pricesV V.! (i - 1)
            next = pricesV V.! i
         in if prev == 0 || not (finiteDouble prev) || not (finiteDouble next)
                then Nothing
                else
                    let ret = next / prev - 1
                     in if finiteDouble ret then Just ret else Nothing

meanList :: [Double] -> Double
meanList xs =
    if null xs then 0 else sum xs / fromIntegral (length xs)

stddevList :: [Double] -> Double
stddevList xs =
    case xs of
        [] -> 0
        [_] -> 0
        _ ->
            let mu = meanList xs
                var = sum (map (\x -> (x - mu) ** 2) xs) / fromIntegral (length xs - 1)
             in sqrt (max 0 var)

efficiencyMalformed :: Double -> Bool
efficiencyMalformed efficiency =
    not (finiteDouble efficiency)
        || efficiency < negate directionalityEfficiencyTol
        || efficiency > 1 + directionalityEfficiencyTol

directionalityWeakBandConfirmed :: Double -> Maybe Int -> Bool
directionalityWeakBandConfirmed = directionalityWeakBandConfirmedWithConfig defaultSignalGateConfig

directionalityWeakBandConfirmedWithConfig :: SignalGateConfig -> Double -> Maybe Int -> Bool
directionalityWeakBandConfirmedWithConfig cfg zScore mChosenDir
    | not (finiteDouble zScore) = False
    | otherwise =
        let zMin = max 0 (sgcDirectionalityWeakBandZMin cfg)
         in case mChosenDir of
                Just dir | dir > 0 -> zScore >= zMin
                Just dir | dir < 0 -> zScore <= negate zMin
                _ -> False

newtype DirectionalityRegimeSummary = DirectionalityRegimeSummary
    { drsMrDominant :: Bool
    }

data DirectionalityRegimeContext
    = DirectionalityRegimeUnavailable
    | DirectionalityRegimeInvalid
    | DirectionalityRegimeAvailable !DirectionalityRegimeSummary

directionalityRegimeContext :: Double -> Maybe RegimeProbs -> DirectionalityRegimeContext
directionalityRegimeContext regimeBankHysteresis mRegimes =
    case mRegimes of
        Nothing -> DirectionalityRegimeUnavailable
        Just regimes ->
            case validateRegimes regimes of
                Nothing -> DirectionalityRegimeInvalid
                Just validRegimes ->
                    let ranked = sortOn (Data.Ord.Down . snd) validRegimes
                        mrDominant =
                            case ranked of
                                (("MR", pMr) : (_, p2) : _) -> pMr - p2 >= regimeBankHysteresis
                                _ -> False
                     in DirectionalityRegimeAvailable DirectionalityRegimeSummary{drsMrDominant = mrDominant}

validateRegimes :: RegimeProbs -> Maybe [(String, Double)]
validateRegimes regimes =
    let entries =
            [ ("TREND", rpTrend regimes)
            , ("MR", rpMR regimes)
            , ("HIGH_VOL", rpHighVol regimes)
            ]
        probs = map snd entries
        total = sum probs
        validProb p = finiteDouble p && p >= 0 && p <= 1
     in if all validProb probs
            && total > 0
            && abs (total - 1) <= directionalityRegimeMassTol
            then Just entries
            else Nothing

clamp01 :: Double -> Double
clamp01 = max 0 . min 1

entryEdgeHeadroomMultiple :: Double
entryEdgeHeadroomMultiple = sgcEntryEdgeHeadroomMultiple defaultSignalGateConfig

entryEdgeSpikeMultiple :: Double
entryEdgeSpikeMultiple = sgcEntryEdgeSpikeMultiple defaultSignalGateConfig

entryEdgeSpikeCredibleCap :: Double
entryEdgeSpikeCredibleCap = sgcEntryEdgeSpikeCredibleCap defaultSignalGateConfig

entryEdgeSpikeConsecutiveLimit :: Int
entryEdgeSpikeConsecutiveLimit = sgcEntryEdgeSpikeConsecutiveLimit defaultSignalGateConfig

validPositiveOrDefault :: Double -> Double -> Double
validPositiveOrDefault fallback value
    | finiteDouble value && value > 0 = value
    | otherwise = fallback

validNonNegativeOrDefault :: Double -> Double -> Double
validNonNegativeOrDefault fallback value
    | finiteDouble value && value >= 0 = value
    | otherwise = fallback

signalGateHeadroomMultiple :: SignalGateConfig -> Double
signalGateHeadroomMultiple cfg =
    validPositiveOrDefault entryEdgeHeadroomMultiple (sgcEntryEdgeHeadroomMultiple cfg)

signalGateSpikeMultiple :: SignalGateConfig -> Double
signalGateSpikeMultiple cfg =
    validNonNegativeOrDefault entryEdgeSpikeMultiple (sgcEntryEdgeSpikeMultiple cfg)

signalGateSpikeCredibleCap :: SignalGateConfig -> Double
signalGateSpikeCredibleCap cfg =
    validPositiveOrDefault entryEdgeSpikeCredibleCap (sgcEntryEdgeSpikeCredibleCap cfg)

signalGateSpikeConsecutiveLimit :: SignalGateConfig -> Int
signalGateSpikeConsecutiveLimit cfg =
    max 0 (sgcEntryEdgeSpikeConsecutiveLimit cfg)

signalGateTrendSlackMultiple :: SignalGateConfig -> Double
signalGateTrendSlackMultiple cfg =
    validNonNegativeOrDefault trendConfirmationSlackMultiple (sgcTrendConfirmationSlackMultiple cfg)

signalGateTrendSlackCap :: SignalGateConfig -> Double
signalGateTrendSlackCap cfg =
    validNonNegativeOrDefault trendConfirmationSlackCap (sgcTrendConfirmationSlackCap cfg)

signalEntryOpenThresholdFeasibilityCap :: Double
signalEntryOpenThresholdFeasibilityCap = entryEdgeSpikeCredibleCap / entryEdgeHeadroomMultiple

signalEntryOpenThresholdFeasible :: Double -> Bool
signalEntryOpenThresholdFeasible rawThreshold =
    case normalizeSignalOpenThreshold rawThreshold of
        Nothing -> False
        Just threshold -> threshold <= signalEntryOpenThresholdFeasibilityCap

signalEntryOpenThresholdFeasibilityReason :: Double -> Maybe String
signalEntryOpenThresholdFeasibilityReason rawThreshold =
    if signalEntryOpenThresholdFeasible rawThreshold
        then Nothing
        else Just "THRESHOLD_INFEASIBLE"

signalEntryHeadroomThresholdCap :: Double -> Double
signalEntryHeadroomThresholdCap rawEdge =
    case normalizeSignalEntryEdge rawEdge of
        Just edge -> edge / entryEdgeHeadroomMultiple
        Nothing -> 0

normalizeSignalOpenThreshold :: Double -> Maybe Double
normalizeSignalOpenThreshold raw
    | finiteDouble raw && raw >= 0 = Just (normalizeSignalThreshold raw)
    | otherwise = Nothing

normalizeSignalFeeFloor :: Double -> Maybe Double
normalizeSignalFeeFloor raw
    | finiteDouble raw && raw >= 0 = Just raw
    | otherwise = Nothing

signalEntryHeadroomOk :: Double -> Maybe Double -> Bool
signalEntryHeadroomOk = signalEntryHeadroomOkWithConfig defaultSignalGateConfig

signalEntryHeadroomOkWithConfig :: SignalGateConfig -> Double -> Maybe Double -> Bool
signalEntryHeadroomOkWithConfig cfg openThreshold = signalEntryFeeBufferOkWithConfig cfg openThreshold 0

signalEntryEdgeSpikeOk :: Double -> Maybe Double -> Bool
signalEntryEdgeSpikeOk = signalEntryEdgeSpikeOkWithConfig defaultSignalGateConfig

signalEntryEdgeSpikeOkWithConfig :: SignalGateConfig -> Double -> Maybe Double -> Bool
signalEntryEdgeSpikeOkWithConfig cfg openThreshold edgeForMethod =
    case normalizeSignalOpenThreshold openThreshold of
        Nothing -> False
        Just threshold ->
            let spikeCap =
                    min
                        (signalGateSpikeMultiple cfg * threshold)
                        (signalGateSpikeCredibleCap cfg)
             in maybe False (\edge -> finiteDouble edge && edge >= 0 && edge <= spikeCap) edgeForMethod

signalEntryEdgeSpikeEntryOk :: Bool -> Double -> Maybe Double -> Bool
signalEntryEdgeSpikeEntryOk = signalEntryEdgeSpikeEntryOkWithConfig defaultSignalGateConfig

signalEntryEdgeSpikeEntryOkWithConfig :: SignalGateConfig -> Bool -> Double -> Maybe Double -> Bool
signalEntryEdgeSpikeEntryOkWithConfig cfg auditOnly openThreshold edgeForMethod =
    auditOnly || signalEntryEdgeSpikeOkWithConfig cfg openThreshold edgeForMethod

signalEntryEdgeSpikeAuditWarning :: Bool -> Double -> Maybe Double -> Maybe String
signalEntryEdgeSpikeAuditWarning = signalEntryEdgeSpikeAuditWarningWithConfig defaultSignalGateConfig

signalEntryEdgeSpikeAuditWarningWithConfig :: SignalGateConfig -> Bool -> Double -> Maybe Double -> Maybe String
signalEntryEdgeSpikeAuditWarningWithConfig cfg auditOnly openThreshold edgeForMethod =
    if auditOnly && not (signalEntryEdgeSpikeOkWithConfig cfg openThreshold edgeForMethod)
        then Just "EDGE_SPIKE"
        else Nothing

{- | Circuit-breaker variant: after 3 consecutive edge-spike warnings,
treat audit-only mode as blocking. This prevents severely miscalibrated
LSTM models from generating infinite permissive audit warnings.
-}
signalEntryEdgeSpikeConsecutiveOk :: Int -> Bool -> Double -> Maybe Double -> Bool
signalEntryEdgeSpikeConsecutiveOk = signalEntryEdgeSpikeConsecutiveOkWithConfig defaultSignalGateConfig

signalEntryEdgeSpikeConsecutiveOkWithConfig :: SignalGateConfig -> Int -> Bool -> Double -> Maybe Double -> Bool
signalEntryEdgeSpikeConsecutiveOkWithConfig cfg consecutiveWarnings auditOnly openThreshold edgeForMethod =
    let ok = signalEntryEdgeSpikeOkWithConfig cfg openThreshold edgeForMethod
     in ok || (auditOnly && consecutiveWarnings < signalGateSpikeConsecutiveLimit cfg)

signalEntryFeeBufferOk :: Double -> Double -> Maybe Double -> Bool
signalEntryFeeBufferOk = signalEntryFeeBufferOkWithConfig defaultSignalGateConfig

signalEntryFeeBufferOkWithConfig :: SignalGateConfig -> Double -> Double -> Maybe Double -> Bool
signalEntryFeeBufferOkWithConfig cfg openThreshold roundTripFeeFloor edgeForMethod =
    case normalizeSignalOpenThreshold openThreshold of
        Nothing -> False
        Just threshold ->
            let requiredHeadroom = signalGateHeadroomMultiple cfg * threshold
             in case normalizeSignalFeeFloor roundTripFeeFloor of
                    Nothing -> False
                    Just feeFloor ->
                        let requiredEdge = requiredHeadroom + feeFloor
                         in maybe False (\edge -> finiteDouble edge && edge >= requiredEdge) edgeForMethod

trendConfirmationSlackMultiple :: Double
trendConfirmationSlackMultiple = sgcTrendConfirmationSlackMultiple defaultSignalGateConfig

trendConfirmationSlackCap :: Double
trendConfirmationSlackCap = sgcTrendConfirmationSlackCap defaultSignalGateConfig

signalTrendSmaConfirmed :: Double -> Double -> Double -> Int -> Bool
signalTrendSmaConfirmed = signalTrendSmaConfirmedWithConfig defaultSignalGateConfig

signalTrendSmaConfirmedWithConfig :: SignalGateConfig -> Double -> Double -> Double -> Int -> Bool
signalTrendSmaConfirmedWithConfig cfg openThreshold currentPrice sma dir
    | not (finiteDouble currentPrice && finiteDouble sma) = True
    | otherwise =
        let slackFrac =
                case normalizeSignalOpenThreshold openThreshold of
                    Just threshold -> min (signalGateTrendSlackCap cfg) (signalGateTrendSlackMultiple cfg * threshold)
                    Nothing -> 0
            lowerBound = sma * (1 - slackFrac)
            upperBound = sma * (1 + slackFrac)
         in case dir of
                d | d > 0 -> currentPrice >= lowerBound
                d | d < 0 -> currentPrice <= upperBound
                _ -> True

signalFundingOiCheck :: Bool -> Maybe Double -> Maybe Double -> Double -> Double -> Maybe Double -> (Bool, Double)
signalFundingOiCheck enabled fundingCap volCap sizeMult fundingPressure oiVolProxy
    | not enabled = (True, 1)
    | otherwise =
        let fundingOk =
                case fundingCap of
                    Nothing -> True
                    Just cap -> finiteDouble fundingPressure && abs fundingPressure <= max 0 cap
            oiVolOk =
                case volCap of
                    Nothing -> True
                    Just cap ->
                        case oiVolProxy of
                            Just oiVol -> finiteDouble oiVol && oiVol <= max 0 cap
                            Nothing -> False
         in if fundingOk && oiVolOk
                then (True, max 0 sizeMult)
                else (False, 0)

signalMetaLabelOk :: Bool -> Double -> Maybe Double -> Double -> Maybe Double -> Bool -> Bool -> Bool
signalMetaLabelOk enabled minEdge edgeForMethod minConfidence methodConfidence requireBand bandOk
    | not enabled = True
    | otherwise =
        let edgeOk =
                case edgeForMethod of
                    Just edge -> finiteDouble edge && edge >= max 0 minEdge
                    Nothing -> False
            confidenceOk =
                case methodConfidence of
                    Just confidence -> finiteDouble confidence && confidence >= max 0 minConfidence
                    Nothing -> False
            bandGateOk = not requireBand || bandOk
         in edgeOk && confidenceOk && bandGateOk

signalMtfConsensusCheck :: Bool -> [Maybe Int] -> Int -> (Bool, Maybe String)
signalMtfConsensusCheck enabled mtfDirs mtfMinAgree
    | not enabled = (True, Nothing)
    | otherwise =
        let dirs = catMaybes mtfDirs
            minAgree = max 1 mtfMinAgree
            consensusOk =
                length dirs >= minAgree
                    && case dirs of
                        [] -> False
                        dir0 : rest -> all (== dir0) rest
         in if consensusOk
                then (True, Nothing)
                else (False, Just "MTF_CONSENSUS")

signalRegimeEdgeOk :: Bool -> Double -> Maybe Double -> (Bool, Maybe String)
signalRegimeEdgeOk enabled minEdge edgeForMethod
    | not enabled = (True, Nothing)
    | otherwise =
        let minEdge' = max 0 minEdge
         in case edgeForMethod of
                Just edge | finiteDouble edge && edge >= minEdge' -> (True, Nothing)
                _ -> (False, Just "REGIME_EDGE")

signalRunPostDirectionGates ::
    Maybe Int ->
    Maybe String ->
    Bool ->
    Bool ->
    (Int -> Bool) ->
    (Int -> Bool) ->
    (Int -> Bool) ->
    Bool ->
    (Int -> (Bool, Maybe String)) ->
    (Bool, Maybe String) ->
    (Bool, Maybe String) ->
    (Bool, Maybe String) ->
    (Int -> Bool) ->
    (Int -> (Bool, Double)) ->
    (Maybe Int, Maybe String)
signalRunPostDirectionGates chosenDir0 mChosenReason volOk volTargetReady trendOk cloudOk priceActionOk signalToNoiseOk nonDirectionalCheck regimeEdgeOk mtfConsensusCheck crossAssetCheck metaLabelOk fundingOiCheck =
    case chosenDir0 of
        Nothing -> (Nothing, mChosenReason)
        Just dir ->
            let (nonDirectionalOk, mNonDirectionalReason) = nonDirectionalCheck dir
                (regimeOk, mRegimeReason) = regimeEdgeOk
                (mtfOk, mMtfReason) = mtfConsensusCheck
                (crossAssetOk, mCrossAssetReason) = crossAssetCheck
                metaOk = metaLabelOk dir
                (fundingOk, _fundingScale) = fundingOiCheck dir
                firstReason =
                    chooseReason
                        [ guardReason volOk "VOLATILITY"
                        , guardReason volTargetReady "VOL_TARGET"
                        , guardReason (trendOk dir) "TREND"
                        , guardReason (cloudOk dir) "CLOUD"
                        , guardReason (priceActionOk dir) "PRICE_ACTION"
                        , guardReason signalToNoiseOk "SIGNAL_TO_NOISE"
                        , gateReason nonDirectionalOk mNonDirectionalReason "NON_DIRECTIONAL"
                        , gateReason regimeOk mRegimeReason "REGIME_EDGE"
                        , gateReason mtfOk mMtfReason "MTF_CONSENSUS"
                        , gateReason crossAssetOk mCrossAssetReason "CROSS_ASSET"
                        , guardReason metaOk "META_LABEL"
                        , guardReason fundingOk "FUNDING_OI"
                        ]
             in case firstReason of
                    Nothing -> (Just dir, mChosenReason)
                    Just reason -> (Nothing, chooseReason [mChosenReason, Just reason])

guardReason :: Bool -> String -> Maybe String
guardReason isOpen reason =
    if isOpen
        then Nothing
        else Just reason

gateReason :: Bool -> Maybe String -> String -> Maybe String
gateReason isOpen mReason fallback =
    if isOpen
        then Nothing
        else Just (fromMaybe fallback mReason)

chooseReason :: [Maybe String] -> Maybe String
chooseReason [] = Nothing
chooseReason (Nothing : rest) = chooseReason rest
chooseReason (reason : _) = reason

signalPredictionSanityOk :: Double -> Maybe Double -> (Bool, Maybe String)
signalPredictionSanityOk _currentPrice Nothing = (True, Nothing)
signalPredictionSanityOk currentPrice (Just predVal)
    | isNaN predVal = (False, Just "MODEL_ANOMALY")
    | predVal < 0 = (False, Just "MODEL_ANOMALY")
    | predVal < currentPrice * 0.01 = (False, Just "MODEL_ANOMALY")
    | predVal > currentPrice * 100 = (False, Just "MODEL_ANOMALY")
    | otherwise = (True, Nothing)

directionalityWeakBandConfirmedWithPrediction :: Double -> Maybe Int -> Maybe Double -> Double -> Bool
directionalityWeakBandConfirmedWithPrediction = directionalityWeakBandConfirmedWithPredictionAndConfig defaultSignalGateConfig

directionalityWeakBandConfirmedWithPredictionAndConfig :: SignalGateConfig -> Double -> Maybe Int -> Maybe Double -> Double -> Bool
directionalityWeakBandConfirmedWithPredictionAndConfig cfg zScore mChosenDir mPrediction _currentPrice =
    case mPrediction of
        Nothing -> directionalityWeakBandConfirmedWithConfig cfg zScore mChosenDir
        Just predVal
            | not (finiteDouble predVal) -> directionalityWeakBandConfirmedWithConfig cfg zScore mChosenDir
            | otherwise ->
                case mChosenDir of
                    Just dir | dir > 0 && predVal > 0 -> True
                    Just dir | dir < 0 && predVal < 0 -> True
                    _ -> directionalityWeakBandConfirmedWithConfig cfg zScore mChosenDir

signalDirectionalitySnapshotImplWithPrediction ::
    Double ->
    Maybe RegimeProbs ->
    V.Vector Double ->
    Int ->
    Maybe Int ->
    Maybe Double ->
    Double ->
    Maybe DirectionalitySnapshot
signalDirectionalitySnapshotImplWithPrediction =
    signalDirectionalitySnapshotImplWithPredictionAndConfig defaultSignalGateConfig

signalDirectionalitySnapshotImplWithPredictionAndConfig ::
    SignalGateConfig ->
    Double ->
    Maybe RegimeProbs ->
    V.Vector Double ->
    Int ->
    Maybe Int ->
    Maybe Double ->
    Double ->
    Maybe DirectionalitySnapshot
signalDirectionalitySnapshotImplWithPredictionAndConfig cfg regimeBankHysteresis mRegimes pricesV t mChosenDir mPrediction currentPrice =
    signalDirectionalitySnapshotImpl'
        cfg
        regimeBankHysteresis
        mRegimes
        pricesV
        t
        mChosenDir
        (\zScore mDir -> directionalityWeakBandConfirmedWithPredictionAndConfig cfg zScore mDir mPrediction currentPrice)

{- | Scale the edge-spike cap by interval duration so longer intervals allow
proportionally larger per-bar edges. Empty or unparseable intervals fall
back to the interval-agnostic cap.
-}
intervalEdgeSpikeMultiplier :: String -> Double
intervalEdgeSpikeMultiplier interval =
    case parseIntervalSeconds interval of
        Nothing -> 1.0
        Just sec
            | sec <= 300 -> 1.0
            | sec <= 3600 -> 1.1
            | sec <= 14400 -> 1.2
            | sec <= 43200 -> 1.4
            | sec <= 86400 -> 1.5
            | otherwise -> 2.0

signalEntryEdgeSpikeOkInterval :: String -> Double -> Maybe Double -> Bool
signalEntryEdgeSpikeOkInterval interval openThreshold edgeForMethod =
    case normalizeSignalOpenThreshold openThreshold of
        Nothing -> False
        Just threshold ->
            let mult = intervalEdgeSpikeMultiplier interval
                spikeCap = min (entryEdgeSpikeMultiple * threshold * mult) (entryEdgeSpikeCredibleCap * mult)
             in maybe False (\edge -> finiteDouble edge && edge >= 0 && edge <= spikeCap) edgeForMethod

signalEntryEdgeSpikeEntryOkInterval :: String -> Bool -> Double -> Maybe Double -> Bool
signalEntryEdgeSpikeEntryOkInterval interval auditOnly openThreshold edgeForMethod =
    auditOnly || signalEntryEdgeSpikeOkInterval interval openThreshold edgeForMethod

signalEntryEdgeSpikeAuditWarningInterval :: String -> Bool -> Double -> Maybe Double -> Maybe String
signalEntryEdgeSpikeAuditWarningInterval interval auditOnly openThreshold edgeForMethod =
    if auditOnly && not (signalEntryEdgeSpikeOkInterval interval openThreshold edgeForMethod)
        then Just "EDGE_SPIKE"
        else Nothing

{- | Series-wide dynamic range as a fraction of the mean: @(max - min) / |mean|@
over the finite points. Returns 'Nothing' when fewer than two finite points are
present. This is the headline statistic for predictor liveness — a predictor
that barely varies across a window is not responding to its input.
-}
dynamicRangePct :: [Maybe Double] -> Maybe Double
dynamicRangePct series =
    case filter finiteDouble (catMaybes series) of
        finite@(_ : _ : _) ->
            let mn = minimum finite
                mx = maximum finite
                mean = sum finite / fromIntegral (length finite)
                denom = abs mean
             in Just (if denom > 0 then (mx - mn) / denom else 0)
        _ -> Nothing

{- | Liveness diagnostics for a single predictor's forecast series over a
backtest split, measured against the price series the method was supposed to
track.

Motivation (observed 2026-06-02): running methods @01@ (LSTM-only), @10@
(Kalman-only) and @blend@ at @--epochs 0@ over a contiguous +20% BTCUSDT 4h
up-trend produced *zero trades*, flat equity, and no warning. Instrumenting the
forecast series partitioned the three "zero-trade" methods, which prior notes
had conflated into one defect:

  * @01@: the untrained LSTM emitted ~94400 on every bar while price ran to
    120750 — dynamic range 0.0074% of its mean vs the price's 20.3%. The model
    ignores its input entirely. This is a genuinely *dead* predictor.
  * @10@: the Kalman forecast tracked the move (range 18.7%, ~92% of the price
    range) yet still opened no position — a *live* predictor gated out
    downstream, a different and real investigation target.

'plPriceTrackingRatio' is the discriminating statistic: a forecast whose
dynamic range is a tiny fraction of the price's own range is not tracking the
market, regardless of regime. Per-bar sanity gates (see
'signalPredictionSanityOk') cannot catch this because each point is individually
in-scale; only the series-wide range relative to price reveals it.
-}
data PredictorLiveness = PredictorLiveness
    { plSeries :: !String
    -- ^ predictor label, e.g. "lstm" | "kalman"
    , plFiniteCount :: !Int
    -- ^ number of finite forecast points assessed
    , plDynamicRangePct :: !Double
    -- ^ (max - min) / |mean| of finite forecasts
    , plMaxStepReturn :: !Double
    -- ^ max |p[t+1] - p[t]| / |p[t]| over consecutive finite forecasts (telemetry)
    , plPriceTrackingRatio :: !Double
    -- ^ plDynamicRangePct / priceDynamicRangePct; ~1 means the forecast moves as
    -- much as price, ~0 means a near-constant (dead) forecast
    }
    deriving (Eq, Show)

{- | Compute liveness for a forecast series, given the price series' own dynamic
range (@(max - min)/|mean|@) as the tracking reference. Returns 'Nothing' when
the forecast has fewer than two finite points. Steps are taken across the
compacted finite series, which can only *overstate* 'plMaxStepReturn' across
gaps — the conservative direction (it never invents liveness that is not there).
-}
predictorLiveness :: String -> Double -> [Maybe Double] -> Maybe PredictorLiveness
predictorLiveness label priceDynamicRangePct series =
    case filter finiteDouble (catMaybes series) of
        finite@(_ : _ : _) ->
            let mn = minimum finite
                mx = maximum finite
                mean = sum finite / fromIntegral (length finite)
                denom = abs mean
                rangePct = if denom > 0 then (mx - mn) / denom else 0
                steps =
                    [ abs (b - a) / abs a
                    | (a, b) <- zip finite (drop 1 finite)
                    , abs a > 0
                    ]
                maxStep = if null steps then 0 else maximum steps
                trackingRatio =
                    if finiteDouble priceDynamicRangePct && priceDynamicRangePct > 0
                        then rangePct / priceDynamicRangePct
                        else 0
             in Just
                    PredictorLiveness
                        { plSeries = label
                        , plFiniteCount = length finite
                        , plDynamicRangePct = rangePct
                        , plMaxStepReturn = maxStep
                        , plPriceTrackingRatio = trackingRatio
                        }
        _ -> Nothing

{- | A forecast tracking less than this fraction of the price's own dynamic
range is treated as not tracking the market (a dead predictor). 5% leaves a
wide margin: the observed dead LSTM tracked 0.04%, the live Kalman tracked 92%.
-}
predictorTrackingFloor :: Double
predictorTrackingFloor = sgcPredictorTrackingFloor defaultSignalGateConfig

{- | A method's predictor stack is degenerate when it produced no closed trades
AND every available forecast series tracked less than 'predictorTrackingFloor'
of the price's dynamic range — i.e. the models were effectively constant, so the
zero-trade outcome was structural rather than a market decision.

Fail-open by design: an empty liveness list yields 'False'. This is a diagnostic
surfaced in the backtest payload, not a trade gate, so the safe default is to
*not* cry wolf. A method that closed trades, or that had a live (tracking)
predictor which was merely gated out, is correctly *not* flagged.
-}
predictorDegenerate :: Int -> [PredictorLiveness] -> Bool
predictorDegenerate = predictorDegenerateWithConfig defaultSignalGateConfig

predictorDegenerateWithConfig :: SignalGateConfig -> Int -> [PredictorLiveness] -> Bool
predictorDegenerateWithConfig cfg closedTrades liveness =
    closedTrades == 0
        && not (null liveness)
        && all (\pl -> plPriceTrackingRatio pl < max 0 (sgcPredictorTrackingFloor cfg)) liveness

-- | JSON encoding of a single predictor's liveness diagnostics.
predictorLivenessToJson :: PredictorLiveness -> Aeson.Value
predictorLivenessToJson pl =
    Aeson.object
        [ "series" Aeson..= plSeries pl
        , "finiteCount" Aeson..= plFiniteCount pl
        , "dynamicRangePct" Aeson..= plDynamicRangePct pl
        , "maxStepReturn" Aeson..= plMaxStepReturn pl
        , "priceTrackingRatio" Aeson..= plPriceTrackingRatio pl
        ]

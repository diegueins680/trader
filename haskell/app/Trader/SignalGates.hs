{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

module Trader.SignalGates (
    DirectionalitySnapshot (..),
    SignalThresholdBoundary (..),
    finiteDouble,
    mkSignalThresholdBoundary,
    normalizeSignalThreshold,
    normalizeSignalEntryEdge,
    signalCrossAssetCheck,
    signalDirectionalitySnapshot,
    directionalityWeakBandConfirmed,
    directionalityWeakBandConfirmedWithPrediction,
    signalDirectionalitySnapshotImplWithPrediction,
    signalEntryOpenThresholdFeasibilityCap,
    signalEntryOpenThresholdFeasibilityReason,
    signalEntryOpenThresholdFeasible,
    signalEntryHeadroomThresholdCap,
    normalizeSignalOpenThreshold,
    normalizeSignalFeeFloor,
    signalEntryHeadroomOk,
    signalEntryEdgeSpikeOk,
    signalEntryEdgeSpikeEntryOk,
    signalEntryEdgeSpikeAuditWarning,
    signalEntryEdgeSpikeOkInterval,
    signalEntryEdgeSpikeEntryOkInterval,
    signalEntryEdgeSpikeAuditWarningInterval,
    signalEntryEdgeSpikeConsecutiveOk,
    signalEntryFeeBufferOk,
    signalTrendSmaConfirmed,
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
directionalityLookbackBars = 24

directionalityChopEfficiencyMax :: Double
directionalityChopEfficiencyMax = 0.08

directionalityMrEfficiencyMax :: Double
directionalityMrEfficiencyMax = 0.35

directionalityWeakBandZMin :: Double
directionalityWeakBandZMin = 0.5

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
signalDirectionalitySnapshotImpl regimeBankHysteresis mRegimes pricesV t mChosenDir =
    signalDirectionalitySnapshotImpl'
        regimeBankHysteresis
        mRegimes
        pricesV
        t
        mChosenDir
        directionalityWeakBandConfirmed

signalDirectionalitySnapshotImpl' ::
    Double ->
    Maybe RegimeProbs ->
    V.Vector Double ->
    Int ->
    Maybe Int ->
    (Double -> Maybe Int -> Bool) ->
    Maybe DirectionalitySnapshot
signalDirectionalitySnapshotImpl' regimeBankHysteresis mRegimes pricesV t mChosenDir weakBandCheck =
    case directionalityWindowMetrics pricesV t of
        Nothing -> Just malformedDirectionalitySnapshot
        Just metrics ->
            let eff0 = wmEfficiency metrics
                eff
                    | efficiencyMalformed eff0 = Nothing
                    | otherwise = Just (clamp01 eff0)
                weakBand = maybe False (<= directionalityMrEfficiencyMax) eff
                hysteresisOk = finiteDouble regimeBankHysteresis && regimeBankHysteresis >= 0
                weakBandRegimeContext =
                    if weakBand
                        then directionalityRegimeContext regimeBankHysteresis mRegimes
                        else DirectionalityRegimeUnavailable
                mReason =
                    case eff of
                        Nothing -> Just "NON_DIRECTIONAL_MALFORMED"
                        Just efficiency
                            | efficiency <= directionalityChopEfficiencyMax -> Just "NON_DIRECTIONAL_CHOP"
                            | efficiency <= directionalityMrEfficiencyMax ->
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
directionalityWindowMetrics pricesV t =
    let start = max 1 (t - directionalityLookbackBars + 1)
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
directionalityWeakBandConfirmed zScore mChosenDir
    | not (finiteDouble zScore) = False
    | otherwise =
        case mChosenDir of
            Just dir | dir > 0 -> zScore >= directionalityWeakBandZMin
            Just dir | dir < 0 -> zScore <= negate directionalityWeakBandZMin
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
entryEdgeHeadroomMultiple = 1.5

entryEdgeSpikeMultiple :: Double
entryEdgeSpikeMultiple = 1000.0

entryEdgeSpikeCredibleCap :: Double
entryEdgeSpikeCredibleCap = 5.0

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
signalEntryHeadroomOk openThreshold = signalEntryFeeBufferOk openThreshold 0

signalEntryEdgeSpikeOk :: Double -> Maybe Double -> Bool
signalEntryEdgeSpikeOk openThreshold edgeForMethod =
    case normalizeSignalOpenThreshold openThreshold of
        Nothing -> False
        Just threshold ->
            let spikeCap = min (entryEdgeSpikeMultiple * threshold) entryEdgeSpikeCredibleCap
             in maybe False (\edge -> finiteDouble edge && edge >= 0 && edge <= spikeCap) edgeForMethod

signalEntryEdgeSpikeEntryOk :: Bool -> Double -> Maybe Double -> Bool
signalEntryEdgeSpikeEntryOk auditOnly openThreshold edgeForMethod =
    auditOnly || signalEntryEdgeSpikeOk openThreshold edgeForMethod

signalEntryEdgeSpikeAuditWarning :: Bool -> Double -> Maybe Double -> Maybe String
signalEntryEdgeSpikeAuditWarning auditOnly openThreshold edgeForMethod =
    if auditOnly && not (signalEntryEdgeSpikeOk openThreshold edgeForMethod)
        then Just "EDGE_SPIKE"
        else Nothing

{- | Circuit-breaker variant: after 3 consecutive edge-spike warnings,
treat audit-only mode as blocking. This prevents severely miscalibrated
LSTM models from generating infinite permissive audit warnings.
-}
signalEntryEdgeSpikeConsecutiveOk :: Int -> Bool -> Double -> Maybe Double -> Bool
signalEntryEdgeSpikeConsecutiveOk consecutiveWarnings auditOnly openThreshold edgeForMethod =
    let ok = signalEntryEdgeSpikeOk openThreshold edgeForMethod
     in ok || (auditOnly && consecutiveWarnings < 3)

signalEntryFeeBufferOk :: Double -> Double -> Maybe Double -> Bool
signalEntryFeeBufferOk openThreshold roundTripFeeFloor edgeForMethod =
    case normalizeSignalOpenThreshold openThreshold of
        Nothing -> False
        Just threshold ->
            let requiredHeadroom = entryEdgeHeadroomMultiple * threshold
             in case normalizeSignalFeeFloor roundTripFeeFloor of
                    Nothing -> False
                    Just feeFloor ->
                        let requiredEdge = requiredHeadroom + feeFloor
                         in maybe False (\edge -> finiteDouble edge && edge >= requiredEdge) edgeForMethod

trendConfirmationSlackMultiple :: Double
trendConfirmationSlackMultiple = 0.5

trendConfirmationSlackCap :: Double
trendConfirmationSlackCap = 0.01

signalTrendSmaConfirmed :: Double -> Double -> Double -> Int -> Bool
signalTrendSmaConfirmed openThreshold currentPrice sma dir
    | not (finiteDouble currentPrice && finiteDouble sma) = True
    | otherwise =
        let slackFrac =
                case normalizeSignalOpenThreshold openThreshold of
                    Just threshold -> min trendConfirmationSlackCap (trendConfirmationSlackMultiple * threshold)
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
directionalityWeakBandConfirmedWithPrediction zScore mChosenDir mPrediction _currentPrice =
    case mPrediction of
        Nothing -> directionalityWeakBandConfirmed zScore mChosenDir
        Just predVal
            | not (finiteDouble predVal) -> directionalityWeakBandConfirmed zScore mChosenDir
            | otherwise ->
                case mChosenDir of
                    Just dir | dir > 0 && predVal > 0 -> True
                    Just dir | dir < 0 && predVal < 0 -> True
                    _ -> directionalityWeakBandConfirmed zScore mChosenDir

signalDirectionalitySnapshotImplWithPrediction ::
    Double ->
    Maybe RegimeProbs ->
    V.Vector Double ->
    Int ->
    Maybe Int ->
    Maybe Double ->
    Double ->
    Maybe DirectionalitySnapshot
signalDirectionalitySnapshotImplWithPrediction regimeBankHysteresis mRegimes pricesV t mChosenDir mPrediction currentPrice =
    signalDirectionalitySnapshotImpl'
        regimeBankHysteresis
        mRegimes
        pricesV
        t
        mChosenDir
        (\zScore mDir -> directionalityWeakBandConfirmedWithPrediction zScore mDir mPrediction currentPrice)

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

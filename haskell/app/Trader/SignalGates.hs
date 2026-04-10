{-# LANGUAGE OverloadedStrings #-}

module Trader.SignalGates (
    SignalThresholdBoundary (..),
    DirectionalitySnapshot (..),
    mkSignalThresholdBoundary,
    normalizeSignalThreshold,
    signalDirectionalitySnapshot,
    signalDirectionalityEntryAllowed,
    signalEntryHeadroomThresholdCap,
    signalEntryHeadroomOk,
    signalEntryFeeBufferOk,
    signalEntryEdgeSpikeOk,
    signalMetaLabelOk,
    signalMtfConsensusCheck,
    signalCrossAssetCheck,
    signalRegimeEdgeOk,
    signalFundingOiCheck,
    signalRunPostDirectionGates,
) where

import Control.Applicative ((<|>))
import Data.Aeson (ToJSON (..), object, (.=))
import Data.List (sortOn)
import Data.Maybe (catMaybes, isJust)
import qualified Data.Ord
import qualified Data.Vector as V
import Trader.Predictors.Types (RegimeProbs (..))

data SignalThresholdBoundary = SignalThresholdBoundary
    { stbConfiguredOpenThreshold :: !Double
    , stbConfiguredCloseThreshold :: !Double
    , stbEffectiveOpenThreshold :: !Double
    , stbEffectiveCloseThreshold :: !Double
    }
    deriving (Eq, Show)

data DirectionalitySnapshot = DirectionalitySnapshot
    { dsLookbackBars :: !Int
    , dsNetReturnPct :: !Double
    , dsRealizedVolPct :: !Double
    , dsEfficiency :: !Double
    , dsZScore :: !Double
    , dsLabel :: !String
    , dsTrendProb :: !(Maybe Double)
    , dsMrProb :: !(Maybe Double)
    , dsHighVolProb :: !(Maybe Double)
    , dsRegimeLeader :: !(Maybe String)
    , dsRegimeGap :: !(Maybe Double)
    , dsNonDirectional :: !Bool
    , dsReason :: !(Maybe String)
    }
    deriving (Eq, Show)

instance ToJSON DirectionalitySnapshot where
    toJSON s =
        object
            [ "lookbackBars" .= dsLookbackBars s
            , "netReturnPct" .= dsNetReturnPct s
            , "realizedVolPct" .= dsRealizedVolPct s
            , "efficiency" .= dsEfficiency s
            , "zScore" .= dsZScore s
            , "label" .= dsLabel s
            , "trendProb" .= dsTrendProb s
            , "mrProb" .= dsMrProb s
            , "highVolProb" .= dsHighVolProb s
            , "regimeLeader" .= dsRegimeLeader s
            , "regimeGap" .= dsRegimeGap s
            , "nonDirectional" .= dsNonDirectional s
            , "reason" .= dsReason s
            ]

maxSignalThreshold :: Double
maxSignalThreshold = 0.999999

normalizeSignalThreshold :: Double -> Double
normalizeSignalThreshold raw =
    if finiteDouble raw && raw > 0
        then min maxSignalThreshold raw
        else 0

mkSignalThresholdBoundary :: Double -> Double -> Double -> Double -> SignalThresholdBoundary
mkSignalThresholdBoundary configuredOpen configuredClose effectiveOpen effectiveClose =
    SignalThresholdBoundary
        { stbConfiguredOpenThreshold = normalizeConfigured configuredOpen
        , stbConfiguredCloseThreshold = normalizeConfigured configuredClose
        , stbEffectiveOpenThreshold = normalizeSignalThreshold effectiveOpen
        , stbEffectiveCloseThreshold = normalizeSignalThreshold effectiveClose
        }
  where
    normalizeConfigured raw =
        if finiteDouble raw && raw >= 0
            then raw
            else 0

entryEdgeSpikeLimit :: Double
entryEdgeSpikeLimit = 4.0

entryEdgeHeadroomMultiple :: Double
entryEdgeHeadroomMultiple = 1.5

maxCredibleSignalEdge :: Double
maxCredibleSignalEdge = 0.5

directionalityLookbackBars :: Int
directionalityLookbackBars = 24

directionalityHighVolVolPct :: Double
directionalityHighVolVolPct = 1.5

directionalityTrendEfficiencyMin :: Double
directionalityTrendEfficiencyMin = 0.45

directionalityTrendZMin :: Double
directionalityTrendZMin = 1.0

directionalityChopEfficiencyMax :: Double
directionalityChopEfficiencyMax = 0.25

directionalityMrEfficiencyMax :: Double
directionalityMrEfficiencyMax = 0.4

directionalityMalformedReason :: String
directionalityMalformedReason = "NON_DIRECTIONAL_MALFORMED"

finiteDouble :: Double -> Bool
finiteDouble x = not (isNaN x || isInfinite x)

directionalityEfficiencyOk :: Double -> Bool
directionalityEfficiencyOk eff =
    finiteDouble eff && eff >= 0 && eff <= 1

directionalityProbOk :: Double -> Bool
directionalityProbOk p = finiteDouble p && p >= 0 && p <= 1

directionalityGapOk :: Double -> Bool
directionalityGapOk g = finiteDouble g && g >= 0 && g <= 1

mkMalformedDirectionalitySnapshot :: Int -> DirectionalitySnapshot
mkMalformedDirectionalitySnapshot windowLen =
    DirectionalitySnapshot
        { dsLookbackBars = max 0 windowLen
        , dsNetReturnPct = 0
        , dsRealizedVolPct = 0
        , dsEfficiency = 0
        , dsZScore = 0
        , dsLabel = "malformed"
        , dsTrendProb = Nothing
        , dsMrProb = Nothing
        , dsHighVolProb = Nothing
        , dsRegimeLeader = Nothing
        , dsRegimeGap = Nothing
        , dsNonDirectional = True
        , dsReason = Just directionalityMalformedReason
        }

directionalitySnapshotWellFormed :: DirectionalitySnapshot -> Bool
directionalitySnapshotWellFormed snap =
    dsLookbackBars snap >= 3
        && finiteDouble (dsNetReturnPct snap)
        && finiteDouble (dsRealizedVolPct snap)
        && directionalityEfficiencyOk (dsEfficiency snap)
        && finiteDouble (dsZScore snap)
        && maybe True directionalityProbOk (dsTrendProb snap)
        && maybe True directionalityProbOk (dsMrProb snap)
        && maybe True directionalityProbOk (dsHighVolProb snap)
        && maybe True directionalityGapOk (dsRegimeGap snap)

signalDirectionalityEntryAllowed :: Maybe DirectionalitySnapshot -> Bool
signalDirectionalityEntryAllowed mSnapshot =
    case mSnapshot of
        Nothing -> False
        Just snap ->
            directionalitySnapshotWellFormed snap
                && not (dsNonDirectional snap)

signalEntryHeadroomThresholdCap :: Double -> Double
signalEntryHeadroomThresholdCap edge =
    let edge' =
            if finiteDouble edge && edge > 0
                then edge
                else 0
     in normalizeSignalThreshold (edge' / entryEdgeHeadroomMultiple)

normalizeSignalFeeFloor :: Double -> Maybe Double
normalizeSignalFeeFloor raw
    | finiteDouble raw = Just (max 0 raw)
    | otherwise = Nothing

signalEntryFeeBufferOk :: Double -> Double -> Maybe Double -> Bool
signalEntryFeeBufferOk openThreshold roundTripFeeFloor edgeForMethod
    | finiteDouble openThreshold =
        let requiredHeadroom =
                entryEdgeHeadroomMultiple * normalizeSignalThreshold openThreshold
         in case normalizeSignalFeeFloor roundTripFeeFloor of
                Nothing -> False
                Just feeFloor ->
                    let requiredEdge = requiredHeadroom + feeFloor
                     in maybe False (\edge -> finiteDouble edge && edge >= requiredEdge) edgeForMethod
    | otherwise = False

signalEntryHeadroomOk :: Double -> Maybe Double -> Bool
signalEntryHeadroomOk openThreshold =
    signalEntryFeeBufferOk openThreshold 0

signalDirectionalitySnapshot :: Double -> Maybe RegimeProbs -> V.Vector Double -> Int -> Maybe DirectionalitySnapshot
signalDirectionalitySnapshot regimeHysteresis mRegimes pricesV idx
    | idx < 0 || idx >= V.length pricesV = Nothing
    | windowLen < 3 = Just malformedSnapshot
    | any badPrice window = Just malformedSnapshot
    | null returns = Just malformedSnapshot
    | otherwise =
        case builtSnapshot of
            Just snapshot -> Just snapshot
            Nothing -> Just malformedSnapshot
  where
    start = max 0 (idx - directionalityLookbackBars + 1)
    windowLen = idx - start + 1
    window = V.toList (V.slice start windowLen pricesV)
    returns =
        [ cur / prev - 1
        | (prev, cur) <- zip window (drop 1 window)
        , prev > 0
        ]
    badPrice px = px <= 0 || isNaN px || isInfinite px
    malformedSnapshot = mkMalformedDirectionalitySnapshot windowLen
    builtSnapshot =
        let netReturnPct =
                case (head window, last window) of
                    (p0, p1)
                        | p0 > 0 -> p1 / p0 - 1
                    _ -> 0
            netDirectional = sum returns
            path = sum (map abs returns)
            efficiency =
                if path <= 1e-12
                    then 0
                    else min 1 (abs netDirectional / path)
            meanRet = sum returns / fromIntegral (length returns)
            variance =
                if length returns < 2
                    then 0
                    else
                        let denom = fromIntegral (length returns - 1)
                         in sum (map (\ret -> (ret - meanRet) * (ret - meanRet)) returns) / denom
            realizedVol = sqrt (max 0 variance)
            zScore =
                if realizedVol <= 1e-12
                    then 0
                    else netDirectional / (realizedVol * sqrt (fromIntegral (length returns)))
            metricsOk =
                finiteDouble netReturnPct
                    && finiteDouble netDirectional
                    && finiteDouble path
                    && directionalityEfficiencyOk efficiency
                    && finiteDouble realizedVol
                    && realizedVol >= 0
                    && finiteDouble zScore
         in if not metricsOk
                then Nothing
                else
                    let label
                            | realizedVol * 100 >= directionalityHighVolVolPct = "high-vol"
                            | efficiency >= directionalityTrendEfficiencyMin && abs zScore >= directionalityTrendZMin =
                                if netDirectional >= 0
                                    then "trend-up"
                                    else "trend-down"
                            | efficiency <= directionalityChopEfficiencyMax = "chop"
                            | otherwise = "range-drift"
                        (trendProb, mrProb, highVolProb, regimeLeader, regimeGap, mrDominant, regimeMalformed) =
                            signalDirectionalityRegimeEvidence regimeHysteresis mRegimes
                        mReason
                            | efficiency <= directionalityChopEfficiencyMax = Just "NON_DIRECTIONAL_CHOP"
                            | efficiency <= directionalityMrEfficiencyMax =
                                case mrDominant of
                                    Just True -> Just "NON_DIRECTIONAL_MR"
                                    Just False -> Nothing
                                    Nothing -> Just directionalityMalformedReason
                            | regimeMalformed = Just directionalityMalformedReason
                            | otherwise = Nothing
                     in Just
                            DirectionalitySnapshot
                                { dsLookbackBars = windowLen
                                , dsNetReturnPct = netReturnPct * 100
                                , dsRealizedVolPct = realizedVol * 100
                                , dsEfficiency = efficiency
                                , dsZScore = zScore
                                , dsLabel = label
                                , dsTrendProb = trendProb
                                , dsMrProb = mrProb
                                , dsHighVolProb = highVolProb
                                , dsRegimeLeader = regimeLeader
                                , dsRegimeGap = regimeGap
                                , dsNonDirectional = isJust mReason
                                , dsReason = mReason
                                }

signalDirectionalityRegimeEvidence ::
    Double ->
    Maybe RegimeProbs ->
    (Maybe Double, Maybe Double, Maybe Double, Maybe String, Maybe Double, Maybe Bool, Bool)
signalDirectionalityRegimeEvidence regimeHysteresis mRegimes =
    case mRegimes of
        Nothing -> (Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, False)
        Just regimes ->
            let trendProb = rpTrend regimes
                mrProb = rpMR regimes
                highVolProb = rpHighVol regimes
             in if all directionalityProbOk [trendProb, mrProb, highVolProb]
                    then
                        let ranked =
                                sortOn
                                    (Data.Ord.Down . snd)
                                    [ ("trend", trendProb)
                                    , ("mr", mrProb)
                                    , ("highVol", highVolProb)
                                    ]
                            gap =
                                case ranked of
                                    ((_, p1) : (_, p2) : _) -> Just (p1 - p2)
                                    _ -> Nothing
                            leader =
                                case ranked of
                                    ((name, _) : _) -> Just name
                                    _ -> Nothing
                            dominant =
                                case (leader, gap) of
                                    (Just "mr", Just g) | directionalityGapOk g -> Just (g >= max 0 regimeHysteresis)
                                    (Just _, Just g) | directionalityGapOk g -> Just False
                                    _ -> Nothing
                         in (Just trendProb, Just mrProb, Just highVolProb, leader, gap, dominant, False)
                    else (Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, True)

signalEntryEdgeSpikeOk :: Double -> Maybe Double -> Bool
signalEntryEdgeSpikeOk openThreshold edgeForMethod =
    let openThreshold' = normalizeSignalThreshold openThreshold
     in case edgeForMethod of
            Just edge ->
                finiteDouble edge
                    && edge >= 0
                    && edge <= maxCredibleSignalEdge
                    && (openThreshold' <= 0 || edge <= entryEdgeSpikeLimit * openThreshold')
            Nothing -> False

signalMetaLabelOk ::
    Bool ->
    Double ->
    Maybe Double ->
    Double ->
    Maybe Double ->
    Bool ->
    Bool ->
    Bool
signalMetaLabelOk enabled minEdge edgeForMethod minConfidence methodConfidence requireBand bandAgree =
    not enabled
        || let edgeOk = minEdge <= 0 || maybe False (>= minEdge) edgeForMethod
               confidenceOk = minConfidence <= 0 || maybe False (>= minConfidence) methodConfidence
               bandOk = not requireBand || bandAgree
            in edgeOk && confidenceOk && bandOk

signalMtfConsensusCheck :: Bool -> [Maybe Int] -> Int -> Int -> (Bool, Maybe String)
signalMtfConsensusCheck enabled mtfDirs minAgree dir =
    if not enabled
        then (True, Nothing)
        else
            let available = catMaybes mtfDirs
                agree = length (filter (== dir) available)
             in if length available < minAgree
                    then (False, Just "MTF_WARMUP")
                    else
                        if agree >= minAgree
                            then (True, Nothing)
                            else (False, Just "MTF_CONSENSUS")

signalCrossAssetCheck :: Bool -> Maybe Int -> Int -> (Bool, Maybe String)
signalCrossAssetCheck enabled crossAssetDir dir =
    if not enabled
        then (True, Nothing)
        else case crossAssetDir of
            Nothing -> (False, Just "CROSS_ASSET")
            Just d ->
                if d == dir
                    then (True, Nothing)
                    else (False, Just "CROSS_ASSET")

signalRegimeEdgeOk :: Bool -> Double -> Maybe Double -> Bool
signalRegimeEdgeOk enabled minEdgeRegime edgeForMethod =
    not enabled
        || minEdgeRegime <= 0
        || maybe False (>= minEdgeRegime) edgeForMethod

signalFundingOiCheck ::
    Bool ->
    Maybe Double ->
    Maybe Double ->
    Double ->
    Double ->
    Maybe Double ->
    (Bool, Double)
signalFundingOiCheck enabled fundingCap volCap sizeMult funding oiVolProxy =
    if not enabled
        then (True, 1.0)
        else
            let clamp01 x = max 0 (min 1 x)
                sizeFloor =
                    if finiteDouble sizeMult
                        then clamp01 sizeMult
                        else 0
                cleanCap mCap =
                    case mCap of
                        Just cap | finiteDouble cap && cap > 0 -> Just cap
                        _ -> Nothing
                fundingCap' = cleanCap fundingCap
                volCap' = cleanCap volCap
                fundingFinite = finiteDouble funding
                oiVolProxyFinite =
                    case oiVolProxy of
                        Just v | finiteDouble v -> Just v
                        _ -> Nothing
                fundingOk =
                    case fundingCap' of
                        Nothing -> True
                        Just cap -> fundingFinite && funding <= cap
                volProxyOk =
                    case volCap' of
                        Nothing -> True
                        Just cap ->
                            case oiVolProxyFinite of
                                Nothing -> False
                                Just v -> v <= cap
                fundingPenalty =
                    case fundingCap' of
                        Just cap
                            | cap > 0 && fundingFinite ->
                                max 0 ((funding - cap) / cap)
                        _ -> 0
                volPenalty =
                    case (volCap', oiVolProxyFinite) of
                        (Just cap, Just v)
                            | cap > 0 ->
                                max 0 ((v - cap) / cap)
                        _ -> 0
                dampRaw = 1 / (1 + fundingPenalty + volPenalty)
                damp = max sizeFloor (min 1 dampRaw)
             in (fundingOk && volProxyOk, damp)

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
    Bool ->
    (Int -> (Bool, Maybe String)) ->
    (Int -> (Bool, Maybe String)) ->
    (Int -> Bool) ->
    (Int -> (Bool, Double)) ->
    (Maybe Int, Maybe String)
signalRunPostDirectionGates chosenDir mPairsOverlayReason volOk volTargetReady trendOk cloudOk priceActionOk signalToNoiseOk nonDirectionalCheck regimeEdgeOk mtfConsensusCheck crossAssetCheck metaLabelOk fundingOiCheck =
    case chosenDir of
        Nothing -> (Nothing, mPairsOverlayReason)
        Just dir ->
            if not volOk
                then (Nothing, Just "MAX_VOLATILITY")
                else
                    if not volTargetReady
                        then (Nothing, Just "VOL_TARGET_WARMUP")
                        else
                            if not (trendOk dir)
                                then (Nothing, Just "TREND_FILTER")
                                else
                                    if not (cloudOk dir)
                                        then (Nothing, Just "KALMAN_CLOUD")
                                        else
                                            if not (priceActionOk dir)
                                                then (Nothing, Just "PRICE_ACTION")
                                                else
                                                    if not signalToNoiseOk
                                                        then (Nothing, Just "SIGNAL_TO_NOISE")
                                                        else
                                                            let (nonDirectionalOk, mNonDirectionalReason) = nonDirectionalCheck dir
                                                             in if not nonDirectionalOk
                                                                    then (Nothing, mNonDirectionalReason <|> Just "NON_DIRECTIONAL")
                                                                    else
                                                                        if not regimeEdgeOk
                                                                            then (Nothing, Just "REGIME_BANK")
                                                                            else
                                                                                let (mtfOk, mMtfReason) = mtfConsensusCheck dir
                                                                                 in if not mtfOk
                                                                                        then (Nothing, mMtfReason)
                                                                                        else
                                                                                            let (crossOk, mCrossReason) = crossAssetCheck dir
                                                                                             in if not crossOk
                                                                                                    then (Nothing, mCrossReason)
                                                                                                    else
                                                                                                        if not (metaLabelOk dir)
                                                                                                            then (Nothing, Just "META_LABEL")
                                                                                                            else
                                                                                                                let (fundingOiOk, _) = fundingOiCheck dir
                                                                                                                 in if not fundingOiOk
                                                                                                                        then (Nothing, Just "FUNDING_OI")
                                                                                                                        else (Just dir, Nothing)

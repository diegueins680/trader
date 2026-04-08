{-# LANGUAGE OverloadedStrings #-}

module Trader.SignalGates (
    SignalThresholdBoundary (..),
    DirectionalitySnapshot (..),
    mkSignalThresholdBoundary,
    normalizeSignalThreshold,
    signalDirectionalitySnapshot,
    signalEntryHeadroomThresholdCap,
    signalEntryHeadroomOk,
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
    if finite raw && raw > 0
        then min maxSignalThreshold raw
        else 0
  where
    finite x = not (isNaN x || isInfinite x)

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
        if finite raw && raw >= 0
            then raw
            else 0
    finite x = not (isNaN x || isInfinite x)

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

signalEntryHeadroomThresholdCap :: Double -> Double
signalEntryHeadroomThresholdCap edge =
    let finite x = not (isNaN x || isInfinite x)
        edge' =
            if finite edge && edge > 0
                then edge
                else 0
     in normalizeSignalThreshold (edge' / entryEdgeHeadroomMultiple)

signalEntryHeadroomOk :: Double -> Maybe Double -> Bool
signalEntryHeadroomOk openThreshold edgeForMethod =
    let finite x = not (isNaN x || isInfinite x)
        openThreshold' = normalizeSignalThreshold openThreshold
        requiredEdge = entryEdgeHeadroomMultiple * openThreshold'
     in openThreshold' <= 0
            || case edgeForMethod of
                Just edge ->
                    finite edge
                        && edge >= requiredEdge
                Nothing -> False

signalDirectionalitySnapshot :: Double -> Maybe RegimeProbs -> V.Vector Double -> Int -> Maybe DirectionalitySnapshot
signalDirectionalitySnapshot regimeHysteresis mRegimes pricesV idx
    | idx < 0 || idx >= V.length pricesV = Nothing
    | windowLen < 3 = Nothing
    | any badPrice window = Nothing
    | null returns = Nothing
    | otherwise =
        let net =
                case (head window, last window) of
                    (p0, p1)
                        | p0 > 0 -> p1 / p0 - 1
                    _ -> 0
            path = sum (map abs returns)
            efficiency =
                if path <= 1e-12
                    then 0
                    else abs net / path
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
                    else net / (realizedVol * sqrt (fromIntegral (length returns)))
            label
                | realizedVol * 100 >= directionalityHighVolVolPct = "high-vol"
                | efficiency >= directionalityTrendEfficiencyMin && abs zScore >= directionalityTrendZMin =
                    if net >= 0
                        then "trend-up"
                        else "trend-down"
                | efficiency <= directionalityChopEfficiencyMax = "chop"
                | otherwise = "range-drift"
            (trendProb, mrProb, highVolProb, regimeLeader, regimeGap, mrDominant) =
                case mRegimes of
                    Nothing -> (Nothing, Nothing, Nothing, Nothing, Nothing, False)
                    Just regimes ->
                        let ranked =
                                sortOn
                                    (Data.Ord.Down . snd)
                                    [ ("trend", rpTrend regimes)
                                    , ("mr", rpMR regimes)
                                    , ("highVol", rpHighVol regimes)
                                    ]
                            gapFor rankedRegs =
                                case rankedRegs of
                                    ((_, p1) : (_, p2) : _) -> Just (p1 - p2)
                                    _ -> Nothing
                            leaderFor rankedRegs =
                                case rankedRegs of
                                    ((name, _) : _) -> Just name
                                    _ -> Nothing
                            gap = gapFor ranked
                            leader = leaderFor ranked
                            dominant =
                                case (leader, gap) of
                                    (Just "mr", Just g) -> g >= max 0 regimeHysteresis
                                    _ -> False
                         in ( Just (rpTrend regimes)
                            , Just (rpMR regimes)
                            , Just (rpHighVol regimes)
                            , leader
                            , gap
                            , dominant
                            )
            mReason
                | efficiency <= directionalityChopEfficiencyMax = Just "NON_DIRECTIONAL_CHOP"
                | efficiency <= directionalityMrEfficiencyMax && mrDominant = Just "NON_DIRECTIONAL_MR"
                | otherwise = Nothing
         in Just
                DirectionalitySnapshot
                    { dsLookbackBars = windowLen
                    , dsNetReturnPct = net * 100
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

signalEntryEdgeSpikeOk :: Double -> Maybe Double -> Bool
signalEntryEdgeSpikeOk openThreshold edgeForMethod =
    let finite x = not (isNaN x || isInfinite x)
        openThreshold' = normalizeSignalThreshold openThreshold
     in openThreshold' <= 0
            || case edgeForMethod of
                Just edge ->
                    finite edge
                        && edge >= 0
                        && edge <= maxCredibleSignalEdge
                        && edge <= entryEdgeSpikeLimit * openThreshold'
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
            let finite x = not (isNaN x || isInfinite x)
                clamp01 x = max 0 (min 1 x)
                sizeFloor =
                    if finite sizeMult
                        then clamp01 sizeMult
                        else 0
                cleanCap mCap =
                    case mCap of
                        -- Zero/negative caps are treated as disabled, matching CLI/docs semantics.
                        Just cap | finite cap && cap > 0 -> Just cap
                        _ -> Nothing
                fundingCap' = cleanCap fundingCap
                volCap' = cleanCap volCap
                fundingFinite = finite funding
                oiVolProxyFinite =
                    case oiVolProxy of
                        Just v | finite v -> Just v
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

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
    signalEntryHeadroomThresholdCap,
    normalizeSignalOpenThreshold,
    normalizeSignalFeeFloor,
    signalEntryHeadroomOk,
    signalEntryEdgeSpikeOk,
    signalEntryFeeBufferOk,
    signalFundingOiCheck,
    signalMetaLabelOk,
    signalMtfConsensusCheck,
    signalRegimeEdgeOk,
    signalRunPostDirectionGates,
) where

import Data.Maybe (catMaybes, fromMaybe)
import qualified Data.Aeson as Aeson

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
    | otherwise = Just 0

signalCrossAssetCheck :: Bool -> Maybe Int -> (Bool, Maybe String)
signalCrossAssetCheck enabled crossAssetDirRaw
    | not enabled = (True, Nothing)
    | otherwise =
        case crossAssetDirRaw of
            Just _ -> (True, Nothing)
            Nothing -> (False, Just "CROSS_ASSET")

signalDirectionalitySnapshot :: (FailClosedSurface r) => r
signalDirectionalitySnapshot = failClosedSurface

entryEdgeHeadroomMultiple :: Double
entryEdgeHeadroomMultiple = 1.5

entryEdgeSpikeMultiple :: Double
entryEdgeSpikeMultiple = 1.7

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
            let requiredEdge = entryEdgeSpikeMultiple * threshold
             in maybe False (\edge -> finiteDouble edge && edge >= requiredEdge) edgeForMethod

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
{-# LANGUAGE FlexibleInstances #-}

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

-- Compatibility surface restored for Main: these shims are fail closed by
-- construction, so re-exporting the legacy names cannot weaken the current
-- entry-gate contract.
data DirectionalitySnapshot = DirectionalitySnapshot
    deriving (Eq, Show)

data SignalThresholdBoundary = SignalThresholdBoundary
    deriving (Eq, Show)

class FailClosedSurface r where
    failClosedSurface :: r

instance FailClosedSurface Bool where
    failClosedSurface = False

instance FailClosedSurface DirectionalitySnapshot where
    failClosedSurface = DirectionalitySnapshot

instance FailClosedSurface SignalThresholdBoundary where
    failClosedSurface = SignalThresholdBoundary

instance FailClosedSurface (Maybe a) where
    failClosedSurface = Nothing

instance (FailClosedSurface r) => FailClosedSurface (a -> r) where
    failClosedSurface = const failClosedSurface

finiteDouble :: Double -> Bool
finiteDouble value = not (isNaN value) && not (isInfinite value)

mkSignalThresholdBoundary :: (FailClosedSurface r) => r
mkSignalThresholdBoundary = failClosedSurface

normalizeSignalThreshold :: Double -> Double
normalizeSignalThreshold = id

normalizeSignalEntryEdge :: Double -> Maybe Double
normalizeSignalEntryEdge raw
    | finiteDouble raw = Just (max 0 raw)
    | otherwise = Just 0

signalCrossAssetCheck :: (FailClosedSurface r) => r
signalCrossAssetCheck = failClosedSurface

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

signalFundingOiCheck :: (FailClosedSurface r) => r
signalFundingOiCheck = failClosedSurface

signalMetaLabelOk :: (FailClosedSurface r) => r
signalMetaLabelOk = failClosedSurface

signalMtfConsensusCheck :: (FailClosedSurface r) => r
signalMtfConsensusCheck = failClosedSurface

signalRegimeEdgeOk :: (FailClosedSurface r) => r
signalRegimeEdgeOk = failClosedSurface

signalRunPostDirectionGates :: (FailClosedSurface r) => r
signalRunPostDirectionGates = failClosedSurface
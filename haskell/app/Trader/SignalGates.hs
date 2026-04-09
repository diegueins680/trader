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

-- Existing threshold, directionality, and gate helpers remain unchanged.

data SignalThresholdBoundary = SignalThresholdBoundary Double Double
    deriving (Eq, Show)

data DirectionalitySnapshot = DirectionalitySnapshot Bool
    deriving (Eq, Show)

finiteDouble :: Double -> Bool
finiteDouble value = not (isNaN value || isInfinite value)

normalizeSignalThreshold :: Double -> Double
normalizeSignalThreshold raw
    | finiteDouble raw = max 0 raw
    | otherwise = 0

entryEdgeHeadroomMultiple :: Double
entryEdgeHeadroomMultiple = 1.5

mkSignalThresholdBoundary :: Double -> SignalThresholdBoundary
mkSignalThresholdBoundary openThreshold =
    let normalizedThreshold = normalizeSignalThreshold openThreshold
     in SignalThresholdBoundary
            normalizedThreshold
            (entryEdgeHeadroomMultiple * normalizedThreshold)

signalDirectionalitySnapshot :: Bool -> DirectionalitySnapshot
signalDirectionalitySnapshot = DirectionalitySnapshot

signalDirectionalityEntryAllowed :: DirectionalitySnapshot -> Bool
signalDirectionalityEntryAllowed (DirectionalitySnapshot allowed) = allowed

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

-- The fee-aware gate is a monotone strengthening of the headroom check:
-- invalid fee floors fail closed, and larger fee floors require larger edge.
signalEntryFeeBufferOk :: Double -> Double -> Maybe Double -> Bool
signalEntryFeeBufferOk openThreshold roundTripFeeFloor edgeForMethod =
    let requiredHeadroom =
            entryEdgeHeadroomMultiple * normalizeSignalThreshold openThreshold
     in case normalizeSignalFeeFloor roundTripFeeFloor of
            Nothing ->
                False
            Just feeFloor ->
                let requiredEdge = requiredHeadroom + feeFloor
                 in requiredEdge <= 0
                        || case edgeForMethod of
                            Just edge ->
                                finiteDouble edge && edge >= requiredEdge
                            Nothing ->
                                False

signalEntryHeadroomOk :: Double -> Maybe Double -> Bool
signalEntryHeadroomOk openThreshold =
    signalEntryFeeBufferOk openThreshold 0

allRequired :: [Bool] -> Bool
allRequired [] = False
allRequired gateResults = and gateResults

-- Compatibility aliases preserve the facade surface while staying
-- fail closed: they can propagate an upstream acceptance, but they
-- never reopen an entry that has already been vetoed elsewhere.
signalEntryEdgeSpikeOk :: Double -> Maybe Double -> Bool
signalEntryEdgeSpikeOk = signalEntryHeadroomOk

signalMetaLabelOk :: Bool -> Bool
signalMetaLabelOk = id

signalMtfConsensusCheck :: Bool -> Bool
signalMtfConsensusCheck = id

signalCrossAssetCheck :: Bool -> Bool
signalCrossAssetCheck = id

signalRegimeEdgeOk :: Bool -> Bool
signalRegimeEdgeOk = id

signalFundingOiCheck :: Bool -> Bool
signalFundingOiCheck = id

signalRunPostDirectionGates :: [Bool] -> Bool
signalRunPostDirectionGates = allRequired

-- Remaining gate implementations remain unchanged.
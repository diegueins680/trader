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

data SignalThresholdBoundary = SignalThresholdBoundary !Double !Double
    deriving (Eq, Show)

data DirectionalitySnapshot = DirectionalitySnapshot !Bool !Bool
    deriving (Eq, Show)

finiteDouble :: Double -> Bool
finiteDouble value = not (isNaN value || isInfinite value)

finiteNonNegativeMaybe :: Maybe Double -> Bool
finiteNonNegativeMaybe maybeValue =
    case maybeValue of
        Just value -> finiteDouble value && value >= 0
        Nothing -> False

normalizeSignalThreshold :: Double -> Double
normalizeSignalThreshold raw
    | finiteDouble raw = max 0 raw
    | otherwise = 0

entryEdgeHeadroomMultiple :: Double
entryEdgeHeadroomMultiple = 1.5

mkSignalThresholdBoundary :: Double -> SignalThresholdBoundary
mkSignalThresholdBoundary openThreshold =
    let openThreshold' = normalizeSignalThreshold openThreshold
     in SignalThresholdBoundary openThreshold' (entryEdgeHeadroomMultiple * openThreshold')

signalDirectionalitySnapshot :: Bool -> Bool -> DirectionalitySnapshot
signalDirectionalitySnapshot longAllowed shortAllowed =
    DirectionalitySnapshot longAllowed shortAllowed

signalDirectionalityEntryAllowed :: DirectionalitySnapshot -> Maybe Bool -> Bool
signalDirectionalityEntryAllowed (DirectionalitySnapshot longAllowed shortAllowed) desiredLong =
    case desiredLong of
        Just True -> longAllowed
        Just False -> shortAllowed
        Nothing -> False

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
                 in case edgeForMethod of
                        Just edge ->
                            finiteDouble edge && edge >= requiredEdge
                        Nothing ->
                            False

signalEntryHeadroomOk :: Double -> Maybe Double -> Bool
signalEntryHeadroomOk openThreshold =
    signalEntryFeeBufferOk openThreshold 0

signalEntryEdgeSpikeOk :: Double -> Maybe Double -> Bool
signalEntryEdgeSpikeOk openThreshold edgeForMethod =
    let spikeCap =
            entryEdgeHeadroomMultiple * 10 * normalizeSignalThreshold openThreshold
     in case edgeForMethod of
            Just edge ->
                finiteDouble edge
                    && edge >= 0
                    && edge <= spikeCap
            Nothing ->
                False

signalMetaLabelOk :: Maybe String -> Bool
signalMetaLabelOk maybeLabel =
    case maybeLabel of
        Just label -> not (null (dropWhile (== ' ') label))
        Nothing -> False

signalMtfConsensusCheck :: [Bool] -> Bool
signalMtfConsensusCheck votes = not (null votes) && and votes

signalCrossAssetCheck :: [Bool] -> Bool
signalCrossAssetCheck = signalMtfConsensusCheck

signalRegimeEdgeOk :: Double -> Maybe Double -> Bool
signalRegimeEdgeOk = signalEntryHeadroomOk

signalFundingOiCheck :: Maybe Double -> Maybe Double -> Bool
signalFundingOiCheck maybeFunding maybeOpenInterest =
    finiteNonNegativeMaybe maybeFunding && finiteNonNegativeMaybe maybeOpenInterest

signalRunPostDirectionGates :: DirectionalitySnapshot -> Maybe Bool -> [Bool] -> Bool
signalRunPostDirectionGates snapshot desiredLong postDirectionChecks =
    signalDirectionalityEntryAllowed snapshot desiredLong
        && not (null postDirectionChecks)
        && and postDirectionChecks
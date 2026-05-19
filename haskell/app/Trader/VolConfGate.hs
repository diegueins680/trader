module Trader.VolConfGate (
    VolConfGatePreset (..),
    VolConfGateBehavior (..),
    VolConfGateCell (..),
    volConfGateCode,
    parseVolConfGatePreset,
    volConfGateChoices,
    volConfGateChoicesCsv,
    volConfGateCell,
    applyVolConfGateBehavior,
    volConfStatefulCloseDirection,
) where

import Data.List (intercalate)

data VolConfGatePreset
    = VolConfGateDisabled
    | VolConfGateV1Default
    | VolConfGateV1HighVolTighter
    | VolConfGateV1HighVolLooser
    | VolConfGateV1ConfStricter
    deriving (Eq, Show)

data VolConfGateBehavior
    = VolConfGateBlock
    | VolConfGateHold
    | VolConfGateAllowEntry
    | VolConfGateAllowExitOnly
    deriving (Eq, Show)

data VolConfGateCell = VolConfGateCell
    { vcgBehavior :: !VolConfGateBehavior
    , vcgSizeMult :: !Double
    }
    deriving (Eq, Show)

volConfGateCode :: VolConfGatePreset -> String
volConfGateCode preset =
    case preset of
        VolConfGateDisabled -> "disabled"
        VolConfGateV1Default -> "vol_conf_v1_default"
        VolConfGateV1HighVolTighter -> "vol_conf_v1_high_vol_tighter"
        VolConfGateV1HighVolLooser -> "vol_conf_v1_high_vol_looser"
        VolConfGateV1ConfStricter -> "vol_conf_v1_conf_stricter"

parseVolConfGatePreset :: String -> Either String VolConfGatePreset
parseVolConfGatePreset raw =
    case raw of
        "disabled" -> Right VolConfGateDisabled
        "vol_conf_v1_default" -> Right VolConfGateV1Default
        "vol_conf_v1_high_vol_tighter" -> Right VolConfGateV1HighVolTighter
        "vol_conf_v1_high_vol_looser" -> Right VolConfGateV1HighVolLooser
        "vol_conf_v1_conf_stricter" -> Right VolConfGateV1ConfStricter
        _ ->
            Left
                ( "Unknown --vol-conf-gate preset '"
                    ++ raw
                    ++ "'. Expected one of: "
                    ++ volConfGateChoicesCsv
                )

volConfGateChoices :: [String]
volConfGateChoices =
    map
        volConfGateCode
        [ VolConfGateDisabled
        , VolConfGateV1Default
        , VolConfGateV1HighVolTighter
        , VolConfGateV1HighVolLooser
        , VolConfGateV1ConfStricter
        ]

volConfGateChoicesCsv :: String
volConfGateChoicesCsv = intercalate ", " volConfGateChoices

data VolBucket
    = VolLow
    | VolMedium
    | VolHigh
    | VolMissing
    deriving (Eq, Show)

data ConfidenceBucket
    = ConfidenceWeak
    | ConfidenceMedium
    | ConfidenceStrong
    deriving (Eq, Show)

volConfGateCell :: VolConfGatePreset -> Maybe Double -> Maybe Double -> VolConfGateCell
volConfGateCell preset mVolatility mConfidence =
    case preset of
        VolConfGateDisabled ->
            mkVolConfGateCell VolConfGateAllowEntry 1.0
        _ ->
            case volBucket preset mVolatility of
                VolMissing -> malformedVolConfGateCell
                volB ->
                    case confidenceBucket preset mConfidence of
                        Nothing -> malformedVolConfGateCell
                        Just confB -> gateCellFor preset volB confB

applyVolConfGateBehavior ::
    (Eq side) =>
    VolConfGateBehavior ->
    Maybe side ->
    Double ->
    Maybe side ->
    Double ->
    (Maybe side, Double)
applyVolConfGateBehavior behavior currentSide currentSize desiredSide desiredSize =
    let currentSize' = max 0 currentSize
        desiredSize' = max 0 desiredSize
        reduceOnly =
            case currentSide of
                Nothing -> (Nothing, 0)
                Just side ->
                    case desiredSide of
                        Nothing -> (Nothing, 0)
                        Just desiredSide'
                            | desiredSide' == side ->
                                (Just side, min currentSize' desiredSize')
                            | otherwise ->
                                (Nothing, 0)
     in case behavior of
            VolConfGateAllowEntry -> (desiredSide, desiredSize')
            VolConfGateHold ->
                case desiredSide of
                    Nothing -> (Nothing, 0)
                    Just _ ->
                        case currentSide of
                            Just side -> (Just side, currentSize')
                            Nothing -> (Nothing, 0)
            VolConfGateBlock -> reduceOnly
            VolConfGateAllowExitOnly -> reduceOnly

volConfStatefulCloseDirection :: VolConfGateBehavior -> Maybe side -> Maybe side -> Maybe side
volConfStatefulCloseDirection behavior preGateDir closeDirBase =
    case behavior of
        VolConfGateAllowEntry -> Nothing
        VolConfGateHold -> Nothing
        VolConfGateBlock -> firstJust preGateDir closeDirBase
        VolConfGateAllowExitOnly -> firstJust preGateDir closeDirBase
  where
    firstJust (Just x) _ = Just x
    firstJust Nothing y = y

volatilityEvidenceMax :: Double
volatilityEvidenceMax = 2.0

volBucket :: VolConfGatePreset -> Maybe Double -> VolBucket
volBucket preset mVolatility =
    case sanitizeFiniteMaybe mVolatility of
        Just vol
            | vol >= 0 && vol <= volatilityEvidenceMax ->
                let highThreshold =
                        case preset of
                            VolConfGateV1HighVolTighter -> 1.0
                            VolConfGateV1HighVolLooser -> 1.4
                            _ -> 1.2
                 in if vol < 0.5
                        then VolLow
                        else
                            if vol < highThreshold
                                then VolMedium
                                else VolHigh
        _ ->
            -- Missing, negative, non-finite, and out-of-range volatility are
            -- malformed risk inputs, so fail closed instead of classifying
            -- them as low volatility.
            VolMissing

confidenceBucket :: VolConfGatePreset -> Maybe Double -> Maybe ConfidenceBucket
confidenceBucket preset mConfidence =
    let weakThreshold =
            case preset of
                VolConfGateV1ConfStricter -> 0.65
                _ -> 0.60
        strongThreshold = 0.80
        classify confidence
            | confidence < weakThreshold = ConfidenceWeak
            | confidence < strongThreshold = ConfidenceMedium
            | otherwise = ConfidenceStrong
     in case mConfidence of
            Nothing -> Just ConfidenceWeak
            Just _ -> classify <$> sanitizeConfidenceUnitInterval mConfidence

gateCellFor :: VolConfGatePreset -> VolBucket -> ConfidenceBucket -> VolConfGateCell
gateCellFor preset volB confB =
    case (volB, confB) of
        (VolLow, ConfidenceWeak) -> mkVolConfGateCell VolConfGateHold 0.0
        (VolLow, ConfidenceMedium) -> mkVolConfGateCell VolConfGateAllowEntry 0.60
        (VolLow, ConfidenceStrong) -> mkVolConfGateCell VolConfGateAllowEntry 1.00
        (VolMedium, ConfidenceWeak) -> mkVolConfGateCell VolConfGateHold 0.0
        (VolMedium, ConfidenceMedium) -> mkVolConfGateCell VolConfGateAllowEntry 0.45
        (VolMedium, ConfidenceStrong) -> mkVolConfGateCell VolConfGateAllowEntry 0.75
        (VolHigh, ConfidenceWeak) -> mkVolConfGateCell VolConfGateBlock 0.0
        (VolHigh, ConfidenceMedium) -> mkVolConfGateCell VolConfGateAllowExitOnly 0.0
        (VolHigh, ConfidenceStrong) ->
            let highStrongSize =
                    case preset of
                        VolConfGateV1HighVolTighter -> 0.25
                        VolConfGateV1HighVolLooser -> 0.45
                        _ -> 0.35
             in mkVolConfGateCell VolConfGateAllowEntry highStrongSize
        (VolMissing, _) -> malformedVolConfGateCell

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)

sanitizeFiniteMaybe :: Maybe Double -> Maybe Double
sanitizeFiniteMaybe mValue =
    case mValue of
        Just x | isFinite x -> Just x
        _ -> Nothing

sanitizeConfidenceUnitInterval :: Maybe Double -> Maybe Double
sanitizeConfidenceUnitInterval mConfidence =
    case sanitizeFiniteMaybe mConfidence of
        Just confidence
            | confidence >= 0 && confidence <= 1 -> Just confidence
        _ -> Nothing

sanitizeFiniteWith :: Double -> Double -> Double
sanitizeFiniteWith fallback x =
    if isFinite x
        then x
        else fallback

clamp :: Double -> Double -> Double -> Double
clamp lo hi x = max lo (min hi x)

mkVolConfGateCell :: VolConfGateBehavior -> Double -> VolConfGateCell
mkVolConfGateCell behavior sizeMult =
    VolConfGateCell behavior (clamp 0 1 (sanitizeFiniteWith 0 sizeMult))

malformedVolConfGateCell :: VolConfGateCell
malformedVolConfGateCell = mkVolConfGateCell VolConfGateAllowExitOnly 0.0

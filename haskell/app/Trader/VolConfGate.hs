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
            VolConfGateCell VolConfGateAllowEntry 1.0
        _ ->
            case volBucket preset mVolatility of
                VolMissing -> VolConfGateCell VolConfGateAllowExitOnly 0.0
                volB ->
                    let confB = confidenceBucket preset mConfidence
                     in gateCellFor preset volB confB

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
                case currentSide of
                    Just side -> (Just side, currentSize')
                    Nothing -> (Nothing, 0)
            VolConfGateBlock -> reduceOnly
            VolConfGateAllowExitOnly -> reduceOnly

volBucket :: VolConfGatePreset -> Maybe Double -> VolBucket
volBucket preset mVolatility =
    case mVolatility of
        Nothing -> VolMissing
        Just rawVol ->
            let vol = max 0 rawVol
                highThreshold =
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

confidenceBucket :: VolConfGatePreset -> Maybe Double -> ConfidenceBucket
confidenceBucket preset mConfidence =
    let weakThreshold =
            case preset of
                VolConfGateV1ConfStricter -> 0.65
                _ -> 0.60
        strongThreshold = 0.80
        confidence =
            case mConfidence of
                -- Missing confidence is treated as weak to keep the gate conservative.
                Nothing -> 0.0
                Just rawConfidence -> max 0 (min 1 rawConfidence)
     in if confidence < weakThreshold
            then ConfidenceWeak
            else
                if confidence < strongThreshold
                    then ConfidenceMedium
                    else ConfidenceStrong

gateCellFor :: VolConfGatePreset -> VolBucket -> ConfidenceBucket -> VolConfGateCell
gateCellFor preset volB confB =
    case (volB, confB) of
        (VolLow, ConfidenceWeak) -> VolConfGateCell VolConfGateHold 0.0
        (VolLow, ConfidenceMedium) -> VolConfGateCell VolConfGateAllowEntry 0.60
        (VolLow, ConfidenceStrong) -> VolConfGateCell VolConfGateAllowEntry 1.00
        (VolMedium, ConfidenceWeak) -> VolConfGateCell VolConfGateHold 0.0
        (VolMedium, ConfidenceMedium) -> VolConfGateCell VolConfGateAllowEntry 0.45
        (VolMedium, ConfidenceStrong) -> VolConfGateCell VolConfGateAllowEntry 0.75
        (VolHigh, ConfidenceWeak) -> VolConfGateCell VolConfGateBlock 0.0
        (VolHigh, ConfidenceMedium) -> VolConfGateCell VolConfGateAllowExitOnly 0.0
        (VolHigh, ConfidenceStrong) ->
            let highStrongSize =
                    case preset of
                        VolConfGateV1HighVolTighter -> 0.25
                        VolConfGateV1HighVolLooser -> 0.45
                        _ -> 0.35
             in VolConfGateCell VolConfGateAllowEntry highStrongSize
        (VolMissing, _) -> VolConfGateCell VolConfGateAllowExitOnly 0.0

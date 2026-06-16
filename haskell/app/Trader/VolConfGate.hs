module Trader.VolConfGate (
    VolConfGatePreset (..),
    VolConfGateBehavior (..),
    VolConfGateCell (..),
    VolConfGateConfig (..),
    defaultVolConfGateConfig,
    volConfGateCode,
    parseVolConfGatePreset,
    volConfGateChoices,
    volConfGateChoicesCsv,
    volConfGateConfigForPreset,
    volConfGateCell,
    volConfGateCellWithConfig,
    sanitizeVolConfGateConfig,
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

data VolConfGateConfig = VolConfGateConfig
    { vcgcVolatilityEvidenceMax :: !Double
    , vcgcLowVolThreshold :: !Double
    , vcgcHighVolThreshold :: !Double
    , vcgcWeakConfidenceThreshold :: !Double
    , vcgcStrongConfidenceThreshold :: !Double
    , vcgcLowMediumSizeMult :: !Double
    , vcgcLowStrongSizeMult :: !Double
    , vcgcMediumMediumSizeMult :: !Double
    , vcgcMediumStrongSizeMult :: !Double
    , vcgcHighStrongSizeMult :: !Double
    }
    deriving (Eq, Show)

defaultVolConfGateConfig :: VolConfGateConfig
defaultVolConfGateConfig = volConfGateConfigForPreset VolConfGateV1Default

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

volConfGateConfigForPreset :: VolConfGatePreset -> VolConfGateConfig
volConfGateConfigForPreset preset =
    sanitizeVolConfGateConfig
        VolConfGateConfig
            { vcgcVolatilityEvidenceMax = 2.0
            , vcgcLowVolThreshold = 0.5
            , vcgcHighVolThreshold =
                case preset of
                    VolConfGateV1HighVolTighter -> 1.0
                    VolConfGateV1HighVolLooser -> 1.4
                    _ -> 1.2
            , vcgcWeakConfidenceThreshold =
                case preset of
                    VolConfGateV1ConfStricter -> 0.65
                    _ -> 0.60
            , vcgcStrongConfidenceThreshold = 0.80
            , vcgcLowMediumSizeMult = 0.60
            , vcgcLowStrongSizeMult = 1.00
            , vcgcMediumMediumSizeMult = 0.45
            , vcgcMediumStrongSizeMult = 0.75
            , vcgcHighStrongSizeMult =
                case preset of
                    VolConfGateV1HighVolTighter -> 0.25
                    VolConfGateV1HighVolLooser -> 0.45
                    _ -> 0.35
            }

sanitizeVolConfGateConfig :: VolConfGateConfig -> VolConfGateConfig
sanitizeVolConfGateConfig cfg =
    let evidenceMax = max 1e-12 (sanitizeFiniteWith 2.0 (vcgcVolatilityEvidenceMax cfg))
        lowVol = clamp 0 evidenceMax (sanitizeFiniteWith 0.5 (vcgcLowVolThreshold cfg))
        highVol = clamp lowVol evidenceMax (sanitizeFiniteWith 1.2 (vcgcHighVolThreshold cfg))
        weakConf = clamp 0 1 (sanitizeFiniteWith 0.60 (vcgcWeakConfidenceThreshold cfg))
        strongConf = clamp weakConf 1 (sanitizeFiniteWith 0.80 (vcgcStrongConfidenceThreshold cfg))
        size = clamp 0 1 . sanitizeFiniteWith 0
     in cfg
            { vcgcVolatilityEvidenceMax = evidenceMax
            , vcgcLowVolThreshold = lowVol
            , vcgcHighVolThreshold = highVol
            , vcgcWeakConfidenceThreshold = weakConf
            , vcgcStrongConfidenceThreshold = strongConf
            , vcgcLowMediumSizeMult = size (vcgcLowMediumSizeMult cfg)
            , vcgcLowStrongSizeMult = size (vcgcLowStrongSizeMult cfg)
            , vcgcMediumMediumSizeMult = size (vcgcMediumMediumSizeMult cfg)
            , vcgcMediumStrongSizeMult = size (vcgcMediumStrongSizeMult cfg)
            , vcgcHighStrongSizeMult = size (vcgcHighStrongSizeMult cfg)
            }

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
volConfGateCell preset =
    volConfGateCellWithConfig (volConfGateConfigForPreset preset) preset

volConfGateCellWithConfig :: VolConfGateConfig -> VolConfGatePreset -> Maybe Double -> Maybe Double -> VolConfGateCell
volConfGateCellWithConfig cfg0 preset mVolatility mConfidence =
    case preset of
        VolConfGateDisabled ->
            mkVolConfGateCell VolConfGateAllowEntry 1.0
        _ ->
            let cfg = sanitizeVolConfGateConfig cfg0
             in case volBucket cfg mVolatility of
                    VolMissing -> malformedVolConfGateCell
                    volB ->
                        case confidenceBucket cfg mConfidence of
                            Nothing -> malformedVolConfGateCell
                            Just confB -> gateCellFor cfg volB confB

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

volBucket :: VolConfGateConfig -> Maybe Double -> VolBucket
volBucket cfg mVolatility =
    case sanitizeFiniteMaybe mVolatility of
        Just vol
            | vol >= 0 && vol <= vcgcVolatilityEvidenceMax cfg ->
                if vol < vcgcLowVolThreshold cfg
                    then VolLow
                    else
                        if vol < vcgcHighVolThreshold cfg
                            then VolMedium
                            else VolHigh
        _ ->
            -- Missing, negative, non-finite, and out-of-range volatility are
            -- malformed risk inputs, so fail closed instead of classifying
            -- them as low volatility.
            VolMissing

confidenceBucket :: VolConfGateConfig -> Maybe Double -> Maybe ConfidenceBucket
confidenceBucket cfg mConfidence =
    let classify confidence
            | confidence < vcgcWeakConfidenceThreshold cfg = ConfidenceWeak
            | confidence < vcgcStrongConfidenceThreshold cfg = ConfidenceMedium
            | otherwise = ConfidenceStrong
     in case mConfidence of
            Nothing -> Just ConfidenceWeak
            Just _ -> classify <$> sanitizeConfidenceUnitInterval mConfidence

gateCellFor :: VolConfGateConfig -> VolBucket -> ConfidenceBucket -> VolConfGateCell
gateCellFor cfg volB confB =
    case (volB, confB) of
        (VolLow, ConfidenceWeak) -> mkVolConfGateCell VolConfGateHold 0.0
        (VolLow, ConfidenceMedium) -> mkVolConfGateCell VolConfGateAllowEntry (vcgcLowMediumSizeMult cfg)
        (VolLow, ConfidenceStrong) -> mkVolConfGateCell VolConfGateAllowEntry (vcgcLowStrongSizeMult cfg)
        (VolMedium, ConfidenceWeak) -> mkVolConfGateCell VolConfGateHold 0.0
        (VolMedium, ConfidenceMedium) -> mkVolConfGateCell VolConfGateAllowEntry (vcgcMediumMediumSizeMult cfg)
        (VolMedium, ConfidenceStrong) -> mkVolConfGateCell VolConfGateAllowEntry (vcgcMediumStrongSizeMult cfg)
        (VolHigh, ConfidenceWeak) -> mkVolConfGateCell VolConfGateBlock 0.0
        (VolHigh, ConfidenceMedium) -> mkVolConfGateCell VolConfGateAllowExitOnly 0.0
        (VolHigh, ConfidenceStrong) -> mkVolConfGateCell VolConfGateAllowEntry (vcgcHighStrongSizeMult cfg)
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

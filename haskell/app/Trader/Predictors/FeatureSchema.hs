module Trader.Predictors.FeatureSchema (
    FeatureRequirement (..),
    TimedFeatureValue (..),
    FeatureField (..),
    FeatureRowV2,
    frv2SchemaId,
    frv2DecisionTimeMs,
    frv2Names,
    frv2Values,
    frv2Available,
    frv2Required,
    featureAvailabilitySchemaIdV2,
    mkFeatureRowV2,
    featureRowModelInputs,
    featureRowSchemaSignature,
) where

import Data.Char (isAlphaNum)
import Data.Int (Int64)
import Data.List (intercalate, nub)
import Data.Maybe (isJust, isNothing)

data FeatureRequirement
    = OptionalFeature
    | RequiredFeature
    deriving (Eq, Show)

data TimedFeatureValue = TimedFeatureValue
    { tfvEventTimeMs :: !Int64
    , tfvAvailabilityTimeMs :: !Int64
    , tfvValue :: !Double
    }
    deriving (Eq, Show)

data FeatureField = FeatureField
    { ffName :: !String
    , ffRequirement :: !FeatureRequirement
    , ffObservation :: !(Maybe TimedFeatureValue)
    }
    deriving (Eq, Show)

data FeatureRowV2 = FeatureRowV2
    { frv2SchemaId :: !String
    , frv2DecisionTimeMs :: !Int64
    , frv2Names :: ![String]
    , frv2Values :: ![Double]
    , frv2Available :: ![Bool]
    , frv2Required :: ![Bool]
    }
    deriving (Eq, Show)

featureAvailabilitySchemaIdV2 :: String
featureAvailabilitySchemaIdV2 = "feature_availability_v2"

mkFeatureRowV2 :: Int64 -> [FeatureField] -> Maybe FeatureRowV2
mkFeatureRowV2 decisionTime fields
    | decisionTime < 0 = Nothing
    | null fields = Nothing
    | not (all (validName . ffName) fields) = Nothing
    | not (allUnique (map ffName fields)) = Nothing
    | anyRequiredUnavailable (zip fields cells) = Nothing
    | otherwise =
        Just
            FeatureRowV2
                { frv2SchemaId = featureAvailabilitySchemaIdV2
                , frv2DecisionTimeMs = decisionTime
                , frv2Names = map ffName fields
                , frv2Values = map (maybe 0 tfvValue) cells
                , frv2Available = map isJust cells
                , frv2Required = map ((== RequiredFeature) . ffRequirement) fields
                }
  where
    cells = map (availableObservation decisionTime . ffObservation) fields
    anyRequiredUnavailable [] = False
    anyRequiredUnavailable ((field, value) : rest) =
        (ffRequirement field == RequiredFeature && isNothing value)
            || anyRequiredUnavailable rest

featureRowModelInputs :: FeatureRowV2 -> [Double]
featureRowModelInputs row = frv2Values row ++ map availabilityValue (frv2Available row)
  where
    availabilityValue True = 1
    availabilityValue False = 0

featureRowSchemaSignature :: FeatureRowV2 -> String
featureRowSchemaSignature row =
    frv2SchemaId row
        ++ "|"
        ++ intercalate "," (zipWith fieldSignature (frv2Names row) (frv2Required row))
  where
    fieldSignature name True = name ++ ":required"
    fieldSignature name False = name ++ ":optional"

availableObservation :: Int64 -> Maybe TimedFeatureValue -> Maybe TimedFeatureValue
availableObservation decisionTime mObservation = do
    observation <- mObservation
    let eventTime = tfvEventTimeMs observation
        availabilityTime = tfvAvailabilityTimeMs observation
        value = tfvValue observation
    if eventTime >= 0
        && availabilityTime >= eventTime
        && availabilityTime <= decisionTime
        && finite value
        then Just observation
        else Nothing

validName :: String -> Bool
validName name = not (null name) && all validCharacter name
  where
    validCharacter character = isAlphaNum character || character `elem` "_.-"

allUnique :: (Eq a) => [a] -> Bool
allUnique values = length values == length (nub values)

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

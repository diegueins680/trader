module Trader.Predictors.ExternalFeaturesV2 (
    externalModelFeatureSchemaIdV2,
    externalModelFeatureSchemaVersionV2,
    externalModelFeatureNamesV2,
    externalModelFeatureSignatureV2,
    externalFeatureRowsV2,
) where

import Control.Monad (guard, join)
import Data.Int (Int64)
import Data.List (intercalate)
import Data.Maybe (mapMaybe)
import qualified Data.Vector as V

import Trader.Predictors.Exogenous (
    AlignedFeatureSeriesV2,
    afsV2AvailabilityTimesMs,
    afsV2Available,
    afsV2EventTimesMs,
    afsV2Values,
 )
import Trader.Predictors.ExternalFeatureSchema (
    ExternalFeature,
    ExternalFeatureInputsV2,
    externalFeatureColumnName,
    externalFeatureFamilies,
    externalFeatureGridMatchesV2,
    externalFeatureSeriesV2,
 )
import Trader.Predictors.FeatureSchema (
    FeatureField (..),
    FeatureRequirement (OptionalFeature),
    FeatureRowV2,
    TimedFeatureValue (..),
    featureAvailabilitySchemaIdV2,
    mkFeatureRowV2,
 )

externalModelFeatureSchemaIdV2 :: String
externalModelFeatureSchemaIdV2 = "external_family_model_features_v2"

externalModelFeatureSchemaVersionV2 :: Int
externalModelFeatureSchemaVersionV2 = 2

{- | Fixed value order. It deliberately matches the legacy external-family
block before the availability masks are appended by 'featureRowModelInputs'.
-}
externalModelFeatureNamesV2 :: [String]
externalModelFeatureNamesV2 =
    concatMap
        ( \feature ->
            let name = externalFeatureColumnName feature
             in [name ++ ".level", name ++ ".delta"]
        )
        externalFeatureFamilies

externalModelFeatureSignatureV2 :: String
externalModelFeatureSignatureV2 =
    externalModelFeatureSchemaIdV2
        ++ "|"
        ++ featureAvailabilitySchemaIdV2
        ++ "|"
        ++ intercalate "," (map (++ ":optional") externalModelFeatureNamesV2)

{- | Convert a timestamp-preserving external-family bundle on an exact bar grid
into availability-aware feature rows. The bundle must have been constructed
from source observations whose provenance is independently admissible; this
pure adapter does not verify external cache or panel artifacts.
-}
externalFeatureRowsV2 :: V.Vector Int64 -> Int64 -> ExternalFeatureInputsV2 -> Maybe [FeatureRowV2]
externalFeatureRowsV2 openTimes intervalMs inputs = do
    decisionTimes <- traverse (decisionTime intervalMs) (V.toList openTimes)
    guard (contiguousGrid intervalMs (V.toList openTimes))
    guard (externalFeatureGridMatchesV2 openTimes intervalMs inputs)
    let presentSeries = mapMaybe (`externalFeatureSeriesV2` inputs) externalFeatureFamilies
        rowCount = V.length openTimes
    guard (not (null presentSeries) && all (validSeries rowCount) presentSeries)
    traverse
        ( \(index, decision) ->
            mkFeatureRowV2
                decision
                ( zipWith
                    optionalField
                    externalModelFeatureNamesV2
                    (featureObservationsAt inputs intervalMs index decision)
                )
        )
        (zip [0 ..] decisionTimes)

optionalField :: String -> Maybe TimedFeatureValue -> FeatureField
optionalField name = FeatureField name OptionalFeature

featureObservationsAt :: ExternalFeatureInputsV2 -> Int64 -> Int -> Int64 -> [Maybe TimedFeatureValue]
featureObservationsAt inputs intervalMs index decision =
    concatMap familyObservations externalFeatureFamilies
  where
    familyObservations :: ExternalFeature -> [Maybe TimedFeatureValue]
    familyObservations feature =
        case externalFeatureSeriesV2 feature inputs of
            Nothing -> [Nothing, Nothing]
            Just series ->
                [ seriesObservation series index decision
                , deltaObservation intervalMs series index decision
                ]

seriesObservation :: AlignedFeatureSeriesV2 -> Int -> Int64 -> Maybe TimedFeatureValue
seriesObservation series index decision = do
    available <- afsV2Available series V.!? index
    guard available
    value <- afsV2Values series V.!? index
    eventTime <- join (afsV2EventTimesMs series V.!? index)
    availabilityTime <- join (afsV2AvailabilityTimesMs series V.!? index)
    guard
        ( finite value
            && eventTime >= 0
            && eventTime <= availabilityTime
            && availabilityTime <= decision
        )
    pure
        TimedFeatureValue
            { tfvEventTimeMs = eventTime
            , tfvAvailabilityTimeMs = availabilityTime
            , tfvValue = value
            }

deltaObservation :: Int64 -> AlignedFeatureSeriesV2 -> Int -> Int64 -> Maybe TimedFeatureValue
deltaObservation intervalMs series index decision = do
    guard (index > 0)
    previous <- seriesObservation series (index - 1) (decision - intervalMs)
    current <- seriesObservation series index decision
    let value = tfvValue current - tfvValue previous
    guard (finite value)
    pure
        TimedFeatureValue
            { tfvEventTimeMs = max (tfvEventTimeMs previous) (tfvEventTimeMs current)
            , tfvAvailabilityTimeMs = max (tfvAvailabilityTimeMs previous) (tfvAvailabilityTimeMs current)
            , tfvValue = value
            }

validSeries :: Int -> AlignedFeatureSeriesV2 -> Bool
validSeries expected series =
    V.length (afsV2Values series) == expected
        && V.length (afsV2Available series) == expected
        && V.length (afsV2EventTimesMs series) == expected
        && V.length (afsV2AvailabilityTimesMs series) == expected

contiguousGrid :: Int64 -> [Int64] -> Bool
contiguousGrid intervalMs openTimes =
    intervalMs > 0
        && not (null openTimes)
        && and
            [ current >= 0 && toInteger current == toInteger previous + toInteger intervalMs
            | (previous, current) <- zip openTimes (drop 1 openTimes)
            ]

decisionTime :: Int64 -> Int64 -> Maybe Int64
decisionTime intervalMs openTime = do
    guard (intervalMs > 0 && openTime >= 0)
    let candidate = toInteger openTime + toInteger intervalMs - 1
    guard (candidate <= toInteger (maxBound :: Int64))
    pure (fromInteger candidate)

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

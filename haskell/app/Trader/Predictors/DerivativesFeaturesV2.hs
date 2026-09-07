module Trader.Predictors.DerivativesFeaturesV2 (
    derivativesModelFeatureSchemaIdV2,
    derivativesModelFeatureSchemaVersionV2,
    derivativesModelFeatureNamesV2,
    derivativesModelFeatureSignatureV2,
    derivativesFeatureRowV2,
    derivativesFeatureRowsV2,
) where

import Control.Monad (guard)
import Data.Char (isAlphaNum, isAscii, toUpper)
import Data.Int (Int64)
import Data.List (intercalate)

import Trader.Predictors.DerivativesPanelSchema (
    DerivativesFeatureV2 (..),
    DerivativesPanelCellV2 (..),
    DerivativesPanelRowV2 (..),
    derivativesPanelCellV2,
 )
import Trader.Predictors.FeatureSchema (
    FeatureField (..),
    FeatureRequirement (OptionalFeature),
    FeatureRowV2,
    TimedFeatureValue (..),
    featureAvailabilitySchemaIdV2,
    mkFeatureRowV2,
 )

derivativesModelFeatureSchemaIdV2 :: String
derivativesModelFeatureSchemaIdV2 = "binance_derivatives_model_features_v2"

derivativesModelFeatureSchemaVersionV2 :: Int
derivativesModelFeatureSchemaVersionV2 = 2

{- | Fixed value order. It deliberately matches the legacy dense derivatives
block before the availability masks are appended by 'featureRowModelInputs'.
-}
derivativesModelFeatureNamesV2 :: [String]
derivativesModelFeatureNamesV2 =
    [ "funding.level"
    , "funding.delta"
    , "open_interest.relative_delta"
    , "basis.level"
    , "taker_flow.centered"
    ]

derivativesModelFeatureSignatureV2 :: String
derivativesModelFeatureSignatureV2 =
    derivativesModelFeatureSchemaIdV2
        ++ "|"
        ++ featureAvailabilitySchemaIdV2
        ++ "|"
        ++ intercalate "," (map (++ ":optional") derivativesModelFeatureNamesV2)

{- | Convert one decoded derivatives row into the availability-aware model
feature contract. The prior row is mandatory only for delta features. Missing,
stale, malformed, or non-finite optional cells remain explicit unavailable
values; malformed row identity or adjacency rejects the entire row.
-}
derivativesFeatureRowV2 :: Int64 -> Maybe DerivativesPanelRowV2 -> DerivativesPanelRowV2 -> Maybe FeatureRowV2
derivativesFeatureRowV2 intervalMs previous current = do
    guard (validRow intervalMs current)
    case previous of
        Nothing -> pure ()
        Just prior -> guard (adjacentRows intervalMs prior current)
    mkFeatureRowV2
        (dpr2DecisionTimeMs current)
        (zipWith optionalField derivativesModelFeatureNamesV2 (featureObservations previous current))
  where
    optionalField name = FeatureField name OptionalFeature

{- | Convert a complete chronological panel. Empty, mixed-symbol, malformed,
or gapped input fails as a whole; extending a valid list cannot change any
already-built prefix row.
-}
derivativesFeatureRowsV2 :: Int64 -> [DerivativesPanelRowV2] -> Maybe [FeatureRowV2]
derivativesFeatureRowsV2 _ [] = Nothing
derivativesFeatureRowsV2 intervalMs rows =
    sequence
        [ derivativesFeatureRowV2 intervalMs previous current
        | (previous, current) <- zip (Nothing : map Just rows) rows
        ]

featureObservations :: Maybe DerivativesPanelRowV2 -> DerivativesPanelRowV2 -> [Maybe TimedFeatureValue]
featureObservations previous current =
    [ levelObservation current DerivativesFundingV2 id
    , deltaObservation previous current DerivativesFundingV2 difference
    , deltaObservation previous current DerivativesOpenInterestV2 relativeDifference
    , levelObservation current DerivativesBasisV2 id
    , levelObservation current DerivativesTakerFlowV2 (subtract 1)
    ]
  where
    difference old new = new - old
    relativeDifference old new
        | abs old <= 1.0e-12 = 0
        | otherwise = (new - old) / abs old

levelObservation :: DerivativesPanelRowV2 -> DerivativesFeatureV2 -> (Double -> Double) -> Maybe TimedFeatureValue
levelObservation row feature transform = do
    observation <- usableObservation row feature
    let value = transform (tfvValue observation)
    guard (finite value)
    pure observation{tfvValue = value}

deltaObservation ::
    Maybe DerivativesPanelRowV2 ->
    DerivativesPanelRowV2 ->
    DerivativesFeatureV2 ->
    (Double -> Double -> Double) ->
    Maybe TimedFeatureValue
deltaObservation previous current feature transform = do
    prior <- previous
    old <- usableObservation prior feature
    new <- usableObservation current feature
    let value = transform (tfvValue old) (tfvValue new)
    guard (finite value)
    pure
        TimedFeatureValue
            { tfvEventTimeMs = max (tfvEventTimeMs old) (tfvEventTimeMs new)
            , tfvAvailabilityTimeMs = max (tfvAvailabilityTimeMs old) (tfvAvailabilityTimeMs new)
            , tfvValue = value
            }

usableObservation :: DerivativesPanelRowV2 -> DerivativesFeatureV2 -> Maybe TimedFeatureValue
usableObservation row feature = do
    cell <- derivativesPanelCellV2 feature row
    guard (dpc2Observed cell && dpc2Fresh cell && finite (dpc2Value cell))
    eventTime <- dpc2EventTimeMs cell
    availabilityTime <- dpc2AvailabilityTimeMs cell
    guard
        ( eventTime >= 0
            && eventTime <= availabilityTime
            && availabilityTime <= dpr2DecisionTimeMs row
        )
    pure
        TimedFeatureValue
            { tfvEventTimeMs = eventTime
            , tfvAvailabilityTimeMs = availabilityTime
            , tfvValue = dpc2Value cell
            }

validRow :: Int64 -> DerivativesPanelRowV2 -> Bool
validRow intervalMs row =
    intervalMs > 0
        && dpr2OpenTimeMs row >= 0
        && validSymbol (dpr2Symbol row)
        && toInteger (dpr2DecisionTimeMs row)
            == toInteger (dpr2OpenTimeMs row) + toInteger intervalMs - 1

adjacentRows :: Int64 -> DerivativesPanelRowV2 -> DerivativesPanelRowV2 -> Bool
adjacentRows intervalMs previous current =
    validRow intervalMs previous
        && dpr2Symbol previous == dpr2Symbol current
        && toInteger (dpr2OpenTimeMs current)
            == toInteger (dpr2OpenTimeMs previous) + toInteger intervalMs

validSymbol :: String -> Bool
validSymbol symbol =
    not (null symbol)
        && symbol == map toUpper symbol
        && all (\character -> isAscii character && isAlphaNum character) symbol

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

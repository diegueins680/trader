module Trader.Predictors.OhlcvInputsV2 (
    CompleteOhlcvInputsV2,
    completeOhlcvSchemaIdV2,
    completeOhlcvSchemaVersionV2,
    completeOhlcvFieldNamesV2,
    completeOhlcvSchemaSignatureV2,
    completeOhlcvInputsV2,
    completeOhlcvRowsV2,
    completeOhlcvLegacyFeatureInputsV2,
    completeOhlcvScopeV2,
    completeOhlcvGridV2,
) where

import Control.Monad (guard)
import Data.Char (isAlphaNum, isAscii, toUpper)
import Data.Int (Int64)
import Data.List (intercalate)
import qualified Data.Vector as V

import Trader.Predictors.FeatureSchema (
    FeatureField (..),
    FeatureRequirement (RequiredFeature),
    FeatureRowV2,
    TimedFeatureValue (..),
    featureAvailabilitySchemaIdV2,
    mkFeatureRowV2,
 )
import Trader.Predictors.Features (FeatureInputs (..), mkFeatureInputs)

data CompleteOhlcvInputsV2 = CompleteOhlcvInputsV2
    { coi2Scope :: !String
    , coi2OpenTimesMs :: !(V.Vector Int64)
    , coi2DecisionTimesMs :: !(V.Vector Int64)
    , coi2AvailabilityTimesMs :: !(V.Vector Int64)
    , coi2IntervalMs :: !Int64
    , coi2Rows :: ![FeatureRowV2]
    , coi2LegacyProjection :: !FeatureInputs
    }
    deriving (Eq, Show)

completeOhlcvSchemaIdV2 :: String
completeOhlcvSchemaIdV2 = "complete_ohlcv_feature_inputs_v2"

completeOhlcvSchemaVersionV2 :: Int
completeOhlcvSchemaVersionV2 = 2

completeOhlcvFieldNamesV2 :: [String]
completeOhlcvFieldNamesV2 =
    [ "market.open"
    , "market.high"
    , "market.low"
    , "market.close"
    , "market.volume"
    ]

completeOhlcvSchemaSignatureV2 :: String
completeOhlcvSchemaSignatureV2 =
    completeOhlcvSchemaIdV2
        ++ "|"
        ++ featureAvailabilitySchemaIdV2
        ++ "|"
        ++ intercalate "," (map (++ ":required") completeOhlcvFieldNamesV2)

{- | Validate a complete-case OHLCV source boundary before it reaches the
unchanged legacy feature formulas. The bucket-end event time is derived from
the exact bar grid; the caller must supply the real first-seen availability and
decision times. All five market fields are required. Missing or malformed core
market evidence rejects the bundle instead of invoking legacy synthetic
open/high/low/volume fallbacks.

The returned legacy projection intentionally contains only OHLCV. Attached
legacy derivatives, external, Coinbase, or other context fields are not carried
through this source-specific boundary.
-}
completeOhlcvInputsV2 ::
    String ->
    V.Vector Int64 ->
    V.Vector Int64 ->
    V.Vector Int64 ->
    Int64 ->
    FeatureInputs ->
    Maybe CompleteOhlcvInputsV2
completeOhlcvInputsV2 scope openTimes decisionTimes availabilityTimes intervalMs inputs = do
    guard (validScope scope)
    guard (contiguousGrid intervalMs (V.toList openTimes))
    let rowCount = V.length openTimes
        closes = fiClose inputs
    opens <- fiOpen inputs
    highs <- fiHigh inputs
    lows <- fiLow inputs
    volumes <- fiVolume inputs
    guard
        ( all
            (== rowCount)
            [ V.length decisionTimes
            , V.length availabilityTimes
            , V.length closes
            , V.length opens
            , V.length highs
            , V.length lows
            , V.length volumes
            ]
        )
    guard
        ( validSchedule
            intervalMs
            (V.toList openTimes)
            (V.toList decisionTimes)
            (V.toList availabilityTimes)
        )
    rows <-
        traverse
            (buildRow openTimes decisionTimes availabilityTimes intervalMs opens highs lows closes volumes)
            [0 .. rowCount - 1]
    pure
        CompleteOhlcvInputsV2
            { coi2Scope = scope
            , coi2OpenTimesMs = openTimes
            , coi2DecisionTimesMs = decisionTimes
            , coi2AvailabilityTimesMs = availabilityTimes
            , coi2IntervalMs = intervalMs
            , coi2Rows = rows
            , coi2LegacyProjection =
                mkFeatureInputs closes (Just opens) (Just highs) (Just lows) (Just volumes)
            }

completeOhlcvRowsV2 :: CompleteOhlcvInputsV2 -> [FeatureRowV2]
completeOhlcvRowsV2 = coi2Rows

completeOhlcvLegacyFeatureInputsV2 :: CompleteOhlcvInputsV2 -> FeatureInputs
completeOhlcvLegacyFeatureInputsV2 = coi2LegacyProjection

completeOhlcvScopeV2 :: CompleteOhlcvInputsV2 -> String
completeOhlcvScopeV2 = coi2Scope

completeOhlcvGridV2 :: CompleteOhlcvInputsV2 -> (V.Vector Int64, V.Vector Int64, V.Vector Int64, Int64)
completeOhlcvGridV2 inputs =
    ( coi2OpenTimesMs inputs
    , coi2DecisionTimesMs inputs
    , coi2AvailabilityTimesMs inputs
    , coi2IntervalMs inputs
    )

buildRow ::
    V.Vector Int64 ->
    V.Vector Int64 ->
    V.Vector Int64 ->
    Int64 ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Int ->
    Maybe FeatureRowV2
buildRow openTimes decisionTimes availabilityTimes intervalMs opens highs lows closes volumes index = do
    openTime <- openTimes V.!? index
    decisionTime <- decisionTimes V.!? index
    availabilityTime <- availabilityTimes V.!? index
    eventTime <- barEndTime intervalMs openTime
    open <- opens V.!? index
    high <- highs V.!? index
    low <- lows V.!? index
    close <- closes V.!? index
    volume <- volumes V.!? index
    guard (validMarketValues open high low close volume)
    let requiredField name value =
            FeatureField
                name
                RequiredFeature
                (Just (TimedFeatureValue eventTime availabilityTime value))
    mkFeatureRowV2
        decisionTime
        ( zipWith
            requiredField
            completeOhlcvFieldNamesV2
            [open, high, low, close, volume]
        )

validMarketValues :: Double -> Double -> Double -> Double -> Double -> Bool
validMarketValues open high low close volume =
    all finite [open, high, low, close, volume]
        && all (> 0) [open, high, low, close]
        && volume >= 0
        && high >= low
        && high >= max open close
        && low <= min open close

validSchedule :: Int64 -> [Int64] -> [Int64] -> [Int64] -> Bool
validSchedule intervalMs openTimes decisionTimes availabilityTimes =
    length openTimes == length decisionTimes
        && length openTimes == length availabilityTimes
        && strictlyAscending decisionTimes
        && nonDecreasing availabilityTimes
        && and
            [ validTiming openTime decisionTime availabilityTime
            | (openTime, decisionTime, availabilityTime) <- zip3 openTimes decisionTimes availabilityTimes
            ]
  where
    validTiming openTime decisionTime availabilityTime =
        case barEndTime intervalMs openTime of
            Nothing -> False
            Just eventTime ->
                let latestDecision = toInteger eventTime + toInteger intervalMs - 1
                 in availabilityTime >= eventTime
                        && availabilityTime <= decisionTime
                        && decisionTime >= eventTime
                        && toInteger decisionTime <= latestDecision

contiguousGrid :: Int64 -> [Int64] -> Bool
contiguousGrid intervalMs openTimes =
    intervalMs > 0
        && not (null openTimes)
        && all (>= 0) openTimes
        && and
            [ toInteger current == toInteger previous + toInteger intervalMs
            | (previous, current) <- zip openTimes (drop 1 openTimes)
            ]

barEndTime :: Int64 -> Int64 -> Maybe Int64
barEndTime intervalMs openTime = do
    guard (intervalMs > 0 && openTime >= 0)
    let candidate = toInteger openTime + toInteger intervalMs
    guard (candidate <= toInteger (maxBound :: Int64))
    pure (fromInteger candidate)

strictlyAscending :: (Ord a) => [a] -> Bool
strictlyAscending values = and (zipWith (<) values (drop 1 values))

nonDecreasing :: (Ord a) => [a] -> Bool
nonDecreasing values = and (zipWith (<=) values (drop 1 values))

validScope :: String -> Bool
validScope scope =
    not (null scope)
        && scope == map toUpper scope
        && all validCharacter scope
  where
    validCharacter character =
        isAscii character && (isAlphaNum character || character `elem` "-_/")

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

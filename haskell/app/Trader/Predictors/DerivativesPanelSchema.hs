{-# LANGUAGE OverloadedStrings #-}

module Trader.Predictors.DerivativesPanelSchema (
    DerivativesFeatureV2 (..),
    DerivativesPanelCellV2 (..),
    DerivativesPanelRowV2 (..),
    derivativesObservationSchemaIdV2,
    derivativesFeatureAvailabilitySchemaIdV2,
    derivativesFeaturesV2,
    derivativesPanelColumnsV2,
    decodeDerivativesPanelV2,
    derivativesPanelCellV2,
    derivativesPanelCellVersionedV2,
    derivativesPanelCellUsableV2,
) where

import Control.Monad (unless)
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (isAlphaNum, isAscii, toUpper)
import qualified Data.Csv as Csv
import qualified Data.HashMap.Strict as HM
import Data.Int (Int64)
import Data.List (nub)
import qualified Data.Map.Strict as Map
import Data.Maybe (isNothing)
import qualified Data.Vector as V
import Text.Read (readMaybe)

import Trader.Predictors.FeatureSchema (featureAvailabilitySchemaIdV2)
import Trader.Text (trim)

data DerivativesFeatureV2
    = DerivativesFundingV2
    | DerivativesOpenInterestV2
    | DerivativesBasisV2
    | DerivativesTakerFlowV2
    deriving (Bounded, Enum, Eq, Ord, Show)

data DerivativesPanelCellV2 = DerivativesPanelCellV2
    { dpc2Value :: !Double
    , dpc2Observed :: !Bool
    , dpc2Fresh :: !Bool
    , dpc2EventTimeMs :: !(Maybe Int64)
    , dpc2AvailabilityTimeMs :: !(Maybe Int64)
    }
    deriving (Eq, Show)

data DerivativesPanelRowV2 = DerivativesPanelRowV2
    { dpr2OpenTimeMs :: !Int64
    , dpr2DecisionTimeMs :: !Int64
    , dpr2Symbol :: !String
    , dpr2Cells :: !(Map.Map DerivativesFeatureV2 DerivativesPanelCellV2)
    }
    deriving (Eq, Show)

derivativesObservationSchemaIdV2 :: String
derivativesObservationSchemaIdV2 = "binance_derivatives_first_seen_v2"

derivativesFeatureAvailabilitySchemaIdV2 :: String
derivativesFeatureAvailabilitySchemaIdV2 = featureAvailabilitySchemaIdV2

derivativesFeaturesV2 :: [DerivativesFeatureV2]
derivativesFeaturesV2 = [minBound .. maxBound]

derivativesPanelColumnsV2 :: [String]
derivativesPanelColumnsV2 = "openTime" : concatMap featureColumns derivativesFeaturesV2

{- | Decode the additive derivatives v2 columns from a research bar CSV. Other
legacy columns may be interleaved, but the v2 columns must occur exactly once
and in canonical relative order. This validates row semantics, not file hashes
or the caller-supplied symbol scope; those remain manifest responsibilities.
-}
decodeDerivativesPanelV2 :: String -> Int64 -> BL.ByteString -> Either String [DerivativesPanelRowV2]
decodeDerivativesPanelV2 expectedSymbol intervalMs bytes = do
    unless (validSymbol expectedSymbol) $
        Left "binance_derivatives_first_seen_v2 expected symbol is invalid"
    unless (intervalMs > 0) $
        Left "binance_derivatives_first_seen_v2 interval is invalid"
    _ <- checkedMultiply intervalMs 2
    (header, records) <-
        Csv.decodeByName bytes :: Either String (Csv.Header, V.Vector Csv.NamedRecord)
    let headerColumns = map BS.unpack (V.toList header)
    unless (length headerColumns == length (nub headerColumns)) $
        Left "binance_derivatives_first_seen_v2 header contains duplicates"
    unless (columnsInOrder derivativesPanelColumnsV2 headerColumns) $
        Left "binance_derivatives_first_seen_v2 header is incompatible"
    rows <- traverse (decodeRow expectedSymbol intervalMs) (V.toList records)
    validateRows rows
    pure rows

derivativesPanelCellV2 :: DerivativesFeatureV2 -> DerivativesPanelRowV2 -> Maybe DerivativesPanelCellV2
derivativesPanelCellV2 feature = Map.lookup feature . dpr2Cells

derivativesPanelCellVersionedV2 :: DerivativesFeatureV2 -> DerivativesPanelRowV2 -> Bool
derivativesPanelCellVersionedV2 feature = Map.member feature . dpr2Cells

derivativesPanelCellUsableV2 :: DerivativesFeatureV2 -> DerivativesPanelRowV2 -> Bool
derivativesPanelCellUsableV2 feature row =
    maybe False dpc2Fresh (derivativesPanelCellV2 feature row)

decodeRow :: String -> Int64 -> Csv.NamedRecord -> Either String DerivativesPanelRowV2
decodeRow symbol intervalMs record = do
    openTime <- requiredCell "openTime" record >>= parseTimestamp "openTime"
    decisionTime <- checkedAdd openTime (intervalMs - 1)
    cells <- traverse (decodeCell intervalMs decisionTime record) derivativesFeaturesV2
    pure
        DerivativesPanelRowV2
            { dpr2OpenTimeMs = openTime
            , dpr2DecisionTimeMs = decisionTime
            , dpr2Symbol = symbol
            , dpr2Cells =
                Map.fromList
                    [ (feature, cell)
                    | (feature, Just cell) <- zip derivativesFeaturesV2 cells
                    ]
            }

decodeCell :: Int64 -> Int64 -> Csv.NamedRecord -> DerivativesFeatureV2 -> Either String (Maybe DerivativesPanelCellV2)
decodeCell intervalMs decisionTime record feature = do
    rawCells <- traverse (`requiredCell` record) (featureColumns feature)
    if all blank rawCells
        then pure Nothing
        else decodePresent rawCells
  where
    decodePresent [rawValue, rawObserved, rawFresh, rawEventTime, rawAvailabilityTime] = do
        let name = featureName feature
        value <- parseFinite (name ++ "V2Value") rawValue
        observed <- parseMask (name ++ "V2Observed") rawObserved
        fresh <- parseMask (name ++ "V2Fresh") rawFresh
        eventTime <- parseOptionalTimestamp (name ++ "V2EventTime") rawEventTime
        availabilityTime <- parseOptionalTimestamp (name ++ "V2AvailabilityTime") rawAvailabilityTime
        validateCell feature intervalMs decisionTime value observed fresh eventTime availabilityTime
        pure
            ( Just
                DerivativesPanelCellV2
                    { dpc2Value = value
                    , dpc2Observed = observed
                    , dpc2Fresh = fresh
                    , dpc2EventTimeMs = eventTime
                    , dpc2AvailabilityTimeMs = availabilityTime
                    }
            )
    decodePresent _ = Left "binance_derivatives_first_seen_v2 internal column width is invalid"

validateCell :: DerivativesFeatureV2 -> Int64 -> Int64 -> Double -> Bool -> Bool -> Maybe Int64 -> Maybe Int64 -> Either String ()
validateCell feature intervalMs decisionTime value observed fresh eventTime availabilityTime
    | not observed = do
        unless (not fresh && value == 0 && isNothing eventTime && isNothing availabilityTime) $
            Left (name ++ " unavailable cell is incoherent")
    | otherwise = do
        event <- maybe (Left (name ++ " observed cell has no event time")) Right eventTime
        available <- maybe (Left (name ++ " observed cell has no availability time")) Right availabilityTime
        unless (event <= available && available <= decisionTime) $
            Left (name ++ " observed timestamps are not causal")
        maxAge <- freshnessLimit feature intervalMs
        age <- checkedSubtract decisionTime event
        unless (fresh == (age <= maxAge)) $
            Left (name ++ " freshness mask disagrees with its timestamp age")
        unless (fresh || value == 0) $
            Left (name ++ " stale cell has a non-zero value")
  where
    name = featureName feature ++ "V2"

featureName :: DerivativesFeatureV2 -> String
featureName feature =
    case feature of
        DerivativesFundingV2 -> "funding"
        DerivativesOpenInterestV2 -> "oi"
        DerivativesBasisV2 -> "basis"
        DerivativesTakerFlowV2 -> "taker"

featureColumns :: DerivativesFeatureV2 -> [String]
featureColumns feature =
    let name = featureName feature ++ "V2"
     in map
            (name ++)
            ["Value", "Observed", "Fresh", "EventTime", "AvailabilityTime"]

freshnessLimit :: DerivativesFeatureV2 -> Int64 -> Either String Int64
freshnessLimit DerivativesFundingV2 _ = Right (9 * 60 * 60 * 1000)
freshnessLimit _ intervalMs = checkedMultiply intervalMs 2

requiredCell :: String -> Csv.NamedRecord -> Either String BS.ByteString
requiredCell name record =
    maybe
        (Left ("binance_derivatives_first_seen_v2 is missing " ++ name))
        Right
        (HM.lookup (BS.pack name) record)

parseTimestamp :: String -> BS.ByteString -> Either String Int64
parseTimestamp name raw =
    case readMaybe timestampText :: Maybe Int64 of
        Just timestamp | timestamp >= 0 -> Right timestamp
        _ ->
            maybe
                (Left (name ++ " is not a valid timestamp"))
                Right
                (parseIntegralDecimalTimestamp timestampText)
  where
    timestampText = trim (BS.unpack raw)

parseIntegralDecimalTimestamp :: String -> Maybe Int64
parseIntegralDecimalTimestamp raw =
    case readMaybe raw :: Maybe Double of
        Just timestamp
            | finite timestamp
            , timestamp >= 0
            , timestamp == fromInteger integral ->
                either (const Nothing) Just (integerToInt64 integral)
          where
            integral = truncate timestamp
        _ -> Nothing


parseOptionalTimestamp :: String -> BS.ByteString -> Either String (Maybe Int64)
parseOptionalTimestamp name raw
    | blank raw = Right Nothing
    | otherwise = Just <$> parseTimestamp name raw

parseFinite :: String -> BS.ByteString -> Either String Double
parseFinite name raw =
    case readMaybe (trim (BS.unpack raw)) of
        Just value | finite value -> Right value
        _ -> Left (name ++ " is not finite")

parseMask :: String -> BS.ByteString -> Either String Bool
parseMask name raw =
    case readMaybe (trim (BS.unpack raw)) :: Maybe Double of
        Just 0 -> Right False
        Just 1 -> Right True
        _ -> Left (name ++ " is not a binary mask")

validateRows :: [DerivativesPanelRowV2] -> Either String ()
validateRows [] = Left "binance_derivatives_first_seen_v2 has no rows"
validateRows rows =
    unless (strictlyIncreasing (map dpr2OpenTimeMs rows)) $
        Left "binance_derivatives_first_seen_v2 open times are not strictly increasing"

validSymbol :: String -> Bool
validSymbol symbol =
    not (null symbol)
        && symbol == map toUpper (trim symbol)
        && all (\character -> isAscii character && isAlphaNum character) symbol

columnsInOrder :: (Eq a) => [a] -> [a] -> Bool
columnsInOrder [] _ = True
columnsInOrder _ [] = False
columnsInOrder expected@(wanted : rest) (actual : remaining)
    | wanted == actual = columnsInOrder rest remaining
    | otherwise = columnsInOrder expected remaining

strictlyIncreasing :: (Ord a) => [a] -> Bool
strictlyIncreasing values = and (zipWith (<) values (drop 1 values))

checkedAdd :: Int64 -> Int64 -> Either String Int64
checkedAdd left right = integerToInt64 (toInteger left + toInteger right)

checkedSubtract :: Int64 -> Int64 -> Either String Int64
checkedSubtract left right = integerToInt64 (toInteger left - toInteger right)

checkedMultiply :: Int64 -> Int64 -> Either String Int64
checkedMultiply left right = integerToInt64 (toInteger left * toInteger right)

integerToInt64 :: Integer -> Either String Int64
integerToInt64 value
    | value < toInteger (minBound :: Int64) = Left "binance_derivatives_first_seen_v2 timestamp arithmetic overflow"
    | value > toInteger (maxBound :: Int64) = Left "binance_derivatives_first_seen_v2 timestamp arithmetic overflow"
    | otherwise = Right (fromInteger value)

blank :: BS.ByteString -> Bool
blank = null . trim . BS.unpack

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

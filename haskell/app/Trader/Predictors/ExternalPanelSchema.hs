{-# LANGUAGE OverloadedStrings #-}

module Trader.Predictors.ExternalPanelSchema (
    ExternalPanelCellV2 (..),
    ExternalPanelRowV2 (..),
    externalPanelSchemaIdV2,
    externalPanelSchemaVersionV2,
    externalPanelFeatureAvailabilitySchemaIdV2,
    externalPanelColumnsV2,
    decodeExternalPanelV2,
    externalPanelCellV2,
    externalPanelCellAvailableV2,
) where

import Control.Monad (unless)
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (toUpper)
import qualified Data.Csv as Csv
import qualified Data.HashMap.Strict as HM
import Data.Int (Int64)
import qualified Data.Map.Strict as Map
import qualified Data.Vector as V
import Text.Read (readMaybe)

import Trader.Predictors.ExternalFeatureSchema (
    ExternalFeature,
    externalFeatureColumnName,
    externalFeatureFamilies,
 )
import Trader.Predictors.FeatureSchema (featureAvailabilitySchemaIdV2)
import Trader.Text (trim)

data ExternalPanelCellV2 = ExternalPanelCellV2
    { epc2Value :: !Double
    , epc2Coverage :: !Double
    }
    deriving (Eq, Show)

data ExternalPanelRowV2 = ExternalPanelRowV2
    { epr2DecisionTimeMs :: !Int64
    , epr2Symbol :: !String
    , epr2Cells :: !(Map.Map ExternalFeature ExternalPanelCellV2)
    }
    deriving (Eq, Show)

externalPanelSchemaIdV2 :: String
externalPanelSchemaIdV2 = "external_feature_panel_v2"

externalPanelSchemaVersionV2 :: Int
externalPanelSchemaVersionV2 = 2

externalPanelFeatureAvailabilitySchemaIdV2 :: String
externalPanelFeatureAvailabilitySchemaIdV2 = featureAvailabilitySchemaIdV2

externalPanelColumnsV2 :: [String]
externalPanelColumnsV2 =
    ["timestamp", "symbol"]
        ++ concatMap familyColumns externalFeatureFamilies
  where
    familyColumns feature =
        let name = externalFeatureColumnName feature
         in [name, name ++ "_coverage"]

{- | Decode the exact materialized panel CSV contract. This validates typed
row semantics, not the separate manifest and artifact hashes; offline callers
must complete `verify-panel` before admitting these bytes to research.
-}
decodeExternalPanelV2 :: BL.ByteString -> Either String [ExternalPanelRowV2]
decodeExternalPanelV2 bytes = do
    (header, records) <-
        Csv.decodeByName bytes :: Either String (Csv.Header, V.Vector Csv.NamedRecord)
    unless (V.toList header == map BS.pack externalPanelColumnsV2) $
        Left "external_feature_panel_v2 header is incompatible"
    rows <- traverse decodeRow (V.toList records)
    validateRows rows
    pure rows

externalPanelCellV2 :: ExternalFeature -> ExternalPanelRowV2 -> Maybe ExternalPanelCellV2
externalPanelCellV2 feature = Map.lookup feature . epr2Cells

externalPanelCellAvailableV2 :: ExternalFeature -> ExternalPanelRowV2 -> Bool
externalPanelCellAvailableV2 feature row =
    maybe False ((> 0) . epc2Coverage) (externalPanelCellV2 feature row)

decodeRow :: Csv.NamedRecord -> Either String ExternalPanelRowV2
decodeRow record = do
    timestamp <- requiredCell "timestamp" record >>= parseTimestamp
    symbol <- BS.unpack <$> requiredCell "symbol" record
    unless (symbol == map toUpper (trim symbol)) $
        Left "external_feature_panel_v2 symbol is not canonical"
    cells <- traverse (decodeCell record) externalFeatureFamilies
    pure
        ExternalPanelRowV2
            { epr2DecisionTimeMs = timestamp
            , epr2Symbol = symbol
            , epr2Cells = Map.fromList (zip externalFeatureFamilies cells)
            }

decodeCell :: Csv.NamedRecord -> ExternalFeature -> Either String ExternalPanelCellV2
decodeCell record feature = do
    let name = externalFeatureColumnName feature
    value <- requiredCell name record >>= parseFinite name
    coverage <- requiredCell (name ++ "_coverage") record >>= parseFinite (name ++ "_coverage")
    unless (coverage >= 0 && coverage <= 1) $
        Left (name ++ " coverage is outside [0, 1]")
    unless (coverage /= 0 || value == 0) $
        Left (name ++ " has a non-zero value without coverage")
    pure ExternalPanelCellV2{epc2Value = value, epc2Coverage = coverage}

requiredCell :: String -> Csv.NamedRecord -> Either String BS.ByteString
requiredCell name record =
    maybe
        (Left ("external_feature_panel_v2 is missing " ++ name))
        Right
        (HM.lookup (BS.pack name) record)

parseTimestamp :: BS.ByteString -> Either String Int64
parseTimestamp raw =
    case readMaybe (trim (BS.unpack raw)) of
        Just timestamp | timestamp >= 0 -> Right timestamp
        _ -> Left "external_feature_panel_v2 timestamp is invalid"

parseFinite :: String -> BS.ByteString -> Either String Double
parseFinite name raw =
    case readMaybe (trim (BS.unpack raw)) of
        Just value | finite value -> Right value
        _ -> Left (name ++ " is not finite")

validateRows :: [ExternalPanelRowV2] -> Either String ()
validateRows [] = Left "external_feature_panel_v2 has no rows"
validateRows rows@(first : _) = do
    let timestamps = map epr2DecisionTimeMs rows
        expectedSymbol = epr2Symbol first
    unless (strictlyIncreasing timestamps) $
        Left "external_feature_panel_v2 timestamps are not strictly increasing"
    unless (all ((== expectedSymbol) . epr2Symbol) rows) $
        Left "external_feature_panel_v2 mixes symbol scopes"

strictlyIncreasing :: (Ord a) => [a] -> Bool
strictlyIncreasing values = and (zipWith (<) values (drop 1 values))

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

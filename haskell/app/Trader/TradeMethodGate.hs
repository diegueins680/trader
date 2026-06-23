{-# LANGUAGE OverloadedStrings #-}

module Trader.TradeMethodGate (
    MethodResultStats (..),
    MethodGateConfig (..),
    MethodGateDecision (..),
    conservativeUnavailableEvidenceSize,
    unavailableEvidenceSizeCap,
    unavailableEvidenceSizeMultiplier,
    loadMethodResultStats,
    methodGateDecision,
) where

import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Char8 as BS
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import qualified Data.Text as T
import qualified Data.Text.Read as TR
import System.Directory (doesFileExist)
import Trader.Method (Method, methodCode, parseMethod)

data MethodResultStats = MethodResultStats
    { mrsTrades :: !Int
    , mrsTotalReturn :: !Double
    }
    deriving (Eq, Show)

data MethodGateConfig = MethodGateConfig
    { mgcMinTrades :: !Int
    , mgcMinTotalReturn :: !Double
    }
    deriving (Eq, Show)

data MethodGateDecision
    = MethodGateAllowed MethodResultStats
    | MethodGateInsufficientEvidence MethodResultStats
    | MethodGateBlocked MethodResultStats
    | MethodGateUnavailable String
    deriving (Eq, Show)

unavailableEvidenceSizeMultiplier :: Double
unavailableEvidenceSizeMultiplier = 0.25

unavailableEvidenceSizeCap :: Double
unavailableEvidenceSizeCap = 0.25

conservativeUnavailableEvidenceSize :: Maybe Double -> Double
conservativeUnavailableEvidenceSize mSize =
    min unavailableEvidenceSizeCap (max 0 baseSize * unavailableEvidenceSizeMultiplier)
  where
    baseSize =
        case mSize of
            Just x | not (isNaN x || isInfinite x) -> x
            _ -> 1

loadMethodResultStats :: FilePath -> IO (Either String (Map T.Text MethodResultStats))
loadMethodResultStats path = do
    exists <- doesFileExist path
    if not exists
        then pure (Left ("missing trade-log evidence: " ++ path))
        else do
            raw <- BS.readFile path
            pure (Right (methodResultStatsFromLines (BS.lines raw)))

methodGateDecision :: MethodGateConfig -> Method -> Map T.Text MethodResultStats -> MethodGateDecision
methodGateDecision cfg method statsByMethod =
    case Map.lookup (methodStatsKey method) statsByMethod of
        Nothing -> MethodGateUnavailable "no result rows for method"
        Just stats
            | mrsTrades stats < mgcMinTrades cfg -> MethodGateInsufficientEvidence stats
            | mrsTotalReturn stats < mgcMinTotalReturn cfg -> MethodGateBlocked stats
            | otherwise -> MethodGateAllowed stats

methodResultStatsFromLines :: [BS.ByteString] -> Map T.Text MethodResultStats
methodResultStatsFromLines =
    foldl addRow Map.empty
  where
    addRow acc raw =
        case Aeson.decodeStrict' raw of
            Just (Aeson.Object obj) ->
                case (objectMethod obj, objectReturn obj) of
                    (Just method, Just ret) -> Map.insertWith combine method (MethodResultStats 1 ret) acc
                    _ -> acc
            _ -> acc
    combine a b = MethodResultStats (mrsTrades a + mrsTrades b) (mrsTotalReturn a + mrsTotalReturn b)

methodStatsKey :: Method -> T.Text
methodStatsKey =
    T.pack . methodCode

objectMethod :: Aeson.Object -> Maybe T.Text
objectMethod obj = do
    raw <- firstValue ["method", "signal_method"] obj >>= valueText
    method <- either (const Nothing) Just (parseMethod (T.unpack raw))
    Just (methodStatsKey method)

objectReturn :: Aeson.Object -> Maybe Double
objectReturn obj =
    firstValue ["pnlPercent", "pnl_pct", "return", "returnPct"] obj >>= valueDouble

firstValue :: [AK.Key] -> Aeson.Object -> Maybe Aeson.Value
firstValue keys obj =
    case keys of
        [] -> Nothing
        k : rest ->
            case KM.lookup k obj of
                Just v -> Just v
                Nothing -> firstValue rest obj

valueText :: Aeson.Value -> Maybe T.Text
valueText (Aeson.String t) =
    let out = T.strip t
     in if T.null out then Nothing else Just out
valueText _ = Nothing

valueDouble :: Aeson.Value -> Maybe Double
valueDouble (Aeson.Number n) =
    finiteDouble (realToFrac n)
valueDouble (Aeson.String raw) =
    let stripped = T.strip raw
        (body, scale) =
            if "%" `T.isSuffixOf` stripped
                then (T.dropEnd 1 stripped, 0.01)
                else (stripped, 1.0)
     in case TR.double body of
            Right (x, rest) | T.null (T.strip rest) -> finiteDouble (x * scale)
            _ -> Nothing
valueDouble v =
    AT.parseMaybe Aeson.parseJSON v >>= finiteDouble

finiteDouble :: Double -> Maybe Double
finiteDouble x =
    if isNaN x || isInfinite x
        then Nothing
        else Just x

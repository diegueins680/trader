{-# LANGUAGE OverloadedStrings #-}

module Trader.PointInTimeUniverse (
    PointInTimeUniverseConfig (..),
    loadPointInTimeUniverse,
    pointInTimeUniverseConfigFromEnv,
) where

import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (isAsciiLower, isDigit, toLower)
import qualified Data.Csv as Csv
import qualified Data.HashMap.Strict as HM
import Data.Int (Int64)
import Data.List (isSuffixOf, sortOn)
import qualified Data.Map.Strict as Map
import Data.Maybe (listToMaybe, mapMaybe)
import qualified Data.Ord
import qualified Data.Vector as V
import System.Directory (doesFileExist)
import System.Environment (lookupEnv)
import System.IO (hPutStrLn, stderr)
import Text.Read (readMaybe)

import Trader.App.Args (parseTimestampMs)
import Trader.Text (normalizeKey, trim)

data PointInTimeUniverseConfig = PointInTimeUniverseConfig
    { pitUniverseCsv :: !(Maybe FilePath)
    , pitUniverseRequireHistorical :: !Bool
    }
    deriving (Eq, Show)

pointInTimeUniverseConfigFromEnv :: IO PointInTimeUniverseConfig
pointInTimeUniverseConfigFromEnv = do
    csvPath <- nonEmptyEnv "TRADER_MARKET_CONTEXT_UNIVERSE_CSV"
    requireHistorical <- readEnvBool "TRADER_MARKET_CONTEXT_REQUIRE_PIT_UNIVERSE" False
    pure
        PointInTimeUniverseConfig
            { pitUniverseCsv = csvPath
            , pitUniverseRequireHistorical = requireHistorical
            }

loadPointInTimeUniverse ::
    PointInTimeUniverseConfig ->
    String ->
    Int ->
    Int64 ->
    IO (Maybe [(String, Double)])
loadPointInTimeUniverse cfg quote topN asOfMs =
    case pitUniverseCsv cfg of
        Nothing -> pure Nothing
        Just path
            | topN <= 0 -> pure (Just [])
            | otherwise -> do
                exists <- doesFileExist path
                if not exists
                    then do
                        hPutStrLn stderr ("WARN: point-in-time universe CSV not found: " ++ path)
                        pure Nothing
                    else do
                        bs <- BL.readFile path
                        case Csv.decodeByName bs of
                            Left err -> do
                                hPutStrLn stderr ("WARN: point-in-time universe CSV decode failed (" ++ path ++ "): " ++ err)
                                pure Nothing
                            Right (hdr, rows) ->
                                let hdrList = V.toList hdr
                                    parsed = mapMaybe (parseUniverseRow hdrList) (V.toList rows)
                                    selected = selectUniverseRows quote topN asOfMs parsed
                                 in pure (Just selected)

data UniverseRow = UniverseRow
    { urTimestampMs :: !Int64
    , urSymbol :: !String
    , urQuoteVolume :: !Double
    }
    deriving (Eq, Show)

parseUniverseRow :: [BS.ByteString] -> Csv.NamedRecord -> Maybe UniverseRow
parseUniverseRow hdrList row = do
    tsRaw <- lookupAnyWithHeaders hdrList timeColumns row
    ts <- parseTimestampMs (trim (BS.unpack tsRaw))
    symRaw <- lookupAnyWithHeaders hdrList symbolColumns row
    let sym = map toUpperAscii (trim (BS.unpack symRaw))
    if null sym
        then Nothing
        else do
            qvRaw <- lookupAnyWithHeaders hdrList quoteVolumeColumns row
            qv <- parseDouble qvRaw
            if qv < 0
                then Nothing
                else Just UniverseRow{urTimestampMs = ts, urSymbol = sym, urQuoteVolume = qv}

selectUniverseRows :: String -> Int -> Int64 -> [UniverseRow] -> [(String, Double)]
selectUniverseRows quote topN asOfMs rows =
    let quoteU = map toUpperAscii quote
        latestBySymbol =
            foldl
                ( \acc row ->
                    if urTimestampMs row <= asOfMs && wantedSymbol quoteU (urSymbol row)
                        then
                            Map.insertWith
                                newer
                                (urSymbol row)
                                row
                                acc
                        else acc
                )
                Map.empty
                rows
        ranked =
            sortOn (Data.Ord.Down . snd) $
                [ (urSymbol row, urQuoteVolume row)
                | row <- Map.elems latestBySymbol
                ]
     in take topN ranked
  where
    newer a b =
        if urTimestampMs a >= urTimestampMs b then a else b

wantedSymbol :: String -> String -> Bool
wantedSymbol quoteU sym =
    quoteU `isSuffixOf` sym
        && let base = take (length sym - length quoteU) sym
               stableBases = ["USDT", "USDC", "BUSD", "TUSD", "FDUSD"]
               leveragedSuffixes = ["UP", "DOWN", "BULL", "BEAR"]
            in base `notElem` stableBases && not (any (`isSuffixOf` base) leveragedSuffixes)

lookupAnyWithHeaders :: [BS.ByteString] -> [String] -> Csv.NamedRecord -> Maybe BS.ByteString
lookupAnyWithHeaders hdrList candidates row =
    listToMaybe
        [ value
        | wanted <- candidates
        , Just key <- [findHeaderKey hdrList wanted]
        , Just value <- [HM.lookup key row]
        ]

findHeaderKey :: [BS.ByteString] -> String -> Maybe BS.ByteString
findHeaderKey hdrList wanted =
    let wantedNorm = normalizeKey wanted
     in listToMaybe [h | h <- hdrList, normalizeKey (BS.unpack h) == wantedNorm]

parseDouble :: BS.ByteString -> Maybe Double
parseDouble raw =
    case readMaybe (trim (BS.unpack raw)) of
        Just v
            | not (isNaN v || isInfinite v) -> Just v
        _ -> Nothing

timeColumns :: [String]
timeColumns =
    [ "openTimeMs"
    , "open_time_ms"
    , "timestampMs"
    , "timestamp"
    , "asOf"
    , "date"
    , "datetime"
    ]

symbolColumns :: [String]
symbolColumns = ["symbol", "ticker", "market", "pair"]

quoteVolumeColumns :: [String]
quoteVolumeColumns = ["quoteVolume", "quote_volume", "volumeUsd", "usdVolume", "notionalVolume", "volume"]

nonEmptyEnv :: String -> IO (Maybe String)
nonEmptyEnv key = do
    mRaw <- lookupEnv key
    pure $ do
        raw <- mRaw
        let cleaned = trim raw
        if null cleaned then Nothing else Just cleaned

readEnvBool :: String -> Bool -> IO Bool
readEnvBool key fallback = do
    mRaw <- lookupEnv key
    pure $
        case fmap (map toLower . trim) mRaw of
            Just "1" -> True
            Just "true" -> True
            Just "yes" -> True
            Just "on" -> True
            Just "0" -> False
            Just "false" -> False
            Just "no" -> False
            Just "off" -> False
            _ -> fallback

toUpperAscii :: Char -> Char
toUpperAscii c =
    if isAsciiLower c
        then toEnum (fromEnum c - 32)
        else c

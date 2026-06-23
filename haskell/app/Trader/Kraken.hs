{-# LANGUAGE OverloadedStrings #-}

module Trader.Kraken (
    KrakenCandle (..),
    krakenBaseUrl,
    fetchKrakenCandles,
    decodeKrakenCandles,
    krakenCandlesCacheStats,
) where

import Control.Exception (throwIO)
import qualified Control.Monad
import Data.Aeson (Value (..), eitherDecode, withArray, withObject, (.:))
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Int (Int64)
import Data.List (sortOn)
import qualified Data.Scientific as Scientific
import qualified Data.Text as T
import Data.Time.Clock (NominalDiffTime)
import Data.Time.Clock.POSIX (getPOSIXTime)
import qualified Data.Vector as V
import Network.HTTP.Client
import Network.HTTP.Types.Status (statusCode)
import System.IO.Unsafe (unsafePerformIO)
import Text.Read (readMaybe)
import Trader.Cache (TtlCache, TtlCacheStats, cacheStats, fetchWithCache, newTtlCacheWithMaxEntries)
import Trader.Http (defaultRetryConfig, getSharedManager, httpLbsWithRetry)
import Trader.MarketDataIntegrity (MarketSeriesBar (..), normalizeClosedMarketSeries)
import Trader.Text (trim)

data KrakenCandle = KrakenCandle
    { kcOpenTime :: !Int64
    , kcOpen :: !Double
    , kcHigh :: !Double
    , kcLow :: !Double
    , kcClose :: !Double
    , kcVolume :: !Double
    }
    deriving (Eq, Show)

krakenBaseUrl :: String
krakenBaseUrl = "https://api.kraken.com"

{-# NOINLINE krakenCandlesCache #-}
krakenCandlesCache :: TtlCache String [KrakenCandle]
krakenCandlesCache = unsafePerformIO (newTtlCacheWithMaxEntries krakenCandlesMaxEntries)

krakenCandlesMaxEntries :: Int
krakenCandlesMaxEntries = 64

krakenCandlesFreshTtl :: NominalDiffTime
krakenCandlesFreshTtl = 30

krakenCandlesStaleTtl :: NominalDiffTime
krakenCandlesStaleTtl = 300

krakenCandlesCacheStats :: IO TtlCacheStats
krakenCandlesCacheStats = cacheStats krakenCandlesCache krakenCandlesStaleTtl

fetchKrakenCandles :: String -> Int -> IO [KrakenCandle]
fetchKrakenCandles pair intervalMin = do
    let key = pair ++ ":" ++ show intervalMin
    fetchWithCache krakenCandlesCache krakenCandlesFreshTtl krakenCandlesStaleTtl key $ do
        mgr <- getSharedManager
        nowSec <- (floor <$> getPOSIXTime) :: IO Int64
        req0 <- parseRequest (krakenBaseUrl ++ "/0/public/OHLC")
        let req =
                setQueryString
                    [ ("pair", Just (BS.pack pair))
                    , ("interval", Just (BS.pack (show intervalMin)))
                    ]
                    req0
        resp <- httpLbsWithRetry defaultRetryConfig (Just "kraken.ohlc") mgr req
        let code = statusCode (responseStatus resp)
        Control.Monad.when (code < 200 || code >= 300) $ throwIO (userError ("Kraken OHLC request failed (HTTP " ++ show code ++ ")"))
        case decodeKrakenCandles pair (responseBody resp) of
            Left err -> throwIO (userError err)
            Right xs ->
                case normalizeClosedKrakenCandles intervalMin (nowSec * 1000) xs of
                    Left err -> throwIO (userError err)
                    Right ok -> pure ok

decodeKrakenCandles :: String -> BL.ByteString -> Either String [KrakenCandle]
decodeKrakenCandles pair raw = do
    v <-
        case eitherDecode raw of
            Left err -> Left ("Failed to decode Kraken OHLC: " ++ err)
            Right ok -> Right ok
    case AT.parseEither (parseKrakenResponse pair) v of
        Left err -> Left ("Failed to parse Kraken OHLC: " ++ err)
        Right xs -> Right xs

parseKrakenResponse :: String -> Value -> AT.Parser [KrakenCandle]
parseKrakenResponse pair =
    withObject "KrakenResponse" $ \o -> do
        errs <- o .: "error" :: AT.Parser [String]
        case errs of
            [] -> pure ()
            _ -> fail ("Kraken error: " ++ show errs)
        res <- o .: "result"
        parseResult pair res

parseResult :: String -> Value -> AT.Parser [KrakenCandle]
parseResult pair =
    withObject "KrakenResult" $ \o -> do
        let wanted = Key.fromText (T.pack pair)
            keys = filter (/= "last") (map Key.toText (KM.keys o))
            pickKey =
                if KM.member wanted o
                    then wanted
                    else case keys of
                        [k] -> Key.fromText k
                        _ -> wanted
        case KM.lookup pickKey o of
            Nothing -> fail "Kraken result missing pair data"
            Just v -> parseCandles v

parseCandles :: Value -> AT.Parser [KrakenCandle]
parseCandles =
    withArray "KrakenCandles" (fmap V.toList . V.mapM parseCandle)

parseCandle :: Value -> AT.Parser KrakenCandle
parseCandle =
    withArray "KrakenCandle" $ \arr -> do
        if V.length arr < 7
            then fail "Kraken candle array too short"
            else do
                t <- parseIndexInt64 0 arr
                open <- parseIndexDouble 1 arr
                high <- parseIndexDouble 2 arr
                low <- parseIndexDouble 3 arr
                close <- parseIndexDouble 4 arr
                volume <- parseIndexDouble 6 arr
                pure KrakenCandle{kcOpenTime = t, kcOpen = open, kcHigh = high, kcLow = low, kcClose = close, kcVolume = volume}

parseIndexInt64 :: Int -> V.Vector Value -> AT.Parser Int64
parseIndexInt64 i arr =
    case arr V.!? i of
        Nothing -> fail "Missing index"
        Just v -> parseInt64Value v

parseIndexDouble :: Int -> V.Vector Value -> AT.Parser Double
parseIndexDouble i arr =
    case arr V.!? i of
        Nothing -> fail "Missing index"
        Just v -> parseDoubleValue v

parseInt64Value :: Value -> AT.Parser Int64
parseInt64Value v =
    case v of
        Number n ->
            case Scientific.toBoundedInteger n of
                Just x -> pure x
                Nothing -> fail "Invalid integer"
        String t ->
            case readMaybeInt64 (T.unpack t) of
                Just x -> pure x
                Nothing -> fail ("Invalid integer: " ++ T.unpack t)
        _ -> fail "Expected integer"

parseDoubleValue :: Value -> AT.Parser Double
parseDoubleValue v =
    case v of
        Number n -> parseFiniteDouble (realToFrac n)
        String t ->
            case readMaybeDouble (T.unpack t) of
                Just x -> parseFiniteDouble x
                Nothing -> fail ("Invalid double: " ++ T.unpack t)
        _ -> fail "Expected number"

readMaybeInt64 :: String -> Maybe Int64
readMaybeInt64 s =
    readMaybe (trim s)

readMaybeDouble :: String -> Maybe Double
readMaybeDouble s =
    readMaybe (trim s)

parseFiniteDouble :: Double -> AT.Parser Double
parseFiniteDouble x =
    if isNaN x || isInfinite x
        then fail "Invalid finite double"
        else pure x

normalizeClosedKrakenCandles :: Int -> Int64 -> [KrakenCandle] -> Either String [KrakenCandle]
normalizeClosedKrakenCandles intervalMin nowMs candles = do
    let intervalMs = fromIntegral (max 1 intervalMin) * 60 * 1000
        pairs = krakenCandlePairs candles
    closed <- normalizeClosedMarketSeries "Kraken candle" intervalMs nowMs pairs
    pure (map snd closed)

krakenCandlePairs :: [KrakenCandle] -> [(MarketSeriesBar, KrakenCandle)]
krakenCandlePairs candles =
    [ ( MarketSeriesBar
            { msbOpenTimeMs = kcOpenTime candle * 1000
            , msbOpen = Just (kcOpen candle)
            , msbHigh = Just (kcHigh candle)
            , msbLow = Just (kcLow candle)
            , msbClose = kcClose candle
            , msbVolume = Just (kcVolume candle)
            }
      , candle
      )
    | candle <- sortOn kcOpenTime candles
    ]

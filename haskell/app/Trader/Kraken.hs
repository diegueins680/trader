{-# LANGUAGE OverloadedStrings #-}
module Trader.Kraken
  ( KrakenCandle(..)
  , krakenBaseUrl
  , fetchKrakenCandles
  ) where

import Control.Exception (throwIO)
import Data.Aeson (Value(..), eitherDecode, withArray, withObject, (.:))
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Char8 as BS
import Data.Int (Int64)
import qualified Data.Text as T
import Data.Time.Clock (NominalDiffTime)
import qualified Data.Vector as V
import Network.HTTP.Client
import Network.HTTP.Types.Status (statusCode)
import Trader.Cache (TtlCache, fetchWithCache, newTtlCache)
import Trader.Http (defaultRetryConfig, getSharedManager, httpLbsWithRetry)
import System.IO.Unsafe (unsafePerformIO)

data KrakenCandle = KrakenCandle
  { kcOpenTime :: !Int64
  , kcHigh :: !Double
  , kcLow :: !Double
  , kcClose :: !Double
  } deriving (Eq, Show)

krakenBaseUrl :: String
krakenBaseUrl = "https://api.kraken.com"

{-# NOINLINE krakenCandlesCache #-}
krakenCandlesCache :: TtlCache String [KrakenCandle]
krakenCandlesCache = unsafePerformIO newTtlCache

krakenCandlesFreshTtl :: NominalDiffTime
krakenCandlesFreshTtl = 30

krakenCandlesStaleTtl :: NominalDiffTime
krakenCandlesStaleTtl = 300

fetchKrakenCandles :: String -> Int -> IO [KrakenCandle]
fetchKrakenCandles pair intervalMin = do
  let key = pair ++ ":" ++ show intervalMin
  fetchWithCache krakenCandlesCache krakenCandlesFreshTtl krakenCandlesStaleTtl key $ do
    mgr <- getSharedManager
    req0 <- parseRequest (krakenBaseUrl ++ "/0/public/OHLC")
    let req =
          setQueryString
            [ ("pair", Just (BS.pack pair))
            , ("interval", Just (BS.pack (show intervalMin)))
            ]
            req0
    resp <- httpLbsWithRetry defaultRetryConfig (Just "kraken.ohlc") mgr req
    let code = statusCode (responseStatus resp)
    if code < 200 || code >= 300
      then throwIO (userError ("Kraken OHLC request failed (HTTP " ++ show code ++ ")"))
      else pure ()
    case eitherDecode (responseBody resp) of
      Left err -> throwIO (userError ("Failed to decode Kraken OHLC: " ++ err))
      Right v ->
        case AT.parseEither (parseKrakenResponse pair) v of
          Left err -> throwIO (userError ("Failed to parse Kraken OHLC: " ++ err))
          Right xs -> pure xs

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
  withArray "KrakenCandles" $ \arr ->
    V.toList <$> V.mapM parseCandle arr

parseCandle :: Value -> AT.Parser KrakenCandle
parseCandle =
  withArray "KrakenCandle" $ \arr -> do
    if V.length arr < 5
      then fail "Kraken candle array too short"
      else do
        t <- parseIndexInt64 0 arr
        high <- parseIndexDouble 2 arr
        low <- parseIndexDouble 3 arr
        close <- parseIndexDouble 4 arr
        pure KrakenCandle { kcOpenTime = t, kcHigh = high, kcLow = low, kcClose = close }

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
    Number n -> pure (round n)
    String t ->
      case readMaybeInt64 (T.unpack t) of
        Just x -> pure x
        Nothing -> fail ("Invalid integer: " ++ T.unpack t)
    _ -> fail "Expected integer"

parseDoubleValue :: Value -> AT.Parser Double
parseDoubleValue v =
  case v of
    Number n -> pure (realToFrac n)
    String t ->
      case readMaybeDouble (T.unpack t) of
        Just x -> pure x
        Nothing -> fail ("Invalid double: " ++ T.unpack t)
    _ -> fail "Expected number"

readMaybeInt64 :: String -> Maybe Int64
readMaybeInt64 s =
  case reads s of
    [(x, "")] -> Just x
    _ -> Nothing

readMaybeDouble :: String -> Maybe Double
readMaybeDouble s =
  case reads s of
    [(x, "")] -> Just x
    _ -> Nothing

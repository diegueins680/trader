{-# LANGUAGE OverloadedStrings #-}
module Trader.Poloniex
  ( PoloniexCandle(..)
  , fetchPoloniexCandles
  ) where

import Control.Exception (throwIO)
import Data.Aeson (Value(..), eitherDecode, withObject, (.:))
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Char8 as BS
import Data.Int (Int64)
import qualified Data.Text as T
import Data.Time.Clock.POSIX (getPOSIXTime)
import qualified Data.Vector as V
import Network.HTTP.Client
import Network.HTTP.Client.TLS (tlsManagerSettings)
import Network.HTTP.Types.Status (statusCode)

data PoloniexCandle = PoloniexCandle
  { pcOpenTime :: !Int64
  , pcHigh :: !Double
  , pcLow :: !Double
  , pcClose :: !Double
  } deriving (Eq, Show)

poloniexBaseUrl :: String
poloniexBaseUrl = "https://poloniex.com/public"

fetchPoloniexCandles :: String -> Int -> Int -> IO [PoloniexCandle]
fetchPoloniexCandles pair periodSec bars = do
  mgr <- newManager tlsManagerSettings
  now <- round <$> getPOSIXTime
  let lookbackBars = max 1 bars
      start = now - lookbackBars * periodSec
      end = now
  req0 <- parseRequest poloniexBaseUrl
  let req =
        setQueryString
          [ ("command", Just "returnChartData")
          , ("currencyPair", Just (BS.pack pair))
          , ("period", Just (BS.pack (show periodSec)))
          , ("start", Just (BS.pack (show start)))
          , ("end", Just (BS.pack (show end)))
          ]
          req0
  resp <- httpLbs req mgr
  let code = statusCode (responseStatus resp)
  if code < 200 || code >= 300
    then throwIO (userError ("Poloniex chart request failed (HTTP " ++ show code ++ ")"))
    else pure ()
  case eitherDecode (responseBody resp) of
    Left err -> throwIO (userError ("Failed to decode Poloniex chart data: " ++ err))
    Right v ->
      case AT.parseEither (parsePoloniexResponse) v of
        Left err -> throwIO (userError ("Failed to parse Poloniex chart data: " ++ err))
        Right xs -> pure xs

parsePoloniexResponse :: Value -> AT.Parser [PoloniexCandle]
parsePoloniexResponse =
  AT.withArray "PoloniexCandles" $ \arr ->
    mapM parseCandle (V.toList arr)

parseCandle :: Value -> AT.Parser PoloniexCandle
parseCandle =
  withObject "PoloniexCandle" $ \o -> do
    t <- parseInt64Value =<< o .: "date"
    high <- parseDoubleValue =<< o .: "high"
    low <- parseDoubleValue =<< o .: "low"
    close <- parseDoubleValue =<< o .: "close"
    pure PoloniexCandle { pcOpenTime = t, pcHigh = high, pcLow = low, pcClose = close }

parseInt64Value :: Value -> AT.Parser Int64
parseInt64Value v =
  case v of
    Number n -> pure (round n)
    String t ->
      case readMaybeInt64 (T.unpack t) of
        Just x -> pure x
        Nothing -> fail "Invalid integer"
    _ -> fail "Expected integer"

parseDoubleValue :: Value -> AT.Parser Double
parseDoubleValue v =
  case v of
    Number n -> pure (realToFrac n)
    String t ->
      case readMaybeDouble (T.unpack t) of
        Just x -> pure x
        Nothing -> fail "Invalid double"
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

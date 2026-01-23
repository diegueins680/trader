{-# LANGUAGE OverloadedStrings #-}
module Trader.Poloniex
  ( PoloniexCandle(..)
  , poloniexBaseUrl
  , fetchPoloniexCandles
  ) where

import Control.Applicative ((<|>))
import Control.Exception (throwIO)
import Data.Aeson (Value(..), eitherDecode, withObject, (.:), (.:?))
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Char8 as BS
import Data.Char (toUpper)
import Data.Int (Int64)
import Data.List (sortOn)
import qualified Data.Text as T
import Data.Time.Clock (NominalDiffTime)
import Data.Time.Clock.POSIX (getPOSIXTime)
import qualified Data.Vector as V
import Network.HTTP.Client
import Network.HTTP.Types.Status (statusCode)
import Trader.Text (dedupeStable, trim)
import Trader.Cache (TtlCache, fetchWithCache, newTtlCache)
import Trader.Http (defaultRetryConfig, getSharedManager, httpLbsWithRetry)
import System.IO.Unsafe (unsafePerformIO)

data PoloniexCandle = PoloniexCandle
  { pcOpenTime :: !Int64
  , pcHigh :: !Double
  , pcLow :: !Double
  , pcClose :: !Double
  } deriving (Eq, Show)

poloniexBaseUrl :: String
poloniexBaseUrl = "https://api.poloniex.com/markets"

poloniexTimeoutMicros :: Int
poloniexTimeoutMicros = 15 * 1000000

{-# NOINLINE poloniexCandlesCache #-}
poloniexCandlesCache :: TtlCache String [PoloniexCandle]
poloniexCandlesCache = unsafePerformIO newTtlCache

poloniexCandlesFreshTtl :: NominalDiffTime
poloniexCandlesFreshTtl = 30

poloniexCandlesStaleTtl :: NominalDiffTime
poloniexCandlesStaleTtl = 300

fetchPoloniexCandles :: String -> String -> Int -> Int -> IO [PoloniexCandle]
fetchPoloniexCandles pair intervalLabel periodSec bars = do
  let key = map toUpper (trim pair) ++ ":" ++ intervalLabel ++ ":" ++ show periodSec ++ ":" ++ show bars
  fetchWithCache poloniexCandlesCache poloniexCandlesFreshTtl poloniexCandlesStaleTtl key $ do
    mgr <- getSharedManager
    now <- round <$> getPOSIXTime
    let lookbackBars = max 1 bars
        endMs = max 0 (fromIntegral now * 1000)
        startMs = max 0 (endMs - fromIntegral lookbackBars * fromIntegral periodSec * 1000)
        candidates = poloniexSymbolCandidates pair
    go mgr startMs endMs candidates Nothing
  where
    go _ _ _ [] Nothing = throwIO (userError "Poloniex chart request failed (no symbol candidates).")
    go _ _ _ [] (Just err) = throwIO err
    go manager startMs endMs (sym : rest) lastErr = do
      res <- fetchSymbol manager startMs endMs sym
      case res of
        Right xs -> pure (sortOn pcOpenTime xs)
        Left code ->
          let err = userError ("Poloniex chart request failed for " ++ sym ++ " (HTTP " ++ show code ++ ")")
              retryable = code `elem` [400, 404, 422]
           in if retryable && not (null rest)
                then go manager startMs endMs rest (Just err)
                else throwIO err

    fetchSymbol manager startMs endMs sym = do
      req0 <- parseRequest (poloniexBaseUrl ++ "/" ++ sym ++ "/candles")
      let req =
            setQueryString
              [ ("interval", Just (BS.pack intervalLabel))
              , ("startTime", Just (BS.pack (show startMs)))
              , ("endTime", Just (BS.pack (show endMs)))
              ]
              req0
          req' =
            req
              { requestHeaders =
                  ("User-Agent", BS.pack "trader-hs/0.1")
                    : ("Accept", BS.pack "application/json")
                    : requestHeaders req
              , responseTimeout = responseTimeoutMicro poloniexTimeoutMicros
              }
      resp <- httpLbsWithRetry defaultRetryConfig (Just "poloniex.candles") manager req'
      let code = statusCode (responseStatus resp)
      if code < 200 || code >= 300
        then pure (Left code)
        else
          case eitherDecode (responseBody resp) of
            Left err -> throwIO (userError ("Failed to decode Poloniex chart data: " ++ err))
            Right v ->
              case AT.parseEither parsePoloniexResponse v of
                Left err -> throwIO (userError ("Failed to parse Poloniex chart data: " ++ err))
                Right xs -> pure (Right xs)

parsePoloniexResponse :: Value -> AT.Parser [PoloniexCandle]
parsePoloniexResponse v =
  case v of
    Array arr -> mapM parseCandle (V.toList arr)
    Object o -> do
      mErr <- o .:? "message" <|> o .:? "error"
      case mErr of
        Just msg -> fail ("Poloniex error: " ++ msg)
        Nothing -> fail "Unexpected Poloniex response."
    _ -> fail "Unexpected Poloniex response."

parseCandle :: Value -> AT.Parser PoloniexCandle
parseCandle v =
  case v of
    Object o -> do
      tVal <- o .:? "ts"
      startVal <- o .:? "startTime"
      dateVal <- o .:? "date"
      v' <- maybe (fail "Poloniex candle missing timestamp") pure (tVal <|> startVal <|> dateVal)
      tRaw <- parseInt64Value v'
      let t = normalizeTimestamp tRaw
      high <- parseDoubleValue =<< o .: "high"
      low <- parseDoubleValue =<< o .: "low"
      close <- parseDoubleValue =<< o .: "close"
      pure PoloniexCandle { pcOpenTime = t, pcHigh = high, pcLow = low, pcClose = close }
    Array arr -> parseCandleArray arr
    _ -> fail "Poloniex candle expected object or array."

parseCandleArray :: V.Vector Value -> AT.Parser PoloniexCandle
parseCandleArray arr = do
  low <- parseDoubleValue =<< parseArrayIndex 0 arr
  high <- parseDoubleValue =<< parseArrayIndex 1 arr
  close <- parseDoubleValue =<< parseArrayIndex 3 arr
  v <-
    case arr V.!? 9 <|> arr V.!? 12 <|> arr V.!? 13 of
      Just t -> pure t
      Nothing -> fail "Poloniex candle missing timestamp"
  tRaw <- parseInt64Value v
  let t = normalizeTimestamp tRaw
  pure PoloniexCandle { pcOpenTime = t, pcHigh = high, pcLow = low, pcClose = close }

parseArrayIndex :: Int -> V.Vector Value -> AT.Parser Value
parseArrayIndex i arr =
  case arr V.!? i of
    Nothing -> fail ("Poloniex candle missing index " ++ show i)
    Just v -> pure v

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

normalizeTimestamp :: Int64 -> Int64
normalizeTimestamp t =
  if t > 1000000000000
    then t `div` 1000
    else t

poloniexSymbolCandidates :: String -> [String]
poloniexSymbolCandidates raw =
  let cleaned =
        map
          (\c -> if c == '-' then '_' else toUpper c)
          (trim raw)
      parts = filter (not . null) (splitOnUnderscore cleaned)
   in case parts of
        [a, b] -> dedupeStable [a ++ "_" ++ b, b ++ "_" ++ a]
        _ -> [cleaned]

splitOnUnderscore :: String -> [String]
splitOnUnderscore s =
  case break (== '_') s of
    (a, "") -> [a]
    (a, _ : rest) -> a : splitOnUnderscore rest

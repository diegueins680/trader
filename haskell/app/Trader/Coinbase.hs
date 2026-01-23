{-# LANGUAGE OverloadedStrings #-}
module Trader.Coinbase
  ( CoinbaseCandle(..)
  , CoinbaseAccount(..)
  , CoinbaseEnv(..)
  , coinbaseBaseUrl
  , newCoinbaseEnv
  , fetchCoinbaseAccounts
  , fetchCoinbaseAvailableBalance
  , fetchCoinbaseCandles
  , placeCoinbaseMarketOrder
  ) where

import Control.Exception (throwIO)
import Crypto.Hash (SHA256)
import Crypto.MAC.HMAC (HMAC, hmac, hmacGetDigest)
import Data.Aeson (FromJSON(..), Value(..), eitherDecode, encode, object, withArray, withObject, (.=), (.:))
import qualified Data.Aeson.Types as AT
import qualified Data.ByteArray as BA
import qualified Data.ByteArray.Encoding as BAE
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BS8
import qualified Data.ByteString.Lazy as BL
import Data.Int (Int64)
import Data.List (find, sortOn)
import qualified Data.Set as Set
import qualified Data.Text as T
import Data.Time.Clock (NominalDiffTime)
import Data.Time.Clock.POSIX (getPOSIXTime, posixSecondsToUTCTime)
import Data.Time.Format (defaultTimeLocale, formatTime)
import qualified Data.Vector as V
import Network.HTTP.Client
import Network.HTTP.Types.Status (statusCode)
import Numeric (showFFloat)
import Trader.Text (trim)
import Trader.Cache (TtlCache, fetchWithCache, newTtlCache)
import Trader.Http (defaultRetryConfig, getSharedManager, httpLbsWithRetry, newHttpManager)
import System.IO.Unsafe (unsafePerformIO)


data CoinbaseCandle = CoinbaseCandle
  { ccOpenTime :: !Int64
  , ccHigh :: !Double
  , ccLow :: !Double
  , ccClose :: !Double
  } deriving (Eq, Show)

data CoinbaseAccount = CoinbaseAccount
  { caCurrency :: !String
  , caAvailable :: !Double
  } deriving (Eq, Show)

instance FromJSON CoinbaseAccount where
  parseJSON =
    withObject "CoinbaseAccount" $ \o -> do
      cur <- o .: "currency"
      availVal <- o .: "available"
      avail <- parseDoubleValue availVal
      pure CoinbaseAccount { caCurrency = cur, caAvailable = avail }

coinbaseBaseUrl :: String
coinbaseBaseUrl = "https://api.exchange.coinbase.com"

coinbaseTimeoutMicros :: Int
coinbaseTimeoutMicros = 15 * 1000000

coinbaseMaxBarsPerRequest :: Int
coinbaseMaxBarsPerRequest = 300

{-# NOINLINE coinbaseCandlesCache #-}
coinbaseCandlesCache :: TtlCache String [CoinbaseCandle]
coinbaseCandlesCache = unsafePerformIO newTtlCache

coinbaseCandlesFreshTtl :: NominalDiffTime
coinbaseCandlesFreshTtl = 30

coinbaseCandlesStaleTtl :: NominalDiffTime
coinbaseCandlesStaleTtl = 300


data CoinbaseEnv = CoinbaseEnv
  { ceManager :: !Manager
  , ceBaseUrl :: !String
  , ceApiKey :: !(Maybe BS.ByteString)
  , ceApiSecret :: !(Maybe BS.ByteString)
  , ceApiPassphrase :: !(Maybe BS.ByteString)
  }

newCoinbaseEnv :: Maybe BS.ByteString -> Maybe BS.ByteString -> Maybe BS.ByteString -> IO CoinbaseEnv
newCoinbaseEnv apiKey apiSecret apiPassphrase = do
  mgr <- newHttpManager
  pure CoinbaseEnv { ceManager = mgr, ceBaseUrl = coinbaseBaseUrl, ceApiKey = apiKey, ceApiSecret = apiSecret, ceApiPassphrase = apiPassphrase }

coinbaseHttp :: CoinbaseEnv -> String -> Request -> IO (Response BL.ByteString)
coinbaseHttp env label req =
  httpLbsWithRetry defaultRetryConfig (Just label) (ceManager env) req

fetchCoinbaseAccounts :: CoinbaseEnv -> IO BL.ByteString
fetchCoinbaseAccounts env = do
  req0 <- parseRequest (ceBaseUrl env ++ "/accounts")
  req <- signCoinbaseRequest env "GET" "/accounts" BS.empty req0
  resp <- coinbaseHttp env "coinbase.accounts" req
  ensure2xx "Coinbase accounts" resp
  pure (responseBody resp)

fetchCoinbaseAvailableBalance :: CoinbaseEnv -> String -> IO Double
fetchCoinbaseAvailableBalance env currency = do
  raw <- fetchCoinbaseAccounts env
  accounts <-
    case eitherDecode raw of
      Left err -> throwIO (userError ("Failed to decode Coinbase accounts: " ++ err))
      Right xs -> pure (xs :: [CoinbaseAccount])
  let target = map toUpperAscii (trim currency)
      match = find (\acct -> map toUpperAscii (caCurrency acct) == target) accounts
  pure (maybe 0 caAvailable match)

placeCoinbaseMarketOrder :: CoinbaseEnv -> String -> String -> Maybe Double -> Maybe Double -> Maybe String -> IO BL.ByteString
placeCoinbaseMarketOrder env product sideRaw mSizeRaw mFundsRaw mClientOrderId = do
  let cleanedProduct = map toUpperAscii (trim product)
  side <-
    case map toLowerAscii (trim sideRaw) of
      "buy" -> pure ("buy" :: String)
      "sell" -> pure ("sell" :: String)
      _ -> throwIO (userError "Invalid Coinbase order side (expected BUY/SELL).")
  let mSize = mSizeRaw >>= positive
      mFunds = mFundsRaw >>= positive
  case (mSize, mFunds) of
    (Nothing, Nothing) -> throwIO (userError "Coinbase market orders require size or funds.")
    _ -> pure ()
  let body =
        encode $
          object $
            [ "type" .= ("market" :: String)
            , "side" .= side
            , "product_id" .= cleanedProduct
            ]
              ++ maybe [] (\q -> ["size" .= renderDoubleText q]) mSize
              ++ maybe [] (\q -> ["funds" .= renderDoubleText q]) mFunds
              ++ maybe [] (\cid -> ["client_oid" .= trim cid]) mClientOrderId
  req0 <- parseRequest (ceBaseUrl env ++ "/orders")
  reqSigned <- signCoinbaseRequest env "POST" "/orders" (BL.toStrict body) req0
  let req =
        reqSigned
          { requestBody = RequestBodyLBS body
          , requestHeaders = ("Content-Type", BS8.pack "application/json") : requestHeaders reqSigned
          , responseTimeout = responseTimeoutMicro coinbaseTimeoutMicros
          }
  resp <- coinbaseHttp env "coinbase.order" req
  ensure2xx "Coinbase order" resp
  pure (responseBody resp)
  where
    positive v = if v > 0 then Just v else Nothing

fetchCoinbaseCandles :: String -> Int -> Int -> IO [CoinbaseCandle]
fetchCoinbaseCandles product granularitySec bars = do
  let key = map toUpperAscii (trim product) ++ ":" ++ show granularitySec ++ ":" ++ show bars
  fetchWithCache coinbaseCandlesCache coinbaseCandlesFreshTtl coinbaseCandlesStaleTtl key $ do
    mgr <- getSharedManager
    now <- round <$> getPOSIXTime
    let totalBars = max 1 bars
        ranges = buildRanges (fromIntegral now) (fromIntegral granularitySec) totalBars
    chunks <- mapM (fetchRange mgr) ranges
    let candles = concat chunks
        sorted = sortOn ccOpenTime candles
    pure (dedupByTime sorted)
  where
    cleaned = map toUpperAscii (trim product)

    fetchRange manager (startSec, endSec) = do
      req0 <- parseRequest (coinbaseBaseUrl ++ "/products/" ++ cleaned ++ "/candles")
      let req =
            setQueryString
              [ ("granularity", Just (BS8.pack (show granularitySec)))
              , ("start", Just (formatIso startSec))
              , ("end", Just (formatIso endSec))
              ]
              req0
          req' =
            req
              { requestHeaders =
                  ("User-Agent", BS8.pack "trader-hs/0.1")
                    : ("Accept", BS8.pack "application/json")
                    : requestHeaders req
              , responseTimeout = responseTimeoutMicro coinbaseTimeoutMicros
              }
      resp <- httpLbsWithRetry defaultRetryConfig (Just "coinbase.candles") manager req'
      ensure2xx "Coinbase candles" resp
      case eitherDecode (responseBody resp) of
        Left err -> throwIO (userError ("Failed to decode Coinbase candles: " ++ err))
        Right v ->
          case AT.parseEither parseCoinbaseResponse v of
            Left err -> throwIO (userError ("Failed to parse Coinbase candles: " ++ err))
            Right xs -> pure xs

buildRanges :: Int64 -> Int64 -> Int -> [(Int64, Int64)]
buildRanges endSec granularitySec bars =
  let bars' = max 1 bars
      g = max 1 granularitySec
      go acc remaining endTime =
        if remaining <= 0
          then reverse acc
          else
            let chunkBars = min coinbaseMaxBarsPerRequest remaining
                spanSec = fromIntegral chunkBars * g
                startTime = max 0 (endTime - spanSec)
                nextEnd = max 0 (startTime - g)
             in go ((startTime, endTime) : acc) (remaining - chunkBars) nextEnd
   in go [] bars' endSec

formatIso :: Int64 -> BS.ByteString
formatIso sec =
  let t = posixSecondsToUTCTime (fromIntegral sec)
   in BS8.pack (formatTime defaultTimeLocale "%Y-%m-%dT%H:%M:%SZ" t)

parseCoinbaseResponse :: Value -> AT.Parser [CoinbaseCandle]
parseCoinbaseResponse =
  withArray "CoinbaseCandles" $ \arr ->
    V.toList <$> V.mapM parseCandle arr

parseCandle :: Value -> AT.Parser CoinbaseCandle
parseCandle =
  withArray "CoinbaseCandle" $ \arr -> do
    if V.length arr < 5
      then fail "Coinbase candle array too short"
      else do
        t <- parseIndexInt64 0 arr
        low <- parseIndexDouble 1 arr
        high <- parseIndexDouble 2 arr
        close <- parseIndexDouble 4 arr
        pure CoinbaseCandle { ccOpenTime = normalizeTimestamp t, ccHigh = high, ccLow = low, ccClose = close }

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
        Nothing -> fail "Invalid integer"
    _ -> fail "Expected integer"

parseDoubleValue :: Value -> AT.Parser Double
parseDoubleValue v =
  case v of
    Number n -> pure (realToFrac n)
    String t ->
      case reads (T.unpack t) of
        [(x, "")] -> pure x
        _ -> fail "Invalid double"
    _ -> fail "Expected number"

readMaybeInt64 :: String -> Maybe Int64
readMaybeInt64 s =
  case reads s of
    [(x, "")] -> Just x
    _ -> Nothing

normalizeTimestamp :: Int64 -> Int64
normalizeTimestamp t =
  if t > 1000000000000
    then t `div` 1000
    else t

dedupByTime :: [CoinbaseCandle] -> [CoinbaseCandle]
dedupByTime xs = go Set.empty xs
  where
    go _ [] = []
    go seen (y:ys)
      | Set.member (ccOpenTime y) seen = go seen ys
      | otherwise = y : go (Set.insert (ccOpenTime y) seen) ys

signCoinbaseRequest :: CoinbaseEnv -> String -> String -> BS.ByteString -> Request -> IO Request
signCoinbaseRequest env method path body req0 = do
  apiKey <- maybe (throwIO (userError "Missing COINBASE_API_KEY")) pure (ceApiKey env)
  secretRaw <- maybe (throwIO (userError "Missing COINBASE_API_SECRET")) pure (ceApiSecret env)
  passphrase <- maybe (throwIO (userError "Missing COINBASE_API_PASSPHRASE")) pure (ceApiPassphrase env)
  ts <- getPOSIXTime
  secret <-
    case (BAE.convertFromBase BAE.Base64 secretRaw :: Either String BS.ByteString) of
      Left err -> throwIO (userError ("Invalid Coinbase API secret (base64): " ++ err))
      Right v -> pure v
  let tsText = BS8.pack (show (floor ts :: Int))
      msg = BS.concat [tsText, BS8.pack method, BS8.pack path, body]
      mac :: HMAC SHA256
      mac = hmac secret msg
      sig :: BS.ByteString
      sig = BAE.convertToBase BAE.Base64 (hmacGetDigest mac)
      sigText = BA.convert sig :: BS.ByteString
      headers =
        [ ("CB-ACCESS-KEY", apiKey)
        , ("CB-ACCESS-SIGN", sigText)
        , ("CB-ACCESS-TIMESTAMP", tsText)
        , ("CB-ACCESS-PASSPHRASE", passphrase)
        , ("User-Agent", BS8.pack "trader-hs/0.1")
        , ("Accept", BS8.pack "application/json")
        ]
  pure req0 { method = BS8.pack method, requestHeaders = headers ++ requestHeaders req0 }

ensure2xx :: String -> Response BL.ByteString -> IO ()
ensure2xx label resp = do
  let code = statusCode (responseStatus resp)
  if code < 200 || code >= 300
    then throwIO (userError (label ++ " request failed (HTTP " ++ show code ++ ")"))
    else pure ()

renderDoubleText :: Double -> String
renderDoubleText x =
  trimTrailingZeros (showFFloat (Just 8) x "")

trimTrailingZeros :: String -> String
trimTrailingZeros s =
  case break (== '.') s of
    (a, "") -> a
    (a, '.':b) ->
      let b' = reverse (dropWhile (== '0') (reverse b))
       in if null b' then a else a ++ "." ++ b'
    _ -> s

toUpperAscii :: Char -> Char
toUpperAscii c =
  if c >= 'a' && c <= 'z'
    then toEnum (fromEnum c - 32)
    else c

toLowerAscii :: Char -> Char
toLowerAscii c =
  if c >= 'A' && c <= 'Z'
    then toEnum (fromEnum c + 32)
    else c

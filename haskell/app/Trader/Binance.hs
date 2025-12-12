{-# LANGUAGE OverloadedStrings #-}
module Trader.Binance
  ( BinanceEnv(..)
  , BinanceOrderMode(..)
  , OrderSide(..)
  , Kline(..)
  , fetchTickerPrice
  , binanceBaseUrl
  , binanceTestnetBaseUrl
  , newBinanceEnv
  , fetchKlines
  , fetchCloses
  , getTimestampMs
  , signQuery
  , placeMarketOrder
  , fetchFreeBalance
  ) where

import Control.Exception (throwIO)
import Data.Aeson (FromJSON(..), eitherDecode, withArray, (.:), withObject)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Base16 as B16
import Data.Int (Int64)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Vector as V
import Network.HTTP.Client
import Network.HTTP.Client.TLS (tlsManagerSettings)
import Network.HTTP.Types.URI (renderSimpleQuery)
import Network.HTTP.Types.Status (statusCode)
import Crypto.MAC.HMAC (HMAC, hmac, hmacGetDigest)
import Crypto.Hash.Algorithms (SHA256)
import Data.ByteArray (convert)
import Data.Time.Clock.POSIX (getPOSIXTime)
import Numeric (showFFloat)
import Text.Read (readMaybe)

data BinanceEnv = BinanceEnv
  { beManager :: Manager
  , beBaseUrl :: String
  , beApiKey :: Maybe BS.ByteString
  , beApiSecret :: Maybe BS.ByteString
  }

data BinanceOrderMode = OrderTest | OrderLive deriving (Eq, Show)

data OrderSide = Buy | Sell deriving (Eq, Show)

data Kline = Kline
  { kOpenTime :: !Int64
  , kClose :: !Double
  } deriving (Eq, Show)

binanceBaseUrl :: String
binanceBaseUrl = "https://api.binance.com"

binanceTestnetBaseUrl :: String
binanceTestnetBaseUrl = "https://testnet.binance.vision"

newBinanceEnv :: String -> Maybe BS.ByteString -> Maybe BS.ByteString -> IO BinanceEnv
newBinanceEnv baseUrl apiKey apiSecret = do
  mgr <- newManager tlsManagerSettings
  pure BinanceEnv { beManager = mgr, beBaseUrl = baseUrl, beApiKey = apiKey, beApiSecret = apiSecret }

instance FromJSON Kline where
  parseJSON = withArray "Kline" $ \arr -> do
    if V.length arr < 6
      then fail "Kline array too short"
      else do
        openTime <- parseIndexInt64 0 arr
        closeTxt <- parseIndexText 4 arr
        close <- parseDoubleText closeTxt
        pure Kline { kOpenTime = openTime, kClose = close }
    where
      parseIndexInt64 i a =
        case a V.!? i of
          Nothing -> fail "Missing index"
          Just v -> parseJSON v
      parseIndexText i a =
        case a V.!? i of
          Nothing -> fail "Missing index"
          Just v -> parseJSON v

parseDoubleText :: Text -> AT.Parser Double
parseDoubleText t =
  case readMaybe (T.unpack t) of
    Just d -> pure d
    Nothing -> fail ("Failed to parse double: " ++ T.unpack t)

fetchKlines :: BinanceEnv -> String -> String -> Int -> IO [Kline]
fetchKlines env symbol interval limit = do
  req0 <- parseRequest (beBaseUrl env ++ "/api/v3/klines")
  let qs =
        renderSimpleQuery
          True
          [ ("symbol", BS.pack (map toUpperAscii symbol))
          , ("interval", BS.pack interval)
          , ("limit", BS.pack (show (max 1 (min 1000 limit))))
          ]
      req = req0 { method = "GET", queryString = qs }
  resp <- httpLbs req (beManager env)
  ensure2xx "klines" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode klines: " ++ e))
    Right ks -> pure ks

fetchCloses :: BinanceEnv -> String -> String -> Int -> IO [Double]
fetchCloses env symbol interval limit = do
  ks <- fetchKlines env symbol interval limit
  pure (map kClose ks)

data TickerPrice = TickerPrice { tpPrice :: Double }

instance FromJSON TickerPrice where
  parseJSON = withObject "TickerPrice" $ \o -> do
    pTxt <- o .: "price"
    p <- parseDoubleText pTxt
    pure (TickerPrice p)

fetchTickerPrice :: BinanceEnv -> String -> IO Double
fetchTickerPrice env symbol = do
  req0 <- parseRequest (beBaseUrl env ++ "/api/v3/ticker/price")
  let qs = renderSimpleQuery True [("symbol", BS.pack (map toUpperAscii symbol))]
      req = req0 { method = "GET", queryString = qs }
  resp <- httpLbs req (beManager env)
  ensure2xx "ticker/price" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode ticker price: " ++ e))
    Right (TickerPrice p) -> pure p

getTimestampMs :: IO Int64
getTimestampMs = do
  t <- getPOSIXTime
  pure (floor (t * 1000))

signQuery :: BS.ByteString -> BS.ByteString -> BS.ByteString
signQuery secret query =
  let mac :: HMAC SHA256
      mac = hmac secret query
      digest = hmacGetDigest mac
   in B16.encode (convert digest)

placeMarketOrder
  :: BinanceEnv
  -> BinanceOrderMode
  -> String      -- symbol
  -> OrderSide
  -> Maybe Double -- quantity (base)
  -> Maybe Double -- quoteOrderQty (quote)
  -> IO BL.ByteString
placeMarketOrder env mode symbol side quantity quoteOrderQty = do
  apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
  secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
  ts <- getTimestampMs
  case (quantity, quoteOrderQty) of
    (Nothing, Nothing) -> throwIO (userError "Provide quantity or quoteOrderQty for MARKET orders")
    _ -> pure ()

  let sideTxt = case side of { Buy -> "BUY"; Sell -> "SELL" }
      baseParams =
        [ ("symbol", BS.pack (map toUpperAscii symbol))
        , ("side", sideTxt)
        , ("type", "MARKET")
        , ("recvWindow", "5000")
        , ("timestamp", BS.pack (show ts))
        ]
      qtyParams =
        case (quantity, quoteOrderQty) of
          (Just q, _) -> [("quantity", renderDouble q)]
          (Nothing, Just qq) -> [("quoteOrderQty", renderDouble qq)]
          _ -> []

      params = baseParams ++ qtyParams
      queryToSign = renderSimpleQuery False params
      sig = signQuery secret queryToSign
      paramsSigned = params ++ [("signature", sig)]
      qs = renderSimpleQuery True paramsSigned

      path =
        case mode of
          OrderTest -> "/api/v3/order/test"
          OrderLive -> "/api/v3/order"

  req0 <- parseRequest (beBaseUrl env ++ path)
  let req =
        req0
          { method = "POST"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx (if mode == OrderTest then "order/test" else "order") resp
  pure (responseBody resp)

fetchFreeBalance :: BinanceEnv -> String -> IO Double
fetchFreeBalance env asset = do
  apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
  secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
  ts <- getTimestampMs

  let params =
        [ ("timestamp", BS.pack (show ts))
        , ("recvWindow", "5000")
        ]
      queryToSign = renderSimpleQuery False params
      sig = signQuery secret queryToSign
      paramsSigned = params ++ [("signature", sig)]
      qs = renderSimpleQuery True paramsSigned

  req0 <- parseRequest (beBaseUrl env ++ "/api/v3/account")
  let req =
        req0
          { method = "GET"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx "account" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode account: " ++ e))
    Right (Account balances) ->
      let sym = map toUpperAscii asset
          match b = map toUpperAscii (baAsset b) == sym
       in case filter match balances of
            (b:_) -> pure (baFree b)
            [] -> pure 0

data Account = Account [Balance]

data Balance = Balance
  { baAsset :: String
  , baFree :: Double
  }

instance FromJSON Account where
  parseJSON = withObject "Account" $ \o -> do
    bals <- o .: "balances"
    pure (Account bals)

instance FromJSON Balance where
  parseJSON = withObject "Balance" $ \o -> do
    asset <- o .: "asset"
    freeTxt <- o .: "free"
    free <- parseDoubleText freeTxt
    pure Balance { baAsset = asset, baFree = free }

renderDouble :: Double -> BS.ByteString
renderDouble x =
  -- Avoid scientific notation; Binance expects decimal strings.
  BS.pack (trimTrailingZeros (showFFloat (Just 8) x ""))

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
  if 'a' <= c && c <= 'z'
    then toEnum (fromEnum c - 32)
    else c

ensure2xx :: String -> Response BL.ByteString -> IO ()
ensure2xx label resp =
  let code = statusCode (responseStatus resp)
   in if code >= 200 && code < 300
        then pure ()
        else throwIO (userError (label ++ " HTTP " ++ show code ++ ": " ++ BS.unpack (BS.take 300 (BL.toStrict (responseBody resp)))))

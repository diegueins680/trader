{-# LANGUAGE OverloadedStrings #-}
module Trader.Binance
  ( BinanceEnv(..)
  , BinanceMarket(..)
  , BinanceOrderMode(..)
  , OrderSide(..)
  , Kline(..)
  , fetchTickerPrice
  , binanceBaseUrl
  , binanceTestnetBaseUrl
  , binanceFuturesBaseUrl
  , binanceFuturesTestnetBaseUrl
  , newBinanceEnv
  , fetchKlines
  , fetchCloses
  , getTimestampMs
  , signQuery
  , placeMarketOrder
  , fetchFreeBalance
  , fetchFuturesPositionAmt
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
  , beMarket :: BinanceMarket
  , beApiKey :: Maybe BS.ByteString
  , beApiSecret :: Maybe BS.ByteString
  }

data BinanceMarket = MarketSpot | MarketMargin | MarketFutures
  deriving (Eq, Show)

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

binanceFuturesBaseUrl :: String
binanceFuturesBaseUrl = "https://fapi.binance.com"

binanceFuturesTestnetBaseUrl :: String
binanceFuturesTestnetBaseUrl = "https://testnet.binancefuture.com"

newBinanceEnv :: BinanceMarket -> String -> Maybe BS.ByteString -> Maybe BS.ByteString -> IO BinanceEnv
newBinanceEnv market baseUrl apiKey apiSecret = do
  mgr <- newManager tlsManagerSettings
  pure BinanceEnv { beManager = mgr, beBaseUrl = baseUrl, beMarket = market, beApiKey = apiKey, beApiSecret = apiSecret }

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
  let path =
        case beMarket env of
          MarketSpot -> "/api/v3/klines"
          MarketMargin -> "/api/v3/klines"
          MarketFutures -> "/fapi/v1/klines"
  req0 <- parseRequest (beBaseUrl env ++ path)
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
  let path =
        case beMarket env of
          MarketSpot -> "/api/v3/ticker/price"
          MarketMargin -> "/api/v3/ticker/price"
          MarketFutures -> "/fapi/v1/ticker/price"
  req0 <- parseRequest (beBaseUrl env ++ path)
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
  -> Maybe Bool   -- reduceOnly (futures only)
  -> IO BL.ByteString
placeMarketOrder env mode symbol side quantity quoteOrderQty reduceOnly = do
  apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
  secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
  ts <- getTimestampMs
  let sideTxt = case side of { Buy -> "BUY"; Sell -> "SELL" }
      baseParams =
        [ ("symbol", BS.pack (map toUpperAscii symbol))
        , ("side", sideTxt)
        , ("type", "MARKET")
        , ("recvWindow", "5000")
        , ("timestamp", BS.pack (show ts))
        ]
      qtyParamsSpotOrMargin =
        case (quantity, quoteOrderQty) of
          (Just q, _) -> [("quantity", renderDouble q)]
          (Nothing, Just qq) -> [("quoteOrderQty", renderDouble qq)]
          _ -> []
      qtyParamsFutures =
        case quantity of
          Just q -> [("quantity", renderDouble q)]
          Nothing -> []
      reduceOnlyParams =
        case reduceOnly of
          Just True -> [("reduceOnly", "true")]
          _ -> []

      (path, label, params) =
        case beMarket env of
          MarketSpot ->
            let p =
                  case (quantity, quoteOrderQty) of
                    (Nothing, Nothing) -> error "Provide quantity or quoteOrderQty for MARKET orders"
                    _ -> baseParams ++ qtyParamsSpotOrMargin
             in ( if mode == OrderTest then "/api/v3/order/test" else "/api/v3/order"
                , if mode == OrderTest then "order/test" else "order"
                , p
                )
          MarketMargin ->
            let p =
                  case (quantity, quoteOrderQty) of
                    (Nothing, Nothing) -> error "Provide quantity or quoteOrderQty for MARKET orders"
                    _ -> baseParams ++ qtyParamsSpotOrMargin
             in case mode of
                  OrderTest -> error "Margin does not support order test; rerun with --binance-live"
                  OrderLive -> ("/sapi/v1/margin/order", "margin/order", p)
          MarketFutures ->
            let p =
                  case quantity of
                    Nothing -> error "Futures MARKET orders require --order-quantity (or compute it from --order-quote in the caller)"
                    Just _ -> baseParams ++ qtyParamsFutures ++ reduceOnlyParams
             in ( if mode == OrderTest then "/fapi/v1/order/test" else "/fapi/v1/order"
                , if mode == OrderTest then "futures/order/test" else "futures/order"
                , p
                )

      queryToSign = renderSimpleQuery False params
      sig = signQuery secret queryToSign
      paramsSigned = params ++ [("signature", sig)]
      qs = renderSimpleQuery True paramsSigned

  req0 <- parseRequest (beBaseUrl env ++ path)
  let req =
        req0
          { method = "POST"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx label resp
  pure (responseBody resp)

fetchFreeBalance :: BinanceEnv -> String -> IO Double
fetchFreeBalance env asset = do
  case beMarket env of
    MarketFutures -> throwIO (userError "fetchFreeBalance is not supported for futures; use fetchFuturesPositionAmt")
    _ -> pure ()
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

  let path =
        case beMarket env of
          MarketSpot -> "/api/v3/account"
          MarketMargin -> "/sapi/v1/margin/account"
          MarketFutures -> "/api/v3/account"
  req0 <- parseRequest (beBaseUrl env ++ path)
  let req =
        req0
          { method = "GET"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx (if beMarket env == MarketMargin then "margin/account" else "account") resp
  case beMarket env of
    MarketSpot ->
      case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode account: " ++ e))
        Right (Account balances) ->
          let sym = map toUpperAscii asset
              match b = map toUpperAscii (baAsset b) == sym
           in case filter match balances of
                (b:_) -> pure (baFree b)
                [] -> pure 0
    MarketMargin ->
      case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode margin account: " ++ e))
        Right (MarginAccount balances) ->
          let sym = map toUpperAscii asset
              match b = map toUpperAscii (mbaAsset b) == sym
           in case filter match balances of
                (b:_) -> pure (mbaNetAsset b)
                [] -> pure 0
    MarketFutures -> pure 0

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

data MarginAccount = MarginAccount [MarginBalance]

data MarginBalance = MarginBalance
  { mbaAsset :: String
  , mbaNetAsset :: Double
  }

instance FromJSON MarginAccount where
  parseJSON = withObject "MarginAccount" $ \o -> do
    bals <- o .: "userAssets"
    pure (MarginAccount bals)

instance FromJSON MarginBalance where
  parseJSON = withObject "MarginBalance" $ \o -> do
    asset <- o .: "asset"
    netTxt <- o .: "netAsset"
    net <- parseDoubleText netTxt
    pure MarginBalance { mbaAsset = asset, mbaNetAsset = net }

fetchFuturesPositionAmt :: BinanceEnv -> String -> IO Double
fetchFuturesPositionAmt env symbol = do
  if beMarket env /= MarketFutures
    then throwIO (userError "fetchFuturesPositionAmt requires MarketFutures")
    else pure ()
  apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
  secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
  ts <- getTimestampMs

  let params =
        [ ("symbol", BS.pack (map toUpperAscii symbol))
        , ("timestamp", BS.pack (show ts))
        , ("recvWindow", "5000")
        ]
      queryToSign = renderSimpleQuery False params
      sig = signQuery secret queryToSign
      paramsSigned = params ++ [("signature", sig)]
      qs = renderSimpleQuery True paramsSigned

  req0 <- parseRequest (beBaseUrl env ++ "/fapi/v2/positionRisk")
  let req =
        req0
          { method = "GET"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx "futures/positionRisk" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode futures positionRisk: " ++ e))
    Right positions ->
      let sym = map toUpperAscii symbol
          match p = map toUpperAscii (fpSymbol p) == sym
       in case filter match positions of
            (p:_) -> pure (fpPositionAmt p)
            [] -> pure 0

data FuturesPosition = FuturesPosition
  { fpSymbol :: String
  , fpPositionAmt :: Double
  }

instance FromJSON FuturesPosition where
  parseJSON = withObject "FuturesPosition" $ \o -> do
    sym <- o .: "symbol"
    amtTxt <- o .: "positionAmt"
    amt <- parseDoubleText amtTxt
    pure FuturesPosition { fpSymbol = sym, fpPositionAmt = amt }

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

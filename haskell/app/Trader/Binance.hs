{-# LANGUAGE OverloadedStrings #-}
module Trader.Binance
  ( BinanceEnv(..)
  , BinanceMarket(..)
  , BinanceOrderMode(..)
  , OrderSide(..)
  , Kline(..)
  , Step(..)
  , SymbolFilters(..)
  , fetchTickerPrice
  , binanceBaseUrl
  , binanceTestnetBaseUrl
  , binanceFuturesBaseUrl
  , binanceFuturesTestnetBaseUrl
  , newBinanceEnv
  , fetchKlines
  , fetchCloses
  , fetchSymbolFilters
  , quantizeDown
  , getTimestampMs
  , signQuery
  , placeMarketOrder
  , placeFuturesTriggerMarketOrder
  , fetchOrderByClientId
  , fetchFreeBalance
  , fetchFuturesAvailableBalance
  , fetchFuturesPositionAmt
  , cancelFuturesOpenOrdersByClientPrefix
  , createListenKey
  , keepAliveListenKey
  , closeListenKey
  ) where

import Control.Applicative ((<|>))
import Control.Exception (SomeException, throwIO, try)
import Data.Aeson (FromJSON(..), eitherDecode, withArray, (.:), withObject)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Base16 as B16
import Data.Char (isSpace)
import Data.Int (Int64)
import Data.List (foldl', isPrefixOf)
import Data.Maybe (fromMaybe, listToMaybe)
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
  , kOpen :: !Double
  , kHigh :: !Double
  , kLow :: !Double
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
        openTxt <- parseIndexText 1 arr
        highTxt <- parseIndexText 2 arr
        lowTxt <- parseIndexText 3 arr
        closeTxt <- parseIndexText 4 arr
        open <- parseDoubleText openTxt
        high <- parseDoubleText highTxt
        low <- parseDoubleText lowTxt
        close <- parseDoubleText closeTxt
        pure Kline { kOpenTime = openTime, kOpen = open, kHigh = high, kLow = low, kClose = close }
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

-- Binance symbol filters (exchangeInfo)

data Step = Step
  { stepScale :: !Integer
  , stepInt :: !Integer
  , stepText :: !Text
  } deriving (Eq, Show)

mkStep :: Text -> Maybe Step
mkStep raw =
  let t = T.strip raw
      s = T.unpack t
   in case break (== '.') s of
        (a, "") -> do
          ai <- readMaybeInteger a
          if ai <= 0 then Nothing else Just Step { stepScale = 1, stepInt = ai, stepText = t }
        (a, '.':b) -> do
          ai <- readMaybeInteger a
          bi <- readMaybeInteger b
          let scale = 10 ^ length b
              val = ai * scale + bi
          if val <= 0 then Nothing else Just Step { stepScale = scale, stepInt = val, stepText = t }
        _ -> Nothing
  where
    readMaybeInteger :: String -> Maybe Integer
    readMaybeInteger "" = Just 0
    readMaybeInteger xs = readMaybe xs

quantizeDown :: Step -> Double -> Double
quantizeDown st x
  | x <= 0 = 0
  | otherwise =
      let scaleD = fromIntegral (stepScale st) :: Double
          scaled = floor (x * scaleD + 1e-9) :: Integer
          stepI = stepInt st
          q = (scaled `div` stepI) * stepI
       in fromIntegral q / scaleD

data SymbolFilters = SymbolFilters
  { sfLotMinQty :: !(Maybe Double)
  , sfLotMaxQty :: !(Maybe Double)
  , sfLotStepSize :: !(Maybe Step)
  , sfMarketMinQty :: !(Maybe Double)
  , sfMarketMaxQty :: !(Maybe Double)
  , sfMarketStepSize :: !(Maybe Step)
  , sfMinNotional :: !(Maybe Double)
  , sfTickSize :: !(Maybe Step)
  } deriving (Eq, Show)

emptySymbolFilters :: SymbolFilters
emptySymbolFilters =
  SymbolFilters
    { sfLotMinQty = Nothing
    , sfLotMaxQty = Nothing
    , sfLotStepSize = Nothing
    , sfMarketMinQty = Nothing
    , sfMarketMaxQty = Nothing
    , sfMarketStepSize = Nothing
    , sfMinNotional = Nothing
    , sfTickSize = Nothing
    }

data ExchangeInfo = ExchangeInfo [ExchangeSymbol]

data ExchangeSymbol = ExchangeSymbol
  { esSymbol :: !String
  , esFilters :: ![Aeson.Object]
  }

instance FromJSON ExchangeInfo where
  parseJSON = withObject "ExchangeInfo" $ \o -> do
    syms <- o .: "symbols"
    pure (ExchangeInfo syms)

instance FromJSON ExchangeSymbol where
  parseJSON = withObject "ExchangeSymbol" $ \o -> do
    sym <- o .: "symbol"
    flt <- o .: "filters"
    pure ExchangeSymbol { esSymbol = sym, esFilters = flt }

fetchSymbolFilters :: BinanceEnv -> String -> IO SymbolFilters
fetchSymbolFilters env symbol = do
  let path =
        case beMarket env of
          MarketSpot -> "/api/v3/exchangeInfo"
          MarketMargin -> "/api/v3/exchangeInfo"
          MarketFutures -> "/fapi/v1/exchangeInfo"
  req0 <- parseRequest (beBaseUrl env ++ path)
  let qs = renderSimpleQuery True [("symbol", BS.pack (map toUpperAscii symbol))]
      req = req0 { method = "GET", queryString = qs }
  resp <- httpLbs req (beManager env)
  ensure2xx "exchangeInfo" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode exchangeInfo: " ++ e))
    Right (ExchangeInfo syms) ->
      case listToMaybe [s | s <- syms, map toUpperAscii (esSymbol s) == map toUpperAscii symbol] of
        Nothing -> throwIO (userError ("exchangeInfo: symbol not found: " ++ symbol))
        Just s -> pure (parseSymbolFilters (esFilters s))

parseSymbolFilters :: [Aeson.Object] -> SymbolFilters
parseSymbolFilters objs =
  foldl' apply emptySymbolFilters objs
  where
    apply acc o =
      case AT.parseMaybe (Aeson..: "filterType") o :: Maybe Text of
        Nothing -> acc
        Just ft ->
          case ft of
            "LOT_SIZE" ->
              acc
                { sfLotMinQty = parseDField o "minQty" <|> sfLotMinQty acc
                , sfLotMaxQty = parseDField o "maxQty" <|> sfLotMaxQty acc
                , sfLotStepSize = parseStepField o "stepSize" <|> sfLotStepSize acc
                }
            "MARKET_LOT_SIZE" ->
              acc
                { sfMarketMinQty = parseDField o "minQty" <|> sfMarketMinQty acc
                , sfMarketMaxQty = parseDField o "maxQty" <|> sfMarketMaxQty acc
                , sfMarketStepSize = parseStepField o "stepSize" <|> sfMarketStepSize acc
                }
            "MIN_NOTIONAL" ->
              acc
                { sfMinNotional = parseDField o "minNotional" <|> sfMinNotional acc
                }
            "PRICE_FILTER" ->
              acc
                { sfTickSize = parseStepField o "tickSize" <|> sfTickSize acc
                }
            _ -> acc

    parseDField o k = do
      t <- AT.parseMaybe (Aeson..: k) o :: Maybe Text
      readMaybe (T.unpack t)

    parseStepField o k = do
      t <- AT.parseMaybe (Aeson..: k) o :: Maybe Text
      mkStep t

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
  -> Maybe String -- newClientOrderId (optional; idempotency)
  -> IO BL.ByteString
placeMarketOrder env mode symbol side quantity quoteOrderQty reduceOnly mClientOrderId = do
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
      clientIdParam =
        case mClientOrderId of
          Nothing -> []
          Just cid | null (trim cid) -> []
          Just cid -> [("newClientOrderId", BS.pack (trim cid))]
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

  (path, label, params) <-
    case beMarket env of
      MarketSpot -> do
        p <-
          case (quantity, quoteOrderQty) of
            (Nothing, Nothing) -> throwIO (userError "Provide quantity or quoteOrderQty for MARKET orders")
            _ -> pure (baseParams ++ qtyParamsSpotOrMargin ++ clientIdParam)
        pure
          ( if mode == OrderTest then "/api/v3/order/test" else "/api/v3/order"
          , if mode == OrderTest then "order/test" else "order"
          , p
          )
      MarketMargin -> do
        p <-
          case (quantity, quoteOrderQty) of
            (Nothing, Nothing) -> throwIO (userError "Provide quantity or quoteOrderQty for MARKET orders")
            _ -> pure (baseParams ++ qtyParamsSpotOrMargin ++ clientIdParam)
        case mode of
          OrderTest -> throwIO (userError "Margin does not support order test; rerun with --binance-live")
          OrderLive -> pure ("/sapi/v1/margin/order", "margin/order", p)
      MarketFutures -> do
        p <-
          case quantity of
            Nothing -> throwIO (userError "Futures MARKET orders require --order-quantity (or compute it from --order-quote in the caller)")
            Just _ -> pure (baseParams ++ qtyParamsFutures ++ reduceOnlyParams ++ clientIdParam)
        pure
          ( if mode == OrderTest then "/fapi/v1/order/test" else "/fapi/v1/order"
          , if mode == OrderTest then "futures/order/test" else "futures/order"
          , p
          )

  let queryToSign = renderSimpleQuery False params
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

placeFuturesTriggerMarketOrder
  :: BinanceEnv
  -> BinanceOrderMode
  -> String      -- symbol
  -> OrderSide
  -> String      -- type (e.g., STOP_MARKET, TAKE_PROFIT_MARKET)
  -> Double      -- stopPrice
  -> Maybe String -- newClientOrderId (optional; idempotency)
  -> IO BL.ByteString
placeFuturesTriggerMarketOrder env mode symbol side orderType stopPrice mClientOrderId = do
  if beMarket env /= MarketFutures
    then throwIO (userError "placeFuturesTriggerMarketOrder requires MarketFutures")
    else pure ()
  if stopPrice <= 0
    then throwIO (userError "stopPrice must be > 0")
    else pure ()
  let orderType' = trim orderType
  if null orderType'
    then throwIO (userError "orderType must be non-empty")
    else pure ()
  apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
  secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
  ts <- getTimestampMs

  let sideTxt = case side of { Buy -> "BUY"; Sell -> "SELL" }
      baseParams =
        [ ("symbol", BS.pack (map toUpperAscii symbol))
        , ("side", sideTxt)
        , ("type", BS.pack orderType')
        , ("stopPrice", renderDouble stopPrice)
        , ("closePosition", "true")
        , ("recvWindow", "5000")
        , ("timestamp", BS.pack (show ts))
        ]
      clientIdParam =
        case mClientOrderId of
          Nothing -> []
          Just cid | null (trim cid) -> []
          Just cid -> [("newClientOrderId", BS.pack (trim cid))]
      params = baseParams ++ clientIdParam

      (path, label) =
        if mode == OrderTest
          then ("/fapi/v1/order/test", "futures/order/test(trigger)")
          else ("/fapi/v1/order", "futures/order(trigger)")

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

fetchOrderByClientId :: BinanceEnv -> String -> String -> IO BL.ByteString
fetchOrderByClientId env symbol clientOrderId = do
  apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
  secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
  ts <- getTimestampMs

  let (path, label) =
        case beMarket env of
          MarketSpot -> ("/api/v3/order", "order/get")
          MarketMargin -> ("/sapi/v1/margin/order", "margin/order/get")
          MarketFutures -> ("/fapi/v1/order", "futures/order/get")

      params =
        [ ("symbol", BS.pack (map toUpperAscii symbol))
        , ("origClientOrderId", BS.pack (trim clientOrderId))
        , ("timestamp", BS.pack (show ts))
        , ("recvWindow", "5000")
        ]

      queryToSign = renderSimpleQuery False params
      sig = signQuery secret queryToSign
      paramsSigned = params ++ [("signature", sig)]
      qs = renderSimpleQuery True paramsSigned

  req0 <- parseRequest (beBaseUrl env ++ path)
  let req =
        req0
          { method = "GET"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx label resp
  pure (responseBody resp)

data FuturesOpenOrder = FuturesOpenOrder
  { fooClientOrderId :: String
  } deriving (Eq, Show)

instance FromJSON FuturesOpenOrder where
  parseJSON = withObject "FuturesOpenOrder" $ \o -> do
    cid <- o .: "clientOrderId"
    pure FuturesOpenOrder { fooClientOrderId = cid }

fetchFuturesOpenOrders :: BinanceEnv -> String -> IO [FuturesOpenOrder]
fetchFuturesOpenOrders env symbol = do
  if beMarket env /= MarketFutures
    then throwIO (userError "fetchFuturesOpenOrders requires MarketFutures")
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

  req0 <- parseRequest (beBaseUrl env ++ "/fapi/v1/openOrders")
  let req =
        req0
          { method = "GET"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx "futures/openOrders" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode futures openOrders: " ++ e))
    Right orders -> pure orders

cancelFuturesOrderByClientId :: BinanceEnv -> String -> String -> IO BL.ByteString
cancelFuturesOrderByClientId env symbol clientOrderId = do
  if beMarket env /= MarketFutures
    then throwIO (userError "cancelFuturesOrderByClientId requires MarketFutures")
    else pure ()
  apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
  secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
  ts <- getTimestampMs

  let params =
        [ ("symbol", BS.pack (map toUpperAscii symbol))
        , ("origClientOrderId", BS.pack (trim clientOrderId))
        , ("timestamp", BS.pack (show ts))
        , ("recvWindow", "5000")
        ]
      queryToSign = renderSimpleQuery False params
      sig = signQuery secret queryToSign
      paramsSigned = params ++ [("signature", sig)]
      qs = renderSimpleQuery True paramsSigned

  req0 <- parseRequest (beBaseUrl env ++ "/fapi/v1/order")
  let req =
        req0
          { method = "DELETE"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx "futures/order/cancel" resp
  pure (responseBody resp)

cancelFuturesOpenOrdersByClientPrefix :: BinanceEnv -> String -> String -> IO Int
cancelFuturesOpenOrdersByClientPrefix env symbol prefix0 = do
  let prefix = trim prefix0
  if null prefix
    then pure 0
    else do
      orders <- fetchFuturesOpenOrders env symbol
      let targetClientIds =
            [ fooClientOrderId o
            | o <- orders
            , prefix `isPrefixOf` fooClientOrderId o
            ]
      results <-
        mapM
          (\cid -> try (cancelFuturesOrderByClientId env symbol cid) :: IO (Either SomeException BL.ByteString))
          targetClientIds
      pure (length [() | Right _ <- results])

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

fetchFuturesAvailableBalance :: BinanceEnv -> String -> IO Double
fetchFuturesAvailableBalance env asset = do
  if beMarket env /= MarketFutures
    then throwIO (userError "fetchFuturesAvailableBalance requires MarketFutures")
    else pure ()
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

  req0 <- parseRequest (beBaseUrl env ++ "/fapi/v2/balance")
  let req =
        req0
          { method = "GET"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx "futures/balance" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode futures balance: " ++ e))
    Right bals ->
      let sym = map toUpperAscii asset
          match b = map toUpperAscii (fbAsset b) == sym
       in case filter match bals of
            (b:_) -> pure (fbAvailableBalance b)
            [] -> pure 0

data FuturesBalance = FuturesBalance
  { fbAsset :: String
  , fbAvailableBalance :: Double
  }

instance FromJSON FuturesBalance where
  parseJSON = withObject "FuturesBalance" $ \o -> do
    sym <- o .: "asset"
    availTxt <- o .: "availableBalance"
    avail <- parseDoubleText availTxt
    pure FuturesBalance { fbAsset = sym, fbAvailableBalance = avail }

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
        else
          let body = responseBody resp
              retryAfter =
                case lookup "Retry-After" (responseHeaders resp) of
                  Nothing -> ""
                  Just v ->
                    let s = trim (BS.unpack v)
                     in if null s then "" else " (Retry-After: " ++ s ++ ")"
              details =
                case eitherDecode body :: Either String BinanceErrorBody of
                  Right be ->
                    let msg = fromMaybe "" (bebMsg be)
                        codeLabel =
                          case bebCode be of
                            Nothing -> ""
                            Just c -> "Binance code " ++ show c ++ ": "
                     in codeLabel ++ msg
                  Left _ -> BS.unpack (BS.take 300 (BL.toStrict body))
           in throwIO (userError (label ++ " HTTP " ++ show code ++ retryAfter ++ ": " ++ details))

data BinanceErrorBody = BinanceErrorBody
  { bebCode :: !(Maybe Int)
  , bebMsg :: !(Maybe String)
  } deriving (Eq, Show)

instance FromJSON BinanceErrorBody where
  parseJSON = withObject "BinanceErrorBody" $ \o -> do
    code <- o Aeson..:? "code"
    msg <- o Aeson..:? "msg"
    pure BinanceErrorBody { bebCode = code, bebMsg = msg }

data ListenKeyResponse = ListenKeyResponse { lkrListenKey :: String }

instance FromJSON ListenKeyResponse where
  parseJSON = withObject "ListenKeyResponse" $ \o -> do
    k <- o .: "listenKey"
    pure (ListenKeyResponse k)

userDataStreamPath :: BinanceMarket -> String
userDataStreamPath market =
  case market of
    MarketSpot -> "/api/v3/userDataStream"
    MarketMargin -> "/sapi/v1/userDataStream"
    MarketFutures -> "/fapi/v1/listenKey"

createListenKey :: BinanceEnv -> IO String
createListenKey env = do
  apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
  let path = userDataStreamPath (beMarket env)
  req0 <- parseRequest (beBaseUrl env ++ path)
  let req =
        req0
          { method = "POST"
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx "listenKey" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode listenKey: " ++ e))
    Right (ListenKeyResponse k) -> pure k

keepAliveListenKey :: BinanceEnv -> String -> IO ()
keepAliveListenKey env listenKey = do
  apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
  let path = userDataStreamPath (beMarket env)
      qs = renderSimpleQuery True [("listenKey", BS.pack listenKey)]
  req0 <- parseRequest (beBaseUrl env ++ path)
  let req =
        req0
          { method = "PUT"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx "listenKey/keepAlive" resp

closeListenKey :: BinanceEnv -> String -> IO ()
closeListenKey env listenKey = do
  apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
  let path = userDataStreamPath (beMarket env)
      qs = renderSimpleQuery True [("listenKey", BS.pack listenKey)]
  req0 <- parseRequest (beBaseUrl env ++ path)
  let req =
        req0
          { method = "DELETE"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- httpLbs req (beManager env)
  ensure2xx "listenKey/close" resp

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

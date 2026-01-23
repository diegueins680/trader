{-# LANGUAGE OverloadedStrings #-}
module Trader.Binance
  ( BinanceEnv(..)
  , BinanceLog(..)
  , BinanceMarket(..)
  , BinanceOrderMode(..)
  , OrderSide(..)
  , BinanceTrade(..)
  , Kline(..)
  , Step(..)
  , SymbolFilters(..)
  , Ticker24h(..)
  , BinanceOpenOrder(..)
  , FuturesPositionRisk(..)
  , fetchTickerPrice
  , fetchTicker24hPrice
  , fetchFuturesMarkPrice
  , fetchTickers24h
  , fetchTopSymbolsByQuoteVolume
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
  , fetchAccountTrades
  , fetchFreeBalance
  , fetchFuturesAvailableBalance
  , fetchFuturesPositionAmt
  , fetchFuturesPositionRisks
  , fetchOpenOrders
  , cancelFuturesOpenOrdersByClientPrefix
  , createListenKey
  , keepAliveListenKey
  , closeListenKey
  ) where

import Control.Applicative ((<|>))
import Control.Exception (SomeException, displayException, fromException, throwIO, try)
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode, object, withArray, (.:), (.=), withObject)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Base16 as B16
import Data.Char (isSpace)
import Data.Int (Int64)
import Data.List (foldl', isPrefixOf, isSuffixOf, sortBy)
import Data.Maybe (fromMaybe, listToMaybe)
import Data.Ord (comparing)
import Data.Text (Text)
import qualified Data.Text as T
import Data.Text.Encoding (decodeUtf8With)
import Data.Text.Encoding.Error (lenientDecode)
import Data.Time.Clock (NominalDiffTime)
import qualified Data.Vector as V
import Network.HTTP.Client
import Network.HTTP.Types.URI (parseQuery, renderSimpleQuery)
import Network.HTTP.Types.Status (statusCode)
import Crypto.MAC.HMAC (HMAC, hmac, hmacGetDigest)
import Crypto.Hash.Algorithms (SHA256)
import Data.ByteArray (convert)
import Data.Time.Clock.POSIX (getPOSIXTime)
import Numeric (showFFloat)
import Text.Read (readMaybe)
import Trader.Text (normalizeKey)
import Trader.Cache (TtlCache, fetchWithCache, newTtlCache)
import Trader.Http (defaultRetryConfig, httpLbsWithRetry, newHttpManager)
import System.IO.Unsafe (unsafePerformIO)

data BinanceEnv = BinanceEnv
  { beManager :: Manager
  , beBaseUrl :: String
  , beMarket :: BinanceMarket
  , beApiKey :: Maybe BS.ByteString
  , beApiSecret :: Maybe BS.ByteString
  , beLogger :: Maybe (BinanceLog -> IO ())
  }

data BinanceLog = BinanceLog
  { blAtMs :: !Int64
  , blMarket :: !BinanceMarket
  , blLabel :: !Text
  , blMethod :: !Text
  , blPath :: !Text
  , blParams :: ![(Text, Text)]
  , blStatus :: !(Maybe Int)
  , blLatencyMs :: !Int
  , blOk :: !Bool
  , blError :: !(Maybe Text)
  } deriving (Eq, Show)

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

data BinanceTrade = BinanceTrade
  { btSymbol :: !String
  , btTradeId :: !Int64
  , btOrderId :: !(Maybe Int64)
  , btPrice :: !Double
  , btQty :: !Double
  , btQuoteQty :: !Double
  , btCommission :: !(Maybe Double)
  , btCommissionAsset :: !(Maybe String)
  , btTime :: !Int64
  , btIsBuyer :: !(Maybe Bool)
  , btIsMaker :: !(Maybe Bool)
  , btSide :: !(Maybe String)
  , btPositionSide :: !(Maybe String)
  , btRealizedPnl :: !(Maybe Double)
  } deriving (Eq, Show)

instance FromJSON BinanceTrade where
  parseJSON = withObject "BinanceTrade" $ \o -> do
    sym <- o .: "symbol"
    tradeId <- o .: "id"
    orderId <- o AT..:? "orderId"
    price <- parseDoubleField o "price"
    qty <- parseDoubleField o "qty"
    quoteQtyRaw <- parseMaybeDoubleField o "quoteQty"
    commission <- parseMaybeDoubleField o "commission"
    commissionAsset <- o AT..:? "commissionAsset"
    ts <- o .: "time"
    isBuyerRaw <- o AT..:? "isBuyer"
    buyerRaw <- o AT..:? "buyer"
    isMakerRaw <- o AT..:? "isMaker"
    makerRaw <- o AT..:? "maker"
    sideRaw <- o AT..:? "side"
    positionSide <- o AT..:? "positionSide"
    realizedPnl <- parseMaybeDoubleField o "realizedPnl"
    let isBuyer = isBuyerRaw <|> buyerRaw
        isMaker = isMakerRaw <|> makerRaw
        sideDerived =
          case sideRaw of
            Just s | not (null (trim s)) -> Just (map toUpperAscii s)
            _ ->
              case isBuyer of
                Just True -> Just "BUY"
                Just False -> Just "SELL"
                Nothing -> Nothing
        quoteQty =
          case quoteQtyRaw of
            Just q -> q
            Nothing -> price * qty
    pure
      BinanceTrade
        { btSymbol = sym
        , btTradeId = tradeId
        , btOrderId = orderId
        , btPrice = price
        , btQty = qty
        , btQuoteQty = quoteQty
        , btCommission = commission
        , btCommissionAsset = commissionAsset
        , btTime = ts
        , btIsBuyer = isBuyer
        , btIsMaker = isMaker
        , btSide = sideDerived
        , btPositionSide = positionSide
        , btRealizedPnl = realizedPnl
        }

instance ToJSON BinanceTrade where
  toJSON t =
    object
      [ "symbol" .= btSymbol t
      , "tradeId" .= btTradeId t
      , "orderId" .= btOrderId t
      , "price" .= btPrice t
      , "qty" .= btQty t
      , "quoteQty" .= btQuoteQty t
      , "commission" .= btCommission t
      , "commissionAsset" .= btCommissionAsset t
      , "time" .= btTime t
      , "isBuyer" .= btIsBuyer t
      , "isMaker" .= btIsMaker t
      , "side" .= btSide t
      , "positionSide" .= btPositionSide t
      , "realizedPnl" .= btRealizedPnl t
      ]

binanceBaseUrl :: String
binanceBaseUrl = "https://api.binance.com"

binanceTestnetBaseUrl :: String
binanceTestnetBaseUrl = "https://testnet.binance.vision"

binanceFuturesBaseUrl :: String
binanceFuturesBaseUrl = "https://fapi.binance.com"

binanceFuturesTestnetBaseUrl :: String
binanceFuturesTestnetBaseUrl = "https://testnet.binancefuture.com"

{-# NOINLINE binanceTickersCache #-}
binanceTickersCache :: TtlCache String [Ticker24h]
binanceTickersCache = unsafePerformIO newTtlCache

{-# NOINLINE binanceExchangeInfoCache #-}
binanceExchangeInfoCache :: TtlCache String SymbolFilters
binanceExchangeInfoCache = unsafePerformIO newTtlCache

{-# NOINLINE binanceKlinesCache #-}
binanceKlinesCache :: TtlCache String [Kline]
binanceKlinesCache = unsafePerformIO newTtlCache

binanceTickersFreshTtl :: NominalDiffTime
binanceTickersFreshTtl = 10

binanceTickersStaleTtl :: NominalDiffTime
binanceTickersStaleTtl = 60

binanceExchangeInfoFreshTtl :: NominalDiffTime
binanceExchangeInfoFreshTtl = 600

binanceExchangeInfoStaleTtl :: NominalDiffTime
binanceExchangeInfoStaleTtl = 3600

binanceKlinesFreshTtl :: NominalDiffTime
binanceKlinesFreshTtl = 5

binanceKlinesStaleTtl :: NominalDiffTime
binanceKlinesStaleTtl = 60

newBinanceEnv :: BinanceMarket -> String -> Maybe BS.ByteString -> Maybe BS.ByteString -> IO BinanceEnv
newBinanceEnv market baseUrl apiKey apiSecret = do
  mgr <- newHttpManager
  pure BinanceEnv { beManager = mgr, beBaseUrl = baseUrl, beMarket = market, beApiKey = apiKey, beApiSecret = apiSecret, beLogger = Nothing }

binanceHttp :: BinanceEnv -> String -> Request -> IO (Response BL.ByteString)
binanceHttp env label req = do
  t0 <- getTimestampMs
  respOrErr <- try (httpLbsWithRetry defaultRetryConfig Nothing (beManager env) req) :: IO (Either SomeException (Response BL.ByteString))
  t1 <- getTimestampMs
  let latencyMs = max 0 (fromIntegral (t1 - t0) :: Int)
      methodTxt = decodeUtf8With lenientDecode (method req)
      pathTxt = decodeUtf8With lenientDecode (path req)
      params = sanitizeQueryParams (queryString req)
      labelTxt = T.pack label
  case respOrErr of
    Left ex -> do
      let errMsg = binanceExceptionSummary ex
      logBinanceRequest env (BinanceLog t1 (beMarket env) labelTxt methodTxt pathTxt params Nothing latencyMs False (Just errMsg))
      throwIO ex
    Right resp -> do
      let code = statusCode (responseStatus resp)
          ok = code >= 200 && code < 300
          errMsg = if ok then Nothing else Just (binanceErrorSummary resp)
      logBinanceRequest env (BinanceLog t1 (beMarket env) labelTxt methodTxt pathTxt params (Just code) latencyMs ok errMsg)
      pure resp

logBinanceRequest :: BinanceEnv -> BinanceLog -> IO ()
logBinanceRequest env entry =
  case beLogger env of
    Nothing -> pure ()
    Just logger -> logger entry

binanceExceptionSummary :: SomeException -> Text
binanceExceptionSummary ex =
  case fromException ex of
    Just (HttpExceptionRequest _ content) -> T.pack (show content)
    Just (InvalidUrlException url reason) -> T.pack ("InvalidUrlException " ++ url ++ ": " ++ reason)
    Nothing -> T.pack (displayException ex)

sanitizeQueryParams :: BS.ByteString -> [(Text, Text)]
sanitizeQueryParams raw =
  let toText = decodeUtf8With lenientDecode
      raw' = BS.dropWhile (== '?') raw
      redactKeys = ["signature", "listenkey"]
      redactIfNeeded key val =
        let keyLower = T.toLower key
         in if keyLower `elem` redactKeys then "<redacted>" else val
   in [ (keyTxt, redactIfNeeded keyTxt valTxt)
      | (k, mv) <- parseQuery raw'
      , let keyTxt = toText k
      , let valTxt = maybe "" toText mv
      ]

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

parseDoubleField :: Aeson.Object -> Text -> AT.Parser Double
parseDoubleField o k = do
  t <- o .: AK.fromText k
  parseDoubleText t

parseMaybeDoubleField :: Aeson.Object -> Text -> AT.Parser (Maybe Double)
parseMaybeDoubleField o k = do
  mt <- o AT..:? AK.fromText k
  case mt of
    Nothing -> pure Nothing
    Just t -> Just <$> parseDoubleText t

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
  let key = beBaseUrl env ++ ":" ++ show (beMarket env) ++ ":" ++ map toUpperAscii symbol
  fetchWithCache binanceExchangeInfoCache binanceExchangeInfoFreshTtl binanceExchangeInfoStaleTtl key $ do
    let path =
          case beMarket env of
            MarketSpot -> "/api/v3/exchangeInfo"
            MarketMargin -> "/api/v3/exchangeInfo"
            MarketFutures -> "/fapi/v1/exchangeInfo"
    req0 <- parseRequest (beBaseUrl env ++ path)
    let qs = renderSimpleQuery True [("symbol", BS.pack (map toUpperAscii symbol))]
        req = req0 { method = "GET", queryString = qs }
    resp <- binanceHttp env "exchangeInfo" req
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
                { sfMinNotional = parseDField o "minNotional" <|> parseDField o "notional" <|> sfMinNotional acc
                }
            "NOTIONAL" ->
              acc
                { sfMinNotional = parseDField o "minNotional" <|> parseDField o "notional" <|> sfMinNotional acc
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
  let key =
        beBaseUrl env
          ++ ":" ++ show (beMarket env)
          ++ ":" ++ map toUpperAscii symbol
          ++ ":" ++ interval
          ++ ":" ++ show limit
  fetchWithCache binanceKlinesCache binanceKlinesFreshTtl binanceKlinesStaleTtl key $
    fetchKlinesRaw env symbol interval limit

fetchKlinesRaw :: BinanceEnv -> String -> String -> Int -> IO [Kline]
fetchKlinesRaw env symbol interval limit = do
  let maxPerRequest = 1000
      wanted = max 1 limit
      path =
        case beMarket env of
          MarketSpot -> "/api/v3/klines"
          MarketMargin -> "/api/v3/klines"
          MarketFutures -> "/fapi/v1/klines"
      symbolKey = BS.pack (map toUpperAscii symbol)
      fetchBatch :: Maybe Int64 -> Int -> IO [Kline]
      fetchBatch mEnd batchLimit = do
        req0 <- parseRequest (beBaseUrl env ++ path)
        let qsBase =
              [ ("symbol", symbolKey)
              , ("interval", BS.pack interval)
              , ("limit", BS.pack (show (max 1 (min maxPerRequest batchLimit))))
              ]
            qs =
              case mEnd of
                Nothing -> qsBase
                Just endTime -> qsBase ++ [("endTime", BS.pack (show endTime))]
            req = req0 { method = "GET", queryString = renderSimpleQuery True qs }
        resp <- binanceHttp env "klines" req
        ensure2xx "klines" resp
        case eitherDecode (responseBody resp) of
          Left e -> throwIO (userError ("Failed to decode klines: " ++ e))
          Right ks -> pure ks

      go :: Int -> Maybe Int64 -> [Kline] -> IO [Kline]
      go remaining mEnd acc = do
        let batchLimit = min maxPerRequest remaining
        ks <- fetchBatch mEnd batchLimit
        if null ks
          then pure acc
          else do
            let ksSorted = sortBy (comparing kOpenTime) ks
                acc' = ksSorted ++ acc
                remaining' = remaining - length ksSorted
                nextEnd = kOpenTime (head ksSorted) - 1
            if remaining' <= 0 || length ksSorted < batchLimit
              then pure acc'
              else go remaining' (Just nextEnd) acc'

  if wanted <= maxPerRequest
    then fetchBatch Nothing wanted
    else go wanted Nothing []

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
  resp <- binanceHttp env "ticker/price" req
  ensure2xx "ticker/price" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode ticker price: " ++ e))
    Right (TickerPrice p) -> pure p

data Ticker24hPrice = Ticker24hPrice { t24LastPrice :: Double }

instance FromJSON Ticker24hPrice where
  parseJSON = withObject "Ticker24hPrice" $ \o -> do
    pTxt <- o .: "lastPrice"
    p <- parseDoubleText pTxt
    pure (Ticker24hPrice p)

fetchTicker24hPrice :: BinanceEnv -> String -> IO Double
fetchTicker24hPrice env symbol = do
  let path =
        case beMarket env of
          MarketSpot -> "/api/v3/ticker/24hr"
          MarketMargin -> "/api/v3/ticker/24hr"
          MarketFutures -> "/fapi/v1/ticker/24hr"
  req0 <- parseRequest (beBaseUrl env ++ path)
  let qs = renderSimpleQuery True [("symbol", BS.pack (map toUpperAscii symbol))]
      req = req0 { method = "GET", queryString = qs }
  resp <- binanceHttp env "ticker/24hr" req
  ensure2xx "ticker/24hr" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode ticker/24hr: " ++ e))
    Right (Ticker24hPrice p) -> pure p

data FuturesMarkPrice = FuturesMarkPrice { fmpMarkPrice :: Double }

instance FromJSON FuturesMarkPrice where
  parseJSON = withObject "FuturesMarkPrice" $ \o -> do
    pTxt <- o .: "markPrice"
    p <- parseDoubleText pTxt
    pure (FuturesMarkPrice p)

fetchFuturesMarkPrice :: BinanceEnv -> String -> IO Double
fetchFuturesMarkPrice env symbol = do
  if beMarket env /= MarketFutures
    then throwIO (userError "fetchFuturesMarkPrice requires MarketFutures")
    else pure ()
  req0 <- parseRequest (beBaseUrl env ++ "/fapi/v1/premiumIndex")
  let qs = renderSimpleQuery True [("symbol", BS.pack (map toUpperAscii symbol))]
      req = req0 { method = "GET", queryString = qs }
  resp <- binanceHttp env "premiumIndex" req
  ensure2xx "premiumIndex" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode premiumIndex: " ++ e))
    Right (FuturesMarkPrice p) -> pure p

data Ticker24h = Ticker24h
  { t24Symbol :: !String
  , t24QuoteVolume :: !Double
  } deriving (Eq, Show)

instance FromJSON Ticker24h where
  parseJSON = withObject "Ticker24h" $ \o -> do
    symTxt <- o .: "symbol"
    qvTxt <- o .: "quoteVolume"
    qv <- parseDoubleText qvTxt
    pure Ticker24h { t24Symbol = T.unpack symTxt, t24QuoteVolume = qv }

fetchTickers24h :: BinanceEnv -> IO [Ticker24h]
fetchTickers24h env = do
  let key = beBaseUrl env ++ ":" ++ show (beMarket env)
  fetchWithCache binanceTickersCache binanceTickersFreshTtl binanceTickersStaleTtl key $ do
    let path =
          case beMarket env of
            MarketSpot -> "/api/v3/ticker/24hr"
            MarketMargin -> "/api/v3/ticker/24hr"
            MarketFutures -> "/fapi/v1/ticker/24hr"
    req0 <- parseRequest (beBaseUrl env ++ path)
    let req = req0 { method = "GET" }
    resp <- binanceHttp env "ticker/24hr" req
    ensure2xx "ticker/24hr" resp
    case eitherDecode (responseBody resp) of
      Left e -> throwIO (userError ("Failed to decode ticker/24hr: " ++ e))
      Right xs -> pure xs

-- | Returns the highest-volume symbols by 24h quote volume for the provided quote asset.
-- Filtering is conservative to avoid leveraged tokens and stable-stable pairs.
fetchTopSymbolsByQuoteVolume :: BinanceEnv -> String -> Int -> IO [(String, Double)]
fetchTopSymbolsByQuoteVolume env quote topN = do
  if topN <= 0
    then pure []
    else do
      tickers <- fetchTickers24h env
      let quoteU = map toUpperAscii quote
          stableBases = ["USDT", "USDC", "BUSD", "TUSD", "FDUSD"]
          leveragedSuffixes = ["UP", "DOWN", "BULL", "BEAR"]
          wanted (Ticker24h symRaw _qv) =
            let sym = map toUpperAscii symRaw
             in quoteU `isSuffixOf` sym
                && let base = take (length sym - length quoteU) sym
                       isStableStable = base `elem` stableBases
                       isLeveraged = any (`isSuffixOf` base) leveragedSuffixes
                    in not isStableStable && not isLeveraged
          ranked =
            sortBy (flip (comparing snd)) $
              [ (map toUpperAscii (t24Symbol t), max 0 (t24QuoteVolume t))
              | t <- filter wanted tickers
              ]
      pure (take topN ranked)

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
  resp <- binanceHttp env label req
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
  resp <- binanceHttp env label req
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
  resp <- binanceHttp env label req
  ensure2xx label resp
  pure (responseBody resp)

fetchAccountTrades :: BinanceEnv -> Maybe String -> Maybe Int -> Maybe Int64 -> Maybe Int64 -> Maybe Int64 -> IO [BinanceTrade]
fetchAccountTrades env mSymbol mLimit mStartTime mEndTime mFromId = do
  apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
  secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
  ts <- getTimestampMs

  symbolParam <-
    case (beMarket env, mSymbol) of
      (MarketFutures, Nothing) -> pure []
      (_, Just sym) -> pure [("symbol", BS.pack (map toUpperAscii sym))]
      (_, Nothing) -> throwIO (userError "binance trades require symbol for spot/margin markets")

  let clampLimit n = max 1 (min 1000 n)
      limitParam =
        case mLimit of
          Nothing -> []
          Just lim -> [("limit", BS.pack (show (clampLimit lim)))]
      startTimeParam =
        case mStartTime of
          Nothing -> []
          Just t -> [("startTime", BS.pack (show (max 0 t)))]
      endTimeParam =
        case mEndTime of
          Nothing -> []
          Just t -> [("endTime", BS.pack (show (max 0 t)))]
      fromIdParam =
        case mFromId of
          Nothing -> []
          Just v -> [("fromId", BS.pack (show (max 0 v)))]
      baseParams =
        [ ("timestamp", BS.pack (show ts))
        , ("recvWindow", "5000")
        ]
      params = symbolParam ++ limitParam ++ startTimeParam ++ endTimeParam ++ fromIdParam ++ baseParams
      queryToSign = renderSimpleQuery False params
      sig = signQuery secret queryToSign
      paramsSigned = params ++ [("signature", sig)]
      qs = renderSimpleQuery True paramsSigned

      (path, label) =
        case beMarket env of
          MarketSpot -> ("/api/v3/myTrades", "account/myTrades")
          MarketMargin -> ("/sapi/v1/margin/myTrades", "margin/myTrades")
          MarketFutures -> ("/fapi/v1/userTrades", "futures/userTrades")

  req0 <- parseRequest (beBaseUrl env ++ path)
  let req =
        req0
          { method = "GET"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- binanceHttp env label req
  ensure2xx label resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode " ++ label ++ ": " ++ e))
    Right trades -> pure trades

data FuturesOpenOrder = FuturesOpenOrder
  { fooClientOrderId :: String
  } deriving (Eq, Show)

data BinanceOpenOrder = BinanceOpenOrder
  { booClientOrderId :: !(Maybe String)
  , booSide :: !(Maybe OrderSide)
  , booReduceOnly :: !(Maybe Bool)
  , booClosePosition :: !(Maybe Bool)
  , booPositionSide :: !(Maybe String)
  } deriving (Eq, Show)

parseOrderSide :: String -> Maybe OrderSide
parseOrderSide raw =
  case map toUpperAscii raw of
    "BUY" -> Just Buy
    "SELL" -> Just Sell
    _ -> Nothing

instance FromJSON FuturesOpenOrder where
  parseJSON = withObject "FuturesOpenOrder" $ \o -> do
    cid <- o .: "clientOrderId"
    pure FuturesOpenOrder { fooClientOrderId = cid }

instance FromJSON BinanceOpenOrder where
  parseJSON = withObject "BinanceOpenOrder" $ \o -> do
    clientOrderId <- o AT..:? "clientOrderId"
    sideRaw <- o AT..:? "side"
    let side = sideRaw >>= parseOrderSide
    reduceOnly <- o AT..:? "reduceOnly"
    closePosition <- o AT..:? "closePosition"
    positionSide <- o AT..:? "positionSide"
    pure
      BinanceOpenOrder
        { booClientOrderId = clientOrderId
        , booSide = side
        , booReduceOnly = reduceOnly
        , booClosePosition = closePosition
        , booPositionSide = positionSide
        }

fetchOpenOrdersWith ::
  FromJSON a =>
  BinanceEnv ->
  String ->
  String ->
  String ->
  IO [a]
fetchOpenOrdersWith env label path symbol = do
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

  req0 <- parseRequest (beBaseUrl env ++ path)
  let req =
        req0
          { method = "GET"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- binanceHttp env label req
  ensure2xx label resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode " ++ label ++ ": " ++ e))
    Right orders -> pure orders

fetchOpenOrders :: BinanceEnv -> String -> IO [BinanceOpenOrder]
fetchOpenOrders env symbol =
  case beMarket env of
    MarketFutures -> fetchOpenOrdersWith env "futures/openOrders" "/fapi/v1/openOrders" symbol
    MarketMargin -> fetchOpenOrdersWith env "margin/openOrders" "/sapi/v1/margin/openOrders" symbol
    MarketSpot -> fetchOpenOrdersWith env "spot/openOrders" "/api/v3/openOrders" symbol

fetchFuturesOpenOrders :: BinanceEnv -> String -> IO [FuturesOpenOrder]
fetchFuturesOpenOrders env symbol = do
  if beMarket env /= MarketFutures
    then throwIO (userError "fetchFuturesOpenOrders requires MarketFutures")
    else pure ()
  fetchOpenOrdersWith env "futures/openOrders" "/fapi/v1/openOrders" symbol

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
  resp <- binanceHttp env "futures/order/cancel" req
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
      label =
        if beMarket env == MarketMargin then "margin/account" else "account"
  req0 <- parseRequest (beBaseUrl env ++ path)
  let req =
        req0
          { method = "GET"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- binanceHttp env label req
  ensure2xx label resp
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
  resp <- binanceHttp env "futures/balance" req
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
  resp <- binanceHttp env "futures/positionRisk" req
  ensure2xx "futures/positionRisk" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode futures positionRisk: " ++ e))
    Right positions ->
      let sym = map toUpperAscii symbol
          match p = map toUpperAscii (fprSymbol p) == sym
          signedAmt p =
            case fmap normalizeKey (fprPositionSide p) of
              Just "short" -> negate (abs (fprPositionAmt p))
              Just "long" -> abs (fprPositionAmt p)
              _ -> fprPositionAmt p
          total = sum [signedAmt p | p <- positions, match p]
       in pure total

data FuturesPositionRisk = FuturesPositionRisk
  { fprSymbol :: !String
  , fprPositionAmt :: !Double
  , fprEntryPrice :: !Double
  , fprMarkPrice :: !Double
  , fprUnrealizedProfit :: !Double
  , fprLiquidationPrice :: !(Maybe Double)
  , fprBreakEvenPrice :: !(Maybe Double)
  , fprLeverage :: !Double
  , fprMarginType :: !(Maybe String)
  , fprPositionSide :: !(Maybe String)
  } deriving (Eq, Show)

instance FromJSON FuturesPositionRisk where
  parseJSON = withObject "FuturesPositionRisk" $ \o -> do
    sym <- o .: "symbol"
    positionAmt <- parseDoubleField o "positionAmt"
    entryPrice <- parseDoubleField o "entryPrice"
    markPrice <- parseDoubleField o "markPrice"
    unrealizedProfit <- parseDoubleField o "unRealizedProfit"
    liquidationPrice <- parseMaybeDoubleField o "liquidationPrice"
    breakEvenPrice <- parseMaybeDoubleField o "breakEvenPrice"
    leverage <- parseDoubleField o "leverage"
    marginType <- o AT..:? "marginType"
    positionSide <- o AT..:? "positionSide"
    pure
      FuturesPositionRisk
        { fprSymbol = sym
        , fprPositionAmt = positionAmt
        , fprEntryPrice = entryPrice
        , fprMarkPrice = markPrice
        , fprUnrealizedProfit = unrealizedProfit
        , fprLiquidationPrice = liquidationPrice
        , fprBreakEvenPrice = breakEvenPrice
        , fprLeverage = leverage
        , fprMarginType = marginType
        , fprPositionSide = positionSide
        }

fetchFuturesPositionRisks :: BinanceEnv -> IO [FuturesPositionRisk]
fetchFuturesPositionRisks env = do
  if beMarket env /= MarketFutures
    then throwIO (userError "fetchFuturesPositionRisks requires MarketFutures")
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

  req0 <- parseRequest (beBaseUrl env ++ "/fapi/v2/positionRisk")
  let req =
        req0
          { method = "GET"
          , queryString = qs
          , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
          }
  resp <- binanceHttp env "futures/positionRisk" req
  ensure2xx "futures/positionRisk" resp
  case eitherDecode (responseBody resp) of
    Left e -> throwIO (userError ("Failed to decode futures positionRisk: " ++ e))
    Right positions -> pure positions

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

binanceErrorSummary :: Response BL.ByteString -> Text
binanceErrorSummary resp =
  case eitherDecode (responseBody resp) :: Either String BinanceErrorBody of
    Right be ->
      let msg = fromMaybe "" (bebMsg be)
          codeLabel =
            case bebCode be of
              Nothing -> ""
              Just c -> "Binance code " ++ show c ++ ": "
       in T.pack (codeLabel ++ msg)
    Left _ ->
      let snippet = BS.unpack (BS.take 200 (BL.toStrict (responseBody resp)))
       in T.pack snippet

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
  resp <- binanceHttp env "listenKey" req
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
  resp <- binanceHttp env "listenKey/keepAlive" req
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
  resp <- binanceHttp env "listenKey/close" req
  ensure2xx "listenKey/close" resp

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

{-# LANGUAGE OverloadedStrings #-}

module Trader.Binance (
    BinanceEnv (..),
    BinanceLog (..),
    BinanceMarket (..),
    BinanceOrderMode (..),
    OrderSide (..),
    BinanceTrade (..),
    FuturesIncome (..),
    Kline (..),
    Step (..),
    SymbolFilters (..),
    Ticker24h (..),
    BinanceOpenOrder (..),
    FuturesPositionRisk (..),
    fetchTickerPrice,
    fetchTicker24hPrice,
    fetchFuturesMarkPrice,
    fetchFundingRateHistory,
    fetchOpenInterestHist,
    fetchTakerLongShortRatio,
    fetchBasisHistory,
    fetchTickers24h,
    fetchTopSymbolsByQuoteVolume,
    binanceBaseUrl,
    binanceTestnetBaseUrl,
    binanceFuturesBaseUrl,
    binanceFuturesTestnetBaseUrl,
    newBinanceEnv,
    fetchKlines,
    fetchKlinesBetween,
    fetchKlinesRaw,
    fetchCloses,
    fetchSymbolFilters,
    quantizeDown,
    getTimestampMs,
    signQuery,
    placeMarketOrder,
    placeFuturesMarketOrderWithPositionSide,
    placeFuturesPostOnlyLimitOrder,
    BookTickerQuote (..),
    fetchBookTickerQuote,
    cancelFuturesOrderByClientId,
    placeFuturesTriggerMarketOrder,
    placeFuturesAlgoTriggerMarketOrder,
    fetchFuturesOpenAlgoOrders,
    cancelFuturesAlgoOrderByClientId,
    FuturesAlgoOpenOrder (..),
    fetchFuturesAccountUid,
    fetchOrderByClientId,
    fetchAccountTrades,
    fetchFuturesIncome,
    fetchFreeBalance,
    fetchFuturesAvailableBalance,
    fetchFuturesPositionAmt,
    fetchFuturesPositionRisks,
    fetchFuturesPositionRisksWithResponseTimeout,
    fetchOpenOrders,
    cancelFuturesOpenOrdersByClientPrefix,
    BinanceProxyHealth (..),
    binanceProxyHealth,
    binanceMarketDataCacheStats,
    binanceExceptionSummary,
    createListenKey,
    keepAliveListenKey,
    closeListenKey,
    futuresPositionRiskLeverageSane,
) where

import Control.Applicative ((<|>))
import Control.Exception (SomeException, displayException, fromException, throwIO, try)
import qualified Control.Monad
import Crypto.Hash.Algorithms (SHA256)
import Crypto.MAC.HMAC (HMAC, hmac, hmacGetDigest)
import Data.Aeson (FromJSON (..), ToJSON (..), eitherDecode, object, withArray, withObject, (.:), (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.Types as AT
import Data.ByteArray (convert)
import Data.ByteArray.Encoding (Base (Base64), convertToBase)
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (isAsciiLower, isSpace, toLower)
import Data.Foldable (traverse_)
import Data.Int (Int64)
import Data.List (foldl', isInfixOf, isPrefixOf, isSuffixOf, sortOn)
import Data.Maybe (fromMaybe, listToMaybe)
import qualified Data.Ord
import Data.Text (Text)
import qualified Data.Text as T
import Data.Text.Encoding (decodeUtf8With)
import Data.Text.Encoding.Error (lenientDecode)
import Data.Time.Clock (NominalDiffTime)
import Data.Time.Clock.POSIX (getPOSIXTime)
import qualified Data.Vector as V
import Network.HTTP.Client
import Network.HTTP.Types.Header (hProxyAuthorization)
import Network.HTTP.Types.Status (statusCode)
import Network.HTTP.Types.URI (parseQuery, renderSimpleQuery)
import Network.URI (URI (..), URIAuth (..), parseURI)
import Numeric (showFFloat)
import System.Environment (lookupEnv)
import System.IO.Unsafe (unsafePerformIO)
import Text.Read (readMaybe)
import Trader.Cache (TtlCache, TtlCacheStats, cacheStats, fetchWithCache, insertCache, newTtlCacheWithMaxEntries)
import Trader.Duration (parseIntervalSeconds)
import Trader.Http (defaultRetryConfig, httpLbsWithRetry, newHttpManager)
import Trader.MarketDataIntegrity (MarketSeriesBar (..), validateMarketSeriesContinuity)
import Trader.Text (normalizeKey)

data BinanceEnv = BinanceEnv
    { beManager :: Manager
    , beBaseUrl :: String
    , beMarket :: BinanceMarket
    , beApiKey :: Maybe BS.ByteString
    , beApiSecret :: Maybe BS.ByteString
    , beLogger :: Maybe (BinanceLog -> IO ())
    , beProxy :: Maybe BinanceProxy
    }

data BinanceProxy = BinanceProxy
    { bpProxy :: !Proxy
    , bpAuthHeader :: !(Maybe BS.ByteString)
    }
    deriving (Eq, Show)

data BinanceProxyHealth = BinanceProxyHealth
    { bphConfigured :: !Bool
    , bphOk :: !Bool
    , bphHost :: !(Maybe Text)
    , bphPort :: !(Maybe Int)
    , bphError :: !(Maybe Text)
    }
    deriving (Eq, Show)

instance ToJSON BinanceProxyHealth where
    toJSON h =
        let status
                | not (bphConfigured h) = ("not_configured" :: String)
                | bphOk h = "ok"
                | otherwise = "error"
         in object
                [ "status" .= status
                , "configured" .= bphConfigured h
                , "ok" .= bphOk h
                , "host" .= bphHost h
                , "port" .= bphPort h
                , "error" .= bphError h
                ]

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
    }
    deriving (Eq, Show)

data BinanceMarket = MarketSpot | MarketMargin | MarketFutures
    deriving (Eq, Show)

data BinanceOrderMode = OrderTest | OrderLive deriving (Eq, Show)

data OrderSide = Buy | Sell deriving (Eq, Show)

data Kline = Kline
    { kOpenTime :: !Int64
    , kCloseTime :: !(Maybe Int64)
    , kOpen :: !Double
    , kHigh :: !Double
    , kLow :: !Double
    , kClose :: !Double
    , kVolume :: !Double
    }
    deriving (Eq, Show)

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
    , btOriginIp :: !(Maybe Text)
    , btExecutorIp :: !(Maybe Text)
    , btOriginInstance :: !(Maybe Text)
    , btEntryIp :: !(Maybe Text)
    , btExitIp :: !(Maybe Text)
    , btEntryInstance :: !(Maybe Text)
    , btExitInstance :: !(Maybe Text)
    , btEntryTime :: !(Maybe Int64)
    , btExitTime :: !(Maybe Int64)
    , btMaxPnl :: !(Maybe Double)
    , btMaxPnlCloseTime :: !(Maybe Int64)
    , btMethod :: !(Maybe Text)
    , btStrategy :: !(Maybe Text)
    , btDecisionSummary :: !(Maybe Text)
    , btDecisionReason :: !(Maybe Text)
    }
    deriving (Eq, Show)

data FuturesIncome = FuturesIncome
    { fiSymbol :: !String
    , fiIncomeType :: !String
    , fiIncome :: !Double
    , fiAsset :: !String
    , fiTime :: !Int64
    }
    deriving (Eq, Show)

newtype BinanceServerTime = BinanceServerTime
    { bstServerTime :: Int64
    }
    deriving (Eq, Show)

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
        originIp <- o AT..:? "originIp"
        executorIp <- o AT..:? "executorIp"
        originInstance <- o AT..:? "originInstance"
        entryIp <- o AT..:? "entryIp"
        exitIp <- o AT..:? "exitIp"
        entryInstance <- o AT..:? "entryInstance"
        exitInstance <- o AT..:? "exitInstance"
        entryTime <- o AT..:? "entryTime"
        exitTime <- o AT..:? "exitTime"
        maxPnl <- parseMaybeDoubleField o "maxPnl"
        maxPnlCloseTime <- o AT..:? "maxPnlCloseTime"
        method <- o AT..:? "method"
        strategy <- o AT..:? "strategy"
        decisionSummary <- o AT..:? "decisionSummary"
        decisionReason <- o AT..:? "decisionReason"
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
        quoteQty <-
            case quoteQtyRaw of
                Just q -> pure q
                Nothing ->
                    let inferred = price * qty
                     in if isNaN inferred || isInfinite inferred
                            then fail "Invalid inferred quoteQty"
                            else pure inferred
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
                , btOriginIp = originIp
                , btExecutorIp = executorIp
                , btOriginInstance = originInstance
                , btEntryIp = entryIp
                , btExitIp = exitIp
                , btEntryInstance = entryInstance
                , btExitInstance = exitInstance
                , btEntryTime = entryTime
                , btExitTime = exitTime
                , btMaxPnl = maxPnl
                , btMaxPnlCloseTime = maxPnlCloseTime
                , btMethod = method
                , btStrategy = strategy
                , btDecisionSummary = decisionSummary
                , btDecisionReason = decisionReason
                }

instance FromJSON FuturesIncome where
    parseJSON = withObject "FuturesIncome" $ \o -> do
        symbol <- fromMaybe "" <$> o AT..:? "symbol"
        incomeType <- o .: "incomeType"
        income <- parseDoubleField o "income"
        asset <- o .: "asset"
        incomeTime <- o .: "time"
        pure
            FuturesIncome
                { fiSymbol = map toUpperAscii symbol
                , fiIncomeType = map toUpperAscii incomeType
                , fiIncome = income
                , fiAsset = map toUpperAscii asset
                , fiTime = incomeTime
                }

instance FromJSON BinanceServerTime where
    parseJSON = withObject "BinanceServerTime" $ \o -> do
        ts <- o .: "serverTime"
        pure (BinanceServerTime ts)

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
            , "originIp" .= btOriginIp t
            , "executorIp" .= btExecutorIp t
            , "originInstance" .= btOriginInstance t
            , "entryIp" .= btEntryIp t
            , "exitIp" .= btExitIp t
            , "entryInstance" .= btEntryInstance t
            , "exitInstance" .= btExitInstance t
            , "entryTime" .= btEntryTime t
            , "exitTime" .= btExitTime t
            , "maxPnl" .= btMaxPnl t
            , "maxPnlCloseTime" .= btMaxPnlCloseTime t
            , "method" .= btMethod t
            , "strategy" .= btStrategy t
            , "decisionSummary" .= btDecisionSummary t
            , "decisionReason" .= btDecisionReason t
            ]

binanceBaseUrl :: String
binanceBaseUrl = "https://api.binance.com"

binanceTestnetBaseUrl :: String
binanceTestnetBaseUrl = "https://testnet.binance.vision"

binanceFuturesBaseUrl :: String
binanceFuturesBaseUrl = "https://fapi.binance.com"

binanceFuturesTestnetBaseUrl :: String
binanceFuturesTestnetBaseUrl = "https://testnet.binancefuture.com"

binanceRecvWindowMs :: BS.ByteString
binanceRecvWindowMs = "10000"

{-# NOINLINE binanceTickersCache #-}
binanceTickersCache :: TtlCache String [Ticker24h]
binanceTickersCache = unsafePerformIO (newTtlCacheWithMaxEntries binanceTickersMaxEntries)

{-# NOINLINE binanceExchangeInfoCache #-}
binanceExchangeInfoCache :: TtlCache String SymbolFilters
binanceExchangeInfoCache = unsafePerformIO (newTtlCacheWithMaxEntries binanceExchangeInfoMaxEntries)

{-# NOINLINE binanceKlinesCache #-}
binanceKlinesCache :: TtlCache String [Kline]
binanceKlinesCache = unsafePerformIO (newTtlCacheWithMaxEntries binanceKlinesMaxEntries)

{-# NOINLINE binanceTimeOffsetCache #-}
binanceTimeOffsetCache :: TtlCache String Int64
binanceTimeOffsetCache = unsafePerformIO (newTtlCacheWithMaxEntries binanceTimeOffsetMaxEntries)

binanceTickersMaxEntries :: Int
binanceTickersMaxEntries = 16

binanceExchangeInfoMaxEntries :: Int
binanceExchangeInfoMaxEntries = 32

binanceKlinesMaxEntries :: Int
binanceKlinesMaxEntries = 128

binanceTimeOffsetMaxEntries :: Int
binanceTimeOffsetMaxEntries = 8

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

binanceTimeOffsetFreshTtl :: NominalDiffTime
binanceTimeOffsetFreshTtl = 10

binanceTimeOffsetStaleTtl :: NominalDiffTime
binanceTimeOffsetStaleTtl = 60

binanceMarketDataCacheStats :: IO [(String, TtlCacheStats)]
binanceMarketDataCacheStats =
    sequence
        [ named "binanceTickers" binanceTickersCache binanceTickersStaleTtl
        , named "binanceExchangeInfo" binanceExchangeInfoCache binanceExchangeInfoStaleTtl
        , named "binanceKlines" binanceKlinesCache binanceKlinesStaleTtl
        , named "binanceTimeOffset" binanceTimeOffsetCache binanceTimeOffsetStaleTtl
        ]
  where
    named label cache staleTtl = do
        stats <- cacheStats cache staleTtl
        pure (label, stats)

newBinanceEnv :: BinanceMarket -> String -> Maybe BS.ByteString -> Maybe BS.ByteString -> IO BinanceEnv
newBinanceEnv market baseUrl apiKey apiSecret = do
    mgr <- newHttpManager
    proxyCfg <- resolveBinanceProxy
    pure
        BinanceEnv
            { beManager = mgr
            , beBaseUrl = baseUrl
            , beMarket = market
            , beApiKey = apiKey
            , beApiSecret = apiSecret
            , beLogger = Nothing
            , beProxy = proxyCfg
            }

resolveBinanceProxy :: IO (Maybe BinanceProxy)
resolveBinanceProxy = do
    mRaw <- lookupEnv "TRADER_BINANCE_PROXY_URL"
    pure (mRaw >>= parseBinanceProxy)

binanceProxyHealth :: IO BinanceProxyHealth
binanceProxyHealth = do
    mRaw <- lookupEnv "TRADER_BINANCE_PROXY_URL"
    case mRaw of
        Nothing -> pure (BinanceProxyHealth False True Nothing Nothing Nothing)
        Just raw ->
            let trimmed = trimString raw
             in if null trimmed
                    then pure (BinanceProxyHealth False True Nothing Nothing Nothing)
                    else case parseBinanceProxyDetails trimmed of
                        Nothing ->
                            pure
                                ( BinanceProxyHealth
                                    { bphConfigured = True
                                    , bphOk = False
                                    , bphHost = Nothing
                                    , bphPort = Nothing
                                    , bphError = Just "Invalid TRADER_BINANCE_PROXY_URL"
                                    }
                                )
                        Just (proxyCfg, hostName, portNum) -> do
                            mgr <- newHttpManager
                            req0 <- parseRequest (binanceBaseUrl ++ "/api/v3/time")
                            let env =
                                    BinanceEnv
                                        { beManager = mgr
                                        , beBaseUrl = binanceBaseUrl
                                        , beMarket = MarketSpot
                                        , beApiKey = Nothing
                                        , beApiSecret = Nothing
                                        , beLogger = Nothing
                                        , beProxy = Just proxyCfg
                                        }
                                req = applyBinanceProxy env req0
                            respOrErr <- try (httpLbs req mgr) :: IO (Either SomeException (Response BL.ByteString))
                            case respOrErr of
                                Left ex ->
                                    pure
                                        ( BinanceProxyHealth
                                            { bphConfigured = True
                                            , bphOk = False
                                            , bphHost = Just (T.pack hostName)
                                            , bphPort = Just portNum
                                            , bphError = Just (binanceExceptionSummary ex)
                                            }
                                        )
                                Right resp ->
                                    let code = statusCode (responseStatus resp)
                                        ok = code >= 200 && code < 300
                                        err =
                                            if ok
                                                then Nothing
                                                else Just (T.pack ("HTTP " ++ show code))
                                     in pure
                                            ( BinanceProxyHealth
                                                { bphConfigured = True
                                                , bphOk = ok
                                                , bphHost = Just (T.pack hostName)
                                                , bphPort = Just portNum
                                                , bphError = err
                                                }
                                            )

parseBinanceProxy :: String -> Maybe BinanceProxy
parseBinanceProxy raw = do
    (proxyCfg, _, _) <- parseBinanceProxyDetails raw
    pure proxyCfg

parseBinanceProxyDetails :: String -> Maybe (BinanceProxy, String, Int)
parseBinanceProxyDetails raw = do
    let trimmed = trimString raw
    if null trimmed
        then Nothing
        else do
            uri <- parseURI trimmed
            auth <- uriAuthority uri
            let hostName = uriRegName auth
                portNum = parseProxyPort (uriPort auth)
            if null hostName || portNum <= 0
                then Nothing
                else
                    let proxyCfg = Proxy (BS.pack hostName) portNum
                        authHeader = parseUserInfo (uriUserInfo auth) >>= proxyAuthHeader
                     in Just (BinanceProxy{bpProxy = proxyCfg, bpAuthHeader = authHeader}, hostName, portNum)

parseProxyPort :: String -> Int
parseProxyPort portRaw =
    case portRaw of
        "" -> 3128
        ':' : rest ->
            case readMaybe rest of
                Just n | n > 0 -> n
                _ -> 0
        _ -> 0

parseUserInfo :: String -> Maybe (String, String)
parseUserInfo raw =
    let trimmed = dropWhileEnd (== '@') raw
     in if null trimmed
            then Nothing
            else case break (== ':') trimmed of
                (u, ':' : p) -> Just (u, p)
                (u, "") -> Just (u, "")
                _ -> Nothing

proxyAuthHeader :: (String, String) -> Maybe BS.ByteString
proxyAuthHeader (user, pass) =
    if null user
        then Nothing
        else
            let raw = BS.pack (user ++ ":" ++ pass)
                encoded = convertToBase Base64 raw
             in Just (BS.concat ["Basic ", encoded])

trimString :: String -> String
trimString = dropWhileEnd isSpace . dropWhile isSpace

applyBinanceProxy :: BinanceEnv -> Request -> Request
applyBinanceProxy env req =
    case beProxy env of
        Nothing -> req
        Just proxyCfg ->
            let headers =
                    case bpAuthHeader proxyCfg of
                        Nothing -> requestHeaders req
                        Just authVal -> (hProxyAuthorization, authVal) : requestHeaders req
             in req{proxy = Just (bpProxy proxyCfg), requestHeaders = headers}

binanceHttp :: BinanceEnv -> String -> Request -> IO (Response BL.ByteString)
binanceHttp env label req0 = do
    t0 <- getTimestampMs
    let req = applyBinanceProxy env req0
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
                volumeTxt <- parseIndexText 5 arr
                open <- parseDoubleText openTxt
                high <- parseDoubleText highTxt
                low <- parseDoubleText lowTxt
                close <- parseDoubleText closeTxt
                volume <- parseDoubleText volumeTxt
                closeTime <-
                    if V.length arr > 6
                        then Just <$> parseIndexInt64 6 arr
                        else pure Nothing
                pure
                    Kline
                        { kOpenTime = openTime
                        , kCloseTime = closeTime
                        , kOpen = open
                        , kHigh = high
                        , kLow = low
                        , kClose = close
                        , kVolume = volume
                        }
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
    case readMaybe (T.unpack (T.strip t)) of
        Just d -> parseFiniteDouble d t
        Nothing -> fail ("Failed to parse double: " ++ T.unpack t)

parseFiniteDouble :: Double -> Text -> AT.Parser Double
parseFiniteDouble d raw =
    if isNaN d || isInfinite d
        then fail ("Failed to parse finite double: " ++ T.unpack raw)
        else pure d

parseDoubleValue :: Aeson.Value -> AT.Parser Double
parseDoubleValue value =
    case value of
        Aeson.String t -> parseDoubleText t
        Aeson.Number n -> parseFiniteDouble (realToFrac n) (T.pack (show n))
        _ -> fail "Expected string or number"

parseDoubleField :: Aeson.Object -> Text -> AT.Parser Double
parseDoubleField o k = do
    value <- o .: AK.fromText k
    parseDoubleValue value

parseMaybeDoubleField :: Aeson.Object -> Text -> AT.Parser (Maybe Double)
parseMaybeDoubleField o k = do
    mt <- o AT..:? AK.fromText k :: AT.Parser (Maybe Aeson.Value)
    case mt of
        Nothing -> pure Nothing
        Just value -> Just <$> parseDoubleValue value

-- Binance symbol filters (exchangeInfo)

data Step = Step
    { stepScale :: !Integer
    , stepInt :: !Integer
    , stepText :: !Text
    }
    deriving (Eq, Show)

mkStep :: Text -> Maybe Step
mkStep raw =
    let t = T.strip raw
        s = T.unpack t
     in case break (== '.') s of
            (a, "") -> do
                ai <- readMaybeInteger a
                if ai <= 0 then Nothing else Just Step{stepScale = 1, stepInt = ai, stepText = t}
            (a, '.' : b) -> do
                ai <- readMaybeInteger a
                bi <- readMaybeInteger b
                let scale = 10 ^ length b
                    val = ai * scale + bi
                if val <= 0 then Nothing else Just Step{stepScale = scale, stepInt = val, stepText = t}
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
    }
    deriving (Eq, Show)

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

newtype ExchangeInfo = ExchangeInfo [ExchangeSymbol]

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
        pure ExchangeSymbol{esSymbol = sym, esFilters = flt}

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
            req = req0{method = "GET", queryString = qs}
        resp <- binanceHttp env "exchangeInfo" req
        ensure2xx "exchangeInfo" resp
        case eitherDecode (responseBody resp) of
            Left e -> throwIO (userError ("Failed to decode exchangeInfo: " ++ e))
            Right (ExchangeInfo syms) ->
                case listToMaybe [s | s <- syms, map toUpperAscii (esSymbol s) == map toUpperAscii symbol] of
                    Nothing -> throwIO (userError ("exchangeInfo: symbol not found: " ++ symbol))
                    Just s -> pure (parseSymbolFilters (esFilters s))

parseSymbolFilters :: [Aeson.Object] -> SymbolFilters
parseSymbolFilters = foldl' apply emptySymbolFilters
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
                ++ ":"
                ++ show (beMarket env)
                ++ ":"
                ++ map toUpperAscii symbol
                ++ ":"
                ++ interval
                ++ ":"
                ++ show limit
    fetchWithCache binanceKlinesCache binanceKlinesFreshTtl binanceKlinesStaleTtl key $
        fetchKlinesRaw env symbol interval limit

fetchKlinesRaw :: BinanceEnv -> String -> String -> Int -> IO [Kline]
fetchKlinesRaw env symbol interval limit = do
    let maxPerRequest = 1000
        wanted = max 1 limit
        fetchWanted = wanted + 1
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
                req = req0{method = "GET", queryString = renderSimpleQuery True qs}
            resp <- binanceHttp env "klines" req
            ensure2xx "klines" resp
            case eitherDecode (responseBody resp) of
                Left e -> throwIO (userError ("Failed to decode klines: " ++ e))
                Right ks -> pure ks

        go :: Int -> Maybe Int64 -> [Kline] -> IO [Kline]
        go remaining mEnd acc = do
            let batchLimit = min maxPerRequest remaining
            ks <- fetchBatch mEnd batchLimit
            case sortOn kOpenTime ks of
                [] -> pure acc
                ksSorted@(firstK : _) -> do
                    let acc' = ksSorted ++ acc
                        batchCount = length ksSorted
                        remaining' = remaining - batchCount
                        nextEnd = kOpenTime firstK - 1
                    if remaining' <= 0 || batchCount < batchLimit
                        then pure acc'
                        else go remaining' (Just nextEnd) acc'

    raw <-
        if fetchWanted <= maxPerRequest
            then fetchBatch Nothing fetchWanted
            else go fetchWanted Nothing []
    now <- getTimestampMs
    case normalizeClosedKlines interval now raw of
        Left err -> throwIO (userError err)
        Right closed -> pure (takeLastKlines wanted closed)

fetchKlinesBetween :: BinanceEnv -> String -> String -> Int64 -> Int64 -> IO [Kline]
fetchKlinesBetween env symbol interval startTime endTime = do
    let startSafe = max 0 startTime
        endSafe = max startSafe endTime
        key =
            beBaseUrl env
                ++ ":"
                ++ show (beMarket env)
                ++ ":"
                ++ map toUpperAscii symbol
                ++ ":"
                ++ interval
                ++ ":"
                ++ show startSafe
                ++ ":"
                ++ show endSafe
    fetchWithCache binanceKlinesCache binanceKlinesFreshTtl binanceKlinesStaleTtl key $
        fetchKlinesBetweenRaw env symbol interval startSafe endSafe

fetchKlinesBetweenRaw :: BinanceEnv -> String -> String -> Int64 -> Int64 -> IO [Kline]
fetchKlinesBetweenRaw env symbol interval startTime endTime = do
    intervalMs <-
        case parseIntervalSeconds interval of
            Just sec | sec > 0 -> pure (fromIntegral sec * 1000)
            _ -> throwIO (userError ("Invalid kline interval: " ++ show interval))
    let maxPerRequest = 1000
        path =
            case beMarket env of
                MarketSpot -> "/api/v3/klines"
                MarketMargin -> "/api/v3/klines"
                MarketFutures -> "/fapi/v1/klines"
        symbolKey = BS.pack (map toUpperAscii symbol)
        fetchBatch batchStart = do
            req0 <- parseRequest (beBaseUrl env ++ path)
            let qs =
                    [ ("symbol", symbolKey)
                    , ("interval", BS.pack interval)
                    , ("limit", BS.pack (show maxPerRequest))
                    , ("startTime", BS.pack (show batchStart))
                    , ("endTime", BS.pack (show endTime))
                    ]
                req = req0{method = "GET", queryString = renderSimpleQuery True qs}
            resp <- binanceHttp env "klines/range" req
            ensure2xx "klines/range" resp
            case eitherDecode (responseBody resp) of
                Left e -> throwIO (userError ("Failed to decode klines: " ++ e))
                Right ks -> pure ks
        go batchStart acc
            | batchStart > endTime = pure acc
            | otherwise = do
                ks <- fetchBatch batchStart
                case sortOn kOpenTime ks of
                    [] -> pure acc
                    ksSorted -> do
                        let acc' = acc ++ ksSorted
                            batchCount = length ksSorted
                            lastOpenTime = kOpenTime (last ksSorted)
                            nextStart = lastOpenTime + intervalMs
                        if batchCount < maxPerRequest || nextStart <= batchStart || nextStart > endTime
                            then pure acc'
                            else go nextStart acc'
    raw <- go startTime []
    now <- getTimestampMs
    case normalizeClosedKlines interval now raw of
        Left err -> throwIO (userError err)
        Right closed -> pure [k | k <- closed, kOpenTime k >= startTime, kOpenTime k <= endTime]

normalizeClosedKlines :: String -> Int64 -> [Kline] -> Either String [Kline]
normalizeClosedKlines interval now ks = do
    intervalMs <-
        case parseIntervalSeconds interval of
            Just sec | sec > 0 -> Right (fromIntegral sec * 1000)
            _ -> Left ("Invalid kline interval: " ++ show interval)
    let sorted = sortOn kOpenTime ks
    validateKlineShapes sorted
    validateStrictKlineOpenTimes sorted
    let closed = filter (klineIsClosed intervalMs now) sorted
    validateMarketSeriesContinuity "Binance kline" intervalMs (map klineMarketSeriesBar closed)
    pure closed

klineMarketSeriesBar :: Kline -> MarketSeriesBar
klineMarketSeriesBar k =
    MarketSeriesBar
        { msbOpenTimeMs = kOpenTime k
        , msbOpen = Just (kOpen k)
        , msbHigh = Just (kHigh k)
        , msbLow = Just (kLow k)
        , msbClose = kClose k
        , msbVolume = Just (kVolume k)
        }

klineIsClosed :: Int64 -> Int64 -> Kline -> Bool
klineIsClosed intervalMs now k =
    let closeTime = fromMaybe (kOpenTime k + intervalMs - 1) (kCloseTime k)
     in closeTime < now

validateKlineShapes :: [Kline] -> Either String ()
validateKlineShapes =
    traverse_ validate
  where
    finite x = not (isNaN x || isInfinite x)
    validate k
        | not (all finite [kOpen k, kHigh k, kLow k, kClose k, kVolume k]) =
            Left ("Invalid kline numeric payload at openTime=" ++ show (kOpenTime k))
        | kVolume k < 0 =
            Left ("Invalid kline negative volume at openTime=" ++ show (kOpenTime k))
        | kHigh k < max (kOpen k) (kClose k) || kHigh k < kLow k || kLow k > min (kOpen k) (kClose k) =
            Left ("Invalid kline OHLC relationship at openTime=" ++ show (kOpenTime k))
        | otherwise = Right ()

validateStrictKlineOpenTimes :: [Kline] -> Either String ()
validateStrictKlineOpenTimes ks =
    case [kOpenTime b | (a, b) <- zip ks (drop 1 ks), kOpenTime a >= kOpenTime b] of
        bad : _ -> Left ("Invalid duplicate/non-increasing kline openTime=" ++ show bad)
        [] -> Right ()

takeLastKlines :: Int -> [Kline] -> [Kline]
takeLastKlines n xs
    | n <= 0 = []
    | otherwise =
        let dropCount = length xs - n
         in if dropCount <= 0 then xs else drop dropCount xs

fetchCloses :: BinanceEnv -> String -> String -> Int -> IO [Double]
fetchCloses env symbol interval limit = do
    ks <- fetchKlines env symbol interval limit
    pure (map kClose ks)

newtype TickerPrice = TickerPrice {tpPrice :: Double}

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
        req = req0{method = "GET", queryString = qs}
    resp <- binanceHttp env "ticker/price" req
    ensure2xx "ticker/price" resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode ticker price: " ++ e))
        Right (TickerPrice p) -> pure p

{- | Best bid/ask from the order book ticker; maker (post-only) entries price
off the touch rather than the last trade.
-}
data BookTickerQuote = BookTickerQuote
    { btqBid :: !Double
    , btqAsk :: !Double
    }
    deriving (Eq, Show)

instance FromJSON BookTickerQuote where
    parseJSON = withObject "BookTickerQuote" $ \o -> do
        bidTxt <- o .: "bidPrice"
        askTxt <- o .: "askPrice"
        bid <- parseDoubleText bidTxt
        ask <- parseDoubleText askTxt
        pure (BookTickerQuote bid ask)

fetchBookTickerQuote :: BinanceEnv -> String -> IO BookTickerQuote
fetchBookTickerQuote env symbol = do
    let path =
            case beMarket env of
                MarketSpot -> "/api/v3/ticker/bookTicker"
                MarketMargin -> "/api/v3/ticker/bookTicker"
                MarketFutures -> "/fapi/v1/ticker/bookTicker"
    req0 <- parseRequest (beBaseUrl env ++ path)
    let qs = renderSimpleQuery True [("symbol", BS.pack (map toUpperAscii symbol))]
        req = req0{method = "GET", queryString = qs}
    resp <- binanceHttp env "ticker/bookTicker" req
    ensure2xx "ticker/bookTicker" resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode book ticker: " ++ e))
        Right quote -> pure quote

{- | Post-only (GTX) futures limit order. GTX never takes liquidity: if the
price would cross the book, the exchange returns the order with status
EXPIRED instead of filling as taker. RESULT response type so any immediate
outcome (posted, expired) is visible to the caller.
-}
placeFuturesPostOnlyLimitOrder ::
    BinanceEnv ->
    BinanceOrderMode ->
    String -> -- symbol
    OrderSide ->
    Double -> -- quantity (base)
    Double -> -- limit price
    Maybe Bool -> -- reduceOnly
    Maybe String -> -- newClientOrderId
    IO BL.ByteString
placeFuturesPostOnlyLimitOrder env mode symbol side quantity price mReduceOnly mClientOrderId = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "placeFuturesPostOnlyLimitOrder requires MarketFutures")
    Control.Monad.when (quantity <= 0) $ throwIO (userError "Futures LIMIT orders require quantity > 0")
    Control.Monad.when (price <= 0) $ throwIO (userError "Futures LIMIT orders require price > 0")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
    let sideTxt = case side of Buy -> "BUY"; Sell -> "SELL"
        baseParams ts =
            [ ("symbol", BS.pack (map toUpperAscii symbol))
            , ("side", sideTxt)
            , ("type", "LIMIT")
            , ("timeInForce", "GTX")
            , ("quantity", renderDouble quantity)
            , ("price", renderDouble price)
            , ("recvWindow", binanceRecvWindowMs)
            , ("timestamp", BS.pack (show ts))
            ]
        reduceOnlyParams =
            case mReduceOnly of
                Just True -> [("reduceOnly", "true")]
                _ -> []
        clientIdParam =
            case mClientOrderId of
                Nothing -> []
                Just cid | null (trim cid) -> []
                Just cid -> [("newClientOrderId", BS.pack (trim cid))]
        (path, label) =
            if mode == OrderTest
                then ("/fapi/v1/order/test", "futures/order/test(limit)")
                else ("/fapi/v1/order", "futures/order(limit)")
        send ts = do
            let params = baseParams ts ++ reduceOnlyParams ++ clientIdParam ++ [("newOrderRespType", "RESULT")]
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
            binanceHttp env label req

    resp <- withBinanceTimestampRetry env send
    ensure2xx label resp
    pure (responseBody resp)

newtype Ticker24hPrice = Ticker24hPrice {t24LastPrice :: Double}

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
        req = req0{method = "GET", queryString = qs}
    resp <- binanceHttp env "ticker/24hr" req
    ensure2xx "ticker/24hr" resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode ticker/24hr: " ++ e))
        Right (Ticker24hPrice p) -> pure p

newtype FuturesMarkPrice = FuturesMarkPrice {fmpMarkPrice :: Double}

instance FromJSON FuturesMarkPrice where
    parseJSON = withObject "FuturesMarkPrice" $ \o -> do
        pTxt <- o .: "markPrice"
        p <- parseDoubleText pTxt
        pure (FuturesMarkPrice p)

fetchFuturesMarkPrice :: BinanceEnv -> String -> IO Double
fetchFuturesMarkPrice env symbol = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "fetchFuturesMarkPrice requires MarketFutures")
    req0 <- parseRequest (beBaseUrl env ++ "/fapi/v1/premiumIndex")
    let qs = renderSimpleQuery True [("symbol", BS.pack (map toUpperAscii symbol))]
        req = req0{method = "GET", queryString = qs}
    resp <- binanceHttp env "premiumIndex" req
    ensure2xx "premiumIndex" resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode premiumIndex: " ++ e))
        Right (FuturesMarkPrice p) -> pure p

{- | A single @(timestampMs, value)@ observation from a Binance derivatives-stats
endpoint. Binance returns the numeric value as a JSON string and the timestamp
as a number; the field names vary by endpoint, so each newtype fixes them.
-}
newtype TsValueRow = TsValueRow {unTsValueRow :: (Int64, Double)}

parseTsValue :: Text -> Text -> Aeson.Value -> AT.Parser TsValueRow
parseTsValue tsKey valKey =
    withObject "TsValueRow" $ \o -> do
        ts <- o .: AK.fromText tsKey
        valTxt <- o .: AK.fromText valKey
        val <- parseDoubleText valTxt
        pure (TsValueRow (ts, val))

-- | Fetch + decode a derivatives-stats array, returning ascending @(ts, value)@.
fetchTsValueSeries ::
    BinanceEnv ->
    String -> -- request label (for errors/metrics)
    String -> -- path
    [(BS.ByteString, BS.ByteString)] -> -- query params
    Text -> -- timestamp JSON key
    Text -> -- value JSON key
    IO [(Int64, Double)]
fetchTsValueSeries env label path params tsKey valKey = do
    req0 <- parseRequest (beBaseUrl env ++ path)
    let req = req0{method = "GET", queryString = renderSimpleQuery True params}
    resp <- binanceHttp env label req
    ensure2xx label resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode " ++ label ++ ": " ++ e))
        Right vals ->
            case AT.parseEither (traverse (parseTsValue tsKey valKey)) vals of
                Left e -> throwIO (userError ("Failed to decode " ++ label ++ " rows: " ++ e))
                Right rows -> pure (sortOn fst (map unTsValueRow rows))

symBytes :: String -> BS.ByteString
symBytes = BS.pack . map toUpperAscii

clampLimit :: Int -> Int -> BS.ByteString
clampLimit hi n = BS.pack (show (max 1 (min hi n)))

{- | Historical perp funding rate, ascending @(fundingTimeMs, rate)@.
@\/fapi\/v1\/fundingRate@ (no auth). Funding settles on a schedule (~8h), so
align to bars with point-in-time forward-fill before use.
-}
fetchFundingRateHistory :: BinanceEnv -> String -> Int -> IO [(Int64, Double)]
fetchFundingRateHistory env symbol limit =
    fetchTsValueSeries
        env
        "fundingRate"
        "/fapi/v1/fundingRate"
        [("symbol", symBytes symbol), ("limit", clampLimit 1000 limit)]
        "fundingTime"
        "fundingRate"

{- | Historical open interest, ascending @(timestampMs, sumOpenInterest)@.
@\/futures\/data\/openInterestHist@. @period@ is one of
@5m,15m,30m,1h,2h,4h,6h,12h,1d@; only ~30 days of history are available at fine
periods. Requires a futures env.
-}
fetchOpenInterestHist :: BinanceEnv -> String -> String -> Int -> IO [(Int64, Double)]
fetchOpenInterestHist env symbol period limit = do
    Control.Monad.when (beMarket env /= MarketFutures) $
        throwIO (userError "fetchOpenInterestHist requires MarketFutures")
    fetchTsValueSeries
        env
        "openInterestHist"
        "/futures/data/openInterestHist"
        [("symbol", symBytes symbol), ("period", BS.pack period), ("limit", clampLimit 500 limit)]
        "timestamp"
        "sumOpenInterest"

{- | Historical taker buy/sell volume ratio (CVD proxy), ascending
@(timestampMs, buySellRatio)@. @\/futures\/data\/takerlongshortRatio@.
-}
fetchTakerLongShortRatio :: BinanceEnv -> String -> String -> Int -> IO [(Int64, Double)]
fetchTakerLongShortRatio env symbol period limit = do
    Control.Monad.when (beMarket env /= MarketFutures) $
        throwIO (userError "fetchTakerLongShortRatio requires MarketFutures")
    fetchTsValueSeries
        env
        "takerlongshortRatio"
        "/futures/data/takerlongshortRatio"
        [("symbol", symBytes symbol), ("period", BS.pack period), ("limit", clampLimit 500 limit)]
        "timestamp"
        "buySellRatio"

{- | Historical futures-vs-index basis rate, ascending @(timestampMs, basisRate)@.
@\/futures\/data\/basis@ (keyed by @pair@ + @contractType=PERPETUAL@).
-}
fetchBasisHistory :: BinanceEnv -> String -> String -> Int -> IO [(Int64, Double)]
fetchBasisHistory env pair period limit = do
    Control.Monad.when (beMarket env /= MarketFutures) $
        throwIO (userError "fetchBasisHistory requires MarketFutures")
    fetchTsValueSeries
        env
        "basis"
        "/futures/data/basis"
        [ ("pair", symBytes pair)
        , ("contractType", BS.pack "PERPETUAL")
        , ("period", BS.pack period)
        , ("limit", clampLimit 500 limit)
        ]
        "timestamp"
        "basisRate"

data Ticker24h = Ticker24h
    { t24Symbol :: !String
    , t24QuoteVolume :: !Double
    }
    deriving (Eq, Show)

instance FromJSON Ticker24h where
    parseJSON = withObject "Ticker24h" $ \o -> do
        symTxt <- o .: "symbol"
        qvTxt <- o .: "quoteVolume"
        qv <- parseDoubleText qvTxt
        pure Ticker24h{t24Symbol = T.unpack symTxt, t24QuoteVolume = qv}

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
        let req = req0{method = "GET"}
        resp <- binanceHttp env "ticker/24hr" req
        ensure2xx "ticker/24hr" resp
        case eitherDecode (responseBody resp) of
            Left e -> throwIO (userError ("Failed to decode ticker/24hr: " ++ e))
            Right xs -> pure xs

{- | Returns the highest-volume symbols by 24h quote volume for the provided quote asset.
Filtering is conservative to avoid leveraged tokens and stable-stable pairs.
-}
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
                    sortOn (Data.Ord.Down . snd) $
                        [ (map toUpperAscii (t24Symbol t), max 0 (t24QuoteVolume t))
                        | t <- filter wanted tickers
                        ]
            pure (take topN ranked)

getTimestampMs :: IO Int64
getTimestampMs = do
    t <- getPOSIXTime
    pure (floor (t * 1000))

binanceTimeOffsetCacheKey :: BinanceEnv -> String
binanceTimeOffsetCacheKey env = beBaseUrl env ++ ":" ++ show (beMarket env) ++ ":timeOffsetMs"

getBinanceTimestampMs :: BinanceEnv -> IO Int64
getBinanceTimestampMs env = do
    let key = binanceTimeOffsetCacheKey env
    offsetOrErr <-
        ( try $
            fetchWithCache binanceTimeOffsetCache binanceTimeOffsetFreshTtl binanceTimeOffsetStaleTtl key $ do
                serverMs <- fetchBinanceServerTime env
                localMs <- getTimestampMs
                pure (serverMs - localMs)
        ) ::
            IO (Either SomeException Int64)
    localMs <- getTimestampMs
    case offsetOrErr of
        Right offset -> pure (localMs + offset)
        Left _ -> pure localMs

getBinanceTimestampMsFresh :: BinanceEnv -> IO Int64
getBinanceTimestampMsFresh env = do
    serverMs <- fetchBinanceServerTime env
    localMs <- getTimestampMs
    let offset = serverMs - localMs
        key = binanceTimeOffsetCacheKey env
    insertCache binanceTimeOffsetCache binanceTimeOffsetStaleTtl key offset
    pure (localMs + offset)

fetchBinanceServerTime :: BinanceEnv -> IO Int64
fetchBinanceServerTime env = do
    let path =
            case beMarket env of
                MarketFutures -> "/fapi/v1/time"
                _ -> "/api/v3/time"
    req0 <- parseRequest (beBaseUrl env ++ path)
    let req = req0{method = "GET"}
    resp <- binanceHttp env "time" req
    ensure2xx "time" resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode time: " ++ e))
        Right (BinanceServerTime ts) -> pure ts

binanceTimestampErrorCode :: Int
binanceTimestampErrorCode = -1021

isBinanceTimestampError :: Response BL.ByteString -> Bool
isBinanceTimestampError resp =
    case eitherDecode (responseBody resp) :: Either String BinanceErrorBody of
        Right be ->
            case bebCode be of
                Just code | code == binanceTimestampErrorCode -> True
                _ ->
                    let msg = maybe "" (map toLower) (bebMsg be)
                     in "timestamp for this request is outside of the recvwindow" `isInfixOf` msg
        Left _ -> False

withBinanceTimestampRetry :: BinanceEnv -> (Int64 -> IO (Response BL.ByteString)) -> IO (Response BL.ByteString)
withBinanceTimestampRetry env send = do
    ts <- getBinanceTimestampMs env
    resp <- send ts
    let code = statusCode (responseStatus resp)
    if code >= 200 && code < 300
        then pure resp
        else
            if isBinanceTimestampError resp
                then do
                    tsFresh <- getBinanceTimestampMsFresh env
                    send tsFresh
                else pure resp

signQuery :: BS.ByteString -> BS.ByteString -> BS.ByteString
signQuery secret query =
    let mac :: HMAC SHA256
        mac = hmac secret query
        digest = hmacGetDigest mac
     in B16.encode (convert digest)

placeMarketOrder ::
    BinanceEnv ->
    BinanceOrderMode ->
    String -> -- symbol
    OrderSide ->
    Maybe Double -> -- quantity (base)
    Maybe Double -> -- quoteOrderQty (quote)
    Maybe Bool -> -- reduceOnly (futures only)
    Maybe String -> -- newClientOrderId (optional; idempotency)
    IO BL.ByteString
placeMarketOrder env mode symbol side quantity quoteOrderQty reduceOnly mClientOrderId = do
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
    let sideTxt = case side of Buy -> "BUY"; Sell -> "SELL"
        baseParams ts =
            [ ("symbol", BS.pack (map toUpperAscii symbol))
            , ("side", sideTxt)
            , ("type", "MARKET")
            , ("recvWindow", binanceRecvWindowMs)
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

    (path, label, buildParams) <-
        case beMarket env of
            MarketSpot -> do
                p <-
                    case (quantity, quoteOrderQty) of
                        (Nothing, Nothing) -> throwIO (userError "Provide quantity or quoteOrderQty for MARKET orders")
                        _ -> pure (\ts -> baseParams ts ++ qtyParamsSpotOrMargin ++ clientIdParam)
                pure
                    ( if mode == OrderTest then "/api/v3/order/test" else "/api/v3/order"
                    , if mode == OrderTest then "order/test" else "order"
                    , p
                    )
            MarketMargin -> do
                p <-
                    case (quantity, quoteOrderQty) of
                        (Nothing, Nothing) -> throwIO (userError "Provide quantity or quoteOrderQty for MARKET orders")
                        _ -> pure (\ts -> baseParams ts ++ qtyParamsSpotOrMargin ++ clientIdParam)
                case mode of
                    OrderTest -> throwIO (userError "Margin does not support order test; rerun with --binance-live")
                    OrderLive -> pure ("/sapi/v1/margin/order", "margin/order", p)
            MarketFutures -> do
                -- Futures defaults to newOrderRespType=ACK, whose response says
                -- status=NEW/executedQty=0 even for market orders that filled
                -- instantly; RESULT returns the real fill (executedQty, avgPrice)
                -- so the caller can update position state on the bar it executed.
                p <-
                    case quantity of
                        Nothing -> throwIO (userError "Futures MARKET orders require --order-quantity (or compute it from --order-quote in the caller)")
                        Just _ -> pure (\ts -> baseParams ts ++ qtyParamsFutures ++ reduceOnlyParams ++ clientIdParam ++ [("newOrderRespType", "RESULT")])
                pure
                    ( if mode == OrderTest then "/fapi/v1/order/test" else "/fapi/v1/order"
                    , if mode == OrderTest then "futures/order/test" else "futures/order"
                    , p
                    )

    let send ts = do
            let params = buildParams ts
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
            binanceHttp env label req

    resp <- withBinanceTimestampRetry env send
    ensure2xx label resp
    pure (responseBody resp)

placeFuturesMarketOrderWithPositionSide ::
    BinanceEnv ->
    BinanceOrderMode ->
    String -> -- symbol
    OrderSide ->
    Double -> -- quantity (base)
    Maybe Bool -> -- reduceOnly
    Maybe String -> -- newClientOrderId (optional; idempotency)
    Maybe String -> -- positionSide (optional; required in Hedge Mode)
    IO BL.ByteString
placeFuturesMarketOrderWithPositionSide env mode symbol side quantity reduceOnly mClientOrderId mPositionSide = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "placeFuturesMarketOrderWithPositionSide requires MarketFutures")
    Control.Monad.when (quantity <= 0) $ throwIO (userError "Futures MARKET orders require quantity > 0")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)

    let sideTxt = case side of Buy -> "BUY"; Sell -> "SELL"
        baseParams ts =
            [ ("symbol", BS.pack (map toUpperAscii symbol))
            , ("side", sideTxt)
            , ("type", "MARKET")
            , ("quantity", renderDouble quantity)
            , ("recvWindow", binanceRecvWindowMs)
            , ("timestamp", BS.pack (show ts))
            ]
        reduceOnlyParams =
            case reduceOnly of
                Just True -> [("reduceOnly", "true")]
                _ -> []
        clientIdParam =
            case mClientOrderId of
                Nothing -> []
                Just cid | null (trim cid) -> []
                Just cid -> [("newClientOrderId", BS.pack (trim cid))]
        positionSideParam =
            case mPositionSide of
                Just ps | not (null (trim ps)) -> [("positionSide", BS.pack (map toUpperAscii (trim ps)))]
                _ -> []
        (path, label) =
            if mode == OrderTest
                then ("/fapi/v1/order/test", "futures/order/test")
                else ("/fapi/v1/order", "futures/order")

        send ts = do
            -- RESULT instead of the futures ACK default: see placeMarketOrder.
            let params = baseParams ts ++ reduceOnlyParams ++ positionSideParam ++ clientIdParam ++ [("newOrderRespType", "RESULT")]
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
            binanceHttp env label req

    resp <- withBinanceTimestampRetry env send
    ensure2xx label resp
    pure (responseBody resp)

placeFuturesTriggerMarketOrder ::
    BinanceEnv ->
    BinanceOrderMode ->
    String -> -- symbol
    OrderSide ->
    String -> -- type (e.g., STOP_MARKET, TAKE_PROFIT_MARKET)
    Double -> -- stopPrice
    Maybe String -> -- newClientOrderId (optional; idempotency)
    IO BL.ByteString
placeFuturesTriggerMarketOrder env mode symbol side orderType stopPrice mClientOrderId = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "placeFuturesTriggerMarketOrder requires MarketFutures")
    Control.Monad.when (stopPrice <= 0) $ throwIO (userError "stopPrice must be > 0")
    let orderType' = trim orderType
    Control.Monad.when (null orderType') $ throwIO (userError "orderType must be non-empty")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)

    let sideTxt = case side of Buy -> "BUY"; Sell -> "SELL"
        baseParams ts =
            [ ("symbol", BS.pack (map toUpperAscii symbol))
            , ("side", sideTxt)
            , ("type", BS.pack orderType')
            , ("stopPrice", renderDouble stopPrice)
            , ("closePosition", "true")
            , ("recvWindow", binanceRecvWindowMs)
            , ("timestamp", BS.pack (show ts))
            ]
        clientIdParam =
            case mClientOrderId of
                Nothing -> []
                Just cid | null (trim cid) -> []
                Just cid -> [("newClientOrderId", BS.pack (trim cid))]

        (path, label) =
            if mode == OrderTest
                then ("/fapi/v1/order/test", "futures/order/test(trigger)")
                else ("/fapi/v1/order", "futures/order(trigger)")
        send ts = do
            let params = baseParams ts ++ clientIdParam
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
            binanceHttp env label req

    resp <- withBinanceTimestampRetry env send
    ensure2xx label resp
    pure (responseBody resp)

placeFuturesAlgoTriggerMarketOrder ::
    BinanceEnv ->
    BinanceOrderMode ->
    String -> -- symbol
    OrderSide ->
    String -> -- type (e.g., STOP_MARKET, TAKE_PROFIT_MARKET)
    Double -> -- triggerPrice
    Maybe String -> -- clientAlgoId (optional; idempotency)
    Maybe String -> -- positionSide (optional; required in Hedge Mode)
    IO BL.ByteString
placeFuturesAlgoTriggerMarketOrder env mode symbol side orderType triggerPrice mClientAlgoId mPositionSide = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "placeFuturesAlgoTriggerMarketOrder requires MarketFutures")
    Control.Monad.when (mode == OrderTest) $ throwIO (userError "Algo orders are not supported in test mode")
    Control.Monad.when (triggerPrice <= 0) $ throwIO (userError "triggerPrice must be > 0")
    let orderType' = trim orderType
    Control.Monad.when (null orderType') $ throwIO (userError "orderType must be non-empty")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)

    let sideTxt = case side of Buy -> "BUY"; Sell -> "SELL"
        algoType = "CONDITIONAL"
        baseParams ts =
            [ ("symbol", BS.pack (map toUpperAscii symbol))
            , ("side", sideTxt)
            , ("algoType", BS.pack algoType)
            , ("type", BS.pack orderType')
            , ("triggerPrice", renderDouble triggerPrice)
            , ("closePosition", "true")
            , ("recvWindow", binanceRecvWindowMs)
            , ("timestamp", BS.pack (show ts))
            ]
        positionSideParam =
            case mPositionSide of
                Nothing -> []
                Just ps | null (trim ps) -> []
                Just ps -> [("positionSide", BS.pack (map toUpperAscii (trim ps)))]
        clientIdParam =
            case mClientAlgoId of
                Nothing -> []
                Just cid | null (trim cid) -> []
                Just cid -> [("clientAlgoId", BS.pack (trim cid))]
        path = "/fapi/v1/algoOrder"
        label = "futures/algoOrder(trigger)"
        send ts = do
            let params = baseParams ts ++ positionSideParam ++ clientIdParam
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
            binanceHttp env label req

    resp <- withBinanceTimestampRetry env send
    ensure2xx label resp
    pure (responseBody resp)

fetchOrderByClientId :: BinanceEnv -> String -> String -> IO BL.ByteString
fetchOrderByClientId env symbol clientOrderId = do
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)

    let (path, label) =
            case beMarket env of
                MarketSpot -> ("/api/v3/order", "order/get")
                MarketMargin -> ("/sapi/v1/margin/order", "margin/order/get")
                MarketFutures -> ("/fapi/v1/order", "futures/order/get")
        send ts = do
            let params =
                    [ ("symbol", BS.pack (map toUpperAscii symbol))
                    , ("origClientOrderId", BS.pack (trim clientOrderId))
                    , ("timestamp", BS.pack (show ts))
                    , ("recvWindow", binanceRecvWindowMs)
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
            binanceHttp env label req
    resp <- withBinanceTimestampRetry env send
    ensure2xx label resp
    pure (responseBody resp)

fetchAccountTrades :: BinanceEnv -> Maybe String -> Maybe Int -> Maybe Int64 -> Maybe Int64 -> Maybe Int64 -> IO [BinanceTrade]
fetchAccountTrades env mSymbol mLimit mStartTime mEndTime mFromId = do
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
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
        (path, label) =
            case beMarket env of
                MarketSpot -> ("/api/v3/myTrades", "account/myTrades")
                MarketMargin -> ("/sapi/v1/margin/myTrades", "margin/myTrades")
                MarketFutures -> ("/fapi/v1/userTrades", "futures/userTrades")
        send ts = do
            let baseParams =
                    [ ("timestamp", BS.pack (show ts))
                    , ("recvWindow", binanceRecvWindowMs)
                    ]
                params = symbolParam ++ limitParam ++ startTimeParam ++ endTimeParam ++ fromIdParam ++ baseParams
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
            binanceHttp env label req

    resp <- withBinanceTimestampRetry env send
    ensure2xx label resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode " ++ label ++ ": " ++ e))
        Right trades -> pure trades

fetchFuturesIncome :: BinanceEnv -> Maybe String -> Maybe String -> Maybe Int64 -> Maybe Int64 -> Maybe Int -> Maybe Int -> IO [FuturesIncome]
fetchFuturesIncome env mSymbol mIncomeType mStartTime mEndTime mPage mLimit = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "fetchFuturesIncome requires MarketFutures")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
    let optionalTextParam name normalizeValue raw =
            case fmap (trim . normalizeValue) raw of
                Just value | not (null value) -> [(name, BS.pack value)]
                _ -> []
        optionalTimeParam name raw =
            case raw of
                Nothing -> []
                Just value -> [(name, BS.pack (show (max 0 value)))]
        pageParam = maybe [] (\value -> [("page", BS.pack (show (max 1 value)))]) mPage
        limitParam = maybe [] (\value -> [("limit", BS.pack (show (max 1 (min 1000 value))))]) mLimit
        send ts = do
            let params =
                    optionalTextParam "symbol" (map toUpperAscii) mSymbol
                        ++ optionalTextParam "incomeType" (map toUpperAscii) mIncomeType
                        ++ optionalTimeParam "startTime" mStartTime
                        ++ optionalTimeParam "endTime" mEndTime
                        ++ pageParam
                        ++ limitParam
                        ++ [ ("timestamp", BS.pack (show ts))
                           , ("recvWindow", binanceRecvWindowMs)
                           ]
                queryToSign = renderSimpleQuery False params
                sig = signQuery secret queryToSign
                paramsSigned = params ++ [("signature", sig)]
                qs = renderSimpleQuery True paramsSigned
            req0 <- parseRequest (beBaseUrl env ++ "/fapi/v1/income")
            let req =
                    req0
                        { method = "GET"
                        , queryString = qs
                        , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
                        }
            binanceHttp env "futures/income" req
    resp <- withBinanceTimestampRetry env send
    ensure2xx "futures/income" resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode futures/income: " ++ e))
        Right rows -> pure rows

newtype FuturesOpenOrder = FuturesOpenOrder
    { fooClientOrderId :: String
    }
    deriving (Eq, Show)

data BinanceOpenOrder = BinanceOpenOrder
    { booClientOrderId :: !(Maybe String)
    , booSide :: !(Maybe OrderSide)
    , booReduceOnly :: !(Maybe Bool)
    , booClosePosition :: !(Maybe Bool)
    , booPositionSide :: !(Maybe String)
    }
    deriving (Eq, Show)

parseOrderSide :: String -> Maybe OrderSide
parseOrderSide raw =
    case map toUpperAscii raw of
        "BUY" -> Just Buy
        "SELL" -> Just Sell
        _ -> Nothing

instance FromJSON FuturesOpenOrder where
    parseJSON = withObject "FuturesOpenOrder" $ \o -> do
        cid <- o .: "clientOrderId"
        pure FuturesOpenOrder{fooClientOrderId = cid}

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
    (FromJSON a) =>
    BinanceEnv ->
    String ->
    String ->
    String ->
    IO [a]
fetchOpenOrdersWith env label path symbol = do
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
    let send ts = do
            let params =
                    [ ("symbol", BS.pack (map toUpperAscii symbol))
                    , ("timestamp", BS.pack (show ts))
                    , ("recvWindow", binanceRecvWindowMs)
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
            binanceHttp env label req
    resp <- withBinanceTimestampRetry env send
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
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "fetchFuturesOpenOrders requires MarketFutures")
    fetchOpenOrdersWith env "futures/openOrders" "/fapi/v1/openOrders" symbol

cancelFuturesOrderByClientId :: BinanceEnv -> String -> String -> IO BL.ByteString
cancelFuturesOrderByClientId env symbol clientOrderId = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "cancelFuturesOrderByClientId requires MarketFutures")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
    let send ts = do
            let params =
                    [ ("symbol", BS.pack (map toUpperAscii symbol))
                    , ("origClientOrderId", BS.pack (trim clientOrderId))
                    , ("timestamp", BS.pack (show ts))
                    , ("recvWindow", binanceRecvWindowMs)
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
            binanceHttp env "futures/order/cancel" req
    resp <- withBinanceTimestampRetry env send
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
    let path =
            case beMarket env of
                MarketSpot -> "/api/v3/account"
                MarketMargin -> "/sapi/v1/margin/account"
                MarketFutures -> "/api/v3/account"
        label =
            if beMarket env == MarketMargin then "margin/account" else "account"
        send ts = do
            let params =
                    [ ("timestamp", BS.pack (show ts))
                    , ("recvWindow", binanceRecvWindowMs)
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
            binanceHttp env label req
    resp <- withBinanceTimestampRetry env send
    ensure2xx label resp
    case beMarket env of
        MarketSpot ->
            case eitherDecode (responseBody resp) of
                Left e -> throwIO (userError ("Failed to decode account: " ++ e))
                Right (Account balances) ->
                    let sym = map toUpperAscii asset
                        match b = map toUpperAscii (baAsset b) == sym
                     in case filter match balances of
                            (b : _) -> pure (baFree b)
                            [] -> pure 0
        MarketMargin ->
            case eitherDecode (responseBody resp) of
                Left e -> throwIO (userError ("Failed to decode margin account: " ++ e))
                Right (MarginAccount balances) ->
                    let sym = map toUpperAscii asset
                        match b = map toUpperAscii (mbaAsset b) == sym
                     in case filter match balances of
                            (b : _) -> pure (mbaNetAsset b)
                            [] -> pure 0
        MarketFutures -> pure 0

newtype Account = Account [Balance]

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
        pure Balance{baAsset = asset, baFree = free}

newtype MarginAccount = MarginAccount [MarginBalance]

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
        pure MarginBalance{mbaAsset = asset, mbaNetAsset = net}

fetchFuturesAvailableBalance :: BinanceEnv -> String -> IO Double
fetchFuturesAvailableBalance env asset = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "fetchFuturesAvailableBalance requires MarketFutures")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
    let send ts = do
            let params =
                    [ ("timestamp", BS.pack (show ts))
                    , ("recvWindow", binanceRecvWindowMs)
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
            binanceHttp env "futures/balance" req
    resp <- withBinanceTimestampRetry env send
    ensure2xx "futures/balance" resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode futures balance: " ++ e))
        Right bals ->
            let sym = map toUpperAscii asset
                match b = map toUpperAscii (fbAsset b) == sym
             in case filter match bals of
                    (b : _) -> pure (fbAvailableBalance b)
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
        pure FuturesBalance{fbAsset = sym, fbAvailableBalance = avail}

newtype FuturesAccountInfo = FuturesAccountInfo
    { faiUid :: Maybe Int64
    }

instance FromJSON FuturesAccountInfo where
    parseJSON = withObject "FuturesAccountInfo" $ \o -> do
        uid <- o AT..:? "uid"
        pure FuturesAccountInfo{faiUid = uid}

fetchFuturesAccountUid :: BinanceEnv -> IO (Maybe Int64)
fetchFuturesAccountUid env = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "fetchFuturesAccountUid requires MarketFutures")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
    let send ts = do
            let params =
                    [ ("timestamp", BS.pack (show ts))
                    , ("recvWindow", binanceRecvWindowMs)
                    ]
                queryToSign = renderSimpleQuery False params
                sig = signQuery secret queryToSign
                paramsSigned = params ++ [("signature", sig)]
                qs = renderSimpleQuery True paramsSigned

            req0 <- parseRequest (beBaseUrl env ++ "/fapi/v2/account")
            let req =
                    req0
                        { method = "GET"
                        , queryString = qs
                        , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
                        }
            binanceHttp env "futures/account" req
    resp <- withBinanceTimestampRetry env send
    ensure2xx "futures/account" resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode futures account: " ++ e))
        Right info -> pure (faiUid info)

fetchFuturesPositionAmt :: BinanceEnv -> String -> IO Double
fetchFuturesPositionAmt env symbol = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "fetchFuturesPositionAmt requires MarketFutures")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
    let send ts = do
            let params =
                    [ ("symbol", BS.pack (map toUpperAscii symbol))
                    , ("timestamp", BS.pack (show ts))
                    , ("recvWindow", binanceRecvWindowMs)
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
            binanceHttp env "futures/positionRisk" req
    resp <- withBinanceTimestampRetry env send
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
    }
    deriving (Eq, Show)

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

futuresPositionRiskLeverageSane :: FuturesPositionRisk -> Bool
futuresPositionRiskLeverageSane fpr =
    let lev = fprLeverage fpr
     in not (isNaN lev || isInfinite lev) && lev > 0 && lev <= 125

fetchFuturesPositionRisks :: BinanceEnv -> IO [FuturesPositionRisk]
fetchFuturesPositionRisks =
    fetchFuturesPositionRisksWithRequest id

fetchFuturesPositionRisksWithResponseTimeout :: Int -> BinanceEnv -> IO [FuturesPositionRisk]
fetchFuturesPositionRisksWithResponseTimeout timeoutMicros =
    fetchFuturesPositionRisksWithRequest $ \req ->
        req{responseTimeout = responseTimeoutMicro (max 1 timeoutMicros)}

fetchFuturesPositionRisksWithRequest :: (Request -> Request) -> BinanceEnv -> IO [FuturesPositionRisk]
fetchFuturesPositionRisksWithRequest adjustRequest env = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "fetchFuturesPositionRisks requires MarketFutures")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
    let send ts = do
            let params =
                    [ ("timestamp", BS.pack (show ts))
                    , ("recvWindow", binanceRecvWindowMs)
                    ]
                queryToSign = renderSimpleQuery False params
                sig = signQuery secret queryToSign
                paramsSigned = params ++ [("signature", sig)]
                qs = renderSimpleQuery True paramsSigned
            req0 <- parseRequest (beBaseUrl env ++ "/fapi/v2/positionRisk")
            let req =
                    adjustRequest $
                        req0
                            { method = "GET"
                            , queryString = qs
                            , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
                            }
            binanceHttp env "futures/positionRisk" req
    resp <- withBinanceTimestampRetry env send
    ensure2xx "futures/positionRisk" resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode futures positionRisk: " ++ e))
        Right positions -> pure positions

data FuturesAlgoOpenOrder = FuturesAlgoOpenOrder
    { faoAlgoId :: !Int64
    , faoClientAlgoId :: !(Maybe String)
    , faoSymbol :: !String
    , faoPositionSide :: !(Maybe String)
    }
    deriving (Eq, Show)

instance FromJSON FuturesAlgoOpenOrder where
    parseJSON = withObject "FuturesAlgoOpenOrder" $ \o -> do
        algoId <- o .: "algoId"
        clientAlgoId <- o AT..:? "clientAlgoId"
        sym <- o .: "symbol"
        positionSide <- o AT..:? "positionSide"
        pure
            FuturesAlgoOpenOrder
                { faoAlgoId = algoId
                , faoClientAlgoId = clientAlgoId
                , faoSymbol = sym
                , faoPositionSide = positionSide
                }

fetchFuturesOpenAlgoOrders :: BinanceEnv -> String -> IO [FuturesAlgoOpenOrder]
fetchFuturesOpenAlgoOrders env symbol = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "fetchFuturesOpenAlgoOrders requires MarketFutures")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
    let send ts = do
            let params =
                    [ ("symbol", BS.pack (map toUpperAscii symbol))
                    , ("timestamp", BS.pack (show ts))
                    , ("recvWindow", binanceRecvWindowMs)
                    ]
                queryToSign = renderSimpleQuery False params
                sig = signQuery secret queryToSign
                paramsSigned = params ++ [("signature", sig)]
                qs = renderSimpleQuery True paramsSigned
            req0 <- parseRequest (beBaseUrl env ++ "/fapi/v1/openAlgoOrders")
            let req =
                    req0
                        { method = "GET"
                        , queryString = qs
                        , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
                        }
            binanceHttp env "futures/openAlgoOrders" req
    resp <- withBinanceTimestampRetry env send
    ensure2xx "futures/openAlgoOrders" resp
    case eitherDecode (responseBody resp) of
        Left e -> throwIO (userError ("Failed to decode futures openAlgoOrders: " ++ e))
        Right orders -> pure orders

cancelFuturesAlgoOrderByClientId :: BinanceEnv -> String -> IO BL.ByteString
cancelFuturesAlgoOrderByClientId env clientAlgoId = do
    Control.Monad.when (beMarket env /= MarketFutures) $ throwIO (userError "cancelFuturesAlgoOrderByClientId requires MarketFutures")
    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)
    secret <- maybe (throwIO (userError "Missing BINANCE_API_SECRET")) pure (beApiSecret env)
    let send ts = do
            let params =
                    [ ("clientAlgoId", BS.pack (trim clientAlgoId))
                    , ("timestamp", BS.pack (show ts))
                    , ("recvWindow", binanceRecvWindowMs)
                    ]
                queryToSign = renderSimpleQuery False params
                sig = signQuery secret queryToSign
                paramsSigned = params ++ [("signature", sig)]
                qs = renderSimpleQuery True paramsSigned
            req0 <- parseRequest (beBaseUrl env ++ "/fapi/v1/algoOrder")
            let req =
                    req0
                        { method = "DELETE"
                        , queryString = qs
                        , requestHeaders = ("X-MBX-APIKEY", apiKey) : requestHeaders req0
                        }
            binanceHttp env "futures/algoOrder/cancel" req
    resp <- withBinanceTimestampRetry env send
    ensure2xx "futures/algoOrder/cancel" resp
    pure (responseBody resp)

renderDouble :: Double -> BS.ByteString
renderDouble x =
    -- Avoid scientific notation; Binance expects decimal strings.
    BS.pack (trimTrailingZeros (showFFloat (Just 8) x ""))

trimTrailingZeros :: String -> String
trimTrailingZeros s =
    case break (== '.') s of
        (a, "") -> a
        (a, '.' : b) ->
            let b' = reverse (dropWhile (== '0') (reverse b))
             in if null b' then a else a ++ "." ++ b'
        _ -> s

toUpperAscii :: Char -> Char
toUpperAscii c =
    if isAsciiLower c
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
    }
    deriving (Eq, Show)

instance FromJSON BinanceErrorBody where
    parseJSON = withObject "BinanceErrorBody" $ \o -> do
        code <- o Aeson..:? "code"
        msg <- o Aeson..:? "msg"
        pure BinanceErrorBody{bebCode = code, bebMsg = msg}

newtype ListenKeyResponse = ListenKeyResponse {lkrListenKey :: String}

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

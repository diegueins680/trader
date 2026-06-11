{-# LANGUAGE OverloadedStrings #-}

module Trader.Coinbase (
    CoinbaseCandle (..),
    CoinbaseAccount (..),
    CoinbaseOrderInfo (..),
    CoinbaseEnv (..),
    coinbaseBaseUrl,
    newCoinbaseEnv,
    fetchCoinbaseAccounts,
    fetchCoinbaseAvailableBalance,
    fetchCoinbaseBaseMinSize,
    fetchCoinbaseOrderById,
    fetchCoinbaseCandles,
    decodeCoinbaseCandles,
    decodeCoinbaseOrderInfo,
    buildRanges,
    coinbaseCandlesCacheStats,
    normalizeCoinbaseCandles,
    placeCoinbaseMarketOrder,
    coinbaseProductFromBinance,
    alignCoinbaseClosesToGrid,
) where

import Control.Applicative ((<|>))
import Control.Exception (throwIO)
import qualified Control.Monad
import Crypto.Hash (SHA256)
import Crypto.MAC.HMAC (HMAC, hmac, hmacGetDigest)
import Data.Aeson (FromJSON (..), Value (..), eitherDecode, encode, object, withArray, withObject, (.:), (.:?), (.=))
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.Types as AT
import qualified Data.ByteArray as BA
import qualified Data.ByteArray.Encoding as BAE
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BS8
import qualified Data.ByteString.Lazy as BL
import Data.Char (isAsciiLower, isAsciiUpper)
import Data.Int (Int64)
import Data.List (find, sortOn)
import qualified Data.Map.Strict as Map
import Data.Maybe (catMaybes, isNothing)
import qualified Data.Scientific as Scientific
import qualified Data.Set as Set
import qualified Data.Text as T
import Data.Time.Clock (NominalDiffTime)
import Data.Time.Clock.POSIX (getPOSIXTime, posixSecondsToUTCTime)
import Data.Time.Format (defaultTimeLocale, formatTime)
import qualified Data.Vector as V
import Network.HTTP.Client
import Network.HTTP.Types.Status (statusCode)
import Numeric (showFFloat)
import System.IO.Unsafe (unsafePerformIO)
import Text.Read (readMaybe)
import Trader.Cache (TtlCache, TtlCacheStats, cacheStats, fetchWithCache, newTtlCacheWithMaxEntries)
import Trader.Http (defaultRetryConfig, getSharedManager, httpLbsWithRetry, newHttpManager)
import Trader.Symbol (splitSymbol)
import Trader.Text (trim)

data CoinbaseCandle = CoinbaseCandle
    { ccOpenTime :: !Int64
    , ccHigh :: !Double
    , ccLow :: !Double
    , ccClose :: !Double
    }
    deriving (Eq, Show)

data CoinbaseAccount = CoinbaseAccount
    { caCurrency :: !String
    , caAvailable :: !Double
    }
    deriving (Eq, Show)

data CoinbaseOrderInfo = CoinbaseOrderInfo
    { coiOrderId :: !(Maybe String)
    , coiClientOrderId :: !(Maybe String)
    , coiStatus :: !(Maybe String)
    , coiFilledSize :: !(Maybe Double)
    , coiExecutedValue :: !(Maybe Double)
    }
    deriving (Eq, Show)

instance FromJSON CoinbaseAccount where
    parseJSON =
        withObject "CoinbaseAccount" $ \o -> do
            cur <- o .: "currency"
            availVal <- o .: "available"
            avail <- parseDoubleValue availVal
            pure CoinbaseAccount{caCurrency = cur, caAvailable = avail}

instance FromJSON CoinbaseOrderInfo where
    parseJSON =
        withObject "CoinbaseOrderInfo" $ \o -> do
            successInfo <- parseNestedInfo o "success_response"
            orderInfo <- parseNestedInfo o "order"
            let nested = catMaybes [successInfo, orderInfo]
                pickNested f = foldr ((<|>) . f) Nothing nested
            oid <- parseMaybeStringField o ["id", "order_id"]
            cid <- parseMaybeStringField o ["client_oid", "client_order_id"]
            st <- parseMaybeStringField o ["status", "order_status"]
            filled <- parseMaybeDoubleField o ["filled_size", "filledSize"]
            executedValue <- parseMaybeDoubleField o ["executed_value", "filled_value", "filledValue", "total_value_after_fees"]
            pure
                CoinbaseOrderInfo
                    { coiOrderId = oid <|> pickNested coiOrderId
                    , coiClientOrderId = cid <|> pickNested coiClientOrderId
                    , coiStatus = st <|> pickNested coiStatus
                    , coiFilledSize = filled <|> pickNested coiFilledSize
                    , coiExecutedValue = executedValue <|> pickNested coiExecutedValue
                    }

parseNestedInfo :: AT.Object -> String -> AT.Parser (Maybe CoinbaseOrderInfo)
parseNestedInfo o key = do
    mv <- o .:? AK.fromString key
    case mv of
        Nothing -> pure Nothing
        Just v -> Just <$> parseJSON v

parseMaybeStringField :: AT.Object -> [String] -> AT.Parser (Maybe String)
parseMaybeStringField _ [] = pure Nothing
parseMaybeStringField o (key : keys) = do
    mv <- o .:? AK.fromString key
    case mv of
        Nothing -> parseMaybeStringField o keys
        Just v -> pure (Just v)

parseMaybeDoubleField :: AT.Object -> [String] -> AT.Parser (Maybe Double)
parseMaybeDoubleField _ [] = pure Nothing
parseMaybeDoubleField o (key : keys) = do
    mv <- o .:? AK.fromString key
    case mv of
        Nothing -> parseMaybeDoubleField o keys
        Just v -> Just <$> parseDoubleValue v

coinbaseBaseUrl :: String
coinbaseBaseUrl = "https://api.exchange.coinbase.com"

coinbaseTimeoutMicros :: Int
coinbaseTimeoutMicros = 15 * 1000000

coinbaseMaxBarsPerRequest :: Int
coinbaseMaxBarsPerRequest = 300

{-# NOINLINE coinbaseCandlesCache #-}
coinbaseCandlesCache :: TtlCache String [CoinbaseCandle]
coinbaseCandlesCache = unsafePerformIO (newTtlCacheWithMaxEntries coinbaseCandlesMaxEntries)

coinbaseCandlesMaxEntries :: Int
coinbaseCandlesMaxEntries = 64

coinbaseCandlesFreshTtl :: NominalDiffTime
coinbaseCandlesFreshTtl = 30

coinbaseCandlesStaleTtl :: NominalDiffTime
coinbaseCandlesStaleTtl = 300

coinbaseCandlesCacheStats :: IO TtlCacheStats
coinbaseCandlesCacheStats = cacheStats coinbaseCandlesCache coinbaseCandlesStaleTtl

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
    pure CoinbaseEnv{ceManager = mgr, ceBaseUrl = coinbaseBaseUrl, ceApiKey = apiKey, ceApiSecret = apiSecret, ceApiPassphrase = apiPassphrase}

coinbaseHttp :: CoinbaseEnv -> String -> Request -> IO (Response BL.ByteString)
coinbaseHttp env label = httpLbsWithRetry defaultRetryConfig (Just label) (ceManager env)

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

fetchCoinbaseBaseMinSize :: CoinbaseEnv -> String -> IO (Maybe Double)
fetchCoinbaseBaseMinSize env product = do
    let cleanedProduct = map toUpperAscii (trim product)
    req0 <- parseRequest (ceBaseUrl env ++ "/products/" ++ cleanedProduct)
    let req =
            req0
                { requestHeaders =
                    ("User-Agent", BS8.pack "trader-hs/0.1")
                        : ("Accept", BS8.pack "application/json")
                        : requestHeaders req0
                , responseTimeout = responseTimeoutMicro coinbaseTimeoutMicros
                }
    resp <- coinbaseHttp env "coinbase.product" req
    ensure2xx "Coinbase product" resp
    payload <-
        case eitherDecode (responseBody resp) of
            Left err -> throwIO (userError ("Failed to decode Coinbase product: " ++ err))
            Right ok -> pure ok
    case AT.parseEither parseCoinbaseBaseMinSize payload of
        Left err -> throwIO (userError ("Failed to parse Coinbase product: " ++ err))
        Right minQty -> pure minQty

fetchCoinbaseOrderById :: CoinbaseEnv -> String -> IO BL.ByteString
fetchCoinbaseOrderById env orderIdRaw = do
    let orderId = trim orderIdRaw
        path = "/orders/" ++ orderId
    Control.Monad.when (null orderId) $
        throwIO (userError "Coinbase order id is empty.")
    req0 <- parseRequest (ceBaseUrl env ++ path)
    reqSigned <- signCoinbaseRequest env "GET" path BS.empty req0
    let req = reqSigned{responseTimeout = responseTimeoutMicro coinbaseTimeoutMicros}
    resp <- coinbaseHttp env "coinbase.order.get" req
    ensure2xx "Coinbase order lookup" resp
    pure (responseBody resp)

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
    let totalBars = max 1 bars
        key = map toUpperAscii (trim product) ++ ":" ++ show granularity ++ ":" ++ show totalBars
    fetchWithCache coinbaseCandlesCache coinbaseCandlesFreshTtl coinbaseCandlesStaleTtl key $ do
        mgr <- getSharedManager
        nowSec <- (floor <$> getPOSIXTime) :: IO Int64
        let ranges = buildRanges nowSec (fromIntegral granularity) totalBars
        chunks <- mapM (fetchRange mgr) ranges
        pure (normalizeCoinbaseCandles totalBars (concat chunks))
  where
    cleaned = map toUpperAscii (trim product)
    granularity = max 1 granularitySec

    fetchRange manager (startSec, endSec) = do
        req0 <- parseRequest (coinbaseBaseUrl ++ "/products/" ++ cleaned ++ "/candles")
        let req =
                setQueryString
                    [ ("granularity", Just (BS8.pack (show granularity)))
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
        case decodeCoinbaseCandles (responseBody resp) of
            Left err -> throwIO (userError err)
            Right xs -> pure xs

decodeCoinbaseCandles :: BL.ByteString -> Either String [CoinbaseCandle]
decodeCoinbaseCandles raw = do
    v <-
        case eitherDecode raw of
            Left err -> Left ("Failed to decode Coinbase candles: " ++ err)
            Right ok -> Right ok
    case AT.parseEither parseCoinbaseResponse v of
        Left err -> Left ("Failed to parse Coinbase candles: " ++ err)
        Right xs -> Right xs

decodeCoinbaseOrderInfo :: BL.ByteString -> Maybe CoinbaseOrderInfo
decodeCoinbaseOrderInfo raw =
    case eitherDecode raw of
        Left _ -> Nothing
        Right info ->
            if coinbaseOrderInfoEmpty info
                then Nothing
                else Just info

coinbaseOrderInfoEmpty :: CoinbaseOrderInfo -> Bool
coinbaseOrderInfoEmpty info =
    isNothing (coiOrderId info)
        && isNothing (coiClientOrderId info)
        && isNothing (coiStatus info)
        && isNothing (coiFilledSize info)
        && isNothing (coiExecutedValue info)

buildRanges :: Int64 -> Int64 -> Int -> [(Int64, Int64)]
buildRanges endSec granularitySec bars =
    let bars' = max 1 bars
        g = max 1 granularitySec
        endSec' = max 0 endSec
        go acc remaining endTime =
            if remaining <= 0 || endTime <= 0
                then reverse acc
                else
                    let chunkBars = min coinbaseMaxBarsPerRequest remaining
                        spanSec = fromIntegral chunkBars * g
                        startTime = max 0 (endTime - spanSec)
                        acc' = (startTime, endTime) : acc
                     in if remaining - chunkBars <= 0 || startTime <= 0
                            then reverse acc'
                            -- Keep adjacent windows contiguous. If the API is end-inclusive,
                            -- downstream dedup still removes the single overlap safely.
                            else go acc' (remaining - chunkBars) startTime
     in go [] bars' endSec'

formatIso :: Int64 -> BS.ByteString
formatIso sec =
    let t = posixSecondsToUTCTime (fromIntegral sec)
     in BS8.pack (formatTime defaultTimeLocale "%Y-%m-%dT%H:%M:%SZ" t)

parseCoinbaseResponse :: Value -> AT.Parser [CoinbaseCandle]
parseCoinbaseResponse =
    withArray "CoinbaseCandles" (fmap V.toList . V.mapM parseCandle)

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
                pure CoinbaseCandle{ccOpenTime = normalizeTimestamp t, ccHigh = high, ccLow = low, ccClose = close}

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

parseCoinbaseBaseMinSize :: Value -> AT.Parser (Maybe Double)
parseCoinbaseBaseMinSize =
    withObject "CoinbaseProduct" $ \o -> do
        mRaw <- o .:? "base_min_size"
        case mRaw of
            Nothing -> pure Nothing
            Just v -> do
                qty <- parseDoubleValue v
                pure $
                    if qty > 0
                        then Just qty
                        else Nothing

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
                Nothing -> fail "Invalid integer"
        _ -> fail "Expected integer"

parseDoubleValue :: Value -> AT.Parser Double
parseDoubleValue v =
    case v of
        Number n -> parseFiniteDouble (realToFrac n)
        String t ->
            case readMaybe (trim (T.unpack t)) of
                Just x -> parseFiniteDouble x
                _ -> fail "Invalid double"
        _ -> fail "Expected number"

readMaybeInt64 :: String -> Maybe Int64
readMaybeInt64 s =
    readMaybe (trim s)

normalizeTimestamp :: Int64 -> Int64
normalizeTimestamp t =
    if abs (toInteger t) >= 1000000000000
        then t `div` 1000
        else t

parseFiniteDouble :: Double -> AT.Parser Double
parseFiniteDouble x =
    if isNaN x || isInfinite x
        then fail "Invalid finite double"
        else pure x

dedupByTime :: [CoinbaseCandle] -> [CoinbaseCandle]
dedupByTime = go Set.empty
  where
    go _ [] = []
    go seen (y : ys)
        | Set.member (ccOpenTime y) seen = go seen ys
        | otherwise = y : go (Set.insert (ccOpenTime y) seen) ys

normalizeCoinbaseCandles :: Int -> [CoinbaseCandle] -> [CoinbaseCandle]
normalizeCoinbaseCandles bars candles =
    takeLast bars' (dedupByTime (sortOn ccOpenTime candles))
  where
    bars' = max 1 bars

takeLast :: Int -> [a] -> [a]
takeLast n xs
    | n <= 0 = []
    | otherwise =
        let k = length xs - n
         in if k <= 0 then xs else drop k xs

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
    pure req0{method = BS8.pack method, requestHeaders = headers ++ requestHeaders req0}

ensure2xx :: String -> Response BL.ByteString -> IO ()
ensure2xx label resp = do
    let code = statusCode (responseStatus resp)
    Control.Monad.when (code < 200 || code >= 300) $ throwIO (userError (label ++ " request failed (HTTP " ++ show code ++ ")"))

renderDoubleText :: Double -> String
renderDoubleText x =
    trimTrailingZeros (showFFloat (Just 8) x "")

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

toLowerAscii :: Char -> Char
toLowerAscii c =
    if isAsciiUpper c
        then toEnum (fromEnum c + 32)
        else c

{- | Map a Binance symbol to the same-asset Coinbase product id, e.g.
@BTCUSDT -> BTC-USD@. All USD-pegged Binance quotes map to Coinbase USD; any
other quote (e.g. a BTC-quoted pair) returns 'Nothing' since there is no clean
same-asset Coinbase product. Used only for cross-exchange input enrichment, so a
'Nothing' simply means "no Coinbase signal for this symbol".
-}
coinbaseProductFromBinance :: String -> Maybe String
coinbaseProductFromBinance symbol =
    let (base, quote) = splitSymbol symbol
     in if null base
            then Nothing
            else case quote of
                "USDT" -> Just (base ++ "-USD")
                "USDC" -> Just (base ++ "-USD")
                "BUSD" -> Just (base ++ "-USD")
                "FDUSD" -> Just (base ++ "-USD")
                "TUSD" -> Just (base ++ "-USD")
                "USD" -> Just (base ++ "-USD")
                _ -> Nothing

{- | Align Coinbase candle closes onto a Binance bar grid given the Binance bar
open times (in ms) and Binance closes.

For each Binance bar we use the Coinbase bucket with the identical open time, or
the most recent earlier one (point-in-time forward-fill — never a future bucket,
so there is no lookahead). Bars with no Coinbase data yet fall back to the
Binance close (a neutral, zero-basis value). Returns 'Nothing' when no Binance
bar overlaps any Coinbase candle, so callers fail open to Binance-only behavior.

The result has the same length as @binanceCloses@.
-}
alignCoinbaseClosesToGrid :: [Int64] -> V.Vector Double -> [CoinbaseCandle] -> Maybe (V.Vector Double)
alignCoinbaseClosesToGrid binanceOpenTimesMs binanceCloses candles
    | Map.null cmap = Nothing
    | not anyCovered = Nothing
    | otherwise = Just (V.fromList (map fst rows))
  where
    n = V.length binanceCloses
    -- ccOpenTime is in epoch seconds; key the lookup by seconds.
    cmap = Map.fromList [(ccOpenTime c, ccClose c) | c <- candles]
    rows = walk Nothing 0 binanceOpenTimesMs
    anyCovered = any snd rows
    walk _ _ [] = []
    walk lastV idx (otMs : rest) =
        let sec = otMs `div` 1000
            cur = Map.lookup sec cmap <|> lastV
            fallbackClose = if idx >= 0 && idx < n then binanceCloses V.! idx else 0
            row = case cur of
                Just c -> (c, True)
                Nothing -> (fallbackClose, False)
         in row : walk cur (idx + 1) rest

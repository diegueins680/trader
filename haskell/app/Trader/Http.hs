{-# LANGUAGE OverloadedStrings #-}

module Trader.Http (
    RetryConfig (..),
    defaultRetryConfig,
    newHttpManager,
    getSharedManager,
    httpLbsWithRetry,
    parseRetryAfterMsAt,
    parseRetryAfterFromHeadersAt,
    boundedBackoffMs,
    jitteredDelayMs,
    defaultTimeoutMicros,
) where

import Control.Concurrent (MVar, modifyMVar, newMVar, threadDelay)
import Control.Exception (SomeException, displayException, throwIO, try)
import Data.ByteString.Char8 (ByteString)
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (isSpace, toLower)
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Data.List (dropWhileEnd, isInfixOf)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe, mapMaybe)
import Data.Time (UTCTime, defaultTimeLocale, parseTimeM)
import Data.Time.Clock.POSIX (getPOSIXTime, utcTimeToPOSIXSeconds)
import Network.HTTP.Client
import Network.HTTP.Client.TLS (tlsManagerSettings)
import Network.HTTP.Types.Header (ResponseHeaders)
import Network.HTTP.Types.Status (statusCode)
import System.Environment (lookupEnv)
import System.IO (hPutStrLn, stderr)
import System.IO.Unsafe (unsafePerformIO)
import System.Random (randomRIO)
import Text.Read (readMaybe)

data RetryConfig = RetryConfig
    { rcMaxRetries :: !Int
    , rcBaseDelayMs :: !Int
    , rcMaxDelayMs :: !Int
    , rcJitterFrac :: !Double
    , rcRetryWrites :: !Bool
    }

data HttpGlobals = HttpGlobals
    { hgRetryConfig :: !RetryConfig
    , hgSharedManager :: !(MVar (Maybe Manager))
    , hgRateLimiter :: !(MVar (Map ByteString Integer))
    , hgHttpLogFlag :: !(IORef (Maybe Bool))
    }

defaultRetryConfigBase :: RetryConfig
defaultRetryConfigBase =
    RetryConfig
        { rcMaxRetries = 5
        , rcBaseDelayMs = 300
        , rcMaxDelayMs = 12000
        , rcJitterFrac = 0.25
        , rcRetryWrites = False
        }

{-# NOINLINE httpGlobals #-}
httpGlobals :: HttpGlobals
httpGlobals = unsafePerformIO initHttpGlobals

initHttpGlobals :: IO HttpGlobals
initHttpGlobals = do
    retryCfg <- retryConfigFromEnv defaultRetryConfigBase
    sharedMgr <- newMVar Nothing
    limiter <- newMVar Map.empty
    logFlag <- newIORef Nothing
    pure
        HttpGlobals
            { hgRetryConfig = retryCfg
            , hgSharedManager = sharedMgr
            , hgRateLimiter = limiter
            , hgHttpLogFlag = logFlag
            }

defaultRetryConfig :: RetryConfig
defaultRetryConfig = hgRetryConfig httpGlobals

defaultTimeoutMicros :: Int
defaultTimeoutMicros = 15 * 1000000

newHttpManager :: IO Manager
newHttpManager =
    newManager
        tlsManagerSettings
            { managerResponseTimeout = responseTimeoutMicro defaultTimeoutMicros
            , managerConnCount = 50
            , managerIdleConnectionCount = 20
            }

getSharedManager :: IO Manager
getSharedManager =
    modifyMVar (hgSharedManager httpGlobals) $ \cached ->
        case cached of
            Just mgr -> pure (cached, mgr)
            Nothing -> do
                mgr <- newHttpManager
                pure (Just mgr, mgr)

httpLbsWithRetry :: RetryConfig -> Maybe String -> Manager -> Request -> IO (Response BL.ByteString)
httpLbsWithRetry cfg mLabel mgr req0 = go 0
  where
    req = req0
    hostBs = host req
    methodBs = method req
    pathBs = path req
    labelTxt = fromMaybe "-" mLabel

    go attempt = do
        applyRateLimit hostBs
        t0 <- getTimeMs
        respOrErr <- try (httpLbs req mgr) :: IO (Either SomeException (Response BL.ByteString))
        t1 <- getTimeMs
        let latencyMs = max 0 (fromIntegral (t1 - t0) :: Int)
        case respOrErr of
            Left ex ->
                ( if shouldRetryException cfg req attempt
                    then
                        ( do
                            delayMs <- computeDelay cfg attempt Nothing
                            logHttpAttempt labelTxt methodBs hostBs pathBs (Left ex) latencyMs attempt True
                            sleepMs delayMs
                            go (attempt + 1)
                        )
                    else
                        ( do
                            logHttpAttempt labelTxt methodBs hostBs pathBs (Left ex) latencyMs attempt False
                            throwIO ex
                        )
                )
            Right resp -> do
                let code = statusCode (responseStatus resp)
                    retryable = shouldRetryStatus cfg req code attempt
                ( if retryable
                        then
                            ( do
                                retryAfter <- retryAfterMs resp
                                delayMs <- computeDelay cfg attempt retryAfter
                                logHttpAttempt labelTxt methodBs hostBs pathBs (Right resp) latencyMs attempt True
                                sleepMs delayMs
                                go (attempt + 1)
                            )
                        else
                            ( do
                                logHttpAttempt labelTxt methodBs hostBs pathBs (Right resp) latencyMs attempt False
                                pure resp
                            )
                    )

shouldRetryException :: RetryConfig -> Request -> Int -> Bool
shouldRetryException cfg req attempt =
    attempt < rcMaxRetries cfg && isRetryableMethod cfg req

shouldRetryStatus :: RetryConfig -> Request -> Int -> Int -> Bool
shouldRetryStatus cfg req code attempt =
    attempt < rcMaxRetries cfg && isRetryableMethod cfg req && isRetryableStatus code

isRetryableMethod :: RetryConfig -> Request -> Bool
isRetryableMethod cfg req =
    case BS.map toLower (method req) of
        "get" -> True
        "head" -> True
        "options" -> True
        "put" -> rcRetryWrites cfg
        "delete" -> rcRetryWrites cfg
        "post" -> rcRetryWrites cfg
        _ -> False

isRetryableStatus :: Int -> Bool
isRetryableStatus code =
    code == 408
        || code == 425
        || code == 429
        || (code >= 500 && code < 600)

computeDelay :: RetryConfig -> Int -> Maybe Int -> IO Int
computeDelay cfg attempt mRetryAfter = do
    let capped = boundedBackoffMs (rcBaseDelayMs cfg) (rcMaxDelayMs cfg) attempt
    jittered <- applyJitter (rcMaxDelayMs cfg) capped (rcJitterFrac cfg)
    pure $
        case mRetryAfter of
            Nothing -> jittered
            Just ra -> max jittered ra

boundedBackoffMs :: Int -> Int -> Int -> Int
boundedBackoffMs baseDelay maxDelay attempt =
    let base = toInteger (max 0 baseDelay)
        cap = toInteger (max 0 maxDelay)
        intCap = toInteger (maxBound :: Int)
        steps = max 0 attempt
        go n current
            | current <= 0 = 0
            | current >= cap = cap
            | n <= 0 = min cap current
            | otherwise = go (n - 1) (min cap (current * 2))
        bounded = go steps base
     in fromInteger (min intCap (max 0 bounded))

applyJitter :: Int -> Int -> Double -> IO Int
applyJitter maxDelay delayMs jitterFrac =
    if delayMs <= 0 || jitterFrac <= 0
        then pure (jitteredDelayMs maxDelay delayMs 0)
        else do
            skew <- randomRIO (-jitterFrac, jitterFrac)
            pure (jitteredDelayMs maxDelay delayMs skew)

jitteredDelayMs :: Int -> Int -> Double -> Int
jitteredDelayMs maxDelay delayMs skew =
    let cap = max 0 maxDelay
        base = max 0 delayMs
        scaled = fromIntegral base * (1 + skew)
        rounded = round scaled
     in max 0 (min cap rounded)

retryAfterMs :: Response BL.ByteString -> IO (Maybe Int)
retryAfterMs resp =
    do
        nowMs <- getTimeMs
        pure (parseRetryAfterFromHeadersAt nowMs (responseHeaders resp))

parseRetryAfterFromHeadersAt :: Integer -> ResponseHeaders -> Maybe Int
parseRetryAfterFromHeadersAt nowMs headers =
    case mapMaybe parseRetryAfterValue headers of
        [] -> Nothing
        (delayMs : _) -> Just delayMs
  where
    parseRetryAfterValue (name, value)
        | name == "Retry-After" = parseRetryAfterMsAt nowMs value
        | otherwise = Nothing

parseRetryAfterMsAt :: Integer -> ByteString -> Maybe Int
parseRetryAfterMsAt nowMs raw =
    let txt = trimSpaces (BS.unpack raw)
        clampMs ms =
            let bounded = max 0 (min ms (toInteger maxSleepMs))
             in fromInteger bounded
     in if null txt
            then Nothing
            else case readMaybe txt :: Maybe Integer of
                Just sec | sec >= 0 -> Just (clampMs (sec * 1000))
                _ ->
                    let parseDate fmt =
                            (parseTimeM True defaultTimeLocale fmt txt :: Maybe UTCTime)
                        formats =
                            [ "%a, %d %b %Y %H:%M:%S GMT"
                            , "%A, %d-%b-%y %H:%M:%S GMT"
                            , "%a %b %e %H:%M:%S %Y"
                            ]
                        toDelayMs t =
                            let targetMs = floor (utcTimeToPOSIXSeconds t * 1000) :: Integer
                                delta = targetMs - nowMs
                             in if delta <= 0
                                    then Nothing
                                    else Just (clampMs delta)
                     in case mapMaybe parseDate formats of
                            [] -> Nothing
                            (t : _) -> toDelayMs t

trimSpaces :: String -> String
trimSpaces = dropWhileEnd isSpace . dropWhile isSpace

sleepMs :: Int -> IO ()
sleepMs ms =
    if ms <= 0
        then pure ()
        else do
            let clampedMs = min maxSleepMs ms
            threadDelay (clampedMs * 1000)

maxSleepMs :: Int
maxSleepMs = (maxBound :: Int) `div` 1000

getTimeMs :: IO Integer
getTimeMs = do
    t <- getPOSIXTime
    pure (floor (t * 1000))

applyRateLimit :: ByteString -> IO ()
applyRateLimit host = do
    let delayMs = rateLimitMsForHost host
    if delayMs <= 0
        then pure ()
        else do
            now <- getTimeMs
            waitMs <- modifyMVar (hgRateLimiter httpGlobals) $ \m -> do
                let nextAllowed = Map.findWithDefault now host m
                    startAt = max now nextAllowed
                    waitFor = max 0 (startAt - now)
                    reserveUntil = startAt + fromIntegral delayMs
                pure (Map.insert host reserveUntil m, waitFor)
            sleepMs (fromIntegral waitMs)

rateLimitMsForHost :: ByteString -> Int
rateLimitMsForHost host =
    let h = map toLower (BS.unpack host)
     in if "binance" `isInfixOf` h
            then 80
            else
                if "coinbase" `isInfixOf` h
                    then 150
                    else
                        ( if ("kraken" `isInfixOf` h) || ("poloniex" `isInfixOf` h)
                            then 200
                            else
                                ( if ".s3." `isInfixOf` h
                                    then 50
                                    else 0
                                )
                        )

logHttpAttempt ::
    String ->
    ByteString ->
    ByteString ->
    ByteString ->
    Either SomeException (Response BL.ByteString) ->
    Int ->
    Int ->
    Bool ->
    IO ()
logHttpAttempt label methodBs host pathBs respOrErr latencyMs attempt willRetry = do
    enabled <- isHttpLogEnabled
    if not enabled || label == "-"
        then pure ()
        else do
            let methodTxt = BS.unpack methodBs
                hostTxt = BS.unpack host
                pathTxt = BS.unpack pathBs
                statusTxt =
                    case respOrErr of
                        Left _ -> "error"
                        Right resp -> show (statusCode (responseStatus resp))
                errTxt =
                    case respOrErr of
                        Left ex -> " err=" ++ sanitize (displayException ex)
                        Right _ -> ""
                retryTxt =
                    if willRetry
                        then " retry=true"
                        else ""
                msg =
                    "http.request"
                        ++ " label="
                        ++ label
                        ++ " method="
                        ++ methodTxt
                        ++ " host="
                        ++ hostTxt
                        ++ " path="
                        ++ pathTxt
                        ++ " status="
                        ++ statusTxt
                        ++ " latency_ms="
                        ++ show latencyMs
                        ++ " attempt="
                        ++ show (attempt + 1)
                        ++ retryTxt
                        ++ errTxt
            hPutStrLn stderr msg

isHttpLogEnabled :: IO Bool
isHttpLogEnabled = do
    let flagRef = hgHttpLogFlag httpGlobals
    cached <- readIORef flagRef
    case cached of
        Just v -> pure v
        Nothing -> do
            env <- lookupEnv "TRADER_HTTP_LOG"
            let v = maybe False isTruthy env
            writeIORef flagRef (Just v)
            pure v

isTruthy :: String -> Bool
isTruthy raw = fromMaybe False (parseEnvBool raw)

sanitize :: String -> String
sanitize =
    map (\c -> if c == '\n' || c == '\r' then ' ' else c)

retryConfigFromEnv :: RetryConfig -> IO RetryConfig
retryConfigFromEnv base = do
    maxRetries <- readEnvInt "TRADER_HTTP_RETRY_MAX" (rcMaxRetries base)
    baseDelay <- readEnvInt "TRADER_HTTP_RETRY_BASE_MS" (rcBaseDelayMs base)
    maxDelay <- readEnvInt "TRADER_HTTP_RETRY_MAX_MS" (rcMaxDelayMs base)
    jitter <- readEnvDouble "TRADER_HTTP_RETRY_JITTER" (rcJitterFrac base)
    retryWrites <- readEnvBool "TRADER_HTTP_RETRY_WRITES" (rcRetryWrites base)
    let maxRetries' = max 0 maxRetries
        baseDelay' = max 0 baseDelay
        maxDelay' = max 0 maxDelay
        jitter' = clampDouble 0 1 jitter
    pure
        base
            { rcMaxRetries = maxRetries'
            , rcBaseDelayMs = baseDelay'
            , rcMaxDelayMs = maxDelay'
            , rcJitterFrac = jitter'
            , rcRetryWrites = retryWrites
            }

readEnvInt :: String -> Int -> IO Int
readEnvInt key fallback = do
    raw <- lookupEnv key
    pure $
        fromMaybe fallback (raw >>= parseEnvInt)

readEnvDouble :: String -> Double -> IO Double
readEnvDouble key fallback = do
    raw <- lookupEnv key
    pure $
        fromMaybe fallback (raw >>= parseEnvFiniteDouble)

readEnvBool :: String -> Bool -> IO Bool
readEnvBool key fallback = do
    raw <- lookupEnv key
    pure $
        fromMaybe fallback (raw >>= parseEnvBool)

clampDouble :: Double -> Double -> Double -> Double
clampDouble lo hi v
    | not (isFiniteDouble v) = lo
    | otherwise = max lo (min hi v)

parseEnvInt :: String -> Maybe Int
parseEnvInt = readMaybe . trimSpaces

parseEnvFiniteDouble :: String -> Maybe Double
parseEnvFiniteDouble txt =
    case readMaybe (trimSpaces txt) of
        Just v | isFiniteDouble v -> Just v
        _ -> Nothing

parseEnvBool :: String -> Maybe Bool
parseEnvBool raw =
    case map toLower (trimSpaces raw) of
        "1" -> Just True
        "true" -> Just True
        "yes" -> Just True
        "on" -> Just True
        "0" -> Just False
        "false" -> Just False
        "no" -> Just False
        "off" -> Just False
        _ -> Nothing

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)

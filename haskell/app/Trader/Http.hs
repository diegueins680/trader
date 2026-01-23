{-# LANGUAGE OverloadedStrings #-}
module Trader.Http
  ( RetryConfig(..)
  , defaultRetryConfig
  , newHttpManager
  , getSharedManager
  , httpLbsWithRetry
  , defaultTimeoutMicros
  ) where

import Control.Concurrent (MVar, modifyMVar, newMVar, readMVar, threadDelay)
import Control.Exception (SomeException, displayException, throwIO, try)
import Data.ByteString.Char8 (ByteString)
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (toLower)
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Data.List (isInfixOf)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import Network.HTTP.Client
import Network.HTTP.Client.TLS (tlsManagerSettings)
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

defaultRetryConfig :: RetryConfig
defaultRetryConfig =
  RetryConfig
    { rcMaxRetries = 3
    , rcBaseDelayMs = 250
    , rcMaxDelayMs = 8000
    , rcJitterFrac = 0.2
    , rcRetryWrites = False
    }

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

{-# NOINLINE sharedManager #-}
sharedManager :: MVar (Maybe Manager)
sharedManager = unsafePerformIO (newMVar Nothing)

getSharedManager :: IO Manager
getSharedManager = do
  cached <- readMVar sharedManager
  case cached of
    Just mgr -> pure mgr
    Nothing ->
      modifyMVar sharedManager $ \_ -> do
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
          case shouldRetryException cfg req attempt of
            False -> do
              logHttpAttempt labelTxt methodBs hostBs pathBs (Left ex) latencyMs attempt False
              throwIO ex
            True -> do
              delayMs <- computeDelay cfg attempt Nothing
              logHttpAttempt labelTxt methodBs hostBs pathBs (Left ex) latencyMs attempt True
              sleepMs delayMs
              go (attempt + 1)
        Right resp -> do
          let code = statusCode (responseStatus resp)
              retryable = shouldRetryStatus cfg req code attempt
          case retryable of
            False -> do
              logHttpAttempt labelTxt methodBs hostBs pathBs (Right resp) latencyMs attempt False
              pure resp
            True -> do
              delayMs <- computeDelay cfg attempt (retryAfterMs resp)
              logHttpAttempt labelTxt methodBs hostBs pathBs (Right resp) latencyMs attempt True
              sleepMs delayMs
              go (attempt + 1)

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
  let base = rcBaseDelayMs cfg * (2 ^ attempt)
      capped = min (rcMaxDelayMs cfg) base
  jittered <- applyJitter capped (rcJitterFrac cfg)
  pure $
    case mRetryAfter of
      Nothing -> jittered
      Just ra -> max jittered ra

applyJitter :: Int -> Double -> IO Int
applyJitter delayMs jitterFrac =
  if delayMs <= 0 || jitterFrac <= 0
    then pure delayMs
    else do
      skew <- randomRIO (-jitterFrac, jitterFrac)
      let scaled = fromIntegral delayMs * (1 + skew)
      pure (max 0 (round scaled))

retryAfterMs :: Response BL.ByteString -> Maybe Int
retryAfterMs resp =
  case lookup "Retry-After" (responseHeaders resp) of
    Nothing -> Nothing
    Just v ->
      case readMaybe (BS.unpack v) :: Maybe Int of
        Just sec | sec > 0 -> Just (sec * 1000)
        _ -> Nothing

sleepMs :: Int -> IO ()
sleepMs ms =
  if ms <= 0
    then pure ()
    else threadDelay (ms * 1000)

getTimeMs :: IO Integer
getTimeMs = do
  t <- getPOSIXTime
  pure (floor (t * 1000))

{-# NOINLINE rateLimiter #-}
rateLimiter :: MVar (Map ByteString Integer)
rateLimiter = unsafePerformIO (newMVar Map.empty)

applyRateLimit :: ByteString -> IO ()
applyRateLimit host = do
  let delayMs = rateLimitMsForHost host
  if delayMs <= 0
    then pure ()
    else do
      now <- getTimeMs
      waitMs <- modifyMVar rateLimiter $ \m -> do
        let lastMs = Map.lookup host m
            gapMs =
              case lastMs of
                Nothing -> 0
                Just prev ->
                  let diffMs = fromIntegral (now - prev)
                   in max 0 (delayMs - diffMs)
            reserveUntil = now + fromIntegral gapMs
        pure (Map.insert host reserveUntil m, gapMs)
      sleepMs (fromIntegral waitMs)

rateLimitMsForHost :: ByteString -> Int
rateLimitMsForHost host =
  let h = map toLower (BS.unpack host)
   in if "binance" `isInfixOf` h
        then 80
        else if "coinbase" `isInfixOf` h
          then 150
          else if "kraken" `isInfixOf` h
            then 200
            else if "poloniex" `isInfixOf` h
              then 200
              else if ".s3." `isInfixOf` h
                then 50
                else 0

{-# NOINLINE httpLogFlag #-}
httpLogFlag :: IORef (Maybe Bool)
httpLogFlag = unsafePerformIO (newIORef Nothing)

logHttpAttempt
  :: String
  -> ByteString
  -> ByteString
  -> ByteString
  -> Either SomeException (Response BL.ByteString)
  -> Int
  -> Int
  -> Bool
  -> IO ()
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
              ++ " label=" ++ label
              ++ " method=" ++ methodTxt
              ++ " host=" ++ hostTxt
              ++ " path=" ++ pathTxt
              ++ " status=" ++ statusTxt
              ++ " latency_ms=" ++ show latencyMs
              ++ " attempt=" ++ show (attempt + 1)
              ++ retryTxt
              ++ errTxt
      hPutStrLn stderr msg

isHttpLogEnabled :: IO Bool
isHttpLogEnabled = do
  cached <- readIORef httpLogFlag
  case cached of
    Just v -> pure v
    Nothing -> do
      env <- lookupEnv "TRADER_HTTP_LOG"
      let v = maybe False isTruthy env
      writeIORef httpLogFlag (Just v)
      pure v

isTruthy :: String -> Bool
isTruthy raw =
  case map toLower raw of
    "1" -> True
    "true" -> True
    "yes" -> True
    "on" -> True
    _ -> False

sanitize :: String -> String
sanitize =
  map (\c -> if c == '\n' || c == '\r' then ' ' else c)

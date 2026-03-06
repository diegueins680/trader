{-# LANGUAGE OverloadedStrings #-}

module Trader.App.Observability (
    Journal,
    Metrics,
    Webhook,
    WebhookEvent (..),
    incCounter,
    journalWrite,
    journalWriteMaybe,
    metricsIncEndpoint,
    metricsRecordBotHalt,
    metricsRecordOrderFailed,
    metricsRecordOrderSent,
    newJournalFromEnv,
    newMetrics,
    newWebhookFromEnv,
    renderMetricsText,
    sanitizeWebhookText,
    stateDirFromEnv,
    stateSubdirFromEnv,
    webhookMaxMessageLen,
    webhookNotifyMaybe,
    webhookNotifyMaybeSync,
) where

import Control.Concurrent (forkIO)
import Control.Concurrent.MVar (MVar, newMVar, withMVar)
import Control.Exception (SomeException, try)
import Control.Monad (when)
import Data.Aeson (encode, object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.HashMap.Strict as HM
import Data.IORef (IORef, atomicModifyIORef', newIORef, readIORef)
import Data.Int (Int64)
import Data.List (isPrefixOf)
import Data.Maybe (fromMaybe)
import Network.HTTP.Client (
    Manager,
    Request,
    RequestBody (..),
    Response,
    httpLbs,
    method,
    newManager,
    parseRequest,
    requestBody,
    requestHeaders,
    responseTimeout,
    responseTimeoutMicro,
 )
import Network.HTTP.Client.TLS (tlsManagerSettings)
import System.Directory (createDirectoryIfMissing)
import System.Environment (lookupEnv)
import System.FilePath ((</>))

import Trader.App.Runtime (splitEnvList)
import Trader.Binance (getTimestampMs)
import Trader.Text (trim)

data Metrics = Metrics
    { mtRequestsTotal :: !(IORef Int64)
    , mtRequestsByEndpoint :: !(IORef (HM.HashMap String Int64))
    , mtOrdersSentTotal :: !(IORef Int64)
    , mtOrdersFailedTotal :: !(IORef Int64)
    , mtBotHaltsTotal :: !(IORef Int64)
    }

newMetrics :: IO Metrics
newMetrics = do
    reqTotal <- newIORef 0
    reqBy <- newIORef HM.empty
    ordersSent <- newIORef 0
    ordersFailed <- newIORef 0
    botHalts <- newIORef 0
    pure
        Metrics
            { mtRequestsTotal = reqTotal
            , mtRequestsByEndpoint = reqBy
            , mtOrdersSentTotal = ordersSent
            , mtOrdersFailedTotal = ordersFailed
            , mtBotHaltsTotal = botHalts
            }

incCounter :: IORef Int64 -> IO ()
incCounter ref = atomicModifyIORef' ref (\n -> (n + 1, ()))

metricsIncEndpoint :: Metrics -> String -> IO ()
metricsIncEndpoint m endpoint = do
    incCounter (mtRequestsTotal m)
    atomicModifyIORef' (mtRequestsByEndpoint m) $ \hm ->
        let next = fromMaybe 0 (HM.lookup endpoint hm) + 1
         in (HM.insert endpoint next hm, ())

metricsRecordOrderSent :: Metrics -> IO ()
metricsRecordOrderSent m = incCounter (mtOrdersSentTotal m)

metricsRecordOrderFailed :: Metrics -> IO ()
metricsRecordOrderFailed m = incCounter (mtOrdersFailedTotal m)

metricsRecordBotHalt :: Metrics -> IO ()
metricsRecordBotHalt m = incCounter (mtBotHaltsTotal m)

escapePromLabel :: String -> String
escapePromLabel = concatMap esc
  where
    esc '"' = "\\\""
    esc '\\' = "\\\\"
    esc c = [c]

renderMetricsText :: Metrics -> Bool -> IO BL.ByteString
renderMetricsText m botRunning = do
    reqTotal <- readIORef (mtRequestsTotal m)
    reqBy <- readIORef (mtRequestsByEndpoint m)
    ordersSent <- readIORef (mtOrdersSentTotal m)
    ordersFailed <- readIORef (mtOrdersFailedTotal m)
    botHalts <- readIORef (mtBotHaltsTotal m)
    let header =
            [ "# HELP trader_http_requests_total Total HTTP requests."
            , "# TYPE trader_http_requests_total counter"
            , "trader_http_requests_total " ++ show reqTotal
            , "# HELP trader_http_requests_by_endpoint_total Total HTTP requests by endpoint."
            , "# TYPE trader_http_requests_by_endpoint_total counter"
            ]
        byEndpoint =
            [ "trader_http_requests_by_endpoint_total{endpoint=\"" ++ escapePromLabel k ++ "\"} " ++ show v
            | (k, v) <- HM.toList reqBy
            ]
        orders =
            [ "# HELP trader_orders_total Total orders (sent/failed)."
            , "# TYPE trader_orders_total counter"
            , "trader_orders_total{result=\"sent\"} " ++ show ordersSent
            , "trader_orders_total{result=\"failed\"} " ++ show ordersFailed
            ]
        bot =
            [ "# HELP trader_bot_halts_total Bot halts."
            , "# TYPE trader_bot_halts_total counter"
            , "trader_bot_halts_total " ++ show botHalts
            , "# HELP trader_bot_running Bot running."
            , "# TYPE trader_bot_running gauge"
            , "trader_bot_running " ++ if botRunning then "1" else "0"
            ]
        txt = unlines (header ++ byEndpoint ++ orders ++ bot)
    pure (BL.fromStrict (BS.pack txt))

data Journal = Journal
    { jPath :: !FilePath
    , jLock :: !(MVar ())
    }

stateDirFromEnv :: IO (Maybe FilePath)
stateDirFromEnv = do
    mDir <- lookupEnv "TRADER_STATE_DIR"
    case trim <$> mDir of
        Just dir | not (null dir) -> pure (Just dir)
        _ -> pure Nothing

stateSubdirFromEnv :: String -> IO (Maybe FilePath)
stateSubdirFromEnv subdir = do
    fmap (</> subdir) <$> stateDirFromEnv

newJournalFromEnv :: IO (Maybe Journal)
newJournalFromEnv = do
    mDir <- lookupEnv "TRADER_JOURNAL_DIR"
    mStateDir <- stateSubdirFromEnv "journal"
    let resolvedDir =
            case trim <$> mDir of
                Just dir | null dir -> Nothing
                Just dir -> Just dir
                Nothing -> mStateDir
    case resolvedDir of
        Nothing -> pure Nothing
        Just dir -> do
            createDirectoryIfMissing True dir
            ts <- getTimestampMs
            lock <- newMVar ()
            pure (Just (Journal (dir </> ("trader-" ++ show ts ++ ".jsonl")) lock))

journalWrite :: Journal -> Aeson.Value -> IO ()
journalWrite j v =
    withMVar (jLock j) $ \_ ->
        BL.appendFile (jPath j) (encode v <> BL.fromStrict (BS.pack "\n"))

journalWriteMaybe :: Maybe Journal -> Aeson.Value -> IO ()
journalWriteMaybe mj v =
    case mj of
        Nothing -> pure ()
        Just j -> journalWrite j v

data Webhook = Webhook
    { whRequest :: !Request
    , whManager :: !Manager
    , whEvents :: !(Maybe [String])
    }

data WebhookEvent = WebhookEvent
    { weType :: !String
    , weContent :: !String
    }
    deriving (Eq, Show)

webhookMaxMessageLen :: Int
webhookMaxMessageLen = 300

sanitizeWebhookText :: Int -> String -> String
sanitizeWebhookText maxLen raw =
    let cleaned = map (\c -> if c == '\n' || c == '\r' then ' ' else c) raw
        trimmed = trim cleaned
        limit = max 0 maxLen
     in if length trimmed <= limit || limit < 4
            then trimmed
            else take (limit - 3) trimmed ++ "..."

normalizeWebhookEvent :: String -> String
normalizeWebhookEvent = map toLowerAscii . trim
  where
    toLowerAscii c
        | 'A' <= c && c <= 'Z' = toEnum (fromEnum c + 32)
        | otherwise = c

parseWebhookEvents :: Maybe String -> Maybe [String]
parseWebhookEvents raw =
    case raw of
        Nothing -> Nothing
        Just s ->
            let events = filter (not . null) (map normalizeWebhookEvent (splitEnvList s))
             in Just events

newWebhookFromEnv :: IO (Maybe Webhook)
newWebhookFromEnv = do
    mUrl <- lookupEnv "TRADER_WEBHOOK_URL"
    case trim <$> mUrl of
        Just url | not (null url) -> do
            eventsEnv <- lookupEnv "TRADER_WEBHOOK_EVENTS"
            req0 <- parseRequest url
            manager <- newManager tlsManagerSettings
            let req =
                    req0
                        { method = "POST"
                        , requestHeaders = ("Content-Type", "application/json") : requestHeaders req0
                        , responseTimeout = responseTimeoutMicro (5 * 1000000)
                        }
            pure (Just (Webhook req manager (parseWebhookEvents eventsEnv)))
        _ -> pure Nothing

webhookAllows :: Webhook -> String -> Bool
webhookAllows wh ev =
    case whEvents wh of
        Nothing -> True
        Just allowed -> normalizeWebhookEvent ev `elem` allowed

webhookNotifyMaybe :: Maybe Webhook -> WebhookEvent -> IO ()
webhookNotifyMaybe mWh ev =
    case mWh of
        Nothing -> pure ()
        Just wh ->
            when (webhookAllows wh (weType ev)) $ do
                _ <- forkIO (webhookSend wh ev)
                pure ()

webhookNotifyMaybeSync :: Maybe Webhook -> WebhookEvent -> IO ()
webhookNotifyMaybeSync mWh ev =
    case mWh of
        Nothing -> pure ()
        Just wh ->
            when (webhookAllows wh (weType ev)) $ webhookSend wh ev

webhookSend :: Webhook -> WebhookEvent -> IO ()
webhookSend wh ev = do
    let payload = object ["content" .= weContent ev]
        req = (whRequest wh){requestBody = RequestBodyLBS (encode payload)}
    _ <- try (httpLbs req (whManager wh)) :: IO (Either SomeException (Response BL.ByteString))
    pure ()

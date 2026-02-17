{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Concurrent (threadDelay)
import Control.Exception (SomeException, try)
import Control.Monad (forM_, forever)
import Data.Aeson (Value (..), decode, encode, object, (.=))
import qualified Data.ByteString.Lazy as BL
import Data.Char (toLower)
import Data.Int (Int64)
import Data.Maybe (fromMaybe)
import Data.String (fromString)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Time.Clock.POSIX (getPOSIXTime)
import Database.PostgreSQL.Simple (Connection, connectPostgreSQL, execute, query)
import Database.PostgreSQL.Simple.FromRow (FromRow (..), field)
import Network.HTTP.Client (Manager, RequestBody (..), httpLbs, method, newManager, parseRequest, requestBody, requestHeaders, responseStatus)
import Network.HTTP.Client.TLS (tlsManagerSettings)
import Network.HTTP.Types (hContentType, statusCode)
import System.Environment (lookupEnv)
import System.Exit (die)
import Text.Read (readMaybe)

data PublishMode
    = PublishNoop
    | PublishStdout
    | PublishKafkaRest
    deriving (Eq, Show)

data OutboxEvent = OutboxEvent
    { oeId :: !Int64
    , oeTenantKey :: !(Maybe Text)
    , oeTopic :: !Text
    , oeEventKey :: !(Maybe Text)
    , oePayloadJson :: !Text
    , oeAttempts :: !Int
    }
    deriving (Eq, Show)

data PublisherCtx = PublisherCtx
    { pcMode :: !PublishMode
    , pcKafkaRestBaseUrl :: !(Maybe String)
    , pcManager :: !Manager
    }

instance FromRow OutboxEvent where
    fromRow =
        OutboxEvent
            <$> field
            <*> field
            <*> field
            <*> field
            <*> field
            <*> field

trim :: String -> String
trim = f . f
  where
    f = reverse . dropWhile (`elem` [' ', '\t', '\n', '\r'])

parseIntEnv :: String -> Int -> IO Int
parseIntEnv key def = do
    mRaw <- lookupEnv key
    case mRaw of
        Nothing -> pure def
        Just raw ->
            case readMaybe (trim raw) of
                Just v | v > 0 -> pure v
                _ -> pure def

parseInt64Env :: String -> Int64 -> IO Int64
parseInt64Env key def = do
    mRaw <- lookupEnv key
    case mRaw of
        Nothing -> pure def
        Just raw ->
            case readMaybe (trim raw) of
                Just v | v > 0 -> pure v
                _ -> pure def

parseInt64EnvAllowZero :: String -> Int64 -> IO Int64
parseInt64EnvAllowZero key def = do
    mRaw <- lookupEnv key
    case mRaw of
        Nothing -> pure def
        Just raw ->
            case readMaybe (trim raw) of
                Just v | v >= 0 -> pure v
                _ -> pure def

parseTextEnv :: String -> IO (Maybe String)
parseTextEnv key = do
    mRaw <- lookupEnv key
    pure $
        case trim <$> mRaw of
            Just v | not (null v) -> Just v
            _ -> Nothing

resolveDbUrl :: IO String
resolveDbUrl = do
    mPrimary <- lookupEnv "TRADER_DB_URL"
    mFallback <- lookupEnv "DATABASE_URL"
    let pick =
            case filter (not . null) (map (trim) (concat [[maybe "" id mPrimary], [maybe "" id mFallback]])) of
                (x : _) -> Just x
                [] -> Nothing
    case pick of
        Nothing -> die "TRADER_DB_URL or DATABASE_URL not set."
        Just url -> pure url

resolveMode :: IO PublishMode
resolveMode = do
    mRaw <- lookupEnv "TRADER_OUTBOX_PUBLISHER_MODE"
    let mode = map toLower (maybe "noop" trim mRaw)
    pure $
        case mode of
            "stdout" -> PublishStdout
            "kafka-rest" -> PublishKafkaRest
            _ -> PublishNoop

getTimestampMs :: IO Int64
getTimestampMs = round . (* 1000) <$> getPOSIXTime

claimOutboxBatch :: Connection -> Int64 -> Int -> IO [OutboxEvent]
claimOutboxBatch conn now limitN =
    query
        conn
        (fromString sql)
        (now, limitN, now)
  where
    sql =
        "WITH cte AS ("
            <> "  SELECT id "
            <> "  FROM outbox_events "
            <> "  WHERE status = 'pending' AND (next_attempt_at_ms IS NULL OR next_attempt_at_ms <= ?) "
            <> "  ORDER BY id ASC "
            <> "  LIMIT ? "
            <> "  FOR UPDATE SKIP LOCKED"
            <> ") "
            <> "UPDATE outbox_events o "
            <> "SET status = 'publishing', attempts = o.attempts + 1, updated_at_ms = ? "
            <> "FROM cte "
            <> "WHERE o.id = cte.id "
            <> "RETURNING o.id, o.tenant_key, o.topic, o.event_key, o.payload_json::text, o.attempts"

markOutboxPublished :: Connection -> Int64 -> Int64 -> IO ()
markOutboxPublished conn now eventId = do
    _ <-
        execute
            conn
            "UPDATE outbox_events SET status = 'published', published_at_ms = ?, updated_at_ms = ?, last_error = NULL WHERE id = ?"
            (now, now, eventId)
    pure ()

markOutboxRetry :: Connection -> Int64 -> Int64 -> Int64 -> Text -> IO ()
markOutboxRetry conn now eventId nextAttempt errMsg = do
    _ <-
        execute
            conn
            "UPDATE outbox_events SET status = 'pending', next_attempt_at_ms = ?, updated_at_ms = ?, last_error = ? WHERE id = ?"
            (nextAttempt, now, errMsg, eventId)
    pure ()

reclaimStalePublishing :: Connection -> Int64 -> Int64 -> IO Int64
reclaimStalePublishing conn now timeoutMs = do
    let cutoff = now - max 1000 timeoutMs
    execute
        conn
        ( "UPDATE outbox_events "
            <> "SET status = 'pending', next_attempt_at_ms = ?, updated_at_ms = ?, last_error = COALESCE(last_error, 'stale publishing lease reclaimed') "
            <> "WHERE status = 'publishing' AND published_at_ms IS NULL AND updated_at_ms < ?"
        )
        (now, now, cutoff)

cleanupPublishedOlderThan :: Connection -> Int64 -> Int64 -> IO Int64
cleanupPublishedOlderThan conn now retentionMs = do
    let cutoff = now - max 0 retentionMs
    execute
        conn
        "DELETE FROM outbox_events WHERE status = 'published' AND published_at_ms IS NOT NULL AND published_at_ms < ?"
        (Only cutoff)

backoffMs :: Int -> Int64
backoffMs attempts =
    let a = max 1 attempts
        raw = (2 ^ min 10 a) * 1000
     in min 300000 (fromIntegral raw)

publishEvent :: PublisherCtx -> OutboxEvent -> IO (Either Text ())
publishEvent ctx event =
    case pcMode ctx of
        PublishNoop -> pure (Left "Publisher mode is noop; set TRADER_OUTBOX_PUBLISHER_MODE=stdout|kafka-rest to enable publishing.")
        PublishStdout -> do
            putStrLn
                ( "outbox.publish"
                    <> " id="
                    <> show (oeId event)
                    <> " topic="
                    <> T.unpack (oeTopic event)
                    <> " key="
                    <> T.unpack (maybe "" id (oeEventKey event))
                )
            pure (Right ())
        PublishKafkaRest ->
            case pcKafkaRestBaseUrl ctx of
                Nothing -> pure (Left "TRADER_OUTBOX_KAFKA_REST_URL is required for kafka-rest mode.")
                Just baseUrl -> do
                    req0 <- parseRequest (baseUrl ++ "/topics/" ++ T.unpack (oeTopic event))
                    let payloadValue =
                            fromMaybe
                                (String (oePayloadJson event))
                                (decode (BL.fromStrict (TE.encodeUtf8 (oePayloadJson event))))
                        bodyValue =
                            object
                                [ "records"
                                    .= [ object
                                            [ "key" .= oeEventKey event
                                            , "value" .= payloadValue
                                            ]
                                       ]
                                ]
                        req =
                            req0
                                { method = "POST"
                                , requestHeaders = [(hContentType, "application/vnd.kafka.json.v2+json")]
                                , requestBody = RequestBodyLBS (encode bodyValue)
                                }
                    resp <- httpLbs req (pcManager ctx)
                    if statusCode (responseStatus resp) >= 200 && statusCode (responseStatus resp) < 300
                        then pure (Right ())
                        else pure (Left ("Kafka REST publish failed with HTTP " <> T.pack (show (statusCode (responseStatus resp)))))

runBatch :: Connection -> PublisherCtx -> Int -> Int64 -> Int64 -> IO ()
runBatch conn ctx batchSize staleTimeoutMs publishedRetentionMs = do
    now0 <- getTimestampMs
    if publishedRetentionMs > 0
        then do
            deleted <- cleanupPublishedOlderThan conn now0 publishedRetentionMs
            if deleted > 0
                then putStrLn ("outbox.cleanup deleted=" <> show deleted)
                else pure ()
        else pure ()
    case pcMode ctx of
        PublishNoop -> pure ()
        _ -> do
            now <- getTimestampMs
            reclaimed <- reclaimStalePublishing conn now staleTimeoutMs
            if reclaimed > 0
                then putStrLn ("outbox.reclaim count=" <> show reclaimed)
                else pure ()
            events <- claimOutboxBatch conn now batchSize
            forM_ events $ \event -> do
                result <- (try (publishEvent ctx event) :: IO (Either SomeException (Either Text ())))
                doneAt <- getTimestampMs
                case result of
                    Left ex -> do
                        let msg = T.pack (take 800 (show ex))
                            nextAt = doneAt + backoffMs (oeAttempts event)
                        markOutboxRetry conn doneAt (oeId event) nextAt msg
                    Right (Left err) -> do
                        let msg = T.take 800 err
                            nextAt = doneAt + backoffMs (oeAttempts event)
                        markOutboxRetry conn doneAt (oeId event) nextAt msg
                    Right (Right ()) -> markOutboxPublished conn doneAt (oeId event)

main :: IO ()
main = do
    dbUrl <- resolveDbUrl
    pollMs <- parseIntEnv "TRADER_OUTBOX_POLL_MS" 1000
    batchSize <- parseIntEnv "TRADER_OUTBOX_BATCH_SIZE" 100
    staleTimeoutMs <- parseInt64Env "TRADER_OUTBOX_PUBLISHING_TIMEOUT_MS" 60000
    publishedRetentionMs <- parseInt64EnvAllowZero "TRADER_OUTBOX_PUBLISHED_RETENTION_MS" 604800000
    mode <- resolveMode
    kafkaRestBaseUrl <- parseTextEnv "TRADER_OUTBOX_KAFKA_REST_URL"
    manager <- newManager tlsManagerSettings
    let ctx = PublisherCtx{pcMode = mode, pcKafkaRestBaseUrl = kafkaRestBaseUrl, pcManager = manager}
    conn <- connectPostgreSQL (TE.encodeUtf8 (T.pack dbUrl))
    putStrLn
        ( "outbox-publisher started"
            <> " mode="
            <> show mode
            <> " pollMs="
            <> show pollMs
            <> " batchSize="
            <> show batchSize
            <> " publishingTimeoutMs="
            <> show staleTimeoutMs
            <> " publishedRetentionMs="
            <> show publishedRetentionMs
            <> " kafkaRestUrlConfigured="
            <> show (maybe False (const True) kafkaRestBaseUrl)
        )
    forever $ do
        runBatch conn ctx batchSize staleTimeoutMs publishedRetentionMs
        threadDelay (pollMs * 1000)

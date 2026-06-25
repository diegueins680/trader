{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Concurrent (threadDelay)
import Control.Exception (SomeException, try)
import Control.Monad (forM_, forever, when)
import Data.Aeson (Value (..), decode, encode, object, (.=))
import qualified Data.ByteString.Lazy as BL
import Data.Char (toLower)
import Data.Int (Int64)
import Data.Maybe (fromMaybe, isJust)
import Data.String (fromString)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Time.Clock.POSIX (getPOSIXTime)
import Database.PostgreSQL.Simple (Connection, Only (..), connectPostgreSQL, execute, query)
import Database.PostgreSQL.Simple.FromRow (FromRow (..), field)
import Network.HTTP.Client (Manager, Request (..), RequestBody (..), httpLbs, method, newManager, parseRequest, requestBody, requestHeaders, responseStatus, responseTimeoutMicro)
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
    , pcKafkaRestTimeoutMicros :: !Int
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
parseIntEnv key def =
    parseNumericEnv key def (> 0) "integer > 0"

parseInt64Env :: String -> Int64 -> IO Int64
parseInt64Env key def =
    parseNumericEnv key def (> 0) "integer > 0"

parseInt64EnvAllowZero :: String -> Int64 -> IO Int64
parseInt64EnvAllowZero key def =
    parseNumericEnv key def (>= 0) "integer >= 0"

parseSecondsEnvMicros :: String -> Int -> IO Int
parseSecondsEnvMicros key defSec = do
    seconds <- parseNumericEnv key defSec (>= 1) "integer >= 1"
    let micros = toInteger seconds * 1000000
        cap = toInteger (maxBound :: Int)
    pure (fromInteger (min cap micros))

parseNumericEnv :: (Read a, Show a) => String -> a -> (a -> Bool) -> String -> IO a
parseNumericEnv key def predicate expected = do
    mRaw <- lookupEnv key
    case mRaw of
        Nothing -> pure def
        Just raw ->
            case readMaybe (trim raw) of
                Just v | predicate v -> pure v
                _ ->
                    die
                        ( "Invalid "
                            ++ key
                            ++ "="
                            ++ show raw
                            ++ " (expected "
                            ++ expected
                            ++ ")."
                        )

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
            case filter (not . null) (map trim [fromMaybe "" mPrimary, fromMaybe "" mFallback]) of
                (x : _) -> Just x
                [] -> Nothing
    case pick of
        Nothing -> die "TRADER_DB_URL or DATABASE_URL not set."
        Just url -> pure url

resolveMode :: IO PublishMode
resolveMode = do
    mRaw <- lookupEnv "TRADER_OUTBOX_PUBLISHER_MODE"
    let mode = map toLower (maybe "noop" trim mRaw)
    case mode of
        "noop" -> pure PublishNoop
        "stdout" -> pure PublishStdout
        "kafka-rest" -> pure PublishKafkaRest
        _ ->
            die
                ( "Invalid TRADER_OUTBOX_PUBLISHER_MODE="
                    ++ show mode
                    ++ " (expected noop|stdout|kafka-rest)."
                )

validatePublisherConfig :: PublishMode -> Maybe String -> IO ()
validatePublisherConfig mode kafkaRestBaseUrl =
    case mode of
        PublishKafkaRest ->
            case kafkaRestBaseUrl of
                Nothing -> die "TRADER_OUTBOX_KAFKA_REST_URL is required when TRADER_OUTBOX_PUBLISHER_MODE=kafka-rest."
                Just baseUrl -> do
                    let probeUrl = trim baseUrl ++ "/topics/trader-healthcheck"
                    parsed <- try (parseRequest probeUrl) :: IO (Either SomeException Request)
                    case parsed of
                        Left _ ->
                            die
                                ( "Invalid TRADER_OUTBOX_KAFKA_REST_URL="
                                    ++ show baseUrl
                                    ++ "; expected an absolute http(s) URL."
                                )
                        Right _ -> pure ()
        _ -> pure ()

getTimestampMs :: IO Int64
getTimestampMs = round . (* 1000) <$> getPOSIXTime

safeDelayMicrosFromMs :: Int -> Int
safeDelayMicrosFromMs ms =
    let boundedMs = max 0 ms
        micros = toInteger boundedMs * 1000
        cap = toInteger (maxBound :: Int)
     in fromInteger (min cap micros)

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
    let expSteps = min 10 (max 1 attempts)
        rawMs :: Int64
        rawMs = (2 ^ expSteps) * 1000
     in min 300000 rawMs

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
                    <> T.unpack (fromMaybe "" (oeEventKey event))
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
                                , responseTimeout = responseTimeoutMicro (pcKafkaRestTimeoutMicros ctx)
                                }
                    resp <- httpLbs req (pcManager ctx)
                    if statusCode (responseStatus resp) >= 200 && statusCode (responseStatus resp) < 300
                        then pure (Right ())
                        else pure (Left ("Kafka REST publish failed with HTTP " <> T.pack (show (statusCode (responseStatus resp)))))

runBatch :: Connection -> PublisherCtx -> Int -> Int64 -> Int64 -> IO ()
runBatch conn ctx batchSize staleTimeoutMs publishedRetentionMs = do
    now0 <- getTimestampMs
    when (publishedRetentionMs > 0) $ do
        deleted <- cleanupPublishedOlderThan conn now0 publishedRetentionMs
        when (deleted > 0) $
            putStrLn ("outbox.cleanup deleted=" <> show deleted)
    case pcMode ctx of
        PublishNoop -> pure ()
        _ -> do
            now <- getTimestampMs
            reclaimed <- reclaimStalePublishing conn now staleTimeoutMs
            when (reclaimed > 0) $
                putStrLn ("outbox.reclaim count=" <> show reclaimed)
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
    kafkaRestTimeoutMicros <- parseSecondsEnvMicros "TRADER_OUTBOX_KAFKA_REST_TIMEOUT_SEC" 15
    mode <- resolveMode
    kafkaRestBaseUrl <- parseTextEnv "TRADER_OUTBOX_KAFKA_REST_URL"
    validatePublisherConfig mode kafkaRestBaseUrl
    manager <- newManager tlsManagerSettings
    let ctx = PublisherCtx{pcMode = mode, pcKafkaRestBaseUrl = kafkaRestBaseUrl, pcManager = manager, pcKafkaRestTimeoutMicros = kafkaRestTimeoutMicros}
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
            <> " kafkaRestTimeoutSec="
            <> show (kafkaRestTimeoutMicros `div` 1000000)
            <> " publishedRetentionMs="
            <> show publishedRetentionMs
            <> " kafkaRestUrlConfigured="
            <> show (isJust kafkaRestBaseUrl)
        )
    forever $ do
        runBatch conn ctx batchSize staleTimeoutMs publishedRetentionMs
        threadDelay (safeDelayMicrosFromMs pollMs)

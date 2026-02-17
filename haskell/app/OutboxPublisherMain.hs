{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Concurrent (threadDelay)
import Control.Exception (SomeException, try)
import Control.Monad (forM_, forever)
import Data.Char (toLower)
import Data.Int (Int64)
import Data.String (fromString)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Database.PostgreSQL.Simple (Connection, connectPostgreSQL, execute, query)
import Database.PostgreSQL.Simple.FromRow (FromRow (..), field)
import System.Environment (lookupEnv)
import System.Exit (die)
import Text.Read (readMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)

data PublishMode
    = PublishNoop
    | PublishStdout
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
    pure $ if mode == "stdout" then PublishStdout else PublishNoop

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

backoffMs :: Int -> Int64
backoffMs attempts =
    let a = max 1 attempts
        raw = (2 ^ min 10 a) * 1000
     in min 300000 (fromIntegral raw)

publishEvent :: PublishMode -> OutboxEvent -> IO (Either Text ())
publishEvent mode event =
    case mode of
        PublishNoop -> pure (Left "Publisher mode is noop; set TRADER_OUTBOX_PUBLISHER_MODE=stdout to enable publishing.")
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

runBatch :: Connection -> PublishMode -> Int -> IO ()
runBatch conn mode batchSize = do
    case mode of
        PublishNoop -> pure ()
        PublishStdout -> do
            now <- getTimestampMs
            events <- claimOutboxBatch conn now batchSize
            forM_ events $ \event -> do
                result <- (try (publishEvent mode event) :: IO (Either SomeException (Either Text ())))
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
    mode <- resolveMode
    conn <- connectPostgreSQL (TE.encodeUtf8 (T.pack dbUrl))
    putStrLn
        ( "outbox-publisher started"
            <> " mode="
            <> show mode
            <> " pollMs="
            <> show pollMs
            <> " batchSize="
            <> show batchSize
        )
    forever $ do
        runBatch conn mode batchSize
        threadDelay (pollMs * 1000)

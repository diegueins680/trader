{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.OpsMigrations (testOpsMigrations) where

import Control.Exception (bracket, bracket_, try)
import Control.Monad (forM_, unless, void)
import qualified Data.ByteString.Char8 as BS
import Data.Either (isLeft)
import Data.Int (Int64)
import Data.String (fromString)
import qualified Data.Text as T
import qualified Data.UUID as UUID
import qualified Data.UUID.V4 as UUID
import Database.PostgreSQL.Simple (Connection, Only (..), SqlError, begin, close, connectPostgreSQL, execute, execute_, query_, rollback, withTransaction)
import Database.PostgreSQL.Simple.Types (Query)
import System.Environment (lookupEnv)
import Trader.Ops.Migrations (ensureOpsDbSchema, topCombosDbFingerprintQuery)

{- | Optional locally; CI supplies a disposable PostgreSQL 16 service. Every
object and mutation stays inside a fresh, random schema which is removed.
-}
testOpsMigrations :: IO ()
testOpsMigrations = do
    mUrl <- lookupEnv "TRADER_TEST_DB_URL"
    case mUrl of
        Nothing -> putStrLn "Ops migration integration skipped: TRADER_TEST_DB_URL is unset."
        Just url ->
            bracket (connectPostgreSQL (BS.pack url)) close $ \conn -> do
                void (execute_ conn "SET client_min_messages=warning")
                uuid <- UUID.nextRandom
                -- Only UUID hex digits and this fixed prefix enter SQL identifiers.
                let schema = "trader_digest_test_" ++ filter (/= '-') (UUID.toString uuid)
                bracket_
                    (void (execute_ conn (fromString ("CREATE SCHEMA " ++ schema))))
                    (void (execute_ conn (fromString ("DROP SCHEMA " ++ schema ++ " CASCADE"))))
                    (setSchema conn schema >> checkMigrations (BS.pack url) schema conn)

setSchema :: Connection -> String -> IO ()
setSchema conn schema = void (execute_ conn (fromString ("SET search_path TO " ++ schema)))

checkMigrations :: BS.ByteString -> String -> Connection -> IO ()
checkMigrations url schema conn = do
    ensureOpsDbSchema conn
    generated <- query_ conn "SELECT count(*) FROM pg_attribute WHERE attrelid='combos'::regclass AND attname IN ('params_json_md5','metrics_json_md5') AND attgenerated='s'" :: IO [Only Int]
    check "fresh schema creates two stored generated digests" (generated == [Only 2])
    -- Reproduce the pre-v6 schema with existing nullable payload rows, then
    -- exercise the real upgrade rather than a separately copied ALTER TABLE.
    void (execute_ conn "ALTER TABLE combos DROP COLUMN params_json_md5, DROP COLUMN metrics_json_md5")
    void (execute_ conn "DELETE FROM ops_schema_migrations WHERE version=6")
    forM_ fixtures $ \row -> void (execute conn insertFixture row)
    payloadsBefore <- payloads conn
    oldBefore <- query_ conn legacyFingerprintQuery :: IO [(Int64, String)]
    ensureOpsDbSchema conn
    ensureOpsDbSchema conn
    payloadsAfter <- payloads conn
    versions <- query_ conn "SELECT count(*) FROM ops_schema_migrations WHERE version=6" :: IO [Only Int]
    check "migration is idempotent and preserves existing JSON" (payloadsBefore == payloadsAfter && versions == [Only 1])
    initial <- equivalent conn
    check "backfilled fingerprint equals the exact pre-migration fingerprint" (initial == oldBefore)
    mismatches <- query_ conn "SELECT count(*) FROM combos WHERE params_json_md5 IS DISTINCT FROM md5(params_json::text) OR metrics_json_md5 IS DISTINCT FROM md5(metrics_json::text)" :: IO [Only Int]
    check "backfill handles JSON, nested values, Unicode and SQL NULL" (mismatches == [Only 0])

    void (execute_ conn "UPDATE combos SET operation_count=operation_count+1, updated_at_ms=99999")
    afterBookkeeping <- equivalent conn
    check "bookkeeping never changes the payload fingerprint" (afterBookkeeping == initial)
    void (execute_ conn "UPDATE combos SET params_json=params_json, metrics_json=metrics_json")
    afterNoOp <- equivalent conn
    check "rewriting unchanged JSON preserves its fingerprint" (afterNoOp == initial)
    forM_ payloadChanges $ \statement -> do
        before <- equivalent conn
        void (execute_ conn statement)
        after <- equivalent conn
        check "every fingerprinted payload column still invalidates" (before /= after)

    bracket (connectPostgreSQL url) close $ \writer -> do
        setSchema writer schema
        before <- equivalent conn
        bracket_ (begin writer) (rollback writer) $ do
            void (execute_ writer changeOtherRow)
            duringWriter <- equivalent writer
            duringReader <- equivalent conn
            check "generated digests change in the writing transaction" (duringWriter /= before)
            check "uncommitted payload changes remain invisible to other readers" (duringReader == before)
        afterRollback <- equivalent conn
        check "rollback restores the payload and digest together" (afterRollback == before)
        void (withTransaction writer (execute_ writer changeOtherRow))
        afterCommit <- equivalent conn
        check "another writer's committed payload invalidates immediately" (afterCommit /= before)

    forged <- try (execute_ conn "UPDATE combos SET params_json_md5='forged'") :: IO (Either SqlError Int64)
    check "writers cannot supply a forged digest" (isLeft forged)
    beforeDelete <- equivalent conn
    void (execute_ conn "DELETE FROM combos WHERE combo_uuid='00000000-0000-4000-8000-000000000003'")
    afterDelete <- equivalent conn
    check "deleting a row invalidates the fingerprint" (beforeDelete /= afterDelete)
    void (execute conn insertFixture ("00000000-0000-4000-8000-000000000005" :: String, Nothing :: Maybe T.Text, Nothing :: Maybe T.Text))
    afterInsert <- equivalent conn
    check "inserting a row invalidates the fingerprint" (afterInsert /= afterDelete)
    void (execute_ conn "TRUNCATE combos CASCADE")
    empty <- equivalent conn
    check "empty tables retain the legacy fingerprint" (empty == [(0, "")])
    putStrLn "Ops migration integration passed (generated digests and transactional invalidation)."

equivalent :: Connection -> IO [(Int64, String)]
equivalent conn = do
    current <- query_ conn topCombosDbFingerprintQuery
    legacy <- query_ conn legacyFingerprintQuery
    check "generated-digest query remains identical to legacy JSON hashing" (current == legacy)
    pure current

payloads :: Connection -> IO [(Maybe String, Maybe String)]
payloads conn = query_ conn "SELECT params_json::text, metrics_json::text FROM combos ORDER BY combo_uuid"

check :: String -> Bool -> IO ()
check label ok = unless ok (ioError (userError label))

fixtures :: [(String, Maybe T.Text, Maybe T.Text)]
fixtures =
    [ ("00000000-0000-4000-8000-000000000001", Just "{\"label\":\"π:你好\",\"nested\":[1,2.5,null]}", Just "{\"equity\":1.2,\"trace\":[0,0.1]}")
    , ("00000000-0000-4000-8000-000000000002", Nothing, Just "{}")
    , ("00000000-0000-4000-8000-000000000003", Just "{}", Nothing)
    ]

insertFixture :: Query
insertFixture =
    "INSERT INTO combos(combo_uuid,params_json,metrics_json,created_at_ms,score,final_equity,annualized_return,open_threshold,close_threshold,objective,source) VALUES(?::uuid,?::jsonb,?::jsonb,1000,0.4,1.2,0.2,0.005,0.005,'return','synthetic')"

changeOtherRow :: Query
changeOtherRow = "UPDATE combos SET params_json='{\"otherWriter\":true}'::jsonb WHERE combo_uuid='00000000-0000-4000-8000-000000000002'"

payloadChanges :: [Query]
payloadChanges =
    [ "UPDATE combos SET params_json='{\"changed\":true}'::jsonb WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET metrics_json='{\"changed\":true}'::jsonb WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET params_json=NULL WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET metrics_json=NULL WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET score=0.9 WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET final_equity=0.8 WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET annualized_return=-0.2 WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET open_threshold=0.002 WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET close_threshold=0.003 WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET objective='sharpe' WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET source='changed' WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET created_at_ms=2000 WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    , "UPDATE combos SET combo_uuid='00000000-0000-4000-8000-000000000004' WHERE combo_uuid='00000000-0000-4000-8000-000000000001'"
    ]

-- Frozen pre-v6 computation: generated columns must change cost, not identity.
legacyFingerprintQuery :: Query
legacyFingerprintQuery =
    "SELECT count(*)::bigint, "
        <> "coalesce(md5(string_agg(combo_uuid::text || ':' || coalesce(created_at_ms::text, '') "
        <> "|| ':' || coalesce(score::text, '') || ':' || coalesce(final_equity::text, '') || ':' || coalesce(annualized_return::text, '') "
        <> "|| ':' || coalesce(open_threshold::text, '') || ':' || coalesce(close_threshold::text, '') "
        <> "|| ':' || coalesce(objective, '') || ':' || coalesce(source, '') "
        <> "|| ':' || coalesce(md5(params_json::text), '') || ':' || coalesce(md5(metrics_json::text), ''), ',' ORDER BY combo_uuid)), '') "
        <> "FROM combos"

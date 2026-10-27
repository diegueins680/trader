{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.OpsMigrations (testOpsMigrations) where

import Control.Exception (bracket, bracket_, try)
import Control.Monad (forM_, unless, void)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Char8 as BS
import Data.Either (isLeft)
import Data.Int (Int64)
import Data.String (fromString)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.UUID as UUID
import qualified Data.UUID.V4 as UUID
import Database.PostgreSQL.Simple (Connection, Only (..), SqlError, begin, close, connectPostgreSQL, execute, execute_, query, query_, rollback, withTransaction)
import Database.PostgreSQL.Simple.Types (PGArray (..), Query)
import System.Environment (lookupEnv)
import Trader.Ops.ComboParameters (persistComboParameters, persistComboReplicas)
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
    checkComboParameters conn
    checkComboReplicas conn
    putStrLn "Ops migration integration passed (generated digests and transactional invalidation)."

checkComboParameters :: Connection -> IO ()
checkComboParameters conn = do
    forM_ fixtures $ \row -> void (execute conn insertFixture row)
    strategyIds <- query_ conn "INSERT INTO strategies(code,label) VALUES('parameter-test','Parameter test') RETURNING id" :: IO [Only Int]
    strategyId <- case strategyIds of
        [Only sid] -> pure sid
        _ -> ioError (userError "parameter fixture strategy missing")
    void $
        execute_
            conn
            ( "CREATE TABLE parameter_insert_audit (statements integer NOT NULL); INSERT INTO parameter_insert_audit VALUES (0); "
                <> "CREATE FUNCTION count_parameter_insert() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN UPDATE parameter_insert_audit SET statements=statements+1; RETURN NULL; END $$; "
                <> "CREATE TRIGGER count_parameter_insert AFTER INSERT ON combo_parameters FOR EACH STATEMENT EXECUTE FUNCTION count_parameter_insert()"
            )
    let combo = UUID.fromWords 0 0x4000 0x80000000 1
        other = UUID.fromWords 0 0x4000 0x80000000 2
        params =
            KM.fromList $
                [ (AK.fromString ("parameter" ++ show n), Aeson.toJSON n)
                | n <- [1 .. 100 :: Int]
                ]
                    ++ [ ("text'π\\key", Aeson.String "quoted ' Unicode 你好 and \\ slash")
                       , ("null", Aeson.Null)
                       , ("bool", Aeson.Bool True)
                       , ("nested", Aeson.object ["values" Aeson..= [Aeson.Null, Aeson.toJSON (2.5 :: Double)]])
                       ]
        readParams uuid = do
            rows <- query conn "SELECT coalesce(jsonb_object_agg(name,value_json),'{}'::jsonb)::text FROM combo_parameters WHERE combo_uuid=?" (Only uuid) :: IO [Only T.Text]
            case rows of
                [Only value] -> either (ioError . userError) pure (Aeson.eitherDecodeStrict' (TE.encodeUtf8 value))
                _ -> ioError (userError "parameter replica missing")
        countStatements = query_ conn "SELECT statements FROM parameter_insert_audit" :: IO [Only Int]
    withTransaction conn (persistComboParameters conn combo (Just strategyId) params)
    stored <- readParams combo
    check "batched parameters preserve every key, nested JSON, null, Unicode and escaping" (stored == Aeson.Object params)
    statements <- countStatements
    check "one hundred parameters use one INSERT statement" (statements == [Only 1])
    wrongStrategy <- query conn "SELECT count(*) FROM combo_parameters WHERE combo_uuid=? AND strategy_id IS DISTINCT FROM ?" (combo, strategyId) :: IO [Only Int]
    check "every parameter retains its strategy association" (wrongStrategy == [Only 0])

    withTransaction conn (persistComboParameters conn other Nothing (KM.singleton "other" (Aeson.Bool True)))
    let replacement = KM.singleton "replacement" Aeson.Null
    withTransaction conn (persistComboParameters conn combo Nothing replacement)
    replaced <- readParams combo
    untouched <- readParams other
    nullStrategies <- query conn "SELECT count(*) FROM combo_parameters WHERE combo_uuid=? AND strategy_id IS NOT NULL" (Only combo) :: IO [Only Int]
    check "replacement removes obsolete keys and supports an absent strategy" (replaced == Aeson.Object replacement && nullStrategies == [Only 0])
    check "replacement leaves other combos untouched" (untouched == Aeson.object ["other" Aeson..= True])

    void (execute_ conn "ALTER TABLE combo_parameters ADD CONSTRAINT reject_parameter_test CHECK (name <> 'reject_parameter')")
    beforeFailure <- countStatements
    failed <- try (withTransaction conn (persistComboParameters conn combo (Just strategyId) (KM.insert "reject_parameter" (Aeson.Bool True) params))) :: IO (Either SqlError ())
    afterFailure <- readParams combo
    afterFailureCount <- countStatements
    check "failed batch rolls back deletion and all inserted values in the caller transaction" (isLeft failed && afterFailure == replaced && afterFailureCount == beforeFailure)
    withTransaction conn (persistComboParameters conn combo Nothing KM.empty)
    cleared <- readParams combo
    check "empty replacement clears all parameters" (cleared == Aeson.Object KM.empty)
    putStrLn "Combo parameter integration passed (batched insert, JSON preservation, replacement and rollback)."

checkComboReplicas :: Connection -> IO ()
checkComboReplicas conn = do
    void $
        execute_
            conn
            ( "CREATE TABLE replica_batch_audit (rows integer NOT NULL); "
                <> "CREATE FUNCTION audit_replica_batch() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN INSERT INTO replica_batch_audit SELECT count(*) FROM inserted; RETURN NULL; END $$; "
                <> "CREATE TRIGGER audit_replica_batch AFTER INSERT ON combos REFERENCING NEW TABLE AS inserted FOR EACH STATEMENT EXECUTE FUNCTION audit_replica_batch()"
            )
    let uuid = UUID.fromWords 0 0x4000 0x80000000
        upsert =
            "INSERT INTO combos (combo_uuid, score, created_at_ms) VALUES (?, ?, ?) ON CONFLICT (combo_uuid) DO UPDATE SET score=EXCLUDED.score, created_at_ms=COALESCE(combos.created_at_ms,EXCLUDED.created_at_ms)"
        replica n stamp params = (uuid n, (uuid n, fromIntegral n :: Double, stamp :: Maybe Int64), Nothing, params)
        rows = [replica n (Just 10) (KM.singleton "value" (Aeson.toJSON n)) | n <- [10000 .. 10200]]
        stored = query_ conn "SELECT combo_uuid,score,created_at_ms,params_json::text,metrics_json::text FROM combos ORDER BY combo_uuid" :: IO [(UUID.UUID, Maybe Double, Maybe Int64, Maybe T.Text, Maybe T.Text)]
        parameters = query_ conn "SELECT combo_uuid,name,value_json::text,strategy_id FROM combo_parameters ORDER BY combo_uuid,name" :: IO [(UUID.UUID, T.Text, Maybe T.Text, Maybe Int)]
    withTransaction conn (persistComboReplicas conn upsert rows)
    batches <- query_ conn "SELECT rows FROM replica_batch_audit ORDER BY rows" :: IO [Only Int]
    check "201 distinct replicas use bounded batches of 100, 100 and 1" (batches == [Only 1, Only 100, Only 100])
    inserted <- query_ conn "SELECT count(*) FROM combo_parameters WHERE combo_uuid >= '00000000-0000-4000-8000-000000002710' AND combo_uuid <= '00000000-0000-4000-8000-0000000027d8'" :: IO [Only Int]
    check "batched replicas retain every parameter row" (inserted == [Only 201])
    strategyIds <- query_ conn "SELECT id FROM strategies WHERE code='parameter-test'" :: IO [Only Int]
    strategyId <- case strategyIds of
        [Only sid] -> pure sid
        _ -> ioError (userError "replica fixture strategy missing")
    let associated =
            [ (uuid 19000, (uuid 19000, 1 :: Double, Nothing :: Maybe Int64), Just strategyId, KM.singleton "associated" Aeson.Null)
            , replica 19001 Nothing (KM.singleton "unassociated" (Aeson.Bool True))
            ]
    withTransaction conn (persistComboReplicas conn upsert associated)
    associations <- query conn "SELECT combo_uuid,strategy_id FROM combo_parameters WHERE combo_uuid=ANY(?::uuid[]) ORDER BY combo_uuid" (Only (PGArray [uuid 19000, uuid 19001])) :: IO [(UUID.UUID, Maybe Int)]
    check "a mixed batch preserves each parameter strategy association" (associations == [(uuid 19000, Just strategyId), (uuid 19001, Nothing)])
    let duplicate = uuid 20000
        other = uuid 20001
        duplicateRows =
            [ replica 20000 Nothing (KM.singleton "obsolete" Aeson.Null)
            , replica 20001 (Just 4) (KM.singleton "untouched" (Aeson.Bool True))
            , replica 20000 (Just 7) (KM.singleton "later" (Aeson.String "Unicode 你好 ' \\"))
            , (duplicate, (duplicate, 99 :: Double, Just (9 :: Int64)), Nothing, KM.empty)
            ]
    withTransaction conn (persistComboReplicas conn upsert duplicateRows)
    duplicated <- query conn "SELECT score,created_at_ms FROM combos WHERE combo_uuid=?" (Only duplicate) :: IO [(Maybe Double, Maybe Int64)]
    duplicateParams <- query conn "SELECT count(*) FROM combo_parameters WHERE combo_uuid=?" (Only duplicate) :: IO [Only Int]
    otherParams <- query conn "SELECT value_json::text FROM combo_parameters WHERE combo_uuid=?" (Only other) :: IO [Only T.Text]
    check "nonadjacent duplicate UUIDs retain last-write order and first non-null creation stamp" (duplicated == [(Just 99, Just 7)])
    check "an empty final duplicate removes earlier parameter keys without touching other combos" (duplicateParams == [Only 0] && otherParams == [Only "true"])
    before <- stored
    beforeParams <- parameters
    beforeAudit <- query_ conn "SELECT count(*) FROM replica_batch_audit" :: IO [Only Int]
    withTransaction conn (persistComboReplicas conn upsert ([] :: [(UUID.UUID, (UUID.UUID, Double, Maybe Int64), Maybe Int, Aeson.Object)]))
    afterEmptyAudit <- query_ conn "SELECT count(*) FROM replica_batch_audit" :: IO [Only Int]
    check "an empty replica batch issues no upsert" (beforeAudit == afterEmptyAudit)
    let invalidRows =
            [ replica n (Just 42) (KM.singleton (if n == 30150 then "reject_parameter" else "valid") (Aeson.toJSON n))
            | n <- [30000 .. 30200]
            ]
    failed <- try (withTransaction conn (persistComboReplicas conn upsert invalidRows)) :: IO (Either SqlError ())
    after <- stored
    afterParams <- parameters
    afterAudit <- query_ conn "SELECT count(*) FROM replica_batch_audit" :: IO [Only Int]
    check "a later invalid parameter rolls back all replica batches and audit effects" (isLeft failed && before == after && beforeParams == afterParams && beforeAudit == afterAudit)
    putStrLn "Combo replica integration passed (bounded statements, duplicates, creation order, empty parameters and rollback)."

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

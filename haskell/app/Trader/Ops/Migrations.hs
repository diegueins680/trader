{-# LANGUAGE OverloadedStrings #-}

module Trader.Ops.Migrations (
    ensureOpsDbSchema,
) where

import Control.Monad (forM_, unless, void)
import Data.Int (Int64)
import qualified Data.List as List
import qualified Data.Set as Set
import Data.String (fromString)
import Data.Time.Clock.POSIX (getPOSIXTime)
import Database.PostgreSQL.Simple (Connection, Only (..), execute, execute_, query_, withTransaction)

data Migration = Migration
    { migVersion :: !Int
    , migDescription :: !String
    , migStatements :: ![String]
    }

migrations :: [Migration]
migrations =
    [ Migration
        { migVersion = 1
        , migDescription = "initial_ops_schema"
        , migStatements =
            [ "CREATE TABLE IF NOT EXISTS platforms ("
                <> "id SERIAL PRIMARY KEY,"
                <> "code TEXT UNIQUE NOT NULL,"
                <> "label TEXT NOT NULL,"
                <> "rest_url TEXT,"
                <> "ws_url TEXT,"
                <> "api_docs_url TEXT,"
                <> "connection_json JSONB,"
                <> "created_at_ms BIGINT,"
                <> "updated_at_ms BIGINT"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS git_commits ("
                <> "id BIGSERIAL PRIMARY KEY,"
                <> "commit_hash TEXT UNIQUE NOT NULL,"
                <> "version TEXT,"
                <> "committed_at_ms BIGINT,"
                <> "created_at_ms BIGINT"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS platform_symbols ("
                <> "id BIGSERIAL PRIMARY KEY,"
                <> "platform_id INTEGER NOT NULL REFERENCES platforms(id) ON DELETE CASCADE,"
                <> "symbol TEXT NOT NULL,"
                <> "market TEXT,"
                <> "base_asset TEXT,"
                <> "quote_asset TEXT,"
                <> "status TEXT,"
                <> "metadata_json JSONB,"
                <> "created_at_ms BIGINT,"
                <> "updated_at_ms BIGINT,"
                <> "UNIQUE (platform_id, symbol, market)"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS bots ("
                <> "id BIGSERIAL PRIMARY KEY,"
                <> "tenant_key TEXT,"
                <> "platform_id INTEGER NOT NULL REFERENCES platforms(id) ON DELETE CASCADE,"
                <> "symbol_id BIGINT REFERENCES platform_symbols(id) ON DELETE SET NULL,"
                <> "symbol TEXT NOT NULL,"
                <> "market TEXT,"
                <> "interval TEXT,"
                <> "live BOOLEAN,"
                <> "trade_enabled BOOLEAN,"
                <> "running BOOLEAN,"
                <> "combo_uuid UUID,"
                <> "args_json JSONB,"
                <> "status_json JSONB,"
                <> "started_at_ms BIGINT,"
                <> "updated_at_ms BIGINT,"
                <> "UNIQUE (tenant_key, platform_id, symbol, market, interval)"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS positions ("
                <> "id BIGSERIAL PRIMARY KEY,"
                <> "platform_id INTEGER NOT NULL REFERENCES platforms(id) ON DELETE CASCADE,"
                <> "symbol_id BIGINT REFERENCES platform_symbols(id) ON DELETE SET NULL,"
                <> "bot_id BIGINT REFERENCES bots(id) ON DELETE SET NULL,"
                <> "symbol TEXT NOT NULL,"
                <> "market TEXT,"
                <> "side TEXT,"
                <> "quantity DOUBLE PRECISION,"
                <> "entry_price DOUBLE PRECISION,"
                <> "mark_price DOUBLE PRECISION,"
                <> "leverage DOUBLE PRECISION,"
                <> "pnl_unrealized DOUBLE PRECISION,"
                <> "position_json JSONB,"
                <> "opened_at_ms BIGINT,"
                <> "updated_at_ms BIGINT"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS position_origins ("
                <> "tenant_key TEXT NOT NULL,"
                <> "platform_id INTEGER NOT NULL REFERENCES platforms(id) ON DELETE CASCADE,"
                <> "symbol TEXT NOT NULL,"
                <> "market TEXT NOT NULL,"
                <> "side TEXT NOT NULL,"
                <> "combo_uuid UUID,"
                <> "opened_at_ms BIGINT,"
                <> "order_id TEXT,"
                <> "updated_at_ms BIGINT,"
                <> "PRIMARY KEY (tenant_key, platform_id, symbol, market)"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS strategies ("
                <> "id SERIAL PRIMARY KEY,"
                <> "code TEXT UNIQUE NOT NULL,"
                <> "label TEXT NOT NULL"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS combos ("
                <> "combo_uuid UUID PRIMARY KEY,"
                <> "strategy_id INTEGER REFERENCES strategies(id),"
                <> "symbol TEXT,"
                <> "interval TEXT,"
                <> "objective TEXT,"
                <> "source TEXT,"
                <> "final_equity DOUBLE PRECISION,"
                <> "annualized_return DOUBLE PRECISION,"
                <> "score DOUBLE PRECISION,"
                <> "open_threshold DOUBLE PRECISION,"
                <> "close_threshold DOUBLE PRECISION,"
                <> "params_json JSONB,"
                <> "metrics_json JSONB,"
                <> "operation_count INTEGER NOT NULL DEFAULT 0,"
                <> "created_at_ms BIGINT,"
                <> "updated_at_ms BIGINT"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS combo_parameters ("
                <> "combo_uuid UUID NOT NULL REFERENCES combos(combo_uuid) ON DELETE CASCADE,"
                <> "name TEXT NOT NULL,"
                <> "value_json JSONB,"
                <> "strategy_id INTEGER REFERENCES strategies(id),"
                <> "PRIMARY KEY (combo_uuid, name)"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS ops ("
                <> "id BIGSERIAL PRIMARY KEY,"
                <> "tenant_key TEXT,"
                <> "at_ms BIGINT NOT NULL,"
                <> "kind TEXT NOT NULL,"
                <> "symbol TEXT,"
                <> "combo_uuid UUID,"
                <> "order_id TEXT,"
                <> "params_json JSONB,"
                <> "args_json JSONB,"
                <> "result_json JSONB,"
                <> "equity DOUBLE PRECISION,"
                <> "git_commit_id BIGINT REFERENCES git_commits(id)"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS async_jobs ("
                <> "job_id TEXT PRIMARY KEY,"
                <> "job_type TEXT NOT NULL,"
                <> "status TEXT NOT NULL,"
                <> "payload_json JSONB NOT NULL,"
                <> "created_at_ms BIGINT,"
                <> "completed_at_ms BIGINT,"
                <> "updated_at_ms BIGINT NOT NULL"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS trade_requests ("
                <> "tenant_scope TEXT NOT NULL,"
                <> "tenant_key TEXT,"
                <> "idempotency_key TEXT NOT NULL,"
                <> "request_hash TEXT NOT NULL,"
                <> "status TEXT NOT NULL,"
                <> "response_json JSONB,"
                <> "error_text TEXT,"
                <> "created_at_ms BIGINT NOT NULL,"
                <> "updated_at_ms BIGINT NOT NULL,"
                <> "completed_at_ms BIGINT,"
                <> "PRIMARY KEY (tenant_scope, idempotency_key)"
                <> ")"
            , "CREATE TABLE IF NOT EXISTS outbox_events ("
                <> "id BIGSERIAL PRIMARY KEY,"
                <> "tenant_key TEXT,"
                <> "topic TEXT NOT NULL,"
                <> "event_key TEXT,"
                <> "payload_json JSONB NOT NULL,"
                <> "status TEXT NOT NULL DEFAULT 'pending',"
                <> "attempts INTEGER NOT NULL DEFAULT 0,"
                <> "next_attempt_at_ms BIGINT,"
                <> "last_error TEXT,"
                <> "created_at_ms BIGINT NOT NULL,"
                <> "updated_at_ms BIGINT NOT NULL,"
                <> "published_at_ms BIGINT"
                <> ")"
            , "ALTER TABLE bots ADD COLUMN IF NOT EXISTS tenant_key TEXT"
            , "ALTER TABLE ops ADD COLUMN IF NOT EXISTS tenant_key TEXT"
            , "ALTER TABLE bots DROP CONSTRAINT IF EXISTS bots_platform_id_symbol_market_interval_key"
            , "ALTER TABLE ops ADD COLUMN IF NOT EXISTS platform_id INTEGER REFERENCES platforms(id)"
            , "ALTER TABLE ops ADD COLUMN IF NOT EXISTS symbol_id BIGINT REFERENCES platform_symbols(id)"
            , "ALTER TABLE ops ADD COLUMN IF NOT EXISTS git_commit_id BIGINT REFERENCES git_commits(id)"
            , "ALTER TABLE git_commits ADD COLUMN IF NOT EXISTS committed_at_ms BIGINT"
            , "CREATE UNIQUE INDEX IF NOT EXISTS bots_tenant_platform_symbol_interval_idx ON bots(tenant_key, platform_id, symbol, market, interval)"
            , "CREATE INDEX IF NOT EXISTS ops_kind_idx ON ops(kind)"
            , "CREATE INDEX IF NOT EXISTS ops_combo_uuid_idx ON ops(combo_uuid)"
            , "CREATE INDEX IF NOT EXISTS ops_symbol_idx ON ops(symbol)"
            , "CREATE INDEX IF NOT EXISTS ops_platform_idx ON ops(platform_id)"
            , "CREATE INDEX IF NOT EXISTS ops_tenant_key_idx ON ops(tenant_key)"
            , "CREATE INDEX IF NOT EXISTS ops_symbol_id_idx ON ops(symbol_id)"
            , "CREATE INDEX IF NOT EXISTS ops_git_commit_id_idx ON ops(git_commit_id)"
            , "CREATE INDEX IF NOT EXISTS git_commits_committed_at_ms_idx ON git_commits(committed_at_ms)"
            , "CREATE INDEX IF NOT EXISTS ops_order_id_idx ON ops(order_id)"
            , "CREATE INDEX IF NOT EXISTS async_jobs_type_status_idx ON async_jobs(job_type, status)"
            , "CREATE INDEX IF NOT EXISTS async_jobs_updated_at_ms_idx ON async_jobs(updated_at_ms)"
            , "CREATE INDEX IF NOT EXISTS trade_requests_tenant_status_idx ON trade_requests(tenant_scope, status)"
            , "CREATE INDEX IF NOT EXISTS trade_requests_updated_at_ms_idx ON trade_requests(updated_at_ms)"
            , "CREATE INDEX IF NOT EXISTS outbox_events_status_next_attempt_idx ON outbox_events(status, next_attempt_at_ms)"
            , "CREATE INDEX IF NOT EXISTS outbox_events_created_at_ms_idx ON outbox_events(created_at_ms)"
            , "CREATE INDEX IF NOT EXISTS outbox_events_tenant_topic_idx ON outbox_events(tenant_key, topic)"
            , "CREATE INDEX IF NOT EXISTS ops_at_ms_idx ON ops(at_ms)"
            , "CREATE INDEX IF NOT EXISTS ops_symbol_at_ms_idx ON ops(symbol, at_ms)"
            , "CREATE UNIQUE INDEX IF NOT EXISTS platforms_code_idx ON platforms(code)"
            , "CREATE INDEX IF NOT EXISTS platform_symbols_platform_idx ON platform_symbols(platform_id)"
            , "CREATE INDEX IF NOT EXISTS platform_symbols_symbol_idx ON platform_symbols(symbol)"
            , "CREATE INDEX IF NOT EXISTS platform_symbols_market_idx ON platform_symbols(market)"
            , "CREATE INDEX IF NOT EXISTS bots_platform_symbol_idx ON bots(platform_id, symbol)"
            , "CREATE INDEX IF NOT EXISTS bots_running_idx ON bots(running)"
            , "CREATE UNIQUE INDEX IF NOT EXISTS positions_bot_id_uniq ON positions(bot_id) WHERE bot_id IS NOT NULL"
            , "CREATE INDEX IF NOT EXISTS positions_platform_symbol_idx ON positions(platform_id, symbol)"
            , "CREATE INDEX IF NOT EXISTS positions_market_idx ON positions(market)"
            , "CREATE INDEX IF NOT EXISTS position_origins_combo_uuid_idx ON position_origins(combo_uuid)"
            , "CREATE INDEX IF NOT EXISTS position_origins_symbol_idx ON position_origins(symbol)"
            , "CREATE INDEX IF NOT EXISTS position_origins_updated_at_ms_idx ON position_origins(updated_at_ms)"
            , "CREATE INDEX IF NOT EXISTS combos_symbol_idx ON combos(symbol)"
            , "CREATE INDEX IF NOT EXISTS combos_interval_idx ON combos(interval)"
            , "CREATE INDEX IF NOT EXISTS combos_strategy_idx ON combos(strategy_id)"
            , "CREATE INDEX IF NOT EXISTS combos_operation_count_idx ON combos(operation_count)"
            , "CREATE INDEX IF NOT EXISTS combos_annualized_return_idx ON combos(annualized_return)"
            , "CREATE INDEX IF NOT EXISTS combo_parameters_name_idx ON combo_parameters(name)"
            ]
        }
    , Migration
        { migVersion = 2
        , migDescription = "combos_store_source_metadata"
        , migStatements =
            [ "ALTER TABLE combos ADD COLUMN IF NOT EXISTS source TEXT"
            , "CREATE INDEX IF NOT EXISTS combos_source_idx ON combos(source)"
            ]
        }
    , Migration
        { migVersion = 3
        , migDescription = "ops_live_schema_columns"
        , migStatements =
            [ "ALTER TABLE ops ADD COLUMN IF NOT EXISTS tenant_key TEXT"
            , "ALTER TABLE ops ADD COLUMN IF NOT EXISTS platform_id INTEGER REFERENCES platforms(id)"
            , "ALTER TABLE ops ADD COLUMN IF NOT EXISTS symbol_id BIGINT REFERENCES platform_symbols(id)"
            , "ALTER TABLE ops ADD COLUMN IF NOT EXISTS args_json JSONB"
            , "ALTER TABLE ops ADD COLUMN IF NOT EXISTS result_json JSONB"
            , "ALTER TABLE ops ADD COLUMN IF NOT EXISTS equity DOUBLE PRECISION"
            , "ALTER TABLE ops ADD COLUMN IF NOT EXISTS git_commit_id BIGINT REFERENCES git_commits(id)"
            , "ALTER TABLE git_commits ADD COLUMN IF NOT EXISTS committed_at_ms BIGINT"
            , "CREATE INDEX IF NOT EXISTS ops_kind_idx ON ops(kind)"
            , "CREATE INDEX IF NOT EXISTS ops_combo_uuid_idx ON ops(combo_uuid)"
            , "CREATE INDEX IF NOT EXISTS ops_symbol_idx ON ops(symbol)"
            , "CREATE INDEX IF NOT EXISTS ops_platform_idx ON ops(platform_id)"
            , "CREATE INDEX IF NOT EXISTS ops_tenant_key_idx ON ops(tenant_key)"
            , "CREATE INDEX IF NOT EXISTS ops_symbol_id_idx ON ops(symbol_id)"
            , "CREATE INDEX IF NOT EXISTS ops_git_commit_id_idx ON ops(git_commit_id)"
            , "CREATE INDEX IF NOT EXISTS git_commits_committed_at_ms_idx ON git_commits(committed_at_ms)"
            , "CREATE INDEX IF NOT EXISTS ops_order_id_idx ON ops(order_id)"
            , "CREATE INDEX IF NOT EXISTS ops_at_ms_idx ON ops(at_ms)"
            , "CREATE INDEX IF NOT EXISTS ops_symbol_at_ms_idx ON ops(symbol, at_ms)"
            ]
        }
    ]

ensureOpsDbSchema :: Connection -> IO ()
ensureOpsDbSchema conn = do
    void $
        execute_
            conn
            ( fromString
                ( "CREATE TABLE IF NOT EXISTS ops_schema_migrations ("
                    <> "version INTEGER PRIMARY KEY,"
                    <> "description TEXT NOT NULL,"
                    <> "applied_at_ms BIGINT NOT NULL"
                    <> ")"
                )
            )
    appliedRows <- query_ conn "SELECT version FROM ops_schema_migrations" :: IO [Only Int]
    let applied = Set.fromList [v | Only v <- appliedRows]
        ordered = List.sortOn migVersion migrations
    forM_ ordered $ \migration ->
        unless (Set.member (migVersion migration) applied) $
            withTransaction conn $ do
                forM_ (migStatements migration) (execute_ conn . fromString)
                now <- getTimestampMs
                void $
                    execute
                        conn
                        "INSERT INTO ops_schema_migrations (version, description, applied_at_ms) VALUES (?, ?, ?)"
                        (migVersion migration, migDescription migration, now)

getTimestampMs :: IO Int64
getTimestampMs = round . (* 1000) <$> getPOSIXTime

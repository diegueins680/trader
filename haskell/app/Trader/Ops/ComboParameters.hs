{-# LANGUAGE OverloadedStrings #-}

module Trader.Ops.ComboParameters (persistComboParameters, persistComboReplicas) where

import Control.Monad (void)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Lazy as BL
import qualified Data.Set as Set
import qualified Data.Text.Encoding as TE
import qualified Data.UUID as UUID
import Database.PostgreSQL.Simple (Connection, execute, executeMany)
import Database.PostgreSQL.Simple.ToRow (ToRow)
import Database.PostgreSQL.Simple.Types (Only (..), PGArray (..), Query)

{- | Replace one combo's parameter replica inside the caller's transaction.
Batch its values into one INSERT instead of one database round trip per key.
-}
persistComboParameters :: Connection -> UUID.UUID -> Maybe Int -> Aeson.Object -> IO ()
persistComboParameters conn comboUuid mStrategyId params =
    persistParameterBatch conn [(comboUuid, mStrategyId, params)]

{- | Persist at most 100 distinct UUIDs per statement inside the caller's
transaction. A repeated UUID ends the batch so upserts, first non-null
creation stamps and live-stat comparisons retain their input order.
-}
persistComboReplicas :: (ToRow row) => Connection -> Query -> [(UUID.UUID, row, Maybe Int, Aeson.Object)] -> IO ()
persistComboReplicas conn upsert = go
  where
    go [] = pure ()
    go rows = do
        let (batch, remaining) = takeBatch 100 Set.empty rows
        void $ executeMany conn upsert [row | (_, row, _, _) <- batch]
        persistParameterBatch conn [(uuid, strategy, params) | (uuid, _, strategy, params) <- batch]
        go remaining

    takeBatch :: Int -> Set.Set UUID.UUID -> [(UUID.UUID, row, Maybe Int, Aeson.Object)] -> ([(UUID.UUID, row, Maybe Int, Aeson.Object)], [(UUID.UUID, row, Maybe Int, Aeson.Object)])
    takeBatch _ _ [] = ([], [])
    takeBatch slots seen rows@(row@(uuid, _, _, _) : remaining)
        | slots <= 0 || Set.member uuid seen = ([], rows)
        | otherwise =
            let (batch, rest) = takeBatch (slots - 1) (Set.insert uuid seen) remaining
             in (row : batch, rest)

persistParameterBatch :: Connection -> [(UUID.UUID, Maybe Int, Aeson.Object)] -> IO ()
persistParameterBatch _ [] = pure ()
persistParameterBatch conn rows = do
    void $ execute conn "DELETE FROM combo_parameters WHERE combo_uuid = ANY(?::uuid[])" (Only (PGArray [uuid | (uuid, _, _) <- rows]))
    void $
        executeMany
            conn
            "INSERT INTO combo_parameters (combo_uuid, name, value_json, strategy_id) VALUES (?, ?, ?, ?)"
            [ (comboUuid, AK.toText key, Just (TE.decodeUtf8 (BL.toStrict (Aeson.encode value))), mStrategyId)
            | (comboUuid, mStrategyId, params) <- rows
            , (key, value) <- KM.toList params
            ]

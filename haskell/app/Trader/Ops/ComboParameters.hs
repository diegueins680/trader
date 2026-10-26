{-# LANGUAGE OverloadedStrings #-}

module Trader.Ops.ComboParameters (persistComboParameters) where

import Control.Monad (void)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Lazy as BL
import qualified Data.Text.Encoding as TE
import qualified Data.UUID as UUID
import Database.PostgreSQL.Simple (Connection, Only (..), execute, executeMany)

{- | Replace one combo's parameter replica inside the caller's transaction.
Batch its values into one INSERT instead of one database round trip per key.
-}
persistComboParameters :: Connection -> UUID.UUID -> Maybe Int -> Aeson.Object -> IO ()
persistComboParameters conn comboUuid mStrategyId params = do
    void $ execute conn "DELETE FROM combo_parameters WHERE combo_uuid = ?" (Only comboUuid)
    void $
        executeMany
            conn
            "INSERT INTO combo_parameters (combo_uuid, name, value_json, strategy_id) VALUES (?, ?, ?, ?)"
            [ (comboUuid, AK.toText key, Just (TE.decodeUtf8 (BL.toStrict (Aeson.encode value))), mStrategyId)
            | (key, value) <- KM.toList params
            ]

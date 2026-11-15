{-# LANGUAGE OverloadedStrings #-}

module Trader.TopComboRetention (
    comboUuidText,
    parseRetainedComboUuids,
    retainedComboUuidsFromEnv,
    retainRankedCombos,
) where

import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import Data.Maybe (fromMaybe)
import qualified Data.Set as Set
import qualified Data.Text as T
import qualified Data.UUID as UUID
import System.Environment (lookupEnv)

-- Retention protects records, never trading eligibility or order authority.
parseRetainedComboUuids :: String -> Either String (Set.Set T.Text)
parseRetainedComboUuids raw = do
    values <- traverse parseOne (filter (not . T.null) (map T.strip (T.splitOn "," (T.pack raw))))
    let uuids = Set.fromList values
    if Set.size uuids > 256
        then Left "TRADER_TOP_COMBOS_RETAIN_UUIDS accepts at most 256 unique UUIDs."
        else Right uuids
  where
    parseOne value =
        case UUID.fromText value of
            Nothing -> Left "TRADER_TOP_COMBOS_RETAIN_UUIDS contains an invalid UUID."
            Just uuid -> Right (UUID.toText uuid)

retainedComboUuidsFromEnv :: IO (Set.Set T.Text)
retainedComboUuidsFromEnv = do
    raw <- lookupEnv "TRADER_TOP_COMBOS_RETAIN_UUIDS"
    either (ioError . userError) pure (parseRetainedComboUuids (fromMaybe "" raw))

comboUuidText :: Aeson.Value -> Maybe T.Text
comboUuidText value =
    case value of
        Aeson.Object obj ->
            case KM.lookup "uuid" obj of
                Just (Aeson.String uuid) -> Just (T.toLower (T.strip uuid))
                _ -> Nothing
        _ -> Nothing

-- Reserve slots for present retained records, then fill in ranking order.
-- When pins exceed the limit, keep every pin; missing pins reserve no slots.
retainRankedCombos :: (a -> Maybe T.Text) -> Set.Set T.Text -> Int -> [a] -> [a]
retainRankedCombos uuidOf retained limit ranked = go ordinarySlots ranked
  where
    isRetained value = maybe False (`Set.member` retained) (uuidOf value)
    ordinarySlots = max 0 (max 0 limit - length (filter isRetained ranked))
    go _ [] = []
    go slots (value : rest)
        | isRetained value = value : go slots rest
        | slots > 0 = value : go (slots - 1) rest
        | otherwise = go slots rest

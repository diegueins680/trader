module Trader.BotStartSemantics (
    botTradeEnabledFromApi,
    botStartSymbolDisabled,
    queuedStartOrderErrorIssue,
    shouldResolveOriginComboOnAutoStart,
    shouldClearPositionOriginOnStart,
    shouldPersistPositionOriginOnSwitch,
    shouldPreserveProvidedComboOnActiveAdopt,
) where

import Data.Char (isSpace, toUpper)
import Data.Maybe (fromMaybe, isJust)

botTradeEnabledFromApi :: Maybe Bool -> Bool
botTradeEnabledFromApi = fromMaybe True

botStartSymbolDisabled :: [String] -> String -> Bool
botStartSymbolDisabled disabled sym =
    normalizeStartSymbol sym `elem` map normalizeStartSymbol disabled
  where
    normalizeStartSymbol = map toUpper . filter (not . isSpace)

queuedStartOrderErrorIssue :: Maybe Int -> Int -> Maybe String
queuedStartOrderErrorIssue mMaxOrderErrors orderErrors
    | orderErrors <= 0 = Nothing
    | otherwise =
        case mMaxOrderErrors of
            Just limit | limit > 0 && orderErrors >= limit -> Just ("order errors=" ++ show orderErrors ++ " reached maxOrderErrors")
            _ -> Nothing

shouldResolveOriginComboOnAutoStart :: Bool -> Bool
shouldResolveOriginComboOnAutoStart adoptActive = adoptActive

shouldPreserveProvidedComboOnActiveAdopt :: Bool -> Maybe a -> Bool
shouldPreserveProvidedComboOnActiveAdopt adoptActive providedCombo = adoptActive && isJust providedCombo

shouldClearPositionOriginOnStart :: Bool -> Bool -> Bool
shouldClearPositionOriginOnStart adoptable adoptActive = adoptable && not adoptActive

shouldPersistPositionOriginOnSwitch :: Bool -> Bool -> Bool -> Bool -> Bool
shouldPersistPositionOriginOnSwitch tradeEnabled live switchedApplied orderSent =
    tradeEnabled && live && switchedApplied && orderSent

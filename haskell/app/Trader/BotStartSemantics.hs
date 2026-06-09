module Trader.BotStartSemantics (
    botTradeEnabledFromApi,
    botStartSymbolDisabled,
    botStartupBacktestRoiAcceptable,
    botStartupBacktestAborts,
    prioritizeBotStartSymbols,
    queuedStartOrderErrorIssue,
    shouldResolveOriginComboOnAutoStart,
    shouldClearPositionOriginOnStart,
    shouldPersistPositionOriginOnSwitch,
    shouldPreserveProvidedComboOnActiveAdopt,
) where

import Data.Char (isSpace, toUpper)
import Data.Maybe (fromMaybe, isJust)

import Trader.Text (dedupeStable)

botTradeEnabledFromApi :: Maybe Bool -> Bool
botTradeEnabledFromApi = fromMaybe True

normalizeStartSymbol :: String -> String
normalizeStartSymbol = map toUpper . filter (not . isSpace)

botStartSymbolDisabled :: [String] -> String -> Bool
botStartSymbolDisabled disabled sym =
    normalizeStartSymbol sym `elem` map normalizeStartSymbol disabled

botStartupBacktestRoiAcceptable :: Maybe Double -> Bool
botStartupBacktestRoiAcceptable (Just finalEquity) =
    finalEquity > 1.0 && not (isNaN finalEquity || isInfinite finalEquity)
botStartupBacktestRoiAcceptable Nothing = False

{- | Decide whether the startup combo backtest guard should abort a bot start.

The guard aborts only when it is enabled AND the backtest produced a
final-equity reading that fails the ROI threshold. Two cases deliberately
never abort (fail open), so that live trading is not held hostage to the
backtest path:

  * the guard is disabled (@enabled = False@) — e.g. the box runs with
    @TRADER_TOP_COMBOS_BACKTEST_ENABLED=false@; and
  * no final-equity reading is available (@Nothing@) — i.e. the backtest
    errored, timed out, or returned no metrics (an infrastructure failure,
    not a verdict on the combo).
-}
botStartupBacktestAborts :: Bool -> Maybe Double -> Bool
botStartupBacktestAborts False _ = False
botStartupBacktestAborts True Nothing = False
botStartupBacktestAborts True mFinalEquity = not (botStartupBacktestRoiAcceptable mFinalEquity)

prioritizeBotStartSymbols :: [String] -> [String] -> [String]
prioritizeBotStartSymbols regularSymbols orphanSymbols =
    filter (not . null) $
        dedupeStable $
            map normalizeStartSymbol (orphanSymbols ++ regularSymbols)

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

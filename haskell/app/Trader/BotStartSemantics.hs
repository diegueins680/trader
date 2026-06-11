module Trader.BotStartSemantics (
    botTradeEnabledFromApi,
    botStartSymbolDisabled,
    botStartupBacktestRoiAcceptable,
    botStartupBacktestAborts,
    BacktestVerdict (..),
    botStartupBacktestVerdict,
    backtestVerdictAborts,
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

{- | Three-valued verdict for the top-combo startup backtest guard.

  * 'BacktestAllow'      — backtest cleared the bar; allow start.
  * 'BacktestAbort'      — backtest produced a verdict that fails the bar;
                           block start (and let upstream prune the combo).
  * 'BacktestNoVerdict'  — backtest did not produce an actionable verdict on
                           the combo (e.g. zero trades fired in the smoke
                           window). Fail open: do not block the start, and
                           do not let upstream prune the combo.
-}
data BacktestVerdict
    = BacktestAllow
    | BacktestAbort
    | BacktestNoVerdict
    deriving (Eq, Show)

{- | Decide the verdict for a startup combo backtest given:

      * whether the guard is enabled,
      * the @finalEquity@ reading (if any), and
      * the @tradeCount@ reading (if any).

    The crucial invariant added 2026-06-10 is:

      A backtest that fired zero trades is /not/ a verdict on the combo's
      profitability — it is a verdict on the smoke /window/. The smoke
      backtest is a short, signal-gated slice; on quiet days the dominant
      outcome is "no trade fired" with @finalEquity == 1.0@ exactly.
      Treating that as a loss (a) blocks otherwise-valid starts and
      (b) silently deletes the combo from top-combos JSON + DB, eroding
      the strategy bank a little more each quiet day. The 2026-06-10
      launchd log shows 124 such erroneous prunes in a single session,
      versus 1 genuine loss.

    Falsification:

      * 'BacktestAllow' \<\=\> guard enabled \& finalEquity is finite \& \> 1.0.
      * 'BacktestAbort' \<\=\> guard enabled \& tradeCount \> 0 \&
                        finalEquity is non-acceptable (sub-threshold or
                        non-finite).
      * 'BacktestNoVerdict' \<\=\> guard enabled \& (no finalEquity reading
                        OR no tradeCount reading OR tradeCount == 0).
      * Guard disabled always yields 'BacktestAllow'.
-}
botStartupBacktestVerdict :: Bool -> Maybe Double -> Maybe Int -> BacktestVerdict
botStartupBacktestVerdict False _ _ = BacktestAllow
botStartupBacktestVerdict True Nothing _ = BacktestNoVerdict
botStartupBacktestVerdict True mFinalEquity mTradeCount =
    if botStartupBacktestRoiAcceptable mFinalEquity
        then BacktestAllow
        else case mTradeCount of
            Just n | n > 0 -> BacktestAbort
            -- Zero-trade or unknown-trade smoke window: not a verdict on the
            -- combo. Do not abort, do not prune.
            _ -> BacktestNoVerdict

-- | Convenience: does this verdict block the start?
backtestVerdictAborts :: BacktestVerdict -> Bool
backtestVerdictAborts BacktestAbort = True
backtestVerdictAborts _ = False

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

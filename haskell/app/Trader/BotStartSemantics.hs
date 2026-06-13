module Trader.BotStartSemantics (
    botTradeEnabledFromApi,
    botStartSymbolDisabled,
    botStartupBacktestRoiAcceptable,
    botStartupBacktestAborts,
    BacktestVerdict (..),
    botStartupBacktestVerdict,
    botStartupBacktestVerdictWithMinTrades,
    backtestVerdictAborts,
    defaultBotStartupBacktestMinTrades,
    botStartupGuardShouldPrune,
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

{- | Default minimum trade count below which a sub-threshold smoke backtest is
NOT treated as an actionable verdict. The 2026-06-12 launchd review showed
smoke windows producing a single "loss" (n=1) that aborted starts for combos
with out-of-sample @finalEquity@ ≥ 1.42 — i.e. n=1 evidence overruling n=many
optimizer evidence. Three trades is the smallest sample where a non-trivial
win-rate-vs-payoff calculation can distinguish "signal lost" from "unlucky
first trade". Tunable via @TRADER_BOT_START_BACKTEST_MIN_TRADES@.
-}
defaultBotStartupBacktestMinTrades :: Int
defaultBotStartupBacktestMinTrades = 3

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
botStartupBacktestVerdict = botStartupBacktestVerdictWithMinTrades 1

{- | Like 'botStartupBacktestVerdict' but only treats a sub-threshold reading
as 'BacktestAbort' when the smoke window saw at least @minTrades@ trades.
Below that, the verdict is 'BacktestNoVerdict': fail open and do not prune.

    Engineering rationale (2026-06-12):

      The 2026-06-11 fix correctly distinguished zero-trade smoke windows
      (no verdict) from traded losses (verdict). But the 2026-06-12 launchd
      log shows a more subtle erosion mode: smoke windows that fire a
      /single/ trade and lose (e.g. @finalEquity == 0.954@ on AAVEUSDT,
      which is one ~5% drawdown — within one daily ATR) were aborting
      starts and prompting deletes of the combo's DB row. A single trade is
      below the noise floor for a combo whose out-of-sample @finalEquity@
      is ≥ 1.42 (i.e. n=many optimizer evidence). Requiring @minTrades@ ≥ 3
      raises the evidence bar to where a sub-threshold reading is more
      consistent with "signal lost" than "unlucky first trade".

      Falsification rows (with @minTrades = 3@):

        * enabled=True finalEquity=Just 1.5  tradeCount=Just 0  → Allow
        * enabled=True finalEquity=Just 0.95 tradeCount=Just 1  → NoVerdict (below minTrades)
        * enabled=True finalEquity=Just 0.95 tradeCount=Just 2  → NoVerdict (below minTrades)
        * enabled=True finalEquity=Just 0.95 tradeCount=Just 3  → Abort
        * enabled=True finalEquity=Just 0.5  tradeCount=Just 12 → Abort
        * enabled=True finalEquity=Just 1.5  tradeCount=Just 1  → Allow (above threshold trumps minTrades)
-}
botStartupBacktestVerdictWithMinTrades :: Int -> Bool -> Maybe Double -> Maybe Int -> BacktestVerdict
botStartupBacktestVerdictWithMinTrades _ False _ _ = BacktestAllow
botStartupBacktestVerdictWithMinTrades _ True Nothing _ = BacktestNoVerdict
botStartupBacktestVerdictWithMinTrades minTradesRaw True mFinalEquity mTradeCount =
    let minTrades = max 1 minTradesRaw
     in if botStartupBacktestRoiAcceptable mFinalEquity
            then BacktestAllow
            else case mTradeCount of
                Just n | n >= minTrades -> BacktestAbort
                -- Zero-trade, unknown-trade, or under-min-trades smoke window:
                -- not a strong enough verdict to abort or prune.
                _ -> BacktestNoVerdict

{- | Should the bot-start guard's @BacktestAbort@ verdict /also/ delete the
combo from the top-combos store and DB?

As of 2026-06-12 the answer is /no/. The periodic refresh path has used
@applyComboUpdatesKeepAllWithStats@ since 2026-06-10 for the same reason:
when one box prunes a combo, the next cross-instance S3 union merge
resurrects it with stale, inflated metrics; whereas a kept, stamped record
wins the merge and falls out of the capped board by rank, fleet-wide. The
bot-start guard had been doing the opposite: prune locally AND delete from
the DB row. Today's launchd log shows the same combo UUIDs being pruned 2–3
times inside one process, which is the fingerprint of that race.

The guard now only /blocks/ the start; pruning is the optimizer's job.
This function exists so the policy is referenced from one place and
falsifiable in tests.
-}
botStartupGuardShouldPrune :: BacktestVerdict -> Bool
botStartupGuardShouldPrune _ = False

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

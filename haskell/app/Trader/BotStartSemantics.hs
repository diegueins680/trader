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
    capBotStartSymbolsPreservingOrphans,
    filterBotStartAttemptsPreservingOrphans,
    prioritizeBotStartSymbols,
    throttleBotStartSymbolsPreservingOrphans,
    queuedStartOrderErrorIssue,
    shouldResolveOriginComboOnAutoStart,
    shouldClearPositionOriginOnStart,
    shouldPersistPositionOriginOnSwitch,
    shouldPreserveProvidedComboOnActiveAdopt,
    adoptionMaxPositionSizeCap,
    capAdoptedMaxPositionSize,
    capAdoptedMaxPositionSizeWithCap,
    AdoptionEvidenceConfig (..),
    defaultAdoptionEvidenceConfig,
    adoptionMinEdgeFloor,
    adoptionMinTradeCount,
    adoptionMinWalkForwardSharpeMean,
    adoptionMaxWalkForwardSharpeStd,
    comboMinEdgeMeetsAdoptionFloor,
    comboMinEdgeMeetsAdoptionFloorWithConfig,
    comboTradeCountMeetsAdoptionFloor,
    comboTradeCountMeetsAdoptionFloorWithConfig,
    comboWalkForwardSharpeMeetsAdoptionFloor,
    comboWalkForwardSharpeMeetsAdoptionFloorWithConfig,
    comboWalkForwardSharpeStdMeetsAdoptionCeiling,
    comboWalkForwardSharpeStdMeetsAdoptionCeilingWithConfig,
) where

import Data.Char (isSpace, toUpper)
import Data.Maybe (fromMaybe, isJust)

import Trader.CostCalibration (venueMinEdgeFloor)
import Trader.Text (dedupeStable)

{- | Hard cap on the 'maxPositionSize' the bot will accept when adopting an
existing top-combo for live trading. Legacy combos on the leaderboard can
carry sizes up to 1.0 from before the cost-floor guards; combined with the
10-20x perp leverage observed on prod (2026-06-13) that produces 18-36%
account hits on a single 2% adverse move, which is the cliff shape of the
incident equity curve. The optimizer's new defaults sample sizes in
[0.15, 0.25]; this cap keeps adoption of pre-fix combos inside the same
envelope so the live fleet de-risks as combos churn.
-}
adoptionMaxPositionSizeCap :: Double
adoptionMaxPositionSizeCap = 0.25

{- | Clamp a combo's 'maxPositionSize' into the safe live-adoption range
@[0, 'adoptionMaxPositionSizeCap']@. Negative or non-finite inputs collapse
to zero. Smaller-than-cap values are preserved so combos that already
sample conservatively are not inflated.
-}
capAdoptedMaxPositionSize :: Double -> Double
capAdoptedMaxPositionSize = capAdoptedMaxPositionSizeWithCap adoptionMaxPositionSizeCap

capAdoptedMaxPositionSizeWithCap :: Double -> Double -> Double
capAdoptedMaxPositionSizeWithCap capRaw raw
    | isNaN raw || isInfinite raw = 0
    | raw < 0 = 0
    | isNaN capRaw || isInfinite capRaw = 0
    | capRaw < 0 = 0
    | otherwise = min capRaw raw

{- | Minimum backtest @tradeCount@ a top-combo must report on its stored
metrics before the bot-start path is allowed to adopt it for live trading.

Engineering rationale (2026-06-14)
==================================

The 2026-06-13 post-fix audit found that 500/500 leaderboard combos sat
below the new minEdge cost floor — the floor caught them. The deeper
problem hidden behind that filter is that the same 500 combos had
__median @tradeCount = 4__ with median Sharpe 8.5 and median annualized
return 8.4. A 4-trade backtest cannot distinguish a real edge from
noise: at a per-trade σ of 1% (typical for daily-bar crypto strategies)
the Sharpe of a 4-trade window of N(0, σ²) returns has a standard
error of ~0.5; an observed Sharpe of 8 is dominated by sampling
variance, not signal. The optimizer's production CLI guard already
rejects trials below 20 round trips (@TRADER_OPTIMIZER_MIN_ROUND_TRIPS@);
the bot-start adoption path must enforce the same floor at adoption
time so a future relaxation of the minEdge filter (e.g. when the
predictors improve and the cost floor is recalibrated) cannot let a
4-trade combo into live trading.

Falsifiable invariants:

  * @adoptionMinTradeCount >= 20@ — the optimizer-side production gate is
    the lower bound; adoption must not be more lenient than the gate that
    produced the combo.
  * 'comboTradeCountMeetsAdoptionFloor' is 'False' on 'Nothing' (no metric
    recorded) — adoption requires evidence, not the absence of evidence.
  * 'comboTradeCountMeetsAdoptionFloor' is monotone: if @n1 <= n2@ then
    @meets n1 ==> meets n2@.
-}
adoptionMinTradeCount :: Int
adoptionMinTradeCount = 20

data AdoptionEvidenceConfig = AdoptionEvidenceConfig
    { aecMinEdgeFloor :: !Double
    , aecMinTradeCount :: !Int
    , aecMinWalkForwardSharpeMean :: !Double
    , aecMaxWalkForwardSharpeStd :: !Double
    }
    deriving (Eq, Show)

defaultAdoptionEvidenceConfig :: AdoptionEvidenceConfig
defaultAdoptionEvidenceConfig =
    AdoptionEvidenceConfig
        { aecMinEdgeFloor = adoptionMinEdgeFloor
        , aecMinTradeCount = adoptionMinTradeCount
        , aecMinWalkForwardSharpeMean = adoptionMinWalkForwardSharpeMean
        , aecMaxWalkForwardSharpeStd = adoptionMaxWalkForwardSharpeStd
        }

{- | Minimum edge a live-adopted top combo must carry. This mirrors the venue
cost floor used by optimizer deployability: a combo whose edge does not beat
the modeled round-trip cost is not a candidate for live capital.
-}
adoptionMinEdgeFloor :: Double
adoptionMinEdgeFloor = venueMinEdgeFloor

comboMinEdgeMeetsAdoptionFloor :: Maybe Double -> Bool
comboMinEdgeMeetsAdoptionFloor =
    comboMinEdgeMeetsAdoptionFloorWithConfig defaultAdoptionEvidenceConfig

comboMinEdgeMeetsAdoptionFloorWithConfig :: AdoptionEvidenceConfig -> Maybe Double -> Bool
comboMinEdgeMeetsAdoptionFloorWithConfig _ Nothing = False
comboMinEdgeMeetsAdoptionFloorWithConfig config (Just edge)
    | isNaN edge || isInfinite edge = False
    | otherwise = edge >= max 0 (aecMinEdgeFloor config)

{- | Adoption-time predicate: does the combo's backtest report at least
'adoptionMinTradeCount' trades? A 'Nothing' reading fails closed because
adoption is a positive assertion ("this combo's evidence is strong enough
to put live capital behind"), not a negative one ("we have no reason to
reject").
-}
comboTradeCountMeetsAdoptionFloor :: Maybe Int -> Bool
comboTradeCountMeetsAdoptionFloor =
    comboTradeCountMeetsAdoptionFloorWithConfig defaultAdoptionEvidenceConfig

comboTradeCountMeetsAdoptionFloorWithConfig :: AdoptionEvidenceConfig -> Maybe Int -> Bool
comboTradeCountMeetsAdoptionFloorWithConfig _ Nothing = False
comboTradeCountMeetsAdoptionFloorWithConfig config (Just n) =
    n >= max 0 (aecMinTradeCount config)

{- | Minimum walk-forward mean Sharpe a top-combo must report before the
bot-start path is allowed to adopt it.

Engineering rationale (2026-06-14)
==================================

The 2026-06-13 fix turned walk-forward Sharpe gates on /by default in the
auto optimizer/ at @minWfSharpeMean = 0.3@ and @maxWfSharpeStd = 1.5@.
That closes the door for freshly-produced trials. Legacy combos on the
leaderboard, however, predate that change: today's snapshot shows
__0/500 combos with any walkForwardSummary at all__. Without an
adoption-time mirror of the optimizer gate the legacy population can
still leak through whenever the minEdge floor moves.

The value mirrors the optimizer's default exactly so the two gates are
falsifiably equal: if one changes, the other must change too, or the
test 'testAdoptionMinWalkForwardSharpeMatchesOptimizerDefault' fails.

Falsifiable invariants:

  * Missing reading ('Nothing'): fail closed.
  * Non-finite reading ('NaN' / +Inf / -Inf): fail closed.
  * Below threshold (@s < adoptionMinWalkForwardSharpeMean@): fail closed.
  * At-threshold equality: pass (the gate is @>=@, matching the optimizer's
    @>= minWfSharpeMean@ contract).
  * Predicate is monotone in the reading.
-}
adoptionMinWalkForwardSharpeMean :: Double
adoptionMinWalkForwardSharpeMean = 0.3

{- | Maximum cross-fold standard deviation of walk-forward Sharpe accepted by
live adoption. The optimizer defaults to @maxWfSharpeStd = 1.5@; adoption uses
the same ceiling so a combo that is profitable in one fold and unstable across
the rest does not become deployable merely because its mean clears the floor.
Set to 0 in an explicit config to disable this ceiling.
-}
adoptionMaxWalkForwardSharpeStd :: Double
adoptionMaxWalkForwardSharpeStd = 1.5

{- | Adoption-time predicate: does the combo's walk-forward summary report a
mean Sharpe that clears 'adoptionMinWalkForwardSharpeMean'? Missing and
non-finite readings fail closed.
-}
comboWalkForwardSharpeMeetsAdoptionFloor :: Maybe Double -> Bool
comboWalkForwardSharpeMeetsAdoptionFloor =
    comboWalkForwardSharpeMeetsAdoptionFloorWithConfig defaultAdoptionEvidenceConfig

comboWalkForwardSharpeMeetsAdoptionFloorWithConfig :: AdoptionEvidenceConfig -> Maybe Double -> Bool
comboWalkForwardSharpeMeetsAdoptionFloorWithConfig _ Nothing = False
comboWalkForwardSharpeMeetsAdoptionFloorWithConfig config (Just s)
    | isNaN s || isInfinite s = False
    | otherwise = s >= aecMinWalkForwardSharpeMean config

comboWalkForwardSharpeStdMeetsAdoptionCeiling :: Maybe Double -> Bool
comboWalkForwardSharpeStdMeetsAdoptionCeiling =
    comboWalkForwardSharpeStdMeetsAdoptionCeilingWithConfig defaultAdoptionEvidenceConfig

comboWalkForwardSharpeStdMeetsAdoptionCeilingWithConfig :: AdoptionEvidenceConfig -> Maybe Double -> Bool
comboWalkForwardSharpeStdMeetsAdoptionCeilingWithConfig config mStd
    | aecMaxWalkForwardSharpeStd config <= 0 = isJust mStd
    | otherwise =
        case mStd of
            Nothing -> False
            Just s
                | isNaN s || isInfinite s -> False
                | otherwise -> s <= aecMaxWalkForwardSharpeStd config

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

As of 2026-06-12 the answer is /no/. The bot-start guard had been pruning
locally AND deleting from the DB row; that made a noisy startup smoke window
destructive. Scheduled stale refreshes are the pruning authority instead:
they can drop below-floor refreshed combos because the top-combos payload now
carries a drop tombstone to prevent stale S3/DB replicas from resurrecting
the old score.

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

{- | Apply the configured portfolio cap without ever dropping a symbol that
already has an exchange position. Orphan adoption is risk reduction and may
temporarily overflow the cap; only flat/new targets are deferred.
-}
capBotStartSymbolsPreservingOrphans :: Int -> [String] -> [String] -> ([String], [String])
capBotStartSymbolsPreservingOrphans maxSymbols regularSymbols orphanSymbols =
    let orphans = prioritizeBotStartSymbols [] orphanSymbols
        regular = filter (`notElem` orphans) (prioritizeBotStartSymbols regularSymbols [])
        regularCapacity = max 0 maxSymbols
     in (orphans ++ take regularCapacity regular, drop regularCapacity regular)

{- | Preserve existing-position adoption attempts even when the new-entry
circuit breaker is open or a symbol has stale backoff state. A successful
exchange inventory scan proves that the deployment can inspect current risk;
that risk must receive an owning worker before optional starts resume.
-}
filterBotStartAttemptsPreservingOrphans :: Bool -> (String -> Bool) -> [String] -> [String] -> ([String], [String])
filterBotStartAttemptsPreservingOrphans circuitOpen allowedByBackoff orphanSymbols missingSymbols =
    let missing = prioritizeBotStartSymbols missingSymbols []
        orphanSet = prioritizeBotStartSymbols [] orphanSymbols
        urgent = filter (`elem` orphanSet) missing
        regular = filter (`notElem` orphanSet) missing
        allowedRegular =
            if circuitOpen
                then []
                else filter allowedByBackoff regular
        blockedRegular = filter (`notElem` allowedRegular) regular
     in (urgent ++ allowedRegular, blockedRegular)

{- | Start every missing orphan immediately. New/flat exposure remains subject
to the per-cycle throttle and waits for the next cycle whenever adoption work
is pending.
-}
throttleBotStartSymbolsPreservingOrphans :: Int -> [String] -> [String] -> ([String], [String])
throttleBotStartSymbolsPreservingOrphans maxRegularStarts orphanSymbols missingSymbols =
    let missing = prioritizeBotStartSymbols missingSymbols []
        orphanSet = prioritizeBotStartSymbols [] orphanSymbols
        urgent = filter (`elem` orphanSet) missing
        regular = filter (`notElem` orphanSet) missing
        regularCapacity = if null urgent then max 0 maxRegularStarts else 0
     in (urgent ++ take regularCapacity regular, drop regularCapacity regular)

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

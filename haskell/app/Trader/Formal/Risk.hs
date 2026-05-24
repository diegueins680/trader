module Trader.Formal.Risk (
    HaltInputs (..),
    RiskVerificationReport (..),
    specRiskHalt,
    verifyFormalRisk,
) where

import Data.Maybe (isNothing)
-- 'HaltInputs' and 'specRiskHalt' are defined in 'Trader.Trading' so the
-- simulation loop can call the canonical spec directly. This module
-- re-exports them and proves properties via 'verifyFormalRisk'.
import Trader.Trading (
    ExitReason (..),
    HaltInputs (..),
    specRiskHalt,
 )

-- ---------------------------------------------------------------------------
-- Verification report
-- ---------------------------------------------------------------------------

data RiskVerificationReport = RiskVerificationReport
    { fvrRiskHaltMonotone :: !Bool
    , fvrRiskHaltResetDaily :: !Bool
    , fvrRiskHaltResetWeekly :: !Bool
    , fvrRiskHaltPreservesOther :: !Bool
    , fvrRiskHaltNoFalsePositive :: !Bool
    }
    deriving (Eq, Show)

-- | Exhaustive enumeration over bounded risk states.
verifyFormalRisk :: RiskVerificationReport
verifyFormalRisk =
    let
        -- Small grid of doubles
        doubles = [0, 0.01, 0.05, 0.1, 0.5, 1.0]
        mDoubles = [Nothing, Just 0, Just 0.05, Just 0.1, Just 1.0]
        prevReasons =
            [ Nothing
            , Just ExitMaxDailyLoss
            , Just ExitMaxWeeklyLoss
            , Just ExitMaxDrawdown
            , Just (ExitOther "NEGATIVE_EXPECTANCY")
            , Just (ExitOther "MANUAL")
            ]
        booleans = [False, True]
        expectancys = [Nothing, Just (-0.1), Just 0, Just 0.01]

        allInputs =
            [ HaltInputs
                { hiPrevHaltReason = pr
                , hiDayChanged = dc
                , hiWeekChanged = wc
                , hiDailyLoss = dl
                , hiWeeklyLoss = wl
                , hiDrawdown = dd
                , hiExpectancy = ex
                , hiMaxDailyLossLim = mdl
                , hiMaxWeeklyLossLim = mwl
                , hiMaxDrawdownLim = mdd
                , hiMinExpectancy = me
                }
            | pr <- prevReasons
            , dc <- booleans
            , wc <- booleans
            , dl <- doubles
            , wl <- doubles
            , dd <- doubles
            , ex <- expectancys
            , mdl <- mDoubles
            , mwl <- mDoubles
            , mdd <- mDoubles
            , me <- mDoubles
            ]

        -- If halted, drawdown must be >= limit OR daily loss >= limit OR
        -- weekly loss >= limit OR expectancy < minExpectancy, OR it is a
        -- preserved previous halt.
        haltMonotone =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                     in case result of
                            Just ExitMaxDrawdown ->
                                maybe False (hiDrawdown hi >=) (hiMaxDrawdownLim hi)
                                    || hiPrevHaltReason hi == Just ExitMaxDrawdown
                            Just ExitMaxDailyLoss ->
                                maybe False (hiDailyLoss hi >=) (hiMaxDailyLossLim hi)
                                    || (hiPrevHaltReason hi == Just ExitMaxDailyLoss && not (hiDayChanged hi))
                            Just ExitMaxWeeklyLoss ->
                                maybe False (hiWeeklyLoss hi >=) (hiMaxWeeklyLossLim hi)
                                    || (hiPrevHaltReason hi == Just ExitMaxWeeklyLoss && not (hiWeekChanged hi))
                            Just (ExitOther "NEGATIVE_EXPECTANCY") ->
                                maybe False (\lim -> maybe False (< lim) (hiExpectancy hi)) (hiMinExpectancy hi)
                                    || hiPrevHaltReason hi == Just (ExitOther "NEGATIVE_EXPECTANCY")
                            Just (ExitOther _) ->
                                hiPrevHaltReason hi == result
                            Nothing -> True
                )
                allInputs

        -- Daily loss halt resets on day change (unless the new day is already
        -- breaching the limit).
        resetDaily =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        stillBreached = maybe False (hiDailyLoss hi >=) (hiMaxDailyLossLim hi)
                     in not
                            ( hiPrevHaltReason hi == Just ExitMaxDailyLoss
                                && hiDayChanged hi
                                && not stillBreached
                            )
                            || (result /= Just ExitMaxDailyLoss)
                )
                allInputs

        -- Weekly loss halt resets on week change (unless the new week is already
        -- breaching the limit).
        resetWeekly =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        stillBreached = maybe False (hiWeeklyLoss hi >=) (hiMaxWeeklyLossLim hi)
                     in not
                            ( hiPrevHaltReason hi == Just ExitMaxWeeklyLoss
                                && hiWeekChanged hi
                                && not stillBreached
                            )
                            || (result /= Just ExitMaxWeeklyLoss)
                )
                allInputs

        -- If previously halted for a non-time-bound reason, it is preserved.
        preservesOther =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                     in case hiPrevHaltReason hi of
                            Just r
                                | r /= ExitMaxDailyLoss && r /= ExitMaxWeeklyLoss ->
                                    result == Just r || result == hiPrevHaltReason hi
                            _ -> True
                )
                allInputs

        -- If no limits are set and no previous halt, result is Nothing.
        noFalsePositive =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                     in not
                            ( isNothing (hiPrevHaltReason hi)
                                && isNothing (hiMaxDailyLossLim hi)
                                && isNothing (hiMaxWeeklyLossLim hi)
                                && isNothing (hiMaxDrawdownLim hi)
                                && isNothing (hiMinExpectancy hi)
                            )
                            || isNothing result
                )
                allInputs
     in
        RiskVerificationReport
            { fvrRiskHaltMonotone = haltMonotone
            , fvrRiskHaltResetDaily = resetDaily
            , fvrRiskHaltResetWeekly = resetWeekly
            , fvrRiskHaltPreservesOther = preservesOther
            , fvrRiskHaltNoFalsePositive = noFalsePositive
            }

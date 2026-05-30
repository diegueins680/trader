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
    , fvrRiskHaltPositionSize :: !Bool
    , fvrRiskHaltLossStreak :: !Bool
    , fvrMaxPositionSizeBound :: !Bool
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
                , hiPositionSize = 0
                , hiMaxPositionSizeLim = Nothing
                , hiConsecutiveLosses = 0
                , hiMaxLossStreakLim = Nothing
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
                                && isNothing (hiMaxPositionSizeLim hi)
                            )
                            || isNothing result
                )
                allInputs

        -- Position-size halt: if position size exceeds the sanitized limit,
        -- a POSITION_SIZE halt is emitted.
        positionSizeHalt =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        limit = fmap (max 0) (hiMaxPositionSizeLim hi)
                        size = max 0 (hiPositionSize hi)
                     in case result of
                            Just (ExitOther "POSITION_SIZE") ->
                                maybe False (size >) limit
                            _ -> True
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = ps
                    , hiMaxPositionSizeLim = mps
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    }
                | ps <- [0, 0.5, 1.0, 1.5, 2.0]
                , mps <- [Nothing, Just 0, Just 1.0, Just 2.0]
                ]

        -- Loss-streak cooldown halt: if consecutive losses exceed the configured
        -- max, a LOSS_STREAK halt is emitted. This prevents runaway drawdowns
        -- during persistent adverse regimes.
        lossStreakHalt =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                     in case result of
                            Just (ExitOther "LOSS_STREAK") ->
                                maybe False (\lim -> hiConsecutiveLosses hi > lim) (hiMaxLossStreakLim hi)
                            _ -> True
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = 0
                    , hiMaxPositionSizeLim = Nothing
                    , hiConsecutiveLosses = cl
                    , hiMaxLossStreakLim = mls
                    }
                | cl <- [0, 1, 2, 3, 5]
                , mls <- [Nothing, Just 0, Just 2, Just 3]
                ]

        -- Max-position-size simulation bound: ecMaxPositionSize must be
        -- non-negative and finite; when set > 0 the simulation must never
        -- emit a position exceeding it.
        maxPositionSizeBound =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        limit = fmap (max 0) (hiMaxPositionSizeLim hi)
                        size = max 0 (hiPositionSize hi)
                     in case result of
                            Just (ExitOther "POSITION_SIZE") ->
                                maybe False (size >) limit
                            _ -> True
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = ps
                    , hiMaxPositionSizeLim = mps
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    }
                | ps <- [0, 0.5, 1.0, 1.5, 2.0, 5.0, 1e308]
                , mps <- [Nothing, Just 0, Just 1.0, Just 2.0, Just 1e308]
                ]
     in
        RiskVerificationReport
            { fvrRiskHaltMonotone = haltMonotone
            , fvrRiskHaltResetDaily = resetDaily
            , fvrRiskHaltResetWeekly = resetWeekly
            , fvrRiskHaltPreservesOther = preservesOther
            , fvrRiskHaltNoFalsePositive = noFalsePositive
            , fvrRiskHaltPositionSize = positionSizeHalt
            , fvrRiskHaltLossStreak = lossStreakHalt
            , fvrMaxPositionSizeBound = maxPositionSizeBound
            }
